import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import glob
import os
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


def get_epsilon(epoch, n_epochs):
    return max(epsilon_final, epsilon_start * (1.0 - epoch / n_epochs))

def get_temperature_sm(epoch, n_epochs, init_temp, final_temp):
    decay_rate = (final_temp / init_temp) ** (1.0 / n_epochs)
    return init_temp * (decay_rate ** epoch)

def get_routes_distance(distance_matrix, routes): 
    total_distance = 0
    for route in routes:
        r = [0]+route+[0]
        total_distance += np.sum([distance_matrix[r[i],r[i+1]] for i in range(len(r)-1)])
    return total_distance

def compute_entropy(probs):
    if probs.dim() == 2:
        probs = probs.squeeze(0)
    ent = -(probs * torch.log(probs + 1e-12)).sum()
    return ent

def calculate_distance_matrix(coords):
    num_nodes = len(coords)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return distance_matrix

def parse_problem_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    customers = []
    for line in lines[5:]:  
        parts = line.strip().split()
        if len(parts) >= 6 and parts[0].isdigit(): 
            customers.append([float(parts[1]), float(parts[2]), int(parts[3])])
    return np.array(customers)  

def get_neighbours(distance_matrix):
    n_vertices = distance_matrix.shape[0]
    neighbours = []
    for i in range(n_vertices):
        index_dist = [(j, distance_matrix[i][j]) for j in range(n_vertices)]
        sorted_index_dist = sorted(index_dist, key=lambda x: x[1])
        neighbours.append([x[0] for x in sorted_index_dist])
    return neighbours

class ConvNeXtModel(nn.Module):
    def __init__(self, input_channels, num_nodes, output_dim,temperature_sm):
        super(ConvNeXtModel, self).__init__()
        self.model=convnext_tiny(pretrain=False)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features,output_dim)
        self.temperature_sm = temperature_sm
    def forward(self, x):
        logits = self.model(x)
        return F.softmax(logits / self.temperature_sm, dim=-1)
    
def prepare_input_tensor(dist_matrix, data, current_routes, absents, device):
    num_nodes = len(data)
    input_channels = 3

    input_tensor = np.zeros((input_channels, num_nodes, num_nodes))

    # 0: Distance_Matrix
    input_tensor[0] = dist_matrix

    # 1: 節點Demand
    demands = np.zeros(num_nodes)
    for i, node in enumerate(data):
        demands[i] = node[2]  
    input_tensor[1] = np.tile(demands, (num_nodes, 1))

    # 2: current_routes(目前節點被分配至哪條路徑)
    route_flags = np.full(num_nodes, -1)  # -1 表示未分配
    for route_idx, route in enumerate(current_routes):
        for node in route:
            route_flags[node] = route_idx
    input_tensor[2] = np.tile(route_flags, (num_nodes, 1))

    return torch.tensor(input_tensor, dtype=torch.float32).to(device)

def ruin(last_routes, neighbours, data, ruin_model, dist_matrix, device, epsilon, in_absents=None):
    def remove_nodes(tr, l_t, c, m):
        m_alpha=0.01
        def string_removal(tr, l_t, c):
            i_c = tr.index(c)
            range1 = max(0, i_c + 1 - l_t)
            range2 = min(i_c, len(tr) - l_t) + 1
            start = np.random.randint(range1, range2)
            return tr[start:start + l_t]

        def split_removal(tr, l_t, c, m):
            additional_l = min(m, len(tr) - l_t)
            l_t_m = l_t + additional_l
            i_c = tr.index(c)
            range1 = max(0, i_c + 1 - l_t_m)
            range2 = min(i_c, len(tr) - l_t_m) + 1
            start = np.random.randint(range1, range2)
            potential_removal = tr[start:start + l_t_m]
            return [potential_removal[i] for i in np.random.choice(l_t_m, l_t, replace=False)]

        if np.random.random() < 0.5:
            newly_removed = string_removal(tr, l_t, c)
        else:
            newly_removed = split_removal(tr, l_t, c, m)
            if m < (len(tr) - l_t) or np.random.random() > m_alpha:
                m += 1
        return m, newly_removed

    def find_t(last_routes, c):
        for i in range(len(last_routes)):
            if c in last_routes[i]:
                return i
        return None

    def routes_summary(last_routes, absents):
        current_routes = []
        for r in last_routes:
            new_r = [x for x in r if x not in absents]
            if len(new_r) > 0:
                current_routes.append(new_r)
        return current_routes
    L_max=10.0
    c_bar=10.0
    m = 1
    l_s_max = min(L_max, np.mean([len(x) for x in last_routes]))
    k_s_max = 4.0 * c_bar / (1.0 + l_s_max) - 1.0
    k_s = int(np.random.random() * k_s_max + 1.0)

    absents = [] if in_absents is None else copy.deepcopy()
    ruined_t_indices = set([])

    input_tensor = prepare_input_tensor(dist_matrix, data, current_routes=last_routes, absents=absents, device=device)
    policy_probs = ruin_model(input_tensor.unsqueeze(0))
    policy_probs = policy_probs.squeeze(0)
    num_actions = policy_probs.size(0)
    action_probs = (1 - epsilon) * policy_probs + epsilon / num_actions
    action_probs = action_probs / action_probs.sum()
    c_seed = torch.multinomial(action_probs, 1).item()
    ruin_log_prob = torch.log(action_probs[c_seed])
    ruin_entropy = compute_entropy(action_probs)

    for c in neighbours[c_seed]:
        if len(ruined_t_indices) >= k_s:
            break
        if c not in absents and c != 0:
            t = find_t(last_routes, c)
            if t in ruined_t_indices:
                continue
            else:
                l_t_max = min(l_s_max, len(last_routes[t]))
                l_t = int(np.random.random() * l_t_max + 1.0)
                m, newly_removed = remove_nodes(last_routes[t], l_t, c, m)
            absents = absents + newly_removed
            ruined_t_indices.add(t)
    current_routes = routes_summary(last_routes, absents)
    return current_routes, absents, ruin_entropy,ruin_log_prob

def recreate(data, dist_matrix, current_routes, absents,vehicle_capacities): 
    def route_add(dist_matrix, current_routes, c, adding_position):
        if adding_position[0]==-1: # adding new route
            current_routes = current_routes+[[c]]
        else:
            chg_r = current_routes[adding_position[0]]
            new_r = chg_r[:adding_position[1]]+[c]+chg_r[adding_position[1]:]
            current_routes[adding_position[0]] = new_r
        return current_routes

    def sort_absents_with_weights(data, absents):

        strategies = ['random', 'demand', 'far', 'close']
        weights = [4, 4, 2, 1] 

        sort_methods = {
            'random': lambda: np.random.permutation(absents),
            'demand': lambda: sorted(absents, key=lambda c: data[c][2], reverse=True), 
            'far':    lambda: sorted(absents, key=lambda c: np.linalg.norm(data[c][:2] - data[0][:2]), reverse=True),  
            'close':  lambda: sorted(absents, key=lambda c: np.linalg.norm(data[c][:2] - data[0][:2]))  
            }
        chosen_strategy = np.random.choice(strategies, p=np.array(weights) / sum(weights))
        
        return sort_methods[chosen_strategy]()

    absents = sort_absents_with_weights(data, absents)
    
    for c in absents:
        probable_place = []
        for ir,r in enumerate(current_routes):
            assigned_capacity = vehicle_capacities[ir % len(vehicle_capacities)]
            if (np.sum([data[x][2] for x in r]) + data[c][2]) > assigned_capacity:
                continue
            for iri in range(len(r)+1):
                if iri == 0:
                    prev_node = 0
                else:
                    prev_node = r[iri - 1]

                if iri == len(r):
                    next_node = 0
                else:
                    next_node = r[iri]
                if np.random.random() < 0.01:
                    continue
                cost_diff = dist_matrix[prev_node, c] + dist_matrix[c, next_node] - dist_matrix[prev_node, next_node]
                probable_place.append((ir, iri, cost_diff))
        if len(probable_place)==0:
            adding_position = (-1,-1,1)
        else:
            adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
        current_routes = route_add(dist_matrix, current_routes, c, adding_position)

    return current_routes

def calculate_total_distance_with_current(routes, dist_matrix):
    total_distance = 0
    depot = 0 
    for route in routes:
        if not route:
            continue
        total_distance += dist_matrix[depot, route[0]]
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i], route[i + 1]]
        total_distance += dist_matrix[route[-1], depot]
    return total_distance

def train_and_save_model(data, ruin_model, save_path, device):
    iterations_best_distances = []
    elapsed_times = []
    optimizer_ruin = torch.optim.Adam(ruin_model.parameters(), lr=0.0001)
    # optimizer_recreate = torch.optim.Adam(recreate_model.parameters(), lr=0.0001)
    dist_matrix = calculate_distance_matrix(data[:, :2])
    alpha_T = (final_T/init_T)**(1.0/n_iter)
    temperature = init_T
    init_routes = [[i] for i in range(1,len(data))]
    best_routes = copy.deepcopy(init_routes)
    best_distance =  get_routes_distance(dist_matrix, best_routes)
    last_routes = copy.deepcopy(init_routes)
    last_distance = get_routes_distance(dist_matrix, best_routes)
    
    iterations_best_distances.append(best_distance)
    print(len(best_routes), best_distance)
    absents = []
    for epoch in range(n_epochs):
        temperature_sm = get_temperature_sm(epoch, n_epochs,init_temp,final_temp)
        ruin_model.temperature_sm = temperature_sm
        epsilon = get_epsilon(epoch, n_epochs)
        neighbours = get_neighbours(dist_matrix)

        ruin_log_probs = []
        ruin_entropies = []
        # recreate_log_probs_list = []
        # recreate_entropies_list = []
        iteration_rewards = []

        for i in range(n_iter):
            current_routes, absents, ruin_entropy, ruin_log_prob= ruin(last_routes=last_routes, neighbours=neighbours, data=data, ruin_model=ruin_model, dist_matrix=dist_matrix, device=device,epsilon=epsilon)
            # current_routes, recreate_entropies, recreate_log_probs = recreate(data=data, dist_matrix=dist_matrix, current_routes=current_routes, absents=absents, device=device, recreate_model=recreate_model)
            current_routes = recreate(data=data, dist_matrix=dist_matrix, current_routes=current_routes, absents=absents,vehicle_capacities=vehicle_capacities)
            current_distance = calculate_total_distance_with_current(current_routes, dist_matrix)
            reward = -current_distance

            if len(current_routes)<len(best_routes) or \
            (current_distance<(last_distance-temperature*np.log(np.random.random())) and \
                len(current_routes)<=len(best_routes)):
                
                if len(current_routes)<len(best_routes) or current_distance<best_distance:
                    best_distance = current_distance
                    best_routes = current_routes

                last_distance = current_distance
                last_routes = current_routes
            temperature*=alpha_T
            elapsed_time = time.time() - start_time
            elapsed_times.append(elapsed_time)
            iterations_best_distances.append(best_distance)
            ruin_log_probs.append(ruin_log_prob)
            ruin_entropies.append(ruin_entropy)

            # if len(recreate_log_probs) > 0:
            #     recreate_log_probs_mean = torch.stack(recreate_log_probs).mean()
            # else:
            #     recreate_log_probs_mean = torch.tensor(0.0, device=device)
            # if len(recreate_entropies) > 0:
            #     recreate_entropies_mean = torch.tensor(recreate_entropies, device=device).mean()
            # else:
            #     recreate_entropies_mean = torch.tensor(0.0, device=device)
            # recreate_log_probs_list.append(recreate_log_probs_mean)
            # recreate_entropies_list.append(recreate_entropies_mean)

            iteration_rewards.append(reward)

        # end for

        # flattened_recreate_log_probs = [lp for sublist in recreate_log_probs_list for lp in sublist]
        # flattened_recreate_entropies = [ent for sublist in recreate_entropies_list for ent in sublist]

        ruin_log_probs_tensor = torch.stack(ruin_log_probs)
        ruin_entropies_tensor = torch.tensor(ruin_entropies, dtype=torch.float32, device=device)
        
        # recreate_log_probs_tensor = torch.stack(recreate_log_probs_list)
        # recreate_entropies_tensor = torch.stack(recreate_entropies_list)
        
        iteration_rewards_tensor = torch.tensor(iteration_rewards, dtype=torch.float32, device=device)

        rewards_mean = iteration_rewards_tensor.mean()
        rewards_std = iteration_rewards_tensor.std() + 1e-8
        iteration_rewards_tensor = (iteration_rewards_tensor - rewards_mean) / rewards_std

        loss_ruin = -(iteration_rewards_tensor * ruin_log_probs_tensor).mean() - entropy_weight * ruin_entropies_tensor.mean()
        # loss_recreate = -(iteration_rewards_tensor * recreate_log_probs_tensor).mean() - entropy_weight * recreate_entropies_tensor.mean()
        optimizer_ruin.zero_grad()
        # optimizer_recreate.zero_grad()
        loss_ruin.backward()
        # loss_recreate.backward()
        optimizer_ruin.step()
        # optimizer_recreate.step()
        
        print(f"Epoch {epoch + 1}/{n_epochs}, Best Epoch {epoch + 1} Distance: {best_distance}")
        print(f"Epoch {epoch + 1}/{n_epochs}, Best Epoch {epoch + 1} Routes: {best_routes}")
        # if (epoch+1) % 100 == 0 and epoch != 0:
        #     torch.save(ruin_model.state_dict(), save_path + f"_ruin_model_{epoch+1}th_epoch.pth")
        #     torch.save(recreate_model.state_dict(), save_path + "_recreate_model.pth")

    return best_distance,best_routes,iterations_best_distances,elapsed_times

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vehicle_capacities = [30, 25, 20]
    iterations_best_distances = []
    elapsed_times = []
    iterations= []
    total_times = []
    folder_path = "./test_data/3v_80c"
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))

    for index, file_path in enumerate(file_paths):
        customers = parse_problem_instance(file_path)
        print(file_path)
        print(customers.shape)
        print("------------------")  
        #40c -> 60*100
        #60c -> 90*100
        #80c -> 120*100
        n_epochs=120
        n_iter=100
        
        # SA temperature
        init_T=100.0
        final_T=1.0

        # softmax temperature
        init_temp=2.0
        final_temp=1.0

        epsilon_start=1.0
        epsilon_final=0.01

        entropy_weight = 0.1
        ruin_model = ConvNeXtModel(input_channels=3, num_nodes=len(customers), output_dim=len(customers),temperature_sm=1.0).to(device)
        start_time = time.time()
        best_distance,best_routes,iterations_best_distances,elapsed_times = train_and_save_model(customers, ruin_model,  save_path="trained_model", device=device)
        end_time = time.time()
        execution_time = end_time - start_time
        total_times.append(execution_time)
        print("\n")
        with open('hcvrp_results_mean.txt', 'a') as f:
            f.write(f"Input File: {file_path}\n")
            f.write(f"Distance: {best_distance}\n")
            f.write(f"Best Routes: {best_routes}\n")
            f.write(f"Best Distances: {[round(value, 3) for value in iterations_best_distances]}\n")
            f.write(f"Elapsed times: {[round(value, 3) for value in elapsed_times]}\n\n")

    # with open('drl_hcvrp_results.txt', 'a') as f:
    #     f.write(f"Best Distance: {best_distance}\n")
    #     f.write(f"Computation Time: {execution_time}\n\n")
