import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import glob
import os
from torchvision.models import convnext_tiny

n_epochs = 1200
n_iter = 10

# SA temperature
init_T  = 100.0
final_T = 1.0

# softmax temperature
init_temp  = 2.0
final_temp = 1.0

# Epsilon-greedy
epsilon_start = 1.0
epsilon_final = 0.01

# Entropy regularization
entropy_weight = 0.1

vehicle_capacities = [30, 25, 20]

def get_epsilon(epoch, n_epochs):
    return max(epsilon_final, epsilon_start * (1.0 - epoch / n_epochs))

def get_temperature_sm(epoch, n_epochs, init_temp, final_temp):
    decay_rate = (final_temp / init_temp) ** (1.0 / n_epochs)
    return init_temp * (decay_rate ** epoch)

def get_routes_distance(distance_matrix, routes): 
    total_distance = 0
    for route in routes:
        r = [0]+route+[0]
        total_distance += np.sum([
            distance_matrix[r[i], r[i+1]] for i in range(len(r)-1)
        ])
    return total_distance

def compute_entropy(probs):
    if probs.dim() == 2:
        probs = probs.squeeze(0)
    ent = -(probs * torch.log(probs + 1e-12)).sum()
    return ent

def calculate_distance_matrix(coords):
    num_nodes = len(coords)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix

def parse_problem_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    customers = []
    for line in lines[5:]:  
        parts = line.strip().split()
        if len(parts) >= 6 and parts[0].isdigit(): 
            x_coord = float(parts[1])
            y_coord = float(parts[2])
            demand  = int(parts[3])
            customers.append([x_coord, y_coord, demand])
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
    def __init__(self, input_channels, num_nodes, l_t_max=10, temperature_sm=1.0):
        super(ConvNeXtModel, self).__init__()
        self.l_t_max = l_t_max
        self.num_nodes = num_nodes
        
        base_model = convnext_tiny(weights=None)
        in_features = base_model.classifier[2].in_features
        # 最終輸出維度 = 節點數 + l_t_max
        base_model.classifier[2] = nn.Linear(in_features, num_nodes + l_t_max)
        
        self.model = base_model
        self.temperature_sm = temperature_sm

    def forward(self, x):
        """
        x shape=(batch_size, channels, N, N)
        輸出 shape=(batch_size, num_nodes + l_t_max)
        其中前 num_nodes => node_logits
             後 l_t_max  => l_t_logits
        """
        logits = self.model(x)  # shape=(batch_size, num_nodes + l_t_max)
        return F.softmax(logits / self.temperature_sm, dim=-1)


def prepare_input_tensor(dist_matrix, data, current_routes, absents, device):
    num_nodes = len(data)
    input_channels = 3

    input_tensor = np.zeros((input_channels, num_nodes, num_nodes), dtype=np.float32)

    # Channel 0: distance matrix
    input_tensor[0] = dist_matrix

    # Channel 1: demand
    demands = np.array([node[2] for node in data], dtype=np.float32)
    for i in range(num_nodes):
        input_tensor[1, i, :] = demands[i]

    # Channel 2: route assignment => -1表示未分配
    route_flags = np.full(num_nodes, -1, dtype=np.float32)
    for route_idx, route in enumerate(current_routes):
        for node in route:
            route_flags[node] = route_idx
    for i in range(num_nodes):
        input_tensor[2, i, :] = route_flags[i]

    return torch.tensor(input_tensor, dtype=torch.float32, device=device)

def remove_one_node_from_route(routes, node_id):
    for r in routes:
        if node_id in r:
            r.remove(node_id)
    routes = [r for r in routes if len(r) > 0]
    return routes

def ruin(last_routes, data, ruin_model, dist_matrix, device, epsilon, in_absents=None):
    absents = [] if in_absents is None else copy.deepcopy(in_absents)
    
    # 1) forward => 取整體輸出 => front=(num_nodes), back=(l_t_max)
    input_tensor = prepare_input_tensor(dist_matrix, data, last_routes, absents, device)
    full_probs = ruin_model(input_tensor.unsqueeze(0)).squeeze(0)  # shape=(num_nodes + l_t_max,)

    num_nodes = len(data)
    l_t_max   = full_probs.size(0) - num_nodes
    
    # 分割出 node_probs / l_t_probs
    node_probs = full_probs[:num_nodes]
    l_t_probs  = full_probs[num_nodes:]

    # 2) sample l_t
    l_t_probs = l_t_probs / l_t_probs.sum()  # normalize
    l_t_idx = torch.multinomial(l_t_probs, 1).item()  # in [0..(l_t_max-1)]
    l_t = l_t_idx + 1  # => [1..l_t_max]
    
    ruin_log_prob = torch.log(l_t_probs[l_t_idx] + 1e-12)
    ruin_entropy  = compute_entropy(l_t_probs)

    # Epsilon for node_probs
    node_probs = node_probs / node_probs.sum()
    def sample_node_with_epsilon(node_probs):
        num_actions = node_probs.size(0)
        action_probs = (1-epsilon)*node_probs + epsilon/num_actions
        action_probs = action_probs / action_probs.sum()
        chosen = torch.multinomial(action_probs, 1).item()
        return chosen, torch.log(action_probs[chosen] + 1e-12), compute_entropy(action_probs)
    
    # 4) 多步移除 => l_t 次
    for _ in range(l_t):
        if len(last_routes)==0:
            break 
        chosen_node, node_lp, node_ent = sample_node_with_epsilon(node_probs)

        ruin_log_prob += node_lp
        ruin_entropy  += node_ent
        
        if (chosen_node not in absents) and (chosen_node != 0):
            absents.append(chosen_node)
            last_routes = remove_one_node_from_route(last_routes, chosen_node)
        
        if len(last_routes)==0:
            break

    return last_routes, absents, ruin_entropy, ruin_log_prob


def recreate(data, dist_matrix, current_routes, absents, vehicle_capacities):
    def route_add(dist_matrix, current_routes, c, adding_position):
        if adding_position[0] == -1: # adding new route
            current_routes = current_routes + [[c]]
        else:
            chg_r = current_routes[adding_position[0]]
            new_r = chg_r[:adding_position[1]] + [c] + chg_r[adding_position[1]:]
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
        chosen_strategy = np.random.choice(strategies, p=np.array(weights)/sum(weights))
        return sort_methods[chosen_strategy]()

    absents = sort_absents_with_weights(data, absents)

    for c in absents:
        probable_place = []
        for ir, r in enumerate(current_routes):
            assigned_capacity = vehicle_capacities[ir % len(vehicle_capacities)]
            if (np.sum([data[x][2] for x in r]) + data[c][2]) > assigned_capacity:
                continue
            for iri in range(len(r)+1):
                prev_node = r[iri-1] if iri>0 else 0
                next_node = r[iri]   if iri<len(r) else 0
                if np.random.random() < 0.01:
                    continue
                cost_diff = dist_matrix[prev_node, c] + dist_matrix[c, next_node] - dist_matrix[prev_node, next_node]
                probable_place.append((ir, iri, cost_diff))

        if len(probable_place) == 0:
            adding_position = (-1, -1, 1)
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
        for i in range(len(route)-1):
            total_distance += dist_matrix[route[i], route[i+1]]
        total_distance += dist_matrix[route[-1], depot]
    return total_distance


def train_and_save_model_multiple(all_data, ruin_model, device):
    optimizer_ruin = torch.optim.Adam(ruin_model.parameters(), lr=0.0001)
    alpha_T = (final_T/init_T)**(1.0/n_iter)

    best_distances_per_instance = []
    best_routes_per_instance = []
    for idx, d in enumerate(all_data):
        customers = d["customers"]
        dist_matrix = calculate_distance_matrix(customers[:, :2])
        init_routes = [[i] for i in range(1, len(customers))]
        best_routes = copy.deepcopy(init_routes)
        best_distance = get_routes_distance(dist_matrix, best_routes)
        best_distances_per_instance.append(best_distance)
        best_routes_per_instance.append(best_routes)

    start_time = time.time()
    for epoch in range(n_epochs):
        ruin_log_probs_all = []
        ruin_entropies_all = []
        iteration_rewards_all = []

        for idx, d in enumerate(all_data):
            customers = d["customers"]
            file_path = d["file_path"]
            v_caps    = d["vehicle_capacities"]

            dist_matrix = calculate_distance_matrix(customers[:, :2])
            last_routes   = copy.deepcopy(best_routes_per_instance[idx])
            last_distance = best_distances_per_instance[idx]
            temperature   = init_T

            for i in range(n_iter):
                current_routes = copy.deepcopy(last_routes)
                absents = []

                ruin_routes, absents, ruin_entropy, ruin_log_prob = ruin(
                    last_routes = current_routes,
                    data        = customers,
                    ruin_model  = ruin_model,
                    dist_matrix = dist_matrix,
                    device      = device,
                    epsilon     = get_epsilon(epoch, n_epochs)
                )

                current_routes = recreate(
                    data=customers,
                    dist_matrix=dist_matrix,
                    current_routes=ruin_routes,
                    absents=absents,
                    vehicle_capacities=v_caps
                )

                current_distance = calculate_total_distance_with_current(current_routes, dist_matrix)
                reward = - current_distance

                if (len(current_routes) < len(last_routes)) or \
                   (current_distance < (last_distance - temperature*np.log(np.random.random()))
                    and len(current_routes) <= len(last_routes)):

                    if (len(current_routes)<len(best_routes_per_instance[idx])) or (current_distance<best_distances_per_instance[idx]):
                        best_distances_per_instance[idx] = current_distance
                        best_routes_per_instance[idx] = copy.deepcopy(current_routes)

                    last_distance = current_distance
                    last_routes   = copy.deepcopy(current_routes)

                temperature *= alpha_T

                ruin_log_probs_all.append(ruin_log_prob)
                ruin_entropies_all.append(ruin_entropy)
                iteration_rewards_all.append(reward)

        ruin_log_probs_tensor   = torch.stack(ruin_log_probs_all)
        ruin_entropies_tensor   = torch.tensor(ruin_entropies_all, dtype=torch.float32, device=device)
        iteration_rewards_tensor= torch.tensor(iteration_rewards_all, dtype=torch.float32, device=device)

        rewards_mean = iteration_rewards_tensor.mean()
        rewards_std  = iteration_rewards_tensor.std() + 1e-8
        iteration_rewards_tensor = (iteration_rewards_tensor - rewards_mean)/rewards_std

        loss_ruin = -(iteration_rewards_tensor * ruin_log_probs_tensor).mean() \
                    - entropy_weight * ruin_entropies_tensor.mean()

        optimizer_ruin.zero_grad()
        loss_ruin.backward()
        optimizer_ruin.step()

        elapsed_time = time.time() - start_time
        mean_best_dist = np.mean(best_distances_per_instance)
        print(f"Epoch {epoch+1}/{n_epochs}, Mean BestDist={mean_best_dist:.6f}, Elapsed={elapsed_time:.6f}s")

        for idx, d in enumerate(all_data):
            print(f"   Instance {d['file_path']}: best_dist={best_distances_per_instance[idx]:.6f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n=== Training done! Total time: {total_time:.6f}s ===")

    save_filename = "ruin_model_final.pth"
    torch.save(ruin_model.state_dict(), save_filename)
    print(f"Model saved to {save_filename}")

    return best_distances_per_instance, best_routes_per_instance

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_path = "./test_data/3v_80c"
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
    file_paths = sorted(file_paths)

    all_data = []
    for fp in file_paths:
        customers = parse_problem_instance(fp)
        d = {
            "customers": customers,
            "file_path": fp,
            "vehicle_capacities": vehicle_capacities
        }
        all_data.append(d)

    if len(all_data) > 1:
        n_cust0 = len(all_data[0]["customers"])
        for d in all_data:
            if len(d["customers"]) != n_cust0:
                raise ValueError("多檔案的客戶數不一致，無法使用同一輸出維度！")

    n_customers = len(all_data[0]["customers"])

    ruin_model = ConvNeXtModel(
        input_channels = 3,
        num_nodes      = n_customers,
        l_t_max        = 10,  
        temperature_sm = 1.0
    ).to(device)

    best_distances, best_routes = train_and_save_model_multiple(
        all_data   = all_data,
        ruin_model = ruin_model,
        device     = device
    )

    for idx, d in enumerate(all_data):
        print(f"[{d['file_path']}] => best_distance={best_distances[idx]} | routes={best_routes[idx]}")
