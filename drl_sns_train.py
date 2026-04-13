import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import glob
import os
from torchvision.models import convnext_tiny


n_epochs = 2000
n_iter = 1        # 每個實例要重複 ruin & recreate 幾次
batch_size = 30  # 每次只處理 n 筆資料
# -------------------------------

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
    def __init__(self, input_channels, num_nodes, output_dim, temperature_sm):
        super(ConvNeXtModel, self).__init__()
        self.model = convnext_tiny(weights=None) 
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, output_dim)
        self.temperature_sm = temperature_sm

    def forward(self, x):
        logits = self.model(x)
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

    absents = [] if in_absents is None else copy.deepcopy(in_absents)
    ruined_t_indices = set([])

    input_tensor = prepare_input_tensor(dist_matrix, data, last_routes, absents, device)
    # batch_size=1
    policy_probs = ruin_model(input_tensor.unsqueeze(0))  # => shape=(1,N)
    policy_probs = policy_probs.squeeze(0)                # => shape=(N,)

    num_actions = policy_probs.size(0)
    action_probs = (1 - epsilon) * policy_probs + epsilon / num_actions
    action_probs = action_probs / action_probs.sum()

    c_seed = torch.multinomial(action_probs, 1).item()
    ruin_log_prob = torch.log(action_probs[c_seed] + 1e-12)
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
    return current_routes, absents, ruin_entropy, ruin_log_prob

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
                if iri == 0:
                    prev_node = 0
                else:
                    prev_node = r[iri - 1]
                if iri == len(r):
                    next_node = 0
                else:
                    next_node = r[iri]

                # 1% 機率跳過
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

    # route_dict = {}

    best_distances_per_instance = []
    best_routes_per_instance = []

    # 計算初始距離
    for idx, d in enumerate(all_data):
        customers = d["customers"]
        dist_matrix = calculate_distance_matrix(customers[:, :2])
        # init_routes = route_dict[idx]
        init_routes = [[i] for i in range(1, len(customers))]
        best_routes = copy.deepcopy(init_routes)
        best_distance = get_routes_distance(dist_matrix, best_routes)
        best_distances_per_instance.append(best_distance)
        best_routes_per_instance.append(best_routes)

    start_time = time.time()

    num_data = len(all_data)
    # 把資料切成批次
    # range(0, num_data, batch_size) => 每次取 batch_size 筆
    for epoch in range(n_epochs):

        epoch_start_time = time.time()
        # batch
        for batch_start in range(0, num_data, batch_size):
            batch_end = min(batch_start + batch_size, num_data)
            sub_data_indices = list(range(batch_start, batch_end))

            ruin_log_probs_batch = []
            ruin_entropies_batch = []
            iteration_rewards_batch = []

            optimizer_ruin.zero_grad()  

            for i_data in sub_data_indices:
                d = all_data[i_data]
                customers = d["customers"]
                file_path = d["file_path"]
                v_caps    = d["vehicle_capacities"]

                dist_matrix = calculate_distance_matrix(customers[:, :2])
                neighbours  = get_neighbours(dist_matrix)

                last_routes   = copy.deepcopy(best_routes_per_instance[i_data])
                last_distance = best_distances_per_instance[i_data]
                temperature   = init_T
                absents       = []

                # 針對該實例跑 n_iter 次
                for _ in range(n_iter):
                    # Ruin
                    current_routes, absents, ruin_entropy, ruin_log_prob = ruin(
                        last_routes  = last_routes,
                        neighbours   = neighbours,
                        data         = customers,
                        ruin_model   = ruin_model,
                        dist_matrix  = dist_matrix,
                        device       = device,
                        epsilon      = get_epsilon(epoch, n_epochs)
                    )

                    # Recreate
                    current_routes = recreate(
                        data             = customers,
                        dist_matrix      = dist_matrix,
                        current_routes   = current_routes,
                        absents          = absents,
                        vehicle_capacities = v_caps
                    )

                    current_distance = calculate_total_distance_with_current(current_routes, dist_matrix)
                    reward = - current_distance

                    # SA
                    if (len(current_routes) < len(last_routes)) or \
                       (current_distance < (last_distance - temperature * np.log(np.random.random()))
                        and len(current_routes) <= len(last_routes)):

                        if (len(current_routes) < len(best_routes_per_instance[i_data])) or \
                           (current_distance < best_distances_per_instance[i_data]):
                            best_distances_per_instance[i_data] = current_distance
                            best_routes_per_instance[i_data]     = copy.deepcopy(current_routes)

                        last_distance = current_distance
                        last_routes   = copy.deepcopy(current_routes)

                    temperature *= alpha_T

                    ruin_log_probs_batch.append(ruin_log_prob)
                    ruin_entropies_batch.append(ruin_entropy)
                    iteration_rewards_batch.append(reward)

            if len(ruin_log_probs_batch) > 0:
                ruin_log_probs_tensor = torch.stack(ruin_log_probs_batch)
                ruin_entropies_tensor = torch.tensor(ruin_entropies_batch, dtype=torch.float32, device=device)
                iteration_rewards_tensor = torch.tensor(iteration_rewards_batch, dtype=torch.float32, device=device)

                rewards_mean = iteration_rewards_tensor.mean()
                rewards_std  = iteration_rewards_tensor.std() + 1e-8
                iteration_rewards_tensor = (iteration_rewards_tensor - rewards_mean) / rewards_std

                loss_ruin = -(iteration_rewards_tensor * ruin_log_probs_tensor).mean() \
                            - entropy_weight * ruin_entropies_tensor.mean()

                loss_ruin.backward()
                optimizer_ruin.step()


        epoch_elapsed_time = time.time() - epoch_start_time
        mean_best_dist = np.mean(best_distances_per_instance)
        print(f"Epoch {epoch+1}/{n_epochs}, Mean BestDist={mean_best_dist:.6f}, "
              f"Elapsed={epoch_elapsed_time:.2f}s in this epoch")

        # 顯示每個檔案當前的 best
        # for idx, d in enumerate(all_data):
        #     print(f"   Instance {d['file_path']}: best_dist={best_distances_per_instance[idx]:.6f}")

        if (epoch + 1) == n_epochs:
            save_filename = f"SNSruin_model_epoch_{epoch+1}.pth"
            torch.save(ruin_model.state_dict(), save_filename)
            print(f"Model saved to {save_filename}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n=== Training done! Total time: {total_time:.2f}s ===")

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
        output_dim     = n_customers,    
        temperature_sm = 1.0
    ).to(device)

    best_distances, best_routes = train_and_save_model_multiple(
        all_data   = all_data,
        ruin_model = ruin_model,
        device     = device
    )

    for idx, d in enumerate(all_data):
        print(f"[{d['file_path']}] => best_distance={best_distances[idx]} | routes={best_routes[idx]}")

