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
n_iter   = 1

init_T   = 100.0
final_T  = 1.0

init_temp   = 2.0
final_temp  = 1.0

epsilon_start = 1.0
epsilon_final = 0.01

entropy_weight = 0.1
vehicle_capacities = [30, 25, 20]

def get_epsilon(epoch, n_epochs):
    return max(epsilon_final, epsilon_start*(1.0 - epoch/n_epochs))

def get_temperature_sm(epoch, n_epochs, init_temp, final_temp):
    decay_rate = (final_temp / init_temp)**(1.0/n_epochs)
    return init_temp*(decay_rate**epoch)

def compute_entropy(probs):
    if probs.dim() == 2:
        probs = probs.squeeze(0)
    ent = -(probs * torch.log(probs + 1e-12)).sum()
    return ent

def calculate_distance_matrix(coords):
    num_nodes = len(coords)
    dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist_matrix[i,j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix

def get_routes_distance(dist_matrix, routes):
    total_distance = 0
    for route in routes:
        if not route: 
            continue
        r = [0] + route + [0]
        for i in range(len(r)-1):
            total_distance += dist_matrix[r[i], r[i+1]]
    return total_distance

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

def remove_one_node_from_route(routes, node_id):
    for r in routes:
        if node_id in r:
            r.remove(node_id)
    routes = [r for r in routes if len(r) > 0]
    return routes


class RuinModel(nn.Module):
    def __init__(self, input_channels, num_nodes, l_t_max=10, temperature_sm=1.0):
        super(RuinModel, self).__init__()
        self.num_nodes = num_nodes
        self.l_t_max   = l_t_max

        base_model = convnext_tiny(weights=None)
        in_features = base_model.classifier[2].in_features
        base_model.classifier[2] = nn.Linear(in_features, num_nodes + l_t_max)
        self.model = base_model
        self.temperature_sm = temperature_sm

    def forward(self, x):
        logits = self.model(x)  # shape=(batch_size, num_nodes + l_t_max)
        return F.softmax(logits / self.temperature_sm, dim=-1)


class RecreateModel(nn.Module):
    def __init__(self, input_channels, num_nodes, temperature_sm=1.0):
        super(RecreateModel, self).__init__()
        self.num_nodes = num_nodes

        base_model = convnext_tiny(weights=None)
        old_conv = base_model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )
        base_model.features[0][0] = new_conv
        in_features = base_model.classifier[2].in_features
        base_model.classifier[2] = nn.Linear(in_features, num_nodes)
        self.model = base_model
        self.temperature_sm = temperature_sm

    def forward(self, x):
        logits = self.model(x)  # shape=(batch_size, num_nodes)
        return F.softmax(logits/self.temperature_sm, dim=-1)


def prepare_input_tensor_ruin(dist_matrix, data, current_routes, absents, device):
    num_nodes = len(data)
    input_channels = 3
    input_tensor = np.zeros((input_channels, num_nodes, num_nodes), dtype=np.float32)

    # channel 0 => dist
    input_tensor[0] = dist_matrix

    # channel 1 => demand row broadcast
    demands = np.array([d[2] for d in data], dtype=np.float32)
    for i in range(num_nodes):
        input_tensor[1, i, :] = demands[i]

    # channel 2 => route assignment
    route_flags = np.full(num_nodes, -1, dtype=np.float32)
    for r_idx, route in enumerate(current_routes):
        for node_id in route:
            route_flags[node_id] = r_idx
    for i in range(num_nodes):
        input_tensor[2, i, :] = route_flags[i]

    return torch.tensor(input_tensor, dtype=torch.float32, device=device)

def prepare_input_tensor_recreate(dist_matrix, data, current_routes, absents, device):
    num_nodes = len(data)
    input_channels = 4
    input_tensor = np.zeros((input_channels, num_nodes, num_nodes), dtype=np.float32)

    # channel 0 => distance
    input_tensor[0] = dist_matrix

    # channel 1 => demand
    demands = np.array([d[2] for d in data], dtype=np.float32)
    for i in range(num_nodes):
        input_tensor[1, i, :] = demands[i]

    # channel 2 => route assignment (-1=absent)
    route_flags = np.full(num_nodes, -1, dtype=np.float32)
    # 計算路徑容量使用 => route_usage[rIdx]
    route_usage = {}
    for r_idx, route in enumerate(current_routes):
        used = sum(data[n][2] for n in route)  # sum demand
        route_usage[r_idx] = used
        for node_id in route:
            route_flags[node_id] = r_idx
    for i in range(num_nodes):
        input_tensor[2, i, :] = route_flags[i]

    # channel 3 => capacity usage or remaining capacity
    # 先計算 vehicle容量 => 
    # e.g. route r => used=route_usage[r], remain= vehicle_capacities[r%len(vehicle_capacities)] - used
    route_remain = {}
    for r_idx, used_val in route_usage.items():
        cap = vehicle_capacities[r_idx % len(vehicle_capacities)]
        remain = cap - used_val
        route_remain[r_idx] = remain

    for i in range(num_nodes):
        r_id = route_flags[i]
        if r_id < 0:
            # absent => fill -1
            for j in range(num_nodes):
                input_tensor[3, i, j] = -1
        else:
            remain_cap = route_remain[r_id]
            for j in range(num_nodes):
                input_tensor[3, i, j] = remain_cap

    return torch.tensor(input_tensor, dtype=torch.float32, device=device)


def ruin(last_routes, data, ruin_model, dist_matrix, device, epsilon, in_absents=None):
    absents = [] if in_absents is None else copy.deepcopy(in_absents)
    
    # 1) forward => 取整體輸出 => front=(num_nodes), back=(l_t_max)
    input_tensor = prepare_input_tensor_ruin(dist_matrix, data, last_routes, absents, device)
    full_probs = ruin_model(input_tensor.unsqueeze(0)).squeeze(0)  # shape=(num_nodes + l_t_max,)

    num_nodes = len(data)
    l_t_max   = 10
    
    # 分割出 node_probs / l_t_probs
    node_probs = full_probs[:num_nodes]
    l_t_probs  = full_probs[num_nodes:]

    # 2) sample l_t
    l_t_probs = l_t_probs / l_t_probs.sum()  # normalize
    l_t_idx = torch.multinomial(l_t_probs, 1).item()  # in [0..(l_t_max-1)]
    l_t = l_t_idx + 1  # => [1..l_t_max]

    ruin_log_prob = torch.log(l_t_probs[l_t_idx] + 1e-12)
    ruin_entropy  = compute_entropy(l_t_probs)

    # 3) 依照 epsilon - mix on node_probs

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
            node_probs[chosen_node] = torch.tensor(0.0, dtype=torch.float32, device=device)
            node_probs = node_probs / (node_probs.sum() + 1e-12)

        if len(last_routes)==0:
            break

    return last_routes, absents, ruin_entropy, ruin_log_prob

def recreate_drl(current_routes, absents, data, dist_matrix, device,
                 recreate_model, epsilon, vehicle_capacities):
    recreate_log_prob = torch.tensor(0.0, dtype=torch.float32, device=device)
    recreate_entropy  = torch.tensor(0.0, dtype=torch.float32, device=device)

    remain_absents = copy.deepcopy(absents)
    inserted_order = []
    while len(remain_absents)>0:
        # 準備 input
        input_tensor = prepare_input_tensor_recreate(dist_matrix, data, current_routes, remain_absents, device)
        # forward => shape=(num_nodes,)
        logits = recreate_model(input_tensor.unsqueeze(0)).squeeze(0)
        
        # 1) 建立副本，避免對 logits in-place 修改
        masked_logits = logits.clone()

        # 2) 對不在 remain_absents 的節點 => 設為 -1e9 (mask)
        for idx_node in range(len(masked_logits)):
            if idx_node not in remain_absents:
                masked_logits[idx_node] = -1e9

        # 3) softmax
        masked_probs = F.softmax(masked_logits, dim=-1)

        # 4) 直接抽樣
        chosen_node = torch.multinomial(masked_probs, 1).item()

        # 5) log_prob & entropy
        logp = torch.log(masked_probs[chosen_node] + 1e-12)
        ent  = - (masked_probs * torch.log(masked_probs + 1e-12)).sum()

        recreate_log_prob += logp
        recreate_entropy  += ent

        # 貪婪插入 => chosen_node => find best route+pos => update current_routes
        current_routes = greedy_insert_one_node(current_routes, chosen_node, data, dist_matrix, vehicle_capacities)
        
        # 移除 chosen_node 出 remain_absents
        remain_absents.remove(chosen_node)
        inserted_order.append(chosen_node)

    return current_routes, recreate_log_prob, recreate_entropy


def greedy_insert_one_node(current_routes, node_id, data, dist_matrix, vehicle_capacities):
    probable_place = []
    demand_node = data[node_id][2]
    for ir, route in enumerate(current_routes):
        cap = vehicle_capacities[ir % len(vehicle_capacities)]
        used = sum(data[n][2] for n in route)
        if used + demand_node > cap:
            continue
        for pos in range(len(route)+1):
            prev_n = route[pos-1] if pos>0 else 0
            next_n = route[pos] if pos<len(route) else 0
            dcost = dist_matrix[prev_n, node_id] + dist_matrix[node_id, next_n] - dist_matrix[prev_n, next_n]
            probable_place.append((ir, pos, dcost))
    
    if len(probable_place)==0:
        # 無路可插 => 開新路徑
        current_routes.append([node_id])
    else:
        best = sorted(probable_place, key=lambda x:x[-1])[0]
        ir, pos, _ = best
        route = current_routes[ir]
        current_routes[ir] = route[:pos] + [node_id] + route[pos:]
    return current_routes


def train_and_save_model_multiple(all_data, ruin_model, recreate_model, device):
    optimizer_ruin = torch.optim.Adam(ruin_model.parameters(), lr=0.0001)
    optimizer_recreate = torch.optim.Adam(recreate_model.parameters(), lr=0.0001)

    best_distances_per_instance = []
    best_routes_per_instance = []

    # 初始化 => 每個instance先給最簡單路徑
    for d in all_data:
        customers = d["customers"]
        dist_matrix = calculate_distance_matrix(customers[:, :2])
        init_routes = [[i] for i in range(1, len(customers))]
        dist_init = get_routes_distance(dist_matrix, init_routes)
        best_distances_per_instance.append(dist_init)
        best_routes_per_instance.append(init_routes)

    start_time = time.time()
    for epoch in range(n_epochs):
        ruin_log_probs_all = []
        ruin_entropies_all = []
        recreate_log_probs_all = []
        recreate_entropies_all = []
        iteration_rewards_all = []

        for idx, d in enumerate(all_data):
            customers = d["customers"]
            dist_matrix = calculate_distance_matrix(customers[:, :2])
            last_routes   = copy.deepcopy(best_routes_per_instance[idx])
            last_distance = best_distances_per_instance[idx]

            for it in range(n_iter):
                ruin_routes, absents, ruin_entropy, ruin_log_prob = ruin(
                    last_routes=copy.deepcopy(last_routes),
                    data=customers,
                    ruin_model=ruin_model,
                    dist_matrix=dist_matrix,
                    device=device,
                    epsilon=get_epsilon(epoch, n_epochs),
                )

                final_routes, rec_log_prob, rec_entropy = recreate_drl(
                    current_routes=ruin_routes,
                    absents=absents,
                    data=customers,
                    dist_matrix=dist_matrix,
                    device=device,
                    recreate_model=recreate_model,
                    epsilon=get_epsilon(epoch, n_epochs),
                    vehicle_capacities=d["vehicle_capacities"]
                )

                current_distance = get_routes_distance(dist_matrix, final_routes)
                reward = - current_distance

                # update best
                if current_distance<last_distance:
                    last_distance = current_distance
                    last_routes   = copy.deepcopy(final_routes)
                    if current_distance<best_distances_per_instance[idx]:
                        best_distances_per_instance[idx]=current_distance
                        best_routes_per_instance[idx]   =copy.deepcopy(final_routes)

                ruin_log_probs_all.append(ruin_log_prob)
                ruin_entropies_all.append(ruin_entropy)
                recreate_log_probs_all.append(rec_log_prob)
                recreate_entropies_all.append(rec_entropy)
                iteration_rewards_all.append(reward)

                # temperature = ...

        # end for each instance
        # policy gradient update
        ruin_log_probs_tensor     = torch.stack(ruin_log_probs_all)
        ruin_entropies_tensor     = torch.tensor(ruin_entropies_all, dtype=torch.float32, device=device)
        recreate_log_probs_tensor = torch.stack(recreate_log_probs_all)
        recreate_entropies_tensor = torch.tensor(recreate_entropies_all, dtype=torch.float32, device=device)

        iteration_rewards_tensor  = torch.tensor(iteration_rewards_all, dtype=torch.float32, device=device)
        r_mean = iteration_rewards_tensor.mean()
        r_std  = iteration_rewards_tensor.std()+1e-8
        iteration_rewards_tensor  = (iteration_rewards_tensor - r_mean)/r_std

        # ruin loss
        loss_ruin = - (iteration_rewards_tensor*ruin_log_probs_tensor).mean() \
                    - entropy_weight*ruin_entropies_tensor.mean()
        # recreate loss
        loss_recreate = - (iteration_rewards_tensor*recreate_log_probs_tensor).mean() \
                        - entropy_weight*recreate_entropies_tensor.mean()

        optimizer_ruin.zero_grad()
        optimizer_recreate.zero_grad()
        loss_ruin.backward()
        loss_recreate.backward()
        optimizer_ruin.step()
        optimizer_recreate.step()

        elapsed_time = time.time()-start_time
        mean_best_dist = np.mean(best_distances_per_instance)
        print(f"Epoch {epoch+1}/{n_epochs}, bestDist={mean_best_dist:.6f}, Elapsed={elapsed_time:.3f}s")

    # end training
    end_time = time.time()
    total_time = end_time-start_time
    print(f"Training done! time={total_time:.3f}s")
    # save model m,
    torch.save(ruin_model.state_dict(), "ruin_model_final.pth")
    torch.save(recreate_model.state_dict(), "recreate_model_final.pth")

    return best_distances_per_instance, best_routes_per_instance


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_path = "./test_data/3v_40c"
    file_paths  = glob.glob(os.path.join(folder_path,"*.txt"))
    file_paths  = sorted(file_paths)

    all_data=[]
    for fp in file_paths:
        customers = parse_problem_instance(fp)
        d={
            "customers": customers, 
            "file_path":fp,
            "vehicle_capacities": vehicle_capacities
        }
        all_data.append(d)

    n_customers = len(all_data[0]["customers"])

    # build ruinModel, recreateModel
    ruin_model = RuinModel(
        input_channels=3,
        num_nodes=n_customers,
        l_t_max=10,
        temperature_sm=1.0
    ).to(device)

    recreate_model = RecreateModel(
        input_channels=4,  # distance,demand,routeAssign,capacity
        num_nodes=n_customers,
        temperature_sm=1.0
    ).to(device)

    # start train
    best_distances, best_routes = train_and_save_model_multiple(
        all_data=all_data,
        ruin_model=ruin_model,
        recreate_model=recreate_model,
        device=device
    )

    # print result
    for idx, d in enumerate(all_data):
        print(f"[{d['file_path']}] => best_distance={best_distances[idx]} | routes={best_routes[idx]}")
