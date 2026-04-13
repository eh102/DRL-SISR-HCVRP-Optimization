import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import glob
import os
from torchvision.models import convnext_tiny

#40c = 2000
#60c = 3000
#80c = 4000
n_iter = 2000
epsilon_start = 1.0
epsilon_final = 0.01
vehicle_capacities = [30, 25, 20]

def get_epsilon(i, n_iter):
    return max(epsilon_final, epsilon_start*(1.0 - i / n_iter))

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
        # override 第一層 conv => in_channels = input_channels
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

    # 0 => dist
    input_tensor[0] = dist_matrix

    # 1 => demand
    demands = np.array([d[2] for d in data], dtype=np.float32)
    for i in range(num_nodes):
        input_tensor[1, i, :] = demands[i]

    # 2 => route assignment
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

    # 0 => dist
    input_tensor[0] = dist_matrix

    # 1 => demand
    demands = np.array([d[2] for d in data], dtype=np.float32)
    for i in range(num_nodes):
        input_tensor[1, i, :] = demands[i]

    # 2 => route assignment
    route_flags = np.full(num_nodes, -1, dtype=np.float32)
    route_usage = {}
    for r_idx, route in enumerate(current_routes):
        used = sum(data[n][2] for n in route)
        route_usage[r_idx] = used
        for node_id in route:
            route_flags[node_id] = r_idx
    for i in range(num_nodes):
        input_tensor[2, i, :] = route_flags[i]

    # 3 => capacity usage
    route_remain = {}
    for r_idx, used_val in route_usage.items():
        cap = vehicle_capacities[r_idx % len(vehicle_capacities)]
        remain = cap - used_val
        route_remain[r_idx] = remain

    for i in range(num_nodes):
        r_id = route_flags[i]
        if r_id < 0:
            for j in range(num_nodes):
                input_tensor[3, i, j] = -1
        else:
            remain_cap = route_remain[r_id]
            for j in range(num_nodes):
                input_tensor[3, i, j] = remain_cap

    return torch.tensor(input_tensor, dtype=torch.float32, device=device)

def ruin(last_routes, data, ruin_model, dist_matrix, device, epsilon, in_absents=None):
    absents = [] if in_absents is None else copy.deepcopy(in_absents)
    
    # 1) forward => (num_nodes + l_t_max) => [node_probs, l_t_probs]
    input_tensor = prepare_input_tensor_ruin(dist_matrix, data, last_routes, absents, device)
    full_probs = ruin_model(input_tensor.unsqueeze(0)).squeeze(0)  # shape=(num_nodes + l_t_max,)

    num_nodes = len(data)
    l_t_max   = 10

    node_probs = full_probs[:num_nodes]
    l_t_probs  = full_probs[num_nodes:]

    # sample l_t
    l_t_probs /= l_t_probs.sum()
    l_t_idx = torch.multinomial(l_t_probs, 1).item()
    l_t = l_t_idx + 1
    
    node_probs /= (node_probs.sum() + 1e-12)
    
    def sample_node(node_probs):
        num_actions = node_probs.size(0)
        node_probs = (1-epsilon)*node_probs + epsilon/num_actions
        node_probs = node_probs / node_probs.sum()
        chosen = torch.multinomial(node_probs, 1).item()
        return chosen

    for _ in range(l_t):
        if len(last_routes)==0:
            break
        chosen_node = sample_node(node_probs)
        if (chosen_node not in absents) and (chosen_node != 0):
            absents.append(chosen_node)
            last_routes = remove_one_node_from_route(last_routes, chosen_node)
            node_probs[chosen_node] = torch.tensor(0.0, dtype=torch.float32, device=device)
            node_probs = node_probs / (node_probs.sum() + 1e-12)
        if len(last_routes)==0:
            break
    return last_routes, absents

def recreate_drl(current_routes, absents, data, dist_matrix, device,
                 recreate_model, epsilon, vehicle_capacities):
    remain_absents = copy.deepcopy(absents)
    while remain_absents:
        # forward => masked_probs => sample node => insert
        input_tensor = prepare_input_tensor_recreate(dist_matrix, data, current_routes, remain_absents, device)
        logits = recreate_model(input_tensor.unsqueeze(0)).squeeze(0)

        masked_logits = logits.clone()
        for idx_node in range(len(masked_logits)):
            if idx_node not in remain_absents:
                masked_logits[idx_node] = -1e9
        masked_probs = F.softmax(masked_logits, dim=-1)
        
        chosen_node = torch.multinomial(masked_probs, 1).item()

        # greedy insert => chosen_node
        current_routes = greedy_insert_one_node(current_routes, chosen_node, data, dist_matrix, vehicle_capacities)

        remain_absents.remove(chosen_node)

    return current_routes

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
        current_routes.append([node_id])
    else:
        best = sorted(probable_place, key=lambda x:x[-1])[0]
        ir, pos, _ = best
        current_routes[ir] = current_routes[ir][:pos] + [node_id] + current_routes[ir][pos:]
    return current_routes

def evaluate_model_multiple(all_data, ruin_model, recreate_model, device):
    best_distances_per_instance = []
    best_routes_per_instance = []

    for d in all_data:
        customers = d["customers"]
        dist_matrix = calculate_distance_matrix(customers[:, :2])
        # 初始解 => trivial
        current_routes = [[i] for i in range(1, len(customers))]
        best_distance = get_routes_distance(dist_matrix, current_routes)
        best_routes   = copy.deepcopy(current_routes)
        # ruin + recreate => n_iter 次 or 1次
        for i in range(n_iter):

            # ruin
            ruin_routes, absents = ruin(
                last_routes=copy.deepcopy(current_routes),
                data=customers,
                ruin_model=ruin_model,
                dist_matrix=dist_matrix,
                device=device,
                epsilon=get_epsilon(i, n_iter)  # or some default
            )
            # recreate
            final_routes = recreate_drl(
                current_routes=ruin_routes,
                absents=absents,
                data=customers,
                dist_matrix=dist_matrix,
                device=device,
                recreate_model=recreate_model,
                epsilon=0.0,
                vehicle_capacities=d["vehicle_capacities"]
            )
            dist_eval = get_routes_distance(dist_matrix, final_routes)
            if dist_eval < best_distance:
                best_distance = dist_eval
                best_routes   = copy.deepcopy(final_routes)
            current_routes = copy.deepcopy(best_routes)
            print("i =",i,"best_distance =",best_distance)

        best_distances_per_instance.append(best_distance)
        best_routes_per_instance.append(best_routes)

    return best_distances_per_instance, best_routes_per_instance

if __name__=="__main__":
    start_time = time.time()
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

    ruin_model = RuinModel(
        input_channels=3,
        num_nodes=n_customers,
        l_t_max=10,
        temperature_sm=1.0
    ).to(device)

    recreate_model = RecreateModel(
        input_channels=4,
        num_nodes=n_customers,
        temperature_sm=1.0
    ).to(device)

    ruin_model.load_state_dict(torch.load("ruin_model_final.pth", map_location=device))
    recreate_model.load_state_dict(torch.load("recreate_model_final.pth", map_location=device))
    ruin_model.eval()
    recreate_model.eval()

    with torch.no_grad():
        best_distances, best_routes = evaluate_model_multiple(
            all_data=all_data,
            ruin_model=ruin_model,
            recreate_model=recreate_model,
            device=device
        )

    for idx, d in enumerate(all_data):
        print(f"File: {d['file_path']}")
        print(f"   Best Distance: {best_distances[idx]:.3f}")
        print(f"   Routes: {best_routes[idx]}")
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)