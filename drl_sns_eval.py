import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import glob
import os
import time

from torchvision.models import convnext_tiny

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# SA temperature
init_T = 100.0
final_T = 1.0

# softmax temperature
init_temp  = 2.0
final_temp = 1.0

# Epsilon-greedy
epsilon_start = 1.0
epsilon_final = 0.05

# Entropy regularization
entropy_weight = 0.1

# 車輛容量
vehicle_capacities = [40,35,30, 25, 20]

# 其他主要參數
n_iter = 1         
batch_size = 30    
# -------------------------------


def get_epsilon(current_iter, total_iter):
    return max(epsilon_final, epsilon_start * (1.0 - current_iter / total_iter))

def get_temperature_sm(current_iter, total_iter, init_temp, final_temp):
    decay_rate = (final_temp / init_temp) ** (1.0 / total_iter)
    return init_temp * (decay_rate ** current_iter)

def get_routes_distance(distance_matrix, routes): 
    total_distance = 0
    for route in routes:
        r = [0] + route + [0]
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

    # 選擇一個 seed 客戶 c_seed
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
            # 若加入該 route 會超過容量，則跳過
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


def evaluate_model_multiple(all_data, ruin_model, device):
    ruin_model.eval()
    alpha_T = (final_T / init_T) ** (1.0 / n_iter)

    best_distances_per_instance = []
    best_routes_per_instance = []
    route_dict = {0:
[
  [29, 46, 55, 56, 24, 40],
  [39, 27, 45, 8, 4, 37],
  [15, 54, 48, 47, 30, 26, 38, 52, 14, 12],
  [25, 2, 13, 36, 60],
  [32, 10, 9],
  [1, 42, 28, 22, 20, 50],
  [6, 58, 59, 53, 19],
  [18, 17, 21, 35, 34],
  [31, 57, 16],
  [23, 33, 5],
  [3, 41],
  [49, 7, 11],
  [43, 44, 51]
],

1:
[
  [42, 41, 11, 27, 48, 33, 24],
  [25, 26, 29, 57, 43],
  [28, 34, 55, 54, 9, 16, 58, 22],
  [31, 10, 39, 51, 50, 3],
  [18, 20, 23, 53, 5, 6],
  [30, 2, 15, 21, 45, 36],
  [59, 32, 49, 60, 40, 19],
  [47, 52, 1, 17, 44, 38],
  [13, 35, 12, 8],
  [7, 37, 56],
  [14],
  [4, 46]
],

2:
[
  [52, 26, 19, 40, 35, 5, 31],
  [38, 42, 6, 47, 4, 59, 12, 44],
  [34, 11, 50, 20, 53, 33, 2],
  [54, 51, 10, 1, 41, 7],
  [17, 36, 28, 49, 60],
  [37, 46, 9, 56, 21, 48],
  [24, 30, 13, 39, 55],
  [15, 22, 27, 25],
  [43, 3, 14, 45],
  [18, 58, 23, 8, 57],
  [16, 32, 29]
],

3:
[
  [58, 60, 29, 51, 41, 52, 44, 1],
  [31, 56, 42, 15, 17, 55],
  [32, 54, 37, 47, 7, 39, 23, 49],
  [22, 12, 50, 8, 6, 57, 30, 35],
  [46, 27, 28, 18],
  [11, 16, 25],
  [4, 59, 24, 38, 5],
  [53, 45, 36],
  [21, 14, 48, 34, 2],
  [9, 33, 20, 40, 13],
  [3],
  [43, 19, 26],
  [10]
],

4:
[
  [31, 8, 34, 18, 16, 12, 1, 46],
  [3, 11, 60, 40, 48, 53],
  [42, 38, 25, 59, 26, 57, 54, 39],
  [30, 41, 9],
  [29, 28, 19, 37, 36, 49],
  [50, 13, 17, 10, 35, 21, 56, 6],
  [51, 14, 45, 52, 44, 4, 24],
  [43, 20, 32, 33, 22],
  [27, 15, 55, 47],
  [5, 7],
  [2, 23, 58]
],

5:
[
  [57, 60, 46, 21, 1, 52],
  [59, 22, 20, 28, 8, 15, 3],
  [47, 49, 58, 31, 44, 24],
  [12, 18, 54, 10, 55],
  [41, 39, 48, 19, 23],
  [14, 17, 53, 4, 2, 36, 51],
  [45, 33, 40, 30, 56, 35],
  [6, 37, 43, 32],
  [26, 16, 11],
  [5, 7, 25],
  [34, 38, 42],
  [27],
  [13, 50],
  [29, 9]
],

6:
[
  [43, 15, 25, 46, 16, 3, 17, 21],
  [37, 10, 11, 13, 22, 6, 51],
  [52, 30, 14, 50, 7, 38, 12],
  [31, 35, 47, 9, 4],
  [18, 23, 48, 57],
  [20, 42, 54, 49, 33, 28, 39, 44],
  [55, 24, 45, 34, 8, 41],
  [59, 58],
  [56, 53, 2],
  [27, 40, 60, 32],
  [5, 36],
  [1, 19, 29, 26]
],

7:
[
  [54, 60, 7, 3, 42, 45, 19],
  [32, 1, 8, 24, 35],
  [17, 10, 33, 56, 23, 16],
  [38, 43, 52],
  [53, 41, 6, 12, 13],
  [49, 11, 28, 57, 55, 18, 31, 36],
  [4, 37, 14, 30, 44],
  [48, 29, 25, 59, 27, 5],
  [21, 34, 26, 50],
  [47, 39, 2],
  [9],
  [46],
  [40, 15, 58],
  [51, 22, 20]
],

8:
[
  [12, 35, 23, 58, 29, 53, 9, 16],
  [30, 3, 46, 51, 39, 32],
  [43, 37, 2, 1, 24],
  [40, 49, 60, 13, 27, 6],
  [8, 56, 31, 7, 50, 11],
  [34, 42, 44, 15, 41, 57, 38],
  [36, 26, 54, 20, 28, 45, 33, 10],
  [18, 48, 55, 14, 4, 5],
  [21, 19],
  [17, 52, 25, 47, 22],
  [59]
],

9:
[
  [10, 49, 41, 55, 56, 30, 57, 47],
  [13, 29, 4, 6, 59, 19, 37, 9],
  [39, 40, 51, 11, 33, 42, 34],
  [54, 1, 50, 32, 27, 14],
  [5, 21, 52],
  [45, 15, 46, 58],
  [8, 17, 18],
  [60, 36, 22, 44, 2],
  [12, 25, 26, 31, 16, 3],
  [20, 24, 35],
  [38],
  [7, 48, 53, 43],
  [28, 23]
],

10:
[
  [3, 58, 50, 14, 10, 6, 33, 48],
  [38, 15, 45, 21, 7, 18, 59],
  [28, 29, 54, 34, 25, 46, 5, 12, 43],
  [24, 23, 37, 20, 44],
  [31, 13, 19, 1, 57],
  [4, 22, 26],
  [42, 32, 55, 17, 11, 41, 30, 8, 35],
  [60],
  [39, 40, 49, 36, 53, 47],
  [52, 16, 9, 27],
  [2, 51, 56]
],

11:
[
  [48, 40, 34, 1, 60, 47, 28],
  [15, 44, 59, 39, 58, 10],
  [32, 3, 4, 25, 13, 18],
  [51, 20, 46, 43, 35],
  [5, 23, 9, 17, 19],
  [6, 53, 33, 56, 45],
  [26, 8, 12, 42, 14],
  [36, 50, 21, 22, 55],
  [41, 11, 38, 7, 57, 29, 30],
  [52, 49, 37],
  [2, 54],
  [16],
  [31, 27, 24]
],

12:
[
  [55, 33, 39, 54, 20, 16, 18, 27],
  [19, 30, 6, 35, 34],
  [8, 12, 41, 26, 28, 31, 22, 1, 52],
  [5, 46, 50, 24, 23],
  [43, 4, 58, 3],
  [15, 48, 47, 44, 49],
  [53, 37, 14, 13, 17, 38],
  [9, 51, 36, 45],
  [59, 29, 42, 21],
  [32, 40, 2, 25, 7, 56, 11, 10],
  [57, 60]
],

13:
[
  [29, 39, 46, 8, 7, 13, 35, 44, 12, 26],
  [3, 57, 40, 27, 37, 53, 2],
  [49, 47, 41, 5, 4, 60, 15, 25],
  [51, 43, 18, 17, 52],
  [38, 24, 22],
  [54, 16, 23, 45, 1, 14, 58],
  [48, 42, 56, 21, 55],
  [50, 31, 6, 10, 30],
  [32, 33, 19, 9, 34],
  [59, 20],
  [36, 11, 28]
],

14:
[
  [28, 25, 57, 8, 1, 3, 20, 45],
  [5, 32, 56, 27, 9, 51, 58],
  [26, 46, 7, 15, 31],
  [38, 43, 36, 6, 23, 22, 37, 29],
  [54, 13, 19, 35],
  [33, 10, 14, 59, 47],
  [60, 50, 49, 44, 24, 53, 55],
  [41, 52, 11, 17],
  [40, 16, 48],
  [18, 39, 30, 4, 34, 21],
  [2],
  [42],
  [12]
],

15:
[
  [42, 12, 52, 44, 6, 56, 4, 40, 57, 50],
  [19, 13, 43, 8, 47, 53],
  [25, 9, 10, 32, 18, 27],
  [33, 2, 36, 26, 39],
  [59, 46, 15, 1, 16, 30],
  [7, 60, 55, 14, 35, 20],
  [24, 49, 17, 37, 28, 23],
  [29, 22, 31, 11],
  [34, 58, 5, 45],
  [21, 3, 38, 48],
  [54],
  [41, 51]
],

16:
[
  [58, 50, 14, 54, 32, 30, 10],
  [21, 34, 41, 53, 31, 16, 12, 44],
  [39, 3, 42, 27, 20, 9],
  [46, 18, 28, 2, 35],
  [11, 33, 52, 26],
  [38, 29, 49, 48, 23, 6, 19, 7, 36],
  [15, 1, 8, 24, 43],
  [5, 37, 40, 13, 57, 60],
  [59, 4, 45],
  [56, 47, 51, 22],
  [17, 25],
  [55]
],

17:
[
  [60, 1, 12, 29, 50, 37, 19, 2],
  [46, 4, 47, 14, 18, 30, 40, 39],
  [24, 16, 5, 54, 51, 45],
  [52, 58, 44, 28, 7, 9],
  [41, 17],
  [21, 33, 53, 34, 11],
  [13, 22, 55, 3, 23, 26],
  [59, 15, 38, 49],
  [32, 57, 36, 42],
  [27, 35, 43, 8, 31, 20],
  [48],
  [56],
  [6, 10, 25]
],

18:
[
  [54, 28, 9, 60, 53, 27],
  [16, 6, 19, 5, 48, 14, 26, 1],
  [30, 58, 40, 12, 29],
  [56, 37, 47, 11, 18, 32],
  [45, 10, 34, 13],
  [24, 3, 39, 7, 17, 42, 52],
  [38, 49, 15, 2, 35],
  [8],
  [22, 23, 4, 33],
  [31, 25, 46, 43],
  [44, 55, 57, 36],
  [50, 20],
  [41],
  [59, 51],
  [21]
],

19:
[
  [56, 10, 11, 42, 13, 6],
  [24, 57, 4, 16, 38, 34],
  [50, 46, 51, 5, 28, 7],
  [12, 47, 21, 44, 39],
  [49, 33, 31],
  [14, 26, 37, 20, 60, 32, 41],
  [27, 3, 22, 9, 36, 2, 54],
  [40, 19, 35, 30, 48],
  [1, 43, 55, 17, 18],
  [59, 58, 8, 45, 15],
  [29, 53],
  [25],
  [23, 52]
],

20:
[
  [54, 38, 11, 21, 5, 50, 55, 43, 46, 31],
  [40, 25, 28, 20, 60, 4, 3],
  [51, 35, 14, 57, 8, 13, 2, 9, 58],
  [29, 56, 33, 41],
  [44, 34, 45],
  [22, 49, 6, 59, 27, 47],
  [16, 10, 36, 48, 15, 17],
  [32, 1, 30, 19],
  [52, 42, 12, 7, 24],
  [53, 39, 26, 37],
  [23, 18]
],

21:
[
  [12, 36, 17, 40, 3, 60, 21, 53, 39],
  [1, 27, 2, 52, 9, 22, 24],
  [47, 19, 41, 45, 30, 23],
  [55, 26, 28, 18, 35],
  [16, 4, 29, 8],
  [7, 5, 59, 54, 38, 51, 15],
  [10, 49, 37, 57, 11, 42],
  [14, 50, 31, 25, 43, 32],
  [34, 20, 44, 13],
  [46, 33, 48],
  [6],
  [58, 56]
],

22:
[
  [36, 4, 49, 26, 46],
  [53, 35, 28, 10, 11],
  [15, 13, 34, 20, 16, 25, 30, 7, 59, 39, 47],
  [52, 48, 22, 19, 2, 38, 50, 55],
  [29, 24, 57, 5],
  [31, 41, 45, 6],
  [32, 21, 3, 14, 27, 17],
  [8, 42],
  [1, 58, 54, 56],
  [18, 43, 33, 44],
  [9, 51],
  [12, 40, 37],
  [60],
  [23]
],

23:
[
  [14, 10, 31, 53, 51, 11, 15, 22, 26],
  [55, 54, 57, 23, 48, 46, 25],
  [42, 29, 12, 52, 17, 32, 24],
  [36, 37, 30, 20, 45, 13],
  [35, 18, 9, 34],
  [41, 19, 38],
  [4, 33, 7, 50, 43, 8],
  [2, 39, 60, 44, 47],
  [28, 49, 40, 59, 5, 16],
  [1, 3, 58, 6],
  [27, 56, 21]
],

24:
[
  [14, 46, 4, 48, 18, 44, 51, 37],
  [32, 52, 59, 17, 55, 19, 1],
  [47, 54, 8, 33, 23, 53],
  [50, 26, 7, 15, 27, 20],
  [3, 11, 57],
  [16, 24, 45, 6, 49, 12],
  [5, 40, 60, 21, 10],
  [13, 56, 43, 31, 42],
  [38, 39, 35, 30, 9],
  [2, 29, 36, 22],
  [41],
  [58],
  [28, 34],
  [25]
],

25:
[
  [46, 4, 20, 16, 18, 59, 39],
  [8, 53, 6, 60, 36, 40, 41, 9],
  [23, 43, 21, 25, 58, 55, 22, 47, 45, 29, 49],
  [52, 48, 51],
  [38, 42, 33, 3],
  [13, 35, 27, 56, 7, 54, 26, 10],
  [2, 19, 5, 34, 32],
  [24, 17, 50],
  [11, 28, 30],
  [14, 57, 31, 1, 37, 15],
  [44],
  [12]
],

26:
[
  [1, 30, 3, 11, 47, 48],
  [4, 52, 56, 39, 16],
  [28, 45, 44, 50, 13, 55],
  [24, 57, 12, 54, 15],
  [17, 60, 32, 25],
  [10, 26, 27, 43, 37, 2],
  [46, 31, 21, 36, 20],
  [53, 5, 35, 22, 18, 23],
  [59, 6, 40, 29, 58],
  [38, 14, 42],
  [34, 19, 7],
  [41, 51, 9, 33, 49],
  [8]
],

27:
[
  [17, 18, 33, 6, 53, 51, 1, 28, 36],
  [37, 32, 47, 55, 11, 48],
  [15, 21, 57, 49, 26, 59, 35],
  [42, 14, 60, 10, 4],
  [34, 40, 9, 29],
  [31, 52, 13, 2],
  [39, 24, 3, 44, 45, 38],
  [25, 12, 56, 30, 50],
  [46, 54, 5, 16],
  [41, 8, 19, 43, 27],
  [7],
  [20],
  [58, 22, 23]
],

28:
[
  [45, 18, 38, 14, 34, 47, 44, 39, 23],
  [19, 8, 17, 25, 1, 24, 30, 52],
  [57, 58, 20, 5, 51, 6, 13, 53, 32, 3, 9, 28],
  [48, 56, 21, 37, 15],
  [40, 50, 35, 26, 60, 42],
  [33, 49, 4, 10, 12, 29, 36],
  [7, 41, 22, 2, 54],
  [11, 16, 27],
  [55, 59, 43, 46, 31]
],

29:
[
  [18, 9, 42, 35, 20, 4, 12, 40],
  [15, 39, 8, 1, 17, 43, 25],
  [36, 30, 27, 58, 56, 47],
  [49, 32, 45, 7, 59, 29],
  [23, 21, 2],
  [33, 53, 6, 38, 60, 13],
  [50, 57, 51, 24, 19, 26],
  [22, 34, 28, 55, 11, 31, 44],
  [5, 48, 16, 14],
  [10, 54, 46],
  [37, 52],
  [3],
  [41]
]


}
    for idx, d in enumerate(all_data):
        customers = d["customers"]
        dist_matrix = calculate_distance_matrix(customers[:, :2])
        init_routes = route_dict[idx]
        # init_routes = [[i] for i in range(1, len(customers))]
        best_routes = copy.deepcopy(init_routes)
        best_distance = get_routes_distance(dist_matrix, best_routes)
        best_distances_per_instance.append(best_distance)
        best_routes_per_instance.append(best_routes)


    num_data = len(all_data)

    start_time = time.time()

    for eval_epoch in range(9000):
        epoch_start_time = time.time()


        for batch_start in range(0, num_data, batch_size):
            batch_end = min(batch_start + batch_size, num_data)
            sub_data_indices = list(range(batch_start, batch_end))


            for i_data in sub_data_indices:
                d = all_data[i_data]
                customers = d["customers"]
                v_caps    = d["vehicle_capacities"]
                dist_matrix = calculate_distance_matrix(customers[:, :2])
                neighbours  = get_neighbours(dist_matrix)

                last_routes   = copy.deepcopy(best_routes_per_instance[i_data])
                last_distance = best_distances_per_instance[i_data]
                temperature   = init_T
                absents       = []

                for _ in range(n_iter):
                    # Ruin
                    current_routes, absents, ruin_entropy, ruin_log_prob = ruin(
                        last_routes  = last_routes,
                        neighbours   = neighbours,
                        data         = customers,
                        ruin_model   = ruin_model,
                        dist_matrix  = dist_matrix,
                        device       = device,
                        epsilon      = get_epsilon(eval_epoch, 1.0) # 這裡 total_iter=1.0, 也可直接給 0
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

                    if (len(current_routes) < len(last_routes)) or \
                       (current_distance < (last_distance - temperature * np.log(np.random.random()))
                        and len(current_routes) <= len(last_routes)):

                        if (len(current_routes) < len(best_routes_per_instance[i_data])) or \
                           (current_distance < best_distances_per_instance[i_data]):
                            best_distances_per_instance[i_data] = current_distance
                            best_routes_per_instance[i_data]    = copy.deepcopy(current_routes)

                        last_distance = current_distance
                        last_routes   = copy.deepcopy(current_routes)

                    temperature *= alpha_T

        epoch_elapsed_time = time.time() - epoch_start_time
        mean_best_dist = np.mean(best_distances_per_instance)
        print(f"[Eval] Epoch {eval_epoch+1}, Mean BestDist={mean_best_dist:.6f}, Elapsed={epoch_elapsed_time:.2f}s")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"=== Evaluation done! Total time: {total_time:.2f}s ===")

    return best_distances_per_instance, best_routes_per_instance


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 準備評估檔案
    folder_path = "./test_data/5v_60c"
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
    file_paths = sorted(file_paths)

    # 讀取實例資料
    all_data = []
    for fp in file_paths:
        customers = parse_problem_instance(fp)
        d = {
            "customers": customers,
            "file_path": fp,
            "vehicle_capacities": vehicle_capacities
        }
        all_data.append(d)

    # 檢查
    n_customers = len(all_data[0]["customers"])
    for d in all_data:
        if len(d["customers"]) != n_customers:
            raise ValueError(f"客戶數不一致: {d['file_path']} "
                             f"與第一個檔案({n_customers} 客戶)不同。")

    # 載入訓練權重
    ruin_model = ConvNeXtModel(
        input_channels = 3,
        num_nodes      = n_customers,   
        output_dim     = n_customers,   
        temperature_sm = 2.0            
    ).to(device)


    saved_model_path = "SNSruin_model_epoch_2000_60c.pth"
    ruin_model.load_state_dict(torch.load(saved_model_path, map_location=device))


    best_distances, best_routes = evaluate_model_multiple(all_data, ruin_model, device)
