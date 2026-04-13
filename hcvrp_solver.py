import copy
import numpy as np
import time

def calculate_distance_matrix(coords):
    distance_matrix = np.zeros([len(coords),len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5
    return distance_matrix

def sisr_hcvrp(data,
              vehicle_capacities,
              n_iter,
              init_T,
              final_T,              
              blink_rate,
              n_iter_fleet=None,
              fleet_gap=None,
              c_bar=10.0,
              L_max=10.0,
              m_alpha=0.01,
              obj_n_routes=None,
              init_route=None,
              verbose_step=None,
              test_obj=None
              ):
    """
    data: An numpy array with a shape of (N, 3), where N is the number of customers.
          Each column represents x coord, y coord, demand, ready time, due time, and
          service time.
    """
    n_customers = len(data)
    n_vehicles = len(vehicle_capacities)
    vehicle_assignments = [vehicle_capacities[i % n_vehicles] for i in range(n_customers)]

    def get_route_distance(distance_matrix, route):
        r = [0]+route+[0]
        result = np.sum([distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)])
        return result
    
    def get_routes_distance(distance_matrix, routes): 
        total_distance = 0
        for route in routes:
            r = [0]+route+[0]
            total_distance += np.sum([distance_matrix[r[i],r[i+1]] for i in range(len(r)-1)])
        return total_distance
    
    def get_neighbours(distance_matrix):
        n_vertices = distance_matrix.shape[0]
        neighbours = []
        for i in range(n_vertices):
            index_dist = [(j, distance_matrix[i][j]) for j in range(n_vertices)]
            sorted_index_dist = sorted(index_dist, key=lambda x: x[1])
            neighbours.append([x[0] for x in sorted_index_dist])
        return neighbours
    
    def ruin(last_routes, neighbours, in_absents=None, isHugeRuin=False):

        def remove_nodes(tr, l_t, c, m): # tr:target routes / l_t:adjacent nodes of length / c:seed node(customer) / m:extend the string seletion scope
            def string_removal(tr, l_t, c):
                i_c = tr.index(c)
                range1 = max(0, i_c+1-l_t)
                range2 = min(i_c, len(tr)-l_t)+1
                start = np.random.randint(range1, range2)
                return tr[start:start+l_t]
            def split_removal(tr, l_t, c, m):
                additional_l = min(m, len(tr)-l_t)
                l_t_m = l_t+additional_l
                i_c = tr.index(c)
                range1 = max(0, i_c+1-l_t_m)
                range2 = min(i_c, len(tr)-l_t_m)+1
                start = np.random.randint(range1, range2)
                potential_removal = tr[start:start+l_t_m]
                return [potential_removal[i] for i in np.random.choice(l_t_m, l_t, replace=False)]
            if np.random.random()<0.5:
                newly_removed = string_removal(tr, l_t, c)
            else:
                newly_removed = split_removal(tr, l_t, c, m)
                if m<(len(tr)-l_t) or np.random.random()>m_alpha:
                    m+=1
            return m, newly_removed
        
        def find_t(last_routes, c):
            for i in range(len(last_routes)):
                if c in last_routes[i]: return i
            return None
        
        def routes_summary(last_routes, absents):
            current_routes = []
            for r in last_routes:
                new_r = [x for x in r if x not in absents]
                if len(new_r)>0:
                    current_routes.append(new_r)
            return current_routes
        
        m = 1
        l_s_max = min(L_max, np.mean([len(x) for x in last_routes])) # the Global length of the string removed at a time a.k.a max number of adjacent removed nodes
        k_s_max = 4.0*c_bar/(1.0+l_s_max)-1.0 # max number of removing that can be made
        k_s = int(np.random.random()*k_s_max+1.0) # Actual number of removing
        c_seed = int(np.random.random()*len(data))
        if in_absents is None:
            absents = []
        else:
            absents = copy.deepcopy(in_absents)
        ruined_t_indices = set([]) # record removed string to prevent repeated removing
        for c in neighbours[c_seed]:
            if len(ruined_t_indices) >= k_s: break
            if c not in absents and c!=0:
                t = find_t(last_routes, c)
                if t in ruined_t_indices: continue
                if isHugeRuin and np.random.random()>0.5:
                    newly_removed = last_routes[t]
                else:
                    l_t_max = min(l_s_max, len(last_routes[t])) # the length of the string removed for the Current Path
                    l_t = int(np.random.random()*l_t_max+1.0) # Actual number of removing for the Current Path
                    m, newly_removed = remove_nodes(last_routes[t], l_t, c, m)
                absents = absents+newly_removed
                ruined_t_indices.add(t)
        current_routes = routes_summary(last_routes, absents)
        return current_routes, absents
    
    def recreate(data, dist_m, current_routes, absents): # finds an optimal permutation of current_routes and absents by a Greedy Method
        
        def route_add(dist_m, current_routes, c, adding_position):
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
                assigned_capacity = vehicle_assignments[ir % n_vehicles]
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
                    if np.random.random() < blink_rate:
                        continue
                    cost_diff = dist_m[prev_node, c] + dist_m[c, next_node] - dist_m[prev_node, next_node]
                    probable_place.append((ir, iri, cost_diff))
            if len(probable_place)==0:
                adding_position = (-1,-1,1)
            else:
                adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
            current_routes = route_add(dist_m, current_routes, c, adding_position)

        return current_routes
    
    alpha_T = (final_T/init_T)**(1.0/n_iter)
    
    coords = data[:,:2]
    distance_matrix = np.zeros([len(coords),len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5
    
    if init_route is not None:
        best_routes = copy.deepcopy(init_route)
    else:
        best_routes = [[i] for i in range(1,len(data))]
    best_distances = []
    elapsed_times = []
    iterations= []
    best_distance = get_routes_distance(distance_matrix, best_routes)
    last_routes = copy.deepcopy(best_routes)
    last_distance = get_routes_distance(distance_matrix, best_routes)
    neighbours = get_neighbours(distance_matrix)
    best_distances.append(best_distance)
    print(len(best_routes), best_distance)
    
    temperature = init_T
    start_time = time.time()
    for i_iter in range(n_iter):
        current_routes, absents = ruin(last_routes, neighbours)
        current_routes = recreate(data, distance_matrix, current_routes, absents)
        
        current_distance = get_routes_distance(distance_matrix, current_routes)
        if len(current_routes)<len(best_routes) or \
           (current_distance<(last_distance-temperature*np.log(np.random.random())) and \
            len(current_routes)<=len(best_routes)):
            
            if len(current_routes)<len(best_routes) or current_distance<best_distance:
                best_distance = current_distance
                best_routes = current_routes
                if test_obj is not None and best_distance<test_obj:
                    break
            last_distance = current_distance
            last_routes = current_routes
        temperature*=alpha_T
        if verbose_step is not None and (i_iter+1)%verbose_step==0:
            elapsed_time = time.time() - start_time
            print(i_iter+1, np.round((i_iter+1)/n_iter*100,4), "%:",
                  len(best_routes), len(last_routes), len(current_routes),
                  best_distance, last_distance, current_distance)
            best_distances.append(best_distance)
            elapsed_times.append(elapsed_time)
            iterations.append(i_iter+1)
    if verbose_step is not None and n_iter%verbose_step!=0:print(i_iter+1, "100.0 %:", best_distance)
    return best_distance, best_routes, best_distances, elapsed_times, iterations
