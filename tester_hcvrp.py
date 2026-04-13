import time
import numpy as np
from hcvrp_solver import sisr_hcvrp
import matplotlib.pyplot as plt
import os
import glob

"""
此程式用於記錄Best Distance對於Iterations和累積時間的變化
"""
total_times = []
distances = []
vehicle_capacities = [30,25,20]
# vehicle_capacities = [40,35,30,25,20] 
folder_path = "./test_data/3v_40c"
file_paths = glob.glob(os.path.join(folder_path, "*.txt"))

def parse_vrp_question(file_path):
    data = []
    with open(file_path, 'r') as f:
        i = 0
        for line in f:
            sline = line.strip()
            if len(sline)==0: continue
            if i>=6:
                data.append([float(i) for i in sline.split()][1:])
            i+=1
    return np.array(data),file_path

for index, file_path in enumerate(file_paths):
    data ,file_path= parse_vrp_question(file_path)
    data = data[:,:3]
    print(file_path)
    print(data.shape)
    print("------------------")
    start_time = time.time()
    np.random.seed(0)
    d, best_routes , best_distances, elapsed_times, iterations= sisr_hcvrp(data, vehicle_capacities, n_iter=10000, init_T=100.0, final_T=1.0,
                            init_route = None, verbose_step=1000,blink_rate=0.01)
    time_cost = time.time()-start_time
    total_times.append(time_cost)
    print(best_routes)
    distances.append(d)
    print("time_cost", time_cost)
    print("distance", d)

    formattd_best_distances = [f"{i:.6f}" for i in best_distances]
    elapsed_times = [f"{time:.2f}" for time in elapsed_times]
    with open('hcvrp_results.txt', 'a') as f:
        f.write(f"Input File: {file_path}\n")
        f.write(f"Distance: {d}\n")
        f.write(f"Best Routes: {best_routes}\n")
        f.write(f"Best Distances: {formattd_best_distances}\n")
        f.write(f"Elapsed times: {elapsed_times}\n")
        f.write(f"Iterations: {iterations}\n\n")
# plt.figure(figsize=(10, 6))
# plt.plot(best_distances, label='Best Route Length')
# plt.xlabel('Iterations')
# plt.ylabel('Best Route Length')
# plt.ylim(0,100)
# plt.title('Convergence of Best Route Length Over Iterations')
# plt.legend()
# plt.grid(True)
# plt.show()