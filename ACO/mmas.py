import os
import tsplib95
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool
from itertools import product
from time import time

from ant import Ant

optimal_value = {
    "berlin52": 7542,
    "rat99": 1211,
    "bier127": 118282,
    "ch130": 6110,
    "a280": 2579,
}

class MMAS:
    def __init__(self, problem_name, max_iter, num_ants, alpha=0.9, beta=1.5, rho=0.9, q=1.0, patience=10, num_workers=4, plots=False, name="MMAS"):
        """
        Args:
            problem_name (str): name of the problem
            max_iter (int): maximum number of iterations
            num_ants (int): number of ants
            alpha (float): pheromone importance
            beta (float): distance priority
            rho (float): evaporation rate
            q (float): pheromone deposit factor
        """
        self.problem_name = problem_name
        self.cords_x = []
        self.cords_y = []
        self.distances = self.get_distance_matrix(problem_name)
        self.num_nodes = len(self.distances)
        self.pheromone = np.ones(shape=(self.num_nodes, self.num_nodes), dtype=np.float64)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.plots = plots
        self.max_iter = max_iter
        self.num_ants = num_ants
        self.patience = patience
        self.const_patience = patience
        self.num_workers = num_workers
        self.global_best = (None, np.inf)
        self.iteration_test = (None, np.inf)
        self.ants = []
        self.iter_offset = 0
        self.name = name
        
        for _ in range(self.num_ants):
            self.ants.append(Ant(self.num_nodes, self.alpha, self.beta))

    def post_iteration(self, iter):
        # for ant in self.ants:
        #     ant.path_cost += self.distances[ant.location][ant.path[0]]
        #     ant.path.append(ant.location)
        #     ant.path.append(ant.path[0])
        #     assert(len(ant.path) == self.num_nodes + 1)
        
        self.ants.sort(key=lambda x: x.path_cost)
        self.old_best = self.global_best
        if self.ants[0].path_cost < self.global_best[1]:
            self.global_best = (self.ants[0].path, self.ants[0].path_cost)
        
        self.iteration_best = (self.ants[0].path, self.ants[0].path_cost)
        
        tau_max = self.q / (self.rho * self.global_best[1] + 1e-10)
        tau_min = tau_max / (2 * self.num_nodes)

        # if self.old_best == self.global_best:
        #     self.patience -= 1
        # else:
        #     self.patience = self.const_patience
        # count if most of value in pheromone matrix is tau_min
        
        self.update_pheromone(tau_min, tau_max, iter)
        cnt = np.sum(abs(self.pheromone - tau_min) < 1e-9)
        print(cnt / (self.num_nodes * self.num_nodes))
        if np.sum(abs(self.pheromone - tau_min) < 1e-9) > self.num_nodes * self.num_nodes * 0.9:
            print("Resetting pheromone matrix")
            self.iter_offset = iter
            self.pheromone = np.ones(shape=(self.num_nodes, self.num_nodes), dtype=np.float64) * tau_max
        # print(self.pheromone)
        # print(tau_min, tau_max)
        self.ants = []
        for i in range(self.num_ants):
            self.ants.append(Ant(self.num_nodes, self.alpha, self.beta))

    def get_distance_matrix(self, problem_name):
        problem = tsplib95.load(f'./tsp_graph/{problem_name}.tsp')
        node_num = len(list(problem.get_nodes()))
        weights = np.zeros(shape=(node_num, node_num), dtype=int)
        for i in problem.get_nodes():
            self.cords_x.append(problem.as_name_dict()['node_coords'][i][0])
            self.cords_y.append(problem.as_name_dict()['node_coords'][i][1])
        for i, j in product(range(node_num), range(node_num)):
            weights[i][j] = problem.get_weight(i+1, j+1)
        return weights
    
    def update_pheromone(self, tau_min, tau_max, iter):
        self.pheromone *= (1.0 - self.rho)
        iter = iter - self.iter_offset
        # scheduling
        if iter < 25:
            path, cost = self.iteration_best
        elif iter < 75:
            if iter % 6 == 0:
                path, cost = self.global_best
            else:
                path, cost = self.iteration_best
        elif iter < 125:
            if iter % 4 == 0:
                path, cost = self.global_best
            else:
                path, cost = self.iteration_best
        elif iter < 250:
            if iter % 3 == 0:
                path, cost = self.global_best
            else:
                path, cost = self.iteration_best
        else:
            if iter % 2 == 0:
                path, cost = self.global_best
            else:
                path, cost = self.iteration_best

        for i in range(len(path)-1):
            self.pheromone[path[i]][path[i+1]] += self.q / cost
            # self.pheromone[path[i+1]][path[i]] += self.q / cost
        
        self.pheromone = np.clip(self.pheromone, tau_min, tau_max)

    def run(self):
        for iter in range(self.max_iter):
            start_time = time()
            with Pool(processes=self.num_workers) as pool:
                self.ants = [pool.apply_async(ant.run, (self.pheromone, self.distances)) for ant in self.ants]
                self.ants = [ant.get() for ant in self.ants]
            
            self.post_iteration(iter)
            print(f"Iter {iter+1}: Current best: {self.global_best[1]}, Time: {time() - start_time}")
            if self.plots:
                self.plot()
            if self.global_best[1] <= optimal_value[self.problem_name]:
                print("Found optimal solution")
                break
                    
        return self.global_best
        
    def plot(self):
        plt.clf()
        graph = plt.plot(self.cords_x, self.cords_y, 'ro')
        plt.title(f"{self.problem_name}, Current Cost: {self.global_best[1]}, Method: {self.name}")
        for i in range(len(self.global_best[0]) - 1):
            plt.plot([self.cords_x[self.global_best[0][i]], self.cords_x[self.global_best[0][i+1]]], [self.cords_y[self.global_best[0][i]], self.cords_y[self.global_best[0][i+1]]], 'g-', lw=1)
        plt.show()
        os.makedirs(f"./plots/{self.name}/{self.problem_name}/", exist_ok=True)
        plt.savefig(f"./plots/{self.name}/{self.problem_name}/{self.problem_name}_{int(self.global_best[1])}.png")
        plt.pause(0.001)