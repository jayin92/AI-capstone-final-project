import tsplib95
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool
from itertools import product
from time import time

from ant import Ant

class TSP:
    def __init__(self, problem_name, max_iter, num_ants, alpha=0.9, beta=1.5, rho=0.9, q=1.0, patience=10, num_workers=4, plots=False):
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
        self.pheromone = np.ones(shape=(self.num_nodes, self.num_nodes))
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
        self.best = (None, np.inf)
        self.ants = []
        
        for _ in range(self.num_ants):
            self.ants.append(Ant(self.num_nodes, self.alpha, self.beta))

    def post_iteration(self):
        for ant in self.ants:
            ant.path_cost += self.distances[ant.location][ant.path[0]]
            ant.path.append(ant.location)
            ant.path.append(ant.path[0])
            assert(len(ant.path) == self.num_nodes + 1)

        self.update_pheromone()
        old_best = self.best
        for i in range(self.num_ants):
            if self.ants[i].path_cost < self.best[1]:
                self.best = (self.ants[i].path, self.ants[i].path_cost)
        if old_best[1] == self.best[1]:
            self.patience -= 1
        else:
            self.patience = self.const_patience
        self.ants = []
        for i in range(self.num_ants):
            self.ants.append(Ant(self.num_nodes, self.alpha, self.beta))

    def get_distance_matrix(self, problem_name):
        problem = tsplib95.load(f'./tsp_graph/{problem_name}.tsp')
        node_num = len(list(problem.get_nodes()))
        weights = np.zeros(shape=(node_num, node_num))
        for i in problem.get_nodes():
            self.cords_x.append(problem.as_name_dict()['node_coords'][i][0])
            self.cords_y.append(problem.as_name_dict()['node_coords'][i][1])
        for i, j in product(range(node_num), range(node_num)):
            weights[i][j] = problem.get_weight(i+1, j+1)
        return weights
    
    def update_pheromone(self):
        for i, j in product(range(self.num_nodes), range(self.num_nodes)):
            self.pheromone[i][j] *= self.rho
        # sort by path cost
        self.ants.sort(key=lambda x: x.path_cost)
        for ant in self.ants[:self.num_ants//2]:
            for i in range(len(ant.path) - 1):
                self.pheromone[ant.path[i]][ant.path[i+1]] += self.q / ant.path_cost
                self.pheromone[ant.path[i+1]][ant.path[i]] += self.q / ant.path_cost

    def run(self):
        while self.max_iter > 0:
            start_time = time()
            with Pool(processes=self.num_workers) as pool:
                self.ants = [pool.apply_async(ant.run, (self.pheromone, self.distances)) for ant in self.ants]
                self.ants = [ant.get() for ant in self.ants]
            
            self.post_iteration()
            if self.patience <= 0:
                break
            self.max_iter -= 1
            print(f"Current best: {self.best[1]}, Time: {time() - start_time}")
            if self.plots:
                self.plot()
                    
        return self.best
        
    def plot(self):
        plt.clf()
        graph = plt.plot(self.cords_x, self.cords_y, 'ro')
        plt.title(f"{self.problem_name}, Current Cost: {self.best[1]}")
        # for i in range(self.num_nodes):
        #     plt.annotate(i, (self.cords_x[i], self.cords_y[i]))
        # for i in range(self.num_nodes):
        #     for j in range(self.num_nodes):
        #         if i != j:
        #             plt.plot([self.cords_x[i], self.cords_x[j]], [self.cords_y[i], self.cords_y[j]], 'k-', lw=0.5)
        for i in range(len(self.best[0]) - 1):
            plt.plot([self.cords_x[self.best[0][i]], self.cords_x[self.best[0][i+1]]], [self.cords_y[self.best[0][i]], self.cords_y[self.best[0][i+1]]], 'g-', lw=1)
        plt.show()
        plt.pause(0.001)