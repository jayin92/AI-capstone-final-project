import os
import tsplib95
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np

from itertools import product

problem_list = [
    "berlin52",
    "rat99",
    "a280",
    "d657",
    "fl3795"
]

alpha = 0.9 # pheromone importance
beta = 1.5 # distance priority
rho = 0.9 # evaporation rate

class Ant:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.path = []
        self.path_cost = 0
        self.visited = []
        self.location = random.randint(0, self.num_nodes-1)

class TSP:
    def __init__(self, problem_name, max_iter, num_ants, alpha=0.9, beta=1.5, rho=0.9, q=1.0, patience=10):
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
        self.max_iter = max_iter
        self.num_ants = num_ants
        self.patience = patience
        self.const_patience = patience
        self.best = (None, np.inf)
        self.ants = []
        
        for i in range(self.num_ants):
            self.ants.append(Ant(self.num_nodes))

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
            self.ants.append(Ant(self.num_nodes))
        

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

    def choose_next(self, ant):
        total = 0.0
        nxt_nodes = []
        weights = []
        for i in range(self.num_nodes):
            if i not in ant.visited:
                val = self.pheromone[ant.location][i] ** self.alpha * \
                    (1.0 / (self.distances[ant.location][i]+1e-8)) ** self.beta
                weights.append(val)
                total += val
                nxt_nodes.append(i)
        probs = [x / total for x in weights]

        return np.random.choice(nxt_nodes, p=probs)

    def run(self):
        while self.max_iter > 0:
            for _ in range(self.num_nodes - 1):
                for ant in self.ants:
                    ant.visited.append(ant.location)
                    ant.path.append(ant.location)
                    nxt_node = self.choose_next(ant)
                    ant.path_cost += self.distances[ant.location][nxt_node]
                    ant.location = nxt_node
            self.post_iteration()
            if self.patience <= 0:
                break
            self.max_iter -= 1
            print(f"Current best: {self.best[1]}")
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


if __name__ == "__main__":
    plt.ion()
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="berlin52", help="choose problem")
    args = parser.parse_args()
    problem_name = args.problem
    tsp = TSP(problem_name, max_iter=100000, num_ants=10, patience=1000, rho=0.1, q=100)
    res = tsp.run()

