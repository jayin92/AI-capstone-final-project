import random
import numpy as np

class Ant:
    def __init__(self, num_nodes, alpha=0.9, beta=1.5):
        self.num_nodes = num_nodes
        self.path = []
        self.path_cost = 0
        self.alpha = alpha
        self.beta = beta
        self.not_visited = set(range(num_nodes))
        self.location = random.randint(0, self.num_nodes-1)
    
    def run(self, pheromone, distances):
        self.pheromone = pheromone
        self.distances = distances
        for _ in range(self.num_nodes - 1):
            self.not_visited.remove(self.location)
            self.path.append(self.location)
            nxt_node = self.choose_next()
            self.path_cost += self.distances[self.location][nxt_node]
            self.location = nxt_node
        
        self.path.append(self.location)
        # self.path.append(self.path[0])
        # self.path_cost += self.distances[self.location][self.path[0]]

        # Run two-opt
        self.path, self.path_cost = self.two_opt()
        self.path_cost += self.distances[self.path[-1]][self.path[0]]
        self.path.append(self.path[0])

        return self

    def choose_next(self):
        total = 0.0
        nxt_nodes = []
        weights = np.zeros(shape=(len(self.not_visited)))
        for idx, i in enumerate(self.not_visited):
            val = self.pheromone[self.location][i] ** self.alpha * \
                (1.0 / (float(self.distances[self.location][i])+1e-8)) ** self.beta
            weights[idx] = val
            total += val
            nxt_nodes.append(i)
        probs = weights / total

        return np.random.choice(nxt_nodes, p=probs)

    def two_opt(self, improvement_threshold=0.01):
        self.best_cost = self.path_cost
        self.best_path = self.path
        improvement_factor = 1
        
        while improvement_factor > improvement_threshold:
            previous_best = self.best_cost
            for swap_first in range(1, self.num_nodes - 2):
                for swap_last in range(swap_first + 1, self.num_nodes - 1):
                    before_start = self.best_path[swap_first - 1]
                    start = self.best_path[swap_first]
                    end = self.best_path[swap_last]
                    after_end = self.best_path[swap_last+1]
                    before = self.distances[before_start][start] + self.distances[end][after_end]
                    after = self.distances[before_start][end] + self.distances[start][after_end]
                    if after < before:
                        self.best_path = self.swap(self.best_path, swap_first, swap_last)
                        self.best_cost = self.best_cost - before + after
            improvement_factor = 1 - self.best_cost/previous_best
        return self.best_path, self.best_cost
    
    def swap(self, path, swap_first, swap_last):
        path_updated = np.concatenate((path[0:swap_first],
                                       path[swap_last:-len(path) + swap_first - 1:-1],
                                       path[swap_last + 1:len(path)]))
        return path_updated.tolist()