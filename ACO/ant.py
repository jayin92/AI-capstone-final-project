import random
import numpy as np

class Ant:
    def __init__(self, num_nodes, alpha=0.9, beta=1.5):
        self.num_nodes = num_nodes
        self.path = []
        self.path_cost = 0
        self.alpha = alpha
        self.beta = beta
        self.visited = []
        self.location = random.randint(0, self.num_nodes-1)
    
    def run(self, pheromone, distances):
        self.pheromone = pheromone
        self.distances = distances
        for _ in range(self.num_nodes - 1):
            self.visited.append(self.location)
            self.path.append(self.location)
            nxt_node = self.choose_next()
            self.path_cost += self.distances[self.location][nxt_node]
            self.location = nxt_node
        
        return self

    def choose_next(self):
        total = 0.0
        nxt_nodes = []
        weights = []
        for i in range(self.num_nodes):
            if i not in self.visited:
                val = self.pheromone[self.location][i] ** self.alpha * \
                    (1.0 / (self.distances[self.location][i]+1e-8)) ** self.beta
                weights.append(val)
                total += val
                nxt_nodes.append(i)
        probs = [x / total for x in weights]

        return np.random.choice(nxt_nodes, p=probs)
