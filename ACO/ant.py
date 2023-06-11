import random

class Ant:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.path = []
        self.path_cost = 0
        self.visited = []
        self.location = random.randint(0, self.num_nodes-1)