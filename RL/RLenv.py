import numpy as np
import random
import math
from collections import namedtuple
import gym
import os
import time

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import tsplib95

State = namedtuple('State', ('W', 'coords', 'partial_solution'))

class TSP_Env(gym.Env):
    def __init__(self, n = 10, name = None, size = 1, seed = 0):
        super().__init__()

        self.coords = []
        self.norm = 1
        self.name = name
        if name is not None:
            tsp = tsplib95.load('./tsp_graph/{}.tsp'.format(name))
            max_ = 0
            min_ = 999999999
            for i in tsp.get_nodes():
                self.coords.append(tsp.as_name_dict()['node_coords'][i])
                max_ = max([max_, tsp.as_name_dict()['node_coords'][i][0], tsp.as_name_dict()['node_coords'][i][1]])
                min_ = min([min_, tsp.as_name_dict()['node_coords'][i][0], tsp.as_name_dict()['node_coords'][i][1]])        
            self.norm = max_ - min_
            self.coords = np.array(self.coords)
            self.norm_coords = self.coords/self.norm
            self.n = len(list(tsp.get_nodes())) # number of nodes
        else:
            self.n = n
            self.grid_size = size
            self.coords = self.grid_size * np.random.uniform(size=(self.n,2))
            
        #set random seed
        self.seed = seed 
        random.seed(seed)
        np.random.seed(seed)
        self.action_space = self.n
        self.observation_space = self.n*5
        self.mask = [0 for _ in range(self.n)]
        
        self.int_dist_mat = distance_matrix(self.coords, self.coords).astype(int)
        self.dist_mat = distance_matrix(self.norm_coords, self.norm_coords)
        self.solution = []

    def reset(self, fix_seed=False):
        """ 
        Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
        """
        if fix_seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.solution = [random.randint(0, self.n-1)]
        for i in range(self.n):
            self.mask[i] = 1 if i in self.solution else 0
 
        return self.state2vec(State(partial_solution=self.solution, W=self.dist_mat, coords=self.coords)), self.mask
        
    def step(self, action):
        next_solution = self.solution + [action]
        reward = -(self.total_distance(next_solution, self.dist_mat) - self.total_distance(self.solution, self.dist_mat))
        next_state = State(partial_solution=next_solution, W=self.dist_mat, coords=self.coords)
        done = len(set(next_state.partial_solution)) == self.dist_mat.shape[0]
        for i in range(self.n):
            self.mask[i] = 1 if i in next_solution else 0
        self.solution = next_solution
        info = {
            'current_solution':self.solution,
            'Dist_matrix':self.dist_mat,
            'Coordinates':self.coords,
            'Distance':self.total_distance(self.solution, self.int_dist_mat)
        }
        return self.state2vec(next_state), reward, self.mask, done, info
    
    def render(self):
        """ 
        Utility function to plot the fully connected graph
        """
        n = len(self.coords)
    
        plt.scatter(self.coords[:,0], self.coords[:,1], s=[50 for _ in range(n)])
        for i in range(n):
            for j in range(n):
                if j < i:
                    plt.plot([self.coords[i,0], self.coords[j,0]], [self.coords[i,1], self.coords[j,1]], 'b', alpha=0.7)
  
    
    def total_distance(self, solution, W):
        if len(solution) < 2:
            return 0  # there is no travel
        
        total_dist = 0
        for i in range(len(solution) - 1):
            total_dist += W[solution[i], solution[i+1]].item()
            
        # if this solution is "complete", go back to initial point
        if len(solution) == W.shape[0]:
            total_dist += W[solution[-1], solution[0]].item()

        return total_dist

    def get_next_neighbor_random(self, state):
        solution, W = state.partial_solution, state.W
        
        if len(solution) == 0:
            return random.choice(range(W.shape[0]))
        already_in = set(solution)
        candidates = list(filter(lambda n: n.item() not in already_in, W[solution[-1]].nonzero()))
        if len(candidates) == 0:
            return None
        return random.choice(candidates).item()
    
    def state2vec(self, state):
        """ Creates a vector representing the history of visited nodes, from a (single) state tuple.
            
            Returns a (Nx5) vector, where for each node we store whether this node is in the sequence,
            whether it is first or last, and its (x,y) coordinates.
        """
        solution = set(state.partial_solution)
        sol_last_node = state.partial_solution[-1] if len(state.partial_solution) > 0 else -1
        sol_first_node = state.partial_solution[0] if len(state.partial_solution) > 0 else -1
        coords = self.norm_coords
        nr_nodes = coords.shape[0]

        xv = np.array([[
            (1 if i in solution else 0),
            (1 if i == sol_first_node else 0),
            (1 if i == sol_last_node else 0),
            coords[i,0],
            coords[i,1]
            ] for i in range(nr_nodes)])
        
        return xv.reshape(-1)
    
    def plot(self):
        plt.clf()
        graph = plt.plot(self.coords[:,0], self.coords[:,1], 'ro')
        plt.title(f"{self.name}, Current Cost: {self.total_distance(self.solution, self.int_dist_mat)}, Method: DQN")
        # for i in range(self.num_nodes):
        #     plt.annotate(i, (self.cords_x[i], self.cords_y[i]))
        # for i in range(self.num_nodes):
        #     for j in range(self.num_nodes):
        #         if i != j:
        #             plt.plot([self.cords_x[i], self.cords_x[j]], [self.cords_y[i], self.cords_y[j]], 'k-', lw=0.5)
        for i in range(len(self.solution)-1):
            plt.plot([self.coords[i][0], self.coords[i+1][0]], [self.coords[i][1], self.coords[i+1][1]], 'g-', lw=1)
        plt.show()
        os.makedirs(f"./RL/plots/", exist_ok=True)
        plt.savefig(f"./RL/plots/{self.name}.png")
        plt.pause(0.001)

if __name__ == '__main__':
    env = TSP_Env(name='berlin52')
    a,_ = env.reset()
    print(a)
    print(env.coords)
    print(env.dist_mat)