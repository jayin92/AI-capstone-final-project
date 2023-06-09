import numpy as np
import random
import math
from collections import namedtuple
import gym
import os
import time

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.signal import medfilt
import tsplib95

State = namedtuple('State', ('W', 'coords', 'partial_solution'))

class TSP_Env(gym.Env):
    def __init__(self, n = 10, name = None, size = 1, seed = 0):
        super().__init__()

        self.coords = []
        if name is not None:
            tsp = tsplib95.load('tsplib95/archives/problems/tsp/{}.tsp'.format(name))
            for i in tsp.get_nodes():
                self.coords.append(tsp.as_name_dict()['node_coords'][i])
            self.n = len(tsp.get_nodes()) # number of nodes
            
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
        
        self.dist_mat = distance_matrix(self.coords, self.coords)
        self.solution = []

    def reset(self):
        """ 
        Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.coords = self.grid_size * np.random.uniform(size=(self.n,2))
        self.dist_mat = distance_matrix(self.coords, self.coords)
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
            'Distance':self.total_distance(self.solution, self.dist_mat)
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
        coords = state.coords
        nr_nodes = coords.shape[0]

        xv = np.array([[
            (1 if i in solution else 0),
            (1 if i == sol_first_node else 0),
            (1 if i == sol_last_node else 0),
            coords[i,0],
            coords[i,1]
            ] for i in range(nr_nodes)])
        
        return xv.reshape(-1)

if __name__ == '__main__':
    env = TSP_Env(n=10, seed=0)
    for _ in range(10):
        print(env.reset())
        print(env.dist_mat)
        print(env.coords)