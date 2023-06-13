# AI-capstone-final-project
The Final Project of AI Capstone in NYCU CS. Using three methods to solve TSP.

## Data Source

[TSPLLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)  
[Spec](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf)  
[The Optimal Solutions](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html)  
[tsplib95 Python Package](https://pypi.org/project/tsplib95/)  


## Problem Instances

| **name** | **nodes** | **optimal distance** |
|:--------:|:---------:|:--------------------:|
| berlin52 |     52    |         7542         |
|  rat99   |    99     |         1211         |
|  bier127 |    127    |       118282         |
|  ch130   |    130    |         6110         |
|   a280   |    280    |         2579         |

## Methods

### Ant Colony System

Ref: 
- [TSP Solving Utilizing Improved Ant Colony Algorithm](https://iopscience.iop.org/article/10.1088/1742-6596/2129/1/012026/pdf)
- https://strikingloo.github.io/ant-colony-optimization-tsp
- [An ant colony optimization method for generalized TSP problem](https://www.sciencedirect.com/science/article/pii/S1002007108002736)

#### Usage

`python ./ACO/main.py`

### Firefly Algorithm

Ref: [Evolutionary Discrete Firefly Algorithm for Travelling Salesman Problem](https://link.springer.com/chapter/10.1007/978-3-642-23857-4_38)

### DQN

Ref: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

#### DQN Introduction

DQN, or Deep Q-Network, is a reinforcement learning algorithm that combines deep learning and Q-learning to solve complex decision-making problems. At its core, DQN leverages a deep neural network to estimate the Q-values of different actions in a given state. The Q-values represent the expected future rewards for taking a particular action in a specific state. By learning these values, the agent can make informed decisions on which actions to take to maximize its cumulative reward. 

The DQN algorithm uses a technique called experience replay, which stores the agent's experiences (state, action, reward, next state) in a replay memory. During training, the agent samples mini-batches of experiences from this memory to decorrelate the data and improve learning efficiency.

The learning process of DQN can be summarized using the following equation:

$$
Q(s, a) \leftarrow (1 - \alpha) \cdot Q(s, a) + \alpha \cdot (r + \gamma \cdot \max_{a'} Q(s', a'))
$$

Here, $Q(s, a)$ represents the estimated Q-value of action $a$ in state $s$. The algorithm updates this estimate using a weighted average of the current estimate $(1 - \alpha) \cdot Q(s, a)$ and a new estimate $\alpha \cdot (r + \gamma \cdot \max_{a'} Q(s', a'))$. In this equation, $r$ represents the immediate reward obtained after taking action $a$ in state $s$, $s'$ represents the resulting state, $\alpha$ is the learning rate, and $\gamma$ is the discount factor that determines the importance of future rewards.

By repeatedly updating the Q-values based on this equation, DQN gradually learns to approximate the optimal Q-function, which leads to better decision-making capabilities for the agent.


#### RL Environment
For the graph $G(V, E)$, $N = |V|$.
- State representation
    
    State representation is Nx5 vector, where for each node we store whether this node is in the sequence, whether it is first or last, and its (x,y) coordinates. The coordinates of each nodes have normalize to one, for the sake of convenience of training neural net.

- Action

    The action is the integer from 1 to N. The env will mask the nodes that have been visited. The solution list is a list contained the nodes that have been visited.

- Reward definition

    The reward is calculated based on the total distance traveled by the solution compared to the current solution.
    ```python
    reward = -(total_dist(next_solution, W) - total_dist(solution, W))
    ```

#### Deep Neual Network Design
 The network consists of five fully connected (dense) layers (hiddent size is 1600) with ReLU activation applied to the output of each layer except for the last one. The input dimension is defined by shape of state tensor, the number of actions is defined by total number of nodes, and the size of the hidden layers is set to hidden_layer_size. The network takes input states and produces corresponding Q-values, which are used to estimate the value of each action in a reinforcement learning context.

#### Training result of different TSP map
- Training hyper parameter:
    - Episode: 2000
    - Exploration rate: 0.03 (Lower means less exploration)
    - Learning rate: 0.005,
    - Learning rate decay rate: 0.95,
    - Gamma: 0.997
    - Batch size: 32,
    - Replay buffer capacity: 10000
    - Dense layer hidden size: 1600
- a280
- berlin52
- bier127
- ch130
- rat99