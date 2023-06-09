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
|  brg180  |    180    |         1950         |
|   a280   |    280    |         2579         |
|   d657   |    657    |         48912        |
|  fl3795  |    3795   |         28772        |

## Methods

### Ant Colony System

Ref: [TSP Solving Utilizing Improved Ant Colony Algorithm](https://iopscience.iop.org/article/10.1088/1742-6596/2129/1/012026/pdf)

### Firefly Algorithm

Ref: [Evolutionary Discrete Firefly Algorithm for Travelling Salesman Problem](https://link.springer.com/chapter/10.1007/978-3-642-23857-4_38)

### DQN

Ref: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
