import tsplib95
problem = tsplib95.load('tsplib95/archives/problems/tsp/berlin52.tsp')

for i in problem.get_nodes():
    print(problem.as_name_dict()['node_coords'][i])