import tsplib95
problem = tsplib95.load('tsp_graph/berlin52.tsp')

for i in problem.get_nodes():
    print(problem.as_name_dict()['node_coords'][i])