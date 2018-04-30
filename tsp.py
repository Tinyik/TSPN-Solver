import os
import numpy as np
import scipy.sparse.csgraph

INPUT_DIR = './inputs'
input0 = os.path.join(INPUT_DIR, '0.in')

def read_file(file):
    with open(file, 'r') as f:
        data = f.readlines()
    data = [line.strip().split() for line in data]
    return data

def data_parser(input_data):
    number_of_kingdoms = int(input_data[0][0])
    list_of_kingdom_names = input_data[1]
    starting_kingdom = input_data[2][0]
    adjacency_matrix = [[entry if entry == 'x' else float(entry) for entry in row] for row in input_data[3:]]
    return number_of_kingdoms, list_of_kingdom_names, starting_kingdom, adjacency_matrix

def matrix(input_file):
    adj = np.genfromtxt(input_file, dtype=float, skip_header=3)
    size = len(adj)
    for i in range(size):
        for j in range(size):
            if np.isnan(adj[i][j]):
                adj[i][j] = np.Infinity
    return adj

def preprocess(input_file):
    data = read_file(input_file)
    number_of_kingdoms, list_of_kingdom_names, starting_kingdom, adjacency_matrix = data_parser(data)
    adjacency_matrix = matrix(input_file)
    return number_of_kingdoms, list_of_kingdom_names, starting_kingdom, adjacency_matrix

number_of_kingdoms, list_of_kingdom_names, starting_kingdom, adjacency_matrix = preprocess(input0)

g = matrix(input0)
shortest_dist = scipy.sparse.csgraph.floyd_warshall(g)

complete_g = np.zeros_like(g)
for i in range(g.shape[0]):
    for j in range(g.shape[1]):
        if g[i,j] == np.inf:
            complete_g[i,j] = shortest_dist[i,j]
        else:
            complete_g[i,j] = g[i,j]

#Set Cover
binarize = lambda x : int(x != np.inf)
binarize = np.vectorize(binarize)
binary = binarize(g)

cost = np.diag(g)
cost = cost/ np.median(cost)

from SetCoverPy import *
sc = setcover.SetCover(binary, cost)
solution, time_used = sc.SolveSCP()
sc.s

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

class CreateDistanceCallback(object):
    """Create callback to calculate distances between points."""
    def __init__(self, adjacency_matrix):
        """Array of distances between points."""

        self.matrix = adjacency_matrix

    def Distance(self, from_node, to_node):
        return int(self.matrix[from_node][to_node])

def solve(list_of_kingdom_names, starting_kingdom, adjacency_matrix, params=[]):
#     city_names = ["New York", "Los Angeles", "Chicago", "Minneapolis", "Denver", "Dallas", "Seattle",
#                 "Boston", "San Francisco", "St. Louis", "Houston", "Phoenix", "Salt Lake City"]
    tsp_size = len(list_of_kingdom_names)
    num_routes = 1    # The number of routes, which is 1 in the TSP.
    # Nodes are indexed from 0 to tsp_size - 1. The depot is the starting node of the route.
    depot = list_of_kingdom_names.index(starting_kingdom)

    # Create routing model
    if tsp_size > 0:
        routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
        search_parameters = pywrapcp.RoutingModel.DefaultModelParameters()

        # Create the distance callback, which takes two arguments (the from and to node indices)
        # and returns the distance between these nodes.
        dist_between_nodes = CreateDistanceCallback(adjacency_matrix)
        dist_callback = dist_between_nodes.Distance
        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
        # Solve, returns a solution if any.
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            # Solution cost.
            print("Total distance: " + str(assignment.ObjectiveValue()) + " miles\n")
            # Inspect solution.
            # Only one route here; otherwise iterate from 0 to routing.vehicles() - 1
            route_number = 0
            index = routing.Start(route_number) # Index of the variable for the starting node.
            route = ''
            while not routing.IsEnd(index):
                # Convert variable indices to node indices in the displayed route.
                route += str(city_names[routing.IndexToNode(index)]) + ' -> '
                index = assignment.Value(routing.NextVar(index))
            route += str(city_names[routing.IndexToNode(index)])
            print("Route:\n\n" + route)
        else:
            print('No solution found.')
    else:
        print('Specify an instance greater than 0.')

contries_to_visit = []
contries_idx = []
for i,c in enumerate(list_of_kingdom_names):
    if sc.s[i]:
        contries_to_visit.append(c)
        contries_idx.append(i)
print(contries_to_visit)
print(contries_idx)

truncated_g = shortest_dist[contries_idx][:, contries_idx]
solve(contries_to_visit, starting_kingdom, truncated_g)