import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from student_utils_sp18 import *
from SetCoverPy import *
from pytsp import atsp_tsp, run, dumps_matrix
from functools import reduce
import numpy as np
import scipy.sparse.csgraph
import signal
from gurobipy import *
from main import *
"""
======================================================================
  Complete the following function.
======================================================================
"""
class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

def solve(number_of_kingdoms, list_of_kingdom_names, starting_kingdom, g, filename, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_kingdom_names: An list of kingdom names such that node i of the graph corresponds to name index i in the list
        starting_kingdom: The name of the starting kingdom for the walk
        adjacency_matrix: The adjacency matrix from the input file

    Output:
        Return 2 things. The first is a list of kingdoms representing the walk, and the second is the set of kingdoms that are conquered
    """
    # return closed_walk, conquered_kingdoms
    gnx =nx.from_numpy_matrix(g)
    paths = list(nx.all_pairs_dijkstra_path(gnx))
    recon = []
    for p in paths:
        a = []
        for i in range(number_of_kingdoms):
            a.append(p[1][i])
        recon.append(a)
    shortest_dist= scipy.sparse.csgraph.floyd_warshall(g)

    # complete_g = np.zeros_like(g)
    # for i in range(g.shape[0]):
    #     for j in range(g.shape[1]):
    #         if g[i,j] == np.inf:
    #             complete_g[i,j] = shortest_dist[i,j]
    #         else:
    #             complete_g[i,j] = g[i,j]
    # for kk in range(complete_g.shape[0]):
    #     complete_g[kk, kk]=0

    # #setcover
    # binarize = lambda x : int(x != 0)
    # binarize = np.vectorize(binarize)
    # binary = binarize(g)
    # for i in range(number_of_kingdoms):
    #     binary[i][i] = 1

    #========================================#
    #Dominating set using Integer Programming
    def get_neighbors_and_self(i):
            if conquer_cost[i] == 0:
                return np.nonzero(g[i])[0].tolist() + [i]
            else:
                return np.nonzero(g[i])[0].tolist()

    m = Model()
    edge_vars={}
    n = len(list_of_kingdom_names)
    conquer_cost = np.diag(g)
    for i in range(n):
        for j in range(i+1):
            edge_vars[i,j] = m.addVar(obj=1, vtype=GRB.BINARY, name='is_path_taken_'+str(i)+'_'+str(j))
            edge_vars[j,i] = edge_vars[i,j]

    for i in range(n):
        edge_vars[i] = m.addVar(obj=1, vtype=GRB.BINARY, name='is_conquered_' + str(i))

    for i in range(n):
        neighbor = get_neighbors_and_self(i)
        m.addConstr(quicksum(edge_vars[j] for j in neighbor) >= 1)

    m.update()
    start = list_of_kingdom_names.index(starting_kingdom)
    dist_from_start = [shortest_dist[start, i] for i in range(n)]
    dist_from_start = dist_from_start / np.average(dist_from_start)
    avg_dist = np.sum(shortest_dist, axis= 1)
    avg_dist = avg_dist/np.average(avg_dist)
    avg_conquer_cost = np.average(conquer_cost)

    closed_walk_dict = {}
    conquered_kingdoms_dict = {}
    cost_dict = {}
    for alpha in [0, 0.01, 0.1, 1, 10, 50]:
        for beta in [0, 0.01, 0.1, 1, 10, 50]:
            print('-------------------- alpha ',alpha, ' beta ', beta,'------------------------')
            m.setObjective(quicksum(conquer_cost[j]*edge_vars[j] + 
                        avg_conquer_cost*edge_vars[j]*(alpha*dist_from_start[j]+ beta*avg_dist[j]) for j in range(n)), GRB.MINIMIZE)
            m._vars = edge_vars
            #m.params.LazyConstraints = 1
            m.optimize()
            conquered = m.getAttr('x', edge_vars)

            should_conquer = [int(conquered[i]) for i in range(n)]
            countries_to_visit = []
            countries_idx = []
            cy = False

            for i,c in enumerate(list_of_kingdom_names):
                if should_conquer[i]:
                    countries_to_visit.append(c)
                    countries_idx.append(i)
                    if start == i:
                        cy = True

            if not cy:
                countries_idx.append(start)

            #Conquer own country and be done
            if len(countries_idx) == 1:
                closed_walk = [list_of_kingdom_names[j] for j in countries_idx]
                assert closed_walk[0] == starting_kingdom
                conquered_kingdoms = countries_to_visit

                closed_walk_dict[(alpha, beta)] = closed_walk
                conquered_kingdoms_dict[(alpha, beta)] = conquered_kingdoms
                total_cost = sum([conquer_cost[list_of_kingdom_names.index(c)] for c in conquered_kingdoms])
                print('total cost', total_cost)
                cost_dict[alpha, beta] = total_cost

            elif len(countries_idx) == 2:
                if countries_idx[1] == start:
                    countries_idx = countries_idx[::-1]
                p = recon[countries_idx[0]][countries_idx[1]]
                p = p + p[:-1][::-1]
                closed_walk = [list_of_kingdom_names[j] for j in p]
                conquered_kingdoms = countries_to_visit

                closed_walk_dict[(alpha, beta)] = closed_walk
                conquered_kingdoms_dict[(alpha, beta)] = conquered_kingdoms

                travel_cost = sum([g[p[i],p[i+1]]for i in range(len(p)-1)])
                total_cost = travel_cost + sum([conquer_cost[list_of_kingdom_names.index(c)] for c in conquered_kingdoms])
                print('total cost', total_cost)
                cost_dict[alpha, beta] = total_cost

            else:
                truncated_g = shortest_dist[countries_idx][:, countries_idx]
                tr_starting_index = countries_idx.index(start)
                    
                matrix_sym = atsp_tsp(truncated_g, strategy="avg")
                outf = "/tmp/myroute.tsp"
                with open(outf, 'w') as dest:
                    dest.write(dumps_matrix(matrix_sym, name="My Route"))
                tour = run(outf, start=tr_starting_index, solver="LKH")
                complete_visits = [countries_idx[i] for i in tour['tour']]

                complete_visits += [complete_visits[0]]
                p = [complete_visits[0]] + reduce(lambda x,y: x+y, [recon[complete_visits[i]][complete_visits[i+1]][1:] for i in range(len(complete_visits) -1)])
                closed_walk = [list_of_kingdom_names[j] for j in p]
                conquered_kingdoms = countries_to_visit
                print('starting kingdom', starting_kingdom)

                closed_walk_dict[(alpha, beta)] = closed_walk
                conquered_kingdoms_dict[(alpha, beta)] = conquered_kingdoms

                travel_cost = sum([g[p[i],p[i+1]]for i in range(len(p)-1)])
                total_cost = travel_cost + sum([conquer_cost[list_of_kingdom_names.index(c)] for c in conquered_kingdoms])
                print('total cost', total_cost)
                cost_dict[alpha, beta] = total_cost

    #calculate costs
    alpha, beta = min(cost_dict.items(), key=lambda key: cost_dict[key[0]])[0]
    print(alpha, beta,'!!')
    closed_walk = closed_walk_dict[alpha, beta]
    conquered_kingdoms = conquered_kingdoms_dict[alpha, beta]

    return closed_walk, conquered_kingdoms



"""
======================================================================
   No need to change any code below this line
======================================================================
"""

def matrix(input_file):
    adj = np.genfromtxt(input_file, dtype=float, skip_header=3)
    size = len(adj)
    for i in range(size):
        for j in range(size):
            if np.isnan(adj[i][j]):
                adj[i][j] = 0
    return adj

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    number_of_kingdoms, list_of_kingdom_names, starting_kingdom, _ = data_parser(input_data)
    adjacency_matrix = matrix(input_file)
    closed_walk, conquered_kingdoms = solve(number_of_kingdoms, list_of_kingdom_names, starting_kingdom, adjacency_matrix, params=params, filename=input_file)

    basename, filename = os.path.split(input_file)
    output_filename = utils.input_to_output(filename)
    output_file = f'{output_directory}/{output_filename}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    utils.write_data_to_file(output_file, closed_walk, ' ')
    utils.write_to_file(output_file, '\n', append=True)
    utils.write_data_to_file(output_file, conquered_kingdoms, ' ', append=True)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
