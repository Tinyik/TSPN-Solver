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
    paths = list(nx.all_pairs_shortest_path(gnx))
    recon = []
    for p in paths:
        a = []
        for i in range(number_of_kingdoms):
            a.append(p[1][i])
        recon.append(a)
    shortest_dist= scipy.sparse.csgraph.floyd_warshall(g)

    complete_g = np.zeros_like(g)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if g[i,j] == np.inf:
                complete_g[i,j] = shortest_dist[i,j]
            else:
                complete_g[i,j] = g[i,j]
    for kk in range(complete_g.shape[0]):
        complete_g[kk, kk]=0 

    #setcover
    binarize = lambda x : int(x != 0)
    binarize = np.vectorize(binarize)
    binary = binarize(g)
    for i in range(number_of_kingdoms):
        binary[i][i] = 1

    cost = np.diag(g)
    cost = cost/ np.median(cost)


    sc = setcover.SetCover(binary, cost)
    solution, time_used = sc.SolveSCP()

    countries_to_visit = []
    countries_idx = []
    starting_index = list_of_kingdom_names.index(starting_kingdom)
    cy = False
    for i,c in enumerate(list_of_kingdom_names):
        if sc.s[i]:
            countries_to_visit.append(c)
            countries_idx.append(i)
            if starting_index == i:
                cy = True
    
    signal.alarm(0)
    try:
        # Use DP
        # truncated_g = shortest_dist[countries_idx][:, countries_idx].tolist()
        # tour = tsp(truncated_g)
        # complete_visits = [countries_idx[i] for i in tour]
        raise TimeoutException
    except TimeoutException:
        # Use approximation
        print("=============================DP DIDN'T WORK=============================")
        signal.alarm(0)

        signal.alarm(10)
        try:
            if len(countries_to_visit) == 1:
                p = countries_to_visit
                closed_walk = [list_of_kingdom_names[j] for j in p]
                conquered_kingdoms = countries_to_visit

                return closed_walk, conquered_kingdoms
            if len(countries_to_visit) == 2:
                p = recon[countries_idx[0]][countries_idx[1]]
                p = p + p[:-1][::-1]
                closed_walk = [list_of_kingdom_names[j] for j in p]
                conquered_kingdoms = countries_to_visit

                return closed_walk, conquered_kingdoms
            else:
                truncated_g = shortest_dist[countries_idx][:, countries_idx]
                if cy:
                    tr_starting_index = 0
                    for i in range(truncated_g.shape[0]):
                        if truncated_g[i][0] == starting_index:
                            tr_starting_index = i
                    #print(len(countries_to_visit),'!!')
                    matrix_sym = atsp_tsp(truncated_g, strategy="avg")
                    outf = "/tmp/myroute.tsp"
                    with open(outf, 'w') as dest:
                        dest.write(dumps_matrix(matrix_sym, name="My Route"))
                    tour = run(outf, start=tr_starting_index, solver="LKH")
                    complete_visits = [countries_idx[i] for i in tour['tour']]
                else:
                    arr = [shortest_dist[starting_index][i] for i in countries_idx]
                    nearest_o = arr.index(min(arr))
                    nearest_n = 0
                    for i in range(truncated_g.shape[0]):
                        if truncated_g[i][0] == nearest_o:
                            nearest_n = i
                    matrix_sym = atsp_tsp(truncated_g, strategy="avg")
                    outf = "/tmp/myroute.tsp"
                    with open(outf, 'w') as dest:
                        dest.write(dumps_matrix(matrix_sym, name="My Route"))
                    tour = run(outf, start=nearest_n, solver="LKH")
                    complete_visits = [starting_index] + [countries_idx[i] for i in tour['tour']]
        except TimeoutException:
            # Both didn't work
            print("=============================BOTH DIDN'T WORK=============================", filename)
            return []
        else:
            # Approximation worked
            complete_visits += [complete_visits[0]]
            p = [complete_visits[0]] + reduce(lambda x,y: x+y, [recon[complete_visits[i]][complete_visits[i+1]][1:] for i in range(1, len(complete_visits) -1)])

    else:
        # DP Worked
        complete_visits += [complete_visits[0]]
        p = [complete_visits[0]] + reduce(lambda x,y: x+y, [recon[complete_visits[i]][complete_visits[i+1]][1:] for i in range(1, len(complete_visits) -1)])

    closed_walk = [list_of_kingdom_names[j] for j in p]
    conquered_kingdoms = countries_to_visit
    print(closed_walk,'cw')
    print(starting_kingdom,'starting kingdom')
    print(countries_to_visit,'to_visit')

    return closed_walk, conquered_kingdoms


    # #print(len(countries_to_visit),'!!')
    # # if len(countries_to_visit) == 1:
    # #     p = countries_to_visit
    # # if len(countries_to_visit) == 2:
    # #     p = recon[countries_idx[0]][countries_idx[1]] 
    # #     p = p + p[:-1][::-1]
    # # else:
    #     # matrix_sym = atsp_tsp(truncated_g, strategy="avg")
    #     # outf = "/tmp/myroute.tsp"
    #     # with open(outf, 'w') as dest:
    #     #     dest.write(dumps_matrix(matrix_sym, name="My Route"))

    #     # signal.alarm(20)
    #     # try:
    #     #     tour = run(outf, start=list_of_kingdom_names.index(starting_kingdom), solver="LKH")
    #     # except TimeoutException:


    #     # complete_visits = [countries_idx[i] for i in tour['tour']]
    #     # complete_visits += [complete_visits[0]]
    # tour = tsp(truncated_g)
    # print(tour,'tour')
    # complete_visits = [countries_idx[i] for i in tour]
    # p = [complete_visits[0]] + reduce(lambda x,y: x+y, [recon[complete_visits[i]][complete_visits[i+1]][1:] for i in range(1, len(complete_visits) -1)])
    # closed_walk = [list_of_kingdom_names[j] for j in p]
    # print(closed_walk)

    # conquered_kingdoms = countries_to_visit
    # print(conquered_kingdoms)

    # return closed_walk, conquered_kingdoms

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
