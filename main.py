#!/usr/bin/env python3

## GroupID-8 (14114002_14114068) - Abhishek Jaisingh & Tarun Kumar
## Date: April 15, 2016
## main.py - Main Travelling Salesman Algorithm

from bitwise_manipulations import *
from math import isinf
from helper import *
import json, time

a = []
random_size = 10
def choose(n):
	global a, random_size
	if n == 1:
		a = getInputFromUser()
	if n == 2:
		print("Enter value of n:")
		random_size = int(input())
		a = generateGraph(random_size)
	if n == 3:
		a = readFromFile()


def generateSubsets(n):
	l = []
	for i in range(2**n):
		l.append(i)
	return sorted(l, key = lambda x : size(x) )


def tsp(a):
	#global a
	n = len(a)
	l = generateSubsets(n)
	cost = [ [-1 for city in range(n)] for subset in l]
	p = [ [-1 for city in range(n)] for subset in l]

	pretty(a)
	t1 = time.time()
	count = 1
	total = len(l)

	for subset in l:
		for dest in range(n):
			if not size(subset):
				cost[subset][dest] = a[0][dest]
				#p[subset][dest] = 0
			elif (not inSubset(0, subset)) and (not inSubset(dest, subset)) :
				mini = float("inf")
				for i in range(n):
					if inSubset(i, subset):
						modifiedSubset = remove(i, subset)
						val = a[i][dest] + cost[modifiedSubset][i]
						
						if val < mini:
							mini = val
							p[subset][dest] = i

				if not isinf(mini):
					cost[subset][dest] = mini
		count += 1
	path = findPath(p)
	print(path)
	t2 = time.time()
	diff = t2 - t1
	print(" => ".join(path))

	Cost = cost[2**n-2][0]
	print(Cost)
	print("Time Taken: %f milliseconds" % (diff * 1000))
	return [int(p)-1 for p in path]


if __name__ =="__main__":
	# choice = int(input("Enter the choice:\n1 - To enter Input\n2 - Generate random Input\n3 - Read from \"input.json\" file\n"))
	# choose(choice)
	m = [[0, 3, 3, 2],
	[3,0,4,1],
	[3,4,0,3],
	[2,1,3,0]]

	tsp(m)