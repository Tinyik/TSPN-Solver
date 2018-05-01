## GroupID-8 (14114002_14114068) - Abhishek Jaisingh & Tarun Kumar
## Date: April 15, 2016
## helper.py - Helper functions for Travelling Salesman Problem

from bitwise_manipulations import *
import time
import random, json


def inSubset(i, s):
	while i > 0 and s > 0:
		s = s >> 1
		i -= 1
	cond = s & 1
	return cond

def remove(i, s):
	x = 1
	x = x << i
	l = length(s)
	l = 2 ** l - 1
	x = x ^ l
	#print ( "i - %d x - %d  s - %d x&s -  %d " % (i, x, s, x & s) )
	return x & s

def findPath(p):
	n = len(p[0])
	number = 2 ** n - 2
	prev = p[number][0]
	path = []
	while prev != -1:
		path.append(prev)
		number = remove(prev, number)
		prev = p[number][prev]
	reversepath = [str(path[len(path)-i-1]+1) for i in range(len(path))]
	reversepath.append("1")
	reversepath.insert(0, "1")
	return reversepath

def pretty(a):
	print("=========================")
	for i in range(len(a)):
		for j in range(len(a[0])):
			print ("%2d"%(a[i][j])),
		print("")
	print("=========================")

def generateGraph(n):
	a = [ [-1 for i in range(n)] for j in range(n)]
	for i in range(n):
		for j in range(n):
			rand = random.randint(0, n)
			if a[i][j] < 0:
				a[i][j] = rand
				a[j][i] = rand
			if i == j:
				a[i][i] = 0
				
	#pretty(a)
	return a

def getInputFromUser():
	n = int(input("Enter number of cities:"))
	print("Enter the values like 1st row, 2nd row and so on.")
	a = [[int(input()) for i in range(n)] for j in range(n)]
	print(a)
	return a

def readFromFile():
	with open('input.json', 'r') as f:
		s = f.read()
		data  = json.loads(s)
		print(data)
		return data
