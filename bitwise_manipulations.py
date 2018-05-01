## GroupID-8 (14114002_14114068) - Abhishek Jaisingh & Tarun Kumar
## Date: April 15, 2016
## bitwise_manipulations.py - Bitwise Manipulation Functions for Travelling Salesman Problem

def size(int_type):
   length = 0
   count = 0
   while (int_type):
       count += (int_type & 1)
       length += 1
       int_type >>= 1
   return count

def length(int_type):
   length = 0
   count = 0
   while (int_type):
       count += (int_type & 1)
       length += 1
       int_type >>= 1
   return length