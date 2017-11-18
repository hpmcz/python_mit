# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:14:07 2017

@author: Michal
"""

def gcdIter(a, b):
    '''
    a, b: positive integers
    
    returns: a positive integer, the greatest common divisor of a & b.
    '''
    if a < b:
        small = a
        big = b
    else:
        small = b
        big = a
    sec_small = small
    while big%small != 0 or sec_small%small != 0 :
        small -= 1
    return small
        