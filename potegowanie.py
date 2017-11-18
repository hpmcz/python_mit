# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 20:47:12 2017

@author: Michal
"""

def iterPower(base, exp):
    '''
    base: int or float.
    exp: int >= 0
 
    returns: int or float, base^exp
    '''
    temp = 1

    while exp > 0:
        exp -= 1
        temp *= base
    return temp
