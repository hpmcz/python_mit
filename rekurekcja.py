# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 20:52:39 2017

@author: Michal
"""

def recurPower(base, exp):
    '''
    base: int or float.
    exp: int >= 0
 
    returns: int or float, base^exp
    '''    
    if exp == 1:
        return base
    else:     
        return base*recurPower(base,exp-1)
        