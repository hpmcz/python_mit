# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:52:29 2017

@author: Michal
"""

def oddTuples(aTup):
    '''
    aTup: a tuple
    
    returns: tuple, every other element of aTup. 
    '''
    aTup2 = ()
    counter = 1
    for a in aTup:
        if counter % 2 != 0:
            aTup2 += (a,)
            counter +=1
        else:
            counter +=1
    
    return aTup2

aTup = ('hi','my','firndo',1,3,)
print(oddTuples(aTup))