# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:48:09 2017

@author: Michal
"""

def isIn(char, aStr):
    ''' '''
    mid = len(aStr) // 2
    return False  
        if not aStr 
        else isIn(char, aStr[:mid])   if char < aStr[mid] 
        else isIn(char, aStr[mid+1:]) if char > aStr[mid] 
        else True
