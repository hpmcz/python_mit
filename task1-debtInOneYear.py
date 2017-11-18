# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:02:57 2017

@author: Michal
"""


balance = 320000
temp = balance
annualInterestRate = 0.2
fixPayment = 0
mInterestRate = annualInterestRate / 12
while balance >= 0:
    balance = temp
    fixPayment += 0.01
    if balance <= 0:
        print(fixPayment)   
    else:
        counter = 12
        while counter != 0:
            counter -= 1
            uBalance = balance - fixPayment
            balance = uBalance + mInterestRate * uBalance                     
print(round(fixPayment,2))