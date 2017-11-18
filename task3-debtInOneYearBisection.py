# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:54:24 2017

@author: Michal
"""
balance = 320000     
annualInterestRate = 0.2

mInterestRate = annualInterestRate / 12
mPaymentLow = balance / 12
mPaymentHigh = (balance * (1+mInterestRate)**12)/12
fixPayment = (mPaymentLow + mPaymentHigh)/2

def findBalance(mPaymentLow, mPaymentHigh, fixPayment, balance):
    """
    Szukanie minimalnej wielkosci raty aby splacic w rok zadluzenie metodą bisekcji
    """
    counter = 12
    temp = balance
    while counter != 0:
        counter -= 1
        uBalance = balance - fixPayment
        balance = uBalance + mInterestRate * uBalance  
    if balance > 0.1:
        mPaymentLow = (mPaymentLow + mPaymentHigh)/2
        fixPayment = (mPaymentLow + mPaymentHigh)/2
        balance = temp
        return findBalance(mPaymentLow, mPaymentHigh, fixPayment, balance)
        
    elif balance < -0.1:
        mPaymentHigh = (mPaymentLow + mPaymentHigh)/2
        fixPayment = (mPaymentLow + mPaymentHigh)/2
        balance = temp
        return findBalance(mPaymentLow, mPaymentHigh, fixPayment, balance)
        
    else:
        return fixPayment
    
answer = findBalance(mPaymentLow, mPaymentHigh, fixPayment, balance)
print(round(answer,2))