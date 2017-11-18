# -*- coding: utf-8 -*-
"""
Monthly interest rate= (Annual interest rate) / 12.0
Minimum monthly payment = (Minimum monthly payment rate) x (Previous balance)
Monthly unpaid balance = (Previous balance) - (Minimum monthly payment)
Updated balance each month = (Monthly unpaid balance) + (Monthly interest rate x Monthly unpaid balance)
"""

balance = 484
annualInterestRate = 0.2
monthlyPaymentRate = 0.04
counter = 12
while counter != 0:
    counter -= 1
    mInterestRate = annualInterestRate / 12.0
    minMonthlyPayment = monthlyPaymentRate * balance
    uBalance = balance - minMonthlyPayment
    balance = uBalance + mInterestRate * uBalance
print(round(balance,2))
           