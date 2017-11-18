# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:37:29 2017

@author: Michal
"""

import pandas as pd
import time
import datetime
import numpy as np

data = pd.read_csv('C:/Users/Michal/Downloads/calllog-2d_2/calllog2.csv',sep=';')
column = data['timeInMilliseconds']
column = column.tolist()
data_get = []
p = []
day_tab = []
#xv = []
for i in range(len(column)):
    a = str(column[i])
    a = a[:-3]
    column[i] = int(a)
    data_gmc = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(column[i])))
    data_get.append(data_gmc)
    p.append(data_gmc.split())         
data['time'] = [x[1] for x in p]
data['data'] = [x[0] for x in p]
# t_time = data['time'].tolist()

data['numberType'] = [1 if x == "B" else 0 for x in data['numberType']]

for i in range(len(column)):
    temp = str(data['data'][i])
    day = datetime.datetime.strptime(temp, '%Y-%m-%d').strftime('%w')
    day_tab.append(day)

data['day_of_the_week'] = day_tab
#for i in range(3086):
    #xv.append(datetime.datetime.strptime(t_time[i], '%H:%M:%S'))
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize
#fig, ax = plt.subplots()
#data['numberType']=[int(i) for i in data['numberType']]
#colors = ['red', 'blue']
#levels = [0, 1]

#cmap, norm = mpl.colors.from_levels_and_colors(levels=levels, colors=colors, extend='max')
#ax.scatter(data['time'], data['day_of_the_week'],c=data['numberType'], cmap=cmap, norm=norm )

"""
Okreslenie czasu polaczen
"""
czas_temp = []
for i in range(len(column)):
    czas_temp.append(datetime.datetime.strptime(data['time'][i], '%H:%M:%S'))    
data['czas_czas'] = czas_temp
z1 = data['czas_czas'][0]
czas_trwania = []
for i in range(len(column)-1):
    if data['callState'][i] == 'CALL_STATE_IDLE':
        temp1 = data['czas_czas'][i] - z1
        czas_trwania.append(temp1)
        z1 = data['czas_czas'][i+1]
        
data['czas_trwania'] = pd.Series(czas_trwania)    
czas_w_sec = []
for i in range(len(data['czas_trwania'])):
    k = data['czas_trwania'][i].total_seconds()
    czas_w_sec.append(k)
data['czas_w_sec'] = czas_w_sec
#
data_all = data[data['callState'] == 'CALL_STATE_IDLE']  
#data_all['czas_w_sec'] = kol2

del data_all['czas_w_sec']
del data_all['czas_trwania']
del data_all['czas_czas']
del data_all['deviceId']
data_all = data_all.reset_index(drop=True)
czas_sec = []
czas_sec = data['czas_w_sec'][0:1263]
data_all['czas_w_sec'] = czas_sec

X = np.array(([data_all['number'],data_all['czas_w_sec'],data_all['day_of_the_week'],data_all['numberType']]),dtype=float)
#Y = np.array(([data_all['numberType'],]),dtype=float)

X = X.T[:-1]
#Y = Y.T[:-1]
#X = np.append(X,Y,1)

X = X[X[:, 1]>0]

Y = X[:, 3]

Y = Y.reshape(len(X),1)

X = np.delete(X,3,1)
"""
X = np.delete(X,253,0)
Y = np.delete(Y,253,0)
X[:,0] = X[:,0]/100000000
X[:,2] = X[:,0]
X = np.delete(X,0,1)
"""
Lambda = 0.0001
X = X/np.amax(X, axis=0)
class NeuralNetwork(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 5
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        #self.W1 = np.array([[-0.0883195,-0.81268303,0.58959586,1.08155204,-0.44852925],[ 0.17149956,1.13480132,0.89209448,0.15314907,-0.02041156],[-0.39043592,-0.71825927,0.09320666,1.17981268,2.16746092],[ 2.36851642,-0.80104342,-1.15392245,1.24153018,0.31799854]])
        #self.W2 = np.array([[ 0.94804279],[-0.74917277],[-0.40225027],[-0.60176395],[-0.41133419]])
        print('w1')
        print(self.W1)
        print('w2')
        print(self.W2)
        
        self.Lambda = Lambda
    def forwardPropagation(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, self.W1)      
        self.a2 = self.sigmoid(self.z2)     
        self.z3 = np.dot(self.a2, self.W2)        
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self, z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
 
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forwardPropagation(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
    
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W1 and W2 for a given X and y:
        self.yHat = self.forwardPropagation(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
      
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2) 
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
    
def computeGradientsCheck(N, X, y):
        paramsInitial = N.getParams()
        chkgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Check Gradient 
            chkgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return chkgrad
    
class Trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 400, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, \
                                 jac=True, method='BFGS', \
                                 args=(X, y), options=options, \
                                 callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
    
NN = NeuralNetwork()

#sigTestValues = np.arange(-5,5,0.01)
#plt.plot(sigTestValues, NN.sigmoid(sigTestValues), linewidth = 2)
#plt.plot(sigTestValues, NN.sigmoidPrime(sigTestValues), linewidth = 2)
#plt.grid(1)
#plt.legend(['f'," f' "])
yHat = NN.forwardPropagation(X)
cost1 = NN.costFunction(X,Y)
dJdW1, dJdW2 = NN.costFunctionPrime(X,Y)

    
learningRate = 2

NN.W1 = NN.W1 + learningRate*dJdW1
NN.W2 = NN.W2 + learningRate*dJdW2
cost2 = NN.costFunction(X,Y)
                        
dJdW1, dJdW2 = NN.costFunctionPrime(X,Y)
NN.W1 = NN.W1 - learningRate*dJdW1
NN.W2 = NN.W2 - learningRate*dJdW2
cost3 = NN.costFunction(X, Y)

T = Trainer(NN)
T.train(X,Y)

yHat = NN.forwardPropagation(X)
