
import pandas as pd
import numpy as np
from scipy import optimize

import datetime

def transformTo1Dparams(X, Theta):
    return np.concatenate((np.asarray(X).reshape((1,-1))[0], Theta.reshape((1,-1))[0]))

def transformTo2Dparams(x, n_movies, n_genres):
    X     = x[:n_movies*n_genres].reshape((-1,n_genres))
    Theta = x[n_movies*n_genres:].reshape((n_genres,-1))
    return X, Theta

def CostFunction(x, Y, R, n_movies, n_genres, C):
    X, Theta = transformTo2Dparams(x, n_movies, n_genres)
    regularization_term = sum(np.power(x,2))/(2*C)
    J = (np.power((R*(np.dot(X, Theta)-Y)),2)).sum()/2 + regularization_term
    return J

def gradientCostFunction(x, Y, R, n_movies, n_genres, C):
    X, Theta = transformTo2Dparams(x, n_movies, n_genres)
    dX     = np.empty(X.shape)
    dTheta = np.empty(Theta.shape)

    tmp = R*(np.dot(X, Theta)-Y)
    for k in range(n_genres):
        for i in range(n_movies):
            dX[i,k]     = np.dot(tmp[i, :], Theta[k,:]) + X[i, k]/C
        for j in range(n_users):
            dTheta[k,j] = np.dot(tmp[:,j], X[:,k])      + Theta[k, j]/C
    dx = transformTo1Dparams(dX, dTheta)
    return dx

def CostFunctionTest(x, Y_all, R, n_movies, n_genres, C, test):
    X, Theta = transformTo2Dparams(x, n_movies, n_genres)

    regularization_term = sum(np.power(x,2))/(2*C)
    R = np.abs(R-1)
    J = (np.power((R*(np.dot(X, Theta)-Y_all)),2)).sum()/2 + regularization_term
    return J

def similarityOfMovies(mid_i, mid_j, X):
    return 1 / float(sum( np.power(X.iloc[mid_i,:]-X.iloc[mid_j,:],2) ))

movie = pd.read_table('ml-100k/u.item', sep='|', header=None)

train = pd.read_table('ml-100k/u1.base', header=None, names=['uid', 'mid', 'rating', 'timestamp'])
test  = pd.read_table('ml-100k/u1.test', header=None, names=['uid', 'mid', 'rating', 'timestamp'])

train.timestamp = [datetime.datetime.fromtimestamp(int(s)) for s in train.timestamp]

n_users  = int(train.describe().loc['max', 'uid'])
n_movies = int(train.describe().loc['max', 'mid'])
n_genres = movie.iloc[:,5:].shape[1]

X = movie.iloc[:,5:]
R = np.zeros( (n_movies, n_users) )
Y = R
for i in range(train.shape[0]):
    R[train.mid[i]-1][train.uid[i]-1]=1
    Y[train.mid[i]-1][train.uid[i]-1]=train.rating[i]

Y_all = Y
for i in range(test.shape[0]):
    Y_all[test.mid[i]-1, test.uid[i]-1] = test.rating[i]
Theta = 0.001*np.random.rand( n_genres, n_users )
X = np.asarray(X, dtype='float')

X_init = X
Theta_init = Theta

# Optimization Process (CG)
alpha = 0.0003
test_iter = 200
test_lambda = np.linspace(10,500,50)
lambda_cost = np.zeros((2,test_lambda.shape[0]))

import matplotlib.pyplot as plt
for r in range(test_lambda.shape[0]):
    X = X_init
    Theta = Theta_init
    Cost = np.zeros((2,test_iter))
    C = test_lambda[r]
    for iter in range(test_iter):
        tmp = R*(np.dot(X, Theta)-Y)
        X = X - alpha * (np.dot(tmp, Theta.T)+X/C)
        tmp = R*(np.dot(X, Theta)-Y)
        Theta = Theta - alpha * (np.dot(X.T, tmp)+Theta/C)
        Cost_ =  CostFunction(transformTo1Dparams(X, Theta), Y, R, n_movies, n_genres, C)
        Cost_test = CostFunctionTest(transformTo1Dparams(X, Theta), Y_all, R, n_movies, n_genres, C, test)
        print Cost_/train.shape[0], Cost_test/test.shape[0], C
        Cost[0, iter] = Cost_
        Cost[1, iter] = Cost_test
    lambda_cost[0,r] = np.min(Cost[0,:])/train.shape[0]
    lambda_cost[1,r] = np.min(Cost[1,:])/test.shape[0]
plt.plot(range(test_lambda.shape[0]), lambda_cost[0,:])
plt.plot(range(test_lambda.shape[0]), lambda_cost[1,:])

plt.xlabel('Lamda')
plt.ylabel('Cost')
plt.show()

# Conclusion1: 0.0002 ~ 0.0003
# Conclusion2: 0.00020 ~ 0.00026
