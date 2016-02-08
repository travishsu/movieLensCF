
import pandas as pd
import numpy as np
from scipy import optimize

import datetime

def CostFunction(X, Theta, Y, R, n_movies, n_genres):
    J = (np.power((R*(np.dot(X, Theta)-Y)),2)).sum()/2
    return J

def gradientCostFunction(X, Theta, Y, R, n_movies, n_genres):
    dX     = np.empty(X.shape)
    dTheta = np.empty(Theta.shape)

    tmp = R*(np.dot(X, Theta)-Y)
    for k in range(n_genres):
        for i in range(n_movies):
            dX[i,k]     = np.dot(tmp[i, :], Theta[k,:])
        for j in range(n_users):
            dTheta[k,j] = np.dot(tmp[:,j], X[:,k])
    return dX, dTheta

def CostFunctionTest(x, Y_all, R, n_movies, n_genres, C, test):
    X, Theta = transformTo2Dparams(x, n_movies, n_genres)

    regularization_term = sum(np.power(x,2))/(2*C)
    R = np.abs(R-1)
    J = (np.power((R*(np.dot(X, Theta)-Y_all)),2)).sum()/2 + regularization_term
    return J

def distanceOfMovies(mid_i, mid_j, X):
    return float(sum( np.power(X.iloc[mid_i,:]-X.iloc[mid_j,:],2) ))
def distanceOfUsers(uid_i, uid_j, Y_inserted):
    return float(sum( np.power(Y_inserted[:,uid_i]-Y_inserted[:,uid_j,],2) ))
def KNNusers(uid, K, Y_inserted):
    distanceAllUsers = np.ones(Y.shape[1])
    for i in range(Y.shape[1]):
        distanceAllUsers[i] = distanceOfUsers(uid, i, Y_inserted)
    indice = np.argsort(distanceAllUsers)
    return Y[np.ix_(range(Y.shape[0]), indice[:K])]
movie = pd.read_table('ml-100k/u.item', sep='|', header=None)

train = pd.read_table('ml-100k/u1.base', header=None, names=['uid', 'mid', 'rating', 'timestamp'])
test  = pd.read_table('ml-100k/u1.test', header=None, names=['uid', 'mid', 'rating', 'timestamp'])

train.timestamp = [datetime.datetime.fromtimestamp(int(s)) for s in train.timestamp]

n_users  = int(train.describe().loc['max', 'uid'])
n_movies = int(train.describe().loc['max', 'mid'])
n_genres = movie.iloc[:,5:].shape[1]

X = movie.iloc[:,5:]
R = np.zeros( (n_movies, n_users) )
Y = np.zeros( (n_movies, n_users) )
for i in range(train.shape[0]):
    R[train.mid[i]-1][train.uid[i]-1]=1
    Y[train.mid[i]-1][train.uid[i]-1]=train.rating[i]
    if np.isnan(train.rating[i]):
        print 2
Y_inserted = Y.copy()
movie_avg = np.sum(Y,axis=1) / np.sum(R,axis=1)
for i in range(movie_avg.shape[0]):
    if np.isnan(movie_avg[i]):
        movie_avg[i] = 0
for uid in range(n_users):
    for i in range(Y.shape[0]):
        if R[i,uid]==0:
            Y_inserted[i,uid] = movie_avg[i]
        if R[i,uid]==0:
            Y_inserted[i,uid] = movie_avg[i]
X = np.asarray(X, dtype='float')
X_init = X

# Optimization Process (CG)
K = n_users/5

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
