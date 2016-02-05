
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

Theta = np.random.rand( n_genres, n_users )

# Optimization Process (CG)
x0 = transformTo1Dparams(X, Theta)
C = 5
print CostFunction(x0, Y, R, n_movies, n_genres, C)
result = optimize.fmin_cg( (lambda x: CostFunction(x, Y, R, n_movies, n_genres, C)), x0,
           fprime=(lambda x: gradientCostFunction(x, Y, R, n_movies, n_genres, C)),
           full_output=True, retall=True, disp=True)
