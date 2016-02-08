import pandas as pd
import numpy as np
import datetime
from scipy import optimize

def distanceOfUsers(uid_i, uid_j, Y_inserted):
    return float(sum( np.power(Y_inserted[:,uid_i]-Y_inserted[:,uid_j,],2) ))
def KNNusers(uid, K, Y_inserted):
    distanceAllUsers = np.ones(Y.shape[1])
    for i in range(Y.shape[1]):
        distanceAllUsers[i] = distanceOfUsers(uid, i, Y_inserted)
    indice = np.argsort(distanceAllUsers)
    return Y[np.ix_(range(Y.shape[0]), indice[1:K+1])]

movie = pd.read_table('ml-100k/u.item', sep='|', header=None)

train = pd.read_table('ml-100k/u2.base', header=None, names=['uid', 'mid', 'rating', 'timestamp'])
test  = pd.read_table('ml-100k/u2.test', header=None, names=['uid', 'mid', 'rating', 'timestamp'])

train.timestamp = [datetime.datetime.fromtimestamp(int(s)) for s in train.timestamp]

n_users  = int(train.describe().loc['max', 'uid'])
n_movies = int(train.describe().loc['max', 'mid'])
n_genres = movie.iloc[:,5:].shape[1]

X = movie.iloc[:,5:]
X = np.asarray(X, dtype='float')
R = np.zeros( (n_movies, n_users) )
Y = R.copy()
for i in range(train.shape[0]):
    R[train.mid[i]-1][train.uid[i]-1]=1
    Y[train.mid[i]-1][train.uid[i]-1]=train.rating[i]

Y_inserted = Y.copy()
indice_movie_rating = [i for i in range(n_movies) if np.sum(R[i,:])!=0]
Y_reduced = Y_inserted[np.ix_(indice_movie_rating,range(Y.shape[1]))]
R_reduced = R[np.ix_(indice_movie_rating,range(Y.shape[1]))]

Y_mu = Y.sum(axis=0)/R.sum(axis=0)
for i in range(n_users):
    Y[np.ix_([k for k in range(n_movies) if R[k,i]==0],[i])] = Y_mu[i]
weights    = np.zeros((n_users,n_users))
diff       = np.empty((n_movies, n_users))
std_rating = np.empty(n_users)

# Pearson correlation
for i in range(n_users):
    diff[:,i] = Y[:,i]-Y_mu[i]
    std_rating[i] = diff[:,i].std()
for i in range(n_users):
    for j in range(n_users):
        weights[i,j] = np.dot(diff[:,i], diff[:,j])/(std_rating[i]*std_rating[j])

# Test
pred = np.empty(test.shape[0])
for test_id in range(test.shape[0]):
    a = test.uid[test_id]
    i = test.mid[test_id]

    pred[test_id] = Y_mu[a] + np.dot(diff[i,:],weights[:,a])/(weights[:,a].sum())

print ((test.rating - pred)**2).sum()/test.shape[0]
error   = test.rating - pred
correct = np.array([error[i] for i in range(test.shape[0]) if np.abs(error[i])<0.5])
correct_proportion = correct.shape[0]/test.shape[0]
