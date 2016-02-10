import pandas as pd
import numpy as np

# Load data
train = pd.read_table('ml-100k/u2.base', header=None, names=['uid', 'mid', 'rating', 'timestamp'])
test  = pd.read_table('ml-100k/u2.test', header=None, names=['uid', 'mid', 'rating', 'timestamp'])

n_users  = int(train.describe().loc['max', 'uid'])
n_movies = int(train.describe().loc['max', 'mid'])

# Transform
Y = np.zeros( (n_movies, n_users) )
R = Y.copy()
for i in range(train.shape[0]):
    Y[train.mid[i]-1][train.uid[i]-1] = train.rating[i]
    R[train.mid[i]-1][train.uid[i]-1] = 1

Y_mu = Y.sum(axis=1)/R.sum(axis=1)

# Allocate
weights    = np.empty((n_movies,n_movies))
diff       = np.empty((n_movies, n_users))
std_rating = np.empty(n_movies)

for i in range(n_movies):
    diff[i,:] = Y[i,:]-Y_mu[i]
    std_rating[i] = np.power(diff[i,:],2).sum()/R[i,:].sum()

# Weighting (Pearson correlation)
for i in range(n_movies):
    for j in range(n_movies):
        weights[i,j] = np.dot(R[i,:]*diff[i,:], R[j,:]*diff[j,:])/(std_rating[i]*std_rating[j])

# Predict
pred_correlation = np.empty(test.shape[0])
pred_cosine      = pred_correlation.copy()
thredhold = 0.00001
for test_id in range(test.shape[0]):
    a = test.uid[test_id]
    i = test.mid[test_id]

    idx_positive_weights    = np.array([j for j in range(weights.shape[0]) if weights[i,j]>thredhold])
    val_positive_weights    = weights[np.ix_([i],idx_positive_weights)][0]
    val_positive_difference = (diff[np.ix_(idx_positive_weights,[a])].T)[0] * (R[np.ix_(idx_positive_weights,[a])].T)[0]
    pred_correlation[test_id] = Y_mu[i] + np.dot(val_positive_difference,val_positive_weights)/(val_positive_weights.sum())

# Validation
error   = test.rating - pred_correlation
correct = np.array([error[i] for i in range(test.shape[0]) if np.abs(error[i])<0.5])
correct_proportion = correct.shape[0]/float(test.shape[0])
print correct_proportion
