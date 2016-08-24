

import numpy as np
import os

n_test = 1000
n_steps = 30
batch_size = 1000
d_neural = 600
d_velocities = 2

# grab data
os.chdir('/Users/michael/Documents/brown/kobe/data')
npzfile = np.load('Flint_2012_e1_PCA.npz')
all_time = npzfile['all_time']
all_velocities = npzfile['all_velocities']
all_neural = npzfile['all_neural']

T = int(all_time) - 30
del all_time


def neural(ind):
    neur = np.zeros((ind.size, d_neural))
    for i0 in range(ind.size):
        s_idx = range(ind[i0], ind[i0] + 30)
        neur[i0, :] = all_neural[:, s_idx].flatten()
    return neur


def velocities(ind):
    return all_velocities[:, ind + 29].T

X = neural(np.arange(5000))
y = velocities(np.arange(5000))

A_est = np.linalg.lstsq(y[:-1, ], y[1:, ])[0].T
S_est = np.cov(y[1:, ].T-np.matmul(A_est, y[:-1, ].T))

C_est = np.linalg.lstsq(y, X)[0].T
Q_est = np.cov(X.T-np.matmul(C_est, y.T))

np.savez('kalman_estimates', A_est, S_est, C_est, Q_est)


