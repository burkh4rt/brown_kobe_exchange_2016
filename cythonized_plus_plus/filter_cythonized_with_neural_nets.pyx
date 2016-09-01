#cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np

import cython
cimport cython

from libc.math cimport exp
from cython.parallel cimport prange

cdef int n_test = 100
cdef int n_steps = 30
cdef int n_particles = 4

cdef int d_neural = 600
cdef int d_velocities = 2

# load Kalman parameters
param_file = np.load('../kalman_estimates.npz')

cdef np.ndarray A_est = param_file['A_est']
cdef np.ndarray S_est = param_file['S_est']
cdef double[:, :] Q_est_inv = np.eye(3)

# load neural network parameters
param_file = np.load('../neural_net_parameters.npz')

cdef np.ndarray f_hidden1_weights = param_file['f_hidden1_weights']
cdef np.ndarray f_hidden1_biases = param_file['f_hidden1_biases']
cdef np.ndarray f_hidden2_weights = param_file['f_hidden2_weights']
cdef np.ndarray f_hidden2_biases = param_file['f_hidden2_biases']
cdef np.ndarray f_hidden3_weights = param_file['f_hidden3_weights']
cdef np.ndarray f_hidden3_biases = param_file['f_hidden3_biases']

cdef float[:,:] g_hidden1_weights = param_file['g_hidden1_weights']
cdef float[:] g_hidden1_biases = param_file['g_hidden1_biases']
cdef float[:,:] g_hidden2_weights = param_file['g_hidden2_weights']
cdef float[:] g_hidden2_biases = param_file['g_hidden2_biases']
cdef float[:,:] g_output_weights = param_file['g_output_weights']
cdef float[:] g_output_biases = param_file['g_output_biases']

data_file = np.load('../Flint_2012_e1_PCA.npz')

cdef int all_time = data_file['all_time']
cdef np.ndarray all_velocities = data_file['all_velocities']
cdef np.ndarray all_neural = data_file['all_neural']
cdef int T = all_time - n_steps

cdef np.ndarray particles = np.random.multivariate_normal(np.zeros(d_velocities), S_est, n_particles)
cdef np.ndarray weights = np.ones((n_particles, 1), dtype=np.double)/n_particles
cdef np.ndarray observation = np.zeros((d_neural, 1))

cdef resample(np.ndarray[np.float64_t, ndim=2] particles, np.ndarray[np.float64_t, ndim=2] weights):
    cdef unsigned int n_particles = particles.shape[0]
    cdef long[:] samples = np.random.multinomial(n_particles, weights.flatten())
    cdef long[:] indices = np.repeat(np.arange(n_particles), samples)
    cdef unsigned int p
    cdef np.ndarray[np.float64_t, ndim=2] new_particles = particles
    for p in range(n_particles):
        new_particles[p, :] = particles[ indices[p], :]
    cdef np.ndarray[np.float64_t, ndim=2] new_weights = np.ones((n_particles, 1))/n_particles
    return new_particles, new_weights

cdef np.ndarray f_hidden1, f_hidden2, f_features
cdef int size1 = 30
cdef int size2 = 10
cdef int size_out = 3
cdef double[:,:] g_hidden1 = np.zeros((n_particles, size1))
cdef double[:,:] g_hidden2 = np.zeros((n_particles, size2))
cdef double[:,:] g_out = np.zeros((n_particles, size_out))
cdef double[:,:] c_particles = particles
cdef double[:,:] c_features = np.zeros((3, 1))
cdef int p, i, j
cdef double[:,:] diffs = np.zeros((n_particles,size_out))
cdef double[:,:] dists = np.zeros((n_particles,size_out))
cdef double[:] log_weights = np.zeros(n_particles)
cdef double max_weight = 0
cdef double sum_weight = 0 

for t in range(n_test):
    # grab observation
    observation = all_neural[:, t:t+30].flatten()[:, None]
    f_hidden1 = np.minimum(np.maximum(np.matmul(observation.T, f_hidden1_weights) + f_hidden1_biases, 0), 6)
    f_hidden2 = np.minimum(np.maximum(np.matmul(f_hidden1, f_hidden2_weights) + f_hidden2_biases, 0), 6)
    f_features = np.matmul(f_hidden2, f_hidden3_weights) + f_hidden3_biases
    c_features = f_features.T
    log_weights[p] = 0
    for p in prange(n_particles, nogil=True):
        for i in range(size1):
            g_hidden1[p, i] = 0
        for i in range(size1):
            for j in range(d_velocities):
                g_hidden1[p, i] += c_particles[p, j] * g_hidden1_weights[j,i]
            g_hidden1[p, i] += g_hidden1_biases[i]
            g_hidden1[p, i] = min(max(g_hidden1[p, i], 0),6)
        for i in range(size2):
            g_hidden2[p, i] = 0
        for i in range(size2):
            for j in range(size1):
                g_hidden2[p, i] += g_hidden1[p, j] * g_hidden2_weights[i,j]
            g_hidden2[p, i] += g_hidden2_biases[i]
            g_hidden2[p, i] = min(max(g_hidden2[p, i], 0),6)
        for i in range(size_out):
            g_out[p, i] = 0
        for i in range(size_out):
            for j in range(size2):
                g_out[p, i] += g_hidden2[p, j] * g_output_weights[i,j]
            g_out[p, i] += g_output_biases[i]
            g_out[p, i] = min(max(g_out[p, i], 0),6)
        for i in range(size_out):
            diffs[p, i] = g_out[p, i] - c_features[i, 0]
            dists[p, i] = 0
            for j in range(size_out):
                dists[p, i] += diffs[p, j]*Q_est_inv[i, j]
            log_weights[p] += -0.5*dists[p, i]*diffs[p, i]
        max_weight = log_weights[0]
        max_weight = max(max_weight, log_weights[p])
        log_weights[p] = exp(log_weights[p]-max_weight)
        sum_weight = 0
        sum_weight += log_weights[p]
        log_weights[p] /= sum_weight
