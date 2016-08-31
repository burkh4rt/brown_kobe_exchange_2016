
import numpy as np
cimport numpy as np

import cython
cimport cython

from libc.math cimport exp, sqrt
from cython.parallel cimport prange

cdef int n_test = 100
cdef int n_steps = 30
cdef int n_particles = 4

cdef int d_neural = 600
cdef int d_velocities = 2

# load Kalman parameters
param_file = np.load('/../kalman_estimates.npz')

cdef np.ndarray A_est = param_file['A_est']
cdef np.ndarray S_est = param_file['S_est']
cdef np.ndarray C_est = param_file['C_est']
cdef np.ndarray Q_est = param_file['Q_est']

cdef np.ndarray Q_est_inv = np.linalg.inv(Q_est)

data_file = np.load('/Users/michael/Documents/brown/kobe/data/Flint_2012_e1_PCA.npz')

cdef int all_time = data_file['all_time']
cdef np.ndarray all_velocities = data_file['all_velocities']
cdef np.ndarray all_neural = data_file['all_neural']
cdef int T = all_time - n_steps

cdef np.ndarray particles = np.random.multivariate_normal(np.zeros(d_velocities), S_est, n_particles)
cdef np.ndarray weights = np.ones((n_particles, 1), dtype=np.double)/n_particles
cdef np.ndarray observation = np.zeros((d_neural, 1))

#!python
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # turn off checks for division

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

#!python
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # turn off checks for division

cdef weight_update(np.ndarray[np.float64_t, ndim=2] particles, np.ndarray[np.float64_t, ndim=2] weights, np.ndarray[np.float64_t, ndim=2] observation, np.ndarray[np.float64_t, ndim=2] C_est, np.ndarray[np.float64_t, ndim=2] Q_est_inv):
    cdef unsigned int p, i, j
    cdef unsigned int n_particles = particles.shape[0]
    cdef unsigned int d_velocities = particles.shape[1]
    cdef unsigned int d_neural = C_est.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] new_weights = np.zeros((n_particles, 1))
    cdef double[:, :] diff = np.zeros((n_particles, d_neural))
    cdef double[:, :] dist = np.zeros((n_particles, d_neural))
    cdef double[:] log_weights = np.zeros(n_particles)
    # particle updates
    #
    for p in prange(n_particles, nogil=True):
        for i in range(d_velocities):
            for j in range(d_neural):
                diff[p, j] += C_est[i, j]*particles[p, i]
            diff[p, i] -= observation[i, 0]
        for i in range(d_neural):
            for j in range(d_neural):
                dist[p, i] += diff[p, j]*Q_est_inv[i,j]
            log_weights[p] += -0.5*dist[p, i]*diff[p, i]
    for p in range(n_particles):
        new_weights[p, 0] = log_weights[p]
    min_value = np.min(np.min(new_weights))
    for p in range(n_particles):
        new_weights[p, 0] = exp(new_weights[p, 0] - min_value)
    sum_value = np.sum(np.sum(new_weights))
    for p in range(n_particles):
        new_weights[p, 0] = new_weights[p, 0]/sum_value
    return new_weights

cdef np.ndarray all_particles = np.zeros((n_test, n_particles, d_velocities))
cdef np.ndarray all_weights = np.zeros((n_test, n_particles, 1))
cdef np.ndarray all_est = np.zeros((n_test, d_velocities))
cdef np.ndarray all_true = np.zeros((n_test, d_velocities))

for t in range(n_test):
    # grab observation
    observation = all_neural[:, t:t+30].flatten()[:, None]
    # resample particles
    particles, weights = resample(particles, weights)
    # update particle locations
    all_particles[t, :, :] = particles = np.matmul(A_est, particles.T).T + np.random.multivariate_normal(np.zeros(d_velocities), S_est, n_particles)
    # update weights
    all_weights[t, :, :] = weights = weight_update(particles, weights, observation, C_est, Q_est_inv)
    all_est[t, :] = np.matmul(weights.T, particles)
    all_true[t, :] = all_velocities[:, t+30]

np.savez('filter_run', all_particles=all_particles, all_weights=all_weights, all_est=all_est, all_true=all_true)

