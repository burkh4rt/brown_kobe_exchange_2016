"""
run with:
mpiexec -n 4 python3 mpi_tf_filter.py
"""

from mpi4py import MPI
import tensorflow as tf
import numpy as np
import os
from scipy.stats import multivariate_normal as mvn

# instantiate MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# describe our data
n_test = 5000
n_steps = 30

d_neural = 600
d_velocities = 2

# move to where data is
os.chdir('/Users/michael/Documents/brown/kobe/data')

# load Kalman parameters
param_file = np.load('kalman_estimates.npz')
A_est = param_file['A_est']
S_est = param_file['S_est']
C_est = param_file['C_est']
Q_est = param_file['Q_est']

# grab data on root process
if rank == 0:
    # grab data
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

# initialize particles
particles = None
weights = None
particles_weights = None
particle = np.zeros(d_velocities)
weight = np.ones(1)
particle_weight = np.hstack((particle, weight))

observation = np.zeros(d_neural)

if rank == 0:
    particles = np.random.multivariate_normal(np.zeros(2), S_est, size)
    weights = np.ones((size, 1))/size
    particles_weights = np.hstack((particles, weights))

for t in range(n_test):
    if rank == 0:
        # resampling step
        samples = np.random.multinomial(size, weights.flatten())
        indices = np.repeat(np.arange(size), samples.flatten())
        particles = particles[indices]
        weights = np.ones((size, 1)) / size
        particles_weights = np.hstack((particles, weights))
        # grab new observation
        observation = neural(np.arange(1) + t)

    # send out the particles to different processes
    comm.Scatter(particles_weights, particle_weight, root=0)
    comm.Bcast(observation, root=0)
    particle = particle_weight[:d_velocities, ]
    weight = particle_weight[d_velocities:, ]

    # update particle location
    particle = np.random.multivariate_normal(np.matmul(A_est, particle), S_est)

    # update particle weight
    diff = np.matmul(C_est, particle) - observation.flatten()
    log_weight = -0.5 * np.matmul(diff.T, diff)
    particle_weight = np.hstack((particle, log_weight))

    comm.Barrier()

    # return
    comm.Gather(particle_weight, particles_weights)

    if rank == 0:
        particles = particles_weights[:, :d_velocities]
        log_weights = particles_weights[:, d_velocities:]
        weights = np.exp(log_weights - np.max(log_weights))
        weights = weights / np.sum(weights)
        estimate = np.matmul(weights.T, particles)
        print('est=', estimate, 'true=', velocities(np.arange(1) + t))

