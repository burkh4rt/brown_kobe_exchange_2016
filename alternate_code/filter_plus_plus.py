"""
run with:
mpiexec -n 4 python3 filter_plus_plus.py
"""

from mpi4py import MPI
# import tensorflow as tf
import numpy as np
import os
import time


# instantiate MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

npt = 100
n_particles = npt*size

# describe our data
n_test = 500
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

Q_est_inv = np.linalg.inv(Q_est)

# grab data on root process
if rank == 0:
    tic = time.clock()
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
particle = np.zeros((npt, d_velocities))
weight = np.ones((npt, 1))
particle_weight = np.hstack((particle, weight))

observation = np.zeros(d_neural)

if rank == 0:
    particles = np.random.multivariate_normal(np.zeros(2), S_est, n_particles)
    weights = np.ones((n_particles, 1))/n_particles
    particles_weights = np.hstack((particles, weights))

# store data
if rank == 0:
    all_particles = np.empty([n_test, n_particles, d_velocities])
    all_weights = np.empty([n_test, n_particles, 1])
    all_true = np.empty([n_test, d_velocities])
    all_est = np.empty([n_test, d_velocities])

for t in range(n_test):
    if rank == 0:
        # resampling step
        samples = np.random.multinomial(n_particles, weights.flatten())
        indices = np.repeat(np.arange(n_particles), samples.flatten())
        particles = particles[indices]
        weights = np.ones((n_particles, 1)) / n_particles
        particles_weights = np.hstack((particles, weights))
        # grab new observation
        observation = neural(np.arange(1) + t)

    # send out the particles to different processes
    comm.Scatter(particles_weights, particle_weight, root=0)
    comm.Bcast(observation, root=0)
    particle = particle_weight[:, :d_velocities]
    weight = particle_weight[:, d_velocities:]

    # update particle location
    particle = np.matmul(A_est, particle.T).T + np.random.multivariate_normal(np.zeros(d_velocities), S_est, npt)

    # update particle weight
    diff = np.matmul(C_est, particle.T).T - observation.flatten()
    log_weight = weight
    for p in range(npt):
        log_weight[p, ] = -0.5 * np.matmul(np.matmul(diff[p, :], Q_est_inv)[:, None].T, diff[p, ])
    particle_weight = np.hstack((particle, log_weight))

    comm.Barrier()

    # return
    comm.Gather(particle_weight, particles_weights)

    if rank == 0:
        all_particles[t, :, :] = particles = particles_weights[:, :d_velocities]
        log_weights = particles_weights[:, d_velocities:]
        weights = np.exp(log_weights - np.max(log_weights))
        all_weights[t, :, :] = weights = weights / np.sum(weights)
        all_est[t, :] = estimate = np.matmul(weights.T, particles)
        all_true[t, :] = true = velocities(np.arange(1) + t)
        print('est=', estimate, 'true=', true)

if rank == 0:
    np.savez('filter_run', all_particles=all_particles, all_weights=all_weights, all_est=all_est, all_true=all_true)
    toc = time.clock() - tic
    print('run_time=', toc, 'sec')

