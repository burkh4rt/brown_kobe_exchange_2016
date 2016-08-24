"""
run with:
mpiexec -n 4 python3 mpi_tf_filter.py
"""

from mpi4py import MPI
import tensorflow as tf
import numpy as np
import scipy as sp
import os

###### load A and S py
# load Kalman parameters
param_file = np.load('kalman_estimates.npz')

# for location update
A_est = param_file['A_est']
S_est = param_file['S_est']

# for weight update
C_est = param_file['C_est']
Q_est = param_file['Q_est']
##########

#TODO: Update for your local environment
file_location = 'C:/Users/Ankan/Documents/Kobe_2016/Project'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#name = MPI.Get_processor_name()

n_test = 1000
n_steps = 30
batch_size = 1000
d_neural = 600
d_velocities = 2

# grab data
if rank == 0:
    # grab data
    os.chdir(file_location)
    npzfile = np.load('Flint_2012_e1_PCA.npz')
    all_time = npzfile['all_time']
    all_velocities = npzfile['all_velocities']
    all_neural = npzfile['all_neural']

    T = int(all_time) - 30
    del all_time

    def neural(ind):
        neur = np.zeros((ind.size, d_neural))
        for i0 in range(ind.size):
            s_idx = range(int(ind[i0]), int(ind[i0]) + 30)
            neur[i0, :] = all_neural[:, s_idx].flatten()
        return neur

    def velocities(ind):
        return all_velocities[:, ind + 29].T

# initialize particles
particles = None
weights = None
particle = np.zeros(d_velocities)   #each particle is a velocity.
weight = np.ones(1)                 
particle_weight = np.hstack((particle, weight))

if rank == 0:
    particles = np.random.multivariate_normal(np.zeros(2), np.eye(2), size)
    weights = np.ones((size, 1))/size
    particles_weights = np.hstack((particles, weights))     #dim 3 horizontal np_array


comm.Scatter(particles_weights, particle_weight)
particle = particle_weight[:d_velocities, ]
weight = particle_weight[d_velocities:, ]

observation = np.zeros(d_neural)

for t in range(n_test):
    if rank == 0:
        # Update observations
        ind = t*np.ones(1)
        observation = neural(ind)

        #Resample #TODO: parallelize
        particle_resampling = np.random.multinomial(1, weights.flatten(), size)
        particles = np.matmul(particle_resampling, particles)
        weights = np.ones((size,1))/size
        particles_weights = np.hstack((particles,weights))

    #Send resampling and observations to other threads
    comm.Bcast(observation, root=0)        
    comm.Scatter(particle_weights, particle_weight)

    #Update resampled particles and uniform weights
    particle = particle_weight[:d_velocities, ]
    weight = particle_weight[d_velocities:, ]

    #Update particles with time
    particle.T = np.matmul(A_est, particle.T) + np.random.normal(np.zeros(2), S_est);

    #Update weights
    weight = sp.stats.multivariate_normal.pdf(observation.T,
                                              mean = np.matmul(C_est, particle.T),
                                              cov = Q_est)
    particle_weight = np.hstack((particle,weight))

    #Pass all weights and particles to root
    comm.Gather(particle_weight, particles_weights)
    particles = particles_weights[:d_velocities, ]
    weights = particles_weights[d_velocities:, ]

    #Renormalize weights
    weights = weights/np.sum(weights)

    #TODO: fix individual weights










