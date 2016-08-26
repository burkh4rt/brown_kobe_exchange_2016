"""
run with:
mpiexec -n 4 python3 mpi_filter_alt_vers.py
"""

from mpi4py import MPI
#import tensorflow as tf
import numpy as np
import os

# instantiate MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# describe our data
n_test = 10
n_steps = 30

d_neural = 600
d_velocities = 2
npt = 5

# move to where data is
os.chdir('.')

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
wide_particles_weights = None
particle = np.zeros(d_velocities*npt)
weight = np.ones(npt)
log_weight = np.zeros(npt)
particle_weight = np.hstack((particle, weight))

observation = np.zeros(d_neural)

if rank == 0:
    particles = np.random.multivariate_normal(np.zeros(d_velocities), S_est, size*npt)
    weights = np.ones((size*npt, 1))/(size*npt)
    particles_weights = np.hstack((particles, weights))

# store data
if rank == 0:		
    all_particles = np.empty([n_test, npt*size, d_velocities])		
    all_weights = np.empty([n_test, npt*size, 1])		
    all_true = np.empty([n_test, d_velocities])		
    all_est = np.empty([n_test, d_velocities])

for t in range(n_test):
    if rank == 0:
        # resampling step
        samples = np.random.multinomial(size*npt, weights.flatten())
        indices = np.repeat(np.arange(size*npt), samples.flatten())
        particles = particles[indices]
        weights = np.ones((size*npt, 1)) / (size*npt)
        particles_weights = np.hstack((particles, weights))
        # grab new observation
        observation = neural(np.arange(1) + t)

        #TODO: TEST
        wide_particles = particles.reshape((size,d_velocities*npt))
        wide_weights = weights.reshape((size,npt))
        wide_particles_weights = np.hstack((wide_particles,wide_weights))

    # send out the particles and observation to different processes
    comm.Scatter(wide_particles_weights, particle_weight, root=0)
    comm.Bcast(observation, root=0)
    
    # unpack particles and weights
    particle = particle_weight[:d_velocities*npt, ]
    weight = particle_weight[d_velocities*npt:, ]


    #particles subset is set of particles handled by thread
    particle_subset = particle.reshape((npt,d_velocities))

    # update particle weight on individual processes
    for i in range(npt):
        particle_subset[i,:] = np.random.multivariate_normal(np.matmul(particle_subset[i,:],A_est.T), S_est)
        diff = np.matmul(C_est, particle_subset[i,:].T).flatten() - observation.flatten()
        log_weight[i] = -0.5 * np.matmul(np.matmul(diff.T, np.linalg.pinv(Q_est)), diff)

    particle_weight = np.hstack((particle, log_weight))

    comm.Barrier()

    # return particle weights to root
    comm.Gather(particle_weight, wide_particles_weights)

    if rank == 0:
        particles = wide_particles_weights[:, :(d_velocities*npt)]
        particles = particles.reshape((size*npt, d_velocities))
        log_weights = wide_particles_weights[:, (npt*d_velocities):]
        log_weights = log_weights.reshape((npt*size),1)
        weights = np.exp(log_weights - np.max(log_weights))
        weights = weights / np.sum(weights)
        estimate = np.matmul(weights.T, particles)

        all_particles[t, :, :] = particles
        all_weights[t, :, :] = weights
        all_est[t, :] = estimate
        all_true[t, :] = true = velocities(np.arange(1) + t)
        print('est=', estimate, 'true=', true)		
	
if rank == 0:		
    np.savez('filter_run', all_particles=all_particles, all_weights=all_weights, all_est=all_est, all_true=all_true)
