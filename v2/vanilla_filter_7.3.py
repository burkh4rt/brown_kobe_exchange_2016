import numpy as np
import os

# name experimental parameters
n_steps = 6
n_particles = 1000

# model information
d_neural = 120
d_velocities = 2

os.chdir('/Users/michael/Documents/brown/kobe/data')

# load Kalman parameters
param_file = np.load('/Users/michael/Documents/brown/kobe/data/kalman_estimates0.npz')

A_est = param_file['A_est']
S_est = 5*param_file['S_est']

# load neural network parameters
param_file = np.load('neural_net_parameters0.npz')

f_hidden1_weights = param_file['f_hidden1_weights']
f_hidden1_biases = param_file['f_hidden1_biases']
f_hidden2_weights = param_file['f_hidden2_weights']
f_hidden2_biases = param_file['f_hidden2_biases']
f_output_weights = param_file['f_output_weights']
f_output_biases = param_file['f_output_biases']

cov_est = 0.1*param_file['cov_est']
cov_est_inv = np.linalg.inv(cov_est)

# gather data
data_file = np.load('/Users/michael/Documents/brown/kobe/data/Flint_2012_e1_PCA00.npz')
all_time = data_file['all_time']
all_velocities = data_file['all_velocities']
all_neural = data_file['all_neural']

n_test = 1000

T = int(all_time) - 6
del all_time

# instantiate particles and weights
particles = np.tile(all_velocities[:, 0], [n_particles,1])
weights = np.ones((n_particles, 1), dtype=np.double)/n_particles
observation = np.zeros((d_neural, 1))

# define a function to resample particles
def resample(particles, weights):
    n_particles = particles.shape[0]
    samples = np.random.multinomial(n_particles, weights.flatten())
    indices = np.repeat(np.arange(n_particles), samples)
    new_particles = particles[indices]
    new_weights = np.ones((n_particles, 1))/n_particles
    return new_particles, new_weights

# instantiate structures to store our results
all_particles = np.zeros((n_test, n_particles, d_velocities))
all_weights = np.zeros((n_test, n_particles, 1))
all_est = np.zeros((n_test, d_velocities))
all_true = np.zeros((n_test, d_velocities))

# loop over observations and run particle filter
for t in range(n_test):
    t0 = t + int(2*T / 3)
    # resample particles
    particles, weights = resample(particles, weights)
    # update particle locations
    all_particles[t, :, :] = particles = np.matmul(A_est, particles.T).T + \
                                         np.random.multivariate_normal(np.zeros(d_velocities), S_est, n_particles)
    # grab observation
    observation = all_neural[:, t0:t0+6].flatten()[:, None].T
    # update weights
    f_hidden1 = np.minimum(np.maximum(np.matmul(observation, f_hidden1_weights) + f_hidden1_biases, 0), 6)
    f_hidden2 = np.minimum(np.maximum(np.matmul(f_hidden1, f_hidden2_weights) + f_hidden2_biases, 0), 6)
    f_out = np.matmul(f_hidden2, f_output_weights) + f_output_biases
    print("f=", f_out)

    diff = f_out - particles
    log_weights = np.zeros((n_particles, 1))
    for p in range(n_particles):
        log_weights[p] = -0.5*np.matmul(np.matmul(diff[p].T, cov_est_inv), diff[p])
    weights = np.exp(log_weights-np.max(np.max(log_weights)))
    all_weights[t, :, :] = weights = weights/np.sum(np.sum(weights))
    print(np.hstack((weights, particles,diff)))
    all_est[t, :] = np.matmul(weights.T, particles)
    all_true[t, :] = all_velocities[:, t0+6]

mse = np.mean(np.mean(np.square(all_true-all_est)))
print(mse)
np.savez('filter_run0', all_particles=all_particles, all_weights=all_weights, all_est=all_est, all_true=all_true)
