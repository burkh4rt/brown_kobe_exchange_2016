import numpy as np

# load Kalman parameters
param_file = np.load('kalman_estimates.npz')

# for location update
A_est = param_file['A_est']
S_est = param_file['S_est']

# for weight update
C_est = param_file['C_est']
Q_est = param_file['Q_est']
