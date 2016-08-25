import numpy as np

res_file = np.load('filter_run.npz')

all_particles = res_file['all_particles']
all_weights = res_file['all_weights']
all_est = res_file['all_est']
all_true = res_file['all_true']
