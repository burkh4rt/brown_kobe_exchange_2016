import numpy as np
from evtk.hl import pointsToVTK

data_file = np.load('/Users/michael/Documents/brown/kobe/cythonized_particle_filter/filter_run.npz')
all_particles = data_file['all_particles']
all_weights = data_file['all_weights']
all_est = data_file['all_est']
all_true = data_file['all_true']
all_particles_t = np.zeros((all_particles.shape[0], all_particles.shape[1], 3))

all_particles_t[:, :, 0:2] = all_particles
for t in range(all_particles.shape[0]):
    all_particles_t[t, :, 2] = t

xs = all_particles_t[:, :, 0].flatten()
ys = all_particles_t[:, :, 1].flatten()
zs = all_particles_t[:, :, 2].flatten()
ws = all_weights.flatten()

pointsToVTK('./particle_data', xs, ys, zs, data={"weights": ws})
