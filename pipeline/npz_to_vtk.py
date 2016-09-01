import numpy as np
import evtk.hl as eh

data_file = np.load('./filter_run.npz')
all_particles = data_file['all_particles']
all_weights = data_file['all_weights']
all_est = data_file['all_est']
all_true = data_file['all_true']
all_particles_t = np.zeros((all_particles.shape[0], all_particles.shape[1], 3))

all_particles_t[:, :, 0:2] = all_particles*100
for t in range(all_particles.shape[0]):
    all_particles_t[t, :, 2] = t

xs = all_particles_t[:, :, 0].flatten()
ys = all_particles_t[:, :, 1].flatten()
zs = all_particles_t[:, :, 2].flatten()
ws = all_weights.flatten()

eh.pointsToVTK('./particle_data', xs, ys, zs, data={"weights": ws})

all_est_t = np.zeros((all_est.shape[0], 3))
all_est_t[:,0:2] = all_est*100
for t in range(all_est.shape[0]):
	all_est_t[t,2] = t	
	
all_est_t = np.repeat(all_est_t,2,axis=0)
all_est_t = all_est_t[1:-1,:]
	
xs = all_est_t[:,0].flatten()
ys = all_est_t[:,1].flatten()
zs = all_est_t[:,2].flatten()

eh.linesToVTK('./est_data', xs, ys, zs)

print all_est_t