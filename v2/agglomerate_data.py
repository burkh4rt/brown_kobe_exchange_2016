import numpy as np
import os
from sklearn.decomposition import PCA

os.chdir('/Users/michael/Documents/brown/kobe/data')

npzfile = np.load('Flint_2012_e1.npz')
all_time = npzfile['all_time']
all_velocities = npzfile['all_velocities']
all_counts = npzfile['all_counts'][0:180,:]

for i in range(2,6):
    npzfile_string = 'Flint_2012_e{0}.npz'.format(i)
    npzfile = np.load(npzfile_string)
    all_velocities = np.hstack((all_velocities,npzfile['all_velocities']))
    all_counts = np.hstack((all_counts, npzfile['all_counts'][0:180,:]))

all_counts0 = np.sum(all_counts[:, 0:319100].reshape(180,-1, 20), axis=2)
all_velocities0 = np.mean(all_velocities[:, 0:319100].reshape(2,-1, 20), axis=2)
all_time0 = all_velocities0.shape[1]

pca = PCA(n_components=20, whiten=1)
all_neural0 = pca.fit_transform(all_counts0.T).T

np.savez_compressed('Flint_2012_e1_PCA00.npz', all_time=all_time0, all_neural=all_neural0, all_velocities=all_velocities0)
