import numpy as np
import os
from sklearn.decomposition import PCA

os.chdir('/Users/michael/Documents/brown/kobe/data')  # move to the directory where our data is stored

# grab data
npzfile = np.load('Flint_2012_e1.npz')
all_time = npzfile['all_time']
all_velocities = npzfile['all_velocities']
all_counts = npzfile['all_counts']
all_LFPs = npzfile['all_LFPs']
all_neural = np.vstack((all_counts, all_LFPs))

all_counts0 = np.sum(all_counts[:, 0:77920].reshape(196,-1, 20), axis=2)
all_velocities0 = np.mean(all_velocities[:, 0:77920].reshape(2,-1, 20), axis=2)
all_time0 = all_velocities0.shape[1]

pca = PCA(n_components=20, whiten=1)
all_neural0 = pca.fit_transform(all_counts0.T).T

np.savez_compressed('Flint_2012_e1_PCA00.npz', all_time=all_time0, all_neural=all_neural0, all_velocities=all_velocities0)
