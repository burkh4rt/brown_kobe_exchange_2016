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
del npzfile, all_counts, all_LFPs

pca = PCA(n_components=20, whiten=1)
all_neural = pca.fit_transform(all_neural.T).T

np.savez_compressed('Flint_2012_e1_PCA.npz', all_time=all_time, all_neural=all_neural, all_velocities=all_velocities)
