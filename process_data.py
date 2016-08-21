import scipy.io as sio
import scipy.interpolate as sp_interp
import numpy as np
import os

os.chdir('/Users/michael/Documents/brown/kobe/data')  # move to the directory where our data is stored
data = sio.loadmat('Flint_2012_e1.mat')  # read matlab data file

# grab & define basic data specs
n_trials = data['Subject'][0, 0]['Trial'].size
n_neurons = max([data['Subject'][0, 0]['Trial'][trial, 0]['Neuron']['Spike'].size for trial in range(n_trials)])
n_LFPs = 95

# pre-allocate data storage structures
all_time = sum(np.squeeze(data['Subject'][0, 0]['Trial'][trial, 0]['Time']).size for trial in range(n_trials))
all_counts = np.empty([n_neurons, all_time])
all_velocities = np.empty([2, all_time])
all_LFPs = np.empty([20*n_LFPs, all_time])

# grab data trial-by-trial and populate our data structures
idx0 = 0
for t in range(n_trials):
    clock_trial = np.squeeze(data['Subject'][0, 0]['Trial'][t, 0]['Time'])
    all_counts_trial = np.empty([n_neurons, clock_trial.size])
    idx_trial = range(idx0, idx0+clock_trial.size)
    all_velocities[:, idx_trial] = data['Subject'][0, 0]['Trial'][t, 0]['HandVel'][:, 0:2].T
    for n in range(n_neurons):
        neuron_spikes_trial = np.squeeze(data['Subject'][0, 0]['Trial'][t, 0]['Neuron']['Spike'][n, 0])
        neuron_spikes_idx = np.digitize(neuron_spikes_trial, clock_trial)
        neuron_spikes_count = np.bincount(np.ndarray.flatten(neuron_spikes_idx), minlength=clock_trial.size)
        all_counts[n, idx_trial] = neuron_spikes_count
    for n in range(n_LFPs):
        single_LFP_trial = np.squeeze(data['Subject'][0, 0]['Trial'][t, 0]['Neuron']['LFP'][n, 0])
        single_LFP_tck = sp_interp.splrep(np.arange(single_LFP_trial.size)/single_LFP_trial.size, single_LFP_trial)
        LFP_eval_pts = np.reshape(np.arange(20*clock_trial.size)/(20*clock_trial.size), [20, clock_trial.size], order='F')
        all_LFPs[20*n:20*(n+1), idx_trial] = sp_interp.splev(LFP_eval_pts, single_LFP_tck)
    idx0 += clock_trial.size

np.savez_compressed('Flint_2012_e1.npz', all_time=all_time, all_counts=all_counts, all_velocities=all_velocities,
                    all_LFPs=all_LFPs)

# exec(open('process_data.py').read())
# get data back with: npzfile = np.load('Flint_2012_e1.npz')
