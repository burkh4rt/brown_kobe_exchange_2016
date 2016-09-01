import numpy as np

#Read data
npzfile = np.load('../pipeline/filter_run.npz')
all_true = npzfile['all_true']
all_est = npzfile['all_est']
all_particles = npzfile['all_particles']
all_weights = npzfile['all_weights']

#Get time array
time = all_true.shape[0]
all_time = np.reshape(np.arange(time), (time,1))
all_time = all_time/1000

#Get size array
normal_size = 20
true_size = np.reshape(np.repeat(20,time), (time,1))

#Get csv for plotting true
plot_true = np.hstack((all_true,all_time))
plot_true = np.hstack((plot_true, true_size))

plot_true = np.vstack((np.arange(4), plot_true))

np.savetxt('filter_run_true.csv', plot_true, delimiter = ',', newline = '\n', fmt = '%1.4e')
