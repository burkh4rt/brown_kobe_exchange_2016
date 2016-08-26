import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation

# tell plt where FFmpeg lives on your machine
# install FFmpeg if necessary with "brew install ffmpeg"
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

os.chdir('/Users/michael/Documents/brown/kobe/data')
filter_data = np.load('filter_run.npz')

all_particles = filter_data['all_particles']  # locations as (n_test, n_particles, 2) np array
all_weights = filter_data['all_weights']  # weights as (n_test, n_particles, 1) np array
all_est = filter_data['all_est']  # filter estimates as (n_test, 2) np array
all_true = filter_data['all_true']  # true realizations as (n_test, 2) np array

# precalculate fixed window size so we don't miss anything
min_part = np.min(all_particles, axis=(0, 1))
max_part = np.max(all_particles, axis=(0, 1))
min_true = np.min(all_true, axis=0)
max_true = np.max(all_true, axis=0)
min_view = np.min(np.vstack((min_part, min_true)), axis=0)
max_view = np.max(np.vstack((max_part, max_true)), axis=0)

# initialize figure and window size
fig = plt.figure()
ax = plt.axes(xlim=(min_view[0], max_view[0]), ylim=(min_view[1], max_view[1]))

# construct empty plots (to later be filled with data) for particles, estimates, & true values
plot_particles = ax.scatter([], [], c='red', s=100, label='particles', alpha=0.7, edgecolor='None')
plot_estimates = ax.scatter([], [], c='green', s=100, label='estimate', alpha=1., edgecolor='None')
plot_true = ax.scatter([], [], c='blue', s=100, label='realization', alpha=1., edgecolor='None')

# legend
ax.legend()


# define a function that populates our plots with the data from a given time step
def print_step(t):
    plot_particles.set_sizes(700 * all_weights[t, :, 0])
    plot_particles.set_offsets(all_particles[t, :, :])
    plot_estimates.set_offsets(all_est[t, :])
    plot_true.set_offsets(all_true[t, :])


# create the animation
anim = matplotlib.animation.FuncAnimation(fig, print_step, frames=range(250))

# use FFmpeg to write animation to file
ffmpeg_writer = matplotlib.animation.FFMpegWriter(fps=10)
anim.save('particle_positions.mp4', writer=ffmpeg_writer)

# uncomment to get a live plot
# plt.show()
