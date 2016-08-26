import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpa

res_file = np.load('filter_run.npz')

all_particles = res_file['all_particles']
all_weights = res_file['all_weights']
all_est = res_file['all_est']
all_true = res_file['all_true']

numpoints = all_particles.shape[1]
tot_time = all_particles.shape[0]

fig = plt.figure()
ax = plt.axes(xlim = (-0.15,0.15), ylim = (-0.15,0.15))

def main():
    ani = mpa.FuncAnimation(fig, scatplot, frames = xrange(tot_time))

    #Use FFmpeg to write animation to file
    #ffmpeg_writer = mpa.FFMpegWriter(fps = 10)
    #ani.save('particle_estimates.mp4', writer = ffmpeg_writer)

    plt.show()
    
    

def scatplot(time):
    fig.clear()
    x_model = all_particles[time,:,0]
    y_model = all_particles[time,:,1]
    c_model = np.chararray((numpoints,1), itemsize = 5)
    c_model[:] = 'red'
    s_model = all_weights[time,:,0]*1000

    x_est = all_est[time,0]
    y_est = all_est[time,1]
    c_est = 'blue'
    s_est = 20

    x_data = all_true[time,0]
    y_data = all_true[time,1]
    c_data = 'black'
    s_data = 20

    x1 = np.append(x_model,x_data)
    y1 = np.append(y_model,y_data)
    c1 = np.append(c_model,c_data)
    s1 = np.append(s_model,s_data)

    x = np.append(x1, x_est)
    y = np.append(y1, y_est)
    c = np.append(c1, c_est)
    s = np.append(s1, s_est)
   
#    scat.set_sizes(s)
#    scat.set_offsets(np.hstack((x,y)))
#    scat.set_color(c) 

    ax = plt.axes(xlim = (-0.15,0.15), ylim = (-0.15,0.15))
    scat = plt.scatter(x,y,c=c,s=s,edgecolor='None')
   
    return scat

print('particles metadata')
print(all_particles.shape)
print('weights metadata')
print(all_weights.shape)
print('est metadata')
print(all_est.shape)
print('true metadata')
print(all_true.shape)

main()


