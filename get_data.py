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

def main():
    fig = plt.figure()

    ani = mpa.FuncAnimation(fig, scatplot, frames = xrange(tot_time))

    plt.show()
    

def scatplot(time):
    x_model = all_particles[time,:,0]
    y_model = all_particles[time,:,1]
    c_model = np.chararray((numpoints,1), itemsize = 5)
    c_model[:] = 'red'
    s_model = all_weights[time,:]*100

    x_est = all_est[time,1]
    y_est = all_est[time,2]
    c_est = 'blue'
    s_model = 20

    x_data = all_true[time,1]
    y_data = all_true[time,2]
    c_data = 'black'
    s_data = 20
   
    scat = plt.scatter(np.vstack((x_model,x_data)),np.vstack((y_model,y_data))
			, s = np.vstack((s_model,s_data)) 
                        , c = np.vstack((c_model, c_data)))

    return scat

print('particles metadata')
print(all_particles.shape)
print('weights metadata')
print(all_weights.shape)
print('est metadata')
print(all_est.shape)
print('true metadata')
print(all_true.shape)

