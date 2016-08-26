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
ax = plt.axes(xlim = (-0.5,0.5), ylim = (-0.5,0.5))

def main():
    ani = mpa.FuncAnimation(fig, scatplot, frames = xrange(tot_time))

    plt.show()
    
#def scatinit():
#    scat = plt.scatter([],[], edgecolor = 'None')
#    return scat 
    

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

    x = np.append(x_model,x_data)
    y = np.append(y_model,y_data)
    c = np.append(c_model,c_data)
    s = np.append(s_model,s_data)
   
#    scat.set_sizes(s)
#    scat.set_offsets(np.hstack((x,y)))
#    scat.set_color(c) 

    ax = plt.axes(xlim = (-0.1,0.1), ylim = (-0.1,0.1))
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

