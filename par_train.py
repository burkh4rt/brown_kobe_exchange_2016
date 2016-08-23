from mpi4py import MPI
import tensorflow as tf
import numpy as np
import time
import math
import sys
import getopt
import resource
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

full_layers = [3]
train_batch = 10
epochs = 30
learning_rate = 0.1
input_shape = [600]
output_size = 2
threads = size
inter_threads = 0
intra_threads = 0
filename = None
valid_pct = 0.1
test_pct = 0.1
stop_time = -1
epoch = 0
error_batch = False
merge_every = 0
top = 1

if 0 == inter_threads:
    inter_threads = threads
if 0 == intra_threads:
    intra_threads = threads

# grab data
os.chdir('/Users/michael/Documents/brown/kobe/data')
npzfile = np.load('Flint_2012_e1_PCA.npz')
all_time = npzfile['all_time']
all_velocities = npzfile['all_velocities']
all_neural = npzfile['all_neural']

T = int(all_time) - 30
del all_time

d_neural = 30 * all_neural.shape[0]
d_velocities = all_velocities.shape[0]


def neural(ind):
    neur = np.zeros((ind.size, d_neural))
    for i0 in range(ind.size):
        s_idx = range(ind[i0], ind[i0] + 30)
        neur[i0, :] = all_neural[:, s_idx].flatten()
    return neur


def velocities(ind):
    return all_velocities[:, ind + 29].T


full_dat = neural(np.arange(3000))
full_lab = velocities(np.arange(3000))
valid_dat = neural(np.arange(3000, 4000))
valid_lab = velocities(np.arange(3000, 4000))
test_dat = neural(np.arange(4000, 5000))
test_lab = velocities(np.arange(4000, 5000))

print(full_dat.shape)
print(full_lab.shape)
print(valid_dat.shape)

input_size = 1
for i in input_shape:
    input_size *= i


# set up network


def weight_variable(shape, saved_state, index):
    if saved_state is None:
        initial = tf.truncated_normal(shape, stddev=0.1)
    else:
        initial = saved_state[0][index]
    return tf.Variable(initial)


def bias_variable(shape, saved_state, index):
    if saved_state is None:
        initial = tf.constant(0.1, shape=shape)
    else:
        initial = saved_state[1][index]
    return tf.Variable(initial)


def create_full_layer(in_size, out_size, layer_list, weight_list,
                      bias_list, saved_state):
    if saved_state is None:
        weight_list.append(tf.Variable(
            tf.random_normal([in_size, out_size], stddev=1.0 / in_size)))
        bias_list.append(tf.Variable(
            tf.random_normal([out_size], stddev=1.0 / in_size)))
    else:
        index = len(weight_list)
        weight_list.append(tf.Variable(saved_state[0][index]))
        bias_list.append(tf.Variable(saved_state[1][index]))
    temp_w = len(weight_list)
    temp_b = len(bias_list)
    temp_l = len(layer_list)
    layer_list.append(tf.nn.sigmoid(tf.matmul(layer_list[temp_l - 1], weight_list[temp_w - 1]) + bias_list[temp_b - 1]))


time_global_start = time.time()


def populate_graph(
        full_layers,
        learning_rate,
        input_shape,
        saved_state):
    weights = []
    biases = []
    layers = []

    x = tf.placeholder(tf.float32, [None, input_size])
    y_ = tf.placeholder(tf.float32, [None, output_size])

    layers.append(x)
    layers.append(tf.reshape(x, [-1] + input_shape))

    full_layers = [input_size] + full_layers
    for i in range(len(full_layers) - 1):
        create_full_layer(full_layers[i], full_layers[i + 1], layers,
                          weights, biases, saved_state)
    if saved_state is None:
        W = tf.Variable(tf.random_normal([full_layers[-1], output_size], stddev=1.0 / full_layers[-1]))
        b = tf.Variable(tf.random_normal([output_size], stddev=1.0 / full_layers[-1]))
    else:
        index = len(weights)
        W = tf.Variable(saved_state[0][index])
        b = tf.Variable(saved_state[1][index])
    weights.append(W)
    biases.append(b)

    w_holder = [tf.placeholder(tf.float32, w.get_shape()) for w in weights]
    b_holder = [tf.placeholder(tf.float32, b.get_shape()) for b in biases]
    w_assign = [w.assign(p) for w, p in zip(weights, w_holder)]
    b_assign = [b.assign(p) for b, p in zip(biases, b_holder)]

    y = tf.matmul(layers[-1], W) + b

    mse_loss = tf.reduce_mean(tf.squared_difference(y_, y), name='mse')

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse_loss)
    init = tf.initialize_all_variables()
    sess = tf.Session(
        config=tf.ConfigProto(
            inter_op_parallelism_threads=inter_threads,
            intra_op_parallelism_threads=intra_threads))

    accuracy_mse = tf.reduce_sum(tf.squared_difference(y_, y))

    sess.run(init)

    ops = {
        "sess": sess,
        "x": x,
        "y_": y_,
        "weights": weights,
        "biases": biases,
        "w_holder": w_holder,
        "b_holder": b_holder,
        "w_assign": w_assign,
        "b_assign": b_assign,
        "train_step": train_step,
        "cross_entropy": mse_loss,
        "accuracy": accuracy_mse,
    }

    return ops


def run_graph(
        data,
        labels,
        train_batch,
        ops,
        saved_state):
    global time_global_start
    global epoch
    global max_accuracy_encountered_value
    global max_accuracy_encountered_epoch
    global max_accuracy_encountered_time

    time_epoch_start = time.time()
    time_comm = 0.0

    sess = ops["sess"]
    x = ops["x"]
    y_ = ops["y_"]
    weights = ops["weights"]
    biases = ops["biases"]
    w_holder = ops["w_holder"]
    b_holder = ops["b_holder"]
    w_assign = ops["w_assign"]
    b_assign = ops["b_assign"]
    train_step = ops["train_step"]
    cross_entropy = ops["cross_entropy"]
    accuracy = ops["accuracy"]

    # use saved state to assign saved weights and biases
    if saved_state is not None:
        feed_dict = {}
        for d, p in zip(saved_state[0], w_holder):
            feed_dict[p] = d
        for d, p in zip(saved_state[1], b_holder):
            feed_dict[p] = d
        sess.run(w_assign + b_assign, feed_dict=feed_dict)

    number_of_batches = int(len(data) / train_batch)
    time_this = time.time()
    min_batches = comm.allreduce(number_of_batches, MPI.MIN)
    time_comm += time.time() - time_this
    if number_of_batches == 0:
        number_of_batches = 1

    for i in range(number_of_batches):
        lo = i * train_batch
        hi = (i + 1) * train_batch
        batch_xs = data[lo:hi]
        batch_ys = labels[lo:hi]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if (i < min_batches) and (merge_every >= 1) and (i % merge_every == 0):
            time_this = time.time()
            r_weights = sess.run(weights)
            r_biases = sess.run(biases)
            for r in r_weights:
                comm.Allreduce(MPI.IN_PLACE, r, MPI.SUM)
                r /= size
            for r in r_biases:
                comm.Allreduce(MPI.IN_PLACE, r, MPI.SUM)
                r /= size
            feed_dict = {}
            for d, p in zip(r_weights, w_holder):
                feed_dict[p] = d
            for d, p in zip(r_biases, b_holder):
                feed_dict[p] = d
            sess.run(w_assign + b_assign, feed_dict=feed_dict)
            time_comm += time.time() - time_this

    # average as soon as we're done with all batches so the error and
    # accuracy reflect the current epoch
    time_this = time.time()
    r_weights = sess.run(weights)
    r_biases = sess.run(biases)
    for r in r_weights:
        comm.Allreduce(MPI.IN_PLACE, r, MPI.SUM)
        r /= size
    for r in r_biases:
        comm.Allreduce(MPI.IN_PLACE, r, MPI.SUM)
        r /= size
    time_comm += time.time() - time_this
    feed_dict = {}
    for d, p in zip(r_weights, w_holder):
        feed_dict[p] = d
    for d, p in zip(r_biases, b_holder):
        feed_dict[p] = d
    sess.run(w_assign + b_assign, feed_dict=feed_dict)
    time_comm += time.time() - time_this

    sum_error = 0.0
    if error_batch:
        for i in range(number_of_batches):
            lo = i * train_batch
            hi = (i + 1) * train_batch
            batch_xs = data[lo:hi]
            batch_ys = labels[lo:hi]
            sum_error += sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
    else:
        sum_error = sess.run(cross_entropy, feed_dict={x: data, y_: labels})
    time_this = time.time()
    sum_error_all = comm.allreduce(sum_error)
    time_comm += time.time() - time_this
    batch_mse = 0.0
    if error_batch:
        test_batch_count = len(test_dat) / train_batch
        if test_batch_count == 0:
            test_batch_count = 1
        for i in range(test_batch_count):
            lo = i * train_batch
            hi = (i + 1) * train_batch
            batch_xs = test_dat[lo:hi]
            batch_ys = test_lab[lo:hi]
            batch_mse += sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    else:
        batch_mse = sess.run(accuracy, feed_dict={x: test_dat, y_: test_lab})
    time_this = time.time()
    batch_mse = comm.allreduce(batch_mse, MPI.SUM)
    acc_count = comm.allreduce(len(test_dat), MPI.SUM)
    batch_mse = float(batch_mse) / acc_count
    time_comm += time.time() - time_this

    time_all = time.time() - time_epoch_start

    if 0 == rank:
        print("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f")(
            epoch + 1,
            time_all,
            time.time() - time_global_start,
            batch_mse,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0,
            time_comm / time_all,
            sum_error_all,
        )
    sys.stdout.flush()

    return r_weights, r_biases


if 0 == rank:
    print("epoch,etime,ctime,accuracy,MB_mem,time_comm,error")

data_threshold = int(len(full_dat) / 2)
active_dat = full_dat
active_lab = full_lab
inactive_dat = np.empty([0] + list(full_dat.shape[1:]), full_dat.dtype)
inactive_lab = np.empty([0] + list(full_lab.shape[1:]), full_lab.dtype)

if stop_time > 0:
    saved_state = None
    ops = populate_graph(
        full_layers,
        learning_rate,
        input_shape,
        saved_state)
    while stop_time > (time.time() - time_global_start):
        saved_state = run_graph(
            active_dat,
            active_lab,
            train_batch,
            ops,
            saved_state)
        epoch += 1

else:
    saved_state = None
    ops = populate_graph(
        full_layers,
        learning_rate,
        input_shape,
        saved_state)
    for epoch in range(epochs):
        saved_state = run_graph(
            active_dat,
            active_lab,
            train_batch,
            ops,
            saved_state)
