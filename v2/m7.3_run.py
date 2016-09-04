import tensorflow as tf
import numpy as np
from math import sqrt
import os

# grab data
npzfile = np.load('/Users/michael/Documents/brown/kobe/data/Flint_2012_e1_PCA00.npz')
all_time = npzfile['all_time']
all_velocities = npzfile['all_velocities']
all_neural = npzfile['all_neural']

os.chdir('/Users/michael/Documents/brown/kobe/data')

"""
data is sampled every 0.01s;
the paper says the neural data 200-300ms beforehand is most informative
so we need the previous 30 observations of neural data for each velocity update
"""

T = int(all_time) - 6
del all_time

d_neural = 6 * 20
d_velocities = all_velocities.shape[0]

all_speeds = np.sum(np.square(all_velocities), axis=0)
fast_idx = np.argsort(-all_speeds)


def neural(ind):
    neur = np.zeros((ind.size, d_neural))
    for i0 in range(ind.size):
        s_idx = range(ind[i0], ind[i0] + 6)
        neur[i0, :] = all_neural[:, s_idx].flatten()
    return neur


def velocities(ind):
    return all_velocities[:, ind + 6].T


g1 = tf.Graph()  # this graph is for building features

d_hid1, d_hid2 = 30, 15

# Tell TensorFlow that the model will be built into the default Graph.
with g1.as_default():
    # Generate placeholders for the images and labels.
    with tf.name_scope('inputs'):
        neural_ = tf.placeholder(tf.float32, shape=[None, d_neural])

    with tf.name_scope('outputs'):
        velocities_ = tf.placeholder(tf.float32, shape=[None, d_velocities])

    with tf.name_scope('keep_prob'):
        keep_prob_ = tf.placeholder("float", name="keep_probability")

    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([d_neural, d_hid1], stddev=1 / sqrt(float(d_hid1))), name='weights')
        biases = tf.Variable(tf.zeros([d_hid1]), name='biases')
        hidden1 = tf.nn.relu6(tf.matmul(neural_, weights) + biases)

    with tf.name_scope('dropout1'):
        hidden1_dropped = tf.nn.dropout(hidden1, keep_prob_)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([d_hid1, d_hid2], stddev=1 / sqrt(float(d_hid2))), name='weights')
        biases = tf.Variable(tf.zeros([d_hid2]), name='biases')
        hidden2 = tf.nn.relu6(tf.matmul(hidden1_dropped, weights) + biases)

    with tf.name_scope('output'):
        weights = tf.Variable(tf.truncated_normal([d_hid2, d_velocities], stddev=1 / sqrt(float(d_velocities))),
                              name='weights')
        biases = tf.Variable(tf.zeros([d_velocities]), name='biases')
        outputs = tf.matmul(hidden2, weights) + biases

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.squared_difference(outputs, velocities_), name='mse')
        tf.histogram_summary('loss', loss)

    optimizer = tf.train.AdagradOptimizer(0.1)
    # optimizer = tf.train.RMSPropOptimizer(0.1)

    # train_op = optimizer.minimize(loss)
    train_op = optimizer.minimize(loss)

    with tf.name_scope('validation'):
        val_op = tf.reduce_mean(tf.squared_difference(outputs, velocities_))
        tf.scalar_summary('validation', val_op)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for training g1
    sess1 = tf.Session(graph=g1)

    # Run the Op to initialize the variables.
    sess1.run(init)

    # training 1
    valid_idx = fast_idx[np.in1d(fast_idx, range(int(2*T / 3)))]
    for j in range(10):
        for i in range(24):
            # randomly grab a training set
            idx = valid_idx[300 * int(6 - i / 4):300 * (int(6 - i / 4) + 1)]
            if i % 10 == 0:  # every 10th step we run our validation step to see how we're doing
                f_dict = {neural_: neural(idx), velocities_: velocities(idx), keep_prob_: 1}
                [summary, vali] = sess1.run([summary_op, val_op], feed_dict=f_dict)
                print('Accuracy at step %s: %s' % (i, vali))
                save_path = saver.save(sess1, "/Users/michael/Documents/brown/kobe/data/writers/1/model.ckpt")
                print("Model saved in file: %s" % save_path)
            else:  # if we're not on a 10th step then we do a regular training step
                f_dict = {neural_: neural(idx), velocities_: velocities(idx), keep_prob_: 0.75}
                [summary, _] = sess1.run([summary_op, train_op], feed_dict=f_dict)


    # estimate error
    idx = np.arange(int(2*T / 3), int(T))
    f_dict = {neural_: neural(idx), velocities_: velocities(idx), keep_prob_: 1}
    f_neur = sess1.run(outputs, feed_dict=f_dict)
    vels = velocities(idx)
    cov_est = np.cov((vels-f_neur).T)
    print(cov_est)


# collect model parameters
f_hidden1_weights = sess1.run('hidden1/weights:0')
f_hidden1_biases = sess1.run('hidden1/biases:0')
f_hidden2_weights = sess1.run('hidden2/weights:0')
f_hidden2_biases = sess1.run('hidden2/biases:0')
f_output_weights = sess1.run('output/weights:0')
f_output_biases = sess1.run('output/biases:0')


# save model parameters
np.savez('neural_net_parameters0', f_hidden1_weights=f_hidden1_weights, f_hidden1_biases=f_hidden1_biases,
         f_hidden2_weights=f_hidden2_weights, f_hidden2_biases=f_hidden2_biases,
         f_output_weights=f_output_weights, f_output_biases=f_output_biases, cov_est=cov_est)

"""
look at output with:
tensorboard --logdir=/Users/michael/Documents/brown/kobe/data/writers/1
"""
