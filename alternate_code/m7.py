import tensorflow as tf
import numpy as np
import scipy.stats as sp
from math import sqrt

# grab data
npzfile = np.load('../Flint_2012_e1_PCA.npz')
all_time = npzfile['all_time']
file_velocities = npzfile['all_velocities']
file_neural = npzfile['all_neural']

"""
data is sampled every 0.01s;
the paper says the neural data 200-300ms beforehand is most informative
so we need the previous 30 observations of neural data for each velocity update
"""

T = int(all_time) - 30
del all_time

d_neural = 30 * file_neural.shape[0]
d_velocities = file_velocities.shape[0]

#Normalize velocities and neural data
# mean_deviated_velocities = file_velocities - np.tile(np.reshape(np.mean(file_velocities,1), (d_velocities,1)), (1,T+30))
# stddev_velocities = np.reshape(np.std(file_velocities, axis=1), (d_velocities,1))
# all_velocities = np.divide(mean_deviated_velocities, np.tile(stddev_velocities, (1,T+30)))

# mean_deviated_neural = file_neural - np.tile(np.reshape(np.mean(file_neural,1), (d_neural/30,1)), (1,T+30))
# stddev_neural = np.reshape(np.std(file_neural, axis=1), (d_neural/30,1))
# all_neural = np.divide(mean_deviated_neural, np.tile(stddev_neural, (1,T+30)))

#all_velocities = all_velocities*10 #Extra normalization factor to scale more easily
#all_neural = all_neural/10

all_velocities = file_velocities
all_neural = file_neural

del file_velocities, file_neural
#del mean_deviated_velocities, mean_deviated_neural

def neural(ind):
    neur = np.zeros((ind.size, d_neural))
    for i0 in range(ind.size):
        s_idx = range(ind[i0], ind[i0] + 30)
        neur[i0, :] = all_neural[:, s_idx].flatten()
    return neur


def velocities(ind):
    return all_velocities[:, ind + 29].T



g1 = tf.Graph()  # this graph is for building features

# choice parameters
d_hid1, d_hid2, d_feat = 100, 50, 3
d_hid1_feat, d_hid2_feat = 30, 10

activation_fn = tf.nn.softsign
training_fn =tf.train.FtrlOptimizer
keep_prob1 = 0.75
keep_prob2 = 0.75


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
        hidden1 = activation_fn(tf.matmul(neural_, weights) + biases)
        tf.histogram_summary('weights1', weights)
        tf.histogram_summary('biases1', biases)


    with tf.name_scope('dropout1'):
        hidden1_dropped = tf.nn.dropout(hidden1, keep_prob_)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([d_hid1, d_hid2], stddev=1 / sqrt(float(d_hid2))), name='weights')
        biases = tf.Variable(tf.zeros([d_hid2]), name='biases')
        hidden2 = activation_fn(tf.matmul(hidden1_dropped, weights) + biases)
        tf.histogram_summary('weights2', weights)
        tf.histogram_summary('biases2', biases)

    with tf.name_scope('dropout2'):
        hidden2_dropped = tf.nn.dropout(hidden2, keep_prob_)

    with tf.name_scope('hidden3'):
        weights = tf.Variable(tf.truncated_normal([d_hid2, d_feat], stddev=1 / sqrt(float(d_feat))), name='weights')
        biases = tf.Variable(tf.zeros([d_feat]), name='biases')
        features = activation_fn(tf.matmul(hidden2_dropped, weights) + biases)
        tf.histogram_summary('weights3', weights)
        tf.histogram_summary('biases3', biases)
        tf.histogram_summary('features', features)

    with tf.name_scope('output'):
        weights = tf.Variable(tf.truncated_normal([d_feat, d_velocities], stddev=1 / sqrt(float(d_velocities))),
                              name='weights')
        biases = tf.Variable(tf.zeros([d_velocities]), name='biases')
        outputs = tf.matmul(features, weights) + biases
        tf.histogram_summary('weights4', weights)
        tf.histogram_summary('biases4', biases)
        tf.histogram_summary('errors1', outputs - velocities_)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.squared_difference(outputs, velocities_), name='mse')
        tf.histogram_summary('loss', loss)

    optimizer = training_fn(0.001)
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
    saver = tf.train.Saver(sharded=True)

    # Create a session for training g1
    sess1 = tf.Session(graph=g1)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter('../writers/1',  sess1.graph)
	
    tf.train.write_graph(g1.as_graph_def(), '../writers/1', 'g1.pbtxt') 

    # Run the Op to initialize the variables.
    sess1.run(init)

    # training 1
    for i in range(301):
        # randomly grab a training set
        idx_tr = np.random.choice(T-20000, 1000, replace=False)
        idx_te = np.random.choice(20000, 1000, replace=False) + T - 20000
		
        if i % 50 == 0:  # every 10th step we run our validation step to see how we're doing
            f_dict = {neural_: neural(idx_te), velocities_: velocities(idx_te), keep_prob_:keep_prob1}
            [summary, vali, feai] = sess1.run([summary_op, val_op, features], feed_dict=f_dict)
            summary_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, vali))
            save_path = saver.save(sess1, "../writers/1/model.ckpt")
            print("Model saved in file: %s" % save_path)
            print("Features are: %s" % feai)
        else:  # if we're not on a 10th step then we do a regular training step
            f_dict = {neural_: neural(idx_tr), velocities_: velocities(idx_tr), keep_prob_: keep_prob1}
            [summary, _] = sess1.run([summary_op, train_op], feed_dict=f_dict)
            summary_writer.add_summary(summary, i)

    learned_variables = tf.all_variables()

g2 = tf.Graph()  # this graph is for mapping velocities to neural via features

# Tell TensorFlow that the model will be built into the default Graph.
with g2.as_default():
    # Generate placeholders for the images and labels.
    with tf.name_scope('inputs'):
        neural_ = tf.placeholder(tf.float32, shape=[None, d_neural])

    with tf.name_scope('outputs'):
        velocities_ = tf.placeholder(tf.float32, shape=[None, d_velocities])

    with tf.name_scope('keep_prob'):
        keep_prob_ = tf.placeholder("float", name="keep_probability")

    with tf.name_scope('feature_injection'):
        with tf.name_scope('hidden1'):
            weights = tf.constant(sess1.run('hidden1/weights:0'))
            biases = tf.constant(sess1.run('hidden1/biases:0'))
            hidden1 = activation_fn(tf.matmul(neural_, weights) + biases)

        with tf.name_scope('hidden2'):
            weights = tf.constant(sess1.run('hidden2/weights:0'))
            biases = tf.constant(sess1.run('hidden2/biases:0'))
            hidden2 = activation_fn(tf.matmul(hidden1, weights) + biases)

        with tf.name_scope('features'):
            weights = tf.constant(sess1.run('hidden3/weights:0'))
            biases = tf.constant(sess1.run('hidden3/biases:0'))
            features = tf.matmul(hidden2, weights) + biases

    with tf.name_scope('map_to_features'):

        with tf.name_scope('hidden1'):
            weights = tf.Variable(tf.truncated_normal([d_velocities, d_hid1_feat],
                                                      stddev=1 / sqrt(float(d_hid1_feat))), name='weights')
            biases = tf.Variable(tf.zeros([d_hid1_feat]), name='biases')
            hidden1 = activation_fn(tf.matmul(velocities_, weights) + biases)
            tf.histogram_summary('weights1', weights)
            tf.histogram_summary('biases1', biases)

        with tf.name_scope('dropout1'):
            hidden1_dropped = tf.nn.dropout(hidden1, keep_prob_)

        with tf.name_scope('hidden2'):
            weights = tf.Variable(tf.truncated_normal([d_hid1_feat, d_hid2_feat],
                                                      stddev=1 / sqrt(float(d_hid2_feat))), name='weights')
            biases = tf.Variable(tf.zeros([d_hid2_feat]), name='biases')
            hidden2 = activation_fn(tf.matmul(hidden1_dropped, weights) + biases)
            tf.histogram_summary('weights2', weights)
            tf.histogram_summary('biases2', biases)

        with tf.name_scope('output'):
            weights = tf.Variable(tf.truncated_normal([d_hid2_feat, d_feat], stddev=1 / sqrt(float(d_feat))),
                                  name='weights')
            biases = tf.Variable(tf.zeros([d_feat]), name='biases')
            outputs = tf.matmul(hidden2, weights) + biases
            tf.histogram_summary('weights3', weights)
            tf.histogram_summary('biases3', biases)
            tf.histogram_summary('errors2', outputs - features)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.squared_difference(outputs, features), name='mse')
        tf.histogram_summary('loss', loss)

    optimizer = training_fn(0.01)
    train_op = optimizer.minimize(loss)

    with tf.name_scope('validation'):
        val_op = tf.reduce_mean(tf.squared_difference(outputs, features))
        tf.scalar_summary('validation', val_op)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(sharded=True)

    # Create a session for training g1
    sess2 = tf.Session(graph=g2)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter('../writers/2', sess2.graph)

    tf.train.write_graph(g2.as_graph_def(), '../writers/2', 'g2.pbtxt') 
	
    # Run the Op to initialize the variables.
    sess2.run(init)

    # training 2
    for i in range(301):
        # randomly grab a training set
        idx_tr = np.random.choice(T - 20000, 10000, replace=False)
       	idx_te = np.random.choice(20000, 10000, replace=False) + T - 20000
        if i % 50 == 0:  # every 10th step we run our validation step to see how we're doing
            f_dict = {neural_: neural(idx_te), velocities_: velocities(idx_te), keep_prob_: keep_prob2}
            [summary, vali] = sess2.run([summary_op, val_op], feed_dict=f_dict)
            summary_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, vali))
            save_path = saver.save(sess2, "../writers/2/model.ckpt")
            print("Model saved in file: %s" % save_path)
        else:  # if we're not on a 10th step then we do a regular training step
            f_dict = {neural_: neural(idx_tr), velocities_: velocities(idx_tr), keep_prob_: keep_prob2}
            [summary, _] = sess2.run([summary_op, train_op], feed_dict=f_dict)
            summary_writer.add_summary(summary, i)

# collect model parameters
f_hidden1_weights = sess1.run('hidden1/weights:0')
f_hidden1_biases = sess1.run('hidden1/biases:0')
f_hidden2_weights = sess1.run('hidden2/weights:0')
f_hidden2_biases = sess1.run('hidden2/biases:0')
f_hidden3_weights = sess1.run('hidden3/weights:0')
f_hidden3_biases = sess1.run('hidden3/biases:0')

g_hidden1_weights = sess2.run('map_to_features/hidden1/weights:0')
g_hidden1_biases = sess2.run('map_to_features/hidden1/biases:0')
g_hidden2_weights = sess2.run('map_to_features/hidden2/weights:0')
g_hidden2_biases = sess2.run('map_to_features/hidden2/biases:0')

# save model parameters
np.savez('neural_net_parameters', f_hidden1_weights=f_hidden1_weights, f_hidden1_biases=f_hidden1_biases,
         f_hidden2_weights=f_hidden2_weights, f_hidden2_biases=f_hidden2_biases,
         f_hidden3_weights=f_hidden3_weights, f_hidden3_biases=f_hidden3_biases,
         g_hidden1_weights=g_hidden1_weights, g_hidden1_biases=g_hidden1_biases,
         g_hidden2_weights=g_hidden2_weights, g_hidden2_biases=g_hidden2_biases)



"""
look at output with:
tensorboard --logdir=../writers/1
tensorboard --logdir=../writers/2
"""

