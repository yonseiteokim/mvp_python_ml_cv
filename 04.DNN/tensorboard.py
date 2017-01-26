# Use 'Deep' neural network to solve XOR problem.

import numpy as np
import tensorflow as tf

xy = np.loadtxt('D:\\ppy\\07train.txt', unpack=True)

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32, name='X-input')
Y = tf.placeholder(tf.float32, name='Y-input')

# Deep network configuration.: Use more layers.
W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='weight1')
W2 = tf.Variable(tf.random_uniform([5, 4], -1.0, 1.0), name='weight2')
W3 = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0), name='weight3')

b1 = tf.Variable(tf.zeros([5]), name="bias1")
b2 = tf.Variable(tf.zeros([4]), name="bias2")
b3 = tf.Variable(tf.zeros([1]), name="bias3")

# Add histogram
w1_hist = tf.summary.histogram("weights1", W1)
w2_hist = tf.summary.histogram("weights2", W2)
w3_hist = tf.summary.histogram("weights3", W3)

b1_hist = tf.summary.histogram("biases1", b1)
b2_hist = tf.summary.histogram("biases2", b2)
b3_hist = tf.summary.histogram("biases3", b3)

y_hist = tf.summary.histogram("y", Y)

# Hypotheses
with tf.name_scope("layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope("layer3") as scope:
    L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)

with tf.name_scope("layer4") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

# Cost function
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1. - hypothesis))
    cost_summ = tf.summary.scalar("cost", cost)

# Minimize cost.
a = tf.Variable(0.1)
with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

# Initializa all variables.
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    # tensorboard merge
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)

    sess.run(init)

    # Run graph.
    for step in range(20001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 2000 == 0:
            summary, _ = sess.run([merged, train], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Check accuracy
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data}))
    print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))