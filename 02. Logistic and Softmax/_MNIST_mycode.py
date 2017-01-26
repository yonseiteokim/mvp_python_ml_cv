from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""
# Easy Version
hypothesis = tf.nn.softmax(tf.matmul(X, W)+b)
cost = tf.reduce_mean(tf.reduce_sum(-Y*tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
"""

# Tensorflow.org Version
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hypothesis))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

init = tf.global_variables_initializer()

training_epoch = 50
display_step = 1
batch_size = 100

session = tf.Session()
session.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})

"""
for epoch in range(training_epoch):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        session.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
        # Compute average loss

    # show logs per epoch step
    avg_cost += session.run(cost, feed_dict={X: batch_xs, Y: batch_ys}) / total_batch
    if epoch % display_step == 0:  # Softmax
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        # print (session.run(b))
"""

print("Optimization Finished!")

# Test model
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(session.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

total_num = int(mnist.test.num_examples)

for r in range(1, 101):
    # print("Label: ", session.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    # print("Prediction: ", session.run(tf.argmax(hypothesis,1), {X: mnist.test.images[r:r+1]}))

    plt.subplot(10, 10, r)
    tmp = mnist.test.images[r:r + 1].reshape(28, 28)
    if (session.run(tf.argmax(mnist.test.labels[r:r + 1], 1)) == session.run(tf.argmax(hypothesis, 1),
                                                                             {X: mnist.test.images[r:r + 1]})):
        plt.imshow(tmp, cmap='Greys', interpolation='nearest')
    else:
        plt.imshow(1 - tmp, cmap='Greys', interpolation='nearest')
    # plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.title(session.run(tf.argmax(hypothesis, 1), {X: mnist.test.images[r:r + 1]}))
    plt.axis('off')

plt.show()

















