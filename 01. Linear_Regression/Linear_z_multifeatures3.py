import tensorflow as tf

x_data = [[1, 1, 1, 1, 1], [1., 0., 3., 0., 5.], [0., 2., 0., 4., 0.]]
y_data = [1, 2, 3, 4, 5]

# try to find values for w and b that compute y_data = W * x_data + b
w = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0)) # size 1x3

# my hypothesis
hypothesis = tf.matmul(w,x_data)

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize
a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# before starting, initialize the variables
init = tf.global_variables_initializer()

# launch
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(w))
