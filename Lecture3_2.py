import tensorflow as tf

#manually programmed gradiend descent vs using library
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W1 = tf.Variable(5.0, name="weight")
W2 = tf.Variable(5.0, name="weight")
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis1 = X * W1
hypothesis2 = X * W2

cost1 = tf.reduce_mean(tf.square(hypothesis1 - Y))
cost2 = tf.reduce_mean(tf.square(hypothesis2 - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W1 * X - Y) * X)
descent = W1 - learning_rate * gradient
update = W1.assign(descent)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(cost2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(21):
        _, cost_val_manual, W_val_manual = sess.run(
            [update, cost1, W1], feed_dict={X: x_data, Y: y_data}
        )
        _, cost_val_op, W_val_op = sess.run(
            [train, cost2, W2], feed_dict={X: x_data, Y: y_data}
        )
        print(step, cost_val_op, W_val_manual, cost_val_op, W_val_op)