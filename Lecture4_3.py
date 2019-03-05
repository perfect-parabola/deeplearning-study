import numpy as np
import tensorflow as tf

#load data with numpy and predict your final score
xy = np.loadtxt('data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([3, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

hypothesis = tf.matmul(X, W) +b
cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    y_pred, cost_val, _ = sess.run([hypothesis, cost, train], feed_dict={X:x_data, Y:y_data})

print("당신의 예측 점수는 ", sess.run(hypothesis, feed_dict={X:[[100,100,100]]}), "점 입니다.")