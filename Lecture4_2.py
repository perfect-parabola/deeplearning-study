import tensorflow as tf

#linear regression for mutl-variable using matrix
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

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
    if i %20 ==0:
        print(i, "Cost: ", cost_val, "\nPrediction:\n", y_pred)