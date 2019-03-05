import tensorflow as tf

#simple linear regression using tensorflow
x_train = [1,2,3]
y_train = [1,2,3]

w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

#tensor flow 에서 variable은 조금 다름 trainable 의 의미 => 알아서 바꿈
hypothesis = x_train*w + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step %20 == 0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))