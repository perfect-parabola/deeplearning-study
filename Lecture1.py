import tensorflow as tf

#check version
print(tf.__version__)

#print constant
hello = tf.constant("Hello")
sess = tf.Session()
print(sess.run(hello))

#add
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
print(node1)
print(node2)
print(node3)
#위에껀 노드지 실행은 안함
'''
Tensor("Const_1:0", shape=(), dtype=float32)
Tensor("Const_2:0", shape=(), dtype=float32)
Tensor("Add:0", shape=(), dtype=float32)
'''

sess = tf.Session()
print(sess.run([node1, node2]))
print(sess.run(node3))
'''
[3.0, 4.0]
7.0
'''

#using placehold
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b
print(sess.run(adder_node, feed_dict={a:3, b:5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2]}))

#Tensor Rank : 차원
#Tensor shape : 가장 안쪽의 element부터 몇개씩 있는지