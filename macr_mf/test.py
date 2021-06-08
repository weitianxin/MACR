import tensorflow as tf
import numpy as np


initializer = tf.contrib.layers.xavier_initializer()
c = tf.Variable(initializer([10]), name='item_embedding')
b = tf.nn.embedding_lookup(c, [1,2,3])
e = tf.const([])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(b))
    print(sess.run(c))