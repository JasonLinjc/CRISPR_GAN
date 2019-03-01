# -*- coding: utf-8 -*-
# @Time     :10/7/18 3:36 PM
# @Auther   :Jason Lin
# @File     :test_tf_data$.py
# @Software :PyCharm
import tensorflow as tf
import numpy as np

x = np.random.sample((100,2))
BATCH_SIZE = 4
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)

# create the iterator
iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el)) # output: [ 0.42116176  0.40666069]
    print(sess.run(el))