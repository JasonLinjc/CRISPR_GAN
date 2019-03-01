# -*- coding: utf-8 -*-
# @Time    : 30/9/2018 6:34 PM
# @Author  : Jason Lin
# @File    : CRISPR_wGAN.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sgRNA_off_decoder
import pickle as pkl
import math
np.random.seed(5)
from tensorflow import set_random_seed
set_random_seed(12)
import Dataset

BATCH_SIZE = 50
width, height = 23*2, 4
input_dim = width * height
EPOCHS = 1000
random_off_dim = 23

off_dim = 23*4
condition_gRNA = 23*4

def my_init(size):
    return tf.random_uniform(size, -0.1, 0.1)

# Parameters of Discriminator
D_W1 = tf.Variable(my_init([input_dim, 92]))
D_b1 = tf.Variable(tf.zeros([92]))
D_W2 = tf.Variable(my_init([92, 46]))
D_b2 = tf.Variable(tf.zeros([46]))
D_W3 = tf.Variable(my_init([46, 1]))
D_b3 = tf.Variable(tf.zeros([1]))
D_variables = [D_W1, D_b1, D_W2, D_b2, D_W3, D_b3]

# Parameters of Generator
G_W1 = tf.Variable(my_init([condition_gRNA + random_off_dim, 46])) # [115, 46]
G_b1 = tf.Variable(tf.zeros([46]))
G_W2 = tf.Variable(my_init([46, 92]))
G_b2 = tf.Variable(tf.zeros([92]))
G_W3 = tf.Variable(my_init([92, off_dim]))
G_b3 = tf.Variable(tf.zeros(off_dim))
G_variables = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]

# Construction of Discriminator
def D(X):
    X = tf.nn.leaky_relu(tf.matmul(X, D_W1) + D_b1)
    X = tf.nn.leaky_relu(tf.matmul(X, D_W2) + D_b2)
    X = tf.matmul(X, D_W3) + D_b3
    return X

# Construction of Generator
def G(X):
    X = tf.nn.leaky_relu(tf.matmul(X, G_W1) + G_b1)
    X = tf.nn.leaky_relu(tf.matmul(X, G_W2) + G_b2)
    X = tf.sigmoid(tf.matmul(X, G_W3) + G_b3)
    return X

def get_real_data():
    real_dataset = pkl.load(open("./data/penghui_encode_code.pkl", "rb"))
    real_dataset = np.array(real_dataset)
    sgRNA_codes = real_dataset[:,0]
    off_codes = real_dataset[:,1]
    return sgRNA_codes, off_codes

def generate_random_offs(sgRNA_code, num=1):
    noise_len = 23
    noise_off_code = np.random.uniform(0, 1, (num, noise_len))
    # print(noise_off_code)
    sgRNA_code = np.array(sgRNA_code)
    # print(sgRNA_code)
    sgRNA_code = np.tile(sgRNA_code, num).reshape(num, 23*4)
    sgRNA_noise_off = np.concatenate((sgRNA_code, noise_off_code), axis=1)
    # print(random_off_codes)
    return sgRNA_code, sgRNA_noise_off

def load_new_sgRNA_seq():
    pass

# Inputs for Generator and Discriminator
condition_gRNA_X = tf.placeholder(tf.float32, shape=[None, condition_gRNA])
real_X = tf.placeholder(tf.float32, shape=[None, condition_gRNA + random_off_dim*4])
random_X = tf.placeholder(tf.float32, shape=[None, condition_gRNA + random_off_dim])
generated_off = G(random_X) # Output the sequence code with length of 23*4
fake_X = tf.concat([condition_gRNA_X, generated_off], 1)

disc_real = D(real_X)
disc_fake = D(fake_X)
# Gradient Penalty

# The loss functions of Discriminator and Generator
gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

alpha = tf.random_uniform(shape=[BATCH_SIZE, 1],
                          minval=0.,
                          maxval=1.)

differences  = fake_X - real_X
interpolates = real_X + (alpha * differences)
gradients = tf.gradients(D(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
disc_cost += 10 * gradient_penalty

# The optimization functions of Discriminator and Generator
D_solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=D_variables)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=G_variables)

# load sgRNA-off data
real_dataset = pkl.load(open("./data/penghui_encode_code.pkl", "rb"))
real_dataset = np.array(real_dataset)
sgRNA_codes = real_dataset[:, 0]
off_codes = real_dataset[:, 1]

sgRNA_off_dataset = Dataset.Dataset_sgRNA_off(sgRNA_codes, off_codes)
sgRNA_off_dataset2 = Dataset.Dataset_sgRNA_off(sgRNA_codes, off_codes)

# load test data
ele_data = pkl.load(open("./data/elevation_data2.pkl", "rb"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(10000):
        total_batch = math.ceil(len(sgRNA_codes)/BATCH_SIZE)
        for _ in range(total_batch):
            for _ in range(5):
                real_batch_X, noise1_batch_X, sgRNA_batch_X = sgRNA_off_dataset.next_batch(BATCH_SIZE)
                random_batch_X = noise1_batch_X
                _, D_loss_ = sess.run([D_solver, disc_cost], feed_dict={real_X: real_batch_X, random_X: random_batch_X,
                                                                     condition_gRNA_X: sgRNA_batch_X})
            _, random_batch_X, sgRNA_batch_X = sgRNA_off_dataset2.next_batch(BATCH_SIZE)
            _, G_loss_ = sess.run([G_solver, gen_cost], feed_dict={random_X: random_batch_X, condition_gRNA_X: sgRNA_batch_X })
            print('epoch %s, D_loss: %s, G_loss: %s' % (e, D_loss_, G_loss_))

        if e > 200:
            print('epoch %s, D_loss: %s, G_loss: %s' % (e, D_loss_, G_loss_))
            sgRNA_condition = ele_data[0][0]
            sgRNA_condition_code, sgRNA_noise_off = generate_random_offs(sgRNA_condition, num=5)
            generative_sgRNA_off = sess.run(fake_X, feed_dict={random_X: sgRNA_noise_off, condition_gRNA_X: sgRNA_condition_code})
            
            decoder = sgRNA_off_decoder.Decoder_sgRNA_off(np.array(generative_sgRNA_off[0]).flatten())
            print(decoder.sgRNA_seq)
            print(decoder.off_seq)
            print("-----------------")
            break


