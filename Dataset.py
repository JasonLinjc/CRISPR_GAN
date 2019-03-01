# -*- coding: utf-8 -*-
# @Time    : 9/10/2018 5:01 PM
# @Author  : Jason Lin
# @File    : Dataset.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import pickle as pkl

class Dataset_sgRNA_off:

    def __init__(self, sgRNA_data, off_data):
        self.sgRNA_data = sgRNA_data
        self.off_data = off_data
        self._num_examples = self.sgRNA_data.shape[0]
        self.noise1_off_data = self.generate_noise_offs()
        self.noise2_off_data = self.generate_noise_offs()
        self._index_in_epoch = 0
        self._epochs_completed = 0


    def generate_noise_offs(self):
        random_off_dim = 23
        # print(self.sgRNA_data.shape[0])
        random_off_codes = np.random.uniform(0, 1, (self._num_examples, random_off_dim))
        return random_off_codes


    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle index
            # self._data = self.data[idx]  # get list of `num` random samples
            self.sgRNA_data = self.sgRNA_data[idx]
            self.off_data = self.off_data[idx]
            self.noise1_off_data = self.noise1_off_data[idx]
            self.noise2_off_data = self.noise2_off_data[idx]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            sgRNA_rest_part = self.sgRNA_data[start:self._num_examples]
            off_rest_part = self.off_data[start:self._num_examples]
            noise1_rest_part = self.noise1_off_data[start:self._num_examples]
            noise2_rest_part = self.noise2_off_data[start:self._num_examples]

            idx0 = np.arange(0, self._num_examples)
            np.random.shuffle(idx0)
            self.sgRNA_data = self.sgRNA_data[idx0]
            self.off_data = self.off_data[idx0]
            self.noise1_off_data = self.noise1_off_data[idx0]
            self.noise2_off_data = self.noise2_off_data[idx0]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            sgRNA_new_part = self.sgRNA_data[start:end]
            off_new_part = self.off_data[start:end]
            noise1_new_part = self.noise1_off_data[start:end]
            noise2_new_part = self.noise2_off_data[start:end]

            sgRNA_data = np.concatenate((sgRNA_rest_part, sgRNA_new_part), axis=0)
            off_data = np.concatenate((off_rest_part, off_new_part), axis=0)
            noise1_data = np.concatenate((noise1_rest_part, noise1_new_part), axis=0)
            noise2_data = np.concatenate((noise2_rest_part, noise1_new_part), axis=0)

            sgRNA_real_off = np.concatenate((sgRNA_data, off_data), axis=1)
            sgRNA_noise1_off = np.concatenate((sgRNA_data, noise1_data), axis=1)
            sgRNA_noise2_off = np.concatenate((sgRNA_data, noise2_data), axis=1)

            return sgRNA_real_off, sgRNA_noise1_off, sgRNA_data
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            sgRNA_data = self.sgRNA_data[start:end]
            off_data = self.off_data[start:end]
            noise1_data = self.noise1_off_data[start:end]
            noise2_data = self.noise2_off_data[start:end]

            sgRNA_real_off = np.concatenate((sgRNA_data, off_data), axis=1)
            sgRNA_noise1_off = np.concatenate((sgRNA_data, noise1_data), axis=1)
            sgRNA_noise2_off = np.concatenate((sgRNA_data, noise2_data), axis=1)

            return sgRNA_real_off, sgRNA_noise1_off, sgRNA_data

"""
real_dataset = pkl.load(open("./data/penghui_encode_code.pkl", "rb"))
real_dataset = np.array(real_dataset)
sgRNA_codes = real_dataset[:, 0]
off_codes = real_dataset[:, 1]

dataset = Dataset_sgRNA_off(sgRNA_codes, off_codes)
a, b, c = dataset.next_batch(50)
a, b, c = dataset.next_batch(50)
print(a[0])
print(b[0])
print(c[0])
"""

