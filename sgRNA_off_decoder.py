# -*- coding: utf-8 -*-
# @Time     :10/10/18 5:01 PM
# @Auther   :Jason Lin
# @File     :sgRNA_off_decoder$.py
# @Software :PyCharm
import numpy as np
import pandas as pd
import pickle as pkl

class Decoder_sgRNA_off:

    def __init__(self, sgRNA_off_code):
        self.code = sgRNA_off_code
        self.get_so_code()
        self.cipher = {1: 'A', 2: 'T', 3: 'G', 4: 'C'}
        self.code_to_seq()
        # code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}

    def get_so_code(self):
        code = np.array(self.code)
        print(code.shape)
        sgRNA_code = code[:4*23]
        off_code = code[4*23:]
        sgRNA_code = sgRNA_code.reshape((23, 4))
        off_code = off_code.reshape((23, 4))
        self.sgRNA_code = sgRNA_code
        self.off_code = off_code

    def code_to_seq(self):
        sgRNA_seq = ""
        off_seq = ""
        for i in range(len(self.sgRNA_code)):
            sgRNA_base_code = self.sgRNA_code[i]
            off_base_code = self.off_code[i]
            sgRNA_max_idx = np.argmax(sgRNA_base_code) + 1
            off_max_idx = np.argmax(off_base_code) + 1

            sgRNA_base = self.cipher[sgRNA_max_idx]
            off_base = self.cipher[off_max_idx]

            sgRNA_seq += sgRNA_base
            off_seq += off_base

        self.sgRNA_seq = sgRNA_seq
        self.off_seq = off_seq


"""
# testing 
ele_data = pkl.load(open("./data/elevation_data2.pkl", "rb"))
sgRNA, off = ele_data[0]
sgRNA_off = np.concatenate((sgRNA, off))

t = Decoder_sgRNA_off(sgRNA_off)
print(t.sgRNA_seq)
print(t.off_seq)
"""
ele_data = pkl.load(open("./data/elevation_data2.pkl", "rb"))
sgRNA, off = ele_data[0]
sgRNA_off = np.concatenate((sgRNA, off))
print(sgRNA_off.shape)