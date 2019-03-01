# -*- coding: utf-8 -*-
# @Time    : 30/9/2018 4:28 PM
# @Author  : Jason Lin
# @File    : gRNA_off_encoder.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import pickle as pkl

class Encoder:

    gRNA_seq = ""
    off_seq = ""
    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}

    # constructor
    def __init__(self, gRNA, off):
        self.off_seq = off
        self.gRNA_seq = gRNA
        self.gRNA_code = self.encode_sgRNA()
        self.off_code = self.encode_off()

    def encode_sgRNA(self):
        gRNA_bases = list(self.gRNA_seq)
        base_code = []
        i = 0
        for base in gRNA_bases:
            if base == "N":
                base = self.off_seq[i]
            base_code += self.code_dict[base]
            i += 0
        return base_code

    def encode_off(self):
        gRNA_bases = list(self.off_seq)
        base_code = []
        for base in gRNA_bases:
            base_code += self.code_dict[base]

        return base_code


    def encode_noise_off(self):
        base_code = []
        for i in range(len(list(self.off_seq))):
            base = np.random.normal(0.5, 1, 4)
            # base += np.abs(np.min(base))
            # base = list(self.normalise_noise(base))
            base_code.append(list(base))
        return base_code

    def normalise_noise(self, codes):
        s = 0.
        for code in codes:
            s += code
        codes = codes/s
        return codes

"""
encoder = Encoder("GCCTCCCCAAAGCCTGGCCAGGG", "GCCTTCCCAAAGCCCGGCCATGG")
b = encoder.encode_noise_off()
print(b)
"""

def encode_penghui_dataset():
    ph_data = pd.read_csv("data/penghui_dataset_positive.csv")
    pair_code = []
    for idx, row in ph_data.iterrows():
        print(idx)
        gRNA_seq = row['sgRNA']
        off_seq = row['offtarget']
        encoder = Encoder(gRNA_seq, off_seq)
        pair_code.append([encoder.gRNA_code, encoder.off_code])
    pkl.dump(pair_code, open("./data/penghui_encode_code.pkl", "wb"))

def elevation_validated_data2():
    sgRNA_offs = pd.read_excel("./data/elevation_validated_data2.xlsx", sheet_name='off-target')
    # print(sgRNA_offs)
    pair_code = []
    for idx, row in sgRNA_offs.iterrows():
        if row['Targetsite'] != "EMX1_site1":
            sgRNA_seq = row["Target Sequence"]
            off_seq = row["Off-target Sequence"]
            sgRNA_off_code = Encoder(sgRNA_seq, off_seq)
            pair_code.append([sgRNA_off_code.gRNA_code, sgRNA_off_code.off_code])
    pkl.dump(pair_code, open("./data/elevation_data2.pkl", "wb"))
    print(np.array(pair_code).shape)
    print(pair_code[0][0])
    print(pair_code[0][1])

elevation_validated_data2()
# encode_penghui_dataset()