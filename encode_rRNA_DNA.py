# -*- coding: utf-8 -*-
# @Time    : 6/16/18 4:45 PM
# @Author  : Jason Lin
# @File    : encode_rRNA_DNA.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import re
import pickle as pkl
from random import randint

class Encoding:

    def get_CD33_data(self):
        cd33_data = pd.read_pickle("./data/cd33.pkl")
        pass

    def encode_gRNA_DNA_seq_pair(self, gRNA_seq, DNA_seq):
        code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
        gRNA_list = list(gRNA_seq)
        DNA_list = list(DNA_seq)
        print(len(gRNA_list))
        if len(gRNA_list) != len(DNA_list):
            print("the length of sgRNA and DNA are not matched!")
            return 0
        pair_code = []

        for i in range(len(gRNA_seq)):
            if gRNA_list[i] == 'N':
                gRNA_list[i] = DNA_list[i]
            gRNA_base_code = code_dict[gRNA_list[i]]
            DNA_based_code = code_dict[DNA_list[i]]

            pair_code.append(list(np.bitwise_or(gRNA_base_code, DNA_based_code)))

        return pair_code

    def encode_crispor_data(self):
        crispr_df = pd.read_csv("./data/crispor_all_data.csv")
        # print(len(crispr_df[crispr_df['label'] == 1]))
        pair_code = []
        label = []
        for idx, row in crispr_df.iterrows():
            print(idx)
            gRNA_seq = row['wt_seq']
            DNA_seq = row['off_seq']
            label.append(row['label'])
            pair_code.append(self.encode_gRNA_DNA_seq_pair(gRNA_seq, DNA_seq))
            print(gRNA_seq, DNA_seq)
            # break
        # crispor_code = [pair_code, label]
        # print(len(crispor_code[1]))
        # pkl.dump(crispor_code, open("./encode_cd33_data/crispor_all_code_data.pkl", "wb"))

    def encode_penghui_data(self):
        ph_data = pd.read_csv("./data/penghui_dataset_oversample1.csv")

        pair_code = []
        label = []
        for idx, row in ph_data.iterrows():
            print(idx)
            gRNA = row['sgRNA']
            off = row['offtarget']
            label.append(row['label'])
            code = self.encode_gRNA_DNA_seq_pair(gRNA, off)
            # print(gRNA, off)
            # print(code)
            pair_code.append(code)
            # break

        penghui_code = [pair_code, label]
        pkl.dump(penghui_code, open("./encode_cd33_data/penghui_code_data_oversample1.pkl", "wb"))


    def encode_gRNA_DNA(self, sgRNA_info, etp_val):
        code_dict = {'A':[1,0,0,0], 'T':[0,1,0,0], 'G':[0,0,1,0], 'C':[0,0,0,1]}
        nucleotides = ['A', 'T', 'G', 'C']
        bad_pam_2 = ['T', 'C']
        bad_pam_3 = ['C', 'A', 'T']
        if sgRNA_info['Category'] == "Mismatch":
            gRNA_seq = list(sgRNA_info['WTSequence'])
            if len(gRNA_seq) == 20:
                gRNA_seq.append(nucleotides[randint(0, 3)])
                if etp_val < 0.3:
                    gRNA_seq.append(bad_pam_2[randint(0,1)])
                    gRNA_seq.append(bad_pam_3[randint(0,2)])
                else:
                    gRNA_seq.append('G')
                    gRNA_seq.append('G')
            print(gRNA_seq)
            print(sgRNA_info['Annotation'])
            mismatch_type, mismatch_pos = sgRNA_info['Annotation'].split(',')
            pair_code = []
            print(len(gRNA_seq))
            for i in range(len(gRNA_seq)):
                if i + 1 == int(mismatch_pos):
                    rna_base, dna_base = mismatch_type.split(":")
                    p_code = np.bitwise_or(code_dict[rna_base], code_dict[dna_base])
                    pair_code.append(list(p_code))
                else:
                    pair_code.append(code_dict[gRNA_seq[i]])
            print(pair_code)
            return pair_code

        if sgRNA_info['Category'] == "PAM":
            sgRNA_seq = list(sgRNA_info['30mer'])
            sgRNA_mut = list(sgRNA_info['30mer_mut'])
            pair_code = []
            for i in range(len(sgRNA_seq)):
                sgRNA_code = code_dict[sgRNA_seq[i]]
                sgRNA_mut_code = code_dict[sgRNA_mut[i]]
                p_code = list(np.bitwise_or(sgRNA_code, sgRNA_mut_code))
                pair_code.append(p_code)
            return pair_code
        # if type == "PAM":

    def build_onehot_dict_for_features(self):
        bases = {'A':['G', 'C', 'T'], 'G':['A', 'T', 'C'], 'C':['A', 'T', 'G'], 'T':['G', 'C', 'A']}
        identity_types = []
        annotation_types = []
        # tran_type = {"transition": [0, 1], "transversion": [1, 0]}
        for base in bases.keys():
            mis_l = bases[base]
            for m in mis_l:
                it = base + ":" + m
                identity_types.append(it)
                for i in range(1,21):
                    at = it + "," + str(i)
                    annotation_types.append(at)
        print(len(annotation_types))
        print(len(identity_types))
        identity_types_dict = {}
        annotation_types_dict = {}
        for i in range(len(identity_types)):
            onehot_idt = np.zeros(len(identity_types))
            onehot_idt[i] = 1
            identity_types_dict[identity_types[i]] = list(onehot_idt)
        print(identity_types_dict['T:A'])
        pkl.dump(identity_types_dict, open("./elevation_features_encoding/identity_types_mis_dict.pkl", "wb"))

        for i in range(len(annotation_types)):
            onehot_ant = np.zeros(len(annotation_types))
            onehot_ant[i] = 1
            annotation_types_dict[annotation_types[i]] = list(onehot_ant)
        print(annotation_types_dict['T:A,1'])
        pkl.dump(annotation_types_dict, open("./elevation_features_encoding/annotation_types_mis_dict.pkl", "wb"))

        bases = ['A', 'T', 'G', 'C']
        pam_list = []
        for b1 in bases:
            for b2 in bases:
                pam_list.append(b1 + b2)

        pam_types_dict = {}
        for i in range(len(pam_list)):
            onehot_pam = np.zeros(len(pam_list))
            onehot_pam[i] = 1
            pam_types_dict[pam_list[i]] = list(onehot_pam)
        print(pam_types_dict)
        pkl.dump(pam_types_dict, open("./elevation_features_encoding/pam_types_dict.pkl", "wb"))

    def extract_onehot_feature_from_cd33(self, sgRNA_info):
        misAnnotation = sgRNA_info['Annotation']
        rna_base, target_base, postion = re.split(":|,", misAnnotation)
        trans_type = ""
        identity = rna_base + ":" + target_base
        # transitions appear more often in genomes
        transition_types = {"A": "G", "G": "A", "C": "T", "T": "C"}
        transversion_types = {"A": ["C", "T"], "C": ["A", "G"], "G": ["T", "C"], "T": ["A", "G"]}
        if transition_types[rna_base] == target_base:
            trans_type = "transition"
        else:
            if target_base in transversion_types[rna_base]:
                trans_type = "transversion"
            else:
                print("Error! There is no mismatch!")
                print(sgRNA_info)
                return 0
        print([int(postion), identity, misAnnotation, trans_type])
        joint_code_dict = pkl.load(open("./elevation_features_encoding/annotation_types_mis_dict.pkl", "rb"))
        identity_code_dict = pkl.load(open("./elevation_features_encoding/identity_types_mis_dict.pkl", "rb"))
        trans_type_dict = {"transition": [1., 0.], "transversion": [0., 1.]}

        code = [[float(postion)], identity_code_dict[identity], joint_code_dict[misAnnotation], trans_type_dict[trans_type]]

        return sum(code, [])


    def encode_cd33(self):
        cd33_data = pd.read_pickle("./data/cd33.pkl")
        cd33_1 = cd33_data[0]
        # cd33_2 = cd33_data[1]
        print(cd33_1.columns)
        # print(cd33_1.ix[0])

        # Encode single mismatch sgRNA of CD33 dataset
        single_mis_cd33 = cd33_1.ix[cd33_1['Category'] == "Mismatch"]
        # print(single_mis_cd33)

        ele_code_train = []
        my_code_train = []
        reg_labels = []
        for idx, row in single_mis_cd33.iterrows():
            sgRNA_seq = row['30mer']
            sgRNA_mut = row['30mer_mut']
            misAnnotation = row['Annotation']
            etp_val = row['Day21-ETP']

            mut_type = row['Category']
            # print(row)
            print(etp_val, mut_type)
            my_code = self.encode_gRNA_DNA(row, etp_val)
            # my_code = [my_code, [etp_val]]
            # print(my_code)
            ele_code = self.extract_onehot_feature_from_cd33(row)
            # ele_code = [ele_code, [etp_val]]
            # two_code = [np.array(my_code), np.array(ele_code), etp_val]
            # cd33_code.append(two_code)
            if len(ele_code) != 255 or len(my_code) != 23:
                print("error")
                return 0
            ele_code_train.append(ele_code)
            my_code_train.append(my_code)
            reg_labels.append(etp_val)
            # print(idx)
        cd33_code = [my_code_train, ele_code_train, reg_labels]
        pkl.dump(cd33_code, open("./encode_cd33_data/cd33_code_pam_data.pkl", "wb"))

if __name__ == "__main__":
    encoding = Encoding()
    # cd33_data = pd.read_pickle("./data/cd33.pkl")
    # cd33_1 = cd33_data[0]
    # print(cd33_1.ix[1])
    # code = encoding.encode_gRNA_DNA(cd33_1.ix[1])
    # print(code)
    # feature = encoding.extract_feature_from_cd33(misAnnotation="G:T,9")
    # print(feature)
    # encoding.build_onehot_dict_for_features()
    # encoding.encode_cd33()

    cd33_data = pd.read_pickle("./data/cd33.pkl")
    cd33_1 = cd33_data[0]
    single_mis_cd33 = cd33_1.ix[cd33_1['Category'] == "Mismatch"]
    # encoding.encode_gRNA_DNA(single_mis_cd33.ix[0])

    # encoding.encode_cd33()
    encoding.encode_penghui_data()