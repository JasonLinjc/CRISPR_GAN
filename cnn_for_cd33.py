# -*- coding: utf-8 -*-
# @Time     :7/17/18 4:01 PM
# @Auther   :Jason Lin
# @File     :cnn_for_cd33$.py
# @Software :PyCharm

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from keras.models import Model
from keras.models import model_from_yaml
import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn import metrics

np.random.seed(5)
from tensorflow import set_random_seed
set_random_seed(12)

def load_penghui_data():
    ph_data = pkl.load(open("./encode_cd33_data/penghui_code_data.pkl", "rb"))
    ph_code = ph_data[0]
    ph_label = ph_data[1]

    train_data = []
    for code in ph_code:
        train_data += code

    train_data = np.array(train_data).reshape(len(ph_code), 1, 23, 4)
    # print(train_data[0])
    return train_data, ph_label

def split_data_for_validation(data):
    skf = StratifiedKFold(n_splits=5)
    # print(skf.get_n_splits())

def load_crispor_data():
    crispor_data = pkl.load(open("./encode_cd33_data/crispor_code_data.pkl", "rb"))
    crispor_code = crispor_data[0]
    crispor_label = crispor_data[1]

    merged_list = []
    for seq in crispor_code:
        merged_list += seq

    my_code = np.array(merged_list).reshape(len(crispor_code), 1, 23, 4)
    print(len(my_code))
    print(len(crispor_label))

    return my_code, crispor_label
    # pass

def load_crispr_data_for_training():
    crispor_data = pkl.load(open("./encode_cd33_data/crispor_all_code_data.pkl", "rb"))
    crispor_code = crispor_data[0]
    crispor_label = crispor_data[1]

    merged_list = []
    for seq in crispor_code:
        merged_list += seq

    my_code = np.array(merged_list).reshape(len(crispor_code), 1, 23, 4)
    print(len(my_code))
    print(len(crispor_label))

    return my_code, crispor_label

def load_cd33_data():
    data = np.array(pkl.load(open("./encode_cd33_data/cd33_code_pam_data.pkl", "rb")))
    my_codes = data[0]
    ele_codes = data[1]
    reg_labels = data[2]

    # print(my_codes)
    merged_list = []
    for l in my_codes:
        merged_list += l
    print(len(my_codes))
    my_codes = np.array(merged_list).reshape(len(my_codes), 1, 23, 4)
    # print(my_codes)

    # For leave-out testing
    # X_train, X_test, y_train, y_test = train_test_split(my_codes, reg_labels, test_size = 0.2, random_state = 1)
    # print(X_train.shape)
    # return X_train, X_test, y_train, y_test
    return my_codes, reg_labels

def cnn_model(X_train, X_test, y_train, y_test):

    # X_train, y_train = load_cd33_data()

    inputs = Input(shape=(1, 23, 4), name='main_input')
    conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(inputs)
    conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(inputs)
    conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(inputs)
    conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(inputs)

    conv_output = keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

    bn_output = BatchNormalization()(conv_output)

    pooling_output = keras.layers.MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

    flatten_output = Flatten()(pooling_output)

    x = Dense(100, activation='relu')(flatten_output)
    x = Dense(23, activation='relu')(x)
    x = keras.layers.Dropout(rate=0.45)(x)

    prediction = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs, prediction)

    crispor = load_crispr_data_for_training()
    crispor_data = crispor[0]
    crispor_label = crispor[1]

    adam_opt = keras.optimizers.adam(lr = 0.0001)
    model.compile(loss='mean_squared_error', optimizer = adam_opt)
    print(model.summary())
    model.fit(X_train, y_train, batch_size=100, epochs=250, shuffle=True)

    # adam_opt = keras.optimizers.adam(lr = 0.00001)
    # model.compile(loss='mean_squared_error', optimizer = adam_opt)
    # X_train, y_train = load_cd33_data()
    # model.fit(X_train, y_train, batch_size=50, epochs=50)
    # model.fit(crispor_data, crispor_label, batch_size=100, epochs=150, shuffle=True)
    """
    # evaluate the model
    scores = model.evaluate(crispor_data, crispor_label, verbose=0)
    print(scores)
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    """
    # later...
    # X_test, y_test = load_crispor_data()
    y_pred = model.predict(X_test).flatten()
    print(y_pred)
    print(y_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    print(auc_score)
    auc_info = [auc_score, fpr, tpr, threshold]
    return auc_info

def load_trained_model():
    # load YAML and create model
    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

# cnn_model()

def kfold_validataion():
    ph_data, ph_label = load_penghui_data()
    ph_label = np.array(ph_label)
    # print(ph_label)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=66)
    skf.get_n_splits(ph_data, ph_label)
    auc_info = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 1
    for train_index, test_index in skf.split(ph_data, ph_label):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = ph_data[train_index], ph_data[test_index]
        y_train, y_test = ph_label[train_index], ph_label[test_index]
        auc_s = cnn_model(X_train, X_test, y_train, y_test)
        auc_score, fpr, tpr, threshold = auc_s
        auc_info.append(auc_s)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc_score)

        plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, auc_score))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def crispor_test():
    ph_data, ph_label = load_penghui_data()
    ph_label = np.array(ph_label)

    crispor_data, crispor_label = load_crispor_data()
    crispor_label = np.array(crispor_label)

    auc_s = cnn_model(ph_data, crispor_data, ph_label, crispor_label)
    print(auc_s)

# X_train, X_test, y_train, y_test = load_cd33_data()
# print(X_train[0][0])

# load_crispor_data()
# print(len(crispor_data[0][0]))
# load_crispr_data_for_training()
# load_crispr_data_for_training()

kfold_validataion()
# crispor_test()