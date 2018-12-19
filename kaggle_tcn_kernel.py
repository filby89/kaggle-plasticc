"""
This script is forked from iprapas's notebook
https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135

#    https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data
#    https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70908
#    https://www.kaggle.com/meaninglesslives/simple-neural-net-for-time-series-classification
#
"""

import sys, os
import argparse
import time
from datetime import datetime as dt
import gc;
import matplotlib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,GlobalAveragePooling1D
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import to_categorical
from keras import backend as K
from keras import regularizers
import keras
from collections import Counter
from sklearn.metrics import confusion_matrix
from keras.backend import set_session
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import GRU, Dense, Activation, Dropout, CuDNNGRU, concatenate, Input, LSTM
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.models import Sequential, Model
from keras.optimizers import Adam
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

matplotlib.use('Agg')

gc.enable()
from functools import partial, wraps

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra

np.warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
import tsfresh
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import feature_calculators
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from numba import jit
from filby import *
from tsfresh_extra import *

np.random.seed(35)


def multi_weighted_logloss(y_true, y_preds, classes, class_weights):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


def multi_weighted_logloss_keras(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss



from keras.preprocessing.sequence import pad_sequences

def featurize_raw():
    pbmap = OrderedDict([(0, 'u'), (1, 'g'), (2, 'r'), (3, 'i'), (4, 'z'), (5, 'Y')])

    # it also helps to have passbands associated with a color
    pbcols = OrderedDict([(0, 'blueviolet'), (1, 'green'), (2, 'red'), \
                          (3, 'orange'), (4, 'black'), (5, 'brown')])

    pbnames = list(pbmap.values())

    metafilename = 'PLAsTiCC-2018/training_set_metadata.csv'
    metadata = Table.read(metafilename, format='csv')
    nobjects = len(metadata)

    lcfilename = 'PLAsTiCC-2018/training_set.csv'
    lcdata = Table.read(lcfilename, format='csv')
    from tqdm import trange

    X_train = []
    y_train = []

    for i in trange(nobjects):
        row = metadata[i]
        thisid = row['object_id']
        target = row['target']

        ind = (lcdata['object_id'] == thisid)
        thislc = lcdata[ind]

        pbind = [(thislc['passband'] == pb) for pb in pbmap]
        t = [thislc['mjd'][mask].data - thislc['mjd'][mask].data[0] for mask in pbind]
        m = [thislc['flux'][mask].data for mask in pbind]
        e = [thislc['flux_err'][mask].data for mask in pbind]
        detected = [thislc['detected'][mask].data for mask in pbind]


        obj = np.concatenate((pad_sequences(t,maxlen=72, dtype='float32',padding="pre"), pad_sequences(m,maxlen=72, dtype='float32',padding="pre"), pad_sequences(e,maxlen=72, dtype='float32',padding="pre"), pad_sequences(detected,maxlen=72, dtype='float32',padding="pre"))).T

        X_train.append(obj)
        y_train.append(target)

    return np.stack(X_train), np.stack(y_train)

from tcn import TCN

def build_model_tcn():

    input0 = Input(shape=(72, 24), dtype='float32', name='raw_lightcurves')

    dense = Dense(128, activation='relu')(input0)

    tcn_output = TCN(nb_filters=64, dropout_rate=0.5, return_sequences=False, nb_stacks=3, kernel_size=2, padding='same', name='TCN1')(dense)  # The TCN layers are here.

    # tcn_output = LSTM(100, return_sequences=False)(input0)  # The TCN layers are here.

    final_output = Dense(14, activation='softmax')(tcn_output)

    return Model(inputs=[input0], outputs=[final_output])


def tcn_modeling_cross_validation(X,
                                   y,
                                   classes,
                                   nr_fold=5,
                                   random_state=7):


    unique_y = np.unique(y)
    class_map = dict()
    for i, val in enumerate(unique_y):
        class_map[val] = i

    y_map = np.array([class_map[val] for val in y])
    y_categorical = to_categorical(y_map)

    y_count = Counter(y_map)
    wtable = np.zeros((len(unique_y),))
    for i in range(len(unique_y)):
        wtable[i] = y_count[i] / y_map.shape[0]

    def mywloss(y_true, y_pred):
        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
        loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0) / wtable))
        return loss

    oof_preds = np.zeros((len(X), np.unique(y).shape[0]))

    epochs = 50
    batch_size = 256

    clfs = []
    folds = StratifiedKFold(n_splits=nr_fold,
                            shuffle=True,
                            random_state=random_state)

    # X[:, :, :6] -= X[:, :1, :6]

    # print("a")
    # ------------- lgbm weights ----------------#
    # w = y.value_counts()
    # weights = {i: np.sum(w) / w[i] for i in w.index}

    for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):

        scaler = TimeSeriesMinMaxScaler()


        checkPoint = ModelCheckpoint("./keras.model", monitor='val_loss', mode='min', save_best_only=True, verbose=0)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=20, min_lr=1e-6, verbose=True)

        x_train, y_train = X[trn_], y_categorical[trn_]
        x_valid, y_valid = X[val_], y_categorical[val_]


        scaler.fit(x_train)
        # print(x_train)
        x_train = scaler.transform(x_train)
        # print(x_train)

        x_valid = scaler.transform(x_valid)

        model = build_model_tcn()
        model.compile(loss=mywloss, optimizer='adam', metrics=['accuracy'])

        # print(x_train.shape)

        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size, shuffle=True, verbose=1, callbacks=[checkPoint, reduce_lr])

        print('Loading Best Model')
        model = tf.keras.models.load_model('./keras.model', custom_objects={'mywloss': mywloss})
        # # Get predicted probabilities for each class
        oof_preds[val_, :] = model.predict(x_valid, batch_size=batch_size)
        # print(multi_weighted_logloss(y_valid, model.predict_proba(x_valid, batch_size=batch_size)))
        clfs.append(model)

        print('no {}-fold loss: {}'.format(fold_ + 1,
                                           multi_weighted_logloss_keras(y_valid, oof_preds[val_, :])))


        # del model
        # gc.collect()

    cnf = confusion_matrix(y_map, np.argmax(oof_preds, axis=-1))

    plot_confusion_matrix(cnf, classes=classes, normalize=True,
                          filename="keras_tcn")

    score = multi_weighted_logloss_keras(y_categorical, oof_preds)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))

    return clfs, score


def main():

    X, y = featurize_raw()
    np.savez('kaggle_tcn_kernel_data', X, y)

    # npzfile = np.load('kaggle_tcn_kernel_data.npz')
    # X = npzfile['arr_0']
    # y = npzfile['arr_1']


    classes = sorted(np.unique(y))
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    class_weights = {c: 1 for c in classes}
    class_weights.update({c: 2 for c in [64, 15]})
    print('Unique classes : {}, {}'.format(len(classes), classes))
    print(class_weights)
    # sanity check: classes

    tcn_modeling_cross_validation(
                        X,
                        y,
                        classes=classes,
                        nr_fold=5, random_state=1)




if __name__ == '__main__':
    main()