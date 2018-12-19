import sys, os
import argparse
import time
from datetime import datetime as dt
import gc;
import matplotlib
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


import math
import feets
import feets.preprocess

fs = feets.FeatureSpace(data=['magnitude', 'time', 'error'], only=['StetsonK', 'SlottedA_length', 'StetsonK_AC'])

import time



feature_calculators.__dict__["FluxPercentileRatioMid80"] = FluxPercentileRatioMid80
feature_calculators.__dict__["FluxPercentileRatioMid20"] = FluxPercentileRatioMid20
feature_calculators.__dict__["FluxPercentileRatioMid35"] = FluxPercentileRatioMid35
feature_calculators.__dict__["FluxPercentileRatioMid50"] = FluxPercentileRatioMid50
feature_calculators.__dict__["FluxPercentileRatioMid65"] = FluxPercentileRatioMid65

feature_calculators.__dict__["SmallKurtosis"] = SmallKurtosis
feature_calculators.__dict__["MedianAbsDev"] = MedianAbsDev
feature_calculators.__dict__["MedianBRP"] = MedianBRP
feature_calculators.__dict__["PercentDifferenceFluxPercentile"] = PercentDifferenceFluxPercentile
feature_calculators.__dict__["PercentAmplitude"] = PercentAmplitude
feature_calculators.__dict__["StetsonK"] = StetsonK
feature_calculators.__dict__["SALT2"] = SALT2
feature_calculators.__dict__["freq_harmonics"] = freq_harmonics


@jit
def haversine_plus(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) from
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Implementing Haversine Formula:
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),
               np.multiply(np.cos(lat1),
                           np.multiply(np.cos(lat2),
                                       np.power(np.sin(np.divide(dlon, 2)), 2))))

    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    return {
        'haversine': haversine,
        'latlon1': np.subtract(np.multiply(lon1, lat1), np.multiply(lon2, lat2)),
    }


@jit
def process_flux(df):
    flux_ratio_sq = np.power(df['flux'].values / df['flux_err'].values, 2.0)

    df_flux = pd.DataFrame({
        'flux_ratio_sq': flux_ratio_sq,
        'flux_by_flux_ratio_sq': df['flux'].values * flux_ratio_sq, },
        index=df.index)

    return pd.concat([df, df_flux], axis=1)


@jit
def process_flux_agg(df):
    flux_w_mean = df['flux_by_flux_ratio_sq_sum'].values / df['flux_ratio_sq_sum'].values
    flux_diff = df['flux_max'].values - df['flux_min'].values

    df_flux_agg = pd.DataFrame({
        'flux_w_mean': flux_w_mean,
        'flux_diff1': flux_diff,
        'flux_diff2': flux_diff / df['flux_mean'].values,
        'flux_diff3': flux_diff / flux_w_mean,
    }, index=df.index)

    return pd.concat([df, df_flux_agg], axis=1)




def featurize(df, df_meta, aggs, fcp, n_jobs=12):
    """
    Extracting Features from train set
    Features from olivier's kernel
    very smart and powerful feature that is generously given here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    per passband features with tsfresh library. fft features added to capture periodicity https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70346#415506
    """
    # df = process_flux_distance(df, df_meta)
    df = process_flux(df)

    agg_df = df.groupby('object_id').agg(aggs)
    agg_df.columns = ['{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]
    agg_df = process_flux_agg(agg_df)  # new feature to play with tsfresh


    # Add more features with
    #
    df_fake = df
    df_fake['extra'] = 0
    agg_df_ts_flux_my = extract_features(df,
                                               column_id='object_id',
                                               column_sort='mjd',
                                                column_kind="extra",
                                               column_value=['mjd', 'flux', 'flux_err', 'passband'],
                                               default_fc_parameters=fcp['my_features'], n_jobs=12)


    agg_df_ts_flux_passband = extract_features(df,
                                               column_id='object_id',
                                               column_sort='mjd',
                                               column_kind='passband',
                                               column_value='flux',
                                               default_fc_parameters=fcp['flux_passband'], n_jobs=n_jobs)

    agg_df_ts_flux = extract_features(df,
                                      column_id='object_id',
                                      column_value='flux',
                                      default_fc_parameters=fcp['flux'], n_jobs=n_jobs)

    agg_df_ts_flux_by_flux_ratio_sq = extract_features(df,
                                                       column_id='object_id',
                                                       column_value='flux_by_flux_ratio_sq',
                                                       default_fc_parameters=fcp['flux_by_flux_ratio_sq'],
                                                       n_jobs=n_jobs)



    # Add smart feature that is suggested here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    df_det = df[df['detected'] == 1].copy()
    agg_df_mjd = extract_features(df_det,
                                  column_id='object_id',
                                  column_value='mjd',
                                  default_fc_parameters=fcp['mjd'], n_jobs=n_jobs)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']

    agg_df_ts_flux_passband.index.rename('object_id', inplace=True)
    agg_df_ts_flux.index.rename('object_id', inplace=True)
    agg_df_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True)
    agg_df_mjd.index.rename('object_id', inplace=True)
    agg_df_ts_flux_my.index = agg_df_mjd.index

    # cesium = pd.read_csv("/gpu-data/filby/plastic/train_set_full_features.csv", index_col='object_id')
    # cesium = cesium[["time_score", "__qso_log_chi2_qsonu___0_", "__median_absolute_deviation___1_", "__median_absolute_deviation___2_", "__median_absolute_deviation___5_"]]
    # print(cesium)

    agg_df_ts = pd.concat([agg_df,
                           agg_df_ts_flux_passband,
                           agg_df_ts_flux,
                           agg_df_ts_flux_by_flux_ratio_sq,
                           agg_df_mjd, agg_df_ts_flux_my], axis=1).reset_index()
    print(agg_df_ts)

    result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')
    return result


def process_meta(filename):
    meta_df = pd.read_csv(filename)

    meta_dict = dict()
    # distance
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values,
                                    meta_df['gal_l'].values, meta_df['gal_b'].values))
    #
    meta_dict['hostgal_photoz_certain'] = np.multiply(
        meta_df['hostgal_photoz'].values,
        np.exp(meta_df['hostgal_photoz_err'].values))

    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)
    return meta_df


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

def lgbm_multi_weighted_logloss(y_true, y_preds):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weights = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    loss = multi_weighted_logloss(y_true, y_preds, classes, class_weights)
    return 'wloss', loss, False



def save_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    plt.figure(figsize=(8, 12))
    import seaborn as sns
    sns.barplot(x='gain', y='feature', data=importances_.sort_values('mean_gain', ascending=False))
    plt.tight_layout()
    plt.savefig('importances.png')
    return importances_



from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow.keras
from collections import Counter
from sklearn.metrics import confusion_matrix
from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def build_model(dropout_rate=0.25, activation='relu', input_shape=100):
    start_neurons = 1024
    # create model
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=input_shape, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 2, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 4, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 8, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate / 2))

    model.add(Dense(14, activation='softmax'))
    return model




def build_model_stack(dropout_rate=0, activation='relu', input_shape=100):
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, activation=activation))
    model.add(Dense(64, input_dim=input_shape, activation=activation))

    model.add(Dense(14, activation='softmax'))
    return model

def stack_modeling_cross_validation(params,
                                   full_train,
                                   y,
                                   classes,
                                   class_weights,
                                   nr_fold=10,
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

    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))

    epochs = 200
    batch_size = 256

    clfs = []
    folds = StratifiedKFold(n_splits=nr_fold,
                            shuffle=True,
                            random_state=random_state)

    # ------------- lgbm weights ----------------#
    # w = y.value_counts()
    # weights = {i: np.sum(w) / w[i] for i in w.index}

    for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):

        scaler = MinMaxScaler()


        checkPoint = ModelCheckpoint("./keras.model", monitor='val_loss', mode='min', save_best_only=True, verbose=0)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=20, min_lr=1e-6, verbose=True)

        x_train, y_train = full_train[trn_], y_categorical[trn_]
        x_valid, y_valid = full_train[val_], y_categorical[val_]


        scaler.fit(x_train)

        x_train = scaler.transform(x_train)

        x_valid = scaler.transform(x_valid)

        model = build_model_stack(dropout_rate=0, activation='relu', input_shape=x_train.shape[1])
        model.compile(loss=mywloss, optimizer='sgd', metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size, shuffle=True, verbose=1, callbacks=[checkPoint, reduce_lr])

        print('Loading Best Model')
        model = tf.keras.models.load_model('./keras.model', custom_objects={'mywloss': mywloss})
        # # Get predicted probabilities for each class
        oof_preds[val_, :] = model.predict_proba(x_valid, batch_size=batch_size)
        # print(multi_weighted_logloss(y_valid, model.predict_proba(x_valid, batch_size=batch_size)))
        clfs.append(model)

        print('no {}-fold loss: {}'.format(fold_ + 1,
                                           multi_weighted_logloss_keras(y_valid, oof_preds[val_, :])))


        # del model
        # gc.collect()

    cnf = confusion_matrix(y_map, np.argmax(oof_preds, axis=-1))

    plot_confusion_matrix(cnf, classes=classes, normalize=True,
                          filename="stacked")

    score = multi_weighted_logloss_keras(y_categorical, oof_preds)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))

    return clfs, score, oof_preds



def nn_modeling_cross_validation(params,
                                   full_train,
                                   y,
                                   classes,
                                   class_weights,
                                   nr_fold=10,
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

    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))

    epochs = 200
    batch_size = 256

    clfs = []
    folds = StratifiedKFold(n_splits=nr_fold,
                            shuffle=True,
                            random_state=random_state)

    # ------------- lgbm weights ----------------#
    # w = y.value_counts()
    # weights = {i: np.sum(w) / w[i] for i in w.index}

    for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):

        scaler = MinMaxScaler()


        checkPoint = ModelCheckpoint("./keras.model", monitor='val_loss', mode='min', save_best_only=True, verbose=0)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=20, min_lr=1e-6, verbose=True)

        x_train, y_train = full_train.iloc[trn_], y_categorical[trn_]
        x_valid, y_valid = full_train.iloc[val_], y_categorical[val_]


        scaler.fit(x_train)

        x_train = scaler.transform(x_train)

        x_valid = scaler.transform(x_valid)

        model = build_model(dropout_rate=0.5, activation='relu', input_shape=x_train.shape[1])
        model.compile(loss=mywloss, optimizer='sgd', metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size, shuffle=True, verbose=0, callbacks=[checkPoint, reduce_lr])

        print('Loading Best Model')
        model = tf.keras.models.load_model('./keras.model', custom_objects={'mywloss': mywloss})
        # # Get predicted probabilities for each class
        oof_preds[val_, :] = model.predict_proba(x_valid, batch_size=batch_size)
        # print(multi_weighted_logloss(y_valid, model.predict_proba(x_valid, batch_size=batch_size)))
        clfs.append(model)

        print('no {}-fold loss: {}'.format(fold_ + 1,
                                           multi_weighted_logloss_keras(y_valid, oof_preds[val_, :])))


        # del model
        # gc.collect()

    cnf = confusion_matrix(y_map, np.argmax(oof_preds, axis=-1))

    plot_confusion_matrix(cnf, classes=classes, normalize=True,
                          filename="keras_nn")

    score = multi_weighted_logloss_keras(y_categorical, oof_preds)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))

    return clfs, score, oof_preds

from imblearn.over_sampling import SMOTE


def smoteAdataset(Xig_train, yig_train, Xig_test, yig_test):
    sm = SMOTE(random_state=2)
    Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())

    return Xig_train_res, pd.Series(yig_train_res), Xig_test, pd.Series(yig_test)

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def lgbm_modeling_cross_validation(params,
                                   full_train,
                                   y,
                                   classes,
                                   class_weights,
                                   nr_fold=10,
                                   random_state=7):

    unique_y = np.unique(y)
    class_map = dict()
    for i, val in enumerate(unique_y):
        class_map[val] = i

    # y = np.array([class_map[val] for val in y])
    y = y.apply(lambda x: class_map[x])

    # Compute weights
    w = y.value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}

    clfs = []
    importances = pd.DataFrame()
    folds = StratifiedKFold(n_splits=nr_fold,
                            shuffle=True,
                            random_state=random_state)

    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]

        trn_xa, trn_y, val_xa, val_y = smoteAdataset(trn_x.values, trn_y.values, val_x.values, val_y.values)
        trn_x = pd.DataFrame(data=trn_xa, columns=trn_x.columns)

        val_x = pd.DataFrame(data=val_xa, columns=val_x.columns)


        clf = LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgbm_multi_weighted_logloss,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )

        clf.my_name = "lgbm"

        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x)#, num_iteration=clf.best_iteration_)
        print('no {}-fold loss: {}'.format(fold_ + 1,
                                           multi_weighted_logloss(val_y, oof_preds[val_, :],
                                                                  classes, class_weights)))

        imp_df = pd.DataFrame({
            'feature': full_train.columns,
            'gain': clf.feature_importances_,
            'fold': [fold_ + 1] * len(full_train.columns),
        })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    score = multi_weighted_logloss(y_true=y, y_preds=oof_preds,
                                   classes=classes, class_weights=class_weights)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))
    df_importances = save_importances(importances_=importances)
    df_importances.to_csv('lgbm_importances.csv', index=False)


    cnf = confusion_matrix(y, np.argmax(oof_preds, axis=1))
    plot_confusion_matrix(cnf, classes=classes, normalize=True,
                          filename="lgbm")



    return clfs, score, oof_preds


def predict_chunk(df_, clfs_, meta_, features, featurize_configs, train_mean, scaler):
    # process all features
    full_test = featurize(df_, meta_,
                          featurize_configs['aggs'],
                          featurize_configs['fcp'])
    full_test.fillna(0, inplace=True)

    # Make predictions
    preds_ = None
    for clf in clfs_:
        if clf.my_name == "lgbm":
            t = full_test[features]
        else:
            t = scaler.transform(full_test[features])

        if preds_ is None:
            preds_ = clf.predict_proba(t)
        else:
            preds_ += clf.predict_proba(t)

    preds_ = preds_ / len(clfs_)

    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds_.shape[0])
    for i in range(preds_.shape[1]):
        preds_99 *= (1 - preds_[:, i])

    # Create DataFrame from predictions
    preds_df_ = pd.DataFrame(preds_,
                             columns=['class_{}'.format(s) for s in clfs_[1].classes_])
    preds_df_['object_id'] = full_test['object_id']
    preds_df_['class_99'] = 0.14 * preds_99 / np.mean(preds_99)
    return preds_df_


def process_test(clfs,
                 features,
                 featurize_configs,
                 train_mean,scaler,
                 filename='predictions.csv',
                 chunks=5000000):
    start = time.time()

    meta_test = process_meta('PLAsTiCC-2018/test_set_metadata.csv')
    # meta_test.set_index('object_id',inplace=True)

    remain_df = None
    for i_c, df in enumerate(pd.read_csv('/gpu-data/filby/plastic/test_set.csv', chunksize=chunks, iterator=True)):
        if chunks *(i_c+1) < 420000000 :
            print("Continuing {}".format(i_c))
            continue
        # Check object_ids
        # I believe np.unique keeps the order of group_ids as they appear in the file
        unique_ids = np.unique(df['object_id'])

        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remain_df = new_remain_df

        preds_df = predict_chunk(df_=df,
                                 clfs_=clfs,
                                 meta_=meta_test,
                                 features=features,
                                 featurize_configs=featurize_configs,
                                 train_mean=train_mean, scaler=scaler)

        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=False)

        del preds_df
        gc.collect()
        print('{:15d} done in {:5.1f} minutes'.format(
            chunks * (i_c + 1), (time.time() - start) / 60), flush=True)

    # Compute last object in remain_df
    preds_df = predict_chunk(df_=remain_df,
                             clfs_=clfs,
                             meta_=meta_test,
                             features=features,
                             featurize_configs=featurize_configs,
                             train_mean=train_mean, scaler=scaler)

    preds_df.to_csv(filename, header=False, mode='a', index=False)
    return


def main(args):
    # Features to compute with tsfresh library. Fft coefficient is meant to capture periodicity

    # agg features
    aggs = {
        'flux': ['min', 'max', 'mean', 'median', 'std'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['mean'],
        'flux_ratio_sq': ['sum', 'skew'],
        'flux_by_flux_ratio_sq': ['sum', 'skew'],
    }

    # tsfresh features
    fcp = {
        'flux': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,
            # 'mean_change': None,
            'length': None,
            "PercentDifferenceFluxPercentile": None,
            "PercentAmplitude": None,
            "MedianBRP": None,
            "MedianAbsDev": None,
            "SmallKurtosis": None,
            'FluxPercentileRatioMid35': None,
            'FluxPercentileRatioMid50': None,
            'FluxPercentileRatioMid65': None,
            'FluxPercentileRatioMid80': None,
            'FluxPercentileRatioMid20': None,
            'kurtosis': None,
            'skewness': None,
            "first_location_of_maximum": None,
            "counts": None
        },

        'my_features': {
            'StetsonK': None,
            # 'freq_harmonics': None
            # "lomb": None
        },

        'flux_by_flux_ratio_sq': {
            'longest_strike_above_mean': None,
            'longest_strike_below_mean': None,
        },

        'flux_passband': {
            'fft_coefficient': [
                    {'coeff': 0, 'attr': 'abs'},
                    {'coeff': 1, 'attr': 'abs'},
            ],
            'kurtosis' : None,
            'skewness' : None,
            'abs_energy': None,
            'mean': None,
            'maximum': None,
            'median': None,
            "sample_entropy": None,
            "quantile": [{"q": 0.7}, {"q": 0.9}],
            "binned_entropy": [{"max_bins": 10}],
            "fft_aggregated": [{"aggtype": "kurtosis"}, {"aggtype": "skew"}],
            "ratio_beyond_r_sigma": [{"r": 0.5}],
            "cid_ce": [{"normalize": True}],
            "autocorrelation": [{"lag": 1}],
            "time_reversal_asymmetry_statistic": [{"lag": 1}],
            "first_location_of_maximum": None,
            # "counts": None, # for dnn

        },
        'mjd': {
            'maximum': None,
            'minimum': None,
            'mean_change': None,
            'mean_abs_change': None,
        },
    }

    best_params = {
        'device': 'gpu',
        'objective': 'multiclass',
        'num_class': 14,
        'boosting_type': 'dart',
        'n_jobs': -1,
        'max_depth': 7,
        'n_estimators': 500,
        'subsample_freq': 2,
        'subsample_for_bin': 5000,
        'min_data_per_group': 100,
        'max_cat_to_onehot': 4,
        'cat_l2': 1.0,
        'cat_smooth': 59.5,
        'max_cat_threshold': 32,
        'metric_freq': 10,
        'verbosity': -1,
        'metric': 'multi_logloss',
        'xgboost_dart_mode': False,
        'uniform_drop': False,
        'colsample_bytree': 0.5,
        'drop_rate': 0.173,
        'learning_rate': 0.0267,
        'max_drop': 5,
        'min_child_samples': 10,
        'min_child_weight': 100.0,
        'min_split_gain': 0.1,
        'num_leaves': 7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.00023,
        'skip_drop': 0.44,
        'subsample': 0.75}

    meta_train = process_meta('PLAsTiCC-2018/training_set_metadata.csv')

    train = pd.read_csv('PLAsTiCC-2018/training_set.csv')

    saved_features = "/gpu-data/filby/plastic/kaggle_tsfresh_features.csv"
    if os.path.exists(saved_features):# and False:
        full_train = pd.read_csv(saved_features)
    else:
        full_train = featurize(train, meta_train, aggs, fcp)
        full_train.to_csv(saved_features, index=False)

    import seaborn as sns
    colormap = plt.cm.RdBu
    plt.figure(figsize=(20, 18))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    corr = full_train.astype(float).corr()
    labels = corr.round(2).astype(str)
    labels[corr < 0.95] = ""
    sns.heatmap(corr, linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=labels, fmt="", xticklabels=1, yticklabels=1, annot_kws={"size": 7, "color": "black"})
    plt.savefig("corr.png")

    if 'target' in full_train:
        y = full_train['target']
        del full_train['target']

    classes = sorted(y.unique())
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    class_weights = {c: 1 for c in classes}
    class_weights.update({c: 2 for c in [64, 15]})
    print('Unique classes : {}, {}'.format(len(classes), classes))
    print(class_weights)
    # sanity check: classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    # sanity check: class_weights = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # if len(np.unique(y_true)) > 14:
    #    classes.append(99)
    #    class_weights[99] = 2

    if 'object_id' in full_train:
        oof_df = full_train[['object_id']]
        del full_train['object_id']
        #del full_train['distmod']
        del full_train['hostgal_specz']
        del full_train['ra'], full_train['decl'], full_train['gal_l'], full_train['gal_b']
        del full_train['ddf']

    train_mean = full_train.mean(axis=0)
    # train_mean.to_hdf('train_data.hdf5', 'data')
    # pd.set_option('display.max_rows', 500)
    # print(full_train.describe().T)
    # import pdb; pdb.set_trace()

    full_train = full_train.replace([np.inf, -np.inf], np.nan)
    full_train.fillna(0, inplace=True)


    full_train = full_train.drop(['flux_mean', '4__mean', '3__mean', '4__mean', '2__mean', '3__fft_coefficient__coeff_0__attr_"abs"', '3__sample_entropy'], axis=1)
    best_params.update({'n_estimators': 2000})


    redshift = full_train['hostgal_photoz']
    print(np.mean(redshift), np.max(redshift), np.min(redshift))


    if args.model == "lgbm":
        eval_func = partial(lgbm_modeling_cross_validation,
                            full_train=full_train,
                            y=y,
                            classes=classes,
                            class_weights=class_weights,
                            nr_fold=5,
                            random_state=1)
        clfs, score, oof_preds = eval_func(best_params)
    elif args.model == "nn":
        eval_func = partial(nn_modeling_cross_validation,
                            full_train=full_train,
                            y=y,
                            classes=classes,
                            class_weights=class_weights,
                            nr_fold=5,
                            random_state=1)
        clfs, score, oof_preds = eval_func(best_params)

    elif args.model == "stack":
        eval_func = partial(lgbm_modeling_cross_validation,
                            full_train=full_train,
                            y=y,
                            classes=classes,
                            class_weights=class_weights,
                            nr_fold=5,
                            random_state=1)



        clfs, score, oof_preds = eval_func(best_params)

        eval_func = partial(nn_modeling_cross_validation,
                            full_train=full_train,
                            y=y,
                            classes=classes,
                            class_weights=class_weights,
                            nr_fold=5,
                            random_state=1)

        clfs1, score1, oof_preds_nn1 = eval_func(best_params)

        train_level_1 = np.concatenate((oof_preds, oof_preds_nn1), axis=1)

        np.save("train_level_1", train_level_1)

        eval_func = partial(stack_modeling_cross_validation,
                            full_train=train_level_1,
                            y=y,
                            classes=classes,
                            class_weights=class_weights,
                            nr_fold=5,
                            random_state=1)

        clfs1, score1, oof_preds_nn1 = eval_func(best_params)


    if args.test:

        filename = '2_subm_{:.6f}_{}.csv'.format(score,
                                               dt.now().strftime('%Y-%m-%d-%H-%M'))
        print('save to {}'.format(filename))
        # TEST
        process_test(clfs,
                     features=full_train.columns,
                     featurize_configs={'aggs': aggs, 'fcp': fcp},
                     scaler=None,
                     filename=filename,
                     train_mean=train_mean,
                     chunks=5000000)

        z = pd.read_csv(filename)
        print("Shape BEFORE grouping: {}".format(z.shape))
        z = z.groupby('object_id').mean()
        print("Shape AFTER grouping: {}".format(z.shape))
        z.to_csv('single_{}'.format(filename), index=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')

    # ========================= Optimizer Parameters ==========================
    parser.add_argument('--model', default="lgbm", type=str)
    parser.add_argument('--test', default=False, type=bool)

    args = parser.parse_args()

    main(args)