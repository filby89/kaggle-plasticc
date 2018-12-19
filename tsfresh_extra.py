import tsfresh
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import feature_calculators
import numpy as np
import math
from scipy.optimize import curve_fit
import sncosmo
from astropy.table import Table
# data = sncosmo.load_example_data()
from gatspy.periodic import LombScargleMultiband, LombScargleMultibandFast
from cesium.time_series import TimeSeries
import cesium.featurize as featurize

model = sncosmo.Model(source='salt2')
import pywt

from collections import OrderedDict


@feature_calculators.set_property("fctype", "combiner")
def wavelet(x, param):
    cA, cD = pywt.dwt(x, 'db2')
    print(np.array(cA).shape, np.array(cD).shape)
    # return cA
feature_calculators.__dict__["wavelet"] = wavelet


@feature_calculators.set_property("fctype", "simple")
def lomba(x):
    model = LombScargleMultibandFast(fit_period=True)
    # print(x)
    mjd = x[:,0]
    flux = x[:,1]
    flux_err = x[:,2]
    passband = x[:,3]


    t_min = max(np.median(np.diff(sorted(mjd))), 0.1)
    t_max = min(10., (mjd.max() - mjd.min()) / 2.)

    model.optimizer.set(period_range=(t_min, t_max), first_pass_coverage=5, quiet=True)
    model.fit(mjd, flux, dy=flux_err, filts=passband)


    return model.best_period


#
feature_calculators.__dict__["lomb"] = lomba

@feature_calculators.set_property("fctype", "simple")
def SALT2(x):
    lsst_band = {
        0: "lsstu",
        1: "lsstg",
        2: "lsstr",
        3: "lssti",
        4: "lsstz",
        5: "lssty",
    }

    x['passband'].replace(lsst_band, inplace=True)
    x['zp'] = 0
    x['zpsys'] = 'ab'

    x = x.rename(columns={'passband': 'band'})

    x.loc[x['band'] == 'lsstu', 'zp'] = 9.17
    x.loc[x['band'] == 'lsstg', 'zp'] = 50.7
    x.loc[x['band'] == 'lsstr', 'zp'] = 43.7
    x.loc[x['band'] == 'lssti', 'zp'] = 32.36
    x.loc[x['band'] == 'lsstz', 'zp'] = 22.68
    x.loc[x['band'] == 'lssty', 'zp'] = 10.58

    # print(x)
    x = Table.from_pandas(x)

    # magsys = sncosmo.CompositeMagSystem(bands={'lsstu': ('ab', 9.16), 'lsstg': ('ab', 50.7), "lsstr": ('ab', 43.7), "lssti": ('ab', 32.36), "lsstz": ('ab', 22.68), "lssty": ("ab", 10.58)})
    # sncosmo.registry.register(magsys, name="my")

    # print(x)
    try:
        result, fitted_model = sncosmo.fit_lc(
            x, model,
            ['z', 't0', 'x0', 'x1', 'c'],  # parameters of model to vary
            bounds={'z': (0.3, 0.7)})  # bounds on parameters (if any)

        print(result)
    except:
        return 0
    return 0




@feature_calculators.set_property("fctype", "simple")
def StetsonK(x):
    magnitude = x[:,1]
    error = x[:,2]

    mean_mag = (np.sum(magnitude / (error * error)) /
                np.sum(1.0 / (error * error)))

    N = len(magnitude)
    sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
              (magnitude - mean_mag) / error)

    K = (1 / np.sqrt(N * 1.0) *
         np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))

    return K



@feature_calculators.set_property("fctype", "combiner")
def counts(x, param=None):
    negatives = np.sum((x < 0))
    positives = np.sum((x > 0))
    ratio = negatives / (positives + negatives)
    return [
        ('negatives', negatives), ('positives', positives), ('ratio_neg', ratio)
    ]

feature_calculators.__dict__["counts"] = counts



@feature_calculators.set_property("fctype", "simple")
def MedianBRP(x):
    magnitude = x

    median = np.median(magnitude)
    amplitude = (np.max(magnitude) - np.min(magnitude)) / 10
    n = len(magnitude)

    count = np.sum(np.logical_and(magnitude < median + amplitude,
                                  magnitude > median - amplitude))

    return float(count) / n

@feature_calculators.set_property("fctype", "simple")
def SmallKurtosis(x):
    magnitude = x

    n = len(magnitude)
    mean = np.mean(magnitude)
    std = np.std(magnitude)

    S = sum(((magnitude - mean) / std) ** 4)

    try:
        c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    except:
        return 0

    return c1 * S - c2


@feature_calculators.set_property("fctype", "simple")
def MedianAbsDev(x):
    magnitude = x

    median = np.median(magnitude)

    devs = (abs(magnitude - median))

    return np.median(devs)


@feature_calculators.set_property("fctype", "simple")
def PercentDifferenceFluxPercentile(x):
    magnitude = x

    median_data = np.median(magnitude)

    sorted_data = np.sort(magnitude)
    lc_length = len(sorted_data)
    F_5_index = math.ceil(0.05 * lc_length)
    F_95_index = math.ceil(0.95 * lc_length)

    if F_95_index == lc_length:
        F_95_index -= 1

    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]

    percent_difference = F_5_95 / median_data

    return percent_difference

@feature_calculators.set_property("fctype", "simple")
def PercentAmplitude(x):
    magnitude = x

    median_data = np.median(magnitude)
    distance_median = np.abs(magnitude - median_data)
    max_distance = np.max(distance_median)

    percent_amplitude = max_distance / median_data

    return percent_amplitude


@feature_calculators.set_property("fctype", "simple")
def FluxPercentileRatioMid20(x):
    magnitude = x

    sorted_data = np.sort(magnitude)
    lc_length = len(sorted_data)

    if lc_length < 10:
        return 0

    F_60_index = math.ceil(0.60 * lc_length)
    F_40_index = math.ceil(0.40 * lc_length)
    F_5_index = math.ceil(0.05 * lc_length)
    F_95_index = math.ceil(0.95 * lc_length)

    if F_95_index == lc_length:
        F_95_index -= 1

    F_40_60 = sorted_data[F_60_index] - sorted_data[F_40_index]
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
    F_mid20 = F_40_60 / F_5_95

    return F_mid20


@feature_calculators.set_property("fctype", "simple")
def FluxPercentileRatioMid35(x):
    magnitude = x

    sorted_data = np.sort(magnitude)
    lc_length = len(sorted_data)
    if lc_length < 10:
        return 0

    F_325_index = math.ceil(0.325 * lc_length)
    F_675_index = math.ceil(0.675 * lc_length)
    F_5_index = math.ceil(0.05 * lc_length)
    F_95_index = math.ceil(0.95 * lc_length)

    if F_95_index == lc_length:
        F_95_index -= 1

    F_325_675 = sorted_data[F_675_index] - sorted_data[F_325_index]
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
    F_mid35 = F_325_675 / F_5_95

    return F_mid35

@feature_calculators.set_property("fctype", "simple")
def FluxPercentileRatioMid50(x):
    magnitude = x

    sorted_data = np.sort(magnitude)
    lc_length = len(sorted_data)
    if lc_length < 10:
        return 0

    F_25_index = math.ceil(0.25 * lc_length)
    F_75_index = math.ceil(0.75 * lc_length)
    F_5_index = math.ceil(0.05 * lc_length)
    F_95_index = math.ceil(0.95 * lc_length)

    if F_95_index == lc_length:
        F_95_index -= 1

    F_25_75 = sorted_data[F_75_index] - sorted_data[F_25_index]
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
    F_mid50 = F_25_75 / F_5_95

    return F_mid50


@feature_calculators.set_property("fctype", "simple")
def FluxPercentileRatioMid65(x):
    magnitude = x

    sorted_data = np.sort(magnitude)
    lc_length = len(sorted_data)
    if lc_length < 10:
        return 0

    F_175_index = math.ceil(0.175 * lc_length)
    F_825_index = math.ceil(0.825 * lc_length)
    F_5_index = math.ceil(0.05 * lc_length)
    F_95_index = math.ceil(0.95 * lc_length)
    if F_95_index == lc_length:
        F_95_index -= 1

    F_175_825 = sorted_data[F_825_index] - sorted_data[F_175_index]
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
    F_mid65 = F_175_825 / F_5_95

    return F_mid65

@feature_calculators.set_property("fctype", "simple")
def FluxPercentileRatioMid80(x):
    magnitude = x

    sorted_data = np.sort(magnitude)
    lc_length = len(sorted_data)
    if lc_length < 10:
        return 0

    F_10_index = math.ceil(0.10 * lc_length)
    F_90_index = math.ceil(0.90 * lc_length)
    F_5_index = math.ceil(0.05 * lc_length)
    F_95_index = math.ceil(0.95 * lc_length)
    if F_95_index == lc_length:
        F_95_index -= 1

    if F_90_index == lc_length:
        F_90_index -= 1

    F_10_90 = sorted_data[F_90_index] - sorted_data[F_10_index]
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
    F_mid80 = F_10_90 / F_5_95

    return F_mid80
