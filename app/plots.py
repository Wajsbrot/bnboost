#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:37:55 2016

@author: Nicolas Thiebaut
@email: nthiebaut@quantmetry.com
@company: Quantmetry
"""
from math import sqrt, ceil, floor
import pandas as pd
import matplotlib.pyplot as plt
from os.path import basename
import numpy as np
from utils import find_categorical
# plt.style.use('ggplot')

import seaborn as sns
sns.set(color_codes=True)


import matplotlib
from matplotlib.ticker import FuncFormatter

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)



def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def make_plots(infile='../data/sample_ctxoeuv_1.csv', max_modalities=10,
               decimal_sep='.', sep=','):
    """ Plot distribution for features in infile

    Parameters
    ----------
    infile: str
        input csv file name
    max_modalities: int
        maximum number of different values for categoorical features
    decimal_sep: str
        decimal values separator
    sep: str
        csv file columns separator


    """
    df = pd.read_csv(infile, sep=sep, decimal=decimal_sep)

    df.dropna(axis=1, how='all', inplace=True)  # remove empty columns
#   df.drop('#rionPaiement', axis=1, inplace=True)  # remove mono-modality cols

    categorical_cols = find_categorical(df)

    # get rid of many-modalities columns
    categorical_cols = [c for c in df[categorical_cols]
                        if (df[c].value_counts().shape[0] < max_modalities)]

    numerical_cols = df.select_dtypes(include=['int', 'float']).columns
    numerical_cols = list(set(numerical_cols).difference(categorical_cols))

    # sns.pairplot(df[numerical_cols], size=5)

    # ----- Plot numerical features ----
    df[numerical_cols][(df >= df.quantile(0.1)) & (df <= df.quantile(0.9)) &
                       (df != 0)].dropna(axis=1, how='all').hist(
                       figsize=(20, 20), normed=True)

    plt.savefig('../figures/'+basename(infile)[:-4]+'_numerical.png')
    plt.clf()

    # ----- Plot categorical features ----
    n = len(categorical_cols)
    height = int(ceil(sqrt(n)))
    width = int(floor(sqrt(n)))
    fig, axes = plt.subplots(nrows=width, ncols=height,
                             figsize=(3*height, 3*width))

    for i in range(1, height*width-n+1):
        print i
        axes[-i, -1].axis('off')  # switch off unused subplots

    for i, c in enumerate(categorical_cols):
        x = i % width
        y = i % height
        df[c].value_counts(normalize=True, dropna=False).plot(kind='bar',
                                                              ax=axes[x, y])
        axes[x, y].set_title(c)
        axes[x, y].yaxis.set_major_formatter(formatter)
        axes[x, y].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
    fig.tight_layout()
    plt.savefig('../figures/'+basename(infile[:-4])+'_categorical.png')


if __name__ == '__main__':
    files = ['../data/train_users_2.csv']
    for f in files:
        make_plots(f, max_modalities=15)
