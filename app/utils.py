#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:12:33 2016

@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
@company: Quantmetry
"""

import numpy as np
import pandas as pd


def find_categorical(df, threshold=5):
    """ Find categorical columns in dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        input dataframe
    threshold: int
        number of modalities below which a column is considered to be
        categorical, even if filled with ints of floats

    Returns
    -------
        list: categorical columns names

    """
    n_unique = df.apply(pd.Series.nunique)  # count unique values in each col
    categorical_cols = n_unique[n_unique <= threshold].index
    non_numerical_cols = df.select_dtypes(exclude=['int', 'float']).columns
    categorical_cols = set(categorical_cols).union(non_numerical_cols)
    return list(categorical_cols)


def group_rare_items(col, lim=0.001, impute=None, n_out=None):
    """ Give all rare items of a pandas Series the same name/value

    Parameters
    ----------
    col: pandas.Series
        Series to be transformed
    lim: float
        limit ratio of the series below which a value is considered rare
    impute: str, int of float
        object to use for rare items replacement, automatic if None
    n_out: int
        number of top values that are not rare. If n_out is None then the rare
        items are defined by the lim parameter

    Returns
    -------
        pandas.Series: transformed Series

    """
    val_counts = col.value_counts(normalize=True)
    if impute is None:
        col.min()-1 if col.dtype in (np.float64, np.int64) else '-rare_item-'

    def g(x):
        if n_out is not None:
            return x if pd.isnull(x) or x in val_counts[:n_out] else impute
        else:
            return x if pd.isnull(x) or val_counts[x] > lim else impute

    return col.apply(g)
