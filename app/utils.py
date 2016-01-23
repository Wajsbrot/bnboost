#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:12:33 2016

@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
@company: Quantmetry
"""

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
