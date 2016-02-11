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
import argparse


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


def get_options():
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    p = argparse.ArgumentParser(description='Generate audit files.',
                                formatter_class=help_formatter)
    p.add_argument('filenames', type=str, nargs='+',
                   help='names of files under investigation')

    args = p.parse_args()
    return args


def nan_indicators(df, cols, threshold=2):
    """ Append nan indicator columns to a DataFrame

    Parameters
    ----------
    df: pandas.DataFrame
        input dataframe
    cols: list of str
        columns names for which indicator columns are appended

    Return
    ------
        pandas.DataFrame
            initiale DataFrame with appended nan indicators columns
    """
    nan_indicators_df = pd.DataFrame()
    for col in cols:
        nan_col = df[col].isnull()
        if nan_col.sum() >= threshold:
            nan_indicators_df['is_'+df[col].name+'_nan'] = \
                nan_col.astype(np.int64)
    return pd.concat([df, nan_indicators_df], copy=False, axis=1)


def values_indicators(df, indicators):
    """ Append indicator columns to a DataFrame

    Parameters
    ----------
    df: pandas.DataFrame
        input dataframe
    indicators: dict
        dictionary containing (column/searched value) pairs

    Returns
    -------
        pandas.DataFrame
    """
    values_indicators_df = pd.DataFrame()
    for col, match in indicators.items():
        values_indicators_df['is_'+df[col].name+'_'+str(match)] = \
            (df[col] == match).astype(np.int64)
    return pd.concat([df, values_indicators_df], copy=False, axis=1)


def xgb_grid_search(param_dict, dtrain):
    """ Perform grid search with cross-validation on XGBoost

    Parameters
    ----------
    param_dict: dict
        parameters with (parameter_name, possible values list) pairs
    dtrain: xgboost.DMatrix
        training data

    Returns
    -------
    xgboost.booster
        optimal model train on the whole training set
    dict
        optimal parameters dictionary

    pandas.Series
        scores for all parameters, stored in multi-indexed Series
    """
    def init_scores(p):
        """ Initialize Multi-indexed Pandas series used to store scores """
        search_values = filter(lambda x: isinstance(p[x], list), p.keys())
        search_params = {k: params[k] for k in search_values}
        n_points = reduce(lambda x,y: x*y, map(lambda x:len(x), search_params.values()))
        scores = pd.Series(np.zeros(n_points), index=pd.MultiIndex.from_product(search_params.values()))
        index_params = pd.MultiIndex.from_product(search_params.values(), names=search_params.keys())
        scores = pd.Series(np.zeros(n_points), index=index_params)
        return scores
    scores = init_scores(param_dict)
    search_param_names = scores.index.names
    # Loop over search_param values
    for sp in scores.index:
        for i, p in enumerate(sp):
            param_dict[search_param_names[i]] = p
        train_scores = xgb.cv(param_dict, dtrain, num_boost_round=5, nfold=3, metrics=(), obj=None, feval=ndcg_score,
                              maximize=True, early_stopping_rounds=None, fpreproc=None, as_pandas=True,
                              show_progress=None, show_stdv=True, seed=0)
        final_score = train_scores['test-ndcg5-mean'].iloc[-1]
        scores[sp] = final_score
    max_score = scores.max()
    max_params = [ round(x, 2) for x in scores.idxmax()]
    for i, p in enumerate(max_params):
        param_dict[search_param_names[i]] = p
    bst = xgb.train(param_dict, dtrain, num_boost_round=5, evals=(), obj=None, feval=None, maximize=False,
                    early_stopping_rounds=None, evals_result=None, verbose_eval=True, learning_rates=None,
                    xgb_model=None)
    return bst, param_dict, scores