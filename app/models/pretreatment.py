#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:51:49 2016

@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
@company: Quantmetry
"""

import numpy as np
import pandas as pd

df_train = pd.read_csv('../data/train_users_2.csv', index_col='id')
df_test = pd.read_csv('../data/test_users.csv', index_col='id')
sessions = pd.read_csv('../data/sessions_features.csv', index_col='user_id')

df_train = df_train.join(sessions)
df_test = df_test.join(sessions)


def clean(df):
    df.replace('-unknown-', np.nan, inplace=True)
    df.loc[df.age > 80, 'age'] = np.nan
    df.loc[df.age < 18, 'age'] = np.nan
    df['timestamp_first_active'] = \
        pd.to_datetime(df.timestamp_first_active.astype(str),
                       format='%Y%m%d%H%M%S')

    df['date_account_created'] = pd.to_datetime(df['date_account_created'])

clean(df_train)
clean(df_test)


def get_top_modalities(col, max_categorical=10):
    return col.value_counts()[:max_categorical]


def relevant_modalities(col, min_occurences=10, lim=None, n_out=None):
    """ Find most represented modalities """
    normalize = lim is not None
    val_counts = col.value_counts(normalize=normalize)
    if n_out is not None:
        return list(val_counts[:n_out].index)
    elif lim is not None:
        return list(val_counts[(val_counts > lim)].index)
    else:
        return list(val_counts[(val_counts > min_occurences)].index)


def group_items(col, relevant_modalities, impute=None):
    """ Group items of col not present in list relevant modalities in one
    modality """
    if impute is None:
        impute = \
         col.min()-1 if col.dtype in (np.float64, np.int64) else '-rare_item-'
    return col.apply(lambda x:
                     x if pd.isnull(x) or x in relevant_modalities else impute)

for c in ['first_browser', 'affiliate_provider', 'signup_flow', 'language']:
    top = relevant_modalities(df_train[c])
    df_train[c] = group_items(df_train[c], top)
    df_test[c] = group_items(df_test[c], top)


def treat_dates(df):
    df['delta_create_active'] = \
        (df['date_account_created']-df['timestamp_first_active']).dt.days
# is_delta = delta_create_active.apply(lambda x: 1 if x > 0 else 0)
# is_delta.name = 'is_delta'

    col_name = 'date_account_created'
    date_col = pd.to_datetime(df[col_name]).dt
    df[col_name + '_year'] = date_col.year
    df[col_name + '_month'] = date_col.month
    df[col_name + '_weekday'] = date_col.weekday

    df['first_active_hour'] = \
        pd.to_datetime(df['timestamp_first_active']).dt.hour
    df.drop(['timestamp_first_active', 'date_account_created',
             'date_first_booking'], axis=1, inplace=True)

treat_dates(df_train)
treat_dates(df_test)


def one_hot_encode(df):
    ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language',
                 'affiliate_channel', 'affiliate_provider',
                 'first_affiliate_tracked', 'signup_app', 'first_device_type',
                 'first_browser', 'date_account_created_year',
                 'date_account_created_month', 'date_account_created_weekday']
    for f in ohe_feats:
        df_dummy = pd.get_dummies(df[f], prefix=f, dummy_na=True)
        df = df.drop([f], axis=1)
        df = pd.concat((df, df_dummy), axis=1, copy=False)
    return df

df_train = one_hot_encode(df_train)
df_test = one_hot_encode(df_test)

df_train.fillna(-999.0, inplace=True)
df_test.fillna(-999.0, inplace=True)

irrelevant_cols = \
    list(set(df_train.columns).symmetric_difference(df_test.columns))
irrelevant_cols.remove('country_destination')

df_train.drop(irrelevant_cols, axis=1, inplace=True, errors='ignore')
df_test.drop(irrelevant_cols, axis=1, inplace=True, errors='ignore')

df_train.to_csv('../data/train.csv')
df_test.to_csv('../data/test.csv')
