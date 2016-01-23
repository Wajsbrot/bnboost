#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:49:02 2016

@author: Nicolas Thiebaut
@email: nthiebaut@quantmetry.com
@company: Quantmetry
"""

import pandas as pd
from os.path import join, basename
from utils import find_categorical


def make_audit_file(infile='../data/sample_adinfo_1.csv', desc_file=None,
                    desc_dir='../out', decimal_sep='.', sep=','):
    """ Generate a description file for datas in a csv file

    Parameters
    ----------
    infile: str
        csv file containing data
    desc_file: str
        csv file in which output is written
    desc_dir: str
        directory where description file is written
    decimal_sep: str
        decimal values separator
    sep: str
        csv file columns separator

    """
    def top_values(df, n_values=5, sep=', ', with_count=True):
        """ Retrieve most represented modalities for each column of a dataframe

        Parameters
        ----------
        df: pandas.DataFrame
            input dataframe
        n_values: int
            numbers of top modalities kept
        sep: str
            modalities separator in output column
        with_count: bool
            if True, write count of modalities in percent next to it

        Returns
        -------
        pandas.Series
            Series with the input DataFrame column names as indices, and the
            top modalities concatenated in a single line as values

        """
        ds = pd.Series([])
        for col_name in df:
            top = df[col_name].value_counts(normalize=True)\
                .iloc[:n_values]
            if with_count:
                top = [str(t)+' ('+str(round(100*top[t], 2))+' %)'
                       for t in top.index]
            else:
                top = top.index.astype(str)
            out = (sep).join(top)
            ds = ds.append(pd.Series([out], index=[col_name]))
        return ds

    if desc_file is None:
        desc_file = join(desc_dir, basename(infile)[:-4] + '_desc.csv')

    df = pd.read_csv(infile, sep=sep, decimal=decimal_sep)

    raw_desc = df.describe(include='all').T
    # raw_desc.rename(columns={'': 'col_name'}, inplace=True)

    null_percent = 100 * df.isnull().sum() / df.shape[0]
    null_percent = null_percent.to_frame(name='null_percent')

    # types = df.dtypes.to_frame(name='types')

    nunique = df.apply(pd.Series.nunique).to_frame(name='n_unique')

    top = top_values(df).to_frame(name='top_10')

    cat_cols = find_categorical(df)
    col_type = pd.Series(raw_desc.index, index=raw_desc.index,
                         name='col_type').isin(cat_cols)
    col_type = col_type.apply(lambda x: 'cat' if x else 'num')

    desc = pd.concat([raw_desc, null_percent, col_type, nunique, top],
                     axis=1, copy=False)

    kept_columns = ['col_type', 'null_percent', 'n_unique', 'mean',
                    'min', 'max', 'top_10']

    desc[kept_columns].to_csv(desc_file, sep=';', float_format='%.2f')


if __name__ == '__main__':
    files = ['../data/train_users_2.csv']
    for f in files:
        make_audit_file(f)
