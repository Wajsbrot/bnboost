#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 03:23:59 2016

@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
@company: Quantmetry
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareItemsGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, max_categorical=10, impute='rare_item'):
        self.col_name = col_name
        self.max_categorical = max_categorical
        self.impute = impute

    def fit(self, X, y=None):
        """ Give all rare items of a pandas Series the same name/value """
        self.top_modalities = \
            X[self.col_name].value_counts()[:self.max_categorical]
        return self

    def transform(self, X):
        def group_rare(x):
            if pd.isnull(x) or x in self.top_modalities:
                return x
            else:
                return self.impute

        return group_rare(X[self.col_name])
