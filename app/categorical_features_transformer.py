#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 03:23:59 2016

@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
@company: Quantmetry
"""

#!/usr/bin/env python

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import NotFittedError

class CategoricalFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col=None, strategy=None, categories=None, tolerance=0.01, new_col=None):
        self.col = col
        self.strategy = strategy
        self.categories = categories
        self.tolerance = tolerance
        self.new_col = new_col

    def fit(self, X, y=None):
        if self.strategy == "dummify":
            # Get list of categories
            categories = self.categories if self.categories else X[self.col].unique()

            # Compute dummies
            def dummify(data):
                def cat_encoder(cat):
                    name = "%s_%s" % ((self.new_col if self.new_col else self.col), cat)
                    return data[self.col].apply(lambda x: 1 if x == cat else 0).to_frame(name = name)
                dummyCols = map(cat_encoder, categories)
                return pd.concat(dummyCols, axis = 1)

            self.__transform = dummify

        elif self.strategy == "project":
            Xy = pd.concat([X, y], axis=1)

            # Get categories frequencies
            val_counts = Xy[self.col].value_counts(normalize=True)

            # Group rare categories
            group_rare_items = lambda x: x if pd.isnull(x) or val_counts[x] > self.tolerance else 'Rare_items'
            Xy[self.col] = Xy[self.col].apply(group_rare_items)

            # Build impact dict
            gmeans = Xy.groupby(self.col)[y.name].mean()
            if 'Rare_items' not in gmeans:
                gmeans['Rare_items'] = 0.0

            # Transform
            def project(x):
                if pd.isnull(x):
                    return x
                else:
                    try:
                        if val_counts[x] > self.tolerance:
                            return gmeans[x]
                        else:
                            return gmeans['Rare_items']
                    except KeyError:
                        return gmeans['Rare_items']

            self.__transform = lambda data: data[self.col].apply(project).to_frame(name=self.new_col)

        elif self.strategy == "zero":
                self.__transform = lambda data: data[self.col].fillna("0").to_frame(name = self.new_col)

        else:
            raise ValueError("Wrong param: unknown strategy %s" % self.strategy)

        return self

    def transform(self, X):
        try:
            return self.__transform(X)
        except AttributeError:
            raise NotFittedError(self.__class__.__name__ + " not fitted")
