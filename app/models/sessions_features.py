#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 01:22:22 2016

@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
@company: Quantmetry
"""
import numpy as np
import pandas as pd

df = pd.read_csv('../data/sessions.csv')
df.replace('-unknown-', np.nan, inplace=True)

grouped = df.groupby(['user_id'])

output = pd.DataFrame()
output[['n_pages', 'dt_clicks']] = \
    grouped['secs_elapsed'].agg(['size', 'mean'])

output['n_devices'] = grouped['device_type'].nunique()
# output['device_type'] = \
#  grouped['device_type'].agg(lambda x: x.value_counts(dropna=False).index[0])

output.to_csv('../data/sessions_features.csv')
