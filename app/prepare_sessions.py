#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 22:23:34 2016

@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
@company: Quantmetry
"""

import numpy as np
import pandas as pd

df = pd.read_csv('../data/sessions.csv')

df.replace('-unknown-', np.nan, inplace=True)

total_time = df.groupby(['user_id'])['secs_elapsed'].sum()
