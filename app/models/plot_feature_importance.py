#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 02:28:51 2016

@author: Nicolas Thiebaut
@email: nkthiebaut@gmail.com
@company: Quantmetry
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

n = 10

fi = pd.read_csv('../out/feature_importances.csv')
fi.sort('importance', ascending=False, inplace=True)

top = fi.iloc[:n]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(n), list(top['importance']),
        color="r", align="center", )
plt.xticks(range(n), list(top['feature']), rotation='vertical')
plt.xlim([-1, n])
plt.savefig('../figures/feature_importances.png')
