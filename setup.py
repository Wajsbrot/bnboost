#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
    name = "bnboost",
    packages = ["bnboost",],
    version = "0.1",
    description = "Codes for the Kaggle Airbnb competition.",
    author = "Nicolas Thiebaut",
    author_email = "nkthiebaut@gmail.com",
    url = "http://",
    download_url = "http://",
    keywords = ["kaggle",],
    install_requires=["xgboost",],
    classifiers = ["Programming Language :: Python :: 2",
		   "Intended Audience :: Developers"]
)


