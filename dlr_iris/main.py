#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 09:49:23 2019

@author: Harshvardhan
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

DATA_PATH = 'data'
DATA_FILE = 'iris.data'

col_names = ['sl', 'sw', 'pl', 'pw', 'labels']

df_data = pd.read_csv(os.path.join(DATA_PATH, DATA_FILE), header=0, names=col_names)
le = LabelEncoder()
le.fit(df_data['labels'])




