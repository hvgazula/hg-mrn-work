# -*- coding: utf-8 -*-
"""
Created on Wed May 30 18:39:34 2018

@author: hgazula
"""

import numpy as np
from load_data_from_mat import return_X_and_y

X, y = return_X_and_y()

start = np.arange(1, X.shape[1], 25)
stop = start - 1
stop = np.append(stop, X.shape[1])

for start_index, start_val in enumerate(start):
    first_arg = start_val
    second_arg = stop[start_index+1]    
    
    file_name = 'job_{:04}_{:04}.sh'.format(first_arg, second_arg)
    with open(file_name, 'a+') as fsub:
        fsub.write('python exhaustive_selection.py {} {}'.format(first_arg, second_arg))

    with open('job.sh', 'a+') as fjob:
        fjob.write(['qsub ' + file_name])
        fjob.write('\n')         
    