# -*- coding: utf-8 -*-
"""
Created on Wed May 30 18:39:34 2018

@author: hgazula
"""

import numpy as np
import os
import stat
from load_data_from_mat import return_X_and_y

X, y = return_X_and_y()

print('Creating job_scripts folder')
jobs_folder = 'job_scripts'
if not os.path.exists(jobs_folder):
    os.makedirs(jobs_folder)

start = np.arange(1, X.shape[1], 25)
stop = start - 1
stop = np.append(stop, X.shape[1])

print('Writing job submission script files')
for start_index, start_val in enumerate(start):
    first_arg = start_val
    second_arg = stop[start_index + 1]

    file_name = 'job_{:04}_{:04}.sh'.format(first_arg, second_arg)
    with open(os.path.join(jobs_folder, file_name), 'w') as fsub:
        fsub.write('echo \'Hello There!\'')
        fsub.write('\n')
        fsub.write('python exhaustive_selection.py {} {}'.format(
            first_arg, second_arg))

    st = os.stat(os.path.join(jobs_folder, file_name))
    os.chmod(os.path.join(jobs_folder, file_name), st.st_mode | stat.S_IEXEC)

    with open('job.sh', 'a') as fjob:
        fjob.write('screen -dm bash ' + os.path.join(jobs_folder, file_name))
        fjob.write('\n')

print('Creating the main job.sh file')
st = os.stat('job.sh')
os.chmod('job.sh', st.st_mode | stat.S_IEXEC)
