#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:30:53 2018

@author: Harshvardhan
"""
import os
import stat

print('Creating job_scripts folder')
jobs_folder = 'job_scripts'
if not os.path.exists(jobs_folder):
    os.makedirs(jobs_folder)

max_runs = 30
folder_index = 2

print('Writing job submission script files')
for run_index in range(1, max_runs + 1):

    file_name = 'run_{:03}.sh'.format(run_index)
    with open(os.path.join(jobs_folder, file_name), 'w') as fsub:
        fsub.write('echo \'Running \' $0')
        fsub.write('\n')
        fsub.write('python sequential_selection.py {} {}'.format(folder_index, 
            run_index))

    st = os.stat(os.path.join(jobs_folder, file_name))
    os.chmod(os.path.join(jobs_folder, file_name), st.st_mode | stat.S_IEXEC)

    with open('job.sh', 'a') as fjob:
        fjob.write('screen -dm bash ' + os.path.join(jobs_folder, file_name))
        fjob.write('\n')

print('Creating the main job.sh file')
st = os.stat('job.sh')
os.chmod('job.sh', st.st_mode | stat.S_IEXEC)
