#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:26:52 2018

@author: Harshvardhan Gazula
@acknowledgments: Eswar Damaraju for showing the relevant MATLAB commands
@notes: Contains code to extract information from the MCIC NIfTI files
"""
import os
import pickle
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.io import loadmat


def nifti_to_data(image_files_location, mask_array):
    """Extracts data from the nifti image_files

    Args:
        image_files_location (string): Path to the folder where the images are
                                        stored
        mask_array (array): Array from Mask file for processing the nii files

    Returns:
        List of numpy arrays from each NIfTI file

    """
    appended_data = []

    image_files = list(image_files_location)

    # Extract Data (after applying mask)
    for image in image_files:
        print(image.split('/')[9])
        image_data = nib.load(image).get_data()
        appended_data.append(image_data[mask_array > 0])

    return appended_data


file_list = pd.read_csv(
    "fBIRNp3_VBM_swc1_list.txt", sep=" ", header=None, names=['file_path'])
file_list["file_index"] = file_list["file_path"].apply(
    lambda x: x.split('/')[9])

QA_list = pd.read_csv(
    "fBIRNp3_QA_withfnm.txt",
    index_col=None,
    header=None,
    names=['meaningless', 'qa_val'],
    delim_whitespace=True)
QA_list['file_index'] = file_list["file_path"].apply(lambda x: x.split('/')[9])
QA_list.drop(columns='meaningless', inplace=True)

file_list["qa_val"] = QA_list["qa_val"]

demo_mat = loadmat("demographics_fbirnp3_ALL.mat")
abc = list(demo_mat["header_ltd"])
file_headers = [str(x[0][0]) for x in abc if x[0].size > 0]

demo_pd = pd.DataFrame(demo_mat["demo_data_ltd"])
demo_pd.drop(columns=[223, 224, 225], inplace=True)
demo_pd.columns = file_headers
demo_pd["SubjectID"] = demo_pd['SubjectID'].apply(
    lambda x: '{:>012.0f}'.format(x))

demo_columns = [
    "SiteID", "Demographics_nDEMOG_CUR_AGE", "Demographics_nDEMOG_DIAGNOSIS"
]

abc = pd.merge(
    file_list,
    demo_pd,
    left_on='file_index',
    right_on='SubjectID',
    how='inner')
abc1 = abc[abc["qa_val"] >= 0.90]

data_location = '/export/mialab/users/hgazula/fbirn_regression/fbirn_data'
mask_location = os.path.join(data_location, 'mask')

# %% Extracting Mask Data
mask_file = os.path.join(mask_location, 'MNI152_T1_2mm_mask.nii')
mask_data = nib.load(mask_file).get_data()

# %% Reading Voxel Info into an Array
print('Extracting Info from NIFTI files')
patient_data = nifti_to_data(abc1["file_path"], mask_data)

voxels = pd.DataFrame(np.vstack(patient_data))

# %% (Redundant) Reading demographic information
patient_demographics = abc1[demo_columns]
patient_demographics = patient_demographics.rename(
    columns={"Demographics_nDEMOG_DIAGNOSIS": "diagnosis",
             "Demographics_nDEMOG_CUR_AGE": "age",
             "SiteID": "site"})
# %% Replacing all diagnosis values with either 'Patient' or 'Control'
# 1 - Patient # 2 - Control
patient_demographics['diagnosis'][patient_demographics['diagnosis'] ==
                                  1] = 'Patient'
patient_demographics['diagnosis'][patient_demographics['diagnosis'] ==
                                  2] = 'Control'

patient_demographics['site'] = patient_demographics['site'].apply(
    lambda x: '{:>02d}'.format(int(x)))
demographics = pd.concat([patient_demographics], axis=0).reset_index(drop=True)

# %% Writing to a file
print('writing data to a pickle file')
with open("final_data.pkl", "wb") as f:
    pickle.dump((demographics, voxels), f)

print('Finished Running Script')
