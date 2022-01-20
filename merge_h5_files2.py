#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:55:16 2020

Merge H5 files

@author: amt
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import glob



'''
#get all data/noise name
dataName = glob.glob('./tmp_LFE/file_*.npy')

# Need to merge .h5 files
allData=np.empty((0,9003))

#merge data
for fileName in dataName:
    d = np.load(fileName)
    allData = np.vstack([allData,d])

#save LFE data
h5f = h5py.File("Cascadia_lfe_data.h5", 'w')  
h5f.create_dataset('waves', data=allData)
h5f.close()
'''





#-------start processing noise data
#get all data/noise name
noiseName = glob.glob('./tmp_LFE/noise_*.npy')

# Need to merge .h5 files
allNoise=np.empty((0,9003))

#merge data
for fileName in noiseName:
    d = np.load(fileName)
    d = d - d.mean(axis=1).reshape(-1,1) #rmean
    allNoise = np.vstack([allNoise,d])

#save LFE data
#h5f = h5py.File("Cascadia_noise_data.h5", 'w')  
h5f = h5py.File("Cascadia_noise_QC_rmean.h5", 'w')  
h5f.create_dataset('waves', data=allNoise)
h5f.close()


