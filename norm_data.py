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

'''
#get all data/noise name
noiseName = glob.glob('./tmp_LFE/noise_*.npy')

# Need to merge .h5 files
allNoise=np.empty((0,9003))

#merge data
for fileName in noiseName:
    d = np.load(fileName)
    allNoise = np.vstack([allNoise,d])

#save LFE data
h5f = h5py.File("Cascadia_noise_data.h5", 'w')  
h5f.create_dataset('waves', data=allNoise)
h5f.close()
'''

'''
data = h5py.File("Cascadia_lfe_data.h5", 'r')
#also remove nan
maxA = np.max(np.abs(data['waves']),axis=1)
idx = np.where(maxA!=0)[0]
print('orig len=',len(data['waves']))
print('after len=',len(idx))
A = data['waves'][idx]/maxA[idx].reshape(-1,1)
#save LFE data
h5f = h5py.File("Cascadia_lfe_data_norm.h5", 'w')  
h5f.create_dataset('waves', data=A)
h5f.close()
'''

'''
data = h5py.File("Cascadia_noise_data.h5", 'r')
A = data['waves']/np.max(np.abs(data['waves']),axis=1).reshape(-1,1)
#norm = np.max(np.abs(data['waves']),axis=1).reshape(-1,1)
#idx = np.where(norm==0)[0]
#print('zeros=',idx)
#save LFE data
h5f = h5py.File("Cascadia_noise_data_norm.h5", 'w')
h5f.create_dataset('waves', data=A)
h5f.close()
'''









