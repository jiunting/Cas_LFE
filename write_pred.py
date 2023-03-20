#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:14:09 2023

@author:Tim Lin
@email:jiunting@uoregon.edu

 Read the raw waveform data from .h5 and get model prediction y
"""

import h5py
import numpy as np
import pandas as pd
from scipy import signal
import unet_tools
import glob
import csv

#===Parameters for ML model, do not change these=====
drop = 0
sr = 100
large = 2
std = 0.4
epsilon=1e-6
run_num = 'S003'
project_name = 'cut_daily'

#===Path for files, change if needed=================
all_T_file = "all_T_0.1_3.npy" # associated waveforms
sav_k_file = "sav_k_0.1_3.npy" # keys for the above dict that passes the threshold
st2detc_file = "st2detc_0.1_3.npy" # to get the id by starttime so that you can read data from h5 file
h5_path = "./Detections_S_new"

def ZEN2inp(Z,E,N,epsilon):
    # convert raw ZEN data to model input
    data_Z_sign = np.sign(Z)
    data_E_sign = np.sign(E)
    data_N_sign = np.sign(N)
    data_Z_val = np.log(np.abs(Z)+epsilon)
    data_E_val = np.log(np.abs(E)+epsilon)
    data_N_val = np.log(np.abs(N)+epsilon)
    data_inp = np.hstack([data_Z_val.reshape(-1,1),data_Z_sign.reshape(-1,1),
                          data_E_val.reshape(-1,1),data_E_sign.reshape(-1,1),
                          data_N_val.reshape(-1,1),data_N_sign.reshape(-1,1),])
    return data_inp

def read_data_from_h5(sta_h5,evid):
    """
    Read ZEN data by id from h5py file. 
    """
    sta = '.'.join(evid.split('_')[0].split('.')[:2])
    if sta not in sta_h5:
        h5file = h5_path+"/"+project_name+"_"+sta+".hdf5"
        F = h5py.File(h5file,'r')
        sta_h5[sta] = F
    else:
        F = sta_h5[sta]
    # start getting data
    ZEN = np.array(F['data'][evid])
    Z = ZEN[0]
    E = ZEN[1]
    N = ZEN[2]
    nor = max(max(np.abs(Z)),max(np.abs(E)),max(np.abs(N)))
    Z = Z/nor
    E = E/nor
    N = N/nor
    return Z, E, N

    
channel_list = {
    'CN.GOBB':'EH',
    'CN.LZB':'BH',
    'CN.MGB':'EH',
    'CN.NLLB':'BH',
    'CN.PFB':'HH',
    'CN.PGC':'BH',
    'CN.SNB':'BH',
    'CN.VGZ':'BH',
    'CN.YOUB':'HH',
    'PO.GOWB':'HH',
    'PO.KLNB':'HH',
    'PO.SILB':'HH',
    'PO.SSIB':'HH',
    'PO.TSJB':'HH',
    'PO.TWGB':'HH',
    'PO.TWKB':'HH',
}


# load model
if drop:
    model=unet_tools.make_large_unet_drop(large,sr,ncomps=3)
else:
    model=unet_tools.make_large_unet(large,sr,ncomps=3)

chks = glob.glob("./checks/large*%s*"%(run_num))
chks.sort()
model.load_weights(chks[-1].replace('.index',''))


# load all detections, and keys that are ready to used.
all_T = np.load(all_T_file, allow_pickle=True).item()
sav_k = np.load(sav_k_file)
st2detc = np.load(st2detc_file, allow_pickle=True).item()


# print the sav_k for example
print(f'There are a total of {len(sav_k)} associated events.')
print(f'The first one is {sav_k[0]}, and there are {all_T[sav_k[0]]["num"]} stations detect an arrival in this time window')
# print the detection for example
for sta in all_T[sav_k[0]]['sta']:
    print(f' {sta}: {st2detc[sta][sav_k[0]]}')


import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


# Load the raw data from h5 files, convert to model input, run model and get arrival prediction.
idx = 0 # This is the example of first associated event, loop this (i.e. range(len(sav_k))) to get more events.
k = sav_k[idx]
sta_h5 = {}
for i_plot, sta in enumerate(all_T[k]['sta']):
    evid = '_'.join([sta+'.'+channel_list[sta], st2detc[sta][k]]) # get the evid and read it from h5 file
    Z,E,N = read_data_from_h5(sta_h5, evid)
    # convert ZEN to model input
    X = ZEN2inp(Z,E,N,epsilon)
    inp = ZEN2inp(Z,E,N,epsilon) # this is the actual model input
    y = model.predict(inp[np.newaxis,:,:])[0] # this is the model prediction (arrival probability)
    plt.plot(np.arange(len(y))/sr, y+i_plot, '-', label=sta)
    plt.plot(np.arange(len(y))/sr, np.ones(len(y))*i_plot, 'k--',lw=0.5)

plt.legend()
plt.xlabel('Time (s)',fontsize=14)
plt.title(f'Window start:{k}',fontsize=14)
plt.savefig('Example_prediction.png')
#plt.show()
plt.close()






