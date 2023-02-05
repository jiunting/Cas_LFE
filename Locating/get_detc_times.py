#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 22:26:17 2023 

@author:Tim Lin
@email:jiunting@uoregon.edu

"""

import numpy as np
import pandas as pd
import glob
from obspy import UTCDateTime
import datetime


def dect_time(file_path: str, thresh=0.1, is_catalog: bool=False, EQinfo: str=None, return_all: bool=False) -> np.ndarray:
    """
    Get detection time for 1). ML detection in .csv file or 2). catalog file.
    If input is .csv file, return the `window` starttime so that is easier to match with other stations
    
    INPUTs
    ======
    file_path: str
        Path of the detection file
    is_catalog: bool
        Input is catalog or not
    thresh: float
        Threshold of selection. Only apply when is_catalog==False (i.e. using ML detection csv files)
    EQinfo: str
        Path of the EQinfo file i.e. "sav_family_phases.npy"
    return_all: bool
        Also return deection time, not just the window starttime
    OUTPUTs
    =======
    sav_OT: np.array[datetime] or None if empty detection
        Detection time in array. For ML detection, time is the arrival at the station. For catalog, time is the origin time
    
    EXAMPLEs
    ========
    #T1 = dect_time('./Data/total_mag_detect_0000_cull_NEW.txt',True,'./Data/sav_family_phases.npy')  #for catalog
    #T2 = dect_time('./results/Detections_S_small/cut_daily_PO.GOWB.csv')                             # for ML detection output
    """
    if is_catalog:
        EQinfo = np.load(EQinfo,allow_pickle=True) #Note! detection file has a shift for each family
        EQinfo = EQinfo.item()
        sav_OT = []
        with open(file_path,'r') as IN1:
            for line in IN1.readlines():
                line = line.strip()
                ID = line.split()[0] #family ID
                OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
                HH = int(line.split()[2])*3600
                SS = float(line.split()[3])
                OT = OT + HH + SS + EQinfo[ID]['catShift'] #detected origin time
                sav_OT.append(OT.datetime)
        sav_OT.sort()
        return pd.Series(sav_OT)
        #sav_OT = np.array(sav_OT)
        #return sav_OT
    else:
        csv = pd.read_csv(file_path)
        if len(csv)==0:
            return None
        T = csv[csv['y']>=thresh]
        if return_all:
            return T.starttime.values, np.array([i.split('_')[-1] for i in T.id])
        else:
            return T.starttime.values
        #net_sta = file_path.split('/')[-1].split('_')[-1].replace('.csv','')
        #prepend = csv.iloc[0].id.split('_')[0]
        #T = pd.to_datetime(csv[csv['y']>=thresh].id,format=prepend+'_%Y-%m-%dT%H:%M:%S.%f')
        #T.reset_index(drop=True, inplace=True) #reset the keys e.g. [1,5,6,10...] to [0,1,2,3,...]
        return T

#------settings-------
thres = 0.1 # y>=thres to be a detection
N_min = 3 # minumum stations have detection at the same time
detc_dir = "../Detections_S_new/*.csv"

csvs = glob.glob(detc_dir)

#load the detection files and get the starttime and detection time for each station
sav_T = {} # net.sta:starttime
st2detc = {} #net.sta: starttime->detection time
for csv in csvs:
    print('Now in :',csv)
    net_sta = csv.split('/')[-1].split('_')[-1].replace('.csv','')
    tmp = dect_time(csv, thresh=0.1, return_all=True)
    if tmp is None:
        continue
    T, Td = tmp
    sav_T[net_sta] = T
    st2detc[net_sta] = {T[i]:Td[i] for i in range(len(T))}

# calculate the number of detections (across stations) in a same time window
# improvement: also consider nearby time window
print('-----Initialize all_T. Min=1, Max=%d'%(len(sav_T)))
all_T = {}
for k in sav_T.keys():
    print('dealing with:',k)
    for t in sav_T[k]:
        if t not in all_T:
            all_T[t] = {'num':1, 'sta':[k]} #['num'] = 1
        else:
            all_T[t]['num'] += 1
            all_T[t]['sta'].append(k)

# get the detections where pass the threshold
sav_k = [] # keys (i.e. window starttime) that pass the filter
for k in all_T.keys():
    if all_T[k]['num']>=N_min:
        sav_k.append(k)

sav_k.sort()

#=====EXAMPLE=====
"""
# The 1th detection. Starting by this time with 15 s long, there're at least N_min detections
print('The 1st detection starting from T:',sav_k[0])
print('There are %d detections:'%(all_T[sav_k[0]]['num']),all_T[sav_k[0]]['sta'])
# From starttime to exact detection time
print('Window starttime=',sav_k[0])
for ista in all_T[sav_k[0]]['sta']:
    print('  %s -> %s'%(ista,st2detc[ista][sav_k[0]]))
"""
#=====EXAMPLE END=====


for k in sav_k:
    print('Window start:',k,'; %d detections.'%(len(all_T[k]['sta'])))
    for ista in all_T[k]['sta']:
        print('  ',st2detc[ista][k],ista)
    break
