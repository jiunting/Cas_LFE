#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 3 19:55:57 2022 

@author:Tim Lin
@email:jiunting@uoregon.edu

"""

import glob
import numpy as np
import pandas as pd


#---------------parameter setting-----------------
N_min = 5 # at least N station detect arrival in the same 15s time window
threshold = 0.5 # decision threshold

use_P = False # add P arrival?
N_min_P = 3
threshold_P = 0.5

#excluded_sta = ['PO.KLNB', ] # excluded stations
excluded_sta = []

fileout = "./arrivals_sta%d_y%.1f.csv"%(N_min,threshold)

#---------------parameter setting END-----------------

def dect_time(file_path: str, thresh=0.1, is_catalog: bool=False, EQinfo: str=None) -> np.ndarray:
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
    
    OUTPUTs
    =======
    sav_OT: np.array[datetime] or None if empty detection
        Detection time in array. For ML detection, time is the arrival at the station. For catalog, time is the origin time
    
    EXAMPLEs
    ========
    #T1 = dect_time('./Data/total_mag_detect_0000_cull_NEW.txt',None,True,'./Data/sav_family_phases.npy')  #for catalog
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
    else:
        csv = pd.read_csv(file_path)
        if len(csv)==0:
            return None, None
        T = csv[csv['y']>=thresh].starttime.values
        T_arr = np.array([i.split('_')[1] for i in csv[csv['y']>=thresh].id.values])
        return T, T_arr


#STEP 1. =====Load all the detections from ML=====
csvs = glob.glob("./Detections_S_new/*.csv")
csvs.sort()

sav_T = {}
sav_T_arr = {} # link starttime to arrival
for csv in csvs:
    print('Now loading :',csv)
    net_sta = csv.split('/')[-1].split('_')[-1].replace('.csv','')
    if net_sta in excluded_sta:
        print(' Skip station:',net_sta)
        continue
    T,T_arr = dect_time(csv, thresh=threshold)
    if T is None:
        continue
    sav_T[net_sta] = T
    sav_T_arr[net_sta] = {i:j for i,j in zip(T,T_arr)}

#STEP 2. =====Count the number of detections in the same starttime window. For simplicity, this doesnt consider nearby windows i.e. +-15 s.
print('-----Initialize all_T, number of detections in each time window, (Min=1, Max=%d)'%(len(sav_T)))
all_T = {} # e.g. all_T['2006-03-01T22:53:00.000000Z'] = {'num':3, 'sta':['CN.LZB', 'CN.PGC', 'PO.SSIB']}
for k in sav_T.keys():
    print('dealing with:',k)
    for t in sav_T[k]:
        if t not in all_T:
            all_T[t] = {'num':1, 'sta':[k]} #['num'] = 1
        else:
            all_T[t]['num'] += 1
            all_T[t]['sta'].append(k)

#STEP 3. =====Apply min N stations filter=====
#N_min = 3 #define in the begining
sav_k = [] # keys that pass the filter
for k in all_T.keys():
    if all_T[k]['num']>=N_min:
        sav_k.append(k)

sav_k.sort()
#sav_k = np.array([UTCDateTime(i) for i in sav_k])
print('Total of candidate templates=%d after N_min=%d filter'%(len(sav_k), N_min))


#STEP 4. ======Add P arrival?, repeat STEP1-3 with P wave======
if use_P:
    csvs = glob.glob("./Detections_P_new/*.csv")
    csvs.sort()
    sav_T_P = {}
    sav_T_arr_P = {} # link starttime to arrival
    for csv in csvs:
        print('Now loading :',csv)
        net_sta = csv.split('/')[-1].split('_')[-1].replace('.csv','')
        if net_sta in excluded_sta:
            print(' Skip station:',net_sta)
            continue
        T,T_arr = dect_time(csv, thresh=threshold_P)
        if T is None:
            continue
        sav_T_P[net_sta] = T
        sav_T_arr_P[net_sta] = {i:j for i,j in zip(T,T_arr)}
    all_T_P = {}
    for k in sav_T_P.keys():
        print('dealing with:',k)
        for t in sav_T_P[k]:
            if t not in all_T_P:
                all_T_P[t] = {'num':1, 'sta':[k]} #['num'] = 1
            else:
                all_T_P[t]['num'] += 1
                all_T_P[t]['sta'].append(k)
    '''
    sav_k_P = [] # keys that pass the filter
    for k in all_T_P.keys():
        if all_T_P[k]['num']>=N_min_P:
            sav_k_P.append(k)
    '''

#STEP final. =====Find the corresponding arrival time and write the output=====
OUT1 = open(fileout,'w')
for st in sav_k: # for each window start time (i.e. st), get the corresponding detected arrival time for each stations
    results = {} #collecting result to write later
    for sta in all_T[st]['sta']: # loop the stations has detection in this start window
        Sarr = sav_T_arr[sta][st]
        results[sta] = {'S':Sarr}
        print('S arr=',Sarr)
    if use_P: # use P or not
        if st in all_T_P:
            if all_T_P[st]['num']>=N_min_P:
                for sta in all_T_P[st]['sta']:
                    Parr = sav_T_arr_P[sta][st]
                    if sta in results: 
                        results[sta]['P'] = Parr #this station has both S and P
                    else:
                        results[sta] = {'P':Parr}
                    print('P arr=',Parr)
    # write to file change the output format!
    for sta in results:
        if use_P:
            if 'P' in results[sta]:
                OUT1.write('%s,%s,'%(sta,results[sta]['P']))
            else:
                OUT1.write('%s,%s,'%(sta,'None'))
        else:
            OUT1.write('%s,'%(sta))
        if 'S' in results[sta]: 
            OUT1.write('%s\n'%(results[sta]['S']))
        else:
            OUT1.write('%s\n'%('None'))
    OUT1.write('#\n')



OUT1.close()



