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
import os
from obspy import UTCDateTime
import datetime
import xarray as xr
import time
import dask
from dask.distributed import Client

# client = Client("127.0.0.1:8786")
#client = Client(n_workers=4, threads_per_worker=1)
#client = Client()
#client

#print('Dashboard link:',client.dashboard_link)

#------settings-------

BATCH_SIZE = 10000 # size to save/compute at a time
run_num = '001'

thres = 0.1 # y>=thres to be a detection
N_min = 3 # minumum stations have detection at the same time
detc_dir = "../Detections_S_new/*.csv"
load_if_possible = True # if not, re-run from .csv file anyway


# save the arival time results (sav_k and st2dect) so you wont load them again
all_T_file = "all_T_%.1f_%d.npy"%(thres,N_min)
sav_k_file = "sav_k_%.1f_%d.npy"%(thres,N_min)
st2detc_file = "st2detc_%.1f_%d.npy"%(thres,N_min)

# save the observations data so you wont read them again from the above sav_k, all_T, and st2detc
all_arrivals_file = "all_arrivals_%.1f_%d.npy"%(thres,N_min)

fout = "EQloc_%s_%.1f_%d_S.txt"%(run_num,thres,N_min)
fout_npy = "res_%s_%.1f_%d_S.npy"%(run_num,thres,N_min)

csvs = glob.glob(detc_dir)


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


def load_from_csv(csvs):
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
    return all_T, sav_k, st2detc



def sort_arrivals(sta_phase, arrival, T0, sort_list):
    """
    Sort the arrivals in the correct order used by grid search
    
    INPUTs
    ======
    sta_phase: list
        List of station name and phase with available arrival
    arrival: list
        List of arrival in string that can be read by UTCDateTime
    T0: str
        Window starttime or anytime that can be read by UTCDateTime
    sort_list: list
        List of sta_phase to be filled
        e.g. ['PFB_P', 'TWKB_P', 'TWGB_P', 'SNB_P','SNB_S', 'KLNB_S', 'TSJB_S']
    OUTPUTs
    =======
    res: np.ndarray
        An array of arrival time relative to the window st. np.nan if no data
    """
    res = [np.nan] * len(sort_list)
    name2idx = {sort_list[i]:i for i in range(len(sort_list))}
    for i,sta_phs in enumerate(sta_phase):
        res[name2idx[sta_phs]] = UTCDateTime(arrival[i]) - UTCDateTime(T0)
    return np.array(res)

def make_arrivals(sav_k,all_T,st2detc):
    all_arrivals = []
    for ik, k in enumerate(sav_k):
        sta_phase = []
        arr = []
        for ista in all_T[k]['sta']:
            sta_phase.append(ista.split('.')[1]+'_'+'S') #only S here
            arr.append(st2detc[ista][k])
        arrivals = sort_arrivals(sta_phase, arr, T0=k, sort_list=Travel['sta_phase'])
        all_arrivals.append(arrivals)
    return np.array(all_arrivals)


def split_dataset(data, batch_size):
    num_batches = (data.shape[0] + batch_size - 1) // batch_size  # round up to nearest integer
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, data.shape[0])
        batch = data[start_idx:end_idx]
        batches.append(batch)
    return batches


def core_funct(tt, arr):
    dt = arr - tt # arr.shape=(28), tt.shape=(1024800,28)
    dT0 = np.nanmean(dt,axis=1)
    dt = abs(dt.T - dT0) #dt.mean(axis=1)) # Calculate origin time shift, (i.e. positive if avg. arrival is later than tt), means T0 too late.
    dt = np.nanmean(dt,axis=0) #dt.nanmean(axis=0)
    return dt, dT0

# Step 1: Load the travel time table
Travel = np.load("./Travel.npy",allow_pickle=True)
Travel = Travel.item()
coords = np.array([coord for coord in Travel['T'].keys()])
TT = np.array([Travel['T'][coord] for coord in Travel['T'].keys()])


# Step 2: Load the observation data
if load_if_possible:
    if os.path.exists(all_arrivals_file):
        all_arrivals = np.load(all_arrivals_file)
        if os.path.exists(sav_k_file) & os.path.exists(all_T_file):
            sav_k = np.load(sav_k_file) # also read sav_k so that you know the starttime for each observation
            all_T = np.load(all_T_file, allow_pickle=True).item() # so you know how many available observations in each starttime
        else:
            all_T, sav_k, st2detc = load_from_csv(csvs)
            np.save(all_T_file,all_T) #save for next time
            np.save(sav_k_file, sav_k)
            np.save(st2detc_file, st2detc)
    else:
        if os.path.exists(all_T_file) & os.path.exists(sav_k_file) & os.path.exists(st2detc_file):
            all_T = np.load(all_T_file, allow_pickle=True).item()
            sav_k = np.load(sav_k_file)
            st2detc = np.load(st2detc_file,allow_pickle=True).item()
        else:
            all_T, sav_k, st2detc = load_from_csv(csvs)
            np.save(all_T_file,all_T) #save for next time
            np.save(sav_k_file, sav_k)
            np.save(st2detc_file, st2detc)
        # when all_T,sav_k, and st2detc are all ready
        all_arrivals = make_arrivals(sav_k,all_T,st2detc)
        np.save(all_arrivals_file,all_arrivals)
else: #redo everything
    all_T, sav_k, st2detc = load_from_csv(csvs)
    all_arrivals = make_arrivals(sav_k,all_T,st2detc)
    np.save(all_T_file,all_T) #save for next time
    np.save(sav_k_file, sav_k)
    np.save(st2detc_file, st2detc)
    np.save(all_arrivals_file,all_arrivals)

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

# Step3: Prepare ds dataset
ds = xr.Dataset(
            data_vars=dict(
                TT=(["grdidx", "sta"], TT,
                   dict(description="arrival time", units="s")),
                lon=(["grdidx"], coords[:,0],
                   dict(description="Longitude", units="degree")),
                lat=(["grdidx"], coords[:,1],
                   dict(description="Latitude", units="degree")),
                dep=(["grdidx"], coords[:,2],
                   dict(description="depth", units="km")),
                arrivals=(["idx_arrival", "sta"], all_arrivals[:100]) # here arrivals.shape=(N,28)
            ),
            coords=dict(
                idx=(["grdidx"], np.arange(TT.shape[0])),
                sta=(["sta"], np.arange(TT.shape[1])),
            )
).chunk({'idx_arrival':1})
ds


#res = xr.apply_ufunc(core_funct, ds.TT, ds.arrivals, vectorize=True,
#                   input_core_dims=[["grdidx", "sta"], ["sta"]], output_core_dims=[['grdidx']], dask="parallelized",
#                   ).compute()

res = xr.apply_ufunc(core_funct, ds.TT, ds.arrivals, vectorize=True,
                   input_core_dims=[["grdidx", "sta"], ["sta"]], output_core_dims=[['grdidx'],['grdidx']], dask="parallelized",
                   )


idx_batches = split_dataset(np.arange(len(res[0])),BATCH_SIZE)

with open(fout,'w') as OUT1:
    OUT1.write('starttime OT lon lat depth residual dt N\n')

for ii,idx in enumerate(idx_batches):
    result = dask.compute(res[0][idx],res[1][idx])
    dT0 = result[1].to_numpy()
    result = result[0].to_numpy()
    #np.save("_".join([fout_npy.split(".npy")[0], "%03d.npy"%ii])  ,result)
    with open(fout,'a') as OUT1:
        for i in range(result.shape[0]):
            idx_min = np.argmin(result[i])
            st = sav_k[idx[i]]
            OT = UTCDateTime(st) - dT0[i,idx_min]
            OUT1.write('%s %s %f %f %.2f %.6f %.6f %d\n'%(sav_k[idx[i]],OT.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],coords[idx_min,0],coords[idx_min,1],coords[idx_min,2],result[i,idx_min],dT0[i,idx_min],len(all_T[sav_k[idx[i]]]['sta'])))



