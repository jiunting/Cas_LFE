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
import xarray as xr
import time
from numba import float64, guvectorize


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
load_if_possible = True # if not, re-run from .csv file anyway


# save the arival time results (sav_k and st2dect) so you wont load them again
all_T_file = "all_T_%.1f_%d.npy"%(thres,N_min)
sav_k_file = "sav_k_%.1f_%d.npy"%(thres,N_min)
st2detc_file = "st2detc_%.1f_%d.npy"%(thres,N_min)

fout = "EQloc2_%.1f_%d_S.txt"%(thres,N_min)

csvs = glob.glob(detc_dir)

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


if load_if_possible:
    try:
        all_T = np.load(all_T_file, allow_pickle=True).item()
        sav_k = np.load(sav_k_file)
        st2detc = np.load(st2detc_file,allow_pickle=True).item()
    except:
        all_T, sav_k, st2detc = load_from_csv(csvs)
        np.save(all_T_file,all_T)
        np.save(sav_k_file, sav_k)
        np.save(st2detc_file, st2detc)
else:
    all_T, sav_k, st2detc = load_from_csv(csvs)
    np.save(all_T_file,all_T)
    np.save(sav_k_file, sav_k)
    np.save(st2detc_file, st2detc)


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


def grid_searching_loc(grid_loc, Travel, arrivals):
    sav_err = []
    for g in grid_loc:
        dt = arrivals - Travel[g]
        dt = dt[~np.isnan(dt)] #remove nan value
        dt = dt - np.mean(dt)
        sav_err.append(np.mean(np.abs(dt))) # error
    return np.array(sav_err)
    

def grid_searching_loc(loc, Travel, arrivals):
    dt = arrivals - Travel[loc]
    dt = dt[~np.isnan(dt)] #remove nan value
    dt = dt - np.mean(dt)
    return np.mean(np.abs(dt)) # error
    
        

# Load the travel time table
Travel = np.load("./Travel.npy",allow_pickle=True)
Travel = Travel.item()
grid_loc = list(Travel['T'].keys()) # 1,024,800 grid nodes to be searched i.e. #Shape of lon,lat,dep=(140, 120, 61)
grid_loc_np = np.array(grid_loc)
coords = np.array([coord for coord in Travel['T'].keys()])
TT = np.array([Travel['T'][coord] for coord in Travel['T'].keys()])

ds_var = xr.Dataset(
            data_vars=dict(
                TT=(["idx", "sta"], TT,
                   dict(description="arrival time", units="s")),
                lon=(["idx"], coords[:,0],
                   dict(description="Longitude", units="degree")),
                lat=(["idx"], coords[:,1],
                   dict(description="Latitude", units="degree")),
                dep=(["idx"], coords[:,2],
                   dict(description="depth", units="km")),
            ),
            coords=dict(
                idx=(["idx"], np.arange(TT.shape[0])),
                sta=(["sta"], np.arange(TT.shape[1])),
            )
    
).chunk({'idx':100})
ds_var


@guvectorize(
    "(float64[:], float64[:], float64[:])",
    "(n), (n) -> ()"
)
def core_funct_v2(travel, arrivals, res):
    dt = arrivals - travel
    dt = dt[~np.isnan(dt)] #remove nan value
    dt = dt - np.mean(dt)
    res[0] = np.mean(np.abs(dt)) # error


# Grid seach each grid nodes-- this is slow
#OUT1 = open(fout,'w')
#OUT1.close()


def run_loop_k(sav_k_batch, all_T_batch):
    for ik, k in enumerate(sav_k_batch):
        print('ik=',ik)
        sta_phase = []
        arrival = []
        for ista in all_T_batch[k]['sta']:
            sta_phase.append(ista.split('.')[1]+'_'+'S') #only S here
            arrival.append(st2detc[ista][k])
        arrivals = sort_arrivals(sta_phase, arrival, T0=k, sort_list=Travel['sta_phase'])
        # grid search for best location
        err = xr.apply_ufunc(core_funct_v2, ds_var.TT, arrivals, input_core_dims=[["sta"], []], dask = 'allowed', output_dtypes=float)
        err = np.array(err)
        idx = np.argmin(err) # error
        #OUT1 = open(fout,'a') #so that you know the progress
        #OUT1.write('%s %f %f %f %f %d\n'%(k,grid_loc_np[idx,0],grid_loc_np[idx,1],grid_loc_np[idx,2],err[idx],len(all_T[k]['sta'])))
        #OUT1.close()

#OUT1.close()

# Parallel processing
n_cores = 4

def split_array(arr, m):
    n = len(arr)
    quotient = n // m
    remainder = n % m
    result = []
    start = 0
    for i in range(m):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient
        #result.append(arr[start:end])
        result.append(arr[start:start+10])
        start = end
    return result

# split the sav_k by n_cores
sav_k_batches = split_array(sav_k, n_cores)
# split the all_T into small batches as well
all_T_batches = []
for sav_k_batch in sav_k_batches:
    tmp = {k:all_T[k] for k in sav_k_batch}
    all_T_batches.append(tmp)
    
assert sum([len(sav_k_batches[ii]) for ii in range(len(sav_k_batches))]) == len(sav_k), 'Check the size!'


from joblib import Parallel, delayed
results = Parallel(n_jobs=n_cores,verbose=10,backend='threading')(delayed(run_loop_k)(sav_k_batches[i], all_T_batches[i]) for i in range(4)  )


results = Parallel(n_jobs=n_cores,verbose=0)(delayed(grid_searching_loc)(grid_loc_batches[i], Travel_batches[i], arrivals) for i in range(len(grid_loc_batches))  )
print(results)

