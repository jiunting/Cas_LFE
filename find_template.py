#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:55:05 2022 

@author:Tim Lin
@email:jiunting@uoregon.edu

"""

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('pdf') #instead using interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import obspy
from obspy import UTCDateTime
import datetime
import glob
from typing import TypedDict, Dict
import os

sampl = 100 #sampling rate
template_length = 15 # template length in sec
search_days = 29 # number of days to be searched
"""
Below define the data location and the channel (e.g. HH, BH, EH) for each station
"""
#In PO, all stations are HH
dir1 = '/projects/amt/shared/cascadia_PO' # .mseed files are in PO or CN directory
net1 = 'PO'
#In CN, stations channels are different, below provide the channel list.
dir2 = '/projects/amt/shared/cascadia_CN'
net2 = 'CN'
CN_list = {'GOBB':'EH',
           'LZB':'BH',
           'MGB':'EH',
           'NLLB':'BH',
           'PFB':'HH',
           'PGC':'BH',
           'SNB':'BH',
           'VGZ':'BH',
           'YOUB':'HH',
           }

def data_process(filePath,sampl=sampl,starttime=None,endtime=None):
    """
    load and process daily .mseed data
    other processing such as detrend, taper, filter are hard coded in the script, modify them accordingly
 
    Inputs
    ======
    filePath: str
        absolute path of .mseed file.
 
    sampl: float
        sampling rate
 
    Outputs
    =======
    D: obspy stream
 
    """
    if not os.path.exists(filePath):
        return None
    try:
        D = obspy.read(filePath)
    except:
        return None #unable to read file either an empty or broken file
    if len(D)!=1:
        D.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
    if starttime is None:
        t1 = D[0].stats.starttime
        t2 = D[0].stats.endtime
    else:
        t1 = starttime
        t2 = endtime
    D.detrend('linear')
    D.taper(0.02) #2% taper
    D.filter('highpass',freq=1.0)
    D.trim(starttime=t1-1,endtime=t2+1,nearest_sample=True, pad=True, fill_value=0)
    D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
    D.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
    return D

def data_cut(Data1,Data2='',t1=UTCDateTime("20100101"),t2=UTCDateTime("20100102")):
    """
    cut data from one or multiple .mseed

    Inputs
    ======
    Data1, Data2: Obspy stream
    t1, t2: start and end time of the timeseries

    Outputs
    =======
    return: numpy array

    """
    if Data2 == '':
        #DD = Data1.slice(starttime=t1,endtime=t2) #slice sometime has issue when data has gap or no data at exact starttime
        DD = Data1.copy()
        DD.trim(starttime=t1-1, endtime=t2+1, nearest_sample=True, pad=True, fill_value=0)
        DD.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
        DD.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
    else:
        DD = Data1+Data2
        DD.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
        DD.trim(starttime=t1-1, endtime=t2+1, nearest_sample=True, pad=True, fill_value=0)
        DD.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
        DD.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
    assert len(DD[0].data)==1501, "cut data not exactly 1501 points"
    return DD[0].data


def daily_select(dataDir, net_sta, comp, ranges=None):
    daily_files = glob.glob(dataDir+"/*."+net_sta+".*."+comp+".mseed") #all daily files
    daily_files.sort()
    daily_files = np.array(daily_files)
    days = np.array([i.split('/')[-1].split('.')[0] for i in daily_files])
    if ranges is not None:
        idx = np.where((days>=ranges[0]) & (days<=ranges[1]))[0]
        daily_files = daily_files[idx]
        days = days[idx]
    return daily_files, days
    

'''
class XstationParams(TypedDict):
    time_range: float # Other stations with arrival within this time range [seconds] will be considered
    n_stations: int   # Consider a template is valid if other (self not included) n stations also have the same arrival (defined by the above time range)

class SearchParams(TypedDict):
    y_min: float       # CNN picker threshold value (select data with y>=y_min)
    group_CC: float    # Threshold of template-target CC to be considered as a group (from 0~1)
    time_max: float    # Maximum time span [days] for template matching. Given the short LFE repeat interval, LFEs should already occur multiple times.
    group_max: int     # Stop matching when detecting more than N events in a same template
    cal_max: int       # Maximum number of calculations for each template. Stop check after this number anyway.
    x_station: XstationParams # Cross station checker. See XstationParams for definition
    ncores: int        # CPUs
    fout_dir: str      # Directory path to save output results

#=========== Example default values==============
search_params: SearchParams = {
    'y_min':0.2,
    'group_CC':0.2,
    'time_max':60,
    'group_max':100,
    'cal_max':10000,
    'x_station':{'time_range':30, 'n_stations':1},
    'ncores':16,
    'fout_dir':"./template_result",
}
#============================================
'''

def get_daily_nums(T):
    # get number of events each day
    # T is the sorted timeseries of the occurrence
    T0 = datetime.datetime(T[0].year,T[0].month,T[0].day)
    T1 = T0 + datetime.timedelta(1)
    sav_num = [] # number of events in a day
    sav_T = []
    n = 1 #set each day number start from 1 to keep daily number continuous, so remember to correct it in the final
    for i in T:
        if T0<=i<T1:
            n += 1
        else:
            if n!=0:
                sav_T.append(T0+datetime.timedelta(0.5))
                sav_num.append(n)
            #update T0,T1 to next day
            T0 = T1
            T1 = T0 + datetime.timedelta(1)
            n = 1
    sav_T.append(T0+datetime.timedelta(0.5))
    sav_num.append(n)
    sav_num = np.array(sav_num)
    sav_num = sav_num-1 #daily number correct by 1
    return np.array(sav_T),np.array(sav_num)

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
        T = csv[csv['y']>=thresh].starttime.values
        #net_sta = file_path.split('/')[-1].split('_')[-1].replace('.csv','')
        #prepend = csv.iloc[0].id.split('_')[0]
        #T = pd.to_datetime(csv[csv['y']>=thresh].id,format=prepend+'_%Y-%m-%dT%H:%M:%S.%f')
        #T.reset_index(drop=True, inplace=True) #reset the keys e.g. [1,5,6,10...] to [0,1,2,3,...]
        return T

#=====get all the detection starttime=====
csvs = glob.glob("./Detections_S_new/*.csv")
csvs.sort()

sav_T = {}
for csv in csvs:
    print('Now in :',csv)
    net_sta = csv.split('/')[-1].split('_')[-1].replace('.csv','')
    T = dect_time(csv, thresh=0.1)
    if T is None:
        continue
    sav_T[net_sta] = T

#=====get the number of detections in the same starttime window=====
#=====future improvement: consider nearby windows i.e. +-15 s
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

#=====Min station filter and get daily detction number=====
N_min = 3
sav_k = [] # keys that pass the filter
for k in all_T.keys():
    if all_T[k]['num']>=N_min:
        sav_k.append(k)

sav_k.sort()
sav_k = np.array([UTCDateTime(i) for i in sav_k])
detc_time, detc_nums = get_daily_nums(sav_k)

# load original detection file (from template matching)
# load family and arrival information
EQinfo = np.load("./sav_family_phases.npy",allow_pickle=True) #Note! detection file has a shift for each family
EQinfo = EQinfo.item()
detcFile = "./total_mag_detect_0000_cull_NEW.txt" #LFE event detection file
sav_OT_template = []
with open(detcFile,'r') as IN1:
    for line in IN1.readlines():
        line = line.strip()
        ID = line.split()[0] #family ID
        OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
        HH = int(line.split()[2])*3600
        SS = float(line.split()[3])
        OT = OT + HH + SS + EQinfo[ID]['catShift'] #detected origin time
        sav_OT_template.append(OT.datetime)

cat_time,cat_daynums = get_daily_nums(sorted(sav_OT_template))
plt.fill_between(detc_time,detc_nums/max(np.abs(detc_nums)),0,color=[1,0.5,0.5])
plt.fill_between(cat_time,cat_daynums/max(np.abs(cat_daynums)),0,color='k',alpha=0.8)
plt.legend(['Model','Catalog'])
plt.show()
plt.close()

# =====select detections that are not in the original catalog and see if that's real=====
#cross-correlation "Coef" cunction for long v.s. short timeseries
from obspy.signal.cross_correlation import correlate_template

#manually select 20060301-20061101 as template
filt_idx = np.where((sav_k>=UTCDateTime('20060301')) & (sav_k<=UTCDateTime('20061101')))[0]
filt_sav_k = sav_k[filt_idx]

all_sta = sav_T.keys()
#run_flag = False
for T0 in filt_sav_k:
    '''
    if T0==UTCDateTime("20060302T023945"):
        run_flag = True
    else:
        run_flag = False
    if not run_flag:
        continue
    '''
    print('Template time series from:',T0,T0+template_length)
    # find available stations
    T0_str = T0.strftime('%Y-%m-%dT%H:%M:%S.%f')+'Z'
    print(' - stations have detections:',all_T[T0_str]['sta'])
    templates = {}
    for net_sta in all_T[T0_str]['sta']:
        net = net_sta.split('.')[0]
        sta = net_sta.split('.')[1]
        if net == 'PO':
            comp = 'HH'
            dataDir = dir1
        elif net == 'CN':
            comp = CN_list[sta]
            dataDir = dir2
        # find the file name
        t1_fileE = dataDir+'/'+T0.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'E.mseed'
        t2_fileE = dataDir+'/'+(T0+template_length).strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'E.mseed'
        if (not os.path.exists(t1_fileE)) or (not os.path.exists(t2_fileE)):
            continue
        if t1_fileE==t2_fileE:
            E = data_process(t1_fileE,sampl=sampl,
                            starttime=UTCDateTime(T0.strftime('%Y%m%d')),
                            endtime=UTCDateTime(T0.strftime('%Y%m%d'))+86400-1/sampl)
            tempE = data_cut(E,Data2='',t1=T0,t2=T0+template_length)
            E.clear()
            daily_files, days = daily_select(dataDir, net_sta, comp+"E",
                          ranges=[T0.strftime('%Y%m%d'), (T0+(search_days*86400)).strftime('%Y%m%d')])
            templates[net+'.'+sta] = {'template':tempE, 'daily_files':daily_files, 'days':days}
            # some test
            #CCF = correlate_template(E[0].data,tempE)
            #np.save('tmpCCF_%s.npy'%(sta),CCF)
            #assert UTCDateTime(T0.year,T0.month,T0.day)+np.argmax(CCF)/sampl==T0
        else:
            continue # for now, just skip the data across days
    #=====use the templates to search on continuous data=====
    # find the intersection time of all stations
    common_days = None
    for k in templates.keys():
        if common_days is None:
            common_days = set(templates[k]['days'])
        else:
            common_days = common_days.intersection(templates[k]['days'])
    common_days = list(common_days)
    common_days.sort()
    # search on those common days
    for i,i_common in enumerate(common_days):
        #if i>3:
        #    continue #only run 3 days for quick testing
        print('  -- searching daily:',i_common)
        #for this day, loop through each stations
        sum_CCF = 0
        n_sta = 0
        for k in templates.keys():
            idx = np.where(templates[k]['days']==i_common)[0][0]
            E = data_process(templates[k]['daily_files'][idx],sampl=sampl,
                            starttime=UTCDateTime(i_common),
                            endtime=UTCDateTime(i_common)+86400-1/sampl)
            if E is None:
                continue
            CCF = correlate_template(E[0].data, templates[k]['template'])
            if np.std(CCF)<1e-5:
                continue #CCF too small, probably just zeros
            sum_CCF += CCF
            n_sta += 1
            E.clear()
            del CCF
        sum_CCF = sum_CCF/len(templates.keys())
        t = (np.arange(86400*sampl)/sampl)[:len(sum_CCF)]
        plt.plot(t,sum_CCF+i,color=[0.2,0.2,0.2],lw=0.5)
        idx = np.where(sum_CCF>=0.5)[0]
        if len(idx)>0:
            print('idx:',idx,'CC:',sum_CCF[idx],'t:',t[idx])
            if i==0:
                #skip detection for itself
                d_idx = np.abs(idx-(T0-UTCDateTime(T0.year,T0.month,T0.day))*sampl)
                rm_idx = np.where(d_idx<=10*sampl)[0]
                idx = np.delete(idx,rm_idx)
            for ii in idx:
                plt.plot(t[ii],sum_CCF[ii],'r.')
        if i==0:
            #plot itself
            plt.plot(T0-UTCDateTime(T0.year,T0.month,T0.day), sum_CCF.max(), 'g.')
            print('plot itself at %.2f sec'%(T0-UTCDateTime(T0.year,T0.month,T0.day)))
    ax=plt.gca()
    ax.tick_params(pad=1.5,length=0.5,size=2.5,labelsize=10)
    plt.title('Template:%s (stack %d stations)'%(T0.isoformat(), n_sta))
    plt.xlim([0,t.max()])
    plt.ylim([-1,search_days+1])
    plt.xlabel('Seconds',fontsize=14,labelpad=0)
    plt.ylabel('Days since template',fontsize=14,labelpad=0)
    plt.savefig('./template_match/CCF_%s.png'%(T0.isoformat()),dpi=300)
    plt.close()
        
