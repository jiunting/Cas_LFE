#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:55:05 2022 

@author:Tim Lin
@email:jiunting@uoregon.edu

Find templates from the detection files e.g. Detections_S_new generated by daily2input.py.
The script do the following steps:
    1. Load all the detected time from all the stations and find if there is any common time in multiple stations (3+ stations here).
        Note that the 'time' is actually the starttime of the detection window, which is always starting from 00, 15, 30, 45 s. so that the common time is easier to determine.
    2. Take that starttime~starttime+15s as a template, cross-correlate on continuous timeseries and make stacked CCF.
    3. Apply threshold to select what time are similar to the template. (CC threshold or MAD)
    4. Group the nearby CC into just one for stacking, otherwise will duplicate and smear the data. (control by CC_range)
    5. Stack: ouput the result (templates) in .npy format and make plot (optional).
"""

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('pdf') #instead using interactive backend
import matplotlib.pyplot as plt
import gc
import pandas as pd
import obspy
from obspy import UTCDateTime
#cross-correlation "Coef" cunction for long v.s. short timeseries
from obspy.signal.cross_correlation import correlate_template
import datetime
import glob
from typing import TypedDict, Dict
import os
import warnings
warnings.simplefilter('error')

sampl = 100 #sampling rate
template_length = 15 # template length in sec
N_min = 3 # minimum number of stations have detection in the same time (same starttime window)
Cent_dist = 30 #[float or False] select stations within x-km from the centroid distance, after this filter check back if the remaining sta meet the N_min.
search_days = 29 # number of days to be searched
MAD_thresh = 8 # 8 times the median absolute deviation
CC_range = 5 # group the CC if they are close in time. [Unit: second] so this value *sampl is the actual pts.
use_comp = ['E','N','Z']

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

if Cent_dist:
    # load station info
    stainfo=np.load('stainfo.npy',allow_pickle=True)
    stainfo = stainfo.item()


def cal_CCC(data1,data2):
    #calculate max CC, and its lag idx
    tmpccf=signal.correlate(data1,data2,'full')
    auto1=signal.correlate(data1,data1,'full')
    auto2=signal.correlate(data2,data2,'full')
    tmpccf=tmpccf/np.sqrt(np.max(auto1)*np.max(auto2))
    maxCCC=np.max(tmpccf)
    lag=tmpccf.argmax()
    return maxCCC,lag

def find_group(idx, CC, t_wind=5, sampl=100):
    '''
    INPUTs
    ======
    idx: list or np.ndarray
        Index to be grouped.
    CC: list or np.ndarray
        CC associated with idx.
    t_wind: float
        Window of the idx to be considered as a `group`. In [second].
    sampl: int (float acceptable)
        Sampling rate of the data
    OUTPUTs
    ======
    new_idx: np.ndarray
        New index after grouping.
    new_CC: np.ndarray
        New CC associated with grouping
    '''
    if len(idx)==0:
        return idx,CC
    prev_idx = idx[0]
    max_CC = CC[0]
    max_CC_idx = idx[0]
    new_idx = []
    new_CC = []
    for i in range(1,len(idx)):
        #print('now in:',i)
        if idx[i]-prev_idx<(t_wind*sampl): # a group
            #print('same group comparing with:',max_CC,CC[i])
            if CC[i]>=max_CC:
                max_CC = CC[i]
                max_CC_idx = idx[i]
        else: #start another group
            #print('change group:',max_CC,CC[i])
            # accept the prev_idx
            new_idx.append(max_CC_idx)
            new_CC.append(max_CC)
            # reset max_CC, max_CC_idx
            max_CC = CC[i]
            max_CC_idx = idx[i]
        prev_idx = idx[i]
    
    new_idx.append(max_CC_idx)
    new_CC.append(max_CC)
    #plt.plot(idx,CC,'r.-')
    #plt.plot(new_idx,new_CC,'ko')
    #plt.show()
    return new_idx, new_CC



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


def filt_cent(stations, stainfo, Cent_dist=30, N_min=3):
    """
    Filter station by centroid and distance. Dont want stations that are too far away to be a template.
    After filter, check if the remaining stations pass the N_min criteria.
    
    INPUTs
    ======
    stations: List or np.ndarray
        A list of stations to be applied.
    stainfo: Dict
        A dictionary of station coordinate e.g. stainfo['PGC'] = [-123.4521, 48.649799999999999, 12.0]
    Cent_dist: float or False
        Filter stations with the centroid distance by initial stations
    N_min: int
        After filtering, check if the stations pass the criteria.
    
    OUTPUTs
    =======
    stations: List
        A list of stations after filtering
    """
    if Cent_dist==False:
        return stations # no filter, just return everything
    sav_stlo = []
    sav_stla = []
    for net_sta in stations:
        net = net_sta.split('.')[0]
        sta = net_sta.split('.')[1]
        stlo,stla,_ = stainfo[sta]
        sav_stlo.append(stlo)
        sav_stla.append(stla)
    cent_lon = np.mean(sav_stlo)
    cent_lat = np.mean(sav_stla)
    res = []
    for net_sta in stations:
        net = net_sta.split('.')[0]
        sta = net_sta.split('.')[1]
        stlo,stla,_ = stainfo[sta]
        dist, _, _ = obspy.geodetics.base.gps2dist_azimuth(lat1=cent_lat, lon1=cent_lon, lat2=stla, lon2=stlo) #dist in m
        if dist*1e-3<=Cent_dist:
            res.append(net_sta)
    if len(res)>=N_min:
        return res
    else:
        return []


#=====get all the detection starttime=====
csvs = glob.glob("./Detections_S_new/*.csv")
csvs.sort()

sav_T = {}
for csv in csvs:
    print('Now loading :',csv)
    net_sta = csv.split('/')[-1].split('_')[-1].replace('.csv','')
    T = dect_time(csv, thresh=0.1)
    if T is None:
        continue
    sav_T[net_sta] = T

#=====get the number of detections in the same starttime window=====
#=====future improvement: consider nearby windows i.e. +-15 s
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

#=====Min station filter and get daily detction number=====
#N_min = 3 #define in the begining
sav_k = [] # keys that pass the filter
for k in all_T.keys():
    if all_T[k]['num']>=N_min:
        sav_k.append(k)

sav_k.sort()
sav_k = np.array([UTCDateTime(i) for i in sav_k])
print('Total of candidate templates=%d after N_min=%d filter'%(len(sav_k), N_min))
print('  (For example:sav_k[0]=%s, startimg from this time T0 to T0+%f s) '%(sav_k[0].isoformat().replace(':',''),template_length))

#
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

cat_time, cat_daynums = get_daily_nums(sorted(sav_OT_template))

"""
#plotting ML detected v.s. catalog
plt.fill_between(detc_time,detc_nums/max(np.abs(detc_nums)),0,color=[1,0.5,0.5])
plt.fill_between(cat_time,cat_daynums/max(np.abs(cat_daynums)),0,color='k',alpha=0.8)
plt.legend(['Model','Catalog'])
plt.show()
plt.close()
"""

# =====select detections that are not in the original catalog and see if that's real=====

# Cant do full scale search.... manually select candidate template between 20060301-20061101, which is the gap in catalog
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
    stations = filt_cent(all_T[T0_str]['sta'], stainfo, Cent_dist, N_min) #adding a centroid distance filter 10/11
    if len(stations)==0:
        continue
    templates = {}
    #for net_sta in all_T[T0_str]['sta']:
    for net_sta in stations:
        net = net_sta.split('.')[0]
        sta = net_sta.split('.')[1]
        if net == 'PO':
            comp = 'HH'
            dataDir = dir1
        elif net == 'CN':
            comp = CN_list[sta]
            dataDir = dir2
        # =====find the file name=====
        comp_complete_flag = True #all the components are completed (no missing component). True to continue process, otherwise, stop this station
        for i_comp in use_comp:
            t1_file = dataDir+'/'+T0.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+i_comp+'.mseed'
            t2_file = dataDir+'/'+(T0+template_length).strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+i_comp+'.mseed'
            if (not os.path.exists(t1_file)) or (not os.path.exists(t2_file)):
                comp_complete_flag = False
                break
            if t1_file==t2_file:
                D = data_process(t1_file,sampl=sampl,
                            starttime=UTCDateTime(T0.strftime('%Y%m%d')),
                            endtime=UTCDateTime(T0.strftime('%Y%m%d'))+86400-1/sampl)
                tempD = data_cut(D,Data2='',t1=T0,t2=T0+template_length)
                D.clear()
                if np.sum(np.isnan(tempD)):
                    comp_complete_flag = False
                    break # template has issue
                daily_files, days = daily_select(dataDir, net_sta, comp+i_comp,
                          ranges=[T0.strftime('%Y%m%d'), (T0+(search_days*86400)).strftime('%Y%m%d')])
                if net+'.'+sta in templates:
                    #check days are consistent with the previous comonent
                    if len(set(templates[net+'.'+sta]['days']).symmetric_difference(set(days)))!=0:
                        comp_complete_flag = False
                        break
                if net+'.'+sta in templates:
                    templates[net+'.'+sta]['template'].update({i_comp:tempD})
                else:
                    templates[net+'.'+sta] = {'template':{i_comp:tempD}, 'daily_files':daily_files, 'days':days, 'comp':comp}
                # some test
                #CCF = correlate_template(E[0].data,tempE)
                #np.save('tmpCCF_%s.npy'%(sta),CCF)
                #assert UTCDateTime(T0.year,T0.month,T0.day)+np.argmax(CCF)/sampl==T0
            else:
                comp_complete_flag = False
                break # for now, just skip the data across days
        if (not comp_complete_flag) and (net+'.'+sta in templates):
            del templates[net+'.'+sta]
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
    # =====search on those common days=====
    sav_flag = False
    for i,i_common in enumerate(common_days):
        #if i>3:
        #    continue #only run 3 days for quick testing
        print('  -- searching daily:',i_common)
        #for this day, loop through each stations
        sum_CCF = 0
        n_sta = 0 #n_sta is actually n_components
        avail_k = []
        for k in templates.keys():
            idx = np.where(templates[k]['days']==i_common)[0][0]
            curr_comp = templates[k]['daily_files'][idx].split('.')[-2]
            sub_sum_CCF = 0
            sub_n_sta = 0 #number of components sum in this station
            comp_complete_flag = True
            for i_comp in use_comp:
                D = data_process(templates[k]['daily_files'][idx].replace(curr_comp, templates[k]['comp']+i_comp),sampl=sampl,
                            starttime=UTCDateTime(i_common),
                            endtime=UTCDateTime(i_common)+86400-1/sampl)
                if D is None:
                    comp_complete_flag = False
                    break #skip this station/component
                try:
                    CCF = correlate_template(D[0].data, templates[k]['template'][i_comp])
                except:
                    CCF = np.nan
                    print('RuntimeWarning capture!')
                    comp_complete_flag = False
                    break
                if np.std(CCF)<1e-5:
                    comp_complete_flag = False
                    break #CCF too small, probably just zeros, skip this station
                # save daily data for later
                if 'tmp_data' in templates[k]:
                    templates[k]['tmp_data'].update({i_comp: D[0].data[:]})
                else:
                    templates[k]['tmp_data'] = {i_comp: D[0].data[:]}
                # stack CCF at the same station, sub_sum_CCF will be added into final CCF if all the components are good.
                sub_sum_CCF += CCF #sum CCF for the same station (multiple components)
                sub_n_sta += 1
                D.clear()
                del CCF
            if comp_complete_flag: #all components are good.
                avail_k.append(k)
                sum_CCF += sub_sum_CCF
                n_sta += sub_n_sta
            else:
                # at least one component has issue
                if 'tmp_data' in templates[k]:
                    del templates[k]['tmp_data']
                D.clear()
                del CCF
        sum_CCF = sum_CCF/n_sta
        thresh = MAD_thresh * np.median(np.abs(sum_CCF - np.median(sum_CCF))) + np.median(sum_CCF)
        print('   ---   Threshold=%f'%(thresh))
        t = (np.arange(86400*sampl)/sampl)[:len(sum_CCF)]
        _days = int((UTCDateTime(i_common)-UTCDateTime(T0.strftime('%Y%m%d')))/86400) #actual day after the template date
        #plt.plot(t,sum_CCF+i,color=[0.2,0.2,0.2],lw=0.5)
        plt.plot(t,sum_CCF+_days,color=[0.2,0.2,0.2],lw=0.5)
        plt.text(t[-1],_days,'%d'%(n_sta))
        #=====find detections=====
        idx = np.where(sum_CCF>=thresh)[0]
        if len(idx)>0:
            if i==0:
                #skip detection for itself
                d_idx = np.abs(idx-(T0-UTCDateTime(T0.year,T0.month,T0.day))*sampl)
                rm_idx = np.where(d_idx<=10*sampl)[0]
                idx = np.delete(idx,rm_idx)
            #e.g. idx: [4613840 4613841 4613842 4613843 4613844 4613845] CC: [ 0.21118094  0.21532219  0.21930467  0.21915041  0.21752349  0.21353063] decide what idx to use
            print('idx:',idx,'CC:',sum_CCF[idx],'t:',t[idx])
            idx, new_CC = find_group(idx, sum_CCF[idx], t_wind=CC_range, sampl=sampl)
            print('after group>>',idx,new_CC,t[idx])
            for ii in idx:
                sav_flag = True
                #plt.plot(t[ii],sum_CCF[ii]+i,'r.')
                plt.plot(t[ii],sum_CCF[ii]+_days,'r.')
                #=====stack data based on the detected time (i.e. idx) for each stations=====
                #for k in templates.keys(): not all the k have data
                for k in avail_k:
                    if 'stack' in templates[k]:
                        for i_comp in use_comp:
                            templates[k]['stack'][i_comp] += templates[k]['tmp_data'][i_comp][ii:ii+int(template_length*sampl+1)]/np.max(np.abs(templates[k]['tmp_data'][i_comp][ii:ii+int(template_length*sampl+1)]))
                        templates[k]['Nstack'] += 1
                        templates[k]['time'].append(UTCDateTime(i_common)+ii/sampl)
                    else:
                        # initial a couple of new keys if there's any stacking
                        templates[k]['stack'] = {}
                        for i_comp in use_comp:
                            templates[k]['stack'][i_comp] = templates[k]['template'][i_comp]/np.max(np.abs(templates[k]['template'][i_comp])) + templates[k]['tmp_data'][i_comp][ii:ii+int(template_length*sampl+1)]/np.max(np.abs(templates[k]['tmp_data'][i_comp][ii:ii+int(template_length*sampl+1)]))
                        templates[k]['Nstack'] = 2
                        templates[k]['time'] = [UTCDateTime(i_common)+ii/sampl]
        if i==0:
            #plot itself
            plt.plot(T0-UTCDateTime(T0.year,T0.month,T0.day), sum_CCF.max(), 'g.')
            print('plot itself at %.2f sec'%(T0-UTCDateTime(T0.year,T0.month,T0.day)))
    for k in templates.keys():
        if 'tmp_data' in templates[k]:
            print('delete tmp_data for',k)
            del templates[k]['tmp_data'] #remove daily data
    if sav_flag: #only save those have matching
        np.save('./template_match/Temp_%s.npy'%(T0.isoformat().replace(':','')),templates)
    else:
        plt.clf()
        plt.close()
        gc.collect()
        continue
    ax=plt.gca()
    ax.tick_params(pad=1.5,length=0.5,size=2.5,labelsize=10)
    #plt.title('Template:%s (stack %d stations)'%(T0.isoformat(), n_sta))
    plt.title('Template:%s'%(T0.isoformat()))
    plt.xlim([0,t.max()])
    plt.ylim([-1,search_days+1])
    plt.xlabel('Seconds',fontsize=14,labelpad=0)
    plt.ylabel('Days since template',fontsize=14,labelpad=0)
    plt.savefig('./template_match/CCF_%s.png'%(T0.isoformat().replace(':','')),dpi=300)
    plt.clf()
    plt.close()
    gc.collect()
