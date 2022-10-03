#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:29:25 2022 

@author:Tim Lin
@email:jiunting@uoregon.edu


Check the template calculation by catalog

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
import datetime
import glob
from typing import TypedDict, Dict
import os
from obspy.signal.cross_correlation import correlate_template

sampl = 100 #sampling rate
template_length = 15 # template length in sec
N_min = 3 # minimum number of stations have detection in the same time (same starttime window)
search_days = 29 # number of days to be searched
MAD_thresh = 8 # 8 times the median absolute deviation
CC_range = 5 # group the CC if they are close in time. [Unit: second] so this value *sampl is the actual pts.


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

# load original detection file (from template matching)
# load family and arrival information
EQinfo = np.load("./data/sav_family_phases.npy",allow_pickle=True) #Note! detection file has a shift for each family
EQinfo = EQinfo.item()
detcFile = "./data/total_mag_detect_0000_cull_NEW.txt" #LFE event detection file
sav_OT_template = []
family = []
with open(detcFile,'r') as IN1:
    for line in IN1.readlines():
        line = line.strip()
        ID = line.split()[0] #family ID
        OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
        HH = int(line.split()[2])*3600
        SS = float(line.split()[3])
        OT = OT + HH + SS #detected template time (not origin time!)
        sav_OT_template.append(OT)
        family.append(ID)

sav_OT_template = np.array(sav_OT_template)
family = np.array(family)
# Note that sav_OT_template + time in EQinfo = actual arrival
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
# use these stations
stas = ['TWGB', 'GOWB', 'TWKB', 'TSJB', 'SILB', 'SSIB', 'VGZ', 'SNB', 'PFB', 'YOUB', 'LZB', 'PGC']
idx = np.where(family=='006')[0]
tmpT = sav_OT_template[idx]
idx = np.where(tmpT>UTCDateTime('20050101'))
T0 = tmpT[idx][0] #use the first one as template to search on daily data
print('Template time series from:',T0,T0+template_length)

templates = {}
for sta in stas:
    if sta in CN_list:
        net = 'CN'
        comp = CN_list[sta]
        dataDir = dir2
    else:
        net = 'PO'
        comp = 'HH'
        dataDir = dir1
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
        daily_files, days = daily_select(dataDir, '.'.join([net,sta]), comp+"E",
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
    avail_k = []
    for k in templates.keys():
        idx = np.where(templates[k]['days']==i_common)[0][0]
        E = data_process(templates[k]['daily_files'][idx],sampl=sampl,
                        starttime=UTCDateTime(i_common),
                        endtime=UTCDateTime(i_common)+86400-1/sampl)
        if E is None:
            continue #skip this station
        CCF = correlate_template(E[0].data, templates[k]['template'])
        if sum(np.isnan(CCF)):
            continue #RuntimeWarning
        if np.std(CCF)<1e-5:
            continue #CCF too small, probably just zeros, skip this station
        avail_k.append(k)
        sum_CCF += CCF
        n_sta += 1
        templates[k]['tmp_data'] = E[0].data[:]
        E.clear()
        del CCF
    sum_CCF = sum_CCF/n_sta
    thresh = MAD_thresh * np.median(np.abs(sum_CCF - np.median(sum_CCF))) + np.median(sum_CCF)
    print('   ---   Threshold=%f'%(thresh))
    t = (np.arange(86400*sampl)/sampl)[:len(sum_CCF)]
    _days = int((UTCDateTime(i_common)-UTCDateTime(T0.strftime('%Y%m%d')))/86400) #actual day after the template date
    #plt.plot(t,sum_CCF+i,color=[0.2,0.2,0.2],lw=0.5)
    plt.plot(t,sum_CCF+_days,color=[0.2,0.2,0.2],lw=0.5)
    plt.text(t[-1],_days,'%d'%(n_sta))
    # find detections
    idx = np.where(sum_CCF>=thresh)[0]
    if len(idx)>0:
        if i==0:
            #skip detection for itself
            d_idx = np.abs(idx-(T0-UTCDateTime(T0.year,T0.month,T0.day))*sampl)
            rm_idx = np.where(d_idx<=10*sampl)[0]
            idx = np.delete(idx,rm_idx)
        print('idx:',idx,'CC:',sum_CCF[idx],'t:',t[idx])
        idx, new_CC = find_group(idx, sum_CCF[idx], t_wind=CC_range, sampl=sampl)
        print('after group>>',idx,new_CC,t[idx])
        for ii in idx:
            #plt.plot(t[ii],sum_CCF[ii]+i,'r.')
            plt.plot(t[ii],sum_CCF[ii]+_days,'r.')
            #stack data for each stations
            #for k in templates.keys():
            for k in avail_k:
                if 'stack' in templates[k]:
                    templates[k]['stack'] += templates[k]['tmp_data'][ii:ii+int(template_length*sampl+1)]/np.max(np.abs(templates[k]['tmp_data'][ii:ii+int(template_length*sampl+1)]))
                    templates[k]['Nstack'] += 1
                    templates[k]['time'].append(UTCDateTime(i_common)+ii/sampl)
                else:
                    # initial a couple of new keys if there's any stacking
                    templates[k]['stack'] = templates[k]['template']/np.max(np.abs(templates[k]['template'])) + templates[k]['tmp_data'][ii:ii+int(template_length*sampl+1)]/np.max(np.abs(templates[k]['tmp_data'][ii:ii+int(template_length*sampl+1)]))
                    templates[k]['Nstack'] = 2
                    templates[k]['time'] = [UTCDateTime(i_common)+ii/sampl]
    if i==0:
        #plot itself
        plt.plot(T0-UTCDateTime(T0.year,T0.month,T0.day), sum_CCF.max(), 'g.')
        print('plot itself at %.2f sec'%(T0-UTCDateTime(T0.year,T0.month,T0.day)))
    break #only test one day
    
for k in templates.keys():
    del templates[k]['tmp_data'] #remove daily data
    
if 'stack' in templates[k]: #only save those have matching
    np.save('./template_match/Temp_%s.npy'%(T0.isoformat().replace(':','')),templates)
else:
    plt.clf()
    plt.close()
    gc.collect()
    #continue
    
ax=plt.gca()
ax.tick_params(pad=1.5,length=0.5,size=2.5,labelsize=10)
plt.title('Template:%s (stack %d stations)'%(T0.isoformat(), n_sta))
plt.xlim([0,t.max()])
plt.ylim([-1,search_days+1])
plt.xlabel('Seconds',fontsize=14,labelpad=0)
plt.ylabel('Days since template',fontsize=14,labelpad=0)
plt.savefig('./template_match/CCF_%s.png'%(T0.isoformat().replace(':','')),dpi=300)
plt.clf()
plt.close()
gc.collect()

