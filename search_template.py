#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:39:28 2022 

@author:Tim Lin
@email:jiunting@uoregon.edu

"""

# for all detections, cross-correlate them to search for template

import numpy as np
from scipy import signal
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import obspy
from obspy import UTCDateTime
from obspy.signal.cross_correlation import correlate_template
import os,sys, time
import datetime
import glob
from numba import jit
from joblib import Parallel, delayed
import seaborn as sns
from typing import TypedDict, Dict



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



#--- process daily data
def data_process(filePath: str, sampl: int=100):
    '''
        load and process daily .mseed data
        filePath: absolute path of .mseed file.
        sampl: sampling rate
        other processing such as detrend, taper, filter are hard coded in the script, modify them accordingly
    '''
    if not os.path.exists(filePath):
        return None
    D = obspy.read(filePath)
    if len(D)!=1:
        D.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
    #t1 = D[0].stats.starttime
    #t2 = D[0].stats.endtime
    # t1, t2 are not necessarily a whole day. Get the t1,t2 from file name instead
    t1 = UTCDateTime(filePath.split('/')[-1].split('.')[0])
    t2 = t1 + 86400
    D.detrend('linear')
    D.taper(0.02) #2% taper
    D.filter('highpass',freq=1.0)
    D.trim(starttime=t1-1,endtime=t2+1,nearest_sample=True, pad=True, fill_value=0)
    D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
    D.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
    return D
    

def data_cut(Data1,Data2='',t1=UTCDateTime("20100101"),t2=UTCDateTime("20100102")):
    '''
        cut data from one or multiple .mseed, return numpy array
        Data1, Data2: Obspy stream
        t1, t2: start and end time of the timeseries
    '''
    sampl = 100 #always 100hz
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
    #assert len(DD[0].data)==1500, "cut data not exactly 3001 points"
    return DD[0].data


def CCC_QC(data1,data2):
    #cross-correlation "Coef" cunction for QC
    CCCF = correlate_template(data1,data2)
    return np.max(np.abs(CCCF))

def QC(data,Type='data'):
    '''
        quality control of data
        return true if data pass all the checking filters, otherwise false
        the same function used before but with a N2=90 instead of 30
    '''
    #nan value in data check
    if np.isnan(data).any():
        return False
    #if they are all zeros
    if np.max(np.abs(data))==0:
        return False
    #normalize the data to maximum 1
    data = data/np.max(np.abs(data))
    #set QC parameters for noise or data
    if Type == 'data':
        N1,N2,min_std,CC = 30,90,0.01,0.98
    else:
        N1,N2,min_std,CC = 30,90,0.05,0.98
    #std window check, std too small probably just zeros
    wind = len(data)//N1
    for i in range(N1):
        #print('data=',data[int(i*wind):int((i+1)*wind)])
        #print('std=',np.std(data[int(i*wind):int((i+1)*wind)]))
        if np.std(data[int(i*wind):int((i+1)*wind)])<min_std :
            return False
    #auto correlation, seperate the data into n segments and xcorr with rest of data(without small data) to see if data are non-corh
    wind = len(data)//N2
    for i in range(N2):
        data_small = data[int(i*wind):int((i+1)*wind)] #small segment
        data_bef = data[:int(i*wind)]
        data_aft = data[int((i+1)*wind):]
        data_whole = np.concatenate([data_bef,data_aft])
        curr_CC = CCC_QC(data_whole,data_small)
        if curr_CC>CC:
            return False
    return True


@jit(nopython=True)
def norm_data(data,pos=1):
    #for normalized cross-correlation
    # pos=1 or 2: input data is first or second
    if pos==1:
        return (data-np.mean(data))/(np.std(data)*len(data))
    else:
        return (data-np.mean(data))/(np.std(data))


def cal_CCC(data1,data2):
    #input data should be normalized first by the norm_data()
    #calculate max CC, and its lag idx
    tmpccf=signal.correlate(data1,data2,'full')
    lag=tmpccf.argmax()
    maxCCC=tmpccf[lag]
    #midd = len(a)-1  #length of the second a, at this idx, refdata align with target data
    #return the shift pts w.r.s to the data1 i.e. (data1,np.roll(data2,shft)) yields max CC
    return maxCCC,lag-(len(data2)-1)


def cal_CCC2(data1,data2):
    """
    A simplified version of cal_CCC which can takes non-normalized data1 and data2
    """
    data1 = (data1-np.mean(data1)) / (np.std(data1)*len(data1))
    data2 = (data2-np.mean(data2)) / (np.std(data2))
    tmpccf=signal.correlate(data1,data2,'full')
    lag=tmpccf.argmax()
    maxCCC=tmpccf[lag]
    return maxCCC,lag-(len(data2)-1)


def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Examples
    --------
    """
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def roll_zeropad3(a, shift, axis=None):
    """
    Same as the roll_zeropad. But roll the 3 component separately.
    Inputs:
    -------
        a : 1D array
            Where a = np.concatenate([Z,E,N]),
        shift: int
            Shifted points for Z, E and N.
    Output:
    -------
        res: 1D array
            With the same length of input a.
            This is equivalent to np.concatenate(roll_zeropad(Z,shift), roll_zeropad(E,shift), roll_zeropad(N,shift))
    """
    l = len(a)//3
    Z = a[:l]
    E = a[l:2*l]
    N = a[2*l:]
    a1 = roll_zeropad(Z,shift)
    a2 = roll_zeropad(E,shift)
    a3 = roll_zeropad(N,shift)
    return np.concatenate([a1,a2,a3])


def batch_CC(n_i,i,target_keys,file_hdf5,search_params):
    """
    Run CC calculations w.r.t. a template waveform i.e. hf['data/'+i]) yields the template data
    Template is an 3-component np array. target_keys is list/array loopable for target data i.e. hf['data/'+target_keys[0]]) yields the 1st target data
    
    INPUTs
    ------
    n_i : int
        The index of template in the (filtered)hdf5/csv detection file. For actual index, check the outer script that calls this function.
    i : string like
        The key of the template data from hdf5 file i.e. hf['data/'+i]) to get the data
    target_keys : string like
        The key of the target data from hdf5 file
    search_params : dict
        The searching parameters, only 'time_max'; 'group_max', 'group_CC' are taken in this function.
        search_params = {
            'time_max':30,        # maximum time span [days] for template matching. Given the short LFE repeat interval, LFEs should already occur multiple times
            'group_max':50,     # stop matching when detecting more than N events in a same template
            'group_max':10000,   # maximum number of calculation for each template.
            'group_CC':0.2,      # threshold of template-target CC to be considered a group
            'y_min':0.2,         # CNN pick threshold value
            'group_percent':0.5, # at least N[0~1] overlap can be considered a group e.g. temp1=1, detc1=[1,3,5,6,7,9]. If temp2=2, detc2=[2,3,5,11], then len(detc2 in detc1)/len(detc1)=2/6, so temp2 is not in the group. This is not implemented yet!
            'ncores':16,
        }
        
    RETURNs
    -------
    group : dict
        The key of the dict is the template index, which is the ith data from the (filtered)csv/hdf5 detection file.
        The content of each key is the target index e.g. groups[n_i] = {'group': [1213, 1216, 1220, 1221, 1222, 1223, 1225] }
    """
    ## ----------params input-------------
    group_CC = search_params['group_CC']
    time_max = search_params['time_max']
    group_max = search_params['group_max']
    cal_max = search_params['cal_max']
    #-------------------------------------
    hf = h5py.File(file_hdf5,'r')
    template = np.array(hf['data/'+i]).reshape(-1)
    group = {}
    #sav_CC = {n_i:[]} #save all the CC for checking
    data1_nor = norm_data(template,pos=1) #normalization for CC calculation
    T0 = UTCDateTime(i.split('_')[-1])
    Tmax = T0 + (time_max*86400) # day to sec
    for n_j,j in enumerate(target_keys):
        if n_j<=n_i:
            continue
        if UTCDateTime(j.split('_')[-1])>Tmax:
            break # j is chronologically ordered, no need to search next j for this template
        if n_j>=cal_max:
            break # stop checking anymore
        if group!={}:
            if len(group[n_i]['group'])>=group_max:
                break
        data2 = np.array(hf['data/'+j]).reshape(-1)
        data2_nor = norm_data(data2,pos=2)
        CC,shift = cal_CCC(data1_nor,data2_nor)
        #sav_CC[n_i].append(CC)
        if CC>=group_CC and np.abs(shift)<len(data2)//3:
            # if shift greater than len(data2)//3 you are matching two different components
            sh_target = roll_zeropad3(data2,shift) # after shifting the target, calculate the CC again
            CC2,shift2 = cal_CCC2(template,sh_target)
            #if not (max(np.abs(sh_target))!=0 and np.abs(shift2)<1):
            # also consider the CC after shift(apdding zero shift)
            if not (max(np.abs(sh_target))!=0 and np.abs(shift2)<1 and CC2>=group_CC ):
                continue # break this target
            #print("=============")
            #print(i,j)
            #print("CC=",CC)
            # add in the group
            # create a new group or merge into the existing group
            if n_i in group:
                group[n_i]['group'].append(n_j)
                group[n_i]['shift'].append(shift)
                group[n_i]['CC1'].append(CC)
                group[n_i]['CC2'].append(CC2)
                #tmp = group[n_i]['template']
                #_,shift = cal_CCC(tmp,data2_nor) #note that to save calculation, tmp is not normalized. i.e. only want shift here
                #tmp = roll_zeropad(tmp,-round((1/len(group[n_i]['group']))*shift)) + roll_zeropad(data2/np.max(np.abs(data2)),round((1-1/len(group[n_i]['group']))*shift))
                #group[n_i]['template'] = tmp/np.max(np.abs(tmp))
                #group[n_i]['template'].append(data2)
                #group[n_i]['CC'].append(CC)
            else:
                #tmp = roll_zeropad(template/np.max(np.abs(template)),-round(0.5*shift)) + roll_zeropad(data2/np.max(np.abs(data2)),round(0.5*shift))
                #tmp = tmp/np.max(np.abs(tmp))
                #group[n_i] = {'group':[n_i,n_j],'template':tmp}
                #group[n_i] = {'group':[n_i,n_j],'template':[template,data2],'CC':[1,CC]}
                group[n_i] = {'group':[n_j], 'shift':[shift], 'CC1':[CC], 'CC2':[CC2]}
    hf.close()
    return group #,sav_CC
    


#csv_file_path = "/Users/jtlin/Documents/Project/Cascadia_LFE/Detections_S_small/cut_daily_PO.GOWB.csv"
#csv = pd.read_csv(csv_file_path)

# time series from the hdf5 is not centered at the arrival
#hdf5_file_path = csv_file_path.replace(".csv",".hdf5")
#hdf5 = h5py.File(hdf5_file_path,'r')

#-----cut detection tcs from daily mseed data-----
# data_path = "/projects/amt/shared/cascadia_PO/"
#data_path = "/Users/jtlin/Documents/Project/Cascadia_LFE/cascadia_"



def make_template(csv_file_path: str, search_params: SearchParams, dect_T_all: Dict[str, pd.Series], load: bool=False, pre_QC: bool=True):
    '''
        Read the arrival from csv then
            load=False:  Cut time series from the daily data centered on the arrival time
            load=True:   Read the time series directly from the .h5py file
    '''
    ## -----Build-in params for timeseries cut (only when load=False), modify them if necessarily------
    sampl = 100
    window_L = 15
    window_R = 15
    data_path_base = "/projects/amt/shared/cascadia_"
    file_hdf5_base = "./template"
    #----------------------------------------------------------------------
    ## -----search parameters----------------------------------------------
    y_min = search_params['y_min']
    group_CC = search_params['group_CC']
    ncores = search_params['ncores']
    #----------------------------------------------------------------------
    csv = pd.read_csv(csv_file_path)
    if len(csv)==0:
        return #empty csv
    #make directory to save results
    if not(os.path.exists(search_params['fout_dir'])):
        os.makedirs(search_params['fout_dir'])
    ##===== load the data directly or cut time series from daily centered at the arrival======
    if load:
        file_hdf5 = csv_file_path.replace('.csv','.hdf5')
    else:
        #Use these hashes so that you can re-use the daily mseed data instead of load them everytime
        prev_dataZ = {}
        prev_dataE = {} #in E comp
        prev_dataN = {} #in Z comp
        net = csv['id'][0].split('.')[0]
        sta = csv['id'][0].split('.')[1]
        comp = "*"
        data_path = data_path_base + net
        file_hdf5 = file_hdf5_base+"/"+"%s.hdf5"%(sta)
        print(' net:%s sta:%s comp:%s'%(net,sta,comp))
        print(' search data from:%s'%(data_path_base))
        print(' create temp data:%s'%(file_hdf5))
        if not(os.path.exists(file_hdf5)):
            hf = h5py.File(file_hdf5,'w')
            hf.create_group('data') #create group of data
            hf.close()
        else:
            print("File: %s already exist! Exit and not overwritting everything"%(file_hdf5))
            sys.exit()
        for ii,i in enumerate(csv['id']):
            #if ii%(len(csv)//10000)==0:
            print('  %.1f of data processed... (%d/%d)'%((ii/len(csv))*100,ii,len(csv)))
            arr = UTCDateTime(i.split('_')[-1])
            t1 = arr-window_L
            t2 = arr+window_R
            # determine which daily file(s) to read
            t1_fileZ = data_path+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'Z.mseed'
            t1_fileE = data_path+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'E.mseed'
            t1_fileN = data_path+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'N.mseed'
            t2_fileZ = data_path+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'Z.mseed'
            t2_fileE = data_path+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'E.mseed'
            t2_fileN = data_path+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'N.mseed'
            # get exact file name or '' if no file found
            t1_fileZ = '' if not glob.glob(t1_fileZ) else glob.glob(t1_fileZ)[0]
            t1_fileE = '' if not glob.glob(t1_fileE) else glob.glob(t1_fileE)[0]
            t1_fileN = '' if not glob.glob(t1_fileN) else glob.glob(t1_fileN)[0]
            t2_fileZ = '' if not glob.glob(t2_fileZ) else glob.glob(t2_fileZ)[0]
            t2_fileE = '' if not glob.glob(t2_fileE) else glob.glob(t2_fileE)[0]
            t2_fileN = '' if not glob.glob(t2_fileN) else glob.glob(t2_fileN)[0]
            if '' in [t1_fileZ,t1_fileE,t1_fileN,t2_fileZ,t2_fileE,t2_fileN]:
                continue # file can't find
            #determine if you need to reload new data
            if t1_fileZ not in prev_dataZ:
                prev_dataZ[t1_fileZ] = data_process(filePath=t1_fileZ,sampl=sampl)
                prev_dataE[t1_fileE] = data_process(filePath=t1_fileE,sampl=sampl) # also for E,N data
                prev_dataN[t1_fileN] = data_process(filePath=t1_fileN,sampl=sampl)
                # release memory if more than 2 saved data
                keys = prev_dataZ.keys()
                if len(keys)>2:
                    #print('prev is full' ,'delete the oldest mseed')
                    tmp_tk = {k.split('/')[-1].split('.')[0]:k for k in keys}
                    tmp_t = list(tmp_tk.keys())
                    tmp_t.sort() # sort and delete the earlist one
                    #print('  ---removed ',tmp_tk[tmp_t[0]])
                    prev_dataZ.pop(tmp_tk[tmp_t[0]])
                    prev_dataE.pop(tmp_tk[tmp_t[0]].replace('Z.mseed','E.mseed'))
                    prev_dataN.pop(tmp_tk[tmp_t[0]].replace('Z.mseed','N.mseed'))
            if t2_fileZ not in prev_dataZ:
                prev_dataZ[t2_fileZ] = data_process(filePath=t2_fileZ,sampl=sampl)
                prev_dataE[t2_fileE] = data_process(filePath=t2_fileE,sampl=sampl) # also for E,N data
                prev_dataN[t2_fileN] = data_process(filePath=t2_fileN,sampl=sampl)
                # release memory if more than 2 saved data
                keys = prev_dataZ.keys()
                if len(keys)>2:
                    #print('prev is full' ,'delete the oldest mseed')
                    tmp_tk = {k.split('/')[-1].split('.')[0]:k for k in keys}
                    tmp_t = list(tmp_tk.keys())
                    tmp_t.sort() # sort and delete the earlist one
                    #print('  ---removed ',tmp_tk[tmp_t[0]])
                    prev_dataZ.pop(tmp_tk[tmp_t[0]])
                    prev_dataE.pop(tmp_tk[tmp_t[0]].replace('Z.mseed','E.mseed'))
                    prev_dataN.pop(tmp_tk[tmp_t[0]].replace('Z.mseed','N.mseed'))
            #------start cut data---------
            if t1_fileZ==t2_fileZ:
                data_Z = data_cut(Data1=prev_dataZ[t1_fileZ],Data2='',t1=t1,t2=t2)
                data_E = data_cut(Data1=prev_dataE[t1_fileE],Data2='',t1=t1,t2=t2)
                data_N = data_cut(Data1=prev_dataN[t1_fileN],Data2='',t1=t1,t2=t2)
            else:
                data_Z = data_cut(Data1=prev_dataZ[t1_fileZ],Data2=prev_dataZ[t2_fileZ],t1=t1,t2=t2)
                data_E = data_cut(Data1=prev_dataE[t1_fileE],Data2=prev_dataE[t2_fileE],t1=t1,t2=t2)
                data_N = data_cut(Data1=prev_dataN[t1_fileN],Data2=prev_dataN[t2_fileN],t1=t1,t2=t2)
            # save the data in hdf5
            hf = h5py.File(file_hdf5,'a')
            hf.create_dataset('data/'+i,data=[data_Z,data_E,data_N]) #this is the raw data (no feature scaling) centered at arrival
            hf.close()
    #===file_hdf5 is done, no matter its loaded directly or re-cut and re-centered from daily mseed===
    net_sta = csv_file_path.split('/')[-1].split('_')[-1].replace('.csv','')
    #hf = h5py.File(file_hdf5,'r')
    #y_min = 0.3
    #group_CC = 0.2
    #group = {}
    #group_rec = set()
    #time_1 = time.time()
    filt_csv = csv[csv['y']>=y_min]
    print(' number of traces=%d after y>=%f filter'%(len(filt_csv),y_min))
    assert len(filt_csv)==len(dect_T_all[net_sta]), "length of filt_csv and dect_T do not match!"
    if pre_QC:
        QC_idx = []
        hf = h5py.File(file_hdf5,'r')
        results = Parallel(n_jobs=ncores,verbose=0,backend='loky')(delayed(QC)(np.array(hf['data/'+tmp_id]).reshape(-1)) for idx,tmp_id in enumerate(filt_csv['id']) )
        results = np.array(results)
        QC_idx = np.where(results==True)[0]
        hf.close()
        filt_csv = filt_csv.iloc[QC_idx]
        print(' number of traces=%d after QC filter'%(len(filt_csv)))
    else:
        QC_idx = None #None is everything
    #------------------------another filter xstation check-------------------------
    # If QC above, use the available QC index
    # Note that "use_idx" is the index used by dect_T_all[net_sta] i.e. after y filter, not after QC
    results = dect_in_others(net_sta, search_params['x_station'], dect_T_all, use_idx=QC_idx)
    assert len(results)==len(filt_csv), "length of the idx do not match! filt_csv[results] will be wrong!"
#    xstation_idx = np.where(results==True)[0]
    print(' number of traces=%d after Xstation check'%(sum(results)))
#    print('results = ',results)
    filt_csv = filt_csv[results]
#    filt_csv = filt_csv.iloc[xstation_idx]
    #------------------------------------------------------------------------------
    print(' number of traces=%d, may take maximum %f min to calculate...'%(len(filt_csv),(len(filt_csv)**2/2)*0.01/60/ncores ))
    results = Parallel(n_jobs=ncores,verbose=10,backend='loky')(delayed(batch_CC)(n_i,i,filt_csv['id'],file_hdf5,search_params) for n_i,i in enumerate(filt_csv['id'])  )
    # collect the results from the above calucation
    group = {}
    group['template_keys'] = filt_csv['id']
    group['template_idxs'] = {}
    #sav_CC = []
    for res in results:
        group['template_idxs'].update(res) #e.g. res = {0: {'group': [1, 3, 4, 5, 6]}, 'shift':[519, 289, 316, 1084, 46], 'CC1':[.....], 'CC2':[.....] }, for template idx=0, idx 1,3,4,5,6 are correlated.
        #for k in cc.keys():
        #    sav_CC.append(cc[k])
    sta = csv_file_path.split('.')[1]
    np.save(search_params['fout_dir']+"/"+"CC_group_%s.npy"%(sta),group)
    print('  -> File saved to:%s'%(search_params['fout_dir']+"/"+"CC_group_%s.npy"%(sta)))
    #np.save('sav_CC_%s.npy'%(sta),sav_CC)
    return
#    for n_i,i in enumerate(filt_csv['id']):
#        print("current i=%s  (%d/%d) "%(i,n_i,len(filt_csv)))
#        print(" %d of data reduced"%(len(group_rec)))
#        print(" run time=",time.time()-time_1)
#        time_1 = time.time()
#        batch_CC(n_i,i,filt_csv['id'],file_hdf5,group,group_rec)
    '''
        for n_j,j in enumerate(filt_csv['id']):
            if n_j<=n_i:
                continue # already calculated before
            # check if n_i or n_j was already grouped, if so skip the calculation
            #if (n_i in group_rec) or (n_j in group_rec): #skip if template is already associated with another data
            if  (n_j in group_rec): #loop every template
                continue
            data1, data2 = np.array(hf['data/'+i]).reshape(-1), np.array(hf['data/'+j]).reshape(-1)
            data1,data2 = norm_data(data1,data2)
            CC,shift = cal_CCC(data1,data2)
            if CC>=group_CC:
                #print("=============")
                #print(i,j)
                #print("CC=",CC)
                # add in the group
                #group_rec.add(n_i)
                group_rec.add(n_j)
                # create a new group or merge into the existing group
                if n_i in group:
                    group[n_i]['group'].append(n_j)
                    tmp = group[n_i]['template']
                    data1, _ = norm_data(tmp,data2) #data2 has been normalized earlier
                    CC,shift = cal_CCC(tmp,data2)
                    tmp = roll_zeropad(tmp,-int(0.1*shift)) + roll_zeropad(np.array(hf['data/'+j]).reshape(-1),int(0.9*shift))
                    group[n_i]['template'] = tmp/np.max(np.abs(tmp))
                else:
                    tmp = roll_zeropad(np.array(hf['data/'+i]).reshape(-1),-int(0.5*shift)) + roll_zeropad(np.array(hf['data/'+j]).reshape(-1),int(0.5*shift))
                    tmp = tmp/max(np.abs(tmp))
                    group[n_i] = {'group':[n_i,n_j],'template':tmp}
    '''
    #hf.close()



def dect_time(file_path: str, search_params: SearchParams, is_catalog: bool=False, EQinfo: str=None) -> np.ndarray:
    """
    Get detection time for 1). ML detection in .csv file or 2). catalog file
    
    INPUTs
    ======
    file_path: str
        Path of the detection file
    is_catalog: bool
        Input is catalog or not
    EQinfo: str
        Path of the EQinfo file i.e. "sav_family_phases.npy"
    
    OUTPUTs
    =======
    sav_OT: np.array[datetime] or None if empty detection
        Detection time in array. For ML detection, time is the arrival at the station. For catalog, time is the origin time
    
    EXAMPLEs
    ========
    #T1 = dect_time('./Data/total_mag_detect_0000_cull_NEW.txt',search_params,True,'./Data/sav_family_phases.npy')  #for catalog
    #T2 = dect_time('./results/Detections_S_small/cut_daily_PO.GOWB.csv',search_params)                             # for ML detection output
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
        net_sta = file_path.split('/')[-1].split('_')[-1].replace('.csv','')
        prepend = csv.iloc[0].id.split('_')[0]
        T = pd.to_datetime(csv[csv['y']>=search_params['y_min']].id,format=prepend+'_%Y-%m-%dT%H:%M:%S.%f')
        T.reset_index(drop=True, inplace=True) #reset the keys e.g. [1,5,6,10...] to [0,1,2,3,...]
        return T




def dect_in_others(self_net_sta: str, x_station: XstationParams, dect_T_all: Dict[str, pd.Series], use_idx: np.ndarray = None) -> np.ndarray:
    """
    For a given time, check if other stations also have arrival at the same time
    
    INPUTs
    ======
    self_net_sta: str
        Self name/key of the dect_T_all to be checked.
    use_idx: np.ndarray
        Index to be used for time checking. None = everything.
    
    OUTPUTs
    =======
    res: np.array
        1). If use_idx is None/unspecified, return array of bool values with the same length of dect_T_all[self_net_sta]
                so one can use: dect_T_all[self_net_sta][res] directly.
        2). If given use_idx, return array of bool values with the same length of dect_T_all[self_net_sta][use_idx]
                This is because the time has been filtered before the function called,
                i.e. filt_t = dect_T_all[self_net_sta][use_idx], so one can use filt_t[res] directly.
    """
    T_self = dect_T_all[self_net_sta] # self
    res = [False] * len(T_self)
    res_n = [0] * len(T_self) # number of other stations have same arrival
    prev_idx = {} # {sta: prev_idx_to_skip}
    if use_idx is None:
        # use all the idx if use_idx is unspecified
        for n_t,t in enumerate(T_self):
            # Loop through every stations. Future optimization: t is sorted and is always increasing, so for other stations, skip the prev_idx if they're False already
            for ksta in dect_T_all.keys():
                if ksta == self_net_sta:
                    continue
                if ksta not in prev_idx:
                    prev_idx[ksta] = 0 # initial new prev_idx start from 0
                T_other = dect_T_all[ksta]
                #if any detection is within the range, sum would be >=1 (==True)
                #if sum(abs(t-T_other) <= datetime.timedelta(seconds=x_station['time_range'])):
                tmp_prev_idx = prev_idx[ksta]
                for is_within in abs(t-T_other[prev_idx[ksta]:]) <= datetime.timedelta(seconds=x_station['time_range']):
                    if is_within:
                        prev_idx[ksta] = tmp_prev_idx
                        break #early break
                    tmp_prev_idx += 1
                if is_within:
                    res_n[n_t] += 1
                    if res_n[n_t] >= x_station['n_stations']:
                        res[n_t] = True
                        continue # continue to next t
    else:
        #total_idx = np.arange(len(T_self))
        #remain_idx = np.setxor1d(total_idx,skip_idx)
        for n_t in use_idx:
            t = T_self[n_t]
            # for this time, loop through every stations
            for ksta in dect_T_all.keys():
                if ksta == self_net_sta:
                    continue
                if ksta not in prev_idx:
                    prev_idx[ksta] = 0 # initial new prev_idx start from 0
                T_other = dect_T_all[ksta]
                #If any detection is within the range, sum would be >=1 (==True)
                #if sum(abs(t-T_other) <= datetime.timedelta(seconds=x_station['time_range'])):
                tmp_prev_idx = prev_idx[ksta]
                for is_within in abs(t-T_other[prev_idx[ksta]:]) <= datetime.timedelta(seconds=x_station['time_range']):
                    if is_within:
                        prev_idx[ksta] = tmp_prev_idx
                        break # early break
                    tmp_prev_idx += 1
                if is_within:
                    res_n[n_t] += 1
                    if res_n[n_t] >= x_station['n_stations']:
                        res[n_t] = True
                        continue # continue to next t
        res = np.array(res)[use_idx]
    return np.array(res)

#res2 = dect_in_others('PO.GOWB',{'time_range':15, 'n_stations':1}, dect_T_all, np.array(range(500)))
#t1 = time.time()
#dect_in_others('PO.GOWB',{'time_range':15, 'n_stations':1}, dect_T_all )
#print(time.time()-t1)


#def run_loop(csv_file_path,search_params):
#    # loop each csv file
#    print('-----------------------------')
#    print('Now in :',csv_file_path)
#    make_template(csv_file_path,search_params,load=True,pre_QC=True) #before looping through all traces, QC again


#csvs_file_path = "/Users/jtlin/Documents/Project/Cascadia_LFE/Detections_S_small"
csvs_file_path = "/Users/jtlin/Documents/Project/Cas_LFE/results/Detections_S_small"
#csvs_file_path = glob.glob(csvs_file_path+"/"+"cut_daily_*GOWB*.csv")
csvs_file_path = glob.glob(csvs_file_path+"/"+"cut_daily_*.csv")
csvs_file_path.sort()

#search_params = {
#    'y_min':0.2,         # CNN pick threshold value
#    'group_CC':0.2,      # threshold of template-target CC to be considered as a group
#    #'group_percent':0.5, # at least N[0~1] overlap can be considered a group e.g. temp1=1, detc1=[1,3,5,6,7,9]. If temp2=2, detc2=[2,3,5,11], then len(detc2 in detc1)/len(detc1)=2/6, so temp2 is not in the group. This is not implemented yet!
#    'time_max':30,    # maximum time span [days] for template matching. Given the short LFE repeat interval, LFEs should already occur multiple times.
#    'group_max':100,     # stop matching when detecting more than N events in a same template
#    'cal_max':3000,     # maximum number of calculations for each template. Stop check after this number anyway.
#    'xstation':{'time_range':15, 'n_percent_station':0.5},   # Cross station check. For each arrival, continue to process if n% of stations  have detections within the time_range
#    'ncores':16,
#}

#----------start running here-------------
# multi-processing is deployed only in the calculation, just loop each station
cat_T = dect_time('./Data/total_mag_detect_0000_cull_NEW.txt',search_params,True,'./Data/sav_family_phases.npy') #for catalog
dect_T_all = {csv_file_path.split('_')[-1].replace('.csv',''): res for csv_file_path in csvs_file_path if (res := dect_time(csv_file_path,search_params)) is not None} # for ML detections

for csv_file_path in csvs_file_path:
    print('start running:', csv_file_path)
    make_template(csv_file_path,search_params,dect_T_all,load=True,pre_QC=True)
#    run_loop(csv_file_path,search_params)

# parallel processing for each stations
#results = Parallel(n_jobs=2,verbose=10)(delayed(run_loop)(csv_file_path) for csv_file_path in csvs_file_path  )

import sys
sys.exit()


#===== plotting timeseries this is just to make some demo for checking======
#groups = np.load('CC_group_GOWB.npy',allow_pickle=True)
#groups = np.load('./template_result/CC_group_TWGB_BACK.npy',allow_pickle=True)
groups = np.load('./template_result/CC_group_GOWB.npy',allow_pickle=True)
groups = groups.item()
hf = h5py.File('./results/Detections_S_small/cut_daily_PO.GOWB.hdf5','r')
#hf = h5py.File('./results/Detections_S_small/cut_daily_PO.TWGB.hdf5','r')
csv = pd.read_csv('./results/Detections_S_small/cut_daily_PO.GOWB.csv')
#csv = pd.read_csv('./results/Detections_S_small/cut_daily_PO.TWGB.csv')
T = np.arange(4500)*0.01
n_plot = 0
props = dict(boxstyle='round', facecolor='white', alpha=0.7)
for template_i in groups['template_idxs'].keys():
    k = groups['template_keys'].iloc[template_i]
    sav_t = [] # save repeating EQs time, first is the template
    template_t = UTCDateTime(k.split('_')[-1])
    sav_t.append(template_t)
    targets_i = groups['template_idxs'][template_i]['group']
    if len(targets_i)<5:
        continue
    template = np.array(hf['data/'+k]).reshape(-1)
    fig, axs = plt.subplots(2,2,figsize=(8,5.5), gridspec_kw={'height_ratios': [2, 1]})
    ax_merged = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    plt.subplot(2,2,1)
    plt.title('Raw detections',fontsize=14)
    plt.grid(False)
    idx = csv[csv['id']==k].idx_max_y
    plt.plot(T,template/max(np.abs(template)),'r')
#    plt.plot(T[idx],template[idx]/max(np.abs(template)),'bo')
#    plt.plot(T[idx+1500],template[idx+1500]/max(np.abs(template)),'bo')
#    plt.plot(T[idx+3000],template[idx+3000]/max(np.abs(template)),'bo')
    plt.text(1,-2,'Z',va='center',bbox=props);plt.text(16,-2,'E',va='center',bbox=props);plt.text(31,-2,'N',va='center',bbox=props)
    plt.subplot(2,2,2)
    plt.title('Shift',fontsize=14)
    plt.grid(False)
    plt.text(1,-2,'Z',va='center',bbox=props);plt.text(16,-2,'E',va='center',bbox=props);plt.text(31,-2,'N',va='center',bbox=props)
    plt.plot(T,template/max(np.abs(template)),'r')
    sum_target = np.zeros(len(template)) # stacked time series
    num_stacks = 0 # number of waveforms stacked
    for n,target_i in enumerate(targets_i):
        kk = groups['template_keys'].iloc[target_i]
        target_t = UTCDateTime(kk.split('_')[-1])
        target = np.array(hf['data/'+kk]).reshape(-1)
        target = target/max(np.abs(target)) #normalize
        # calculate CCC
        plt.subplot(2,2,1)
        #CCC,shift = cal_CCC2(template,target) #do not calculate again, use the info already saved.
        CCC = groups['template_idxs'][template_i]['CC1'][n]
        shift = groups['template_idxs'][template_i]['shift'][n]
        #print('CCC:',CCC,'CCC file:',groups['template_idxs'][template_i]['CC1'][n])
        #print('shift:',shift,'shift file:',groups['template_idxs'][template_i]['shift'][n])
        #------If the shift is greater than 1500 pts, that means you're matching two different components
        plt.plot(T,target-n-1,'k',linewidth=0.5)
        #plt.text(0,-n-1,'sh=%d'%(shift),fontsize=5,va='center')
        plt.subplot(2,2,2)
        sh_target = roll_zeropad3(target,shift)
        CCC2,shift2 = cal_CCC2(template,sh_target)
        # stack the data if it is not just zeros and the CC is basically zero lag after the first shifting
        if max(np.abs(sh_target))!=0 and np.abs(shift2)<1 and CCC2>=0.2:
            plt.plot(T,sh_target-n-1,'k',linewidth=0.5)
            sum_target += (sh_target)
            num_stacks += 1
            sav_t.append(target_t)
            plt.text(0,-n-1,'CC=%.2f'%(CCC2),fontsize=5,va='center')
    plt.subplot(2,2,1)
    plt.plot([15,15],[1,-n-3],color=[0.5,0.5,0.5])
    plt.plot([30,30],[1,-n-3],color=[0.5,0.5,0.5])
    plt.xlim([0,45])
    plt.ylim([-n-3,1])
    plt.xlabel('Time (s)',fontsize=12,labelpad=-1)
    ax=plt.gca()
    ax.tick_params(labelleft=False,pad=0,length=0,size=1)
    plt.subplot(2,2,2)
    #sum_target /= num_stacks
    sum_target /= max(np.abs(sum_target))
    plt.plot([15,15],[1,-n-3],color=[0.5,0.5,0.5])
    plt.plot([30,30],[1,-n-3],color=[0.5,0.5,0.5])
    plt.xlim([0,45])
    plt.ylim([-n-3,1])
    plt.xlabel('Time (s)',fontsize=12,labelpad=-1)
    ax=plt.gca()
    ax.tick_params(labelleft=False,pad=0,length=0,size=1)
    #------plot stacking tcs------
    ax_merged.plot(T,template/max(np.abs(template)),'r')
    ax_merged.text(0.5,0.1,'raw template')
    ax_merged.plot(T,sum_target-2.1,'b')
    CCC3,shift3 = cal_CCC2(template,sum_target)
    ax_merged.text(0.5,-2,'stack CC=%.2f'%(CCC3))
    ax_merged.plot([15,15],[-3.1,1],color=[0.5,0.5,0.5])
    ax_merged.plot([30,30],[-3.1,1],color=[0.5,0.5,0.5])
    ax_merged.text(7.5,0.5,'Z',va='center',bbox=props);ax_merged.text(22.5,0.5,'E',va='center',bbox=props);ax_merged.text(37.5,0.5,'N',va='center',bbox=props)
    ax_merged.set_xlabel('Time (s)',fontsize=14,labelpad=0)
    ax_merged.tick_params(labelleft=False,pad=0,length=0,size=1)
    ax_merged.set_xlim([0,45])
    ax_merged.set_ylim([-3.1,1])
    ax_merged.set_title('Shift and stack (N=%d)'%(num_stacks))
    plt.suptitle('Template: %s'%(k),fontsize=14)
    plt.subplots_adjust(left=0.08,top=0.88,right=0.97,bottom=0.1,wspace=0.05,hspace=0.25)
    plt.show()
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(template/max(np.abs(template)),'k')
    plt.plot(sum_target,'r',linewidth=0.7)
    plt.title('CC=%f'%(CCC3))
    plt.subplot(2,1,2)
    residual = template/max(np.abs(template))-sum_target
    plt.plot(residual)
    plt.title('std=%f'%(np.std(residual)))
    plt.show()
    n_plot += 1
    if n_plot>0:
        break



'''
# plotting results
groups = np.load('CC_group_GOWB.npy',allow_pickle=True)
groups = groups.item()
CCs = np.load('sav_CC_GOWB.npy',allow_pickle=True)
corr = np.ones([len(CCs),len(CCs)])
n = len(CCs)
for i in range(n):
    corr[i,i+1:n] = CCs[i]
    corr[i+1:n,i] = CCs[i]

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask,square=True,vmin=0,vmax=0.7, cbar_kws={"shrink": .5})
plt.show()
'''


# community detection
groups = np.load('CC_group_GOWB.npy',allow_pickle=True)
groups = groups.item()
import networkx as nx
G = nx.Graph()

for i,k in enumerate(groups.keys()):
    elist = [(k,g) for g in groups[k]['group']]
    G.add_edges_from(elist)


# trim the graph by number of connections
thres_conn = max([G.degree[n] for n in G.nodes])
rate = 0.3
max_iter = 10
for iter in range(max_iter):
    #thres_conn *= rate
    thres_conn = 20
    rm_n = []
    for n in G.nodes:
        if G.degree[n]<round(thres_conn):
            rm_n.append(n)
    print(' number of nodes=',len(G.nodes))
    print('   remove nodes=',len(rm_n))
    for n in rm_n:
        G.remove_node(n)


nx.draw(G,with_labels=False,node_size=6,edge_color=[0.8,0.8,0.8])
plt.show()


from networkx.algorithms import community
#k = 6 #groups
#GG = community.asyn_fluidc(G,k=k)
#GG = [next(GG) for i in range(k)]
#GG1 = next(GG)
#GG2 = next(GG)
#GG3 = next(GG)

#GG = community.girvan_newman(G) # very slow for large network
GG = community.greedy_modularity_communities(G)

# set different color
color_palette = sns.color_palette("tab10",len(GG))

colors = [] # color for plotting each community
for n in G:
    for ig,g in enumerate(GG):
        if n in g:
            colors.append(color_palette[ig])

my_pos = nx.spring_layout(G, seed = 0) # set the position so that plot will be the same
nx.draw(G,pos=my_pos,node_color=colors,with_labels=False,node_size=3,width=0.1,edge_color=[0.8,0.8,0.8])
plt.show()
