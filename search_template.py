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
import os,sys, time
import glob
from numba import jit
from joblib import Parallel, delayed



#--- process daily data
def data_process(filePath,sampl=100):
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

'''
def cal_CCC(data1,data2):
    #cross-correlation "Coef" cunction, return the max CC
    from obspy.signal.cross_correlation import correlate_template
    CCCF=correlate_template(data1,data2)
    return np.max(np.abs(CCCF))

def cal_CCC(data1,data2):
    #calculate max CC, and its lag idx
    tmpccf=signal.correlate(data1,data2,'full')
    auto1=signal.correlate(data1,data1,'full')
    auto2=signal.correlate(data2,data2,'full')
    tmpccf=tmpccf/np.sqrt(np.max(auto1)*np.max(auto2))
    return tmpccf
    maxCCC=np.max(tmpccf)
    lag=tmpccf.argmax()
    #midd = len(a)-1  #length of the second a, at this idx, refdata align with target data
    shft = lag-(len(data2)-1) # the shift pts w.r.s to the data1 i.e. (data1,np.roll(data2,shft)) yields max CC
    return maxCCC,shft
'''

@jit(nopython=True)
def norm_data(data1,data2):
    #for normalized cross-correlation
    return (data1-np.mean(data1))/(np.std(data1)*len(data1)), (data2-np.mean(data2))/(np.std(data2))


def cal_CCC(data1,data2):
    #input data should be normalized first by the norm_data()
    #calculate max CC, and its lag idx
    tmpccf=signal.correlate(data1,data2,'full')
    lag=tmpccf.argmax()
    maxCCC=tmpccf[lag]
    #midd = len(a)-1  #length of the second a, at this idx, refdata align with target data
    #return the shift pts w.r.s to the data1 i.e. (data1,np.roll(data2,shft)) yields max CC
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


'''
def batch_CC(template,target_keys):
    # do CC calculations w.r.t. a template i.g. data1
    # template is an np array
    # loopable target_keys for target data 
    for j in target_keys:
        data2 = np.array(hf['data/'+j]).reshape(-1)
        data1,data2 = norm_data(template,data2)
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
                group[n_i] = {'group':[n_j],'template':tmp}
'''     

#csv_file_path = "/Users/jtlin/Documents/Project/Cascadia_LFE/Detections_S_small/cut_daily_PO.GOWB.csv"
#csv = pd.read_csv(csv_file_path)

# time series from the hdf5 is not centered at the arrival
#hdf5_file_path = csv_file_path.replace(".csv",".hdf5")
#hdf5 = h5py.File(hdf5_file_path,'r')

#-----cut detection tcs from daily mseed data-----
# data_path = "/projects/amt/shared/cascadia_PO/"
#data_path = "/Users/jtlin/Documents/Project/Cascadia_LFE/cascadia_"


def make_template(csv_file_path,load=False):
    '''
        read the arrival from csv then
            load=False:  cut time series from the daily data centered on the arrival time
            load=True:   read the time series directly from the .h5py file
    '''
    ## -----build-in params for timeseries cut (only when load=False)------
    sampl = 100
    window_L = 15
    window_R = 15
    data_path_base = "/projects/amt/shared/cascadia_"
    file_hdf5_base = "./template"
    #----------------------------------------------------------------------
    csv = pd.read_csv(csv_file_path)
    if len(csv)==0:
        return #empty csv
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
    #===file_hdf5 is done, no matter its loaded directly or re-cut from daily mseed===
    hf = h5py.File(file_hdf5,'r')
    y_min = 0.3
    group_CC = 0.2
    group = {}
    group_rec = set()
    time_1 = time.time()
    filt_csv = csv[csv['y']>=y_min]
    for n_i,i in enumerate(filt_csv['id']):
        print("current i=%s  (%d/%d) "%(i,n_i,len(filt_csv)))
        print(" %d of data reduced"%(len(group_rec)))
        print(" run time=",time.time()-time_1)
        time_1 = time.time()
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
        
    hf.close()
    np.save('test_group.npy',group)




csvs_file_path = "/projects/amt/jiunting/Cascadia_LFE/Detections_S"
#csvs_file_path = glob.glob(csvs_file_path+"/"+"cut_daily_*GOWB*.csv")
csvs_file_path = glob.glob(csvs_file_path+"/"+"cut_daily_*.csv")
csvs_file_path.sort()

def run_loop(csv_file_path):
    print('-----------------------------')
    print('Now in :',csv_file_path)
    make_template(csv_file_path,load=True)

# parallel processing
results = Parallel(n_jobs=8,verbose=0)(delayed(run_loop)(csv_file_path) for csv_file_path in csvs_file_path  )



