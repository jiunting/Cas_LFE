#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:55:16 2020

Merge H5 files

@author: amt
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import glob



def cal_CCC(data1,data2):
    #cross-correlation "Coef" cunction, return the max CC
    from obspy.signal.cross_correlation import correlate_template
    CCCF=correlate_template(data1,data2)
    return np.max(np.abs(CCCF))

def QC(data,Type='data',return_norm=True):
    '''
    quality control of data
    return true if data pass all the checking filters, otherwise false
    '''
    #nan value in data check
    if np.isnan(data).any():
        return False,None
    #if they are all zeros
    if np.max(np.abs(data))==0:
        return False,None
    #demean
    data = data-np.mean(data)
    #normalize the data to maximum 1
    if not return_norm:
        data_dmean = data[:] #copy the data without normalized
    data = data/np.max(np.abs(data))
    #set QC parameters for noise or data
    if Type == 'data':
        N1,N2,min_std,CC = 30,30,0.01,0.98
    else:
        N1,N2,min_std,CC = 30,30,0.05,0.98
    #std window check, std too small probably just zeros
    wind = len(data)//N1
    T = np.concatenate([np.arange(3001)/100,np.arange(3001)/100+30,np.arange(3001)/100+60])
    for i in range(N1):
        #print('data=',data[int(i*wind):int((i+1)*wind)])
        #print('std=',np.std(data[int(i*wind):int((i+1)*wind)]))
        if np.std(data[int(i*wind):int((i+1)*wind)])<min_std :
            return False,None
    #auto correlation, seperate the data into n segments and xcorr with rest of data(without small data) to see if data are non-corh
    wind = len(data)//N2
    for i in range(N2):
        data_small = data[int(i*wind):int((i+1)*wind)] #small segment
        data_bef = data[:int(i*wind)]
        data_aft = data[int((i+1)*wind):]
        data_whole = np.concatenate([data_bef,data_aft])
        curr_CC = cal_CCC(data_whole,data_small)
        if curr_CC>CC:
            return False,None
    if return_norm:
        return True,data
    else:
        return True,data_dmean



#get all data/noise name
#dataName = glob.glob('/projects/amt/jiunting/Cascadia_LFE/Data/ID_*_*.*.*_S.h5')
dataName = glob.glob('/projects/amt/jiunting/Cascadia_LFE/Data/ID_*_*.*.*_P.h5') 
#noiseName = glob.glob('/projects/amt/jiunting/Cascadia_LFE/Data_noise/*_noise.h5')

#dataName = glob.glob('/projects/amt/jiunting/Cascadia_LFE/Data_QC_rmean/ID_*_*.*.*_S.h5')
#dataName = glob.glob('/projects/amt/jiunting/Cascadia_LFE/Data_QC_rmean_CC_0.1/ID_*_*.*.*_S.h5')
#noiseName = glob.glob('/projects/amt/jiunting/Cascadia_LFE/Data_noise/*_noise.h5')

#small scale test
#dataName = glob.glob('/projects/amt/jiunting/Cascadia_LFE/Data/ID_026_PO.SILB.HH_S.h5')
#noiseName = glob.glob('/projects/amt/jiunting/Cascadia_LFE/Data_noise/PO.T*.h5')

#dataName = glob.glob('/projects/amt/jiunting/Cascadia_LFE/Data_QC_rmean_NotCC_0.2/ID_*_*.*.*_S.h5')

# plots?
plots=0

# Need to merge .h5 files
allData=np.empty((0,9003))
allNoise=np.empty((0,9003))

#deal with data
NData = 0
NDataDrop = 0
NFile = 0
for i_file,lfeFile in enumerate(dataName):
    #allData=np.empty((0,9003)) #instead of stacking new on the whole, save every chunks and stack later
    OUT1 = open('lfe_data.log','a')
    OUT1.write('Now in %d out of %d\n'%(i_file,len(dataName)))
    OUT1.close()
    tmp = h5py.File(lfeFile, 'r')
    data = tmp['waves']
    tmp_allData=np.empty((0,9003)) #data in each .h5 file after QC
    for i_d in data:
        c,tmp_data = QC(i_d,Type='data',return_norm=False) # 2021/9/2 just save the data to Data_QC_rmean(without normalized)
        if c:
            tmp_allData = np.vstack([tmp_allData,tmp_data])
    #====save the data from /projects/amt/jiunting/Cascadia_LFE/Data to /projects/amt/jiunting/Cascadia_LFE/Data_QC_rmean
    #=== batch processing(data have been QC) ===   
    #data = np.array(data)
    #data = data/np.max(np.abs(data),axis=1).reshape(-1,1) #normalize
    #===========================================
    #allData = np.vstack([allData,tmp_allData]) comment this on 2021/9/2
    #allData = np.vstack([allData,data])
    tmp.close()
    h5f = h5py.File('/projects/amt/jiunting/Cascadia_LFE/Data_QC_rmean/'+lfeFile.split('/')[-1], 'w') # cp data from Data to Data_QC_rmean
    h5f.create_dataset('waves', data=tmp_allData)
    h5f.close()
    continue #2021/9/2
    #save every 100 files as a chunk
    if i_file%100==0 and i_file!=0:
        np.save('tmp_%03d.npy'%(i_file),allData)
        #np.save('tmp_%03d.npy'%(NFile),allData)
        allData=np.empty((0,9003)) #reset instead of stacking new on the whole, save every chunks and stack later
    NFile += 1
    '''
    for i_d,d in enumerate(data):
        if QC(d,Type='data'):
            #allData = np.concatenate((allData,d))
            allData = np.vstack([allData,d])
            NData += 1
        else:
            NDataDrop += 1
        #print log file
        if i_d%500==0:
            OUT1 = open('lfe_data.log','a')
            OUT1.write('-- %d out of %d traces\n'%(i_d,len(data)))
            OUT1.close()
    np.save('./tmp_LFE/file_%03d.npy'%(NFile),allData)
    NFile += 1
    '''

#stop here, 2021/9/2
import sys
sys.exit()

if len(allData)!=0:
    np.save('tmp_%03d.npy'%(i_file),allData)

tmp_all = glob.glob('tmp_*.npy')
tmp_all.sort()
allData=np.empty((0,9003))
for i_tmp in tmp_all:
    D = np.load(i_tmp)
    allData = np.vstack([allData,D])


print(allData.shape)
#print('Number of data:',NData)
print('Number of data dropped:',NDataDrop)
#save LFE data
#h5f = h5py.File("Cascadia_lfe_QC_rmean_NotCC0.2.h5", 'w')  
h5f = h5py.File("Cascadia_lfe_QC_rmean_norm_P.h5", 'w')
#h5f = h5py.File("Cascadia_lfe_QC_rmean_norm_check.h5", 'w')  
h5f.create_dataset('waves', data=allData)
h5f.close()

import sys
sys.exit()



#deal with noise
NNoise = 0
NNoiseDrop = 0
NFile = 0
n_traces = 0
allNoise=np.empty((0,9003))
for i_file,noiseFile in enumerate(noiseName):
    #allNoise=np.empty((0,9003))
    OUT1 = open('noise_data.log','a')
    OUT1.write('Now in %d out of %d\n'%(i_file,len(noiseName)))
    OUT1.close()
    tmp = h5py.File(noiseFile, 'r')
    noise = tmp['waves']
    noise = np.array(noise)
    for i_n,n in enumerate(noise):
        if QC(n,Type='noise'):
            #allNoise = np.concatenate((allNoise,n))   
            allNoise = np.vstack([allNoise,n])
            NNoise += 1
            n_traces += 1
        else:
            NNoiseDrop += 1 
        if i_n%500==0:
            OUT1 = open('noise_data.log','a')
            OUT1.write('-- %d out of %d traces\n'%(i_n,len(noise)))
            OUT1.close()
        if n_traces==2500:
            np.save('./tmp_LFE/noise_%03d.npy'%(NFile),allNoise)
            NFile += 1
            n_traces = 0 
            allNoise = np.empty((0,9003))

if allNoise.shape[0]!=0:
    np.save('./tmp_LFE/noise_%03d.npy'%(NFile),allNoise)



#print(allNoise.shape)
print('Number of data:',NNoise)
print('Number of noise dropped:',NNoiseDrop)
# save noise files
#h5f = h5py.File("Cascadia_noise_data.h5", 'w')  
#h5f.create_dataset('waves', data=allNoise)
#h5f.close()  
