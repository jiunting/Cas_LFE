# filter the existing .h5 data with CC values
# Compare the CC current data with the stacked data

import h5py
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


def cal_CCC(data1,data2):
    #cross-correlation "Coef" cunction, return the max CC
    from obspy.signal.cross_correlation import correlate_template
    CCCF=correlate_template(data1,data2)
    return np.max(np.abs(CCCF))

def QC(data,Type='data'):
    '''
    quality control of data
    return true if data pass all the checking filters, otherwise false
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
            return False
    #auto correlation, seperate the data into n segments and xcorr with rest of data(without small data) to see if data are non-corh
    wind = len(data)//N2
    for i in range(N2):
        data_small = data[int(i*wind):int((i+1)*wind)] #small segment
        data_bef = data[:int(i*wind)]
        data_aft = data[int((i+1)*wind):]
        data_whole = np.concatenate([data_bef,data_aft])
        curr_CC = cal_CCC(data_whole,data_small)
        if curr_CC>CC:
            return False
    return True


#------CC level-------
CC = 0.2
dir_data = '/projects/amt/jiunting/Cascadia_LFE/Data'

'''
# make dir if dir not exist
dir_out = '/projects/amt/jiunting/Cascadia_LFE/Data_CC_%3.1f/'%(CC)
if not(os.path.exists(dir_out)):
    os.makedirs(dir_out)
'''


'''
#output QC, rmean clean data first
all_data = glob.glob(dir_data+'/'+'ID_*_S.h5')
#ID_159_PO.TWKB.HH_S.h5 
for i_data in all_data:
    fileName = i_data.split('/')[-1]
    print('---------------------------------------------')
    print('fileName = ',fileName)
    data = h5py.File(i_data,'r')
    idx = []
    for i,i_trace in enumerate(data['waves']):
        if QC(i_trace):
            idx.append(i)
    idx = np.array(idx)
    print('original length=',len(data['waves']))
    print('new length=',len(idx))
    data_mean = data['waves'][idx,:].mean(axis=1).reshape(-1,1)
    data_demean = data['waves'][idx,:] - data_mean
    #save the de-mean result (without normalization)
    h5f = h5py.File('./Data_QC_rmean/'+fileName,'w')
    h5f.create_dataset('waves',data=data_demean)
    h5f.close()
    data.close()
    #(data['waves'][idx]-data_mean)/np.max(np.abs((data['waves'][idx]-data_mean)),axis=1)
    #stacked_data = data['waves'].mean(axis=0)
''' 


dir_data = '/projects/amt/jiunting/Cascadia_LFE/Data_noise'
#output QC, rmean clean noise first
all_data = glob.glob(dir_data+'/'+'*noise.h5')
#ID_159_PO.TWKB.HH_S.h5 
for i_data in all_data:
    fileName = i_data.split('/')[-1]
    print('---------------------------------------------')
    print('fileName = ',fileName)
    data = h5py.File(i_data,'r')
    idx = []
    for i,i_trace in enumerate(data['waves']):
        if QC(i_trace):
            idx.append(i)
    idx = np.array(idx)
    print('original length=',len(data['waves']))
    print('new length=',len(idx))
    data_mean = data['waves'][idx,:].mean(axis=1).reshape(-1,1)
    data_demean = data['waves'][idx,:] - data_mean
    #save the de-mean result (without normalization)
    h5f = h5py.File('./Noise_QC_rmean/'+fileName,'w')
    h5f.create_dataset('waves',data=data_demean)
    h5f.close()
    data.close()
    #(data['waves'][idx]-data_mean)/np.max(np.abs((data['waves'][idx]-data_mean)),axis=1)
    #stacked_data = data['waves'].mean(axis=0)





