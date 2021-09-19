#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:35:20 2021

@author: timlin
"""

#Use detection time to cut timeseries from Talapas dataset
import numpy as np
from obspy import UTCDateTime
import obspy
import glob
import os
import h5py

'''
#load family and arrival information from HypoINV file
EQinfo = np.load('sav_family_phases.npy',allow_pickle=True)
EQinfo = EQinfo.item()
'''

#LFE detection file
detcFile = 'total_mag_detect_0000_cull_NEW.txt' 
#detcFile = 'test.txt' 

#seconds before and after arrival
timeL = 15
timeR = 15
sampl = 100 #sampling rate
dir1 = '/projects/amt/shared/cascadia_PO' # .mseed files are in PO or CN directory
net1 = 'PO'
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


def data_process(filePath,sampl=sampl):
    '''
    load and process daily .mseed data
    filePath: absolute path of .mseed file.
    sampl: sampling rate
    other processing such as detrend, taper, filter are hard coded in the script, modify them accordingly
    '''
    if not os.path.exists(filePath):
        return ''
    D = obspy.read(filePath)
    if len(D)!=1:
        D.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
    t1 = D[0].stats.starttime
    t2 = D[0].stats.endtime
    D.detrend('linear')
    D.taper(0.02) #2% taper
    D.filter('highpass',freq=1.0)
    D.trim(starttime=t1-1,endtime=t2+1,nearest_sample=True, pad=True, fill_value=0)
    D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
    D.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
    return D


def data_cut(Data1,Data2='',t1=UTCDateTime("20100101"),t2=UTCDateTime("20100102")):
    '''
    cut data from one or multiple .mseed
    Data1, Data2: Obspy stream 
    t1, t2: start and end time of the timeseries
    '''
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
    assert len(DD[0].data)==3001, "cut data not exactly 3001 points"
    return DD[0].data


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
    #N1 = 30
    wind = len(data)//N1
    T = np.concatenate([np.arange(3001)/100,np.arange(3001)/100+30,np.arange(3001)/100+60])
    for i in range(N1):
        #print(np.std(data[int(i*wind):int((i+1)*wind)]))
        #if np.std(data[int(i*wind):int((i+1)*wind)])<1e-3:
        #curr_data = data[int(i*wind):int((i+1)*wind)]
        #curr_data = curr_data/np.max(np.abs(curr_data))
        #print(np.std(data[int(i*wind):int((i+1)*wind)]))
        #plt.plot(T,data,'k')
        #plt.plot(T[int(i*wind):int((i+1)*wind)],data[int(i*wind):int((i+1)*wind)],'r')
        #plt.title('std=%f'%(np.std(data[int(i*wind):int((i+1)*wind)])))
        #plt.show()
        if np.std(data[int(i*wind):int((i+1)*wind)])<min_std:
            return False
    #differential check
    #if np.abs(np.diff(data)).min()<1e-5:
    #    return False
    #auto correlation, seperate the data into n segments and xcorr with rest of data(without small data) to see if data are non-corh
    #N2 = 15
    wind = len(data)//N2
    for i in range(N2):
        data_small = data[int(i*wind):int((i+1)*wind)] #small segment
        data_bef = data[:int(i*wind)]
        data_aft = data[int((i+1)*wind):]
        data_whole = np.concatenate([data_bef,data_aft])
        curr_CC = cal_CCC(data_whole,data_small) 
        #print(curr_CC)
        if curr_CC>CC:
            return False
    return True



#read detection file and find all the time
sav_times = [] #save all the detection time
with open(detcFile,'r') as IN1:
    for line in IN1.readlines():
        line = line.strip()
        ID = line.split()[0] #family ID
        OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
        HH = (int(line.split()[2])-1)*3600  #HR from 1-24
        SS = float(line.split()[3])
        OT = OT + HH + SS #this is NOT origin time, but close
        #to get the true OT, you'll need to relate the time to lfe_svi.arc
        sav_times.append(OT)

sav_times = np.array(sav_times)

#sort all the detected time
sav_times.sort()

#N noise data for each stations
#N_noise = 30000
N_noise = 48300
min_sept = 180 #minimum separation in second

#get station list from LFE observations (i.e. if there's LFE traces at that station, we want also noise data)
DataDir = '/projects/amt/jiunting/Cascadia_LFE/Data'
LFEs = glob.glob(DataDir+'/*.h5')
sav_NetStaComp = []
for LFE in LFEs:
    NetStaComp = LFE.split('/')[-1].split('_')[2] #this is for example CN.PGC.BH
    sav_NetStaComp.append(NetStaComp)

#sav_NetStaComp uniq station name
sav_NetStaComp = list(set(sav_NetStaComp))


def get_randTime(t1,t2,catTime=sav_times,min_sept=min_sept,N_noise=N_noise):
    #make random time that are not too close to event origins
    #t1,t2: random time spans from t1 to t2
    #catTime: catalog detections
    #min_sept: the min separation seconds between any catalog event and the random time
    #N: number of random events
    T_range = t2-t1 #sav_times[-1]-sav_times[0] #get the range of random number
    num = 0 #number of noise data
    sav_randOT = [] #save all the random generated time
    while num < N_noise:
        randOT = t1+np.random.rand()*T_range #rand() from 0~1 scale to 0~T_range
        randOT = UTCDateTime(randOT.strftime('%Y%m%d%H%M%S.%f')[:-3])
        #check if accept or reject
        if np.min(np.abs(sav_times-randOT))>min_sept:
            #and the random event do not repeat
            if len(sav_randOT)!=0:
                #check if the new one has already exist (very close to the existing random time)
                if np.min(np.abs(sav_randOT-randOT))<=min_sept:
                    idx = np.where(np.abs(sav_randOT-randOT)==np.min(np.abs(sav_randOT-randOT)))[0]
                    print('new random time too close to existing random time:',randOT,sav_randOT[idx[0]])
                    continue
                num += 1 #accept this random time
                sav_randOT = np.hstack([sav_randOT,randOT])
            else:
                num += 1 #first generated random time, accept this random time
                sav_randOT = np.hstack([sav_randOT,randOT])
                #print('accepted! n=',num)
        else:
            idx = np.where(np.abs(sav_times-randOT)==np.min(np.abs(sav_times-randOT)))[0]
            print('random time too close to catalog:',randOT,sav_times[idx[0]])
            continue
    sav_randOT.sort()     
    sav_randOT = np.array(sav_randOT)
    return sav_randOT


#example of using the get_randTime
#sav_rndOT = get_randTime(UTCDateTime('20040101'),UTCDateTime('20050101'),sav_times,300,1000)

'''
#-----this only needs to be run ones---------
#create many random time and save to sav_randOT.npy that can be used later
from joblib import Parallel, delayed
def run_loop(t1=UTCDateTime("20030101")):
    sav_randOT = get_randTime(t1,t2=t1+(86400*364.95),catTime=sav_times,min_sept=min_sept,N_noise=50000) #first test with 20
    return sav_randOT

n_cores = 16
t1 = [UTCDateTime("20030101")+86400*365*i  for i in range(12)] #what we want is from 20030101 to ~20140101
results = Parallel(n_jobs=n_cores,verbose=10)(delayed(run_loop)(i) for i in t1  )
#merge the results
sav_results = []
for result in results:
    sav_results = np.hstack([sav_results,result])

sav_results.sort()
np.save('sav_randOT.npy',sav_results)
#---------------------------------------------
'''

#load noise time
sav_randOT = np.load('sav_randOT.npy',allow_pickle=True)



for NetStaComp in sav_NetStaComp:
    net = NetStaComp.split('.')[0]
    sta = NetStaComp.split('.')[1]
    comp = NetStaComp.split('.')[2]
    print('-Runing:','.'.join([net,sta,comp]))
    sav_DD = [] #save all the nose time series for net.sta.comp
    OUT1 = open('./Data_noise/%s.%s.%s_noise.log'%(net,sta,comp),'w')
    #get data time range
    dataDir = '/projects/amt/shared/cascadia_'+net
    allData = glob.glob(dataDir+'/'+'*.'+sta+'*'+comp+'Z.mseed')
    allData.sort()
    data_t1 = UTCDateTime(allData[0].split('/')[-1].split('.')[0])
    data_t2 = UTCDateTime(allData[-1].split('/')[-1].split('.')[0])

    #filter the sav_randOT within the time range provided by data
    filt_sav_randOT = sav_randOT.copy()
    idx = np.where((filt_sav_randOT>=data_t1) & 
                   (filt_sav_randOT<=data_t2))[0]
    filt_sav_randOT = filt_sav_randOT[idx]
    np.random.shuffle(filt_sav_randOT) #shuffle this random time

    #but not just use the time totally random, we want the time same day be processed together to speed up when loading daily data
    #below show a way to do so
    #filt_sav_randOT = filt_sav_randOT[:N_noise] #not just pick the target number because some random time may not work(missing data)
    #separate the random data into multiple groups (5 groups)
    gp_filt_sav_randOT = [[]]*5
    for i,tmp_randOT in enumerate(filt_sav_randOT):
        rem = i%5
        gp_filt_sav_randOT[rem].append(tmp_randOT)
    
    #sort the time for each group then concate them
    filt_sav_randOT = []
    for i in range(len(gp_filt_sav_randOT)):
        gp_filt_sav_randOT[i].sort()
        filt_sav_randOT = np.hstack([filt_sav_randOT,gp_filt_sav_randOT[i]])


    #now you have (5) groups of filt_sav_randOT data sorted in time, try them group-by-group until reach the target N_noise
    num = 0 #attemp to get N_noise
    #Similar to get_data.py, use these hashes so that you can re-use the daily mseed data instead of load them everytime
    prev_dataZ = {} #previous data with structure = {'file1':{'name':filePath,'data':obspySt}, 'file2':{'name':filePath,'data':obspySt}}
    prev_dataE = {} #in E comp
    prev_dataN = {} #in Z comp
    for rndOT in filt_sav_randOT:
        if num==N_noise:
            break
        rndOT = rndOT + np.random.rand()*30 - 15 #allow some additional small perturbation, but not overlapping with the catalog window
        rndOT = UTCDateTime(rndOT.strftime('%Y%m%d%H%M%S.%f')[:-3]) #accuracy to 0.001 s
        t1 = rndOT-timeL
        t2 = rndOT+timeR
        print('  start cutting data from:',t1,t2)
        #locate the file, read the file
        #loc is always empty, comps are always comp[ENZ]
        t1_fileZ = dataDir+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'Z.mseed'
        t1_fileE = dataDir+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'E.mseed'
        t1_fileN = dataDir+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'N.mseed'         
        t2_fileZ = dataDir+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'Z.mseed'
        t2_fileE = dataDir+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'E.mseed'
        t2_fileN = dataDir+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'N.mseed'

        #determine if you need to reload new data
        if sta in prev_dataZ:
            #check if the current and previous are same
            if t1_fileZ != prev_dataZ[sta]['file1']['name']:
                #print('New data: %s and %s are different, update the new data t1'%(t1_fileZ,prev_dataZ[sta]['file1']['name']))
                #read the daily mseed data
                prev_dataZ[sta]['file1'] = {'name':t1_fileZ,'data':data_process(filePath=t1_fileZ,sampl=sampl)}
                prev_dataE[sta]['file1'] = {'name':t1_fileE,'data':data_process(filePath=t1_fileE,sampl=sampl)}
                prev_dataN[sta]['file1'] = {'name':t1_fileN,'data':data_process(filePath=t1_fileN,sampl=sampl)}
            if t2_fileZ != prev_dataZ[sta]['file2']['name']:
                #print('New data: %s and %s are different, update the new data t2'%(t1_fileZ,prev_dataZ[sta]['file1']['name']))
                #read the daily mseed data
                prev_dataZ[sta]['file2'] = {'name':t2_fileZ,'data':data_process(filePath=t2_fileZ,sampl=sampl)}
                prev_dataE[sta]['file2'] = {'name':t2_fileE,'data':data_process(filePath=t2_fileE,sampl=sampl)}
                prev_dataN[sta]['file2'] = {'name':t2_fileN,'data':data_process(filePath=t2_fileN,sampl=sampl)}
        else:
            #initiate a new prev_dataZ[sta]
            #print('sta: %s not in prev_dataZ...initial a new sta'%(sta))
            prev_dataZ[sta] = {'file1':{'name':t1_fileZ,'data':data_process(filePath=t1_fileZ,sampl=sampl)},
                               'file2':{'name':t2_fileZ,'data':data_process(filePath=t2_fileZ,sampl=sampl)}}      
            prev_dataE[sta] = {'file1':{'name':t1_fileE,'data':data_process(filePath=t1_fileE,sampl=sampl)},
                               'file2':{'name':t2_fileE,'data':data_process(filePath=t2_fileE,sampl=sampl)}}      
            prev_dataN[sta] = {'file1':{'name':t1_fileN,'data':data_process(filePath=t1_fileN,sampl=sampl)},
                               'file2':{'name':t2_fileN,'data':data_process(filePath=t2_fileN,sampl=sampl)}} 


        #-------start cutting data-------
        if t1.strftime('%Y%m%d') == t2.strftime('%Y%m%d'):
            #only load t1 file
            #print('prev_dataZ=',prev_dataZ)
            if not (os.path.exists(prev_dataZ[sta]['file1']['name']) & os.path.exists(prev_dataE[sta]['file1']['name']) & os.path.exists(prev_dataN[sta]['file1']['name'])):
                continue #either Z,E or N file are missing
            else:
                #print(' --Begin cut from',t1_fileZ)
                D_Z = data_cut(prev_dataZ[sta]['file1']['data'],Data2='',t1=t1,t2=t2)
                D_E = data_cut(prev_dataE[sta]['file1']['data'],Data2='',t1=t1,t2=t2)
                D_N = data_cut(prev_dataN[sta]['file1']['data'],Data2='',t1=t1,t2=t2)
                #concatenate ZEN traces
                #assert len(D_Z[0].data)==len(D_E[0].data)==len(D_N[0].data)==(timeL+timeR)*sampl+1, "length are different! check the data processing"
                assert len(D_Z)==len(D_E)==len(D_N)==(timeL+timeR)*sampl+1, "length are different! check the data processing"
                #DD = np.concatenate([D_Z[0].data,D_E[0].data,D_N[0].data])
                DD = np.concatenate([D_Z,D_E,D_N])
                sav_DD.append(DD)
                #also save log
                OUT1.write('%s - %s  1\n'%(t1.strftime('%Y%m%d%H%M%S.%f')[:-3],t2.strftime('%Y%m%d%H%M%S.%f')[:-3]))
                num += 1
        else:
            #time window across different day, read both
            if not (os.path.exists(prev_dataZ[sta]['file1']['name']) & os.path.exists(prev_dataE[sta]['file1']['name']) & os.path.exists(prev_dataN[sta]['file1']['name'])) and not(os.path.exists(prev_dataZ[sta]['file2']['name']) & os.path.exists(prev_dataE[sta]['file2']['name']) & os.path.exists(prev_dataN[sta]['file2']['name'])) :
                continue #either Z,E or N file for either t1 or t2 are missing
            else:
                #print(' --Begin cut from both',t1_fileZ,t2_fileZ)
                D_Z = data_cut(prev_dataZ[sta]['file1']['data'],Data2=prev_dataZ[sta]['file2']['data'],t1=t1,t2=t2)
                D_E = data_cut(prev_dataE[sta]['file1']['data'],Data2=prev_dataE[sta]['file2']['data'],t1=t1,t2=t2)
                D_N = data_cut(prev_dataN[sta]['file1']['data'],Data2=prev_dataN[sta]['file2']['data'],t1=t1,t2=t2)
                #concatenate ZEN traces
                #assert len(D_Z[0].data)==len(D_E[0].data)==len(D_N[0].data)==(timeL+timeR)*sampl+1, "length are different! check the data processing"
                assert len(D_Z)==len(D_E)==len(D_N)==(timeL+timeR)*sampl+1, "length are different! check the data processing"
                #DD = np.concatenate([D_Z[0].data,D_E[0].data,D_N[0].data])
                DD = np.concatenate([D_Z,D_E,D_N])
                sav_DD.append(DD)
                #also save log
                OUT1.write('%s - %s  2\n'%(t1.strftime('%Y%m%d%H%M%S.%f')[:-3],t2.strftime('%Y%m%d%H%M%S.%f')[:-3]))
                num += 1

    #save the h5py data
    h5f = h5py.File('./Data_noise/%s.%s.%s_noise.h5'%(net,sta,comp),'w')
    h5f.create_dataset('waves',data=sav_DD)
    h5f.close()

    #close log file
    OUT1.close()



