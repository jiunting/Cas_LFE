#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:38:50 2021

@author: timlin

5/26 2022: 1. Write out template OT in h5py file so that I know which trace is using. 2. QC check before write out (not normalized yet)
"""

#Use detection time to cut timeseries from Talapas dataset
import numpy as np
from obspy import UTCDateTime
import obspy
import glob,os
import h5py
from obspy.signal.cross_correlation import correlate_template


#load family and arrival information from HypoINV file
EQinfo = np.load('sav_family_phases.npy',allow_pickle=True)
EQinfo = EQinfo.item()

#LFE detection file
detcFile = 'total_mag_detect_0000_cull_NEW.txt'

#cut seconds before and after arrival
timeL = 15
timeR = 15
sampl = 100 #sampling rate
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


def data_process(filePath,sampl=sampl):
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
    assert len(DD[0].data)==3001, "cut data not exactly 3001 points"
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





currentFam = '-1' #current family ID, when changing ID, output .h5 file.
sta_P1 = {} #this record Phase1 (P wave)
sta_P2 = {} # Phase2 (S wave?)

#Use these hashes so that you can re-use the daily mseed data instead of load them everytime
prev_dataZ = {} #previous data with structure = {'file1':{'name':filePath,'data':obspySt}, 'file2':{'name':filePath,'data':obspySt}}
prev_dataE = {} #in E comp
prev_dataN = {} #in Z comp
with open(detcFile,'r') as IN1:
    for line in IN1.readlines():
        line = line.strip()
        print('Line=',line)
        ID = line.split()[0] #family ID
        OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
        HH = (int(line.split()[2])-1)*3600  #HR from 1-24
        SS = float(line.split()[3])
        OT = OT + HH + SS #detected origin time. always remember!!! this is not the real OT. The shifted time in the sav_family_phases.npy have been corrected accordingly.
        Mag = float(line.split()[4])
        #if Mag<2.49:
        #    continue #so that only save large LFEs for testing
        #If changing to new ID, save existing results and reset everything for the next run.
        if ID != currentFam:
            print('-change ID from:',currentFam,ID)
            #print('sta_P1=',sta_P1)
            #print('sta_P1 key=',sta_P1.keys())
            for name in sta_P1.keys():
                #save the P wave to h5py data
                h5f = h5py.File('./Data_QC_rmean/ID_%s_%s_P.h5'%(currentFam,name),'w')
                h5f.create_dataset('waves/'+OT.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-2],data=sta_P1[name])
                h5f.close()
            for name in sta_P2.keys():
                #save the S wave to h5py data
                h5f = h5py.File('./Data_QC_rmean/ID_%s_%s_S.h5'%(currentFam,name),'w')
                h5f.create_dataset('waves'+OT.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-2],data=sta_P2[name])
                h5f.close()
            #reset sta_P1 and sta_P2
            currentFam = ID
            sta_P1 = {}
            sta_P2 = {}

        #for this family ID, check arrivals (for each station) from EQinfo
        for sta in EQinfo[ID]['sta'].keys():
            P1 = EQinfo[ID]['sta'][sta]['P1'] #Phase 1 (P)
            P2 = EQinfo[ID]['sta'][sta]['P2'] #Phase 2 (S)

            """
            Define where to find the data, only search data in CN and PO here.
            Note there are some data in C8 too. modify below to find the corrected path
            """
            #some data are in cascadia_CN
            if sta in CN_list:
                dataDir = dir2
                net = net2
                comp = CN_list[sta]
            else: #otherwist look for cascadia_PO, and if data not found, data_process below will just return empty string.
                dataDir = dir1
                net = net1
                comp = 'HH' #PO stations are always HH

            #print('station:%s'%(sta))
            """
            Start processing P wave data
            """
            if P1 != -1:
                arr = OT+P1
                t1 = arr - timeL # cut timeseries start time
                t2 = arr + timeR # cut timeseries end time. Note that t1 and t2 may across +1 day.
                #locate the file, read the file. If the file does not exist, the following data_process will return empty string.
                #loc is always empty, comps are always comp[ENZ]
                t1_fileZ = dataDir+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'Z.mseed'
                t1_fileE = dataDir+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'E.mseed'
                t1_fileN = dataDir+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'N.mseed'
                t2_fileZ = dataDir+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'Z.mseed'
                t2_fileE = dataDir+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'E.mseed'
                t2_fileN = dataDir+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'N.mseed'
                
                #Determine if you need to reload new data
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

                if t1.strftime('%Y%m%d') == t2.strftime('%Y%m%d'):
                    #only load t1 file
                    #print('prev_dataZ=',prev_dataZ)
                    if not (os.path.exists(prev_dataZ[sta]['file1']['name']) & os.path.exists(prev_dataE[sta]['file1']['name']) & os.path.exists(prev_dataN[sta]['file1']['name'])):
                        #continue #either Z,E or N file are missing
                        pass #should go to S phase
                    else:
                        #print(' --Begin cut from',t1_fileZ)
                        D_Z = data_cut(prev_dataZ[sta]['file1']['data'],Data2='',t1=t1,t2=t2)
                        D_E = data_cut(prev_dataE[sta]['file1']['data'],Data2='',t1=t1,t2=t2)
                        D_N = data_cut(prev_dataN[sta]['file1']['data'],Data2='',t1=t1,t2=t2)
                        #concatenate ZEN traces
                        #assert len(D_Z[0].data)==len(D_E[0].data)==len(D_N[0].data)==(timeL+timeR)*sampl+1, "length are different! check the data processing"
                        assert len(D_Z)==len(D_E)==len(D_N)==(timeL+timeR)*sampl+1, "length are different! check the data processing"
                        #DD = np.concatenate([D_Z[0].data,D_E[0].data,D_N[0].data])
                        D_Z = D_Z-np.mean(D_Z)
                        D_E = D_E-np.mean(D_E)
                        D_N = D_N-np.mean(D_N)
                        DD = np.concatenate([D_Z,D_E,D_N])
                        if QC(DD, Type='data'):
                            DD = DD/max(abs(DD))
                            sav_name = '.'.join([net,sta,comp])
                            if not sav_name in sta_P1:
                                sta_P1[sav_name] = [DD]
                            else:
                                sta_P1[sav_name].append(DD)
                        #np.save('./Data/Fam_%s_%s_%s_P1.npy'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),DD)
                        #DD.write('./Data/Fam_%s_%s_%s_P1.mseed'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),format='MSEED')
                        #DD.clear()
                else:
                    #time window across different day, read both
                    if not (os.path.exists(prev_dataZ[sta]['file1']['name']) & os.path.exists(prev_dataE[sta]['file1']['name']) & os.path.exists(prev_dataN[sta]['file1']['name'])) and not(os.path.exists(prev_dataZ[sta]['file2']['name']) & os.path.exists(prev_dataE[sta]['file2']['name']) & os.path.exists(prev_dataN[sta]['file2']['name'])) :
                        #continue #either Z,E or N file are missing
                        pass
                    else:
                        #print(' --Begin cut from both',t1_fileZ,t2_fileZ)
                        D_Z = data_cut(prev_dataZ[sta]['file1']['data'],Data2=prev_dataZ[sta]['file2']['data'],t1=t1,t2=t2)
                        D_E = data_cut(prev_dataE[sta]['file1']['data'],Data2=prev_dataE[sta]['file2']['data'],t1=t1,t2=t2)
                        D_N = data_cut(prev_dataN[sta]['file1']['data'],Data2=prev_dataN[sta]['file2']['data'],t1=t1,t2=t2)
                        #concatenate ZEN traces
                        #assert len(D_Z[0].data)==len(D_E[0].data)==len(D_N[0].data)==(timeL+timeR)*sampl+1, "length are different! check the data processing"
                        assert len(D_Z)==len(D_E)==len(D_N)==(timeL+timeR)*sampl+1, "length are different! check the data processing"
                        #DD = np.concatenate([D_Z[0].data,D_E[0].data,D_N[0].data])
                        D_Z = D_Z-np.mean(D_Z)
                        D_E = D_E-np.mean(D_E)
                        D_N = D_N-np.mean(D_N)
                        DD = np.concatenate([D_Z,D_E,D_N])
                        if QC(DD, Type='data'):
                            DD = DD/max(abs(DD))
                            sav_name = '.'.join([net,sta,comp])
                            if not sav_name in sta_P1:
                                sta_P1[sav_name] = [DD]
                            else:
                                sta_P1[sav_name].append(DD)
                        #np.save('./Data/Fam_%s_%s_%s_P1.npy'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),DD)
                        #DD.write('./Data/Fam_%s_%s_%s_P1.mseed'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),format='MSEED')
                        #DD.clear()

                #print('  --sta:%5s %s'%(sta,arr.isoformat()))
            #continue #do not go S wave
            
            """
            Start processing S wave data, this is basically the same as processing P wave above
            """
            if P2 != -1:
                arr = OT+P2
                t1 = arr - timeL
                t2 = arr + timeR
                #locate the file, read the file
                #loc is always empty, comps are only HH[ENZ]
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
                        #read the daily mseed data
                        prev_dataZ[sta]['file1'] = {'name':t1_fileZ,'data':data_process(filePath=t1_fileZ,sampl=sampl)}
                        prev_dataE[sta]['file1'] = {'name':t1_fileE,'data':data_process(filePath=t1_fileE,sampl=sampl)}
                        prev_dataN[sta]['file1'] = {'name':t1_fileN,'data':data_process(filePath=t1_fileN,sampl=sampl)}
                    if t2_fileZ != prev_dataZ[sta]['file2']['name']:
                        #read the daily mseed data
                        prev_dataZ[sta]['file2'] = {'name':t2_fileZ,'data':data_process(filePath=t2_fileZ,sampl=sampl)}
                        prev_dataE[sta]['file2'] = {'name':t2_fileE,'data':data_process(filePath=t2_fileE,sampl=sampl)}
                        prev_dataN[sta]['file2'] = {'name':t2_fileN,'data':data_process(filePath=t2_fileN,sampl=sampl)}
                else:
                    #initiate a new prev_dataZ[sta]
                    prev_dataZ[sta] = {'file1':{'name':t1_fileZ,'data':data_process(filePath=t1_fileZ,sampl=sampl)},
                                       'file2':{'name':t2_fileZ,'data':data_process(filePath=t2_fileZ,sampl=sampl)}}
                    prev_dataE[sta] = {'file1':{'name':t1_fileE,'data':data_process(filePath=t1_fileE,sampl=sampl)},
                                       'file2':{'name':t2_fileE,'data':data_process(filePath=t2_fileE,sampl=sampl)}}
                    prev_dataN[sta] = {'file1':{'name':t1_fileN,'data':data_process(filePath=t1_fileN,sampl=sampl)},
                                       'file2':{'name':t2_fileN,'data':data_process(filePath=t2_fileN,sampl=sampl)}}

                if t1.strftime('%Y%m%d') == t2.strftime('%Y%m%d'):
                    #only load t1 file
                    if not (os.path.exists(prev_dataZ[sta]['file1']['name']) & os.path.exists(prev_dataE[sta]['file1']['name']) & os.path.exists(prev_dataN[sta]['file1']['name'])):
                        continue #no other phases need to run
                    #print(' --Begin cut from',t1_fileZ)
                    D_Z = data_cut(prev_dataZ[sta]['file1']['data'],Data2='',t1=t1,t2=t2)
                    D_E = data_cut(prev_dataE[sta]['file1']['data'],Data2='',t1=t1,t2=t2)
                    D_N = data_cut(prev_dataN[sta]['file1']['data'],Data2='',t1=t1,t2=t2)
                    #concatenate ZEN traces
                    assert len(D_Z)==len(D_E)==len(D_N)==(timeL+timeR)*sampl+1, "length are different! check the data processing"
                    D_Z = D_Z-np.mean(D_Z)
                    D_E = D_E-np.mean(D_E)
                    D_N = D_N-np.mean(D_N)
                    DD = np.concatenate([D_Z,D_E,D_N])
                    if QC(DD, Type='data'):
                        DD = DD/max(abs(DD))
                        sav_name = '.'.join([net,sta,comp])
                        if not sav_name in sta_P2:
                            sta_P2[sav_name] = [DD]
                        else:
                            sta_P2[sav_name].append(DD)
                    #np.save('./Data/Fam_%s_%s_%s_P2.npy'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),DD)
                    #DD.write('./Data/Fam_%s_%s_%s_P2.mseed'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),format='MSEED')
                    #DD.clear()
                else:
                    #time window across different day, read both
                    if not (os.path.exists(prev_dataZ[sta]['file1']['name']) & os.path.exists(prev_dataE[sta]['file1']['name']) & os.path.exists(prev_dataN[sta]['file1']['name'])):
                        continue
                    if not (os.path.exists(prev_dataZ[sta]['file2']['name']) & os.path.exists(prev_dataE[sta]['file2']['name']) & os.path.exists(prev_dataN[sta]['file2']['name'])):
                        continue
                    #print(' --Begin cut from both',t1_fileZ,t2_fileZ)
                    D_Z = data_cut(prev_dataZ[sta]['file1']['data'],Data2=prev_dataZ[sta]['file2']['data'],t1=t1,t2=t2)
                    D_E = data_cut(prev_dataE[sta]['file1']['data'],Data2=prev_dataE[sta]['file2']['data'],t1=t1,t2=t2)
                    D_N = data_cut(prev_dataN[sta]['file1']['data'],Data2=prev_dataN[sta]['file2']['data'],t1=t1,t2=t2)
                    #concatenate ZEN traces
                    assert len(D_Z)==len(D_E)==len(D_N)==(timeL+timeR)*sampl+1, "length are different! check the data processing"
                    D_Z = D_Z-np.mean(D_Z)
                    D_E = D_E-np.mean(D_E)
                    D_N = D_N-np.mean(D_N)
                    DD = np.concatenate([D_Z,D_E,D_N])
                    if QC(DD, Type='data'):
                        DD = DD/max(abs(DD))
                        sav_name = '.'.join([net,sta,comp])
                        if not sav_name in sta_P2:
                            sta_P2[sav_name] = [DD]
                        else:
                            sta_P2[sav_name].append(DD)
                    #np.save('./Data/Fam_%s_%s_%s_P2.npy'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),DD)
                    #DD.write('./Data/Fam_%s_%s_%s_P2.mseed'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),format='MSEED')
                    #DD.clear()

                #print('  --sta:%5s %s'%(sta,arr.isoformat()))
    
#save the last family
for name in sta_P1.keys():
    #save the h5py data
    h5f = h5py.File('./Data_QC_rmean/ID_%s_%s_P.h5'%(currentFam,name),'w')
    h5f.create_dataset('waves'+OT.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-2],data=sta_P1[name])
    h5f.close()
for name in sta_P2.keys():
    #save the h5py data
    h5f = h5py.File('./Data_QC_rmean/ID_%s_%s_S.h5'%(currentFam,name),'w')
    h5f.create_dataset('waves'+OT.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-2],data=sta_P2[name])
    h5f.close()












