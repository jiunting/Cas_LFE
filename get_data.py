#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:38:50 2021

@author: timlin
"""

#Use detection time to cut timeseries from Talapas dataset
import numpy as np
from obspy import UTCDateTime
import obspy
import glob,os
import h5py

#load family and arrival information from HypoINV file
EQinfo = np.load('sav_family_phases.npy',allow_pickle=True)
EQinfo = EQinfo.item()

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







currentFam = '-1' #current family ID
sta_P1 = {} #this record Phase1 (P wave)
sta_P2 = {} # Phase2 (S wave?)

num = 0

#Use these hashes so that you can re-use the daily mseed data instead of load them everytime
prev_dataZ = {} #previous data with structure = {'file1':{'name':filePath,'data':obspySt}, 'file2':{'name':filePath,'data':obspySt}}
prev_dataE = {} #in E comp
prev_dataN = {} #in Z comp
with open(detcFile,'r') as IN1:
    for line in IN1.readlines():
        num += 1
        line = line.strip()
        print('Line=',line)
        ID = line.split()[0] #family ID
        OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
        HH = (int(line.split()[2])-1)*3600  #HR from 1-24
        SS = float(line.split()[3])
        OT = OT + HH + SS #detected origin time. always remember!!! this is not the real OT. The shifted time in the sav_family_phases.npy have been corrected accordingly.
        Mag = float(line.split()[4])
        #if Mag<2.49:
        #    continue

        #if new ID, reset everything
        if ID != currentFam:
            print('-change ID from:',currentFam,ID)
            #print('sta_P1=',sta_P1)
            #print('sta_P1 key=',sta_P1.keys())
            for name in sta_P1.keys():
                #save the h5py data
                h5f = h5py.File('./Data/ID_%s_%s_P.h5'%(currentFam,name),'w')
                h5f.create_dataset('waves',data=sta_P1[name])
                h5f.close()
            for name in sta_P2.keys():
                #save the h5py data
                h5f = h5py.File('./Data/ID_%s_%s_S.h5'%(currentFam,name),'w')
                h5f.create_dataset('waves',data=sta_P2[name])
                h5f.close()
            #reset sta_P1 and sta_P2
            currentFam = ID
            sta_P1 = {}
            sta_P2 = {}
            
        #if num>950:
        #     continue

        #for this family ID, check arrivals from EQinfo
        for sta in EQinfo[ID]['sta'].keys():
            P1 = EQinfo[ID]['sta'][sta]['P1'] #Phase 1
            P2 = EQinfo[ID]['sta'][sta]['P2'] #Phase 2

            #some data are in cascadia_CN
            if sta in CN_list:
                dataDir = dir2
                net = net2
                comp = CN_list[sta]
            else:
                dataDir = dir1
                net = net1
                comp = 'HH' #PO stations are always HH

            #print('station:%s'%(sta))

            if P1 != -1:
                arr = OT+P1
                t1 = arr - timeL
                t2 = arr + timeR
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

                if t1.strftime('%Y%m%d') == t2.strftime('%Y%m%d'):
                    #only load t1 file
                    #print('prev_dataZ=',prev_dataZ)
                    if not (os.path.exists(prev_dataZ[sta]['file1']['name']) & os.path.exists(prev_dataE[sta]['file1']['name']) & os.path.exists(prev_dataN[sta]['file1']['name'])):
                        #continue #either Z,E or N file are missing
                        pass #should go test S phase
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
                        DD = np.concatenate([D_Z,D_E,D_N])
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
                    DD = np.concatenate([D_Z,D_E,D_N])
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
                    DD = np.concatenate([D_Z,D_E,D_N])                  
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
    h5f = h5py.File('./Data/ID_%s_%s_P.h5'%(currentFam,name),'w')
    h5f.create_dataset('waves',data=sta_P1[name])
    h5f.close()
for name in sta_P2.keys():
    #save the h5py data
    h5f = h5py.File('./Data/ID_%s_%s_S.h5'%(currentFam,name),'w')
    h5f.create_dataset('waves',data=sta_P2[name])
    h5f.close()












