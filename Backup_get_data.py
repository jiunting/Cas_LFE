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

#load family and arrival information from HypoINV file
EQinfo = np.load('sav_family_phases.npy',allow_pickle=True)
EQinfo = EQinfo.item()

#LFE detection file
detcFile = 'total_mag_detect_0000_cull_NEW.txt' 

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


def data_process(filePath1,filePath2='',t1=UTCDateTime("20100101"),t2=UTCDateTime2000("20100102"),sampl=sampl):
    '''
    process data from one or multiple .mseed
    filePath1, filePath2: absolute path of .mseed file. leave filePath2 blank if only unsing one mseed file 
    t1, t2: start and end time of the timeseries
    sampl: sampling rate
    '''
    if filePath2 == '':
        D = obspy.read(filePath1)
        if len(D)!=1:
            D.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
        D.detrend('linear')
        D.taper(0.02) #2% taper
        D.filter('highpass',frep=1.0)
        D.trim(starttime=t1-1,endtime=t2+1,nearest_sample=True, pad=True, fill_value=0)
        D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
        D.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
    else:
        D = obspy.read(filePath1)
        D2 = obspy.read(filePath2)
        D += D2
        D.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
        D.detrend('linear')
        D.taper(0.02)
        D.filter('highpass',frep=1.0)
        D.trim(starttime=t1-1,endtime=t2+1,nearest_sample=True, pad=True, fill_value=0)
        D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
        D.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
    return D



with open(detcFile,'r') as IN1:
    for line in IN1.readlines():
        line = line.strip()
        #print(line)
        ID = line.split()[0] #family ID
        OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
        HH = (int(line.split()[2])-1)*3600  #HR from 1-24
        SS = float(line.split()[3])
        OT = OT + HH + SS #detected origin time
        Mag = float(line.split()[4])
        #if Mag<2.49:
        #    continue
        
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
                
                if t1.strftime('%Y%m%d') == t2.strftime('%Y%m%d'):
                    #only load t1 file
                    if not (os.path.exists(t1_fileZ) & os.path.exists(t1_fileE) & os.path.exists(t1_fileN)):
                        continue
                    #print(' --Begin cut from',t1_fileZ)
                    D = obspy.read(t1_fileZ)
                    if len(D)!=1:
                        D.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
                    D.detrend('linear')
                    D.trim(starttime=t1-1,endtime=t2+1,nearest_sample=True, pad=True, fill_value=0)
                    D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
                    D.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
                    D.write('./Data/Fam_%s_%s_%s_P1.mseed'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),format='MSEED')
                    D.clear()
                else:
                    #time window across different day, read both
                    if not (os.path.exists(t1_fileZ) & os.path.exists(t1_fileE) & os.path.exists(t1_fileN)):
                        continue
                    if not (os.path.exists(t2_fileZ) & os.path.exists(t2_fileE) & os.path.exists(t2_fileN)):
                        continue

                    #print(' --Begin cut from both',t1_fileZ,t2_fileZ)
                    D = obspy.read(t1_fileZ)
                    D2 = obspy.read(t2_fileZ)
                    D += D2
                    D.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
                    D.detrend('linear')
                    D.trim(starttime=t1-1,endtime=t2+1,nearest_sample=True, pad=True, fill_value=0)
                    D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
                    D.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
                    D.write('./Data/Fam_%s_%s_%s_P1.mseed'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),format='MSEED')
                    D.clear()

                print('  --sta:%5s %s'%(sta,arr.isoformat()))
            #continue #test P1 first
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

                if t1.strftime('%Y%m%d') == t2.strftime('%Y%m%d'):
                    #only load t1 file
                    if not os.path.exists(t1_fileZ):
                        continue
                    #print(' --Begin cut from',t1_fileZ)
                    D = obspy.read(t1_fileZ)
                    if len(D)!=1:
                        D.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
                    D.detrend('linear')
                    D.trim(starttime=t1-1,endtime=t2+1,nearest_sample=True, pad=True, fill_value=0)
                    D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
                    D.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
                    D.write('./Data/Fam_%s_%s_%s_P2.mseed'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),format='MSEED')
                    D.clear()
                else:
                    #time window across different day, read both
                    if (not os.path.exists(t1_fileZ)) or (not os.path.exists(t1_fileZ)):
                        continue
                    #print(' --Begin cut from both',t1_fileZ,t2_fileZ)
                    D = obspy.read(t1_fileZ)
                    D2 = obspy.read(t2_fileZ)
                    D += D2
                    D.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
                    D.detrend('linear')
                    D.trim(starttime=t1-1,endtime=t2+1,nearest_sample=True, pad=True, fill_value=0)
                    D.interpolate(sampling_rate=sampl, starttime=t1,method='linear')
                    D.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
                    D.write('./Data/Fam_%s_%s_%s_P2.mseed'%(ID,sta,arr.strftime('%Y%m%d_%H%M%S.%f')[:-2]),format='MSEED')
                    D.clear()

                print('  --sta:%5s %s'%(sta,arr.isoformat()))
                
                pass
    













