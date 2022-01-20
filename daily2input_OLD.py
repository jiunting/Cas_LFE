### convert daily timeseries .mseed data to model input ###

import numpy as np
import obspy
from obspy import UTCDateTime
import glob
import os

### some functions
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
    sampl = 100
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


#--- where are the data
dir1 = '/projects/amt/shared/cascadia_PO' # .mseed files are in PO or CN directory
net1 = 'PO' #PO stations are always HH
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

#--- get all the stations
phases = np.load('sav_family_phases.npy',allow_pickle=True)
phases = phases.item()
all_sta = []
for k in phases.keys():
    all_sta += list(phases[k]['sta'].keys())

all_sta = list(set(all_sta))


import time
time0 = time.time()
#--- for each station, cut data to ML's input format
for sta in all_sta:
    sta = 'TWKB'
    print('Start sta=',sta)
    if sta in CN_list:
        #data under CN
        net = net2
        tar_dir = dir2
        chn = CN_list[sta]
    else:
        net = net1
        tar_dir = dir1
        chn = 'HH'

    # get all the dailydata list
    print(tar_dir+'/*.'+net+'.'+sta+'..'+chn+'Z.mseed')
    D_Zs = glob.glob(tar_dir+'/*.'+net+'.'+sta+'..'+chn+'Z.mseed')
    D_Zs.sort()

    # for each Z, find other components
    sav_DD = []
    for D_Z in D_Zs:
        print(' data found:',D_Z)
        comp = D_Z.split('/')[-1].split('.')[-2]
        D_E = D_Z.replace(comp,comp[:2]+'E')
        D_N = D_Z.replace(comp,comp[:2]+'N')
        # also check if the E,N data exist
        if (not os.path.exists(D_E)) or (not os.path.exists(D_N)):
            print('data: %s or %s does not exist!'%(D_E,D_N))
            continue
        
        # start processing 3C data
        D_Z = data_process(D_Z,sampl=100) # sampling rate = 100
        D_E = data_process(D_E,sampl=100)
        D_N = data_process(D_N,sampl=100)

        # cut daily data into 30 s data
        t_st = D_Z[0].stats.starttime
        while t_st<(D_Z[0].stats.endtime-30):
            t_ed = t_st + 30
            #print('process data from :',t_st,t_ed)
            data_Z = data_cut(D_Z,Data2='',t1=t_st,t2=t_ed)
            data_E = data_cut(D_E,Data2='',t1=t_st,t2=t_ed)
            data_N = data_cut(D_N,Data2='',t1=t_st,t2=t_ed)
            assert len(data_Z)==len(data_E)==len(data_N)==3001, "length are different! check the data processing"
            DD = np.concatenate([data_Z,data_E,data_N])
            sav_DD.append(DD)
            t_st += 30
    time1 = time.time()
    print('========Total runtime=',time1-time0)
    sav_DD = np.array(sav_DD)
    np.save('sav_DD.npy',sav_DD)
    break




