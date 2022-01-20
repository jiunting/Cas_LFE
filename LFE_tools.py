import numpy as np
import obspy
from obspy import UTCDateTime
import os


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
    assert len(DD[0].data)==3001, "cut data not exactly 3001 points"
    return DD[0].data



def ZEN2inp(Z,E,N,epsilon):
    # convert raw ZEN data to input feature
    data_Z_sign = np.sign(Z)
    data_E_sign = np.sign(E)
    data_N_sign = np.sign(N)
    data_Z_val = np.log(np.abs(Z)+epsilon)
    data_E_val = np.log(np.abs(E)+epsilon)
    data_N_val = np.log(np.abs(N)+epsilon)
    data_inp = np.hstack([data_Z_val.reshape(-1,1),data_Z_sign.reshape(-1,1),
                          data_E_val.reshape(-1,1),data_E_sign.reshape(-1,1),
                          data_N_val.reshape(-1,1),data_N_sign.reshape(-1,1),])
    return data_inp


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


