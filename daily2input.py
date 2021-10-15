### convert daily timeseries .mseed data to model input ###

import numpy as np
import obspy
from obspy import UTCDateTime
import glob
import os
import sys
import h5py
import unet_tools
import matplotlib
matplotlib.use('pdf') #instead using interactive backend
import matplotlib.pyplot as plt
import datetime
from joblib import Parallel, delayed


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


def detc_sta(model_name,sta,outdir='Detections_S',resume=False,save_hdf5=False):
    '''
    Detect LFEs from continuous data. Model, data path and output name are hard-coded in the function
    but modify them are encouraged
        Input:
            sta:        station name e.g. "PGC"
            outdir:     output directory
            resume:     resume the calculation from the last stop. If the stop is within a day (daily data incomplete), delete the whole date in .csv file
                        Careful! resume mode do NOT work when save_hdf5=True due to the hdf5 file corrupt.
            save_hdf5:  save the 15s detection waveforms (this is slow when saving over 1M of waveforms)
            
    '''
    #--- parameters
    #model_path = 'large_1.0_unet_lfe_std_0.4.tf.02'
    #model_path = 'large_1.0_unet_lfe_std_0.4.tf.002' # this is the S wave model
    #model_path = 'large_1.0_unet_lfe_std_0.4.tf.006' # this is the P wave model
    model_path = model_name
    drop = False
    N_size = float(model_path.split('/')[-1].split('_')[1])
    run_num = model_path.split('/')[-1].split('.tf.')[-1]

    #--- build model artechetecture
    if drop:
        model = unet_tools.make_large_unet_drop(N_size,sr=100,ncomps=3)
    else:
        model = unet_tools.make_large_unet(N_size,sr=100,ncomps=3)

    # load weights
    model.load_weights(model_path)

    #------where are the data, and what's its channel-------
    # These help the Line#197 where to find *Z.mseed data based on the station name(the only given input)
    # the data can only be in PO or CN for my case
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
    #------define where are the data END-----------

    # epsilon value shouldn't change
    epsilon = 1e-6
    # what decision value are you using
    thres = 0.1

    if sta in CN_list:
        #data under CN
        net = net2
        tar_dir = dir2
        chn = CN_list[sta]
        loc = '' #loc can be '00','01' etc. but is always empty in this case
    else:
        net = net1
        tar_dir = dir1
        chn = 'HH'
        loc = '' #loc is always empty in this case

    # get all the dailydata list for this station
    print('get data form: ',tar_dir+'/*.'+net+'.'+sta+'.'+loc+'.'+chn+'Z.mseed')
    D_Zs = glob.glob(tar_dir+'/*.'+net+'.'+sta+'.'+loc+'.'+chn+'Z.mseed')
    D_Zs.sort()
  
    print(' Total of %d data found'%(len(D_Zs)))
    if len(D_Zs)==0:
        return # for this sta input, none data can be found

    # create output directory & file here
    if not(os.path.exists(outdir)):
        os.makedirs(outdir)
    # csv file to save detection information
    file_csv = outdir+'/'+'cut_daily_%s.%s.csv'%(net,sta)
    # hdf5 file to save waveforms
    if save_hdf5:
        file_hdf5 = outdir+'/'+'cut_daily_%s.%s.hdf5'%(net,sta)
    #file_csv = './Detections_S/cut_daily_%s.%s.csv'%(net,sta)
    #file_hdf5 = './Detections_S/cut_daily_%s.%s.hdf5'%(net,sta)
    #file_csv = './Detections_P/cut_daily_%s.%s.csv'%(net,sta)
    #file_hdf5 = './Detections_P/cut_daily_%s.%s.hdf5'%(net,sta)
    if not(os.path.exists(file_csv)):
        OUT1 = open(file_csv,'w')
        OUT1.write('network,sta,chn,stlon,stlat,stdep,starttime,endtime,y,idx_max_y,id\n')
        OUT1.close()
    else:
        if resume:
            print("File: %s already exist! Resume calculation......"%(file_csv))
        else:    
            print("File: %s already exist! Exit and not overwritting everything"%(file_csv))
            sys.exit()

    # create hdf5 file
    if save_hdf5:
        if resume:
            print("WARNING! resume on hdf5 file: %s can cause unknown issue, terminate for safty"%(file_hdf5))
            sys.exit()
        else:
            if not(os.path.exists(file_hdf5)):
                hf = h5py.File(file_hdf5,'w')
                hf.create_group('data') #create group of data
                hf.close()
            else:
                print("File: %s already exist! Exit and not overwritting everything"%(file_hdf5))
                sys.exit()

    # get station loc
    try:
        stlon,stlat,stdep = stainfo[sta] #find station location
    except:
        stlon,stlat,stdep = -1,-1,-1 # no station location information

    # for each Z, find other components
    #num = 0
    # e.g. D_Z = "/projects/amt/shared/cascadia_CN/20140422.CN.YOUB..HHZ.mseed"
    if resume:
        import pandas as pd
        csv = pd.read_csv(file_csv)
        prev_T = pd.to_datetime(csv['starttime'],format='%Y-%m-%dT%H:%M:%S.%fZ')
        prev_T = list(set([i_T.strftime('%Y%m%d')  for i_T in prev_T]))
        prev_T.sort()

    #======start looping all the daily data=======
    for D_Z in D_Zs:
        if resume:
           cur_T = D_Z.split('/')[-1].split('.')[0]
           if cur_T in prev_T:
               continue # continue to the next loop

        OUT1 = open(file_csv,'a')
        if save_hdf5:
            hf = h5py.File(file_hdf5,'a')
        print('--currently at:',D_Z)
        comp = D_Z.split('/')[-1].split('.')[-2]
        D_E = D_Z.replace(comp,comp[:2]+'E')
        D_N = D_Z.replace(comp,comp[:2]+'N')
        # also check if the E,N data exist
        if (not os.path.exists(D_E)) or (not os.path.exists(D_N)):
            print('Missing at least one component! data: %s or %s does not exist!'%(D_E,D_N))
            OUT1.close()
            if save_hdf5:
                hf.close()
            continue
        
        # process the 3C data to be exact 1 day, detrend and high-pass
        try:
            D_Z = data_process(D_Z,sampl=100) # sampling rate = 100
            D_E = data_process(D_E,sampl=100)
            D_N = data_process(D_N,sampl=100)
        except:
            OUT1.close()
            if save_hdf5:
                hf.close()
            continue
        times = D_Z[0].times()
        t_start = D_Z[0].stats.starttime
        t_end = D_Z[0].stats.endtime
        assert len(D_Z[0].data)==len(D_E[0].data)==len(D_N[0].data)==8640001, "Check data_process!"
        #only need numpy array
        D_Z = D_Z[0].data
        D_E = D_E[0].data
        D_N = D_N[0].data

        # cut daily data into 15 s data
        #t_st = D_Z[0].stats.starttime
        wid_sec = 15
        sr = 100
        wid_pts = wid_sec * sr
        wid_T = np.arange(wid_pts)*(1.0/sr)
        i_st = 0
        i_ed = i_st + wid_pts
        sav_data = []
        sav_data_Z = []
        sav_data_E = []
        sav_data_N = []
        #time_bef = datetime.datetime.now()
        while i_ed<=8640001:
            data_Z = D_Z[i_st:i_ed]
            data_E = D_E[i_st:i_ed]
            data_N = D_N[i_st:i_ed]
            #normalize data by the 3-Comps max value, this is what the model was trained
            norm_val = max(max(np.abs(data_Z)),max(np.abs(data_E)),max(np.abs(data_N))) 
            TT = times[i_st:i_ed]
            tr_starttime = t_start + TT[0]
            tr_endtime = t_start + TT[-1]
            #tr_id = '.'.join([net,sta,chn])+'_'+tr_starttime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4]
            #print('T=',TT[0],TT[-1],tr_starttime,tr_endtime)
            i_st = i_ed  #the first point of the 2nd round is same as the last point of the 1st round
            i_ed = i_st + wid_pts
            # dealing with data
            #DD = np.concatenate([data_Z,data_E,data_N])
            #data_inp = ZEN2inp(data_Z,data_E,data_N,epsilon)
            data_inp = ZEN2inp(data_Z/norm_val,data_E/norm_val,data_N/norm_val,epsilon)  #normalize the data
            sav_data.append(data_inp)  # run model prediction in batch
            sav_data_Z.append(data_Z)  #save raw data (without normalized and feature engineering)
            sav_data_E.append(data_E)
            sav_data_N.append(data_N)

        #==== batch prediction, this is way faster====
        sav_data = np.array(sav_data)
        tmp_y = model.predict(sav_data) #prediction in 2d array
        idx_lfe = np.where(tmp_y.max(axis=1)>=thres)[0]
        # get each lfe info
        for i_lfe in idx_lfe:
            # QC check first
            D_merge = np.hstack([sav_data_Z[i_lfe],sav_data_E[i_lfe],sav_data_N[i_lfe]])
            D_merge = D_merge/np.max(np.abs(D_merge)) #normalized
            if not QC(D_merge):
                continue # reject 
            # find the maximum y
            idx_maxy = np.where(tmp_y[i_lfe]==np.max(tmp_y[i_lfe]))[0][0]
            #get time from index and max y for each trace
            tr_starttime = t_start + i_lfe*wid_sec 
            tr_endtime = t_start + (i_lfe+1)*wid_sec - (1.0/sr)
            lfe_time = tr_starttime + wid_T[idx_maxy]
            tr_id = '.'.join([net,sta,chn])+'_'+lfe_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4] #trace id
            if resume:
                # some data are old, overwrite the old data, add the new data
                OUT1.write('%s,%s,%s,%.4f,%.4f,%f,%s,%s,%.2f,%d,%s\n'%(net,sta,chn,stlon,stlat,stdep,tr_starttime,tr_endtime,tmp_y[i_lfe][idx_maxy],idx_maxy,tr_id))
                '''
                # test if a corrupted hdf5 file can be fixed......
                try:
                    del hf['data/'+tr_id]
                except:
                    #print('dataset not exist')
                    pass
                hf.create_dataset('data/'+tr_id,data=[sav_data_Z[i_lfe],sav_data_E[i_lfe],sav_data_N[i_lfe]]) #this is the raw data (no feature scaling)
                
                chk_k = 'data/'+tr_id
                tmp_data = hf.get(chk_k)
                tmp_data = hf.get(chk_k)
                if tmp_data:
                    #overwrite the existing data
                    tmp_data[0] = sav_data_Z[i_lfe]
                    tmp_data[1] = sav_data_E[i_lfe]
                    tmp_data[2] = sav_data_N[i_lfe]
                else:
                    # data do not exist, create new dataset
                    #print('data do not exist:','data/'+tr_id)
                    try:
                        hf.create_dataset('data/'+tr_id,data=[sav_data_Z[i_lfe],sav_data_E[i_lfe],sav_data_N[i_lfe]]) #this is the raw data (no feature scaling)
                    except:
                        # hdf5 dataset corrupt
                        print("hdf5 dataset corrupted, no way to recover........?")
                        hf.close()
                        sys.exit()
                        # create dataset but add an additional "S"
                        #hf.create_dataset('data/'+tr_id+'S',data=[sav_data_Z[i_lfe],sav_data_E[i_lfe],sav_data_N[i_lfe]]) #this is the raw data (no feature scaling)
                '''

            else:
                # writing new data
                OUT1.write('%s,%s,%s,%.4f,%.4f,%f,%s,%s,%.2f,%d,%s\n'%(net,sta,chn,stlon,stlat,stdep,tr_starttime,tr_endtime,tmp_y[i_lfe][idx_maxy],idx_maxy,tr_id))
                if save_hdf5:
                    hf.create_dataset('data/'+tr_id,data=[sav_data_Z[i_lfe],sav_data_E[i_lfe],sav_data_N[i_lfe]]) #this is the raw data (no feature scaling)
        
        #save result every sta,daily
        OUT1.close()
        if save_hdf5:
            hf.close()

        # early stop for debug purpose
        #num += 1
        #if num>10:
        #    break
        





# main 
'''
#--- get the station list here
# this gives a list of names: e.g.  ['PGC','GOBB','KLNB'.....]
phases = np.load('sav_family_phases.npy',allow_pickle=True)
phases = phases.item()
all_sta = []
for k in phases.keys():
    all_sta += list(phases[k]['sta'].keys())

all_sta = list(set(all_sta))

print('number of station:',len(all_sta)) # how many stations are used in the catalog

# --- load station info(location) file
stainfo = np.load('stainfo.npy',allow_pickle=True)
stainfo = stainfo.item()

#check if station coord are completed
for s in all_sta:
    if not (s in stainfo):
        print("station: %s not found"%(s))
'''

# --- load station info(location) file
#stainfo = np.load('stainfo.npy',allow_pickle=True)
#stainfo = stainfo.item()

# provide a list of station name directly
all_sta = ["SILB","SSIB","GOWB"]
model_name = "large_1.0_unet_lfe_std_0.4.tf.002" #S-wave model
outdir = "Detections_S_new"
resume = True
save_hdf5 = False
n_cores = 8

# parallel processing
results = Parallel(n_jobs=n_cores,verbose=0)(delayed(detc_sta)(model_name,sta,outdir,resume,save_hdf5) for sta in all_sta  )


