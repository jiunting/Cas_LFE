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


'''
#--- parameters
#model_path = 'large_2.0_unet_lfe_std_0.4.tf.014'
model_path = 'large_1.0_unet_lfe_std_0.4.tf.02'
#model_path = 'large_0.5_unet_lfe_std_0.4.tf.01'
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


epsilon = 1e-6
thres = 0.1
'''

#dir1 = '/Users/timlin/Documents/Project/Cascadia_LFE'
#dir2 = '/Users/timlin/Documents/Project/Cascadia_LFE'

#--- get all the stations
phases = np.load('sav_family_phases.npy',allow_pickle=True)
phases = phases.item()
all_sta = []
for k in phases.keys():
    all_sta += list(phases[k]['sta'].keys())

all_sta = list(set(all_sta))

print('number of station:',len(all_sta))

# --- load station info(location) file
stainfo = np.load('stainfo.npy',allow_pickle=True)
stainfo = stainfo.item()

#check if station coord are completed
for s in all_sta:
    if not (s in stainfo):
        print("station: %s not found"%(s))

'''
# create a csv file to save file information
if not(os.path.exists('cut_daily_all.csv')):
    OUT1 = open('cut_daily_all.csv','w')
    OUT1.write('network,sta,chn,stlon,stlat,stdep,starttime,endtime,y,idx_max_y,id\n')
    OUT1.close()
else:
    print("File: cut_daily_all.csv already exist! Exit and not overwritting everything")
    sys.exit()


# create hdf5 file
hf = h5py.File('cut_daily_all.hdf5','a')
hf.create_group('data') #create group of data
hf.close()
'''

#OUT2 = open('all_sta.txt','w') # all the station used
#OUT2.write('#sta stlon stlat elev\n')
#--- for each station, cut data to ML's input format
#for sta in all_sta:

def detc_sta(sta):
    #from LFE_tools import data_process, data_cut, ZEN2inp, cal_CCC, QC

    #--- parameters
    run_num = 'P003'
    #model_path = 'large_1.0_unet_lfe_std_0.4.tf.02'
    #model_path = 'large_1.0_unet_lfe_std_0.4.tf.002' # this is the S wave model
    #model_path = 'large_1.0_unet_lfe_std_0.4.tf.006' # this is the P wave model
    drop = False
    N_size = 2
    #N_size = float(model_path.split('/')[-1].split('_')[1])
    #run_num = model_path.split('/')[-1].split('.tf.')[-1]

    #--- build model artechetecture
    if drop:
        model = unet_tools.make_large_unet_drop(N_size,sr=100,ncomps=3)
    else:
        model = unet_tools.make_large_unet(N_size,sr=100,ncomps=3)

    # load weights
    chks = glob.glob("./checks/large*%s*"%(run_num))
    chks.sort()
    model.load_weights(chks[-1].replace('.index',''))
    #model.load_weights(model_path)

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
    dir3 = '/projects/amt/shared/cascadia_C8'
    net3 = 'C8'
    C8_list = {'BPCB':'HH',
               'GLBC':'HH',
               'JRBC':'HH',
               'LCBC':'HH',
               'MGCB':'HH',
               'PHYB':'HH',
               'SHDB':'HH',
               'SHVB':'HH',
               'SOKB':'HH',
               'TWBB':'HH',
            }


    epsilon = 1e-6
    thres = 0.1


    #OUT1 = open('cut_daily_all.csv','a')
    #hf = h5py.File('cut_daily_all.hdf5','a')
    #sta = 'SNB'
    if sta in CN_list:
        #data under CN
        net = net2
        tar_dir = dir2
        chn = CN_list[sta]
        loc = '' #loc can be '00','01' etc. but is always empty in this case
    elif sta in C8_list:
        net = net3
        tar_dir = dir3
        chn = C8_list[sta]
        loc = ''  
    else:
        net = net1
        tar_dir = dir1
        chn = 'HH'
        loc = '' #loc is always empty in this case

    # get all the dailydata list
    print('get data form: ',tar_dir+'/*.'+net+'.'+sta+'.'+loc+'.'+chn+'Z.mseed')
    D_Zs = glob.glob(tar_dir+'/*.'+net+'.'+sta+'.'+loc+'.'+chn+'Z.mseed')
    D_Zs.sort()
  
    print(' Total of %d data found'%(len(D_Zs)))
    if len(D_Zs)==0:
        return # for this sta input, none data can be found

    # create output file here
    # create a csv file to save file information
    #file_csv = './Detections_P/cut_daily_%s.%s.csv'%(net,sta)
    #file_hdf5 = './Detections_P/cut_daily_%s.%s.hdf5'%(net,sta)
    #file_csv = './Detections_P_new/cut_daily_%s.%s.csv'%(net,sta)  #the previous Detections_P_new has renamed to Detections_P_addSta i.e. additional station detc
    #file_hdf5 = './Detections_P_new/cut_daily_%s.%s.hdf5'%(net,sta)
    file_csv = './Detections_P_C8_new/cut_daily_%s.%s.csv'%(net,sta)  #the previous Detections_P_new has renamed to Detections_P_addSta i.e. additional station detc
    file_hdf5 = './Detections_P_C8_new/cut_daily_%s.%s.hdf5'%(net,sta)
    #file_csv = './Detections_P/cut_daily_%s.%s.csv'%(net,sta)
    #file_hdf5 = './Detections_P/cut_daily_%s.%s.hdf5'%(net,sta)
    if not(os.path.exists(file_csv)):
        OUT1 = open(file_csv,'w')
        #OUT1.write('network,sta,chn,stlon,stlat,stdep,starttime,endtime,y,idx_max_y,id\n')
        OUT1.write('starttime,endtime,y,idx_max_y,id\n')
        OUT1.close()
    else:
        print("File: cut_daily_all.csv already exist! Exit and not overwritting everything")
        sys.exit()

    # create hdf5 file
    hf = h5py.File(file_hdf5,'a')
    hf.create_group('data') #create group of data
    hf.close()

    # get station loc
    stlon,stlat,stdep = stainfo[sta] #need to find station location
    #OUT2.write('%s %f %f %f\n'%(sta,stlon,stlat,stdep))

    # for each Z, find other components
    #num = 0
    for D_Z in D_Zs:
        OUT1 = open(file_csv,'a')
        hf = h5py.File(file_hdf5,'a')
        print('--currently at:',D_Z)
        comp = D_Z.split('/')[-1].split('.')[-2]
        D_E = D_Z.replace(comp,comp[:2]+'E')
        D_N = D_Z.replace(comp,comp[:2]+'N')
        # also check if the E,N data exist
        if (not os.path.exists(D_E)) or (not os.path.exists(D_N)):
            print('Missing at least one component! data: %s or %s does not exist!'%(D_E,D_N))
            hf.close()
            OUT1.close()
            continue
        
        # process the 3C data to be exact 1 day, detrend and high-pass
        try:
            D_Z = data_process(D_Z,sampl=100) # sampling rate = 100
            D_E = data_process(D_E,sampl=100)
            D_N = data_process(D_N,sampl=100)
        except:
            hf.close()
            OUT1.close()
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
            sav_data_Z.append(data_Z)  #raw data without processing
            sav_data_E.append(data_E)
            sav_data_N.append(data_N)

        #batch prediction, this is way faster
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
            #OUT1.write('%s,%s,%s,%.4f,%.4f,%f,%s,%s,%.2f,%d,%s\n'%(net,sta,chn,stlon,stlat,stdep,tr_starttime,tr_endtime,tmp_y[i_lfe][idx_maxy],idx_maxy,tr_id))
            OUT1.write('%s,%s,%.2f,%d,%s\n'%(tr_starttime,tr_endtime,tmp_y[i_lfe][idx_maxy],idx_maxy,tr_id))
            hf.create_dataset('data/'+tr_id,data=[sav_data_Z[i_lfe],sav_data_E[i_lfe],sav_data_N[i_lfe]]) #this is the raw data (no feature scaling)
            #print('  save csv:',tr_id)
            #print('  save h5py:','data/'+tr_id)

        #time_aft = datetime.datetime.now()
        #print('total run time:',(time_aft-time_bef)/60.0)
        
        #save result every sta,daily
        hf.close()
        OUT1.close()

        # early stop for debug purpose
        #num += 1
        #if num>10:
        #    break
        
        #sav_data = np.array(sav_data)
        #np.save('testQQQ.npy',sav_data)
        #sys.exit()
    #save result every sta
    #hf.close()
    #OUT1.close()
    #break




#OUT2.close()


#process one-by-one
#print('testing station:','SSIB')
#detc_sta('SSIB')
#print('testing station:','TWKB')
#detc_sta('TWKB')

# parallel processing
#all_sta = ['SSIB','TWKB']

all_sta = ['BPCB','GLBC','JRBC','LCBC','MGCB','PHYB','SHDB','SHVB','SOKB','TWBB'] #C8 stations
n_cores = len(all_sta)
results = Parallel(n_jobs=n_cores,verbose=10)(delayed(detc_sta)(sta) for sta in all_sta  )





