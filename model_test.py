#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:54:21 2020

Train a CNN to detect LFEs please

@author: amt

6/18 2022: Test model with traiching evID, modified by Tim Lin
"""
import h5py
import matplotlib
matplotlib.use('pdf') #instead using interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import unet_tools
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import glob
from obspy import UTCDateTime
import obspy

class data_generator(keras.utils.Sequence):
    #######Generator should inherit the "Sequence" class in order to run multi-processing of fit_generator###########
    """
    Generator for Unet training. Modified the x_data, sig_inds part on 6/3 2022

    INPUTs
    ======
    batch_size: int
        Batch size for each training step
    x_data_file: str
        Name for big h5py file with all the data. Can be called by the key in the .csv file
    sig_inds: list or np.ndarray
        List of event index. Use this to get the keys that call h5py data.
        e.g. sig_inds[0] = 1753 (the event of catalog[1753]), then available data are csv[csv['idxCatalog']==1753]
            famID net   sta comp PS                                       evID  idxCatalog
                1  CN   PFB   HH  S   2008-05-20T01:46:29.7000_001.CN.PFB.HH.S        1753
                1  CN   SNB   BH  S   2008-05-20T01:46:29.7000_001.CN.SNB.BH.S        1753
                1  CN   VGZ   BH  S   2008-05-20T01:46:29.7000_001.CN.VGZ.BH.S        1753
                1  CN  YOUB   HH  S  2008-05-20T01:46:29.7000_001.CN.YOUB.HH.S        1753
            There are 4 traces recorded the event 1753, use the evIDs to get data.
    n_data_file: str
        Noise data. n_data['waves'] yields everything already
    noise_inds: list or np.ndarray
        Indexes get used
    csv_file: str
        A csv file to get the evID from idxCatalog
    sr: int
        Sampling rate
    std: float
        Standard deviation of the gaussian for arrival labels.

    OUTs
    ====
    Training or testing data mixed with signal and noise
    """
    def __init__(self,batch_size,x_data_file,n_data_file,sig_inds,noise_inds,csv_file,sr,std,valid=False):
        self.batch_size = batch_size
        self.x_data_file = x_data_file
        self.n_data_file = n_data_file
        self.sig_inds = sig_inds
        self.noise_inds = noise_inds
        self.csv_file = csv_file
        self.sr = sr
        self.std = std
        self.valid = valid

    def __len__(self):
        #length of noise
        return len(self.noise_inds)

    def __getitem__(self,index):
        # index is a dummy since each traces will be shifted and so is different
        np.random.seed()
        batch_size = self.batch_size
        x_data_file = self.x_data_file
        sig_inds = self.sig_inds
        n_data_file = self.n_data_file
        noise_inds = self.noise_inds
        csv_file = self.csv_file
        sr = self.sr
        std = self.std
        valid = self.valid

        x_data = h5py.File(x_data_file,'r')
        n_data = h5py.File(n_data_file,'r')
        csv = pd.read_csv(csv_file)
        #while True:
        # randomly select a starting index for the data batch
        #start_of_data_batch=np.random.choice(len(sig_inds)-batch_size//2)
        # randomly select a starting index for the noise batch
        #start_of_noise_batch=np.random.choice(len(noise_inds)-batch_size//2)
        np.random.shuffle(sig_inds)
        np.random.shuffle(noise_inds)
        if valid:
            start_of_noise_batch=0
            start_of_data_batch=0
        # get range of indicies for data and noise. Data have been shuffled, so just select from begining to half batch size
        datainds = sig_inds[:batch_size//2] # NOTE that this is the index, each index has at least one id which will be processed later
        noiseinds = noise_inds[:batch_size//2]
        noiseinds.sort()

        #length of each component
        nlen = int(sr*30)+1

        # Get keys from looping each datainds until getting enough data traces i.e. batch_size//2
        accum_n = 0
        comp1_data = []
        comp2_data = []
        comp3_data = []
        for tmpidx in datainds:
            tmp_keys = csv[csv['idxCatalog']==tmpidx]['evID']
            n = len(tmp_keys) #number of data to be added
            if (accum_n + n) < batch_size//2:
                for k in tmp_keys:
                    data_all = x_data['waves/'+k]
                    comp1_data.append(data_all[:nlen])
                    comp2_data.append(data_all[nlen:2*nlen])
                    comp3_data.append(data_all[2*nlen:])
                accum_n += n
            else:
                res_n = batch_size//2 - accum_n # number of traces needed
                #print('  number of traces remaining...',res_n)
                tmp_keys = [k for k in tmp_keys]
                np.random.shuffle(tmp_keys)
                #print('    randomly draw from %d keys'%len(tmp_keys))
                for i in range(res_n):
                    #print('  adding',i+1)
                    data_all = x_data['waves/'+tmp_keys[i]]
                    comp1_data.append(data_all[:nlen])
                    comp2_data.append(data_all[nlen:2*nlen])
                    comp3_data.append(data_all[2*nlen:])
                break
        assert len(comp1_data)==batch_size//2, "size incorrect!"
        comp1_data = np.array(comp1_data)
        comp2_data = np.array(comp2_data)
        comp3_data = np.array(comp3_data)
        # grab batch
        #=====added these lines by Tim=====
        #normalize the data and noise
        #data_all = x_data['waves'][datainds]/np.max(np.abs(x_data['waves'][datainds]),axis=1).reshape(-1,1)
        #noise_all = n_data['waves'][noiseinds]/np.max(np.abs(n_data['waves'][noiseinds]),axis=1).reshape(-1,1)
        #comp1=np.concatenate((data_all[:,:nlen],noise_all[:,:nlen]))
        #comp2=np.concatenate((data_all[:,nlen:2*nlen],noise_all[:,nlen:2*nlen]))
        #comp3=np.concatenate((data_all[:,2*nlen:],noise_all[:,2*nlen:]))
        #==================================
        comp1=np.concatenate((comp1_data,n_data['waves'][noiseinds,:nlen]))
        comp2=np.concatenate((comp2_data,n_data['waves'][noiseinds,nlen:2*nlen]))
        comp3=np.concatenate((comp3_data,n_data['waves'][noiseinds,2*nlen:]))
        # make target data vector for batch
        target=np.concatenate((np.ones_like(datainds),np.zeros_like(noiseinds)))
        # make structure to hold target functions
        batch_target=np.zeros((batch_size,nlen))
        # shuffle things so not always signal first and then noise
        inds=np.arange(batch_size)
        np.random.shuffle(inds)
        comp1=comp1[inds,:]
        comp2=comp2[inds,:]
        comp3=comp3[inds,:]
        target=target[inds]
        # some params
        winsize=15 # winsize in seconds
        # this just makes a nonzero value where the pick is
        for ii, targ in enumerate(target):
            #print(ii,targ)
            if targ==0:
                #batch_target[ii,:]=1/nlen*np.ones((1,nlen))   #this is label for noise
                batch_target[ii,:]=np.zeros((1,nlen))   #this is label for noise
            elif targ==1:
                batch_target[ii,:]=signal.gaussian(nlen,std=int(std*sr)) #this is data
        # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        time_offset=np.random.uniform(0,winsize,size=batch_size)
        # initialize arrays to hold shifted data
        new_batch=np.zeros((batch_size,int(winsize*sr),3))
        new_batch_target=np.zeros((batch_size,int(winsize*sr)))
        # this loop shifts data and targets and stores results
        # for data, arrivals at exactly 15 s, randomly cut the 30 s data from min: 0-15s to max: 15-30s
        # do the same process for target
        for ii,offset in enumerate(time_offset):
            bin_offset=int(offset*sr) #HZ sampling Frequency
            start_bin=bin_offset
            end_bin=start_bin+int(winsize*sr)
            c1 = comp1[ii,start_bin:end_bin]
            c2 = comp2[ii,start_bin:end_bin]
            c3 = comp3[ii,start_bin:end_bin]
            # normalize to make sure max is 1 but keep the ratio
            nor_val = max(max(np.abs(c1)), max(np.abs(c2)), max(np.abs(c3)))
            new_batch[ii,:,0]=c1/nor_val
            new_batch[ii,:,1]=c2/nor_val
            new_batch[ii,:,2]=c3/nor_val
            new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
        # does feature log, add additional channel for sign of log(abs(x))
        new_batch_sign=np.sign(new_batch)
        new_batch_val=np.log(np.abs(new_batch)+epsilon)
        batch_out=[]
        for ii in range(new_batch_target.shape[0]):
            batch_out.append(np.hstack( [new_batch_val[ii,:,0].reshape(-1,1), new_batch_sign[ii,:,0].reshape(-1,1),
                                          new_batch_val[ii,:,1].reshape(-1,1), new_batch_sign[ii,:,1].reshape(-1,1),
                                          new_batch_val[ii,:,2].reshape(-1,1), new_batch_sign[ii,:,2].reshape(-1,1)] ) )
        batch_out=np.array(batch_out)
        return batch_out,new_batch_target


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

def inp2ZEN(data_inp,epsilon):
    # reverse the input data (i.e. from the ZEN2inp function) to ZEN time series
    Z = (np.exp(data_inp[:,0])-epsilon) * data_inp[:,1]
    E = (np.exp(data_inp[:,2])-epsilon) * data_inp[:,3]
    N = (np.exp(data_inp[:,4])-epsilon) * data_inp[:,5]
    return Z,E,N
    
def data_process(filePath,sampl=100):
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
    import os
    import obspy
    if not os.path.exists(filePath):
        return None
    try:
        D = obspy.read(filePath)
    except:
        return None #unable to read file either an empty or broken file
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
        DD.interpolate(sampling_rate=sr, starttime=t1,method='linear')
        DD.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
    else:
        DD = Data1+Data2
        DD.merge(method=1,interpolation_samples=-1,fill_value='interpolate')
        DD.trim(starttime=t1-1, endtime=t2+1, nearest_sample=True, pad=True, fill_value=0)
        DD.interpolate(sampling_rate=sr, starttime=t1,method='linear')
        DD.trim(starttime=t1, endtime=t2, nearest_sample=True, pad=True, fill_value=0)
    #assert len(DD[0].data)==1501, "cut data not exactly 1500 points"
    return DD[0].data#[:1500]



def make_confusion_plot(save_fig=None):
    '''
        To plot example of TP,FN,FP,TN
        only use the function when you have TP_idx, FN_idx...., X and y loaded.
    '''
    fig = plt.figure()
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    T = np.arange(1500)/100 # to second
    plt.subplot(2,2,1) #for TP
    tmpid = np.random.choice(TP_idx)
    tmp_Z, tmp_E, tmp_N = inp2ZEN(X[tmpid],epsilon)
    plt.plot(T,tmp_Z+1.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.plot(T,tmp_E+2.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.plot(T,tmp_N+3.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.text(0.5,1.4,'Z',fontsize=12,)
    plt.text(0.5,2.4,'E',fontsize=12,)
    plt.text(0.5,3.4,'N',fontsize=12,)
    plt.text(7.3,3.3,'TP',fontsize=14,color=[1,0,0],bbox=props)
    h1 = plt.plot(T,y[tmpid],'k',linewidth=1.5)
    h2 = plt.plot(T,y_pred[tmpid],'b',linewidth=1.5)
    plt.legend((h1[0],h2[0]),('True','Model'),loc=1)
    plt.xlim([0,15]);plt.ylim([-0.2,3.8])
    ax = plt.gca()
    ax.tick_params(labelbottom=False)
    ax.set_yticks([0,1])
    plt.subplot(2,2,2) #for FN
    tmpid = np.random.choice(FN_idx)
    tmp_Z, tmp_E, tmp_N = inp2ZEN(X[tmpid],epsilon)
    plt.plot(T,tmp_Z+1.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.plot(T,tmp_E+2.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.plot(T,tmp_N+3.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.text(7.3,3.3,'FN',fontsize=14,color=[1,0,0],bbox=props)
    plt.plot(T,y[tmpid],'k',linewidth=1.5)
    plt.plot(T,y_pred[tmpid],'b',linewidth=1.5)
    plt.xlim([0,15]);plt.ylim([-0.2,3.8])
    ax = plt.gca()
    ax.tick_params(labelbottom=False,labelleft=False)
    ax.set_yticks([0,1])
    plt.subplot(2,2,3) #for FP
    tmpid = np.random.choice(FP_idx)
    tmp_Z, tmp_E, tmp_N = inp2ZEN(X[tmpid],epsilon)
    plt.plot(T,tmp_Z+1.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.plot(T,tmp_E+2.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.plot(T,tmp_N+3.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.text(7.3,3.3,'FP',fontsize=14,color=[1,0,0],bbox=props)
    plt.plot(T,y[tmpid],'k',linewidth=1.5)
    plt.plot(T,y_pred[tmpid],'b',linewidth=1.5)
    plt.xlim([0,15]);plt.ylim([-0.2,3.8])
    plt.xlabel('Time (s)',fontsize=14,labelpad=0)
    ax = plt.gca()
    ax.set_yticks([0,1])
    plt.subplot(2,2,4) #for TN
    tmpid = np.random.choice(TN_idx)
    tmp_Z, tmp_E, tmp_N = inp2ZEN(X[tmpid],epsilon)
    plt.plot(T,tmp_Z+1.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.plot(T,tmp_E+2.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.plot(T,tmp_N+3.2,'-',color=[0.7,0.7,0.7],linewidth=0.5)
    plt.text(7.3,3.3,'TN',fontsize=14,color=[1,0,0],bbox=props)
    plt.plot(T,y[tmpid],'k',linewidth=1.5)
    plt.plot(T,y_pred[tmpid],'b',linewidth=1.5)
    plt.xlim([0,15]);plt.ylim([-0.2,3.8])
    plt.xlabel('Time (s)',fontsize=14,labelpad=0)
    ax = plt.gca()
    ax.set_yticks([0,1])
    ax.tick_params(labelleft=False)
    plt.subplots_adjust(left=0.05,top=0.95,right=0.97,bottom=0.1,wspace=0.14,hspace=0.12)
    if save_fig:
        plt.savefig(save_fig,dpi=300)
        plt.show()
    else:
        plt.show()



def get_catalog(catalog_file: str, EQinfo_file: str) -> pd.DataFrame:
    """
    Convert catalog file to pandas format with columns =
    ['famID','mag','catOT','OT']
        famID: family ID e.g. 001
        mag: magnitude of LFE
        catOT: catalog origintime. This is the time of when template start
        OT: real OT. The real origin time after correction
    
    INPUT:
    ======
    catalog_file: str
        catalog file name
    EQinfo_file: str
        EQ info file name. The file save LFE family info
        
    OUTPUT:
    =======
    res: pd.DataFrame
        pandas format
    """
    EQinfo = np.load(EQinfo_file,allow_pickle=True)
    EQinfo = EQinfo.item()
    
    head = ['famID','lon','lat','dep','mag','catOT','OT']
    sav_all = []
    with open(catalog_file,'r') as IN1:
        for line in IN1.readlines():
            line = line.strip()
            ID = line.split()[0] #family ID
            OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
            HH = (int(line.split()[2])-1)*3600  #HR from 1-24
            SS = float(line.split()[3])
            OT = OT + HH + SS  #detected origin time. always remember!!! this is not the real OT. The shifted time in the sav_family_phases.npy have been corrected accordingly.
            real_OT = OT + EQinfo[ID]['catShift']  # set to real OT, not template starttime
            Mag = float(line.split()[4])
            # get EQINFO from EQinfo
            evlo = -EQinfo[ID]['eqLoc'][0]
            evla = EQinfo[ID]['eqLoc'][1]
            evdp = EQinfo[ID]['eqLoc'][2]
            sav_all.append([ID,evlo,evla,evdp,Mag,OT.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-2],real_OT.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-2]])
    df = pd.DataFrame(sav_all, columns = head)
    return df


# Model and data to be tested
#--------------------------------------------------------
catalog_file = "total_mag_detect_0000_cull_NEW.txt"
EQinfo_file = "sav_family_phases.npy"
drop = 0
sr = 100
wd = 30 # 30sec of data. No longer needed
epsilon=1e-6
large = 0.5
std = 0.4
run_num = 'S001'
x_data_file = "Data_QC_rmean_norm/merged20220602_S.h5"
csv_file = "Data_QC_rmean_norm/merged20220602_S.csv"
n_data_file = "Cascadia_noise_QC_rmean_norm.h5"

# load station info
stainfo=np.load('stainfo.npy',allow_pickle=True)
stainfo = stainfo.item()

# load EQinfo
EQinfo = np.load(EQinfo_file,allow_pickle=True)
EQinfo = EQinfo.item()

# test parameters
min_stations = 5 # at least 5 station record available to be tested
min_mag = 2.0 # only test with large magnitude
timeL = 0
timeR = 15 # window length needs to match the model input
#time_skip = 30 # skip the first 30 s
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
#---------------------------------------------------------

# loading catalog file to pandas
catalog = get_catalog(catalog_file, EQinfo_file)

# loading testing index
sig_test_inds = np.load("./Index/index_%s/sig_test_inds.npy"%(run_num))
noise_test_inds = np.load("./Index/index_%s/noise_test_inds.npy"%(run_num))

#loading h5py data and csv file
x_data = h5py.File(x_data_file,'r')
csv = pd.read_csv(csv_file)
n_data = h5py.File(n_data_file,'r')

'''
To get data
print(csv.iloc[0].evID) # all the available evID, e.g. 2005-09-05T19:21:37.6250_001.CN.LZB.BH.S
x_data['waves/'+'2005-09-05T19:21:37.6250_001.CN.LZB.BH.S'] # shape=(9003,) sorted in ZEN, 100 Hz., +-15s centered on arrival.
'''

# load model
if drop:
    model=unet_tools.make_large_unet_drop(large,sr,ncomps=3)
else:
    model=unet_tools.make_large_unet(large,sr,ncomps=3)

chks = glob.glob("./checks/large*%s*"%(run_num))
chks.sort()
model.load_weights(chks[-1].replace('.index',''))


# Part 1. Test the model with known LFE data
n_test = 0
for idx in sig_test_inds:
#for idx in sig_train_inds:
    evIDs = csv[csv['idxCatalog']==idx].evID
    if len(evIDs)<min_stations:
        continue
    if catalog.iloc[idx].mag<min_mag:
        continue
    real_OT = catalog.iloc[idx].OT
    print('processing:',evIDs)
    real_OT = UTCDateTime(catalog.iloc[idx].OT) #+ 999999
    #real_OT = np.random.choice(sav_randOT)
    #org_real_OT = UTCDateTime(catalog.iloc[idx].OT)
    cat_OT = UTCDateTime(catalog.iloc[idx].catOT)
    famID = catalog.iloc[idx].famID
    # source loc
    evlo = catalog.iloc[idx].lon
    evla = catalog.iloc[idx].lat
    evdp = catalog.iloc[idx].dep
    if real_OT.date!=(real_OT+300).date:
        continue # cross days, skip to save time
    plt.close()
    for n,ev in enumerate(evIDs):
        t1 = real_OT - timeL
        t2 = real_OT + timeR
        sta = ev.split('_')[-1].split('.')[2]
        stlo = stainfo[sta][0]
        stla = stainfo[sta][1]
        # calculate distance
        st_dist = obspy.geodetics.locations2degrees(lat1=evla,long1=evlo,lat2=stla,long2=stlo)
        #S = vel_mod.get_travel_times(source_depth_in_km=evdp, distance_in_degree=st_dist, phase_list=('S','s'), receiver_depth_in_km=0)
        print('  distance=',st_dist)
        #arr = cat_OT - EQinfo[famID]['sta'][sta]['P2']  - real_OT # theorical arrival to be marked, for S wave
        travel = EQinfo[famID]['sta'][sta]['P2'] - EQinfo[famID]['catShift']
        if travel>15:
            #pass # do everything even the arrival is outside of the 15s window
            continue
        #some data are in cascadia_CN
        if sta in CN_list:
            dataDir = dir2
            net = net2
            comp = CN_list[sta]
        else:
            dataDir = dir1
            net = net1
            comp = 'HH' #PO stations are always HH
        t1_fileZ = dataDir+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'Z.mseed'
        t1_fileE = dataDir+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'E.mseed'
        t1_fileN = dataDir+'/'+t1.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'N.mseed'
        t2_fileZ = dataDir+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'Z.mseed'
        t2_fileE = dataDir+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'E.mseed'
        t2_fileN = dataDir+'/'+t2.strftime('%Y%m%d')+'.'+net+'.'+sta+'..'+comp+'N.mseed'
        # read data from hard-drive, process them and cut to 15s window
        print('  reading data :',t1_fileZ)
        Z = data_process(filePath=t1_fileZ,sampl=sr)
        E = data_process(filePath=t1_fileE,sampl=sr)
        N = data_process(filePath=t1_fileN,sampl=sr)
        # don't need this!
        if Z is None or E is None or N is None:
            continue
        # cut date
        Z = data_cut(Z,t1=t1,t2=t2)[:-1]
        E = data_cut(E,t1=t1,t2=t2)[:-1]
        N = data_cut(N,t1=t1,t2=t2)[:-1]
        nor = max(max(np.abs(Z)),max(np.abs(E)),max(np.abs(N)))
        Z = Z/nor
        E = E/nor
        N = N/nor
        t = np.arange(len(Z))*(1/sr)
        # convert ZEN to model input
        X = ZEN2inp(Z,E,N,epsilon)
        y_pred = model.predict(X.reshape(1,X.shape[0],X.shape[1]))
        #print('    travelT=',travel,S[0].time,t[y_pred[0].argmax()],max(y_pred[0]))
        '''
        plt.plot(t,10*E+st_dist*111,color=[0.8,0.8,0.8],linewidth=0.8)
        plt.plot(travel,st_dist*111,'r*',markersize=12)
        if max(y_pred[0])>0.1:
            plt.plot(t[y_pred[0].argmax()],st_dist*111,'bo',markersize=8)
        plt.plot(t,10*y_pred[0]+st_dist*111,'b',linewidth=0.8)
        plt.plot([0,15],[st_dist*111,st_dist*111],'k--',linewidth=0.3)
        '''
        plt.plot(t,E+n,color=[0.8,0.8,0.8],linewidth=0.8)
        plt.plot(t,y_pred[0]+n)
    plt.title('OT:%s M=%.1f'%(real_OT.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-2],catalog.iloc[idx].mag),fontsize=14)
    plt.xlim([0,15])
    plt.xlabel('Origin Time (s)',fontsize=14)
    #plt.xlabel('Origin:%s'%(org_real_OT.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-2]),fontsize=14)
    plt.ylabel('Hypo Dist (km)',fontsize=14)
    plt.savefig('testplot_%03d.png'%(n_test))
    plt.close()
    n_test += 1
    if n_test==10:
        break




# Part 2. Test the model with noise data
n_test = 0
n_per_plot = 10
sav_ymax = []
for idx in noise_test_inds:
    rnd_start = np.random.choice(range(1500))
    Z = n_data['waves'][idx][:3001]
    E = n_data['waves'][idx][3001:6002]
    N = n_data['waves'][idx][6002:]
    # cut to a 15s timeseries starting randomly
    Z = Z[rnd_start:rnd_start+1500]
    E = E[rnd_start:rnd_start+1500]
    N = N[rnd_start:rnd_start+1500]
    nor = max(max(np.abs(Z)),max(np.abs(E)),max(np.abs(N)))
    Z = Z/nor
    E = E/nor
    N = N/nor
    t = np.arange(len(Z))*(1/sr)
    if n_per_plot==10:
        n_test += 1
        plt.xlim([0,15])
        plt.xlabel('Time (s)',fontsize=14)
        plt.savefig('testplot_%03d.png'%(n_test))
        plt.close() #reset the plot
        n_per_plot = 0
        plt.figure()
    # convert ZEN to model input
    X = ZEN2inp(Z,E,N,epsilon)
    y_pred = model.predict(X.reshape(1,X.shape[0],X.shape[1]))
    print('=====================')
    print('  ymax=',y_pred[0].max())
    sav_ymax.append(y_pred[0].max())
    print('=====================')
    plt.plot(t,E+n_per_plot,color=[0.8,0.8,0.8],linewidth=0.8)
    if max(y_pred[0])>0.1:
        plt.plot(t[y_pred[0].argmax()],n_per_plot,'bo',markersize=8)
    plt.plot(t,y_pred[0]+n_per_plot,'b',linewidth=0.8)
    plt.plot([0,15],[n_per_plot,n_per_plot],'k--',linewidth=0.3)
    plt.close()
    n_per_plot += 1
    if n_test==50:
        break




import sys
sys.exit()

'''
# pull out the data by index
i = 3
#for this event get all data
sav_X = []
for n,ev in enumerate(csv[csv['idxCatalog']==sig_test_inds[i]].evID):
    ev
    data = np.array(x_data['waves/%s'%(ev)])
    Z = data[:wd*sr+1][750:2250]
    E = data[wd*sr+1:(wd*sr+1)*2][750:2250]
    N = data[(wd*sr+1)*2:][750:2250]
    nor = max(max(np.abs(Z)),max(np.abs(E)),max(np.abs(N)))
    Z = Z/nor
    E = E/nor
    N = N/nor
    #plt.plot(E+n,'k')
    #plt.plot()
    # only use center 15 s
    X = ZEN2inp(Z,E,N,epsilon)
    #y_pred = model.predict(X.reshape(1,1500,6))
    #plt.plot(y_pred[0]+n,'r')

plt.savefig('pred_test.png')
plt.close()

sav_X = np.array(sav_X)
y_pred = model.predict(sav_X)
'''

x_data.close()
n_data.close()

# generate data from indexes
my_data = data_generator(32,x_data_file,n_data_file,sig_test_inds,noise_test_inds,csv_file,sr,std)


#models = ['large_0.5_unet_lfe_std_0.4.tf.008','large_1.0_unet_lfe_std_0.4.tf.009','large_0.5_unet_lfe_std_0.4.tf.010','large_1.0_unet_lfe_std_0.4.tf.011']
#texts = ['large0.5 CC0.1','large1 CC0.1','large0.5 CC0.2','large1 CC0.2']
#larges = [0.5,1,0.5,1]


idx_path = '/projects/amt/jiunting/Cascadia_LFE/Index'
#models = ['large_0.5_unet_lfe_std_0.4.tf.012','large_1.0_unet_lfe_std_0.4.tf.013','large_2.0_unet_lfe_std_0.4.tf.014','large_4.0_unet_lfe_std_0.4.tf.015']
#models = ['large_0.5_unet_lfe_std_0.4.tf.01','large_1.0_unet_lfe_std_0.4.tf.02','large_2.0_unet_lfe_std_0.4.tf.03','large_4.0_unet_lfe_std_0.4.tf.04']


# normalized training data
models = ['large_0.5_unet_lfe_std_0.4.tf.001','large_1.0_unet_lfe_std_0.4.tf.002']
#models = ['large_0.5_unet_lfe_std_0.4.tf.005','large_1.0_unet_lfe_std_0.4.tf.006'] #P wave model
#models = ['large_0.5_unet_lfe_std_0.4.tf.01','large_1.0_unet_lfe_std_0.4.tf.02','large_2.0_unet_lfe_std_0.4.tf.03','large_4.0_unet_lfe_std_0.4.tf.04' ]
texts = ['size 0.5','size 1','size 2','size 4']
larges = [0.5,1,2,4]
colors = [[0,0.8,0],[1,0,1],[0.8,0,0],[0,0,1]]

#x_data = h5py.File('Cascadia_lfe_QC_rmean.h5', 'r')
#n_data = h5py.File('Cascadia_noise_QC_rmean.h5', 'r')

x_data = h5py.File('Cascadia_lfe_QC_rmean_norm.h5', 'r') # for S wave
#x_data = h5py.File('Cascadia_lfe_QC_rmean_norm_P.h5', 'r') # for P wave
n_data = h5py.File('Cascadia_noise_QC_rmean_norm.h5', 'r') #for noise

#X = np.load('Xtest_large.npy')
#y = np.load('ytest_large.npy')

#X = np.load('Xtest_NotCC0.1.npy')
#y = np.load('ytest_NotCC0.1.npy')

threshs = np.arange(0,1,0.01) # testing thresholds


#plot training/validation loss
# looping models
fig = plt.figure()
for i,model_path in enumerate(models):
    # load model architecture
    if drop:
        model = unet_tools.make_large_unet_drop(larges[i],sr,ncomps=3)
    else:
        model = unet_tools.make_large_unet(larges[i],sr,ncomps=3)

    # load loss
    loss = np.genfromtxt(model_path+'.csv',delimiter=',',skip_header=1)
    plt.plot(loss[:,0]+1,loss[:,2],'o-',color=colors[i])
    plt.plot(loss[:,0]+1,loss[:,4],'^-',color=colors[i])

#plt.grid(True)
plt.xlim([0.5,50.5])
plt.xlabel('Epoch',fontsize=14)
plt.ylabel('Loss',fontsize=14)
#add a subplot for legend
ax1 = fig.add_axes((0.48,0.65,0.4,0.2),fc=[1,1,1])
plt.plot(3,-2.2,'o',markersize=10,markeredgecolor=[0,0,0],markerfacecolor=[1,1,1])
plt.text(3.3,-2.3,'training loss')
plt.plot(3,-2.8,'^',markersize=10,markeredgecolor=[0,0,0],markerfacecolor=[1,1,1])
plt.text(3.3,-2.9,'validation loss')
#colors
plt.plot([0,0.8],[-2,-2],color=colors[0])
plt.text(1,-2.1,'Size 0.5')
plt.plot([0,0.8],[-2.5,-2.5],color=colors[1])
plt.text(1,-2.6,'Size 1')
plt.plot([0,0.8],[-3,-3],color=colors[2])
plt.text(1,-3.1,'Size 2')
plt.plot([0,0.8],[-3.5,-3.5],color=colors[3])
plt.text(1,-3.6,'Size 4')
plt.xlim([-0.3,6])
plt.ylim([-3.8,-1.7])
plt.xticks([],[])
plt.yticks([],[])


plt.savefig('testlosses.png')
#plt.savefig('testlosses_P.png')
plt.close()

#import sys
#sys.exit()


print('==========start loading model=============')

#models = ['./checks/large_2.0_unet_lfe_std_0.4.tf.03_0024.ckpt']
models = ['large_0.5_unet_lfe_std_0.4.tf.001','large_1.0_unet_lfe_std_0.4.tf.002']  #S wave
run_nums = ['001','002']
#models = ['large_0.5_unet_lfe_std_0.4.tf.005','large_1.0_unet_lfe_std_0.4.tf.006']  #P wave
#run_nums = ['005','006']
larges = [0.5,1]
stds = [0.4] #std width 

# looping models
for i,model_path in enumerate(models):
    if i==0:
        continue
    # load model architecture
    if drop:
        model = unet_tools.make_large_unet_drop(larges[i],sr,ncomps=3)
    else:
        model = unet_tools.make_large_unet(larges[i],sr,ncomps=3)
    # load weights
    model.load_weights(model_path)

    # get run id number from file name (final epoch/model)
    #run_num = model_path.split('.')[-1]
    run_num = run_nums[i]
    #sig_valid_inds = np.load(idx_path+'/index_'+run_num+'/sig_valid_inds.npy')
    #noise_valid_inds = np.load(idx_path+'/index_'+run_num+'/noise_valid_inds.npy')
    sig_test_inds = np.load(idx_path+'/index_'+run_num+'/sig_test_inds.npy')
    noise_test_inds = np.load(idx_path+'/index_'+run_num+'/noise_test_inds.npy')

    # uncomment following to also include validation data
    #sig_test_inds = np.hstack([sig_test_inds,sig_valid_inds])
    #noise_test_inds = np.hstack([noise_test_inds,noise_valid_inds])

    '''
    # ========get testing data, this can be generated previously=========
    #my_data = my_data_generator(10000,x_data,n_data,sig_test_inds,noise_test_inds,sr,std,valid=False)  #when valid=True, want all the data, batch_size is automatically
    my_data = my_data_generator(100000,x_data,n_data,sig_test_inds,noise_test_inds,sr,std,valid=False)  #when valid=True, want all the data, batch_size is automatically
    #my_data = my_data_generator(1,x_data,n_data,sig_test_inds,noise_test_inds,sr,std,valid=True)  #when valid=True, want all the data, batch_size is automatically
    # "i'll stop here"
    X,y = next(my_data)
    '''
    #np.save('X.npy',X)
    #np.save('y_all_valid.npy',y)

    # ========save the test data for faster loading next time========
    #np.save('X_test_%s.npy'%(run_num),X)
    #np.save('y_test_%s.npy'%(run_num),y)
    X = np.load('X_test_%s.npy'%(run_num))
    y = np.load('y_test_%s.npy'%(run_num))
    
    # model predictions
    y_pred = model.predict(X)

    # turn 2D labels into 1D labels. i.e. LFE or noise
    y_true = np.where(np.max(y,axis=1) == 1, 1, 0)
    
    # testing different thresh
    sav_TPR = []
    sav_FPR = []
    sav_prec = []
    sav_acc = []
    sav_thres = []
    for thresh in threshs:
        thresh = 0.1
        # turn 2D prediction to 1D labels LFE or noise depending on threshold
        #print('testing thres=%f'%(thresh))
        y_pred_TF = np.where(np.max(y_pred,axis=1) >= thresh, 1, 0)
        TP = np.sum((y_true==1) & (y_pred_TF==1))
        TN = np.sum((y_true==0) & (y_pred_TF==0))
        FP = np.sum((y_true==0) & (y_pred_TF==1))
        FN = np.sum((y_true==1) & (y_pred_TF==0))
        #print(' summing of all=%d'%(TP+TN+FP+FN))
        #print('TP,FN=',TP,FN)
        #print('FP,TN=',FP,TN)
        #print('TP,FP=',TP,FP)
        try:
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            prec = TP / (TP + FP)
            acc = (TP + TN) / (TP + TN + FP + FN)
        except:
            continue
        sav_TPR.append(TPR)
        sav_FPR.append(FPR)
        sav_prec.append(prec)
        sav_acc.append(acc)
        sav_thres.append(thresh)
        #----------make some plots------------
        TP_idx = np.where((y_true==1) & (y_pred_TF==1))[0]
        TN_idx = np.where((y_true==0) & (y_pred_TF==0))[0]
        FP_idx = np.where((y_true==0) & (y_pred_TF==1))[0]
        FN_idx = np.where((y_true==1) & (y_pred_TF==0))[0]
        for nfig in range(50):
            save_fig = './Figs_confusion/fig_%03d.png'%(nfig)
            print('Start making figure:',nfig,save_fig)
            make_confusion_plot(save_fig=save_fig)
        import sys
        sys.exit()
        #-----------plot end------------------


    # calculate AUC, area under ROC curve
    AUC = np.abs(np.trapz(sav_TPR,sav_FPR)) #FPR starts from right to left making integration negative
     
    # make figure
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.plot(sav_thres,np.array(sav_prec)*100,color='tab:blue',label='Precision')
    plt.plot(sav_thres,np.array(sav_TPR)*100,color='tab:red',label='Recall')
    plt.plot(sav_thres,np.array(sav_acc)*100,color='k',label='Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('threshold',fontsize=14,labelpad=0)
    plt.ylabel('percentage',fontsize=14,labelpad=-1)
    plt.subplot(1,2,2)
    plt.plot(sav_FPR,sav_TPR,color=[0,0,0])
    plt.scatter(sav_FPR,sav_TPR,c=threshs)
    plt.xlabel('FPR',fontsize=14,labelpad=0)
    plt.ylabel('TPR',fontsize=14,labelpad=-1)
    plt.title('AUC=%.2f'%(AUC))
    #plt.text(0.5,0.18,texts[i])
    plt.grid(True)
    plt.subplots_adjust(left=0.095,top=0.88,right=0.98,bottom=0.1,wspace=0.25,hspace=0.1)
    # add colorbar
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = matplotlib.cm.ScalarMappable(norm=norm)
    cmap.set_array([])
    cbaxes = fig.add_axes([0.8, 0.18, 0.1, 0.02])
    clb = plt.colorbar(cmap,cax=cbaxes, orientation='horizontal',label='threshold')
    clb.set_label('threshold',rotation=0,labelpad=-1)
    #plt.savefig('model_%02d.png'%(i))
    plt.savefig('model_run%s.png'%(run_num))
    plt.close()


    #========Also consider prediction timing, not just the whole trace==============
    # testing different thresh
    sav_TPR2 = []
    sav_FPR2 = []
    sav_prec2 = []
    sav_acc2 = []
    sav_thres2 = []
    for thresh in threshs:
        # turn 2D prediction to 1D labels LFE or noise depending on threshold
        # check the arrival and value
        tmp_TP = 0
        tmp_TN = 0
        tmp_FP = 0
        tmp_FN = 0
        for n in range(y.shape[0]):
            #y[n] v.s. y_pred[n]
            #tmpidx_y = np.where(y[n]>=thresh)[0]
            tmpidx_y = np.where(y[n]==1)[0] # true arrival time is exactly at where y=1
            tmpidx_y_pred = np.where(y_pred[n]>=thresh)[0]
            if len(tmpidx_y)==0:
                # this is just noise
                if len(tmpidx_y_pred)==0:
                    # correct prediction: noise v.s. noise
                    tmp_TN += 1
                else:
                    # false positive
                    tmp_FP += 1
            else:
                # this is actual LFE signal, should also consider travel time
                assert len(tmpidx_y)==1, "1 appears more than once?"
                if len(tmpidx_y_pred)==0:
                    # false negative
                    tmp_FN += 1
                else:
                    # signal v.s. signal, should also consider travel time
                    #tmpidx_y_max = np.where(y[n]==np.max(y[n]))[0][0]
                    tmpidx_y_max = tmpidx_y[0]
                    tmpidx_y_pred_max = np.where(y_pred[n]==np.max(y_pred[n]))[0][0] #assuming that each 15 s trace has only one LFE data!!
                    #if np.abs(tmpidx_y_max-tmpidx_y_pred_max)<stds[i]*sr:
                    if np.abs(tmpidx_y_max-tmpidx_y_pred_max)<=2.0*sr:
                        tmp_TP += 1
                    else:
                        #tmp_FN += 1 # although it detect the LFE, but it's not at the right arrival time
                        tmp_FP += 1 # although it detect the LFE, but it's not at the right arrival time (detect LFE at the part where should be just noise)

        assert (tmp_TP + tmp_TN + tmp_FP + tmp_FN) == y.shape[0], "length different! something wrong"
        try:
            # calculate the rates
            TPR = tmp_TP / (tmp_TP + tmp_FN)
            FPR = tmp_FP / (tmp_FP + tmp_TN)
            prec = tmp_TP / (tmp_TP + tmp_FP)
            acc = (tmp_TP + tmp_TN) / (tmp_TP + tmp_TN + tmp_FP + tmp_FN)
        except:
            continue
        sav_TPR2.append(TPR)
        sav_FPR2.append(FPR)
        sav_prec2.append(prec)
        sav_acc2.append(acc)
        sav_thres2.append(thresh)

    # calculate AUC, area under ROC curve
    AUC = np.abs(np.trapz(sav_TPR2,sav_FPR2)) #FPR starts from right to left making integration negative

    # make figure
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.plot(sav_thres2,np.array(sav_prec2)*100,color='tab:blue',label='Precision')
    plt.plot(sav_thres2,np.array(sav_TPR2)*100,color='tab:red',label='Recall')
    plt.plot(sav_thres2,np.array(sav_acc2)*100,color='k',label='Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('threshold',fontsize=14,labelpad=0)
    plt.ylabel('percentage',fontsize=14,labelpad=-1)
    plt.subplot(1,2,2)
    plt.plot(sav_FPR2,sav_TPR2,color=[0,0,0])
    plt.scatter(sav_FPR2,sav_TPR2,c=sav_thres2)
    plt.xlabel('FPR',fontsize=14,labelpad=0)
    plt.ylabel('TPR',fontsize=14,labelpad=-1)
    plt.title('AUC=%.2f'%(AUC))
    #plt.text(0.5,0.18,texts[i])
    plt.grid(True)
    plt.subplots_adjust(left=0.095,top=0.88,right=0.98,bottom=0.1,wspace=0.25,hspace=0.1)
    # add colorbar
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = matplotlib.cm.ScalarMappable(norm=norm)
    cmap.set_array([])
    cbaxes = fig.add_axes([0.8, 0.18, 0.1, 0.02])
    clb = plt.colorbar(cmap,cax=cbaxes, orientation='horizontal',label='threshold')
    clb.set_label('threshold',rotation=0,labelpad=-1)
    #plt.savefig('model_%02d.png'%(i))
    plt.savefig('model_run%s_time.png'%(run_num))
    plt.close()



print("COMPUTATION FINISH")

import sys
sys.exit()


# LOAD THE DATA
print("LOADING DATA")
#n_data = h5py.File('Cascadia_noise_data_norm.h5', 'r')
#x_data = h5py.File('Cascadia_lfe_data_norm.h5', 'r')
#n_data = h5py.File('Cascadia_noise_QC_rmean.h5', 'r')
#x_data = h5py.File('Cascadia_lfe_QC_rmean.h5', 'r')

#test4
#n_data = h5py.File('Cascadia_noise_QC_rmean.h5', 'r')
#x_data = h5py.File('Cascadia_lfe_QC_rmean_CC0.1.h5', 'r')

#test5
#n_data = h5py.File('Cascadia_noise_QC_rmean.h5', 'r')
#x_data = h5py.File('Cascadia_lfe_QC_rmean_CC0.2.h5', 'r')


#test6 #make sure data and noise have the same traces
#n_data = h5py.File('Cascadia_noise_QC_rmean.h5', 'r')
#x_data = h5py.File('Cascadia_lfe_QC_rmean_CC0.1.h5', 'r')

#test7 #make sure data and noise have the same traces
#n_data = h5py.File('Cascadia_noise_QC_rmean.h5', 'r')
#x_data = h5py.File('Cascadia_lfe_QC_rmean_CC0.2.h5', 'r')

#test8,9 #make sure data and noise have the same traces
#n_data = h5py.File('Cascadia_noise_QC_rmean.h5', 'r')
#x_data = h5py.File('Cascadia_lfe_QC_rmean_CC0.1.h5', 'r')

#test10,11 #make sure data and noise have the same traces
#n_data = h5py.File('Cascadia_noise_QC_rmean.h5', 'r')
#x_data = h5py.File('Cascadia_lfe_QC_rmean_CC0.2.h5', 'r')

#make some validation dataset for the model with only trained CC0.1 or 0.2
n_data = h5py.File('Cascadia_noise_QC_rmean.h5', 'r')
x_data = h5py.File('Cascadia_lfe_QC_rmean_NotCC0.1.h5', 'r')

#model_save_file="large_"+str(large)+"_unet_lfe_std_"+str(std)+".tf."+run_num  
        
#if drop:
#    model_save_file="drop_"+model_save_file
    
# MAKE TRAINING AND TESTING DATA
print("MAKE TRAINING AND TESTING DATA")
np.random.seed(0)
siginds=np.arange(x_data['waves'].shape[0])
noiseinds=np.arange(n_data['waves'].shape[0])
np.random.shuffle(siginds)
np.random.shuffle(noiseinds)

#sig_train_inds=np.sort(siginds[:int(0.75*len(siginds))])
#noise_train_inds=np.sort(noiseinds[:int(0.75*len(noiseinds))])
#sig_test_inds=np.sort(siginds[int(0.75*len(siginds)):])
#noise_test_inds=np.sort(noiseinds[int(0.75*len(noiseinds)):])

# modified 04/02 2021 Tim
Len_siginds = len(siginds) #length of siginds
Len_noiseinds = len(noiseinds) #length of noiseinds
data_len = np.min([Len_siginds,Len_noiseinds]) #equal number of noise/data

#sig_train_inds=np.sort(siginds[:int(0.75*data_len)])
#noise_train_inds=np.sort(noiseinds[:int(0.75*data_len)])
#sig_test_inds=np.sort(siginds[int(0.75*data_len):])
#noise_test_inds=np.sort(noiseinds[int(0.75*data_len):])

sig_test_inds=np.sort(siginds[:int(1*data_len)])
noise_test_inds=np.sort(noiseinds[:int(1*data_len)])



# do the shifts and make batches
print("SETTING UP GENERATOR")
def my_data_generator(batch_size,x_data,n_data,sig_inds,noise_inds,sr,std,valid=False):
    ### if valid=True, want all the data. batch_size is automatically
    while True:
        # randomly select a starting index for the data batch
        start_of_data_batch=np.random.choice(len(sig_inds)-batch_size//2)
        # randomly select a starting index for the noise batch
        start_of_noise_batch=np.random.choice(len(noise_inds)-batch_size//2)
        if valid:
            start_of_noise_batch=0
            start_of_data_batch=0
            #batch_size = 
            # get range of indicies from data
            datainds=sig_inds[:]
            # get range of indicies from noise
            noiseinds=noise_inds[:]
        else:
            # get range of indicies from data
            datainds=sig_inds[start_of_data_batch:start_of_data_batch+batch_size//2]
            # get range of indicies from noise
            noiseinds=noise_inds[start_of_noise_batch:start_of_noise_batch+batch_size//2]
        #length of each component
        nlen=int(sr*30)+1
        # grab batch
        #=====added these lines by Tim=====
        #normalize the data and noise
        #data_all = x_data['waves'][datainds]/np.max(np.abs(x_data['waves'][datainds]),axis=1).reshape(-1,1)
        #noise_all = n_data['waves'][noiseinds]/np.max(np.abs(n_data['waves'][noiseinds]),axis=1).reshape(-1,1)
        #comp1=np.concatenate((data_all[:,:nlen],noise_all[:,:nlen]))
        #comp2=np.concatenate((data_all[:,nlen:2*nlen],noise_all[:,nlen:2*nlen]))
        #comp3=np.concatenate((data_all[:,2*nlen:],noise_all[:,2*nlen:]))
        #==================================
        comp1=np.concatenate((x_data['waves'][datainds,:nlen],n_data['waves'][noiseinds,:nlen]))
        comp2=np.concatenate((x_data['waves'][datainds,nlen:2*nlen],n_data['waves'][noiseinds,nlen:2*nlen]))
        comp3=np.concatenate((x_data['waves'][datainds,2*nlen:],n_data['waves'][noiseinds,2*nlen:]))
        # make target data vector for batch
        target=np.concatenate((np.ones_like(datainds),np.zeros_like(noiseinds)))
        # make structure to hold target functions
        batch_target=np.zeros((batch_size,nlen))
        # shuffle things (not sure if this is needed)
        #inds=np.arange(batch_size)
        inds=np.arange(len(datainds)+len(noiseinds))
        np.random.shuffle(inds)
        comp1=comp1[inds,:]
        comp2=comp2[inds,:]
        comp3=comp3[inds,:]
        target=target[inds]
        # some params
        winsize=15 # winsize in seconds
        # this just makes a nonzero value where the pick is
        for ii, targ in enumerate(target):
            #print(ii,targ)
            if targ==0:
                batch_target[ii,:]=1/nlen*np.ones((1,nlen))   #this is label for noise
            elif targ==1:
                batch_target[ii,:]=signal.gaussian(nlen,std=int(std*sr)) #this is data
        # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        time_offset=np.random.uniform(0,winsize,size=batch_size)
        # initialize arrays to hold shifted data
        new_batch=np.zeros((batch_size,int(winsize*sr),3))
        new_batch_target=np.zeros((batch_size,int(winsize*sr)))        
        # this loop shifts data and targets and stores results
        # for data, arrivals at exactly 15 s, randomly cut the 30 s data from min: 0-15s to max: 15-30s
        # do the same process for target
        for ii,offset in enumerate(time_offset):
            bin_offset=int(offset*sr) #HZ sampling Frequency
            start_bin=bin_offset 
            end_bin=start_bin+int(winsize*sr) 
            new_batch[ii,:,0]=comp1[ii,start_bin:end_bin]
            new_batch[ii,:,1]=comp2[ii,start_bin:end_bin]
            new_batch[ii,:,2]=comp3[ii,start_bin:end_bin]
            new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
        # does feature log, add additional channel for sign of log(abs(x))
        new_batch_sign=np.sign(new_batch)
        new_batch_val=np.log(np.abs(new_batch)+epsilon)
        batch_out=[]
        for ii in range(new_batch_target.shape[0]):
            batch_out.append(np.hstack( [new_batch_val[ii,:,0].reshape(-1,1), new_batch_sign[ii,:,0].reshape(-1,1), 
                                          new_batch_val[ii,:,1].reshape(-1,1), new_batch_sign[ii,:,1].reshape(-1,1),
                                          new_batch_val[ii,:,2].reshape(-1,1), new_batch_sign[ii,:,2].reshape(-1,1)] ) )
        batch_out=np.array(batch_out)
        yield(batch_out,new_batch_target)

# generate batch data
print("FIRST PASS WITH DATA GENERATOR")
#my_data=my_data_generator(32,x_data,n_data,sig_train_inds,noise_train_inds,sr,std)
my_data=my_data_generator(100000,x_data,n_data,sig_test_inds,noise_test_inds,sr,std)  # generate some testing dataset example
x,y=next(my_data)

#np.save('Xtest_large.npy',x) #a large dataset mixed with training/validation data but the amount of training data included is limited
#np.save('ytest_large.npy',y)
np.save('Xtest_NotCC0.1.npy',x) #a large dataset mixed with training/validation data but the amount of training data included is limited
np.save('ytest_NotCC0.1.npy',y)

#np.save('Xtest_%s.npy'%(run_num),x)
#np.save('ytest_%s.npy'%(run_num),y)
import sys 
sys.exit()


# PLOT GENERATOR RESULTS
if plots:
    for ind in range(5):
        fig, (ax0,ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)
        t=1/sr*np.arange(x.shape[1])
        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel('Amplitude', color='tab:red')
        ax0.plot(t, (np.exp(x[ind,:,0])-epsilon)*x[ind,:,1], color='tab:red', label='data')
        ax0.tick_params(axis='y')
        ax0.legend(loc="lower right")
        ax1 = ax0.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.set_ylabel('Prediction', color='black')  # we already handled the x-label with ax1
        ax1.plot(t, y[ind,:], color='black', linestyle='--', label='target')
        ax1.legend(loc="upper right")
        ax2.plot(t, x[ind,:,0], color='tab:green', label='ln(data amp)')
        ax2.plot(t, x[ind,:,1], color='tab:blue', label='data sign')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax2.legend(loc="lower right")
        # plt.show()
        
# BUILD THE MODEL
print("BUILD THE MODEL")
if drop:
    model=unet_tools.make_large_unet_drop(large,sr,ncomps=3)    
else:
    model=unet_tools.make_large_unet(large,sr,ncomps=3)  
        
# ADD SOME CHECKPOINTS
print("ADDING CHECKPOINTS")
checkpoint_filepath = './checks/'+model_save_file+'_{epoch:04d}.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True, verbose=1,
    #monitor='val_acc', mode='max', save_best_only=True)
    monitor='val_accuracy', mode='max', save_best_only=True)

# TRAIN THE MODEL
print("TRAINING!!!")
if train:
    batch_size=32
    if resume:
        print('Resuming training results from '+model_save_file)
        model.load_weights(checkpoint_filepath)
    else:
        print('Training model and saving results to '+model_save_file)
        
    csv_logger = tf.keras.callbacks.CSVLogger(model_save_file+".csv", append=True)
    history=model.fit_generator(my_data_generator(batch_size,x_data,n_data,sig_train_inds,noise_train_inds,sr,std),
                        steps_per_epoch=(len(sig_train_inds)+len(noise_train_inds))//batch_size,
                        validation_data=my_data_generator(batch_size,x_data,n_data,sig_test_inds,noise_test_inds,sr,std),
                        validation_steps=(len(sig_test_inds)+len(noise_test_inds))//batch_size,
                        epochs=epos, callbacks=[model_checkpoint_callback,csv_logger])
    model.save_weights("./"+model_save_file)
else:
    print('Loading training results from '+model_save_file)
    model.load_weights("./result_files/"+model_save_file)
    
# plot the results
if plots:
    # training stats
    training_stats = np.genfromtxt("./result_files/"+model_save_file+'.csv', delimiter=',',skip_header=1)
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(training_stats[:,0],training_stats[:,1])
    ax1.plot(training_stats[:,0],training_stats[:,3])
    ax1.legend(('acc','val_acc'))
    ax2.plot(training_stats[:,0],training_stats[:,2])
    ax2.plot(training_stats[:,0],training_stats[:,4])
    ax2.legend(('loss','val_loss'))
    ax2.set_xlabel('Epoch')
    ax1.set_title(model_save_file)

# See how things went
my_test_data=my_data_generator(20,x_data,n_data,sig_test_inds,noise_test_inds,sr,std,valid=True)
x,y=next(my_test_data)

test_predictions=model.predict(x)

# PLOT A FEW EXAMPLES
if plots:
    for ind in range(15):
        fig, ax1 = plt.subplots()
        t=1/100*np.arange(x.shape[1])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        trace=np.multiply(np.power(x[ind,:,0],10),x[ind,:,1])
        ax1.plot(t, trace, color='tab:red') #, label='data')
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Prediction')  # we already handled the x-label with ax1
        ax2.plot(t, test_predictions[ind,:], color='tab:blue') #, label='prediction')
        ax2.plot(t, y[ind,:], color='black', linestyle='--') #, label='target')
        ax2.tick_params(axis='y')
        ax2.set_ylim((-0.1,2.1))
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(('prediction','target'))
        plt.show()
