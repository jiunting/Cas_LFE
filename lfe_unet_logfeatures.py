#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:54:21 2020

Train a CNN to detect LFEs please

@author: amt

6/2 2022: Train the model with tracking evID, modified by Tim Lin
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import unet_tools
import tensorflow.keras as keras
import tensorflow as tf
import argparse
import pandas as pd

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-subset", "--subset", help="train on a subset or no?", type=int)
parser.add_argument("-pors", "--pors", help="train P or S network", type=int)
parser.add_argument("-train", "--train", help="do you want to train?", type=int)
parser.add_argument("-drop", "--drop", help="want to add a drop layer to the network", type=int)
parser.add_argument("-plots", "--plots", help="want plots", type=int)
parser.add_argument("-resume", "--resume", help="want to resume training?", type=int)
parser.add_argument("-large", "--large", help="what size do you want the network to be?", type=float)
parser.add_argument("-epochs", "--epochs", help="how many epochs", type=int)
parser.add_argument("-std", "--std", help="standard deviation of target", type=float)
parser.add_argument("-sr", "--sr", help="sample rate in hz", type=int)
parser.add_argument("-runid", "--runid", help="running id", type=str)
args = parser.parse_args()

subset=args.subset #True # train on a subset or the full monty?
ponly=args.pors #True # 1 - P+Noise, 2 - S+noise
train=args.train #True # do you want to train?
drop=args.drop #True # drop?
plots=args.plots #False # do you want to make some plots?
resume=args.resume #False # resume training
large=args.large # large unet
epos=args.epochs # how many epocs?
std=args.std # how long do you want the gaussian STD to be?
sr=args.sr
run_num = args.runid

epsilon=1e-6

#run_num = '04'





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

#test 12,13,14,15 back to no CC threshold, but same data/noise traces
#n_data = h5py.File('Cascadia_noise_QC_rmean.h5', 'r')
#x_data = h5py.File('Cascadia_lfe_QC_rmean.h5', 'r')


#=========start new training==========
#test 001~004 same as 12~15, but use normalized data (S waves)
#n_data = h5py.File('Cascadia_noise_QC_rmean_norm.h5', 'r')
#x_data = h5py.File('Cascadia_lfe_QC_rmean_norm.h5', 'r')

#test 005~008 save as 001~004 but P waves
#n_data = h5py.File('Cascadia_noise_QC_rmean_norm.h5', 'r')
#x_data = h5py.File('Cascadia_lfe_QC_rmean_norm_P.h5', 'r')


#========New training on 2022 6========
#test S00X~S00X or P00X~P00X
#x_data = h5py.File('Data_QC_rmean_norm/merged20220602_S.h5', 'r') # for S wave
#csv = pd.read_csv('Data_QC_rmean_norm/merged20220602_S.csv')
#n_data = h5py.File('Cascadia_noise_QC_rmean_norm.h5', 'r') # noise part is the same
x_data_file = "Data_QC_rmean_norm/merged20220602_S.h5"
csv_file = "Data_QC_rmean_norm/merged20220602_S.csv"
n_data_file = "Cascadia_noise_QC_rmean_norm.h5"

csv = pd.read_csv(csv_file)
n_data = h5py.File(n_data_file,'r')

model_save_file="large_"+str(large)+"_unet_lfe_std_"+str(std)+".tf."+run_num  
        
if drop:
    model_save_file="drop_"+model_save_file
    
# MAKE TRAINING AND TESTING DATA
print("MAKE TRAINING AND TESTING DATA")
print(" Partitioning Based on event(6/3 2022)")
np.random.seed(0)
# Signal group by the same event
uniq_events = list(set(csv['idxCatalog']))

#siginds=np.arange(x_data['waves'].shape[0])
noiseinds=np.arange(n_data['waves'].shape[0])

#np.random.shuffle(siginds)
np.random.shuffle(uniq_events)
np.random.shuffle(noiseinds)

#sig_train_inds=np.sort(siginds[:int(0.75*len(siginds))])
#noise_train_inds=np.sort(noiseinds[:int(0.75*len(noiseinds))])
#sig_test_inds=np.sort(siginds[int(0.75*len(siginds)):])
#noise_test_inds=np.sort(noiseinds[int(0.75*len(noiseinds)):])

# 75-25 training and testing split on events. This also suggests roughly same 75-25 splitting on traces
len_siginds = len(uniq_events) #length of unique events
sig_train_inds = uniq_events[:int(len_siginds*0.75)]
sig_test_inds =  uniq_events[int(len_siginds*0.75):]

# number of training signals = events * number of records per event
#len_siginds = sum([sum(csv['idxCatalog']==i) for i in sig_train_inds])# too slow! just estimate
len_siginds = int((len(csv)/len(uniq_events)) * len(sig_train_inds)) # traces per event


# modified 04/02 2021 Tim: make sure data/noise have roughtly same amount
len_noiseinds = len(noiseinds) #length of noiseinds
data_len = np.min([len_siginds,len_noiseinds]) # I always have more data than signal

# noise part
noise_train_inds = np.sort(noiseinds[:int(0.75*data_len)])
noise_test_inds = np.sort(noiseinds[int(0.75*data_len):data_len])


# save training/testing indexes for later analysis
import os
#check file/dir exist,otherwise mkdir
if not(os.path.exists('./Index/index_%s'%(run_num))):
    os.makedirs('./Index/index_%s'%(run_num))

np.save('./Index/index_%s/sig_train_inds.npy'%(run_num),sig_train_inds) # the indexes are event idx from the catalog. one idx can have many traces
np.save('./Index/index_%s/noise_train_inds.npy'%(run_num),noise_train_inds)
#np.save('./Index/index_%s/sig_valid_inds.npy'%(run_num),sig_valid_inds)
#np.save('./Index/index_%s/noise_valid_inds.npy'%(run_num),noise_valid_inds)
np.save('./Index/index_%s/sig_test_inds.npy'%(run_num),sig_test_inds)
np.save('./Index/index_%s/noise_test_inds.npy'%(run_num),noise_test_inds)


# do the shifts and make batches
print("SETTING UP GENERATOR")
#def my_data_generator(batch_size,x_data,n_data,sig_inds,noise_inds,csv,sr,std,valid=False):


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

# generate batch data
print("FIRST PASS WITH DATA GENERATOR")
my_data=data_generator(32,x_data_file,n_data_file,sig_train_inds,noise_train_inds,csv_file,sr,std)
#my_data=my_data_generator(128,x_data,n_data,sig_test_inds,noise_test_inds,sr,std)  # generate some testing dataset example
X,y = my_data.__getitem__(0)
np.save('Xtest_%s.npy'%(run_num),X)
np.save('ytest_%s.npy'%(run_num),y)
#import sys 
#sys.exit()


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
    #monitor='val_accuracy', mode='max', save_best_only=True)
    monitor='val_loss', mode='min', save_best_only=False,period=1)

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
    # create generators
    g_train = data_generator(batch_size,x_data_file,n_data_file,sig_train_inds,noise_train_inds,csv_file,sr,std)
    g_valid = data_generator(batch_size,x_data_file,n_data_file,sig_test_inds,noise_test_inds,csv_file,sr,std)
    history=model.fit_generator(g_train,
                        steps_per_epoch=(2*len(noise_train_inds))//batch_size,
                        validation_data=g_valid,
                        validation_steps=(2*len(noise_test_inds))//batch_size,
                        use_multiprocessing=True,
                        workers=8,
                        epochs=epos,
                        callbacks=[model_checkpoint_callback,csv_logger])
    model.save_weights("./"+model_save_file)
    #np.save('./Test'+run_num+'.npy',history.history) #training losses will also be in .csv file
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
