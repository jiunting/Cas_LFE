#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:56:43 2020

Makes figure 3

@author: amt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
from scipy import signal
import unet_tools
from sklearn.metrics import accuracy_score, precision_score, recall_score

# OPTIONS
subset=0 #True # train on a subset or the full monty?
epos=50 # how many epocs?
epsilon=1e-6
firstflag=1
save=0


for sr in [100]: #, 100]:
    # LOAD THE DATA
    for ponly in [1]: # 1 - P+Noise, 2 - S+noise    
        print("LOADING DATA")
        # file="testing_data_sr_"+str(sr)+"_ponly_"+str(ponly)+".pkl"
        # if save:
        #     if sr==40:
        #         if ponly:
        #             n_data, _ = pickle.load( open( 'pnsn_ncedc_N_training_data.pkl', 'rb' ) )
        #             x_data, _ = pickle.load( open( 'pnsn_ncedc_P_training_data.pkl', 'rb' ) ) 
        #         else:
        #             n_data, _ = pickle.load( open( 'pnsn_ncedc_N_training_data.pkl', 'rb' ) )
        #             x_data, _ = pickle.load( open( 'pnsn_ncedc_S_training_data.pkl', 'rb' ) ) 
        #     elif sr==100:
        #         if ponly:
        #             n_data, _ = pickle.load( open( 'pnsn_ncedc_N_100_training_data.pkl', 'rb' ) )
        #             x_data, _ = pickle.load( open( 'pnsn_ncedc_P_100_training_data.pkl', 'rb' ) ) 
        #         else:
        #             n_data, _ = pickle.load( open( 'pnsn_ncedc_N_100_training_data.pkl', 'rb' ) )
        #             x_data, _ = pickle.load( open( 'pnsn_ncedc_S_100_training_data.pkl', 'rb' ) ) 
            
        #     # MAKE FEATURES AND TARGET VECTOR N=0, P/S=2
        #     print("MAKE FEATURES AND TARGET VECTOR")
        #     features=np.concatenate((n_data,x_data))
        #     target=np.concatenate((np.zeros(n_data.shape[0]),np.ones(x_data.shape[0])))
        #     del n_data
        #     del x_data
            
        #     # MAKE TRAINING AND TESTING DATA
        #     print("MAKE TRAINING AND TESTING DATA")
        #     np.random.seed(0)
        #     inds=np.arange(target.shape[0])
        #     np.random.shuffle(inds)
        #     train_inds=inds[:int(0.75*len(inds))]
        #     test_inds=inds[int(0.75*len(inds)):]
        #     x_test=features[test_inds,:]
        #     y_test=target[test_inds]
        
        #     # Saving the objects:
        #     with open(file, 'wb') as f:  # Python 3: open(..., 'wb')
        #         pickle.dump([x_test, y_test], f)
        # else:
        #     # Getting back the objects:
        #     with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
        #         x_test, y_test = pickle.load(f)
        
        # # do the shifts and make batches
        # print("SETTING UP GENERATOR")
        # def my_data_generator(batch_size,dataset,targets,sr,std,valid=False):
        #     while True:
        #         start_of_batch=np.random.choice(dataset.shape[0]-batch_size)
        #         if valid:
        #             start_of_batch=0
        #         #print('start of batch: '+str(start_of_batch))
        #         # grab batch
        #         batch=dataset[start_of_batch:start_of_batch+batch_size,:]
        #         # make target data for batch
        #         batch_target=np.zeros_like(batch)
        #         # some params
        #         winsize=15 # winsize in seconds
        #         # this just makes a nonzero value where the pick is
        #         # batch_target[:, batch_target.shape[1]//2]=targets[start_of_batch:start_of_batch+batch_size]
        #         for ii, targ in enumerate(targets[start_of_batch:start_of_batch+batch_size]):
        #             #print(ii,targ)
        #             if targ==0:
        #                 batch_target[ii,:]=1/dataset.shape[1]*np.ones((1,dataset.shape[1]))
        #             elif targ==1:
        #                 batch_target[ii,:]=signal.gaussian(dataset.shape[1],std=int(std*sr))
        #         # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        #         time_offset=np.random.uniform(0,winsize,size=batch_size)
        #         new_batch=np.zeros((batch_size,int(winsize*sr)))
        #         new_batch_target=np.zeros((batch_size,int(winsize*sr)))        
        #         for ii,offset in enumerate(time_offset):
        #             bin_offset=int(offset*sr) #HZ sampling Frequency
        #             start_bin=bin_offset 
        #             end_bin=start_bin+int(winsize*sr) # keep 4s worth of samples
        #             new_batch[ii,:]=batch[ii,start_bin:end_bin]
        #             new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
        #         # does feature log
        #         new_batch_sign=np.sign(new_batch)
        #         new_batch_val=np.log(np.abs(new_batch)+epsilon)
        #         batch_out=[]
        #         for ii in range(new_batch_target.shape[0]):
        #             batch_out.append(np.hstack( [new_batch_val[ii,:].reshape(-1,1), new_batch_sign[ii,:].reshape(-1,1) ] ) )
        #         batch_out=np.array(batch_out)
        #         yield(batch_out,new_batch_target)

        
        threshs=np.arange(0.01,1,0.01)
        # prec=np.zeros((12,len(threshs)))
        # reca=np.zeros((12,len(threshs)))
        # accu=np.zeros((12,len(threshs)))
        # firstflag=1
        # count=0
        # for drop in [0, 1]: # True # drop?
        #     for large in [0.5, 1]: # large unet
        #         for std in [0.05, 0.1, 0.2]: #[0.1, 0.2]: # how long do you want the gaussian STD to be?
                    
        #             if ponly==1:
        #                 model_save_file="unet_logfeat_250000_pn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"        
        #             else:
        #                 model_save_file="unet_logfeat_250000_sn_eps_"+str(epos)+"_sr_"+str(sr)+"_std_"+str(std)+".tf"     
            
        #             if large:
        #                 fac=large
        #                 model_save_file="large_"+str(fac)+"_"+model_save_file
                    
        #             if drop:
        #                 model_save_file="drop_"+model_save_file
                    
        #             # BUILD THE MODEL
        #             print("BUILD THE MODEL")
        #             if drop:
        #                 model=unet_tools.make_large_unet_drop(fac,sr)    
        #             else:
        #                 model=unet_tools.make_large_unet(fac,sr)
                    
        #             # LOAD THE MODEL
        #             print('Loading training results from '+model_save_file)
        #             model.load_weights("./result_files/"+model_save_file)  
                    
        #             if firstflag:
        #                 # GET PERFORMANCE STATS
        #                 # this plots accuracy, precision, and recall for each model
        #                 my_test_data=my_data_generator(10000,x_test,y_test,sr,std)
        #                 x,y=next(my_test_data)
        #                 firstflag=0

        #             test_predictions=model.predict(x)
        #             metrics=np.zeros((3,len(threshs)))
        #             for ii,thresh in enumerate(threshs):
        #                 y_true=np.where(np.max(y,axis=1) > thresh, 1, 0)
        #                 y_pred=np.where(np.max(test_predictions,axis=1) > thresh, 1, 0)
        #                 metrics[0,ii]=accuracy_score(y_true,y_pred)
        #                 metrics[1,ii]=precision_score(y_true,y_pred)
        #                 metrics[2,ii]=recall_score(y_true,y_pred)
                    
        #             accu[count,:]=metrics[0,:]
        #             prec[count,:]=metrics[1,:]
        #             reca[count,:]=metrics[2,:]
        #             print(accu)
        #             count+=1
        
file="metrics_sr_"+str(sr)+"_ponly_"+str(ponly)+".pkl"
count=0        
import matplotlib.pylab as pl

colors = pl.cm.rainbow(np.linspace(0,1,4))
colors = [[0.5,0,0],
          [0,0.5,0],
          [0,0,0.5],
          [0.5,0.5,0]]
fig2 = plt.figure(constrained_layout=True,figsize=(10,6))
gs = fig2.add_gridspec(1, 1)
ax1 = fig2.add_subplot(gs[0, 0])
for drop in [0,1]: # True # drop?
    for large in [0.5, 1]: # large unet
        for std in [0.05, 0.1, 0.2]: #[0.1, 0.2]: # how long do you want the gaussian STD to be?
            
            # with open(file, 'wb') as f:  # Python 3: open(..., 'wb')
            #   pickle.dump([accu,prec,reca], f)
    
            # Getting back the objects:
            with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
                accu,prec,reca = pickle.load(f)
                accu*=100
                prec*=100
                reca*=100
                f1=2*(prec*reca)/(prec+reca)
                print(count)
                if std==0.05:
                    if drop==1:
                        ax1.plot(threshs[:-1],accu[count,:-1],color=colors[count],linestyle='solid',label="Acc. "+str(large)+" w/drop $\sigma$="+str(std))
                        ax1.plot(threshs[:-1],prec[count,:-1],color=colors[count],linestyle='dashed',label="Prec. "+str(large)+" w/drop $\sigma$="+str(std))
                        ax1.plot(threshs[:-1],reca[count,:-1],color=colors[count],linestyle='dotted',label="Rec. "+str(large)+" w/drop $\sigma$="+str(std))
                        ax1.plot(threshs[:-1],f1[count,:-1],color=(0,0,0),linestyle='solid',label="F1 "+str(large)+" w/drop $\sigma$="+str(std))
                    else:
                        ax1.plot(threshs[:-1],accu[count,:-1],color=colors[count],linestyle='solid',label="Acc. "+str(large)+" w/o drop $\sigma$="+str(std))
                        ax1.plot(threshs[:-1],prec[count,:-1],color=colors[count],linestyle='dashed',label="Prec. "+str(large)+" w/o drop $\sigma$="+str(std))
                        ax1.plot(threshs[:-1],reca[count,:-1],color=colors[count],linestyle='dotted',label="Rec. "+str(large)+" w/o drop $\sigma$="+str(std))
                        ax1.plot(threshs[:-1],f1[count,:-1],color=(0,0,0),linestyle='solid',label="F1 "+str(large)+" w/o drop $\sigma$="+str(std))
                    count+=1
ax1.legend(ncol=2,prop={'size': 12})
ax1.set_xlabel('Decision Threshold',fontsize=14)
ax1.set_ylabel('Percent',fontsize=14)
ax1.set_xlim((0,1))
ax1.set_ylim((50,100))
ax1.tick_params(axis="x", labelsize=12)
ax1.tick_params(axis="y", labelsize=12)