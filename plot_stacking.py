import numpy as np
import glob
import matplotlib
matplotlib.use('pdf') #instead using interactive backend
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from scipy import signal


data_path = '/projects/amt/jiunting/Cascadia_LFE/Data_QC_rmean/'
data_all_P = glob.glob(data_path+'*_P.h5')
data_all_S = glob.glob(data_path+'*_S.h5')

#/projects/amt/jiunting/Cascadia_LFE/Data_QC_rmean/ID_162_CN.VGZ.BH_P.h5
T = np.concatenate([np.arange(3001)/100,np.arange(3001)/100+30,np.arange(3001)/100+60])
n_plt = 20 # only plot first n traces
#stack_max = 1000
#for d in data_all_S:
for d in data_all_P:
    print('Now loading:',d)
    fam = d.split('/')[-1].split('_')[1]
    net_sta = d.split('/')[-1].split('_')[2]
    phase = d.split('/')[-1].split('_')[3].split('.')[0] #P or S
    dtfl = h5py.File(d, 'r')
    A = dtfl.get('waves')
    AA = np.array(A)
    sumA = np.zeros_like(T)
    # create subplot with the fig height=2:1
    fig, axs = plt.subplots(2,1,figsize=(8,5.5), gridspec_kw={'height_ratios': [2, 1]})
    for i in range(len(AA)):
        if i<n_plt:
            #plt.plot(T,AA[i]/max(np.abs(AA[i]))+i,'-',color=[0.7,0.7,0.7],linewidth=0.8) #only plot first n traces
            axs[0].plot(T,AA[i]/max(np.abs(AA[i]))+i,'-',color=[0.7,0.7,0.7],linewidth=0.8) #only plot first n traces
        # stacking continuing
        #if i>=stack_max:
        #    break
        sumA += AA[i]/max(np.abs(AA[i]))
        
    sumA = (sumA/max(np.abs(sumA))) * 2 #make stacking slightly larger
    axs[0].plot(T,sumA+n_plt,'r',linewidth=1)
    #props = dict(boxstyle='square', facecolor='white', alpha=0.7)
    axs[0].text(2,n_plt+0.5,'stack (N=%d)'%(i+1),fontsize=12)
    axs[0].plot([30,30],[-1,n_plt+2],'k',linewidth=1.5)
    axs[0].plot([60,60],[-1,n_plt+2],'k',linewidth=1.5)
    axs[0].plot([90,90],[-1,n_plt+2],'k',linewidth=1.5)
    axs[0].set_xlim([0,90])
    axs[0].set_ylim([-1,n_plt+2])
    axs[0].set_yticks([])
    #ax1=plt.gca()
    #ax1.tick_params(pad=1)
    axs[0].tick_params(axis='both',pad=1,labelsize=12)
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    axs[0].text(5,0.3,'Z',fontsize=14,bbox=props)
    axs[0].text(35,0.3,'E',fontsize=14,bbox=props)
    axs[0].text(65,0.3,'N',fontsize=14,bbox=props)
    #axs[0].set_xticklabels(axs[0].get_xticklabels(),fontsize=12)
    #axs[0].set_xlabel('Time (s)',fontsize=14,labelpad=0)
    axs[0].set_ylabel('Traces',fontsize=14,labelpad=0)
    axs[0].set_title('%s family:%s phase:%s'%(net_sta,fam,phase),fontsize=15)
    # plot 2nd subplot
    #axs[1].plot(T,sumA+n_plt,'r',linewidth=1)
    sumA = (sumA/max(np.abs(sumA)))*(1/4.0)
    Z_stack = sumA[:3001]
    E_stack = sumA[3001:6002]
    N_stack = sumA[6002:9003]
    assert len(Z_stack)==len(E_stack)==len(N_stack)==3001,"npts not consistent!"
    axs[1].plot(T[:3001],N_stack+1*(1/4.0),'r',linewidth=1)
    axs[1].text(8,1.1*(1/4.0),'N stack',fontsize=14)
    axs[1].plot(T[:3001],E_stack+2*(1/4.0),'r',linewidth=1)
    axs[1].text(8,2.1*(1/4.0),'E stack',fontsize=14)
    axs[1].plot(T[:3001],Z_stack+3*(1/4.0),'r',linewidth=1)
    axs[1].text(8,3.1*(1/4.0),'Z stack',fontsize=14)
    axs[1].plot(T[:3001],signal.gaussian(3001,std=int(0.4*100)),'b-') # plot gaussian
    axs[1].set_xlim([7.5,22.5])
    axs[1].tick_params(axis='both',pad=1,labelsize=12)
    axs[1].set_ylabel('Target',fontsize=14,labelpad=0)
    axs[1].set_xlabel('Time (s)',fontsize=14,labelpad=0)
    plt.subplots_adjust(left=0.08,top=0.95,right=0.97,bottom=0.08,wspace=0.07,hspace=0.1)
    fig.savefig('./Figs_data_stacking/fam_%s_%s_%s.png'%(fam,net_sta,phase))
    #fig.close()
    plt.close()
    dtfl.close()
    #break
