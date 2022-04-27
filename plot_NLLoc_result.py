#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 00:09:33 2022 

@author:Tim Lin
@email:jiunting@uoregon.edu

"""

# plot results from the NLLoc

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

eqs = pd.read_csv('./NLLoc_out/UnetLFE_nlloc_Sonly.sum',sep='\s+')
#eqs = pd.read_csv('./NLLoc_out/UnetLFE_nlloc_PS.sum',sep='\s+')
t = pd.to_datetime(eqs['#OT'],format='%Y-%m-%dT%H:%M:%S')
loc_info = np.load('./NLLoc_out/LocDetail.npy',allow_pickle=True)
#loc_info = np.load('./NLLoc_out/LocDetail_PS.npy',allow_pickle=True)
loc_info = loc_info.item()

#plot accumulated eq number over time
plt.plot(t,np.arange(len(t)))
plt.grid(True)
plt.xlabel('year',fontsize=14)
plt.ylabel('accum. number',fontsize=14)
plt.show()

#plot distribution
plt.scatter(eqs['evlon'],eqs['evlat'],c=eqs['evdep'],s=3,cmap='jet')
clb = plt.colorbar()
clb.set_label('depth(km)')
plt.show()


# filter events
thres_phs = 6 #minimum stations
thres_RMS = 0.1 #RMS for fitting
sav_idx = []
for k in range(len(eqs)):
    if (loc_info[k]['quality']['Nphs']>=thres_phs) & (loc_info[k]['quality']['RMS']<=thres_RMS):
        sav_idx.append(k)

sav_idx = np.array(sav_idx)
#plot distribution
plt.scatter(eqs['evlon'][sav_idx],eqs['evlat'][sav_idx],c=eqs['evdep'][sav_idx],s=3,cmap='jet',vmin=0,vmax=80)
plt.title('sta>=%d RMS<=%.1f'%(thres_phs,thres_RMS))
clb = plt.colorbar()
clb.set_label('depth(km)')
plt.show()

