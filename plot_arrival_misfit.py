#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:11:46 2022 

@author:Tim Lin
@email:jiunting@uoregon.edu


Plot the arrival time misfit.
Run the model_test_timing.py first to get arrival misfit information
"""

import numpy as np
import matplotlib.pyplot as plt

thresh = 0.1
sr = 100
#A = np.load('sav_misfit_P003.npy',allow_pickle=True)
A = np.load('sav_misfit_S003.npy',allow_pickle=True)
A = A.item()

"""
>>>A.keys()
dict_keys(['misfit', 'mag', 'dist', 'famID', 'max_y'])
"""

misfit = np.array(A['misfit'])
misfit -= 750 # arrival is always at the center
misfit = misfit*(1.0/sr)
mag = np.array(A['mag'])
dist = np.array(A['dist'])
famID = np.array(A['famID'])
max_y = np.array(A['max_y'])

idx = np.where(max_y>=thresh)[0]
misfit = misfit[idx]
mag = mag[idx]
dist = dist[idx]
famID = famID[idx]
max_y = max_y[idx]


#make boxplot
fig, axs = plt.subplots(1,5,figsize=(8,5.5)) #create grids
ax_mrg1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
wd = 10 #km from 0~95
#out_dots = dict(markerfacecolor=[0.1,0.1,0.1],markeredgecolor=[0.1,0.1,0.1],mew=0.1, marker='d',markersize=6)
#plt.plot(dist,misfit,'k.',ms=1)
gps = np.arange(0,95,wd*2)
all_box_res = []
for gp in gps:
    #idx = np.where((np.abs(sav_all_Mw-gp)<=wd) & (~np.isnan(res)) )[0]
    idx = np.where((np.abs(dist-gp)<=wd) )[0]
    all_box_res.append(misfit[idx])

#bp = ax_mrg1.boxplot(all_box_res,positions=gps,widths=wd*2,patch_artist=True,flierprops=out_dots)
vp = ax_mrg1.violinplot(all_box_res,positions=gps,widths=wd*2)
for patch in vp['bodies']:
    patch.set_facecolor('darkorange')
    patch.set_edgecolor([0,0.,0])

vp['cbars'].set_linewidth(0.8)
vp['cmins'].set_linewidth(0.8)
vp['cmaxes'].set_linewidth(0.8)
vp['cbars'].set_color('k')
vp['cmins'].set_color('k')
vp['cmaxes'].set_color('k')

plt.xticks([0,20,40,60,80],['0','20','40','60','80'])
plt.ylim([-8,8])
plt.ylabel('Predicted - Real (s)',fontsize=14,labelpad=0)
plt.xlabel('Distance (km)',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,length=0.5,size=0.5,labelsize=12)

ax_mrg2 = plt.subplot2grid((1, 5), (0, 2), colspan=2)
wd = 0.2 #km from 0~95
out_dots = dict(markerfacecolor=[0.1,0.1,0.1],markeredgecolor=[0.1,0.1,0.1],mew=0.1, marker='d',markersize=6)
#plt.plot(mag,misfit,'k.',ms=1)
gps = np.arange(1.0,2.6,wd*2)
all_box_res = []
for gp in gps:
    #idx = np.where((np.abs(sav_all_Mw-gp)<=wd) & (~np.isnan(res)) )[0]
    idx = np.where((np.abs(mag-gp)<=wd) )[0]
    all_box_res.append(misfit[idx])

vp = ax_mrg2.violinplot(all_box_res,positions=gps,widths=wd*2)
for patch in vp['bodies']:
    patch.set_facecolor('aqua')
    patch.set_edgecolor([0,0.,0])

vp['cbars'].set_linewidth(0.8)
vp['cmins'].set_linewidth(0.8)
vp['cmaxes'].set_linewidth(0.8)
vp['cbars'].set_color('k')
vp['cmins'].set_color('k')
vp['cmaxes'].set_color('k')
    
plt.xticks([1. , 1.4, 1.8, 2.2],['1.0' , '1.4', '1.8', '2.2'])
plt.ylim([-8,8])
plt.xlabel('Magnitude',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,length=0.5,size=0.5,labelsize=12)
ax1.tick_params(labelleft=False)


plt.subplot(1,5,5)
plt.hist(misfit, bins=50, orientation="horizontal",facecolor='r');
plt.ylim([-8,8])
plt.xlabel('Frequency',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=0.5,length=0.5,size=0.5,labelsize=12)
ax1.tick_params(labelleft=False)

plt.savefig('misfit_S_%.1f.png'%(thresh),dpi=450)
plt.show()
