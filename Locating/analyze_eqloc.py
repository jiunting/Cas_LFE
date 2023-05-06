#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 21:14:10 2023 

@author:Tim Lin
@email:jiunting@uoregon.edu

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from obspy import UTCDateTime
import datetime
from numba import njit
import time

#fname = "./EQloc_0.2_5_S.txt"
fname = "./EQloc_001_0.1_3_S.txt"

A = pd.read_csv(fname,sep='\s+',header=0)
#A['T'] = pd.to_datetime(A['T'],format='%Y-%m-%dT%H:%M:%S.%fZ')
T = [UTCDateTime(t) for t in A['OT']]

def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start} seconds to run")
        return result
    return wrapper
    



@time_it
@njit
def moving_average(numbers, window_size):
    """
    Returns the moving average of the given list of numbers using the specified window size.
    """
    moving_averages = []
    for i in range(len(numbers) - window_size + 1):
        window = numbers[i:i+window_size]
        window_average = np.mean(window)
        moving_averages.append(window_average)
    return moving_averages
    
    
def utc_to_decimal_year(utc):
    """
    Converts an ObsPy UTCDateTime object to decimal year.
    """
    total_t = UTCDateTime(utc.year+1, 1, 1) - UTCDateTime(utc.year, 1, 1)
    return utc.year + (utc - UTCDateTime(utc.year, 1, 1)) / total_t
    

def get_daily_nums(T):
    # get number of events each day
    # T is the sorted timeseries of the occurrence
    T0 = datetime.datetime(T[0].year,T[0].month,T[0].day)
    T1 = T0 + datetime.timedelta(1)
    sav_num = [0] # number of events in a day
    sav_T = []
    for i in T:
        #print('Time',i,'between?',T0,T1)
        while True: #keep trying the current i, until find the right time window T0-T1
            if T0<=i<T1:
                sav_num[-1] += 1
                break # move on to next i., use the same T0-T1
            else:
                #i is outside the T0-T1 window, and this `must` be because i > (T0 ~ T1), update time, use the same i.
                sav_T.append(T0+datetime.timedelta(0.5)) # deal with sav_num[-1]'s time
                # update time window
                T0 = T1
                T1 = T0 + datetime.timedelta(1)
                sav_num.append(0) # create new counter for next time
    sav_T.append(T0+datetime.timedelta(0.5))
    sav_num = np.array(sav_num)
    return np.array(sav_T),np.array(sav_num)

daily_T, daynums = get_daily_nums(T)
daily_T = np.array([utc_to_decimal_year(UTCDateTime(i)) for i in daily_T])

plt.subplot(2,1,1)
plt.plot(daily_T, daynums)
plt.xlabel('year')
plt.ylabel('daily num')
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(daily_T, np.cumsum(daynums))
plt.xlabel('year')
plt.ylabel('cumsum')
plt.grid(True)
#plt.show()
"""
csv = pd.read_csv('EQ_catalog.csv') # USGS earthquake catalog
tEQ = [UTCDateTime(t) for t in csv['time']]
tEQ.sort()
daily_T_EQ, daynums_EQ = get_daily_nums(tEQ)
tEQ = np.array([utc_to_decimal_year(UTCDateTime(t)) for t in csv['time']])
plt.subplot(2,1,1)
plt.plot(tt,csv['mag']*(2000/4.79),'r-')


"""


# convert UTCDateTime to decimal year
decT = np.array([utc_to_decimal_year(t) for t in T])


# see movement along lon,lat,dep
window = 200
lon_avg = moving_average(np.array(A['lon']), window)
lat_avg = moving_average(np.array(A['lat']), window)
dep_avg = moving_average(np.array(A['depth']), window)
T_avg = moving_average(np.array(decT), window)
#plt.subplot(2,1,1)
#plt.scatter(A['lon'], A['lat'], c=decT, s=10, cmap='jet');plt.colorbar()
plt.scatter(lon_avg, lat_avg, c=T_avg, s=0.05, cmap='jet');plt.colorbar()
plt.xlim([-124.5,-123])
plt.ylim([48.1,49.3])
plt.show()

#plt.subplot(2,1,2)
#plt.plot(daily_T,np.cumsum(daynums))


plt.figure()
plt.subplot(2,1,1)
#plt.plot(decT,A['lon']-np.mean(lon_avg),'k',lw=0.2)
plt.plot(T_avg,lon_avg-np.mean(lon_avg),'r',label='Lon mov_avg(%d)'%(window))
plt.plot(T_avg,lat_avg-np.mean(lat_avg),'b',label='Lat mov_avg(%d)'%(window))
plt.plot(T_avg,(dep_avg-np.mean(dep_avg))*0.1,'k',label='Dep mov_avg(%d)'%(window))
plt.xlim([2005, 2014.5])
plt.legend()
plt.ylabel('$\Delta$ Lon')
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(daily_T,np.cumsum(daynums),'r')
plt.xlim([2005, 2014.5])
plt.ylabel('Cumulative num')
plt.grid(True)
plt.show()


# load daily LFEs detections from file using N=3, y>0.1, so it is more
AA = np.load('daily_T_num.npy')
daily_T, daynums = AA[0], AA[1]


# making plot
fig, ax = plt.subplots(2,1,figsize=(7.5,9))

ax[0].plot(daily_T, np.cumsum(daynums),'k')
ax[0].plot(trem_time,np.cumsum(trem_daynums)*5e1,'m')
ax[0].grid(True)


# manually define SSE and iSSE windows
SSE_windows = [(2005.65, 2005.78), (2007.06, 2007.1), (2008.33, 2008.42), (2009.34, 2009.44), (2010.61, 2010.69), (2011.61, 2011.68), (2012.66, 2012.75), (2013.7, 2013.77)]
for i, w in enumerate(SSE_windows):
    if i==0:
        ax[0].axvline(w[0],color='r',lw=0.5,label='SSE') #st
    else:
        ax[0].axvline(w[0],color='r',lw=0.5) #ed
    ax[0].axvline(w[1],color='r',lw=0.5) #ed
    # fill between
    y = [0, 1e6]
    ax[0].fill_betweenx(y, w[0]*np.ones(len(y)),w[1]*np.ones(len(y)),hatch='.',color=[0.8,0.,0.])

iSSE_windows = [(2005,2005.6), (2006.31, 2006.88), (2007.44, 2008.2), (2008.6, 2009.2), (2010.14, 2010.57), (2010.99, 2011.55), (2011.92, 2012.62), (2013.23, 2013.68)]
for i, w in enumerate(iSSE_windows):
    if i==0:
        ax[0].axvline(w[0],color='b',lw=0.5,label='inter-SSE') #st
    else:
        ax[0].axvline(w[0],color='b',lw=0.5) #st
    ax[0].axvline(w[1],color='b',lw=0.5) #ed
    # fill between
    y = [0, 1e6]
    ax[0].fill_betweenx(y, w[0]*np.ones(len(y)),w[1]*np.ones(len(y)),hatch='x',color=[0.8,0.8,0.8])

# find LFE locations during SSE occurrence
sav_loc_SSE = []
for i in range(len(SSE_windows)):
    st,ed = SSE_windows[i]
    idx = np.where((decT>=st) & (decT<ed))[0]
    sav_loc_SSE.append([np.mean([st,ed]), np.mean(A['lon'].iloc[idx]), np.mean(A['lat'].iloc[idx])])

sav_loc_iSSE = []
for i in range(len(iSSE_windows)):
    st,ed = iSSE_windows[i]
    idx = np.where((decT>=st) & (decT<ed))[0]
    sav_loc_iSSE.append([np.mean([st,ed]), np.mean(A['lon'].iloc[idx]), np.mean(A['lat'].iloc[idx])])

sav_loc_SSE, sav_loc_iSSE = np.array(sav_loc_SSE), np.array(sav_loc_iSSE)

# Make a plot for the cycles on the map
cycles = range(len(sav_loc_SSE))
cm = plt.cm.jet(plt.Normalize(min(cycles),max(cycles))(cycles))

for c in cycles:
    ax[1].plot(sav_loc_iSSE[c,1], sav_loc_iSSE[c,2],'*',ms=15,markeredgecolor=[0,0,0],lw=0.2,color=cm[c])
    ax[1].plot(sav_loc_SSE[c,1], sav_loc_SSE[c,2],'^',ms=10,color=cm[c])
    #plot vectors
    ax[1].quiver(sav_loc_iSSE[c,1], sav_loc_iSSE[c,2], sav_loc_SSE[c,1]-sav_loc_iSSE[c,1], sav_loc_SSE[c,2]-sav_loc_iSSE[c,2],angles='xy', scale_units='xy', scale=1 )

norm = matplotlib.colors.Normalize(vmin=min(cycles), vmax=max(cycles))
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='jet')
cmap.set_array([])
# put the bar inside the plot
cbaxes = fig.add_axes([0.16, 0.18, 0.24, 0.02])
clb = fig.colorbar(cmap,cax=cbaxes, orientation='horizontal',label='Cycles')

#ax[1].set_xlim([-124.5,-123])
#ax[1].set_ylim([48.1,49.3])
ax[0].set_xlim([2004.95,2014])
ax[0].set_ylim([0,1e6])

ax[1].set_xlim([-123.92,-123.65])
ax[1].set_ylim([48.65,48.93])
ax[1].axis('equal')
plt.show()
    

# load tremor to see if it got the same thing
tremor = pd.read_csv("../tremor_events-2009-08-06T00_00_00-2014-12-31T23_59_59.csv")
tremor = tremor[(tremor['lon']>=-124.5) & (tremor['lon']<=-123) & (tremor['lat']>=48.1) & (tremor['lat']<=49.3)]
tremor_t = [UTCDateTime(tt).datetime for tt in tremor['starttime']]
tremor_dect = np.array([utc_to_decimal_year(UTCDateTime(t)) for t in tremor_t])

trem_time,trem_daynums = get_daily_nums(sorted(tremor_t))
trem_time = np.array([utc_to_decimal_year(UTCDateTime(i)) for i in trem_time])
plt.plot(trem_time,np.cumsum(trem_daynums))


# find Tremor locations during SSE/iSSE occurrence
sav_Tremloc_SSE = []
for i in range(len(SSE_windows)):
    st,ed = SSE_windows[i]
    idx = np.where((trem_time>=st) & (trem_time<ed))[0]
    sav_Tremloc_SSE.append([np.mean([st,ed]), np.mean(tremor['lon'].iloc[idx]), np.mean(tremor['lat'].iloc[idx])])

sav_Tremloc_iSSE = []
for i in range(len(iSSE_windows)):
    st,ed = iSSE_windows[i]
    idx = np.where((trem_time>=st) & (trem_time<ed))[0]
    sav_Tremloc_iSSE.append([np.mean([st,ed]), np.mean(tremor['lon'].iloc[idx]), np.mean(tremor['lat'].iloc[idx])])

sav_Tremloc_SSE, sav_Tremloc_iSSE = np.array(sav_Tremloc_SSE), np.array(sav_Tremloc_iSSE)



# making plot
fig, ax = plt.subplots(2,1,figsize=(7.5,9))

ax[0].plot(daily_T, np.cumsum(daynums),'k')
ax[0].plot(trem_time,np.cumsum(trem_daynums)*5e1,'m')
ax[0].grid(True)
cycles = range(len(sav_loc_SSE))
cm = plt.cm.jet(plt.Normalize(min(cycles),max(cycles))(cycles))

for c in cycles:
    ax[1].plot(sav_loc_iSSE[c,1], sav_loc_iSSE[c,2],'*',ms=15,markeredgecolor=[0,0,0],lw=0.2,color=cm[c])
    ax[1].plot(sav_loc_SSE[c,1], sav_loc_SSE[c,2],'^',ms=10,color=cm[c])
    #for tremor
    ax[1].plot(sav_Tremloc_iSSE[c,1], sav_Tremloc_iSSE[c,2],'.',ms=15,markeredgecolor=[0,0,0],lw=0.2,color=cm[c])
    ax[1].plot(sav_Tremloc_SSE[c,1], sav_Tremloc_SSE[c,2],'s',ms=10,color=cm[c])
    #plot vectors
    ax[1].quiver(sav_loc_iSSE[c,1], sav_loc_iSSE[c,2], sav_loc_SSE[c,1]-sav_loc_iSSE[c,1], sav_loc_SSE[c,2]-sav_loc_iSSE[c,2],angles='xy', scale_units='xy', scale=1 )
    ax[1].quiver(sav_Tremloc_iSSE[c,1], sav_Tremloc_iSSE[c,2], sav_Tremloc_SSE[c,1]-sav_Tremloc_iSSE[c,1], sav_Tremloc_SSE[c,2]-sav_Tremloc_iSSE[c,2],angles='xy', scale_units='xy', scale=1 )

plt.show()


#moving average for Tremor

Tremlon_avg = moving_average(np.array(tremor['lon']), window)
Tremlat_avg = moving_average(np.array(tremor['lat']), window)
Tremdep_avg = moving_average(np.array(tremor['depth']), window)
TremT_avg = moving_average(np.array(tremor_dect), window)
#plt.subplot(2,1,1)
#plt.scatter(A['lon'], A['lat'], c=decT, s=10, cmap='jet');plt.colorbar()
plt.scatter(Tremlon_avg, Tremlat_avg, c=TremT_avg, s=0.5, cmap='jet');plt.colorbar()
plt.xlim([-124.5,-123])
plt.ylim([48.1,49.3])
plt.show()

