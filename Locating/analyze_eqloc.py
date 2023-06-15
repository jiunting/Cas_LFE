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


def get_daily_nums(T, dt=1, T_st: str = None, return_idx = False):
    # get number of events each day
    # T is the sorted timeseries of the occurrence
    # dt is the time window for calculating number of events
    # T_st is the time to start with: None or str, If None, use the first date as T0
    # return_idx is to also return the index of each time window
    if T_st != None:
        assert type(T_st)==str, "T_st is str in YYYYMMDD e.g. 20100302"
        T0 = datetime.datetime(int(T_st[:4]), int(T_st[4:6]), int(T_st[6:8]))
        T1 = T0 + datetime.timedelta(dt)
    else:
        T0 = datetime.datetime(T[0].year,T[0].month,T[0].day)
        T1 = T0 + datetime.timedelta(dt)
    sav_num = [0] # number of events in a day
    sav_T = []
    sav_idx = [[]]
    for ii,i in enumerate(T):
        #print('Time',i,'between?',T0,T1)
        while True: #keep trying the current i, until find the right time window T0-T1
            if T0<=i<T1:
                sav_num[-1] += 1
                sav_idx[-1].append(ii)
                break # move on to next i., use the same T0-T1
            else:
                #i is outside the T0-T1 window, and this `must` be because i > (T0 ~ T1), update time, use the same i.
                sav_T.append(T0+datetime.timedelta(dt*0.5)) # deal with sav_num[-1]'s time
                # update time window
                T0 = T1
                T1 = T0 + datetime.timedelta(dt)
                sav_num.append(0) # create new counter for next time
                sav_idx.append([])
    sav_T.append(T0+datetime.timedelta(dt*0.5))
    sav_num = np.array(sav_num)
    if return_idx:
        return np.array(sav_T),np.array(sav_num), sav_idx
    return np.array(sav_T),np.array(sav_num)


def ML2M0(ML):
    # convert local magnitude to moment
    return 10**(1.5*ML+9.0)

def get_contour2D(x,y,z,sampl=0.01):
    from scipy.interpolate import griddata,interp2d
    #plt.scatter() makes sense, but it is not the plt.contour format
    #the function return the X,Y,Z in plt.contour format
    x=np.array(x)
    y=np.array(y)
    z=np.array(z)
    #f = interp2d(x, y, z, kind='cubic') #the interpolate function
    xx=np.arange(x.min(),x.max(),sampl)
    yy=np.arange(y.min(),y.max(),sampl)
    X,Y=np.meshgrid(xx,yy)
    Z=griddata((x,y),z,(X,Y),method='linear',fill_value=0)
    return X,Y,Z

#from mpl_toolkits.basemap import Basemap
#m = Basemap(projection='merc',llcrnrlat=48.1-1,urcrnrlat=49.3+1,llcrnrlon=-124.5-1,urcrnrlon=-123+1,lat_ts=48.7,resolution='f')
#m.drawcoastlines()
def plot_map():
    plt.figure(figsize=(6.2,8.2))
    IN1 = open('Vancouver_coast_nation.txt','r')
    curr_line = []
    for line in IN1.readlines():
        if line[0]=='>':
            if len(curr_line)!=0:
                curr_line = np.array(curr_line)
                plt.plot(curr_line[:,0],curr_line[:,1],'--',color=[0.5,0.5,0.5],lw=1.)
                curr_line = []
            continue
        elems = line.split()
        curr_line.append([float(elems[0]), float(elems[1])])
    IN1.close()
    IN1 = open('Vancouver_coast.txt','r')
    curr_line = []
    for line in IN1.readlines():
        if line[0]=='>':
            if len(curr_line)!=0:
                curr_line = np.array(curr_line)
                plt.plot(curr_line[:,0],curr_line[:,1],'-',color=[0.7,0.7,0.7],lw=0.6)
                curr_line = []
            continue
        elems = line.split()
        curr_line.append([float(elems[0]), float(elems[1])])
    IN1.close()


# LFE Locating result file
#fname = "./EQloc_0.2_5_S.txt"
#fname = "./EQloc_001_0.1_3_S.txt"
fname = "./EQloc_005_0.3_3_S.txt" # slab geometry
#fname = "./EQloc_002_0.1_3_S.txt" # locating with C8 stations

window_len = 10 # group N days together. Set to 1 means to calculate daily number.

# Tremor catlog
tremor_file = "../tremor_events-2009-08-06T00_00_00-2014-12-31T23_59_59.csv"

# EQ catalog
EQ_file = "./EQ_catalog_CN.txt" # earthquake catalog from CN, more EQ than NEIC

# load LFE data and do filter
A = pd.read_csv(fname,sep='\s+',header=0)

# ----------residual filtering based on number of stations----------
# IDEA: Different number of station used have different threshold, smaller stations are easier to fit.
# Get the misfit distribution for each stations i.e. use 3 stations, 5 stations or 8 stations etc.
# Find the first N% of sorted misfit
#first_N = 0.25 # a float from 0-1 select the residual threshold based on each groups
first_N = 0.10 # a float from 0-1 select the residual threshold based on each groups
sav_thres = {}
for N in np.unique(A['N']):
    tmp_residual = list(A[A['N']==N]['residual'])
    tmp_residual.sort()
    sav_thres[N] = tmp_residual[int((len(tmp_residual)-1)*first_N)]

thres_25 = np.array([sav_thres[N] for N in A['N']])
A['thres_25'] = thres_25
A = A[A['residual']<=A['thres_25']]
# ----------------------END of filter------------------------------------

#A['T'] = pd.to_datetime(A['T'],format='%Y-%m-%dT%H:%M:%S.%fZ')
T = [UTCDateTime(t) for t in A['OT']]
# convert UTCDateTime to decimal year
T_dec = np.array([utc_to_decimal_year(t) for t in T])


# load tremor data
tremor = pd.read_csv(tremor_file)
tremor = tremor[(tremor['lon']>=-124.6) & (tremor['lon']<=-122.9) & (tremor['lat']>=48.0) & (tremor['lat']<=49.4)]
T_tremor = np.array([UTCDateTime(t) for t in tremor['starttime']])
T_tremor_dec = np.array([utc_to_decimal_year(t) for t in T_tremor])
sor_idx = np.argsort(T_tremor_dec)
tremor = tremor.iloc[sor_idx]
T_tremor = T_tremor[sor_idx]
T_tremor_dec = T_tremor_dec[sor_idx]

# Calcaulte number of events in a time window. If window is 1 then its daily event number.
window_T_LFE, window_nums_LFE,sav_idx_LFE = get_daily_nums(T, window_len, '20030101',True) # for LFE
# convert datetime to decimal year for plotting
window_T_LFE_dec = np.array([utc_to_decimal_year(UTCDateTime(i)) for i in window_T_LFE])

# Same process with tremor
window_T_tremor, window_nums_tremor, sav_idx_tremor = get_daily_nums(T_tremor, window_len, '20030101',True) # for tremor
window_T_tremor_dec = np.array([utc_to_decimal_year(UTCDateTime(i)) for i in window_T_tremor])


plt.subplot(2,1,1)
plt.plot(window_T_LFE_dec, window_nums_LFE,'b')
plt.xlabel('year')
plt.ylabel('daily num')
plt.grid(axis='y')
plt.subplot(2,1,2)
plt.plot(T_dec, np.cumsum(np.ones(len(T_dec))),'k')
plt.plot(window_T_LFE_dec, np.cumsum(window_nums_LFE)) # the same thing, for debug purpose
plt.xlabel('year')
plt.ylabel('cumsum')
plt.grid(axis='y')
#plt.show()

"""

csv = pd.read_csv('EQ_catalog.csv') # USGS earthquake catalog
csv = csv[csv['depth']]
tEQ = np.array([UTCDateTime(t) for t in csv['time']])
tEQ.sort()
daily_T_EQ, daynums_EQ = get_daily_nums(tEQ, window_len, '20030101') #tEQ in UTCDateTime  for calculating number
daily_T_EQ = np.array([utc_to_decimal_year(UTCDateTime(t)) for t in daily_T_EQ])
tEQ = np.array([utc_to_decimal_year(UTCDateTime(t)) for t in csv['time']]) #tEQ in decimal year for plotting
plt.subplot(2,1,1)
plt.plot(daily_T_EQ, daynums_EQ*(0.8*daynums.max()/daynums_EQ.max()), 'm')
plt.plot([tEQ, tEQ], [np.zeros(len(csv)),csv['mag']*(0.7*daynums.max()/csv['mag'].max())],'ro-',lw=0.3,markersize=0.5)
#csv_filt = csv[csv['mag']>=1.0]
#plt.plot([tEQ[csv_filt.index],tEQ[csv_filt.index]], [np.zeros(len(csv_filt)),csv_filt['mag']*(0.7*daynums.max()/csv_filt['mag'].max())],'ro-',lw=0.3,markersize=0.5)

"""

EQ = pd.read_csv(EQ_file, sep='|') # CN earthquake catalog
EQ = EQ[(EQ['Depth/km']>10) & (EQ['Time']<'2015')]
T_EQ = np.array([UTCDateTime(t) for t in EQ['Time']])
sor_idx = np.argsort(T_EQ)
T_EQ = T_EQ[sor_idx]
T_EQ_dec = np.array([utc_to_decimal_year(UTCDateTime(i)) for i in T_EQ])
window_T_EQ, window_nums_EQ, sav_idx_EQ = get_daily_nums(T_EQ, window_len, '20030101',True)
# convert datetime to decimal year for plotting
window_T_EQ_dec = np.array([utc_to_decimal_year(UTCDateTime(t)) for t in window_T_EQ])
mag = np.array(EQ['Magnitude'])[sor_idx] # apply the same sorting with T_EQ
M0 = np.array([ML2M0(i) for i in mag])
window_M0 = np.array([sum(M0[i]) for i in sav_idx_EQ])

# overlay on LFE plot
plt.subplot(2,1,2)
ax1 = plt.gca()
ax2 = ax1.twinx() #twinx means same x axis (wanna plot different y)
ax2.set_xlim(ax1.get_xlim())
ax2.plot(window_T_EQ_dec, np.cumsum(window_M0),'r')
ax2.set_ylabel('accumulated M0')
#plt.plot(window_T_EQ_dec, window_nums_EQ*(0.8*window_nums_LFE.max()/window_nums_EQ.max()), 'm')
#plt.plot([T_EQ_dec,T_EQ_dec], [np.zeros(len(T_EQ)),mag*(0.7*window_nums_LFE.max()/mag.max())],'r-',lw=0.3,markersize=0.5)
#ax2.plot([T_EQ_dec,T_EQ_dec], [np.zeros(len(T_EQ)),mag],'r-',lw=0.3,markersize=0.5)
#plt.plot([T_EQ,T_EQ], [np.zeros(len(T_EQ)),M0*(0.7*window_nums_LFE.max()/mag.max())],'ro-',lw=0.3,markersize=0.5)
ax2.grid(False)
plt.show()

#-----------Start some spatial analysis------------
# see movement along lon,lat,dep
window = 200
lon_avg = moving_average(np.array(A['lon']), window)
lat_avg = moving_average(np.array(A['lat']), window)
dep_avg = moving_average(np.array(A['depth']), window)
T_avg = moving_average(np.array(T_dec), window)
#plt.subplot(2,1,1)
#plt.scatter(A['lon'], A['lat'], c=decT, s=10, cmap='jet');plt.colorbar()
plt.scatter(lon_avg[:], lat_avg[:], c=T_avg[:], s=0.05, cmap='jet');plt.colorbar()
plt.xlim([-124.5,-123])
plt.ylim([48.1,49.3])
plt.show()

# generate contour or pcolor data to plot on map
x_range = (-124.5,-123)
y_range = (48.1,49.3)
dx = dy = 0.01
X = np.arange(x_range[0],x_range[1],dx)
Y = np.arange(y_range[0],y_range[1],dy)
XX, YY = np.meshgrid(X,Y)
ZZ = np.zeros(XX.shape)
# put the points in the grid-nodes
for i in range(len(lon_avg)):
    ix = np.where(X>=lon_avg[i])[0][0] - 1
    iy = np.where(Y>=lat_avg[i])[0][0] - 1
    ZZ[iy, ix] += 1


plot_map()
plt.pcolor(XX, YY, ZZ)
plt.xlim([-124.5,-123])
plt.ylim([48.1,49.3])
ax1 = plt.gca()
ax1.tick_params(pad=1.5,length=0,size=4.,labelsize=11)
for i in ax1.get_xticklines():
    i.set_visible(True)

for i in ax1.get_yticklines():
    i.set_visible(True)

plt.colorbar()
plt.show()


#----Plot moving avg of lon, lat, depth corrdinates time series------
plt.figure()
plt.subplot(3,1,1)
plt.plot(T_avg,lon_avg,'r')
#plt.plot(T_avg,lat_avg-np.mean(lat_avg),'b',label='Lat mov_avg(%d)'%(window))
#plt.plot(T_avg,(dep_avg-np.mean(dep_avg))*0.1,'k',label='Dep mov_avg(%d)'%(window))
plt.ylabel('Lon')
plt.grid(True)
ax1=plt.gca()
ax2 = ax1.twinx() #twinx means same x axis (wanna plot different y)
ax2.set_xlim(ax1.get_xlim())
ax2.plot(window_T_LFE_dec,window_nums_LFE,'k')
plt.xlim([2005, 2014.5])
#plt.ylabel('$\Delta$ Lon')

plt.subplot(3,1,2)
plt.plot(T_avg,lat_avg,'r')
plt.ylabel('Lat')
plt.grid(True)
ax1=plt.gca()
ax2 = ax1.twinx() #twinx means same x axis (wanna plot different y)
ax2.set_xlim(ax1.get_xlim())
ax2.plot(window_T_LFE_dec,window_nums_LFE,'k')
plt.xlim([2005, 2014.5])
#plt.ylabel('$\Delta$ Lon')

plt.subplot(3,1,3)
plt.plot(T_avg,dep_avg,'r')
plt.ylabel('Dep')
plt.grid(True)
ax1=plt.gca()
ax2 = ax1.twinx() #twinx means same x axis (wanna plot different y)
ax2.set_xlim(ax1.get_xlim())
ax2.plot(window_T_LFE_dec,window_nums_LFE,'k')
plt.xlim([2005, 2014.5])
#plt.ylabel('$\Delta$ Lon')
plt.show()
#----Plot moving avg of lon, lat, depth corrdinates time series END------



# load daily LFEs detections from file using N=3, y>0.1, so it has more
#AA = np.load('daily_T_num.npy')
#daily_T, daynums = AA[0], AA[1]
plt.plot(T_dec, np.cumsum(np.ones(len(T_dec))), 'k')
plt.grid(True)
plt.title('Manually mark the LFE initiation!')
plt.show()


LFE_start = [2005.6744, 2007.063, 2008.3409, 2009.3470, 2010.6201, 2011.6202, 2012.6655, 2013.7328]
#LFE_start = [2006.3091, 2010.142, ]
plt.plot(T_dec, np.cumsum(np.ones(len(T_dec))), 'k')
plt.grid(True)
for t in LFE_start:
    plt.axvline(t, color = 'r')

plt.title('Manual picks for large SSE start')
plt.xlim([2005,2014.5])
plt.show()


# ---------Plot LFE initiation v.s. during large SSE-----------
#x_range = (-124.5,-123)
#y_range = (48.1,49.3)
x_range = (-124.8,-122.7)
y_range = (47.8,49.6)
dx = dy = 0.02
X = np.arange(x_range[0],x_range[1],dx)
Y = np.arange(y_range[0],y_range[1],dy)
XX, YY = np.meshgrid(X,Y)
props = dict(boxstyle='round', facecolor='white', alpha=0.7)
for i_t,t in enumerate(LFE_start):
    # method 1. find idx_bf, idx_ad by number of points
    #idx_bf = np.where( T_dec < (t-0.01/365.25))[0][-20:]
    #idx_af = np.where( T_dec > (t+0.01/365.25))[0][:10000]
    # method 2. find idx_bf, idx_ad by time
    #idx_bf = np.where( (T_dec<(t-0.01/365.25)) & (T_dec>(t-3/365.25)) )[0]
    #idx_af = np.where( (T_dec>(t+0.01/365.25)) & (T_dec<(t+30/365.25)) )[0]
    # method 3. similar to 2, find idx_bf, idx_ad by time. idx_bf is the initiation of SSE
    idx_bf = np.where( T_dec>t )[0][:50] # use the first 50 LFE locations
    idx_af = np.where( (T_dec>t) & (T_dec<(t+7/365.25)))[0][50:] # skip the first 50 points
    # repeat the same thing for tremor
    idx_tremor_bf = np.where( T_tremor_dec>t )[0][:50]
    idx_tremor_af = np.where( (T_tremor_dec>t) & (T_tremor_dec<(t+7/365.25)))[0][50:]
    # see if any EQ within the time. Seems not...
    #idx_EQ = np.where((T_EQ_dec>t-(5/365.25)) & (T_EQ_dec<(t+15/365.25))  )[0]
    # calculate pre-SSE coordinate
    lon_bf, lat_bf = np.mean(A.iloc[idx_bf]['lon']), np.mean(A.iloc[idx_bf]['lat'])
    lon_tremor_bf, lat_tremor_bf = np.mean(tremor.iloc[idx_tremor_bf]['lon']), np.mean(tremor.iloc[idx_tremor_af]['lat'])
    # make moving average
    window = 5
    lon_avg = moving_average(np.array(A.iloc[idx_af]['lon']), window)
    lat_avg = moving_average(np.array(A.iloc[idx_af]['lat']), window)
    # same thing for tremor
    lon_tremor_avg = moving_average(np.array(tremor.iloc[idx_tremor_af]['lon']), window)
    lat_tremor_avg = moving_average(np.array(tremor.iloc[idx_tremor_af]['lat']), window)
    # SSE coordinate
    ZZ = np.zeros(XX.shape)
    # put the points in the grid-nodes
    for i in range(len(lon_avg)):
        #ix = np.where(X>=A.iloc[i]['lon'])[0][0] - 1 #individual point is too noisy, use moving avg instead
        #iy = np.where(Y>=A.iloc[i]['lat'])[0][0] - 1
        ix = np.where(X>=lon_avg[i])[0][0] - 1
        iy = np.where(Y>=lat_avg[i])[0][0] - 1
        ZZ[iy, ix] += 1
    # tremor coordinate
    ZZ_tremor = np.zeros(XX.shape)
    # put the points in the grid-nodes
    for i in range(len(lon_tremor_avg)):
        #ix = np.where(X>=A.iloc[i]['lon'])[0][0] - 1 #individual point is too noisy, use moving avg instead
        #iy = np.where(Y>=A.iloc[i]['lat'])[0][0] - 1
        ix = np.where(X>=lon_tremor_avg[i])[0][0] - 1
        iy = np.where(Y>=lat_tremor_avg[i])[0][0] - 1
        ZZ_tremor[iy, ix] += 1
    # make plot
    plot_map()
    plt.pcolor(XX, YY, ZZ)
    plt.contour(XX,YY,ZZ_tremor,10 , cmap='hot')
    # plot initiation point
    plt.plot(lon_bf, lat_bf, '*', color=[0,0,1],markeredgecolor=[1,1,1], markersize=16)
    plt.plot(lon_tremor_bf, lat_tremor_bf, '*', color=[1,0,1],markeredgecolor=[1,1,1], markersize=16)
    plt.xlim([-124.5,-123])
    plt.ylim([48.1,49.3])
    ax1 = plt.gca()
    ax1.tick_params(pad=1.5,length=0,size=4.,labelsize=11)
    for i in ax1.get_xticklines():
        i.set_visible(True)
    
    for i in ax1.get_yticklines():
        i.set_visible(True)
    
    plt.text(-124.4,49.2,'T= %.2f'%(t),fontsize=14,bbox=props) # box
    #plt.title('Large SSE start: %.2f'%(t), fontsize=14)
    #plt.colorbar()
    #plt.savefig('./fig_LFE_loc/LFE_loc_T%01d.png'%(i_t),dpi=200)
    plt.show()
    break







#===============================END of Analysis============================

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
tremor = pd.read_csv(tremor_file)
tremor = tremor[(tremor['lon']>=-124.6) & (tremor['lon']<=-122.9) & (tremor['lat']>=48.0) & (tremor['lat']<=49.4)]
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

