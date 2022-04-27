import numpy as np
import matplotlib
matplotlib.use('pdf') #instead using interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from obspy import UTCDateTime
import datetime

# load family and arrival information
EQinfo = np.load('sav_family_phases.npy',allow_pickle=True)
EQinfo = EQinfo.item()

# load original detection file (from template matching)
detcFile = 'total_mag_detect_0000_cull_NEW.txt' #LFE event detection file
sav_OT_template = []
with open(detcFile,'r') as IN1:
    for line in IN1.readlines():
        line = line.strip()
        ID = line.split()[0] #family ID
        OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
        HH = int(line.split()[2])*3600
        SS = float(line.split()[3])
        OT = OT + HH + SS + EQinfo[ID]['catShift'] #detected origin time
        sav_OT_template.append(OT.datetime)



#for this family ID, check arrivals from EQinfo
sav_eqloc = []
for ID in EQinfo.keys():
    sav_eqloc.append(EQinfo[ID]['eqLoc'])

sav_eqloc = np.array(sav_eqloc)



detc = pd.read_csv('cut_daily_all.csv')
# plot all stations
sav_stname = []
sav_stlon = []
sav_stlat = []
sav_stidx = []
for sta in list(set(detc['sta'])):
    sta_st_index = detc[detc['sta']==sta].index[0]
    tmp = detc.iloc[sta_st_index]
    sav_stname.append(tmp.sta)
    sav_stlon.append(tmp.stlon)
    sav_stlat.append(tmp.stlat)
    sav_stidx.append(sta_st_index)

# show station, LFEs location
plt.scatter(sav_eqloc[:,0]*-1,sav_eqloc[:,1],c=sav_eqloc[:,2],cmap=plt.cm.jet)
plt.plot(sav_stlon,sav_stlat,'k^',markersize=12,markeredgecolor=[1,0,0])
for i in range(len(sav_stname)):
    plt.text(sav_stlon[i],sav_stlat[i],sav_stname[i])

clb=plt.colorbar()
clb.set_label('depth(km)', rotation=90,labelpad=1,fontsize=14)
plt.savefig('map_sta_LFE.png')
plt.show()

import sys
sys.exit()


# -------plot detection timeseries--------
# get start/end index of each station
tmp_stidx = sav_stidx[:]
tmp_stidx.append(len(detc))
tmp_stidx.sort()
tmp_stidx = np.array(tmp_stidx)
now_idx = {k:i for k,i in zip(sav_stname,sav_stidx)}
for k in now_idx:
    idx = np.where(tmp_stidx==now_idx[k])[0][0]
    now_idx[k] = [now_idx[k],tmp_stidx[idx+1]-1]

# convert arr to DateTime
prepend = None
sav_tcs_sta = {}
for k in now_idx:
    print('Now in,',k)
    prepend = detc.iloc[now_idx[k][0]].id.split('_')[0]
    detc_sta = pd.to_datetime(detc.iloc[now_idx[k][0]:now_idx[k][1]+1].id,format=prepend+'_%Y-%m-%dT%H:%M:%S.%f')
    sav_tcs_sta[k] = detc_sta
    prepend = None

# show data
plt.figure()
for n,k in enumerate(sav_tcs_sta):
    plt.plot(sav_tcs_sta[k],np.ones(len(sav_tcs_sta[k]))+n,'.-',linewidth=0.3)
    plt.text(sav_tcs_sta[k].iloc[0],1+n,k,fontsize=14)

plt.plot(sav_OT_template,np.zeros(len(sav_OT_template)),'k.-',linewidth=0.1)
plt.show()


#-------plot detection end--------------------




sav_tcs_sta = {}
for i in range(len(detc)):
    if i%5000==0:
        print(i,len(detc))
    ID = detc.iloc[i].id.split('_')
    sta = ID[0]
    arr = UTCDateTime(ID[-1])
    if not(sta in sav_tcs_sta):
        sav_tcs_sta[sta] = [arr]
    else:
        sav_tcs_sta[sta].append(arr)



detc_st = pd.to_datetime(detc['id'][0:10000],format='PO.TSJB.HH_%Y-%m-%dT%H:%M:%S.%f')









# ---start associate detection by window's starttime. starttime has an exact format increasing 15 s
# find starting/ending for each station to get the earliest/final detection starttime
first_st = UTCDateTime("21000101")
for i in sav_stidx:
    if UTCDateTime(detc.iloc[i].starttime)<first_st:
        first_st = UTCDateTime(detc.iloc[i].starttime)

# grid search every datetime
trial_st = first_st

## convert pandas to datetime then grid search from whole array
## this is toooooo slow! when considering detc is a over 3M array
#detc_st = pd.to_datetime(detc['starttime'],format='%Y-%m-%dT%H:%M:%S.%fZ')
#detc_st = np.array([UTCDateTime(i_detc_st) for i_detc_st in detc_st])

# instead, use starting/ending index for each station for searching reference
tmp_stidx = sav_stidx[:]
tmp_stidx.append(len(detc))
tmp_stidx.sort()
tmp_stidx = np.array(tmp_stidx)
now_idx = {k:i for k,i in zip(sav_stname,sav_stidx)}
for k in now_idx:
    idx = np.where(tmp_stidx==now_idx[k])[0][0]
    now_idx[k] = [now_idx[k],tmp_stidx[idx+1]-1]


# initialize searching
association = {}
thres_s = 60 # if time difference gt this, stop associate
n = 0
while trial_st<UTCDateTime("20150101"):
    if n%50000==0:
        print('now trying trail_st:',trial_st)
    # now reference starttime at trial_st
    for k in now_idx:
        # see if station's index reaches to end
        if now_idx[k][0]>now_idx[k][1]:
            continue #this station done!
        # station hasn't done yet
        if UTCDateTime(detc.iloc[now_idx[k][0]].starttime)==trial_st:
            str_trial_st = trial_st.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4] #in string format
            if str_trial_st in association:
                association[str_trial_st].append(now_idx[k][0])
            else:
                association[str_trial_st] = [now_idx[k][0]]
            now_idx[k][0] += 1 #index move forward
            continue
        elif trial_st-UTCDateTime(detc.iloc[now_idx[k][0]].starttime)>thres_s:
            now_idx[k][0] += 1 #index move forward
        elif UTCDateTime(detc.iloc[now_idx[k][0]].starttime)-trial_st>thres_s:
            continue #don't do anything, until trial_st reaches to it
        else:
            print('-------------------------------------------------')
            print('Something unexpected! sta,idx=',k,now_idx[k])
            print('Something unexpected!',trial_st,UTCDateTime(detc.iloc[now_idx[k][0]].starttime))
    n += 1
    trial_st += 15


#-----------------------------------------------------------------------------------------------------


looping detections
sav_all_arr = {}
for i in range(len(detc)):
    sta = detc.iloc[i].sta
    arr = UTCDateTime(detc.iloc[i].id.split('_')[-1])
    wid_st = UTCDateTime(detc.iloc[i].starttime)
    wid_ed = UTCDateTime(detc.iloc[i].endtime)
    if sta in sav_all_arr:
        sav_all_arr[sta].append(arr)
    else:
        sav_all_arr[sta] = [arr]







