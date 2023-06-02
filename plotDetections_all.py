# plot detection time for all the stations

import numpy as np
import matplotlib
matplotlib.use('pdf') #instead using interactive backend
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import unet_tools
import glob
from obspy import UTCDateTime
import datetime
import seaborn as sns


def dect_time(file_path: str, thresh=0.1, is_catalog: bool=False, EQinfo: str=None, return_all: bool=False) -> np.ndarray:
    """
    Get detection time for 1). ML detection in .csv file or 2). catalog file.
    If input is .csv file, return the `window` starttime so that is easier to match with other stations
    
    INPUTs
    ======
    file_path: str
        Path of the detection file
    is_catalog: bool
        Input is catalog or not
    thresh: float
        Threshold of selection. Only apply when is_catalog==False (i.e. using ML detection csv files)
    EQinfo: str
        Path of the EQinfo file i.e. "sav_family_phases.npy"
    return_all: bool
        Also return deection time, not just the window starttime
    OUTPUTs
    =======
    sav_OT: np.array[datetime] or None if empty detection
        Detection time in array. For ML detection, time is the arrival at the station. For catalog, time is the origin time
    
    EXAMPLEs
    ========
    #T1 = dect_time('./Data/total_mag_detect_0000_cull_NEW.txt',True,'./Data/sav_family_phases.npy')  #for catalog
    #T2 = dect_time('./results/Detections_S_small/cut_daily_PO.GOWB.csv')                             # for ML detection output
    """
    if is_catalog:
        EQinfo = np.load(EQinfo,allow_pickle=True) #Note! detection file has a shift for each family
        EQinfo = EQinfo.item()
        sav_OT = []
        with open(file_path,'r') as IN1:
            for line in IN1.readlines():
                line = line.strip()
                ID = line.split()[0] #family ID
                OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
                HH = int(line.split()[2])*3600
                SS = float(line.split()[3])
                OT = OT + HH + SS + EQinfo[ID]['catShift'] #detected origin time
                sav_OT.append(OT.datetime)
        sav_OT.sort()
        return pd.Series(sav_OT)
        #sav_OT = np.array(sav_OT)
        #return sav_OT
    else:
        csv = pd.read_csv(file_path)
        if len(csv)==0:
            return None
        T = csv[csv['y']>=thresh]
        if return_all:
            return T.starttime.values, np.array([i.split('_')[-1] for i in T.id])
        else:
            return T.starttime.values
        #net_sta = file_path.split('/')[-1].split('_')[-1].replace('.csv','')
        #prepend = csv.iloc[0].id.split('_')[0]
        #T = pd.to_datetime(csv[csv['y']>=thresh].id,format=prepend+'_%Y-%m-%dT%H:%M:%S.%f')
        #T.reset_index(drop=True, inplace=True) #reset the keys e.g. [1,5,6,10...] to [0,1,2,3,...]
        return T


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


"""
# Old function is inaccurate when T breaks more than 2 days!!
def get_daily_nums(T):
    # get number of events each day
    # T is the sorted timeseries of the occurrence
    T0 = datetime.datetime(T[0].year,T[0].month,T[0].day)
    T1 = T0 + datetime.timedelta(1)
    sav_num = [] # number of events in a day
    sav_T = []
    n = 1 #set each day number start from 1 to keep daily number continuous, so remember to correct it in the final
    for i in T:
        if T0<=i<T1:
            n += 1
        else:
            if n!=0:
                sav_T.append(T0+datetime.timedelta(0.5))
                sav_num.append(n)
            #update T0,T1 to next day
            T0 = T1
            T1 = T0 + datetime.timedelta(1)
            n = 1
    sav_T.append(T0+datetime.timedelta(0.5))
    sav_num.append(n)
    sav_num = np.array(sav_num)
    sav_num = sav_num-1 #daily number correct by 1
    return np.array(sav_T),np.array(sav_num)
"""

# load original detection file (from template matching)
# load family and arrival information
EQinfo = np.load('sav_family_phases.npy',allow_pickle=True) #Note! detection file has a shift for each family
EQinfo = EQinfo.item()
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

cat_time,cat_daynums = get_daily_nums(sorted(sav_OT_template))

# load tremor
tremor = pd.read_csv("tremor_events-2009-08-06T00_00_00-2014-12-31T23_59_59.csv")
tremor = tremor[(tremor['lon']>=-124.5) & (tremor['lon']<=-123) & (tremor['lat']>=48.1) & (tremor['lat']<=49.3)]
tremor_t = [UTCDateTime(tt).datetime for tt in tremor['starttime']]
trem_time,trem_daynums = get_daily_nums(sorted(tremor_t),15)
trem_time = np.array([utc_to_decimal_year(UTCDateTime(i)) for i in trem_time])

# load model detection .csv file
#csvs = glob.glob('./Detections_S_new/*.csv')
#csvs = glob.glob('./Detections_P/*.csv')
csvs = glob.glob('./Detections_S_new/*.csv')
#csvs = glob.glob('./Detections_S_C8_new/*.csv')
#csvs = glob.glob('./Detections_P_new/*.csv')

thres = 0.1 #plot everything >= thres

#PO.SILB.HH_2005-01-01T16:53:28.41
sav_detcTime = {}
for csv_path in csvs:
    csv = pd.read_csv(csv_path)
    if len(csv)==0:
        continue
    netSta = csv_path.split('/')[-1].split('_')[-1].replace('.csv','')
    print('Now reading:',netSta)
    prepend = csv.iloc[0].id.split('_')[0]
    sav_detcTime[netSta] = pd.to_datetime(csv[csv['y']>=thres].id,format=prepend+'_%Y-%m-%dT%H:%M:%S.%f')
    sav_detcTime[netSta].reset_index(drop=True, inplace=True) #reset the keys e.g. [1,5,6,10...] to [0,1,2,3,...]

# plotting detection timeseries 
sns.set()
plt.figure()
#plt.plot(sav_OT_template,np.zeros(len(sav_OT_template)),'k.',markersize=1)
#plt.plot(cat_time,(cat_daynums/max(np.abs(cat_daynums)))*0.5,'-',color=[0.5,0.5,0.5])
plt.fill_between(cat_time,(cat_daynums/max(np.abs(cat_daynums)))*0.8,0,color=[0.5,0.5,0.5])
plt.text(max(sav_OT_template),0,'Catalog',fontsize=10)
for n,k in enumerate(sav_detcTime.keys()):
    print(' plotting k=%s, len=%d'%(k,len(sav_detcTime[k])))
    #plt.plot(sav_detcTime[k],np.ones(len(sav_detcTime[k]))*(n+1),'.',markersize=1)
    sav_T,sav_daynums = get_daily_nums(sav_detcTime[k]) #get daily number tcs
    h = plt.fill_between(sav_T,(sav_daynums/max(np.abs(sav_daynums)))*0.8+(n+1),(n+1),linewidth=0.1)
    h.set_edgecolor(h.get_facecolor())
    if k=='CN.PFB':
        plt.text(max(sav_detcTime[k]),n+1,k,ha='center',fontsize=10)
    else:
        plt.text(max(sav_detcTime[k]),n+1,k,fontsize=10)
    
plt.yticks([],[])
plt.xlabel('Time (year)',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=1.0,length=1.5,size=1.5,labelsize=12) 
#plt.xlim([datetime.datetime(2005,1,1), datetime.datetime(2005,1,30)])

#plt.savefig('detection_tcs_all_y%.1f.png'%(thres))
#plt.savefig('detection_tcs_all_y%.1f.pdf'%(thres))
#plt.savefig('detection_new_tcs_all_y%.1f.png'%(thres))
#plt.savefig('detection_new_tcs_all_y%.1f.pdf'%(thres))

#plt.savefig('detection_tcs_all_P_y0.1.png')
#plt.savefig('detection_tcs_all_P_y0.1.pdf')
#plt.savefig('detection_tcs_all_S_new_y0.1_1005.png',dpi=450)
plt.savefig('detection_tcs_all_S_new_y0.1_050523.png',dpi=450)
#plt.savefig('detection_tcs_all_S_C8_new_y0.1_1005.png',dpi=300)
#plt.savefig('detection_tcs_all_P_new_y0.1_0906.png',dpi=300)
plt.close()




# 2. plot merged detection timeseries (copy from find_template.py)
#=====get the number of detections in the same starttime window=====
#=====future improvement: consider nearby windows i.e. +-15 s
sav_T = {} # net.sta:starttime
#st2detc = {} #net.sta: st->dect
for csv in csvs:
    print('Now in :',csv)
    net_sta = csv.split('/')[-1].split('_')[-1].replace('.csv','')
    T = dect_time(csv, thresh=0.1)
    #T, Td = dect_time(csv, thresh=0.1, return_all=True)
    if T is None:
        continue
    sav_T[net_sta] = T
    #st2detc[net_sta] = {T[i]:Td[i] for i in range(len(T))}

print('-----Initialize all_T. Min=1, Max=%d'%(len(sav_T)))
all_T = {}
for k in sav_T.keys():
    print('dealing with:',k)
    for t in sav_T[k]:
        if t not in all_T:
            all_T[t] = {'num':1, 'sta':[k]} #['num'] = 1
        else:
            all_T[t]['num'] += 1
            all_T[t]['sta'].append(k)

#=====Min station filter and get daily detction number=====
N_min = 3
sav_k = [] # keys that pass the filter
for k in all_T.keys():
    if all_T[k]['num']>=N_min:
        sav_k.append(k)

sav_k.sort()
sav_k = np.array([UTCDateTime(i) for i in sav_k])
detc_time, detc_nums = get_daily_nums(sav_k) # daily detections from all the stations if pass the N_min filter


plt.fill_between(detc_time,detc_nums/max(np.abs(detc_nums)),0,color=[1,0.5,0.5])
plt.fill_between(cat_time,cat_daynums/max(np.abs(cat_daynums)),0,color='k',alpha=0.8)
plt.legend(['Model','Catalog'])
plt.xlabel('Time (year)',fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=1.0,length=1.5,size=1.5,labelsize=12)
#plt.savefig('detection_tcs_merged_S_new_y0.1_1005.png',dpi=300)
plt.savefig('detection_tcs_merged_S_new_y0.1_050523.png',dpi=300)
#plt.savefig('detection_tcs_merged_S_C8_new_y0.1_1005.png',dpi=300)
plt.close()



# 3. plot accumulated number
Tst = max(min(cat_time),min(detc_time))
Tend = min(max(cat_time),max(detc_time))
idx_cat_st = np.where(cat_time==Tst)[0][0]
idx_detc_st = np.where(detc_time==Tst)[0][0]
idx_cat_end = np.where(cat_time==Tend)[0][0]
idx_detc_end = np.where(detc_time==Tend)[0][0]
idx_trem_end = np.where(trem_time<=Tend)[0][-1]

h1 = plt.plot(detc_time[idx_detc_st:idx_detc_end+1], np.cumsum(detc_nums[idx_detc_st:idx_detc_end+1]), 'r', lw=3, label='Model')
h2 = plt.plot(cat_time[idx_cat_st:idx_cat_end+1], np.cumsum(cat_daynums[idx_cat_st:idx_cat_end+1]), 'k', lw=3, label='Catalog')
#h2 = plt.plot(cat_time, np.cumsum(cat_daynums), 'k', lw=3, label='Catalog')
#plt.legend()
plt.ylabel('Cumulative LFE Number', fontsize=14,labelpad=0)
plt.xlabel('Time (year)', fontsize=14,labelpad=0)
ax1=plt.gca()
ax1.tick_params(pad=1.0,length=1.5,size=1.5,labelsize=12)
ax1.grid(axis='y')
# add tremor plot
ax2 = ax1.twinx()
ax2.set_xlim(ax1.get_xlim())
ax2.grid(False)
#h3 = plt.plot(trem_time, np.cumsum(trem_daynums), 'b', lw=3, label='Tremor')
h3 = plt.plot(trem_time[:idx_trem_end], np.cumsum(trem_daynums[:idx_trem_end]), 'b', lw=3, label='Tremor')
ax2.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
ax2.spines['right'].set_color('b') 
ax2.tick_params(axis='y', colors='b')
ax2.yaxis.label.set_color('blue') 
ax2.tick_params(pad=1.0,length=1.5,size=1.5,labelsize=12)
plt.ylabel('Cumulative Tremor Number',fontsize=14,labelpad=0,color='b')
plt.legend([h1[0],h2[0],h3[0]],['Model','Catalog','Tremor'])
#plt.savefig('detection_tcs_merged_cum_S_new_y0.1_1026.png',dpi=450)
plt.savefig('detection_tcs_merged_cum_S_new_y0.1_050523.png',dpi=450)
#plt.savefig('detection_tcs_merged_cum_S_C8_new_y0.1_1005.png',dpi=300)
plt.close()







