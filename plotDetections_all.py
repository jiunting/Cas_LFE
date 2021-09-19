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


def get_daily_nums(T):
    # get number of events each day
    # T is the sorted timeseries of the occurrence
    T0 = datetime.datetime(T[0].year,T[0].month,T[0].day)
    T1 = T0 + datetime.timedelta(1)
    sav_num = [] # number of events in a day
    sav_T = []
    n = 0
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
    return np.array(sav_T),np.array(sav_num)


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

# load model detection .csv file
csvs = glob.glob('./Detections_S/*.csv')
#csvs = glob.glob('./Detections_P/*.csv')

thres = 0.5 #plot everything >= thres

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
plt.text(max(sav_OT_template),0,'catalog',fontsize=10)
for n,k in enumerate(sav_detcTime.keys()):
    print(' plotting k=%s, len=%d'%(k,len(sav_detcTime[k])))
    #plt.plot(sav_detcTime[k],np.ones(len(sav_detcTime[k]))*(n+1),'.',markersize=1)
    sav_T,sav_daynums = get_daily_nums(sav_detcTime[k]) #get daily number tcs
    h = plt.fill_between(sav_T,(sav_daynums/max(np.abs(sav_daynums)))*0.8+(n+1),(n+1),linewidth=0.1)
    h.set_edgecolor(h.get_facecolor())
    plt.text(max(sav_detcTime[k]),n+1,k,fontsize=10)
    
plt.yticks([],[])

plt.savefig('detection_tcs_all_y%.1f.png'%(thres))
plt.savefig('detection_tcs_all_y%.1f.pdf'%(thres))
#plt.savefig('detection_tcs_all_P_y0.1.png')
plt.close()






