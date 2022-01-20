import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


sav_sta = {}
sav_lon = []
sav_lat = []
stainfo = pd.read_csv('stations_all_4.dat',sep='\s+')
for i in range(len(stainfo)):
    #staname = '.'.join([stainfo.iloc[i]['NETWORK'],stainfo.iloc[i]['STATION']])  #network name is a mess.....
    staname = stainfo.iloc[i]['STATION'] #just use the station code, WARNING! could have same name but different station
    if not staname in sav_sta:
        sav_sta[staname] = [stainfo.iloc[i]['LON'],stainfo.iloc[i]['LAT'],stainfo.iloc[i]['ELEVATION']]
        sav_lon.append(stainfo.iloc[i]['LON'])
        sav_lat.append(stainfo.iloc[i]['LAT'])


np.save('stainfo.npy',sav_sta)
