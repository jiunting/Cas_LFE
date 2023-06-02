#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:19:26 2020

Converts srRays files to smaller .pkl files for I/O purposes

@author: amt
"""

import pickle
import pandas as pd
import numpy as np
import scipy.io as sio
import mat73
import matplotlib.pyplot as plt
import stingray_utils
import glob

staname = 'VGZ'
phase = 'S'


file = "/Users/jtlin/Documents/Project/Cas_LFE/Locating/svi_lfes/srOutput/srRays_%s_%s.mat"%(staname,phase)
refsta = mat73.loadmat(file)
reftts = refsta['srRays']['time'] # shape of x,y,z = 120,140,61. Centered at (-123.75, 48.7) with spacing=1km. Z=0~60km.

iz = 10 #check the ith depth

plt.subplot(1,2,1)
plt.pcolor(refsta['srRays']['xg'],refsta['srRays']['yg'],reftts[:,:,iz].T,cmap='plasma')
plt.title('%s (depth=%.1f km)'%(staname,refsta['srRays']['zg'][iz]),fontsize=14)
plt.xlabel('X (km)',fontsize=14)
plt.ylabel('Y (km)',fontsize=14)
plt.colorbar(label='time(s)')
#plt.show()


#quick plot and check
#plt.plot(refsta['srRays']['xg'],reftts[:,0,:])
#plt.xlabel('X (km)',fontsize=14)
#plt.ylabel('Time (s)',fontsize=14)
#plt.show()


# convert x/y to lon/lat
# step 1. enumerate every x, looping y
lon = []
lat = []
for iy in refsta['srRays']['yg']:
    ilons, ilats = stingray_utils.xy2map(refsta['srRays']['xg'], iy, refsta['srRays']['srGeometry'])
    lon.append(ilons)
    lat.append(ilats)

lon = np.array(lon)
lat = np.array(lat)

plt.subplot(1,2,2)
plt.pcolor(lon,lat,reftts[:,:,iz].T,cmap='plasma')
plt.title('%s (depth=%.1f km)'%(staname,refsta['srRays']['zg'][iz]),fontsize=14)
plt.xlabel('lon',fontsize=14)
plt.ylabel('lat',fontsize=14)
plt.colorbar(label='time(s)')
plt.show()

"""
@author: Tim, Jan 14 2023
Shape of lon=lat=(140, 120). e.g. lon[0] = the 0th row on the map (i.e. varing longitudes, fixing latitude)
Note that the longitudes and latitudes are slightly difference because of the earth's geometry
Want the travel time from lon/lat/depth to every "staname" with both P and S phase

"""

def xy2lonlat(refsta):
    """
    INPUTs:
        xg: from refsta['srRays']['xg'], 1D array
        yg: from refsta['srRays']['yg'], 1D array
    OUTPUTs:
        lon, lat: 2D array
            Looping the lon, fix lat first e.g. lon[0] has the same/slightly different lat
    """
    lon = []
    lat = []
    for iy in refsta['srRays']['yg']:
        ilons, ilats = stingray_utils.xy2map(refsta['srRays']['xg'], iy, refsta['srRays']['srGeometry'])
        lon.append(ilons)
        lat.append(ilats)
    return np.array(lon), np.array(lat)
    

# Save all the travel times (from grid nodes to stations)
Travel = {'T':{},'sta_phase':[]} #travel time from which grid to what station/phase
stanames = glob.glob("/Users/jtlin/Documents/Project/Cas_LFE/Locating/svi_lfes/srOutput/srRays_*_S.mat")
stanames = [ista.split('/')[-1].split('_')[1] for ista in stanames]
for ip,phase in enumerate(['P','S']):
    for i in range(len(stanames)):
        staname = stanames[i]
        print('In sta:',staname)
        file = "/Users/jtlin/Documents/Project/Cas_LFE/Locating/svi_lfes/srOutput/srRays_%s_%s.mat"%(staname,phase)
        refsta = mat73.loadmat(file)
        reftts = refsta['srRays']['time']
        if i==0 and ip==0:
            lon, lat = xy2lonlat(refsta) # only run once since all the grid nodes (lon/lat/dep) are the same
        #lon.shape=lat.shape=(140,120)
        for ilat in range(lat.shape[0]): #0~139
        #for ilat in range(lat.shape[0])[::5]: #0~139
            for ilon in range(lon.shape[1]): #0~119
            #for ilon in range(lon.shape[1])[::5]: #0~119
                for idep,dep in enumerate(refsta['srRays']['zg']):
                #for idep,dep in enumerate(refsta['srRays']['zg'][::5]):
                #for idep,dep in enumerate(refsta['srRays']['zg'][25::5]):
                    #print(lon[ilat,ilon],lat[ilat,ilon],dep,'Travel=%f'%(reftts[ilon,ilat,idep])) #note the shape are different!
                    if (lon[ilat,ilon],lat[ilat,ilon],dep) in Travel['T']:
                        Travel['T'][(lon[ilat,ilon],lat[ilat,ilon],dep)].append(reftts[ilon,ilat,idep]) #append a time at this grid node
                    else:
                        Travel['T'][(lon[ilat,ilon],lat[ilat,ilon],dep)] = [reftts[ilon,ilat,idep]]
        # save the travel time order so that you know what station/phase is for in Travel['T'][lon,lat,dep]
        Travel['sta_phase'].append('_'.join([staname,phase]))

# convert list to array
for ig in Travel['T'].keys():
    Travel['T'][ig] = np.array(Travel['T'][ig])

# Finally save the travel time
np.save('Travel.npy',Travel)
#np.save('Travel_reduced.npy',Travel)
#np.save('Travel_reduced_25km.npy',Travel)

# -----------Filter the Travel by slab geometry--------------
coords = np.array([coord for coord in Travel['T'].keys()]) # all the coords

slab = np.genfromtxt('cas_slab_rmNan.xyz')
idx=np.where((slab[:,0]>=-124.5) & (slab[:,0]<=-123) & (slab[:,1]>=48.1) & (slab[:,1]<=49.3))[0]
slab2 = slab[idx]
# make a plot to see the geometry
ax = plt.figure().add_subplot(projection='3d')
ax.plot(slab2[:,0],slab2[:,1],-slab2[:,2],'.')
plt.show()

sav_idx = []
for i in range(slab2.shape[0]):
    d = ((coords[:,0]-slab2[i,0])**2 + (coords[:,1]-slab2[i,1])**2 + (-coords[:,2]-slab2[i,2])**2) ** 0.5
    sav_idx.append(np.argmin(d))
    
coords2 = coords[sav_idx]
# plot and check
ax = plt.figure().add_subplot(projection='3d')
ax.plot(slab2[:,0],slab2[:,1],-slab2[:,2],'k.', label='slab')
ax.plot(coords2[:,0],coords2[:,1],coords2[:,2],'r.',label='closest node to slab')
plt.legend()
plt.show()

Travel2 = {'sta_phase':Travel['sta_phase']}
Travel2['T'] = {tuple(c):Travel['T'][tuple(c)] for c in coords2}

# save the travel time
np.save('Travel_slab.npy',Travel2)


# -----------Filter the Travel by slab geometry END--------------

# END here

#
#geom = sio.loadmat("/Users/jtlin/Documents/Project/Cas_LFE/Locating/svi_lfes/srInput/srGeometry_SVI.mat")


"""
# load positions
stas = pd.read_csv('crackattack_r1_positions.dat', sep=' ',header=None)
stas.columns=['deployment','station','nodeid','latitude','longitude','utmy','utmx'] 
stas=stas.sort_values(by=['latitude'],ascending=False)
stas=stas.reset_index(drop=True)
stas=stas.drop([22])
stas=stas.drop([29])

for refstaid in stas['station']:
    print(refstaid)
    file='/Users/amt/Documents/rattlesnake_ridge/ray_tracing/srRays_'+str(refstaid)+'.mat'
    refsta=sio.loadmat(file)
    #reftts=np.empty((151,251,76))
    reftts=refsta['srRays']['time'][0][0] 
    reftts=reftts[:,:,:40]
    with open('/Users/amt/Documents/rattlesnake_ridge/ray_tracing/tts_'+str(refstaid)+'.pkl', 'wb') as f:
        pickle.dump(reftts, f)
"""
