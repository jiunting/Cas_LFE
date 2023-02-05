#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:09:06 2021

Script to get elevations of JESTER SITES and save to .mat file

@author: amt
"""
import pandas as pd
from scipy.io import savemat
import numpy as np
import datetime
import matplotlib.pyplot as plt
import math
import stingray_utils
from scipy.interpolate import griddata, RegularGridInterpolator
import xarray as xr

srControl=True   
srGeometry=True
srStation=True
srEvent=False
srModel=True
srElevation=True

# Model grid
# minlon,maxlon,minlat,maxlat=-124.5,-123,48.1,49.3 # desired grid
# define central lat and lon
midlon=(-124.5-123)/2
midlat=(48.1+49.3)/2
dx=dy=dz=1 # model node spacing
xoffset=-60.0
yoffset=-70.0
maxdep=60.0
xdist=120.0
ydist=140.0
#minlonact, minlatact= minlon, minlat
_,maxlon=stingray_utils.ll2dxdy((midlat,midlon),90,xdist/2*1000)
_,minlon=stingray_utils.ll2dxdy((midlat,midlon),270,xdist/2*1000)
maxlat,_=stingray_utils.ll2dxdy((midlat,midlon),0,ydist/2*1000)
minlat,_=stingray_utils.ll2dxdy((midlat,midlon),180,ydist/2*1000)
print(minlon,maxlon,minlat,maxlat)

if srControl:

    # -------------------------------------------------------------------
    # this makes srControl
    condict={}
    condict['tf_latlon']=1 
    condict['tf_anisotropy']=0 
    condict['tf_line_integrate']=1
    condict['arcfile']='arc7.mat'
    condict['tf_waterpath']=0
    condict['tf_carve']=1
    condict['carve']={}
    condict['carve']['buffer']=2
    condict['carve']['zvalue']=[[2]]
    savemat("/Users/amt/Documents/cascadia_lfe_locations/srInput/srControl_SVI.mat", {'srControl': condict})

if srGeometry:

    # -------------------------------------------------------------------
    # this makes srGeometry
    geodict={}
    geodict['longitude']=midlon
    geodict['latitude']=midlat
    geodict['rotation']=0.0
    geodict['tf_latlon']=True
    geodict['tf_flat']=0.0
    savemat("/Users/amt/Documents/cascadia_lfe_locations/srInput/srGeometry_SVI.mat", {'srGeometry': geodict})

if srStation:

    # -------------------------------------------------------------------
    # this makes srStation
    df=pd.read_csv("/Users/amt/Documents/cascadia_lfe_locations/vancouver_station_list.txt", header=None, names=['Network','Station','Lat','Lon','Elev(m)'], delimiter=' ')
    dfdict={}
    dfdict['name']=df['Station'].values.reshape(len(df),1)
    dfdict['latitude']=df['Lat'].values.reshape(len(df),1)
    dfdict['longitude']=df['Lon'].values.reshape(len(df),1)
    dfdict['elevation']=df['Elev(m)'].values.reshape(len(df),1)/1000
    savemat("/Users/amt/Documents/cascadia_lfe_locations/srInput/srStation_SVI.mat", {'srStation': dfdict})

if srEvent:

    # -------------------------------------------------------------------
    # this makes srEvent file
    eqdf=pd.read_csv("/Users/amt/Documents/StingrayGIT_BPV/StingrayGIT/amandaworking/earthquakes.csv")
    datetimes=pd.to_datetime(eqdf['DateTime'])
    origins=np.zeros((len(eqdf),1))
    for ii in range(len(eqdf)):
        origins[ii]=(datetimes-datetime.datetime.utcfromtimestamp(0)).values[ii].item()/1e9
    eqdict={}
    eqdict['id']=eqdf['EventID'].values.reshape(len(eqdf),1)
    eqdict['type']=3*np.ones((len(eqdf),1))
    eqdict['latitude']=eqdf['Latitude'].values.reshape(len(eqdf),1)
    eqdict['longitude']=eqdf['Longitude'].values.reshape(len(eqdf),1)
    eqdict['origintime']=origins
    eqdict['depth']=eqdf['Depth'].values.reshape(len(eqdf),1)
    eqdict['datum']=np.zeros((len(eqdf),1))
    savemat("/Users/amt/Documents/StingrayGIT_BPV/StingrayGIT/amandaworking/srInput/srEvent_SVI.mat", {'srEvent': eqdict})

if srModel:
    # load velmod
    mod=pd.read_csv('Savard_VpVs.txt', header=None, names=['longitude','latitude','depth','Vp','Vs','DWS'], delimiter=' ')   
    # define output model boundaries and other properties
    botdx=stingray_utils.distance((minlat,minlon),(minlat,maxlon))
    topdx=stingray_utils.distance((maxlat,minlon),(maxlat,maxlon))
    leftdy=stingray_utils.distance((minlat,minlon),(maxlat,minlon))
    rightdy=stingray_utils.distance((minlat,maxlon),(maxlat,maxlon))
    print('botdx: '+str(botdx))
    print('topdx: '+str(topdx))
    print('leftdy: '+str(leftdy))
    print('rightdy: '+str(rightdy))
    gridlength=np.max([botdx,topdx])
    gridheight=np.max([leftdy,rightdy])
    print('gridlength: '+str(gridlength))
    print('gridheight: '+str(gridheight))

    nx=int(np.ceil(xdist)//dx)
    ny=int(np.ceil(ydist)//dy)
    nz=int(maxdep//dz+1)
 
    # need to determine lats and lons at which output model points will be
    lats=np.zeros(ny)
    lons=np.zeros(nx)
    for ii in range(nx):
        _,lons[ii]=stingray_utils.ll2dxdy((minlat,minlon),90,dx*1e3*ii)
    for ii in range(ny):
        lats[ii],_=stingray_utils.ll2dxdy((minlat,minlon),0,dy*1e3*ii)
    depths=np.arange(0,maxdep+dz,dz)
    latgrid,longrid,depgrid=np.meshgrid(lats,lons,depths)
            
    # this plots the Savard model and grid
    fig,ax=plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(15,10))
    depth=12
    minimod=mod[mod['depth']==depth]
    ec=ax[0].scatter(minimod['longitude'].values, minimod['latitude'].values, s=20, c=minimod['Vp'].values, 
                  vmin=np.min([minimod['Vp'].values, minimod['Vs'].values]), vmax=np.max([minimod['Vp'].values, minimod['Vs'].values]))
    ax[1].scatter(minimod['longitude'].values, minimod['latitude'].values, s=20, c=minimod['Vs'].values, 
                  vmin=np.min([minimod['Vp'].values, minimod['Vs'].values]), vmax=np.max([minimod['Vp'].values, minimod['Vs'].values]))
    ax[0].plot(longrid[depgrid==1],latgrid[depgrid==1],'ko')
    ax[1].plot(longrid[depgrid==1],latgrid[depgrid==1],'ko')
    ax[0].title.set_text('P velocity')
    ax[1].title.set_text('S velocity')
    plt.colorbar(ec)
    
    # Savard model has depth=0 at sea level but Stingray needs depth=0 at elevation
    # need to interp elevaation at each output lat and lon in stingray grid
    f = '/Users/amt/Documents/cascadia_lfe_locations/svi_topo.nc'
    grid = xr.open_dataset(f)
    elevlon = grid['lon'].to_numpy()
    elevlat = grid['lat'].to_numpy()
    elev = grid['elevation'].to_numpy()    
    elev_interp = RegularGridInterpolator((elevlat,elevlon), elev, method="linear")
    elevgrid=elev_interp((np.ravel(latgrid[depgrid==0]),np.ravel(longrid[depgrid==0])))
    elevgrid=elevgrid.reshape(*latgrid.shape[:2])
    
    # this plots the elevation interpolation
    fig,ax=plt.subplots(nrows=1,ncols=1,sharey=True,figsize=(15,10))
    X,Y=np.meshgrid(elevlon,elevlat)
    ec=ax.scatter(X, Y, s=20, c=elev,vmin=0, vmax=1439)
    ax.scatter(longrid[depgrid==0], latgrid[depgrid==0], s=30, c=elevgrid, vmin=0, vmax=1439, edgecolors='black', zorder=10)
    # ax.plot(longrid[depgrid==0],latgrid[depgrid==0],'ko')
    ax.title.set_text('Elevation (m)')
    plt.colorbar(ec)
    
    # interp Savard velocities w/ griddata since data are irregular
    # cannot do a single interpolation because irregular grids dont allow you
    # to extrapolate
    Pslo=np.zeros((nx,ny,nz))
    Sslo=np.zeros((nx,ny,nz))
    points=(mod['longitude'].values,mod['latitude'].values,mod['depth'].values)
    newpoints=(np.ravel(longrid),np.ravel(latgrid),np.ravel(depgrid))
    pvals=griddata(points, mod['Vp'].values, newpoints) # p slowness
    svals=griddata(points, mod['Vs'].values, newpoints) # p slowness
    Pslo = 1/pvals.reshape(*longrid.shape) # p slowness
    Sslo = 1/svals.reshape(*longrid.shape) # s slowness
    
    # interp velocities w/ RegularGridInterpolator
    # necessary so that can extrap to depths above sea level
    points=(lons,lats,depths)
    Pinterp = RegularGridInterpolator(points, Pslo, bounds_error=False, fill_value=None)
    Sinterp = RegularGridInterpolator(points, Sslo, bounds_error=False, fill_value=None)
    outlats=latgrid.copy().reshape(-1)
    outlons=longrid.copy().reshape(-1)
    outdeps=depgrid.copy().reshape(-1)
    outdepcorr=np.zeros_like(outlats)
    Pslo_output=np.zeros_like(outdeps)
    Sslo_output=np.zeros_like(outdeps)
    print(depgrid[0,0,0])
    for ii in range(len(outlats)):
        print(ii)
        targlat, targlon=outlats[ii], outlons[ii]
        xp=np.where((latgrid==targlat) & (longrid==targlon) & (depgrid==0))[0]
        yp=np.where((latgrid==targlat) & (longrid==targlon) & (depgrid==0))[1]
        outdeps[ii]=outdeps[ii]+elevgrid[xp[0],yp[0]]/1000
    Pslo_output=Pinterp((outlons,outlats,outdeps))
    Sslo_output=Sinterp((outlons,outlats,outdeps))
    Pslo_output_cube=Pslo_output.reshape(*longrid.shape)
    Sslo_output_cube=Sslo_output.reshape(*longrid.shape)
    
    # this makes the model file
    fig,ax=plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(15,10))
    depth=42
    depthind=int(depth/dz)
    minimod=mod[mod['depth']==depth]
    ec=ax[0].scatter(minimod['longitude'].values, minimod['latitude'].values, s=50, c=minimod['Vp'].values, 
                  vmin=3, vmax=8, edgecolors='black', zorder=10)
    ax[1].scatter(minimod['longitude'].values, minimod['latitude'].values, s=50, c=minimod['Vs'].values, 
                  vmin=3, vmax=8, edgecolors='black', zorder=10)
    ax[0].scatter(longrid[:,:,depthind], latgrid[:,:,depthind], s=20, c=1/Pslo_output_cube[:,:,depthind], 
                  vmin=3, vmax=8)
    ax[1].scatter(longrid[:,:,depthind], latgrid[:,:,depthind], s=20, c=1/Sslo_output_cube[:,:,depthind], 
                  vmin=3, vmax=8)       
    ax[0].title.set_text('P velocity')
    ax[1].title.set_text('S velocity')
    #plt.colorbar(ec)
    ax[0].set_xlim((minlon-0.1,maxlon+0.2))
    ax[1].set_xlim((minlon-0.1,maxlon+0.2))
    ax[0].set_ylim((minlat-0.1,maxlat+0.2))
    ax[1].set_ylim((minlat-0.1,maxlat+0.2))
    
    # make dict
    modeldict={}
    modeldict['ghead']=[xoffset,yoffset,nx,ny,nz,dx,dy,dz]
    modeldict['P']={}
    modeldict['P']['u']=Pslo_output_cube
    modeldict['S']={}
    modeldict['S']['u']=Sslo_output_cube
    savemat("/Users/amt/Documents/cascadia_lfe_locations/srInput/srModel_SVI.mat", {'srModel': modeldict})       

if srElevation:
    # import pygmt
    # grid = pygmt.datasets.load_earth_relief(resolution='15s', region=reg) # this gets the dataset. region is [minlon, maxlon, minlat, maxlat]
    # grid.to_netcdf('msh_topo_15s.nc')

    f = '/Users/amt/Documents/cascadia_lfe_locations/svi_topo.nc'
    grid = xr.open_dataset(f)
    lon = grid['lon'].to_numpy()
    lat = grid['lat'].to_numpy()
    elev = grid['elevation'].to_numpy()
    x_min = np.min(lon)
    x_max = np.max(lon)
    y_min = np.min(lat)
    y_max = np.max(lat)
    dx = np.unique(np.diff(lon))[0]
    dy = np.unique(np.diff(lat))[0]
    nx = len(lon)
    ny = len(lat)
    elevdict = {}
    elevdict['header'] = [x_min,x_max,y_min,y_max,dx,dy,nx,ny]
    elevdict['data'] = elev.T
    # make sure this starts at min(lon)
    #velevdict['longitude'] = lon.reshape(1,len(lon))
    # make sure this starts at min(lat)
    # elevdict['latitude'] = lat.reshape(len(lat),1)
    savemat("/Users/amt/Documents/cascadia_lfe_locations/srInput/srElevation_SVI.mat", {'srElevation': elevdict})
    