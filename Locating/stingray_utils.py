#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 09:02:42 2021

Various utilities for stingray

@author: amt
"""

import math
import scipy.io as sio
import numpy as np
from geographiclib.geodesic import Geodesic
import h5py

def distance(origin, destination):
    
    """
    A script to determine distance in km between two coordinates

    Parameters
    ----------
    origin : float
        lat, lon pair of origin
    destination : float
        lat, lon pair of destination

    Returns
    -------
    d : float
        distance between origin and destination in km 
    """
    
    geod = Geodesic.WGS84 
    g = geod.Inverse(origin[0], origin[1], destination[0], destination[1])

    return g['s12']/1000

def ll2dxdy(origin,bearing,dist):
    """
    A script to determine latitude and longitude from lat0, lon0, bearing and distance in km

    Parameters
    ----------
    origin : float
        lat, lon pair of origin
    bearing : float
        heading in degrees E of N
    dist : float
        distance in kilometers

    Returns
    -------
    latitude and longitude at new position

    """
    geod = Geodesic.WGS84
    g = geod.Direct(origin[0], origin[1], bearing, dist)
    return g['lat2'], g['lon2']

def srStation2dict(file):
    
    """
    mat2dict Converts srStation.mat to python dictionary

    Parameters
    ----------
    file : str
        path to stingray .mat file

    Returns
    -------
    srStation : dictionary
        dictionary of srRays contents 
    """
    
    stadict=sio.loadmat(file)
    srStation={}
    srStation['nsta']=stadict['srStation']['nsta'][0][0][0][0]
    tmp=[]
    for ii in range(srStation['nsta']):
        tmp.append(stadict['srStation']['name'][0][0][ii][0][0])
    srStation['name']=tmp
    srStation['latitude']=stadict['srStation']['latitude'][0][0].flatten()
    srStation['longitude']=stadict['srStation']['longitude'][0][0].flatten()
    srStation['elevation']=stadict['srStation']['elevation'][0][0].flatten()
    srStation['x']=stadict['srStation']['x'][0][0].flatten()
    srStation['y']=stadict['srStation']['y'][0][0].flatten()
    srStation['filename']=stadict['srStation']['filename'][0][0][0]
    srStation['statics_P']=stadict['srStation']['statics'][0][0][0][0][0].flatten()
    srStation['statics_S']=stadict['srStation']['statics'][0][0][0][0][1].flatten()
    
    return srStation
    
def srRays2dict(file):
    
    """
    mat2dict Converts srRays.mat travel time files to python dictionaries

    Parameters
    ----------
    file : str
        path to stingray .mat file

    Returns
    -------
    srrays : dictionary
        dictionary of srRays contents 
    """
    srrays={}
    # try to load regular mat file
    try:
        ttdict=sio.loadmat(file)      
        srrays['phasetype']=ttdict['srRays']['phasetype'][0][0][0][0]
        srrays['phase']=ttdict['srRays']['phase'][0][0][0]
        srrays['model']=ttdict['srRays']['model'][0][0][0][0][0]
        srrays['ghead']=ttdict['srRays']['ghead'][0][0][0][:]
        srrays['time']=ttdict['srRays']['time'][0][0] 
        srrays['iprec']=ttdict['srRays']['iprec'][0][0] 
        srrays['nx']=ttdict['srRays']['nx'][0][0][0][0]
        srrays['ny']=ttdict['srRays']['ny'][0][0][0][0]
        srrays['nz']=ttdict['srRays']['nz'][0][0][0][0]
        srrays['gx']=ttdict['srRays']['gx'][0][0][0][0]
        srrays['gy']=ttdict['srRays']['gy'][0][0][0][0]
        srrays['gz']=ttdict['srRays']['gz'][0][0][0][0]
        srrays['nodes']=ttdict['srRays']['nodes'][0][0][0][0]
        srrays['xg']=ttdict['srRays']['xg'][0][0].flatten()
        srrays['yg']=ttdict['srRays']['yg'][0][0].flatten()
        srrays['zg']=ttdict['srRays']['zg'][0][0].flatten()
        srrays['maxx']=ttdict['srRays']['minx'][0][0][0][0]
        srrays['maxy']=ttdict['srRays']['miny'][0][0][0][0]
        srrays['maxz']=ttdict['srRays']['minz'][0][0][0][0]
        srrays['minx']=ttdict['srRays']['minx'][0][0][0][0]
        srrays['miny']=ttdict['srRays']['miny'][0][0][0][0]
        srrays['minz']=ttdict['srRays']['minz'][0][0][0][0]
        srrays['elevation']=ttdict['srRays']['elevation'][0][0][0][:]
        srrays['longitude']=ttdict['srRays']['srGeometry'][0][0][0][0][0][0][0]
        srrays['latitude']=ttdict['srRays']['srGeometry'][0][0][0][0][1][0][0]
        srrays['rotation']=ttdict['srRays']['srGeometry'][0][0][0][0][2][0][0]
        srrays['tf_latlon']=ttdict['srRays']['srGeometry'][0][0][0][0][3][0][0]
        srrays['tf_flat']=ttdict['srRays']['srGeometry'][0][0][0][0][4][0][0]
        srrays['Re']=ttdict['srRays']['srGeometry'][0][0][0][0][5][0][0]
        srrays['filename']=ttdict['srRays']['srGeometry'][0][0][0][0][6][0][:]
        srrays['modelname']=ttdict['srRays']['modelname'][0][0][0]
        # TODO: add srControl
    except:
        ttdict=h5py.File(file,'r')
        srrays['phasetype']=ttdict['srRays']['phasetype'][:].flatten()
        srrays['phase']=ttdict['srRays']['phase'][:].flatten()
        #srrays['model']=ttdict['srRays']['model'][0][0][0][0][0]
        srrays['ghead']=ttdict['srRays']['ghead'][:].flatten()
        srrays['time']=ttdict['srRays']['time']
        srrays['iprec']=ttdict['srRays']['iprec']
        srrays['nx']=ttdict['srRays']['nx'][0][0]
        srrays['ny']=ttdict['srRays']['ny'][0][0]
        srrays['nz']=ttdict['srRays']['nz'][0][0]
        srrays['gx']=ttdict['srRays']['gx'][0][0]
        srrays['gy']=ttdict['srRays']['gy'][0][0]
        srrays['gz']=ttdict['srRays']['gz'][0][0]
        srrays['nodes']=ttdict['srRays']['nodes'][0][0]
        srrays['xg']=ttdict['srRays']['xg'][0]
        srrays['yg']=ttdict['srRays']['yg'][0]
        srrays['zg']=ttdict['srRays']['zg'][0]
        srrays['maxx']=ttdict['srRays']['minx'][0][0]
        srrays['maxy']=ttdict['srRays']['miny'][0][0]
        srrays['maxz']=ttdict['srRays']['minz'][0][0]
        srrays['minx']=ttdict['srRays']['minx'][0][0]
        srrays['miny']=ttdict['srRays']['miny'][0][0]
        srrays['minz']=ttdict['srRays']['minz'][0][0]
        srrays['elevation']=ttdict['srRays']['elevation']
        srrays['longitude']=ttdict['srRays']['srGeometry']['longitude'][0][0]
        srrays['latitude']=ttdict['srRays']['srGeometry']['latitude'][0][0]
        srrays['rotation']=ttdict['srRays']['srGeometry']['rotation'][0][0]
        srrays['tf_latlon']=ttdict['srRays']['srGeometry']['tf_latlon'][0][0]
        srrays['tf_flat']=ttdict['srRays']['srGeometry']['tf_flat'][0][0]
        srrays['Re']=ttdict['srRays']['srGeometry']['Re'][0][0]
        # srrays['filename']=ttdict['srRays']['srGeometry'][0][0][0][0][6][0][:]
        # srrays['modelname']=ttdict['srRays']['modelname'][0][0][0]
        # TODO: add srControl
    return srrays

def xy2map(xlc,ylc,ttdict):
    ''' xy2map Convert local cartesian to geodetic or UTM coordinates
    %           (Stingray utility)
    %
    %
    %  Local cartesian reference frame can be rotated with respect to latitude
    %  and longitude or with respect to easting and northing.
    %
    % INPUT
    %          xlc: x value in local cartesian, km
    %          ylc: y value in local cartesian, km
    %          ttdict: Stingray structure
    %
    % OUTPUT
    %         mapx: either decimal longitude or easting
    %         mapy: either decimal latitude or northing
    %
    % NOTES
    % + B. VanderBeek re-wrote to improve accuracy (AUG-2019)
    % --> The regular srModel grid is assumed to represent points that conform
    %     to Earth's curvature with z being the radial direction. Previous
    %     mapping assumed true cartesian coordinates (i.e. as you increase
    %     distance from origin the Earth's surface falls away from you). This
    %     becomes inaccurate for large models.
    % --> Distances and bearings at Earth's surface are calculated using the
    %     the x,y cartesian srModel coordinates and used to derive geographic
    %     coordinates using Matlab's mapping tools
    % --> Of course, distances in srModel are distorted with depth. This can be
    %     account for using the Earth flattening transform (see FlatEarth.m)
    % --> New mapping improves 1D teleseismic travel-time agreement with IASP91
    %
    %  Copyright 2010 Blue Tech Seismics, Inc. '''
    
    # Adjust for rotation. Rotation angle is measured counterclockwise positive
    # in degrees
    xe = np.cos(np.deg2rad(ttdict['rotation']))*xlc - np.sin(np.deg2rad(ttdict['rotation']))*ylc
    yn = np.sin(np.deg2rad(ttdict['rotation']))*xlc + np.cos(np.deg2rad(ttdict['rotation']))*ylc
    
    if ttdict['tf_latlon'] == 1:
        # Get arc-distances and bearings to srModel points
        D = (180/(ttdict['Re']*np.pi))*np.sqrt((xe**2) + (yn**2))
        AZ = np.rad2deg(np.arctan2(xe,yn))
        AZ[D==0] = 0 # Necessary?
        AZ = AZ % 360 #  = wrapTo360(AZ);
        # Get geographic position from distances and bearings
        #[mapy,mapx] = reckon(ttdict['latitude'],ttdict['longitude'],D,AZ)
        geod = Geodesic.WGS84 
        mapx=np.zeros(len(AZ))
        mapy=np.zeros(len(AZ))
        for ii in range(len(mapx)):
            g = geod.ArcDirect(ttdict['latitude'],ttdict['longitude'], AZ[ii], D[ii])
            mapx[ii], mapy[ii]=g['lon2'], g['lat2']
        
        # Check mapping
        if any(np.isnan(mapx)) or any(np.isnan(mapy)):
            raise RuntimeError('NaN''s in mapping from cartesian to geographic coordinates')
        
    elif ttdict['tf_latlon'] == 0:
        # UTM coordinates
        mapx = ttdict['easting'] + xe
        mapy = ttdict['northing'] + yn
    return mapx, mapy

def map2xy(mapx,mapy,srGeometry,ttdict):

    # % MAP2XY -- Convert lat/lon positions to local cartesian
    # %           (Stingray utility)
    # %
    # %  [x,y] = map2xy(mapx,mapy,srGeometry)
    # %
    # %  Returns cartesian values in kilometers relative to the geographic center
    # %  of the experiment found in srGeometry.  Cartesian reference frame can be
    # %  rotated with respect to latitude and longitude or with respect to
    # %  easting and northing.  
    # %
    # % INPUT
    # %   mapx: either decimal longitude or easting
    # %   mapy: either decimal latitude or northing
    # %
    # % OUTPUT
    # %      x: x value in local cartesian, km
    # %      y: y value in local cartesian, km
    # %
    # %
    # % NOTES
    # % + B. VanderBeek re-wrote to improve accuracy (AUG-2019)
    # % --> The regular srModel grid is assumed to represent points that conform
    # %     to Earth's curvature with z being the radial direction. Previous
    # %     mapping assumed true cartesian coordinates (i.e. as you increase
    # %     distance from origin the Earth's surface falls away from you). This
    # %     becomes inaccurate for large models.
    # % --> Distances and bearings at Earth's surface are calculated using the
    # %     the x,y cartesian srModel coordinates and used to derive geographic
    # %     coordinates using Matlab's mapping tools
    # % --> Of course, distances in srModel are distorted with depth. This can be
    # %     account for using the Earth flattening transform (see FlatEarth.m)
    # % --> New mapping improves 1D teleseismic travel-time agreement with IASP91
    # %
    # %  Copyright 2010 Blue Tech Seismics, Inc.


    if ttdict['tf_latlon'] == 1:
        # Convert arc distances and bearings to cartesian distances
        [D,AZ] = distance(ttdict['latitude'],ttdict['longitude'],mapy,mapx)
        D      = D*ttdict['Re']*np.pi/180
        xe     = D*np.sin(np.deg2rad(AZ))
        yn     = D*np.cos(np.deg2rad(AZ))
        
        # Check mapping
        if np.any(np.isnan(xe)) or np.any(np.isnan(yn)):
            raise RuntimeError('NaN''s in mapping from geographic coordinates to cartesian')
        
    elif srGeometry.tf_latlon == 0:
        # UTM coordinates
        xe = mapx - ttdict['easting']
        yn = mapy - ttdict['northing']
    
    # Adjust for rotation. Rotation angle is measured counterclockwise positive
    # in degrees.
    xlc = np.cos(np.deg2rad(-ttdict['rotation']))*xe - np.sin(np.deg2rad(-ttdict['rotation']))*yn
    ylc = np.sin(np.deg2rad(-ttdict['rotation']))*xe + np.cos(np.deg2rad(-ttdict['rotation']))*yn
    return xlc, ylc 

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n