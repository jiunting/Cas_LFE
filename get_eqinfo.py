#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:46:45 2021

@author: timlin
"""

#get origin time from total_mag_detect_0000_cull_NEW.txt
#get phase from HypoInv Arc file

import numpy as np
from obspy import UTCDateTime

fileName = 'lfe_svi.arc'

line = '230001010000186948 3078123 5527 3294  0 24215 11   935816  9220970  70  0     34   13  88  66 12  00  00  0  0GSC      25    0 00   0 00         0   0  00   0  00  NC01'
line = '200301010000067148 1111123 3118 3008  0 22273 25  17 1477 12218012 109  0     60   16 107 119  7  00  00  0  0GSC      24    0 00   0 00         0   0  00   0  00  NC01'
sav_OT = []
sav_evlat = []
sav_evlon = []
sav_evdep = []
sav_ID = []
sav_family_phases = {}
with open(fileName,'r') as IN1:
    for line in IN1.readlines():
        #determine if this is header line
        try:
            int(line[:16])
            headerLine = line
            print(headerLine)
            OT = UTCDateTime('.'.join([line[:14],line[14:16]]))
            familyID = '%03d'%(int(headerLine[:4])-2000)
            #evlat = float(line[16:18]) + float('.'.join([line[18:21].strip(),  line[21:23].strip()  ]))/60.0
            #evlon = float(line[23:26]) + float('.'.join([line[26:29].strip(),  line[29:31].strip()  ]))/60.0
            #add 0 to prevent error for ' '
            assert (float('0'+line[18:21].strip()) + float('0'+line[21:23].strip())*0.01)<60,'degree minute'
            assert (float('0'+line[26:29].strip()) + float('0'+line[29:31].strip())*0.01)<60,'degree minute'
            evlat = float(line[16:18]) + (float('0'+line[18:21].strip()) + float('0'+line[21:23].strip())*0.01)/60.0
            evlon = float(line[23:26]) + (float('0'+line[26:29].strip()) + float('0'+line[29:31].strip())*0.01)/60.0
            evdep = float('.'.join([line[31:34], line[34:36]]))
            sav_OT.append(OT)
            sav_evlat.append(evlat)
            sav_evlon.append(evlon)
            sav_evdep.append(evdep)
            sav_ID.append(familyID)
            sav_family_phases[familyID] = {'eqLoc':[evlon,evlat,evdep],'catShift':OT-UTCDateTime(OT.strftime("%Y-%m-%d")),'sta':{}}
        except:
            if line == '\n':
                continue
            print(headerLine)
            stn = line[:5].strip()
            net = line[5:7] #this should always be PO in the file
            sav_family_phases[familyID]['sta'][stn] = {'P1':-1,'P2':-1}
            #dealing with P1
            #if line[29:32].strip() + line[32:34].strip():
            if not line[29:34].strip() in ['0','']:
                #not empty, update the Phase(P1) value from -1 to P1Arr
                P1 = float(line[29:32]) + float(line[32:34])*0.01
                #arrival time is P1-OT
                P1Arr = P1 #- (OT.second + OT.microsecond*1e-6)
                assert P1Arr > 0, "arrival time doesnt make sense!"
                sav_family_phases[familyID]['sta'][stn]['P1'] = P1Arr
            
            if not line[41:46].strip() in ['0','']:
                P2 = float(line[41:44]) + float(line[44:46])*0.01
                P2Arr = P2 #- (OT.second + OT.microsecond*1e-6)
                assert P2Arr > 0, "arrival time doesnt make sense!"
                sav_family_phases[familyID]['sta'][stn]['P2'] = P2Arr


#save the result for later use
np.save('sav_family_phases.npy',sav_family_phases)

import matplotlib.pyplot as plt

sav_evlon, sav_evlat, sav_evdep = np.array(sav_evlon), np.array(sav_evlat), np.array(sav_evdep)
plt.scatter(-sav_evlon,sav_evlat,c=sav_evdep,cmap='jet')
#plt.plot(-sav_evlon,sav_evlat,'ko')
for i in range(len(sav_evlon)):
    plt.text(-sav_evlon[i],sav_evlat[i],sav_ID[i])
plt.colorbar()
plt.show()


