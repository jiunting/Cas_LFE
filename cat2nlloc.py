# convert catalog file to NLLoc format
import numpy as np
from obspy import UTCDateTime


# outfile name
outfile = 'total_catalog.nlloc'



# load original detection file (from template matching)
# load family and arrival information
EQinfo = np.load('sav_family_phases.npy',allow_pickle=True) #Note! detection file has a shift for each family
EQinfo = EQinfo.item()
detcFile = 'total_mag_detect_0000_cull_NEW.txt' #LFE event detection file
sav_OT_template = []
sav_fam = []
sav_mag = []
with open(detcFile,'r') as IN1:
    for line in IN1.readlines():
        line = line.strip()
        ID = line.split()[0] #family ID
        OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
        HH = int(line.split()[2])*3600
        SS = float(line.split()[3])
        #OT = OT + HH + SS + EQinfo[ID]['catShift'] #detected origin time
        OT = OT + HH + SS  #detected template time, so that OT+P1 or P2 is the arrival time
        sav_OT_template.append(OT.datetime) # save the catalog time
        sav_fam.append(ID)
        sav_mag.append(line.split()[4])


sav_OT_template = np.array(sav_OT_template)
sav_fam = np.array(sav_fam)
sav_mag = np.array(sav_mag)

# sort the time
sor_idx = np.argsort(sav_OT_template)
sav_OT_template = sav_OT_template[sor_idx]
sav_fam = sav_fam[sor_idx]
sav_mag = sav_mag[sor_idx]

sta_list = ['LZB','YOUB','KLNB','TWKB','SNB','SILB','TSJB','TWGB','PFB','GOWB','NLLB','PGC','SSIB','VGZ']
phase_list = ['S'] #only keep S wave


sav_all_arr = []
sav_all_phases = []
sav_all_sta_name = []
sav_ID = []
for i in range(len(sav_OT_template)):
    ID = sav_fam[i]
    # get arrival time info
    arrs = []      # save arrival time
    phases = []    # save what phase
    sta_name = []  # save what sta
    for sta in EQinfo[ID]['sta']:
        if sta not in sta_list:
            continue
        if 'P' in phase_list:
            P1 = EQinfo[ID]['sta'][sta]['P1']
            if P1 != -1:
                arrs.append(UTCDateTime(sav_OT_template[i])+P1)
                phases.append('P')
                sta_name.append(sta)
        if 'S' in phase_list:
            P2 = EQinfo[ID]['sta'][sta]['P2']
            if P2 != -1:
                arrs.append(UTCDateTime(sav_OT_template[i])+P2)
                phases.append('S')
                sta_name.append(sta)
    # sort by first arrival in the net (i.e. all avail stations)
    arrs, phases, sta_name = np.array(arrs), np.array(phases), np.array(sta_name)
    sor_idx = np.argsort(arrs)
    arrs, phases, sta_name = arrs[sor_idx], phases[sor_idx], sta_name[sor_idx]
    #save the arrs,phases, sta_name into list to be used later
    sav_all_arr.append(arrs)
    sav_all_phases.append(phases)
    sav_all_sta_name.append(sta_name) 
    sav_ID.append(ID)

# first arrival for each event, depending on how far/close the stations are
sav_first_arr = np.array([i[0] for i in sav_all_arr])
sav_all_arr, sav_all_phases, sav_all_sta_name, sav_ID = np.array(sav_all_arr), np.array(sav_all_phases), np.array(sav_all_sta_name), np.array(sav_ID)
sor_idx = np.argsort(sav_first_arr)
sav_all_arr, sav_all_phases, sav_all_sta_name, sav_ID = sav_all_arr[sor_idx], sav_all_phases[sor_idx], sav_all_sta_name[sor_idx], sav_ID[sor_idx]


#===write to file===
with open(outfile,'w') as OUT1:
    for i in range(len(sav_ID)):
        OUT1.write('#Fam:%s\n'%(sav_ID[i]))
        for ii in range(len(sav_all_arr[i])):
            arr = sav_all_arr[i][ii]
            sta = sav_all_sta_name[i][ii]
            phs = sav_all_phases[i][ii]
            OUT1.write('%6s ?  ?    ? %1s      ? %s GAU  1.00e-01 -1.00e+00 -1.00e+00 -1.00e+00\n'%(sta,phs,arr.strftime('%Y%m%d %H%M %S.%f')[:19]))
    









