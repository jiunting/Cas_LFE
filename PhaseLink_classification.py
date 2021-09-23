# classify the phase_link output .nlloc file by LFE families

import numpy as np
from obspy import UTCDateTime


EQinfo = np.load('sav_family_phases.npy',allow_pickle=True) #Note! detection file has a shift for each family
EQinfo = EQinfo.item()

# table sorted by the sta defined here
sta_list = ['LZB','YOUB','KLNB','TWKB','SNB','SILB','TSJB','TWGB','PFB','GOWB','NLLB','PGC','SSIB','VGZ']
phase_list = ['P2'] #only use S wave i.e. P2
phase_conv = {'P1':'P','P2':'S'}

# get inter-station time difference
sav_table = {} 
for fam in EQinfo.keys():
    sav_t = []
    sav_sta_phase = []
    for sta in sta_list:
        for phs in phase_list:
            sav_sta_phase.append('_'.join([sta,phase_conv[phs]]))
            if sta not in EQinfo[fam]['sta']:
                sav_t.append(np.nan)
                continue
            if phs in EQinfo[fam]['sta'][sta] and EQinfo[fam]['sta'][sta][phs]!=-1:
                sav_t.append(EQinfo[fam]['sta'][sta][phs])
            else:
                sav_t.append(np.nan)
    sav_t, sav_sta_phase = np.array(sav_t), np.array(sav_sta_phase)
    '''
    # use all the available stations, phases, without limiting the shape of the table
    # this will need to change the table to match target data later
    for sta in EQinfo[fam]['sta'].keys():
        if sta not in sta_list:
            continue
        P1 = EQinfo[fam]['sta'][sta]['P1']
        P2 = EQinfo[fam]['sta'][sta]['P2']
        if 'P' in phase_list:
            if P1!=-1:
                sav_t.append(P1)
                sav_sta_phase.append('_'.join([sta,'P']))
        if 'S' in phase_list:
            if P2!=-1:
                sav_t.append(P2)
                sav_sta_phase.append('_'.join([sta,'S']))
    sav_t, sav_sta_phase = np.array(sav_t), np.array(sav_sta_phase)
    '''
    #=== convert arrival to differential arrival time table===
    #i.e. arrival/travel time difference between station pairs
    arr_diff = np.zeros([len(sav_t),len(sav_t)]) #initialize the 2D diff arrival time array
    for i in range(len(sav_t)):
        arr_diff[i] = sav_t-sav_t[i]
    sav_table[fam] = {'columns':sav_sta_phase,'table':arr_diff}
        
np.save('diff_travelTime_table.npy',sav_table)    


'''
#======= plot the checking table ========
for k in sav_table.keys():
    plt.figure(figsize=(8.5,8.5))
    plt.imshow(sav_table[k]['table'],cmap='seismic')
    plt.gca().invert_yaxis()
    Ncol = len(sav_table[k]['columns'])
    plt.xticks(list(range(Ncol)),list(sav_table[k]['columns']),rotation=90,fontsize=14)
    plt.yticks(list(range(Ncol)),list(sav_table[k]['columns']),rotation=0,fontsize=14)
    plt.title('family: %s'%(k),fontsize=14)
    plt.grid(False)
    clb = plt.colorbar()
    clb.set_label('dT (s)', rotation=90,labelpad=-1,fontsize=12)
    plt.savefig('./Time_diff_table/table%s.png'%(k))
    plt.close()
    #plt.show()
'''

# read PhaseLink output file (.nlloc) and use the checking table to classify the family
'''
Example
VGZ    CN   ?    ? S      ? 20050907 0754  7.2900 GAU  1.00e-01 -1.00e+00 -1.00e+00 -1.00e+00
SILB   PO   ?    ? S      ? 20050907 0754 10.8700 GAU  1.00e-01 -1.00e+00 -1.00e+00 -1.00e+00
PGC    CN   ?    ? S      ? 20050907 0754 11.8400 GAU  1.00e-01 -1.00e+00 -1.00e+00 -1.00e+00
SNB    CN   ?    ? S      ? 20050907 0754 16.3000 GAU  1.00e-01 -1.00e+00 -1.00e+00 -1.00e+00
'''


#IN1 = open('../PhaseLink/CasLFEs_S_y0.1.nlloc','r')
filename = '../PhaseLink/CasLFEs_S_y0.1.nlloc'
sav_sta = []
sav_phs = []
sav_t = []

with open(filename,'r') as IN1:
    for line in IN1:
        if len(line.split())!=14:
            # this is header line
            sav_sta = []
            sav_phs = []
            sav_t = []
            continue
        sta = line.split()[0]
        phs = line.split()[4]
        t = UTCDateTime(line.split()[6]+'T'+line.split()[7]) + float(line.split()[8])
        sav_sta.append(sta)
        sav_phs.append(phs)
        sav_t.append(t)















