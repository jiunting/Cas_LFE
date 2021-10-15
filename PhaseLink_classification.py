# classify the phase_link output .nlloc file by LFE families

import numpy as np
from obspy import UTCDateTime
import copy


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

# keep only table value
sav_table_value = []
sav_table_fam = []
for k in sav_table.keys():
    sav_table_value.append(sav_table[k]['table'])
    sav_table_fam.append(k)

sav_table_value, sav_table_fam = np.array(sav_table_value), np.array(sav_table_fam)

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


#filename = '../PhaseLink/CasLFEs_S_y0.7.nlloc'
filename = '../PhaseLink/CasLFEs_S_y0.1.nlloc'
#filename = './test.nlloc'

out_name = 'family_classify_S_y0.1.txt'

filt_N = 1 # minimum number of observations 





def match_fam(sav_sta_phase,sav_t_part,OUT):
    '''
        Input:
            sav_sta_phase:         list of station_phase
            sav_t_part:            list of absolute arrival time
            OUT:                   output file handle
        Output:
            write to OUT handle
    '''
    sav_sta_phase,sav_t_part = np.array(sav_sta_phase), np.array(sav_t_part)
    sav_t = [] # full t, same length as sta_list * phase_list
    for sta in sta_list:
        for phs in phase_list:
            if '_'.join([sta,phase_conv[phs]]) in sav_sta_phase:
                idx = np.where('_'.join([sta,phase_conv[phs]])==sav_sta_phase )[0][0]
                sav_t.append(sav_t_part[idx]-UTCDateTime("20000101"))
            else:
                sav_t.append(np.nan)
    # convert arrival time to differential arrival time
    sav_t = np.array(sav_t)
    arr_diff = np.zeros([len(sav_t),len(sav_t)]) #initialize the 2D diff arrival time array
    for i in range(len(sav_t)):
        arr_diff[i] = sav_t-sav_t[i]
    #sav_table_obs.append(arr_diff) # dont want to save everything
    #=== start differential arr matching (compare with table sav_table)===
    #residuals = np.abs(arr_diff - sav_table_value) # residuals between current obs and 130 families
    #===== only take the upper triangle, and without diagonal(self) term=========
    sav_table_value_triu = np.array([tmp_tab[np.triu_indices(len(sav_t),k=1)]  for tmp_tab in sav_table_value])
    residuals = np.abs(arr_diff[np.triu_indices(len(sav_t),k=1)] - sav_table_value_triu ) # residuals between current obs and 130 families
    sav_mean = [] #mean of the misfit value
    sav_n_obs = [] # number of observations
    for ii in range(len(residuals)):
        tmp_residuals = residuals[ii].reshape(-1)
        idx_val = np.where( ~np.isnan(tmp_residuals) )[0]
        if len(idx_val)!=0:
            tmp_mean = tmp_residuals[idx_val].mean()
            sav_mean.append(tmp_mean)
            #sav_n_obs.append(len(idx_val)**0.5)
            sav_n_obs.append(len(idx_val))
        else:
            sav_mean.append(float("inf")) # or append another nan
            sav_n_obs.append(0)
    # find the best fit family
    sav_mean = np.array(sav_mean)
    sav_n_obs = np.array(sav_n_obs)
    idx_gtNsta = np.where(sav_n_obs>=filt_N)[0] # minimum number of observations
    if len(idx_gtNsta)==0:
        # not enough stations left
        OUT.write('%s,%f,%s,%d,%d\n'%('000',0,"2000-01-01T00:00:00.00",0,sav_n_obs.max()))
        return      
    sav_mean = sav_mean[idx_gtNsta]
    sav_n_obs = sav_n_obs[idx_gtNsta]
    tmp_table_fam = sav_table_fam[idx_gtNsta]
    fam_idx = np.where( sav_mean==sav_mean.min() )[0][0]
    #========= write to file ==============
    OUT.write('%s,%f,%s,%d,%d\n'%(tmp_table_fam[fam_idx],sav_mean[fam_idx],first_t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-4],sav_n_obs[fam_idx],sav_n_obs.max()))
    return


sav_sta_phase = []
sav_t_part = [] # only have partial t, not the full sav_t
#sav_table_obs = []

OUT1 = open(out_name,'w')
OUT1.write('FamilyID,mean_residual,first_arr,N_obs,max_N_obs\n')
with open(filename,'r') as IN1:
    for line in IN1:
        if len(line.split())!=14:
            # this is header line
            if len(sav_sta_phase)!=0:
                match_fam(sav_sta_phase,sav_t_part,OUT1) # this done all the work
                # previous stuff done! 
            # reset list for the next event
            sav_sta_phase = []
            sav_t_part = []
            continue
        sta = line.split()[0]
        phs = line.split()[4]
        t = UTCDateTime(line.split()[6]+'T'+line.split()[7]) + float(line.split()[8])
        if sav_sta_phase==[]:
            import copy
            first_t = copy.deepcopy(t)
        sav_sta_phase.append('_'.join([sta,phs]))
        sav_t_part.append(t)

if len(sav_sta_phase)!=0:
    match_fam(sav_sta_phase,sav_t_part,OUT1)


OUT1.close()



