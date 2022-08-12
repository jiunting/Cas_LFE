# Test the model and do statistic

import h5py
import matplotlib
matplotlib.use('pdf') #instead using interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import unet_tools
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import glob
from obspy import UTCDateTime
import obspy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def ZEN2inp(Z,E,N,epsilon):
    # convert raw ZEN data to input feature
    data_Z_sign = np.sign(Z)
    data_E_sign = np.sign(E)
    data_N_sign = np.sign(N)
    data_Z_val = np.log(np.abs(Z)+epsilon)
    data_E_val = np.log(np.abs(E)+epsilon)
    data_N_val = np.log(np.abs(N)+epsilon)
    data_inp = np.hstack([data_Z_val.reshape(-1,1),data_Z_sign.reshape(-1,1),
                          data_E_val.reshape(-1,1),data_E_sign.reshape(-1,1),
                          data_N_val.reshape(-1,1),data_N_sign.reshape(-1,1),])
    return data_inp


def get_catalog(catalog_file: str, EQinfo_file: str) -> pd.DataFrame:
    """
    Convert catalog file to pandas format with columns =
    ['famID','mag','catOT','OT']
        famID: family ID e.g. 001
        mag: magnitude of LFE
        catOT: catalog origintime. This is the time of when template start
        OT: real OT. The real origin time after correction
    
    INPUT:
    ======
    catalog_file: str
        catalog file name
    EQinfo_file: str
        EQ info file name. The file save LFE family info
        
    OUTPUT:
    =======
    res: pd.DataFrame
        pandas format
    """
    EQinfo = np.load(EQinfo_file,allow_pickle=True)
    EQinfo = EQinfo.item()
    head = ['famID','lon','lat','dep','mag','catOT','OT']
    sav_all = []
    with open(catalog_file,'r') as IN1:
        for line in IN1.readlines():
            line = line.strip()
            ID = line.split()[0] #family ID
            OT = UTCDateTime('20'+line.split()[1]) #YYMMDD
            HH = (int(line.split()[2])-1)*3600  #HR from 1-24
            SS = float(line.split()[3])
            OT = OT + HH + SS  #detected origin time. always remember!!! this is not the real OT. The shifted time in the sav_family_phases.npy have been corrected accordingly.
            real_OT = OT + EQinfo[ID]['catShift']  # set to real OT, not template starttime
            Mag = float(line.split()[4])
            # get EQINFO from EQinfo
            evlo = -EQinfo[ID]['eqLoc'][0]
            evla = EQinfo[ID]['eqLoc'][1]
            evdp = EQinfo[ID]['eqLoc'][2]
            sav_all.append([ID,evlo,evla,evdp,Mag,OT.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-2],real_OT.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-2]])
    df = pd.DataFrame(sav_all, columns = head)
    return df


def cal_metrics(y, y_pred, thresholds, test_time=True):
    sav_TPR = []
    sav_FPR = []
    sav_prec = []
    sav_acc = []
    sav_threshold = []
    for threshold in thresholds:
        # 1. first consider binary classification
        y_pred_max = np.max(y_pred, axis=1)
        y_max = np.max(y, axis=1)
        #TP1 = np.where((y_max>=threshold) & (y_pred_max>=threshold))[0]
        #TN1 = np.where((y_max<threshold) & (y_pred_max<threshold))[0]
        #FP1 = np.where((y_max<threshold) & (y_pred_max>=threshold))[0]
        #FN1 = np.where((y_max>=threshold) & (y_pred_max<threshold))[0]
        TP = np.where((y_max==1) & (y_pred_max>=threshold))[0]
        TN = np.where((y_max==0) & (y_pred_max<threshold))[0]
        FP = np.where((y_max==0) & (y_pred_max>=threshold))[0]
        FN = np.where((y_max==1) & (y_pred_max<threshold))[0]
        assert (len(TP)+len(TN)+len(FP)+len(FN))==y.shape[0],'metrics do not match the total number!'
        if (len(TP) + len(FN))!=0:
            TPR = len(TP) / (len(TP) + len(FN))
        else:
            TPR = 0
        if (len(FP) + len(TN))!=0:
            FPR = len(FP) / (len(FP) + len(TN))
        else:
            FPR = 0
        if (len(TP) + len(FP))!=0:
            prec = len(TP) / (len(TP) + len(FP))
        else:
            prec = 0
        acc = (len(TP) + len(TN)) / y.shape[0]
        sav_TPR.append(TPR)
        sav_FPR.append(FPR)
        sav_prec.append(prec)
        sav_acc.append(acc)
        sav_threshold.append(threshold)
    # calculate AUC, area under ROC curve
    AUC = np.abs(np.trapz(sav_TPR,sav_FPR)) #FPR starts from right to left making integration negative
    return sav_threshold, sav_TPR, sav_FPR, sav_prec, sav_acc, AUC


# Model and data to be tested
#--------------------------------------------------------
catalog_file = "total_mag_detect_0000_cull_NEW.txt"
EQinfo_file = "sav_family_phases.npy"
drop = 0
sr = 100

epsilon=1e-6
std = 0.4
#run_num = 'S001'
#large = 0.5
#run_num = 'S002'
#large = 1.0
#run_num = 'S003'
#large = 2.0
run_num = 'P001'
large = 0.5
#run_num = 'P002'
#large = 1.0
#run_num = 'P003'
#large = 2.0
#x_data_file = "Data_QC_rmean_norm/merged20220602_S.h5"
#csv_file = "Data_QC_rmean_norm/merged20220602_S.csv"
x_data_file = "Data_QC_rmean_norm/merged20220602_P.h5"
csv_file = "Data_QC_rmean_norm/merged20220602_P.csv"
n_data_file = "Cascadia_noise_QC_rmean_norm.h5"

# load station info
stainfo=np.load('stainfo.npy',allow_pickle=True)
stainfo = stainfo.item()

# load EQinfo
EQinfo = np.load(EQinfo_file,allow_pickle=True)
EQinfo = EQinfo.item()

# loading catalog file to pandas
catalog = get_catalog(catalog_file, EQinfo_file)

# loading testing(or training) index
sig_test_inds = np.load("./Index/index_%s/sig_test_inds.npy"%(run_num))
noise_test_inds = np.load("./Index/index_%s/noise_test_inds.npy"%(run_num))

#loading h5py data and csv file
x_data = h5py.File(x_data_file,'r')
csv = pd.read_csv(csv_file)
n_data = h5py.File(n_data_file,'r')

"""
To get data
print(csv.iloc[0].evID) # all the available evID, e.g. 2005-09-05T19:21:37.6250_001.CN.LZB.BH.S
x_data['waves/'+'2005-09-05T19:21:37.6250_001.CN.LZB.BH.S'] # shape=(9003,) sorted in ZEN, 100 Hz., +-15s centered on arrival.
"""

# load model
if drop:
    model=unet_tools.make_large_unet_drop(large,sr,ncomps=3)
else:
    model=unet_tools.make_large_unet(large,sr,ncomps=3)

chks = glob.glob("./checks/large*%s*"%(run_num))
chks.sort()
model.load_weights(chks[-1].replace('.index',''))


# parameters for model testing
Ntests = 10 # number of testing
winsize=15   # winsize for input data in seconds
rnd_start = True # for each trace, randomly cut the time series. If False: always centered at arrival(center of the 30 sec trace)
sig_max = 2000 # number of sig/noise data, note that noise data has the same number with sig
mag_range = (0,3) #magnitude range to be tested
dist_range = (0,100) #distance range to be tested

# length of each component
nlen = int(sr*30)+1

versions = ['v1','v2','v3']
mag_ranges = [(0,3),(2.2,3),(0,3)]
dist_ranges = [(0,100),(0,100),(0,10)]
sav_stats = {}
for i_ver, version in enumerate(versions):
    print('Version:',version)
    sav_stats.update({version:{'fpr':[],'tpr':[],'thresholds':[],'AUC':[]}})
    # ===== Start generating data =====
    for Ntest in range(Ntests):
        print(' %d/%d tests'%(Ntest+1,Ntests))
        X = []
        y = []
        # 1. start with sig data
        n_sig = 0 #number of sig data
        done_flag = False # are you done?
        shf_sig_test_inds = sig_test_inds.copy()
        np.random.shuffle(shf_sig_test_inds)
        for idx in shf_sig_test_inds:
            if done_flag:
                break
            # EQ info
            mag = catalog.iloc[idx].mag
            evlo = catalog.iloc[idx].lon
            evla = catalog.iloc[idx].lat
            evdp = catalog.iloc[idx].dep
            if mag<mag_ranges[i_ver][0] or mag>mag_ranges[i_ver][1]:
                continue
            #print('In idx:%s, number of accumulated traces=%d'%(idx,n_sig))
            rows = csv[csv['idxCatalog']==idx] #for this earthquake, get all the available records
            for _, row in rows.iterrows():
                sta = row.sta
                stlo, stla, _ = stainfo[row.sta]
                dist, _, _ = obspy.geodetics.base.gps2dist_azimuth(lat1=evla,lon1=evlo,lat2=stla,lon2=stlo)
                dist = dist*1e-3 #m to km
                if dist<dist_ranges[i_ver][0] or dist>dist_ranges[i_ver][1]:
                    continue
                i_evid = row.evID
                #print(' ',i_evid)
                n_sig += 1
                data_all = np.array(x_data['waves/%s'%(i_evid)]) #(9003,)
                data_comp1 = data_all[:nlen]
                data_comp2 = data_all[nlen:2*nlen]
                data_comp3 = data_all[2*nlen:]
                targ = signal.gaussian(nlen,std=int(std*sr))
                if rnd_start:
                    # random start time idx from 1~1500
                    st = np.random.choice(np.arange(1,1501))
                else:
                    # always select the center trace
                    st = 750
                data_comp1 = data_comp1[st:st+1500]
                data_comp2 = data_comp2[st:st+1500]
                data_comp3 = data_comp3[st:st+1500]
                targ = targ[st:st+1500] #note when rnd_start=False, targ[750] = 1.0 (select the middle point) 
                # normalize to make sure max is 1 but keep the ratio
                nor_val = max(max(np.abs(data_comp1)), max(np.abs(data_comp1)), max(np.abs(data_comp1)))
                inp = ZEN2inp(data_comp1/nor_val, data_comp2/nor_val, data_comp3/nor_val, epsilon)
                X.append(inp)
                y.append(targ)
                if n_sig>=sig_max:
                    done_flag = True 
                    break
        # 2. add the same number of noise data as the sig
        sh_noise_test_inds = np.random.choice(noise_test_inds, n_sig)
        for i in sh_noise_test_inds:
            noise_all = n_data['waves'][i,:] # (9003,)
            noise_comp1 = noise_all[:nlen]
            noise_comp2 = noise_all[nlen:2*nlen]
            noise_comp3 = noise_all[2*nlen:]
            #targ = signal.gaussian(nlen,std=int(std*sr))
            if rnd_start:
                # random start time idx from 0~1501
                st = np.random.choice(np.arange(1501))
            else:
                # always select the center trace
                st = 750
            noise_comp1 = noise_comp1[st:st+1500]
            noise_comp2 = noise_comp2[st:st+1500]
            noise_comp3 = noise_comp3[st:st+1500]
            #targ = targ[st:st+1500] 
            targ = np.zeros(1500)
            # normalize to make sure max is 1 but keep the ratio
            nor_val = max(max(np.abs(noise_comp1)), max(np.abs(noise_comp1)), max(np.abs(noise_comp1)))
            inp = ZEN2inp(noise_comp1/nor_val, noise_comp2/nor_val, noise_comp3/nor_val, epsilon)
            X.append(inp)
            y.append(targ)
        # === start model testing ===
        X = np.array(X)
        y = np.array(y)
        y_pred = model.predict(X)
        #1. simple classification problem
        fpr, tpr, thresholds = roc_curve(np.max(y,axis=1),np.max(y_pred,axis=1))
        #sav_threshold, sav_TPR, sav_FPR, sav_prec, sav_acc, AUC = cal_metrics(y, y_pred, thresholds)
        AUC = auc(fpr, tpr)
        sav_stats[version]['fpr'].append(fpr)
        sav_stats[version]['tpr'].append(tpr)
        sav_stats[version]['thresholds'].append(thresholds)
        sav_stats[version]['AUC'].append(AUC)
        #2. calculate arrival time misfit
        #sav_threshold, sav_TPR, sav_FPR, sav_prec, sav_acc, AUC = cal_metrics(y, y_pred, thresholds, dt=2.0) #2 s tolarance
        #arr_idx_y = np.array([np.where(iy==max(iy))[0][0] if max(iy)==1 else -1 for iy in y])

np.save('./Stats/model_test_%s.npy'%(run_num),sav_stats)




# ===== plot performance curves =====
sav_stats = np.load('./Stats/model_test_%s.npy'%(run_num),allow_pickle=True)
sav_stats = sav_stats.item()

colors = ['b','g','m']

for ik,k in enumerate(sav_stats.keys()):
    # get fpr as X axis, tpr as Y axis
    sav_fpr = []
    sav_tpr = []
    for i in range(len(sav_stats[k]['AUC'])):
        interp_tpr = np.interp(np.arange(-0.001,1,0.001), sav_stats[k]['fpr'][i], sav_stats[k]['tpr'][i])
        sav_fpr.append(np.arange(-0.001,1,0.001))
        sav_tpr.append(interp_tpr)
        #plt.plot(sav_stats[k]['fpr'][i], sav_stats[k]['tpr'][i],'k.')
        #plt.plot(np.arange(-0.001,1,0.001), interp_tpr,'r')
    # calculate mean, and std
    sav_fpr = np.array(sav_fpr)
    sav_tpr = np.array(sav_tpr)
    mean_fpr = np.mean(sav_fpr,axis=0)
    mean_tpr = np.mean(sav_tpr,axis=0)
    std_fpr = np.std(sav_fpr,axis=0)
    std_tpr = np.std(sav_tpr,axis=0)
    plt.plot(mean_fpr, mean_tpr, colors[ik],lw = 3, alpha=0.8,label=r'%s, AUC=%.2f$\pm$%.3f'%(k,np.mean(sav_stats[k]['AUC']),np.std(sav_stats[k]['AUC'])) )
    fill_upper = np.minimum(mean_tpr + std_tpr, 1)
    fill_lower = np.minimum(mean_tpr - std_tpr, 1)
    plt.fill_between(mean_fpr,fill_upper, fill_lower, color=colors[ik], alpha=0.2)
    #plt.plot(mean_fpr, mean_tpr+std_tpr, 'b--',lw=3, alpha=0.8)
    #plt.plot(mean_fpr, mean_tpr-std_tpr, 'b--',lw=3, alpha=0.8)
    #plt.savefig('check_pred.png')
    #plt.close()

plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('FPR',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.legend()
plt.savefig('./Stats/Stats_%s.png'%(run_num),dpi=150)
plt.close()



# ===== plot performance curves from .npy files=====
colors = ['b','g','m']
run_nums = ['S001','S002','S003','P001','P002','P003']
for run_num in run_nums:
    sav_stats = np.load('./Stats/model_test_%s.npy'%(run_num),allow_pickle=True)
    sav_stats = sav_stats.item()
    print('%s:  AUC=%.3f,%.3f,%.3f'%(run_num, np.mean(sav_stats['v1']['AUC']), np.mean(sav_stats['v2']['AUC']), np.mean(sav_stats['v3']['AUC'])  ))
    for ik,k in enumerate(sav_stats.keys()):
        # get fpr as X axis, tpr as Y axis
        sav_fpr = []
        sav_tpr = []
        for i in range(len(sav_stats[k]['AUC'])):
            interp_tpr = np.interp(np.arange(-0.001,1,0.001), sav_stats[k]['fpr'][i], sav_stats[k]['tpr'][i])
            sav_fpr.append(np.arange(-0.001,1,0.001))
            sav_tpr.append(interp_tpr)
        # calculate mean, and std
        sav_fpr = np.array(sav_fpr)
        sav_tpr = np.array(sav_tpr)
        mean_fpr = np.mean(sav_fpr,axis=0)
        mean_tpr = np.mean(sav_tpr,axis=0)
        std_fpr = np.std(sav_fpr,axis=0)
        std_tpr = np.std(sav_tpr,axis=0)
        plt.plot(mean_fpr, mean_tpr, colors[ik],lw = 3, alpha=0.8,label=r'%s, AUC=%.3f$\pm$%.3f'%(k,np.mean(sav_stats[k]['AUC']),np.std(sav_stats[k]['AUC'])) )
        fill_upper = np.minimum(mean_tpr + std_tpr, 1)
        fill_lower = np.minimum(mean_tpr - std_tpr, 1)
        plt.fill_between(mean_fpr,fill_upper, fill_lower, color=colors[ik], alpha=0.2)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax1 = plt.gca()
    ax1.tick_params(pad=-1)
    plt.xlabel('FPR',fontsize=14,labelpad=0)
    plt.ylabel('TPR',fontsize=14,labelpad=0)
    plt.legend()
    plt.savefig('./Stats/Stats_%s.png'%(run_num),dpi=150)
    plt.close()


