import numpy as np
import matplotlib
matplotlib.use('pdf') #instead using interactive backend
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import unet_tools

#csv = pd.read_csv('cut_daily_all.csv')
#Data = h5py.File('cut_daily_all.hdf5','r')
csv = pd.read_csv('./Detections_S/cut_daily_PO.SILB.csv')
Data = h5py.File('./Detections_S/cut_daily_PO.SILB.hdf5','r')



model_path = 'large_1.0_unet_lfe_std_0.4.tf.002' # this is the S wave model
#model_path = 'large_1.0_unet_lfe_std_0.4.tf.006' # this is the P wave model
drop = False
N_size = float(model_path.split('/')[-1].split('_')[1])
run_num = model_path.split('/')[-1].split('.tf.')[-1]

#--- build model artechetecture
if drop:
    model = unet_tools.make_large_unet_drop(N_size,sr=100,ncomps=3)
else:
    model = unet_tools.make_large_unet(N_size,sr=100,ncomps=3)

# load weights
model.load_weights(model_path)


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


for n,i in enumerate(csv[csv['y']<0.2].index):
#for n,i in enumerate(csv[csv.index>6707000].index):
    evid = csv.iloc[i].id
    idx_max_y = csv.iloc[i].idx_max_y
    D = np.array(Data.get('data/'+evid))
    #double check if model get same result
    norm_val = max(max(np.abs(D[0])),max(np.abs(D[1])),max(np.abs(D[2])))
    data_inp = ZEN2inp(D[0]/norm_val,D[1]/norm_val,D[2]/norm_val,epsilon=1e-6)
    data_inp = data_inp.reshape(1,-1,6)
    y_pred = model.predict(data_inp)
    D_merge = np.hstack([D[0],D[1],D[2]])
    D_merge = D_merge/max(np.abs(D_merge))
    plt.figure()
    plt.plot(np.arange(len(D_merge))*0.01,D_merge,'-',color=[0.5,0.5,0.5],linewidth=1) #sr always 100
    plt.plot(np.arange(len(D_merge))*0.01,np.hstack([y_pred.reshape(-1),y_pred.reshape(-1),y_pred.reshape(-1)]),'r',linewidth=0.5)
    #plot where the max detection is in 3-comps
    idx_max_y = np.array([idx_max_y, idx_max_y+1500, idx_max_y+3000])
    #plt.plot(np.arange(len(D_merge))[idx_max_y]*0.01,D_merge[idx_max_y],'ro') #sr always 100
    plt.plot(np.arange(len(D_merge))[idx_max_y]*0.01,np.hstack([y_pred.reshape(-1),y_pred.reshape(-1),y_pred.reshape(-1)])[idx_max_y],'ro') #sr always 100
    # control plot 
    #plt.text(0,n-0.2,evid,fontsize=12)
    plt.title('ID:%s y=%.2f'%(evid,csv.iloc[i].y),fontsize=14)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.ylim([-1,1])
    YLIM = plt.ylim()
    plt.text(2,0.85*YLIM[1],'Z',fontsize=14,bbox=props)
    plt.text(17,0.85*YLIM[1],'E',fontsize=14,bbox=props)
    plt.text(32,0.85*YLIM[1],'N',fontsize=14,bbox=props)
    plt.plot([15,15],YLIM,'k-',linewidth=1.5)
    plt.plot([30,30],YLIM,'k-',linewidth=1.5)
    plt.xlim([0,45])
    #plt.ylim(YLIM)
    plt.xlabel('Time (s)',fontsize=14)
    #plt.savefig('./Figs1/fig%03d.png'%(n)) 
    plt.savefig('./Figs2/fig%03d.png'%(n)) 
    plt.close()
    if n>100:
        break

Data.close()

