import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import unet_tools

#--- parameters
#detc = 'cut_daily_all_2.csv'
#data = 'cut_daily_all_2.hdf5'
detc = 'cut_daily_test.csv'
data = 'cut_daily_test.hdf5'

#model_path = 'large_2.0_unet_lfe_std_0.4.tf.014'
model_path = 'large_1.0_unet_lfe_std_0.4.tf.02'
drop = False
N_size = float(model_path.split('/')[-1].split('_')[1])
run_num = model_path.split('/')[-1].split('.tf.')[-1]
epsilon = 1e-6

#--- build model artechetecture
if drop:
    model = unet_tools.make_large_unet_drop(N_size,sr=100,ncomps=3)
else:
    model = unet_tools.make_large_unet(N_size,sr=100,ncomps=3)

# load weights
model.load_weights(model_path)


#'network','sta','chn','stlon','stlat','stdep','starttime','endtime','y','id'
detc = pd.read_csv(detc)
data = h5py.File(data,'r')

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


def cal_CCC(data1,data2):
    #cross-correlation "Coef" cunction, return the max CC
    from obspy.signal.cross_correlation import correlate_template
    CCCF=correlate_template(data1,data2)
    return np.max(np.abs(CCCF))


def QC(data,Type='data'):
    '''
        quality control of data
        return true if data pass all the checking filters, otherwise false
        '''
    #nan value in data check
    if np.isnan(data).any():
        return False
    #if they are all zeros
    if np.max(np.abs(data))==0:
        return False
    #normalize the data to maximum 1
    data = data/np.max(np.abs(data))
    #set QC parameters for noise or data
    if Type == 'data':
        N1,N2,min_std,CC = 30,30,0.01,0.98
    else:
        N1,N2,min_std,CC = 30,30,0.05,0.98
    #std window check, std too small probably just zeros
    wind = len(data)//N1
    for i in range(N1):
        #print('data=',data[int(i*wind):int((i+1)*wind)])
        #print('std=',np.std(data[int(i*wind):int((i+1)*wind)]))
        if np.std(data[int(i*wind):int((i+1)*wind)])<min_std :
            return False
    #auto correlation, seperate the data into n segments and xcorr with rest of data(without small data) to see if data are non-corh
    wind = len(data)//N2
    for i in range(N2):
        data_small = data[int(i*wind):int((i+1)*wind)] #small segment
        data_bef = data[:int(i*wind)]
        data_aft = data[int((i+1)*wind):]
        data_whole = np.concatenate([data_bef,data_aft])
        curr_CC = cal_CCC(data_whole,data_small)
        if curr_CC>CC:
            return False
    return True



all_idx = detc[detc['y']==0.17].index
#all_idx = detc[(detc['y']<0.5) & (detc['y']>0.1)].index
#detc['id'][(detc['y']>=0.5)]
for n,idx in enumerate(all_idx):
    if n<20:
        continue
    evid = detc.iloc[idx].id
    max_y = detc.iloc[idx].y
    D = data.get('data/'+evid)
    D_merge = np.hstack([D[0],D[1],D[2]])
    D_merge = D_merge/np.max(np.abs(D_merge))
    if QC(D_merge):
        plt.plot(np.arange(len(D_merge))*0.01,D_merge+n,'-',color=[0.5,0.5,0.5],linewidth=1) #sr always 100
    else:
        plt.plot(np.arange(len(D_merge))*0.01,D_merge+n,'b')
    tmp_inp = ZEN2inp(D[0],D[1],D[2],epsilon)
    tmp_inp = tmp_inp.reshape(1,-1,6)
    y_pred = model.predict(tmp_inp)
    print('max y=',y_pred.max(),max_y)
    y_pred = np.hstack([y_pred[0],y_pred[0],y_pred[0]])
    plt.plot(np.arange(len(D_merge))*0.01,y_pred+n,'r',linewidth=0.8)
    plt.text(0,n-0.2,evid,fontsize=12)
    if n==50:
        break

props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(5,n+0.7,'Z',fontsize=14,bbox=props)
plt.text(20,n+0.7,'E',fontsize=14,bbox=props)
plt.text(35,n+0.7,'N',fontsize=14,bbox=props)

YLIM = plt.ylim()
plt.plot([15,15],YLIM,'k-',linewidth=1.5)
plt.plot([30,30],YLIM,'k-',linewidth=1.5)
plt.xlim([0,45])
plt.ylim(YLIM)
plt.xlabel('Time (s)',fontsize=14)
plt.title('ID:%s y=%.2f'%(evid,max_y))
plt.show()




all_idx = detc[detc['y']==0.2].index
for n,idx in enumerate(all_idx):
    evid = detc.iloc[idx].id
    max_y = detc.iloc[idx].y
    D = data.get('data/'+evid)
    D_merge = np.hstack([D[0],D[1],D[2]])
    #D_merge = D_merge/np.max(np.abs(D_merge))
    plt.plot(np.arange(len(D_merge))*0.01,D_merge,'-',color=[0.5,0.5,0.5],linewidth=1)
    tmp_inp = ZEN2inp(D[0],D[1],D[2],epsilon)
    tmp_inp = tmp_inp.reshape(1,-1,6)
    y_pred = model.predict(tmp_inp)
    print('max y=',y_pred.max(),max_y)
    y_pred = np.hstack([y_pred[0],y_pred[0],y_pred[0]])
    scale = np.max(D_merge)
    plt.plot(np.arange(len(D_merge))*0.01,y_pred*scale,'r',linewidth=0.8)
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    YLIM = plt.ylim()
    plt.text(5,0.8*YLIM[1],'Z',fontsize=14,bbox=props)
    plt.text(20,0.8*YLIM[1],'E',fontsize=14,bbox=props)
    plt.text(35,0.8*YLIM[1],'N',fontsize=14,bbox=props)
    plt.plot([15,15],YLIM,'k-',linewidth=1.5)
    plt.plot([30,30],YLIM,'k-',linewidth=1.5)
    plt.xlim([0,45])
    plt.ylim(YLIM)
    plt.xlabel('Time (s)',fontsize=14)
    plt.title('ID:%s y=%.2f'%(evid,max_y))
    plt.show()



#----Test multiply the data---------
D_merge = np.hstack([D[0],D[1],D[2]])
#D_merge = D_merge/np.max(np.abs(D_merge))
plt.plot(np.arange(len(D_merge))*0.01,D_merge,'-',color=[0.5,0.5,0.5],linewidth=1)
tmp_inp = ZEN2inp(D[0],D[1],D[2],epsilon)
tmp_inp = tmp_inp.reshape(1,-1,6)
y_pred = model.predict(tmp_inp)
print('max y=',y_pred.max(),max_y)
y_pred = np.hstack([y_pred[0],y_pred[0],y_pred[0]])
scale = np.max(D_merge)
plt.plot(np.arange(len(D_merge))*0.01,y_pred*scale,'r',linewidth=0.8)
props = dict(boxstyle='round', facecolor='white', alpha=1)
YLIM = plt.ylim()
plt.text(5,0.8*YLIM[1],'Z',fontsize=14,bbox=props)
plt.text(20,0.8*YLIM[1],'E',fontsize=14,bbox=props)
plt.text(35,0.8*YLIM[1],'N',fontsize=14,bbox=props)
plt.plot([15,15],YLIM,'k-',linewidth=1.5)
plt.plot([30,30],YLIM,'k-',linewidth=1.5)
plt.xlim([0,45])
plt.ylim(YLIM)
plt.xlabel('Time (s)',fontsize=14)
plt.title('ID:%s y=%.2f'%(evid,y_pred.max()))
plt.show()

#
sc = 1e #scale the data and test if model break
D_merge = np.hstack([D[0],D[1],D[2]])*sc
#D_merge = D_merge/np.max(np.abs(D_merge))
plt.plot(np.arange(len(D_merge))*0.01,D_merge,'-',color=[0.5,0.5,0.5],linewidth=1)
tmp_inp = ZEN2inp(D[0]*sc,D[1]*sc,D[2]*sc,epsilon)
tmp_inp = tmp_inp.reshape(1,-1,6)
y_pred = model.predict(tmp_inp)
print('max y=',y_pred.max(),max_y)
y_pred = np.hstack([y_pred[0],y_pred[0],y_pred[0]])
scale = np.max(D_merge)
plt.plot(np.arange(len(D_merge))*0.01,y_pred*scale,'r',linewidth=0.8)
props = dict(boxstyle='round', facecolor='white', alpha=1)
YLIM = plt.ylim()
plt.text(5,0.8*YLIM[1],'Z',fontsize=14,bbox=props)
plt.text(20,0.8*YLIM[1],'E',fontsize=14,bbox=props)
plt.text(35,0.8*YLIM[1],'N',fontsize=14,bbox=props)
plt.plot([15,15],YLIM,'k-',linewidth=1.5)
plt.plot([30,30],YLIM,'k-',linewidth=1.5)
plt.xlim([0,45])
plt.ylim(YLIM)
plt.xlabel('Time (s)',fontsize=14)
plt.title('ID:%s y=%.2f'%(evid,y_pred.max()))
plt.show()


for i,sc in enumerate(np.arange(10)):
    sc = 10**sc
    tmp_inp = ZEN2inp(D[0]*sc,D[1]*sc,D[2]*sc,epsilon)
    tmp_inp = tmp_inp.reshape(1,-1,6)
    y_pred = model.predict(tmp_inp)
    plt.plot(range(len(y_pred.T)),y_pred.T+i,'k')
    plt.plot([0,1500],[i+0.1,i+0.1],'r--')

plt.show()


