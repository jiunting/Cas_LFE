import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import unet_tools
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score



drop = False
large = 1
sr = 100
#checkpoint_filepath = '/Users/timlin/Documents/Project/Cascadia_LFE/checks/large_1.0_unet_lfe_std_0.4.tf_0006.ckpt'
#checkpoint_filepath = 'large_0.5_unet_lfe_std_0.4.tf.008_0004.ckpt'  #model 008
#checkpoint_filepath = 'large_1.0_unet_lfe_std_0.4.tf.009_0004.ckpt'  #model 009
#checkpoint_filepath = 'large_0.5_unet_lfe_std_0.4.tf.010_0017.ckpt'  #model 010
checkpoint_filepath = 'large_1.0_unet_lfe_std_0.4.tf.011_0017.ckpt'  #model 011


if drop:
    model = unet_tools.make_large_unet_drop(large,sr,ncomps=3)
else:
    model = unet_tools.make_large_unet(large,sr,ncomps=3)

model.load_weights(checkpoint_filepath)


# load testing data
X = np.load('Xtest_010.npy') #CC0.1
y = np.load('ytest_010.npy')

y_pred = model.predict(X)

#plot the first n examples
N = 64
T = np.arange(1500)*0.01
for i in range(N):
    plt.plot(T,y[i]+i,'tab:red')
    plt.plot(T,y_pred[i]+i,'tab:blue')

plt.xlim([0,15])
plt.ylim([-0.5,N])
plt.xlabel('Time(s)')
plt.ylabel('traces')



i = 22
# plot the recovered data
epsilon = 1e-6
#comb_D = np.hstack([(np.exp(X[22][:,0])-epsilon)*X[22][:,1], (np.exp(X[22][:,2])-epsilon)*X[22][:,3], (np.exp(X[22][:,4])-epsilon)*X[22][:,5]])
plt.subplot(4,1,1)
plt.plot(T,(np.exp(X[i][:,0])-epsilon)*X[i][:,1],'k',label='Z'); plt.legend()
plt.xlim([0,15])
plt.xticks([],[])
plt.subplot(4,1,2)
plt.plot(T,(np.exp(X[i][:,2])-epsilon)*X[i][:,3],'k',label='E'); plt.legend()
plt.xlim([0,15])
plt.xticks([],[])
plt.subplot(4,1,3)
plt.plot(T,(np.exp(X[i][:,4])-epsilon)*X[i][:,5],'k',label='N'); plt.legend()
plt.xlim([0,15])
plt.xticks([],[])
plt.subplot(4,1,4)
plt.plot(T,y[i],'tab:red',label='true')
plt.plot(T,y_pred[i],'tab:blue',label='predict')
plt.xlim([0,15])
plt.ylim([0,1])
plt.xlabel('Time(s)')
plt.legend()

plt.subplots_adjust(left=0.08,top=0.88,right=0.97,bottom=0.1,wspace=0.0,hspace=0.0)
plt.show()



#-----get metrics---------
sav_acc = []
sav_prec = []
sav_rec = [] # or true positive rate
sav_FPR = [] #false positive rate FP/(FP+TN)
y_true = np.where(np.max(y,axis=1) == 1, 1, 0)
for thresh in np.arange(0,1,0.01):
    #y_true = np.where(np.max(y,axis=1) >= thresh, 1, 0)
    y_pred_TF = np.where(np.max(y_pred,axis=1) >= thresh, 1, 0)
    acc = accuracy_score(y_true,y_pred_TF)
    prec = precision_score(y_true,y_pred_TF)
    rec = recall_score(y_true,y_pred_TF)
    TP = np.sum((y_true==1) & (y_pred_TF==1))
    TN = np.sum((y_true==0) & (y_pred_TF==0))
    FP = np.sum((y_true==0) & (y_pred_TF==1))
    FN = np.sum((y_true==1) & (y_pred_TF==0))
    FPR = FP/(FP+TN)
    print('----------------')
    print('%.2f    |    %.2f'%(TP,FP))
    print('%.2f    |    %.2f'%(FN,TN))
    #print('precision=',prec,TP/(TP+FP))
    #print('recall=',rec,TP/(TP+FN))
    sav_acc.append(acc)
    sav_prec.append(prec)
    sav_rec.append(rec)
    sav_FPR.append(FPR)

#plot ROC curve, calculate AUC
plt.plot(sav_FPR,sav_rec,color=[0,0,0])
plt.scatter(sav_FPR,sav_rec,c=np.arange(0,1,0.01))
AUC = m.auc(sav_FPR,sav_rec)
plt.xlabel('FPR',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.title('AUC=%.2f'%(AUC))
plt.grid(True)
plt.colorbar()
plt.show()
