import obspy
import matplotlib.pyplot as plt
import glob
import numpy as np
import seaborn as sns
sns.set()
import h5py

files = glob.glob('./Data/*SSIB*P1.mseed')
#files = glob.glob('./Data/*TWKB*P1*.mseed')
#files = glob.glob('./Data/*SILB*P2*.mseed')
files = glob.glob('./Data/Fam_003_SILB*_P1.mseed')
#files = glob.glob('./Data/Fam_058_SILB*_P2.mseed')
files = glob.glob('./Data/Fam_*_TSJB*_P2.mseed')
files = glob.glob('./Data/Fam_001_TWKB*_P2.mseed')

files = glob.glob('./Data/Fam_001_TWKB*_P2.npy')


T = np.concatenate([np.arange(3001)/100,np.arange(3001)/100+30,np.arange(3001)/100+60])

file = 'ID_001_CN.LZB.BH_S.h5'
file = 'CN.SNB.BH_noise.h5'
file = 'ID_299_CN.YOUB.HH_S.h5'
#dtfl = h5py.File('./Data_test/'+file, 'r')
dtfl = h5py.File(file, 'r')
A = dtfl.get('waves')
AA = np.array(A)
#plt.subplot(1,2,1)
sumA = np.zeros_like(AA[i])
N = 0
for i in range(len(AA)):
    if QC(AA[i],Type='data'):
        if N<30:
            plt.plot(T,AA[i]/np.max(AA[i])+N,color=[0.5,0.5,0.5],linewidth=0.5)
            N += 1
        sumA += AA[i]/np.max(np.abs(AA[i]))
    else:
        continue
        plt.plot(T,AA[i]/np.max(AA[i])+i,'r',linewidth=1)


    '''
    if i%100==0:
        #plt.plot(T,sumA+i,'b')
        plt.ylabel('N-traces')
        plt.xlabel('Time(s)')
        plt.show()
        sumA = np.zeros_like(AA[i])
        N = 0 #reset number of traces
    '''

plt.plot([30,30],[-1,N+3],'k',linewidth=1.5)
plt.plot([60,60],[-1,N+3],'k',linewidth=1.5)
plt.plot([90,90],[-1,N+3],'k',linewidth=1.5)
plt.xlim([0,90])
plt.ylim([-1,N+3])
plt.yticks([],[])
props = dict(boxstyle='round', facecolor='white', alpha=0.7)
plt.text(5,N*0.05,'Z',fontsize=14,bbox=props)
plt.text(35,N*0.05,'E',fontsize=14,bbox=props)
plt.text(65,N*0.05,'N',fontsize=14,bbox=props)


sumA = 3*sumA/sumA.max()
plt.plot(T,sumA+N,'k',linewidth=0.8)
#plt.ylabel('N-traces')
plt.xlabel('Time(s)')



plt.plot(T,0.3*i*AA[:i].sum(axis=0)/AA[:i].sum(axis=0).max()+i+1,'r',linewidth=0.5)
YLIM = plt.ylim()
plt.plot([30,30],YLIM,'k',linewidth=1)
plt.plot([60,60],YLIM,'k',linewidth=1)
plt.xlim([0,90])
props = dict(boxstyle='round', facecolor='white', alpha=1)
plt.text(5,-1,'Vertical',bbox=props)
plt.text(35,-1,'East',bbox=props)
plt.text(65,-1,'North',bbox=props)
plt.xlabel('Time (s)')
plt.ylabel('N traces')
plt.title('Family:%s Station:%s'%(file.split('_')[1],file.split('_')[2]))
plt.show()


#dtfl = h5py.File('./Data/ID_001_CN.YOUB.HH_S.h5', 'r')
dtfl = h5py.File('./Data/ID_001_CN.YOUB.HH_S.h5', 'r')
A = dtfl.get('waves')
AA = np.array(A)
#plt.subplot(1,2,2)
for i in range(len(AA)):
    if QC(AA[i]):
        plt.plot(AA[i]/np.max(AA[i])+i,'k',linewidth=0.5)
    if i==100:
        break
plt.plot(AA[:100].sum(axis=0)/AA[:100].sum(axis=0).max(),'r',linewidth=0.5)




#files = ['./Data/Fam_003_SILB_20050909_150206_P2.mseed',
#         './Data/Fam_003_TWKB_20050909_150206_P2.mseed',
#         './Data/Fam_003_TSJB_20050909_150208_P2.mseed',
#         './Data/Fam_003_SSIB_20050909_150209_P2.mseed',
#         ]
#files = ['./Data/Fam_041_SILB_20050909_150205_P2.mseed',
#         './Data/Fam_041_TWKB_20050909_150206_P2.mseed',
#         './Data/Fam_041_SSIB_20050909_150209_P2.mseed',
#         ]
n = 0
sumD = []
for file in files:
    print(file)
    #D = obspy.read(file)
    #D.detrend('linear')
    #D.taper(0.02)
    #D.filter("bandpass",freqmin=1,freqmax=8)
    D = np.load(file)
    
    data_norm = D/np.max(np.abs(D))
    time = D[0].times()
    plt.plot(time,data_norm+n,color=[0.5,0.5,0.5],linewidth=0.3)
    plt.fill_between(time,np.zeros_like(time)+n,data_norm+n,where=np.zeros_like(time)+n>data_norm+n,color=[0.5,0.5,0.5],interpolate=True)
    #plt.text(30,n,file.split('_')[2])
    sumD.append(D[0].data/np.max(np.abs(D[0].data)))
    n += 1
    if n==50:
        break

YLIM = plt.ylim()
plt.plot([15,15],YLIM,'r--')
sumD = np.array(sumD)
plt.plot(D[0].times(),n+np.sum(sumD,axis=0),'k',linewidth=1.5)
plt.text(2,55,'stack',fontsize=15)
plt.xlim([0,30])
plt.ylim([-1,65])
plt.ylabel('N detections')
plt.xlabel('Time (s)')
plt.title('Family 001 PO.TWKB.HHZ')
plt.grid(False)
