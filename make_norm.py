import h5py
import numpy as np

#F1 = h5py.File('Cascadia_lfe_QC_rmean.h5','r') #data
F1 = h5py.File('Cascadia_noise_QC_rmean.h5','r') #noise

#h5f = h5py.File("Cascadia_lfe_QC_rmean_norm.h5", 'w') #data
#h5f.create_dataset('waves', data=F1['waves']/np.max(np.abs(F1['waves']),axis=1).reshape(-1,1))

h5f = h5py.File("Cascadia_noise_QC_rmean_norm.h5", 'w') #noise
h5f.create_dataset('waves', data=F1['waves']/np.max(np.abs(F1['waves']),axis=1).reshape(-1,1))

h5f.close()

F1.close()
