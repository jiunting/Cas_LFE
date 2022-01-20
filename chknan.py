#check if nan in data

import numpy as np
import h5py


data = h5py.File('Cascadia_noise_data_norm.h5','r')
maxD = np.max(data['waves'])
print('shape should be 1:',np.shape(maxD))
if np.isnan(maxD):
    print('somewhere in noise data is nan')

data = h5py.File('Cascadia_lfe_data_norm.h5','r')
maxD = np.max(data['waves'])
print('shape should be 1:',np.shape(maxD))
if np.isnan(maxD):
    print('somewhere in lfe waveform is nan')
