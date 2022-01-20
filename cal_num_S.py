import glob
import h5py
import numpy as np


dataName = glob.glob('./Data/*_P.h5')

N = 0
for i_file,lfeFile in enumerate(dataName):
    #OUT1 = open('lfe_data.log','a')
    #OUT1.write('Now in %d out of %d\n'%(i_file,len(dataName)))
    #OUT1.close()
    tmp = h5py.File(lfeFile, 'r')
    data = tmp['waves']
    #data = np.array(data)
    print('In %d out of %d N=%d after=%d '%(i_file,len(dataName),N,N+len(data)))
    N = N + len(data)

print('number of data=',N)
