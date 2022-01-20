# check if the csv, hdf5 data matches
import h5py
import pandas as pd
import glob


file_path = '/projects/amt/jiunting/Cascadia_LFE/Detections_P'
#file_path = '/projects/amt/jiunting/Cascadia_LFE/Detections_S/Detections_S_BUG'

csvs = glob.glob(file_path+'/*.csv')

for csv_file in csvs:
    hdf5_file = csv_file.replace('.csv','.hdf5')
    print('now checking:',csv_file)
    # load csv file
    csv = pd.read_csv(csv_file)

    # load hdf5
    hf = h5py.File(hdf5_file,'r')
    for i in range(len(csv)):
        if len(csv)<=100:
            if i%10==0:
                print('  %d of %d'%(i,len(csv)))
        elif 100<len(csv)<=1000:
            if i%100==0:
                print('  %d of %d'%(i,len(csv)))
        else:
            if i%1000==0:
                print('  %d of %d'%(i,len(csv)))
        a = 0
        evid = csv.iloc[i].id
        a = hf['data/'+evid]
        assert a.shape==(3,1500), "shape doesn't match!"

    hf.close()
