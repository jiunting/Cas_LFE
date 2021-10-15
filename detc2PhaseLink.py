# convert all the detection output (daily2input.py) to phase_link PhaseLink prediction file

import pandas as pd
import glob

# for S wave
#csv_path = '/projects/amt/jiunting/Cascadia_LFE/Detections_S'
#outfile = 'CasLFEs_S_y0.1.out'
#outfile = 'CasLFEs_S_y0.7.out'
#thres = 0.1 #minumum y value
#thres = 0.7 #minumum y value

# for P wave
csv_path = '/projects/amt/jiunting/Cascadia_LFE/Detections_P'
outfile = 'CasLFEs_P_y0.1.out'
thres = 0.1 #minumum y value


#csvs = glob.glob(csv_path+'/'+'cut_daily_*.csv')
csvs = glob.glob(csv_path+'/'+'cut_daily_CN.PGC.csv')

all_data = []
for csv in csvs:
    tmp_csv = pd.read_csv(csv)
    all_data.append(tmp_csv)

all_data = pd.concat(all_data)
all_data.reset_index(drop=True, inplace=True)

'''
Example format
network                                     PO
sta                                       TWGB
chn                                         HH
stlon                                 -124.256
stlat                                  48.6076
stdep                                      127
starttime          2005-01-01T07:11:00.000000Z
endtime            2005-01-01T07:11:14.990000Z
y                                         0.51
idx_max_y                                   54
id           PO.TWGB.HH_2005-01-01T07:11:00.54
'''

# apply filter
idx = all_data[all_data['y']>=thres].index

# writing output
with open('tmplog.out','w') as OUT2:
    pass

OUT1 = open(outfile,'w')
Nper = len(idx)//10000
for i in idx:
    if i%Nper==0:
        OUT2 = open('tmplog.out','a')
        OUT2.write('Now at %d total of %d (%f) \n'%(i,len(idx), (i/len(idx))*100  ))
        OUT2.close()
    OUT1.write('%s %s %s %s %.3f %.1f\n'%(all_data.iloc[i].network,all_data.iloc[i].sta,'S',all_data.iloc[i].id.split('_')[-1],all_data.iloc[i].y,0.5))

OUT1.close()
