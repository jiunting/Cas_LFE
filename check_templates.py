#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:23:54 2022 

@author:Tim Lin
@email:jiunting@uoregon.edu

Check templates and stacked data from find_template.py
"""
import numpy as np
import matplotlib.pyplot as plt
import glob


templates_path = "./template_match"
sampl = 100 #sampling rate
template_length = 15 # template length in sec
min_Nsta = 3 # number of station involved
min_Nstack = 5

templates_files = glob.glob(templates_path+"/Temp_*.npy")
templates_files.sort()

#templates_files = ["./template_match/Temp_2006-03-01T225500.npy"]
#templates_files = ["./template_match/Temp_2006-03-03T120130.npy"]
#templates_files = ["./template_match/Temp_2005-09-03T071338.425000.npy"]
#templates_files = ["./template_match/Temp_2005-09-03T071350.145000.npy"]
#templates_files = ["./template_match/Temp_2005-09-18T024010.895000.npy"]
templates_files = ["./template_match/Temp_2005-09-18T024019.635000.npy"]
#templates_files = ["./template_match/Temp_2006-03-01T225300.npy"]
#templates_files = ["./template_match/Temp_2006-03-03T120230.npy"]

t = np.arange(int(sampl*template_length+1))/sampl
props = dict(boxstyle='round', facecolor='white', alpha=0.7)
i_comp = 'E'
for templates in templates_files:
    print('Now dealing with:',templates,templates.replace('Temp','CCF').replace('.npy','.png'))
    template_time = templates.split('/')[-1].split('_')[1].replace('.npy','')
    templates = np.load(templates, allow_pickle=True)
    templates = templates.item()
    if len(templates.keys())<min_Nsta:
        continue
    plt.figure(figsize=(12,6.5))
    for i,sta in enumerate(templates.keys()):
        Nstack = templates[sta]['Nstack']
        if Nstack<min_Nstack:
            print('- station %s not enough stacking'%(sta))
            continue
        plt.subplot(1,2,1)
        try:
            plt.plot(t, i*1.5 + templates[sta]['template'][i_comp]/np.max(np.abs(templates[sta]['template'][i_comp])),'-',lw=1.5,alpha=0.9)
        except:
            plt.plot(t, i*1.5 + templates[sta]['template']/np.max(np.abs(templates[sta]['template'])),'-',lw=1.5,alpha=0.9)
        plt.text(0.8,i*1.5+0.3, sta,fontsize=12,bbox=props) #plot station name
        plt.subplot(1,2,2)
        try:
            plt.plot(t, i*1.5 + templates[sta]['stack'][i_comp]/np.max(np.abs(templates[sta]['stack'][i_comp])),'-',lw=1.5,alpha=0.9)
        except:
            plt.plot(t, i*1.5 + templates[sta]['stack']/np.max(np.abs(templates[sta]['stack'])),'-',lw=1.5,alpha=0.9)
        plt.text(t[-1],i*1.5,'%d'%templates[sta]['Nstack'])
    plt.subplot(1,2,1)
    plt.title('Template (%s)'%(template_time),fontsize=14)
    plt.xlim([0,15])
    plt.xlabel('Time (s)',fontsize=14,labelpad=0)
    plt.yticks([],[])
    plt.subplot(1,2,2)
    plt.title('Stacked',fontsize=14)
    plt.xlim([0,15])
    plt.xlabel('Time (s)',fontsize=14,labelpad=0)
    plt.yticks([],[])
    #plt.suptitle(template_time)
    plt.subplots_adjust(left=0.08,top=0.88,right=0.97,bottom=0.1,wspace=0.07)
    plt.show()
    #break


SNB
SSIB
KLNB
