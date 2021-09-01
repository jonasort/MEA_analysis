from neo.io import NeuroExplorerIO
import os
import glob
import neo
import elephant
import numpy as np
import pandas as pd
import quantities as pq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from quantities import ms, s, Hz
from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process
from elephant import neo_tools as nt

from elephant.spade import spade
from elephant.spade import pvalue_spectrum
from elephant.spade import concepts_mining
from elephant.spade import concept_output_to_patterns
import quantities as pq


scale_factor_for_second = 1e-06
scale_factor_for_milisecond = 1e-03
tick=40


filepath = r"../input/burststart-spade"


filelist = glob.glob('*.npy')
filelist
filenamelist = []
for i in filelist:
    filenamelist.append(i.split('.')[0])



def spikedic_to_neospiketrains(reloadedspikedic, recordinglength):
    
    spiketrains = [] 
    keylist_spiketrains = []
    for key in reloadedspikedic:
        key_array=np.asarray(reloadedspikedic[key])
        key_array_sec=key_array*scale_factor_for_second*tick
        
        # adjust the minimal amount of spikes per spiketrain
        if len(key_array_sec)>0:
            st = neo.SpikeTrain(list(key_array_sec), units='sec', t_stop=recordinglength)
            spiketrains.append(st)
            keylist_spiketrains.append(key)
        
    return spiketrains, keylist_spiketrains 




bin_size = 5 * pq.ms # time resolution to discretize the spiketrains
winlen = 20 # maximal pattern length in bins (i.e., sliding window)
dither = 20 * pq.ms
spectrum = '3d#'
alpha = 0.05
stat='fdr_bh'


fileidentifier = 'binsize-'+str(int(bin_size))+'_winlen-'+str(winlen)+'_dither-'+str(int(dither))+'_spectrum-'+spectrum+'_statcor-'+stat




reloadedspikedic = np.load(filelist[0],allow_pickle='TRUE').item() 



#for i in range(0,len(filenamelist)):
for i in range(0,1): 
    
    filename = filelist[i]
    filenamebase = filename.split('.npy')[0]
    
    #normalerweise wird hier das .item() statement benutzt. 
    #Dieses scheint den gecutteten Spikedictionaries nicht zu passen da es als dictionary innerhalb eines arrays abgebildet ist
    reloadedspikedic = np.load(filelist[i],allow_pickle='TRUE').item() 
    
    spikes = []
    for key in reloadedspikedic:
        spikes.append(reloadedspikedic[key])
    spikearray = np.sort(np.concatenate(spikes, axis = 0))
    spikelist = list(spikearray)

    spikearray_sec=spikearray*scale_factor_for_second*tick
    recordinglength = round(spikearray_sec[-1]) + 1
    print(filelist[i] + ' - Recording length: '+str(recordinglength))
    spiketrains, keylist_spiketrains = spikedic_to_neospiketrains(reloadedspikedic, recordinglength)
    
    result_spade = spade(spiketrains, bin_size=bin_size, winlen=winlen, n_surr=2000, min_occ=4, min_spikes=2, min_neu=2, stat_corr=stat, 
                        spectrum=spectrum, alpha=alpha)
    
    os.chdir('/kaggle/working')
    np.save(filenamebase+'_SPADE_'+fileidentifier+'.npy', result_spade)
    print('Finished '+filename)
    os.chdir(filepath)
print('Get me a beer.')