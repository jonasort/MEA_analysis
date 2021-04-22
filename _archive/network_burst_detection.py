#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:08:53 2020

@author: jonas

Script for network burst detection
"""

import os
import sys
import numpy as np
import neo
import pandas as pd
import h5py
from hdfviewer.widgets.HDFViewer import HDFViewer
from hdfviewer.widgets.PathSelector import PathSelector
import McsPy
import sys, importlib, os
import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import bokeh.io
import bokeh.plotting
from bokeh.palettes import Spectral11
from scipy.signal import butter, lfilter, freqz, find_peaks, correlate
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import McsPy
import McsPy.McsData
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
%load_ext autoreload
%autoreload 2
from Butterworth_Filter import butter_bandpass, butter_bandpass_filter
import glob
from quantities import ms, s, Hz
#from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process
import math
from collections import Counter
import scipy




'''
convert channels into neo.spiketrains
'''


def convert_to_neo_spiketrains(spikedictionary):
    '''

    Parameters
    ----------
    expects a dictionary of spikes like this: dic = {str 'channellabel'= key : array ([detectedspikes])=value}

    Returns
    -------
    a dictionary with the channellabels as keys and neo.spiketrains as values

    '''
    spiketraindic = {}
    listexcluded = []
    for key in spikedictionary:
        stop = len(spikedictionary[key])-1
        if stop == -1:
            listexcluded.append(key)
        else:
            spiketrain = neo.SpikeTrain((spikedictionary[key].tolist()), units="microseconds", t_start = 0.0, t_stop = spikedictionary[key][stop])
            spiketraindic[key]=spiketrain
        
    return spiketraindic

plt.eventplot(spikearray, linelengths=0.75, color='black')



array= np.array([reloadedspikedic[k] for k in sorted(reloadedspikedic.keys())]).flatten()
arraysec = array*scale_factor_for_second*40





# detect firing rate for the whole network
# step 1: create one list with every detectet spike

spikes = []

for key in spikedic:
    spikes.append(spikedic[key])
spikelist = np.sort(np.concatenate(spikes, axis = 0))

# step 2: bin all spikes into 10 ms bin 
# calculate to usec 


def create_bins(lower_bound, width, quantity):
    """ create_bins returns an equal-width (distance) partitioning. 
        It returns an ascending list of tuples, representing the intervals.
        A tuple bins[i], i.e. (bins[i][0], bins[i][1])  with i > 0 
        and i < quantity, satisfies the following conditions:
            (1) bins[i][0] + width == bins[i][1]
            (2) bins[i-1][0] + width == bins[i][0] and
                bins[i-1][1] + width == bins[i][1]
    """
    

    bins = []
    for low in range(lower_bound, 
                     lower_bound + quantity*width + 1, width):
        bins.append((low, low+width))
    return bins


def create_ln_bins(start_in_microseconds, stop_in_microseconds, totalbins):
    logbins=[]
    lb = np.logspace(np.log10(start_in_microseconds),np.log10(stop_in_microseconds), totalbins)
    for i in range(0, len(lb)-1):
        logbins.append((lb[i], lb[i+1]))
    return logbins
        
logbins = create_ln_bins(1000, 10000000, 40)
    
    
    
    
    

def find_bin(value, bins):
    """ bins is a list of tuples, like [(0,20), (20, 40), (40, 60)],
        binning returns the smallest index i of bins so that
        bin[i][0] <= value < bin[i][1]
    """
    
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1


from collections import Counter

binned_spikes = []

for value in spikes_usec:
    bin_index = find_bin(value, bins)
    #print(value, bin_index, bins[bin_index])
    binned_spikes.append(bin_index)
    
frequencies = Counter(binned_spikes)
firing_rate_hz = frequencies *1000




import math
spikes_usec = spikelist*tick
binsize = 10000
numberofbins = math.ceil(max(spikes_usec)/binsize)
bins = create_bins(0, binsize, numberofbins)



def find_binned_spikes(data, bins):
    '''
    Parameters
    ----------
    data : for network spike binning --> expects an 1D array with all spikes detected for the network
    bins : list of tuples of expected bins

    Returns
    -------
    binlist : list of lists where lists contain all spikes for the respective bins

    '''
    binlist =[]
    binspike =[]
    for i in range(0, len(bins)):
        binspike = []
        for a in data:    
            if bins[i][0] <= a < bins[i][1]:
                binspike.append(a)
        binlist.append(binspike)
            
    return binlist



binlist = find_binned_spikes(spikes_usec, bins)            
binarray = np.array([np.array(xi) for xi in binlist]) 

bincount = []
for i in range(0, (len(binarray)-1)):
    bincount.append(len(binarray[i]))
    
    
    
    
# if desired the bincount list can be converted to a df

bindf = pd.DataFrame({'spikes_per_ms': bincount})
smalldf = pd.DataFrame({'spikes_per_ms': bincount[11000:]})

bindf["CMA"] = bindf.spikes_per_ms.expanding().mean()
bindf["SMA_50ms"] = bindf.spikes_per_ms.rolling(5, min_periods=1).mean()
bindf["SMA_100ms"] = bindf.spikes_per_ms.rolling(10, min_periods=1).mean()

smalldf["CMA"] = smalldf.spikes_per_ms.expanding().mean()
smalldf["SMA_50ms"] = smalldf.spikes_per_ms.rolling(5, min_periods=1).mean()
smalldf["SMA_100ms"] = smalldf.spikes_per_ms.rolling(10, min_periods=1).mean()
smalldf["SMA_1000ms"] = smalldf.spikes_per_ms.rolling(100, min_periods=1).mean()

colors = ['blue', 'orange', 'green', 'red']


bindf[['spikes_per_ms', 'CMA', 'SMA_50ms', 'SMA_100ms']].plot(color=colors, linewidth=3, figsize=(12,6))
smalldf[['spikes_per_ms', 'CMA', 'SMA_1000ms', 'SMA_100ms']].plot(color=colors, linewidth=1, figsize=(12,6))

    


# Code to detect ISI for every channel of the MEA    

def get_isi_singlechannel(spikedic, tick):
    '''
    Parameters
    ----------
    spikedic : dictionary with all detected spikes for a channel
        DESCRIPTION.

    Returns
    -------
    isidic : keys = channels, values = List of tuples where tuple[0]=detected spike and tuple[1]=isi to the next detected spike
    isi_alone_dic : keys = channels, values = list of isi alone
    CAVE returns are in microseconds
    '''
    
    isidic ={}     
    isilist = []
    isi_alone_dic = {}
    isislist =[]

    for key in spikedic:
        isilist = []
        isislist = []
        if len(spikedic[key])>=2:
            for i in range(0, (len(spikedic[key])-1)):
                isi = spikedic[key][i], (spikedic[key][i+1]-spikedic[key][i])*tick
                isi_alone = (spikedic[key][i+1]-spikedic[key][i])*tick
                isilist.append(isi)
                isislist.append(isi_alone)
        isidic[key]=isilist
        isi_alone_dic[key]=isislist
        
    return isidic, isi_alone_dic
    
isidic, isi_alone = get_isi_singlechannel(spikedic, tick)
# from the isi dic create pd df
O9df = pd.DataFrame({'isi':isi_alone['O9']})
# you can visualise as above
    


# function that bins spikes for one channel 

binsize_isi = 10000 #5ms
number_of_isi_bins = math.ceil(int(max(isi_alone.values())[0])/binsize_isi)# gets the biggest isi and divides it by requested binsize
# ggf. number of bins einfach auf xx Sekunden begrenzen
isibins = create_bins(0, binsize_isi, number_of_isi_bins) 
                              

def bin_isi(isi_alone_dic, binsize, binmax=bool, binmaxnumber=None):
    '''

    Parameters
    ----------
    isi_alone_dic : dic
        dictionary with all ISI for every channel
    binsize: int
        expects int in microseconds that defines bin-width
    Returns
    -------
    histo_ISI_dic:
        dic with key:channellabel, value: list with bincounts per bin

    '''
    isi_bins = []
    isi_bins_list = []
    isi_bin_count = []
    histo_ISI_dic = {}
    for key in isi_alone_dic:
        if binmax==True:
            isi_bin_count=[]
            isibins=create_bins(0, binsize, binmaxnumber)
            isi_bins_list=[] 
            for i in range(0, len(isibins)):
                isi_bins=[]
                for a in isi_alone_dic[key]:
                    if isibins[i][0] <= a < isibins[i][1]:
                        isi_bins.append(a)
                isi_bins_list.append(isi_bins)
            for i in range(0, (len(isi_bins_list)-1)):
                isi_bin_count.append(len(isi_bins_list[i]))
            histo_ISI_dic[key]=isi_bin_count
        #else:
            # noch schreiben für variable maximalnummer an bins
            
    return histo_ISI_dic



def bin_lnisi(isi_alone_dic, start_in_microseconds, stop_in_microseconds, totalbins, binmax):
    '''

    Parameters
    ----------
    isi_alone_dic : dic
        dictionary with all ISI for every channel 
    binsize: int
        expects int in microseconds that defines bin-width
    Returns
    -------
    histo_ISI_dic:
        dic with key:channellabel, value: list with bincounts per logarithmic spaced bins

    '''
    isi_bins = []
    isi_bins_list = []
    isi_bin_count = []
    histo_ln_ISI_dic = {}
    for key in isi_alone_dic:
        if binmax==True:
            isi_bin_count=[]
            isibins=create_ln_bins(start_in_microseconds, stop_in_microseconds, totalbins)
            isi_bins_list=[] 
            for i in range(0, len(isibins)):
                isi_bins=[]
                for a in isi_alone_dic[key]:
                    if isibins[i][0] <= a < isibins[i][1]:
                        isi_bins.append(a)
                isi_bins_list.append(isi_bins)
            for i in range(0, (len(isi_bins_list)-1)):
                isi_bin_count.append(len(isi_bins_list[i]))
            histo_ln_ISI_dic[key]=isi_bin_count
        #else:
            # noch schreiben für variable maximalnummer an bins
            
    return histo_ln_ISI_dic, isibins
            


# created ein dictionary mit 5ms bins und gezählten ISI in pro bin
histo_ISI_dic=bin_isi(isi_alone, binsize=10000, binmax=True, binmaxnumber=300)

# create pd df for one channel+calculate CMA
df_isi_E4 = pd.DataFrame({'ISI_per_5ms_bins':histo_ISI_dic['E4']})
df_isi_E4["CMA"] = df_isi_E4.ISI_per_5ms_bins.expanding().mean()

# plot exemplary channel
df_isi_E4[['ISI_per_5ms_bins', 'CMA']].plot(color=colors, linewidth=3, figsize=(16,6), title = "ISI Histogram + CMA of O10")
                
            
#ln isi
histo_ln_ISI_dic, logbins=bin_lnisi(isi_alone, start_in_microseconds=1, stop_in_microseconds=10000000, binmax=True, totalbins=50)

# create pd df for one channel+calculate CMA
df_ln_isi_O10 = pd.DataFrame({'ISI_per_5ms_bins':histo_ln_ISI_dic['O9']})
df_ln_isi_O10["CMA"] = df_ln_isi_O10.ISI_per_5ms_bins.expanding().mean()

# plot exemplary channel
df_ln_isi_O10[['ISI_per_5ms_bins', 'CMA']].plot(color=colors, linewidth=3, figsize=(16,6), title = "ln(ISI) Histogram + CMA of O10")
                








df= pd.DataFrame({'ISI_per_5ms_bins':histo_ISI_dic[j]})
df["CMA"] = df.ISI_per_5ms_bins.expanding().mean()
df[['ISI_per_5ms_bins', 'CMA']].plot(color=colors, linewidth=3, figsize=(16,6), title="Histogram of "+str(j))



# see a ISI histogramplot with CMA for every channel
for key in histo_ISI_dic:
    j = key
    df= pd.DataFrame({'ISI_per_5ms_bins':histo_ISI_dic[j]})
    df["CMA"] = df.ISI_per_5ms_bins.expanding().mean()
    df[['ISI_per_5ms_bins', 'CMA']].plot(color=colors, linewidth=3, figsize=(16,6), title="Histogram of "+str(j))


# Implementation of MEA-CMA as published by Välkki et al (2017)
    
def get_allchannel_ISI_bins(histo_ISI_dic):
    '''
    Parameters
    ----------
    histo_ISI_dic : dic mit den einzelnen ISI für jeden Channel. Cave, die Values müssen alle die gleiche
                    Länge haben, sonst funktioniert die zip Funktion nicht.
        DESCRIPTION.

    Returns
    -------
    network_ISI_binned = array of all ISI of the whole network binned

    '''
    network_ISI = []
    for key in histo_ISI_dic:
        list1 = histo_ISI_dic[key]
        if len(list1)>len(network_ISI):
            network_ISI=list1
        else:
            list2 = network_ISI
            network_ISI = [a + b for a, b in zip(list1, list2)]
    return np.array(network_ISI)


network_ISI=get_allchannel_ISI_bins(histo_ISI_dic)
plt.hist(network_ISI, normed=False, bins=1000, histtype='stepfilled', color='steelblue',edgecolor='none')



colors = ["blue", "green"]

df= pd.DataFrame({'ISI_per_5ms_bins':network_ISI})
df["CMA"] = df.ISI_per_5ms_bins.expanding().mean()
df[['ISI_per_5ms_bins', 'CMA']].plot(color=colors, linewidth=3, figsize=(16,6), title="Histogram of "+str(j))


# Sample Data for stimulation free period





            
def get_burst_threshold(df_with_CMA):
    
    networkburstthreshold_ISI = 200000 #wie im paper maximal bei 200 ms als isi
    skewness = scipy.stats.skew(network_ISI)
    if skewness < 1:
        alpha1 = 1
        alpha2 = 0.5
    elif skewness >= 1 and skewness <4:
        alpha1 = 0.7
        alpha2 = 0.3
    elif skewness >=4 and skewness <9:
        alpha1 = 0.5
        alpha2 = 0.3
    elif skewness >=9:
        alpha1 = 0.3
        alpha2 = 0.1
    maxCMA = max(df_with_CMA['CMA'])
    CMAalpha = maxCMA*alpha1
    CMAalpha2 = maxCMA*alpha2
    return CMAalpha, CMAalpha2, maxCMA, alpha1, alpha2
    

CMAalpha, CMAalpha2, maxCMA, alpha1, alpha2=get_burst_threshold(df)
            

def ISI_threshold_min(df_with_CMA, CMAalpha, CMAalpha2, binsize_in_micros):
    indexfactor = df[df['CMA']>CMAalpha].index[-1] + 1
    indexfactor2 = df[df['CMA']>CMAalpha2].index[-1] + 1
    threshold_intraburst = float(indexfactor*binsize_in_micros)
    threshold_burst_related = float(indexfactor2*binsize_in_micros)
    
    return threshold_intraburst, threshold_burst_related

threshold_intraburst, threshold_burst_related = ISI_threshold_min(df, CMAalpha, CMAalpha2, 10000)


def find_burst_starts(isi_alone, threshold_intraburst, spikedic):
    burststartdic = {}
    #burststartlist = []
    for key in isi_alone:
        if len(isi_alone[key])<4:
            break
        burststartlist=[]
        counter = 0
        for i in range(0, len(isi_alone[key])-4):
            #print(str(i) + ": "+str(isi_alone[key][i]))
            setter=0
            #if (i+counter)>len(isi_alone[key]):
             #   break
            #if len(isi_alone[key])<4:
             #   break
            if isi_alone[key][i+counter]<threshold_intraburst:
                setter +=1
                if isi_alone[key][i+counter+setter] < threshold_burst_related:
                    setter +=1
                    if isi_alone[key][i+counter+setter] < threshold_burst_related:
                        burststartlist.append(spikedic[key][i])
                        while isi_alone[key][i+counter+setter]<threshold_burst_related and (i+counter+setter)< (len(isi_alone[key])-4):
                            setter +=1
                        #if i+counter+setter> (len(isi_alone[key])-4):
                            #counter = counter + setter -1
                         #   break
                        #else:
                        counter = counter + setter
        burststartdic[key]=burststartlist
    return burststartdic
                        
 
def find_burst_starts(isi_alone, threshold_intraburst, spikedic):
    burststartdic = {}
    noburstlist = []
    #burststartlist = []
    for key in isi_alone:
        print(key)
        if len(isi_alone[key])<3:
            noburstlist.append(isi_alone[key])
        burststartlist=[]
        counter = 0
        while counter < (len(isi_alone[key])-4):
            setter = 0
            if isi_alone[key][counter]<threshold_intraburst:
                setter +=1
                if isi_alone[key][counter+setter] < threshold_intraburst:
                    setter +=1
                    if isi_alone[key][counter+setter] < threshold_intraburst:
                        burststartlist.append(spikedic[key][counter])
                        setter += 1
                        while isi_alone[key][counter+setter]<threshold_intraburst and (counter+setter)< (len(isi_alone[key])-4):
                            setter +=1
                            print('a '+str(setter))
                        setter +=1
                    else:
                        counter +=1
                else:
                    counter +=1
                counter = counter + setter + 1
            else:
                counter +=1
            print(str(key) + str(counter))
        burststartdic[key]=burststartlist
        
    return burststartdic   


E10dic = is


for key in burststartdic:
    for i in range(0, len(burststartdic[key])-1):
        a= (burststartdic[key][i+1])*tick-(burststartdic[key][i])*tick
        if a < 50000:
            print(str(key) +": "+ str(i) + " with " + str(a))




' To Do : hier herausfinden, warum die indexe nicht passen. Das darf eigentlich nicht sein'
' Wenn erledigt, dann das ganze auf den Restingteil andwenden und visuell die spiketrains kontrollieren'  
    
    
    
    
    
    
#sum_list = [a + b for a, b in zip(list1, list2)]








# Get a list of all spiktrains as pyspike spiketrains


def get_pyspike_spiketrains(spikedic):
    '''

    Parameters
    ----------
    spikedic : dictionary with all detected spikes 
        DESCRIPTION.

    Returns
    -------
    spk_st_list : list of all channels as pyspike spiketrains
        DESCRIPTION.

    '''
    spk_st_list =[]
    for key in spikedic:
        st = SpikeTrain(spikedic[key], (0, 3000000))
        spk_st_list.append(st)
    
    return spk_st_list

spike_trains = get_pyspike_spiketrains(spikedic)
isi_profile = spk.isi_profile(spike_trains[0], spike_trains[1])
x, y = isi_profile.get_plottable_data()
plt.plot(x, y, '--k')
print("ISI distance: %.8f" % isi_profile.avrg())
plt.show()





def get_pyspike_spiketrains(spikedic, a, b):
    '''

    Parameters
    ----------
    spikedic : dictionary with all detected spikes 
        DESCRIPTION.

    Returns
    -------
    spk_st_list : list of all channels as pyspike spiketrains
        DESCRIPTION.

    '''
    pyspike_st_dic ={} 
    lazylist = []

    for key in spikedic:
        lazylist = []
        st = SpikeTrain(lazylist, edges=(0, 300000))
        pyspike_st_dic[key]=st
    
    return pyspike_st_dic

a=187579*tick*scale_factor_for_second
b=190079*tick*scale_factor_for_second




pyspikedic = get_pyspike_spiketrains(spikedic, a, b)


isi_profile = spk.isi_profile(pyspikedic['G11'], pyspikedic['E10'])
x, y = isi_profile.get_plottable_data()
plt.plot(x, y, '--k')
print("ISI distance: %.8f" % isi_profile.avrg())
plt.show()







#spike synchronization
spike_profile = spk.spike_sync_profile(pyspikedic['E10'], pyspikedic['G11'])
x, y = spike_profile.get_plottable_data()


pyspike_st_list = []
pyspike_st_order =[]

for key in pyspikedic:
    if len(pyspikedic[key])>0:
        pyspike_st_order.append(str(key))
        pyspike_st_list.append(pyspikedic[key])
    
    
F_init = spk.spike_train_order(pyspike_st_list)
D_init = spk.spike_directionality_matrix(pyspike_st_list)
phi, _ = spk.optimal_spike_train_sorting(pyspike_st_list)
F_opt = spk.spike_train_order(spike_trains, indices=phi)
print("Synfire Indicator of optimized spike train sorting:" +str(F_opt))

D_opt = spk.permutate_matrix(D_init, phi)


plt.figure()
plt.imshow(D_init)
plt.title("Initial Directionality Matrix")

plt.figure()
plt.imshow(D_opt)
plt.title("Optimized Directionality Matrix")

plt.show()

isi_prof = isi_profile ( spike_trains ) 
x , y = isi_prof . g e t _ p l o t t a b l e _ d a t a () plot (x , y , ’ -k ’)
















