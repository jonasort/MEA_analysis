#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 07:17:41 2020

@author: Jonas Ort
"""


'''
This script detects spikes from MEA files and then stores the timestamps for
each slice into a .csv file.

This way we want to reduce the amount of data storage used.
'''
# import
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
import scipy



'''
Define directories
'''
# spikedic als masterdictionäre mit dem Label {"R12":array}
spikedic = {}

#part that chooses each file of a directory, exerpts name then 
direcfile = '/Users/jonas/Documents/Code/MEAS_Analysis/'
output ='/Users/jonas/Documents/Code/MEAS_Analysis/OUTPUT/'
meafile = 'HCxA_Chr2_light_5V_100light_2000stop_position8.h5'
filename = meafile
os.chdir(direcfile)

'''
Functions
'''


def get_MEA_Signal(analog_stream, channel_idx, from_in_s=0, to_in_s=None):
    '''
    Extracts one Channels (channel_idx) Sginal 
    
    :param analog_stream = the analogstream from one recording
    :param channel_idx = the channel index of the channel where you extract the values from
    :param from_in_s= starting point of the range you want to observe in seconds
    :param to_in_s= ending point of the range you want to observe. Default is None (i.e. whole range)
    
    Returns: the signal in uV, time stamps in sec, the sampling frequency
    
    
    '''
    ids = [c.channel_id for c in analog_stream.channel_infos.values()]
    channel_id = ids[channel_idx]
    channel_info = analog_stream.channel_infos[channel_id]
    sampling_frequency = channel_info.sampling_frequency.magnitude

    # get start and end index
    from_idx = max(0, int(from_in_s * sampling_frequency))
    if to_in_s is None:
        to_idx = analog_stream.channel_data.shape[1]
    else:
        to_idx = min(analog_stream.channel_data.shape[1], int(to_in_s * sampling_frequency))

    # get the timestamps for each sample
    time = analog_stream.get_channel_sample_timestamps(channel_id, from_idx, to_idx)

    # scale time to seconds:
    scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second

    # get the signal
    signal = analog_stream.get_channel_in_range(channel_id, from_idx, to_idx)

    # scale signal to µV:
    scale_factor_for_uV = Q_(1,signal[1]).to(ureg.uV).magnitude
    signal_in_uV = signal[0] * scale_factor_for_uV
    
    return signal_in_uV, time_in_sec, sampling_frequency, scale_factor_for_second


def get_MEA_Channel_labels(np_analog_for_filter):
    '''
    Gives a List of all MEA Channel Labels (e.g. R12) in the order they appear
    within the recording.
    
    :param analogstream_data = an numpy array shape(channels, data)
    
    '''
    labellist = []
    for i in range(0, len(np_analog_for_filter)):
        #channel_idx = i
        ids = [c.channel_id for c in analog_stream_0.channel_infos.values()]
        channel_id = ids[i]
        channel_info = analog_stream_0.channel_infos[channel_id]
        #print(channel_info.info['Label'])
        labellist.append(channel_info.info['Label'])
    return labellist
    

def detect_threshold_crossings(signal, fs, threshold, dead_time):
    """
    Detect threshold crossings in a signal with dead time and return them as an array

    The signal transitions from a sample above the threshold to a sample below the threshold for a detection and
    the last detection has to be more than dead_time apart from the current one.

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold: The threshold for the signal
    :param dead_time: The dead time in seconds.
    """
    dead_time_idx = dead_time * fs
    threshold_crossings = np.diff((signal <= threshold).astype(int) > 0).nonzero()[0]
    distance_sufficient = np.insert(np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    while not np.all(distance_sufficient):
        # repeatedly remove all threshold crossings that violate the dead_time
        threshold_crossings = threshold_crossings[distance_sufficient]
        distance_sufficient = np.insert(np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    return threshold_crossings


def get_next_minimum(signal, index, max_samples_to_search):
    """
    Returns the index of the next minimum in the signal after an index

    :param signal: The signal as a 1-dimensional numpy array
    :param index: The scalar index
    :param max_samples_to_search: The number of samples to search for a minimum after the index
    """
    search_end_idx = min(index + max_samples_to_search, signal.shape[0])
    min_idx = np.argmin(signal[index:search_end_idx])
    return index + min_idx


def align_to_minimum(signal, fs, threshold_crossings, search_range):
    """
    Returns the index of the next negative spike peak for all threshold crossings

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold_crossings: The array of indices where the signal crossed the detection threshold
    :param search_range: The maximum duration in seconds to search for the minimum after each crossing
    """
    search_end = int(search_range*fs)
    aligned_spikes = [get_next_minimum(signal, t, search_end) for t in threshold_crossings]
    return np.array(aligned_spikes)


def find_triggers(dset_trigger, trigger_n,tick):
    Trigger_An=dset_trigger[trigger_n]
    diff_An=np.diff(Trigger_An)
    peaks, _ = find_peaks(diff_An, height = 2000) #MEA60=0.75
    peaks_off, _ = find_peaks(-diff_An, height = 2000) #""
    
    if trigger_n ==0:
        odd_peaks= peaks
        odd_peaks_off= peaks_off
    else:
        odd_peaks=peaks
        odd_peaks_off=peaks_off
    #x=np.arange(len(Trigger_An))*tick
    #plt.plot(x, Trigger_An)
    return odd_peaks, odd_peaks_off, diff_An

def spike_on_off(trigger_on, trigger_off, spikedic, tick):
    """
    Takes the dictionary with all spikes and sorts them into either a dictionary for
    spikes while trigger on (=ONdic) or off (=OFFdic)
    
    :param trigger_on =basically created through the find_triggers function and marks points were stimulation is turned on
    :param trigger_off =see trigger_on but for stimulation off
    :spikedic = dictionary of spikes for each electrode
    :tick
    """
    on=[]
    off=[]
    ONdic ={}
    OFFdic={}
    
    if trigger_off[len(trigger_off)-1]>trigger_on[len(trigger_on)-1]:
        Trigger_An=trigger_on*tick
    else:
        Trigger_An=[]
        for n in range(0,len(trigger_on)-1):
            Trigger_An.append(trigger_on[n]*tick)   
        Trigger_An=np.array(Trigger_An)
            
    if trigger_off[0]>trigger_on[0]:
        Trigger_Aus=trigger_off*tick
    else:
        Trigger_Aus=[]
        for n in range(1,len(trigger_off)):
            Trigger_Aus.append(trigger_off[n]*tick)
        Trigger_Aus=np.array(Trigger_Aus)
    
    Trigger_Aus2=np.insert(Trigger_Aus,0,0)
    
    for key in spikedic:
        ON = []
        OFF = []
        for i in spikedic[key]: #i mit 40 multiplizieren, da Trigger an und aus mit Tick multipliziert sind
            if any(Trigger_An[foo] < i*tick < Trigger_Aus[foo]  for foo in np.arange(len(Trigger_Aus)-1)):
                ON.append(i)
            elif any(Trigger_Aus2[foo]  < i*tick < Trigger_An[foo]  for foo in np.arange(len(Trigger_An))):
                OFF.append(i)
        ONdic[key]=np.asarray(ON)
        OFFdic[key]=np.asarray(OFF)
    
    return ONdic, OFFdic


def extract_waveforms(signal, fs, spikes_idx, pre, post):
    """
    Extract spike waveforms as signal cutouts around each spike index as a spikes x samples numpy array

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param spikes_idx: The sample index of all spikes as a 1-dim numpy array
    :param pre: The duration of the cutout before the spike in seconds
    :param post: The duration of the cutout after the spike in seconds
    """
    cutouts = []
    pre_idx = int(pre * fs)
    post_idx = int(post * fs)
    for index in spikes_idx:
        if index-pre_idx >= 0 and index+post_idx <= signal.shape[0]:
            cutout = signal[(index-pre_idx):(index+post_idx)]
            cutouts.append(cutout)
    return np.stack(cutouts)


#def plot_waveforms(cutouts, fs, pre, post, n=100, color='k', show=True):
    """
    not working yet!!!
    Plot an overlay of spike cutouts

    :param cutouts: A spikes x samples array of cutouts
    :param fs: The sampling frequency in Hz
    :param pre: The duration of the cutout before the spike in seconds
    :param post: The duration of the cutout after the spike in seconds
    :param n: The number of cutouts to plot, or None to plot all. Default: 100
    :param color: The line color as a pyplot line/marker style. Default: 'k'=black
    :param show: Set this to False to disable showing the plot. Default: True
    """
    if n is None:
        n = cutouts.shape[0]
    n = min(n, cutouts.shape[0])
    time_in_us = np.arange(-pre*1000, post*1000, 1e3/fs)
    if show:
        _ = plt.figure(figsize=(12,6))

    for i in range(n):
        _ = plt.plot(time_in_us, cutouts[i,]*1e6, color, linewidth=1, alpha=0.3)
        _ = plt.xlabel('Time (%s)' % ureg.ms)
        _ = plt.ylabel('Voltage (%s)' % ureg.uV)
        _ = plt.title('Cutouts')

    if show:
        plt.show()





# set filter cuts in Hz
lowcut = 100
highcut = 4000

# Length of cutouts around shapes
pre = 0.001 # 1 ms
post= 0.002 # 2 ms

tick = 40

filelist= glob.glob(direcfile+"/*.h5")

resting_spikedic={}
cutouts_dic ={} 

for file in filelist:
    filename = file
    channel_raw_data = McsPy.McsData.RawData(filename)
    analog_stream_0 = channel_raw_data.recordings[0].analog_streams[0]
    analog_stream_0_data = analog_stream_0.channel_data
    np_analog_stream_0_data = np.transpose(channel_raw_data.recordings[0].analog_streams[0].channel_data)
    np_analog_for_filter = np.transpose(np_analog_stream_0_data)
    np_analog_stream_1_data = np.transpose(channel_raw_data.recordings[0].analog_streams[1].channel_data)
    np_analog_stream_1_data_transpose = np.transpose(np_analog_stream_1_data)
    
    for i in range (0, len(np_analog_for_filter)):
        channel_idx = i
        labellist = get_MEA_Channel_labels(np_analog_for_filter)
        signal_in_uV, time_in_sec, sampling_frequency, scale_factor_for_second = get_MEA_Signal(analog_stream_0, channel_idx)
        bandpassfilteredsignal = butter_bandpass_filter(signal_in_uV, lowcut, highcut, sampling_frequency)
        threshold = -5* np.std(bandpassfilteredsignal)
        crossings = detect_threshold_crossings(bandpassfilteredsignal, sampling_frequency, threshold, dead_time=0.001)
        channellabel = labellist[i]
        #spikedic[channellabel] = crossings
        restingspikedic[channellabel] = crossings[crossings>2750000] # nur crossings, wenn keine stimuli sind
        
        
        #CAVE
        #spikedic hat die ticks der threshold crossings
        #für Umrechnung in Sekunden muss spikedic*scale_factor_for_second*40 
        #(für die microseconds) gerechnet werden
        
        
        
        # bei Bedarf zusätzlich mit den Cutouts der Waveforms möglich
        cutouts = extract_waveforms(bandpassfilteredsignal, sampling_frequency, crossings, pre, post)
        cutouts_dic[channellabel] = cutouts
        #creates two dictionaris (ON/OFF)
        trigger_on, trigger_off, diff_An = find_triggers(np_analog_stream_1_data_transpose, 3, 40)
        ONdic, OFFdic = spike_on_off(trigger_on, trigger_off, spikedic, tick)
        trigger_on_seconds=trigger_on*tick*scale_factor_for_second

        #firing rate CAVE: Falsch da sich die off spikes auf die Gesamtzeit bezgen werden
        frdic = {}
        for key in ONdic:
            total = len(spikedic[key])/time_in_sec[-1]
            frdic[key]=total
#            rate_on = len(ONdic[key])/time_in_sec[-1]
 #           frdic[key+'_on'] = rate_on
  #          rate_off = len(OFFdic[key])/time_in_sec[-1]
   #         frdic[key+'_off'] = rate_off
    #        total = len(spikedic[key])/time_in_sec[-1]
     #       frdic[key+'_total'] = total
            
        
        #for plotting
        spikearray= np.array([reloadedspikedic[k] for k in sorted(reloadedspikedic.keys())]).flatten()*40*scale_factor_for_second
        OFFspikearray= np.array([OFFdic[k] for k in sorted(OFFdic.keys())]).flatten()*40*scale_factor_for_second
        ONspikearray= np.array([ONdic[k] for k in sorted(ONdic.keys())]).flatten()*40*scale_factor_for_second




np.save(filename+'_crossings_dict.npy', spikedic)
np.save(filename+'resting_spikedic.npy', resting_spikedic) 
np.save(filename+'_wavecutouts_dict.npy', cutouts_dic)

# Load
reloadedspikedic = np.load(filename+'_crossings_dict.npy',allow_pickle='TRUE').item()
restingspikedic = np.load(filename+'resting_spikedic.npy', allow_pickle='TRUE').item()
coutouts_dic = np.load(filename+'_wavecutouts_dict.npy', allow_pickle='TRUE').item()







'''
plotting sandbox
'''
plt.figure(figsize=(30, 10))
plt.eventplot((ONspikearray)*scale_factor_for_second*40, linelengths=0.75, color='black')
plt.xlabel('Time (s)', fontsize=16)
plt.yticks([0,1], labels=["Channels MEA"], fontsize=16)
plt.title("Figure 1")




E4_array = np.array(spikedic['E4']).flatten()*40*scale_factor_for_second
E4_burststarts = np.array(burststartdic['E4']).flatten()*scale_factor_for_second
trigger_on_seconds=trigger_on*tick*scale_factor_for_second

fig = plt.figure(figsize=(30, 10))
ax = fig.add_subplot(1, 1, 1)
ax.eventplot(E4_burststarts, color='red')
ax.eventplot(E4_array, linelengths=0.5, color='black')
ax.eventplot(trigger_on_seconds, linelengths=0.75, color='blue')
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Channel E4 - Eventplot with burststarts')

plt.show()



G11_array = np.array(spikedic['G11']).flatten()*40*scale_factor_for_second
G11_burststarts = np.array(burststartdic['G11']).flatten()*scale_factor_for_second

fig = plt.figure(figsize=(30, 10))
ax = fig.add_subplot(1, 1, 1)
ax.eventplot(G11_burststarts, color='red')
ax.eventplot(G11_array, linelengths=0.5, color='black')
ax.eventplot(trigger_on_ms, linelengths=0.75, color='blue')
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Channel G11 - Eventplot with burststarts')

plt.show()



def plot_burststarts(spikedic, burststartdic, key):
    
    array = np.array(spikedic[i]).flatten()*40*scale_factor_for_second
    burststarts = np.array(burststartdic[i]).flatten()*scale_factor_for_second
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.eventplot(E10_burststarts, color='red')
    ax.eventplot(E10_array, linelengths=0.5, color='black')
    ax.set_xlabel('Time (s)', fontsize=16)
    #ax.set_ylabel('Channel % - Eventplot with burststarts' %str(key))
    ax.set_ylabel('Channel '+str(i) +' Burststarts')
    
    
    return plt.show()

    
    #plt.savefig(direcfile + '/burstplot_'+str(i))
    
    
    
    
    df= pd.DataFrame({'ISI_per_5ms_bins':histo_ISI_dic[j]})
    df["CMA"] = df.ISI_per_5ms_bins.expanding().mean()
    df[['ISI_per_5ms_bins', 'CMA']].plot(color=colors, linewidth=3, figsize=(16,6), title="Histogram of "+str(j))

trigger_on_seconds=trigger_on*tick*scale_factor_for_second

# plots a Burstplot + trigger for every channel
for key in burststartdic:
    array = np.array(spikedic[key]).flatten()*40*scale_factor_for_second
    burststarts = np.array(burststartdic[key]).flatten()*scale_factor_for_second
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.eventplot(burststarts, linelengths = 1.0, color='red')
    ax.eventplot(array, linelengths=0.5, color='black')
    ax.eventplot(trigger_on_seconds, linelengths=0.75, color='blue')
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_title('Channel '+str(key) +' Burststarts')
    #ax.set_ylabel('Channel % - Eventplot with burststarts' %str(key))
    #ax.set_ylabel('Channel '+str(key) +' Burststarts')
    
    
    plt.show()
    fig.savefig(output+ '/burstplot_'+str(key)+'.png')
    plt.close()
    
    
#ZHG4UAXQCJYABJ6WI5SFMG6Q


direcfile= direcfile+'burstplots/'


plt.style.use('seaborn')

# Versuch einen Parallelplot der interessanten Spiketrains zu bauen
interesting_keys=['C6', 'C11', 'E1', 'E8', 'E9', 'E10', 'E13', 'F7', 'G7', 'G10', 'G11', 'G14']
interesting_keys=['E1', 'G10', 'I8', 'I9', 'K10', 'L8', 'M7', 'O4', 'O5']


interestingdic = {}
for i in interesting_keys:
    interestingdic[i]=spikedic[i]
    
interestingarray= np.array([interestingdic[k] for k in sorted(interestingdic.keys())]).flatten()*40*scale_factor_for_second

# Plot with several channels paralell
fig = plt.figure(figsize=(30,10))
i = 1
for key in interestingdic:
    if i == 1:
        ax= fig.add_subplot(len(interestingdic), 1, i)
    else:
        ax= fig.add_subplot(len(interestingdic), 1, i, sharex=ax)  
    array = np.array(spikedic[key]).flatten()*40*scale_factor_for_second
    burststarts = np.array(burststartdic[key]).flatten()*scale_factor_for_second
    ax.eventplot(burststarts, linelengths = 1.0, color='red')
    ax.eventplot(array, linelengths=0.5, color='black')
    ax.eventplot(trigger_on_seconds, linelengths=0.75, color='blue')
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel(str(key))
    i +=1


def dictionary_cutter(spikedic, burststartdic, trigger_on_seconds, interesting_keys, starttime, stoptime, tick, scale_factor_for_second):
    '''
    Kürzt das spikedic und das burststartdic sowie die trigger auf einen gewünschten Zeitraum ein, welcher
    in Sekunden (starttime, stoptime) angegeben werden muss.
    
    Parameters
    ----------
    spikedic : TYPE
        DESCRIPTION.
    burststartdic : TYPE
        DESCRIPTION.
    trigger_on_seconds : TYPE
        DESCRIPTION.
    interesting_keys : TYPE
        DESCRIPTION.
    starttime : TYPE
        DESCRIPTION.
    stoptime : TYPE
        DESCRIPTION.
    tick : TYPE
        DESCRIPTION.
    scale_factor_for_second : TYPE
        DESCRIPTION.

    Returns
    -------
    cutted_spikes : TYPE
        DESCRIPTION.
    cutted_bursts : TYPE
        DESCRIPTION.

    '''
    # alle benötigten leeren objekte erstellen
    interestingdic = {}
    cutted_spikes ={} 
    cutted_bursts ={}
    uselist = []
    
    scaler_to_seconds = tick*scale_factor_for_second
    for i in interesting_keys:
        interestingdic[i]=spikedic[i]
    interestingarray= np.array([interestingdic[k] for k in sorted(interestingdic.keys())]).flatten()*40*scale_factor_for_second
    
    # cutter --> soll beobachteten Raum einschneiden
    start = starttime
    stop = stoptime
    for key in spikedic:
        uselist = []
        for i in range(0,len(spikedic[key])-1):
            if spikedic[key][i]*scaler_to_seconds >= start and spikedic[key][i]*scaler_to_seconds < stop:
                uselist.append(spikedic[key][i])
        cutted_spikes[key]=uselist
    for key in burststartdic:
        uselist = []
        for i in range(0, len(burststartdic[key])-1):
            if burststartdic[key][i]*scale_factor_for_second >= start and burststartdic[key][i]*scale_factor_for_second < stop:
                uselist.append(burststartdic[key][i])
        cutted_bursts[key]=uselist
    uselist = []
    for i in trigger_on_seconds:
        if i >= start and i < stop:
            uselist.append(i)
        cutted_trigger=np.array(uselist)
            
    return cutted_spikes, cutted_bursts, cutted_trigger
  
'''
Mit obiger Funktion und untem durchgeführten plot lassten sich die gewünschten channels zu einem jeweiligen
Zeitraum mit detektiertem Burst und 

'''
cutted_spikes, cutted_bursts, cutted_trigger = dictionary_cutter(spikedic, burststartdic, trigger_on_seconds, interesting_keys, starttime=53, stoptime=60, tick=tick, scale_factor_for_second=scale_factor_for_second)
  

# neues interesting dic mit den cutted values
interestingdic = {}
for i in interesting_keys:
    interestingdic[i]=cutted_spikes[i]

# entsprechender Plot, welcher bei ax.set_xlim noch angepasst werden muss an die Zeiten die beobachtet werden  
fig = plt.figure(figsize=(20,10))
i = 1
for key in interestingdic:
    if i == 1:
        ax= fig.add_subplot(len(interestingdic), 1, i)
    else:
        ax= fig.add_subplot(len(interestingdic), 1, i, sharex=ax)
    array = np.array(cutted_spikes[key]).flatten()*40*scale_factor_for_second
    burststarts = np.array(cutted_bursts[key]).flatten()*scale_factor_for_second
    ax.eventplot(burststarts, linelengths = 1.0, color='red')
    ax.eventplot(array, linelengths=0.5, color='black')
    ax.eventplot(cutted_trigger, linelengths=0.75, color='blue')
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel(str(key))
    ax.set_xlim(53, 60) #CAVE: hier die Zeiten anpassen etnsprechend dem cuttdictionary
    ax.set_ylim(0, 2)
    i +=1






'''
SPIKE Synchrony, Leader/Follower, Delayplot Sandbox
'''

import pyspike as spk
from pyspike import SpikeTrain

ssdic = {}


# für interesting keys Bestimmung anhand der Firing Rate
for key in frdic:
    if frdic[key]>0.5:
        ssdic[key]=frdic[key]


interesting_keys=list(ssdic.keys())

interestingdic = {} # ggf. in Sekunden
for i in interesting_keys:
    interestingdic[i]=spikedic[i]*scale_factor_for_second*tick
    
    

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
        st = SpikeTrain(spikedic[key], edges=(a, b))
        pyspike_st_dic[key]=st
    
    return pyspike_st_dic


# immer an gewünschte Zeit anpassen
a=53.69
b=54.1


pyspikedic = get_pyspike_spiketrains(interestingdic, a, b) #creates a dic with 

spiketrainlist = []
st_orderlist=[]

for key in pyspikedic:
    spiketrainlist.append(pyspikedic[key])
    st_orderlist.append(key)





'''
Create Spike Sync Profile
'''
spike_profile = spk.spike_sync_profile(spiketrainlist)
x, y = spike_profile.get_plottable_data()

fig = plt.figure(figsize=(20,10))
ax2 = fig.add_subplot()
ax2.plot(x, y)
ax2.set_xlim(a, b)


'''
Für das Spikesyncprofile ein entsprechender zusätzlicher Plot, der Sypike Synchronität anzeigt
'''
fig = plt.figure(figsize=(120,40))
i = 1
for key in interestingdic:
    if i == 1:
        ax= fig.add_subplot(len(interestingdic)+1, 1, i)
    else:
        ax= fig.add_subplot(len(interestingdic)+1, 1, i, sharex=ax)
    array = np.array(cutted_spikes[key]).flatten()*40*scale_factor_for_second
    burststarts = np.array(cutted_bursts[key]).flatten()*scale_factor_for_second
    ax.eventplot(burststarts, linelengths = 1.0, color='red')
    ax.eventplot(array, linelengths=0.5, color='black')
    ax.eventplot(cutted_trigger, linelengths=0.75, color='blue')
    ax.set_xlabel('Time (s)', fontsize=16)
    ax.set_ylabel(str(key))
    ax.set_xlim(a, b) #CAVE: hier die Zeiten anpassen etnsprechend dem cuttdictionary
    ax.set_ylim(0, 2)
    i +=1
ax2=fig.add_subplot(len(interestingdic)+1, 1, i, sharex=ax)
ax2.plot(x,y, 'd--')
ax.set_xlim(a, b)












plt.figure()
isi_distance = spk.isi_distance_matrix(spiketrainlist)
plt.imshow(isi_distance, interpolation='none')
plt.title("ISI-distance")

plt.figure()
spike_distance = spk.spike_distance_matrix(spiketrainlist)
plt.imshow(spike_distance, interpolation='none')
plt.title("SPIKE-distance")

plt.figure()
spike_sync = spk.spike_sync_matrix(spiketrainlist)
plt.imshow(spike_sync, interpolation='none')
plt.title("SPIKE-Sync")

plt.show()




spiketrainorder = spk.spike_train_order(spiketrainlist)







df = pd.DataFrame(spike_sync, index=st_orderlist, columns=st_orderlist)
sns.clustermap(df, cmap='vlag', linewidths=.75, figsize=(20, 20))
sns.heatmap(df, cmap='vlag', linewidths=.75)



stopm = spk.spike_train_order_multi(spiketrainlist)


E = spk.spike_train_order_profile(spiketrainlist)

plt.figure()
x, y = E.get_plottable_data()
plt.plot(x, y, '-ob')
plt.ylim(-1.1, 1.1)
plt.xlim(a,b)
plt.xlabel("t")
plt.ylabel("E")
plt.title("Spike Train Order Profile")

plt.show()










df = pd.DataFrame(D_init, index=st_orderlist, columns=st_orderlist)
df = pd.DataFrame(D_opt)

E = spk.spike_directionality_values(spiketrainlist)

'''

'''
F_init = spk.spike_train_order(spiketrainlist)
print "Initial Synfire Indicator for 20 Poissonian spike trains:", F_init

D_init = spk.spike_directionality_matrix(spiketrainlist)
phi, _ = spk.optimal_spike_train_sorting(spiketrainlist)
F_opt = spk.spike_train_order(spiketrainlist, indices=phi)
print "Synfire Indicator of optimized spike train sorting:", F_opt

D_opt = spk.permutate_matrix(D_init, phi)

plt.figure()
plt.imshow(D_init)
plt.title("Initial Directionality Matrix")

plt.figure()
plt.imshow(D_opt)
plt.title("Optimized Directionality Matrix")

plt.show()



















M = 20
spike_trains = [spk.generate_poisson_spikes(1.0, [0, 100]) for m in range(M)]

F_init = spk.spike_train_order(spike_trains)
print "Initial Synfire Indicator for 20 Poissonian spike trains:", F_init

D_init = spk.spike_directionality_matrix(spike_trains)
phi, _ = spk.optimal_spike_train_sorting(spike_trains)
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



'''
Ideen --> pyspike auf das burststartdic anwenden, also quasi starts der Bursts beobachten
und von dort aus die leader/follower Definition laufen lassen.

Ab da möglicherweise den Delay Plot von Julia umschreiben. 




'''

