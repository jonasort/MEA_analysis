#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:53:07 2020

@author: jonas
"""

scriptdirectory = "C:/Users/User/Documents/JO/Python Scripts/MEA-master/MEA-master"
inputdirectory = "D:/Files_Reutlingen_Jenny/191017_hdf5"
outputdirectory = "D:/Files_Reutlingen_Jenny191017_output"

import os


os.chdir(scriptdirectory)



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
#import pyspike as spk
#from pyspike import SpikeTrain
#from get_recording_infos import get_channel_infos



'''
FUNCTIONS
'''

def get_channel_infos(filedirectory, meafile):
    channel_raw_data = McsPy.McsData.RawData(os.path.join(filedirectory, meafile))
    print(channel_raw_data.recordings)
    print(channel_raw_data.comment)
    print(channel_raw_data.date)
    print(channel_raw_data.clr_date)
    print(channel_raw_data.date_in_clr_ticks)
    print(channel_raw_data.file_guid)
    print(channel_raw_data.mea_name)
    print(channel_raw_data.mea_sn)
    print(channel_raw_data.mea_layout)
    print(channel_raw_data.program_name)
    print(channel_raw_data.program_version)
    analognumber = len(channel_raw_data.recordings[0].analog_streams.keys())
    print('In total '+ str(analognumber) + " analog_streams were identified.\n")
    for i in range(len(channel_raw_data.recordings[0].analog_streams.keys())):
        keylist = []
        stream = channel_raw_data.recordings[0].analog_streams[i]
        for key in stream.channel_infos.keys():
                keylist.append(key)
        channel_id = keylist[0]
        datapoints = channel_raw_data.recordings[0].analog_streams[i].channel_data.shape[1]
        samplingfrequency = stream.channel_infos[channel_id].sampling_frequency
        ticks = stream.channel_infos[channel_id].info['Tick']
        time = stream.get_channel_sample_timestamps(channel_id)
        scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
        time_in_sec = time[0] * scale_factor_for_second
        timelengthrecording_ms = time[0][-1]+ticks
        timelengthrecording_s = (time[0][-1]+ticks)*scale_factor_for_second
        print("analog_stream Nr. " + str(i) + ": ")
        print("datapoints measured = " + str(datapoints))
        print("sampling frequency = " + str(samplingfrequency))
        print("ticks = " + str(ticks))
        print("total recordingtime is: " + str(timelengthrecording_s) + "seconds \n")




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


def find_triggers(dset_trigger, tick):
    for i in range(0,len(dset_trigger)-1):
        trigger_n=i
        Trigger_An=dset_trigger[trigger_n]
        diff_An=np.diff(Trigger_An)
        peaks, _ = find_peaks(diff_An, height = 2000) #MEA60=0.75
        peaks_off, _ = find_peaks(-diff_An, height = 2000) #""
        if len(peaks)>=0:
            break
    
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
    Trigger_An=[]
    Trigger_Aus=[]
    
    if len(trigger_off)==0:
        Trigger_An=[]
    elif trigger_off[len(trigger_off)-1]>trigger_on[len(trigger_on)-1]:
        Trigger_An=trigger_on*tick
    else:
        Trigger_An=[]
        for n in range(0,len(trigger_on)-1):
            Trigger_An.append(trigger_on[n]*tick)   
        Trigger_An=np.array(Trigger_An)

            
    if len(trigger_on)==0:
        Trigger_Aus=[]
    elif trigger_off[0]>trigger_on[0]:
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
            if len(Trigger_An)==0:
                OFF.append(i)
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
    if len(cutouts)>0:
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
        
        
'''
WORKING SCRIPT.


DOES: 
    - import the Data from the .h5 file
    - transposes it to be worked on
    - 

'''






'''
PLEASE DEFINE THESE PARAMETERS: 
    
    lowcut/highcut for the bandpassfilter
    pre/post as cutout around spikes for waveformanalysis
'''


# set filter cuts in Hz
lowcut = 150
highcut = 5000

# Length of cutouts around shapes
pre = 0.001 # 1 ms
post= 0.002 # 2 ms


os.chdir(inputdirectory)


filelist= glob.glob("*.h5")

resting_spikedic={}
spikedic={}
cutouts_dic ={} 
keylist = []

for file in filelist:
    resting_spikedic={}
    spikedic={}
    cutouts_dic ={} 
    keylist = []
    filename = file
    print('Working on file: ' +filename)
    channel_raw_data = McsPy.McsData.RawData(filename)
    get_channel_infos(inputdirectory, filename)
    analog_stream_0 = channel_raw_data.recordings[0].analog_streams[0]
    stream = analog_stream_0
    for key in stream.channel_infos.keys():
        keylist.append(key)
    channel_id = keylist[0]
    tick = stream.channel_infos[channel_id].info['Tick']
    time = stream.get_channel_sample_timestamps(channel_id)
    scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
    timelengthrecording_ms = time[0][-1]+tick
    timelengthrecording_s = (time[0][-1]+tick)*scale_factor_for_second
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
        spikedic[channellabel] = crossings
        #restingspikedic[channellabel] = crossings[crossings>2750000] # nur crossings, wenn keine stimuli sind
    for i in range (190, 200):
        channel_idx = i
        labellist = get_MEA_Channel_labels(np_analog_for_filter)
        signal_in_uV, time_in_sec, sampling_frequency, scale_factor_for_second = get_MEA_Signal(analog_stream_0, channel_idx)
        bandpassfilteredsignal = butter_bandpass_filter(signal_in_uV, lowcut, highcut, sampling_frequency)
        threshold = -5* np.std(bandpassfilteredsignal)
        crossings = detect_threshold_crossings(bandpassfilteredsignal, sampling_frequency, threshold, dead_time=0.001)
        channellabel = labellist[i]
        spikedic[channellabel] = crossings
        
        #CAVE
        #spikedic hat die ticks der threshold crossings
        #für Umrechnung in Sekunden muss spikedic*scale_factor_for_second*40 
        #(für die microseconds) gerechnet werden
        
        
        
        # bei Bedarf zusätzlich mit den Cutouts der Waveforms möglich
        cutouts = extract_waveforms(bandpassfilteredsignal, sampling_frequency, crossings, pre, post)
        
        cutouts_dic[channellabel] = cutouts
        #creates two dictionaris (ON/OFF)
    trigger_on, trigger_off, diff_An = find_triggers(np_analog_stream_1_data_transpose, tick)
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
        #spikearray= np.array([reloadedspikedic[k] for k in sorted(reloadedspikedic.keys())]).flatten()*40*scale_factor_for_second
        #OFFspikearray= np.array([OFFdic[k] for k in sorted(OFFdic.keys())]).flatten()*40*scale_factor_for_second
        #ONspikearray= np.array([ONdic[k] for k in sorted(ONdic.keys())]).flatten()*40*scale_factor_for_second

    
    os.chdir(outputdirectory)

    np.save(filename+'_crossings_dict.npy', spikedic)
    #np.save(filename+'resting_spikedic.npy', resting_spikedic) 
    np.save(filename+'_wavecutouts_dict.npy', cutouts_dic)
    np.save(filename+'_firingrate_dict.npy', frdic)
    
    os.chdir(inputdirectory)

# Load
#reloadedspikedic = np.load(filename+'_crossings_dict.npy',allow_pickle='TRUE').item()
#restingspikedic = np.load(filename+'resting_spikedic.npy', allow_pickle='TRUE').item()
#coutouts_dic = np.load(filename+'_wavecutouts_dict.npy', allow_pickle='TRUE').item()

        
        
        
