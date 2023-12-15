#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:17:17 2023

@author: jonas
"""

import os
import sys
import numpy as np
import neo
import pandas as pd
import h5py
import ast
import McsPy
import sys, importlib, os
import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, find_peaks, correlate, gaussian, filtfilt
from scipy import stats
from scipy import signal
from scipy import stats
from scipy import signal
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import McsPy
import McsPy.McsData
from McsPy import ureg, Q_
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import scipy
from scipy.stats import pearsonr
import time
import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
import copy
import pickle
import fnmatch

# Plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
#networkx
import plotly.graph_objects as go
import networkx as nx
import matplotlib.patches as mpatche







'''
Functions
'''


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y    
    
     


def detect_peaks(y):
    threshold =5 * np.std(y) #np.median(np.absolute(y)/0.6745)
    peaks, _ = find_peaks(-y, height= threshold, distance=50)   
    return peaks,y,threshold



def get_channel_infos(filedirectory, meafile):
    channel_raw_data = McsPy.McsData.RawData(os.path.join(filedirectory, 
                                                          meafile))
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
    print('In total '+ str(analognumber) 
          + " analog_streams were identified.\n")
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
        print("total recordingtime is: " 
              + str(timelengthrecording_s) + "seconds \n")




def get_MEA_Signal(analog_stream, channel_idx, from_in_s=0, to_in_s=None):
    '''
    Extracts one Channels (channel_idx) Sginal 
    
    :param analog_stream = the analogstream from one recording
    :param channel_idx   = the channel index of the channel where you 
                            extract the values from
    :param from_in_s     = starting point of the range you want to observe 
                            in seconds
    :param to_in_s       = ending point of the range you want to observe. 
                            Default is None (i.e. whole range)
    
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
        to_idx = min(
            analog_stream.channel_data.shape[1], 
            int(to_in_s * sampling_frequency)
            )

    # get the timestamps for each sample
    time = analog_stream.get_channel_sample_timestamps(
        channel_id, from_idx, to_idx
        )

    # scale time to seconds:
    scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second

    # get the signal
    signal = analog_stream.get_channel_in_range(channel_id, from_idx, to_idx)

    # scale signal to µV:
    scale_factor_for_uV = Q_(1,signal[1]).to(ureg.uV).magnitude
    signal_in_uV = signal[0] * scale_factor_for_uV
    
    return signal_in_uV, time_in_sec, sampling_frequency, scale_factor_for_second


def get_MEA_Channel_labels(np_analog_for_filter, analog_stream_0):
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
    Detect threshold crossings in a signal with dead time and return 
    them as an array

    The signal transitions from a sample above the threshold to a sample 
    below the threshold for a detection and
    the last detection has to be more than dead_time apart 
    from the current one.

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold: The threshold for the signal
    :param dead_time: The dead time in seconds.
    """
    dead_time_idx = dead_time * fs
    threshold_crossings = np.diff(
        (signal <= threshold).astype(int) > 0).nonzero()[0]
    distance_sufficient = np.insert(
        np.diff(threshold_crossings) >= dead_time_idx, 0, True
        )
    while not np.all(distance_sufficient):
        # repeatedly remove all threshold crossings that violate the dead_time
        threshold_crossings = threshold_crossings[distance_sufficient]
        distance_sufficient = np.insert(
            np.diff(threshold_crossings) >= dead_time_idx, 0, True
            )
    return threshold_crossings


def get_next_minimum(signal, index, max_samples_to_search):
    """
    Returns the index of the next minimum in the signal after an index

    :param signal: The signal as a 1-dimensional numpy array
    :param index: The scalar index
    :param max_samples_to_search: The number of samples to search for a 
                                    minimum after the index
    """
    search_end_idx = min(index + max_samples_to_search, signal.shape[0])
    min_idx = np.argmin(signal[index:search_end_idx])
    return index + min_idx


def align_to_minimum(signal, fs, threshold_crossings, search_range, first_time_stamp=0):
    """
    Returns the index of the next negative spike peak for all threshold crossings

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold_crossings: The array of indices where the signal 
                                crossed the detection threshold
    :param search_range: The maximum duration in seconds to search for the 
                         minimum after each crossing
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
    
    :param trigger_on =basically created through the find_triggers function 
                        and marks points were stimulation is turned on
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
            cutout = signal[int((index-pre_idx)):int((index+post_idx))]
            cutouts.append(cutout)
    if len(cutouts)>0:
        return np.stack(cutouts)
    
 
def plot_waveforms(cutouts, fs, pre, post, n=100, color='k', show=True):
    """
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
        
        
def butter_lowpass_filter(data, cutoff, fs, order, nyq):

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
        
        
        
def get_next_maximum(signal, index, max_samples_to_search):
    """
    Returns the index of the next maximum in the signal after an index

    :param signal: The signal as a 1-dimensional numpy array
    :param index: The scalar index
    :param max_samples_to_search: The number of samples to search for a 
                                    minimum after the index
    """
    search_end_idx = min(index + max_samples_to_search, signal.shape[0])
    max_idx = np.argmax(signal[index:search_end_idx])
    return index + max_idx



def lfp_crossing_detection(lowpass_filtered_signal, lfp_threshold, 
                           tick, scale_factor_for_second,
                           time_in_sec, minimal_length):
    
    '''
    parameters 
    
    lowpass_filtered_signal : array like / list
        the lowpass filtered signal which is considered as the LFP
    
    lfp_threshold : float / int
        the threshold when there is a crossing we regard it as LFP
        deviation
        
    minimal_length : float
        minimal length of a LFP deviation to be considered relevant,
        default is 50ms
        the value is given in seconds
        
    _________
    
    returns:
    
    lfp_down_crossing : list
        list of tuples (a,b) with a start of a deviation, b stop in seconds
        
    lfp_up_crossing : list
        analogue to lfp_down_crossing
        
    amplitudes_down : list
        every down crossing has its local minimum which is the maximal negative amplitude
    
    
    amplitudes_down : list
        every up crossing has its local minimum which is the maximal negative amplitude
    
    '''

    # dicts will have tuples with a start and stop of the lfp crossing
    lfp_up_crossing = []
    lfp_down_crossing = []
    amplitudes_up = []
    amplitudes_down = []
    
    # lfp crosses below threshold
    for i in range(0, len(lowpass_filtered_signal)-1):
        start = 0
        stop = 0
        if (lowpass_filtered_signal[i] < -lfp_threshold) and (lowpass_filtered_signal[i-1] >= -lfp_threshold):
            start = i
            while (lowpass_filtered_signal[i] < -lfp_threshold) and (i < len(lowpass_filtered_signal)-1):
                stop = i
                i += 1
            # filter for at least 50ms  of LFP deviation
            start_seconds = start*scale_factor_for_second*tick + time_in_sec[0] #added since recording do not always start at 0
            stop_seconds = stop*scale_factor_for_second*tick +time_in_sec[0]#added since recording do not always start at 0
            difference_seconds = stop_seconds - start_seconds
            if difference_seconds >= 0.05: # in seconds --> 50 ms
            
                lfp_down_crossing.append((start_seconds, stop_seconds))
                amplitude_point = get_next_minimum(lowpass_filtered_signal, start, stop-start)
                amplitude_down = lowpass_filtered_signal[amplitude_point]
                amplitudes_down.append(amplitude_down)
            
    # lfp crosses above threshold
    
    for i in range(0, len(lowpass_filtered_signal)-1):
        start = 0
        stop = 0
        if (lowpass_filtered_signal[i] > lfp_threshold) and (lowpass_filtered_signal[i-1] <= lfp_threshold):
            start = i
            while (lowpass_filtered_signal[i] > lfp_threshold) and (i < len(lowpass_filtered_signal)-1):
                stop = i
                i += 1
            # filter for at least 50ms  of LFP deviation
            start_seconds = start*scale_factor_for_second*tick +time_in_sec[0]#added since recording do not always start at 0
            stop_seconds = stop*scale_factor_for_second*tick +time_in_sec[0]#added since recording do not always start at 0
            difference_seconds = stop_seconds - start_seconds
            if difference_seconds >= 0.05: # in seconds --> 50 ms
            
                lfp_up_crossing.append((start_seconds, stop_seconds))
                amplitude_point = get_next_maximum(lowpass_filtered_signal, start, stop-start)
                amplitude_up = lowpass_filtered_signal[amplitude_point]
                amplitudes_up.append(amplitude_up)


    return lfp_down_crossing, lfp_up_crossing, amplitudes_down, amplitudes_up




def get_isi_single_channel(spikedic, tick, scale_factor_for_milisecond):
    
    '''
    input: 
        spikedic with keys = channellabels, values = spiketimes in raw ticks
    
    
    returns: 

        dictionary with keys = channellabels, values = isi per channel in miliseconds
        
        
    nota bene:
        the amount of spikes is not filtered, we still need to factor out non relevant channels
    
    '''
    
    # set the empty dictionary and temporary list
    isi_dictionary = {}
    isi_temp_list =[]
    
    
    for key in spikedic:
        isi_temp_list =[]
        spikes = spikedic[key]
        spikes = [spike * tick * scale_factor_for_milisecond for spike in spikes]
        
        if len(spikes) >= 2:
            for i in range(0, len(spikes)-1): 

                # calculate the isi
                isi =  spikes[i+1] - spikes[i] 
                isi_temp_list.append(isi)

        isi_dictionary[key] = isi_temp_list
        
    
    return isi_dictionary



def find_random_spikes(spikedic, networkbursts, tick, scale_factor_for_second):
    '''

    Parameters
    ----------
    spikedic : dic
        keys = channellabel, values = spiketimes as tick.
    networkbursts : list of tuples (a,b)
        a = start of a networkactivity
        b = stop of a networkactivity
        in seconds
        
        

    Returns
    -------
    random_nrandom_spike_per_channeldic : dic
        keys = channellabels
        values = 
    number_rand_nrandom_spike_per_channeldic : dic
        DESCRIPTION.
    total_non_random_spikes : int
        DESCRIPTION.
    total_random_spikes : int
        DESCRIPTION.

    '''
    total_non_random_spikes = 0
    total_random_spikes = 0
    singlechannel_random = []
    singlechannel_non_random = []
    
    
    random_nrandom_spike_per_channeldic = {}
    number_rand_nrandom_spike_per_channeldic = {}
    
    for key in spikedic:
        singlechannel_random = []
        singlechannel_non_random = []
        for i in networkbursts:
            start = i[0]
            stop = i[1]
            for j in spikedic[key]:
                if start < (j*tick*scale_factor_for_second) < stop:
                    singlechannel_non_random.append(j)
    
                    
        allchannelspikes = spikedic[key].copy()
        print(len(allchannelspikes)-len(singlechannel_non_random))
        
        
        singlechannel_random = [i for i in allchannelspikes if i not in singlechannel_non_random]
        print(singlechannel_random)
        '''
        for k in singlechannel_non_random:
            print(k)
            allchannelspikes.remove(k)
        singlechannel_random = allchannelspikes.copy()
        '''    
           
        random_nrandom_spike_per_channeldic[key] = (singlechannel_random,
                                                    singlechannel_non_random)
        number_rand_nrandom_spike_per_channeldic[key] = (len(singlechannel_random),
                                                         len(singlechannel_non_random))
        
        total_non_random_spikes += len(singlechannel_non_random)
        total_random_spikes += len(singlechannel_random)
 
    
    return random_nrandom_spike_per_channeldic, number_rand_nrandom_spike_per_channeldic, total_non_random_spikes, total_random_spikes

            
    
    


def gaussian_smoothing(y, window_size=10, sigma=2):

    filt = signal.gaussian(window_size, sigma)

    return signal.convolve(y, filt, mode='full')





def invert_layerdic(layer_dic):
    
    '''
    Expects a dictionary with key = layer, value = list of channellabels
    
    Returns a dictionary with key = channellabels, value = layer
    '''
    layerdic_invert = {}

    for key in layer_dic:
        for i in layer_dic[key]:
            layerdic_invert[i]=key
            
            
    return layerdic_invert




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


def find_bin(value, bins):
    """ bins is a list of tuples, like [(0,20), (20, 40), (40, 60)],
        binning returns the smallest index i of bins so that
        bin[i][0] <= value < bin[i][1]
    """
    
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1



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




def get_isi_singlechannel(spikedic, tick):
    '''
    Parameters
    ----------
    spikedic : dictionary with all detected spikes for a channel
        DESCRIPTION.

    Returns
    -------
    isidic : keys = channels, values = List of tuples where tuple[0]=detected spike and tuple[1]=isi to the next detected spike
    isi_alone_dic : keys = channels, values = list of isi alone in microseconds!
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
                isi = spikedic[key][i]*tick, (spikedic[key][i+1]-spikedic[key][i])*tick #CL tick für beide dazu
                isi_alone = (spikedic[key][i+1]-spikedic[key][i])*tick
                isilist.append(isi)
                isislist.append(isi_alone)
        isidic[key]=isilist
        isi_alone_dic[key]=isislist
        
    return isidic, isi_alone_dic



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


def get_burst_threshold(df_with_CMA, network_ISI):
    '''
    

    Parameters
    ----------
    df_with_CMA : TYPE
        DESCRIPTION.

    Returns
    -------
    CMAalpha : TYPE
        DESCRIPTION.
    CMAalpha2 : TYPE
        DESCRIPTION.
    maxCMA : TYPE
        DESCRIPTION.
    alpha1 : TYPE
        DESCRIPTION.
    alpha2 : TYPE
        DESCRIPTION.

    '''
    
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


def ISI_threshold_min(df, CMAalpha, CMAalpha2, binsize_in_micros):
    '''
    '''
    indexfactor = df[df['CMA']>CMAalpha].index[-1] + 1
    indexfactor2 = df[df['CMA']>CMAalpha2].index[-1] + 1
    threshold_intraburst = float(indexfactor*binsize_in_micros)
    threshold_burst_related = float(indexfactor2*binsize_in_micros)
    
    return threshold_intraburst, threshold_burst_related




def find_burst_starts_and_length(isi_alone, threshold_intraburst, spikedic, tick):
    '''
    Parameters
    ----------
    isi_alone : dict
        k = channellabel, values = interspike intervals in microseconds
    threshold_intraburst : float
        the calculated threshold for a single channel burst in microseconds
    spikedic : dict
        k = channellabel, values = spiketimes in ticks
        

    Returns
    -------
    burststart_end_dic : dict
        k = channellabel, values = tuple(a,b) with a = start of a burst x, b= end of burst x 
        with all times in microseconds

    '''
    burststartdic = {}
    noburstlist = []
    #burststartlist = []
    for key in isi_alone:
        #print(key)
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
                        burststart_spike = spikedic[key][counter]*tick
                        burstend_spike = spikedic[key][counter+setter]*tick
                        #burststartlist.append((spikedic[key][counter])*tick) #CL: zusätzlich times tick to get all timestamps in µs
                        setter += 1
                        while isi_alone[key][counter+setter]<threshold_intraburst and (counter+setter)< (len(isi_alone[key])-4):
                            setter +=1
                            burstend_spike = spikedic[key][counter+setter]*tick
                            #print('burst '+str(setter))
                        burststartlist.append((burststart_spike, burstend_spike))
                        setter +=1
                    else:
                        counter +=1
                else:
                    counter +=1
                counter = counter + setter + 1
            else:
                counter +=1
            #print(str(key) + str(counter))
        burststartdic[key]=burststartlist
        
    return burststartdic   



def extract_burststarts(burststartenddic):

    burststart_dic = {}
    burstlist = []
    
    for key in burststartenddic:
        burstlist = []
        start_ends = burststartenddic[key]
        for i in start_ends:
            burstlist.append(i[0])
        burststart_dic[key] = burstlist
        
    return burststart_dic



def subdivide_spiketrain(spiketrain, sub_start = 0, sub_stop = 10, tick=40, scale_factor_for_second=1e-06):
    '''
    Excpects: 
        a spiketrain with tick datapoints
        default ticks are 40
        default scale_factor_for_seconds = 1e-06
        provide the start and stop of the desired sub in seconds
    
    Does:
        converts the desired seconds into data ticks
        checks if the spikes of the given spiketrain is in the desired subs
        substracts the starting time -->
        
    Returns:
        a spiketrain dictionary that again starts from zero
    
    '''
    sub_start_tick = sub_start / (tick*scale_factor_for_second)
    sub_stop_tick = sub_stop / (tick*scale_factor_for_second)
    sub_spiketrain = {}
  
    for key in spiketrain: 
        list_per_key = []
        for i in spiketrain[key]:
            if (i>=sub_start_tick ) & (i<sub_stop_tick):
                list_per_key.append(int(i-sub_start_tick))
        sub_spiketrain[key]=list_per_key

    return sub_spiketrain



def get_interburst_intervals(burststart_end_dic):
    
    '''
    parameters:
    
    burststart_end_dic : dic
    keys = channellabels
    values = list of tuples tuple (a, b) with a = burststarts, b = burstends in µs
    
    
    ______________________
    
    returns:
    
    ibi_dic : dic
    keys = channellabels
    values = list of all interburstintervals in µs
    
    
    ______________________
    
    nota bene:
    
    interburst intervals are defined as non-bursting intervals between bursts.
    That means it is from burst-end1 to burststart2.
    
    '''
    
    
    ibi_dic = {}
    ibi_list = []
    
    for key in burststart_end_dic:
        ibi_list = []
        bursts = burststart_end_dic[key]
        for i in range(0, len(bursts)-1): # we leave the last burst out
            burst_end = bursts[i][1]
            next_start = bursts[i+1][0]
            
            interburst_interval = next_start - burst_end
            
            ibi_list.append(interburst_interval)
        
        ibi_dic[key] = ibi_list
        
    return ibi_dic


def invert_layerdic(layer_dic):
    
    '''
    Expects a dictionary with key = layer, value = list of channellabels
    
    Returns a dictionary with key = channellabels, value = layer
    '''
    layerdic_invert = {}

    for key in layer_dic:
        for i in layer_dic[key]:
            layerdic_invert[i]=key
            
            
    return layerdic_invert
            


def get_channel_infos(filedirectory, meafile):
    channel_raw_data = McsPy.McsData.RawData(os.path.join(filedirectory, 
                                                          meafile))
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
    print('In total '+ str(analognumber) 
          + " analog_streams were identified.\n")
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
        print("total recordingtime is: " 
              + str(timelengthrecording_s) + "seconds \n")




def get_MEA_Signal(analog_stream, channel_idx, from_in_s=0, to_in_s=None):
    '''
    Extracts one Channels (channel_idx) Sginal 
    
    :param analog_stream = the analogstream from one recording
    :param channel_idx   = the channel index of the channel where you 
                            extract the values from
    :param from_in_s     = starting point of the range you want to observe 
                            in seconds
    :param to_in_s       = ending point of the range you want to observe. 
                            Default is None (i.e. whole range)
    
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
        to_idx = min(
            analog_stream.channel_data.shape[1], 
            int(to_in_s * sampling_frequency)
            )

    # get the timestamps for each sample
    time = analog_stream.get_channel_sample_timestamps(
        channel_id, from_idx, to_idx
        )

    # scale time to seconds:
    scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second

    # get the signal
    signal = analog_stream.get_channel_in_range(channel_id, from_idx, to_idx)

    # scale signal to µV:
    scale_factor_for_uV = Q_(1,signal[1]).to(ureg.uV).magnitude
    signal_in_uV = signal[0] * scale_factor_for_uV
    
    return signal_in_uV, time_in_sec, sampling_frequency, scale_factor_for_second



def detect_threshold_crossings(signal, fs, threshold, dead_time):
    """
    Detect threshold crossings in a signal with dead time and return 
    them as an array

    The signal transitions from a sample above the threshold to a sample 
    below the threshold for a detection and
    the last detection has to be more than dead_time apart 
    from the current one.

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold: The threshold for the signal
    :param dead_time: The dead time in seconds.
    """
    dead_time_idx = dead_time * fs
    threshold_crossings = np.diff(
        (signal <= threshold).astype(int) > 0).nonzero()[0]
    distance_sufficient = np.insert(
        np.diff(threshold_crossings) >= dead_time_idx, 0, True
        )
    while not np.all(distance_sufficient):
        # repeatedly remove all threshold crossings that violate the dead_time
        threshold_crossings = threshold_crossings[distance_sufficient]
        distance_sufficient = np.insert(
            np.diff(threshold_crossings) >= dead_time_idx, 0, True
            )
    return threshold_crossings


def get_next_minimum(signal, index, max_samples_to_search):
    """
    Returns the index of the next minimum in the signal after an index

    :param signal: The signal as a 1-dimensional numpy array
    :param index: The scalar index
    :param max_samples_to_search: The number of samples to search for a 
                                    minimum after the index
    """
    search_end_idx = min(index + max_samples_to_search, signal.shape[0])
    min_idx = np.argmin(signal[index:search_end_idx])
    return index + min_idx


def align_to_minimum(signal, fs, threshold_crossings, search_range, first_time_stamp=0):
    """
    Returns the index of the next negative spike peak for all threshold crossings

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold_crossings: The array of indices where the signal 
                                crossed the detection threshold
    :param search_range: The maximum duration in seconds to search for the 
                         minimum after each crossing
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
    
    :param trigger_on =basically created through the find_triggers function 
                        and marks points were stimulation is turned on
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
            cutout = signal[int((index-pre_idx)):int((index+post_idx))]
            cutouts.append(cutout)
    if len(cutouts)>0:
        return np.stack(cutouts)
    
    
def plot_waveforms(cutouts, fs, pre, post, n=100, color='k', show=True):
    """
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

        
def butter_lowpass_filter(data, cutoff, fs, order, nyq):

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def get_isi_singlechannel(spikedic, tick):
    '''
    Parameters
    ----------
    spikedic : dictionary with all detected spikes for a channel
        DESCRIPTION.

    Returns
    -------
    isidic : keys = channels, values = List of tuples where tuple[0]=detected spike and tuple[1]=isi to the next detected spike
    isi_alone_dic : keys = channels, values = list of isi alone in microseconds!
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
                isi = spikedic[key][i]*tick, (spikedic[key][i+1]-spikedic[key][i])*tick #CL tick für beide dazu
                isi_alone = (spikedic[key][i+1]-spikedic[key][i])*tick
                isilist.append(isi)
                isislist.append(isi_alone)
        isidic[key]=isilist
        isi_alone_dic[key]=isislist
        
    return isidic, isi_alone_dic



def subdivide_spiketrain(spiketrain, sub_start = 0, sub_stop = 10, tick=40, scale_factor_for_second=1e-06):
    '''
    Excpects: 
        a spiketrain with tick datapoints
        default ticks are 40
        default scale_factor_for_seconds = 1e-06
        provide the start and stop of the desired sub in seconds
    
    Does:
        converts the desired seconds into data ticks
        checks if the spikes of the given spiketrain is in the desired subs
        substracts the starting time -->
        
    Returns:
        a spiketrain dictionary that again starts from zero
    
    '''
    sub_start_tick = sub_start / (tick*scale_factor_for_second)
    sub_stop_tick = sub_stop / (tick*scale_factor_for_second)
    sub_spiketrain = {}
  
    for key in spiketrain: 
        list_per_key = []
        for i in spiketrain[key]:
            if (i>=sub_start_tick ) & (i<sub_stop_tick):
                list_per_key.append(int(i-sub_start_tick))
        sub_spiketrain[key]=list_per_key

    return sub_spiketrain






def find_network_burst_components(network_bursts_seconds, 
                                  Bursts, spikedic_MAD, ups, 
                                  up_amplitudes, downs, 
                                  down_amplitudes, inverted_layerdic,
                                  tick):
    
    '''
    ______________________
    parameters
    
    network_bursts_seconds : list of tuples
        tuples are all filtered network bursts (i.e., the gaussian smoothed firing rate that
        crosses the mean of the smoothed firing rate)
        
        tuple(a, b) with a = burststart, b = burststop in seconds
        
    
    Bursts : dict
        key = channellabel
        value = list of tuples (a, b) with a = burststart, b = burststop in µseconds
        
    spikedic_MAD : dict
    
        key = channellabel
        value = list of spikes in ticks --> times tick and scale_factor_second_to_receive
            the spikes in seconds
            
    _______________________
    returns
        
    network_bursts_dictionary : dict
        key = tuple (a, b) with a = networkburststart, b = networkburststop in seconds
        
        value = tuple (a,b,c) with a=the number of single channel bursting channels,
                b = the number of active (i.e., spiking) channels, and c = array of all 
                single channel bursting channels
                
    relevant_relevant_channels : list
    
        list with all channels that are active at any network burst
        can be used to filter the original signal when extracting the LFP
    
    
    '''





    
    network_bursts_dictionary = {}
    # relevant channels is basically all channels that burst at any time in one list
    relevant_channels = []

    for i in network_bursts_seconds:
        
        print('Working on networkburst  ', i)
        network_features_dic = {}
        
        network_key = str(i)
        burst_list = []
        bursting_channels = []
        active_channels = []
        
        
        # get all channels that burst while the network burst is going on
        total_number_burst_starts = 0
        total_number_burst_ends = 0
        for key in Bursts:   
            for b in Bursts[key]:
                # if either start or end of single channel burst is within the network burst
                burst_start = b[0]*1e-06
                burst_stop = b[1]*1e-06
                   
                    
                # every burst that starts and every burst that stops
                # is counted into the bursting channels and for the
                # total number of bursts
               
                if i[0] <= burst_start <= i[1]:
                    bursting_channels.append(key)
                    relevant_channels.append(key)
                    total_number_burst_starts +=1
                    
                    
                if i[0] <= burst_stop <= i[1]:
                    bursting_channels.append(key)
                    relevant_channels.append(key)
                    total_number_burst_ends +=1
        
        # all channels that have a spike
        spikecount = 0
        for key in spikedic_MAD:
            for s in spikedic_MAD[key]:
                s = s*tick*1e-06
                if i[0] <= s <= i[1]:
                    spikecount += 1
                    active_channels.append(key)
                    
        
        # extract all channels that show a lfp up deviation here
        # with the index the amplitudes are retrieved
        # and added to the list to calculate the mean amplitude 
        lfp_up_list = []
        lfp_up_amplitudes = []
        for key in ups:
            for up in ups[key]:
                up_start = up[0]
                up_stop = up[1]
                up_index = ups[key].index(up)
                if (i[0] <= up_start <= i[1]) or (i[0] <= up_stop <= i[1]):
                    lfp_up_list.append(key)
                    amplitude = up_amplitudes[key][up_index]
                    lfp_up_amplitudes.append(amplitude)
        average_up = np.nanmean(lfp_up_amplitudes)            
        
        # extract all channels that show a lfp down deviation here
        # with the index the amplitudes are retrieved
        # and added to the list to calculate the mean amplitude 
        lfp_down_list = []                          
        lfp_down_amplitudes = []
        for key in downs:
            for down in downs[key]:
                down_start = down[0]
                down_stop = down[1]
                down_index = downs[key].index(down)
                if (i[0] <= down_start <= i[1]) or (i[0] <=down_stop <= i[1]):
                    lfp_down_list.append(key)
                    amplitude = down_amplitudes[key][down_index]
                    lfp_down_amplitudes.append(amplitude)
        average_down = np.nanmean(lfp_down_amplitudes)
        
        
        #active channels
        active_channels = np.unique(active_channels)
        network_features_dic['active_channels'] = active_channels
        
        networkburst_layerlist = []
        for c in active_channels:
            try:
                layer = inverted_layerdic[c]
                networkburst_layerlist.append(layer)
            except:
                print('Layer for channel {} missing.'.format(c))
        
        
        # time_length networkburst
        nb_start = i[0]
        nb_stop = i[1]
        timelength_networkburst = nb_stop - nb_start
        network_features_dic['timelength_network_burst_s'] = timelength_networkburst
        
        
        #dictionary with spikes for the now calculated networkburst per channel
        spikes_per_channel_networkburst = {}
        
        #dictionary with number of spikes for the now calculated networkburst per channel
        number_spikes_per_channel_networkburst = {}
        
        # dictionary for firing rate per channel per networkburst
        fr_per_channel_networkburst = {}
        
        
        # filter all spikes that occur in this networkburst
        for key in spikedic_MAD:
            spikelist = []
            for s in spikedic_MAD[key]:
                s = s*tick*1e-06
                if i[0] <= s <= i[1]:
                    spikelist.append(s)
            spikes_per_channel_networkburst[key] = spikelist
            number_spikes_per_channel_networkburst[key] = len(spikelist)
            fr_per_channel_networkburst[key]= len(spikelist)/timelength_networkburst #this is calculated but not used, probably no benefit

               
        
        # get the interspike_intervals for every networkburst per channel
        to_discard, isi_per_channel_networkburst = get_isi_singlechannel(
                                                spikes_per_channel_networkburst,
                                                tick = tick)
        
        
        # now the parameters above are calculated for the layers
        spikes_per_layer_networkburst = {}
        
        isi_mean_per_layer_networkburst = {}
        isi_std_per_layer_networkburst = {}
        
        for key in layerdic:
            tmp_layerlist_isi = []
            tmp_layerlist_spikes = []
            for c in layerdic[key]:
                #for spike in number_spikes_per_channel_networkburst:
                tmp_layerlist_spikes.append(number_spikes_per_channel_networkburst[c])
                try:
                    tmp_layerlist_isi.append(list(isi_per_channel_networkburst[c]))
                except:
                    pass
            # flatten the resulting lists of list
            tmp_layerlist_isi = [x for y in tmp_layerlist_isi for x in y]
        
            
            # add information to the corresponding dictionary
            spikes_per_layer_networkburst[key] = sum(tmp_layerlist_spikes)
            isi_mean_per_layer_networkburst[key] = np.mean(tmp_layerlist_isi)
            isi_std_per_layer_networkburst[key] = np.std(tmp_layerlist_isi)
                
        
        
        
        # add features to the dictionary
        # bursting channels
        bursting_channels = np.unique(bursting_channels)
        network_features_dic['bursting_channels'] = bursting_channels
        
        # number of bursting channels
        n_bursting_channels = len(bursting_channels)
        network_features_dic['number_of_bursting_channels'] = n_bursting_channels
        
        # number of bursting channels
        network_features_dic['number_burst_starts'] = total_number_burst_starts
        network_features_dic['number_burst_ends'] = total_number_burst_starts
        
        
        #number of active channels
        n_active_channels = len(active_channels)
        network_features_dic['number_of_active_channels'] = n_active_channels
        
        #total number of spikes 
        network_features_dic['number_of_spikes'] = spikecount
        
        # firing rate networkburst
        firing_rate_networkburst = spikecount/timelength_networkburst
        network_features_dic['firing_rate_Hz'] = firing_rate_networkburst
        
        #up lfps:
        network_features_dic['channels_lfp_up'] = np.unique(lfp_up_list)
        network_features_dic['number_channels_with_lfp_up'] = len(np.unique(lfp_up_list))
        network_features_dic['mean_up_lfp_amplitude'] = average_up
        
        #down lfps:
        network_features_dic['channels_lfp_down'] = np.unique(lfp_down_list)
        network_features_dic['number_channels_with_lfp_down'] = len(np.unique(lfp_down_list))
        network_features_dic['mean_down_lfp_amplitude'] = average_down
        
        #anatomy_registration active channels
        network_features_dic['n_layer1_channels'] = networkburst_layerlist.count("layer1")
        network_features_dic['n_layer2-3_channels'] = networkburst_layerlist.count("layer2-3")
        network_features_dic['n_layer4_channels'] = networkburst_layerlist.count("layer4")
        network_features_dic['n_layer5-6_channels'] = networkburst_layerlist.count("layer5-6")
        network_features_dic['n_whitematter_channels'] = networkburst_layerlist.count("whitematter")
        
        
        
        #anatomy registration spikes
        network_features_dic['n_spikes_layer1'] = spikes_per_layer_networkburst["layer1"]
        network_features_dic['n_spikes_layer23'] = spikes_per_layer_networkburst["layer2-3"]    
        network_features_dic['n_spikes_layer4'] = spikes_per_layer_networkburst["layer4"]
        network_features_dic['n_spikes_laye5-6'] = spikes_per_layer_networkburst["layer5-6"]
        network_features_dic['n_spikes_whitematter'] = spikes_per_layer_networkburst["whitematter"]
        
        # anatomy registration mean isi
        network_features_dic['isi_mean_layer1'] = isi_mean_per_layer_networkburst["layer1"]
        network_features_dic['isi_mean_layer23'] = isi_mean_per_layer_networkburst["layer2-3"]
        network_features_dic['isi_mean_layer4'] = isi_mean_per_layer_networkburst["layer4"]
        network_features_dic['isi_mean_layer56'] = isi_mean_per_layer_networkburst["layer5-6"]
        network_features_dic['isi_mean_whitematter'] = isi_mean_per_layer_networkburst["whitematter"]
        
        # anatomy registration std isi
        network_features_dic['isi_std_layer1'] = isi_std_per_layer_networkburst["layer1"]
        network_features_dic['isi_std_layer23'] = isi_std_per_layer_networkburst["layer2-3"]
        network_features_dic['isi_std_layer4'] = isi_std_per_layer_networkburst["layer4"]
        network_features_dic['isi_std_layer56'] = isi_std_per_layer_networkburst["layer5-6"]
        network_features_dic['isi_std_whitematter'] = isi_std_per_layer_networkburst["whitematter"]

      
        
        
        
        
        network_bursts_dictionary[network_key] = (network_features_dic)
    
    return network_bursts_dictionary, relevant_channels




def plot_networkburst_traces(network_bursts_dictionary, filebase):
    
    
    for burst in network_bursts_dictionary:
        plt.ioff()
        k = burst

        nb_start = float(k.split("(")[1].split(")")[0].split(',')[0])
        nb_stop = float(k.split("(")[1].split(")")[0].split(',')[1])

        limit0, limit1 = np.around(nb_start-0.5, decimals=2), np.around(nb_stop+0.5, decimals=2)


        # netowrkburst of interest
        nboi= network_bursts_dictionary[k]
        bursting_channels = nboi['bursting_channels']
        active_channels = nboi['active_channels']
        down_channels = list(network_bursts_dictionary[k]['channels_lfp_up'])
        up_channels = list(network_bursts_dictionary[k]['channels_lfp_up'])
        lfp_channels = up_channels + down_channels
        
   
        channels_of_interest = list(active_channels)

        channels_of_interest.sort()

        number_traces = len(channels_of_interest)

        fig = plt.figure(figsize=(10, number_traces))

        gs1 = gridspec.GridSpec(number_traces, 1)
        gs1.update(wspace=0.025, hspace=0.05) 
        axs = []
        
        
        # change to the subfolder containing the relevant bandpass and lowpass dic
        # relevant_subfolder = 
        # os.chdir()

        for i in range(1, number_traces+1):


            key = channels_of_interest[i-1]

            #no get all signals to plot and the lfp_down and lfp_ups
            bandpass_signal = bandpass_dic[key]
            # in the lowpass_dic there are still additional returns from the old butter filter function
            lowpass_signal = lowpass_dic[key]
            ups = lfp_ups[key]
            downs = lfp_downs[key]



            axs.append(fig.add_subplot(gs1[i-1]))


            axs[-1] = plt.plot(time_in_sec, bandpass_signal, c="#048ABF", linewidth = 0.5)
            axs[-1] = plt.plot(time_in_sec, lowpass_signal, c='#F20505', linewidth=1)
            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            #ax.spines['bottom'].set_visible(False)
            #ax.spines['left'].set_visible(False)
            #ax.get_xaxis().set_visible(False)
            frameon = False

            ax = plt.axvspan(nb_start, nb_stop, color='#C1D96C', alpha=0.1)

            for i in downs:
                ax = plt.axvspan(i[0], i[1], color='#5D7CA6', alpha=0.2)
            for i in ups:
                ax = plt.axvspan(i[0], i[1], color='#BF214B', alpha=0.2)



            # plt xlim for zooming in the time
            plt.xlim(limit0, limit1)
            plt.ylim(-70, 70)
            plt.yticks(fontsize='xx-small')
            plt.ylabel(key)

        fig.suptitle(filebase + ' network burst from '+str(limit0)+' to '+str(limit1))

        fig.savefig(
        filebase + '_lfp_and_bandpasssignal_cutout_from_' + str(limit0) +'_to_'+str(limit1)+'.png',
        bbox_inches='tight', dpi=300)
        plt.close()







def main():
    
    inputdirectory = input('Please enter the file directory: ')
    os.chdir(inputdirectory)
    filelist= glob.glob("*.h5")
    layerdictionary_list = glob.glob('*layerdic*')
    print(filelist)
    

    bool_location = 0
    while bool_location != ('A' or 'R'):
        bool_location = input('Enter A if this file is from Aachen and R if it is from Reutlingen: ')
        if bool_location != ('A' or 'R'):
            print('Please insert a valid input.')


    
    
    timestr = datetime.today().strftime('%Y-%m-%d')
    
    # to save memory:
    plt.ioff()
    
    # set filter cuts in Hz
    lowcut = 150
    highcut = 4500
    
    # Length of cutouts around shapes
    pre = 0.001 # 1 ms
    post= 0.002 # 2 ms
    
    # divide recording in n seconds long subrecordings
    dividing_seconds = 120
    
    
    #this creates one overview spikedic for all recordings
    record_overview_dic = {}
    master_filelist = []
    
    resting_spikedic={}
    spikedic={}
    spikedic_MAD={}
    artefactsdic_MAD={}
    cutouts_dic ={} 
    keylist = []
    
    lfp_ups = {}
    lfp_downs = {}
    lfp_amplitudes_up = {}
    lfp_amplitueds_down = {}
    
    cs_lfp_ups = {}
    cs_lfp_downs = {}
    cs_lfp_amplitudes_up = {}
    cs_lfp_amplitudes_down = {}
    
    lowpass_signal_dic = {}
    bandpass_signal_dic = {}
    convolved_lowpass_signal_dic = {}
    
    
    # inputdirectory = '/Users/jonas/Documents/DATA/MEA_DATA_Aachen_sample'
        
    '''

    1. MAIN SCRIPT FOR SPIKE EXTRACTION


    '''    
    # for-loop files
    for file in filelist:
        resting_spikedic={}
        spikedic={}
        cutouts_dic ={} 
        keylist = []
        filename = file
        filedatebase = filename.split('T')[0]
        filenamebase = filename.split('__')[1]
        #filebase = filename.split('.')[0]
        filebase = filedatebase + '_' + filenamebase
        
        
        if filebase not in master_filelist:
            master_filelist.append(filebase)

        #create the outputdirectory
        mainoutputdirectory = os.path.join(inputdirectory, 'output')
        outputdirectory = os.path.join(mainoutputdirectory, filebase)
        try:
            Path(outputdirectory).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        
        outputdirectory_spikeextraction = os.path.join(outputdirectory, 'time_series_correlation')
        try:
            Path(outputdirectory_spikeextraction).mkdir(parents=True, exist_ok=False)

        except FileExistsError:
            pass
        
        
        
        
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
        first_recording_timepoint = time[0][0]
        scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
        scale_factor_for_milisecond = scale_factor_for_second/1000
        time_in_sec = time[0]*scale_factor_for_second
        timelengthrecording_ms = time[0][-1]+tick
        timelengthrecording_s = (time[0][-1]+tick)*scale_factor_for_second
        fs = int(stream.channel_infos[channel_id].sampling_frequency.magnitude)
        
        analog_stream_0_data = analog_stream_0.channel_data
        np_analog_stream_0_data = np.transpose(
            channel_raw_data.recordings[0].analog_streams[0].channel_data
            )
        
        # the stream needs to be changed because MCS hates me
        np_analog_for_filter = np.transpose(np_analog_stream_0_data)
        #np_analog_stream_1_data = np.transpose(
            #channel_raw_data.recordings[0].analog_streams[0].channel_data
            #)
        #np_analog_stream_1_data_transpose = np.transpose(np_analog_stream_1_data)
        
        # delete these streams to save memory
        del np_analog_stream_0_data
        del analog_stream_0_data
        
        
        n = len(np_analog_for_filter)
        
        raw_signal_dic = {}

        # Initialize the adjacency matrix
        adjacency_matrix = np.zeros((n, n))
        
        for i in range (0, len(np_analog_for_filter)):
            
            channel_idx = i
            labellist = get_MEA_Channel_labels(np_analog_for_filter, analog_stream_0)
            signal_in_uV, time_in_sec, sampling_frequency, scale_factor_for_second = get_MEA_Signal(
                analog_stream_0, channel_idx)
            channellabel = labellist[i]
            raw_signal_dic[channellabel] = signal_in_uV


        
        # Iterate over each pair of electrodes and calculate the Pearson correlation coefficient
        for i, data_i in enumerate(raw_signal_dic.values()):
            for j, data_j in enumerate(raw_signal_dic.values()):
                if i != j:
                    corr_coeff, _ = pearsonr(data_i, data_j)
                    adjacency_matrix[i, j] = corr_coeff
                else:
                    adjacency_matrix[i, j] = 1  # the correlation of the signal with itself is 1
        
        # Print the adjacency matrix
        print(adjacency_matrix)
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        bandpass_signal_dic = {}
        raw_signal_dic = {}
                
                
                
                # second for loop for every channel:
        for i in range (0, len(np_analog_for_filter)):
        # for every channel we get the signal, filter it, define a threshold
        # see the crossings, align them to the next minimum (=spikes)
        # fill the dictionary with the tickpoints
        # and finally plot everything
            
            # for long file the cutout dics should not be saved to spare memory
            # for short files it is possible to keep the cutouts and save them
            
            
            '''
            
            Get raw  and bandpassfiltered signal
            
            '''
            cutouts_dic ={}
        
            channel_idx = i
            labellist = get_MEA_Channel_labels(np_analog_for_filter, analog_stream_0)
            signal_in_uV, time_in_sec, sampling_frequency, scale_factor_for_second = get_MEA_Signal(
                analog_stream_0, channel_idx)
            
            
            
            


# Get the number of electrodes
n = len(electrode_data)

# Initialize the adjacency matrix
adjacency_matrix = np.zeros((n, n))

# Iterate over each pair of electrodes and calculate the Pearson correlation coefficient
for i, data_i in enumerate(electrode_data.values()):
    for j, data_j in enumerate(electrode_data.values()):
        if i != j:
            corr_coeff, _ = pearsonr(data_i, data_j)
            adjacency_matrix[i, j] = corr_coeff
        else:
            adjacency_matrix[i, j] = 1  # the correlation of the signal with itself is 1

# Print the adjacency matrix
print(adjacency_matrix)

            
            '''
            bandpassfilteredsignal = butter_bandpass_filter(
                signal_in_uV, lowcut, highcut, sampling_frequency
                )
        
            # This Part is for finding MAD spikes + plotting
            noise_mad = np.median(np.absolute(bandpassfilteredsignal)) / 0.6745
            threshold = -5* noise_mad
            artefact_threshold = -8* noise_mad
          
            channellabel = labellist[i]

            bandpass_signal_dic[channellabel] = bandpassfilteredsignal
            raw_signal_dic[channellabel] = signal_in_uV
            #artefactsdic_MAD[channellabel] = artefacts
            '''
        
