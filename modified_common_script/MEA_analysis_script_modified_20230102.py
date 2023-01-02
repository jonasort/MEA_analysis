
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun 21 August 00:00:28 2022

@author: jonas ort, department of neurosurgery RWTH Aachen
"""
'''
Imports
'''
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
    
    bool_channelmap = 0
    while bool_channelmap != 1:
        string_input = input('Do you want to use a labeldictionary? Enter y or n: ')
        if string_input == 'y':
            bool_channelmap = 1
        elif string_input == 'n':
            bool_channelmap = 1
        else:
            print('Please insert a valid input.')
    bool_channelmap = string_input
            
            
            
    bool_location = 0
    while bool_location != ('A' or 'R'):
        bool_location = input('Enter A if this file is from Aachen and R if it is from Reutlingen: ')
        if bool_location != ('A' or 'R'):
            print('Please insert a valid input.')


    bool_modules = 0
    while bool_modules != ('b'):
        bool_modules = input('If you want the basic analysis (spikes only), enter b. If you want extended analysis (including lfp times), enter e: ')
        if bool_modules != ('b'):
            print('Pleas insert a valid input.')
    
    
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
        
        outputdirectory_spikeextraction = os.path.join(outputdirectory, 'spike_extraction')
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
        
        
        signal_cuts = []
        
        starting_point = 0
        stopping_point = 0
        while starting_point < timelengthrecording_s:
            if starting_point + dividing_seconds >= int(timelengthrecording_s):
                stopping_point = int(timelengthrecording_s)
            
            else:
                stopping_point =stopping_point + dividing_seconds
            signal_cuts.append((starting_point, stopping_point))
        
            # set the window one step further:
            starting_point = starting_point + dividing_seconds
        
        # unfortunately another for loop to get through the subrecordings
        
        for i in signal_cuts:
            
            
            
            starting_point = i[0]
            stopping_point = i[1]
        
            if stopping_point - starting_point > 10:    
            
                #timestr = time.strftime("%d%m%Y")
                outpath = Path(
                    outputdirectory_spikeextraction, filebase + '_from_'+str(starting_point) + 
                    '_to_' +str(stopping_point) + '_analyzed_on_'+timestr)
                try:
                    Path(outpath).mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    print ("Creation of the directory %s failed" % outpath)
                else:
                    print ("Successfully created the directory %s " % outpath)
                    
                os.chdir(outpath)
                
                
                
                # second for loop for every channel:
                for i in range (0, len(np_analog_for_filter)):
                # for every channel we get the signal, filter it, define a threshold
                # see the crossings, align them to the next minimum (=spikes)
                # fill the dictionary with the tickpoints
                # and finally plot everything
                    
                    # for long file the cutout dics should not be saved to spare memory
                    # for short files it is possible to keep the cutouts and save them
                    
                    
                    '''
                    
                    1.1 SPIKE DETECTION
                    
                    '''
                    cutouts_dic ={}
                
                    channel_idx = i
                    labellist = get_MEA_Channel_labels(np_analog_for_filter, analog_stream_0)
                    signal_in_uV, time_in_sec, sampling_frequency, scale_factor_for_second = get_MEA_Signal(
                        analog_stream_0, channel_idx, from_in_s=starting_point,
                        to_in_s=stopping_point
                        )
                    bandpassfilteredsignal = butter_bandpass_filter(
                        signal_in_uV, lowcut, highcut, sampling_frequency
                        )
                
                    # This Part is for finding MAD spikes + plotting
                    noise_mad = np.median(np.absolute(bandpassfilteredsignal)) / 0.6745
                    threshold = -5* noise_mad
                    artefact_threshold = -8* noise_mad
                    crossings = detect_threshold_crossings(
                        bandpassfilteredsignal, sampling_frequency, 
                        threshold, dead_time=0.001
                        )
                    spikes=align_to_minimum(
                        bandpassfilteredsignal, fs, crossings, search_range=0.003, 
                        first_time_stamp=first_recording_timepoint
                        )
                    
                    if len(spikes) < 10:
                        artefact_crossings = detect_threshold_crossings(
                            bandpassfilteredsignal, sampling_frequency, 
                            artefact_threshold, dead_time=0.001
                            )
                        artefacts = align_to_minimum(
                            bandpassfilteredsignal, fs, artefact_crossings, search_range=0.003, 
                            first_time_stamp=first_recording_timepoint
                            )
                    
                    
                    # this line accoutns for a starting point of the recording that is != 0
                    spikes = spikes + int(time_in_sec[0]/(scale_factor_for_second*tick)) 
                    channellabel = labellist[i]
                    spikedic_MAD[channellabel] = spikes
                    bandpass_signal_dic[channellabel] = bandpassfilteredsignal
                    #artefactsdic_MAD[channellabel] = artefacts
                    print('iteration ' + str(i) + ' channel: ' +str(channellabel))
                    
                  
                        
                      
                    
                    '''
                    
                    1.2. LFP_CROSSINGS
                    
                    '''
                    
                    if bool_modules == 'e':
                    
                    
                        T = timelengthrecording_s         # Sample Period
                        fs = fs      # sample rate, Hz
                        cutoff = 100  # desired cutoff frequency of the filter of lowpass
                        nyq = 0.5 * fs  # Nyquist Frequency
                        order = 2       # sin wave can be approx represented as quadratic
                        n = int(T * fs) # total number of samples
                        
                        butter_lowpass_filtered_signal = butter_lowpass_filter(signal_in_uV, 
                                                                               cutoff, fs, order, 
                                                                               nyq=nyq)
                        
                        
                        lfp_std = np.std(butter_lowpass_filtered_signal)
                        lfp_mean = np.mean(butter_lowpass_filtered_signal)
                        threshold_LFP = 3*lfp_std
                        
                        
                        down_cross, up_cross, amp_down, amp_up = lfp_crossing_detection(
                            butter_lowpass_filtered_signal, threshold_LFP, tick=tick, 
                            scale_factor_for_second=scale_factor_for_second, 
                            time_in_sec=time_in_sec, minimal_length=0.03)
                        
                        
                        '''
                        Modulation for the LFP Start
                        '''
                        convolved_signal = np.convolve(butter_lowpass_filtered_signal, np.ones(3000)/3000, mode='full')
                        
                        
                        length_cutter = len(butter_lowpass_filtered_signal)
                        
                        cs = convolved_signal[:length_cutter]
                        
                        cs_threshold = np.std(cs)*2
                        
                        cs_down_cross, cs_up_cross, cs_amp_down, cs_amp_up = lfp_crossing_detection(
                            convolved_signal, cs_threshold, tick=tick, 
                            scale_factor_for_second=scale_factor_for_second, 
                            time_in_sec=time_in_sec, minimal_length=0.01)
                        
                        
                        
                        
                        '''
                        Modulation for the LFP Stop
                        
                        '''
                        
                        
                        
                        
                        lfp_ups[channellabel] = up_cross
                        lfp_downs[channellabel] = down_cross
                        lfp_amplitudes_up[channellabel] = amp_up
                        lfp_amplitueds_down[channellabel] = amp_down
                        
                        cs_lfp_ups[channellabel] = cs_up_cross
                        cs_lfp_downs[channellabel] = cs_down_cross
                        cs_lfp_amplitudes_up[channellabel] = cs_amp_up
                        cs_lfp_amplitudes_down[channellabel] = cs_amp_down
            
                        
                        
                        
                        
                        lowpass_signal_dic[channellabel] = butter_lowpass_filtered_signal
                        convolved_lowpass_signal_dic[channellabel] = convolved_signal
            
            
                        
                        '''
                        PLOTTING OF SPIKES + LFP
                        
                        '''
                        
                        # if there are detected spikes get the waveforms, plot the channel and waveforms and save
                        
                        try:
                        
                        
                        
                            if len(spikes) > 3:
                                
                                #only extract cutouts when they are relevant
                                cutouts = extract_waveforms(
                                        bandpassfilteredsignal, sampling_frequency, spikes, 
                                        pre, post
                                        )
                                cutouts_dic[channellabel] = cutouts
                                
                                
                                plt.style.use("seaborn-white")
                                
                                
                                # figure 1: signal with threshold
                                fig1, ax = plt.subplots(1, 1, figsize=(20, 10))
                                ax = plt.plot(time_in_sec, bandpassfilteredsignal, c="#45858C")
                                ax = plt.plot([time_in_sec[0], time_in_sec[-1]], [threshold, threshold], c="#297373")
                                ax = plt.plot(spikes*tick*scale_factor_for_second, [threshold-1]*(spikes*tick*scale_factor_for_second).shape[0], 'ro', ms=2, c="#D9580D")
                                ax = plt.title('Channel %s' %channellabel)
                                ax = plt.xlabel('Time in Sec, Threshold: %s' %threshold)
                                ax = plt.ylabel('µ volt')
                                ax = plt.plot(time_in_sec, butter_lowpass_filtered_signal, c='#F29829', linewidth=0.5)
                                for i in down_cross:
                                    ax = plt.axvspan(i[0], i[1], color='#5D7CA6', alpha=0.2)
                                for i in up_cross:
                                    ax = plt.axvspan(i[0], i[1], color='#BF214B', alpha=0.2)
                                fig_1_name = filebase+'_signal_'+channellabel+'MAD_THRESHOLD.png'
                                if not os.path.exists(outpath):
                                    os.mkdir(outpath) 
                                fullfig_1_name = Path(outpath, fig_1_name)
                                fig1.savefig(fullfig_1_name)
                                plt.close(fig1) 
                                plt.clf()
                                                     
                                
                                
                                #figure 2: waveforms 
                                fig2, ax2 = plt.subplots(1, 1, figsize=(12,6))
                                #ax2 is a plot of the waveform cutouts
                                n = 100
                                n = min(n, cutouts.shape[0])
                                time_in_us = np.arange(-pre*1000, post*1000, 1e3/fs)
                                cutout_mean = np.mean(cutouts, axis=0)
                                for i in range(n):
                                    ax2 = plt.plot(time_in_us, cutouts[i,]*1e6, color='black', linewidth=1, alpha=0.3)
                                    ax2 = plt.plot(time_in_us, cutout_mean*1e6, color="red", linewidth=1, alpha=0.3)
                                    ax2 = plt.xlabel('Time (%s)' % ureg.ms)
                                    ax2 = plt.ylabel('Voltage (%s)' % ureg.uV)
                                    ax2 = plt.title('Cutouts of Channel %s' %channellabel)
                                    
                                fig_2_name = filebase+'_waveforms_'+channellabel+'MAD_THRESHOLD.png'
                                if not os.path.exists(outpath):
                                    os.mkdir(outpath) 
                                fullfig_2_name = Path(outpath, fig_2_name)
                                fig2.savefig(fullfig_2_name)
                                plt.close(fig2) 
                                plt.clf()
                                    
                                    
                            
                            
                            else:
                                
                                plt.style.use("seaborn-white")
                                
                                # withouth spikes, so we always have an overview file
                                # figure 1: signal with threshold
                                fig1, ax = plt.subplots(1, 1, figsize=(20, 10))
                                ax = plt.plot(time_in_sec, bandpassfilteredsignal, c="#45858C")
                                ax = plt.plot([time_in_sec[0], time_in_sec[-1]], [threshold, threshold], c="#297373")
                                #ax = plt.plot(spikes*tick*scale_factor_for_second, [threshold-1]*(spikes*tick*scale_factor_for_second).shape[0], 'ro', ms=2, c="#D9580D")
                                ax = plt.title('Channel %s' %channellabel)
                                ax = plt.xlabel('Time in Sec, Threshold: %s' %threshold)
                                ax = plt.ylabel('µ volt')
                                ax = plt.plot(time_in_sec, butter_lowpass_filtered_signal, c='#F29829', linewidth=0.5)
                                for i in down_cross:
                                    ax = plt.axvspan(i[0], i[1], color='#5D7CA6', alpha=0.2)
                                for i in up_cross:
                                    ax = plt.axvspan(i[0], i[1], color='#BF214B', alpha=0.2)
                                fig_1_name = filebase+'_signal_'+channellabel+'MAD_THRESHOLD.png'
                                if not os.path.exists(outpath):
                                    os.mkdir(outpath) 
                                fullfig_1_name = Path(outpath, fig_1_name)
                                fig1.savefig(fullfig_1_name)
                                plt.close(fig1) 
                                plt.clf()
                            
                            # cutoutdics are deleted to spare memory
                            del cutouts_dic
                            
                            
                            
                            if len(up_cross)>=1: 
                                
                                counter = 0
                                #figure 3: zoom in on LFPs up
                                
                                for i in up_cross:
                                    
                                    counter += 1
                                        
                                    start = i[0]
                                    stop = i[1]
                                    
                                    # frist get the relevant range of time to plot
                                    plotting_time = [tis for tis in time_in_sec if start <= tis <= stop]
                                    
                                    #extract the index from it
                                    start_ind = list(time_in_sec).index(plotting_time[0])
                                    stop_ind = list(time_in_sec).index(plotting_time[-1])
                                    
                                    
                                    diff = stop_ind - start_ind
                                    plotstart_ind = int(max(time_in_sec[0], start_ind - diff))
                                    plotstop_ind = int((min(stop_ind + diff, len(time_in_sec)-1)))
                                    
                                    # the indexes are converted to time in seconds
                                    time_start = time_in_sec[plotstart_ind]
                                    time_stop = time_in_sec[plotstop_ind]
                                    
                                    
                                    plotting_spikes = spikes*tick*scale_factor_for_second
                                    
                                    
                                    plotting_spikes = [spike for spike in plotting_spikes if time_start < spike < time_stop ]
                                    plotting_time = [tis for tis in time_in_sec if time_start <= tis < time_stop ]
                                    #plotting_bandpass =  [j for j in list(bandpassfilteredsignal) if time_start <= j  < time_stop]
                                    #plotting_lowpass = [k for k in list(butter_lowpass_filtered_signal) if time_start <= k  < time_stop]
                                    
                                    
                                    fig3, ax = plt.subplots(1, 1, figsize=(10, 5))
                                    ax = plt.plot(plotting_time, bandpassfilteredsignal[plotstart_ind:plotstop_ind], c='#45858C', linewidth=0.5)
                                    ax = plt.plot(plotting_time, butter_lowpass_filtered_signal[plotstart_ind:plotstop_ind], c='#F29829', linewidth=0.5)
                                    ax = plt.plot([plotting_time[0], plotting_time[-1]], [threshold_LFP, threshold_LFP], c="#A6036D", lw=1)
                                    ax = plt.plot([plotting_time[0], plotting_time[-1]], [-threshold_LFP, -threshold_LFP], c="#A6036D", lw=1)
                                    ax = plt.plot(plotting_spikes, [threshold-1]*np.asarray(plotting_spikes).shape[0], 'ro', ms=2, c="#D9580D")
                                    ax = plt.axvspan(i[0], i[1], color='#BF214B', alpha=0.2)
                                    
                                    
                                    print(time_in_sec[plotstart_ind])
                                    print(time_in_sec[plotstop_ind])
                                    ax = plt.title('Channel %s' %channellabel)
                                    ax = plt.xlabel('Time in Sec')
                                    ax = plt.ylabel('µ volt')
                            
                                    save_start_str = str(start).replace('.', 'p')
                                    save_stop_str = str(stop).replace('.', 'p')
                                    
                                    fig_3_name = filebase+'_lfp_Up_number_'+str(counter)+'_'+channellabel+'_.png'
                                    if not os.path.exists(outpath):
                                        os.mkdir(outpath)
                                    
                                    fullfig_3_name = Path(outpath, fig_3_name)
                                    fig3.savefig(fullfig_3_name)
                                    plt.close(fig3) 
                                    plt.clf()
                                    
                                 
                                        
                                        
                                    
                            if len(down_cross)>=1:     
                                #figure4: zoom in LFPs down
                                
                                counter = 0 
                                
                                for i in down_cross:
                                    
                                    counter += 1
                                
                                    start = i[0]
                                    stop = i[1]
                                    
                                    # frist get the relevant range of time to plot
                                    plotting_time = [tis for tis in time_in_sec if start <= tis <= stop]
                                    
                                    #extract the index from it
                                    start_ind = list(time_in_sec).index(plotting_time[0])
                                    stop_ind = list(time_in_sec).index(plotting_time[-1])
                                    
                                    
                                    diff = stop_ind - start_ind
                                    plotstart_ind = int(max(time_in_sec[0], start_ind - diff))
                                    plotstop_ind = int((min(stop_ind + diff, len(time_in_sec)-1)))
                                    
                                    # the indexes are converted to time in seconds
                                    time_start = time_in_sec[plotstart_ind]
                                    time_stop = time_in_sec[plotstop_ind]
                                    
                                    
                                    plotting_spikes = spikes*tick*scale_factor_for_second
                                    
                                    
                                    plotting_spikes = [spike for spike in plotting_spikes if time_start < spike < time_stop ]
                                    plotting_time = [tis for tis in time_in_sec if time_start <= tis < time_stop ]
                                    #plotting_bandpass =  [j for j in list(bandpassfilteredsignal) if time_start <= j  < time_stop]
                                    #plotting_lowpass = [k for k in list(butter_lowpass_filtered_signal) if time_start <= k  < time_stop]
                                    
                                    
                                    fig4, ax = plt.subplots(1, 1, figsize=(10, 5))
                                    ax = plt.plot(plotting_time, bandpassfilteredsignal[plotstart_ind:plotstop_ind], c='#45858C', linewidth=0.5)
                                    ax = plt.plot(plotting_time, butter_lowpass_filtered_signal[plotstart_ind:plotstop_ind], c='#F29829', linewidth=0.5)
                                    ax = plt.plot([plotting_time[0], plotting_time[-1]], [threshold_LFP, threshold_LFP], c="#A6036D", lw=1)
                                    ax = plt.plot([plotting_time[0], plotting_time[-1]], [-threshold_LFP, -threshold_LFP], c="#A6036D", lw=1)
                                    ax = plt.plot(plotting_spikes, [threshold-1]*np.asarray(plotting_spikes).shape[0], 'ro', ms=2, c="#D9580D")
                                   
                                    ax = plt.axvspan(i[0], i[1], color='#5D7CA6', alpha=0.2)
                                    
                                
                                    ax = plt.title('Channel %s - ' %channellabel)
                                    ax = plt.xlabel('Time in Sec')
                                    ax = plt.ylabel('µ volt')
                                    
                                    save_start_str = str(start).replace('.', 'p')
                                    save_stop_str = str(stop).replace('.', 'p')
                        
                                    
                                    fig_4_name = filebase+'_lfp_Down_number_'+str(counter)+'_'+channellabel+'_.png'
                                    
                                    
                                    if not os.path.exists(outpath):
                                        os.mkdir(outpath)
                                    
                                    fullfig_4_name = Path(outpath, fig_4_name)
                                    fig4.savefig(fullfig_4_name)
                                    plt.close(fig4) 
                                    plt.clf()
                                    
                                    
                                    
                                    
                                    
                            
                        except:
                            
                            print('During plotting of LFP an error occured.')
                            pass
                        
                
                            
                        
                # end of the channel for-loop, from here on
                # plot of rasterplot for channels with > 0.5 htz 
                # create an array of the spikes in scale of seconds
            
                spikedic_seconds = {}
                for key in spikedic_MAD:
                    relevant_factor = timelengthrecording_s*0.05
                    if len(spikedic_MAD[key])>relevant_factor:
                        sec_array = spikedic_MAD[key]*tick*scale_factor_for_second
                        spikedic_seconds[key]=sec_array
                spikearray_seconds = np.asarray(list(spikedic_seconds.values()))
                
                '''
                
                1.3. NETWORK ACTIVITY
                
                
                
                maybe delete this whole part, since it is not needed and only gives
                out the plot for the subrecording
                
                '''
            
                # get a 1-D array with every detected spike
                scale_factor_for_milisecond = 1e-03
                full_spike_list = []
                full_spike_list_seconds = []
                for key in spikedic_MAD:
                    if len(spikedic_MAD[key])>relevant_factor:
                        x = list(np.asarray(spikedic_MAD[key])*scale_factor_for_milisecond*tick)
                        full_spike_list = full_spike_list + x
                
                        xs = list(np.asarray(spikedic_MAD[key])*scale_factor_for_second*tick)
                        full_spike_list_seconds = full_spike_list_seconds + xs
                full_spikes = sorted(full_spike_list)
                full_spikes_seconds = sorted(full_spike_list_seconds)
            
            
                #define bins 
                binsize = 0.005 #seconds
                bins= np.arange(0, timelengthrecording_s+binsize, binsize)
                
                # make a histogram 
                full_spikes_binned = np.histogram(full_spikes_seconds, bins)[0]
                
               
                #trial of population burst plot as inspired by Andrea Corna
                bins = int(timelengthrecording_s / binsize)+1
                
                firing_rate_histogram = np.histogram(full_spikes_seconds, bins=bins)
                firing_rate = firing_rate_histogram[0]*200 #conversion to hertz
                
                
                def gaussian_smoothing(y, window_size=10, sigma=2):
                    
                    filt = signal.gaussian(window_size, sigma)
                    
                    return signal.convolve(y, filt, mode='same')
                
                N = int(1.5/binsize) # für eine Secunde, das Sliding window, also letztlich number of bins
                
                # gaussian smmothing fo the firing rate and moving average
                fr_gau = gaussian_smoothing(firing_rate)
                ma_fr_gau = np.convolve(fr_gau, np.ones(N)/N, mode='same')
                #plt.plot(ma_fr_gau)
                
                
                
                
                # we look for the mean of the MA as threshold
                # we arrange this mean in an array for plotting
                mean_ma_fr_gau = np.mean(ma_fr_gau)
                network_burst_threshold = mean_ma_fr_gau
                
                
                # to compare the amount of network bursts over different recordings we want the threshold
                # to be form the baseline file
                #baselinefile = filelist[1]
                
                #baseline_file_threshold = np.load(baselinefile+'_info_dict.npy',
                 #                                 allow_pickle = True).item()['network_burst_threshold']
                
                
                # Cave: funktioniert so nur, wenn das File Komplett durchläuft
                #if file == filelist[0]:
                 #   network_burst_threshold = mean_ma_fr_gau
                #else:
                 #   network_burst_threshold = baseline_file_threshold
            
                
                shape_for_threshold = np.shape(ma_fr_gau)
                network_burst_threshold_array = np.full(shape_for_threshold, network_burst_threshold)
                
            
                # now we identify the burts from the network and will extract an array with 
                # tuples containing the burst start and end times
                bursts= []
                burst_start = []
                burst_seconds_start = []
                burst_end = []
                burst_seconds_end = []
                for index in range(0, len(ma_fr_gau[:-N])):
                    if ma_fr_gau[index+N] > network_burst_threshold:
                        if ma_fr_gau[index+N-1] <= network_burst_threshold:
                            burst_start.append(index+N)
                        if index == 0:
                            burst_start.append(0)
                            #burst_seconds_start.append((index+N)*0.005)
                    else:
                        if (ma_fr_gau[index+N-1] > network_burst_threshold) and (len(burst_start)>0):
                            if index+N > len(ma_fr_gau):
                                ending = len(ma_fr_gau)
                            else: 
                                ending = index + N
                                
                            burst_end.append(ending)
                            #burst_seconds_end.append((ending)*0.005)
                bursts = list(zip(burst_start, burst_end))
                
                
                
                # now we need to reconvert the bins towards seconds:
                    
                for i in burst_start:
                    burst_seconds_start.append(firing_rate_histogram[1][i])
                for i in burst_end:
                    burst_seconds_end.append(firing_rate_histogram[1][i])
               
                bursts_seconds = list(zip(burst_seconds_start, burst_seconds_end))
                # bursts sind jetzt im 5ms bin 
                
                
                
                # plot for firing rate and identified bursting activity
                
                fig = plt.figure(figsize = (20,8))
                gs = fig.add_gridspec(2, hspace = 0, height_ratios=[1,5])
                axs = gs.subplots(sharex=False, sharey=False)
                axs[0].plot(ma_fr_gau, color= 'black')
                axs[0].set_ylabel('Firing Rate [Hz]')
                axs[1].eventplot(spikearray_seconds, color = 'black', linewidths = 0.2,
                                 linelengths = 0.5, colors = 'black')
                axs[1].set_ylabel('Relevant Channels')
                
                for ax in axs:
                    for i in bursts_seconds:
                        axs[1].axvspan(i[0], i[1], facecolor = '#5B89A6', alpha = 0.3)
                fig.savefig(filebase + 'raster_firingrate_plot.png', dpi=300)
                plt.close(fig)
                
                
                '''
                
                continue after possible deleted part
                ^^^^
                '''
                
                # lastly we save important information of the recording into a dictionary
                # this way, we can easily access them for further analysis
                
                info_dic = {}
                info_dic['tick']=tick
                info_dic['timelengthrecording_s']=timelengthrecording_s
                info_dic['timelengthrecording_ms']=timelengthrecording_ms
                info_dic['first_recording_timepoint']=first_recording_timepoint
                info_dic['scale_factor_for_second']=scale_factor_for_second
                info_dic['time_in_sec']=time_in_sec
                info_dic['sampling_frequency']=fs
                firing_rate_recording = len(full_spikes_seconds) / timelengthrecording_s
                info_dic['full_recording_fr'] = firing_rate_recording
                
                
                if file == filelist[0]:
                    info_dic['network_burst_threshold_basline']=network_burst_threshold
                
                
                # remove for reutlingen data
                if bool_location == 'A':
                    filename = filebase
                
    
                np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_spikes_MAD_dict.npy', spikedic_MAD) 
                np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_info_dict.npy', info_dic)
                
                if bool_modules == 'e':
                    np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_LFP_UPS.npy', lfp_ups)
                    np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_LFP_DOWNS.npy', lfp_downs)
                    np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_LFP_Amplitudes_UPS.npy', lfp_amplitudes_up)
                    np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_LFP_Amplitudes_DOWNS.npy', lfp_amplitueds_down)
                    np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_lowpass_signal.npy', lowpass_signal_dic) 
                    np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_bandpass_signal.npy', bandpass_signal_dic) 
                    
                
                    np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_cs_lfp_ups.npy', cs_lfp_ups) 
                    np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_cs_lfp_downs.npy', cs_lfp_downs) 
                    np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_cs_lfp_amplitueds_up.npy', cs_lfp_amplitudes_up)
                    np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_cs_lfp_amplitueds_down.npy', cs_lfp_amplitudes_down)
            
                
                # recordoverview dic filling to paste into excel later on
            
                
                
                os.chdir(inputdirectory)
        
        
        
    
    
    
    
    '''

    SCRIPT 2 Like

    '''
    
    
    os.chdir(mainoutputdirectory)
    outputfolderlist = glob.glob('*')
    
    #from here we will loop to each main outputfolder
    for mainfolder in outputfolderlist:
        os.chdir(os.path.join(mainoutputdirectory, mainfolder))
        working_directory = os.path.join(mainoutputdirectory, mainfolder, 'spike_extraction')
        filename = mainfolder.split('/')[-1]
        
        
        
        masterdictionary = {}
        
        
                
         
        # now a data structure is created where we can store all necessary information
        # i.e., it is a dicionary of dictionaries that will be pickled
        
        Basics = {}
        Infos_Recording = {}
        Infos_Analysis = {}
        Infos_Anatomy = {}
        main_recording_dictionary ={}
        
        Infos_Recording['filename']=filename
        
        slice_id = filename.split('_')[4]
        Infos_Recording['slice_id']= slice_id
        
        medium = filename.split('_')[3]
        Infos_Recording['medium']=medium
        
        drug = filename.split('_')[5]
        Infos_Recording['drug']=drug

        stimulation = filename.split('_')[6]
        Infos_Recording['stimulation']=stimulation
        
        tissue = filename.split('_')[1]
        Infos_Recording['tissue']=tissue
        
        recording_date_info = filename.split('_')[0]
        Infos_Recording['recording_date']=recording_date_info
        
        
        # the folderlist will contain all 120second long subfolders
        # the filename is 
        os.chdir(working_directory)
        folderlist = glob.glob(filename+'*')
        
        
        
        
        # get into every folder and find the dictionaries
        # replace them in a two meta-dictionaries (infodics and spikedics)
        infodics = {}
        spikedics = {}
        
        for folder in folderlist:
            os.chdir(os.path.join(working_directory, folder))
            # cave: here the slicing needs to be adjusted dependent on reutlingen filenames
            if bool_location == 'A':
                timekey = folder.split('_')[9:12]
            else:
                timekey = folder.split('_')[6:9]
            timekey = '_'.join(timekey)
            
            # load the info_dic_file
            info_dic_filename = glob.glob('*info*npy')
            print(info_dic_filename)
            print(os.getcwd())
            info_dic = np.load(info_dic_filename[0], allow_pickle=True).item()
            infodics[timekey] = info_dic
            
            # load the spikedic_file
            spike_dic_filename = glob.glob('*spikes_MAD*')[0]
            spikedic_MAD = np.load(spike_dic_filename, allow_pickle=True).item()
            spikedics[timekey] = spikedic_MAD
        
        
        # separately save all infodics
        np.save(os.path.join(mainoutputdirectory, mainfolder, 'infodics_'+filename+'.npy'), infodics)
        
        # get the first of all infodics
        first_info_dic_key = list(infodics.keys())[0]
        infodic = infodics[first_info_dic_key]
        
        '''
        ADD the info_dics to our pickle data
        '''
        
        Infos_Recording['info_dics_subrecordings'] = infodics
        Infos_Recording['recordings_date'] = recording_date_info
        Infos_Recording['timelengthrecording_s'] = infodic['timelengthrecording_s']
        
        
        # the parameter infodic is available through our loop
        # it contains the information of the last inofdic we loaded
        tick = infodic['tick']
        first_recording_timepoint = infodic['first_recording_timepoint']
        scale_factor_for_second = infodic['scale_factor_for_second']
        timelengthrecording_s = infodic['timelengthrecording_s']
        
        # we attach them in the first level of the Infos_Recording to 
        # have faster access to it
        Infos_Recording['scale_factor_for_second'] = scale_factor_for_second
        Infos_Recording['tick'] = tick
        
        
        
        
        
        '''
        2.1 
        JOIN subdivided spikedics to the full spikedic
        
        nb: the spike dics contain all spikes in the original tick data points
        the are continuing meaning that for a spikedic starting at 600 seconds of the
        recordings, the start is not zero but form 600 already. thus, they can simply be 
        concatenated.
        '''
        
        timekeys = list(spikedics.keys())
        channelkeys = list(spikedics[timekeys[0]].keys())
        
        
        # we now need to use a double loop to get all dictionary keys and join them into a big full recording dictionary
        spikedic_MAD_full = {}
        temp_spikelist = []
        
        for i in channelkeys:
            temp_spikelist = []
            for j in timekeys:
                spikes = list(spikedics[j][i])
                temp_spikelist.append(spikes)
            
            #join the lists
            temp_spikelista = sum(temp_spikelist, [])
            #remove the duplicates
            temp_spikelistb = list(set(temp_spikelista))
            
            #sort the list
            temp_spikelistc = sorted(temp_spikelistb)
            
            #assign them to their channel in the full dictionary
            spikedic_MAD_full[i] = temp_spikelistc
        
        
        # join the spikedic to the main_recording dictionary
        spikedic_MAD = spikedic_MAD_full
        main_recording_dictionary['spikedic_MAD'] = spikedic_MAD
        
        # and save it separately
        np.save(os.path.join(mainoutputdirectory, mainfolder, filename +'_full_spikedic.npy'),           
                spikedic_MAD_full)
        
        
        # relevant factor: minimal amount of spikes to be relevant
        # create an array of the spikes in scale of seconds
        active_channels = 0
        spikedic_seconds = {}
        for key in spikedic_MAD:
            relevant_factor = timelengthrecording_s*0.05
            if len(spikedic_MAD[key])>relevant_factor:
                sec_array = np.asarray(spikedic_MAD[key])*tick*scale_factor_for_second
                spikedic_seconds[key]=sec_array
                active_channels += 1
        spikearray_seconds = np.asarray(list(spikedic_seconds.values()))  
        
        # add them to the sub dictionaries
        Basics['active_channels'] = active_channels
        Basics['relevant_factor'] = relevant_factor
        
        
        # get a 1-D array with every detected spike
        scale_factor_for_milisecond = 1e-03
        full_spike_list = []
        full_spike_list_seconds = []
        for key in spikedic_MAD:
            if len(spikedic_MAD[key])>relevant_factor:
                x = list(np.asarray(spikedic_MAD[key])*scale_factor_for_milisecond*tick)
                full_spike_list = full_spike_list + x
        
                xs = list(np.asarray(spikedic_MAD[key])*scale_factor_for_second*tick)
                full_spike_list_seconds = full_spike_list_seconds + xs
        full_spikes = sorted(full_spike_list)
        full_spikes_seconds = sorted(full_spike_list_seconds)
        
        
        # calculate the mean firing rate for the whole recording
        mean_fr_whole_recording = np.around(
            (len(full_spikes_seconds) / timelengthrecording_s), 3)
        
        # add them to the sub dictionaries
        Basics['mean_fr_whole_recording'] = mean_fr_whole_recording
        
        
        
        
        '''
        2.2
        NETWORK BURSTING ACTIVITY
        
        For the whole concatenated recording, we now define the networkburst
        using the mean firing rate.
        
        How?
        We use gaussian smoothing, and a moving average over that smoothing.
        From this moving average we calculate the mean.
        
        A network burst is defined as the mean of that moving average + one standard
        deviation.
        
        '''
        
        
        # define bins 
        binsize = 0.005 #seconds
        bins= np.arange(0, timelengthrecording_s+binsize, binsize)
        
        # make a histogram 
        full_spikes_binned = np.histogram(full_spikes_seconds, bins)[0]
        
        
        #trial of population burst plot as inspired by Andrea Corna
        bins = int(timelengthrecording_s / binsize)+1
        
        #conversion to hertz
        firing_rate_histogram = np.histogram(full_spikes_seconds, bins=bins)
        firing_rate = firing_rate_histogram[0]*200 
        
        
        
        # sliding window of the moving average
        N = int(1/binsize) 
        
        # gaussian smmothing fo the firing rate and moving average
        fr_gau = gaussian_smoothing(firing_rate)
        
        
        ma_fr_gau = np.convolve(fr_gau, np.ones(N)/N, mode='full')
        
        # we look for the mean of the MA as threshold
        # we arrange this mean in an array for plotting
        mean_ma_fr_gau = np.mean(ma_fr_gau)
        std_ma_fr_gau = np.std(ma_fr_gau)
        network_burst_threshold = mean_ma_fr_gau #+ 1*std_ma_fr_gau
        shape_for_threshold = np.shape(ma_fr_gau)
        network_burst_threshold_array = np.full(shape_for_threshold, network_burst_threshold)
        
        # extraction of the network bursting activity
        # now we identify the burts from the network and will extract an array with 
        # tuples containing the burst start and end times
        bursts= []
        burst_start = []
        burst_seconds_start = []
        burst_end = []
        burst_seconds_end = []
        
        
        
        # filtering the actual network bursts in 5 ms bins
        bursts= []
        burst_start = []
        burst_seconds_start = []
        burst_end = []
        burst_seconds_end = []
        for index in range(0, len(ma_fr_gau[:-N])):
            if ma_fr_gau[index+N] > network_burst_threshold:
                if ma_fr_gau[index+N-1] <= network_burst_threshold:
                    burst_start.append(index)
                if index == 0:
                    burst_start.append(0)
                    #burst_seconds_start.append((index+N)*0.005)
            else:
                if (ma_fr_gau[index+N-1] > network_burst_threshold) and (len(burst_start)>0):
                    if index+N > len(ma_fr_gau):
                        ending = len(ma_fr_gau)
                    else: 
                        ending = index + N
        
                    burst_end.append(ending)
                    #burst_seconds_end.append((ending)*0.005)
        bursts = list(zip(burst_start, burst_end))
        
        
        
        # now we need to reconvert the bins towards seconds:
        for i in burst_start:
            burst_seconds_start.append(firing_rate_histogram[1][i])
        for i in burst_end:
            if i >= len(firing_rate_histogram[1]):
                burst_seconds_end.append(firing_rate_histogram[1][-1])
            else:
                burst_seconds_end.append(firing_rate_histogram[1][i])
        
        bursts_seconds = list(zip(burst_seconds_start, burst_seconds_end))
        # bursts sind jetzt im 5ms bin   
        
        # since we reference the bursts back to the seconds and those have different lengths
        # we need to correct for bursts that are overlapping
        bursts_seconds_corrected = []
        for i in range(0, len(bursts_seconds)-1):
            
            first_b = bursts_seconds[i]
            old_first_start = first_b[0]
            old_first_end = first_b[1]
            
            second_b = bursts_seconds[i+1]
            old_second_start = second_b[0]
            old_second_end = second_b[1]
            
            if old_second_start < old_first_end:
                new_first_stop = old_second_start - 0.1 # we substract one msecond
            
                first_b = (old_first_start, new_first_stop)
            
            bursts_seconds_corrected.append(first_b)
            
        bursts_seconds = bursts_seconds_corrected
        
        
        # add the network bursts to the main_recording_dictionary
        main_recording_dictionary['network_bursts_seconds'] = bursts_seconds
        
        
        # we plot the final rasterplot + firing rate for the whole recording
        # for sanity checking
        fig = plt.figure(figsize = (12,6))
        gs = fig.add_gridspec(2, hspace = 0, height_ratios=[1,5])
        axs = gs.subplots(sharex=False, sharey=False)
        axs[0].plot(ma_fr_gau, color= 'black', linewidth = 0.2)
        axs[0].set_ylabel('Firing Rate [Hz]')
        axs[1].eventplot(spikearray_seconds, color = 'black', linewidths = 0.3,
                         linelengths = 1, colors = 'black')
        axs[1].set_ylabel('Relevant Channels')
        
        for ax in axs:
            for i in bursts_seconds:
                axs[1].axvspan(i[0], i[1], facecolor = '#5B89A6', alpha = 0.3)
        fig.savefig(os.path.join(mainoutputdirectory, mainfolder, 
                                 filename+ '__raster_firingrate_plot.png'), dpi=300)
        
        
        
        # now we calculate the individual firing rates per channel
        whole_recording_firingrate_dic = {}
        
        # i.e, number of spikes divided by duration -> results in number per second
        for key in spikedic_MAD:
            fr_channel = len(spikedic_MAD[key])/timelengthrecording_s 
            whole_recording_firingrate_dic[key] = fr_channel
        
        # add it to the main dictionary
        main_recording_dictionary['fr_dic'] = whole_recording_firingrate_dic
        
        
        '''
        2.3 
        basic spiking statistics to the recording
        
        '''
        # create the dictionary with isi + add it
        isi_dictionary = get_isi_single_channel(spikedic_MAD_full, tick=tick,
                                                scale_factor_for_milisecond=scale_factor_for_milisecond)
        main_recording_dictionary['isi_dictionary'] = isi_dictionary
        
        
        # get the average isi and std
        # creating list to easily calculate the whole mean and std
        isi_averages = []
        isi_standarddeviations = []
        
        # creat dictionaries to do the same for every channel
        isi_average_dic = {}
        isi_standarddeviations_dic = {}
        
        
        for key in isi_dictionary:
            if len(isi_dictionary[key]) > relevant_factor:
                
                # for the relevant channels we attain the mean
                mean_isi = np.mean(isi_dictionary[key])
                isi_averages.append(mean_isi)
                
                # and the standard deviation
                std_isi = np.std(isi_dictionary[key])
                isi_standarddeviations.append(std_isi)
                
                isi_average_dic[key] = mean_isi
                isi_standarddeviations_dic[key] = std_isi
                
                
                
        mean_isi_relevant_channels = np.mean(isi_averages)
        mean_isi_std = np.mean(isi_standarddeviations)
        
        main_recording_dictionary['isi_average_dic'] = isi_average_dic
        main_recording_dictionary['isi_std_dic'] = isi_standarddeviations_dic
        
        Basics = {}
        
        Basics['active_channels'] = active_channels
        Basics['relevant_factor'] = relevant_factor
        Basics['mean_fr_whole_recording'] = mean_fr_whole_recording
        
        
        
        # find spikes that appear random (not in network activity)
        a, b, c, d = find_random_spikes(spikedic_MAD, bursts_seconds_corrected,
                                        tick=tick, 
                                        scale_factor_for_second=scale_factor_for_second)
        
        
        # dictionary with key = channel,
        # value = tuple with two list cotaining the random and nrandom spikes
        random_nrandom_spike_per_channeldic = a
        
        # dictionary with key=channel,
        # value = tuple with int as number of spikes random and nrandom
        number_rand_nrandom_spike_per_channeldic = b
        
        total_non_random_spikes = c
        total_random_spikes = d
        
        Basics['number_random_spikes'] = total_random_spikes
        Basics['number_notrandom_spikes'] = total_non_random_spikes
        main_recording_dictionary['number_rand_notrand_spikes_per_channel'] = number_rand_nrandom_spike_per_channeldic
        main_recording_dictionary['rand_notrand_spikes_per_channel'] = random_nrandom_spike_per_channeldic

        
        # add missing information to the main recording dic
        Infos_Analysis['relevant_factor'] = relevant_factor
        main_recording_dictionary['Infos_Recording'] = Infos_Recording
        main_recording_dictionary['Infos_Analysis'] = Infos_Analysis
        main_recording_dictionary['Infos_Anatomy'] = Infos_Anatomy
        main_recording_dictionary['Basics'] = Basics
        
        # and finally pickle the main_recording_dictionary
        with open(os.path.join(mainoutputdirectory, 
                               mainfolder+'/MAIN_RECORDING_Dictionary_'+filename+'.pkl'), 'wb') as f:
                  pickle.dump(main_recording_dictionary, f)
            
    '''
    Works Until Here, in essence end of Script 2
    
    '''  
    
    
    
    
    '''
    
    Script 3
    '''
    # goes through each of the created folders
    # creates the filename
    for mainfolder in outputfolderlist:
        os.chdir(os.path.join(mainoutputdirectory, mainfolder))
        filename = mainfolder.split('/')[-1]
        filename_substring = '_'.join(filename.split('_')[1:])       
        
        MAIN_RECORDING_DICTIONARY = pickle.load(
            open(os.path.join(mainoutputdirectory, 
                              mainfolder, 'MAIN_RECORDING_Dictionary_'+filename+'.pkl'), 
                 "rb"))

        
        
        MAIN_RECORDING_DICTIONARY['Infos_Recording'].keys()
        tick = MAIN_RECORDING_DICTIONARY['Infos_Recording']['tick']
        timelengthrecording_s = MAIN_RECORDING_DICTIONARY['Infos_Recording']['timelengthrecording_s']
        scale_factor_for_second = MAIN_RECORDING_DICTIONARY['Infos_Recording']['scale_factor_for_second']
        scale_factor_for_milisecond = scale_factor_for_second/1000
        spikedic_MAD = MAIN_RECORDING_DICTIONARY['spikedic_MAD']

        # get the isi distribution
        st_channel = spikedic_MAD
        binsize_for_ISI = 5000 #in microseconds

        isidic, isi_alone = get_isi_singlechannel(st_channel, tick) #creates two dictionaries
        histo_ISI_dic=bin_isi(isi_alone, binsize=binsize_for_ISI, 
                              binmax=True, binmaxnumber=200) # dictionary für jeden channel mit 300x 10ms bins (binsize) und der Menge an ISI für die jeweilige Länge
        network_ISI=get_allchannel_ISI_bins(histo_ISI_dic) #gibt ein array mit bins entsprechend der bins aus der Vorfunktion

        colors = ['green', 'blue', 'orange', 'purple']
        df= pd.DataFrame({'ISI_per_10ms_bins':network_ISI}) #aus Network_ISI wird ein pdDF um die weiteren Schritte durchführen zu können
        df["CMA"] = df.ISI_per_10ms_bins.expanding().mean()
        df[['ISI_per_10ms_bins', 'CMA']].plot(color=colors, linewidth=3, 
                                              figsize=(10,4), 
                                              title="Histogram of ISI-bins 10ms whole network")


        # calculate the adaptive threshold
        CMAalpha, CMAalpha2, maxCMA, alpha1, alpha2=get_burst_threshold(df, network_ISI=network_ISI) # threshold calculation
        
        
        # if maxCMA is =0, the spikedic is empty
        # this if statement prevents the script from failing, since 
        # the ISI_threshold_min does not work for an empty script
        if maxCMA > 0:
            threshold_intraburst, threshold_burst_related = ISI_threshold_min(df, CMAalpha, 
                                                                              CMAalpha2, binsize_for_ISI) #set thresholds
    
            print('intraburst: ', threshold_intraburst, '   related: ', threshold_burst_related)
            
        else:
            threshold_burst_related, threshold_intraburst = 140000, 140000


        # final threshold is calculated from the burst related within our defined limits
        final_threshold = 0
        if threshold_burst_related > 140000:
            final_threshold = 140000
        elif threshold_burst_related < 60000:
            final_threshold = 60000
        else:
            final_threshold = threshold_burst_related
        print('The final threshold for this recoding is: {}'.format(final_threshold))


        # add to main recording dic
        # add the final threshold to the 
        Infos_Analysis = MAIN_RECORDING_DICTIONARY['Infos_Analysis']
        Infos_Analysis['isi_burst_threshold_base'] = final_threshold

        MAIN_RECORDING_DICTIONARY['Infos_Analysis'] = Infos_Analysis



        # calculate the burststarts
        burststart_end_dic = find_burst_starts_and_length(isi_alone, final_threshold, st_channel,
                                                          tick=tick) 

        # add them to the Main dictionary
        MAIN_RECORDING_DICTIONARY['Bursts'] = burststart_end_dic



        # extract all burststarts for the spade analysis + save it 
        burststart_dic = extract_burststarts(burststart_end_dic)
        np.save(filename+'_burst_starts_dictionary.npy', burststart_dic)



         # create an array of the spikes in scale of seconds
        active_channels = 0
        spikedic_seconds = {}
        for key in burststart_dic:
            relevant_factor = timelengthrecording_s*0.05
            if len(burststart_dic[key])>relevant_factor:
                sec_array = np.asarray(burststart_dic[key])*tick*scale_factor_for_second
                spikedic_seconds[key]=sec_array
                active_channels += 1
        spikearray_seconds = np.asarray(list(spikedic_seconds.values())) 


        # calculate and save inter burst intervals and save them to main recording dic
        burst_ibi_dic = get_interburst_intervals(burststart_end_dic)
        MAIN_RECORDING_DICTIONARY['Interburst-Intervals'] = burst_ibi_dic



        # for every unit, the whole time of bursts is calculated and put into a dictionary
        bursting_time_per_unit_dic = {}
        for key in burststart_end_dic:
            time = 0
            for i in burststart_end_dic[key]:
                bursttime = i[1] - i[0]
                time = time + bursttime
            bursting_time_per_unit_dic[key] = time*scale_factor_for_second # kein tick, vorher bereits drin



        # for every unit, the whole time of bursts is calculated and put into a dictionary
        bursts_per_unit_dic = {}
        for key in burststart_end_dic:
            number_of_bursts = 0
            for i in burststart_end_dic[key]:
                number_of_bursts += 1
            bursts_per_unit_dic[key] = number_of_bursts

        # save both
        MAIN_RECORDING_DICTIONARY['bursting_time_per_channel'] = bursting_time_per_unit_dic
        MAIN_RECORDING_DICTIONARY['bursts_per_channel'] = bursts_per_unit_dic
        
        
        '''
        Load in Layerdic
        '''
        save_cwd = os.getcwd()
        
        os.chdir(inputdirectory)
        # find the right layerdic from the layerdic 
        # matches = [match for match in layerdictionary_list if '*'+filename+'*' in match]
        
        
        if bool_channelmap == 'y':
            
            layerdic_filename = glob.glob('*'+'layerdic'+'*'+filename_substring+'*.txt')[0]
        
            
            with open(layerdic_filename) as f:
                data = f.read()
          
            # reconstructing the data as a dictionary
            layerdic = ast.literal_eval(data)
            
            # change back to the last directory
            os.chdir(save_cwd)

    
    
        '''
        Burst Connections
        '''
        # this function takes the dictionary with every burst start and stop and returns a dictionary
        # where every unit is a key and the values are tuples consisting of keys of connected units (i.e., bursting together)
        # and the number of shared bursts
        burst_connections_dic = {}


        for key in burststart_end_dic:
            other_keys = list(burststart_end_dic.keys())
            other_keys.remove(key)
            connected_unit_list = []
            for j in other_keys:
                number_of_bursts = 0
                time_shared = 0
                for i in burststart_end_dic[key]:
                    start, end = i[0], i[1]
                    for m in burststart_end_dic[j]:
                        other_start = m[0]
                        other_end = m[1]
                        if (other_start > start) & (other_start < end):
                            if other_end >= end:
                                time_shared = time_shared + (end - other_start)
                            else:
                                time_shared = time_shared + (other_end - other_start)
                            number_of_bursts += 1
                            
                if number_of_bursts > 0:
                    connected_unit_list.append((j, number_of_bursts, time_shared*scale_factor_for_second))        
            burst_connections_dic[key] = connected_unit_list


        # we now calculate the burst connections with at least 0.1Hz
        simple_burst_connection = {}

        for key in burst_connections_dic:
            listed_connections = []
            for i in burst_connections_dic[key]:
                if i[1] > timelengthrecording_s*0.1: # Länge des Recordings * mindestens 0.1Hz -> alle zehn Sekunden
                #if i[1] > int(mean_fr_whole_recording*0.1):
                
               #if i[2] > 3: # Länge der gesharedten bursts in sec
                #if (i[1] > 10) & (i[2] > 1): # Länge der gesharedten bursts in sec
                    listed_connections.append(i[0])
            simple_burst_connection[key] = listed_connections


        MAIN_RECORDING_DICTIONARY['burst_connections'] = simple_burst_connection
    
        
    
        '''
        Get all possible MEA Channels depending on the bool_loc
        '''
        
        if bool_location == 'A':
            all_channels = ['D1', 'E1', 'F1', 'G1', 'H1', 'J1', 'J2', 'K1', 'K2', 'L1', 'L2', 'L3', 'M1', 'M2', 
                                'M3', 'M4', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 
                                'O7', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'R2', 'R3', 'R4', 'R5', 
                                'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'B1', 'B2', 'C1', 'C2', 'D2', 'E2', 'F2', 'G2', 'G3', 'H2', 'H3', 'J3', 'K3', 'K4', 
                                 'L4', 'L5', 'M5', 'M6', 'M7', 'N7', 'N8', 'O8', 'O9', 'O10', 'O11', 'P10', 'P11', 
                                 'P12', 'P13', 'P14', 'P15', 'P16', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'B3', 'B4', 'B5', 'B6', 
                                 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 
                                  'C11', 'C12', 'C13', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 
                                 'D13', 'D14', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 
                                 'E15', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 
                                 'F16', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 
                                 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'J4', 
                                 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'K5', 'K6', 
                                 'K7', 'K8', 'K9', 'K10', 'K11', 'K12', 'K13', 'K14', 'K15', 'K16', 'L6', 'L7', 'L8', 'L9', 
                                 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 
                                 'M14', 'M15', 'M16', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'O12', 'O13', 
                                 'O14', 'O15', 'O16', 'A12', 'A13', 'A14', 'A15', 'B13', 'B14', 'B15', 'B16', 'C14', 'C15', 'C16', 'D15', 'D16', 'E16']
        
        if bool_location == 'R':
            all_channels = ['D1', 'E1', 'F1', 'G1', 'H1', 'I1', 'I2', 'K1', 'K2', 'L1', 'L2', 'L3', 'M1', 'M2', 
                                    'M3', 'M4', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 
                                    'O7', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'R2', 'R3', 'R4', 'R5', 
                                    'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'B1', 'B2', 'C1', 'C2', 'D2', 'E2', 'F2', 'G2', 'G3', 'H2', 'H3', 'I3', 'K3', 'K4', 
                                     'L4', 'L5', 'M5', 'M6', 'M7', 'N7', 'N8', 'O8', 'O9', 'O10', 'O11', 'P10', 'P11', 
                                     'P12', 'P13', 'P14', 'P15', 'P16', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'B3', 'B4', 'B5', 'B6', 
                                     'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 
                                      'C11', 'C12', 'C13', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 
                                     'D13', 'D14', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 
                                     'E15', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 
                                     'F16', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 
                                     'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'I4', 
                                     'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'K5', 'K6', 
                                     'K7', 'K8', 'K9', 'K10', 'K11', 'K12', 'K13', 'K14', 'K15', 'K16', 'L6', 'L7', 'L8', 'L9', 
                                     'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 
                                     'M14', 'M15', 'M16', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'O12', 'O13', 
                                     'O14', 'O15', 'O16', 'A12', 'A13', 'A14', 'A15', 'B13', 'B14', 'B15', 'B16', 'C14', 'C15', 'C16', 'D15', 'D16', 'E16']
        
        if bool_channelmap == 'y':
        
            layer_colors={'layer1':'#F28749', 
                          'layer2-3':'#B5D932', 
                          'layer4':'#8C3A67', 
                          'layer5-6':'#3F93A6', 
                          'whitematter':'#C9F2EE',
                          'noslice': '#FFFFFF'}
            
            inverted_layerdic = invert_layerdic(layerdic)
    
            Infos_Anatomy = {}
            Infos_Anatomy['layerdic_invert']=inverted_layerdic
            Infos_Anatomy['layerdic']=layerdic
    
            MAIN_RECORDING_DICTIONARY['Infos_Anatomy'] = Infos_Anatomy
    
            # this block creates coordinates as on a MEA Grid for each channel
            # coordinates are between 0 and 1 via np.linspace
    
            # CAVE: I needs to be exchanged for J for Aachen Data, or i.e. the not MC Rack obtained data
            if bool_location == 'A':
                columnlist =['A','B','C','D','E','F','G','H','J','K','L','M','N','O','P','R']
            if bool_location == 'R':
                columnlist =['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','R']
    
            mea_coordinates = np.linspace(0,1,16)
            mea_positional_coordinates_dic = {}
    
            for i in all_channels:
                x_num = columnlist.index(i[0])
                x_coord = mea_coordinates[x_num]
                y_num = 17-int(i[1:]) # minus 1 since python starts counting at zero
                y_coord = 1-(mea_coordinates[-y_num])
                mea_positional_coordinates_dic[i] = [x_coord, y_coord]
    
    
            #normalize the bursting time per unit 
            normalized_bursting_time_per_unit_dic = {}
            time_list = list(bursting_time_per_unit_dic.values())
            maximum = max(time_list)
            minimum = min(time_list)
    
            #### Redo For Loop muss nach außen
            try:
                
                for key in bursting_time_per_unit_dic:
                    value = bursting_time_per_unit_dic[key]
                    normalized = (value - minimum)/(maximum-minimum)
                    normalized_bursting_time_per_unit_dic[key] = normalized
            except ZeroDivisionError:
                print('ZeroDivisionError - ')
                
    
            '''
            The Burst Graph is produced
            '''
    
            burst_conn_graph = nx.Graph()
            for key in simple_burst_connection:
                for i in simple_burst_connection[key]:
                    burst_conn_graph.add_edge(key, i)
    
            burst_conn_graph.number_of_nodes(), burst_conn_graph.number_of_edges()
    
            G = burst_conn_graph
    
    
            for i in G.nodes():
                
                try:
                    node_key = i
                    coordinate = mea_positional_coordinates_dic[node_key]
                    G.nodes[node_key]['pos']=coordinate
                    G.nodes[node_key]['layer']=inverted_layerdic[i]
                    G.nodes[node_key]['color']=layer_colors[inverted_layerdic[i]]
                    
                    try:
                        G.nodes[node_key]['degree_centrality']=nx.degree_centrality(G)[i]
                    except:
                        print('degree centrality failed')
                        
                    try:
                        G.nodes[node_key]['betweenness_centrality']=nx.betweenness_centrality(G, k=10, endpoints = True)[i]
                    except:
                        print('betweennes centrality failed')
                        
                    try:
                        G.nodes[node_key]['bursting_time_normalized']=normalized_bursting_time_per_unit_dic[i]
                    except:
                        print('normalized bursting time not possible')
                except KeyError:
                    print('channel ', node_key, ' failed')
    
    
            pos = nx.get_node_attributes(G, 'pos')
            layer = nx.get_node_attributes(G, 'layer')
            color = nx.get_node_attributes(G, 'color')
            burst_time = nx.get_node_attributes(G, 'bursting_time_normalized')
            try:
                centrality = nx.betweenness_centrality(G, k=10, endpoints = True)
            except:
                print('Degree Centrality Exception encountered')
            
            
            
            MAIN_RECORDING_DICTIONARY['GRAPH_shared_bursts'] = G
    
    
        with open(os.path.join(mainoutputdirectory, mainfolder, 'MAIN_RECORDING_Dictionary_'+filename+'.pkl'), 'wb') as f:
            pickle.dump(MAIN_RECORDING_DICTIONARY, f)
        

    '''
    SCRIPT 4
    
    - for every recording we get all the basis information
    - we create a dataframe with all the basics
    
    
    
    '''
    df_list = []
    MAINDATAFRAME = pd.DataFrame()
        
    for mainfolder in outputfolderlist:
        os.chdir(os.path.join(mainoutputdirectory, mainfolder))
        filename = mainfolder.split('/')[-1]
        filename_substring = '_'.join(filename.split('_')[1:])       
        
        MAIN_RECORDING_DICTIONARY = pickle.load(
            open(os.path.join(mainoutputdirectory, 
                              mainfolder, 'MAIN_RECORDING_Dictionary_'+filename+'.pkl'), 
                 "rb"))

        
        
        MAIN_RECORDING_DICTIONARY['Infos_Recording'].keys()
        tick = MAIN_RECORDING_DICTIONARY['Infos_Recording']['tick']
        timelengthrecording_s = MAIN_RECORDING_DICTIONARY['Infos_Recording']['timelengthrecording_s']
        scale_factor_for_second = MAIN_RECORDING_DICTIONARY['Infos_Recording']['scale_factor_for_second']
        scale_factor_for_milisecond = scale_factor_for_second/1000
        spikedic_MAD = MAIN_RECORDING_DICTIONARY['spikedic_MAD']


        network_bursts_seconds = MAIN_RECORDING_DICTIONARY['network_bursts_seconds']
        spikedic_MAD = MAIN_RECORDING_DICTIONARY['spikedic_MAD']
        fr_dic = MAIN_RECORDING_DICTIONARY['fr_dic']
        isi_dictionary = MAIN_RECORDING_DICTIONARY['isi_dictionary']
        isi_average_dic = MAIN_RECORDING_DICTIONARY['isi_average_dic']
        Infos_Recording = MAIN_RECORDING_DICTIONARY['Infos_Recording']
        Infos_Analysis = MAIN_RECORDING_DICTIONARY['Infos_Analysis']
        if bool_channelmap == 'y':
            Infos_Anatomy = MAIN_RECORDING_DICTIONARY['Infos_Anatomy']
        Bursts = MAIN_RECORDING_DICTIONARY['Bursts']
        Interburst_Intervals = MAIN_RECORDING_DICTIONARY['Interburst-Intervals']
        bursting_time_per_channel = MAIN_RECORDING_DICTIONARY['bursting_time_per_channel']
        bursts_per_channel = MAIN_RECORDING_DICTIONARY['bursts_per_channel']
        
        if bool_channelmap == 'y':
            burst_connections = MAIN_RECORDING_DICTIONARY['burst_connections']
        
            inverted_layerdic = Infos_Anatomy['layerdic_invert']
            layerdic = Infos_Anatomy['layerdic']
        Basics = MAIN_RECORDING_DICTIONARY['Basics']
        scale_factor_for_second = 1e-06
        timelengthrecording_s = Infos_Recording['timelengthrecording_s']
        
        infos_dic_subrecordings = Infos_Recording['info_dics_subrecordings']
        throwkey = list(infos_dic_subrecordings.keys())[0]
        subrec_infos = infos_dic_subrecordings[throwkey]
        
        
        # make dataframe
        # line per recording
        # alle basicstats
        # aufgelöst per layer
        '''
        CREATE MAIN DATAFRAME DICTIONARY
        '''
        
        MDF_dic = {}
        
        MDF_dic['filename'] = filename
        MDF_dic['slice_id'] = Infos_Recording['slice_id']
        MDF_dic['recording_date'] = Infos_Recording['recording_date']
        MDF_dic['tissue'] = Infos_Recording['tissue']
        MDF_dic['medium'] = Infos_Recording['medium']
        MDF_dic['drug'] = Infos_Recording['drug']
        MDF_dic['stimulation'] = Infos_Recording['stimulation']
        MDF_dic['sampling_frequency'] = subrec_infos['sampling_frequency']
        MDF_dic['timelength_recording_seconds'] = timelengthrecording_s
        MDF_dic['active_channels'] = Basics['active_channels']
        MDF_dic['relevant_factor'] = Basics['relevant_factor']
        MDF_dic['mean_fr_whole_recording'] = Basics['mean_fr_whole_recording']
        MDF_dic['number_random_spikes'] = Basics['number_random_spikes']
        MDF_dic['number_notrandom_spikes'] = Basics['number_notrandom_spikes']
        
        # calculate number of single channel bursts
        number_single_channel_bursts =  0
        for key in Bursts:
            num = len(Bursts[key])
            number_single_channel_bursts += num
        MDF_dic['number_single_channel_bursts'] = number_single_channel_bursts
        

        # claculate mean length of single channel bursts
        single_channel_bursts_length_list =[]
        for key in Bursts:
            for b in Bursts[key]:
                b_length = b[1]-b[0]
                b_length_ms = b_length * scale_factor_for_milisecond
                single_channel_bursts_length_list.append(b_length_ms)
                
        mean_single_channel_burst_length_ms = np.mean(single_channel_bursts_length_list)
        MDF_dic['mean_single_channel_burst_length_ms'] = mean_single_channel_burst_length_ms
        
        std_single_channel_burst_length_ms = np.std(single_channel_bursts_length_list)
        MDF_dic['std_single_channel_burst_length_ms'] = std_single_channel_burst_length_ms

        
        # calculate number of network bursts
        MDF_dic['number_network_bursts'] = len(network_bursts_seconds)

        
        # calculate mean length of network bursts
        networkbursts_length_list_s = []
        for i in network_bursts_seconds:
            nb_length_s = i[1]-i[0]
            networkbursts_length_list_s.append(nb_length_s)
        
        MDF_dic['mean_networkbursts_length_s'] = np.mean(networkbursts_length_list_s)

        
        # calculate std of length of network burts
        MDF_dic['std_networkbursts_length_s'] = np.std(networkbursts_length_list_s)

        
        
        
        if bool_channelmap == 'y':
            
            
            # calculate number of electrodes per layer
            inverted_layers_list = list(inverted_layerdic.values())
            MDF_dic['number_channels_covered_layer1'] = inverted_layers_list.count('layer1')
            MDF_dic['number_channels_covered_layer2-3'] = inverted_layers_list.count('layer2-3')
            MDF_dic['number_channels_covered_layer4'] = inverted_layers_list.count('layer4')
            MDF_dic['number_channels_covered_layer5-6'] = inverted_layers_list.count('layer5-6')
            MDF_dic['number_channels_covered_whitematter'] = inverted_layers_list.count('whitematter')
            MDF_dic['number_channels_covered_noslice'] = inverted_layers_list.count('noslice')

             
            # calculate fraction of electrodes per layer
            MDF_dic['fraction_channels_covered_layer1'] = inverted_layers_list.count('layer1')/252
            MDF_dic['fraction_channels_covered_layer2-3'] = inverted_layers_list.count('layer2-3')/252
            MDF_dic['fraction_channels_covered_layer4'] = inverted_layers_list.count('layer4')/252
            MDF_dic['fraction_channels_covered_layer5-6'] = inverted_layers_list.count('layer5-6')/252
            MDF_dic['fraction_channels_covered_whitematter'] = inverted_layers_list.count('whitematter')/252
            MDF_dic['fraction_channels_covered_noslice'] = inverted_layers_list.count('noslice')/252
            
            
            # calculate spikes per layer
            all_layers = list(layerdic.keys())
            spikes_per_layer_dic = {}
            for l in all_layers:
                spike_number_layer = 0
                for c in layerdic[l]:
                    spike_number_layer += len(spikedic_MAD[c])
                spikes_per_layer_dic[l] = spike_number_layer
           
                    
            MDF_dic['number_of_spikes_layer1'] = spikes_per_layer_dic['layer1']
            MDF_dic['number_of_spikes_layer2-3'] = spikes_per_layer_dic['layer2-3']
            MDF_dic['number_of_spikes_layer4'] = spikes_per_layer_dic['layer4']
            MDF_dic['number_of_spikes_layer5-6'] = spikes_per_layer_dic['layer5-6']
            MDF_dic['number_of_spikes_whitematter'] = spikes_per_layer_dic['whitematter']
            MDF_dic['number_of_spikes_noslice'] = spikes_per_layer_dic['noslice']
            
            
            # firing rate per layer in hertz
            MDF_dic['fr_recording_layer1'] = spikes_per_layer_dic['layer1']/timelengthrecording_s
            MDF_dic['fr_recording_layer2-3'] = spikes_per_layer_dic['layer2-3']/timelengthrecording_s
            MDF_dic['fr_recording_layer4'] = spikes_per_layer_dic['layer4']/timelengthrecording_s
            MDF_dic['fr_recording_layer5-6'] = spikes_per_layer_dic['layer5-6']/timelengthrecording_s
            MDF_dic['fr_recording_whitematter'] = spikes_per_layer_dic['whitematter']/timelengthrecording_s
            MDF_dic['fr_recording_noslice'] = spikes_per_layer_dic['noslice']/timelengthrecording_s
                        
            # calculate number of single channel bursts per layer
            bursts_per_layer_dic = {}
            for l in all_layers:
                burst_number_layer = 0
                for c in layerdic[l]:
                    burst_number_layer += bursts_per_channel[c]
                bursts_per_layer_dic[l] = burst_number_layer
                
            MDF_dic['number_of_singlechannelbursts_layer1'] = bursts_per_layer_dic['layer1']
            MDF_dic['number_of_singlechannelbursts_layer2-3'] = bursts_per_layer_dic['layer2-3']
            MDF_dic['number_of_singlechannelbursts_layer4'] = bursts_per_layer_dic['layer4']
            MDF_dic['number_of_singlechannelbursts_layer5-6'] = bursts_per_layer_dic['layer5-6']
            MDF_dic['number_of_singlechannelbursts_whitematter'] = bursts_per_layer_dic['whitematter']
            MDF_dic['number_of_singlechannelbursts_noslice'] = bursts_per_layer_dic['noslice']
               
            
            # calculate single channel mean duration of a burst time per layer
            
            mean_burstlength_s_per_layer_dic = {}
            for l in all_layers:
                bursttime_seconds_layer = []
                for c in layerdic[l]:
                    bursttime_seconds_layer.append(bursting_time_per_channel[c])
                mean_burstlength_s_per_layer_dic[l] = np.mean(bursttime_seconds_layer)
            
            MDF_dic['mean_length_singlechannelburst_layer1'] = mean_burstlength_s_per_layer_dic['layer1']
            MDF_dic['mean_length_singlechannelburst_layer2-3'] = mean_burstlength_s_per_layer_dic['layer2-3']
            MDF_dic['mean_length_singlechannelburst_layer4'] = mean_burstlength_s_per_layer_dic['layer4']
            MDF_dic['mean_length_singlechannelburst_layer5-6'] = mean_burstlength_s_per_layer_dic['layer5-6']
            MDF_dic['mean_length_singlechannelburst_whitematter'] = mean_burstlength_s_per_layer_dic['whitematter']
            MDF_dic['mean_length_singlechannelburst_noslice'] = mean_burstlength_s_per_layer_dic['noslice']
                
                
        recording_df = pd.DataFrame(MDF_dic, index=[0])
        df_list.append(recording_df)
        
        
        MAINDATAFRAME = pd.concat(df_list, ignore_index=True)
        
        os.chdir(mainoutputdirectory)
        MAINDATAFRAME.to_excel('output_MEA_recordings_overview.xlsx')
        MAINDATAFRAME.to_csv('output_MEA_recordings_overview.csv')


    print('Finished the analysis. Check your outputfolder.')
        

                
if __name__ == '__main__':
    main()