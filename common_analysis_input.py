#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 00:00:28 2022

@author: jonas
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




















'''
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def print_usage():
    eprint("Usage: python mea_script.py <input_directory>")
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
    
     


def detect_peaks(data):
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
        
        
def butter_lowpass_filter(data, cutoff, fs, order):

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



def lfp_crossing_detection(lowpass_filtered_signal, lfp_threshold, minimal_length = 0.05):
    
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




def get_isi_single_channel(spikedic):
    
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


def gaussian_smoothing(y, window_size=10, sigma=2):

    filt = signal.gaussian(window_size, sigma)

    return signal.convolve(y, filt, mode='full')





def invert_layerdic(layer_dic):
    
    '''
    Expects a dictionary with key = layer, value = list of channellabels
    
    Returns a dictionary with key = channellabels, value = layer
    '''
    layerdic_invert = {}

    for key in layerdic:
        for i in layerdic[key]:
            layerdic_invert[i]=key
            
            
    return layerdic_invert








def main():
    
    inputdirectory = input('Please enter the file directory: ')
    os.chdir(inputdirectory)
    filelist= glob.glob("*.h5")
    print(filelist)
    
    bool_channelmap = input('Do you want to use a labeldictionary? Enter y or n: ')    
    bool_location = input('Enter A if this file is from Aachen and R if it is from Reutlingen')    

    bool_modules = input('If you want the basic analysis (spikes only), enter b. If you want extended analysis (including lfp times), enter e: ')
    
    
    
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
        outputdirectory = os.path.join(inputdirectory, 'output', filebase)
        try:
            os.mkdir(outputdirectory)
        except OSError:
            pass
        
        outputdirectory_spikeextraction = os.path.join(outputdirectory, 'spike_extraction')
        try:
            os.mkdir(outputdirectory_spikeextraction)
        except OSError:
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
                    os.mkdir(outpath)
                except OSError:
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
                        
                        butter_lowpass_filtered_signal = butter_lowpass_filter(signal_in_uV, cutoff, fs, order)
                        
                        
                        lfp_std = np.std(butter_lowpass_filtered_signal)
                        lfp_mean = np.mean(butter_lowpass_filtered_signal)
                        threshold_LFP = 3*lfp_std
                        
                        
                        down_cross, up_cross, amp_down, amp_up = lfp_crossing_detection(
                            butter_lowpass_filtered_signal, threshold_LFP, minimal_length=0.03)
                        
                        
                        '''
                        Modulation for the LFP Start
                        '''
                        convolved_signal = np.convolve(butter_lowpass_filtered_signal, np.ones(3000)/3000, mode='full')
                        
                        
                        length_cutter = len(butter_lowpass_filtered_signal)
                        
                        cs = convolved_signal[:length_cutter]
                        
                        cs_threshold = np.std(cs)*2
                        
                        cs_down_cross, cs_up_cross, cs_amp_down, cs_amp_up = lfp_crossing_detection(
                            convolved_signal, cs_threshold, minimal_length=0.01)
                        
                        
                        
                        
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
    
    
    
    outputfolderlist = glob.glob(outputdirectory)
    
    #from here we will loop to each main outputfolder
    for folder in folderlist:
        os.chdir(folder)
        working_directory = os.path.join(folder, 'spike_extraction')
        filename = folder.split('/')[-1]
        
        
        
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
        folderlist = glob.glob(filename+'*')
        
        
        
        
        # get into every folder and find the dictionaries
        # replace them in a two meta-dictionaries (infodics and spikedics)
        infodics = {}
        spikedics = {}
        
        for folder in folderlist:
            os.chdir(os.path.join(working_directory, folder))
            # cave: here the slicing needs to be adjusted dependent on reutlingen filenames
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
        np.save(os.path.join(output_directory, 'infodics_'+filename+'.npy'), infodics)
        
        # get the first of all infodics
        first_info_dic_key = list(infodics.keys())[0]
        infodic = infodics[first_info_dic_key]
        
        '''
        ADD the info_dics to our pickle data
        '''
        
        Infos_Recording['info_dics_subrecordings'] = infodics
        Infos_Recording['recordings_date'] = recordingdate
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
        np.save(os.path.join(output_directory, filename +'_full_spikedic.npy'), spikedic_MAD_full)
        
        
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
        fig.savefig(os.path.join(output_directory, 
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
        add the basic spiking statistics to the recording
        
        '''
        # create the dictionary with isi + add it
        isi_dictionary = get_isi_single_channel(spikedic_MAD_full)
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
        
        
        '''
        load the anatomy as dicionary and invert it, add it to the infos
        '''
        
        # uncheck these when finished
        #layerdic_invert = invert_layerdic(layerdic)
        #Infos_Anatomy['layerdic'] = layerdic
        #Infos_Anatomy['layerdic_invert'] = layerdic_invert
        
        
        # add missing information to the main recording dic
        Infos_Analysis['relevant_factor'] = relevant_factor
        main_recording_dictionary['Infos_Recording'] = Infos_Recording
        main_recording_dictionary['Infos_Analysis'] = Infos_Analysis
        main_recording_dictionary['Infos_Anatomy'] = Infos_Anatomy
        main_recording_dictionary['Basics'] = Basics
        
        # and finally pickle the main_recording_dictionary
        with open(os.path.join(output_directory+'/MAIN_RECORDING_Dictionary_'+filename+'.pkl'), 'wb') as f:
                  pickle.dump(main_recording_dictionary, f)
            
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()