#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:26:06 2023

@author: jonas ort, md 
department of neurosurgery RWTH Aachen

"""


import os
import sys
import glob
import time
import copy
import pickle
import fnmatch
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import ast
import scipy
from scipy.signal import (butter, lfilter, freqz, find_peaks, correlate, gaussian, 
                          filtfilt)
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

import McsPy
import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg, Q_

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import networkx as nx
import plotly.graph_objects as go

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')



'''

Funtions

'''


'''

FILTERING BASICS

'''

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Create a Butterworth bandpass filter.

    Parameters:
    lowcut : float
        The lower frequency boundary of the bandpass filter in Hz.
    highcut : float
        The higher frequency boundary of the bandpass filter in Hz.
    fs : float
        The sampling rate of the signal in Hz.
    order : int, optional
        The order of the filter (default is 4).

    Returns:
    b, a : ndarray, ndarray
        Numerator (b) and denominator (a) polynomials of the IIR filter.
    """

    # Nyquist frequency is half the sampling rate
    nyq = 0.5 * fs

    # Calculate low and high cutoff frequencies relative to Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq

    # Generate the coefficients of the Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    return b, a


    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a Butterworth bandpass filter to a given dataset.

    Parameters:
    data : array_like
        The data to be filtered.
    lowcut : float
        The lower frequency boundary of the bandpass filter in Hz.
    highcut : float
        The higher frequency boundary of the bandpass filter in Hz.
    fs : float
        The sampling rate of the signal in Hz.
    order : int, optional
        The order of the filter (default is 4).

    Returns:
    y : ndarray
        The filtered output data.
    """

    # Generate the coefficients of the Butterworth bandpass filter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)

    # Apply the filter to the data using lfilter
    y = lfilter(b, a, data)

    return y
    
    
     
def detect_peaks(y):
    """
    Detect peaks in a given dataset.

    Parameters:
    y : array_like
        The input data in which peaks are to be detected.

    Returns:
    peaks : ndarray
        Indices of the peaks in `y`.
    y : array_like
        The original input data (returned as is).
    threshold : float
        The calculated threshold value used for peak detection.
    """

    # Calculate the threshold as five times the standard deviation of y
    # Alternatively, uncomment the next line to use a robust measure of dispersion
    # based on the median absolute deviation (MAD)
    threshold = 5 * np.std(y)
    # threshold = np.median(np.absolute(y) / 0.6745)

    # Use scipy's find_peaks function to find indices of peaks in -y
    # 'height' specifies the required height of the peaks
    # 'distance' specifies the required minimum horizontal distance (in number of samples) 
    # between neighboring peaks
    peaks, _ = find_peaks(-y, height=threshold, distance=50)

    return peaks, y, threshold



'''

READ IN DATA

'''

def get_channel_infos(filedirectory, meafile):
    """
    Extracts and prints information about channels from a MEA file.

    Parameters:
    filedirectory : str
        The directory where the MEA file is located.
    meafile : str
        The name of the MEA file.

    Returns:
    None
    """

    # Load the MEA file data
    channel_raw_data = McsPy.McsData.RawData(os.path.join(filedirectory, meafile))

    # Print various information about the MEA file
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

    # Get the number of analog streams and print it
    analognumber = len(channel_raw_data.recordings[0].analog_streams.keys())
    print(f'In total {analognumber} analog_streams were identified.\n')

    # Loop through each analog stream to print more detailed information
    for i in range(analognumber):
        keylist = []
        stream = channel_raw_data.recordings[0].analog_streams[i]

        # Extract channel IDs
        for key in stream.channel_infos.keys():
            keylist.append(key)
        channel_id = keylist[0]

        # Extract and print various channel-related information
        datapoints = stream.channel_data.shape[1]
        samplingfrequency = stream.channel_infos[channel_id].sampling_frequency
        ticks = stream.channel_infos[channel_id].info['Tick']
        time = stream.get_channel_sample_timestamps(channel_id)
        scale_factor_for_second = Q_(1, time[1]).to(ureg.s).magnitude
        time_in_sec = time[0] * scale_factor_for_second
        timelengthrecording_ms = time[0][-1] + ticks
        timelengthrecording_s = (time[0][-1] + ticks) * scale_factor_for_second
        print(f"analog_stream Nr. {i}: ")
        print(f"datapoints measured = {datapoints}")
        print(f"sampling frequency = {samplingfrequency}")
        print(f"ticks = {ticks}")
        print(f"total recording time is: {timelengthrecording_s} seconds \n")
        
        

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
    


'''

DETECT SPIKES

'''



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
        
        
'''

LOCAL FIELD POTENTIAL DETECTION

'''  
        
        
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

        
def lfp_crossing_detection(lowpass_filtered_signal, lfp_threshold, tick, scale_factor_for_second, time_in_sec, minimal_length):
    '''
    Detects LFP crossings in a lowpass-filtered signal in both directions.

    Parameters:
    lowpass_filtered_signal : array_like
        The lowpass filtered signal considered as the LFP.
    lfp_threshold : float
        The threshold for detecting LFP deviation.
    tick : float
        The duration of one tick in the recording.
    scale_factor_for_second : float
        Scaling factor to convert from ticks to seconds.
    time_in_sec : array_like
        Array of time points corresponding to each data point in the signal.
    minimal_length : float
        Minimal length of an LFP deviation to be considered relevant (in seconds).
    
    Returns:
    lfp_down_crossing : list of tuples
        List of (start, stop) times for downward LFP crossings.
    lfp_up_crossing : list of tuples
        List of (start, stop) times for upward LFP crossings.
    amplitudes_down : list
        List of maximal negative amplitudes for each downward crossing.
    amplitudes_up : list
        List of maximal positive amplitudes for each upward crossing.
    '''

    # Helper function to add crossing and amplitude information
    def add_crossing(start_idx, stop_idx, crossing_list, amplitude_list, direction):
        start_seconds = start_idx * scale_factor_for_second * tick + time_in_sec[0]
        stop_seconds = stop_idx * scale_factor_for_second * tick + time_in_sec[0]
        difference_seconds = stop_seconds - start_seconds

        if difference_seconds >= minimal_length:
            crossing_list.append((start_seconds, stop_seconds))
            if direction == 'down':
                amplitude_point = get_next_minimum(lowpass_filtered_signal, start_idx, stop_idx-start_idx)
            else:
                amplitude_point = get_next_maximum(lowpass_filtered_signal, start_idx, stop_idx-start_idx)
            amplitude = lowpass_filtered_signal[amplitude_point]
            amplitude_list.append(amplitude)

    lfp_up_crossing = []
    lfp_down_crossing = []
    amplitudes_up = []
    amplitudes_down = []

    # Initialize variables to track the state of crossings
    in_downward_crossing = False
    in_upward_crossing = False
    start_down, start_up = 0, 0

    # Iterate through the signal to detect crossings
    for i in range(1, len(lowpass_filtered_signal)):
        current_value = lowpass_filtered_signal[i]

        # Check for the start of a downward crossing
        if current_value < -lfp_threshold and not in_downward_crossing:
            in_downward_crossing = True
            start_down = i

        # Check for the end of a downward crossing
        elif current_value >= -lfp_threshold and in_downward_crossing:
            in_downward_crossing = False
            stop_down = i
            add_crossing(start_down, stop_down, lfp_down_crossing, amplitudes_down, 'down')

        # Check for the start of an upward crossing
        if current_value > lfp_threshold and not in_upward_crossing:
            in_upward_crossing = True
            start_up = i

        # Check for the end of an upward crossing
        elif current_value <= lfp_threshold and in_upward_crossing:
            in_upward_crossing = False
            stop_up = i
            add_crossing(start_up, stop_up, lfp_up_crossing, amplitudes_up, 'up')

    return lfp_down_crossing, lfp_up_crossing, amplitudes_down, amplitudes_up



def get_isi_single_channel(spikedic, tick, scale_factor_for_milisecond):
    '''
    Calculates the interspike interval (ISI) in milliseconds for each channel.

    Parameters:
    spikedic : dict
        A dictionary with keys as channel labels and values as spike times in raw ticks.
    tick : float
        The duration of one tick in the recording.
    scale_factor_for_milisecond : float
        Scaling factor to convert from ticks to milliseconds.

    Returns:
    isi_dictionary : dict
        A dictionary with keys as channel labels and values as lists of ISIs for each channel.

    Note:
    The function does not filter out channels based on the number of spikes. Channels with fewer than two spikes will have an empty list of ISIs.
    '''

    # Initialize a dictionary to store ISI values for each channel
    isi_dictionary = {}

    # Iterate through each channel in the spike dictionary
    for key, spikes in spikedic.items():
        # Convert spike times from ticks to milliseconds
        spikes_ms = [spike * tick * scale_factor_for_milisecond for spike in spikes]

        # Calculate ISIs only if there are at least two spikes
        isi_temp_list = []
        if len(spikes_ms) >= 2:
            for i in range(len(spikes_ms) - 1):
                # Calculate the ISI and append to the temporary list
                isi = spikes_ms[i + 1] - spikes_ms[i]
                isi_temp_list.append(isi)

        # Store the ISI list in the dictionary
        isi_dictionary[key] = isi_temp_list

    return isi_dictionary



def extract_spikes_thresholdbased(channel_idx, np_analog_for_filter, 
                                  analog_stream_0, 
                                  starting_point, 
                                  stopping_point, 
                                  lowcut, 
                                  highcut, 
                                  fs, 
                                  tick, 
                                  first_recording_timepoint):
    """
    Processes the signal for a given channel.

    Parameters:
    channel_idx : int
        Index of the channel to be processed.
    np_analog_for_filter : ndarray
        Array for filtering.
    analog_stream_0 : AnalogStreamType
        Analog stream from the MEA data.
    starting_point : float
        Start time of the signal in seconds.
    stopping_point : float
        Stop time of the signal in seconds.
    lowcut : float
        Low cutoff frequency for bandpass filter.
    highcut : float
        High cutoff frequency for bandpass filter.
    fs : float
        Sampling frequency.
    tick : float
        Duration of one tick in the recording.
    first_recording_timepoint : float
        First time stamp in the recording.

    Returns:
    channellabel : str
        Label of the processed channel.
    spikes : list
        Detected spikes in the channel.
    bandpassfilteredsignal : ndarray
        Bandpass-filtered signal of the channel.
    """

    labellist = get_MEA_Channel_labels(np_analog_for_filter, analog_stream_0)
    signal_in_uV, time_in_sec, sampling_frequency, scale_factor_for_second = get_MEA_Signal(
        analog_stream_0, channel_idx, from_in_s=starting_point, to_in_s=stopping_point
    )
    bandpassfilteredsignal = butter_bandpass_filter(signal_in_uV, lowcut, highcut, sampling_frequency)

    # Spike Detection
    noise_mad = np.median(np.absolute(bandpassfilteredsignal)) / 0.6745
    threshold = -5 * noise_mad
    crossings = detect_threshold_crossings(bandpassfilteredsignal, sampling_frequency, threshold, dead_time=0.001)
    spikes = align_to_minimum(bandpassfilteredsignal, fs, crossings, search_range=0.003, first_time_stamp=first_recording_timepoint)
    
    # Adjust for starting point
    spikes = spikes + int(time_in_sec[0] / (scale_factor_for_second * tick))
    channellabel = labellist[channel_idx]

    # Artifact Detection (optional, currently commented out)
    # artefact_threshold = -8 * noise_mad
    # artefact_crossings = detect_threshold_crossings(bandpassfilteredsignal, sampling_frequency, artefact_threshold, dead_time=0.001)
    # artefacts = align_to_minimum(bandpassfilteredsignal, fs, artefact_crossings, search_range=0.003, first_time_stamp=first_recording_timepoint)

    print('iteration ' + str(channel_idx) + ' channel: ' + str(channellabel))

    return channellabel, spikes, bandpassfilteredsignal





    


