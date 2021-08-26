# -*- coding: utf-8 -*-

""" 
Created on Fri Jul 31 09:53:07 2020

@author: jonas ort, department of neurosurgery, RWTH AACHEN, medical faculty

ADJUSTMENTS:    - define three directories, they should all exist BEFORE running this script 
                - adjust pre and post cutouts 
                - adjust high- and low-cut filters

DEPENDENCIES:   - see imports   
                - plot_signal_and_spikes.py 
                - Butterworth_Filter.py

INPUT:          - .h5 File of a MEA recording as generated by the MCS Data Manager,
                    can be be arranged in one folder

OUTPUT:         - spikes_STD_dict.npy = Dictionary of Spiketimes as detected by 
                    threshold crossing of -5* STD 
                - spikes_MAD_dict.npy = Dictionary of Spiketimes as detected by 
                    threshold crossing of -5* MEDIAN ABSOLUTE DEVIATION 
                - wavecutouts_dict.npy = Dictionary of all Wavecutouts for each 
                    channel of each slice - firingrate_dict.npy = Dictionary of 
                    the firingrate for each channel 
                - For each channel with at least 1 Spike detected, a plot of the 
                    bandpass- filtered Signal, the threshold and the spikes will 
                    be created

NOTA BENE:      - Cutting out waveforms takes up a lot of memory and increases 
                    the working time profoundly. 
                - Most Outputs are set on inactive with #. Uncheck what you need. 
                - Depending on the working system, the length of an array that can 
                    be passed towards the plotting function is limited. Recordings 
                    above 120 seconds will run out of memory on a 64GB RAM machine, 
                    causing the script to stop.

ACKNOWLEDGEMENTS: - This script uses the code snippets of a Tutorial by MSC PyDataTools 
                    and were adjusted to the researchers needs.

OVERVIEW: 1. Directories and Inputs 2. Funcions 3. Working Script

"""

"""
DIRECTORIES
"""

# adjust the directories first!
scriptdirectory = "C:/Users/User/Documents/JO/gitkraken/MEA_analysis/Tübingen_Branch"
inputdirectory = r"D:\Files_Reutlingen_Jenny\19-04-25"


"""
IMPORTS
"""


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
from scipy.signal import butter, lfilter, freqz, find_peaks, correlate, gaussian
from scipy import stats
from scipy import signal
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

import warnings
warnings.filterwarnings('ignore')

from Butterworth_Filter import butter_bandpass, butter_bandpass_filter
import glob
import scipy
import matplotlib.pyplot as plt
from plot_signal_and_spikes import plot_signal_and_spikes_from_bandpassfilteredsignal
import time
import glob







timestr = time.strftime("%d%m%Y")
outputdirectory=r"D:\Files_Reutlingen_Jenny\19-04-25\190425_paper"



'''
_____________________________FUNCTIONS____________________________________
'''

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
    

#@jit(nopython=True)
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

#@jit(nopython=True)
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
    
 
#@jit(nopython=True)    
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
Actual Script
'''

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

# get filelist
os.chdir(inputdirectory)
filelist= glob.glob("*.h5")


resting_spikedic={}
spikedic={}
spikedic_MAD={}
artefactsdic_MAD={}
cutouts_dic ={} 
keylist = []


# for-loop files
for file in filelist:
    resting_spikedic={}
    spikedic={}
    cutouts_dic ={} 
    keylist = []
    filename = file
    #filedatebase = filename.split('T')[0]
    #filenamebase = filename.split('__')[1]
    filebase = filename.split('.')[0]
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
        if starting_point + dividing_seconds > int(timelengthrecording_s):
            stopping_point = int(timelengthrecording_s)
            break
        else:
            stopping_point = stopping_point + dividing_seconds
        signal_cuts.append((starting_point, stopping_point))
        
        
        # set the window one step further:
        starting_point = stopping_point
    
    # unfortunately another for loop to get through the subrecordings
    
    for i in signal_cuts:
        starting_point = i[0]
        stopping_point = i[1]
    
    
        #timestr = time.strftime("%d%m%Y")
        outpath = os.path.join(
            outputdirectory, filebase + '_from_'+str(starting_point) + 
            '_to_' +str(stopping_point) + '_analyzed_on_'+timestr).replace("\\","/")
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
            cutouts_dic ={}
        
            channel_idx = i
            labellist = get_MEA_Channel_labels(np_analog_for_filter)
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
            
            
            raw_spikes = spikes + first_recording_timepoint/tick
            spikes = raw_spikes + int(time_in_sec[0]/(scale_factor_for_second*tick))
            channellabel = labellist[i]
            spikedic_MAD[channellabel] = spikes
            #artefactsdic_MAD[channellabel] = artefacts
            print('iteration ' + str(i) + 'channel: ' +str(channellabel))
            
            
            # if there are detected spikes get the waveforms, plot the channel and waveforms and save
            if len(spikes > 0):
                cutouts = extract_waveforms(
                        bandpassfilteredsignal, sampling_frequency, raw_spikes, 
                        pre, post
                        )
                cutouts_dic[channellabel] = cutouts
                
                
                plt.style.use("seaborn-white")
                
                
                # figure 1: signal with threshold
                fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                ax = plt.plot(time_in_sec, bandpassfilteredsignal, c="#1E91D9")
                ax = plt.plot([time_in_sec[0], time_in_sec[-1]], [threshold, threshold], c="#297373")
                ax = plt.plot(spikes*tick*scale_factor_for_second, [threshold-1]*(spikes*tick*scale_factor_for_second).shape[0], 'ro', ms=2, c="#D9580D")
                ax = plt.title('Channel %s' %channellabel)
                ax = plt.xlabel('Time in Sec, Threshold: %s' %threshold)
                ax = plt.ylabel('µ volt')
                
                fig.savefig(filebase+'_signal_'+channellabel+'MAD_THRESHOLD_artefact.png')
                plt.close(fig) 
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
        
                fig2.savefig(filebase+'_waveforms_'+channellabel+'_.png')
                plt.close(fig2)
                plt.clf()
                
                '''
                delete cutouts dic to spare memory usage
                '''
                
                del cutouts_dic
           
                
                
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
        fig.savefig('raster_firingrate_plot.png', dpi=300)
        plt.close(fig)
        
        
        
        
        # lastly we save important information of the recording into a dictionary
        # this way, we can easily access them for further analysis
        
        info_dic = {}
        info_dic['tick']=tick
        info_dic['timelengthrecording_s']=timelengthrecording_s
        info_dic['first_recording_timepoint']=first_recording_timepoint
        info_dic['scale_factor_for_second']=scale_factor_for_second
        
        if file == filelist[0]:
            info_dic['network_burst_threshold_basline']=network_burst_threshold
        
        
        
    
            
        #np.save(filename+'_spikes_STD_dict.npy', spikedic)
        np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_spikes_MAD_dict.npy', spikedic_MAD) 
        #np.save(filename+'_wavecutouts_dict.npy', cutouts_dic)
        #np.save(filename+'_firingrate_dict.npy', frdic)
        np.save(filename+'_'+str(starting_point)+'_'+str(stopping_point)+'_info_dict.npy', info_dic)
        os.chdir(inputdirectory)
        
        
        #outpath = r'D:\MEA_DATA_Aachen\ANALYZED\2021-05-10_cortex_div4_hCSF_ID039_nodrug_spont_1_analyzed_on_23072021'
        #os.chdir(outpath)
        #spikedic_MAD = np.load(filename+'_spikes_MAD_dict.npy', allow_pickle=True).item()
        
        

print('Finished. You can close me, Henner.')














