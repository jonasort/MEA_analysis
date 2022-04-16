# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 09:36:23 2021

@author: jonas ort, MD, department of neurosurgery, RWTH Aachen Medical Faculty
"""

'''
DIRECTORIES
'''


scriptdirectory = r"//Users/naila/Documents/GitHub/MEA_analysis/CSA_JO"

# outputdirectory must already contain the .pkl file
output_directory = r"/Users/naila/Documents/DATA/ANALYZED/Victoria Witzig/ID006"

# filenam
filename = 'cortexmouse_div21_biometra_ID006_nodrug_spont_1'

# directory where the .npy dictionaries of lfp_signal are stored
temp_dir = r"/Users/naila/Documents/DATA/ANALYZED/Victoria Witzig/ID006/spike extraction"





'''

IMPORTS

'''
import os
os.chdir(scriptdirectory)


import sys
import numpy as np
import pandas as pd
import importlib

import McsPy
import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg, Q_

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.signal import butter, lfilter, freqz, find_peaks, correlate, gaussian, filtfilt
from scipy import stats
from scipy import signal

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dep_Butterworth_Filter import butter_bandpass, butter_bandpass_filter

import glob
#from plot_signal_and_spikes import plot_signal_and_spikes_from_bandpassfilteredsignal
import time

from neo.core import AnalogSignal
import quantities as pq

import pickle


'''

FUNCTIONS

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
                                  down_amplitudes, inverted_layerdic):
    
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


'''

SCRIPT

'''


os.chdir(output_directory)



# import the MAIN Dictionary
MAIN_RECORDING_DICTIONARY = pickle.load(open(
    os.path.join(output_directory+'/MAIN_RECORDING_Dictionary_'+filename+'.pkl'),
    "rb")
    )


network_bursts_seconds = MAIN_RECORDING_DICTIONARY['network_bursts_seconds']
spikedic_MAD = MAIN_RECORDING_DICTIONARY['spikedic_MAD']
fr_dic = MAIN_RECORDING_DICTIONARY['fr_dic']
isi_dictionary = MAIN_RECORDING_DICTIONARY['isi_dictionary']
isi_average_dic = MAIN_RECORDING_DICTIONARY['isi_average_dic']
Infos_Recording = MAIN_RECORDING_DICTIONARY['Infos_Recording']
Infos_Analysis = MAIN_RECORDING_DICTIONARY['Infos_Analysis']
Infos_Anatomy = MAIN_RECORDING_DICTIONARY['Infos_Anatomy']
Bursts = MAIN_RECORDING_DICTIONARY['Bursts']
Interburst_Intervals = MAIN_RECORDING_DICTIONARY['Interburst-Intervals']
bursting_time_per_channel = MAIN_RECORDING_DICTIONARY['bursting_time_per_channel']
bursts_per_channel = MAIN_RECORDING_DICTIONARY['bursts_per_channel']
burst_connections = MAIN_RECORDING_DICTIONARY['burst_connections']
Basics = MAIN_RECORDING_DICTIONARY['Basics']
inverted_layerdic = Infos_Anatomy['layerdic_invert']
layerdic = Infos_Anatomy['layerdic']
tick = 40
scale_factor_for_second = 1e-06
timelengthrecording_s = Infos_Recording['timelengthrecording_s']



# import all the extra files with the lfp amplitude deviations
os.chdir(temp_dir)


# create folderlist of the temp directory
folderlist = glob.glob('*'+filename+'*')


lfp_downs_dic = {}
lfp_downs_amp_dic = {}
lfp_ups_dic = {}
lfp_ups_amp_dic = {} 
#full_bandpass_dic = {}
#full_lowpass_dic = {}


for i in folderlist:
    
    subdirectory = os.path.join(temp_dir, i)
    os.chdir(subdirectory)
    
    # join the timekey (we need a dictionary structure to join all the subparts
    # for long recordings)
    timekey = '_'.join(i.split('_')[-6:-3])

    # get the respective filenames
    lowpassfile = glob.glob('*lowpass*.npy')[0]
    bandpassfile = glob.glob('*bandpass*.npy')[0]
    
    lfp_down_file = glob.glob('*LFP_DOWNS*.npy')[0]
    lfp_up_file = glob.glob('*LFP_UPS*.npy')[0]
    
    lfp_amp_down_file = glob.glob('*LFP_Amp_DOWNS*.npy')[0]
    lfp_amp_up_file = glob.glob('*LFP_Amp_UPS*.npy')[0]
    
    cs_lfp_down_file = glob.glob('*cs_lfp_down*.npy')[0]
    cs_lfp_up_file = glob.glob('*cs_lfp_up*.npy')[0]
    
    cs_lfp_amp_down_file = glob.glob('*cs_lfp_amp*do*.npy')[0]
    cs_lfp_amp_up_file = glob.glob('*cs_lfp_amp*up*.npy')[0]



    # bandpass and lowpass signal
    lowpass_dic = np.load(lowpassfile, allow_pickle=True).item()
    bandpass_dic = np.load(bandpassfile, allow_pickle=True).item()
    
    # lfp deviations
    lfp_downs = np.load(lfp_down_file, allow_pickle=True).item()
    lfp_ups = np.load(lfp_up_file, allow_pickle=True).item()
    
    # lfp amplitudes
    lfp_amplit_downs = np.load(lfp_amp_down_file, allow_pickle=True).item()
    lfp_amplit_ups = np.load(lfp_amp_up_file, allow_pickle=True).item()
    
    # cs lfp deviations
    cs_lfp_downs = np.load(cs_lfp_down_file, allow_pickle=True).item()
    cs_lfp_ups = np.load(cs_lfp_up_file, allow_pickle=True).item()
    
    # cs lfp amplitudes
    cs_lfp_amplit_downs = np.load(cs_lfp_amp_down_file, allow_pickle=True).item()
    cs_lfp_amplit_ups = np.load(cs_lfp_amp_up_file, allow_pickle=True).item()
    
    lfp_downs_dic[timekey] = lfp_downs
    lfp_downs_amp_dic[timekey] = lfp_amplit_downs
    lfp_ups_dic[timekey] = lfp_ups
    lfp_ups_amp_dic[timekey] = lfp_amplit_ups 
   # full_lowpass_dic[timekey] = lowpass_dic
   # full_bandpass_dic[timekey] = bandpass_dic
    


# these dictionaries have to be rejoined again in order to pass them into the network
# dictionary function

'''

Rejoining the Dictionaries for Long Recordings

This step is necessary to receive a joined version of all subdivided

'''

from collections import defaultdict

# join the lfp_down_dic
joined_lfp_down_dic = {}
dd = defaultdict(list)
for timekey in  lfp_downs_dic:
    for key, value in lfp_downs_dic[timekey].items():
        for i in value:
            dd[key].append(i)
for key in dd:
    sort_values = sorted(dd[key])
    joined_lfp_down_dic[key] = sort_values


# join the lfp_up_dic
joined_lfp_up_dic = {}
dd = defaultdict(list)
for timekey in  lfp_ups_dic:
    for key, value in lfp_ups_dic[timekey].items():
        for i in value:
            dd[key].append(i)
for key in dd:
    sort_values = sorted(dd[key])
    joined_lfp_up_dic[key] = sort_values


# join the lfp_downs_amp_dic
joined_lfp_downs_amp_dic = {}
dd = defaultdict(list)
for timekey in  lfp_downs_amp_dic:
    for key, value in lfp_downs_amp_dic[timekey].items():
        for i in value:
            dd[key].append(i)
for key in dd:
    sort_values = sorted(dd[key])
    joined_lfp_downs_amp_dic[key] = sort_values


# join the lfp_downs_amp_dic
joined_lfp_ups_amp_dic = {}
dd = defaultdict(list)
for timekey in  lfp_ups_amp_dic:
    for key, value in lfp_ups_amp_dic[timekey].items():
        for i in value:
            dd[key].append(i)
for key in dd:
    sort_values = sorted(dd[key])
    joined_lfp_ups_amp_dic[key] = sort_values





#we create the network_burst_dictionary that will be used to get our data into
#a dataframe format

network_bursts_dictionary, network_relevant_channels = find_network_burst_components(
                                  network_bursts_seconds, 
                                  Bursts, spikedic_MAD, joined_lfp_up_dic, 
                                  joined_lfp_ups_amp_dic, joined_lfp_down_dic,
                                  joined_lfp_downs_amp_dic, inverted_layerdic)



# the dictionary for dataframe is now iteratively filled with the information
# from 
dictionary_for_dataframe = {}
df = pd.DataFrame.from_records([dictionary_for_dataframe])

for key in network_bursts_dictionary:
    
    # fill up all basic recoding infos
    dictionary_for_dataframe["filename"] = filename
    dictionary_for_dataframe["recording_date"] = Infos_Recording["recordings_date"]
    dictionary_for_dataframe["timelength_recording_s"] = Infos_Recording["timelengthrecording_s"]
    dictionary_for_dataframe["firingrate_whole_recording_Hz"] = Basics["mean_fr_whole_recording"]
    dictionary_for_dataframe["active_channels_whole_recording"] = Basics["active_channels"]
    
    
    
    dictionary_for_dataframe["network_burst_seconds"] = key
    dictionary_for_dataframe["timelength_network_burst_s"] = network_bursts_dictionary[key]["timelength_network_burst_s"]
    dictionary_for_dataframe["bursting_channels"] = network_bursts_dictionary[key]["bursting_channels"]
    dictionary_for_dataframe["number_of_bursting_channels"] = network_bursts_dictionary[key]["number_of_bursting_channels"]
    dictionary_for_dataframe["number_burst_starts"] = network_bursts_dictionary[key]["number_burst_starts"]
    dictionary_for_dataframe["number_burst_ends"] = network_bursts_dictionary[key]["number_burst_ends"]
    dictionary_for_dataframe["active_channels_total"] = network_bursts_dictionary[key]["active_channels"]
    dictionary_for_dataframe["number_of_active_channels"] = network_bursts_dictionary[key]["number_of_active_channels"]
    dictionary_for_dataframe["number_of_spikes"] = network_bursts_dictionary[key]["number_of_spikes"]
    dictionary_for_dataframe["channels_lfp_up"] = network_bursts_dictionary[key]["channels_lfp_up"]
    dictionary_for_dataframe["number_channels_with_lfp_up"] = network_bursts_dictionary[key]["number_channels_with_lfp_up"]
    dictionary_for_dataframe["mean_up_lfp_amplitude"] = network_bursts_dictionary[key]["mean_up_lfp_amplitude"]
    dictionary_for_dataframe["channels_lfp_down"] = network_bursts_dictionary[key]["channels_lfp_down"]
    dictionary_for_dataframe["number_channels_with_lfp_down"] = network_bursts_dictionary[key]["number_channels_with_lfp_down"]
    dictionary_for_dataframe["mean_down_lfp_amplitude"] = network_bursts_dictionary[key]["mean_down_lfp_amplitude"]
    dictionary_for_dataframe["n_layer1_active_channels"] = network_bursts_dictionary[key]["n_layer1_channels"]
    dictionary_for_dataframe["n_layer23_active_channels"] = network_bursts_dictionary[key]["n_layer2-3_channels"]
    dictionary_for_dataframe["n_layer4_active_channels"] = network_bursts_dictionary[key]["n_layer4_channels"]
    dictionary_for_dataframe["n_layer56_active_channels"] = network_bursts_dictionary[key]["n_layer5-6_channels"]
    dictionary_for_dataframe["n_whitematter_active_channels"] = network_bursts_dictionary[key]["n_whitematter_channels"]
    
    # number of spikes per layer for the respective networkburst
    dictionary_for_dataframe["n_spikes_layer1"] = network_bursts_dictionary[key]["n_spikes_layer1"]
    dictionary_for_dataframe["n_spikes_layer23"] = network_bursts_dictionary[key]["n_spikes_layer23"]
    dictionary_for_dataframe["n_spikes_layer4"] = network_bursts_dictionary[key]["n_spikes_layer4"]
    dictionary_for_dataframe["n_spikes_layer56"] = network_bursts_dictionary[key]["n_spikes_laye5-6"]
    dictionary_for_dataframe["n_spikes_whitematter"] = network_bursts_dictionary[key]["n_spikes_whitematter"]
    
    # isi mean for the respective networkburst
    dictionary_for_dataframe["isi_mean_layer1_microsec"] = network_bursts_dictionary[key]["isi_mean_layer1"]
    dictionary_for_dataframe["isi_mean_layer23_microsec"] = network_bursts_dictionary[key]["isi_mean_layer23"]
    dictionary_for_dataframe["isi_mean_layer4_microsec"] = network_bursts_dictionary[key]["isi_mean_layer4"]
    dictionary_for_dataframe["isi_mean_layer56_microsec"] = network_bursts_dictionary[key]["isi_mean_layer56"]
    dictionary_for_dataframe["isi_mean_whitematter_microsec"] = network_bursts_dictionary[key]["isi_mean_whitematter"]
    
    # isi std for the respective networkburst
    dictionary_for_dataframe["isi_std_layer1"] = network_bursts_dictionary[key]["isi_std_layer1"]
    dictionary_for_dataframe["isi_std_layer23"] = network_bursts_dictionary[key]["isi_std_layer23"]
    dictionary_for_dataframe["isi_std_layer4"] = network_bursts_dictionary[key]["isi_std_layer4"]
    dictionary_for_dataframe["isi_std_layer56"] = network_bursts_dictionary[key]["isi_std_layer56"]
    dictionary_for_dataframe["isi_std_whitematter'"] = network_bursts_dictionary[key]["isi_std_whitematter"]
    
    # channels that are covered by slice
    dictionary_for_dataframe["channels_covered_layer1"] = len(layerdic['layer1'])
    dictionary_for_dataframe["channels_covered_layer23"] = len(layerdic['layer2-3'])
    dictionary_for_dataframe["channels_covered_layer4"] = len(layerdic['layer4'])
    dictionary_for_dataframe["channels_covered_layer56"] = len(layerdic['layer5-6'])
    dictionary_for_dataframe["channels_covered_whitematter"] = len(layerdic['whitematter'])
    



    

    df = df.append(pd.DataFrame.from_records([dictionary_for_dataframe]))


# save the dataframe
with open(os.path.join(output_directory+'/DATAFRAME_'+filename+'.pkl'), 'wb') as f:
    pickle.dump(df, f)





'''

to do 
start = 0
stop = timelengthrecording_s
tick_micro = 0.00004
time_in_sec = np.arange(start, stop+tick_micro, tick_micro)


plot_networkburst_traces(network_bursts_dictionary=network_bursts_dictionary,
                         filebase=filename)

'''




































