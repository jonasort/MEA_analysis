#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:50:37 2023

@author: jonas
"""

'''

IMPORTS

'''

# Standard library imports
import os
import sys
import glob
import ast
from pathlib import Path


# Related third party imports
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy.signal import butter, lfilter, freqz, find_peaks, correlate, gaussian, filtfilt
from scipy import stats
from scipy import signal
from IPython.core.interactiveshell import InteractiveShell
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import quantities as pq
from quantities import ms, s, Hz
import pickle

# Local application/library specific imports
import neo
from neo.io import NeuroExplorerIO
import McsPy
import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg, Q_
import elephant
from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process
from elephant import neo_tools as nt
from elephant.spade import spade
from elephant.spade import pvalue_spectrum
from elephant.spade import concepts_mining
from elephant.spade import concept_output_to_patterns

InteractiveShell.ast_node_interactivity = 'all'





'''

FUNCTIONS

'''

def spikedic_to_neospiketrains(spikedic, recordinglength):
    
    spiketrains = [] 
    keylist_spiketrains = []
    for key in spikedic:
        key_array_sec=np.asarray(spikedic[key])
        
        # adjust the minimal amount of spikes per spiketrain
        if len(key_array_sec)>0:
            st = neo.SpikeTrain(list(key_array_sec), units='sec', t_stop=recordinglength)
            spiketrains.append(st)
            keylist_spiketrains.append(key)
        
    return spiketrains, keylist_spiketrains







def cut_reloaded_spikedic(spikedic, scale_factor_for_second, start=0, stop=60):

    cutted_dic = {}
    temp_list = []
    
    for key in spikedic:
        templist = []
        spikes = spikedic[key]
        for i in spikes:
            j = i*scale_factor_for_second # the ticks have already been applied in the burststarts
            if (j>=start) and (j<stop):
                templist.append(i)
        cutted_dic[key] = templist
        
    return cutted_dic




def remove_artefact_spikes(spikes, recording_length, tick, scale_factor_for_second, window=0.01):
    
    removing_times = []
    i = 0
    removed_spikes = {} 
    kicked_out = {}
    while i < recording_length:
        activechannels = 0
        for key in spikes:
            for s in spikes[key]:
                if i <= s*tick*scale_factor_for_second < (i + window):
                    activechannels += 1
        if activechannels >= 100:
            removing_times.append(i)
        i += window

    
    print(removing_times)
    for key in spikes:
        new_spikes = []
        kicked_out_list = []

        s_list_save = list(spikes[key])
        s_list = list(spikes[key])
        for s in s_list_save:
            for time in removing_times:
                if time <= s*tick*scale_factor_for_second < (time + window*2):
                    kicked_out_list.append(s)
                    try:
                        s_list.remove(s)
                    except ValueError:
                        pass
                #else:
                 #   new_spikes.append(s)
        removed_spikes[key] = s_list
        kicked_out[key] = kicked_out_list

    return removed_spikes, kicked_out




def gaussian_smoothing(y, window_size=10, sigma=500):

    filt = signal.gaussian(window_size, sigma)

    return signal.convolve(y, filt, mode='same')



def convert_spiketimes_to_seconds(spike_dict, scale_factor, tick):
    """
    Convert spike times to seconds.
    
    Parameters:
    spike_dict (dict): Dictionary with channel labels as keys and lists of integer spike times as values.
    scale_factor (int): Scale factor to convert spike times to seconds.
    tick (int): Tick of the recordings.
    
    Returns:
    dict: A dictionary with channel labels as keys and lists of spike times in seconds as values.
    """
    
    # Convert spike_dict values to seconds
    spike_dict_seconds = {k: [tick_val * scale_factor * tick for tick_val in v] for k, v in spike_dict.items()}
    
    return spike_dict_seconds




def extract_spikes_in_lfp_periods(spike_dict_seconds, lfp_start_times, window=1.0):
    """
    Extract spikes that occur within a window period after each LFP start time, and adjust the spike times so
    that they appear as if the LFP periods occurred consecutively.
    
    Parameters:
    spike_dict_seconds (dict): Dictionary with channel labels as keys and lists of spike times in seconds as values.
    lfp_start_times (list): List of LFP start times.
    window (float, optional): Window period after each LFP start time. Default is 1.0.
    
    Returns:
    dict: A dictionary with channel labels as keys and lists of adjusted spike times within the LFP periods as values.
    """
    
    # Initialize an empty dictionary to store the spikes within the LFP periods
    spikes_in_lfp_periods = {k: [] for k in spike_dict_seconds.keys()}

    # Initialize a variable to keep track of the total time of the previous LFP periods
    total_previous_time = 0.0

    # Loop over the LFP start times
    for start_time in lfp_start_times:
        end_time = start_time + window

        # Loop over the channels
        for channel, spike_times in spike_dict_seconds.items():
            # Add spikes that fall within the window period after the LFP start time, 
            # and adjust the spike times as if the LFP periods occurred consecutively
            spikes_in_lfp_periods[channel].extend([(t - start_time + total_previous_time) for t in spike_times if start_time <= t < end_time])
            
        # Add the current window period to the total time of the previous LFP periods
        total_previous_time += window
            
    return spikes_in_lfp_periods





'''

SCRIPT

'''

# for testing:
#/Users/jonas/Documents/DATA/SPADE_LFP_202305/SPADE_Trial/output/2022-04-06_cortex_div13_biometra_ID2203039CE_nodrug_spont_2


def main():
    
    #inputdirectory = input('Please enter the file directory: ')
    os.chdir(inputdirectory)
    filelist = glob.glob('*.pkl')
    
    print(filelist)
    df_list = []
    for i in filelist:
        filename = i
        filename = i.split('Dictionary_')[1].split('.pkl')[0]

        print('Working on ' + filename)
        print('this is element ' + str(filelist.index(i)) + ' of ' + str(len(filelist)))
        
        
        #create the outputdirectory
        mainoutputdirectory = os.path.join(inputdirectory, 'output_spade')
        outputdirectory = os.path.join(mainoutputdirectory, filename)
        try:
            Path(outputdirectory).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        
        
        
        '''
        all basic parameters are imported
        '''
        MAIN_RECORDING_DICTIONARY = pickle.load(open('MAIN_RECORDING_Dictionary_'+filename+'.pkl', "rb"))
        spikedic_MAD = MAIN_RECORDING_DICTIONARY['spikedic_MAD']
        tick = MAIN_RECORDING_DICTIONARY['Infos_Recording']['tick']
        timelengthrecording_s = MAIN_RECORDING_DICTIONARY['Infos_Recording']['timelengthrecording_s']
        info_dics_subrecordings = MAIN_RECORDING_DICTIONARY['Infos_Recording']['info_dics_subrecordings']
        Infos_Recording = MAIN_RECORDING_DICTIONARY['Infos_Recording']
        key_timepoint = list(info_dics_subrecordings.keys())[0]
        first_recording_timepoint = info_dics_subrecordings[key_timepoint]['first_recording_timepoint']
        subrec_infos = info_dics_subrecordings[key_timepoint]
        scale_factor_for_second = MAIN_RECORDING_DICTIONARY['Infos_Recording']['scale_factor_for_second']
        relevant_factor = timelengthrecording_s*0.05
        fr_dic = MAIN_RECORDING_DICTIONARY['fr_dic']
        
        '''
        removal of artefacts
        '''
        
        spikedic_MAD, kicked_out = remove_artefact_spikes(spikes=spikedic_MAD, 
                                                          recording_length=timelengthrecording_s, 
                                                          tick=tick,
                                                          scale_factor_for_second=scale_factor_for_second)
        
        '''
        remove
        
        
        '''
        active_channels = 0
        spikedic_seconds = {}
        for key in spikedic_MAD:
            sec_array = np.asarray(spikedic_MAD[key])*tick*scale_factor_for_second
            spikedic_seconds[key]=sec_array
            active_channels += 1
            
        spikearray_seconds = np.asarray(list(spikedic_seconds.values()))  
        
        
        '''
        get one-dimensional array of every detected spike
        '''
        scale_factor_for_milisecond = 1e-03
        full_spike_list = []
        full_spike_list_seconds = []
        for key in spikedic_MAD:
            # without the relevant factor --every channel will be plotted
            x = list(np.asarray(spikedic_MAD[key])*scale_factor_for_milisecond*tick)
            full_spike_list = full_spike_list + x
    
            xs = list(np.asarray(spikedic_MAD[key])*scale_factor_for_second*tick)
            full_spike_list_seconds = full_spike_list_seconds + xs
        full_spikes = sorted(full_spike_list)
        full_spikes_seconds = sorted(full_spike_list_seconds)
        
        mean_fr_whole_recording = np.around((len(full_spikes_seconds) / timelengthrecording_s), 3)
        
        
        
        #define bins 
        binsize = 0.005 #seconds
        #bins= np.arange(0, timelengthrecording_s+binsize, binsize)
        
        #trial of population burst plot as inspired by Andrea Corna
        bins = int(timelengthrecording_s / binsize)+1
        
        firing_rate_histogram = np.histogram(full_spikes_seconds, bins=bins)
        firing_rate = firing_rate_histogram[0]*200 #conversion to hertz
        #firing_rate = firing_rate_histogram[0]
        
        
        # using firing rate histogram already conversed to hertz
        N = int(1/binsize) # für eine Sekunde, das Sliding window, also letztlich number of bins
        plot_N = int(0.01/binsize)
        # gaussian smmothing fo the firing rate and moving average
        fr_gau = gaussian_smoothing(firing_rate)
        plot_fr_gau = gaussian_smoothing(firing_rate)
        ma_fr_gau = np.convolve(fr_gau, np.ones(N)/N, mode='full')
        plotting_ma_fr_gau = np.convolve(fr_gau, np.ones(plot_N)/plot_N, mode='full')
        
        # we look for the mean of the MA as threshold
        # we arrange this mean in an array for plotting
        mean_ma_fr_gau = np.mean(ma_fr_gau)
        std_ma_fr_gau = np.std(ma_fr_gau)
        network_burst_threshold = mean_ma_fr_gau
        shape_for_threshold = np.shape(ma_fr_gau)
        network_burst_threshold_array = np.full(shape_for_threshold, network_burst_threshold)
        
        
        '''
        PLOTTING OF THE corrected Rasterplot
        '''
        
        fig = plt.figure(figsize = (10,6))
        gs = fig.add_gridspec(2, hspace = 0, height_ratios=[2,5])
        axs = gs.subplots(sharex=False, sharey=False)
        axs[0].plot(ma_fr_gau, color= 'black', lw= 0.5)
        axs[0].set_ylabel('Firing Rate [Hz]')
        axs[0].set_ylim(0, 5000)
        axs[1].eventplot(spikearray_seconds, color = 'black', linewidths = 0.5,
                         linelengths = 1, colors = 'black')
        axs[1].set_ylabel('Relevant Channels')
        fig.suptitle(filename)
        
        
        
        '''
        remove until here

        '''
        lfp_list = [4.56, 73.9, 99.36, 147.26, 167.29, 179.9, 194.14, 217.1, 238.38, 254.1]

    
        spikedic_in_seconds = convert_spiketimes_to_seconds(spike_dict=spikedic_MAD, 
                                                            scale_factor=scale_factor_for_second, 
                                                            tick=tick)
        
        spikes_in_lfp_periods = extract_spikes_in_lfp_periods(spike_dict_seconds=spikedic_in_seconds, 
                                                              lfp_start_times = lfp_list, 
                                                              window=1.0)
        
        
        
        # add the parameters for SPADE
        bin_size = 5 * pq.ms # time resolution to discretize the spiketrains
        winlen = 10 # maximal pattern length in bins (i.e., sliding window)
        dither = 20 * pq.ms
        spectrum = '#'
        alpha = 0.05
        stat='fdr_bh'
        
        # get the recordinglenght needed to create the neo spikearray
        
        spikes = []
        for key in spikes_in_lfp_periods:
            spikes.append(spikes_in_lfp_periods[key])
            spikearray = np.sort(np.concatenate(spikes, axis = 0))
            spikelist = list(spikearray)

        recordinglength = round(spikearray[-1]) + 1
        
        
        # create the neo spiketrain
        spiketrains, keylist_spiketrains = spikedic_to_neospiketrains(spikedic=spikes_in_lfp_periods, 
                                                                      recordinglength=recordinglength)
            
        
        result_spade = spade(spiketrains, bin_size=bin_size, winlen=winlen, n_surr=500, min_occ=4, min_spikes=2, min_neu=2, stat_corr=stat, 
                       spectrum=spectrum, alpha=alpha)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

