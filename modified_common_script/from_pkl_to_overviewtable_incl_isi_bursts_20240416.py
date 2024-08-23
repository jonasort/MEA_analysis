#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 22:11:10 2023

@author: jonas
"""



'''

IMPORTS

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
import logging
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


from multiprocessing import Process, Queue


'''

FUNCTIONS

'''

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



def bin_spike_dictionary(spike_dic_sec, bin_length_ms, recording_length_sec):
    
    binned_spikedic_sec = {}
    
    # get the number of needed bins
    number_of_bins = int(recording_length_sec / (bin_length_ms*0.001))
    
    
    for key in spike_dic_sec:
        binned = np.histogram(spike_dic_sec[key], bins = number_of_bins, range= (0, recording_length_sec))
        binned_spikedic_sec[key] = binned[0]
    
    return binned_spikedic_sec



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

            


def find_shared_spiking_activity(binned_spikedic):
    
    spike_connection_dic = {}
    spike_connection_dic_simple = {}

    
    for key in binned_spikedic:
        other_keys = list(binned_spikedic.keys())
        other_keys.remove(key)
        connections = []
        connections_simple = []
        

        for j in other_keys: 
            
            number_shared = 0
            for i in binned_spikedic[key]:
                if i > 0:
                    if binned_spikedic[j][i] > 0:
                        number_shared += 1
                
            if number_shared > 0:
                connections.append((j, number_shared))
                connections_simple.append(j)

        spike_connection_dic[key] = connections
        spike_connection_dic_simple[key] = connections_simple

        

    return spike_connection_dic, spike_connection_dic_simple



def get_dict_values(df, dic):
    
    for i in df.columns:
        for j in range(0, 16):
            keystring = i+str(df.index[j])
            #print(keystring)
            if keystring in dic.keys():
                df.loc[df.index[j],i]=dic[keystring]
            
                
    
    return df


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

from multiprocessing import Process, Queue

'''
def calculate_sigma(graph, result_queue):
    try:
        sigma_value = nx.sigma(graph)
        result_queue.put(sigma_value)
    except Exception as e:
        result_queue.put(e)
'''


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






'''

SCRIPT

'''

def main():
    
    inputdirectory = input('Please enter the file directory: ')
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
        mainoutputdirectory = os.path.join(inputdirectory, 'output')
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
        Basics = MAIN_RECORDING_DICTIONARY['Basics']
        
        '''
        removal of artefacts
        '''
        
        spikedic_MAD, kicked_out = remove_artefact_spikes(spikes=spikedic_MAD, 
                                                          recording_length=timelengthrecording_s, 
                                                          tick=tick,
                                                          scale_factor_for_second=scale_factor_for_second)
        
        active_channels = 0
        active_relevant_channels = 0
        spikedic_seconds = {}
        for key in spikedic_MAD:
            sec_array = np.asarray(spikedic_MAD[key])*tick*scale_factor_for_second
            spikedic_seconds[key]=sec_array
            if len(sec_array)> 1:
                active_channels += 1
            if len(sec_array)> relevant_factor:
                active_relevant_channels += 1
            
            
        #spikearray_seconds = np.asarray(list(spikedic_seconds.values()))  
        spikearray_seconds = list(spikedic_seconds.values())
        
        
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
        
        #for ax in axs:
        #    for i in bursts_seconds:
        #        axs[1].axvspan(i[0], i[1], facecolor = '#5B89A6', alpha = 0.3)

        
        fig.savefig(os.path.join(outputdirectory, filename+'__raster_firingrate_plot_solo.png'), dpi = 300, bbox_inches='tight')
        fig.savefig(os.path.join(outputdirectory, filename+'__raster_firingrate_plot_solo.png'), dpi = 300, bbox_inches='tight')
        plt.close(fig)


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
        MAIN_RECORDING_DICTIONARY['network_bursts_seconds'] = bursts_seconds
        
        
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
        fig.savefig(os.path.join(outputdirectory, 
                                 filename+ '__raster_firingrate_plot.png'), dpi=300)
        
        
        
        # now we calculate the individual firing rates per channel
        whole_recording_firingrate_dic = {}
        
        # i.e, number of spikes divided by duration -> results in number per second
        for key in spikedic_MAD:
            fr_channel = len(spikedic_MAD[key])/timelengthrecording_s 
            whole_recording_firingrate_dic[key] = fr_channel
        
        # add it to the main dictionary
        MAIN_RECORDING_DICTIONARY['fr_dic'] = whole_recording_firingrate_dic
        
        
        '''
        2.3 
        basic spiking statistics to the recording
        
        '''
        # create the dictionary with isi + add it
        isi_dictionary = get_isi_single_channel(spikedic_MAD, tick=tick,
                                                scale_factor_for_milisecond=scale_factor_for_milisecond)
        MAIN_RECORDING_DICTIONARY['isi_dictionary'] = isi_dictionary
        
        
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
        
        MAIN_RECORDING_DICTIONARY['isi_average_dic'] = isi_average_dic
        MAIN_RECORDING_DICTIONARY['isi_std_dic'] = isi_standarddeviations_dic
        
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
        MAIN_RECORDING_DICTIONARY['number_rand_notrand_spikes_per_channel'] = number_rand_nrandom_spike_per_channeldic
        MAIN_RECORDING_DICTIONARY['rand_notrand_spikes_per_channel'] = random_nrandom_spike_per_channeldic

        
        # add missing information to the main recording dic
        Infos_Analysis = MAIN_RECORDING_DICTIONARY['Infos_Analysis']
        Infos_Analysis['relevant_factor'] = relevant_factor
        MAIN_RECORDING_DICTIONARY['Infos_Recording'] = Infos_Recording
        MAIN_RECORDING_DICTIONARY['Infos_Analysis'] = Infos_Analysis
        MAIN_RECORDING_DICTIONARY['Basics'] = Basics







        '''
        
        INTRACHANNEL BURSTS
        
        '''

        try:
            # get the isi distribution
            st_channel = spikedic_MAD
            binsize_for_ISI = 5000 #in microseconds
        
            isidic, isi_alone = get_isi_singlechannel(st_channel, tick) #creates two dictionaries
            histo_ISI_dic=bin_isi(isi_alone, binsize=binsize_for_ISI, 
                                  binmax=True, binmaxnumber=200) # dictionary für jeden channel mit 300x 10ms bins (binsize) und der Menge an ISI für die jeweilige Länge
            network_ISI=get_allchannel_ISI_bins(histo_ISI_dic) #gibt ein array mit bins entsprechend der bins aus der Vorfunktion
        
        
        
            
            colors = ['green', 'blue', 'orange', 'purple']
            df= pd.DataFrame({'ISI_per_10ms_bins':network_ISI}) #aus Network_ISI wird ein pdDF um die weiteren Schritte durchführen zu können
            
            try:
                df["CMA"] = df.ISI_per_10ms_bins.expanding().mean()
                df[['ISI_per_10ms_bins', 'CMA']].plot(color=colors, linewidth=3, 
                                                      figsize=(10,4), 
                                                      title="Histogram of ISI-bins 10ms whole network")
            
            
                # calculate the adaptive threshold
                CMAalpha, CMAalpha2, maxCMA, alpha1, alpha2=get_burst_threshold(df, network_ISI=network_ISI) # threshold calculation
            except:
                pass
                maxCMA = 0
                
                
            
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
            #np.save(filename+'_burst_starts_dictionary.npy', burststart_dic)
        
        
        
        
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
            
            
            # average burst statistics
            
            total_number_single_channel_bursts = 0
            total_number_channels_bursting = 0
            for value in bursts_per_unit_dic.values():
                total_number_single_channel_bursts += value
                
                if value > 0:
                    total_number_channels_bursting += 1
            
            total_length_bursts_of_all_channels = 0
            for value in bursting_time_per_unit_dic.values():
                total_length_bursts_of_all_channels += value
                
                
            mean_singlechannel_bursting_time_ms = total_length_bursts_of_all_channels/total_number_single_channel_bursts
            
            
            mean_interburst_interval = np.mean(list(burst_ibi_dic.values()))
            
            
        except:
            total_number_single_channel_bursts = 0
            total_number_channels_bursting = 0
            total_length_bursts_of_all_channels = 0
            mean_singlechannel_bursting_time_ms = 0
            mean_interburst_interval = 0

            
            
            pass
        
        

        






        '''
        
        Calculate Connected Channels depending on the
        
        '''
        # binned spiking dic
        dic = bin_spike_dictionary(spike_dic_sec = spikedic_seconds,
                            bin_length_ms = 200,
                           recording_length_sec = timelengthrecording_s)
        
        # calculate the connections
        connections, cs = find_shared_spiking_activity(binned_spikedic=dic)
        
        
        '''
        MEA Coordinates for Plotting
        '''
        
        columnlist =['A','B','C','D','E','F','G','H','J','K','L','M','N','O','P','R']

        mea_coordinates = np.linspace(0,1,16)
        mea_positional_coordinates_dic = {}
        
        for i in all_channels:
            x_num = columnlist.index(i[0])
            x_coord = mea_coordinates[x_num]
            y_num = 17-int(i[1:]) # minus 1 since python starts counting at zero
            y_coord = 1-(mea_coordinates[-y_num])
            mea_positional_coordinates_dic[i] = [x_coord, y_coord]
            
        '''
        CREATE THE GRAPH
        '''
        ## actual graph

        connections_graph = nx.Graph()
        for key in cs:
            for i in cs[key]:
                connections_graph.add_edge(key, i)
            
        connections_graph.number_of_nodes(), connections_graph.number_of_edges()
        G = connections_graph
        
        
        # calculate the corrected firing rate dic
        fr_dic_corrected = {}
        for key in spikedic_MAD:
            number_spikes = len(spikedic_MAD[key])
            fr = number_spikes/timelengthrecording_s
            fr_dic_corrected[key] = fr
            
            
        '''
        ADD Graph Properties
        '''
        for i in G.nodes():
        
            try:
                node_key = i
                coordinate = mea_positional_coordinates_dic[node_key]
                G.nodes[node_key]['pos']=coordinate
                G.nodes[node_key]['firing_rate']=fr_dic_corrected[i]
        
                try:
                    G.nodes[node_key]['degree_centrality']=nx.degree_centrality(G)[i]
                except:
                    print('degree centrality failed')
        
                try:
                    G.nodes[node_key]['betweenness_centrality']=nx.betweenness_centrality(G, k=10, endpoints = True)[i]
                except:
                    print('betweennes centrality failed')
        
            except KeyError:
                print('channel ', node_key, ' failed')
        
        
        pos = nx.get_node_attributes(G, 'pos')
        
        
        
        '''
        
        Calculate Graph Metrics
        
        '''


        # Configure logging
        logging.basicConfig(filename=filename + '_graph_analysis.log',  # Name of the log file
                    filemode='a',  # 'a' means append (add log entries to the file); 'w' would overwrite the file each time
                    level=logging.INFO,  # Logging level
                    format='%(asctime)s - %(levelname)s - %(message)s')  # Format of log messages
        
        
        # Initialize all variables to None as a default
        dgc, mean_degree_centrality, cc, closeness_centrality, evc, eigenvector_centrality = None, None, None, None, None, None
        average_shortest_path, average_clustering_coefficient, non_randomness, small_world_sigma, small_world_omega, diameter, node_connectivity = None, None, None, None, None, None, None
        
        try:
            # Degree Centrality
            if nx.is_connected(G):
                dgc = dict(nx.degree_centrality(G))
                mean_degree_centrality = np.round(np.mean(list(dgc.values())), 4)
            else:
                logging.warning("Graph is disconnected. Skipping degree centrality calculations.")
        except Exception as e:
            logging.error(f"Error calculating degree centrality: {e}")
        
        try:
            # Closeness Centrality
            if nx.is_connected(G):
                cc = dict(nx.closeness_centrality(G))
                closeness_centrality = np.round(np.mean(list(cc.values())), 4)
            else:
                logging.warning("Graph is disconnected. Skipping closeness centrality calculations.")
        except Exception as e:
            logging.error(f"Error calculating closeness centrality: {e}")
        
        try:
            # Eigenvector Centrality
            evc = dict(nx.eigenvector_centrality(G))
            eigenvector_centrality = np.round(np.mean(list(evc.values())), 4)
        except Exception as e:
            logging.error(f"Error calculating eigenvector centrality: {e}")
        
        try:
            # Average Shortest Path Length
            if nx.is_connected(G):
                average_shortest_path = nx.average_shortest_path_length(G)
            else:
                logging.warning("Graph is disconnected. Skipping average shortest path length calculation.")
        except Exception as e:
            logging.error(f"Error calculating average shortest path length: {e}")
        
        try:
            # Average Clustering Coefficient
            average_clustering_coefficient = nx.average_clustering(G)
        except Exception as e:
            logging.error(f"Error calculating average clustering coefficient: {e}")
        
        try:
            # Non-randomness
            non_randomness = nx.non_randomness(G)
        except Exception as e:
            logging.error(f"Error calculating non-randomness: {e}")
            
            
        '''
        try:
            # Small World Sigma
            small_world_sigma = nx.sigma(G)
        except Exception as e:
            logging.error(f"Error calculating small world sigma: {e}")
        
        try:
            # Small World Omega
            small_world_omega = nx.omega(G)
        except Exception as e:
            logging.error(f"Error calculating small world omega: {e}")
            
        '''
        
        '''

        if G.number_of_edges() == 0 or not nx.is_connected(G):
            print("Graph is empty or disconnected. Skipping `sigma` and `omega` calculations.")
        else:
            # Proceed with calculations if the graph is neither empty nor disconnected
            sigma = nx.sigma(G)
            omega = nx.omega(G)
            
            
        '''
            
        
                    
        

        try:
            # Diameter
            if nx.is_connected(G):
                diameter = nx.diameter(G)
            else:
                logging.warning("Graph is disconnected. Skipping diameter calculation.")
        except Exception as e:
            logging.error(f"Error calculating diameter: {e}")
        
        try:
            # Node Connectivity
            node_connectivity = nx.node_connectivity(G)
        except Exception as e:
            logging.error(f"Error calculating node connectivity: {e}")

            
        # Get the degrees of all nodes
        degrees = [degree for node, degree in G.degree()]

        
        # Count the number of occurrences of each degree value
        degree_count = {}
        for degree in degrees:
            if degree in degree_count:
                degree_count[degree] += 1
            else:
                degree_count[degree] = 1
        
        # Prepare the data for plotting
        degrees, counts = zip(*degree_count.items())
        
        
        # Sort the degrees and corresponding counts
        sorted_degrees_counts = sorted(zip(degrees, counts))
        sorted_degrees, sorted_counts = zip(*sorted_degrees_counts)
        
        # Plotting
        fig = plt.figure(figsize=(10, 6))
        plt.loglog(sorted_degrees, sorted_counts, 'bo-', markersize=8, alpha=0.6)  # Log-log plot
        plt.title('Degree Distribution on Log-Log Scale')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        fig.savefig(os.path.join(outputdirectory, 'degree_distribution'+filename+'_.png'), 
                    dpi = 300, bbox_inches='tight')
        fig.savefig(os.path.join(outputdirectory, 'digree_distribution'+filename+'_.eps'), 
                    dpi = 300, bbox_inches='tight')
        plt.show()
  
   
        
        '''
        PLOTTING OF THE GRAPH
        '''
        # degree centrality
        
        try:
            if dgc != None:
                fig, ax = plt.subplots(1,1, figsize=(10,10))
        
                nx.draw_networkx_edges(
                    G,
                    pos, 
                    alpha=0.1, 
                    edge_color='#000000',
                    ax=ax,
                )
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=list(dgc.keys()),
                    node_size=100,
                    node_color=list(dgc.values()),
                    cmap=plt.cm.Reds,
                    ax=ax
                )
                
                
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                #ax.set_xticks(columnlist)
                ax.grid(ls=':')
                #plt.axis("off")
                #ax.legend(handles=layer_colors.values(), labels=layer_colors.keys())
                plt.title('Graph - '+ filename + 'mean degree centrality = '+ str(mean_degree_centrality))
                
                fig.savefig(os.path.join(outputdirectory, 'graph_degreenes_centrality'+filename+'_.png'), 
                            dpi = 300, bbox_inches='tight')
                fig.savefig(os.path.join(outputdirectory, 'graph_degreenes_centrality'+filename+'_.eps'), 
                            dpi = 300, bbox_inches='tight')
                
                plt.close(fig)
        except:
            pass
            
        
        
        
        # eigenvector centrality
        try:
            if evc != None:
                fig, ax = plt.subplots(1,1, figsize=(10,10))
                
                
                nx.draw_networkx_edges(
                    G,
                    pos, 
                    alpha=0.1, 
                    edge_color='#000000',
                    ax=ax,
                )
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=list(evc.keys()),
                    node_size=100,
                    node_color=list(evc.values()),
                    cmap=plt.cm.Reds,
                    ax=ax
                )
                
                
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                #ax.set_xticks(columnlist)
                ax.grid(ls=':')
                #plt.axis("off")
                #ax.legend(handles=layer_colors.values(), labels=layer_colors.keys())
                plt.title('Graph - '+ filename + 'mean eigenvector centrality = '+ str(eigenvector_centrality))
                
                fig.savefig(os.path.join(outputdirectory, 'eigenvector_centrality'+filename+'_.png'), 
                            dpi = 300, bbox_inches='tight')
                fig.savefig(os.path.join(outputdirectory, 'eigenvector_centrality'+filename+'_.eps'), 
                            dpi = 300, bbox_inches='tight')
                
                plt.close(fig)
        except:
            pass








        '''
        PLOTTING OF MEA GRID
        
        '''
        # next we plot this on a mea array:
        
        mea_array=np.empty((16,16,))
        mea_array[:]=np.nan
        columnlist =['A','B','C','D','E','F','G','H','J','K','L','M','N','O','P','R']
        df = pd.DataFrame(data=mea_array,columns=columnlist)
        df.index+=1
        
        df1 = pd.DataFrame(data=mea_array,columns=columnlist)
        df1.index+=1
        
        df2 = pd.DataFrame(data=mea_array,columns=columnlist)
        df2.index+=1
        
        df3 = pd.DataFrame(data=mea_array,columns=columnlist)
        df3.index+=1
        
        use_df_copy = df.copy()
        use_df_copy1 = df1.copy()
        use_df_copy2 = df2.copy()
        use_df_copy3 = df3.copy()
        
        df_firing_rate = get_dict_values(use_df_copy, fr_dic_corrected)
        
        f, ax = plt.subplots(1, 1, figsize = (12,10))

        sns.heatmap(df_firing_rate, annot=False, linewidths=.5, ax=ax, cmap="rocket_r", vmax=2)
        ax.set_title('firing rate per channel [Hz]; mean fr all channels = ' + str(mean_fr_whole_recording))
        
        f.savefig(os.path.join(outputdirectory, filename+'whole_MEA_Heatmap.png'), dpi = 300, bbox_inches='tight')
        f.savefig(os.path.join(outputdirectory, filename+'whole_MEA_Heatmap.eps'), dpi = 300, bbox_inches='tight')
        plt.close(fig)

        
        
        '''
        Create DF for the Recording
        
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
        MDF_dic['active_channels'] = active_channels
        MDF_dic['relevant_active_channels'] = active_relevant_channels

        
        MDF_dic['mean_fr_whole_recording'] = mean_fr_whole_recording
        MDF_dic['mean_degree_centrality'] = mean_degree_centrality
        MDF_dic['closeness_centrality'] = closeness_centrality
        MDF_dic['mean_eigenvector_centrality'] = eigenvector_centrality
        MDF_dic['average_shortest_path_length'] = average_shortest_path
        MDF_dic['average_clustering_coefficient'] = average_clustering_coefficient
        MDF_dic['non_randomness'] = str(non_randomness)
        MDF_dic['sw_sigma'] = small_world_sigma
        MDF_dic['sw_omega'] = small_world_omega
        MDF_dic['diameter'] = diameter
        MDF_dic['node_connectivity'] = node_connectivity
        
        
        MDF_dic['number_channels_with_burst'] = total_number_channels_bursting
        MDF_dic['number_bursts_on_channels'] = total_number_single_channel_bursts
        MDF_dic['mean_singlechannel_bursting_time_ms'] = mean_singlechannel_bursting_time_ms
        #MDF_dic['mean_interspike_interval'] = mean_isi
        MDF_dic['mean_interspike_interval_relevant_channels'] = mean_isi_relevant_channels
        MDF_dic['mean_interburst_interval'] = mean_interburst_interval

        
        
        recording_df = pd.DataFrame(MDF_dic, index=[0])
        df_list.append(recording_df)
    
    
    MAINDATAFRAME = pd.concat(df_list, ignore_index=True)
    
    os.chdir(mainoutputdirectory)
    MAINDATAFRAME.to_excel('output_MEA_recordings_overview.xlsx')
    MAINDATAFRAME.to_csv('output_MEA_recordings_overview.csv')
    
    print('Finished the analysis. Check your outputfolder.')
        

                
if __name__ == '__main__':
    main()