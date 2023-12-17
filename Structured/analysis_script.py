#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:45:31 2023

@author: jonas
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


# import MEA functions

import meafunctions as mf



modulebool_spikes = False
modulebool_lfp = False
modulebool_waveforms = False
modulebool_location = False


def main():
    #inputdirectory = input('Please enter the file directory: ')
    inputdirectory = '/Users/jonas/Documents/DATA/Data_Tuscany'
    os.chdir(inputdirectory)
    filelist= glob.glob("*.h5")
    layerdictionary_list = glob.glob('*layerdic*')
    print(filelist)
    

    modulebool_location = 0
    while modulebool_location != ('A' or 'R'):
        modulebool_location = input('Enter A if this file is from Aachen and R if it is from Reutlingen: ')
        if modulebool_location != ('A' or 'R'):
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
    
    
    '''
    
    LOOP to go through each file and split it into 120sec. parts
    
    '''
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
        mf.get_channel_infos(inputdirectory, filename)
        
        
        
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
                
                
                
                '''
                
                SPIKE EXTRACTION
                
                '''
                    
                for i in range (0, len(np_analog_for_filter)):
                    
                    
                    channellabel, spikes, bandpassfilteredsignal = mf.extract_spikes_thresholdbased(channel_idx = i, 
                                                     np_analog_for_filter = np_analog_for_filter, 
                                                     analog_stream_0 = analog_stream0, 
                                                     starting_point = startingpoint, 
                                                     stopping_point = stopping_point, 
                                                     lowcut = lowcut, 
                                                     highcut = highcut, 
                                                     fs = fs, 
                                                     tick = tick, 
                                                     first_recording_timepoint = first_recording_timepoint)
                    
                    spikedic_MAD[channellabel] = spikes

                    




























