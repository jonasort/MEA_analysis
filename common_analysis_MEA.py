#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:56:28 2022

@author: jonas ort, md
"""



'''
input: .h5 files

output:
    excel sheet with recording names, basic parameters, firing rate, firing rate per layer, burst rate, burst times
    mean isi, mean ibi, unit bursts
'''




# imports
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import numpy as np
import glob
import os
import sys
import numpy as np
import neo
import pandas as pd
import h5py
import McsPy
import sys, importlib, os
import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

from time import strftime




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





inputdirectory  = r"/Users/naila/Documents/DATA/PREPROCESSED/trial_vicky"

os.chdir(inputdirectory)
probe_file="/Users/naila/Documents/GitHub/MEA_analysis/Spikesorting/MCS_MEA_256_100µM_spacing.prb"

filelist = glob.glob('*.h5')

filename = filelist[0]
# import the traces and use the numpy extractor




channel_raw_data = McsPy.McsData.RawData(filename)
analog_stream_0 = channel_raw_data.recordings[0].analog_streams[0]
stream = analog_stream_0
keylist = []
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
np_analog_list = list(np_analog_for_filter)
labels = 





numpyrecording = si.NumpyRecording(traces_list=np_analog_stream_0_data, 
                                   sampling_frequency=fs, 
                                   )

sorting = ss.run_spykingcircus(recording=numpyrecording  ,
                                 output_folder=inputdirectory,
                                 docker_image="spikeinterface/spyking-circus-base")


# import the recording and use spikeinterface spikesorting to extract the units

# filter units by quality metrics

# make spike dic with every channel

# import layerdic if available

# calculate firing rate
# fr per layer
# calculate bursts
# bursts per layer
# calculate network bursts
# calculate 
