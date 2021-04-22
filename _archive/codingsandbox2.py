# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import bokeh.io
import bokeh.plotting
from bokeh.palettes import Spectral11
%matplotlib inline


direcfile = '/Users/jonas/Documents/Code/MEAS_Analysis/'
meafile = 'HCxA_Chr2_light_5V_100light_2000stop_position8.h5'
channelmap = 'channel_map_MEA256.txt'

/Users/jonas/Documents/Code/MEAS_Analysis/HCxA_Chr2_light_5V_100light_2000stop_position8.h5

channel_raw_data = McsPy.McsData.RawData(os.path.join(direcfile, meafile))
print(channel_raw_data.recordings)

channel_raw_data.recordings.keys()
print(channel_raw_data.recordings[0].analog_streams.keys())

analog_stream_0 = channel_raw_data.recordings[0].analog_streams[0]
analog_stream_0_data = analog_stream_0.channel_data
print(analog_stream_0_data)


# rearragne analog stream for plotting
np_analog_stream_0_data = np.transpose(analog_stream_0_data)
print("Old shape:", analog_stream_0_data.shape)
print("New shape:", np_analog_stream_0_data.shape)
print()
print(np_analog_stream_0_data)


# shape of data
np_analog_stream_0_data = np.transpose(channel_raw_data.recordings[0].analog_streams[0].channel_data)
print(np_analog_stream_0_data.shape)

# check for specific range of data, e.g. after stimulation
np_data_range = np_analog_stream_0_data[4500:4801, 4:8]
print(np_data_range)

#Triggers
analog_stream_1 = channel_raw_data.recordings[0].analog_streams[1]
analog_stream_1_data = analog_stream_1.channel_data






directory = '/Users/jonas/Documents/Code/MEAS_Analysis/HCxA_Chr2_light_5V_100light_2000stop_position8.h5'
filename = 'HCxA_Chr2_light_5V_100light_2000stop_position8.h5'
path= directory+'/'+filename

os.path.abspath(directory)

 threshold_bI=0.3         
 min_fr=0.5               
 binsize_robust=0.01
 threshold_robustness=0.3
 min_robustness=0.1


class MCRack():
    def __init__(self,path,min_fr,threshold_bI,threshold_robustness,min_robustness,position,folder,filename,binsize_robust, trigger_n=3, nr_channels=60):
        self.min_fr=min_fr
        self.position=position
        self.threshold_bI=threshold_bI
        self.threshold_robustness=threshold_robustness
        self.trigger_n=trigger_n
        self.filename=filename
        self.foler=folder
        self.nr_channels=nr_channels
        self.binsize_robust=binsize_robust
        f = h5py.File(path,'r')
        ticks = f['/Data/Recording_0/AnalogStream/Stream_1/InfoChannel'][()]["Tick"]*10**(-6) #Umrechnen frames in Zeit [s]
        self.tick=ticks[trigger_n]
        self.dset_tot = f['/Data/Recording_0/AnalogStream/Stream_0/ChannelData'][()] #raw data set 
        self.dset_trigger = f['/Data/Recording_0/AnalogStream/Stream_1/ChannelData'].value[()] #data set trigger
        self.channel_map=pd.read_csv("channel_map_MEA256.txt") 
        self.spikes=[]
        for i in range(nr_channels):
           self.dset = self.dset_tot[i]
           peaks= get_peaks(self.dset)[0]
           self.spikes.append(peaks)           
        for foo,peaks in enumerate(self.spikes):
            self.spikes[foo]=peaks*self.tick
            
        self.trigger_peaks,self.trigger_off, diff_trigger= find_triggers(self.dset_trigger,trigger_n, self.tick)
        f.close

        self.spikes_ON_total, self.spikes_Off_total=self._calc_onoff()
        self._calc_bias(threshold_bI,min_fr)
        self.cdict=self.channel_map["Channelname"].to_dict()
        self.firing_rate()
        self.robust_spikes(binsize_robust,threshold_robustness,min_robustness)


rackfile=MCRack(path=path, min_fr=min_fr,threshold_bI=threshold_bI,threshold_robustness=threshold_robustness,min_robustness=min_robustness,position=position,folder=folder, filename=filename,binsize_robust=binsize_robust 
                        , nr_channels=252, trigger_n=3)



def disjoint_groups(groups):
    """`groups` should be a list of sets"""
    groups = groups[:]  # copy, so as not to change original
    for group1 in groups:
        for group2 in groups:
            if group1 != group2:
                if group2.issubset(group1):
                    groups.remove(group2)
                elif group1.issubset(group2):
                    groups.remove(group1)
    return groups
