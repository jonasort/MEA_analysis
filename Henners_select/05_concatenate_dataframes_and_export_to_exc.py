# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:34:49 2022


@author: jonas ort, MD, department of neurosurgery, RWTH Aachen Medical Faculty
"""

'''
DIRECTORIES
'''


scriptdirectory = r"C:/Users/User/Documents/JO/gitkraken/MEA_analysis/Tübingen_Branch"

'''
EDIT HERE, HENNER

input directory = directory where the dataframes from 04 script are stored
output directory = directory where you want the the .xlsx files to be exported to
'''
input_directory = r"D:\Files_Reutlingen_Jenny\dataframes"
output_directory = r"D:\Files_Reutlingen_Jenny\dataframes"



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
from sklearn import preprocessing

from Butterworth_Filter import butter_bandpass, butter_bandpass_filter

import glob
from plot_signal_and_spikes import plot_signal_and_spikes_from_bandpassfilteredsignal
import time

from neo.core import AnalogSignal
import quantities as pq

from elephant.spectral import welch_psd
from elephant.current_source_density import estimate_csd
import pickle
from pathlib import Path


import qgrid
import plotly.express as px



'''
SCRIPT
'''


# this part loads all DATAFRAME
# get all available DATAFRRAME.pkl files
os.chdir(input_directory)
folderlist = glob.glob('*DATAFRAME*.pkl')


df_from_each_file = (pickle.load(open(Path(input_directory, filename), "rb")) for filename in folderlist)
df   = pd.concat(df_from_each_file, ignore_index=True)

filenamelist = []
cut = lambda x: x.split('DATAFRAME_')[1].split('.pkl')[0]
filenamelist = [cut(i) for i in folderlist]
filenamelist


'''
Dataframe cleaning
'''

# dichotomizing column is added whether at least one deviation 
df.loc[df['number_channels_with_lfp_down'] == 0, 'any_lfp_deviation'] = 0
df.loc[df['number_channels_with_lfp_up'] == 0, 'any_lfp_deviation'] = 0
df.loc[df['number_channels_with_lfp_down'] > 0, 'any_lfp_deviation'] = 1
df.loc[df['number_channels_with_lfp_down'] > 0, 'any_lfp_deviation'] = 1


# filter out artefacts by number of ative channels > 250
df = df.loc[(df['number_of_active_channels'] <= 250)]

# filter out channels with low active channels, i.e., < 10 % of totally active channels
# and at least five active channels
df = df.loc[(df['number_of_active_channels'] > 0.1*df['active_channels_whole_recording'])]
df = df.loc[(df['number_of_active_channels'] > 5)]

# get a clean meadium from the filename
df['medium'] = df['filename'].str.split('_').str[3]
df['drug'] = df['filename'].str.split('_').str[4]
df.loc[(df['medium'] == 'aCSF'), 'num_medium'] = 1
df.loc[(df['medium'] == 'hCSF'), 'num_medium'] = 2

# additional feature: ratio of bursting channels to active channels
df = df.assign(bursting_to_active_channels_ratio=lambda df: df.number_of_bursting_channels / df.number_of_active_channels)

# intra networkburst firing rate
df = df.assign(networkburst_firing_rate=lambda df: df.number_of_spikes/df.timelength_network_burst_s)

# getting the anatomy in percent
df = df.assign(layer1_percent_of_active_channels=lambda df: df.n_layer1_active_channels/df.number_of_active_channels)
df = df.assign(layer23_percent_of_active_channels=lambda df: df.n_layer23_active_channels/df.number_of_active_channels)
df = df.assign(layer4_percent_of_active_channels=lambda df: df.n_layer4_active_channels/df.number_of_active_channels)
df = df.assign(layer56_percent_of_active_channels=lambda df: df.n_layer56_active_channels/df.number_of_active_channels)
df = df.assign(whitematter_percent_of_active_channels=lambda df: df.n_whitematter_active_channels/df.number_of_active_channels)

# getting the anatomy in percent of all channels covered by the respective layer
df = df.assign(layer1_percent_of_covered_channels=lambda df: df.n_layer1_active_channels/df.channels_covered_layer1)
df = df.assign(layer23_percent_of_covered_channels=lambda df: df.n_layer23_active_channels/df.channels_covered_layer23)
df = df.assign(layer4_percent_of_covered_channels=lambda df: df.n_layer4_active_channels/df.channels_covered_layer4)
df = df.assign(layer56_percent_of_covered_channels=lambda df: df.n_layer56_active_channels/df.channels_covered_layer56)
df = df.assign(whitematter_percent_of_covered_channels=lambda df: df.n_whitematter_active_channels/df.channels_covered_whitematter)

# firing rate per layer for each networkburst
df = df.assign(layer1_firing_rate=lambda df: df.n_spikes_layer1/df.timelength_network_burst_s)
df = df.assign(layer23_firing_rate=lambda df: df.n_spikes_layer23/df.timelength_network_burst_s)
df = df.assign(layer4_firing_rate=lambda df: df.n_spikes_layer4/df.timelength_network_burst_s)
df = df.assign(layer56_firing_rate=lambda df: df.n_spikes_layer56/df.timelength_network_burst_s)
df = df.assign(whitematter_firing_rate=lambda df: df.n_spikes_whitematter/df.timelength_network_burst_s)

# number of channels with any lfp
df = df.assign(number_channels_with_any_lfp = lambda df: df.number_channels_with_lfp_up + df.number_channels_with_lfp_down)


# save all frames into an excel file
for i in filenamelist:
    saveframe = df.loc[(df['filename'] == i)]
    saveframe = saveframe.reset_index()
    saveframe.to_excel('overview_features_networkbursts_' + i + '.xlsx')


print('Your dataframes haves been exported as excel .xlsx files')