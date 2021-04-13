# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:56:17 2021

@author: jonas ort, MD, department of neurosurgery RWTH Aachen 
"""

#import spikeinterface modules
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import numpy as np
import glob

#import everything else
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

filedirec = r"D:/Files_Reutlingen_Jenny/191021_hdf5"
outputdirectory_HS='D:/Files_Reutlingen_Jenny/191021_extra_Spikesorting/output_HS'
outputdirectory_HS_phy='D:/Files_Reutlingen_Jenny/191021_extra_Spikesorting/output_HS_exportphy'
probe_file="C:/Users/User/Documents/JO/gitkraken/MEA_analysis/Spikesorting/MCS_MEA_256_100µM_spacing.prb"


os.chdir(filedirec)

filelist = glob.glob('*.h5')
filename=filelist[0]


#load the recordings and apply the filter
recording_MEA=se.MCSH5RecordingExtractor(os.path.join(filedirec, filename))
recording_f = st.preprocessing.bandpass_filter(recording_MEA, freq_min=150, freq_max=5000, filter_type='butter', order=2)
recording_cmr = st.preprocessing.common_reference(recording_f, reference='median')


#prints general information
print('Num. channels = {}'.format(len(recording_MEA.get_channel_ids())))
print('Sampling frequency = {} Hz'.format(recording_MEA.get_sampling_frequency()))
print('Num. timepoints = {}'.format(recording_MEA.get_num_frames()))
print('Stdev. on third channel = {}'.format(np.std(recording_MEA.get_traces(channel_ids=2))))
print('Location of third electrode = {}'.format(recording_MEA.get_channel_property(channel_id=2, property_name='location')))


# load the probe file
recording_cmrprobe=recording_MEA.load_probe_file(probe_file=probe_file)


# check probe file locations and plot for control
recording_cmrprobe.get_channel_locations()
sw.plot_electrode_geometry(recording_cmrprobe, color='C0', label_color='r', figure=None, ax=None)


# check if Herdingspikes is within the installed spikesorters
print(ss.installed_sorters())
ss.get_default_params("herdingspikes")

#HerdingSpikes spike sorting
sorting_HS = ss.run_herdingspikes(recording_cmrprobe, output_folder=outputdirectory_HS)
sorting_curated = st.curation.threshold_num_spikes(sorting=sorting_HS, threshold=10, threshold_sign='less')
st.postprocessing.export_to_phy(recording=recording_cmrprobe, sorting=sorting_curated, output_folder=outputdirectory_HS_phy)
print('Units found with Herding Spikes:', sorting_SC.get_unit_ids())

