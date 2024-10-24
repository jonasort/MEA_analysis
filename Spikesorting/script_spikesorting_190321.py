# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:57:54 2021

@author: jonas ort, department of neurosurgery, rwth aachen university


Script for Spikeinterface based Spikesorting of 256 MEA recordings using 
SpykingCircus as Sorter + Exprot to Phy. 

ADJUSTMENTS:
    - use
    
DEPENDENCIES:
    - see imports
    - SpykingCircus must be installed

INPUT:
    - .h5 File of a MEA recording as generated by the MCS Data Manager, can be
      be arranged in one folder

OUTPUT:
    - Folder for SC Data, including files
    - Folder for Phy Export, including files
    

NOTA BENE:
    - The Output volume is quite huge (approximately factor 9 of the input). For
      1GB of original Recording data in total around 9GB output will follow
    
ACKNOWLEDGEMENTS:
    - Developers of SpykingCircus, Spikeinterface and Phy



OVERVIEW:
    1. Directories and Inputs
    2. Funcions
    3. Working Script


"""


#import spikeinterface modules
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw


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
import numpy as np
import glob
import time



# define inputdirectory and 
inputdirectory = "D:/Files_Reutlingen_Jenny/191023"
probe_file="C:/Users/User/Documents/JO/gitkraken/MEA_analysis/Spikesorting/MCS_MEA_256_100µM_spacing.prb"

timestr = time.strftime("%d%m%Y")
outputdirectory=inputdirectory+('_output_spikesorting_%s' % timestr)



try:
    os.mkdir(outputdirectory)
except OSError:
    print("Creation of the directory %s failed" % outputdirectory)
else:
    print("Successfully created the directory %s " % outputdirectory)

os.chdir(inputdirectory)


filelist = glob.glob('spon*.h5')

for file in filelist:
    os.chdir(inputdirectory)
    filename=file
    filenamebase=file.split('.')[0]
    str_outputdirectory_SC='SC_'+filenamebase
    str_outputdirectory_Phy='Phy_'+filenamebase
    
    print('Working on '+filename)
    
    # load the recording
    recording_MEA=se.MCSH5RecordingExtractor(os.path.join(inputdirectory, filename))
    recording_f = st.preprocessing.bandpass_filter(recording_MEA, freq_min=150, freq_max=5000, filter_type='butter', order=2)
    recording_cmr = st.preprocessing.common_reference(recording_f, reference='median')
    
    print('Num. channels = {}'.format(len(recording_MEA.get_channel_ids())))
    print('Sampling frequency = {} Hz'.format(recording_MEA.get_sampling_frequency()))
    print('Num. timepoints = {}'.format(recording_MEA.get_num_frames()))
    print('Stdev. on third channel = {}'.format(np.std(recording_MEA.get_traces(channel_ids=2))))
    print('Location of third electrode = {}'.format(recording_MEA.get_channel_property(channel_id=2, property_name='location')))


    recording_cmrprobe=recording_cmr.load_probe_file(probe_file)
    recording_cmrprobe.get_channel_locations()
    
    os.chdir(outputdirectory)
    try:
        os.mkdir(str_outputdirectory_SC)
    except OSError:
        print("Creation of the directory %s failed" % str_outputdirectory_SC)
    else:
        print("Successfully created the directory %s " % str_outputdirectory_SC)
        
    try:
        os.mkdir(str_outputdirectory_Phy)
    except OSError:
        print("Creation of the directory %s failed" % str_outputdirectory_Phy)
    else:
        print("Successfully created the directory %s " % str_outputdirectory_Phy)
    

    sorting_SC = ss.run_spykingcircus(recording_cmrprobe, output_folder=str_outputdirectory_SC)
    sorting_curated = st.curation.threshold_num_spikes(sorting=sorting_SC, threshold=10, threshold_sign='less')
    st.postprocessing.export_to_phy(recording=recording_cmrprobe, sorting=sorting_curated, output_folder=str_outputdirectory_Phy)


    print('Units found with Spyking Circus:', sorting_SC.get_unit_ids())
    
    
print('Job done.')















