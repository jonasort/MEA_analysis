# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:42:02 2021

@author: jonas ort, department of neurosurgery, RWTH AACHEN, medical faculty
"""

#import spikeinterface modules
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw


#import everything else

import glob
import os
import sys
import numpy as np

from time import strftime






''' _____________________FUNCTIONS______________________________________'''


def divide_recording_to_sub(recording, sublength_seconds):
  
    '''
    parameters: recording = recording extractor
                sublength_seconds = int, how long shall the subrecordings be
                
    returns: dictionary with key=str: sec_xxx-xxx, value=subrecording extractor
    '''
    
    
    subrecording_dic = {}
    fs = recording.get_sampling_frequency()
    recording_length = recording.get_num_frames()
    recording_seconds = recording_length/fs
    end_frame = int(recording_seconds)
    
    for snippet in range(0, end_frame, sublength_seconds):
        sub_start = snippet
        sub_end = snippet + sublength_seconds
        if sub_end > end_frame:
            sub_end = end_frame
        sub_str = 'sec_'+str(sub_start)+'-'+str(sub_end)
    
        subrecording_dic[sub_str] = se.SubRecordingExtractor(
            recording_cmrprobe, start_frame = sub_start*fs,
            end_frame = sub_end*fs)
    
    return subrecording_dic



    




'''________________________WORKING_SCRIPT_______'____________________________'''



''
'''
1. Create subrecordings, Caches, run the spikesorter
'''

# load in the recordings from the .h5 file
recording_MEA=se.MCSH5RecordingExtractor(snakemake.input[0], stream_id=0)

# bandpassfilter the recording
recording_f = st.preprocessing.bandpass_filter(
    recording_MEA, freq_min=150, freq_max=4500, filter_type='butter', order=2)


#remove bad channels automatically
recording_removed_bad = st.preprocessing.remove_bad_channels(
    recording_MEA, seconds = 30)




# common reference
recording_cmr = st.preprocessing.common_reference(
    recording_removed_bad, reference='median')

# load the probe file
recording_cmrprobe=recording_cmr.load_probe_file(
    probe_file=snakemake.input[1])

# divide the recording into subrecords, subrecords is a dictionary
subrecording_dic = divide_recording_to_sub(recording_cmrprobe, 300)


for key in subrecording_dic:
  sub_cache = se.CacheRecordingExtractor(
      subrecording_dic[key])
  filepath = os.path.join(
      outpath, filebase+str(key)+'_filtered_data.dat'
      ).replace('\\','/')
  sub_cache.move_to(filepath) 
  sub_cache.dump_to_dict()
  filepathpickle = os.path.join(
      outpath, filebase+str(key)+'_recording.pkl'
      ).replace('\\','/')
  sub_cache.dump_to_pickle(filepathpickle)
# create the cache for the subrecordings
create_cache_for_subrecordings(
    subrecording_dic=subrecords, filebase=filebase, outpath=outpath)


    
