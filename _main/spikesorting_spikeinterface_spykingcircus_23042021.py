# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:42:02 2021

@author: User
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
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

from time import strftime

filedirec = r"D:\Files_Reutlingen_Jenny\main_191021extra"
inputdirectory = r"D:\Files_Reutlingen_Jenny\main_191021extra\191021_extra"
os.chdir(inputdirectory)

filelist = glob.glob('*.h5')
filebase = filename.split('.')[0]


timestr = strftime("%d%m%Y")
outputdirectory = os.path.join(filedirec, '_output_Spikesorting_'+ timestr).replace('\\','/')

probe_file="C:/Users/User/Documents/JO/gitkraken/MEA_analysis/Spikesorting/MCS_MEA_256_100µM_spacing.prb"
outputdirectory_SC='D:/Files_Reutlingen_Jenny/main_191021extra/191021_extra_Spikesorting/output_Spykingcirucs'


outpath=os.path.join(outputdirectory+'_'+filename.split('.')[0]+'_spikesorting').replace("\\","/")
try:
    os.mkdir(outpath)
except OSError:
    print ("Creation of the directory %s failed" % outpath)
else:
    print ("Successfully created the directory %s " % outpath)
    


os.chdir(outpath)





''' _____________________FUNCTIONS______________________________________'''


def divide_recording_to_sub(recording, sublength_seconds):
    
    
    subrecording_dic = {}
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




def create_cache_for_subrecordings(subrecording_dic, filebase, outpath):

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
        
    return 'Finished dumping the subrecords. To load, use the load_dumped_recordings function'






def load_dumped_subrecordings(directory, filebase):

    subrecording_dic = {}
    filelist = glob.glob('*.pkl')
    for file in filelist:
        key = file.split(filebase)[1].split('.')[0].split('_recording')[0]
        subrecording_dic[key]=se.load_extractor_from_pickle(file)
        
    return subrecording_dic



def run_spykingcircus_on_sub(subrecording_dic, directory):
    

    sorted_dic={}
    for key in loaded:
        outpath_SC=os.path.join(outpath, 'sorted_'+str(key)).replace('\\', '/')
        '''
        try:
            os.mkdir(outpath_SC)
        except OSError:
            print ("Creation of the directory %s failed" % outpath)
        else:
            print ("Successfully created the directory %s " % outpath)
        '''
        sorted_dic[key]=ss.run_spykingcircus(
            loaded[key], output_folder=outpath_SC)

    return sorted_dic
    


def load_dumped_sorted_dic(outpath):

    sorted_dic = {}
    filelist = glob.glob('*sorted*')
    for file in filelist:
        key = file.split('sorted_')[1]
        sorted_dic[key]=se.SpykingCircusSortingExtractor(file)
        
    return sorted_dic
        
    
    




'''________________________WORKING_SCRIPT___________________________________'''


recording_MEA=se.MCSH5RecordingExtractor(
    os.path.join(inputdirectory, filename)
    )

recording_f = st.preprocessing.bandpass_filter(
    recording_MEA, freq_min=150, freq_max=5000, filter_type='butter', order=2
    )

recording_cmr = st.preprocessing.common_reference(
    recording_f, reference='median'
    )

recording_cmrprobe=recording_MEA.load_probe_file(
    probe_file="C:/Users/User/Documents/JO/gitkraken/MEA_analysis/Spikesorting/MCS_MEA_256_100µM_spacing.prb"
    )

subrecords = divide_recording_to_sub(recording_cmrprobe, 300)

create_cache_for_subrecordings(
    subrecording_dic=subrecords, filebase=filebase, outpath=outpath
    )

loaded = load_dumped_subrecordings(outpath, filebase)

sorted_dic = run_spykingcircus_on_sub(loaded, outpath)
sorted_dic = load_dumped_sorted_dic(outpath)
    