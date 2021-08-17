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

# main directory of the folder to analyse
filedirec = r"D:\MEA_DATA_Aachen\ANALYZED\20210514_cortex_div8"
# sub directory with the actual data
inputdirectory = r"D:\MEA_DATA_Aachen\PREPROCESSED\20210514_cortex_div8"

os.chdir(inputdirectory)






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




def create_cache_for_subrecordings(subrecording_dic, filebase, outpath):
  
    '''
    parameters: 
        subrecording_dic = dictionary with all subrecordings to be cached
        filebase = str, name of the based file
        outpath = directory where files will be directed
        
    returns: 
        print statement after function is finished, will dump and save
        the cached as .pkl in outpath
    '''
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
   
    '''
    parameters:
        diretory = where subrecordings are saved
        filebase = namebase of the recording to be loaded
        
    returns:
        a dictionary with keys=dictkeys ('sec_xxx-xxx'), values=subrecordings
        extractors
    '''
    beforedirectory = os.getcwd()
    os.chdir(directory)
    subrecording_dic = {}
    filelist = glob.glob('*.pkl')
    for file in filelist:
        key = file.split(filebase)[1].split('.')[0].split('_recording')[0]
        subrecording_dic[key]=se.load_extractor_from_pickle(file)
    
    os.chdir(beforedirectory)
    return subrecording_dic



def run_spykingcircus_on_sub(subrecording_dic, directory):
    
    '''
    parameters:
        subrecording dic = a dictionary with keys=dictkeys ('sec_xxx-xxx'), 
        values=subrecordings
        directory = path where subrecordings are saved
        
    returns:
        a dictionary with key = dictkeys ('sec_xxx-xxx'), values = sortings

    '''
    sorted_dic={}
    for key in subrecording_dic:
        outpath_SC=os.path.join(outpath, 'sorted_'+str(key)).replace('\\', '/')
        '''
        try:
            os.mkdir(outpath_SC)
        except OSError:
            print ("Creation of the directory %s failed" % outpath)
        else:
            print ("Successfully created the directory %s " % outpath)
        '''
        # sorted_dic[key]=ss.run_spykingcircus(
        #     subrecording_dic[key], output_folder=outpath_SC
        #     )
        sorted_dic[key]=ss.run_sorter('spykingcircus',
            subrecording_dic[key], output_folder=outpath_SC
            )
        

    return sorted_dic
    


def load_dumped_sorted_dic(outpath):

    '''
    parameters: 
        directory where sortings of spyking circus are saved
        
    returns:
        dictionary with key = dictkeys ('sec_xxx-xxx'), values = sortings
    '''
    sorted_dic = {}
    filelist = glob.glob('*sorted*')
    for file in filelist:
        key = file.split('sorted_')[1]
        sorted_dic[key]=se.SpykingCircusSortingExtractor(file)
        
    return sorted_dic
        
    
    




'''________________________WORKING_SCRIPT___________________________________'''


# create the filelist of all .h5 files
filelist = glob.glob('*.h5')


for i in filelist:
    filename = i
    print('Working on %s' %filename)


    filebase = filename.split('__')[1]
    
    # for overview when the analysis was performed: create a timestring
    timestr = strftime("%d%m%Y")
    outputdirectory = os.path.join(filedirec, '_output_Spikesorting_'+ timestr).replace('\\','/')
    
    probe_file="C:/Users/User/Documents/JO/gitkraken/MEA_analysis/Spikesorting/MCS_MEA_256_100µM_spacing.prb"
    #outputdirectory_SC='D:/Files_Reutlingen_Jenny/main_191021extra/191021_extra_Spikesorting/output_Spykingcirucs'
    
    # one outpath is created for every datafile
    outpath=os.path.join(outputdirectory+'_'+filename.split('__')[1]+'_spikesorting').replace("\\","/")
    try:
        os.mkdir(outpath)
    except OSError:
        print ("Creation of the directory %s failed" % outpath)
    else:
        print ("Successfully created the directory %s " % outpath)
        
    
    
    os.chdir(outpath)
    
    
    '''
    1. Create subrecordings, Caches, run the spikesorter
    '''
    
    # load in the recordings from the .h5 file
    recording_MEA=se.MCSH5RecordingExtractor(
        os.path.join(inputdirectory, filename), stream_id=0)
    
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
        probe_file="C:/Users/User/Documents/JO/gitkraken/MEA_analysis/Spikesorting/MCS_MEA_256_100µM_spacing.prb")
    
    # divide the recording into subrecords, subrecords is a dictionary
    subrecords = divide_recording_to_sub(recording_cmrprobe, 130)
    
    # create the cache for the subrecordings
    create_cache_for_subrecordings(
        subrecording_dic=subrecords, filebase=filebase, outpath=outpath)
    
    loaded = load_dumped_subrecordings(outpath, filebase)
    
    sorted_dic = run_spykingcircus_on_sub(loaded, outpath)
    
    # the dic can be loaded 
    sorted_dic = load_dumped_sorted_dic(outpath)
    
print('Finished the sorting-process.')
    
