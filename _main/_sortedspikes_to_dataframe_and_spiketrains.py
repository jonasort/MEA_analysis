# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:43:16 2021

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
import itertools

from time import strftime





'''____________________________PATHS______________________________________'''

mainpath = r'D:\MEA_DATA_Aachen\ANALYZED\20210510_cortex_div4'
outpath = r'D:\MEA_DATA_Aachen\ANALYZED\20210510_cortex_div4\_output_Spikesorting_07062021_cortex_div4_hCSF_ID039_nodrug_spont_1_spikesorting'




labeldic = np.load(
    r'C:\Users\User\Documents\JO\gitkraken\MEA_analysis\basic_inputs\labeldictionary_MCS_256MEA.npy', 
    allow_pickle='TRUE').item()



'''____________________________FUNCTIONS___________________________________'''

# loading functions

def load_dumped_subrecordings(directory, filebase):

    os.chdir(directory)
    subrecording_dic = {}
    filelist = glob.glob('*recording.pkl')
    for file in filelist:
        key = file.split(filebase)[1].split('.')[0].split('_recording')[0]
        subrecording_dic[key]=se.load_extractor_from_pickle(file)
        
    return subrecording_dic
        
    
def load_dumped_sorted_dic(outpath):

    sorted_dic = {}
    filelist = glob.glob('*sorted*')
    for file in filelist:
        key = file.split('sorted_')[1]
        sorted_dic[key]=se.SpykingCircusSortingExtractor(file)
        
    return sorted_dic


# function to get the filebase from folder
def get_filebase_from_foldername(mainpath_folder):
    
    separator = '_'
    filebase = separator.join(mainpath_folder.split('\\')[-1].split('_')[4:-1])
    
    return filebase




def get_dictionary_keys(outpath_folder):

    dictkeys = []
    os.chdir(outpath_folder)
    folders_sorting = glob.glob('*sorted*')
    for i in folders_sorting:
        dk = i.split('sorted_')[1]
        dictkeys.append(dk)
    
    return dictkeys, print('Dictkeys are %s. Your CWD may have changed. Please check.' %dictkeys)



# function to convert basic information to dataframe
def units_to_pandas_DataFrame(sorted_dic, recording_cache, dictkey, 
                              layerdic_invert, recordingdate='not given'):
    
    # creates pandas DataFrame only including the unit ids. 
    # the order will be confused because of the unit 
    #numbering ('1' instead of '001')
    
    unit_ids = sorted_dic[dictkey].get_unit_ids()
    
    # check for this sorted statement!!!
    
    unitframe = pd.DataFrame(
        sorted_dic[dictkey].get_unit_ids(), 
        columns=['unit_ids']
        )
    
    
    list_not_empty_spiketrains = []
    for i in unit_ids:
        st_len = len(sorted_dic[dictkey].get_unit_spike_train(unit_id=i))
        if st_len > 0:
            list_not_empty_spiketrains.append(i)
    
    
    # calculate as many paramters as possible outside of the loop
    recordings_seconds = recording_cache[dictkey].get_num_frames()/recording_cache[dictkey].get_sampling_frequency()
    
    
    try:
        features = st.postprocessing.compute_unit_template_features(
            recording_cache[dictkey], 
            sorted_dic[dictkey], 
            as_dataframe=True,
            memmap = False
            )
    except ValueError:
        print ("Quality Metrics not raised an error")
        feature_error = 1
      
    else:
        feature_error = 0
        
        
        
    try:
        quality_metrics = st.validation.compute_quality_metrics(
            sorted_dic[dictkey], 
            recording_cache[dictkey], 
            metric_names=['firing_rate', 'isi_violation', 'snr', 
                          'amplitude_cutoff', 'presence_ratio'],
            as_dataframe=True,
            memmap = False
            )
        
    except ValueError:
        print ("Quality Metrics not raised an error")
        qm_error = 1
      
    else:
        qm_error = 0
    
    
    
    for i in list_not_empty_spiketrains:  
        
        unitframe.loc[(unitframe['unit_ids']==i), 'unit_index']=unitframe.loc[(
            unitframe['unit_ids']==i)].index
        
        
        # add channel with maximum amplitude
        unitframe.loc[(unitframe['unit_ids']==i), 'max_channel']=st.postprocessing.get_unit_max_channels(
            recording_cache[dictkey], sorted_dic[dictkey], unit_ids=[i]
                                                                                                        )
    
        # get the channel label as on MCS MEA 256 chips
        max_channel = int(unitframe.loc[(unitframe['unit_ids']==i)]['max_channel'])
        unitframe.loc[(unitframe['unit_ids']==i), 'channellabel']= labeldic[max_channel]
        
        # get the number of spiks per unit
        unitframe.loc[(unitframe['unit_ids']==i), 'n_spikes']=len(
            sorted_dic[dictkey].get_unit_spike_train(unit_id=i))
        
        # add firing rate
        unitframe.loc[(unitframe['unit_ids']==i), 'firing_rate']= unitframe.loc[(
            unitframe['unit_ids']==i)]['n_spikes']/recordings_seconds
        
        # add layer
        labelkey = unitframe.loc[(unitframe['unit_ids']==i), 'channellabel'].values[0]
        unitframe.loc[(unitframe['unit_ids']==i), 'layer']= layerdic_invert[labelkey]
        
        # add features
        if feature_error == 0:
            unitframe.loc[(unitframe['unit_ids']==i), 'ft_peak_to_valley']=features.loc[i]['peak_to_valley']
            unitframe.loc[(unitframe['unit_ids']==i), 'ft_halfwidth']=features.loc[i]['halfwidth']
            unitframe.loc[(unitframe['unit_ids']==i), 'ft_peak_trough_ratio']=features.loc[i]['peak_trough_ratio']
            unitframe.loc[(unitframe['unit_ids']==i), 'ft_repolarization_slope']=features.loc[i]['repolarization_slope']
            unitframe.loc[(unitframe['unit_ids']==i), 'ft_recovery_slope']=features.loc[i]['recovery_slope']

        # add quality metrics
        if qm_error == 0:
            unitframe.loc[(unitframe['unit_ids']==i), 'qm_firing_rate']=quality_metrics.loc[i]['firing_rate']
            unitframe.loc[(unitframe['unit_ids']==i), 'qm_isi_violation']=quality_metrics.loc[i]['isi_violation']
            unitframe.loc[(unitframe['unit_ids']==i), 'qm_amplitude_cutoff']=quality_metrics.loc[i]['amplitude_cutoff']
            unitframe.loc[(unitframe['unit_ids']==i), 'qm_presence_ratio']=quality_metrics.loc[i]['presence_ratio']
            unitframe.loc[(unitframe['unit_ids']==i), 'qm_snr']=quality_metrics.loc[i]['snr']
            
        
        unitframe.loc[(unitframe['unit_ids']==i), 'file']=filebase
        unitframe.loc[(unitframe['unit_ids']==i), 'subrecording']=dictkey
        unitframe.loc[(unitframe['unit_ids']==i), 'recordingdate']=recordingdate
        # verify if this line works
        #unitframe.loc[(unitframe['unit_ids']==i), 'medium']=unitframe.loc[(unitframe['unit_ids']==i)]['file'].split('_')[0]
    return unitframe




def invert_layerdic(layer_dic):
    
    '''
    Expects a dictionary with key = layer, value = list of channellabels
    
    Returns a dictionary with key = channellabels, value = layer
    '''
    layerdic_invert = {}

    for key in layerdic:
        for i in layerdic[key]:
            layerdic_invert[i]=key
            
            
    return layerdic_invert


def create_complete_DataFrame(frame_dic):
    '''
    Expects: a Dictionary with keys = dictkeys, e.g. 'sec_0-300' and val = DF
    
    Returns: A complete DF from all frames contained in the given frame_dictionary
    
    '''
    
    
    completeframe = frame_dic[list(frame_dic.keys())[0]]
    number_of_keys = len(list(frame_dic.keys()))
    
    for i in range(1, number_of_keys):
        completeframe = completeframe.append(frame_dic[list(frame_dic.keys())[i]])
        
    return completeframe



# for single subrecordings
def spiketrains_to_spikedictionary_channel_subrecording(sorting_dic, unitframe, dictkey):
    
    
    '''
    Expects the Sorting Dic, a basic Dataframe and the dictkey
    
    Returns
        1. spikedictionary_channel -> the spiketrains per channel
        2. spikedictionar_neuron -> spiketrain per sorted neuron
        3. spikedictionary_channel_neuron -> nested dictionary
    '''
    
    labels = unitframe['channellabel'].unique()
    unit_ids = sorted_dic[dictkey].get_unit_ids()
    spikechannellist = []
    spikedictionary_channel = {}
    spikedictionary_neuron = {}
    spikedictionary_channel_neuron = {}
    sub_spikedictionary_channel_neuron ={}
    
    for label in labels:
        sub_spikedictionary_channel_neuron ={}
        spikechannellist = []
        unit_i = list(unitframe.loc[(unitframe['channellabel']==label)]['unit_ids'])
        for i in unit_i:
            spiketrains = sorted_dic[dictkey].get_unit_spike_train(i)
            spikechannellist.append(list(spiketrains))
            spikedictionary_neuron[i]=spiketrains
            sub_spikedictionary_channel_neuron[i]=spiketrains
        spikechannellist = sorted(list(itertools.chain.from_iterable(spikechannellist)))
        spikedictionary_channel[label]=spikechannellist
        spikedictionary_channel_neuron[label]=sub_spikedictionary_channel_neuron
        
    return spikedictionary_channel, spikedictionary_neuron, spikedictionary_channel_neuron


'''_____________________________SCRIPT______________________________________'''


os.chdir(mainpath)
folderlist = glob.glob('*_output_Spikesorting*spikesorting*')


# get a filebaselist for every folder in the mainpath, that we we can filter 

filebase_list = []
for i in folderlist:
    filebase = get_filebase_from_foldername(i)
    filebase_list.append(filebase)
    
#change path to folder list subfolder (i.e., outpath)
os.chdir(outpath)


# within outpath, get the different keys that comprise our spikesorting dicitonary
dictionary_keys = get_dictionary_keys(outpath)

filebase = filebase_list[0]

# in function 
loaded = load_dumped_subrecordings(outpath, filebase)
sorted_dic = load_dumped_sorted_dic(outpath)

# create a dictionary with keys = dictkeys (e.g., 'sec_0-300'), returns
#to oragnize the dataframes we first use a dictionary


frame_dic = {}

recdate = mainpath.split('\\')[3].split('_')[0]


'''
Load the data and dump it towards a pd.DF

'''

# give the layerdic
layerdic = {'layer1':[], 
            'layer2-3':['L1', 'M1', 'M2', 'M3', 'M15', 'M16', 'N1', 'N2', 'N3',
                        'N4', 'N5', 'N6', 'N7','N8', 'N9', 'N10', 'N11', 'N12',
                        'N13', 'N14', 'N15', 'N16', 'O1', 'O2', 'O3', 'O4', 
                        'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 
                        'O13', 'O14', 'O15', 'O16', 'P1', 'P2', 'P3', 'P4', 
                        'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 
                        'P13', 'P14', 'P15', 'P16', 'R2', 'R3', 'R4', 'R5', 
                        'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 
                        'R13', 'R14', 'R15'],
           'layer4':['K1', 'K2', 'K14', 'K15', 'K16', 'L2', 'L3', 'L4', 'L5', 
                     'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14',
                     'L15', 'L16', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 
                     'M11', 'M12', 'M13', 'M14'],
           'layer5-6':['K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10', 'K11', 
                       'K12', 'K13', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 
                       'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 
                       'I16', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 
                       'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 
                      'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 
                      'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16',
                      'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 
                      'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16',
                       'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 
                       'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16',
                       'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 
                       'D10', 'D11'],
           'whitematter':['D12', 'D13', 'D14', 'D15', 'D16', 'C1', 'C2', 'C3', 
                          'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 
                          'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'B1', 'B2', 
                          'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 
                          'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 
                          'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 
                          'A10', 'A11', 'A12', 'A13', 'A14', 'A15' ]}

inverted_layerdic = invert_layerdic(layerdic)

for key in sorted_dic:
    frame_dic[key] = units_to_pandas_DataFrame(
                        sorted_dic=sorted_dic,
                        recording_cache=loaded,
                        dictkey=key,
                        layerdic_invert=inverted_layerdic,
                        recordingdate=recdate)
    
# the frame dictionary is saved, from here, the dataframes can easily be 
# loaded again
np.save(filebase+'_DataFrame_dictionary.npy', frame_dic)


# load the framedic
frame_dic = np.load(filebase+'_DataFrame_dictionary.npy', allow_pickle='TRUE').item()

# runs the function and creates the completeframe. for 17.10.2019 the frame is already completed including the 
completeframe = create_complete_DataFrame(frame_dic=frame_dic)

#save the completeframe
completeframe.to_pickle(str('DF_' + filebase + '_' + 'complete') + '_df.pkl')

# load the completeframe
completeframe = pd.read_pickle(str('DF_' + filebase + '_' + 'complete') + '_df.pkl')

'''
Create the Spiketrains

'''

u = dictionary_keys[0][0]

st_channel, st_neurons, st_channels_neurons = spiketrains_to_spikedictionary_channel_subrecording(
    sorting_dic = sorted_dic,
    unitframe = completeframe,
    dictkey=u)



# save spiketrains

np.save(filebase+'_spiketrains_per_channel.npy', st_channel)
np.save(filebase+'_spiketrains_per_unit.npy', st_neurons)
np.save(filebase+'_spiketrains_nested_per_channel_per_unit.npy', st_channels_neurons)


# load spiketrains

# check if the spiketrains are somewhat filtered
st_channel = np.load(filebase+'_spiketrains_per_channel.npy', allow_pickle='TRUE').item()
st_neurons = np.load(filebase+'_spiketrains_per_unit.npy', allow_pickle='TRUE').item()
st_channels_neurons = np.load(filebase+'_spiketrains_nested_per_channel_per_unit.npy', allow_pickle='TRUE').item()




















































