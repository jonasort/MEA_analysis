# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:09:05 2021

@author: jonas ort, MD rwth aachen university, department of neurosurgery
"""

import os
import glob
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
import copy
import pickle


# this directory for the extracted spikes and lfp dictionaries
working_directory = r"D:/Files_Reutlingen_Jenny/19-04-15/190415_paper/spike_extraction"

# second directory/recording where the rest of analysis will be stored
output_directory = r"D:/Files_Reutlingen_Jenny/19-04-15/190415_paper"


# for reutlingen files: manually correct the filename, medium, recording date
# cave: the filename is essential to only grab the correct folders later
filename= '190415_03_Cortex-synChR2-A_aCSF_rewashafterhcsf'
medium = 'aCSF'
recordingdate = '2019-04-15'


# change to the working_directory
os.chdir(working_directory)




'''

Functions

'''

def get_isi_single_channel(spikedic):
    
    '''
    input: 
        spikedic with keys = channellabels, values = spiketimes in raw ticks
    
    
    returns: 

        dictionary with keys = channellabels, values = isi per channel in miliseconds
        
        
    nota bene:
        the amount of spikes is not filtered, we still need to factor out non relevant channels
    
    '''
    
    # set the empty dictionary and temporary list
    isi_dictionary = {}
    isi_temp_list =[]
    
    
    for key in spikedic:
        isi_temp_list =[]
        spikes = spikedic[key]
        spikes = [spike * tick * scale_factor_for_milisecond for spike in spikes]
        
        if len(spikes) >= 2:
            for i in range(0, len(spikes)-1): 

                # calculate the isi
                isi =  spikes[i+1] - spikes[i] 
                isi_temp_list.append(isi)

        isi_dictionary[key] = isi_temp_list
        
    
    return isi_dictionary


def gaussian_smoothing(y, window_size=10, sigma=2):

    filt = signal.gaussian(window_size, sigma)

    return signal.convolve(y, filt, mode='full')





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




'''

ACTUAL SCRIPT

'''



# for aachen files: write loop to get the above info automatically
'''
TO DO WRITE AACHEN LOOP
'''


# now a data structure is created where we can store all necessary information
# i.e., it is a dicionary of dictionaries that will be pickled

Basics = {}
Infos_Recording = {}
Infos_Analysis = {}
Infos_Anatomy = {}
main_recording_dictionary ={}


Infos_Recording['medium']=medium


# the folderlist will contain all 120second long subfolders
# the filename is 
folderlist = glob.glob(filename+'*')




# get into every folder and find the dictionaries
# replace them in a two meta-dictionaries (infodics and spikedics)
infodics = {}
spikedics = {}

for folder in folderlist:
    os.chdir(os.path.join(working_directory, folder))
    # cave: here the slicing needs to be adjusted dependent on reutlingen filenames
    timekey = folder.split('_')[6:9]
    timekey = '_'.join(timekey)
    
    # load the info_dic_file
    info_dic_filename = glob.glob('*info*npy')
    print(info_dic_filename)
    print(os.getcwd())
    info_dic = np.load(info_dic_filename[0], allow_pickle=True).item()
    infodics[timekey] = info_dic
    
    # load the spikedic_file
    spike_dic_filename = glob.glob('*spikes_MAD*')[0]
    spikedic_MAD = np.load(spike_dic_filename, allow_pickle=True).item()
    spikedics[timekey] = spikedic_MAD


# separately save all infodics
np.save(os.path.join(output_directory, 'infodics_'+filename+'.npy'), infodics)

# get the first of all infodics
first_info_dic_key = list(infodics.keys())[0]
infodic = infodics[first_info_dic_key]

'''
ADD the info_dics to our pickle data
'''

Infos_Recording['info_dics_subrecordings'] = infodics
Infos_Recording['recordings_date'] = recordingdate
Infos_Recording['timelengthrecording_s'] = infodic['timelengthrecording_s']


# the parameter infodic is available through our loop
# it contains the information of the last inofdic we loaded
tick = infodic['tick']
first_recording_timepoint = infodic['first_recording_timepoint']
scale_factor_for_second = infodic['scale_factor_for_second']
timelengthrecording_s = infodic['timelengthrecording_s']

# we attach them in the first level of the Infos_Recording to 
# have faster access to it
Infos_Recording['scale_factor_for_second'] = scale_factor_for_second
Infos_Recording['tick'] = tick





'''
JOIN subdivided spikedics to the full spikedic

nb: the spike dics contain all spikes in the original tick data points
the are continuing meaning that for a spikedic starting at 600 seconds of the
recordings, the start is not zero but form 600 already. thus, they can simply be 
concatenated.
'''

timekeys = list(spikedics.keys())
channelkeys = list(spikedics[timekeys[0]].keys())


# we now need to use a double loop to get all dictionary keys and join them into a big full recording dictionary
spikedic_MAD_full = {}
temp_spikelist = []

for i in channelkeys:
    temp_spikelist = []
    for j in timekeys:
        spikes = list(spikedics[j][i])
        temp_spikelist.append(spikes)
    
    #join the lists
    temp_spikelista = sum(temp_spikelist, [])
    #remove the duplicates
    temp_spikelistb = list(set(temp_spikelista))
    
    #sort the list
    temp_spikelistc = sorted(temp_spikelistb)
    
    #assign them to their channel in the full dictionary
    spikedic_MAD_full[i] = temp_spikelistc


# join the spikedic to the main_recording dictionary
spikedic_MAD = spikedic_MAD_full
main_recording_dictionary['spikedic_MAD'] = spikedic_MAD

# and save it separately
np.save(os.path.join(output_directory, filename +'_full_spikedic.npy'), spikedic_MAD_full)


# relevant factor: minimal amount of spikes to be relevant
# create an array of the spikes in scale of seconds
active_channels = 0
spikedic_seconds = {}
for key in spikedic_MAD:
    relevant_factor = timelengthrecording_s*0.05
    if len(spikedic_MAD[key])>relevant_factor:
        sec_array = np.asarray(spikedic_MAD[key])*tick*scale_factor_for_second
        spikedic_seconds[key]=sec_array
        active_channels += 1
spikearray_seconds = np.asarray(list(spikedic_seconds.values()))  

# add them to the sub dictionaries
Basics['active_channels'] = active_channels
Basics['relevant_factor'] = relevant_factor


# get a 1-D array with every detected spike
scale_factor_for_milisecond = 1e-03
full_spike_list = []
full_spike_list_seconds = []
for key in spikedic_MAD:
    if len(spikedic_MAD[key])>relevant_factor:
        x = list(np.asarray(spikedic_MAD[key])*scale_factor_for_milisecond*tick)
        full_spike_list = full_spike_list + x

        xs = list(np.asarray(spikedic_MAD[key])*scale_factor_for_second*tick)
        full_spike_list_seconds = full_spike_list_seconds + xs
full_spikes = sorted(full_spike_list)
full_spikes_seconds = sorted(full_spike_list_seconds)


# calculate the mean firing rate for the whole recording
mean_fr_whole_recording = np.around(
    (len(full_spikes_seconds) / timelengthrecording_s), 3)

# add them to the sub dictionaries
Basics['mean_fr_whole_recording'] = mean_fr_whole_recording




'''
NETWORK BURSTING ACTIVITY

For the whole concatenated recording, we now define the networkburst
using the mean firing rate.

How?
We use gaussian smoothing, and a moving average over that smoothing.
From this moving average we calculate the mean.

A network burst is defined as the mean of that moving average + one standard
deviation.

'''


# define bins 
binsize = 0.005 #seconds
bins= np.arange(0, timelengthrecording_s+binsize, binsize)

# make a histogram 
full_spikes_binned = np.histogram(full_spikes_seconds, bins)[0]


#trial of population burst plot as inspired by Andrea Corna
bins = int(timelengthrecording_s / binsize)+1

#conversion to hertz
firing_rate_histogram = np.histogram(full_spikes_seconds, bins=bins)
firing_rate = firing_rate_histogram[0]*200 



# sliding window of the moving average
N = int(1/binsize) 

# gaussian smmothing fo the firing rate and moving average
fr_gau = gaussian_smoothing(firing_rate)


ma_fr_gau = np.convolve(fr_gau, np.ones(N)/N, mode='full')

# we look for the mean of the MA as threshold
# we arrange this mean in an array for plotting
mean_ma_fr_gau = np.mean(ma_fr_gau)
std_ma_fr_gau = np.std(ma_fr_gau)
network_burst_threshold = mean_ma_fr_gau #+ 1*std_ma_fr_gau
shape_for_threshold = np.shape(ma_fr_gau)
network_burst_threshold_array = np.full(shape_for_threshold, network_burst_threshold)

# extraction of the network bursting activity
# now we identify the burts from the network and will extract an array with 
# tuples containing the burst start and end times
bursts= []
burst_start = []
burst_seconds_start = []
burst_end = []
burst_seconds_end = []



# filtering the actual network bursts in 5 ms bins
bursts= []
burst_start = []
burst_seconds_start = []
burst_end = []
burst_seconds_end = []
for index in range(0, len(ma_fr_gau[:-N])):
    if ma_fr_gau[index+N] > network_burst_threshold:
        if ma_fr_gau[index+N-1] <= network_burst_threshold:
            burst_start.append(index)
        if index == 0:
            burst_start.append(0)
            #burst_seconds_start.append((index+N)*0.005)
    else:
        if (ma_fr_gau[index+N-1] > network_burst_threshold) and (len(burst_start)>0):
            if index+N > len(ma_fr_gau):
                ending = len(ma_fr_gau)
            else: 
                ending = index + N

            burst_end.append(ending)
            #burst_seconds_end.append((ending)*0.005)
bursts = list(zip(burst_start, burst_end))



# now we need to reconvert the bins towards seconds:
for i in burst_start:
    burst_seconds_start.append(firing_rate_histogram[1][i])
for i in burst_end:
    if i >= len(firing_rate_histogram[1]):
        burst_seconds_end.append(firing_rate_histogram[1][-1])
    else:
        burst_seconds_end.append(firing_rate_histogram[1][i])

bursts_seconds = list(zip(burst_seconds_start, burst_seconds_end))
# bursts sind jetzt im 5ms bin   

# since we reference the bursts back to the seconds and those have different lengths
# we need to correct for bursts that are overlapping
bursts_seconds_corrected = []
for i in range(0, len(bursts_seconds)-1):
    
    first_b = bursts_seconds[i]
    old_first_start = first_b[0]
    old_first_end = first_b[1]
    
    second_b = bursts_seconds[i+1]
    old_second_start = second_b[0]
    old_second_end = second_b[1]
    
    if old_second_start < old_first_end:
        new_first_stop = old_second_start - 0.1 # we substract one msecond
    
        first_b = (old_first_start, new_first_stop)
    
    bursts_seconds_corrected.append(first_b)
    
bursts_seconds = bursts_seconds_corrected


# add the network bursts to the main_recording_dictionary
main_recording_dictionary['network_bursts_seconds'] = bursts_seconds


# we plot the final rasterplot + firing rate for the whole recording
# for sanity checking
fig = plt.figure(figsize = (12,6))
gs = fig.add_gridspec(2, hspace = 0, height_ratios=[1,5])
axs = gs.subplots(sharex=False, sharey=False)
axs[0].plot(ma_fr_gau, color= 'black', linewidth = 0.2)
axs[0].set_ylabel('Firing Rate [Hz]')
axs[1].eventplot(spikearray_seconds, color = 'black', linewidths = 0.3,
                 linelengths = 1, colors = 'black')
axs[1].set_ylabel('Relevant Channels')

for ax in axs:
    for i in bursts_seconds:
        axs[1].axvspan(i[0], i[1], facecolor = '#5B89A6', alpha = 0.3)
fig.savefig(os.path.join(output_directory, 
                         filename+ '__raster_firingrate_plot.png'), dpi=300)



# now we calculate the individual firing rates per channel
whole_recording_firingrate_dic = {}

# i.e, number of spikes divided by duration -> results in number per second
for key in spikedic_MAD:
    fr_channel = len(spikedic_MAD[key])/timelengthrecording_s 
    whole_recording_firingrate_dic[key] = fr_channel

# add it to the main dictionary
main_recording_dictionary['fr_dic'] = whole_recording_firingrate_dic


'''
add the basic spiking statistics to the recording

'''
# create the dictionary with isi + add it
isi_dictionary = get_isi_single_channel(spikedic_MAD_full)
main_recording_dictionary['isi_dictionary'] = isi_dictionary


# get the average isi and std
# creating list to easily calculate the whole mean and std
isi_averages = []
isi_standarddeviations = []

# creat dictionaries to do the same for every channel
isi_average_dic = {}
isi_standarddeviations_dic = {}


for key in isi_dictionary:
    if len(isi_dictionary[key]) > relevant_factor:
        
        # for the relevant channels we attain the mean
        mean_isi = np.mean(isi_dictionary[key])
        isi_averages.append(mean_isi)
        
        # and the standard deviation
        std_isi = np.std(isi_dictionary[key])
        isi_standarddeviations.append(std_isi)
        
        isi_average_dic[key] = mean_isi
        isi_standarddeviations_dic[key] = std_isi
        
        
        
mean_isi_relevant_channels = np.mean(isi_averages)
mean_isi_std = np.mean(isi_standarddeviations)

main_recording_dictionary['isi_average_dic'] = isi_average_dic
main_recording_dictionary['isi_std_dic'] = isi_standarddeviations_dic

Basics = {}

Basics['active_channels'] = active_channels
Basics['relevant_factor'] = relevant_factor
Basics['mean_fr_whole_recording'] = mean_fr_whole_recording


'''
load the anatomy as dicionary and invert it, add it to the infos
'''

# uncheck these when finished
#layerdic_invert = invert_layerdic(layerdic)
#Infos_Anatomy['layerdic'] = layerdic
#Infos_Anatomy['layerdic_invert'] = layerdic_invert


# add missing information to the main recording dic
Infos_Analysis['relevant_factor'] = relevant_factor
main_recording_dictionary['Infos_Recording'] = Infos_Recording
main_recording_dictionary['Infos_Analysis'] = Infos_Analysis
main_recording_dictionary['Infos_Anatomy'] = Infos_Anatomy
main_recording_dictionary['Basics'] = Basics

# and finally pickle the main_recording_dictionary
with open(os.path.join(output_directory+'/MAIN_RECORDING_Dictionary_'+filename+'.pkl'), 'wb') as f:
          pickle.dump(main_recording_dictionary, f)