# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 17:12:43 2021

@author: jonas ort, MD rwth aachen university, department of neurosurgery
"""

import glob
import pickle
import os
import sys
import pandas as pd
import scipy
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
#networkx
import plotly.graph_objects as go
import networkx as nx
import matplotlib.patches as mpatches


'''
Directories

'''

working_directory =r"D:\MEA_DATA_Aachen\ANALYZED\trial_temp\Henner\ID046"

output_directory = r"D:\MEA_DATA_Aachen\ANALYZED\trial_temp\Henner\ID046"

filename = "2021-05-17_cortex_div11_aCSF_ID046_30µMNorepinephrine_spont_1"

os.chdir(working_directory)


'''
Functions

'''


def create_bins(lower_bound, width, quantity):
    """ create_bins returns an equal-width (distance) partitioning. 
        It returns an ascending list of tuples, representing the intervals.
        A tuple bins[i], i.e. (bins[i][0], bins[i][1])  with i > 0 
        and i < quantity, satisfies the following conditions:
            (1) bins[i][0] + width == bins[i][1]
            (2) bins[i-1][0] + width == bins[i][0] and
                bins[i-1][1] + width == bins[i][1]
    """
    

    bins = []
    for low in range(lower_bound, 
                     lower_bound + quantity*width + 1, width):
        bins.append((low, low+width))
    return bins


def find_bin(value, bins):
    """ bins is a list of tuples, like [(0,20), (20, 40), (40, 60)],
        binning returns the smallest index i of bins so that
        bin[i][0] <= value < bin[i][1]
    """
    
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1



def find_binned_spikes(data, bins):
    '''
    Parameters
    ----------
    data : for network spike binning --> expects an 1D array with all spikes detected for the network
    bins : list of tuples of expected bins

    Returns
    -------
    binlist : list of lists where lists contain all spikes for the respective bins

    '''
    binlist =[]
    binspike =[]
    for i in range(0, len(bins)):
        binspike = []
        for a in data:    
            if bins[i][0] <= a < bins[i][1]:
                binspike.append(a)
        binlist.append(binspike)
            
    return binlist




def get_isi_singlechannel(spikedic, tick):
    '''
    Parameters
    ----------
    spikedic : dictionary with all detected spikes for a channel
        DESCRIPTION.

    Returns
    -------
    isidic : keys = channels, values = List of tuples where tuple[0]=detected spike and tuple[1]=isi to the next detected spike
    isi_alone_dic : keys = channels, values = list of isi alone in microseconds!
    CAVE returns are in microseconds
    '''
    
    isidic ={}     
    isilist = []
    isi_alone_dic = {}
    isislist =[]

    for key in spikedic:
        isilist = []
        isislist = []
        if len(spikedic[key])>=2:
            for i in range(0, (len(spikedic[key])-1)):
                isi = spikedic[key][i]*tick, (spikedic[key][i+1]-spikedic[key][i])*tick #CL tick für beide dazu
                isi_alone = (spikedic[key][i+1]-spikedic[key][i])*tick
                isilist.append(isi)
                isislist.append(isi_alone)
        isidic[key]=isilist
        isi_alone_dic[key]=isislist
        
    return isidic, isi_alone_dic



def bin_isi(isi_alone_dic, binsize, binmax=bool, binmaxnumber=None):
    '''

    Parameters
    ----------
    isi_alone_dic : dic
        dictionary with all ISI for every channel
    binsize: int
        expects int in microseconds that defines bin-width
    Returns
    -------
    histo_ISI_dic:
        dic with key:channellabel, value: list with bincounts per bin

    '''
    isi_bins = []
    isi_bins_list = []
    isi_bin_count = []
    histo_ISI_dic = {}
    for key in isi_alone_dic:
        if binmax==True:
            isi_bin_count=[]
            isibins=create_bins(0, binsize, binmaxnumber)
            isi_bins_list=[] 
            for i in range(0, len(isibins)):
                isi_bins=[]
                for a in isi_alone_dic[key]:
                    if isibins[i][0] <= a < isibins[i][1]:
                        isi_bins.append(a)
                isi_bins_list.append(isi_bins)
            for i in range(0, (len(isi_bins_list)-1)):
                isi_bin_count.append(len(isi_bins_list[i]))
            histo_ISI_dic[key]=isi_bin_count
        #else:
            # noch schreiben für variable maximalnummer an bins
            
    return histo_ISI_dic


def bin_lnisi(isi_alone_dic, start_in_microseconds, stop_in_microseconds, totalbins, binmax):
    '''

    Parameters
    ----------
    isi_alone_dic : dic
        dictionary with all ISI for every channel 
    binsize: int
        expects int in microseconds that defines bin-width
    Returns
    -------
    histo_ISI_dic:
        dic with key:channellabel, value: list with bincounts per logarithmic spaced bins

    '''
    isi_bins = []
    isi_bins_list = []
    isi_bin_count = []
    histo_ln_ISI_dic = {}
    for key in isi_alone_dic:
        if binmax==True:
            isi_bin_count=[]
            isibins=create_ln_bins(start_in_microseconds, stop_in_microseconds, totalbins)
            isi_bins_list=[] 
            for i in range(0, len(isibins)):
                isi_bins=[]
                for a in isi_alone_dic[key]:
                    if isibins[i][0] <= a < isibins[i][1]:
                        isi_bins.append(a)
                isi_bins_list.append(isi_bins)
            for i in range(0, (len(isi_bins_list)-1)):
                isi_bin_count.append(len(isi_bins_list[i]))
            histo_ln_ISI_dic[key]=isi_bin_count
        #else:
            # noch schreiben für variable maximalnummer an bins
            
    return histo_ln_ISI_dic, isibins


def get_allchannel_ISI_bins(histo_ISI_dic):
    '''
    Parameters
    ----------
    histo_ISI_dic : dic mit den einzelnen ISI für jeden Channel. Cave, die Values müssen alle die gleiche
                    Länge haben, sonst funktioniert die zip Funktion nicht.
        DESCRIPTION.

    Returns
    -------
    network_ISI_binned = array of all ISI of the whole network binned

    '''
    network_ISI = []
    for key in histo_ISI_dic:
        list1 = histo_ISI_dic[key]
        if len(list1)>len(network_ISI):
            network_ISI=list1
        else:
            list2 = network_ISI
            network_ISI = [a + b for a, b in zip(list1, list2)]
    return np.array(network_ISI)


def get_burst_threshold(df_with_CMA):
    '''
    

    Parameters
    ----------
    df_with_CMA : TYPE
        DESCRIPTION.

    Returns
    -------
    CMAalpha : TYPE
        DESCRIPTION.
    CMAalpha2 : TYPE
        DESCRIPTION.
    maxCMA : TYPE
        DESCRIPTION.
    alpha1 : TYPE
        DESCRIPTION.
    alpha2 : TYPE
        DESCRIPTION.

    '''
    
    networkburstthreshold_ISI = 200000 #wie im paper maximal bei 200 ms als isi
    skewness = scipy.stats.skew(network_ISI)
    if skewness < 1:
        alpha1 = 1
        alpha2 = 0.5
    elif skewness >= 1 and skewness <4:
        alpha1 = 0.7
        alpha2 = 0.3
    elif skewness >=4 and skewness <9:
        alpha1 = 0.5
        alpha2 = 0.3
    elif skewness >=9:
        alpha1 = 0.3
        alpha2 = 0.1
    maxCMA = max(df_with_CMA['CMA'])
    CMAalpha = maxCMA*alpha1
    CMAalpha2 = maxCMA*alpha2
    return CMAalpha, CMAalpha2, maxCMA, alpha1, alpha2


def ISI_threshold_min(df_with_CMA, CMAalpha, CMAalpha2, binsize_in_micros):
    '''
    '''
    indexfactor = df[df['CMA']>CMAalpha].index[-1] + 1
    indexfactor2 = df[df['CMA']>CMAalpha2].index[-1] + 1
    threshold_intraburst = float(indexfactor*binsize_in_micros)
    threshold_burst_related = float(indexfactor2*binsize_in_micros)
    
    return threshold_intraburst, threshold_burst_related


def find_burst_starts(isi_alone, threshold_intraburst, spikedic):
    '''
    Parameters
    ----------
    isi_alone : TYPE
        DESCRIPTION.
    threshold_intraburst : TYPE
        DESCRIPTION.
    spikedic : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    burststartdic = {}
    noburstlist = []
    #burststartlist = []
    for key in isi_alone:
        #print(key)
        if len(isi_alone[key])<3:
            noburstlist.append(isi_alone[key])
        burststartlist=[]
        counter = 0
        while counter < (len(isi_alone[key])-4):
            setter = 0
            if isi_alone[key][counter]<threshold_intraburst:
                setter +=1
                if isi_alone[key][counter+setter] < threshold_intraburst:
                    setter +=1
                    if isi_alone[key][counter+setter] < threshold_intraburst:
                        burststartlist.append((spikedic[key][counter])*tick) #CL: zusätzlich times tick to get all timestamps in ms
                        setter += 1
                        while isi_alone[key][counter+setter]<threshold_intraburst and (counter+setter)< (len(isi_alone[key])-4):
                            setter +=1
                            #print('burst '+str(setter))
                        setter +=1
                    else:
                        counter +=1
                else:
                    counter +=1
                counter = counter + setter + 1
            else:
                counter +=1
            #print(str(key) + str(counter))
        burststartdic[key]=burststartlist
        
    return burststartdic   


def find_burst_starts_and_length(isi_alone, threshold_intraburst, spikedic):
    '''
    Parameters
    ----------
    isi_alone : dict
        k = channellabel, values = interspike intervals in microseconds
    threshold_intraburst : float
        the calculated threshold for a single channel burst in microseconds
    spikedic : dict
        k = channellabel, values = spiketimes in ticks
        

    Returns
    -------
    burststart_end_dic : dict
        k = channellabel, values = tuple(a,b) with a = start of a burst x, b= end of burst x 
        with all times in microseconds

    '''
    burststartdic = {}
    noburstlist = []
    #burststartlist = []
    for key in isi_alone:
        #print(key)
        if len(isi_alone[key])<3:
            noburstlist.append(isi_alone[key])
        burststartlist=[]
        counter = 0
        while counter < (len(isi_alone[key])-4):
            setter = 0
            if isi_alone[key][counter]<threshold_intraburst:
                setter +=1
                if isi_alone[key][counter+setter] < threshold_intraburst:
                    setter +=1
                    if isi_alone[key][counter+setter] < threshold_intraburst:
                        burststart_spike = spikedic[key][counter]*tick
                        burstend_spike = spikedic[key][counter+setter]*tick
                        #burststartlist.append((spikedic[key][counter])*tick) #CL: zusätzlich times tick to get all timestamps in µs
                        setter += 1
                        while isi_alone[key][counter+setter]<threshold_intraburst and (counter+setter)< (len(isi_alone[key])-4):
                            setter +=1
                            burstend_spike = spikedic[key][counter+setter]*tick
                            #print('burst '+str(setter))
                        burststartlist.append((burststart_spike, burstend_spike))
                        setter +=1
                    else:
                        counter +=1
                else:
                    counter +=1
                counter = counter + setter + 1
            else:
                counter +=1
            #print(str(key) + str(counter))
        burststartdic[key]=burststartlist
        
    return burststartdic   

def find_burst_starts_and_length(isi_alone, threshold_intraburst, spikedic):
    '''
    Parameters
    ----------
    isi_alone : dict
        k = channellabel, values = interspike intervals in microseconds
    threshold_intraburst : float
        the calculated threshold for a single channel burst in microseconds
    spikedic : dict
        k = channellabel, values = spiketimes in ticks
        

    Returns
    -------
    burststart_end_dic : dict
        k = channellabel, values = tuple(a,b) with a = start of a burst x, b= end of burst x 
        with all times in microseconds

    '''
    burststartdic = {}
    noburstlist = []
    #burststartlist = []
    for key in isi_alone:
        #print(key)
        if len(isi_alone[key])<3:
            noburstlist.append(isi_alone[key])
        burststartlist=[]
        counter = 0
        while counter < (len(isi_alone[key])-3):
            setter = 0
            if isi_alone[key][counter]<threshold_intraburst:
                setter +=1
                if isi_alone[key][counter+setter] < threshold_intraburst:
                    setter +=1
                    if isi_alone[key][counter+setter] < threshold_intraburst:
                        setter += 1
                        burststart_spike = spikedic[key][counter]*tick
                        burstend_spike = spikedic[key][counter+setter]*tick
                        #burststartlist.append((spikedic[key][counter])*tick) #CL: zusätzlich times tick to get all timestamps in µs
                        
                        while isi_alone[key][counter+setter]<threshold_intraburst and (counter+setter)< (len(isi_alone[key])-3):
                            setter +=1
                            burstend_spike = spikedic[key][counter+setter]*tick
                            
                            #print('burst '+str(setter))
                        burststartlist.append((burststart_spike, burstend_spike))
                        #setter +=1
                        counter = counter + setter
                    else:
                        counter +=1
                else:
                    counter +=1
                
            else:
                counter +=1
            #print(str(key) + str(counter))
        burststartdic[key]=burststartlist
        
    return burststartdic  


def extract_burststarts(burststartenddic):

    burststart_dic = {}
    burstlist = []
    
    for key in burststartenddic:
        burstlist = []
        start_ends = burststartenddic[key]
        for i in start_ends:
            burstlist.append(i[0])
        burststart_dic[key] = burstlist
        
    return burststart_dic



def subdivide_spiketrain(spiketrain, sub_start = 0, sub_stop = 10, tick=40, scale_factor_for_second=1e-06):
    '''
    Excpects: 
        a spiketrain with tick datapoints
        default ticks are 40
        default scale_factor_for_seconds = 1e-06
        provide the start and stop of the desired sub in seconds
    
    Does:
        converts the desired seconds into data ticks
        checks if the spikes of the given spiketrain is in the desired subs
        substracts the starting time -->
        
    Returns:
        a spiketrain dictionary that again starts from zero
    
    '''
    sub_start_tick = sub_start / (tick*scale_factor_for_second)
    sub_stop_tick = sub_stop / (tick*scale_factor_for_second)
    sub_spiketrain = {}
  
    for key in spiketrain: 
        list_per_key = []
        for i in spiketrain[key]:
            if (i>=sub_start_tick ) & (i<sub_stop_tick):
                list_per_key.append(int(i-sub_start_tick))
        sub_spiketrain[key]=list_per_key

    return sub_spiketrain



def get_interburst_intervals(burststart_end_dic):
    
    '''
    parameters:
    
    burststart_end_dic : dic
    keys = channellabels
    values = list of tuples tuple (a, b) with a = burststarts, b = burstends in µs
    
    
    ______________________
    
    returns:
    
    ibi_dic : dic
    keys = channellabels
    values = list of all interburstintervals in µs
    
    
    ______________________
    
    nota bene:
    
    interburst intervals are defined as non-bursting intervals between bursts.
    That means it is from burst-end1 to burststart2.
    
    '''
    
    
    ibi_dic = {}
    ibi_list = []
    
    for key in burststart_end_dic:
        ibi_list = []
        bursts = burststart_end_dic[key]
        for i in range(0, len(bursts)-1): # we leave the last burst out
            burst_end = bursts[i][1]
            next_start = bursts[i+1][0]
            
            interburst_interval = next_start - burst_end
            
            ibi_list.append(interburst_interval)
        
        ibi_dic[key] = ibi_list
        
    return ibi_dic


def invert_layerdic(layer_dic):
    
    '''
    Expects a dictionary with key = layer, value = list of channellabels
    
    Returns a dictionary with key = channellabels, value = layer
    '''
    layerdic_invert = {}

    for key in layer_dic:
        for i in layer_dic[key]:
            layerdic_invert[i]=key
            
            
    return layerdic_invert
            

'''
SCRIPT

'''


filelist = glob.glob('*'+filename+'*.pkl')

# load in the pickle
MAIN_RECORDING_DICTIONARY = pickle.load(
    open(os.path.join(output_directory+'\\MAIN_RECORDING_Dictionary_'+filename+'.pkl'), 
         "rb"))


MAIN_RECORDING_DICTIONARY['Infos_Recording'].keys()
tick = MAIN_RECORDING_DICTIONARY['Infos_Recording']['tick']
timelengthrecording_s = MAIN_RECORDING_DICTIONARY['Infos_Recording']['timelengthrecording_s']
scale_factor_for_second = MAIN_RECORDING_DICTIONARY['Infos_Recording']['scale_factor_for_second']
scale_factor_for_milisecond = scale_factor_for_second/1000
spikedic_MAD = MAIN_RECORDING_DICTIONARY['spikedic_MAD']

# get the isi distribution
st_channel = spikedic_MAD
binsize_for_ISI = 5000 #in microseconds

isidic, isi_alone = get_isi_singlechannel(st_channel, tick) #creates two dictionaries
histo_ISI_dic=bin_isi(isi_alone, binsize=binsize_for_ISI, 
                      binmax=True, binmaxnumber=200) # dictionary für jeden channel mit 300x 10ms bins (binsize) und der Menge an ISI für die jeweilige Länge
network_ISI=get_allchannel_ISI_bins(histo_ISI_dic) #gibt ein array mit bins entsprechend der bins aus der Vorfunktion

colors = ['green', 'blue', 'orange', 'purple']
df= pd.DataFrame({'ISI_per_10ms_bins':network_ISI}) #aus Network_ISI wird ein pdDF um die weiteren Schritte durchführen zu können
df["CMA"] = df.ISI_per_10ms_bins.expanding().mean()
df[['ISI_per_10ms_bins', 'CMA']].plot(color=colors, linewidth=3, 
                                      figsize=(10,4), 
                                      title="Histogram of ISI-bins 10ms whole network")


# calculate the adaptive threshold
CMAalpha, CMAalpha2, maxCMA, alpha1, alpha2=get_burst_threshold(df) # threshold calculation
threshold_intraburst, threshold_burst_related = ISI_threshold_min(df, CMAalpha, 
                                                                  CMAalpha2, binsize_for_ISI) #set thresholds

print('intraburst: ', threshold_intraburst, '   related: ', threshold_burst_related)


# final threshold is calculated from the burst related within our defined limits
final_threshold = 0
if threshold_burst_related > 140000:
    final_threshold = 140000
elif threshold_burst_related < 60000:
    final_threshold = 60000
else:
    final_threshold = threshold_burst_related
print('The final threshold for this recoding is: {}'.format(final_threshold))


# add to main recording dic
# add the final threshold to the 
Infos_Analysis = MAIN_RECORDING_DICTIONARY['Infos_Analysis']
Infos_Analysis['isi_burst_threshold_base'] = final_threshold

MAIN_RECORDING_DICTIONARY['Infos_Analysis'] = Infos_Analysis



# calculate the burststarts
burststart_end_dic = find_burst_starts_and_length(isi_alone, final_threshold, st_channel) 

# add them to the Main dictionary
MAIN_RECORDING_DICTIONARY['Bursts'] = burststart_end_dic



# extract all burststarts for the spade analysis + save it 
burststart_dic = extract_burststarts(burststart_end_dic)
np.save(filename+'_burst_starts_dictionary.npy', burststart_dic)



 # create an array of the spikes in scale of seconds
active_channels = 0
spikedic_seconds = {}
for key in burststart_dic:
    relevant_factor = timelengthrecording_s*0.05
    if len(burststart_dic[key])>relevant_factor:
        sec_array = np.asarray(burststart_dic[key])*tick*scale_factor_for_second
        spikedic_seconds[key]=sec_array
        active_channels += 1
spikearray_seconds = np.asarray(list(spikedic_seconds.values())) 


# calculate and save inter burst intervals and save them to main recording dic
burst_ibi_dic = get_interburst_intervals(burststart_end_dic)
MAIN_RECORDING_DICTIONARY['Interburst-Intervals'] = burst_ibi_dic



# for every unit, the whole time of bursts is calculated and put into a dictionary
bursting_time_per_unit_dic = {}
for key in burststart_end_dic:
    time = 0
    for i in burststart_end_dic[key]:
        bursttime = i[1] - i[0]
        time = time + bursttime
    bursting_time_per_unit_dic[key] = time*scale_factor_for_second # kein tick, vorher bereits drin



# for every unit, the whole time of bursts is calculated and put into a dictionary
bursts_per_unit_dic = {}
for key in burststart_end_dic:
    number_of_bursts = 0
    for i in burststart_end_dic[key]:
        number_of_bursts += 1
    bursts_per_unit_dic[key] = number_of_bursts

# save both
MAIN_RECORDING_DICTIONARY['bursting_time_per_channel'] = bursting_time_per_unit_dic
MAIN_RECORDING_DICTIONARY['bursts_per_channel'] = bursts_per_unit_dic



'''
Burst Connections
'''
# this function takes the dictionary with every burst start and stop and returns a dictionary
# where every unit is a key and the values are tuples consisting of keys of connected units (i.e., bursting together)
# and the number of shared bursts
burst_connections_dic = {}


for key in burststart_end_dic:
    other_keys = list(burststart_end_dic.keys())
    other_keys.remove(key)
    connected_unit_list = []
    for j in other_keys:
        number_of_bursts = 0
        time_shared = 0
        for i in burststart_end_dic[key]:
            start, end = i[0], i[1]
            for m in burststart_end_dic[j]:
                other_start = m[0]
                other_end = m[1]
                if (other_start > start) & (other_start < end):
                    if other_end >= end:
                        time_shared = time_shared + (end - other_start)
                    else:
                        time_shared = time_shared + (other_end - other_start)
                    number_of_bursts += 1
                    
        if number_of_bursts > 0:
            connected_unit_list.append((j, number_of_bursts, time_shared*scale_factor_for_second))        
    burst_connections_dic[key] = connected_unit_list









# we now calculate the burst connections with at least 0.1Hz
simple_burst_connection = {}

for key in burst_connections_dic:
    listed_connections = []
    for i in burst_connections_dic[key]:
        if i[1] > timelengthrecording_s*0.1: # Länge des Recordings * mindestens 0.1Hz -> alle zehn Sekunden
        #if i[1] > int(mean_fr_whole_recording*0.1):
        
       #if i[2] > 3: # Länge der gesharedten bursts in sec
        #if (i[1] > 10) & (i[2] > 1): # Länge der gesharedten bursts in sec
            listed_connections.append(i[0])
    simple_burst_connection[key] = listed_connections


MAIN_RECORDING_DICTIONARY['burst_connections'] = simple_burst_connection


# aachen MEA
all_channels = ['D1', 'E1', 'F1', 'G1', 'H1', 'J1', 'J2', 'K1', 'K2', 'L1', 'L2', 'L3', 'M1', 'M2', 
                        'M3', 'M4', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 
                        'O7', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'R2', 'R3', 'R4', 'R5', 
                        'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'B1', 'B2', 'C1', 'C2', 'D2', 'E2', 'F2', 'G2', 'G3', 'H2', 'H3', 'J3', 'K3', 'K4', 
                         'L4', 'L5', 'M5', 'M6', 'M7', 'N7', 'N8', 'O8', 'O9', 'O10', 'O11', 'P10', 'P11', 
                         'P12', 'P13', 'P14', 'P15', 'P16', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'B3', 'B4', 'B5', 'B6', 
                         'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 
                          'C11', 'C12', 'C13', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 
                         'D13', 'D14', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 
                         'E15', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 
                         'F16', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 
                         'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'J4', 
                         'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'K5', 'K6', 
                         'K7', 'K8', 'K9', 'K10', 'K11', 'K12', 'K13', 'K14', 'K15', 'K16', 'L6', 'L7', 'L8', 'L9', 
                         'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 
                         'M14', 'M15', 'M16', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'O12', 'O13', 
                         'O14', 'O15', 'O16', 'A12', 'A13', 'A14', 'A15', 'B13', 'B14', 'B15', 'B16', 'C14', 'C15', 'C16', 'D15', 'D16', 'E16']


# Reutlingen MEA
"""
all_channels = ['D1', 'E1', 'F1', 'G1', 'H1', 'I1', 'I2', 'K1', 'K2', 'L1', 'L2', 'L3', 'M1', 'M2', 
                        'M3', 'M4', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 
                        'O7', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'R2', 'R3', 'R4', 'R5', 
                        'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'B1', 'B2', 'C1', 'C2', 'D2', 'E2', 'F2', 'G2', 'G3', 'H2', 'H3', 'I3', 'K3', 'K4', 
                         'L4', 'L5', 'M5', 'M6', 'M7', 'N7', 'N8', 'O8', 'O9', 'O10', 'O11', 'P10', 'P11', 
                         'P12', 'P13', 'P14', 'P15', 'P16', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'B3', 'B4', 'B5', 'B6', 
                         'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 
                          'C11', 'C12', 'C13', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 
                         'D13', 'D14', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 
                         'E15', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 
                         'F16', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 
                         'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'I4', 
                         'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'K5', 'K6', 
                         'K7', 'K8', 'K9', 'K10', 'K11', 'K12', 'K13', 'K14', 'K15', 'K16', 'L6', 'L7', 'L8', 'L9', 
                         'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 
                         'M14', 'M15', 'M16', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'O12', 'O13', 
                         'O14', 'O15', 'O16', 'A12', 'A13', 'A14', 'A15', 'B13', 'B14', 'B15', 'B16', 'C14', 'C15', 'C16', 'D15', 'D16', 'E16']


"""


#CAVE: Always make sure to insert the acutal layerdic for the specific recording

#  layerdic ID0046, 11 DIV, OP 2021_05_06
layerdic = {'layer1':["B16","C16","D16","E16","F16","G16","H16","J16","K16","L16","M16","N16","O16","P16"],

 

 

            'layer2-3': ["A11","A12","A13","A14","A15","B11","B12","B13","B14","B15","C11","C12","C13","C14","C15",

"D11","D12","D13","D14","D15","E11","E12","E13","E14","E15","F11","F12","F13","F14","F15",

"G11","G12","G13","G14","G15","H11","H12","H13","H14","H15","J11","J12","J13","J14","J15",

"K11","K12","K13","K14","K15","L11","L12","L13","L14","L15","M11","M12","M13","M14","M15",

"N11","N12","N13","N14","N15","O11","O12","O13","O14","O15","P11","P12","P13","P14","P15",

"R11","R12","R13","R14","R15"],

 

           'layer4':    ["A9","A10","B9","B10","C9","C10","D9","D10","E9","E10","F9","F10","G9","G10","H9","H10",

"J9","J10","K9","K10","L9","L10","M9","M10","N9","N10","O9","O10","P9","P10","R9","R10"],

 

 

           'layer5-6':  [ "A2", "A3", "A4", "A5", "A6", "A7", "A8",

"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",

"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",

"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",

"E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",

"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",

"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8",

"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8",

"J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8",

"K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8",

"L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",

"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8",

"N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8",

"O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8",

"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",

"R2", "R3", "R4", "R5", "R6", "R7", "R8",

],

 

           'whitematter':[]}

 


layer_colors={'layer1':'#F28749', 'layer2-3':'#B5D932', 'layer4':'#8C3A67', 'layer5-6':'#3F93A6', 'whitematter':'#C9F2EE'}

inverted_layerdic = invert_layerdic(layerdic)

Infos_Anatomy = {}
Infos_Anatomy['layerdic_invert']=inverted_layerdic
Infos_Anatomy['layerdic']=layerdic

MAIN_RECORDING_DICTIONARY['Infos_Anatomy'] = Infos_Anatomy

# this block creates coordinates as on a MEA Grid for each channel
# coordinates are between 0 and 1 via np.linspace

# CAVE: I needs to be exchanged for J for Aachen Data, or i.e. the not MC Rack obtained data
columnlist =['A','B','C','D','E','F','G','H','J','K','L','M','N','O','P','R']
mea_coordinates = np.linspace(0,1,16)
mea_positional_coordinates_dic = {}

for i in all_channels:
    x_num = columnlist.index(i[0])
    x_coord = mea_coordinates[x_num]
    y_num = 17-int(i[1:]) # minus 1 since python starts counting at zero
    y_coord = 1-(mea_coordinates[-y_num])
    mea_positional_coordinates_dic[i] = [x_coord, y_coord]


#normalize the bursting time per unit 
normalized_bursting_time_per_unit_dic = {}
time_list = list(bursting_time_per_unit_dic.values())
maximum = max(time_list)
minimum = min(time_list)

#### Redo For Loop muss nach außen
try:
    
    for key in bursting_time_per_unit_dic:
        value = bursting_time_per_unit_dic[key]
        normalized = (value - minimum)/(maximum-minimum)
        normalized_bursting_time_per_unit_dic[key] = normalized
except ZeroDivisionError:
    print('ZeroDivisionError - ')
    

'''
The Burst Graph is produced
'''

burst_conn_graph = nx.Graph()
for key in simple_burst_connection:
    for i in simple_burst_connection[key]:
        burst_conn_graph.add_edge(key, i)

burst_conn_graph.number_of_nodes(), burst_conn_graph.number_of_edges()

G = burst_conn_graph


for i in G.nodes():
    
    try:
        node_key = i
        coordinate = mea_positional_coordinates_dic[node_key]
        G.nodes[node_key]['pos']=coordinate
        G.nodes[node_key]['layer']=inverted_layerdic[i]
        G.nodes[node_key]['color']=layer_colors[inverted_layerdic[i]]
        
        try:
            G.nodes[node_key]['degree_centrality']=nx.degree_centrality(G)[i]
        except:
            print('degree centrality failed')
            
        try:
            G.nodes[node_key]['betweenness_centrality']=nx.betweenness_centrality(G, k=10, endpoints = True)[i]
        except:
            print('betweennes centrality failed')
            
        try:
            G.nodes[node_key]['bursting_time_normalized']=normalized_bursting_time_per_unit_dic[i]
        except:
            print('normalized bursting time not possible')
    except KeyError:
        print('channel ', node_key, ' failed')


pos = nx.get_node_attributes(G, 'pos')
layer = nx.get_node_attributes(G, 'layer')
color = nx.get_node_attributes(G, 'color')
burst_time = nx.get_node_attributes(G, 'bursting_time_normalized')
try:
    centrality = nx.betweenness_centrality(G, k=10, endpoints = True)
except:
    print('Degree Centrality Exception encountered')
'''
Plot the Graph Result for sanity Check


Does not yet work:
    
______________

# degree centrality
for i in G.nodes():
    try:
    
        node_key = i
        coordinate = mea_positional_coordinates_dic[node_key]
        G.nodes[node_key]['pos']=coordinate
        G.nodes[node_key]['layer']=inverted_layerdic[i]
        G.nodes[node_key]['color']=layer_colors[inverted_layerdic[i]]
        G.nodes[node_key]['degree_centrality']=nx.degree_centrality(G)[i]
        G.nodes[node_key]['bursting_time_normalized']=normalized_bursting_time_per_unit_dic[i]
    except KeyError:
        print('KeyError with channel ', node_key)

pos = nx.get_node_attributes(G, 'pos')
layer = nx.get_node_attributes(G, 'layer')
color = nx.get_node_attributes(G, 'color')
bursting_time_normalized = nx.get_node_attributes(G, 'bursting_time_normalized')

# plot the figures
fig, ax = plt.subplots(1,1, figsize=(10,10))


nodes = G.nodes()
# colors = [mapping[G.nodes[n]['color']] for n in nodes]
node_size = [v * 10000 for v in centrality.values()]
#node_size = [v * 5000 for v in bursting_time_normalized.values()]


nx.draw_networkx(
    G,
    pos,
    node_size=node_size,
    node_color=color.values(),
    edge_color='gainsboro',
    ax=ax
)


ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(mea_coordinates)
ax.set_yticks(mea_coordinates)
ax.set_xticklabels(columnlist)
ax.set_yticklabels([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
ax.set_title('Shared Single Channel Bursts - Network; Degree Centrality')

ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

ax.grid(ls=':')
#plt.axis("off")
ax.legend(handles=layer_colors.values(), labels=layer_colors.keys())
layer1 = mpatches.Patch(color='#F28749', label='layer 1')
layer23 = mpatches.Patch(color='#B5D932', label='layer 2/3')
layer4 = mpatches.Patch(color='#8C3A67', label='layer 4')
layer56 = mpatches.Patch(color='#3F93A6', label='layer 5/6')
wm = mpatches.Patch(color='#C9F2EE', label='whitematter')
plt.legend(handles=[layer1, layer23, layer4, layer56, wm])

#plt.yticks(mea_coordinates, columnlist)
#plt.xticks(list(mea_coordinates), columnlist)
plt.show()

fig.savefig(filename+'_betweenness_centrality_0p1Gz_04102021.png', dpi = 300, bbox_inches='tight', transparent=True)


'''


MAIN_RECORDING_DICTIONARY['GRAPH_shared_bursts'] = G


with open(os.path.join(output_directory+'\\MAIN_RECORDING_Dictionary_'+filename+'.pkl'), 'wb') as f:
    pickle.dump(MAIN_RECORDING_DICTIONARY, f)































