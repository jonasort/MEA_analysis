#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:49:08 2024

@author: 
jonas ort, md, msc rwth
department of neurosurgery rwth university
sprezzatura coding collective

"""
# main_analysis.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("/Users/jonas/Documents/GitHub/MEA_analysis/OOC_transcription"))))

from MEAsprezz.data_handler import MEAData
from MEAsprezz.SpikeDetector import SpikeDetector

file_path = '/Users/jonas/Documents/DATA/MEA_DATA_Aachen_sample/datasample/2021-05-17T11-28-41__cortex_div11_aCSF_ID046_30µMNE_spont_1__.h5'



from MEAsprezz.data_handler import MEAData
import os
import glob
import numpy as np
from datetime import datetime

def main():
    # Get input directory from user
    input_directory = input('Please enter the file directory: ')
    os.chdir(input_directory)
    file_list = glob.glob("*.h5")

    # Get analysis preferences from user
    bool_channelmap = input('Do you want to use a labeldictionary? Enter y or n: ')
    bool_location = input('Enter A if this file is from Aachen and R if it is from Reutlingen: ')
    bool_modules = input('If you want the basic analysis (spikes only), enter b. If you want extended analysis (including lfp times), enter e: ')

    # Time string for output naming
    time_str = datetime.today().strftime('%Y-%m-%d')

    # Set filter parameters
    lowcut = 150
    highcut = 4500

    # Length of cutouts around spikes
    pre = 0.001  # 1 ms
    post = 0.002  # 2 ms

    # Divide recording in n seconds long subrecordings
    dividing_seconds = 120

    # Process each file
    for file in file_list:
        print(f'Working on file: {file}')
        
        # Load the MEA data
        mea_data = MEAData(os.path.join(input_directory, file))
        
        # Get recording info
        recording_info = mea_data.get_recording_info()
        print(f"Recording duration: {recording_info['timelength_recording_s']} seconds")
        print(f"Sampling frequency: {recording_info['sampling_frequency']} Hz")
        
        # Get subdivision ranges
        sub_ranges = mea_data.subdivide_recording(dividing_seconds)
        
        # Process each subdivision
        for start, stop in sub_ranges:
            print(f"Processing subdivision from {start} to {stop} seconds")
            
            # Process each channel
            for channel_idx in range(recording_info['number_of_channels']):
                # Get channel data
                signal, time = mea_data.get_channel_data(channel_idx, from_in_s=start, to_in_s=stop)
                
                # Here you would call your spike detection function
                # For example:
                # spikes = detect_spikes(signal, time, recording_info['sampling_frequency'])
                
                # Store or process the detected spikes
                # ...

        # After processing all subdivisions, you can aggregate the results
        # ...

    print('Finished the analysis. Check your output folder.')

if __name__ == '__main__':
    main()
    

spike_detector = SpikeDetector(sampling_frequency=recording_info['sampling_frequency'])

    
signal, time = mea_data.get_channel_data(channel_idx, from_in_s=start, to_in_s=stop)
channel_label = mea_data.get_channel_labels()[channel_idx]
  
spike_times, waveforms = spike_detector.detect_spikes(signal, time, channel_label)
  