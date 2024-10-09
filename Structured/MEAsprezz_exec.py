#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:54:51 2024

@author: 
jonas ort, md, msc rwth
department of neurosurgery rwth university
sprezzatura coding collective
"""
import os
import numpy as np
import McsPy.McsData
from McsPy import ureg, Q_
import MEAsprezz


# In your main function or analysis script
file_path = "path/to/your/mea/data.h5"
mea_data = MEAData(file_path)

# Get information about the recording
recording_info = mea_data.get_recording_info()
print(f"Recording duration: {recording_info['timelength_recording_s']} seconds")

# Get data for a specific channel
channel_idx = 0
signal, time = mea_data.get_channel_data(channel_idx, from_in_s=0, to_in_s=10)  # First 10 seconds

# Get time ranges for subdividing the recording
sub_ranges = mea_data.subdivide_recording(sub_length_s=120)

# Iterate over subdivisions
for start, end in sub_ranges:
    # Process each subdivision...
    pass