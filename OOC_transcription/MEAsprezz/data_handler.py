#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:49:08 2024

@author: 
jonas ort, md, msc rwth
department of neurosurgery rwth university
sprezzatura coding collective

"""

import os
import numpy as np
import McsPy.McsData
from McsPy import ureg, Q_

class MEAData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.analog_stream = None
        self.channel_data = None
        self.channel_ids = None
        self.sampling_frequency = None
        self.time_stamps = None
        self.scale_factor_for_second = None
        self.timelength_recording_s = None
        self.load_data()

    def load_data(self):
        """Load the MEA data from the file."""
        self.raw_data = McsPy.McsData.RawData(self.file_path)
        self.analog_stream = self.raw_data.recordings[0].analog_streams[0]
        self.channel_data = self.analog_stream.channel_data
        self.channel_ids = [c.channel_id for c in self.analog_stream.channel_infos.values()]
        self._set_recording_info()

    def _set_recording_info(self):
        """Set basic recording information."""
        channel_info = self.analog_stream.channel_infos[self.channel_ids[0]]
        self.sampling_frequency = channel_info.sampling_frequency.magnitude
        time = self.analog_stream.get_channel_sample_timestamps(self.channel_ids[0])
        self.scale_factor_for_second = Q_(1, time[1]).to(ureg.s).magnitude
        self.time_stamps = time[0] * self.scale_factor_for_second
        self.timelength_recording_s = self.time_stamps[-1]

    def get_channel_data(self, channel_idx, from_in_s=0, to_in_s=None):
        """Get data for a specific channel."""
        channel_id = self.channel_ids[channel_idx]
        from_idx = max(0, int(from_in_s * self.sampling_frequency))
        to_idx = min(self.channel_data.shape[1], int(to_in_s * self.sampling_frequency)) if to_in_s else None
        signal = self.analog_stream.get_channel_in_range(channel_id, from_idx, to_idx)
        scale_factor_for_uV = Q_(1, signal[1]).to(ureg.uV).magnitude
        return signal[0] * scale_factor_for_uV, self.time_stamps[from_idx:to_idx]

    def get_channel_labels(self):
        """Get labels for all channels."""
        return [self.analog_stream.channel_infos[ch_id].info['Label'] for ch_id in self.channel_ids]

    def get_recording_info(self):
        """Get basic recording information."""
        return {
            'sampling_frequency': self.sampling_frequency,
            'timelength_recording_s': self.timelength_recording_s,
            'scale_factor_for_second': self.scale_factor_for_second,
            'number_of_channels': len(self.channel_ids)
        }

    def subdivide_recording(self, sub_length_s):
        """Generate time ranges for subdividing the recording."""
        return [(i, min(i + sub_length_s, self.timelength_recording_s)) 
                for i in np.arange(0, self.timelength_recording_s, sub_length_s)]