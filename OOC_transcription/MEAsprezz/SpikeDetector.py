#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:14:47 2024

@author: jonas
"""

import numpy as np
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt

class SpikeDetector:
    def __init__(self, sampling_frequency, lowcut=150, highcut=4500, 
                 threshold_factor=5, dead_time=0.001, pre=0.001, post=0.002):
        self.sampling_frequency = sampling_frequency
        self.lowcut = lowcut
        self.highcut = highcut
        self.threshold_factor = threshold_factor
        self.dead_time = dead_time
        self.pre = pre
        self.post = post
        self.spike_dict = {}
        
            
    def detect_spikes(self, signal, time, channel_label):
        # Ensure signal and time have the same length
        min_length = min(len(time), len(signal))
        signal = signal[:min_length]
        time = time[:min_length]
    
        filtered_signal = self._bandpass_filter(signal)
        threshold = self._calculate_threshold(filtered_signal)
        crossings = self._detect_threshold_crossings(filtered_signal, threshold)
        spikes = self._align_to_minima(filtered_signal, crossings)
        
        # Ensure spikes are within valid range
        spikes = spikes[spikes < len(time)]
        
        spike_times = time[spikes]
        self.spike_dict[channel_label] = spike_times
        waveforms = self._extract_waveforms(filtered_signal, spikes)
        return spike_times, waveforms


    def _bandpass_filter(self, signal):
        nyq = 0.5 * self.sampling_frequency
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(4, Wn=[low, high], btype='band')
        return lfilter(b, a, signal)

    def _calculate_threshold(self, signal):
        noise_mad = np.median(np.absolute(signal)) / 0.6745
        return -self.threshold_factor * noise_mad

    def _detect_threshold_crossings(self, signal, threshold):
        crossings = np.diff((signal <= threshold).astype(int) > 0).nonzero()[0]
        if len(crossings) == 0:
            return np.array([])
        dead_time_idx = int(self.dead_time * self.sampling_frequency)
        return crossings[np.insert(np.diff(crossings) >= dead_time_idx, 0, True)]

    def _align_to_minima(self, signal, crossings):
        search_range = int(0.002 * self.sampling_frequency)  # 2ms search range
        spikes = []
        for crossing in crossings:
            if crossing + search_range < len(signal):
                spikes.append(crossing + np.argmin(signal[crossing:crossing+search_range]))
        return np.array(spikes, dtype=int)  # Ensure integer type

    def _extract_waveforms(self, signal, spikes):
        pre_idx = int(self.pre * self.sampling_frequency)
        post_idx = int(self.post * self.sampling_frequency)
        waveforms = []
        for spike in spikes:
            start = max(0, spike - pre_idx)
            end = min(len(signal), spike + post_idx)
            waveform = signal[start:end]
            # Pad the waveform if it's at the edge of the signal
            if len(waveform) < pre_idx + post_idx:
                pad_left = max(0, pre_idx - (spike - start))
                pad_right = max(0, (spike + post_idx) - end)
                waveform = np.pad(waveform, (pad_left, pad_right), mode='edge')
            waveforms.append(waveform)
        return np.array(waveforms)

    def plot_spikes(self, signal, time, channel_label):
        if channel_label not in self.spike_dict:
            print(f"No spikes detected for channel {channel_label}")
            return
    
        spike_times = self.spike_dict[channel_label]
        
        # Use the shorter length to avoid dimension mismatch
        min_length = min(len(time), len(signal))
        time = time[:min_length]
        signal = signal[:min_length]
    
        plt.figure(figsize=(20, 10))
        plt.plot(time, signal, c="#45858C", label='Signal')
        plt.plot(spike_times, np.interp(spike_times, time, signal), 'ro', ms=2, c="#D9580D", label='Spikes')
        plt.title(f'Channel {channel_label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (µV)')
        plt.legend()
        plt.show()

    def plot_average_waveform(self, channel_label):
        if channel_label not in self.spike_dict:
            print(f"No spikes detected for channel {channel_label}")
            return

        waveforms = self._extract_waveforms(self.filtered_signal, self.spike_dict[channel_label])
        if len(waveforms) == 0:
            print(f"No waveforms extracted for channel {channel_label}")
            return

        avg_waveform = np.mean(waveforms, axis=0)
        time_ms = np.linspace(-self.pre*1000, self.post*1000, len(avg_waveform))

        plt.figure(figsize=(12, 6))
        for waveform in waveforms[:100]:  # Plot first 100 waveforms
            plt.plot(time_ms, waveform, color='gray', alpha=0.3)
        plt.plot(time_ms, avg_waveform, color='red', linewidth=2)
        plt.title(f'Average Waveform for Channel {channel_label}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (µV)')
        plt.show()