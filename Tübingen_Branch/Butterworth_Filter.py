#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 07:11:09 2020

@author: jonas ort
"""

import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, find_peaks, correlate
from scipy import signal
from scipy import stats
from scipy.stats import spearmanr,pearsonr
import pandas as pd
import cProfile
import h5py
import time
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import collections
import math
from matplotlib.patches import Ellipse



    

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y    
    
fs = 10000          #sample rate
lowcut = 150        #cutoff frequencies (in Hz)
highcut = 4500
#data = dset    
#y = butter_bandpass_filter(data, lowcut, highcut, fs, order=4)        


def detect_peaks(data):
    threshold =5 * np.std(y) #np.median(np.absolute(y)/0.6745)
    peaks, _ = find_peaks(-y, height= threshold, distance=50)   
    return peaks,y,threshold