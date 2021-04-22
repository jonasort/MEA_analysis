# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:19:40 2020

@author: User
"""


scriptdirectory = "C:/Users/User/Documents/JO/Python Scripts/MEA-master/MEA-master"
inputdirectory = "C:/Users/User/Documents/Jenny/Other MEA data"
outputdirectory = "C:/Users/User/Documents/Jenny/Other MEA data"

import os


os.chdir(scriptdirectory)



import sys
import numpy as np
import neo
import pandas as pd
import h5py
from hdfviewer.widgets.HDFViewer import HDFViewer
from hdfviewer.widgets.PathSelector import PathSelector
import McsPy
import sys, importlib, os
import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import bokeh.io
import bokeh.plotting
from bokeh.palettes import Spectral11
from scipy.signal import butter, lfilter, freqz, find_peaks, correlate
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import McsPy
import McsPy.McsData
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

from Butterworth_Filter import butter_bandpass, butter_bandpass_filter
import glob
import scipy


from neo.io import RawMCSIO
from neo.io import RawBinarySignalIO

os.chdir(inputdirectory)


reader = RawMCSIO(filename="DayDIV14_20190830_Plate6.raw")
reader = RawBinarySignalIO(filename="DayDIV14_20190830_Plate6.raw")
reader

RawBinarySignalIO.supported_objects

block = reader.read_block()
seg[0]
