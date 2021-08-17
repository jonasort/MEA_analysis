# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:35:11 2021

@author: jonas ort, MD, department of neurosurgery RWTH Aachen 

"""

scriptdirectory = "C:/Users/User/Documents/JO/gitkraken/MEA_analysis/Tübingen_Branch"
inputdirectory = r"D:\MEA_DATA_Aachen\PREPROCESSED\20210517_cortex_div11"



import os
os.chdir(scriptdirectory)

import sys
import numpy as np
import neo
import pandas as pd
import h5py

from neo.io import NeuroExplorerIO

os.chdir(inputdirectory)


reader = NeuroExplorerIO(filename='2021-05-17T12-21-29__cortex_div11_hCSF_ID046_nodrug_spont_2__.nex')
