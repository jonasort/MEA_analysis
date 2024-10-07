#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:09:18 2024

@author: 
jonas ort, md, msc rwth
department of neurosurgery rwth university
sprezzatura coding collective
"""

# MEAsprezz/__init__.py

# Import the MEAData class from the data_handler module
from .data_handler import MEAData

# As you create more classes, you can import them here
# For example:
# from .spike_detector import SpikeDetector
# from .burst_detector import BurstDetector
# from .lfp_analyzer import LFPAnalyzer

# If you want to define what gets imported with "from MEAsprezz import *"
# you can define __all__
__all__ = ['MEAData']

# You can also add any initialization code for your package here if needed

# Optionally, you can define a version for your package
__version__ = '0.1.0'

# You could also add a brief description of the package
__description__ = 'MEAsprezz: A funky package for analyzing Micro Electrode Array data'