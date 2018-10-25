#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd



#function to get current directory
def getCurrentDirectory():
    import os
    print(os.listdir('../'))



#function to read csv file
def readCSVfile(path):
    crimes_original = pd.read_csv(path, low_memory=False)





