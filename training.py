#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:26:44 2018

@author: eshwarmannuru
"""

import os
import sys

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
         ]

def createfolders(path):
    
    for l in letters:
        directory = path + "/" + l
        if not os.path.exists(directory):
            os.makedirs(directory)
        

p = sys.argv[1]
createfolders(p)
    