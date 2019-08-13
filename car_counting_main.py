# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:06:38 2019

@author: YOGENDER
"""

import tensorflow as tf

# Object detection imports
#from utilss import backbone
from api import object_counting_api_kf
import pandas as pd

input_video = "./input_images_and_videos/10min_video.mp4" # Input video name with directory


fps = 30  # Change it with your input video fps
roi = [110, 700 ] # roi line position
deviation = 10 # Threshold value for box position (i.e how much deviate from roi position and must less than deviation) for car counting 
use_normalized_coordinates=False # Flag counter which define the coordinate is normalized or not i.e True means coordinate value between 0 & 1 and 
                                 # False means we using original coordinates

entry_cord = [1320,1450,1380,1640]      # Coordinates of extracted frame from original frame at entry point[y1, y2, x1, x2, width, height] 
exit_cord = [1936,2420,793,2052]        # Coordinates of extracted frame from original frame at exit point[y1, y2, x1, x2, width, height] 

height_exit = exit_cord[1]- exit_cord[0] # Height of the extracted frame of Exit
width_exit = exit_cord[3] - exit_cord[2] # Width of the extracted frame of Exit

height_entry = entry_cord[1]- entry_cord[0] # Height of the extracted frame of Entry
width_entry= entry_cord[3] - entry_cord[2] # Width of the extracted frame of Entry


line_thickness = 4 # Bounding box line thickness


object_counting_api_kf.car_counting(input_video, fps,  roi, deviation, use_normalized_coordinates, entry_cord, exit_cord, height_exit, width_exit, width_entry, height_entry, line_thickness)