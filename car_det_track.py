#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""@author: ambakick
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
#from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

import helpers
from detector import PersonDetector
import tracker
import cv2
detect = PersonDetector()
# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

max_age = 0  # no.of consecutive unmatched detection before 
             # a track is deleted

min_hits = 1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque([str(i) for i in range(10000)])

debug = False

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.5):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            #print(det,'det')
            IOU_mat[t,d] = helpers.box_iou2(trk,det)
            #print(t,'t', d,'d')
            #print(IOU_mat[t,d],"IOU")
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    #print(IOU_mat,'all_iou')
    matched_idx = linear_assignment(-IOU_mat)        
    #print(matched_idx,'matched_idx')
    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
       # print(matches, 'matches=0')
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       
    


def pipeline(img):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug
    
    frame_count+=1
    
    img_dim = (img.shape[1], img.shape[0])
    z_box = detect.get_localization(img)
    #print(z_box)# Real coordinates  measurement
    if debug:
       print('Frame:', frame_count)
       
    x_box =[]
    if debug: 
        for i in range(len(z_box)):
           img1= helpers.draw_box_label(img, z_box[i], box_color=(255, 0, 0))
           plt.imshow(img1)
        plt.show()
    #print(tracker_list)
    if len(tracker_list) > 0:
        for trk in tracker_list:
            #print(trk.box)
            x_box.append(trk.box)
    
    #print(len(tracker_list))
    #print(x_box,'x_box')
    matched, unmatched_dets, unmatched_trks \
    = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.5)  
    if debug:
         print('Detection: ', z_box)
         print('x_box: ', x_box)
         print('matched:', matched)
         print('unmatched_det:', unmatched_dets)
         print('unmatched_trks:', unmatched_trks)
    
    #print(matched,'matched')     
    # Deal with matched detections     
    if matched.size >0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.hits += 1
            #print(tmp_trk,'matched.size>0')
    
    # Deal with unmatched detections      
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker() # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            #print(x, 'x in unmatched')
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            #print(tmp_trk.id)
            tracker_list.append(tmp_trk)
            x_box.append(xx)
            #print(x_box,'unmatched_dets >0')
    
    # Deal with unmatched tracks       
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box =xx
            x_box[trk_idx] = xx
            #print(x_box, 'unmatched_trks')
                   
    #print(tracker_list, 'after unmatched_trks')   
    # The list of tracks to be annotated  
    good_tracker_list =[]
    boxes=[]
    ids=[]
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
             good_tracker_list.append(trk)
             boxes.append(trk.box)
             ids.append(trk.id)
             x_cv2 = trk.box
             #print(x_cv2,'x_cv2')
             if debug:
                 print('updated box: ', x_cv2)
                 print()
             img= helpers.draw_box_label(trk.id,img, x_cv2) # Draw the bounding boxes on the 
                                             # images
    #print(boxes, 'boxes')
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  
    
    for trk in deleted_tracks:
            track_id_list.append(trk.id)
    
    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    
    if debug:
       print('Ending tracker_list: ',len(tracker_list))
       print('Ending good tracker_list: ',len(good_tracker_list))
    
    #cv2.imshow("frame",img)
    return img,np.array(boxes),ids
    
