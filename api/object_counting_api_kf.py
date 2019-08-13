
import tensorflow as tf
import csv
import cv2
import numpy as np
from utilss import visualization_utils as vis_util
from datetime import datetime
import pandas as pd
from car_det_track import pipeline 



def car_counting(input_video, fps, roi, deviation,use_normalized_coordinates, entry_cord, exit_cord, height_exit, width_exit, width_Entry, height_Entry, line_thickness):
        
        # Variables
        total_passed_vehicle_Exit = 0 # using for total count of car throughout video or camera input to count vehicles at Exit
        total_passed_vehicle_Entry = 0 # using for total count of car throughout video or camera input to count vehicles at Entry
        time_stamp = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
       
        #initialize .csv
        log_name = 'csv_log/'+'car_log_run_'+ time_stamp +'.csv'
        with open(log_name, 'w') as f:
                writer = csv.writer(f)  
                csv_line = "frame,frame_no,frame_time,current_Timestamp,Object_type,xmin,ymin,xmax,ymax,id,door,car_count,exit(1),entrance(2)"                 
                writer.writerows([csv_line.split(',')])

        #initializing Video Writer for Entry point
        video_name = 'video_log/Entry/'+'car_log_run_'+ time_stamp +'.avi'
        fourcc_entry = cv2.VideoWriter_fourcc(*'XVID')
        output_movie_Entry = cv2.VideoWriter(video_name, fourcc_entry, fps, (width_Entry, height_Entry))
        
        #initializing Video Writer for Exit point
        video_name = 'video_log/Exit/'+'car_log_run_'+ time_stamp +'.avi'
        fourcc_exit = cv2.VideoWriter_fourcc(*'XVID')
        output_movie_Exit = cv2.VideoWriter(video_name, fourcc_exit, fps, (width_exit, height_exit))
        
        # Intializing VideoCapture object
        cap = cv2.VideoCapture(input_video)
        
        
        # for all the frames that are extracted from input video
        while(cap.isOpened()):
                ret, frame = cap.read() # frame is image from video and ret contain boolean value True or False if frame read correctly then it return True otherwise False
                

                if not  ret:
                    print("end of the video file...")
                    break

#------------------------------------------------------------------------------------------------------------------------------------------------------------------                
                # Processing frame for Exit Point
                
                roi_exit = int(roi[1])
                
                # Extracting Frame from original frame for Exit Point for Processing
                input_frame_Exit = frame[exit_cord[0]:exit_cord[1],exit_cord[2]:exit_cord[3] ].copy()
                
                # Kalman filter detected boxes coordinates output with ids
                
                _,boxes_exit,ids_exit=pipeline(input_frame_Exit.copy()) 
                

                # Visualization of the results of a detection and return counter .        
                counter = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(1,ids_exit,cap.get(1), input_frame_Exit,
                                                                                                             1,
                                                                                                             boxes_exit,                                                                                                             
                                                                                                             use_normalized_coordinates,
                                                                                                             line_thickness,
                                                                                                             log_name,
                                                                                                             x_reference = roi_exit,
                                                                                                             deviation = deviation)
                # Importing csv log file
                conclusion_table = pd.read_csv(log_name, index_col='frame' )
                
                # loading index for counting no. of record in the file 
                index = list(conclusion_table.index)
                
                # counting total car passing through roi line and it depend on the counter i.e it can be 0, 1
                total_passed_vehicle_Exit = total_passed_vehicle_Exit + counter

                # Assigning the row_name based on the index length 
                if len(index) == 0 :
                    row_name = 1    # row_name is one if index length is zero
                else :  
                    row_name = index[-1] # if record is exist than row_name will be last record row_name
                
                row_name = row_name   # Assigning row name
                
                
                if counter == 1:
                  cv2.line(input_frame_Exit, (roi_exit, 0), (roi_exit, 484), (0, 0xFF, 0), 5) # when the vehicle passed over line and make the color of ROI line green if counter is 1
                  
                  # Saving values to csv file at appropiate column name 
                  conclusion_table.at[row_name, 'car_count'] = total_passed_vehicle_Exit
                  conclusion_table.at[row_name, 'exit(1)'] = 1
                  conclusion_table.at[row_name, 'entrance(2)'] = 0
                  conclusion_table.to_csv(log_name)

                else:
                  cv2.line(input_frame_Exit, (roi_exit, 0), (roi_exit, 484), (0, 0, 0xFF), 5) # if counter is zero then it keep roi line color red

                

                # insert information text to video frame
                
                # Assigning Font Style
                font = cv2.FONT_HERSHEY_SIMPLEX 
                
                # Writing text on the frame
                cv2.putText(input_frame_Exit,'Detected Cars: ' + str(total_passed_vehicle_Exit), (10, 35), font, 0.4, (0, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_SIMPLEX,) 
            
                # Writing frame and making video of it
                output_movie_Exit.write(input_frame_Exit)
                
                # Displaying frame in separate window called "Exit Point"
                cv2.imshow('Exit Point',input_frame_Exit)
                
                
 #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------               
                
                
                '''#  Processing Frame-by-Frame for Entry Point 
                
                roi_entry= int(roi[0])
                
                # Extracting Frame from original frame for Entry Point for Processing
                input_frame_Entry = frame[entry_cord[0]:entry_cord[1],entry_cord[2]:entry_cord[3] ].copy()
                #input_frame = cv2.resize(input_frame, (847,180), interpolation = cv2.INTER_AREA)

                # Kalman filter detected boxes coordinates output with ids
                _,boxes_entry,ids_entry=pipeline(input_frame_Entry.copy()) 
                

                # Visualization of the results of a detection and return counter .        
                counter = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(2, ids_entry, cap.get(1), input_frame_Entry,
                                                                                                             1,
                                                                                                             boxes_entry,                                                                                                             
                                                                                                             use_normalized_coordinates,
                                                                                                             line_thickness,
                                                                                                             log_name,
                                                                                                             x_reference = roi_entry,
                                                                                                             deviation = deviation)
                
                # counting total car passing through roi line and it depend on the counter i.e it can be 0, 1 
                #(i.e 0 means no car pass through roi line and 1 means means car pass through roi line)
                total_passed_vehicle_Entry = total_passed_vehicle_Entry + counter
                
                # Importing csv log file
                conclusion_table = pd.read_csv(log_name, index_col='frame' )
                
                # loading index for counting no. of record in the file 
                index = list(conclusion_table.index)
                
                

                # Assigning the row_name based on the index length 
                if len(index) == 0 :
                    row_name = 1    # row_name is one if index length is zero
                else :  
                    row_name = index[-1] # if record is exist than row_name will be last record row_name
                
                row_name = row_name   # Assigning row name  
                
                if counter == 1:
                  cv2.line(input_frame_Entry, (roi_entry, 0), (roi_entry, 484), (0, 0xFF, 0), 5) # when the vehicle passed over line and make the color of ROI line green if counter is 1
                  
                  # Saving values to csv file at appropiate column name 
                  conclusion_table.at[row_name, 'car_count'] = total_passed_vehicle_Entry
                  conclusion_table.at[row_name, 'exit(1)'] = 0
                  conclusion_table.at[row_name, 'entrance(2)'] = 2
                  conclusion_table.to_csv(log_name)

                else:
                  cv2.line(input_frame_Entry, (roi_entry, 0), (roi_entry, 484), (0, 0, 0xFF), 5) # if counter is zero then it keep roi line color red

                

                # insert information text to video frame
                
                # Assigning Font Style
                font = cv2.FONT_HERSHEY_SIMPLEX 
                
                # Writing text on the frame
                cv2.putText(input_frame_Entry,'Detected Cars: ' + str(total_passed_vehicle_Entry), (10, 35), font, 0.4, (0, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_SIMPLEX,) 
            
                # Writing frame and making video of it
                output_movie_Entry.write(input_frame_Entry)
                
                # Displaying frame in separate window called "Entry Point"
                cv2.imshow('Entry Point',input_frame_Entry)'''
                
                # Defination of terminate Process by pressing key in our case is 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

              

        cap.release()
        cv2.destroyAllWindows()

