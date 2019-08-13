#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

import tensorflow as tf
import csv
import cv2
import numpy as np
from utilss import visualization_utils as vis_util
from datetime import datetime
import pandas as pd
from sort.sort import *
from Person_det_track import pipeline


def car_counting_main(input_video, detection_graph, category_index,fps, width, height, roi, deviation, entry_cord, exit_cord, total_passed_vehicle ):
         

       
        # Creating New Csv file with Date and Time
       '''name = 'car_log_run_'+str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))+'.csv'
       with open(name, 'w') as f:
                writer = csv.writer(f)  
                csv_line = "frame, Timestamp, Object_type, car_count, exit, enter"                 
                writer.writerows([csv_line.split(',')])'''
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('project_complete1.avi', fourcc)
        fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
        output_movie2 = cv2.VideoWriter('project_complete2.avi', fourcc2, fps, (847, 180))
        # input video
        cap = cv2.VideoCapture(input_video)
        
        total_passed_vehicle = 0
        total_passed_vehicle2 = 0
        
        #width_heigh_taken = True
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()
                cv2.imwrite('frame1.jpg',frame)

                if not  ret:
                    print("end of the video file...")
                    break
                roi = 400
                input_frame = frame[1958:2480, 1181:2006].copy()
                #input_frame = cv2.resize(input_frame, (847,180), interpolation = cv2.INTER_AREA)

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                #image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                ''' (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})'''

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                _,boxess,ids=pipeline(input_frame.copy())
                print(boxess,'boxess')
                print(ids, 'ids')

                # Visualization of the results of a detection.        
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(1,ids,cap.get(1),
                                                                                                             input_frame,
                                                                                                             1,
                                                                                                             is_color_recognition_enabled,
                                                                                                             boxess,
                                                                                                            
                                                                                                             category_index,
                                                                                                             x_reference = roi,
                                                                                                             deviation = 10,
                                                                                                             use_normalized_coordinates=False,
                                                                                                             line_thickness=4)
                conclusion_table = pd.read_csv(r'D:\Ankit sir\Computer Vision\refine object_counting\tensorflow_object_counting_api-master - Copy/object_counting_report.csv', index_col='frame' )
                #conclusion_df = pd.DataFrame(conclusion_table)
                index = list(conclusion_table.index)
                #print(index, 'outer')
                total_passed_vehicle = total_passed_vehicle + counter

                # when the vehicle passed over line and counted, make the color of ROI line green
                if len(index) == 0 :
                    row_name = 1
                else :  
                    row_name = index[-1]
                row_name = row_name     
                if counter == 1:
                  cv2.line(input_frame, (roi, 0), (roi, 484), (0, 0xFF, 0), 5)
                  
                  conclusion_table.at[row_name, 'car_count'] = total_passed_vehicle
                  conclusion_table.at[row_name, 'exit(1)'] = 1
                  conclusion_table.at[row_name, 'entrance(2)'] = 0
                  conclusion_table.to_csv(r'D:\Ankit sir\Computer Vision\refine object_counting\tensorflow_object_counting_api-master - Copy/object_counting_report.csv')

                else:
                  cv2.line(input_frame, (roi, 0), (roi, 484), (0, 0, 0xFF), 5)

                #total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Cars: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )


                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi-10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )

                output_movie.write(input_frame)
                print ("writing frame")
                cv2.imshow('object counting_1',input_frame)
                cv2.imwrite('results_crop/'+str(cap.get(1))+'.jpg',input_frame)
                '''roi=700
                input_frame_1 = frame[ 1329:1509,1403:2250].copy() 

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded_1 = np.expand_dims(input_frame_1, axis=0)

                # Actual detection.
                (boxes_1, scores_1, classes_1, num_1) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded_1})

                
                #print(boxes_1, 'boxes_1')
                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter1, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(2,cap.get(1),
                                                                                                             input_frame_1,
                                                                                                             1,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(boxes_1),
                                                                                                             np.squeeze(classes_1).astype(np.int32),
                                                                                                             np.squeeze(scores_1),
                                                                                                             category_index,
                                                                                                             x_reference = roi,
                                                                                                             deviation = 10,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4
                                                                                                             )
                conclusion_table = pd.read_csv('object_counting_report.csv', index_col = 'frame')
                index=list(conclusion_table.index)
                total_passed_vehicle2 = total_passed_vehicle2 + counter1

                # when the vehicle passed over line and counted, make the color of ROI line green
                if len(index) == 0 :
                    row_name = 1
                else :  
                    row_name = index[-1]
                row_name = row_name     
                if counter1 == 1:
                  cv2.line(input_frame_1, (roi, 0), (roi, 180), (0, 0xFF, 0), 5)
                  #conclusion_table.at[row_name, 'Timestamp'] = str(datetime.now())
                  #conclusion_table.at[row_name, 'Object_type'] = 'car'
                  conclusion_table.at[row_name, 'car_count'] = total_passed_vehicle2
                  conclusion_table.at[row_name, 'exit(1)'] = 0
                  conclusion_table.at[row_name, 'entrance(2)'] = 2
                  conclusion_table.to_csv('D:/Ankit sir/tensorflow_object_counting_api-master - Copy/object_counting_report.csv')
                else:
                  cv2.line(input_frame_1, (roi, 0), (roi, 180), (0, 0, 0xFF), 5)

                #total_passed_vehicle2 = total_passed_vehicle2 + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame_1,
                    'Detected cars: ' + str(total_passed_vehicle2),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )


                cv2.putText(
                    input_frame_1,
                    'ROI Line',
                    (545, roi-10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )

                output_movie2.write(input_frame_1)
                print ("writing frame")
                cv2.imshow('object counting_2',input_frame_1)
                cv2.imwrite('results_crop/'+str(cap.get(1)+500)+'.jpg',input_frame_1)'''
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                ''''if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])'''         

            cap.release()
            cv2.destroyAllWindows()

def cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, roi, deviation):
        total_passed_vehicle = 0        

        #initialize .csv
        with open('object_counting_report.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
                writer.writerows([csv_line.split(',')])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        # input video
        cap = cv2.VideoCapture(input_video)

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

               # Visualization of the results of a detection.        
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_y_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             2,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(boxes),
                                                                                                             np.squeeze(classes).astype(np.int32),
                                                                                                             np.squeeze(scores),
                                                                                                             category_index,
                                                                                                             y_reference = roi,
                                                                                                             deviation = deviation,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)

                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:                  
                  cv2.line(input_frame, (0, roi), (width, roi), (0, 0xFF, 0), 5)
                else:
                  cv2.line(input_frame, (0, roi), (width, roi), (0, 0, 0xFF), 5)
                
                total_passed_vehicle = total_passed_vehicle + counter

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Vehicles: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )               
                
                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi-10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )

                output_movie.write(input_frame)
                print ("writing frame")
                #cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                '''if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         '''

            cap.release()
            cv2.destroyAllWindows()


def object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height):
        #initialize .csv
        with open('object_counting_report.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
                writer.writerows([csv_line.split(',')])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        # input video
        cap = cv2.VideoCapture(input_video)

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        height = 0
        width = 0
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if(len(counting_mode) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                else:
                    cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                
                output_movie.write(input_frame)
                print ("writing frame")
                #cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         

            cap.release()
            cv2.destroyAllWindows()

def targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_object, fps, width, height):
        #initialize .csv
        with open('object_counting_report.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
                writer.writerows([csv_line.split(',')])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))
        # input video
        cap = cv2.VideoCapture(input_video)

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        the_result = "..."
        width_heigh_taken = True
        height = 0
        width = 0
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      targeted_objects=targeted_object,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if(len(the_result) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                else:
                    cv2.putText(input_frame, the_result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                
                #cv2.imshow('object counting',input_frame)

                output_movie.write(input_frame)
                print ("writing frame")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         

            cap.release()
            cv2.destroyAllWindows()

def single_image_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height):     
        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        height = 0
        width = 0
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')            

       

        input_frame = cv2.imread(input_video)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(input_frame, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # insert information text to video frame
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Visualization of the results of a detection.        
        counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1,input_frame,
                                                                                              1,
                                                                                              is_color_recognition_enabled,
                                                                                              np.squeeze(boxes),
                                                                                              np.squeeze(classes).astype(np.int32),
                                                                                              np.squeeze(scores),
                                                                                              category_index,
                                                                                              use_normalized_coordinates=True,
                                                                                              line_thickness=4)
        if(len(counting_mode) == 0):
            cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
        else:
            cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
        
        cv2.imshow('tensorflow_object counting_api',input_frame)        
        cv2.waitKey(0)

        return counting_mode       

