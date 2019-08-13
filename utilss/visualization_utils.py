#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""

# Imports
import collections
import functools
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import tensorflow as tf
import cv2
import numpy
import os
import pandas as pd

# string utils - import
from utilss.string_utils import custom_string_util

# image utils - image saver import
from utilss.image_utils import image_saver

#  predicted_speed predicted_color module - import
from utilss.object_counting_module import object_counter
#  predicted_speed predicted_color module - import
from utilss.object_counting_module import object_counter_x_axis

# color recognition module - import
from utilss.color_recognition_module import color_recognition_api

from datetime import datetime

# Variables
is_vehicle_detected = [0]
ROI_POSITION = [0]
DEVIATION = [0]
is_color_recognition_enable = [0]
mode_number = [0]
x_axis = [0]

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

current_path = os.getcwd()


def draw_bounding_box_on_image_array(current_frame_number, image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box in normalized coordinates (same below).
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  is_vehicle_detected, csv_line, update_csv = draw_bounding_box_on_image(current_frame_number,image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))
  return is_vehicle_detected, csv_line, update_csv

def draw_bounding_box_on_image(current_frame_number,image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  csv_line = "" # to create new csv line consists of vehicle type, predicted_speed, color and predicted_direction
  update_csv = False # update csv for a new vehicle that are passed from ROI - just one new line for each vehicles
  is_vehicle_detected = [0]
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  
    

  predicted_direction = "n.a." # means not available, it is just initialization

  image_temp = numpy.array(image)
  detected_vehicle_image = image_temp[int(top):int(bottom), int(left):int(right)]

  '''if(bottom > ROI_POSITION): # if the vehicle get in ROI area, vehicle predicted_speed predicted_color algorithms are called - 200 is an arbitrary value, for my case it looks very well to set position of ROI line at y pixel 200'''
  if(x_axis[0] == 1):
    predicted_direction, is_vehicle_detected, update_csv = object_counter_x_axis.count_objects_x_axis(top, bottom, right, left, detected_vehicle_image, ROI_POSITION[0], ROI_POSITION[0]+DEVIATION[0], ROI_POSITION[0]+(DEVIATION[0]*2), DEVIATION[0])
  elif(mode_number[0] == 2):
    predicted_direction, is_vehicle_detected, update_csv = object_counter.count_objects(top, bottom, right, left, detected_vehicle_image, ROI_POSITION[0], ROI_POSITION[0]+DEVIATION[0], ROI_POSITION[0]+(DEVIATION[0]*2), DEVIATION[0])

  if(1 in is_color_recognition_enable):
    predicted_color = color_recognition_api.color_recognition(detected_vehicle_image)    
  
  try:
    font = ImageFont.truetype('arial.ttf', 16)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  if(1 in is_color_recognition_enable):
    display_str_list[0] = predicted_color + " " + display_str_list[0]
    csv_line = predicted_color + "," + str (predicted_direction) # csv line created
  else:
    display_str_list[0] = display_str_list[0]
    csv_line = str (predicted_direction) # csv line created
  
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height

  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin
    return is_vehicle_detected, csv_line, update_csv








def visualize_boxes_and_labels_on_image_array_x_axis(door,ids,current_frame_number,
                                              image,
                                              mode,                                              
                                              boxes,                                              
                                              use_normalized_coordinates,
                                              line_thickness,
                                              log_name,
					                          targeted_objects=None,
                                              x_reference=None,
                                              deviation=None,
                                              instance_masks=None,
                                              keypoints=None,
                                             
                                              max_boxes_to_draw=20,
                                              min_score_thresh=.5,
                                              agnostic_mode=False,
                                              ):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width], can
      be None
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  csv_line_util = "not_available"
  counter = 0
  print(image.shape[0])
  print(image.shape[1])
  im_width = image.shape[0]
  im_height = image.shape[1]
  ROI_POSITION.insert(0,x_reference)
  DEVIATION.insert(0,deviation)
  x_axis.insert(0,1)
  is_vehicle_detected = []
  mode_number.insert(0,mode)
  
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  
  
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
      #print(i,'i')
      #print(boxes[i])
      box = tuple(boxes[i].tolist())
      if not agnostic_mode:
          '''if classes[i] in category_index.keys():
            class_name = category_index[classes[i]]['name']        
          else:
            class_name = 'Car'''              
          display_str = 'ID:{}'.format(ids[i])
      else:
         pass
      class_name ='car' #display_str = 'score: {}%'.format(int(100 * scores[i]))        
      
      box_to_display_str_map[box].append(display_str)
      if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
      else:
          box_to_color_map[box] = 'Green'#STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]

  if(mode == 1):
    counting_mode = ""
  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    '''if instance_masks is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color
      )'''
        
    display_str_list=box_to_display_str_map[box]
    print(display_str_list[0][3:],'id')

    if(mode == 1 and targeted_objects == None):
      counting_mode = counting_mode + str(display_str_list)

    elif(mode == 1 and targeted_objects in display_str_list[0]):
      counting_mode = counting_mode + str(display_str_list)

    if ((targeted_objects != None) and (targeted_objects in display_str_list[0])):
            #if instance_masks is not None:
              #draw_mask_on_image_array(image, box_to_instance_masks_map[box], color=color)
        
            is_vehicle_detected, csv_line, update_csv = draw_bounding_box_on_image_array(current_frame_number,
                image,
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates) 
      
            #if keypoints is not None:
              #draw_keypoints_on_image_array(
                  #image,
                  #box_to_keypoints_map[box],
                  #color=color,
                 # radius=line_thickness / 2,
                 # use_normalized_coordinates=use_normalized_coordinates)

    elif (targeted_objects == None):
            #if instance_masks is not None:
              #draw_mask_on_image_array(image, box_to_instance_masks_map[box], color=color)

            is_vehicle_detected, csv_line, update_csv = draw_bounding_box_on_image_array(current_frame_number,
                image,
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates)
            
            (left, right, top, bottom) = (xmin , xmax , ymin , ymax )
            if(1 in is_vehicle_detected):
                conclusion_table = pd.read_csv(log_name, index_col = 'frame' )
                index=list(conclusion_table.index)
                if len(index)==0:
                    row_name = 1
                    conclusion_table.at[row_name, 'frame_no'] = current_frame_number
                    conclusion_table.at[row_name, 'frame_time'] = str(current_frame_number/30)
                    conclusion_table.at[row_name, 'current_Timestamp'] = str(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
                    conclusion_table.at[row_name, 'Object_type'] = 'car'    
                    conclusion_table.at[row_name, 'xmin'] = left
                    conclusion_table.at[row_name, 'ymin'] = top
                    conclusion_table.at[row_name, 'xmax'] = right
                    conclusion_table.at[row_name, 'ymax'] = bottom
                    conclusion_table.at[row_name, 'id'] = display_str_list[0][3:]
                    conclusion_table.at[row_name, 'door'] = door
                    conclusion_table.to_csv(log_name)
                else :
                    #print(index[-1])
                    row_name = index[-1] + 1
                if int(display_str_list[0][3:]) == conclusion_table.iloc[-1,-5]  and door==conclusion_table.iloc[-1,-4]  :
                    is_vehicle_detected.remove(1)
                
                else :    
                    conclusion_table.at[row_name, 'frame_no'] = current_frame_number
                    conclusion_table.at[row_name, 'frame_time'] = str(current_frame_number/13.68)
                    conclusion_table.at[row_name, 'current_Timestamp'] = str(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
                    conclusion_table.at[row_name, 'Object_type'] = 'car'    
                    conclusion_table.at[row_name, 'xmin'] = left
                    conclusion_table.at[row_name, 'ymin'] = top
                    conclusion_table.at[row_name, 'xmax'] = right
                    conclusion_table.at[row_name, 'ymax'] = bottom
                    conclusion_table.at[row_name, 'id'] = display_str_list[0][3:]
                    conclusion_table.at[row_name, 'door'] = door
                    conclusion_table.to_csv(log_name)
                
                
      
            

  if(1 in is_vehicle_detected):
        counter = 1
        del is_vehicle_detected[:]
        is_vehicle_detected = []        
        csv_line_util = class_name + "," + csv_line 

  if(mode == 1):
    counting_mode = counting_mode.replace("['", " ").replace("']", " ").replace("%", "")
    counting_mode = ''.join([i for i in counting_mode.replace("['", " ").replace("']", " ").replace("%", "") if not i.isdigit()])
    counting_mode = str(custom_string_util.word_count(counting_mode))
    counting_mode = counting_mode.replace("{", "").replace("}", "")
    #print('mode==1')

    return counter

  else:
    return counter
