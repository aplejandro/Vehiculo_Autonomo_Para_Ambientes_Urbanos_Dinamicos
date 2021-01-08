import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import pickle
import random
import glob
import csv

import laneline
from image_functions import *
from pipeline import *

from collections import deque
from moviepy.editor import VideoFileClip

def color_mask(img):
    """
    New color mask fucntion that combines three different color space filters
    to increasee detection accuracy.
    """

    hsv = HSV(img)
    hls = HLS(img)
    binary = np.zeros_like(img[:, :, 0])

    # Green filter
    #RGB
    lower_green = (90,235,70)
    upper_green = (210,255,170)
    rgb_g_mask = cv2.inRange(img, lower_green, upper_green)
    #HSV
    lower_green_hsv = (45, 80, 230)
    upper_green_hsv = (60, 180, 255)
    hsv_g_mask = cv2.inRange(hsv, lower_green_hsv, upper_green_hsv)
    #HLS
    lower_green_hls = (45,150,205)
    upper_green_hls = (60,220,255)
    hls_g_mask = cv2.inRange(hls, lower_green_hls, upper_green_hls)
    mask_g = cv2.bitwise_and(hsv_g_mask, hls_g_mask, rgb_g_mask)
    green_filter = binary | mask_g

    # Red filter
    #RGB
    lower_red = (240,150,75)
    upper_red = (255,180,95)
    rgb_r_mask = cv2.inRange(img, lower_red, upper_red)
    #HSV
    lower_red_hsv = (0, 155, 240)
    upper_red_hsv = (20, 195, 255)
    hsv_r_mask = cv2.inRange(hsv, lower_red_hsv, upper_red_hsv)
    #HLS
    lower_red_hls = (0,160,240)
    upper_red_hls = (20,190,255)
    hls_r_mask = cv2.inRange(hls, lower_red_hls, upper_red_hls)

    mask_r = cv2.bitwise_and(hsv_r_mask, hls_r_mask, rgb_r_mask)
    red_filter = binary | mask_r

    return green_filter, red_filter

def HLS(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def read_all_imgs(path, color=cv2.IMREAD_COLOR):

    images = []
    filenames = []

    filelist = os.listdir(path)
    for file in filelist:

        try:
            img = cv2.imread(path + file, color)
        except:
            img = None

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            filenames.append(file)

    return images, filenames

def read_all_txt(path):

    filenames = []

    filelist = os.listdir(path)
    for file in filelist:
        filenames.append(path + file)

    return filenames

def is_file_empty(file_path):
    """ Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0

def traffic_light_state(img, y_top, x_left, y_bottom, x_right):
    ROI = img[y_top:y_bottom, x_left:x_right]
    green, red = color_mask(ROI)
    nz_green = green.nonzero()
    nz_red = red.nonzero()
    nzg = nz_green[0]
    nzr = nz_red[0]
    if len(nzg) > len(nzr):
        return True
    else:
        return False

def all_trafic_light_state(img, y_top, x_left, y_bottom, x_right):
    if x_left > 300 and x_left < 1200:
        base = x_right - x_left
        height = y_bottom - y_top
        area = base * height
        if area > 140:
            ROI = img[y_top:y_bottom, x_left:x_right]
            green, red = color_mask(ROI)
            nz_green = green.nonzero()
            nz_red = red.nonzero()
            nzg = nz_green[0]
            nzr = nz_red[0]
            if len(nzg) > len(nzr):
                return 1
            else:
                return -1
        else:
            return 0
    else:
        return 0

print("Reading images...")
path_detected = "D:/TESIS_IVAN/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/examples/output_map_2/detections_images/"
path_original = "D:/TESIS_IVAN/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/examples/output_map_2/rgb_3/"
path_txt = "D:/TESIS_IVAN/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/examples/output_map_2/detections_txt/"


detected_images, detected_images_names = read_all_imgs(path_detected)
original_images, original_images_names = read_all_imgs(path_original)
txt_names = read_all_txt(path_txt)


print("Analizing " + str(len(detected_images)) + " detected images")
print("Analizing " + str(len(original_images)) + " original images")
print("Analizing " + str(len(txt_names)) + " text files")

# for i in range(len(txt_names)):
#     name = original_images_names[i].split('/')[-1]
#     print("Imagen: " + name)
#     is_empty = is_file_empty(txt_names[i])
#     if is_empty:
#         print("Traffic Light Not Detected")
#         continue
#     else:
#         with open(txt_names[i], 'r') as file:
#             reader = csv.reader(file)
#             tl_flag = False
#             for row in reader:
#                 if row[0] == "Traffic Light":
#                     traffic_light_row = row
#                     print(traffic_light_row)
#                     tl_flag = True
#                     break
#             if tl_flag:
#                 if int(traffic_light_row[2]) > 300 and int(traffic_light_row[2]) < 1000:
#                     base = int(traffic_light_row[4]) - int(traffic_light_row[2])
#                     height = int(traffic_light_row[3]) - int(traffic_light_row[1])
#                     area = base * height
#                     if area < 140:
#                         print("Traffic Light Detected but too small to analyze")
#                     else:
#                         state = traffic_light_state(original_images[i],int(traffic_light_row[1]),int(traffic_light_row[2]),int(traffic_light_row[3]),int(traffic_light_row[4]))
#                         if state:
#                             print("Traffic Light Detected: Go")
#                         else:
#                             print("Traffic Light Detected: Stop")
#                 else:
#                     print("Traffic Light Detected but not in range")
#             else:
#                 print("Traffic Light Not Detected")


font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
thickness = 2

for i in range(len(txt_names)):
    name = detected_images_names[i].split('/')[-1]
    print("Imagen: " + name)
    is_empty = is_file_empty(txt_names[i])
    if is_empty:
        print("Traffic Light Not Detected")
        new_image = cv2.putText(detected_images[i], 'Traffic Light Not Detected', org, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
        continue
    else:
        with open(txt_names[i], 'r') as file:
            reader = csv.reader(file)
            traffic_light_row = []
            tl_flag = False
            for row in reader:
                if row[0] == "Traffic Light":
                    tl_flag = True
                    traffic_light_row.append(row)
            if tl_flag:
                sum_state = 0
                for tl in traffic_light_row:
                    state = all_trafic_light_state(original_images[i],int(tl[1]),int(tl[2]),int(tl[3]),int(tl[4]))
                    #print(state)
                    sum_state += state
                if sum_state == 0:
                    print("Traffic Light detected but too small to analyze or not in range")
                    new_image = cv2.putText(detected_images[i], 'Traffic Light Detected but too small to analyze or not in range', org, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
                elif sum_state > 0:
                    print("Traffic Light Detected: Go")
                    new_image = cv2.putText(detected_images[i], 'Traffic Light Detected: Go', org, font, fontScale, (0,255,0), thickness, cv2.LINE_AA)
                else:
                    print("Traffic Light Detected: Stop")
                    new_image = cv2.putText(detected_images[i], 'Traffic Light Detected: Stop', org, font, fontScale, (255,0,0), thickness, cv2.LINE_AA)
            else:
                print("Traffic Light Not Detected")
                new_image = cv2.putText(detected_images[i], 'Traffic Light Not Detected', org, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
    mpimg.imsave('output_map_2/Final/' + name, new_image )
