import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import pickle
import random
import glob

import laneline
from image_functions import *
from pipeline import *

from collections import deque
from moviepy.editor import VideoFileClip




path = "D:/TESIS_IVAN/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/examples/primeras_salidas/output_map_1/detections_images/"


tst_imgs, tst_names = read_all_imgs(path)


print("Analizing " + str(len(tst_imgs)) + " images")






#New color mask 2
print("Applying Color Mask")
ncmask_2 = []
for img in tst_imgs:
    new_color_masked = new_color_mask_2(img)
    ncmask_2.append(new_color_masked)
print("Done")

print("Applying Direction Gradient")
direction = []
for img in ncmask_2:
    dir_img = dir_thresh(img, kernel_size = 7, thresh = (-0.3,0.3))
    direction.append(dir_img)
print("Done")

print("Applying Median Filter")
median_bin = []
for img in direction:
    median_img = MEDIAN_BLUR(img, kernel_size = 11)
    #median_img = MEDIAN_BLUR(median_img, kernel_size = 3)
    median_img = inverse(median_img)
    median_bin.append(median_img)
print("Done")

perspective = []
Minverse = []
print("Applying Perspective Transform")
for img in median_bin:
    perspective_img,M,Minv = perspective_transform(img)
    perspective.append(perspective_img)
    Minverse.append(Minv)
print("Done")

#Processed images
raw = []
left_raw = []
right_raw = []
print("Detecting Lines")
for img in perspective:
    left_fit_raw, right_fit_raw, out_img_raw = find_line_raw(img, return_img=True, plot_boxes=True, plot_line=True)
    raw.append(out_img_raw)
    left_raw.append(left_fit_raw)
    right_raw.append(right_fit_raw)
print("Done")


print("Processing Images")
processed = []
i = 0
for img in raw:
    print("Image: " + str(i))
    processed_img = plot_lanelines(tst_imgs[i],perspective[i],left_raw[i], right_raw[i], Minverse[i], img)
    mpimg.imsave('primeras_salidas/output_map_1/lanes_and_detections/' + tst_names[i], processed_img)
    processed.append(processed_img)
    i += 1
print("Done")
