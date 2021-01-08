import matplotlib.pyplot as plt
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




path = "D:/TESIS_IVAN/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/examples/output_maps/"


ori_imgs, tst_fn = read_all_imgs(path)

tst_imgs = []

for img in ori_imgs:
    tst_imgs.append(img)

print("Analizando " + str(len(tst_imgs)) + " imagenes")

#Classical color mask yellow and white
# cmask = []
# dst_bin = []
# for img in tst_imgs:
#     color_masked = color_mask(img, return_bin=False)
#     binary = color_mask(img, return_bin=True)
#     cmask.append(color_masked)
#     dst_bin.append(binary)
#
# for img1, img2 in zip(tst_imgs, cmask):
#     plot_two(img1, img2, 'Undistorted Image', 'HSV Mask', cmap2='gray')

# for img1, img2 in zip(tst_imgs, dst_bin):
#     plot_two(img1, img2, 'Undistorted Image', 'HSV Mask Binary', cmap2='gray')

#New color mask
# ncmask = []
# for img in tst_imgs:
#     new_color_masked = new_color_mask(img)
#     ncmask.append(new_color_masked)

# for img1, img2 in zip(tst_imgs, ncmask):
#     plot_two(img1, img2, 'Undistorted Image', '3 Color Spaces Mask', cmap2='gray')

# hls = []
# hsv = []
# for img in tst_imgs:
#     hls_masked=HLS(img)
#     hsv_masked=HSV(img)
#     hls.append(hls_masked)
#     hsv.append(hsv_masked)

cmask_2 = []
dst_bin_2 = []
for img in tst_imgs:
    color_masked_2 = color_mask_2(img, return_bin=False)
    binary_2 = color_mask_2(img, return_bin=True)
    cmask_2.append(color_masked_2)
    dst_bin_2.append(binary_2)

# for img1, img2, img3 in zip(tst_imgs, hsv, hls):
#     plot_three(img1, img2, img3, 'Undistorted Image', 'HSV Mask','HLS Mask', cmap2='gray', cmap3='gray')

# for img1, img2 in zip(tst_imgs, cmask_2):
#     plot_two(img1, img2, 'Undistorted Image', 'HSV Mask', cmap2='gray')

# for img1, img2 in zip(tst_imgs, dst_bin_2):
#     plot_two(img1, img2, 'Undistorted Image', 'HSV Mask Binary', cmap2='gray')

#New color mask 2
ncmask_2 = []
for img in tst_imgs:
    new_color_masked = new_color_mask_2(img)
    ncmask_2.append(new_color_masked)

# for img1, img2 in zip(tst_imgs, ncmask_2):
#     plot_two(img1, img2, 'Undistorted Image', '3 Color Spaces Mask', cmap2='gray')

# gauss= []
# for img in ncmask_2:
#     gauss_img = GAUSS(img, kernel_size = 3)
#     gauss.append(gauss_img)

# median= []
# for img in ncmask_2:
#     median_img = MEDIAN(img, kernel_size = 3)
#     median.append(median_img)

# for img1, img2, img3 in zip(tst_imgs, gauss, median):
#     plot_three(img1, img2, img3, 'Original Image', 'Gauss','Median', cmap2='gray', cmap3='gray')

# sobel_gauss = []
# sobel_median = []
# for img in ncmask_2:
#     sobel_g = sobel(img,sobel_kernel=5)
#     sobel_m = sobel(img,sobel_kernel=5)
#     sobel_gauss.append(sobel_g)
#     sobel_median.append(sobel_m)

# for img1, img2, img3 in zip(tst_imgs, sobel_gauss, sobel_median):
#     plot_three(img1, img2, img3, 'Original Image', 'Gauss','Median', cmap2='gray', cmap3='gray')

direction = []
for img in ncmask_2:
    dir_img = dir_thresh(img, kernel_size = 7, thresh = (-0.3,0.3))
    direction.append(dir_img)

# inverted = []
# for img in direction:
#     inv_img = inverse(img)
#     inverted.append(inv_img)

# for img1, img2 in zip(tst_imgs, direction):
#     plot_two(img1, img2, 'Original Image', 'Direction Binary', cmap2='gray')

median_bin = []
for img in direction:
    median_img = MEDIAN_BLUR(img, kernel_size = 11)
    #median_img = MEDIAN_BLUR(median_img, kernel_size = 3)
    median_img = inverse(median_img)
    median_bin.append(median_img)

# for img1, img2 in zip(tst_imgs, median_bin):
#     plot_two_points(img1, img2, 'Original Image', 'Dir-Med Binary', cmap2='gray')

perspective = []
Minverse = []
for img in median_bin:
    perspective_img,M,Minv = perspective_transform(img)
    perspective.append(perspective_img)
    Minverse.append(Minv)

# for img1, img2 in zip(median_bin, perspective):
#     plot_two_points(img1, img2, 'Original Image', 'Perspective',cmap1='gray', cmap2='gray')

histograms = []
for img in perspective:
    hist_img = hist(img)
    histograms.append(hist_img)

# for img1, img2 in zip(perspective, histograms):
#     plot_hist(img1, img2, 'Perspective',cmap1='gray')

#Processed images
raw = []
recursive = []
left_raw = []
right_raw = []
left_rec = []
right_rec = []
i = 1
for img in perspective:
    #out_img_raw = fit_polynomial(img)
    #print("Imagen: " + str(i))
    left_fit_raw, right_fit_raw, out_img_raw = find_line_raw(img, return_img=True, plot_boxes=True, plot_line=True)
    i += 1
    #left_fit_rec, right_fit_rec, out_img_rec = find_line_recursive(img, left_fit, right_fit, margin=100, return_img=True, plot_boxes=True, plot_line=True)
    raw.append(out_img_raw)
    #recursive.append(out_img_rec)
    left_raw.append(left_fit_raw)
    #left_rec.append(left_fit_rec)
    right_raw.append(right_fit_raw)
    #right_rec.append(right_fit_rec)

# for img1, img2 in zip(perspective,raw):
#      plot_two(img1, img2,'Perspective','Raw',cmap1='gray',cmap2='gray')

processed = []
i = 0
for img in raw:
    print("Imagen: " + str(i))
    processed_img = plot_lanelines(tst_imgs[i],perspective[i],left_raw[i], right_raw[i], Minverse[i], img)
    processed.append(processed_img)
    i += 1

for img1, img2 in zip(tst_imgs,processed):
    plot_two(img1, img2,'Original','Processed')

# processed = []
# i = 1
# for img in tst_imgs:
#     print("Imagen " + str(i))
#     color_masked = new_color_mask_2(img)
#     dir_img = dir_thresh(color_masked, kernel_size = 7, thresh = (-0.3,0.3))
#     median_img = MEDIAN_BLUR(dir_img, kernel_size = 11)
#     inverted_img = inverse(median_img)
#     perspective_img, M, Minv = perspective_transform(img)
#     left_fit_raw, right_fit_raw, out_img_raw = find_line_raw(perspective_img, return_img=True)
#     processed_img = plot_lanelines(img,perspective_img,left_raw, right_fit_raw, Minv, out_img_raw)
#     processed.append(processed_img)
#     i += 1
#
#
# for img1, img2 in zip(tst_imgs,processed):
#     plot_two(img1, img2,'Original','Procesed')
