import cv2
import sys
import csv
import math
from skimage import morphology
import os
import numpy as np
import matplotlib.pyplot as plt

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

def GAUSS(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def read_all_txt(path):

    filenames = []

    filelist = os.listdir(path)
    for file in filelist:
        filenames.append(path + file)

    return filenames

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

# def traffic_light_state(img, y_top, x_left, y_bottom, x_right):
#     plt.subplot(231),plt.imshow(img),plt.title('Original')
#     plt.show()
#     ROI = img[y_top:y_bottom, x_left:x_right]
#     plt.subplot(232),plt.imshow(ROI),plt.title('ROI')
#     plt.show()
#     gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
#     plt.subplot(233),plt.imshow(gray,'gray'),plt.title('Gray')
#     plt.show()
#     gauss = GAUSS(gray, kernel_size = 3)
#     ret, binary = cv2.threshold(gauss,200,255,cv2.THRESH_BINARY)
#     k = np.ones((3,3),np.uint8)
#     plt.subplot(234),plt.imshow(binary,'gray'),plt.title('Binary')
#     plt.show()
#     opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
#     median = cv2.medianBlur(opening,3)
#     plt.subplot(235),plt.imshow(median,'gray'),plt.title('Binary')
#     plt.show()

def traffic_light_state(img, y_top, x_left, y_bottom, x_right):
    plt.subplot(231),plt.imshow(img),plt.title('Original')
    plt.show()
    ROI = img[y_top:y_bottom, x_left:x_right]
    plt.subplot(232),plt.imshow(ROI),plt.title('ROI')
    plt.show()
    green, red = color_mask(ROI)
    plt.subplot(233),plt.imshow(green, 'gray'),plt.title('Green Filter')
    plt.show()
    plt.subplot(234),plt.imshow(red, 'gray'),plt.title('Red Filter')
    plt.show()
    nz_green = green.nonzero()
    nz_red = red.nonzero()
    nzg = nz_green[0]
    nzr = nz_red[0]
    # print(len(nzg))
    # print(len(nzr))
    if len(nzg) > len(nzr):
        return True
    else:
        return False



def plot_three(img1, img2, img3, title1='Img_1', title2='Img_2', title3='Img_3', cmap1=None, cmap2=None, cmap3=None):
    implot = plt.figure(figsize=(32, 32))

    ax1 = implot.add_subplot(3, 1, 1)
    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(img1, cmap=cmap1)
    ax1.set_title(title1)

    ax2 = implot.add_subplot(3, 1, 2)
    ax2.grid(False)
    ax2.axis('off')
    ax2.imshow(img2, cmap=cmap2)
    ax2.set_title(title2)

    ax3 = implot.add_subplot(3, 1, 3)
    ax3.grid(False)
    ax3.axis('off')
    ax3.imshow(img3, cmap=cmap3)
    ax3.set_title(title3)

    plt.show()

def HLS(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

path_detected = "D:/TESIS_IVAN/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/examples/output_map_3/detections_images/"
path_original = "D:/TESIS_IVAN/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/examples/output_map_3/rgb_3/"
path_txt = "D:/TESIS_IVAN/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/examples/output_map_3/detections_txt/"


detected_images, detected_images_names = read_all_imgs(path_detected)
original_images, original_images_names = read_all_imgs(path_original)
txt_names = read_all_txt(path_txt)

#Imagen 00005683
img_i = 200
detected = detected_images[img_i]
name = detected_images_names[img_i]
original = original_images[img_i]
text = txt_names[img_i]
hsv = HSV(original)
hls = HLS(original)

plot_three(original, hsv, hls,'Original', 'HSV', 'HLS',cmap2 = 'gray', cmap3='gray')
print("Image: " + name)

with open(text, 'r') as file:
    reader = csv.reader(file)
    tl_flag = False
    for row in reader:
        if row[0] == "Traffic Light":
            traffic_light_row = row
            tl_flag =  True
            break
    if tl_flag:
        if int(traffic_light_row[2]) > 300 and int(traffic_light_row[2]) < 1000:
            base = int(traffic_light_row[4]) - int(traffic_light_row[2])
            height = int(traffic_light_row[3]) - int(traffic_light_row[1])
            area = base * height
            if area < 140:
                print("Traffic Light Detected but too small to analize or not in the correct orientation")
                #print(traffic_light_row)
            else:
                state = traffic_light_state(original,int(traffic_light_row[1]),int(traffic_light_row[2]),int(traffic_light_row[3]),int(traffic_light_row[4]))
                if state:
                    print("Traffic Light Detected: Go")
                else:
                    print("Traffic Light Detected: Stop")
    else:
        print("Traffic Light Not Detected")
