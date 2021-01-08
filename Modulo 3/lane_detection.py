#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image
#%matplotlib inline

import math
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[225,0, 0], thickness=15):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    x_left = []
    y_left = []
    x_right = []
    y_right = []
    ysize = img.shape[0]
    xsize = img.shape[1]
    y_top = int(ysize*0.65) # need y coordinates of the top and bottom of left and right lane
    y_bottom = int(ysize) #  to calculate x values once a line is found

    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                m = float(((y2-y1)/(x2-x1)))
                if (m > 0.5): # left lane, slope > tan(26.52)
                        x_left.append(x1)
                        x_left.append(x2)
                        y_left.append(y1)
                        y_left.append(y2)
                if (m < -0.5): # Right lane, slope < tan(153.48)
                        x_right.append(x1)
                        x_right.append(x2)
                        y_right.append(y1)
                        y_right.append(y2)
        # only execute if there are points found that meet criteria, this eliminates borderline cases i.e. rogue frames
        if (x_left!=[]) and (y_left!=[]) and (x_right!=[]) and (y_right!=[]):
            left_line_coeffs = np.polyfit(x_left, y_left, 1)
            left_x_top = int((y_top - left_line_coeffs[1])/left_line_coeffs[0])
            left_x_bottom = int((y_bottom - left_line_coeffs[1])/left_line_coeffs[0])
            right_line_coeffs = np.polyfit(x_right, y_right, 1)
            right_x_top = int((y_top - right_line_coeffs[1])/right_line_coeffs[0])
            right_x_bottom = int((y_bottom - right_line_coeffs[1])/right_line_coeffs[0])
            cv2.line(img, (left_x_top, y_top), (left_x_bottom, y_bottom), color, thickness)
            cv2.line(img, (right_x_top, y_top), (right_x_bottom, y_bottom), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def show_imgs(original_img, processed_img):
    plt.figure(figsize=(20,10))
    plt.subplot(121),plt.imshow(original_img)
    plt.title('Original Image')
    plt.subplot(122),plt.imshow(processed_img)
    plt.title('Processed Image')
    plt.show()

def detection_pipeline(image, kernel_size=5, canny_low=100, canny_high=150, rho=10, theta=np.pi/180,
                       threshold=200, min_line_len=150, max_line_gap=50):
    gray = grayscale(image)
    blur = gaussian_blur(gray, kernel_size)
    canny_edges = canny(blur, canny_low, canny_high)
    ysize = image.shape[0]
    xsize = image.shape[1]
    left_bottom = [0.05*xsize, ysize]
    right_bottom = [0.95*xsize , ysize]
    left_top = [xsize*0.4,ysize*0.65]
    right_top = [xsize*0.7,ysize*0.7]
    vertices = np.array([[left_bottom,left_top,right_top, right_bottom]], np.int32)
    roi = region_of_interest(canny_edges, [vertices])
    hough = hough_lines(roi, rho, theta, threshold, min_line_len, max_line_gap)
    final_img = weighted_img(hough, image, α=0.8, β=1., γ=0.)
    return final_img

test_images = os.listdir("output_map_1/rgb_3/")
print(test_images)
path = 'output_map_1/rgb_3/'
for img_name in test_images:
    print("Processing image: " + img_name)
    img = cv2.imread('output_map_1/rgb_3/' + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lane_detected_imgs = detection_pipeline(img)
    mpimg.imsave('output_map_1/lanes_detection/' + img_name, lane_detected_imgs)
print("Done")
show_imgs(img, lane_detected_imgs)
