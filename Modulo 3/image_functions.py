import os
import cv2
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


def HLS(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def HLS_RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_HLS2RGB)


def HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def HSV_RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def GRAY(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def GAUSS(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def MEDIAN_BLUR(img,kernel_size):
    return cv2.medianBlur(img, kernel_size)

def inverse(img):
    # for row in range(img.shape[0]):
    #     for px in range(img.shape[1]):
    #         if img[row][px] == 0:
    #             img[row][px] = 1
    #         else:
    #             img[row][px] = 0
    return cv2.bitwise_not(img) - 254


def calibrate_camera(images, pattern):
    """
    Function to calibrate camera based on multiple images of a checkerboard pattern.
    Source:
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    Inputs:
    images - Numpy array of all calibration images
    pattern - tuple with the Checkerboard pattern from calibration images in the format (r, c) where
              r is the number of crossings per row and c is the number of crossings per column
    """
    # termination criteria to refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Create a meshgrid of points
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    # Create Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for img in images:
        gray = GRAY(img)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern, None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Now that we have our object points and corners we can perform our camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs


def undistort_image(image, camera_params):

    ret, mtx, dist, rvecs, tvecs = camera_params

    dst = cv2.undistort(image, mtx, dist, None, mtx)

    return dst


def correct_and_plot(image, camera_params):
    implot = plt.figure(figsize=(16, 16))

    ax1 = implot.add_subplot(1, 2, 1)
    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(image)
    ax1.set_title('Original Image')

    ax2 = implot.add_subplot(1, 2, 2)
    ax2.grid(False)
    ax2.axis('off')
    ax2.imshow(undistort_image(image, camera_params))
    ax2.set_title('Undistorted Image')

    plt.show()

def plot_hist(img1, hist,title1='Img_1', cmap1=None):
    implot = plt.figure(figsize=(16, 16))

    ax1 = implot.add_subplot(1, 2, 1)
    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(img1, cmap=cmap1)
    ax1.set_title(title1)

    ax2 = implot.add_subplot(1, 2, 2)
    ax2.grid(False)
    ax2.axis('off')
    ax2.plot(hist)

    plt.show()


def plot_two(img1, img2, title1='Img_1', title2='Img_2', cmap1=None, cmap2=None):
    implot = plt.figure(figsize=(16, 16))

    ax1 = implot.add_subplot(1, 2, 1)
    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(img1, cmap=cmap1)
    ax1.set_title(title1)

    ax2 = implot.add_subplot(1, 2, 2)
    ax2.grid(False)
    ax2.axis('off')
    ax2.imshow(img2, cmap=cmap2)
    ax2.set_title(title2)

    plt.show()

def plot_two_points(img1, img2, title1='Img_1', title2='Img_2', cmap1=None, cmap2=None):
    implot = plt.figure(figsize=(16, 16))

    ax1 = implot.add_subplot(1, 2, 1)
    ax1.grid(False)
    ax1.axis('off')
    ax1.imshow(img1, cmap=cmap1)
    # src[0] = [np.uint(img_w * 0.375), np.uint(img_h * 2 / 3)]
    # src[1] = [np.uint(img_w * 0.625), np.uint(img_h * 2 / 3)]
    # src[2] = [img_w, img_h]
    # src[3] = [0, img_h]
    #src = np.float32([[466., 510.],[160.,720.],[1097.,720.],[791., 510.]])
    ax1.plot(450,510,'.', color = (1,0,0))
    ax1.plot(160,720,'.',color = (1,0,0))
    ax1.plot(1097,720,'.',color = (1,0,0))
    ax1.plot(806,510,'.',color = (1,0,0))
    ax1.set_title(title1)

    ax2 = implot.add_subplot(1, 2, 2)
    ax2.grid(False)
    ax2.axis('off')
    ax2.imshow(img2, cmap=cmap2)
    # ax2.plot(791,479,'.', color = (1,0,0))
    # ax2.plot(1069,714,'.', color = (1,0,0))
    # ax2.plot(188,714,'.', color = (1,0,0))
    # ax2.plot(466,479,'.', color = (1,0,0))
    ax2.set_title(title2)

    plt.show()

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

def color_mask(img, return_bin=False):
    """
    This function will convert an image to HSV color space and apply two different masks:
    one for the white and another one for yellow and will return a color masked image in the HSV color space.
    """

    hsv = HSV(img)

    # White Filter:
    lower_w = np.array([0, 0, 200])
    upper_w = np.array([180, 25, 255])

    hsv_w_mask = cv2.inRange(hsv, lower_w, upper_w)

    # Yellow Filter
    lower_y = np.array([15, 75, 75])
    upper_y = np.array([35, 255, 255])

    hsv_y_mask = cv2.inRange(hsv, lower_y, upper_y)

    mask = cv2.bitwise_or(hsv_y_mask, hsv_w_mask)

    color_masked = cv2.bitwise_and(hsv,hsv,mask=mask)

    if return_bin:
        gray = GRAY(color_masked)
        binary = np.zeros_like(gray)
        binary[(gray > 0)] = 1
        return binary

    return color_masked

def color_mask_2(img, return_bin=False):
    """
    This function will convert an image to HSV color space and apply two different masks:
    one for the white and another one for yellow and will return a color masked image in the HSV color space.
    """

    hsv = HSV(img)

    # White Filter:
    lower_w = (15, 5, 200)
    upper_w = (120, 25, 255)

    hsv_w_mask = cv2.inRange(hsv, lower_w, upper_w)


    # Yellow Filter
    lower_y = (19, 50, 150)
    upper_y = (35, 215, 255)

    hsv_y_mask = cv2.inRange(hsv, lower_y, upper_y)

    mask = cv2.bitwise_or(hsv_y_mask, hsv_w_mask)

    color_masked = cv2.bitwise_and(hsv,hsv,mask=mask)

    if return_bin:
        gray = GRAY(color_masked)
        binary = np.zeros_like(gray)
        binary[(gray > 0)] = 1
        return binary

    return color_masked


def new_color_mask(img):
    """
    New color mask fucntion that combines three different color space filters
    to increasee detection accuracy.
    """

    hsv = HSV(img)
    hls = HLS(img)
    binary = np.zeros_like(img[:, :, 0])

    # Yellow filter
    lower_yellow = (15, 75, 75)
    upper_yellow = (35, 255, 255)
    hsv_y_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_hls_yellow = (10, 0, 20)
    upper_hls_yellow = (30, 255, 255)
    hls_y_mask = cv2.inRange(hls, lower_hls_yellow, upper_hls_yellow)

    # White filter
    hsv_sensitivity = 75
    lower_hsv_white = (0, 0, 255 - hsv_sensitivity)
    upper_hsv_white = (180, 25, 255)
    hsv_w_mask = cv2.inRange(hsv, lower_hsv_white, upper_hsv_white)

    hls_sensitivity = 75
    lower_hls_white = (0, 255 - hls_sensitivity, 0)
    upper_hls_white = (255, 255, hls_sensitivity)
    hls_w_mask = cv2.inRange(hls, lower_hls_white, upper_hls_white)

    rgb_sensitivity = 55
    lower_rgb_white = (255 - rgb_sensitivity, 255 - rgb_sensitivity, 255 - rgb_sensitivity)
    upper_rgb_white = (255, 255, 255)
    rgb_w_mask = cv2.inRange(img, lower_rgb_white, upper_rgb_white)

    bit_layer = binary | hsv_y_mask | hls_y_mask | hsv_w_mask | hls_w_mask | rgb_w_mask

    return bit_layer

def new_color_mask_2(img):
    """
    New color mask fucntion that combines three different color space filters
    to increasee detection accuracy.
    """

    hsv = HSV(img)
    hls = HLS(img)
    binary = np.zeros_like(img[:, :, 0])

    # Yellow filter
    #RGB
    lower_yellow = (180,140,35)
    upper_yellow = (255,230,175)
    rgb_y_mask = cv2.inRange(img, lower_yellow, upper_yellow)
    #HSV
    lower_yellow_hsv = (19, 50, 150)
    upper_yellow_hsv = (35, 215, 255)
    hsv_y_mask = cv2.inRange(hsv, lower_yellow_hsv, upper_yellow_hsv)
    #HLS
    lower_yellow_hls = (20,114,90)
    upper_yellow_hls = (32,203,255)
    hls_y_mask = cv2.inRange(hls, lower_yellow_hls, upper_yellow_hls)

    # White filter
    #RGB
    lower_white = (200,190,175)
    upper_white = (225,230,225)
    rgb_w_mask = cv2.inRange(img, lower_white, upper_white)
    #HSV
    lower_white_hsv = (18, 5, 200)
    upper_white_hsv = (120, 35, 255)
    hsv_w_mask = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)
    #HLS
    lower_white_hls = (17,150,9)
    upper_white_hls = (115,235,90)
    hls_w_mask = cv2.inRange(hls, lower_white_hls, upper_white_hls)

    mask_y = cv2.bitwise_and(hsv_y_mask, hls_y_mask, rgb_y_mask)
    mask_w = cv2.bitwise_and(hsv_w_mask, hls_w_mask, rgb_w_mask)
    bit_layer = binary | mask_w | mask_y

    return bit_layer


def sobel(img, orient='x', thres=(0, 255),kernel_size=3):

    # Check if image has 3 channels as the only accepted formats are RGB or HSV


    # Sobel transform as suggested by lessons
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=kernel_size))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=kernel_size))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Thresholding
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thres[0]) & (scaled_sobel <= thres[1])] = 1

    return binary_output


def complex_sobel(image, sobel_kernel=5, params={'thres_gradx': (0, 255),
                                                 'thres_mag': (0, 255),
                                                 'thres_dir': (0, np.pi / 2),
                                                 'thres_s': (0, 255)}):

    result = np.zeros(image.shape[:2], dtype=np.uint8)

    gray = GRAY(image)
    hsv = HSV(image)
    s = hsv[:, :, 1]

    for c in [gray, s]:
        gradx = abs_sobel_thresh(c, orient='x', sobel_kernel=sobel_kernel, thresh=params['thres_gradx'])
        mag_bin = mag_thresh(c, sobel_kernel=sobel_kernel, thresh=params['thres_mag'])
        dir_bin = dir_thresh(c, sobel_kernel=sobel_kernel, thresh=params['thres_dir'])
        result[((gradx == 1)) | ((mag_bin == 1) & (dir_bin == 1))] = 1

    s_bin = np.zeros_like(s)
    s_bin[(s >= params['thres_s'][0]) & (s <= params['thres_s'][1])] = 1

    result[(s_bin == 1)] = 1
    return result


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    # Thresholding
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    return mag_binary


def dir_thresh(image, kernel_size=3, thresh=(0, np.pi / 2)):

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    scaled_sobel = np.uint8(255 * absgraddir / np.max(absgraddir))

    # Thresholding
    dir_binary = np.zeros_like(scaled_sobel)
    dir_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return dir_binary


def perspective_transform_2(image, calc_points=True):

    img_w = image.shape[1]
    img_h = image.shape[0]

    src = np.zeros((4, 2), dtype=np.float32)

    if calc_points:
        src[0] = [np.uint(img_w * 0.375), np.uint(img_h * 2 / 3)]
        src[1] = [np.uint(img_w * 0.625), np.uint(img_h * 2 / 3)]
        src[2] = [img_w, img_h]
        src[3] = [0, img_h]
    else:
        # Coordinates for the ROI based on the fixed camera mount and fixed car position in this exercise:
        src[0] = [480., 480.]                    # [ img width * 0.375, img_height * 2/3]
        src[1] = [800., 480.]                    # [ img width * 0.625, img_height * 2/3]
        src[2] = [1280., 720.]                   # [ img width * 1,     img_height * 1  ]
        src[3] = [0., 720.]                      # [ img width * 0,     img_height * 1  ]

    # Calculate the destination points
    dst = np.zeros((4, 2), dtype=np.float32)
    dst[0] = [0., 0.]
    dst[1] = [img_w, 0.]
    dst[2] = [img_w, img_h]
    dst[3] = [0., img_h]

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (img_w, img_h))
    return warped, M, Minv


def perspective_transform(image):

    img_size = (image.shape[1],image.shape[0])
    img_w = image.shape[1]
    img_h = image.shape[0]

    # Source points:
    src = np.zeros((4, 2), dtype=np.float32)
    src[0] = [450.,520.]
    src[1] = [160.,719.]
    #src[1] = [1279.,719.]
    src[2] = [1097.,719.]
    #src[2] = [1279.,719.]
    src[3] = [806.,520.]
    # Destination points
    dst = np.zeros((4, 2), dtype=np.float32)
    dst[0] = [0.,0.]
    dst[1] = [0.,img_h]
    dst[2] = [img_w,img_h]
    dst[3] = [img_w,0.]

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, img_size, flags = cv2.INTER_LINEAR)
    return warped, M, Minv


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    print(len(nonzeroy))
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img

def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis = 0)

    return histogram
