import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

undist_params = pickle.load(open("./undist_params.p", "rb"))
src = undist_params["src"]
dst = undist_params["dst"]
mtx = undist_params["mtx"]
dist = undist_params["dist"]

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

def undistort_and_transform(img, src, dst, mtx, dist):

    img_size = (img.shape[1], img.shape[0])

    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # and the inverse as well
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M, Minv

def threshold_image(img, s_thresh=(210, 255), sx_thresh=(30, 45), b_thresh=(150, 255)):
    img = np.copy(img)

    # Convert to HSL color space and separate the L channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Convert to HSL color space and separate the L channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
    lab_b_channel = lab[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel S of HLS
    s_binary = np.zeros_like(l_channel)
    s_binary[(l_channel >= s_thresh[0]) & (l_channel <= s_thresh[1])] = 1

    # Threshold color channel B of LAB
    b_binary = np.zeros_like(lab_b_channel)
    b_binary[(lab_b_channel >= b_thresh[0]) & (lab_b_channel <= b_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((b_binary, sxbinary, s_binary))

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1) | (b_binary == 1)] = 1

    return color_binary, combined_binary

def find_curvature_and_offset(leftx, rightx, lefty, righty, img_height_pix, img_width_pix):
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature at the base of the image (img_height_pix)
    left_curverad = np.sign(left_fit_cr[0]) * (
    ((1 + (2 * left_fit_cr[0] * img_height_pix * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0]))
    right_curverad = np.sign(right_fit_cr[0]) * (((1 + (
        2 * right_fit_cr[0] * img_height_pix * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0]))

    img_height_m = img_height_pix * ym_per_pix
    img_width_m = img_width_pix * xm_per_pix

    left_intercept = np.polyval(left_fit_cr, img_height_m)
    right_intercept = np.polyval(right_fit_cr, img_width_m)

    center = (left_intercept + right_intercept) / 2.0

    offset_m = center - img_width_m / 2.0

    return left_curverad, right_curverad, offset_m

def find_polynomials(binary_warped, plotgraph=False):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 8
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    if plotgraph:
        plt.figure(figsize=(16, 8))
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    l_curverad, r_curverad, offset_m = find_curvature_and_offset(leftx, rightx, lefty, righty, binary_warped.shape[0],
                                                                 binary_warped.shape[1])

    return left_fit, right_fit, l_curverad, r_curverad, offset_m, out_img


def find_polynomials_skip_windowing(binary_warped, left_fit, right_fit, plotgraph=False):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    result = np.zeros_like(out_img)

    if plotgraph:
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.figure(figsize=(16, 8))
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    l_curverad, r_curverad, offset_m = find_curvature_and_offset(leftx, rightx, lefty, righty, binary_warped.shape[0],
                                                                 binary_warped.shape[1])

    return left_fit, right_fit, l_curverad, r_curverad, offset_m, result

def draw_lane(undistorted_img, warped, left_fit, right_fit, minv, l_radius, r_radius, deviation_offset, error_vec):
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, minv, (undistorted_img.shape[1], undistorted_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)

    # Add Curvature and Deviation Text
    l_radius_txt = str(np.round(l_radius, 2))
    r_radius_txt = str(np.round(r_radius, 2))
    deviation_offset_txt = "{:.3f}".format(deviation_offset)
    font = cv2.FONT_HERSHEY_PLAIN
    curvature_txt = "left_curvature: " + l_radius_txt + " [m], right_curvature: " + r_radius_txt + " [m]"
    deviation_txt = "deviation: " + deviation_offset_txt + " [m]"
    curvature_err_txt = "Lane curvatures differ"
    lane_width_err_txt = "Bad lane width"
    parallelism_err_txt = "Lanes not Parallel"

    cv2.putText(result, curvature_txt, (30, 60), font, 2, (255, 255, 0), 3)
    cv2.putText(result, deviation_txt, (30, 110), font, 2, (255, 255, 0), 3)
    if(error_vec[0]==1):
        cv2.putText(result, curvature_err_txt, (30, 160), font, 2, (255, 0, 0), 3)
    if(error_vec[1]==1):
        cv2.putText(result, lane_width_err_txt, (30, 210), font, 2, (255, 0, 0), 3)
    if(error_vec[2]==1):
        cv2.putText(result, parallelism_err_txt, (30, 260), font, 2, (255, 0, 0), 3)


    return result

def sanity_check(warped, left_fit, right_fit, left_curverad, right_curverad):
    maxAllowablePolynomialDerivativeDifference = 1.0
    maxAllowableCurvatureDifference = 1500
    maxAllowableLaneWidthMeters = 5.7
    minAllowableWidthMeters = 2.7

    errorVector = [0, 0, 0]

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    # Check if lines have similar curvature
    if np.absolute(left_curverad - right_curverad) > maxAllowableCurvatureDifference:
        errorVector[0] = 0

    # Check that they are separated by approximately the right distance horizontally
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)
    maxWidthPerceived = np.amax(xm_per_pix * np.abs(right_fitx - left_fitx))
    minWidthPerceived = np.amin(xm_per_pix * np.abs(right_fitx - left_fitx))
    if (maxWidthPerceived > maxAllowableLaneWidthMeters) | (minWidthPerceived < minAllowableWidthMeters):
        errorVector[1] = 1

    # Checking that they are roughly parallel
    left_derivative = left_fit[0] * ploty * 2 + left_fit[1]
    right_derivative = right_fit[0] * ploty * 2 + right_fit[1]
    maxPolynomialDerivativeDifference = np.amax(np.absolute(left_derivative - right_derivative))
    if maxAllowablePolynomialDerivativeDifference < maxPolynomialDerivativeDifference:
        errorVector[2] = 1

    return errorVector

class Lane():
    pass
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.left_curvature = None
        self.right_curvature = None
        self.vehicle_offset_m = None
        self.lane_frame_age = 0

def add_lane_to_history(history, left_fit, right_fit, left_curverad, right_curverad, vehicle_offset_m):
    new_lane = Lane()
    new_lane.left_fit = left_fit
    new_lane.right_fit = right_fit
    new_lane.left_curvature = left_curverad
    new_lane.right_curvature = right_curverad
    new_lane.vehicle_offset_m = vehicle_offset_m
    history.insert(0, new_lane)
    return new_lane

lanes_history = []


def identify_lanes(image):
    # Binarize and warp image
    result_colored, result_binary = threshold_image(image)
    warped, M, Minv = undistort_and_transform(result_binary, src, dst, mtx, dist)
    n = 14

    # Fit polynomial
    if len(lanes_history) == 0:
        left_fit, right_fit, left_curverad, right_curverad, vehicle_offset_m, debug_img = find_polynomials(warped,
                                                                                                           False)
        error = sanity_check(warped, left_fit, right_fit, left_curverad, right_curverad)
        if error == [0, 0, 0]:
            add_lane_to_history(lanes_history, left_fit, right_fit, left_curverad, right_curverad, vehicle_offset_m)
    else:
        most_recent_lane = lanes_history[0]
        left_fit, right_fit, left_curverad, right_curverad, vehicle_offset_m, debug_img = find_polynomials_skip_windowing(
            warped, most_recent_lane.left_fit, most_recent_lane.right_fit, False)
        error = sanity_check(warped, left_fit, right_fit, left_curverad, right_curverad)
        if error == [0, 0, 0]:
            add_lane_to_history(lanes_history, left_fit, right_fit, left_curverad, right_curverad, vehicle_offset_m)

        left_fits = []
        right_fits = []
        for lane in lanes_history:
            left_fits.insert(0, lane.left_fit)
            right_fits.insert(0, lane.right_fit)

        left_fit = np.array(left_fits).mean(0)
        right_fit = np.array(right_fits).mean(0)

        if len(lanes_history) > n:
            lanes_history.pop()

        for i, lane in enumerate(lanes_history):
            lane.lane_frame_age += 1
            if lane.lane_frame_age > 2 * n:
                lanes_history.pop(i)

    undist = cv2.undistort(image, mtx, dist, None, mtx)

    return undist, warped, left_fit, right_fit, Minv, left_curverad, right_curverad, vehicle_offset_m, error

