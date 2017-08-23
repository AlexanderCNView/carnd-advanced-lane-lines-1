from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import glob
import matplotlib.image as mpimg


def calibrate():
    images = glob.glob('camera_cal/calibration*.jpg')
    objpoints = []
    imgpoints = []
    dim = (9, 6)
    objp = np.zeros((dim[0] * dim[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dim[0], 0:dim[1]].T.reshape(-1, 2)

    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, dim, None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


def undistort(img, mtx, dist):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted


def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from vertices. The rest of the image is set to black.
    """

    #  defining a blank mask to start with
    mask = np.zeros_like(img)
    img_height, img_width = img.shape[0], img.shape[1]
    # Defining region of interest
    vertices = np.array([[(50, img_height), (img_width / 2, img_height / 2),
                          (img_width / 2, img_height / 2),
                          (img_width - 50, img_height)]], dtype=np.int32)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        ignore_mask_color = (255,) * img.shape[2]  # i.e. 3 or 4 depending on your image
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Binarize channels
def binarize_channels(pixels, thresh_min, thresh_max):
    binarized = np.zeros_like(pixels)
    binarized[(pixels >= thresh_min) & (pixels <= thresh_max)] = 1
    return binarized


def draw_lane(img, warped_img, left_fitx, right_fitx, ploty, Minv):
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    final = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return final


def draw_src_dst(img, src):
    img_polygon = np.copy(img)
    cv2.polylines(img_polygon, [np.array(src, dtype=np.int32).reshape((-1, 1, 2))], True, (255, 0, 0), 5)
    return img_polygon


def fit_polynomial(warped, img_height):

    histogram = np.sum(warped[img_height // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(img_height / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

    temp = np.zeros_like(warped)
    temp[np.nonzero(warped)] = 255
    out_img = np.dstack((temp, temp, temp))
    out_img = out_img.astype('uint8')

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
        win_y_low = img_height - (window + 1) * window_height
        win_y_high = img_height - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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

    ploty = np.linspace(0, img_height - 1, img_height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Calculate curvature radius
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    lane_width = (left_fitx - right_fitx)[-1]
    xm_per_pix = 3.7 / lane_width  # meters per pixel in x dimension
    ym_per_pix = 30 / 720  # meters per pixel in y dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    # Calculate deviation from the center
    camera_position = out_img.shape[1] / 2
    center_offset_meters = (camera_position - ((left_fitx[-1] + right_fitx[-1]) / 2)) * xm_per_pix
    return ploty, left_fitx, right_fitx, left_curverad, right_curverad, center_offset_meters, out_img


def annotate_image(img, left_curverad, right_curverad, offset):
    annotated = np.copy(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, 'Left Radius: {0:.2f}m'.format(left_curverad),
                (850, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, 'Right Radius: {0:.2f}m'.format(right_curverad),
                (850, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, 'Off Center: {0:.2f}m'.format(offset),
                (850, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def compose_final(img, img1, img2, img3, img4):
    sub_w, sub_h = 266, 150
    final_img = np.copy(img)
    final_img[0:sub_h, 0:sub_w] = cv2.resize(img1, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
    final_img[0:sub_h, sub_w:sub_w * 2] = cv2.resize(img2, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
    final_img[sub_h:sub_h * 2, 0:sub_w] = cv2.resize(img3, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
    final_img[sub_h:sub_h * 2, sub_w:sub_w * 2] = cv2.resize(img4, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
    return final_img


def process_image(img):
    mtx, dist = calibrate()
    img = undistort(img, mtx, dist)
    img_width, img_height = img.shape[1], img.shape[0]
    r_channel, g_channel, b_channel = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel, l_channel, s_channel = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]

    s_binary = binarize_channels(s_channel, 150, 255)
    r_binary = binarize_channels(r_channel, 230, 255)
    g_binary = binarize_channels(g_channel, 150, 255)
    l_binary = binarize_channels(l_channel, 140, 255)

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(g_binary == 1) & (r_binary == 1) | ((s_binary == 1) & (l_binary == 1))] = 1
    combined_binary = region_of_interest(combined_binary)

    src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])
    dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])

    img_with_src = draw_src_dst(img, src)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    binary_warped = cv2.warpPerspective(combined_binary, M,
                                        dsize=(img_width, img_height),
                                        flags=cv2.INTER_LINEAR)

    color_warped = cv2.warpPerspective(img, M,
                                       dsize=(img_width, img_height),
                                       flags=cv2.INTER_LINEAR)
    color_warped = draw_src_dst(color_warped, dst)

    ploty, left_fitx, right_fitx, left_curverad, right_curverad, center_offset_meters, out_img = fit_polynomial(binary_warped, img_height=img_height)
    lane_img = draw_lane(img, binary_warped, left_fitx, right_fitx, ploty, Minv)
    annotated = annotate_image(lane_img, left_curverad, right_curverad, center_offset_meters)
    combined_binary_temp = np.zeros_like(combined_binary).astype('uint8')
    combined_binary_temp[np.nonzero(combined_binary)] = 255
    combined_binary_stacked = np.dstack((combined_binary_temp, combined_binary_temp, combined_binary_temp))
    final = compose_final(img=annotated, img1=img_with_src,
                          img2=combined_binary_stacked, img3=color_warped, img4=out_img)
    return final


video_filename = 'project'
clip1 = VideoFileClip(video_filename + '_video.mp4')
clip_out = clip1.fl_image(process_image)
clip_out.write_videofile(video_filename + '_video_out.mp4', audio=False)
