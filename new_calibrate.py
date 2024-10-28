import os
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

ARUCO_DICT = cv2.aruco.DICT_6X6_50
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.139
MARKER_LENGTH = 0.084

PATH_TO_CALIBRATION_IMAGES = '/data/calibration_info/264_calibration_dataset/'
CURRENT_LEN = 264
MAX_COUNT_ACCEPTED_IMAGES = 500

# Resize the image to a smaller size (e.g., half the original size)
SCALE_PERCENT = 50

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary)

# Create the image with the ChArUco board
# size_ratio = SQUARES_Y / SQUARES_X
# LENGTH_PX = 640   # total length of the page in pixels
# MARGIN_PX = 20    # size of the margin in pixels
# SAVE_NAME = 'ChArUco_Marker.png'
# img = board.draw((LENGTH_PX, int(LENGTH_PX * size_ratio)), None, MARGIN_PX, 1)
# cv2.imshow("img", img)
# cv2.waitKey(2000)
# cv2.imwrite(SAVE_NAME, img)


def get_images():
    rejected_images = 0
    accepted_images = 0
    all_charuco_corners = []
    all_charuco_ids = []
    frame_accepted = []
    working_images = []
    images = os.listdir(PATH_TO_CALIBRATION_IMAGES)
    images.sort()
    images = shuffle(images, random_state=3)
    gray = None
    h, w = cv2.imread(PATH_TO_CALIBRATION_IMAGES+images[0], 0).shape[:2]

    for idx, i in enumerate(images):
        image = PATH_TO_CALIBRATION_IMAGES + i
        frame = cv2.imread(image)
        assert w == frame.shape[1] and h == frame.shape[0], "All the images must have same shape"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_copy = frame.copy()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

        if len(marker_corners) > 0:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
            cv2.aruco.drawDetectedMarkers(gray_copy, marker_corners, marker_ids)
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 23:
                accepted_images += 1
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                frame_accepted.append(gray_copy)
                working_images.append(image)
                if accepted_images >= MAX_COUNT_ACCEPTED_IMAGES:
                    break
            else:
                rejected_images += 1
                small_width = int(gray_copy.shape[0] * SCALE_PERCENT / 100)
                small_height = int(gray_copy.shape[1] * SCALE_PERCENT / 100)
                small_image = cv2.resize(gray_copy, (small_height, small_width), interpolation=cv2.INTER_AREA)
                cv2.imshow(f'REJECTED_{image}', small_image)
                os.remove(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    image_shape = gray.shape[:2][::] if gray is not None else None

    print('Accepted: ', accepted_images)
    print('Rejected: ', rejected_images)
    cv2.destroyAllWindows()
    return all_charuco_corners, all_charuco_ids, image_shape, frame_accepted, working_images


def calibrate_camera(all_charuco_corners, all_charuco_ids, image_shape):
    # Calibrate camera
    start_time = time.time()
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners,
                                                                                        all_charuco_ids, board,
                                                                                        image_shape, None, None)
    print("RMS error: ", retval)
    print('Camera matrix: ', camera_matrix)
    print('Dist matrix: ', dist_coeffs)
    print("Elapsed time in minutes:", (time.time() - start_time)/60)
    np.save(f'{CURRENT_LEN}_camera_matrix.npy', camera_matrix)
    np.save(f'{CURRENT_LEN}_dist_coeffs.npy', dist_coeffs)
    np.save(f'{CURRENT_LEN}_rvecs.npy', rvecs)
    np.save(f'{CURRENT_LEN}_tvecs.npy', tvecs)
    return retval, camera_matrix, dist_coeffs, rvecs, tvecs


def display_images(frame_accepted, camera_matrix, dist_coeffs):
    # Iterate through displaying all the images
    for frame in frame_accepted:
        undistorted_image = cv2.undistort(frame, camera_matrix, dist_coeffs)
        small_width = int(undistorted_image.shape[0] * SCALE_PERCENT / 100)
        small_height = int(undistorted_image.shape[1] * SCALE_PERCENT / 100)
        small_undistorted_image = cv2.resize(undistorted_image, (small_height, small_width), interpolation=cv2.INTER_AREA)
        cv2.imshow('Undistorted Image', small_undistorted_image)

        if cv2.waitKey(300) & 0xFF == ord('q'):
            break


all_charuco_corners, all_charuco_ids, image_shape, frame_accepted, working_images = get_images()
retval, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(all_charuco_corners, all_charuco_ids, image_shape)
# camera_matrix = np.load(f'/data/Calibrate_Camera/{CURRENT_LEN}_camera_matrix.npy')
# dist_coeffs = np.load(f'/data/Calibrate_Camera/{CURRENT_LEN}_dist_coeffs.npy')
# rvecs = np.load(f'/data/Calibrate_Camera/{CURRENT_LEN}_rvecs.npy')
# tvecs = np.load(f'/data/Calibrate_Camera/{CURRENT_LEN}_tvecs.npy')
display_images(frame_accepted, camera_matrix, dist_coeffs)
