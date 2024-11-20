import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

ARUCO_DICT = cv2.aruco.DICT_6X6_50
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.139
MARKER_LENGTH = 0.084
CORNER_THRESHOLD = 0.5

PATH_TO_CALIBRATION_IMAGES = '/data/calibration_info/camTIS33320064_focus120cm_len121_papermarker/'
# the folder name must be created according to the principle of as
# 'camera<TYPE+BARCODE>_focus<VALUE>_len<VALUE>_<TYPE_OF_MARKER>'
# TYPE_OF_MARKER can be papermarker or screenmarker
FOLDER_FOR_INTRINSICS = "camTIS33320064_focus120cm_len121_papermarker"
FOLDER_FOR_RVECS_TVECS = "camTIS33320064_focus120cm_len121_papermarker"
MAX_COUNT_ACCEPTED_IMAGES = 500

# Resize the image to a smaller size (e.g., half the original size)
SCALE_PERCENT = 50

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary)

# Create the image with the ChArUco board
# size_ratio = SQUARES_Y / SQUARES_X
# LENGTH_PX = 1400   # total length of the page in pixels
# MARGIN_PX = 20    # size of the margin in pixels
# SAVE_NAME = 'ChArUco_Marker.png'
# img = board.draw((LENGTH_PX, int(LENGTH_PX * size_ratio)), None, MARGIN_PX, 1)
# cv2.imshow("img", img)
# cv2.waitKey(2000)
# cv2.imwrite(SAVE_NAME, img)
# play = True
# while play:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         play = False
# cv2.destroyAllWindows()


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

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    for idx, i in enumerate(images):
        image = PATH_TO_CALIBRATION_IMAGES + i
        frame = cv2.imread(image)
        assert w == frame.shape[1] and h == frame.shape[0], "All the images must have same shape"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_copy = frame.copy()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        all_possible_charuco_corners = (SQUARES_X-1)*(SQUARES_Y-1)
        if len(marker_corners) > 0:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
            cv2.aruco.drawDetectedMarkers(gray_copy, marker_corners, marker_ids)
            if (charuco_corners is not None and charuco_ids is not None
                    and len(charuco_corners) >= CORNER_THRESHOLD * all_possible_charuco_corners):
                print(len(charuco_corners), image)
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
        else:
            rejected_images += 1
            small_width = int(gray_copy.shape[0] * SCALE_PERCENT / 100)
            small_height = int(gray_copy.shape[1] * SCALE_PERCENT / 100)
            small_image = cv2.resize(gray_copy, (small_height, small_width), interpolation=cv2.INTER_AREA)
            cv2.imshow(f'REJECTED_{image}', small_image)
            os.remove(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # should be [width, height] -> https://docs.opencv.org/4.5.4/dc/dbb/tutorial_py_calibration.html
    image_shape = gray.shape[:2][::-1] if gray is not None else None
    print("Image shape:", image_shape)
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
    np.save(f'{FOLDER_FOR_INTRINSICS}/camera_matrix.npy', camera_matrix)
    np.save(f'{FOLDER_FOR_INTRINSICS}/dist_coeffs.npy', dist_coeffs)
    np.save(f'{FOLDER_FOR_INTRINSICS}/rvecs.npy', rvecs)
    np.save(f'{FOLDER_FOR_INTRINSICS}/tvecs.npy', tvecs)
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


def plot_reprojection_error_graph(working_images, img_points, camera_matrix, dist_coeffs, rvecs, tvecs):
    print("Location of intrinsics info:", FOLDER_FOR_INTRINSICS)
    # get 3D points (pattern_points) for board Charuco
    pattern_points = board.chessboardCorners.reshape(-1, 1, 3)

    # Combine it to a dataframe
    calibration_df = pd.DataFrame({
        "image_names": working_images,
        "img_points": img_points,  # all_charuco_corners
    })
    calibration_df.sort_values("image_names")
    calibration_df.reset_index(drop=True)

    reprojection_error = []

    for i in range(len(calibration_df)):
        img_point = np.array(img_points[i], dtype=np.float32)
        rvec = np.array(rvecs[i], dtype=np.float32)
        tvec = np.array(tvecs[i], dtype=np.float32)
        imgpoints2, _ = cv2.projectPoints(pattern_points, rvec, tvec, camera_matrix, dist_coeffs)
        temp_error = cv2.norm(img_point, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        reprojection_error.append(temp_error)
    calibration_df['reprojection_error'] = pd.Series(reprojection_error)
    # Save the DataFrame to a CSV file
    calibration_df.to_csv(f'{FOLDER_FOR_INTRINSICS}/calibration_data.csv', index=False)

    avg_error = np.sum(np.array(reprojection_error)) / len(calibration_df.img_points)
    print(f"Reproject error for dataset {PATH_TO_CALIBRATION_IMAGES}: {avg_error}")
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(20)

    # Plot the data for all reprojection errors but show only selected image names on x-axis
    ax.scatter(range(len(reprojection_error)), reprojection_error, label='Reprojection error', marker='o')

    # Plot the average line for all image names
    y_mean = [avg_error] * len(calibration_df.image_names)
    ax.plot(range(len(reprojection_error)), y_mean, label='Mean Reprojection error', linestyle='--')
    # Reducing the number of labels to display on the x-axis (eg every 25 labels)
    N = max(len(reprojection_error) // 25, 1)
    selected_indices = np.arange(0, len(calibration_df.image_names), N)
    selected_image_names = [calibration_df.image_names[i] for i in selected_indices]

    # Set x-axis ticks and labels to show only selected image names
    ax.set_xticks(selected_indices)
    ax.set_xticklabels(selected_image_names, rotation=45)

    # Make a legend
    ax.legend(loc='upper right')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    # name x and y axis
    ax.set_title(f"Reprojection error plot for {FOLDER_FOR_INTRINSICS=}\n{PATH_TO_CALIBRATION_IMAGES=}")
    ax.set_xlabel("Image_names")
    ax.set_ylabel("Reprojection error in pixels")

    plt.savefig(f'{FOLDER_FOR_INTRINSICS}/reprojection_error.png')
    plt.show()

# all_charuco_corners, all_charuco_ids, image_shape, frame_accepted, working_images = get_images()
# retval, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(all_charuco_corners, all_charuco_ids, image_shape)
# camera_matrix = np.load(f'{FOLDER_FOR_INTRINSICS}/camera_matrix.npy')
# dist_coeffs = np.load(f'{FOLDER_FOR_INTRINSICS}/dist_coeffs.npy')
# rvecs = np.load(f'{FOLDER_FOR_RVECS_TVECS}/rvecs.npy')
# tvecs = np.load(f'{FOLDER_FOR_RVECS_TVECS}/tvecs.npy')
# plot_reprojection_error_graph(working_images, all_charuco_corners, camera_matrix, dist_coeffs, rvecs, tvecs)
# display_images(frame_accepted, camera_matrix, dist_coeffs)
