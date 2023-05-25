#  (C) Copyright
#  Logivations GmbH, Munich 2010-2023
import glob
import os
import time

import cv2
import numpy as np

from lv import DEFAULT_FILE_PATH
from lv.tracking.constants import STANDARD_RES

##########################################
IMAGE_PATH = "/data/calibration_images/120/black_and_white"
SAVE_RESULT_PATH = "/code/deep_cv/appconfig/tracking/camera_settings/120_degrees_lens"
CAMERA_RES = STANDARD_RES
##########################################


aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard_create(
    squaresX=10,
    squaresY=10,
    squareLength=2,
    markerLength=1.8,
    dictionary=aruco_dict,
)

images = glob.glob(f"{IMAGE_PATH}/*.jpg")

print(f"Found {len(images)} images")

allCharucoCorners = []
allCharucoIds = []

print("Image processing...")
t1 = time.time()
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if len(corners) > 0:
        _, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        allCharucoCorners.append(charucoCorners)
        allCharucoIds.append(charucoIds)

print(f"Image processing completed, spend {time.time() - t1} s.")
print("Starting camera calibration...")
t1 = time.time()
_, cameraMatrix, distCoeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=allCharucoCorners,
    charucoIds=allCharucoIds,
    board=board,
    imageSize=gray.shape[::-1],
    cameraMatrix=None,
    distCoeffs=None,
)


print(
    f"Calibration completed (time spend: {time.time() - t1}, saving results..."
)

for res in [("mtx", cameraMatrix), ("dist", distCoeffs)]:
    path = f"{SAVE_RESULT_PATH}/gazebo_{res[0]}_{CAMERA_RES[1]}.npy"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    np.save(path, res[1])
