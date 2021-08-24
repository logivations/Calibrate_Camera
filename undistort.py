import os
import numpy as np
import cv2

# RMS error: 1.7529967106893658
mtx = np.array(
    [
        [1.69790744e03, 0.00000000e00, 1.61754387e03],
        [0.00000000e00, 1.70255112e03, 1.26906632e03],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
dist = np.array(
    [
        [
            -8.45809117e-01,
            8.41710399e00,
            -3.39425761e-03,
            1.55864247e-03,
            3.55850671e-01,
            -7.78445012e-01,
            8.22459580e00,
            1.44433693e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ]
    ]
)
DIR = "/data/waveshare_calibration_images/capture_single/capture"
images = os.listdir(DIR)
print(images)
w, h = (3264, 2464)

newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

for idx, image in enumerate(images):
    image = DIR + "/" + image
    print(image)
    # frame = cv2.imread(image,0)
    frame = cv2.imread(image)
    print(frame.shape)
    # ret,frame = cap.read()q
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.undistort(gray, mtx, dist, newCameraMatrix=newcameramtx)
    im = np.abs((gray - gray2))
    im = cv2.applyColorMap(im, cv2.COLORMAP_BONE)
    print(type(im[1, 1]))
    print("max", np.max(im))
    print("min", np.min(im))
    cv2.imshow("original", cv2.resize(frame, fx=0.25, fy=0.25, dsize=None))

    # cv2.imshow('image', cv2.resize(im, fx=0.25, fy=0.25, dsize=None))
    cv2.imshow("image undistort", cv2.resize(gray2, fx=0.25, fy=0.25, dsize=None))
    if cv2.waitKey() & 0xFF == ord("q"):
        break
