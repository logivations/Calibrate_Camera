import time
import cv2.aruco as A
import cv2
import numpy as np
import os
import random
from sklearn.utils import shuffle
print (cv2.__version__)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
board = cv2.aruco.CharucoBoard_create(5,7,.051,.03,dictionary)
USE_MP = False
#img = board.draw((200*3,200*3))

#Dump the calibration board to a file
#cv2.imwrite('charuco.png',img)
"""use pdf instead """

#Start capturing images for calibration
#cap = cv2.VideoCapture(0)
rejected = 0
acctepted = 0
count = 200 # max number of images to use for calibration
allCorners = []
allIds = []
decimator = 0
# path to images to be used for calibration
impath = './images_1640_1232_C95/'
# name of saved matrices Picam_v2_dist_+cal_name ... will be saved to PWD
cal_name = '1640_C95'
images = os.listdir(impath)
images.sort()
images = shuffle(images,random_state=2)

def read_image(i):
    image = impath + i
    print (image)
    frame = cv2.imread(image)
    print (frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray,dictionary)
    cv2.aruco.drawDetectedMarkers(frame, res[0], res[1])
    if len(res[0])>0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)
        if res2[1] is not None and res2[2] is not None and len(res2[1])>10:
            return res2[1],res2[2]
    return [], []

import multiprocessing
pool = multiprocessing.Pool(8)
#allCorners, allIds = zip(*pool.map(read_image,range(len(images))))#[], []
for ac,ai in pool.map(read_image, images):
    if len(ai) != 0:
        allCorners.append(ac)
        allIds.append(ai)
image = impath + images[0]
frame = cv2.imread(image)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
imsize = gray.shape
print (imsize)

cv2.destroyAllWindows()
#Calibration fails for lots of reasons. Release the video if we do

def calibrate_char(k=0):
    cal = None
    try:
        allCorners1, allIds1 = shuffle(allCorners, allIds)
        allIds1 = allIds1[:int(len(allIds) / 1.5)]
        allCorners1 = allCorners1[:int(len(allIds) / 1.5)]
        #cal = cv2.aruco.calibrateCameraCharuco(allCorners1, allIds1, board, imsize, None, None)
        objPoints, imgPoints = [board.chessboardCorners for i in allIds1], allCorners1
        cal = cv2.calibrateCameraRO(objPoints, imgPoints, imsize, 4, None, None)
    except Exception as e:
        print('error calib:', e)
        pass  # some combinations of images are not calibrateable
    return cal
import pkgutil
cal = None
cal1 = []
try:
    if USE_MP:
        try:
            import multiprocessing

            pool = multiprocessing.Pool(1)
            cal1 = pool.map(calibrate_char, range(0, 10))

            print(len(cal1))
            cal1 = [cl for cl in cal1 if cl is not None]
            for c in cal1:
                if cal is None:
                    cal = c
                if c[0] < cal[0]:
                    cal = c
                print(cal[0], c[0])
        except Exception as e:
            print ('exception in mp calib',e)
    else:

        for i in range(10):
            try:
                allCorners1, allIds1 = shuffle(allCorners,allIds)
                length = count if count < int(len(allIds)/2) else int(len(allIds)/2)
                print (length)
                allIds1 = allIds1[:length]
                allCorners1 = allCorners1[:length]
                cal1.append(calibrate_char())
                # cal1.append(cv2.aruco.calibrateCameraCharuco(allCorners1, allIds1, board, imsize, None, None))

                if cal is None:
                    cal = cal1[-1]
                if cal1[-1][0]<cal[0]:
                    cal = cal1[-1]
                print(i,cal[0],cal1[-1][0])
            except Exception as e:
                print('error calib:',e)
                pass # some combinations of images are not calibrateable
    np.save('Picam_v2_mtx_%s.npy'%cal_name, cal[1])
    np.save('Picam_v2_dist_%s.npy'%cal_name, cal[2])
    np.save('Picam_v2_rvec_%s.npy'%cal_name, cal[3])
    np.save('Picam_v2_tvec_%s.npy'%cal_name, cal[4])
    #np.save('Picam_v2_allCorners_1640_ncm_92.npy', allCorners)
    #np.save('Picam_v2_board_1640_ncm_92.npy', board)

    print ('RMS error: ', cal[0])
#    print ('rvec: ', np.array(cal[3]).shape)
#    print ('tvec: ', np.array(cal[4]).shape)

    print ('Camera MAtrix: ', cal[1])
    print ('Dist Matrix: ', cal[2])
except Exception as e:
    print ('exception occured',e)
    #cap.release()

#cap.release()
#mtx = np.load('/workspace/deep_cv/appconfig/forklift/Picam_v2_mtx_1640_old.npy')
#dist = np.load('/workspace/deep_cv/appconfig/forklift/Picam_v2_dist_1640_old.npy')
#mtx = np.load('Picam_v2_mtx_1640_ncm_92.npy')
#dist = np.load('Picam_v2_dist_1640_ncm_92.npy')
mtx = cal[1]
dist = cal[2]
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1280,720)
for idx, i in enumerate(images):
    image = impath + i
    print (image)
    # frame = cv2.imread(image,0)
    frame = cv2.imread(image)
    print (frame.shape)
    # ret,frame = cap.read()q
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.undistort(gray,mtx,dist)
    im = np.abs((gray-gray2))
    im = cv2.applyColorMap(im, cv2.COLORMAP_BONE)
    print(type(im[1,1]))
    print('max',np.max(im))
    print('min',np.min(im))
    cv2.imshow('image', im)
    if cv2.waitKey() & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()