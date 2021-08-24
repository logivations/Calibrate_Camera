import numpy as np
import cv2
import cv2.aruco as aruco
import os

"""
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
"""

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
print (aruco_dict)
# second parameter is id number
# last parameter is total image size
for i in range(50):
    img_wb = np.ones((1000, 1000)) * 255
    filepath = os.path.join("Marker_images", "wb_" + str(i) + ".jpg")
    print(filepath)
    img = aruco.drawMarker(aruco_dict, i, 700)
    img_wb[150:850, 150:850] = img
    cv2.imwrite(filepath, img_wb)
    # cv2.imshow('frame', img)
    # cv2.waitKey(100)
# cv2.destroyAllWindows()
