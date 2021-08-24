import numpy as np
import cv2
import cv2.aruco as aruco
import os

"""
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
"""
BOARD = [[45, 46], [47, 48]]
NAME = "calibration_3"
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
print (aruco_dict)
# second parameter is id number
# last parameter is total image size
img_total = np.ones((4000, 4000)) * 255
img_total_wb = np.ones((4600, 4600)) * 255
filepath = os.path.join(NAME + ".png")
for row, val_r in enumerate(BOARD):
    for col, val_c in enumerate(val_r):
        print(row, col)
        img_wb = np.ones((2000, 2000)) * 255
        img = aruco.drawMarker(aruco_dict, val_c, 1400)
        img_wb[300:1700, 300:1700] = img
        img_total[row * 2000 : (row + 1) * 2000, col * 2000 : (col + 1) * 2000] = img_wb
        # cv2.imwrite(filepath, img_wb)

img_total_wb[300:4300, 300:4300] = img_total
font = 3  # cv2.FONT_HERSHEY_SIMPLEX
img_total_wb = cv2.putText(img_total_wb, NAME, (1850, 100), font, 4, (0, 0, 0), 4, 4)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 1000, 1000)
cv2.imshow("image", img_total_wb)

cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite(filepath, img_total_wb)
