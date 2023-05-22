#  (C) Copyright
#  Logivations GmbH, Munich 2010-2023
import cv2
import cv2.aruco as aruco


aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(
    squaresX=10,
    squaresY=10,
    squareLength=2,
    markerLength=1.8,
    dictionary=aruco_dict,
)

board_image = board.draw(outSize=(10000, 10000))
cv2.imwrite("../calibration_checkerboard.jpg", board_image)
print("Calibration marker saved")

