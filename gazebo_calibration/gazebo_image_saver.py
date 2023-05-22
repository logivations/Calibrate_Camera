#  (C) Copyright
#  Logivations GmbH, Munich 2010-2023
import curses
import datetime
import os

import cv2
import numpy as np

from lv import DEFAULT_FILE_PATH
from lv.frame_grabbers.generic_ros_frame_grabber import (
    GenericRosFrameGrabber,
)


topic = f"/warehouse_light/camera104"
frame_grabber = GenericRosFrameGrabber(
    topic=topic,
    n_buffers=5,
    create_preview=True,
    stream_frames=True,
    topic_color_image=None,
    topic_camera_info=None,
)

last_rgb, last_grayscale = None, None

def save_frame(rgb=False):
    nonlocal last_rgb, last_grayscale
    path = f"{DEFAULT_FILE_PATH}/calibration_images/{'rgb' if rgb else 'black_and_white'}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    frame = frame_grabber.get_frame()
    image = frame.preview_img if rgb else frame.img
    if (rgb and last_rgb is not None and np.all(image == last_rgb)) or (
        last_grayscale is not None and np.all(image == last_grayscale)
    ):
        print(f"\rWarning: The same {'RGB' if rgb else 'grayscale'} image!")
        return None
    else:
        print(f"\rSaving {'RGB' if rgb else 'black and white'}")
        if rgb:
            last_rgb = image.copy()
        else:
            last_grayscale = image.copy()

        img_path = f"{path}/{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]}.jpg"
        cv2.imwrite(
            img_path,
            frame.preview_img if rgb else frame.img,
        )


def image_saver(stdscr):
    while True:
        key_pressed = stdscr.getkey()
        last_pressed_key = key_pressed
        if "g" == last_pressed_key:
            save_frame(rgb=False)
        elif "c" == last_pressed_key:
            save_frame(rgb=True)


if __name__ == "__main__":
    # run from the terminal: python3 -m lv.gazebo_image_saver
    curses.wrapper(image_saver)
