#  (C) Copyright
#  Logivations GmbH, Munich 2010-2023
import asyncio
import datetime
import os
import threading
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi import Request
from starlette.responses import StreamingResponse
from lv.frame_grabbers.generic_ros_frame_grabber import (
    GenericRosFrameGrabber,
)
from lv.frame_grabbers.video_stream_utils import concatenate_images
from lv.tracking.constants import ALL_CAMERAS_CONFIGS, STANDARD_RES
from lv.utils import camera_utils

############################################
BASE_IMAGE_DIR = "/data/calibration_images"
LENS_DEGREE = 120
IMAGE_RES = STANDARD_RES
TOPIC = "/warehouse_light/camera104"

UNDISTORT = False
############################################
# import math
# HFoV_deg = 60
# HFoV_rad = HFoV_deg * (math.pi / 180)
# print(HFoV_rad)


class RealTimeImageSaver:
    def __init__(self):
        self.last_frame = None
        self.last_saved_rgb = None
        self.last_saved_gray = None
        self.timestamp = None

        self.lock = threading.Lock()

        self.frame_grabber = GenericRosFrameGrabber(
                topic=TOPIC,
                n_buffers=5,
                create_preview=True,
                stream_frames=False,
                topic_color_image=None,
                topic_camera_info=None,
            )


    async def save_frame(self, rgb=False):
        try:
            self.lock.acquire()
            previous_image = self.last_saved_rgb if rgb else self.last_saved_gray
            image_to_save = self.last_frame.preview_img if rgb else self.last_frame.img
            if previous_image is not None and np.all(previous_image == image_to_save):
                msg = f"Warning: The same {'RGB' if rgb else 'grayscale'} image!"
                print(msg)
            else:
                text = f"Saving {'RGB' if rgb else 'black and white'}"
                print(text)
                path = f"{BASE_IMAGE_DIR}/{LENS_DEGREE}/{'rgb' if rgb else 'black_and_white'}"
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)

                img_path = f"{path}/{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]}.jpg"
                msg = f"{text}: {img_path}"
                cv2.imwrite(img_path, image_to_save)
                print(f"{'RGB' if rgb else 'Black and white'} image saved. Path: {img_path}")
                if rgb:
                    self.last_saved_rgb = image_to_save.copy()
                else:
                    self.last_saved_gray = image_to_save.copy()
        except BaseException as e:
            msg = f"Could not save image: {e}"
            print(msg)
        finally:
            self.lock.release()
        return msg

    async def get_images_for_stream(self):
        camera_configs = ALL_CAMERAS_CONFIGS[LENS_DEGREE]
        mtx = camera_configs.mtx
        dist = camera_configs.dist

        while True:
            frame = self.frame_grabber.get_frame(consume_frame=False)
            if frame is None:
                continue

            with self.lock:
                self.last_frame = frame

            images = [frame.img, frame.preview_img]

            if UNDISTORT:
                undist_images = []
                for _, img in enumerate(images):
                    try:
                        undist_images.append(camera_utils.undistort(
                            img,
                            mtx=mtx,
                            dist=dist,
                            calibration_resolution=IMAGE_RES,
                            interpolation=cv2.INTER_LINEAR,
                        ))
                    except:
                        print("Error when undistoring image")
                else:
                    images = undist_images

            concatenated_image = concatenate_images(images)

            try:
                ret, jpeg = cv2.imencode(
                    ".jpg", concatenated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                )
            except:
                print("could not encode image")
                continue
            await asyncio.sleep(0.01)
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"

app = FastAPI(async_mode=True)

image_generator = RealTimeImageSaver()
templates = Jinja2Templates(directory="/code/deep_cv/lv/gz_calibration/templates")

@app.get("/stream")
async def video_feed():
    """TODO: Add docstring."""
    return StreamingResponse(
        image_generator.get_images_for_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.post("/save_rgb")
async def save_rgb_image():
    msg = await image_generator.save_frame(rgb=True)
    return msg

@app.post("/save_gray")
async def save_gray_image():
    msg = await image_generator.save_frame(rgb=False)
    return msg

@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse("stream.html", {"request": request})


# start: /usr/local/bin/uvicorn --app-dir /code/deep_cv lv.gz_calibration.gazebo_image_saver:app --host 0.0.0.0 --port 8020
# stop:  sudo pkill -9 -f gazebo_image_saver; lsof -t -i :8020 | xargs kill -9
