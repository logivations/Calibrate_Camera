"""
Save images for calibration 

usage: python save_im_calib.py --width=3280 --height=2464 --vis=True --save_im=True

"""
import time, datetime
import numpy as np
import cv2
import os
from numpy.linalg import inv
from time_utils import time_it
from logger import logger
from fire import Fire
import picamera
import picamera.array
from time import sleep
# marker_size = aruco_markers_props.markersize
save_folder = './images_1640_412/'
#dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
#parameters = aruco.DetectorParameters_create()
# https://docs.opencv.org/3.2.0/d5/dae/tutorial_aruco_detection.html

markerSize = 16
next_id = 1


def get_trans_rot_vec_split(T_mtx):
    # TODO: transfor to the object of matrix
    rvec = T_mtx[0:, 0]
    tvec = T_mtx[0:, 1]
    return rvec, tvec


def get_static_path(folder_name, file_name):
    script_dir = os.path.dirname(__file__)
    path = folder_name if file_name is None else "%s/%s" % (folder_name, file_name)
    return os.path.join(script_dir, path)


# file_path = get_static_path('resources','cameraParameters.xml')
# myLoadedData = cv2.FileStorage(file_path,cv2.FileStorage_READ)
# print myLoadedData

#mtx_path = get_static_path('resources', "%s_mtx.npy" % 'Picam_v2')
#cam_mtx = np.load(mtx_path)
#dist_path = get_static_path('resources', "%s_dist.npy" % 'Picam_v2')
#dist_coef = np.load(dist_path)
      
@time_it
def analyze(frame, vis=False,save_im=True, frame_num=0):
    
    logger.info("-----start capture-----")
    start_time = time.time()
    # Capture frame-by-frame
    
    #pf= frame[:,:,0]   

    if save_im:
        #BGR=cv2.cvtColor(frame,cv2.COLOR_YUV2BGR)
        cv2.imwrite(save_folder+str(frame_num)+'.png',frame[:,:,0])
        k = cv2.waitKey(50) & 0xFF
        if k == ord('q'):
            print('exit')
            os.exit()
    if vis:
        #cv2.imshow('output', markedFrame1[1000:1500,1000:1500,0])
        cv2.imshow('output', frame[:,:,0])
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print('exit')
            os.exit()

    print (time.time()-start_time)
    start_time = time.time()
    logger.info("-----end capture-----")


class camera_server(picamera.array.PiYUVAnalysis):
    def __init__(self, camera, vis=False, save_im=False):
        super(camera_server, self).__init__(camera)
        self.frame_num = 0
        self.frame = np.zeros((3, 3))
        self.starttime = time.time()
        self.vis = vis
        self.save_im=save_im
        

    def analyse(self, a):
        self.frame = a
        self.frame_num += 1
        if self.frame_num % 30 == 0:
            #print '\n\n\n\n fps'
            #print 30/(time.time()-self.starttime)
            self.starttime = time.time()
        analyze(self.frame, self.vis, self.save_im, self.frame_num)

def start_demo(device=0, fps=10, width=1640, height=1232,  vis=False, save_im=True):
    #print "st"
    
    if vis :
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)    
        cv2.resizeWindow("output", 640, 480)

    try:
        camera = picamera.PiCamera(sensor_mode=2)
        # http://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        camera.resolution = (width, height)
        camera.framerate = fps
        camera.shutter_speed = 3000
        
        output = camera_server(camera, vis, save_im)
        camera.start_recording(output, 'yuv')
        
        camera.wait_recording(5000)
    except Exception as e:
        #print e
        pass

    finally:
        pass

        cv2.destroyAllWindows()
        #camera.stop_recording()


if __name__ == '__main__':
    Fire(start_demo)
