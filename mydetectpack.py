import cv2
import numpy as np
import sys
import platform
import os
import glob


from pathlib import Path
from sklearn.isotonic import isotonic_regression
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class myloadimgs:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):


        self.img=path
        ni, nv = 1,0

        self.img_size = img_size
        self.stride = stride
        
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride

        
        self.cap = None


    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        

        
        
        # Read image
        self.count += 1
        im0 = self.img
        s = ''
        path=''
        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, s  
    
    
class PIDtrace:
    '''
    input must be np.ndarray
    input  current_vector(matrix is also ok) and dim_target\n
    output vector ,direction and size is calculated by pid
    '''
    def __init__(self,kp,ki,kd,shape) :
        self.kp=kp
        self.kd=kd
        self.ki=ki
        self.shape=shape
        self.error=np.zeros(shape)
        self.integral=np.zeros(shape)
        self.diff=np.zeros(shape)
        self.pre_error=np.zeros(shape)
    def update(self,act,exp):
        '''
        act=actual_value=star_location\n
        exp=expectation=target_location
        '''
        act,exp=check_and_change_shape(act,exp,self.shape)
        self.error=exp-act
        self.integral+=self.error
        self.diff=self.error-self.pre_error
        self.pre_error=self.error
        pid_value=self.kp*self.error+self.ki*self.integral+self.kd*self.diff
        return pid_value
    
def draw_pid_vector(img:np.ndarray,act,pid_value):
    act,pid_value=check_and_change_shape(act,pid_value,(2,1))
    start_point=[int(act[0]),int(act[1])]
    end_point=act+pid_value
    end_point=[int(end_point[0]),int(end_point[1])]
    cv2.arrowedLine(img,start_point,end_point,(128,255,128))
    return img
    
    


def check_and_change_shape(x,y,shape:tuple)->np.ndarray:
    '''input list or np.ndarray,return np.ndarray'''
    
    x=np.reshape(x,shape)
    y=np.reshape(y,shape)
    return x,y
    
        
        
         
  