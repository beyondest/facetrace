from camera import control
import img_operation as imo
import os_operation as oso
import cv2
import mydetect
import mydetectpack as mp
import numpy as np
from camera import mvsdk
import os



myweights='armor.pt'
yolov5s='yolov5s.pt'
img_path='people.jpg'

process_imgsz=(640,640)
camera_center=np.array([320,320])

num=0




'''camera init part'''
hcamera=control.camera_init(mvsdk.CAMERA_MEDIA_TYPE_BGR8)
control.isp_init(hcamera,2000)
#out=control.save_video_camera_init(out_path,name='fuck2.mp4',codec='AVC1')
camera_info=control.get_all(hcamera)
control.print_getall(camera_info)
control.camera_open(hcamera)
pframebuffer_address=control.camera_setframebuffer()


#press esc to end
while (cv2.waitKey(1) & 0xFF) != 27:
    num+=1
    dst=control.grab_img(hcamera,pframebuffer_address)
    
    dst=cv2.resize(dst,process_imgsz,interpolation=cv2.INTER_LINEAR)
    dst,dia_list=mydetect.myrun(source=dst,weights=myweights,draw_img=True)
    
    
    
    #out.write(dst)
    #cv2.circle(dst,camera_center.astype(np.uint16),10,(125,125,255),-1)
    cv2.imshow('press esc to end',dst)
    out_path=os.path.join('out',f'{num}.jpg')
    cv2.imwrite(out_path,dst)
    
    
cv2.destroyAllWindows()
control.camera_close(hcamera,pframebuffer_address)
#out.release()
