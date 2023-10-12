from camera import control
import img_operation as imo
import os_operation as oso
import cv2
import mydetect
import mydetectpack as mp
import numpy as np
from camera import mvsdk




weights_path='best.pt'
img_path='people.jpg'

process_imgsz=(640,640)
camera_center=np.array([320,320])

kp=0.01
ki=0.01
kd=0.01
pid_shape=(2,1)





'''camera init part'''
hcamera=control.camera_init(mvsdk.CAMERA_MEDIA_TYPE_BGR8)
control.isp_init(hcamera,2000)
#out=control.save_video_camera_init(out_path,name='fuck2.mp4',codec='AVC1')
camera_info=control.get_all(hcamera)
control.print_getall(camera_info)
control.camera_open(hcamera)
pframebuffer_address=control.camera_setframebuffer()
'''pid init'''
pid=mp.PIDtrace(kp,ki,kd,pid_shape)

#press esc to end
while (cv2.waitKey(1) & 0xFF) != 27:
    dst=control.grab_img(hcamera,pframebuffer_address)
    
    dst=cv2.resize(dst,process_imgsz,interpolation=cv2.INTER_LINEAR)
    dst,dia_list=mydetect.myrun(source=dst,weights=weights_path,draw_img=True)
    
    if len(dia_list)>0:
    
        dst,center=mydetect.drawrec_and_getcenter(dia_list,dst)
        pid_value=pid.update(camera_center,center)
        dst=mp.draw_pid_vector(dst,camera_center,pid_value)
    
    #out.write(dst)
    cv2.circle(dst,camera_center.astype(np.uint16),10,(125,125,255),-1)
    cv2.imshow('press esc to end',dst)
    
    
cv2.destroyAllWindows()
control.camera_close(hcamera,pframebuffer_address)
#out.release()
