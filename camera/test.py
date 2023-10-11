from matplotlib import contour
import mvsdk
import control
import cv2
import os
out_path='D:/tmdvs/pywork/vsRM/for_calibrate'
num=0

hcamera=control.camera_init()
control.isp_init(hcamera,500)
#out=control.save_video_camera_init(out_path,name='fuck2.mp4',codec='AVC1')


camera_info=control.get_all(hcamera)
control.print_getall(camera_info)


control.camera_open(hcamera)
pframebuffer_address=control.camera_setframebuffer()

#press esc to end
while (cv2.waitKey(1) & 0xFF) != 27:
    dst=control.grab_img(hcamera,pframebuffer_address)
    #out.write(dst)
    control.camera_show(dst)
    num+=1
    abs_path=os.path.join(out_path,f'{num}.png')
    cv2.imwrite(abs_path,dst)
    
cv2.destroyAllWindows()
control.camera_close(hcamera,pframebuffer_address)
#out.release()



