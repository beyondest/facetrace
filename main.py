from camera import control
import img_operation as imo
import os_operation as oso
import cv2
import mydetect

weights_path='best.pt'
img_path='people.jpg'
camera_center=[320,320]

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
    
    
    dia_list=mydetect.myrun(source=dst,weights=weights_path)
    drawed_img,center=mydetect.drawrec_and_getcenter(dia_list)
    
    
    #out.write(dst)
    control.camera_show(drawed_img)
    
    
cv2.destroyAllWindows()
control.camera_close(hcamera,pframebuffer_address)
#out.release()
