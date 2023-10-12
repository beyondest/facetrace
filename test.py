path='people.jpg'
path2='best.pt'
from sympy import true
import mydetect as md
import mydetectpack as mp
import cv2
img=cv2.imread(path)
img,dia_list=md.myrun(weights=path2,source=img,draw_img=True)
cv2.imshow('fu',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
