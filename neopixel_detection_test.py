import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_EXPOSURE, -12)
detector = cv2.SimpleBlobDetector()
ret,img = cap.read()
writer = cv2.VideoWriter('output.avi',cv2.VideoWriter.fourcc('M','J','P','G'),30,(640,480))
while(1):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blobimg = 255 - gray
    cv2.imshow('image',img)
    x = cv2.waitKey(1)
    if x == 27:
        break
    writer.write(img)

cap.release()
writer.release()
