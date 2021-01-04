import numpy as np
import cv2 as cv

img = cv.imread('../images/20210104_114541.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray,200,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)

cv.imwrite('Out.jpg', img)