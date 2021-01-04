import numpy as np
import cv2 as cv

img = cv.imread('montagnes.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray,200,0.01,10)
corners = np.int0(corners)


curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 

# Sanity check
assert prev_pts.shape == curr_pts.shape
	 
# Filter only valid points
idx = np.where(status==1)[0]
prev_pts = prev_pts[idx]
curr_pts = curr_pts[idx]

for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)

cv.imwrite('corner_detection_montagnes.jpg', img)