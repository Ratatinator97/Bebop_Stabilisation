#!/usr/bin/env python
import rospy
from std_msgs.msg import Empty
from std_msgs.msg import UInt8
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist 
from cv_bridge import CvBridge
import sys, tty, termios
import numpy as np
import cv2 as cv
import math
import struct

bridge = CvBridge()
prev_img = []

def callback(msg):
    #print(msg)
    pass
def callback2(msg):
    print("Image cb called !")
    global prev_img
    global bridge
    # to skip first frame
    if prev_img == []:
        print("Fist img")
        curr_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        prev_img = curr_img
    else:

        # Convert to gray scales
        prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
        # Detect features to track
        prev_pts = cv.goodFeaturesToTrack(prev_gray,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=10)
        # Get the current img
        curr_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # Convert to gray scales
        curr_gray = cv.cvtColor(curr_img,cv.COLOR_BGR2GRAY)
        # Track feature points
        curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
        # Sanity check
        assert prev_pts.shape == curr_pts.shape 
        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        for i in curr_pts:
            x,y = i.ravel()
            cv.circle(curr_img,(x,y),3,255,-1)
        name = '../temp/'+str(i).zfill(10)+'.jpg'
        cv.imwrite(name, curr_img)
        print("Image saved !")


if __name__ == '__main__':
    rospy.init_node('decollage', anonymous=True)
    odometry = rospy.Subscriber("bebop/odom", Odometry, callback)
    images_raw = rospy.Subscriber("bebop/image_raw", Image, callback2)
    #pilot = rospy.Publisher("cmd_vel", Twist, queue_size=10)
    #rospy.Publisher("[namespace]/takeoff", Empty, queue_size=10).publish()
    #flip = rospy.Publisher("[namespace]/flip", UInt8, queue_size=10)
    #flip.publish(0)
    #rospy.Publisher("[namespace]/land", Empty, queue_size=10).publish()
    while not rospy.is_shutdown():
        continue

