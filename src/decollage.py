#!/usr/bin/env python
import rospy
from std_msgs.msg import Empty
from std_msgs.msg import UInt8
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist 
from cv_bridge import CvBridge
from pathlib import Path
import sys, tty, termios
import numpy as np
import cv2 as cv
import math
import struct
import os
import csv

bridge = CvBridge()
prev_gray = []
j = 0
session_name = ""
transforms = []

def callback(msg):
    #print(msg)
    pass
def callback2(msg):
    print("Image cb called !")
    global prev_gray
    global bridge
    global j
    global session_name
    global transforms
    # to skip first frame
    if prev_gray == []:
        print("First img")
        curr_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        curr_gray =  cv.cvtColor(curr_img,cv.COLOR_BGR2GRAY)
        prev_gray = curr_gray
    else:
        # Detect features to track
        prev_pts = cv.goodFeaturesToTrack(prev_gray,
                                     maxCorners=1000,
                                     qualityLevel=0.01,
                                     minDistance=30)
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

        # Save raw image
        cv.imwrite('../raw/'+session_name+str(j).zfill(10)+'.jpg', curr_img)
        for i in curr_pts:
            x,y = i.ravel()
            cv.circle(curr_img,(x,y),3,255,-1)
        j += 1
        # Save annotated image
        cv.imwrite('../temp/'+session_name+str(j).zfill(10)+'.jpg', curr_img)
        print("Image saved !")


        m = cv.estimateAffinePartial2D(prev_pts, curr_pts)
        dx = m[0,2]
        dy = m[1,2]
        # Rotation angle
        da = np.arctan2(m[1,0], m[0,0])
        # Store transformation
        transforms.append([dx,dy,da])


        prev_gray = curr_gray

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    fields = ['dx', 'dy', 'da']
    with open('../data/transforms', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(transforms)
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    session_name = input("Enter the name of the session: ")
    if not session_name in os.listdir('../raw/'):
        os.mkdir("../raw/"+session_name)
    if not session_name in os.listdir('../temp/'):
        os.mkdir("../temp/"+session_name)
    
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

