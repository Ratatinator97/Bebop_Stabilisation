#!/usr/bin/env python
import rospy
from std_msgs.msg import Empty
from std_msgs.msg import UInt8
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist 
import sys, tty, termios
import numpy as np
import cv2 as cv
import math
import struct

def callback(msg):
    print(msg)
    pass
def callback2(msg):
    print(np.size(msg))
    pass
if __name__ == '__main__':
    rospy.init_node('decollage', anonymous=True)
    print("Node initialized")
    odometry = rospy.Subscriber("bebop/odom", Odometry, callback)
    print("Subscribed to odom")
    images_raw = rospy.Subscriber("bebop/image_raw", Image, callback2)
    print("Subscribed to Image")
    #pilot = rospy.Publisher("cmd_vel", Twist, queue_size=10)
    #rospy.Publisher("bebop/takeoff", Empty, queue_size=10).publish()
    #flip = rospy.Publisher("bebop/flip", UInt8, queue_size=10)
    #flip.publish(0)
    #rospy.Publisher("bebop/land", Empty, queue_size=10).publish()
    while not rospy.is_shutdown():
        continue

