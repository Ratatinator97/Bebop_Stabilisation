
#! /usr/bin/python

import rospy
import numpy as np
import os
import csv
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Empty
# OpenCV2 for saving an image
import cv2 as cv
import sys
from time import time

import signal
import math

counter = 0
x_error = 0
y_error = 0
kpx = 0
kpy = 0
ki = 0
N_FRAMES = 10
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
        
class images_motion(object):

    def __init__(self):
        self.takeoff_pub = rospy.Publisher('/bebop/takeoff', Empty, queue_size=1)
        self.land_pub = rospy.Publisher('/bebop/land', Empty, queue_size=1)
        self.twist_pub = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=30)


        self.odo_sub = rospy.Subscriber("/bebop/odom", Odometry, self.callback)
        self.raw_imb_sub = rospy.Subscriber("/bebop/image_raw", Image, self.callback2)

        self.empty_msg = Empty()
        self.bridge = CvBridge()
        self.prev_gray = []
        self.j = 0
        self.transforms = []
        self.correction = []
        self.orientation = []
         
        self.session_name = raw_input("Enter the name of the session: ")
        if not self.session_name in os.listdir('../raw/'):
            os.mkdir("../raw/"+self.session_name)
        if not self.session_name in os.listdir('../temp/'):
            os.mkdir("../temp/"+self.session_name)
        if not self.session_name in os.listdir('../data/'):
            os.mkdir("../data/"+self.session_name)
        self.file_odom = open("../data/"+self.session_name+"/odometry.csv", "wb")
        self.writer = csv.writer(self.file_odom)
        self.writer.writerow( ('Timestamp', 'x', 'y', 'z') ) 

    def callback(self, msg):
        x, y, z = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        timestamp = msg.header.stamp.secs + (msg.header.stamp.nsecs*pow(10,-9))
        self.orientation.append([timestamp,x,y,z])
            
    def callback2(self, msg):
        global counter
        global x_error
        global y_error
        global N_FRAMES
        global kpx
        global kpy
        timestamp = time()
        # to skip first frame
        if self.prev_gray == []:
            curr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            curr_gray =  cv.cvtColor(curr_img,cv.COLOR_BGR2GRAY)
            self.prev_gray = curr_gray
        else:
            # Detect features to track
            prev_pts = cv.goodFeaturesToTrack(self.prev_gray,
                                        maxCorners=1000,
                                        qualityLevel=0.05,
                                        minDistance=30)
            # Get the current img
            curr_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Convert to gray scales
            curr_gray = cv.cvtColor(curr_img,cv.COLOR_BGR2GRAY)
            # Track feature points
            curr_pts, status, err = cv.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts, None) 
            # Sanity check
            assert prev_pts.shape == curr_pts.shape 
            # Filter only valid points
            idx = np.where(status==1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Save raw image
            cv.imwrite('../raw/'+self.session_name+'/'+str(self.j).zfill(10)+'.jpg', curr_img)
            for i in curr_pts:
                x,y = i.ravel()
                cv.circle(curr_img,(x,y),3,255,-1)
            self.j += 1
            # Save annotated image
            cv.imwrite('../temp/'+self.session_name+'/'+str(self.j).zfill(10)+'.jpg', curr_img)

            m, _ = cv.estimateAffinePartial2D(prev_pts, curr_pts)
            dx = m[0][2]
            dy = m[1][2]

            if counter < N_FRAMES:
                x_error += dx
                y_error += dy
                counter += 1
            else:
                self.transforms.append([timestamp, x_error, y_error])
                x_error = x_error/N_FRAMES
                x_error = x_error*kpx
                y_error = y_error*kpy
                if x_error > 1:
                    x_error = 1
                elif x_error < -1:
                    x_error = -1
                if y_error > 1:
                    y_error = 1
                elif y_error < -1:
                    y_error = -1
                twist_msg = Twist()
                twist_msg.linear.y = -x_error
                twist_msg.linear.z = -y_error
                print("commande x "+format(x_error, '.5f'))
                print("commande y "+format(y_error, '.5f'))
                self.correct_velocity(twist_msg)
                self.correction.append([timestamp, x_error, y_error])
                x_error = 0
                y_error = 0
                counter = 0

            self.prev_gray = curr_gray
        
    def save_and_quit(self):
        # Image processing saving
        fields = ['Timestamp', 'x_error', 'y_error']
        
        with open('../data/'+self.session_name+'/transforms.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(self.transforms)
        with open('../data/'+self.session_name+'/correction.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(self.correction)
        # close odometry .csv
        self.writer.writerows(self.orientation)
        self.file_odom.close()
        print('Movements saved ! Exiting')

    def takeoff(self):
        global kpy
        global kpx
        print("Taking off...")
        rospy.sleep(0.5)
        self.takeoff_pub.publish(self.empty_msg)
        rospy.sleep(3)
        kpx=0.05
        kpy=0.005


    def abbort_mission(self):
        print("LANDING !! ABORT MISSION !! LANDING !!")
        rospy.sleep(0.5)
        self.land_pub.publish(self.empty_msg)

    def correct_velocity(self, twist_msg):
        rospy.sleep(0.5)
        self.twist_pub.publish(twist_msg)

def signal_handler(sig, frame):
    global pim
    pim.save_and_quit()
    pim.abbort_mission()
    print('Movements saved ! Exiting')
    sys.exit(0)

def main(args):
    global pim
    rospy.init_node('process_images_node', anonymous=True)
    rospy.sleep(1)
    signal.signal(signal.SIGINT, signal_handler)
    pim.takeoff()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shut down")
    cv.destroyAllWindows()
pim = images_motion()
if __name__ == '__main__':
          main(sys.argv)


