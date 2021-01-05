
#! /usr/bin/python

import rospy
import numpy as np
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Empty
# OpenCV2 for saving an image
import cv2
import sys
from time import time

class images_motion(object):

    def __init__(self):
        self.takeoff_pub = rospy.Publisher('/bebop/takeoff', Empty, queue_size=1)
        self.land_pub = rospy.Publisher('/bebop/land', Empty, queue_size=1)

        self.odo_sub = rospy.Subscriber("/bebop/odom", Odometry, self.callback)
        self.raw_imb_sub = rospy.Subscriber("/bebop/image_raw", Image, self.callback2)

        self.empty_msg = Empty()
        self.bridge = CvBridge()
        self.prev_gray = []
        self.j = 0
        self.transforms = []
        self.file_odom = open("../data/"+session_name+"/odometry.csv", "wb")
        self.writer = csv.writer(file_odom)
        writer.writerow( ('Timestamp', 'Twist Linear', 'Twist Angular') )  

        self.session_name = raw_input("Enter the name of the session: ")
        if not session_name in os.listdir('../raw/'):
            os.mkdir("../raw/"+session_name)
        if not session_name in os.listdir('../temp/'):
            os.mkdir("../temp/"+session_name)
        if not session_name in os.listdir('../data/'):
            os.mkdir("../data/"+session_name)

    def callback(self, msg):
        print(str(msg.header.stamp.secs)+" : "+str(msg.header.stamp.nsecs))
        print("Linear: "+str(msg.twist.twist.linear))
        print("Linear: "+str(msg.twist.twist.angular))
        self.writer.writerow( (str(msg.header.stamp.secs)+","+str(msg.header.stamp.nsecs),
            str(msg.twist.twist.linear), 
            str(msg.twist.twist.angular) ) )
            
    def callback2(self, msg):
        timestamp = time()
        print("Image cb called !")
        # to skip first frame
        if self.prev_gray == []:
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
            cv.imwrite('../raw/'+self.session_name+'/'+str(self.j).zfill(10)+'.jpg', curr_img)
            for i in curr_pts:
                x,y = i.ravel()
                cv.circle(curr_img,(x,y),3,255,-1)
            self.j += 1
            # Save annotated image
            cv.imwrite('../temp/'+self.session_name+'/'+str(self.j).zfill(10)+'.jpg', curr_img)
            print("Image saved !")


            m, _ = cv.estimateAffinePartial2D(prev_pts, curr_pts)
            print(m)
            dx = m[0][2]
            dy = m[1][2]
            # Rotation angle
            da = np.arctan2(m[1][0], m[0][0])
            # Store transformation
            transforms.append([timestamp, dx,dy,da])


            prev_gray = curr_gray
        
    def save_and_quit(self):
        # Image processing saving
        fields = ['Timestamp', 'dx', 'dy', 'da']
        
        with open('../data/'+self.session_name+'/transforms.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(self.transforms)
        # close odometry .csv
        self.file_odom.close()
        print('Movements saved ! Exiting')

    def takeoff(self):
        print("Taking off...")
        rospy.sleep(0.5)
        self.takeoff_pub.publish(self.empty_msg)

    def abbort_mission(self):
        print("LANDING !! ABORT MISSION !! LANDING !!")
        rospy.sleep(0.5)
        self.land_pub.publish(self.empty_msg)

def signal_handler(sig, frame):
    pim.save_and_quit()
    print('Movements saved ! Exiting')
    sys.exit(0)

def main(args):
    pim = images_motion()
    rospy.init_node('process_images_node', anonymous=True)
    rospy.sleep(1)
    #pim.takeoff()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        pim.save_and_quit()
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
          main(sys.argv)


