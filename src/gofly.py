#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Empty
import time

def drone():
    takeoff = rospy.Publisher('/bebop/takeoff', Empty, queue_size=10)
    land = rospy.Publisher('/bebop/land', Empty, queue_size=10)

    rospy.init_node('drone', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    empty_msg = Empty()
    while not rospy.is_shutdown():
        takeoff.publish(empty_msg)
        time.sleep(5)
        land.publish(empty_msg)
        time.sleep(10)
        rate.sleep()

if __name__ == '__main__':
    try:
        drone()
    except rospy.ROSInterruptException:
        pass