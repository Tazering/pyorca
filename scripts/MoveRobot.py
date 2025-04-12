#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

def move_robot():
    rospy.init_node('test_velocity_pub')
    pub = rospy.Publisher('/r1/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        cmd = Twist()
        cmd.linear.x = 0.1
        cmd.angular.z = 0.5
        pub.publish(cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass
