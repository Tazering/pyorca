#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

rospy.init_node('test_cmd_vel_publisher')

# Create a publisher for /r1/cmd_vel
pub = rospy.Publisher('/r1/cmd_vel', Twist, queue_size=10)

# Wait for the publisher to register
rospy.sleep(1)

# Create a simple Twist message
move_cmd = Twist()
move_cmd.linear.x = 0.2  # Move forward
move_cmd.angular.z = 0.0  # No rotation

# Publish the message
pub.publish(move_cmd)

rospy.loginfo("Published Twist message to /r1/cmd_vel")

