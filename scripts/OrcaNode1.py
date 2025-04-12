#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path

def trajectory_callback(msg):
    if msg.poses:
        # Let's just move the robot toward the first point in the trajectory
        goal_pose = msg.poses[0].pose
        rospy.loginfo(f"Moving toward goal: {goal_pose.position.x}, {goal_pose.position.y}")
        
        # Create a simple control command to move the robot
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.1  # Move forward
        cmd_vel.angular.z = 0.0  # No rotation
        
        # Publish the command
        cmd_vel_pub.publish(cmd_vel)

def orca_controller():
    rospy.init_node('orca_1', anonymous=True)
    rospy.Subscriber('/r1/default_traj', Path, trajectory_callback)
    
    global cmd_vel_pub
    cmd_vel_pub = rospy.Publisher('/r1/cmd_vel', Twist, queue_size=10)
    
    rospy.spin()

if __name__ == '__main__':
    try:
        orca_controller()
    except rospy.ROSInterruptException:
        pass
