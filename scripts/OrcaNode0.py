#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from math import pi, atan2, sqrt
import numpy as np
from ORCA import Orca
from pyorca import Agent

class OrcaNode0:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.orca = Orca()  # Instantiate ORCA class
        self.agent = None
        self.current_position = np.array([1.0, 3.0])
        self.current_velocity = np.array([0.0, 0.0])
        self.goal = np.array([4.0, 2.0])
        
        # Initialize ROS
        rospy.init_node(f"orca_controller0")
        self.cmd_vel_pub = rospy.Publisher(f"/{self.robot_name}/cmd_vel", Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber(f"/{self.robot_name}/odom", Odometry, self.odom_callback)

        # Set up ORCA agent for this robot
        self.agent = self.orca.add_agent(self.current_position, self.current_velocity, 0.2, 1.0, self.goal)

        # Start the main control loop
        self.main_loop()

    def odom_callback(self, msg):
        """Callback for the robot's odometry data."""
        self.current_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.current_velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        
        # Update agent with the current position and velocity
        self.agent.position = self.current_position
        self.agent.velocity = self.current_velocity

    def update_goals(self):
        """Update the goal for the robot. Can be set dynamically."""
        # For example, you can update this based on a goal publisher or path planner
        # In this case, we keep it static
        self.orca.set_waypoints(self.agent, [self.goal])

    def main_loop(self):
        """Main control loop where ORCA computations happen."""
        rate = rospy.Rate(20)  # Run at 20 Hz
        while not rospy.is_shutdown():
            # Update goals and agent states
            self.update_goals()

            # Compute new velocities for all agents (in this case just one agent)
            self.orca.update_all_velocities()

            # Get the new velocity for this agent
            new_speed, new_heading = self.orca.compute_velocity(self.agent)
            
            # Create Twist message to send to ROS
            twist = Twist()
            twist.linear.x = new_speed * np.cos(new_heading)
            twist.linear.y = new_speed * np.sin(new_heading)
            
            # Publish the velocity command
            self.cmd_vel_pub.publish(twist)

            # Sleep to maintain the loop rate
            rate.sleep()

if __name__ == '__main__':
    try:
        robot_name = rospy.get_param("~robot_name", "r0")
        orca_node = OrcaNode0(robot_name)
    except rospy.ROSInterruptException:
        pass