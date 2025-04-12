#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import pi, atan2, sqrt
import numpy as np
from ORCA import Orca
from pyorca import Agent
from std_msgs.msg import String

class OrcaNode0:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.orca = Orca()  # Instantiate ORCA class
        self.agent = None
        self.current_position = np.array([1.0, 3.0])
        self.current_velocity = np.array([0.0, 0.0])
        self.goal = np.array([4.0, .5])

        # === Track other robots ===
        self.other_robots = {"r1"}
        self.other_agents = {}
        for other_name in self.other_robots:
            self.other_agents[other_name] = {
                "position": np.array([0.0, 0.0]),
                "velocity": np.array([0.0, 0.0]),
                "radius": 0.4,
                "max_speed": .4
            }

        if self.agent is not None:
            rospy.loginfo(f"[{self.robot_name}] Updating ORCA agents. Self: {self.agent.position}, Other: {[a['position'] for a in self.other_agents.values()]}")


        # === ROS Init ===
        rospy.init_node(f"orca_controller0")
        self.cmd_vel_pub = rospy.Publisher(f"/{self.robot_name}/cmd_vel", Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber(f"/{self.robot_name}/odom", Odometry, self.odom_callback)

        for other_name in self.other_robots:
            rospy.Subscriber(f"/{other_name}/odom", Odometry, self.generate_odom_callback(other_name))

        # Set up ORCA agent for this robot
        self.agent = self.orca.add_agent(self.current_position, self.current_velocity, 0.2, 0.4, self.goal)

        rospy.Subscriber(f"/{self.robot_name}/adjusted_traj", String, self.trajectory_callback)

        # Start the main control loop
        self.main_loop()

    def odom_callback(self, msg):
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])

        if self.agent is None:
            self.agent = self.orca.add_agent(
                position=position,
                velocity=velocity,
                radius=0.2,
                max_speed=1.0,
                goal= self.goal
            )
            rospy.loginfo(f"[{self.robot_name}] Initialized ORCA agent at {position}")
            return

        # Now that agent is initialized, update its state
        self.agent.position = position
        self.agent.velocity = velocity


    def generate_odom_callback(self, robot_id):
        def callback(msg):
            pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
            vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
            self.other_agents[robot_id]["position"] = pos
            self.other_agents[robot_id]["velocity"] = vel
        return callback

    def trajectory_callback(self, msg):
        """Callback for the adjusted trajectory of the robot."""
        # Parse the received trajectory data
        # Assuming the trajectory is sent as a string, you might need to parse it into a usable format (like a list of waypoints)
        # For simplicity, we'll just set the last goal as the new goal from the trajectory
        self.goal = np.array([float(coord) for coord in msg.data.split(',')])

    def update_goals(self):
        """Update the goal for the robot. Can be set dynamically."""
        self.orca.set_waypoints(self.agent, [self.goal])

    def main_loop(self):
        """Main control loop where ORCA computations happen."""
        rate = rospy.Rate(20)  # Run at 20 Hz
        while not rospy.is_shutdown():
            self.update_goals()

            # === Inject other agents into ORCA ===
            self.orca.agents = [self.agent]  # Reset to only include self

            for agent_data in self.other_agents.values():
                ghost = Agent(
                    position=agent_data["position"],
                    velocity=agent_data["velocity"],
                    radius=agent_data["radius"],
                    max_speed=agent_data["max_speed"],
                    goal=None
                )
                self.orca.agents.append(ghost)

            self.orca.update_all_velocities()
            new_speed, new_heading = self.orca.compute_velocity(self.agent)

            # Publish new velocity
            twist = Twist()
            twist.linear.x = new_speed * np.cos(new_heading)
            twist.linear.y = new_speed * np.sin(new_heading)
            self.cmd_vel_pub.publish(twist)

            rate.sleep()

if __name__ == '__main__':
    try:
        robot_name = rospy.get_param("~robot_name", "r0")
        orca_node = OrcaNode0(robot_name)
    except rospy.ROSInterruptException:
        pass
