#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
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
        self.goal = np.array([4.0, 4.0])

        # === Track other robots ===
        self.other_robots = {"r0"}
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
        rospy.init_node(f"orca_controller1")
        self.cmd_vel_pub = rospy.Publisher(f"/{self.robot_name}/cmd_vel", Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber(f"/{self.robot_name}/odom", Odometry, self.odom_callback)

        for other_name in self.other_robots:
            rospy.Subscriber(f"/{other_name}/odom", Odometry, self.generate_odom_callback(other_name))

        # Set up ORCA agent for this robot
        self.agent = self.orca.add_agent(self.current_position, self.current_velocity, 0.2, 0.4, self.goal)

        rospy.Subscriber(f"/{self.robot_name}/adjusted_traj", Path, self.trajectory_callback)

        self.current_trajectory = []
        self.following_trajectory = True

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
        self.current_trajectory = np.array([
            [pose.pose.position.x, pose.pose.position.y] for pose in msg.poses
        ])        
        self.following_trajectory = True

    def update_goal_from_trajectory(self):
        """Select the closest future point in trajectory as the new goal."""
        if len(self.current_trajectory) == 0:
            return

        dists = np.linalg.norm(self.current_trajectory - self.agent.position, axis=1)
        idx = np.argmin(dists)
        if idx < len(self.current_trajectory):
            self.goal = self.current_trajectory[idx]
            self.current_trajectory = self.current_trajectory[idx+1:]

    def update_goals(self):
        """Update the goal for the robot. Can be set dynamically."""
        if self.following_trajectory and len(self.current_trajectory) > 0:
            self.goal = self.current_trajectory[0]
        else:
            self.orca.set_waypoints(self.agent, [self.goal])

    def check_proximity_to_other_agents(self):
        """Check if another agent is too close."""
        threshold_distance = 1.0  # The distance threshold for switching to ORCA
        for other_agent in self.other_agents.values():
            distance = np.linalg.norm(self.agent.position - other_agent["position"])
            if distance < threshold_distance:
                return True  # Too close to another agent
        return False

    def main_loop(self):
        """Main control loop where ORCA computations happen."""
        rate = rospy.Rate(20)  # Run at 20 Hz
        while not rospy.is_shutdown():
            self.update_goals()

            # === Inject other agents into ORCA ===
            if self.check_proximity_to_other_agents():
                self.following_trajectory = False
                rospy.loginfo(f"[{self.robot_name}] ==================== Switching to ORCA due to proximity ============")
            else:
                self.following_trajectory = True
            
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

            if self.following_trajectory:
                self.update_goal_from_trajectory()
                direction = self.goal - self.agent.position
                distance = np.linalg.norm(direction)
                

                if distance > 0.05:  # A threshold to decide if the goal is reached
                    direction = direction/distance
                    desired_speed = min(0.4, distance)
                    heading = atan2(direction[1], direction[0])
                else:
                    desired_speed = 0.05
                    heading = 0.0

            else:
                desired_speed, heading = self.orca.compute_velocity(self.agent)


            # Publish new velocity
            twist = Twist()
            twist.linear.x = desired_speed * np.cos(heading)
            twist.linear.y = desired_speed * np.sin(heading)
            self.cmd_vel_pub.publish(twist)

            rate.sleep()

if __name__ == '__main__':
    try:
        robot_name = rospy.get_param("~robot_name", "r1")
        orca_node = OrcaNode0(robot_name)
    except rospy.ROSInterruptException:
        pass
