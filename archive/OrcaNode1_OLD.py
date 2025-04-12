#!/usr/bin/env python3
import rospy
import numpy as np
import time
import math

from std_msgs.msg import Float32, Int32
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Path, Odometry
from ORCA import Orca

class OrcaController:
    def __init__(self):
        rospy.init_node('orca_1')
        rospy.loginfo("Orca Controller Node Started")

        self.orca = Orca()
        self.agent = None

        self.goal_x = rospy.get_param("r1/goal_x", 4.0)
        self.goal_y = rospy.get_param("r1/goal_y", 0.0)
        #combine goal_x and y into a Point
        self.adjusted_goal_position = Point()
        self.adjusted_goal_position.x =  self.goal_x
        self.adjusted_goal_position.y = self.goal_y
        #print("goal position: ", self.adjusted_goal_position)

        # Instance variables (replacing globals)
        self.default_linear_velocity = 0.2
        self.default_collision_detected = False
        self.adjusted_collision_detected = False
        self.previous_lin_vel = 0.2
        self.other_velocity = 0.0
        #self.default_goal_position = None
        self.consecutive_collisions = 0
        self.desired_heading = 0.0
        self.current_heading = 0.0
        self.collision_detected = 0
        self.default_trajectory_received = False
        self.default_trajectory_points = []  # List of waypoints (geometry_msgs/Point)
        self.default_traj_index = 0          # Current index into trajectory
        self.adjusted_trajectory_received = False
        self.adjusted_trajectory_points = []  # List of waypoints (geometry_msgs/Point)
        self.adjusted_traj_index = 4 
        # Controller parameters and constants
        self.original_velocity = 0.2  # Base linear velocity
        self.robot_importance = 0.8   # Importance factor (0.0 to 1.0)
        self.Kp = 1.0               # Proportional gain for PD controller
        self.Kd = 0.05                 # Derivative gain for PD controller
        self.prev_error = 0.0
        self.prev_time = rospy.get_time()
        # Publisher and Subscribers
        self.cmd_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=10)
        # self.cmd_pub = rospy.Publisher("/ROSBOT1/cmd_vel", Twist, queue_size=10)
        self.new_vel_pub = rospy.Publisher("/r1/new_velocity", Float32, queue_size=10)
        rospy.Subscriber("/r1/other_velocity", Float32, self.other_velocity_callback)
        rospy.Subscriber("/r1/consecutive_collisions", Int32, self.collisions_callback)
        rospy.Subscriber("/r1/adjusted_traj", Path, self.adjusted_traj_callback)
        rospy.Subscriber("/r1/default_traj", Path, self.default_traj_callback)
        rospy.Subscriber("/r1/odom", Odometry, self.pose_callback)
        rospy.Subscriber("/r1/collision_detected_default", Int32, self.default_collision_callback)
        rospy.Subscriber("/r1/collision_detected_adjusted", Int32, self.adjusted_collision_callback)


        # Control loop at 10 Hz
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)
        # Trajectory timer (will be started once a trajectory is received)
        self.traj_timer = None
        

    def start_timer(self):
        """Starts a timer that updates the desired heading every 1 second."""
        if self.traj_timer is not None:
            self.traj_timer.shutdown()  # Cancel existing timer if needed
        #rospy.loginfo("Starting trajectory timer: 0.5  second per waypoint")
        self.traj_timer = rospy.Timer(rospy.Duration(0.5), self.traj_timer_callback)

    def other_velocity_callback(self, msg):
        self.other_velocity = msg.data

    def collisions_callback(self, msg):
        self.consecutive_collisions = msg.data

    def adjusted_traj_callback(self, msg):
        if len(msg.poses) >= 1:
            # Extract positions from the poses in the Path message
            self.adjusted_trajectory_points = [pose.pose.position for pose in msg.poses]
            self.adjusted_traj_index = 4  # reset the trajectory index
            self.adjusted_trajectory_received = True
            
            self.start_timer()  # start or restart the trajectory timer
            #rospy.loginfo("Trajectory received with %d waypoints.", len(self.adjusted_trajectory_points))
        else:
            rospy.logwarn("Path message has no poses; Failed in traj_callback")
            pass

    def default_traj_callback(self, msg):
        if len(msg.poses) >= 1:
            # Extract positions from the poses in the Path message
            #self.default_trajectory_points = [pose.pose.position for pose in msg.poses]
            #self.default_traj_index = 0  # reset the trajectory index
            self.default_trajectory_received = True
            #self.default_goal_position = self.default_trajectory_points[-1]
            #rospy.loginfo("Trajectory received with %d waypoints.", len(self.default_trajectory_points))
        else:
            rospy.logwarn("Path message has no poses; Failed in traj_callback")

    def pose_callback(self, msg):
        # Extract the current heading (yaw) from the quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_heading = math.atan2(siny_cosp, cosy_cosp)
        # Also store current position for heading computation
        self.current_position = msg.pose.pose.position

        rospy.loginfo("Current position: [%.2f, %.2f], Current velocity: [%.2f, %.2f]", 
               self.current_position.x, self.current_position.y,
               self.previous_lin_vel, self.other_velocity)

        position = np.array([self.current_position.x, self.current_position.y])
        velocity = np.array([self.previous_lin_vel, 0.0])

        if self.agent is None:
            self.agent = self.orca.add_agent(
                position = position,
                velocity = velocity,
                radius = 0.2,
                max_speed = 0.5
            )
        else:
            self.agent.position = position
            self.agent.velocity = velocity

    def default_collision_callback(self, msg):
        self.default_collision_detected = msg.data

    def adjusted_collision_callback(self, msg):
        self.adjusted_collision_detected = msg.data

    def traj_timer_callback(self, event):
        """Called every 0.5 seconds to update the desired heading based on the next waypoint."""
        if not self.adjusted_trajectory_received:
            return  # nothing to do if no trajectory
        
        # Check if we've reached the end of the trajectory
        if self.adjusted_traj_index >= len(self.adjusted_trajectory_points):
            #rospy.loginfo("Reached the end of the trajectory. Stopping trajectory timer.")
            self.traj_timer.shutdown()  # Shut down the timer
            self.adjusted_trajectory_received = False  # Reset the flag
            return
        
        # Get the next waypoint target
        target = self.adjusted_trajectory_points[self.adjusted_traj_index]
        #print("Target waypoint: ", target)
        #print("Current position: ", self.current_position)
        #print("Current heading: ", self.current_heading)
        #print("Waypoint number: ", self.adjusted_traj_index)
        # Ensure we have a current position before computing the desired heading
        if not hasattr(self, 'current_position'):
            rospy.logwarn("Current position not available yet; cannot update desired heading.")
            return
        
        # Compute the desired heading from current position to target waypoint
        dx = target.x - self.current_position.x
        dy = target.y - self.current_position.y
        self.desired_heading = math.atan2(dy, dx)
        #rospy.loginfo("DX and DY are %.3f, %.3f", dx, dy)
        #rospy.loginfo("Waypoint %d: Updated desired heading to %.3f radians, Current Heading %.3f", 
        #            self.adjusted_traj_index, self.desired_heading, self.current_heading)
        
        self.adjusted_traj_index += 1


    def modify_velocity(self, other_robot_velocity, num_consecutive_collisions_detected):
        # Update agent positions and velocities using ORCA
        self.orca.update_all_velocities()
        rospy.loginfo("Agent velocity: [%.2f, %.2f]", self.agent.velocity[0], self.agent.velocity[1])

        # Extract the new velocity for the robot
        new_velocity = self.agent.velocity
        new_lin_vel = np.linalg.norm(new_velocity)  # Linear speed
        new_heading = math.atan2(new_velocity[1], new_velocity[0])  # Heading (angular velocity)
        
        # Control logic for angular velocity remains the same
        error = self.desired_heading - self.current_heading
        error = math.atan2(math.sin(error), math.cos(error))
        
        dt = rospy.get_time() - self.prev_time
        if dt == 0:
            dt = 0.1
        angular_vel = self.Kp * error + self.Kd * (error - self.prev_error) / dt
        
        # Publish the updated command
        cmd_vel = Twist()
        cmd_vel.linear.x = new_lin_vel
        cmd_vel.angular.z = angular_vel
        rospy.loginfo("Publishing cmd_vel: linear %.3f, angular %.3f", cmd_vel.linear.x, cmd_vel.angular.z)
        rospy.loginfo("Calculated velocities: linear x: %.2f, angular z: %.2f", linear_velocity, angular_velocity)

        self.cmd_pub.publish(cmd_vel)

        # Store the current linear velocity for the next iteration
        self.previous_lin_vel = new_lin_vel
        self.prev_error = error

        rospy.loginfo("New ORCA velocity: [%.2f, %.2f]", new_velocity[0], new_velocity[1])

        return new_lin_vel

    def control_loop(self, event):
        rospy.loginfo("Entered control_loop")

        # Check if we are at our goal position
        if not hasattr(self, 'current_position'):
            return
        if not self.default_trajectory_received:
            return

        
        if self.adjusted_goal_position is not None:
            dx = self.adjusted_goal_position.x - self.current_position.x
            dy = self.adjusted_goal_position.y - self.current_position.y
            distance = math.sqrt(dx**2 + dy**2)
            #print("Distance to goal: ", distance)
            if distance < 0.25:
                rospy.loginfo("Reached the goal position.")
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0

                rospy.loginfo("Publishing cmd_vel: linear %.3f, angular %.3f", cmd_vel.linear.x, cmd_vel.angular.z)

                rospy.loginfo("Flags: default_traj=%s, adjusted_traj=%s, goal=(%.2f, %.2f), pos=(%.2f, %.2f)",
                self.default_trajectory_received,
                self.adjusted_trajectory_received,
                self.adjusted_goal_position.x, self.adjusted_goal_position.y,
                self.current_position.x, self.current_position.y)

                self.cmd_pub.publish(cmd_vel)
                return
            else:
                current_time = rospy.get_time()
                dt = current_time - self.prev_time if self.prev_time is not None else 0.1
                
                # If no collision is detected on the default speed, revert back to that speed
                if not self.default_collision_detected and not self.adjusted_collision_detected:
                    new_lin_vel = self.default_linear_velocity
                    #rospy.loginfo("R1 No collision detected for either path, reverting to default speed %.3f", new_lin_vel)
                elif self.default_collision_detected and not self.adjusted_collision_detected:
                    new_lin_vel = self.previous_lin_vel
                    #rospy.loginfo("R1 default velocity collision, no adjusted collision. keeping adjusted speed %.3f", new_lin_vel)
                elif not self.default_collision_detected and self.adjusted_collision_detected:
                    new_lin_vel = self.default_linear_velocity
                    #rospy.loginfo("R1 adjused velocity collision, no default collision. going back to default speed %.3f", new_lin_vel)
                elif self.default_collision_detected and self.adjusted_collision_detected:
                    new_lin_vel = self.modify_velocity(self.other_velocity, self.consecutive_collisions)
                    #rospy.loginfo("R1 collision detected, adjusting velocity to %.3f", new_lin_vel)
                else:
                    #rospy.loginfo("R1 ERROR IN THE CONTROL LOOP")
                    return
                
                new_vel = Float32()
                new_vel.data = new_lin_vel
                self.new_vel_pub.publish(new_vel)

                # Compute the heading error (desired - current)
                error = self.desired_heading - self.current_heading
                # Normalize the error to [-pi, pi]
                error = math.atan2(math.sin(error), math.cos(error))
                derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
                angular_vel = self.Kp * error + self.Kd * derivative
                # Clip angular velocity to safe limits
                angular_vel = max(-2.0, min(angular_vel, 2.0))
                self.prev_error = error
                self.prev_time = current_time

                # Build and publish the Twist command
                cmd_vel = Twist()
                cmd_vel.linear.x = new_lin_vel
                cmd_vel.angular.z = angular_vel
                
                # Publish only if we have an active trajectory
                if self.adjusted_trajectory_received:
                    self.cmd_pub.publish(cmd_vel)
                    #rospy.loginfo("Publishing cmd_vel: linear %.3f, angular %.3f", new_lin_vel, angular_vel)
                else:
                    pass
                    #rospy.loginfo("No active trajectory.")
                self.previous_lin_vel = new_lin_vel


if __name__ == '__main__':
    try:
        OrcaController()
        rospy.spin()
    except Exception as e:
        rospy.logerr("Exception in ORCA controller: %s", str(e))