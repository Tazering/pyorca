#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path

class OrcaController:
    def __init__(self):
        # Read robot namespace from parameter
        self.robot_ns = rospy.get_param("~robot_ns", "r1")

        # Set up publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher(f"/r0/cmd_vel", Twist, queue_size=10)
        rospy.loginfo(f"Publishing to: /r0/cmd_vel")
        self.goal_sub = rospy.Subscriber(f"/r0/orca_goal", PoseStamped, self.goal_callback)
        self.trajectory_sub = rospy.Subscriber(f"/r0/adjusted_traj", Path, self.trajectory_callback)

        # Current target goal
        self.current_goal = None
        self.current_trajectory = None

        rospy.loginfo(f"[{self.robot_ns}] ORCA controller initialized.")

    def goal_callback(self, msg):
        self.current_goal = msg.pose
        rospy.loginfo(f"[r0] Received ORCA goal: "
                      f"({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def trajectory_callback(self, msg):
        self.current_trajectory = msg
        rospy.loginfo(f"[r0] Received trajectory with {len(msg.poses)} points.")

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.current_goal:
                if self.current_trajectory and len(self.current_trajectory.poses) > 0:
                    goal_pose = self.current_trajectory.poses[0].pose  # Just for example: use the first trajectory point

                    # Simple movement towards the goal point
                    cmd = Twist()
                    cmd.linear.x = 0.2  # Move forward
                    cmd.angular.z = self.calculate_angular_velocity(goal_pose)
                    self.cmd_vel_pub.publish(cmd)
                    rospy.loginfo_throttle(2.0, f"[r0] Moving towards goal at "
                                                f"({goal_pose.position.x}, {goal_pose.position.y})")

            rate.sleep()
    
    def calculate_angular_velocity(self, goal_pose):
        # Calculate the angular velocity needed to turn towards the goal
        angle_to_goal = self.calculate_angle_to_goal(goal_pose)
        return angle_to_goal  # Example: simple angular velocity towards goal

    def calculate_angle_to_goal(self, goal_pose):
        # Placeholder for angle calculation logic
        return 0.0  # Update this with actual logic to compute angle to goal

if __name__ == '__main__':
    rospy.init_node('orca_controller1')
    try:
        controller = OrcaController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
