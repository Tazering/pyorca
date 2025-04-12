#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class MovingGoalPublisher:
    def __init__(self):
        rospy.loginfo("Initialized Moving Goal Publisher")

        self.lookahead_idx = rospy.get_param("~lookahead_idx", 2)
        self.robot_ns = rospy.get_param("~robot_ns", "r0")

        self.goal_pub = rospy.Publisher(f"/r0/orca_goal", PoseStamped, queue_size=1)
        self.trajectory_sub = rospy.Subscriber(f"/r0/adjusted_traj", Path, self.trajectory_callback)

        self.current_goal = None

    def trajectory_callback(self, path_msg):
        rospy.loginfo("Trajectory callback triggered.")
        if len(path_msg.poses) == 0:
            rospy.logwarn("Received empty trajectory.")
            return

        rospy.loginfo(f"Received trajectory with {len(path_msg.poses)} points.")
        
        # Choose lookahead point
        idx = min(self.lookahead_idx, len(path_msg.poses) - 1)
        target_pose = path_msg.poses[idx]

        # Save and publish
        self.current_goal = target_pose
        self.goal_pub.publish(target_pose)
        rospy.loginfo_throttle(2.0, f"Published new ORCA goal at index {idx}")


if __name__ == '__main__':
    rospy.init_node('moving_goal_publisher0')
    try:
        MovingGoalPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
