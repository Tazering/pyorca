#!/usr/bin/env python3
"""
ROS node for Hybrid A* Planning using Current Odometry as the Start

Subscribes to:
    - "/r0/odom" (nav_msgs/Odometry)
    - "/r0/new_velocity" (std_msgs/Float32)

Publishes:
    - "/r0/default_traj" (nav_msgs/Path)
    - "/r0/adjusted_traj" (nav_msgs/Path)
    - "/obstacle_markers" (visualization_msgs/MarkerArray)

This node uses the current odometry message as the starting point for Hybrid A* planning.
The goal position and planning parameters (such as turning limits and step size) are obtained from ROS parameters.
The computed path is published as a nav_msgs/Path message at 1 Hz.
The static obstacles are published as visualization markers so that they can be displayed in RViz.
"""

import rospy
import math
import random
import numpy as np
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped, Point
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

# --- Environment and Hybrid A* Functions (reused) ---
GRID_SIZE = 5

def point_in_polygon(x, y, polygon):
    """
    Determines if the point (x,y) is inside a polygon.
    Uses the ray-casting algorithm.
    ADDED for static obstacle checking.
    """
    num = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(num + 1):
        p2x, p2y = polygon[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def is_point_in_obstacle(x, y, obstacles, padding=0.1):
    """
    Check if a point is inside any of the polygon obstacles.
    (Padding is not used in this simple example.)
    ADDED for static obstacle checking.
    """
    for poly in obstacles:
        if point_in_polygon(x, y, poly):
            return True
    return False

def collision_check_segment(p1, p2, obstacles, walls, padding=0.2, steps=5):
    """
    Check for collision along the segment between p1 and p2 by sampling points.
    ADAPTED to use polygon obstacles from the XML.
    """
    (x1, y1) = p1
    (x2, y2) = p2
    for i in range(steps + 1):
        alpha = i / steps
        xx = x1 + alpha * (x2 - x1)
        yy = y1 + alpha * (y2 - y1)
        # Boundaries: adjusted based on the static environment dimensions.
        if xx < 0 or xx > 5 or yy < 0 or yy > 5:
            return True
        if is_point_in_obstacle(xx, yy, obstacles, padding):
            return True
    # The 'walls' parameter is left unused here.
    return False

def collision_check_hybrid(subpts, obstacles, walls, padding=0.2):
    """
    Check collision along a hybrid arc by checking each segment.
    EDITED to use updated collision_check_segment.
    """
    for i in range(len(subpts) - 1):
        if collision_check_segment(subpts[i], subpts[i+1], obstacles, walls, padding, steps=4):
            return True
    return False

def propagate_arc(x0, y0, h0, steer_deg, step=0.2, n_sub=2):
    ds = step / n_sub
    da = steer_deg / n_sub
    x, y, h = x0, y0, h0
    out_pts = [(x, y)]
    for _ in range(n_sub):
        h_next = h + da
        rad = math.radians(h_next)
        x_next = x + ds * math.cos(rad)
        y_next = y + ds * math.sin(rad)
        x, y, h = x_next, y_next, h_next
        out_pts.append((x, y))
    return (x, y, h), out_pts

def hybrid_astar(obstacles, walls, start_xy, goal_xy, step,
                 heading_deg=0.0, left_deg=30.0, right_deg=30.0,
                 goal_radius=0.2, max_iter=200000, padding=0.2):
    from heapq import heappush, heappop
    (sx, sy) = start_xy
    def keyify(x, y, h):
        return (round(x, 2), round(y, 2), round(h % 360, 1))
    def heuristic(x, y):
        return math.hypot(x - goal_xy[0], y - goal_xy[1])
    start_state = (sx, sy, heading_deg)
    start_key = keyify(*start_state)
    cost = {start_key: 0.0}
    parent = {start_key: None}
    frontier = []
    heappush(frontier, (0.0, start_key))
    step_turn = 15.0
    steer_values = []
    steer_min = -abs(right_deg)
    steer_max = abs(left_deg)
    val = steer_min
    while val <= steer_max + 1e-9:
        steer_values.append(val)
        val += step_turn
    if 0.0 not in steer_values:
        steer_values.append(0.0)
    steer_values = sorted(set(steer_values))
    expansions = 0
    while frontier and expansions < max_iter:
        expansions += 1
        _, current_key = heappop(frontier)
        cx, cy, ch = current_key
        current_cost = cost[current_key]
        if math.hypot(cx - goal_xy[0], cy - goal_xy[1]) < goal_radius:
            path = []
            tmp_key = current_key
            while tmp_key is not None:
                path.append(tmp_key)
                tmp_key = parent[tmp_key]
            path.reverse()
            return path
        for steer_deg in steer_values:
            (nx, ny, nh), arc_pts = propagate_arc(cx, cy, ch, steer_deg, step, n_sub=4)
            if not collision_check_hybrid(arc_pts, obstacles, walls, padding):
                next_key = keyify(nx, ny, nh)
                new_cost = current_cost + step
                if next_key not in cost or new_cost < cost[next_key]:
                    cost[next_key] = new_cost
                    parent[next_key] = current_key
                    f = new_cost + heuristic(nx, ny)
                    heappush(frontier, (f, next_key))
    return []

def build_static_environment():
    """
    Constructs a static environment using obstacles defined in the XML.
    Each obstacle is a polygon given by a list of (x, y) points.
    ADDED to replace the random environment.
    """
    obstacles = []
    # Bottom wall: spans x from 0 to 5, y ~ 0 to 0.1.
    obstacles.append([(0, 0), (5, 0), (5, 0.1), (0, 0.1)])
    # Top wall: spans x from 0 to 5, y ~ 4.9 to 5.
    obstacles.append([(0, 4.9), (5, 4.9), (5, 5.0), (0, 5.0)])
    # Left wall: spans y from 0 to 5, x ~ 0 to 0.1.
    obstacles.append([(0, 0), (0.1, 0), (0.1, 5), (0, 5)])
    # Right wall: spans y from 0 to 5, x ~ 4.9 to 5.
    obstacles.append([(4.9, 0), (5.0, 0), (5.0, 5), (4.9, 5)])
    # Interior wall - lower segment: near x=3, from y=0 to 2.4.
    obstacles.append([(2.0, 0), (3.0, 0), (3.0, 2.2), (2.0, 2.2)])
    # Interior wall - upper segment: near x=3, from y=2.6 to 5.
    obstacles.append([(2.0,2.8), (3.0,2.8), (3.0, 5.0), (2.0, 5.0)])
    walls = []  # No separate walls list is needed.
    return obstacles, walls

# --- Hybrid A* Planner Node Using Timer for 1 Hz Planning ---
class HybridAStarPlanner:
    """
    Subscribes to an odometry topic, updates the latest odometry,
    and runs the Hybrid A* planning at 1 Hz using a timer callback.
    Also publishes static obstacles as visualization markers for RViz.
    """
    def __init__(self):
        self.path_pub = rospy.Publisher("/r0/default_traj", Path, queue_size=10, latch=True)
        self.path_pub_adjusted = rospy.Publisher("/r0/adjusted_traj", Path, queue_size=10, latch=True)
        self.obstacle_marker_pub = rospy.Publisher("/obstacle_markers", MarkerArray, queue_size=10, latch=True)
        self.current_odom = None
        self.new_velocity = 0.1
        
        rospy.Subscriber("/r0/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/r0/new_velocity", Float32, self.velocity_callback)
        # Set up timers to run the planning at 1 Hz.
        self.timer = rospy.Timer(rospy.Duration(2.0), self.timer_callback)
        self.timer_2 = rospy.Timer(rospy.Duration(2.0), self.timer_callback_2)
        
        # Publish the static obstacles as markers.
        self.publish_obstacle_markers()
        rospy.loginfo("Hybrid A* planner node started and obstacle markers published.")

    def velocity_callback(self, msg):
        # Update the new_velocity.
        self.new_velocity = msg.data

    def odom_callback(self, msg):
        # Update the latest odometry message.
        self.current_odom = msg

    def publish_obstacle_markers(self):
        """Publishes the static obstacles as a MarkerArray for visualization in RViz."""
        obstacles, walls = build_static_environment()
        marker_array = MarkerArray()
        for i, poly in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05  # Line width.
            # Set the color to red.
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(0)  # 0 means forever.

            marker.points = []
            for (x, y) in poly:
                p = Point()
                p.x = x
                p.y = y
                p.z = 0.0
                marker.points.append(p)
            # Close the polygon by appending the first point at the end (if not already closed).
            if len(poly) > 0:
                marker.points.append(marker.points[0])
            marker_array.markers.append(marker)
        
        self.obstacle_marker_pub.publish(marker_array)
        rospy.loginfo("Published %d obstacle markers.", len(marker_array.markers))

    def timer_callback(self, event):
        if self.current_odom is None:
            rospy.logwarn("No odometry received yet.")
            return

        # Extract current pose from the odometry.
        x = self.current_odom.pose.pose.position.x
        y = self.current_odom.pose.pose.position.y
        q = self.current_odom.pose.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quaternion)
        yaw_deg = math.degrees(yaw)
        start_xy = (x, y)
        start_heading = yaw_deg

        # Read goal and planning parameters from ROS parameters.
        goal_x = rospy.get_param("r0/goal_x", 4)
        goal_y = rospy.get_param("r0/goal_y", 2)
        goal_xy = (goal_x, goal_y)
        left_deg = rospy.get_param("r0/left_deg", 30.0)
        right_deg = rospy.get_param("r0/right_deg", 30.0)
        step = rospy.get_param("r0/step", 0.1)

        # Build the planning environment.
        obstacles, walls = build_static_environment()

        # Compute the Hybrid A* path.
        path = hybrid_astar(obstacles, walls, start_xy, goal_xy, step,
                            heading_deg=start_heading, left_deg=left_deg, right_deg=right_deg,
                            goal_radius=0.2, max_iter=40000, padding=0.2)
        if not path:
            rospy.logwarn("R0 No path found by Hybrid A* from current odometry.")
            return

        # Create the Path message.
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "odom"
        for waypoint in path:
            x_wp, y_wp, yaw_wp = waypoint
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = x_wp
            pose_stamped.pose.position.y = y_wp
            quat = quaternion_from_euler(0, 0, math.radians(yaw_wp))
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]
            path_msg.poses.append(pose_stamped)

        # Publish the planned path.
        self.path_pub.publish(path_msg)
        rospy.loginfo("Published default Hybrid A* path.")

    def timer_callback_2(self, event):
        if self.current_odom is None:
            rospy.logwarn("No odometry received yet.")
            return

        # Extract current pose from the odometry.
        x = self.current_odom.pose.pose.position.x
        y = self.current_odom.pose.pose.position.y
        q = self.current_odom.pose.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quaternion)
        yaw_deg = math.degrees(yaw)
        start_xy = (x, y)
        start_heading = yaw_deg

        # Read goal and planning parameters from ROS parameters.
        goal_x = rospy.get_param("r0/goal_x", 4.0)
        goal_y = rospy.get_param("r0/goal_y", 2.0)
        goal_xy = (goal_x, goal_y)
        left_deg = rospy.get_param("r0/left_deg", 15.0)
        right_deg = rospy.get_param("r0/right_deg", 15.0)
        step = rospy.get_param("r0/step", 0.25)

        # Build the planning environment.
        obstacles, walls = build_static_environment()

        # Compute the Hybrid A* path using new_velocity.
        path = hybrid_astar(obstacles, walls, start_xy, goal_xy, self.new_velocity,
                            heading_deg=start_heading, left_deg=left_deg, right_deg=right_deg,
                            goal_radius=0.5, max_iter=40000, padding=0.2)
        if not path:
            rospy.logwarn("No path found by Hybrid A* from current odometry.")
            return

        # Create the Path message.
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "odom"
        for waypoint in path:
            x_wp, y_wp, yaw_wp = waypoint
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = x_wp
            pose_stamped.pose.position.y = y_wp
            quat = quaternion_from_euler(0, 0, math.radians(yaw_wp))
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]
            path_msg.poses.append(pose_stamped)

        # Publish the adjusted path.
        self.path_pub_adjusted.publish(path_msg)
        rospy.loginfo("Published adjusted Hybrid A* path.")

def main():
    rospy.init_node("trajectory_planner_r0", anonymous=False)
    planner = HybridAStarPlanner()
    rospy.spin()

if __name__ == '__main__':
    main()