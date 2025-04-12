cmake_minimum_required(VERSION 3.0.2)
project(pyorca)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_python_setup()

catkin_install_python(PROGRAMS
  scripts/moving_goal_publisher0.py
  scripts/moving_goal_publisher1.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
