cmake_minimum_required(VERSION 3.0.2)
project(sagittarius_sort_pipeline)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  actionlib
  actionlib_msgs
  cv_bridge
  sagittarius_object_color_detector
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    sensor_msgs
    std_msgs
    actionlib
    actionlib_msgs
    cv_bridge
    sagittarius_object_color_detector
)

catkin_install_python(PROGRAMS
  nodes/sort_main.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
