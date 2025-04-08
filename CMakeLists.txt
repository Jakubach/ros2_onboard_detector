cmake_minimum_required(VERSION 3.5)
project(onboard_detector)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -O3)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)


include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/dynamicDetector.cpp
  src/uvDetector.cpp
  src/dbscan.cpp
  src/kalmanFilter.cpp
)

add_executable(detector_node src/detector_node.cpp)

target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(detector_node ${PROJECT_NAME} rclcpp::rclcpp)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  image_transport
  vision_msgs
  PCL
  visualization_msgs
  nav2_msgs
  tf2_ros
  tf2
  tf2_geometry_msgs
  pcl_conversions
)

install(TARGETS
  detector_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
  launch
  cfg
  DESTINATION share/${PROJECT_NAME}/
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
)

ament_package()