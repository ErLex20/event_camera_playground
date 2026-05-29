cmake_minimum_required(VERSION 3.8)
project(evo_utils)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(camera_calibration_parsers REQUIRED)
find_package(minkindr REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED libglog)

add_library(${PROJECT_NAME} SHARED
  src/camera.cpp
  src/utils_geometry.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  ${OpenCV_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  ${OpenCV_LIBS}
  ${GLOG_LIBRARIES})

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  image_geometry
  camera_calibration_parsers
  minkindr)

install(
  DIRECTORY include/
  DESTINATION include)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp sensor_msgs image_geometry camera_calibration_parsers minkindr Eigen3 OpenCV)
ament_export_include_directories(include)

ament_package()
