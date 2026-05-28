cmake_minimum_required(VERSION 3.12)
project(event_detector_cpp)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options(-Wno-array-bounds) # Eigen warning
endif()

cmake_policy(SET CMP0146 OLD)
cmake_policy(SET CMP0074 OLD)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(dua_app_management REQUIRED)
find_package(dua_common_interfaces REQUIRED)
find_package(dua_cv_bridge REQUIRED)
find_package(dua_geometry_interfaces REQUIRED)
find_package(dua_hardware_interfaces REQUIRED)
find_package(dua_node_cpp REQUIRED)
find_package(dua_qos_cpp REQUIRED)
find_package(dv-processing REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(event_camera_codecs REQUIRED)
find_package(event_camera_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(params_manager_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(simple_serviceclient_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(Threads REQUIRED)

# Configure include directories from external libraries
include_directories(
  include)

# Generate parameters source code
generate_init_parameters(
  YAML_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/event_detector_cpp/params.yaml"
  OUT_FILE "init_parameters.cpp")

# Configure Event Detector node source files
file(GLOB EVENT_DETECTOR_NODE_SRCS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "src/event_detector_cpp/*.cpp")
list(APPEND EVENT_DETECTOR_NODE_SRCS init_parameters.cpp)

# Event Detector node
add_library(event_detector_cpp SHARED ${EVENT_DETECTOR_NODE_SRCS})
target_compile_definitions(event_detector_cpp PRIVATE COMPOSITION_BUILDING_DLL)
target_include_directories(event_detector_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/event_detector_cpp>)
target_compile_features(event_detector_cpp PUBLIC c_std_99 cxx_std_17) # Require C99 and C++17
target_link_libraries(event_detector_cpp
  Eigen3::Eigen
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
  Threads::Threads
  dv::processing
)
ament_target_dependencies(event_detector_cpp
  dua_common_interfaces
  dua_cv_bridge
  dua_geometry_interfaces
  dua_hardware_interfaces
  dua_node_cpp
  dua_qos_cpp
  event_camera_codecs
  event_camera_msgs
  params_manager_cpp
  rclcpp
  rclcpp_components
  sensor_msgs
  simple_serviceclient_cpp
  std_msgs
  std_srvs
)

rclcpp_components_register_nodes(event_detector_cpp "event_detector_cpp::EventDetector")

# Event Detector standalone application
add_executable(event_detector_cpp_app
  src/event_detector_cpp_app.cpp)
target_include_directories(event_detector_cpp_app PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(event_detector_cpp_app PUBLIC c_std_99 cxx_std_17) # Require C99 and C++17
target_link_libraries(event_detector_cpp_app event_detector_cpp)
ament_target_dependencies(event_detector_cpp_app
  dua_app_management
  rclcpp)

# Install Event Detector component
install(TARGETS event_detector_cpp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Install Event Detector app
install(TARGETS event_detector_cpp_app
  DESTINATION lib/${PROJECT_NAME})

# Install Config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

# Install Launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
