cmake_minimum_required(VERSION 3.8)
project(minkindr)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# glog (system, via pkg-config or find_package)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED libglog)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  ${GLOG_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen ${GLOG_LIBRARIES})

install(
  DIRECTORY include/
  DESTINATION include)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)
ament_export_include_directories(include)

ament_package()
