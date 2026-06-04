# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_event_detector_cpp_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED event_detector_cpp_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(event_detector_cpp_FOUND FALSE)
  elseif(NOT event_detector_cpp_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(event_detector_cpp_FOUND FALSE)
  endif()
  return()
endif()
set(_event_detector_cpp_CONFIG_INCLUDED TRUE)

# output package information
if(NOT event_detector_cpp_FIND_QUIETLY)
  message(STATUS "Found event_detector_cpp: 0.1.0 (${event_detector_cpp_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'event_detector_cpp' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT event_detector_cpp_DEPRECATED_QUIET)
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(event_detector_cpp_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${event_detector_cpp_DIR}/${_extra}")
endforeach()
