/*
 * ros_params_helper.h  (ROS2 port)
 *
 * Original (ROS1) author: cforster, from libpointmatcher_ros.
 *
 * ROS2 port for the merged EVO node. The original EVO stages read parameters
 * through rpg_common_ros::param<T>(nh, name, default), using two node handles:
 *   - the global handle (nh_)  -> shared parameters (e.g. world_frame_id)
 *   - the private handle (nhp_) -> node-private parameters (e.g. frame_size)
 *
 * In the single merged node every stage shares one rclcpp::Node, so the private
 * parameters of the different stages would collide (both mapping and tracking
 * define frame_size, etc.). To preserve the original call sites *verbatim* while
 * avoiding collisions, the node handle is replaced by a ParamProvider that
 * carries the node pointer plus a namespace prefix:
 *   - nh_  = {node, ""}            -> shared parameters
 *   - nhp_ = {node, "<stage>."}    -> stage-private parameters
 *
 * Parameters are declared lazily on first access (using the supplied default),
 * so YAML/launch overrides given as "<prefix><name>: value" still take effect.
 */

#pragma once

#include <rclcpp/rclcpp.hpp>

#include <string>
#include <type_traits>

namespace rpg_common_ros {

/// Replacement for ros::NodeHandle in the EVO stages.
struct ParamProvider {
  rclcpp::Node * node = nullptr;  ///< owning node
  std::string prefix;             ///< parameter namespace prefix ("" or "<stage>.")

  ParamProvider() = default;
  ParamProvider(rclcpp::Node * n, const std::string & p) : node(n), prefix(p) {}

  bool hasParam(const std::string & name) const
  {
    return node->has_parameter(prefix + name);
  }
};

/**
 * Read parameter <prefix><name>, declaring it (with defaultValue) on first
 * access so that YAML/launch overrides are honoured.
 */
template <typename T>
T param(const ParamProvider & nh, const std::string & name, const T & defaultValue)
{
  const std::string full = nh.prefix + name;
  // ROS2 parameters only support bool / int64 / double / string (+ arrays),
  // while the EVO stages use bool / int / size_t / float / double / string.
  // Dispatch on the requested C++ type, declaring with the matching ROS2 type
  // so that YAML/launch overrides are picked up.
  if constexpr (std::is_same_v<T, bool>) {
    if (!nh.node->has_parameter(full)) {
      nh.node->declare_parameter(full, rclcpp::ParameterValue(static_cast<bool>(defaultValue)));
    }
    return nh.node->get_parameter(full).as_bool();
  } else if constexpr (std::is_integral_v<T>) {
    if (!nh.node->has_parameter(full)) {
      nh.node->declare_parameter(full, rclcpp::ParameterValue(static_cast<int64_t>(defaultValue)));
    }
    return static_cast<T>(nh.node->get_parameter(full).as_int());
  } else if constexpr (std::is_floating_point_v<T>) {
    if (!nh.node->has_parameter(full)) {
      nh.node->declare_parameter(full, rclcpp::ParameterValue(static_cast<double>(defaultValue)));
    }
    return static_cast<T>(nh.node->get_parameter(full).as_double());
  } else {  // std::string and other directly-supported types
    if (!nh.node->has_parameter(full)) {
      nh.node->declare_parameter(full, rclcpp::ParameterValue(defaultValue));
    }
    return nh.node->get_parameter(full).get_value<T>();
  }
}

inline bool hasParam(const ParamProvider & nh, const std::string & name)
{
  return nh.hasParam(name);
}

}  // namespace rpg_common_ros

namespace rpg_ros = rpg_common_ros;
