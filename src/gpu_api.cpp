#include "bfnew/gpu_api.hpp"

#include <stdexcept>

namespace bfnew {
namespace {

[[nodiscard]] bool valid_engine(const EngineKind engine) noexcept {
  switch (engine) {
    case EngineKind::jacobi_pull:
    case EngineKind::dense_chaotic_push:
    case EngineKind::frontier_push:
      return true;
  }
  return false;
}

[[nodiscard]] bool valid_control_mode(const ControlMode mode) noexcept {
  switch (mode) {
    case ControlMode::persistent_cooperative:
    case ControlMode::chunked_host_poll:
    case ControlMode::per_round_host_poll:
      return true;
  }
  return false;
}

[[nodiscard]] bool valid_grid_policy(const GridPolicy policy) noexcept {
  switch (policy) {
    case GridPolicy::occupancy_derived:
    case GridPolicy::fixed_blocks_per_wgp:
      return true;
  }
  return false;
}

[[nodiscard]] bool valid_instrumentation(
    const InstrumentationLevel instrumentation) noexcept {
  switch (instrumentation) {
    case InstrumentationLevel::none:
    case InstrumentationLevel::light:
    case InstrumentationLevel::debug:
      return true;
  }
  return false;
}

[[nodiscard]] bool valid_stop_reason(const std::uint32_t encoded) noexcept {
  return encoded <= static_cast<std::uint32_t>(DeviceStopReason::device_failure);
}

[[nodiscard]] std::uint32_t expected_error_bits(
    const DeviceStopReason reason) noexcept {
  switch (reason) {
    case DeviceStopReason::queue_overflow:
      return device_error::queue_overflow;
    case DeviceStopReason::invalid_controller_state:
      return device_error::invalid_controller_state;
    case DeviceStopReason::device_failure:
      return device_error::device_failure;
    case DeviceStopReason::none:
    case DeviceStopReason::converged:
    case DeviceStopReason::maximum_rounds:
      return device_error::none;
  }
  return device_error::none;
}

}  // namespace

GpuRunOptionsError validate_gpu_run_options(const GpuRunOptions& options) noexcept {
  if (!valid_engine(options.engine)) {
    return GpuRunOptionsError::invalid_engine;
  }
  if (!valid_control_mode(options.control_mode)) {
    return GpuRunOptionsError::invalid_control_mode;
  }
  if (options.rounds_per_chunk == 0U) {
    return GpuRunOptionsError::invalid_rounds_per_chunk;
  }
  if (options.block_size == 0U) {
    return GpuRunOptionsError::invalid_block_size;
  }
  if (!valid_grid_policy(options.grid_policy)) {
    return GpuRunOptionsError::invalid_grid_policy;
  }
  if (options.grid_policy == GridPolicy::fixed_blocks_per_wgp &&
      options.blocks_per_wgp == 0U) {
    return GpuRunOptionsError::invalid_blocks_per_wgp;
  }
  if (options.grid_policy == GridPolicy::occupancy_derived &&
      options.blocks_per_wgp != 0U) {
    return GpuRunOptionsError::invalid_blocks_per_wgp;
  }
  if (!valid_instrumentation(options.instrumentation)) {
    return GpuRunOptionsError::invalid_instrumentation;
  }
  if (options.maximum_rounds == 0U) {
    return GpuRunOptionsError::invalid_maximum_rounds;
  }
  if (options.enable_per_lane_convergence > 1U) {
    return GpuRunOptionsError::invalid_per_lane_flag;
  }
  return GpuRunOptionsError::none;
}

DeviceController initialize_device_controller(
    const GpuRunOptions& options,
    const LaneMask valid_lane_mask,
    const std::uint64_t initial_frontier_size) {
  if (validate_gpu_run_options(options) != GpuRunOptionsError::none) {
    throw std::invalid_argument{"cannot initialize a controller from invalid options"};
  }
  if (valid_lane_mask == 0U) {
    throw std::invalid_argument{"device controller requires at least one valid lane"};
  }
  if (options.engine == EngineKind::frontier_push &&
      initial_frontier_size == 0U) {
    throw std::invalid_argument{
        "frontier controller requires a nonempty initial frontier"};
  }

  DeviceController controller;
  controller.valid_lane_mask = valid_lane_mask;
  controller.active_lane_mask = valid_lane_mask;
  controller.execute_lane_mask = valid_lane_mask;
  controller.engine_kind = static_cast<std::uint32_t>(options.engine);
  controller.enable_per_lane_convergence = options.enable_per_lane_convergence;
  controller.maximum_rounds = options.maximum_rounds;
  controller.distance_read_slot = 0U;
  controller.distance_write_slot =
      options.engine == EngineKind::jacobi_pull ? 1U : 0U;
  controller.frontier_read_slot = 0U;
  controller.frontier_write_slot = 1U;
  controller.frontier_size[0] =
      options.engine == EngineKind::frontier_push ? initial_frontier_size : 0U;
  return controller;
}

DeviceControllerError validate_device_controller(
    const DeviceController& controller) noexcept {
  const auto engine = static_cast<EngineKind>(controller.engine_kind);
  if (!valid_engine(engine)) {
    return DeviceControllerError::invalid_engine;
  }
  if (controller.valid_lane_mask == 0U) {
    return DeviceControllerError::empty_valid_lanes;
  }
  const LaneMask combined = controller.active_lane_mask |
                            controller.changed_lane_mask |
                            controller.converged_lane_mask |
                            controller.execute_lane_mask |
                            controller.next_frontier_lane_mask;
  if ((combined & ~controller.valid_lane_mask) != 0U) {
    return DeviceControllerError::mask_outside_valid_lanes;
  }
  if ((controller.active_lane_mask & controller.converged_lane_mask) != 0U) {
    return DeviceControllerError::active_converged_overlap;
  }
  if (controller.done == 0U &&
      controller.execute_lane_mask != controller.active_lane_mask) {
    return DeviceControllerError::execute_mask_mismatch;
  }
  if (controller.done != 0U && controller.execute_lane_mask != 0U) {
    return DeviceControllerError::execute_mask_mismatch;
  }
  if (controller.enable_per_lane_convergence > 1U) {
    return DeviceControllerError::invalid_per_lane_flag;
  }
  if (controller.maximum_rounds == 0U) {
    return DeviceControllerError::invalid_round_limit;
  }
  if (controller.rounds_completed > controller.maximum_rounds) {
    return DeviceControllerError::completed_round_overflow;
  }
  if (controller.distance_read_slot > 1U ||
      controller.distance_write_slot > 1U ||
      (engine == EngineKind::jacobi_pull &&
       controller.distance_read_slot == controller.distance_write_slot) ||
      (engine != EngineKind::jacobi_pull &&
       (controller.distance_read_slot != 0U ||
        controller.distance_write_slot != 0U))) {
    return DeviceControllerError::invalid_distance_slots;
  }
  if (controller.frontier_read_slot > 1U ||
      controller.frontier_write_slot > 1U ||
      controller.frontier_read_slot == controller.frontier_write_slot) {
    return DeviceControllerError::invalid_frontier_slots;
  }
  if (engine == EngineKind::frontier_push && controller.done == 0U &&
      controller.frontier_size[controller.frontier_read_slot] == 0U) {
    return DeviceControllerError::invalid_frontier_slots;
  }
  if (controller.done > 1U) {
    return DeviceControllerError::invalid_done_flag;
  }
  if (!valid_stop_reason(controller.stop_reason)) {
    return DeviceControllerError::invalid_stop_reason;
  }
  const bool has_stop_reason =
      controller.stop_reason != static_cast<std::uint32_t>(DeviceStopReason::none);
  if ((controller.done != 0U) != has_stop_reason) {
    return DeviceControllerError::inconsistent_done_state;
  }
  const auto reason = static_cast<DeviceStopReason>(controller.stop_reason);
  if ((controller.error_bits & ~device_error::known_mask) != 0U) {
    return DeviceControllerError::unknown_error_bits;
  }
  if (controller.error_bits != expected_error_bits(reason)) {
    return DeviceControllerError::inconsistent_error_state;
  }
  if (controller.done != 0U &&
      (controller.changed_lane_mask != 0U ||
       controller.next_frontier_lane_mask != 0U)) {
    return DeviceControllerError::inconsistent_terminal_state;
  }
  if (reason == DeviceStopReason::converged &&
      (controller.active_lane_mask != 0U ||
       controller.converged_lane_mask != controller.valid_lane_mask)) {
    return DeviceControllerError::inconsistent_terminal_state;
  }
  if (reason == DeviceStopReason::maximum_rounds &&
      controller.rounds_completed != controller.maximum_rounds) {
    return DeviceControllerError::inconsistent_terminal_state;
  }
  return DeviceControllerError::none;
}

DeviceRunStatusError validate_device_run_status(
    const DeviceRunStatus& status) noexcept {
  if (status.final_distance_slot > 1U) {
    return DeviceRunStatusError::invalid_distance_slot;
  }
  if (status.converged > 1U) {
    return DeviceRunStatusError::invalid_converged_flag;
  }
  if (status.valid_lane_mask == 0U) {
    return DeviceRunStatusError::empty_valid_lanes;
  }
  const LaneMask combined = status.reached_target_mask |
                            status.bounding_box_miss_mask |
                            status.active_lane_mask |
                            status.converged_lane_mask;
  if ((combined & ~status.valid_lane_mask) != 0U) {
    return DeviceRunStatusError::mask_outside_valid_lanes;
  }
  if ((status.active_lane_mask & status.converged_lane_mask) != 0U) {
    return DeviceRunStatusError::active_converged_overlap;
  }
  if ((status.reached_target_mask & status.bounding_box_miss_mask) != 0U) {
    return DeviceRunStatusError::reached_miss_overlap;
  }
  if (!valid_stop_reason(status.stop_reason)) {
    return DeviceRunStatusError::invalid_stop_reason;
  }
  const auto reason = static_cast<DeviceStopReason>(status.stop_reason);
  if ((status.converged != 0U) != (reason == DeviceStopReason::converged)) {
    return DeviceRunStatusError::inconsistent_convergence_state;
  }
  if (reason == DeviceStopReason::converged &&
      (status.active_lane_mask != 0U ||
       status.converged_lane_mask != status.valid_lane_mask)) {
    return DeviceRunStatusError::inconsistent_terminal_state;
  }
  if ((status.error_bits & ~device_error::known_mask) != 0U) {
    return DeviceRunStatusError::unknown_error_bits;
  }
  if (status.error_bits != expected_error_bits(reason)) {
    return DeviceRunStatusError::inconsistent_error_state;
  }
  if (status.error_bits != device_error::none &&
      status.bounding_box_miss_mask != 0U) {
    return DeviceRunStatusError::error_reported_as_bounding_box_miss;
  }
  return DeviceRunStatusError::none;
}

}  // namespace bfnew
