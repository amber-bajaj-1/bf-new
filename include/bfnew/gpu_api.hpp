#pragma once

#include "bfnew/query.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace bfnew {

using LaneMask = std::uint32_t;
inline constexpr std::uint32_t maximum_batch_lanes = 32U;

enum class EngineKind : std::uint32_t {
  jacobi_pull = 0U,
  dense_chaotic_push = 1U,
  frontier_push = 2U,
};

enum class ControlMode : std::uint32_t {
  persistent_cooperative = 0U,
  chunked_host_poll = 1U,
  per_round_host_poll = 2U,
};

enum class GridPolicy : std::uint32_t {
  occupancy_derived = 0U,
  fixed_blocks_per_wgp = 1U,
};

enum class InstrumentationLevel : std::uint32_t {
  none = 0U,
  light = 1U,
  debug = 2U,
};

enum class DeviceStopReason : std::uint32_t {
  none = 0U,
  converged = 1U,
  maximum_rounds = 2U,
  queue_overflow = 3U,
  invalid_controller_state = 4U,
  device_failure = 5U,
};

namespace device_error {

inline constexpr std::uint32_t none = 0U;
inline constexpr std::uint32_t queue_overflow = 1U << 0U;
inline constexpr std::uint32_t invalid_controller_state = 1U << 1U;
inline constexpr std::uint32_t device_failure = 1U << 2U;
inline constexpr std::uint32_t known_mask =
    queue_overflow | invalid_controller_state | device_failure;

}  // namespace device_error

struct GpuRunOptions {
  EngineKind engine{EngineKind::jacobi_pull};
  ControlMode control_mode{ControlMode::chunked_host_poll};
  std::uint32_t rounds_per_chunk{8U};
  std::uint32_t block_size{256U};
  GridPolicy grid_policy{GridPolicy::occupancy_derived};
  std::uint32_t blocks_per_wgp{};
  InstrumentationLevel instrumentation{InstrumentationLevel::none};
  std::uint64_t maximum_rounds{1U};
  std::uint32_t enable_per_lane_convergence{1U};
};

enum class GpuRunOptionsError : std::uint8_t {
  none,
  invalid_engine,
  invalid_control_mode,
  invalid_rounds_per_chunk,
  invalid_block_size,
  invalid_grid_policy,
  invalid_blocks_per_wgp,
  invalid_instrumentation,
  invalid_maximum_rounds,
  invalid_per_lane_flag,
};

[[nodiscard]] GpuRunOptionsError validate_gpu_run_options(
    const GpuRunOptions& options) noexcept;

struct DeviceController {
  LaneMask valid_lane_mask{};
  LaneMask active_lane_mask{};
  LaneMask changed_lane_mask{};
  LaneMask converged_lane_mask{};
  LaneMask execute_lane_mask{};
  std::uint32_t engine_kind{};
  std::uint32_t enable_per_lane_convergence{};
  std::uint64_t rounds_completed{};
  std::uint64_t maximum_rounds{};
  std::uint32_t distance_read_slot{};
  std::uint32_t distance_write_slot{};
  std::uint32_t frontier_read_slot{};
  std::uint32_t frontier_write_slot{};
  std::uint64_t frontier_size[2]{};
  LaneMask next_frontier_lane_mask{};
  std::uint32_t done{};
  std::uint32_t stop_reason{};
  std::uint32_t error_bits{};
};

enum class DeviceControllerError : std::uint8_t {
  none,
  invalid_engine,
  empty_valid_lanes,
  mask_outside_valid_lanes,
  active_converged_overlap,
  execute_mask_mismatch,
  invalid_per_lane_flag,
  invalid_round_limit,
  completed_round_overflow,
  invalid_distance_slots,
  invalid_frontier_slots,
  invalid_done_flag,
  invalid_stop_reason,
  inconsistent_done_state,
  inconsistent_terminal_state,
  unknown_error_bits,
  inconsistent_error_state,
};

[[nodiscard]] DeviceController initialize_device_controller(
    const GpuRunOptions& options,
    LaneMask valid_lane_mask,
    std::uint64_t initial_frontier_size = 0U);

[[nodiscard]] DeviceControllerError validate_device_controller(
    const DeviceController& controller) noexcept;

struct DeviceWorkStatistics {
  std::uint64_t edges_examined{};
  std::uint64_t successful_decreases{};
  std::uint64_t active_vertices{};
  std::uint64_t active_lane_rounds{};
  std::uint64_t maximum_queue_size{};
  std::uint64_t host_checks{};
  std::uint64_t host_synchronizations{};
  std::uint64_t controller_copies{};
  std::uint64_t kernel_dispatches{};
  std::uint64_t expansion_count{};
  std::uint64_t atomic_attempts{};
  std::uint64_t successful_atomic_updates{};
  std::uint64_t queue_claims{};
  std::uint64_t duplicate_suppressions{};
  std::uint64_t mask_operations{};
  std::uint64_t overflow_events{};
  // Dense-chaotic instrumentation. A high-contention destination has at
  // least two admitted incoming edges for the query. The count is unique per
  // query, while changed-flag updates count block-level publications.
  std::uint64_t high_contention_destinations{};
  std::uint64_t changed_flag_updates{};
  std::uint64_t full_edge_rounds{};
  // Frontier-push instrumentation. An empty-frontier round is the executed
  // terminal round that produces no next work; a small frontier contains
  // fewer than one wave32 of current entries.
  std::uint64_t empty_frontier_rounds{};
  std::uint64_t small_frontier_rounds{};
};

struct DeviceRunStatus {
  std::uint32_t final_distance_slot{};
  std::uint32_t converged{};
  std::uint64_t rounds_completed{};
  LaneMask reached_target_mask{};
  LaneMask bounding_box_miss_mask{};
  LaneMask valid_lane_mask{};
  LaneMask active_lane_mask{};
  LaneMask converged_lane_mask{};
  std::uint32_t stop_reason{};
  std::uint32_t error_bits{};
};

enum class DeviceRunStatusError : std::uint8_t {
  none,
  invalid_distance_slot,
  invalid_converged_flag,
  empty_valid_lanes,
  mask_outside_valid_lanes,
  active_converged_overlap,
  reached_miss_overlap,
  invalid_stop_reason,
  inconsistent_convergence_state,
  inconsistent_terminal_state,
  unknown_error_bits,
  inconsistent_error_state,
  error_reported_as_bounding_box_miss,
};

[[nodiscard]] DeviceRunStatusError validate_device_run_status(
    const DeviceRunStatus& status) noexcept;

struct GpuSsspResult {
  std::uint32_t engine_kind{};
  std::uint32_t control_mode{};
  DeviceRunStatus status{};
  DeviceWorkStatistics work{};
};

class GpuSsspEngine {
 public:
  virtual ~GpuSsspEngine() = default;

  [[nodiscard]] virtual EngineKind kind() const noexcept = 0;
  [[nodiscard]] virtual GpuSsspResult run(
      const RouteQuery& query,
      const GpuRunOptions& options) = 0;
};

static_assert(std::is_standard_layout_v<DeviceController>);
static_assert(std::is_trivially_copyable_v<DeviceController>);
static_assert(std::is_standard_layout_v<GpuRunOptions>);
static_assert(std::is_trivially_copyable_v<GpuRunOptions>);
static_assert(std::is_standard_layout_v<DeviceWorkStatistics>);
static_assert(std::is_trivially_copyable_v<DeviceWorkStatistics>);
static_assert(std::is_standard_layout_v<DeviceRunStatus>);
static_assert(std::is_trivially_copyable_v<DeviceRunStatus>);
static_assert(std::is_standard_layout_v<GpuSsspResult>);
static_assert(std::is_trivially_copyable_v<GpuSsspResult>);

// The host compiler and HIP compiler must agree on these copied ABI records.
static_assert(sizeof(GpuRunOptions) == 48U);
static_assert(alignof(GpuRunOptions) == 8U);
static_assert(sizeof(DeviceController) == 96U);
static_assert(alignof(DeviceController) == 8U);
static_assert(sizeof(DeviceRunStatus) == 48U);
static_assert(alignof(DeviceRunStatus) == 8U);
static_assert(sizeof(DeviceWorkStatistics) == 168U);
static_assert(alignof(DeviceWorkStatistics) == 8U);
static_assert(sizeof(GpuSsspResult) == 224U);
static_assert(alignof(GpuSsspResult) == 8U);
static_assert(offsetof(GpuRunOptions, maximum_rounds) == 32U);
static_assert(offsetof(GpuRunOptions, enable_per_lane_convergence) == 40U);
static_assert(offsetof(DeviceController, rounds_completed) == 32U);
static_assert(offsetof(DeviceController, frontier_size) == 64U);
static_assert(offsetof(DeviceController, next_frontier_lane_mask) == 80U);
static_assert(offsetof(DeviceController, error_bits) == 92U);
static_assert(offsetof(DeviceRunStatus, rounds_completed) == 8U);
static_assert(offsetof(DeviceRunStatus, reached_target_mask) == 16U);
static_assert(offsetof(DeviceRunStatus, error_bits) == 40U);
static_assert(
    offsetof(DeviceWorkStatistics, high_contention_destinations) == 128U);
static_assert(offsetof(DeviceWorkStatistics, changed_flag_updates) == 136U);
static_assert(offsetof(DeviceWorkStatistics, full_edge_rounds) == 144U);
static_assert(offsetof(DeviceWorkStatistics, empty_frontier_rounds) == 152U);
static_assert(offsetof(DeviceWorkStatistics, small_frontier_rounds) == 160U);
static_assert(offsetof(GpuSsspResult, status) == 8U);
static_assert(offsetof(GpuSsspResult, work) == 56U);

}  // namespace bfnew
