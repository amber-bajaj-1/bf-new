#pragma once

#include "bfnew/device_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/query.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace bfnew {

// Phase 10 stores one in-place distance word per vertex. The word is the
// unsigned IEEE-754 representation of a finite nonnegative float or +infinity.
// In that restricted domain unsigned integer order is exactly float order.
struct DenseScratchLayout {
  std::uint64_t vertex_count{};
  std::uint64_t distance_bits_bytes{};
  std::uint64_t total_bytes{};

  constexpr bool operator==(const DenseScratchLayout&) const noexcept = default;
};

[[nodiscard]] DenseScratchLayout make_dense_scratch_layout(
    std::size_t vertex_count);

[[nodiscard]] bool is_dense_atomic_float(float value) noexcept;
[[nodiscard]] std::uint32_t dense_atomic_float_bits(float value);
[[nodiscard]] float dense_atomic_bits_float(std::uint32_t bits);

struct DenseAtomicMinResult {
  std::uint32_t final_bits{};
  bool decreased{};
};

// Portable model of the bitwise atomic-min decision. The HIP implementation
// uses the same unsigned comparison inside a CAS loop.
[[nodiscard]] DenseAtomicMinResult dense_atomic_min_bits(
    std::uint32_t current_bits,
    std::uint32_t candidate_bits);

struct DenseDistanceView {
  std::uint32_t* distance_bits{};
  std::uint32_t vertex_count{};
};

#if defined(__HIPCC__)
#define BFNEW_DENSE_HOST_DEVICE __host__ __device__
#else
#define BFNEW_DENSE_HOST_DEVICE
#endif

[[nodiscard]] BFNEW_DENSE_HOST_DEVICE inline DenseDistanceView
bind_dense_distance_view(
    void* const scratch,
    const std::uint64_t scratch_bytes,
    const std::uint32_t vertex_count) noexcept {
  const std::uint64_t required_bytes =
      static_cast<std::uint64_t>(vertex_count) * sizeof(std::uint32_t);
  if (scratch == nullptr || vertex_count == 0U ||
      scratch_bytes < required_bytes) {
    return {};
  }
  return DenseDistanceView{
      static_cast<std::uint32_t*>(scratch),
      vertex_count,
  };
}

[[nodiscard]] BFNEW_DENSE_HOST_DEVICE inline bool valid_dense_distance_view(
    const DenseDistanceView& view) noexcept {
  return view.distance_bits != nullptr && view.vertex_count != 0U;
}

enum class DenseAdvanceResult : std::uint32_t {
  no_op = 0U,
  continue_execution = 1U,
  converged = 2U,
  maximum_rounds = 3U,
  invalid_controller_state = 4U,
};

namespace detail {

BFNEW_DENSE_HOST_DEVICE inline void fail_dense_controller(
    DeviceController& controller) noexcept {
  if (controller.valid_lane_mask == 0U) {
    controller.valid_lane_mask = LaneMask{1U};
  }
  controller.active_lane_mask = 0U;
  controller.changed_lane_mask = 0U;
  controller.converged_lane_mask = 0U;
  controller.execute_lane_mask = 0U;
  controller.next_frontier_lane_mask = 0U;
  controller.engine_kind =
      static_cast<std::uint32_t>(EngineKind::dense_chaotic_push);
  controller.enable_per_lane_convergence =
      controller.enable_per_lane_convergence > 1U
          ? 1U
          : controller.enable_per_lane_convergence;
  if (controller.maximum_rounds == 0U) {
    controller.maximum_rounds = 1U;
  }
  if (controller.rounds_completed > controller.maximum_rounds) {
    controller.rounds_completed = controller.maximum_rounds;
  }
  controller.distance_read_slot = 0U;
  controller.distance_write_slot = 0U;
  controller.frontier_read_slot = 0U;
  controller.frontier_write_slot = 1U;
  controller.frontier_size[0] = 0U;
  controller.frontier_size[1] = 0U;
  controller.done = 1U;
  controller.stop_reason =
      static_cast<std::uint32_t>(DeviceStopReason::invalid_controller_state);
  controller.error_bits = device_error::invalid_controller_state;
}

}  // namespace detail

// Dense push never swaps distance slots. A lane converges only after an
// actually executed complete admitted-edge scan reports no strict decrease.
[[nodiscard]] BFNEW_DENSE_HOST_DEVICE inline DenseAdvanceResult
advance_dense_controller(DeviceController& controller) noexcept {
  if (controller.done != 0U) {
    return DenseAdvanceResult::no_op;
  }

  const bool valid =
      controller.engine_kind ==
          static_cast<std::uint32_t>(EngineKind::dense_chaotic_push) &&
      controller.valid_lane_mask != 0U && controller.active_lane_mask != 0U &&
      controller.execute_lane_mask == controller.active_lane_mask &&
      ((controller.active_lane_mask | controller.changed_lane_mask |
        controller.converged_lane_mask | controller.execute_lane_mask) &
       ~controller.valid_lane_mask) == 0U &&
      (controller.active_lane_mask | controller.converged_lane_mask) ==
          controller.valid_lane_mask &&
      (controller.changed_lane_mask & ~controller.execute_lane_mask) == 0U &&
      (controller.converged_lane_mask & controller.active_lane_mask) == 0U &&
      controller.maximum_rounds != 0U &&
      controller.rounds_completed < controller.maximum_rounds &&
      controller.distance_read_slot == 0U &&
      controller.distance_write_slot == 0U &&
      controller.enable_per_lane_convergence <= 1U &&
      controller.next_frontier_lane_mask == 0U &&
      controller.stop_reason == static_cast<std::uint32_t>(DeviceStopReason::none) &&
      controller.error_bits == device_error::none;
  if (!valid) {
    detail::fail_dense_controller(controller);
    return DenseAdvanceResult::invalid_controller_state;
  }

  ++controller.rounds_completed;
  const LaneMask changed =
      controller.changed_lane_mask & controller.execute_lane_mask;
  if (controller.enable_per_lane_convergence != 0U) {
    const LaneMask newly_converged = controller.execute_lane_mask & ~changed;
    controller.converged_lane_mask |= newly_converged;
    controller.active_lane_mask &= changed;
  } else if (changed == 0U) {
    controller.converged_lane_mask = controller.valid_lane_mask;
    controller.active_lane_mask = 0U;
  }
  controller.changed_lane_mask = 0U;
  controller.next_frontier_lane_mask = 0U;

  if (controller.active_lane_mask == 0U) {
    controller.execute_lane_mask = 0U;
    controller.done = 1U;
    controller.stop_reason =
        static_cast<std::uint32_t>(DeviceStopReason::converged);
    return DenseAdvanceResult::converged;
  }
  if (controller.rounds_completed == controller.maximum_rounds) {
    controller.execute_lane_mask = 0U;
    controller.done = 1U;
    controller.stop_reason =
        static_cast<std::uint32_t>(DeviceStopReason::maximum_rounds);
    return DenseAdvanceResult::maximum_rounds;
  }

  controller.execute_lane_mask = controller.active_lane_mask;
  return DenseAdvanceResult::continue_execution;
}

[[nodiscard]] BFNEW_DENSE_HOST_DEVICE inline DeviceRunStatus
make_dense_run_status(
    const DeviceController& controller,
    const LaneMask reached_target_mask,
    const LaneMask bounding_box_miss_mask) noexcept {
  const auto reason = static_cast<DeviceStopReason>(controller.stop_reason);
  const bool completed = reason == DeviceStopReason::converged;
  const LaneMask safe_reached =
      completed ? reached_target_mask & controller.valid_lane_mask : 0U;
  const LaneMask safe_miss =
      completed && controller.error_bits == device_error::none
          ? bounding_box_miss_mask & controller.valid_lane_mask & ~safe_reached
          : 0U;
  return DeviceRunStatus{
      0U,
      completed ? 1U : 0U,
      controller.rounds_completed,
      safe_reached,
      safe_miss,
      controller.valid_lane_mask,
      controller.active_lane_mask,
      controller.converged_lane_mask,
      controller.stop_reason,
      controller.error_bits,
  };
}

#undef BFNEW_DENSE_HOST_DEVICE

enum class DenseHostSchedule : std::uint8_t {
  csr_forward,
  csr_reverse,
  alternating,
};

struct HostDenseRunResult {
  std::vector<float> distances;
  std::vector<std::uint32_t> distance_bits;
  DeviceController controller{};
  GpuSsspResult result{};
  std::uint64_t queued_round_pairs{};
  std::uint64_t completed_host_chunks{};
};

[[nodiscard]] HostDenseRunResult run_host_dense_chaotic_push(
    const DeviceGraphLayout32& graph,
    const RouteQuery& query,
    std::span<const LaneMask> tile_lane_masks,
    std::span<const LaneMask> csr_run_lane_masks,
    const GpuRunOptions& options,
    DenseHostSchedule schedule = DenseHostSchedule::csr_forward);

static_assert(std::is_standard_layout_v<DenseScratchLayout>);
static_assert(std::is_trivially_copyable_v<DenseScratchLayout>);
static_assert(std::is_standard_layout_v<DenseDistanceView>);
static_assert(std::is_trivially_copyable_v<DenseDistanceView>);

}  // namespace bfnew
