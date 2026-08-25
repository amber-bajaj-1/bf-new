#pragma once

#include "bfnew/device_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/query.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace bfnew {

// Phase 9 uses exactly two graph-sized float columns.  The checked host
// planner keeps their offsets explicit so neither a host launcher nor a device
// kernel needs to infer a layout from an allocation size.
struct JacobiScratchLayout {
  std::uint64_t vertex_count{};
  std::uint64_t distance_slot_bytes{};
  std::uint64_t distance_slot_offsets[2]{};
  std::uint64_t total_bytes{};

  constexpr bool operator==(const JacobiScratchLayout&) const noexcept = default;
};

[[nodiscard]] JacobiScratchLayout make_jacobi_scratch_layout(
    std::size_t vertex_count);

// A kernel receives a const read column and a distinct mutable write column.
// This type makes the no-in-place-update contract visible to both compilers.
struct JacobiDistanceView {
  const float* read_distances{};
  float* write_distances{};
  std::uint32_t vertex_count{};
  std::uint32_t read_slot{};
  std::uint32_t write_slot{};
};

#if defined(__HIPCC__)
#define BFNEW_JACOBI_HOST_DEVICE __host__ __device__
#else
#define BFNEW_JACOBI_HOST_DEVICE
#endif

[[nodiscard]] BFNEW_JACOBI_HOST_DEVICE inline JacobiDistanceView
bind_jacobi_distance_slots(
    void* const scratch,
    const std::uint64_t scratch_bytes,
    const std::uint32_t vertex_count,
    const std::uint32_t read_slot,
    const std::uint32_t write_slot) noexcept {
  const std::uint64_t slot_bytes =
      static_cast<std::uint64_t>(vertex_count) * sizeof(float);
  const std::uint64_t required_bytes = slot_bytes * 2U;
  if (scratch == nullptr || vertex_count == 0U ||
      scratch_bytes < required_bytes || read_slot > 1U || write_slot > 1U ||
      read_slot == write_slot) {
    return {};
  }

  auto* const base = static_cast<float*>(scratch);
  return JacobiDistanceView{
      base + static_cast<std::uint64_t>(read_slot) * vertex_count,
      base + static_cast<std::uint64_t>(write_slot) * vertex_count,
      vertex_count,
      read_slot,
      write_slot,
  };
}

[[nodiscard]] BFNEW_JACOBI_HOST_DEVICE inline JacobiDistanceView
bind_jacobi_distance_view(
    void* const scratch,
    const std::uint64_t scratch_bytes,
    const std::uint32_t vertex_count,
    const DeviceController& controller) noexcept {
  return bind_jacobi_distance_slots(
      scratch,
      scratch_bytes,
      vertex_count,
      controller.distance_read_slot,
      controller.distance_write_slot);
}

[[nodiscard]] BFNEW_JACOBI_HOST_DEVICE inline bool
valid_jacobi_distance_view(const JacobiDistanceView& view) noexcept {
  return view.vertex_count != 0U && view.read_slot <= 1U &&
         view.write_slot <= 1U && view.read_slot != view.write_slot &&
         view.read_distances != nullptr && view.write_distances != nullptr &&
         view.read_distances != view.write_distances;
}

enum class JacobiAdvanceResult : std::uint32_t {
  no_op = 0U,
  continue_execution = 1U,
  converged = 2U,
  maximum_rounds = 3U,
  invalid_controller_state = 4U,
};

namespace detail {

BFNEW_JACOBI_HOST_DEVICE inline void fail_jacobi_controller(
    DeviceController& controller) noexcept {
  // Publish a validator-clean terminal error record even when the triggering
  // field itself was corrupt. This keeps controller failures distinct from
  // bounding-box misses and gives the host a safe final-slot identity.
  if (controller.valid_lane_mask == 0U) {
    controller.valid_lane_mask = LaneMask{1U};
  }
  controller.active_lane_mask = 0U;
  controller.changed_lane_mask = 0U;
  controller.converged_lane_mask = 0U;
  controller.next_frontier_lane_mask = 0U;
  controller.execute_lane_mask = 0U;
  controller.engine_kind =
      static_cast<std::uint32_t>(EngineKind::jacobi_pull);
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
  if (controller.distance_read_slot > 1U ||
      controller.distance_write_slot > 1U ||
      controller.distance_read_slot == controller.distance_write_slot) {
    controller.distance_read_slot = 0U;
    controller.distance_write_slot = 1U;
  }
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

// Consumes exactly one completed Jacobi round.  Both ordinary advance kernels
// and the designated persistent controller owner use this transition.  A done
// controller is a cheap no-op, which is required for rounds already queued in a
// host-polling chunk.
[[nodiscard]] BFNEW_JACOBI_HOST_DEVICE inline JacobiAdvanceResult
advance_jacobi_controller(DeviceController& controller) noexcept {
  if (controller.done != 0U) {
    return JacobiAdvanceResult::no_op;
  }

  const bool valid =
      controller.engine_kind == static_cast<std::uint32_t>(EngineKind::jacobi_pull) &&
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
      controller.distance_read_slot <= 1U &&
      controller.distance_write_slot <= 1U &&
      controller.distance_read_slot != controller.distance_write_slot &&
      controller.enable_per_lane_convergence <= 1U &&
      controller.next_frontier_lane_mask == 0U &&
      controller.stop_reason == static_cast<std::uint32_t>(DeviceStopReason::none) &&
      controller.error_bits == device_error::none;
  if (!valid) {
    detail::fail_jacobi_controller(controller);
    return JacobiAdvanceResult::invalid_controller_state;
  }

  ++controller.rounds_completed;
  const std::uint32_t former_read_slot = controller.distance_read_slot;
  controller.distance_read_slot = controller.distance_write_slot;
  controller.distance_write_slot = former_read_slot;

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

  // A no-change round wins over the round limit.  For a V-vertex chain, the
  // required final no-change scan may legitimately be round V.
  if (controller.active_lane_mask == 0U) {
    controller.execute_lane_mask = 0U;
    controller.done = 1U;
    controller.stop_reason =
        static_cast<std::uint32_t>(DeviceStopReason::converged);
    return JacobiAdvanceResult::converged;
  }
  if (controller.rounds_completed == controller.maximum_rounds) {
    controller.execute_lane_mask = 0U;
    controller.done = 1U;
    controller.stop_reason =
        static_cast<std::uint32_t>(DeviceStopReason::maximum_rounds);
    return JacobiAdvanceResult::maximum_rounds;
  }

  controller.execute_lane_mask = controller.active_lane_mask;
  return JacobiAdvanceResult::continue_execution;
}

[[nodiscard]] BFNEW_JACOBI_HOST_DEVICE inline DeviceRunStatus
make_jacobi_run_status(
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
      controller.distance_read_slot,
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

#undef BFNEW_JACOBI_HOST_DEVICE

struct HostJacobiRunResult {
  // This is a portable semantic/control model used by CPU-only tests.  It is
  // not device-execution evidence.
  std::vector<float> distances;
  std::array<std::vector<float>, 2> distance_slots;
  DeviceController controller{};
  GpuSsspResult result{};
  std::uint64_t queued_round_pairs{};
  std::uint64_t completed_host_chunks{};
};

[[nodiscard]] HostJacobiRunResult run_host_jacobi_pull(
    const DeviceGraphLayout32& graph,
    const RouteQuery& query,
    std::span<const LaneMask> tile_lane_masks,
    std::span<const LaneMask> csc_run_lane_masks,
    const GpuRunOptions& options);

static_assert(std::is_same_v<
              decltype(JacobiDistanceView::read_distances),
              const float*>);
static_assert(std::is_same_v<
              decltype(JacobiDistanceView::write_distances),
              float*>);
static_assert(std::is_standard_layout_v<JacobiDistanceView>);
static_assert(std::is_trivially_copyable_v<JacobiDistanceView>);
static_assert(std::is_standard_layout_v<JacobiScratchLayout>);
static_assert(std::is_trivially_copyable_v<JacobiScratchLayout>);

}  // namespace bfnew
