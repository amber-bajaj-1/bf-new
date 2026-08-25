#pragma once

#include "bfnew/dense_chaotic_push.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/query.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace bfnew {

// Phase 11 keeps all hot state in one retained scratch allocation: one atomic
// distance word per vertex, two bounded vertex queues, and one 64-bit enqueue
// generation per vertex. A zero requested queue capacity means vertex_count.
struct FrontierScratchLayout {
  std::uint64_t vertex_count{};
  std::uint64_t queue_capacity{};
  std::uint64_t distance_bits_offset{};
  std::uint64_t distance_bits_bytes{};
  std::uint64_t frontier_offsets[2]{};
  std::uint64_t frontier_bytes_each{};
  std::uint64_t enqueue_generation_offset{};
  std::uint64_t enqueue_generation_bytes{};
  std::uint64_t total_bytes{};

  constexpr bool operator==(const FrontierScratchLayout&) const noexcept =
      default;
};

[[nodiscard]] FrontierScratchLayout make_frontier_scratch_layout(
    std::size_t vertex_count,
    std::size_t queue_capacity = 0U);

struct FrontierScratchView {
  std::uint32_t* distance_bits{};
  std::uint32_t* frontiers[2]{};
  std::uint64_t* enqueue_generation{};
  std::uint32_t vertex_count{};
  std::uint32_t queue_capacity{};
};

#if defined(__HIPCC__)
#define BFNEW_FRONTIER_HOST_DEVICE __host__ __device__
#else
#define BFNEW_FRONTIER_HOST_DEVICE
#endif

[[nodiscard]] BFNEW_FRONTIER_HOST_DEVICE inline FrontierScratchView
bind_frontier_scratch(
    void* const scratch,
    const std::uint64_t scratch_bytes,
    const std::uint32_t vertex_count,
    const std::uint32_t requested_queue_capacity) noexcept {
  const std::uint32_t queue_capacity =
      requested_queue_capacity == 0U ? vertex_count : requested_queue_capacity;
  if (scratch == nullptr || vertex_count == 0U || queue_capacity == 0U ||
      queue_capacity > vertex_count ||
      reinterpret_cast<std::uintptr_t>(scratch) % alignof(std::uint64_t) !=
          0U) {
    return {};
  }
  const std::uint64_t distance_bytes =
      static_cast<std::uint64_t>(vertex_count) * sizeof(std::uint32_t);
  const std::uint64_t frontier_bytes =
      static_cast<std::uint64_t>(queue_capacity) * sizeof(std::uint32_t);
  const std::uint64_t frontier_zero_offset = distance_bytes;
  const std::uint64_t frontier_one_offset =
      frontier_zero_offset + frontier_bytes;
  const std::uint64_t generation_unaligned =
      frontier_one_offset + frontier_bytes;
  const std::uint64_t generation_offset =
      (generation_unaligned + alignof(std::uint64_t) - 1U) &
      ~(static_cast<std::uint64_t>(alignof(std::uint64_t)) - 1U);
  const std::uint64_t generation_bytes =
      static_cast<std::uint64_t>(vertex_count) * sizeof(std::uint64_t);
  const std::uint64_t required_bytes = generation_offset + generation_bytes;
  if (scratch_bytes < required_bytes) {
    return {};
  }
  auto* const bytes = static_cast<std::uint8_t*>(scratch);
  return FrontierScratchView{
      reinterpret_cast<std::uint32_t*>(bytes),
      {
          reinterpret_cast<std::uint32_t*>(bytes + frontier_zero_offset),
          reinterpret_cast<std::uint32_t*>(bytes + frontier_one_offset),
      },
      reinterpret_cast<std::uint64_t*>(bytes + generation_offset),
      vertex_count,
      queue_capacity,
  };
}

[[nodiscard]] BFNEW_FRONTIER_HOST_DEVICE inline bool
valid_frontier_scratch_view(const FrontierScratchView& view) noexcept {
  return view.distance_bits != nullptr && view.frontiers[0] != nullptr &&
         view.frontiers[1] != nullptr && view.enqueue_generation != nullptr &&
         view.vertex_count != 0U && view.queue_capacity != 0U &&
         view.queue_capacity <= view.vertex_count &&
         view.frontiers[0] != view.frontiers[1] &&
         reinterpret_cast<std::uintptr_t>(view.distance_bits) %
                 alignof(std::uint32_t) ==
             0U &&
         reinterpret_cast<std::uintptr_t>(view.frontiers[0]) %
                 alignof(std::uint32_t) ==
             0U &&
         reinterpret_cast<std::uintptr_t>(view.frontiers[1]) %
                 alignof(std::uint32_t) ==
             0U &&
         reinterpret_cast<std::uintptr_t>(view.enqueue_generation) %
                 alignof(std::uint64_t) ==
             0U;
}

enum class FrontierAdvanceResult : std::uint32_t {
  no_op = 0U,
  continue_execution = 1U,
  converged = 2U,
  maximum_rounds = 3U,
  queue_overflow = 4U,
  invalid_controller_state = 5U,
  device_failure = 6U,
};

namespace detail {

BFNEW_FRONTIER_HOST_DEVICE inline void canonicalize_frontier_error(
    DeviceController& controller,
    const DeviceStopReason reason,
    const std::uint32_t error_bits) noexcept {
  if (controller.valid_lane_mask == 0U) {
    controller.valid_lane_mask = LaneMask{1U};
  }
  controller.active_lane_mask = 0U;
  controller.changed_lane_mask = 0U;
  controller.converged_lane_mask = 0U;
  controller.execute_lane_mask = 0U;
  controller.next_frontier_lane_mask = 0U;
  controller.engine_kind = static_cast<std::uint32_t>(EngineKind::frontier_push);
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
  controller.frontier_read_slot =
      controller.frontier_read_slot > 1U ? 0U : controller.frontier_read_slot;
  controller.frontier_write_slot = 1U - controller.frontier_read_slot;
  controller.done = 1U;
  controller.stop_reason = static_cast<std::uint32_t>(reason);
  controller.error_bits = error_bits;
}

}  // namespace detail

// The engine round has already materialized frontier_size[write] and the
// next-frontier lane mask. This sole transition owner swaps queues after every
// executed round, clears the recycled queue size, and gives empty-frontier
// convergence precedence over the maximum-round limit.
[[nodiscard]] BFNEW_FRONTIER_HOST_DEVICE inline FrontierAdvanceResult
advance_frontier_controller(DeviceController& controller) noexcept {
  if (controller.done != 0U) {
    return FrontierAdvanceResult::no_op;
  }

  const bool structural_state_valid =
      controller.engine_kind ==
          static_cast<std::uint32_t>(EngineKind::frontier_push) &&
      controller.valid_lane_mask != 0U && controller.active_lane_mask != 0U &&
      controller.execute_lane_mask == controller.active_lane_mask &&
      ((controller.active_lane_mask | controller.changed_lane_mask |
        controller.converged_lane_mask | controller.execute_lane_mask |
        controller.next_frontier_lane_mask) &
       ~controller.valid_lane_mask) == 0U &&
      (controller.active_lane_mask | controller.converged_lane_mask) ==
          controller.valid_lane_mask &&
      (controller.converged_lane_mask & controller.active_lane_mask) == 0U &&
      controller.changed_lane_mask == 0U &&
      controller.maximum_rounds != 0U &&
      controller.rounds_completed < controller.maximum_rounds &&
      controller.distance_read_slot == 0U &&
      controller.distance_write_slot == 0U &&
      controller.frontier_read_slot <= 1U &&
      controller.frontier_write_slot <= 1U &&
      controller.frontier_read_slot != controller.frontier_write_slot &&
      controller.frontier_size[controller.frontier_read_slot] != 0U &&
      controller.enable_per_lane_convergence <= 1U &&
      controller.stop_reason ==
          static_cast<std::uint32_t>(DeviceStopReason::none);
  const bool recognized_error =
      controller.error_bits == device_error::none ||
      controller.error_bits == device_error::queue_overflow ||
      controller.error_bits == device_error::device_failure;
  if (!structural_state_valid || !recognized_error) {
    detail::canonicalize_frontier_error(
        controller,
        DeviceStopReason::invalid_controller_state,
        device_error::invalid_controller_state);
    return FrontierAdvanceResult::invalid_controller_state;
  }

  const std::uint32_t old_read = controller.frontier_read_slot;
  const std::uint32_t old_write = controller.frontier_write_slot;
  const std::uint64_t next_size = controller.frontier_size[old_write];
  const bool next_present =
      (controller.next_frontier_lane_mask & controller.valid_lane_mask) != 0U;
  if (controller.error_bits == device_error::none &&
      (next_present != (next_size != 0U))) {
    detail::canonicalize_frontier_error(
        controller,
        DeviceStopReason::invalid_controller_state,
        device_error::invalid_controller_state);
    return FrontierAdvanceResult::invalid_controller_state;
  }

  ++controller.rounds_completed;
  controller.frontier_read_slot = old_write;
  controller.frontier_write_slot = old_read;
  controller.frontier_size[controller.frontier_write_slot] = 0U;
  controller.changed_lane_mask = 0U;
  controller.next_frontier_lane_mask = 0U;

  if (controller.error_bits == device_error::queue_overflow) {
    detail::canonicalize_frontier_error(
        controller,
        DeviceStopReason::queue_overflow,
        device_error::queue_overflow);
    return FrontierAdvanceResult::queue_overflow;
  }
  if (controller.error_bits == device_error::device_failure) {
    detail::canonicalize_frontier_error(
        controller,
        DeviceStopReason::device_failure,
        device_error::device_failure);
    return FrontierAdvanceResult::device_failure;
  }

  if (next_size == 0U) {
    controller.active_lane_mask = 0U;
    controller.converged_lane_mask = controller.valid_lane_mask;
    controller.execute_lane_mask = 0U;
    controller.done = 1U;
    controller.stop_reason =
        static_cast<std::uint32_t>(DeviceStopReason::converged);
    return FrontierAdvanceResult::converged;
  }
  if (controller.rounds_completed == controller.maximum_rounds) {
    controller.execute_lane_mask = 0U;
    controller.done = 1U;
    controller.stop_reason =
        static_cast<std::uint32_t>(DeviceStopReason::maximum_rounds);
    return FrontierAdvanceResult::maximum_rounds;
  }

  controller.active_lane_mask = controller.valid_lane_mask;
  controller.execute_lane_mask = controller.valid_lane_mask;
  return FrontierAdvanceResult::continue_execution;
}

[[nodiscard]] BFNEW_FRONTIER_HOST_DEVICE inline DeviceRunStatus
make_frontier_run_status(
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

#undef BFNEW_FRONTIER_HOST_DEVICE

struct HostFrontierRunResult {
  std::vector<float> distances;
  std::vector<std::uint32_t> distance_bits;
  std::vector<std::uint64_t> current_frontier_sizes;
  DeviceController controller{};
  GpuSsspResult result{};
  std::uint64_t queued_round_pairs{};
  std::uint64_t completed_host_chunks{};
};

[[nodiscard]] HostFrontierRunResult run_host_frontier_push(
    const DeviceGraphLayout32& graph,
    const RouteQuery& query,
    std::span<const LaneMask> tile_lane_masks,
    std::span<const LaneMask> csr_run_lane_masks,
    const GpuRunOptions& options,
    std::size_t queue_capacity = 0U);

static_assert(std::is_standard_layout_v<FrontierScratchLayout>);
static_assert(std::is_trivially_copyable_v<FrontierScratchLayout>);
static_assert(std::is_standard_layout_v<FrontierScratchView>);
static_assert(std::is_trivially_copyable_v<FrontierScratchView>);

}  // namespace bfnew
