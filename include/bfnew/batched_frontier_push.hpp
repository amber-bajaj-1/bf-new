#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/frontier_push.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace bfnew {

// Phase 16 retains one independent atomic distance word for each query lane.
// Query lanes are contiguous within a vertex.
[[nodiscard]] constexpr std::uint64_t batched_frontier_distance_index(
    const std::uint32_t vertex,
    const std::uint32_t lane,
    const std::uint32_t lane_width) noexcept {
  return static_cast<std::uint64_t>(vertex) * lane_width + lane;
}

[[nodiscard]] constexpr bool supported_batched_frontier_width(
    const std::uint32_t lane_width) noexcept {
  return lane_width == 1U || lane_width == 8U || lane_width == 16U ||
         lane_width == 32U;
}

// Checked retained-memory planner. Storage order is distance words, two
// graph-sized wave32 activity-mask slots, then two merged vertex queues. Each
// queue entry is one vertex shared by all active lanes in that vertex's mask.
// A zero queue-capacity request means V.
struct BatchedFrontierScratchLayout {
  std::uint64_t vertex_count{};
  std::uint32_t lane_width{};
  std::uint32_t reserved{};
  std::uint64_t queue_capacity{};
  std::uint64_t distance_bits_offset{};
  std::uint64_t distance_bits_bytes{};
  std::uint64_t activity_mask_offsets[2]{};
  std::uint64_t activity_mask_bytes_each{};
  std::uint64_t frontier_offsets[2]{};
  std::uint64_t frontier_bytes_each{};
  std::uint64_t total_bytes{};

  constexpr bool operator==(
      const BatchedFrontierScratchLayout&) const noexcept = default;
};

[[nodiscard]] BatchedFrontierScratchLayout
make_batched_frontier_scratch_layout(
    std::size_t vertex_count,
    std::uint32_t lane_width,
    std::size_t queue_capacity = 0U);

enum class BatchedFrontierAdvanceResult : std::uint32_t {
  no_op = 0U,
  continue_execution = 1U,
  converged = 2U,
  maximum_rounds = 3U,
  queue_overflow = 4U,
  invalid_controller_state = 5U,
  device_failure = 6U,
};

#if defined(__HIPCC__)
#define BFNEW_BATCHED_FRONTIER_HOST_DEVICE __host__ __device__
#else
#define BFNEW_BATCHED_FRONTIER_HOST_DEVICE
#endif

// Consumes exactly one completed batched frontier round. The next-frontier
// lane union is the only set eligible to execute again in either convergence
// mode. Enabled mode publishes lanes absent from that union as converged;
// disabled mode defers that publication until the complete queue is empty.
// Empty-frontier convergence takes precedence over the maximum-round bound.
[[nodiscard]] BFNEW_BATCHED_FRONTIER_HOST_DEVICE inline
BatchedFrontierAdvanceResult advance_batched_frontier_controller(
    DeviceController& controller) noexcept {
  if (controller.done != 0U) {
    return BatchedFrontierAdvanceResult::no_op;
  }

  const bool structural_state_valid =
      controller.engine_kind ==
          static_cast<std::uint32_t>(EngineKind::frontier_push) &&
      controller.valid_lane_mask != 0U &&
      controller.active_lane_mask != 0U &&
      controller.execute_lane_mask == controller.active_lane_mask &&
      ((controller.active_lane_mask | controller.changed_lane_mask |
        controller.converged_lane_mask | controller.execute_lane_mask |
        controller.next_frontier_lane_mask) &
       ~controller.valid_lane_mask) == 0U &&
      (controller.active_lane_mask & controller.converged_lane_mask) == 0U &&
      (controller.next_frontier_lane_mask &
       ~controller.execute_lane_mask) == 0U &&
      (controller.enable_per_lane_convergence != 0U
           ? (controller.active_lane_mask |
              controller.converged_lane_mask) == controller.valid_lane_mask
           : controller.converged_lane_mask == 0U) &&
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
    return BatchedFrontierAdvanceResult::invalid_controller_state;
  }

  const std::uint32_t old_read = controller.frontier_read_slot;
  const std::uint32_t old_write = controller.frontier_write_slot;
  const std::uint64_t next_size = controller.frontier_size[old_write];
  const LaneMask next_lanes =
      controller.next_frontier_lane_mask & controller.valid_lane_mask;
  if (controller.error_bits == device_error::none &&
      ((next_lanes != 0U) != (next_size != 0U))) {
    detail::canonicalize_frontier_error(
        controller,
        DeviceStopReason::invalid_controller_state,
        device_error::invalid_controller_state);
    return BatchedFrontierAdvanceResult::invalid_controller_state;
  }

  const LaneMask executed_lanes = controller.execute_lane_mask;
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
    return BatchedFrontierAdvanceResult::queue_overflow;
  }
  if (controller.error_bits == device_error::device_failure) {
    detail::canonicalize_frontier_error(
        controller,
        DeviceStopReason::device_failure,
        device_error::device_failure);
    return BatchedFrontierAdvanceResult::device_failure;
  }

  if (next_size == 0U) {
    controller.active_lane_mask = 0U;
    controller.converged_lane_mask = controller.valid_lane_mask;
    controller.execute_lane_mask = 0U;
    controller.done = 1U;
    controller.stop_reason =
        static_cast<std::uint32_t>(DeviceStopReason::converged);
    return BatchedFrontierAdvanceResult::converged;
  }

  if (controller.enable_per_lane_convergence != 0U) {
    controller.converged_lane_mask |= executed_lanes & ~next_lanes;
  }
  controller.active_lane_mask = next_lanes;

  if (controller.rounds_completed == controller.maximum_rounds) {
    controller.execute_lane_mask = 0U;
    controller.done = 1U;
    controller.stop_reason =
        static_cast<std::uint32_t>(DeviceStopReason::maximum_rounds);
    return BatchedFrontierAdvanceResult::maximum_rounds;
  }

  controller.execute_lane_mask = next_lanes;
  return BatchedFrontierAdvanceResult::continue_execution;
}

#undef BFNEW_BATCHED_FRONTIER_HOST_DEVICE

struct BatchedFrontierWorkStatistics {
  // One entry owner follows Phase 11's one-thread-per-worklist-vertex policy.
  // The difference between vertex/lane pairs and entries is the queue work
  // saved by representing a vertex once with a wave32 lane mask.
  std::uint64_t frontier_vertex_entries{};
  std::uint64_t active_vertex_lane_pairs{};
  std::uint64_t shared_vertex_entries_saved{};
  std::uint64_t multi_lane_frontier_vertex_entries{};

  std::uint64_t csr_runs_considered{};
  std::uint64_t csr_runs_visited{};
  std::uint64_t csr_runs_skipped{};
  std::uint64_t active_lanes_over_visited_runs{};
  // One CSR edge record is requested once by an entry owner and can serve
  // several admitted lane relaxations. These are logical, not cache counters.
  std::uint64_t csr_edge_loads{};
  std::uint64_t lane_edge_relaxations{};
  std::uint64_t shared_edge_lane_work_saved{};
  std::uint64_t multi_lane_csr_edge_loads{};

  std::uint64_t distance_atomic_source_loads{};
  std::uint64_t distance_atomic_attempts{};
  std::uint64_t successful_distance_atomic_updates{};

  // The GPU implementation maps these portable semantic events to atomic
  // exchange/OR operations. A next-mask OR occurs once per edge with at least
  // one successful lane, never once per lane.
  std::uint64_t current_mask_atomic_exchanges{};
  std::uint64_t next_mask_atomic_ors{};
  std::uint64_t controller_mask_atomic_ors{};
  std::uint64_t unique_next_vertex_lane_activations{};
  std::uint64_t queue_claims{};
  // Exact identities:
  //   queue_entries_saved_by_lane_merging =
  //       unique_next_vertex_lane_activations - queue_claims
  //   same_lane_duplicate_suppressions =
  //       successful_distance_atomic_updates -
  //       unique_next_vertex_lane_activations
  //   duplicate_suppressions = both preceding categories summed.
  std::uint64_t queue_entries_saved_by_lane_merging{};
  std::uint64_t same_lane_duplicate_suppressions{};
  std::uint64_t duplicate_suppressions{};
  std::uint64_t maximum_queue_size{};
  std::uint64_t overflow_events{};

  std::uint64_t initial_source_lane_activations{};
  std::uint64_t initial_queue_entries{};
  std::uint64_t initial_queue_entries_saved_by_lane_merging{};

  std::uint64_t frontier_rounds{};
  std::uint64_t empty_frontier_rounds{};
  std::uint64_t small_frontier_rounds{};
  std::uint64_t active_lane_rounds{};
  std::uint64_t valid_lane_round_capacity{};
  std::uint64_t lane_width_round_capacity{};
  std::uint64_t wave32_lane_round_capacity{};
  std::uint64_t unused_wave_lane_round_capacity{};
  std::uint64_t inactive_valid_lane_rounds{};
  std::uint64_t padded_lane_round_capacity{};
  std::uint64_t padded_lane_semantic_work{};

  // A lane's tail starts after its first complete round with no next entry.
  // Frontier masks already prevent later semantic work in both convergence
  // modes, so the avoided-work field is deliberately and truthfully zero.
  std::uint64_t tail_lane_rounds{};
  std::uint64_t tail_lane_rounds_without_frontier_work{};
  std::uint64_t semantic_lane_edge_work_avoided_by_per_lane_convergence{};

  std::uint64_t union_tile_lane_positions{};
  std::uint64_t selected_tile_lane_positions{};
  std::uint64_t distance_reset_bytes{};
  std::uint64_t activity_mask_reset_bytes{};
  std::uint64_t source_seed_write_bytes{};
  std::uint64_t frontier_queue_storage_bytes{};

  constexpr bool operator==(
      const BatchedFrontierWorkStatistics&) const noexcept = default;
};

struct HostBatchedFrontierRunResult {
  std::vector<float> distances;
  std::vector<std::uint32_t> distance_bits;
  DeviceController controller{};
  GpuSsspResult result{};
  BatchedFrontierWorkStatistics batch_work{};

  // Exact bounded semantic traces and retained terminal state for queue/mask
  // invariant tests. Queue cells beyond frontier_size[slot] are stale and not
  // semantic; activity masks must be zero except for entries in a live slot.
  std::vector<std::uint64_t> current_frontier_sizes;
  std::vector<LaneMask> current_frontier_lane_unions;
  std::array<std::vector<std::uint32_t>, 2U> frontier_queues;
  std::array<std::vector<LaneMask>, 2U> frontier_lane_masks;
  std::uint64_t queue_capacity{};

  // A convergence round is the one-based first complete round that creates no
  // next entry for that lane. Padded or unfinished lanes retain zero.
  std::vector<std::uint64_t> frontier_rounds_by_lane;
  std::vector<std::uint64_t> convergence_round_by_lane;
  std::vector<std::uint64_t> tail_rounds_by_lane;

  std::uint64_t queued_round_pairs{};
  std::uint64_t completed_host_chunks{};
};

// Portable Phase 16 semantic/controller model. Both retained run masks and
// compact nonzero descriptors are accepted. It is a merged vertex frontier,
// never a dense full-edge scan or an adaptive scheduling path.
[[nodiscard]] HostBatchedFrontierRunResult run_host_batched_frontier_push(
    const DeviceGraphLayout32& graph,
    std::span<const RouteQuery> queries,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description,
    const GpuRunOptions& options,
    std::size_t queue_capacity = 0U);

static_assert(std::is_standard_layout_v<BatchedFrontierScratchLayout>);
static_assert(std::is_trivially_copyable_v<BatchedFrontierScratchLayout>);

}  // namespace bfnew
