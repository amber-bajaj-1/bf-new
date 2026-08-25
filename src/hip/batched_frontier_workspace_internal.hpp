#pragma once

#include "bfnew/batch_layout.hpp"

#include <cstdint>
#include <type_traits>

namespace bfnew::hip::detail {

struct DeviceBatchFrontierView {
  std::uint32_t lane_width{};
  LaneMask valid_lane_mask{};
  std::uint32_t selected_range_count{};
  std::uint32_t union_vertex_count{};
  std::uint32_t queue_capacity{};
  const std::uint32_t* source_offsets{};
  const std::uint32_t* target_offsets{};
  const BatchVertexRange* selected_ranges{};
  const std::uint32_t* selected_range_vertex_offsets{};
  std::uint64_t* lane_convergence_rounds{};
};

// Algorithmic device counters. Cache traffic and occupancy are deliberately
// not represented here; the engine reports profiler counters as unavailable.
struct DeviceBatchFrontierStatistics {
  std::uint64_t worklist_vertices{};
  std::uint64_t active_vertex_lanes{};
  std::uint64_t multi_lane_worklist_vertices{};
  std::uint64_t csr_runs_considered{};
  std::uint64_t csr_runs_skipped{};
  std::uint64_t csr_nonzero_runs_visited{};
  std::uint64_t csr_edge_records_loaded{};
  std::uint64_t multi_lane_csr_edge_records{};
  std::uint64_t admitted_lane_edge_pairs{};
  std::uint64_t distance_atomic_source_loads{};
  std::uint64_t distance_atomic_attempts{};
  std::uint64_t distance_atomic_successes{};
  std::uint64_t active_lanes_across_nonzero_runs{};
  std::uint64_t current_mask_atomic_exchanges{};
  std::uint64_t next_mask_atomic_ors{};
  std::uint64_t controller_mask_atomic_ors{};
  std::uint64_t lane_enqueue_transitions{};
  std::uint64_t queue_claims{};
  std::uint64_t queue_entries_saved_by_lane_merging{};
  std::uint64_t same_lane_duplicate_suppressions{};
  std::uint64_t duplicate_suppressions{};
  std::uint64_t maximum_queue_size{};
  std::uint64_t overflow_events{};
  std::uint64_t frontier_rounds{};
  std::uint64_t empty_frontier_rounds{};
  std::uint64_t small_frontier_rounds{};
};

static_assert(std::is_standard_layout_v<DeviceBatchFrontierView>);
static_assert(std::is_trivially_copyable_v<DeviceBatchFrontierView>);
static_assert(std::is_standard_layout_v<DeviceBatchFrontierStatistics>);
static_assert(std::is_trivially_copyable_v<DeviceBatchFrontierStatistics>);

}  // namespace bfnew::hip::detail
