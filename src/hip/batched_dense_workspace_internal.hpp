#pragma once

#include "bfnew/batch_layout.hpp"

#include <cstdint>
#include <type_traits>

namespace bfnew::hip::detail {

struct DeviceBatchDenseView {
  std::uint32_t lane_width{};
  LaneMask valid_lane_mask{};
  std::uint32_t selected_range_count{};
  std::uint32_t union_vertex_count{};
  const std::uint32_t* source_offsets{};
  const std::uint32_t* target_offsets{};
  const BatchVertexRange* selected_ranges{};
  const std::uint32_t* selected_range_vertex_offsets{};
  std::uint64_t* lane_convergence_rounds{};
};

struct DeviceBatchDenseStatistics {
  std::uint64_t csr_runs_considered{};
  std::uint64_t csr_runs_skipped{};
  std::uint64_t csr_nonzero_runs_visited{};
  std::uint64_t csr_edge_records_loaded{};
  std::uint64_t admitted_lane_edge_pairs{};
  std::uint64_t atomic_source_loads{};
  std::uint64_t atomic_min_attempts{};
  std::uint64_t successful_atomic_updates{};
  std::uint64_t active_lanes_across_nonzero_runs{};
  std::uint64_t active_source_lane_evaluations{};
  std::uint64_t changed_round_publications{};
  std::uint64_t full_edge_rounds{};
};

static_assert(std::is_standard_layout_v<DeviceBatchDenseView>);
static_assert(std::is_trivially_copyable_v<DeviceBatchDenseView>);
static_assert(std::is_standard_layout_v<DeviceBatchDenseStatistics>);
static_assert(std::is_trivially_copyable_v<DeviceBatchDenseStatistics>);

}  // namespace bfnew::hip::detail
