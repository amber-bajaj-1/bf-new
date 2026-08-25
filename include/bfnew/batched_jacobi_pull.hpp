#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/jacobi_pull.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace bfnew {

// Phase 14 keeps query lanes adjacent for every vertex. The element for
// (vertex, lane) is therefore vertex * lane_width + lane in both slots.
[[nodiscard]] constexpr std::uint64_t batched_jacobi_distance_index(
    const std::uint32_t vertex,
    const std::uint32_t lane,
    const std::uint32_t lane_width) noexcept {
  return static_cast<std::uint64_t>(vertex) * lane_width + lane;
}

[[nodiscard]] constexpr bool supported_batched_jacobi_width(
    const std::uint32_t lane_width) noexcept {
  return lane_width == 1U || lane_width == 8U || lane_width == 16U ||
         lane_width == 32U;
}

struct BatchedJacobiWorkStatistics {
  // A visited run has at least one executing admitted lane. A skipped run was
  // inspected but became empty after intersecting its prepared admission mask
  // with execute_lane_mask. Compact descriptors can still be skipped after
  // their lanes converge.
  std::uint64_t csc_runs_considered{};
  std::uint64_t csc_runs_visited{};
  std::uint64_t csc_runs_skipped{};
  std::uint64_t active_lanes_over_visited_runs{};

  // One edge load can serve every active query lane in a run. The second count
  // is the logical per-lane relaxation work enabled by those shared loads.
  std::uint64_t csc_edge_loads{};
  std::uint64_t lane_edge_relaxations{};
  std::uint64_t destination_lane_writes{};
  std::uint64_t successful_decreases{};

  // Capacity denominators make utilization and padding auditable without
  // embedding floating-point ratios in the evidence record.
  std::uint64_t active_lane_rounds{};
  std::uint64_t valid_lane_round_capacity{};
  std::uint64_t wave_lane_round_capacity{};
  std::uint64_t inactive_valid_lane_rounds{};
  std::uint64_t padded_lane_round_capacity{};
  std::uint64_t padded_lane_semantic_work{};

  // Tail rounds are rounds after a lane's first complete no-change scan and
  // before batch termination. They are avoided when per-lane convergence is
  // enabled and executed (as harmless full copies) when it is disabled.
  std::uint64_t tail_lane_rounds{};
  std::uint64_t tail_lane_rounds_avoided{};
  std::uint64_t tail_lane_rounds_executed{};
  std::uint64_t lane_edge_relaxations_avoided_by_early_convergence{};

  // Union inflation is union_tile_lane_positions /
  // selected_tile_lane_positions. The latter is the sum of popcounts of the
  // prepared tile masks.
  std::uint64_t union_tile_lane_positions{};
  std::uint64_t selected_tile_lane_positions{};

  // Modeled device reset traffic covers the two admitted vertex/lane columns;
  // source seeding is separate because it overwrites initialized entries. The
  // portable reference allocates fresh vectors and may initialize additional
  // nonsemantic cells as a host-container convenience; those host writes are
  // deliberately not presented as device selected-reset traffic.
  std::uint64_t distance_reset_bytes{};
  std::uint64_t source_seed_write_bytes{};

  constexpr bool operator==(
      const BatchedJacobiWorkStatistics&) const noexcept = default;
};

struct HostBatchedJacobiRunResult {
  // Portable semantic/control evidence only. This is not GPU timing or
  // throughput evidence. Both slots and final_distances use vertex-major,
  // contiguous-lane indexing.
  std::vector<float> final_distances;
  std::array<std::vector<float>, 2U> distance_slots;
  DeviceController controller{};
  GpuSsspResult result{};
  BatchedJacobiWorkStatistics batch_work{};

  // Vectors have lane_width entries. Zero is the sentinel for a padded lane
  // or for a valid lane that did not reach a no-change round before a maximum-
  // round stop. Real completed rounds are one-based.
  std::vector<std::uint64_t> rounds_executed_by_lane;
  std::vector<std::uint64_t> convergence_round_by_lane;
  std::vector<std::uint64_t> tail_rounds_by_lane;

  // Set only after bitwise comparison of both slots over every vertex selected
  // for that lane. Cells outside a lane's tiles are intentionally nonsemantic.
  LaneMask converged_slots_bitwise_identical_mask{};
  std::uint64_t queued_round_pairs{};
  std::uint64_t completed_host_chunks{};
};

// Executes the Phase 14 semantics on the CPU for bounded differential tests.
// The immutable graph is the checked Phase 8 device image; batch and
// description are the Phase 13 plan/preparation records. Both retained masks
// and compact nonzero descriptors are accepted. The standard plan family
// remains 32/16/8; callers request width one explicitly for baseline evidence.
[[nodiscard]] HostBatchedJacobiRunResult run_host_batched_jacobi_pull(
    const DeviceGraphLayout32& graph,
    std::span<const RouteQuery> queries,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description,
    const GpuRunOptions& options);

}  // namespace bfnew
