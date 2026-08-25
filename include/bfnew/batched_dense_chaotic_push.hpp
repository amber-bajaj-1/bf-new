#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/dense_chaotic_push.hpp"
#include "bfnew/device_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace bfnew {

// Phase 15 keeps the atomic distance words for one vertex adjacent by query
// lane: index = vertex * lane_width + lane.
[[nodiscard]] constexpr std::uint64_t batched_dense_distance_index(
    const std::uint32_t vertex,
    const std::uint32_t lane,
    const std::uint32_t lane_width) noexcept {
  return static_cast<std::uint64_t>(vertex) * lane_width + lane;
}

[[nodiscard]] constexpr bool supported_batched_dense_width(
    const std::uint32_t lane_width) noexcept {
  return lane_width == 1U || lane_width == 8U || lane_width == 16U ||
         lane_width == 32U;
}

struct BatchedDenseWorkStatistics {
  // A considered run belongs to a selected source row that has at least one
  // executing lane. A visited run has at least one executing admitted lane;
  // otherwise it is skipped. Compact descriptors omit endpoint-empty runs but
  // can still become skipped after all of their lanes freeze.
  std::uint64_t csr_runs_considered{};
  std::uint64_t csr_runs_visited{};
  std::uint64_t csr_runs_skipped{};
  std::uint64_t active_lanes_over_visited_runs{};

  // One algorithmic CSR edge-record request can serve every admitted lane.
  // This is not a measured cache transaction or proof of one physical load.
  // Each logical lane-edge performs one atomic-compatible source load and one
  // independent destination atomic-min attempt.
  std::uint64_t csr_edge_loads{};
  std::uint64_t lane_edge_relaxations{};
  std::uint64_t atomic_source_loads{};
  std::uint64_t atomic_min_attempts{};
  // The portable model always knows this value. The HIP timing path at
  // InstrumentationLevel::none leaves it zero/unavailable; Light or Debug
  // supplies the measured device count.
  std::uint64_t successful_atomic_updates{};
  std::uint64_t active_source_lane_evaluations{};
  std::uint64_t changed_round_publications{};
  std::uint64_t full_edge_rounds{};

  std::uint64_t active_lane_rounds{};
  std::uint64_t valid_lane_round_capacity{};
  std::uint64_t lane_width_round_capacity{};
  std::uint64_t wave32_lane_round_capacity{};
  std::uint64_t unused_wave_lane_round_capacity{};
  std::uint64_t inactive_valid_lane_rounds{};
  std::uint64_t padded_lane_round_capacity{};
  std::uint64_t padded_lane_semantic_work{};
  std::uint64_t frontier_semantic_work{};

  // A wave32 owns one selected source row and serially services its requested
  // edge steps. These exact integer terms expose lane occupancy for those
  // edge steps without presenting it as measured GPU occupancy or cache
  // behavior.
  std::uint64_t edge_wave_lane_capacity{};
  std::uint64_t unused_edge_wave_lane_capacity{};

  std::uint64_t tail_lane_rounds{};
  std::uint64_t tail_lane_rounds_avoided{};
  std::uint64_t tail_lane_rounds_executed{};
  std::uint64_t lane_edge_relaxations_avoided_by_early_convergence{};

  std::uint64_t union_tile_lane_positions{};
  std::uint64_t selected_tile_lane_positions{};

  // Dense push has one selected atomic word per admitted vertex/lane. Source
  // seeds are the +0-valued subset of those initialization writes.
  // The portable model uses a fresh full host vector for convenience; its
  // nonsemantic host initialization is not reported as device reset traffic.
  std::uint64_t distance_reset_bytes{};
  std::uint64_t source_seed_write_bytes{};

  constexpr bool operator==(const BatchedDenseWorkStatistics&) const noexcept =
      default;
};

struct HostBatchedDenseRunResult {
  // Both vectors are vertex-major with contiguous lanes. distance_bits is the
  // authoritative in-place atomic representation; distances is its exact
  // float projection for differential tests.
  std::vector<float> distances;
  std::vector<std::uint32_t> distance_bits;
  DeviceController controller{};
  GpuSsspResult result{};
  BatchedDenseWorkStatistics batch_work{};

  // Zero denotes padding or a valid lane that did not prove a no-change scan
  // before a maximum-round/error exit. Real scan numbers are one-based.
  std::vector<std::uint64_t> rounds_executed_by_lane;
  std::vector<std::uint64_t> convergence_round_by_lane;
  std::vector<std::uint64_t> tail_rounds_by_lane;

  std::uint64_t queued_round_pairs{};
  std::uint64_t completed_host_chunks{};
};

// Portable Phase 15 semantic/control model for bounded differential tests.
// It accepts either exact Phase 13 CSR run representation and performs only
// in-place atomic-bit-compatible dense relaxation; it has no frontier state.
[[nodiscard]] HostBatchedDenseRunResult
run_host_batched_dense_chaotic_push(
    const DeviceGraphLayout32& graph,
    std::span<const RouteQuery> queries,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description,
    const GpuRunOptions& options,
    DenseHostSchedule schedule = DenseHostSchedule::csr_forward);

}  // namespace bfnew
