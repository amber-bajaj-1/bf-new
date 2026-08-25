#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/hip/batched_workspace.hpp"
#include "bfnew/hip/compact_path_results.hpp"
#include "bfnew/hip/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bfnew::hip {

// The default leaves the common edge fields as compiler-visible uniform
// loads.  The explicit wave broadcast remains a selectable experiment; it is
// not a default or an optimization claim until the two paths are measured on
// the target kernel and workload.
enum class BatchedJacobiLoadStrategy : std::uint32_t {
  compiler_uniform = 0U,
  explicit_wave_broadcast = 1U,
};

[[nodiscard]] constexpr bool valid_batched_jacobi_load_strategy(
    const BatchedJacobiLoadStrategy strategy) noexcept {
  return strategy == BatchedJacobiLoadStrategy::compiler_uniform ||
         strategy == BatchedJacobiLoadStrategy::explicit_wave_broadcast;
}

// The full-layout Phase 13 decision uses two vertex-major arrays.  Within one
// vertex, lanes are contiguous.  Width is restricted to the Phase 14 matrix.
[[nodiscard]] std::size_t batched_jacobi_distance_scratch_bytes(
    std::uint32_t vertex_count,
    std::uint32_t lane_width);

struct BatchedJacobiOptionalHardwareCounters {
  // Phase 14 does not manufacture profiler evidence.  These remain zero and
  // unavailable unless a later profiler adapter populates them from a
  // separate rocprofiler-SDK run.
  std::uint32_t available{};
  std::uint64_t l2_read_bytes{};
  std::uint64_t l2_write_bytes{};
  std::uint64_t texture_load_instructions{};
  std::uint64_t texture_store_instructions{};
  std::uint64_t waves{};
};

struct BatchedJacobiRunMetrics {
  float preparation_gpu_milliseconds{};
  // Results are materialized only after the complete batch stops, so this is
  // also the observable SSSP latency for every query returned by the batch.
  // Per-lane convergence rounds below expose earlier internal completion.
  float sssp_device_timeline_milliseconds{};
  float result_transfer_gpu_milliseconds{};
  double end_to_end_wall_milliseconds{};

  std::uint32_t ordinary_active_blocks_per_wgp{};
  std::uint32_t cooperative_grid_blocks{};
  std::uint32_t cooperative_active_blocks_per_wgp{};
  std::uint32_t kernel_registers_per_thread{};
  std::uint32_t lane_width{};
  std::uint32_t valid_lane_count{};
  BatchedJacobiLoadStrategy load_strategy{
      BatchedJacobiLoadStrategy::compiler_uniform};

  std::uint64_t engine_round_dispatches{};
  std::uint64_t controller_advance_dispatches{};
  std::uint64_t convergence_host_checks{};

  // Device counters are collected only for Light/Debug instrumentation.
  // A run visit is counted once per destination/run/round, not once per lane.
  std::uint64_t csc_runs_considered{};
  std::uint64_t csc_runs_visited{};
  std::uint64_t csc_runs_skipped{};
  std::uint64_t csc_edge_records_loaded{};
  std::uint64_t admitted_lane_edge_pairs{};
  std::uint64_t lane_edge_relaxations_avoided_by_early_convergence{};
  std::uint64_t active_lanes_across_nonzero_runs{};
  std::uint64_t active_vertex_lane_evaluations{};

  std::uint64_t executed_lane_rounds{};
  std::uint64_t lane_rounds_avoided_by_early_convergence{};
  std::uint64_t padded_lane_rounds{};
  std::uint64_t unused_wave_lane_rounds{};

  // Reset traffic includes both distance slots and only selected range/lane
  // cells. Invalid/padded and nonselected cells are neither cleared nor read.
  std::uint64_t distance_reset_bytes{};
  // Selected source cells are written as zero instead of infinity in the same
  // fused initialization stores. This is an auditable subset of reset bytes,
  // not additional physical traffic.
  std::uint64_t source_seed_write_bytes{};
  std::uint64_t padded_distance_reset_bytes{};
  std::uint64_t union_tile_lane_positions{};
  std::uint64_t selected_tile_lane_positions{};

  // Algorithmic byte requests, not measured L2 traffic. They deliberately
  // remain separate from the optional hardware counters above.
  std::uint64_t edge_record_read_bytes_requested{};
  std::uint64_t source_distance_read_bytes_requested{};

  double average_active_lanes_per_nonzero_run{};
  double lane_round_utilization{};
  double batch_queries_per_second{};
  BatchedJacobiOptionalHardwareCounters hardware_counters{};
};

struct BatchedJacobiRunOutput {
  GpuSsspResult result{};

  // Final labels are vertex-major: distances[vertex * lane_width + lane].
  // They are empty for run_status_only().
  std::vector<float> distances;
  bool distances_downloaded{};

  // Zero denotes a padded lane or a valid lane that did not establish a
  // no-change round before an error/round limit.  With convergence disabled,
  // the first proven no-change round is still recorded without freezing the
  // lane, which makes tail work directly observable.
  std::vector<std::uint64_t> lane_convergence_rounds;
  std::vector<std::uint64_t> lane_executed_rounds;
  std::vector<std::uint64_t> lane_tail_rounds;

  // A bit is set only when every vertex in that valid lane is bitwise equal
  // across the two downloaded columns after normal convergence.
  LaneMask converged_slots_bitwise_identical_mask{};
  BatchedJacobiRunMetrics metrics{};
};

// Batched Jacobi intentionally has a batch-specific surface instead of
// pretending that one BatchDeviceDescription is one RouteQuery.  It reuses
// the immutable Phase 8 resident graph, controller/status/scratch workspace,
// stream lease, and a dense device-only CSC mask image materialized for the
// selected columns during initialization.
class BatchedJacobiPullEngine final {
 public:
  BatchedJacobiPullEngine(
      const WeightedGraph& host_graph,
      const TileRunLayout64& tile_runs,
      const ResidentDeviceGraph& resident_graph,
      ReusableBatchedJacobiWorkspace& workspace,
      const HipStream& stream);
  ~BatchedJacobiPullEngine();

  BatchedJacobiPullEngine(const BatchedJacobiPullEngine&) = delete;
  BatchedJacobiPullEngine& operator=(const BatchedJacobiPullEngine&) = delete;
  BatchedJacobiPullEngine(BatchedJacobiPullEngine&&) = delete;
  BatchedJacobiPullEngine& operator=(BatchedJacobiPullEngine&&) = delete;

  [[nodiscard]] constexpr EngineKind kind() const noexcept {
    return EngineKind::jacobi_pull;
  }

  [[nodiscard]] BatchedJacobiRunOutput run_status_only(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      BatchedJacobiLoadStrategy load_strategy =
          BatchedJacobiLoadStrategy::compiler_uniform);

  // Executes the same engine and GPU all-targets finalizer, then transfers
  // exactly one DeviceRunStatus. It does not copy the final controller,
  // counters, lane traces, or either V*W distance slot. Ordinary host-poll
  // control still performs its required intermediate controller polls.
  [[nodiscard]] DeviceRunStatus run_compact_status(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      BatchedJacobiLoadStrategy load_strategy =
          BatchedJacobiLoadStrategy::compiler_uniform);

  // Runs reconstruction before the batch lease is retired. Only fixed target
  // summaries and compact path arenas cross D2H; neither Jacobi distance slot
  // is downloaded.
  [[nodiscard]] CompactPathBatchOutput run_compact_paths(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      ReusableCompactPathWorkspace& compact_workspace,
      BatchedJacobiLoadStrategy load_strategy =
          BatchedJacobiLoadStrategy::compiler_uniform);

  [[nodiscard]] BatchedJacobiRunOutput run_with_distances(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      BatchedJacobiLoadStrategy load_strategy =
          BatchedJacobiLoadStrategy::compiler_uniform);

 private:
  [[nodiscard]] BatchedJacobiRunOutput run_impl(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      BatchedJacobiLoadStrategy load_strategy,
      bool download_distances,
      bool compact_status_only,
      ReusableCompactPathWorkspace* compact_workspace,
      CompactPathBatchOutput* compact_output);

  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
