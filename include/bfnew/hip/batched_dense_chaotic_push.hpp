#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/batched_dense_chaotic_push.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/hip/batched_dense_workspace.hpp"
#include "bfnew/hip/compact_path_results.hpp"
#include "bfnew/hip/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bfnew::hip {

// The compiler-uniform path is the default. Explicit wave broadcast remains a
// controlled experiment; neither strategy is claimed faster before target-GPU
// measurement of this exact kernel and workload.
enum class BatchedDenseLoadStrategy : std::uint32_t {
  compiler_uniform = 0U,
  explicit_wave_broadcast = 1U,
};

[[nodiscard]] constexpr bool valid_batched_dense_load_strategy(
    const BatchedDenseLoadStrategy strategy) noexcept {
  return strategy == BatchedDenseLoadStrategy::compiler_uniform ||
         strategy == BatchedDenseLoadStrategy::explicit_wave_broadcast;
}

// One atomic-compatible uint32 distance word per vertex/query lane. Storage
// is vertex-major and lane-contiguous: vertex * lane_width + lane.
[[nodiscard]] std::size_t batched_dense_distance_scratch_bytes(
    std::uint32_t vertex_count,
    std::uint32_t lane_width);

struct BatchedDenseOptionalHardwareCounters {
  // No profiler values are synthesized. These remain zero/unavailable until
  // populated by a future rocprofiler-SDK adapter or an external profile run.
  std::uint32_t available{};
  std::uint64_t l2_read_bytes{};
  std::uint64_t l2_write_bytes{};
  std::uint64_t atomic_stall_cycles{};
  std::uint64_t write_stall_cycles{};
  std::uint64_t waves{};
};

struct BatchedDenseRunMetrics {
  float preparation_gpu_milliseconds{};
  // Results are visible only when the complete batch stops; this is therefore
  // the observable latency of each returned query. Per-lane convergence rounds
  // expose earlier internal completion when freezing is enabled.
  float sssp_device_timeline_milliseconds{};
  float result_transfer_gpu_milliseconds{};
  double end_to_end_wall_milliseconds{};

  std::uint32_t ordinary_active_blocks_per_wgp{};
  std::uint32_t cooperative_grid_blocks{};
  std::uint32_t cooperative_active_blocks_per_wgp{};
  std::uint32_t kernel_registers_per_thread{};
  std::uint32_t lane_width{};
  std::uint32_t valid_lane_count{};
  BatchedDenseLoadStrategy load_strategy{
      BatchedDenseLoadStrategy::compiler_uniform};

  std::uint64_t engine_round_dispatches{};
  std::uint64_t controller_advance_dispatches{};
  std::uint64_t convergence_host_checks{};
  std::uint64_t unused_wave_lane_rounds{};

  // Algorithmic requests, not measured cache traffic.
  std::uint64_t edge_record_read_bytes_requested{};
  std::uint64_t atomic_source_read_bytes_requested{};
  std::uint64_t atomic_destination_access_bytes_requested{};

  double average_active_lanes_per_nonzero_run{};
  double lane_round_utilization{};
  // Derived from the device-timeline span. Host-poll spans include the host's
  // observation gaps, so final throughput comparisons must use the separately
  // retained end-to-end wall samples from the deferred campaign.
  double batch_queries_per_second{};
  BatchedDenseOptionalHardwareCounters hardware_counters{};
};

struct BatchedDenseRunOutput {
  GpuSsspResult result{};

  // Empty for run_status_only(). Both views are vertex-major and preserve the
  // exact one-slot atomic representation when downloaded.
  std::vector<float> distances;
  std::vector<std::uint32_t> distance_bits;
  bool distances_downloaded{};

  // Zero denotes padding or a lane that did not prove a no-change scan before
  // an error/round-limit exit. Scan numbers are one-based.
  std::vector<std::uint64_t> lane_convergence_rounds;
  std::vector<std::uint64_t> lane_executed_rounds;
  std::vector<std::uint64_t> lane_tail_rounds;

  BatchedDenseWorkStatistics batch_work{};
  BatchedDenseRunMetrics metrics{};
};

class BatchedDenseChaoticPushEngine final {
 public:
  BatchedDenseChaoticPushEngine(
      const WeightedGraph& host_graph,
      const TileRunLayout64& tile_runs,
      const ResidentDeviceGraph& resident_graph,
      ReusableBatchedDenseWorkspace& workspace,
      const HipStream& stream);
  ~BatchedDenseChaoticPushEngine();

  BatchedDenseChaoticPushEngine(const BatchedDenseChaoticPushEngine&) = delete;
  BatchedDenseChaoticPushEngine& operator=(
      const BatchedDenseChaoticPushEngine&) = delete;
  BatchedDenseChaoticPushEngine(BatchedDenseChaoticPushEngine&&) = delete;
  BatchedDenseChaoticPushEngine& operator=(
      BatchedDenseChaoticPushEngine&&) = delete;

  [[nodiscard]] constexpr EngineKind kind() const noexcept {
    return EngineKind::dense_chaotic_push;
  }

  [[nodiscard]] BatchedDenseRunOutput run_status_only(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      BatchedDenseLoadStrategy load_strategy =
          BatchedDenseLoadStrategy::compiler_uniform);

  // Runs the normal GPU finalizer and copies exactly one compact terminal
  // status. No controller, counters, lane traces, or V*W labels are returned.
  [[nodiscard]] DeviceRunStatus run_compact_status(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      BatchedDenseLoadStrategy load_strategy =
          BatchedDenseLoadStrategy::compiler_uniform);

  [[nodiscard]] CompactPathBatchOutput run_compact_paths(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      ReusableCompactPathWorkspace& compact_workspace,
      BatchedDenseLoadStrategy load_strategy =
          BatchedDenseLoadStrategy::compiler_uniform);

  [[nodiscard]] BatchedDenseRunOutput run_with_distances(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      BatchedDenseLoadStrategy load_strategy =
          BatchedDenseLoadStrategy::compiler_uniform);

 private:
  [[nodiscard]] BatchedDenseRunOutput run_impl(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      BatchedDenseLoadStrategy load_strategy,
      bool download_distances,
      bool compact_status_only,
      ReusableCompactPathWorkspace* compact_workspace,
      CompactPathBatchOutput* compact_output);

  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
