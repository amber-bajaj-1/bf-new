#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/batched_frontier_push.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/hip/batched_frontier_workspace.hpp"
#include "bfnew/hip/compact_path_results.hpp"
#include "bfnew/hip/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bfnew::hip {

// Returns the exact public Phase 16 layout size: V*W distance words, two
// graph-sized activity-mask arrays, then two bounded vertex queues. A zero
// queue request means V; an explicit request must be in [1,V].
[[nodiscard]] std::size_t batched_frontier_scratch_bytes(
    std::uint32_t vertex_count,
    std::uint32_t lane_width,
    std::uint32_t queue_capacity = 0U);

struct BatchedFrontierOptionalHardwareCounters {
  // No profiler evidence is synthesized by the runtime. These stay zero and
  // unavailable until a future external/rocprofiler-SDK collection populates
  // them for this exact kernel.
  std::uint32_t available{};
  std::uint64_t l2_read_bytes{};
  std::uint64_t l2_write_bytes{};
  std::uint64_t atomic_stall_cycles{};
  std::uint64_t waves{};
};

struct BatchedFrontierRunMetrics {
  float preparation_gpu_milliseconds{};
  float sssp_device_timeline_milliseconds{};
  float result_transfer_gpu_milliseconds{};
  double end_to_end_wall_milliseconds{};

  std::uint32_t ordinary_grid_blocks{};
  std::uint32_t ordinary_active_blocks_per_wgp{};
  std::uint32_t cooperative_grid_blocks{};
  std::uint32_t cooperative_active_blocks_per_wgp{};
  std::uint32_t kernel_registers_per_thread{};
  std::uint32_t lane_width{};
  std::uint32_t valid_lane_count{};
  std::uint32_t queue_capacity{};

  std::uint64_t engine_round_dispatches{};
  std::uint64_t controller_advance_dispatches{};
  std::uint64_t convergence_host_checks{};

  // Algorithmic byte requests, not measured cache traffic.
  std::uint64_t edge_record_read_bytes_requested{};
  std::uint64_t distance_atomic_read_bytes_requested{};
  std::uint64_t distance_atomic_min_bytes_requested{};
  std::uint64_t activity_mask_atomic_bytes_requested{};

  double average_active_lanes_per_frontier_vertex{};
  double average_active_lanes_per_nonzero_run{};
  double frontier_lane_utilization{};
  double lane_round_utilization{};
  double batch_queries_per_second{};
  BatchedFrontierOptionalHardwareCounters hardware_counters{};
};

struct BatchedFrontierRunOutput {
  GpuSsspResult result{};
  DeviceController final_controller{};

  // Empty for run_status_only(). Downloaded words preserve the exact atomic
  // representation and are projected bitwise to float.
  std::vector<float> distances;
  std::vector<std::uint32_t> distance_bits;
  bool distances_downloaded{};

  std::vector<std::uint64_t> frontier_rounds_by_lane;
  std::vector<std::uint64_t> lane_convergence_rounds;
  std::vector<std::uint64_t> lane_tail_rounds;

  BatchedFrontierWorkStatistics batch_work{};
  BatchedFrontierRunMetrics metrics{};
};

// Phase 16 retains Phase 11's simple one-owner-thread-per-worklist-vertex
// schedule. Each queue vertex carries one 32-bit query-lane mask. Edge
// balancing, wave aggregation, and adaptive scheduling are not implemented.
class BatchedFrontierPushEngine final {
 public:
  BatchedFrontierPushEngine(
      const WeightedGraph& host_graph,
      const TileRunLayout64& tile_runs,
      const ResidentDeviceGraph& resident_graph,
      ReusableBatchedFrontierWorkspace& workspace,
      const HipStream& stream,
      std::uint32_t queue_capacity = 0U);
  ~BatchedFrontierPushEngine();

  BatchedFrontierPushEngine(const BatchedFrontierPushEngine&) = delete;
  BatchedFrontierPushEngine& operator=(
      const BatchedFrontierPushEngine&) = delete;
  BatchedFrontierPushEngine(BatchedFrontierPushEngine&&) = delete;
  BatchedFrontierPushEngine& operator=(
      BatchedFrontierPushEngine&&) = delete;

  [[nodiscard]] constexpr EngineKind kind() const noexcept {
    return EngineKind::frontier_push;
  }

  [[nodiscard]] BatchedFrontierRunOutput run_status_only(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options);

  // Runs the existing on-device all-targets finalizer and copies exactly one
  // DeviceRunStatus. It excludes final controller/counter/lane-trace and V*W
  // distance transfers; ordinary modes retain mandatory controller polling.
  [[nodiscard]] DeviceRunStatus run_compact_status(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options);

  [[nodiscard]] CompactPathBatchOutput run_compact_paths(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      ReusableCompactPathWorkspace& compact_workspace);

  [[nodiscard]] BatchedFrontierRunOutput run_with_distances(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options);

 private:
  [[nodiscard]] BatchedFrontierRunOutput run_impl(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      bool download_distances,
      bool compact_status_only,
      ReusableCompactPathWorkspace* compact_workspace,
      CompactPathBatchOutput* compact_output);

  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
