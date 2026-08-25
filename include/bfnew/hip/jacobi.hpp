#pragma once

#include "bfnew/device_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/hip/runtime.hpp"
#include "bfnew/jacobi_pull.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bfnew::hip {

// Phase 9 keeps the two Jacobi distance columns in the shared engine-scratch
// allocation.  This helper is also the single checked source of the workspace
// requirement used by callers and tests.
[[nodiscard]] std::size_t jacobi_distance_scratch_bytes(
    std::uint32_t vertex_count);

struct JacobiRunMetrics {
  // Adjacent GPU-event intervals. Preparation covers stream-ordered query
  // staging, the SSSP interval covers initialization through terminal-status
  // materialization, and result transfer covers the requested D2H records.
  float preparation_gpu_milliseconds{};
  float sssp_device_timeline_milliseconds{};
  float result_transfer_gpu_milliseconds{};
  double end_to_end_wall_milliseconds{};

  // Phase 9 compatibility aliases. They are populated from the SSSP device
  // timeline and end-to-end wall fields respectively.
  float gpu_milliseconds{};
  double wall_milliseconds{};

  // Nonzero only for PersistentCooperative.  The occupancy value is measured
  // from the real Jacobi kernel, never inherited from the capability probe.
  std::uint32_t cooperative_grid_blocks{};
  std::uint32_t cooperative_active_blocks_per_wgp{};

  // These count submitted logical round/advance kernels.  Chunked execution
  // may submit no-op pairs beyond convergence within its final K-round chunk;
  // rounds_completed in the shared status counts only rounds that executed.
  std::uint64_t engine_round_dispatches{};
  std::uint64_t controller_advance_dispatches{};
  std::uint64_t convergence_host_checks{};
};

struct JacobiRunOutput {
  GpuSsspResult result{};
  std::vector<float> distances;
  JacobiRunMetrics metrics{};
  bool distances_downloaded{};

  // On normal convergence the complete-column rule requires the two distance
  // slots to be bitwise identical. Both requested full columns or both compact
  // selected ranges are downloaded after the device stops, so parity is
  // explicit over the returned correctness region.
  bool converged_slots_bitwise_identical{};
};

// A standalone, synchronous CSC pull engine.  The resident graph and retained
// workspace are supplied by Phase 8 and remain owned by the caller.  One engine
// instance is bound to one stream, matching the workspace lease contract.
class JacobiPullEngine final : public GpuSsspEngine {
 public:
  JacobiPullEngine(
      const WeightedGraph& host_graph,
      const TileRunLayout64& tile_runs,
      const ResidentDeviceGraph& resident_graph,
      ReusableDeviceWorkspace& workspace,
      const HipStream& stream);
  ~JacobiPullEngine() override;

  JacobiPullEngine(const JacobiPullEngine&) = delete;
  JacobiPullEngine& operator=(const JacobiPullEngine&) = delete;
  JacobiPullEngine(JacobiPullEngine&&) = delete;
  JacobiPullEngine& operator=(JacobiPullEngine&&) = delete;

  [[nodiscard]] EngineKind kind() const noexcept override {
    return EngineKind::jacobi_pull;
  }

  // The common interface returns controller/status/instrumentation only.
  [[nodiscard]] GpuSsspResult run(
      const RouteQuery& query,
      const GpuRunOptions& options) override;

  // Production/status path used by fair timing. It transfers the terminal
  // controller, status, and requested instrumentation, but no graph-sized
  // distance columns. Consequently distances is empty and the two-column
  // identity flag is false in the returned output.
  [[nodiscard]] JacobiRunOutput run_status_only(
      const RouteQuery& query,
      const GpuRunOptions& options);

  // Phase 9 correctness validation additionally needs the final labels.
  [[nodiscard]] JacobiRunOutput run_with_distances(
      const RouteQuery& query,
      const GpuRunOptions& options);

  // Correctness-only compact readback. Distances are concatenated in the
  // canonical selected-tile order, then vertex order within each tile range.
  [[nodiscard]] JacobiRunOutput run_with_selected_distances(
      const RouteQuery& query,
      const GpuRunOptions& options);

 private:
  enum class DistanceReadback : std::uint8_t {
    none,
    full_graph,
    selected_ranges,
  };

  [[nodiscard]] JacobiRunOutput run_impl(
      const RouteQuery& query,
      const GpuRunOptions& options,
      DistanceReadback readback);

  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
