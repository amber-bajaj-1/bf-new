#pragma once

#include "bfnew/dense_chaotic_push.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/hip/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bfnew::hip {

// Dense chaotic push owns exactly one atomic-compatible 32-bit distance word
// per vertex in the shared engine-scratch allocation. It has no frontier,
// predecessor, or second distance column.
[[nodiscard]] std::size_t dense_distance_scratch_bytes(
    std::uint32_t vertex_count);

struct DenseRunMetrics {
  float preparation_gpu_milliseconds{};
  float sssp_device_timeline_milliseconds{};
  float result_transfer_gpu_milliseconds{};
  double end_to_end_wall_milliseconds{};

  // Phase 10 compatibility aliases.
  float gpu_milliseconds{};
  double wall_milliseconds{};
  std::uint32_t cooperative_grid_blocks{};
  std::uint32_t cooperative_active_blocks_per_wgp{};
  std::uint64_t engine_round_dispatches{};
  std::uint64_t controller_advance_dispatches{};
  std::uint64_t convergence_host_checks{};
};

struct DenseRunOutput {
  GpuSsspResult result{};
  std::vector<float> distances;
  DenseRunMetrics metrics{};
  bool distances_downloaded{};
};

// A standalone CSR push engine bound to one resident graph, reusable
// workspace, and stream. The graph remains immutable and resident across runs.
class DenseChaoticPushEngine final : public GpuSsspEngine {
 public:
  DenseChaoticPushEngine(
      const WeightedGraph& host_graph,
      const TileRunLayout64& tile_runs,
      const ResidentDeviceGraph& resident_graph,
      ReusableDeviceWorkspace& workspace,
      const HipStream& stream);
  ~DenseChaoticPushEngine() override;

  DenseChaoticPushEngine(const DenseChaoticPushEngine&) = delete;
  DenseChaoticPushEngine& operator=(const DenseChaoticPushEngine&) = delete;
  DenseChaoticPushEngine(DenseChaoticPushEngine&&) = delete;
  DenseChaoticPushEngine& operator=(DenseChaoticPushEngine&&) = delete;

  [[nodiscard]] EngineKind kind() const noexcept override {
    return EngineKind::dense_chaotic_push;
  }

  [[nodiscard]] GpuSsspResult run(
      const RouteQuery& query,
      const GpuRunOptions& options) override;

  // Transfers only terminal controller/status and requested instrumentation;
  // the returned distance vector is empty.
  [[nodiscard]] DenseRunOutput run_status_only(
      const RouteQuery& query,
      const GpuRunOptions& options);

  [[nodiscard]] DenseRunOutput run_with_distances(
      const RouteQuery& query,
      const GpuRunOptions& options);

  // Correctness-only compact readback in selected-tile/range order.
  [[nodiscard]] DenseRunOutput run_with_selected_distances(
      const RouteQuery& query,
      const GpuRunOptions& options);

 private:
  enum class DistanceReadback : std::uint8_t {
    none,
    full_graph,
    selected_ranges,
  };

  [[nodiscard]] DenseRunOutput run_impl(
      const RouteQuery& query,
      const GpuRunOptions& options,
      DistanceReadback readback);

  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
