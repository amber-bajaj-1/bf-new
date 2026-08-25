#pragma once

#include "bfnew/device_layout.hpp"
#include "bfnew/frontier_push.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/hip/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bfnew::hip {

[[nodiscard]] std::size_t frontier_scratch_bytes(
    std::uint32_t vertex_count,
    std::uint32_t queue_capacity = 0U);

struct FrontierRunMetrics {
  float preparation_gpu_milliseconds{};
  float sssp_device_timeline_milliseconds{};
  float result_transfer_gpu_milliseconds{};
  double end_to_end_wall_milliseconds{};

  // Phase 11 compatibility aliases.
  float gpu_milliseconds{};
  double wall_milliseconds{};
  std::uint32_t cooperative_grid_blocks{};
  std::uint32_t cooperative_active_blocks_per_wgp{};
  std::uint64_t engine_round_dispatches{};
  std::uint64_t controller_advance_dispatches{};
  std::uint64_t convergence_host_checks{};
};

struct FrontierRunOutput {
  GpuSsspResult result{};
  std::vector<float> distances;
  DeviceController final_controller{};
  FrontierRunMetrics metrics{};
  std::uint32_t queue_capacity{};
  bool distances_downloaded{};
};

// Standalone one-thread-per-frontier-entry CSR push. The optional capacity is
// primarily a validation seam for explicit overflow behavior; zero reserves
// one entry per graph vertex, which cannot overflow under correct deduplication.
class FrontierPushEngine final : public GpuSsspEngine {
 public:
  FrontierPushEngine(
      const WeightedGraph& host_graph,
      const TileRunLayout64& tile_runs,
      const ResidentDeviceGraph& resident_graph,
      ReusableDeviceWorkspace& workspace,
      const HipStream& stream,
      std::uint32_t queue_capacity = 0U);
  ~FrontierPushEngine() override;

  FrontierPushEngine(const FrontierPushEngine&) = delete;
  FrontierPushEngine& operator=(const FrontierPushEngine&) = delete;
  FrontierPushEngine(FrontierPushEngine&&) = delete;
  FrontierPushEngine& operator=(FrontierPushEngine&&) = delete;

  [[nodiscard]] EngineKind kind() const noexcept override {
    return EngineKind::frontier_push;
  }

  [[nodiscard]] GpuSsspResult run(
      const RouteQuery& query,
      const GpuRunOptions& options) override;

  // Transfers only terminal controller/status and requested instrumentation;
  // the returned distance vector is empty while final_controller remains
  // available for queue-parity and terminal-state reporting.
  [[nodiscard]] FrontierRunOutput run_status_only(
      const RouteQuery& query,
      const GpuRunOptions& options);

  [[nodiscard]] FrontierRunOutput run_with_distances(
      const RouteQuery& query,
      const GpuRunOptions& options);

  // Correctness-only compact readback in selected-tile/range order.
  [[nodiscard]] FrontierRunOutput run_with_selected_distances(
      const RouteQuery& query,
      const GpuRunOptions& options);

 private:
  enum class DistanceReadback : std::uint8_t {
    none,
    full_graph,
    selected_ranges,
  };

  [[nodiscard]] FrontierRunOutput run_impl(
      const RouteQuery& query,
      const GpuRunOptions& options,
      DistanceReadback readback);

  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
