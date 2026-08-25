#pragma once

#include "bfnew/device_layout.hpp"
#include "bfnew/engine_shootout.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/hip/runtime.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace bfnew::hip {

enum class ShootoutDistanceComparison : std::uint8_t {
  bitwise = 0U,
  within_four_ulps = 1U,
};

// Compact one-time index used by the representative-corpus pilot. Building it
// visits each immutable CSR tile run once. Per-query selected-edge estimates
// then inspect only tile-pair rows rather than rescanning every graph vertex or
// edge for every query.
class ShootoutTilePairIndex final {
 public:
  ShootoutTilePairIndex(
      const WeightedGraph& graph,
      const TileRunLayout64& tile_runs);

  [[nodiscard]] std::uint64_t selected_vertex_count(
      std::span<const TileId> selected_tiles) const;
  [[nodiscard]] std::uint64_t selected_edge_count(
      std::span<const TileId> selected_tiles) const;
  [[nodiscard]] std::uint32_t tile_count() const noexcept;

 private:
  struct DestinationCount {
    TileId destination{};
    std::uint64_t edges{};

    constexpr bool operator==(const DestinationCount&) const noexcept =
        default;
  };

  std::vector<std::uint64_t> vertex_counts_;
  std::vector<std::vector<DestinationCount>> outgoing_;
};

struct ShootoutEngineExecution {
  ShootoutSample sample;
  // Populated only for correctness, packed in selected-tile/range order.
  // Timing/counter/profiler and pilot paths call run_status_only and must leave
  // this vector empty.
  std::vector<float> distances;
};

// Sequential standalone-query executor. It owns no graph/workspace/stream;
// those retained objects outlive it. begin_stage validates one persisted
// manifest/catalog and, for every post-correctness stage, requires the complete
// matching correctness matrix before the first engine call.
class EngineShootoutExecutor final {
 public:
  EngineShootoutExecutor(
      const WeightedGraph& host_graph,
      const TileRunLayout64& tile_runs,
      const ResidentDeviceGraph& resident_graph,
      ReusableDeviceWorkspace& workspace,
      const HipStream& stream);
  ~EngineShootoutExecutor();

  EngineShootoutExecutor(const EngineShootoutExecutor&) = delete;
  EngineShootoutExecutor& operator=(const EngineShootoutExecutor&) = delete;
  EngineShootoutExecutor(EngineShootoutExecutor&&) = delete;
  EngineShootoutExecutor& operator=(EngineShootoutExecutor&&) = delete;

  // Executes the actual ordinary and persistent kernels independently at
  // maximum_rounds=1. The instrumentation parameter selects the exact None or
  // Debug kernel variants whose legality is being discovered. This is
  // configuration discovery, never timing evidence.
  [[nodiscard]] std::vector<ShootoutKernelLimit> probe_kernel_limits(
      const RouteQuery& query,
      InstrumentationLevel instrumentation);

  // Deterministic selection metadata. The Jacobi pilot is status-only and uses
  // per-round control so it does not require a cooperative launch. It must
  // converge normally; its runtime is never retained as performance evidence.
  [[nodiscard]] std::uint64_t run_jacobi_pilot(
      const RouteQuery& query,
      std::uint64_t maximum_rounds,
      std::uint32_t block_size = 256U);

  void begin_stage(
      const ShootoutManifest& manifest,
      const ShootoutConfigurationCatalog& catalog,
      std::span<const ShootoutSample> correctness_samples,
      ShootoutRunKind run_kind,
      InstrumentationLevel instrumentation);

  // expected_distances is mandatory only for correctness and must contain one
  // label per selected vertex, packed in canonical selected-tile/range order.
  // No expected-label span may be supplied to a status-only stage. Timing
  // rejects any retained-workspace allocation event observed across the call.
  [[nodiscard]] ShootoutEngineExecution execute(
      const RouteQuery& query,
      const ShootoutScheduleEntry& schedule,
      std::span<const float> expected_distances = {},
      ShootoutDistanceComparison comparison =
          ShootoutDistanceComparison::within_four_ulps);

  // Profile-case replay puts one unrecorded in-process warmup before these
  // named, stream-ordered marker kernels. External trace/PMC extraction must
  // retain only kernels between the begin/end marker symbols.
  void emit_profile_range_begin_marker();
  void emit_profile_range_end_marker();

  [[nodiscard]] std::uint64_t workspace_allocation_events() const noexcept;

 private:
  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
