#include "bfnew/hip/engine_shootout.hpp"

#include "bfnew/hip/dense_chaotic_push.hpp"
#include "bfnew/hip/frontier_push.hpp"
#include "bfnew/hip/jacobi.hpp"
#include "bfnew/sssp.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bfnew::hip {
namespace {

constexpr std::array<std::uint32_t, 3U> required_block_sizes{
    128U, 256U, 512U};

void require(const bool condition, const char* const message) {
  if (!condition) {
    throw std::invalid_argument{message};
  }
}

[[nodiscard]] bool same_workload(
    const ShootoutWorkloadIdentity& left,
    const ShootoutWorkloadIdentity& right) noexcept {
  return left == right;
}

[[nodiscard]] bool valid_stage_instrumentation(
    const ShootoutRunKind kind,
    const InstrumentationLevel instrumentation) noexcept {
  switch (kind) {
    case ShootoutRunKind::algorithm_counters:
      return instrumentation == InstrumentationLevel::debug;
    case ShootoutRunKind::warmup:
    case ShootoutRunKind::correctness:
    case ShootoutRunKind::timing:
    case ShootoutRunKind::trace:
    case ShootoutRunKind::pmc:
      return instrumentation == InstrumentationLevel::none;
  }
  return false;
}

[[nodiscard]] GpuRunOptions options_for(
    const ShootoutTuning& tuning,
    const InstrumentationLevel instrumentation) noexcept {
  GpuRunOptions options;
  options.engine = tuning.engine;
  options.control_mode = tuning.control_mode;
  options.rounds_per_chunk = tuning.rounds_per_chunk;
  options.block_size = tuning.block_size;
  options.grid_policy = tuning.grid_policy;
  options.blocks_per_wgp = tuning.blocks_per_wgp;
  options.instrumentation = instrumentation;
  options.maximum_rounds = tuning.maximum_rounds;
  options.enable_per_lane_convergence = 1U;
  return options;
}

struct NormalizedOutput {
  GpuSsspResult result{};
  std::vector<float> distances;
  double preparation_gpu_milliseconds{};
  double sssp_device_timeline_milliseconds{};
  double result_transfer_gpu_milliseconds{};
  double end_to_end_wall_milliseconds{};
  std::uint32_t cooperative_grid_blocks{};
  std::uint32_t cooperative_active_blocks_per_wgp{};
};

template <typename Output>
[[nodiscard]] NormalizedOutput normalize_output(Output output) {
  NormalizedOutput normalized;
  normalized.result = output.result;
  normalized.distances = std::move(output.distances);
  normalized.preparation_gpu_milliseconds =
      output.metrics.preparation_gpu_milliseconds;
  normalized.sssp_device_timeline_milliseconds =
      output.metrics.sssp_device_timeline_milliseconds;
  normalized.result_transfer_gpu_milliseconds =
      output.metrics.result_transfer_gpu_milliseconds;
  normalized.end_to_end_wall_milliseconds =
      output.metrics.end_to_end_wall_milliseconds;
  normalized.cooperative_grid_blocks = output.metrics.cooperative_grid_blocks;
  normalized.cooperative_active_blocks_per_wgp =
      output.metrics.cooperative_active_blocks_per_wgp;
  return normalized;
}

void require_valid_metric(const double value, const char* const name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::runtime_error{std::string{"shootout engine produced invalid "} +
                             name};
  }
}

[[nodiscard]] bool distances_agree(
    const std::span<const float> actual,
    const std::span<const float> expected,
    const ShootoutDistanceComparison comparison) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    if (comparison == ShootoutDistanceComparison::bitwise) {
      if (std::bit_cast<std::uint32_t>(actual[index]) !=
          std::bit_cast<std::uint32_t>(expected[index])) {
        return false;
      }
    } else if (!nonnegative_distance_within_ulps(
                   actual[index], expected[index], 4U)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::uint64_t tile_pair_key(
    const TileId source,
    const TileId destination) noexcept {
  return (static_cast<std::uint64_t>(source.value()) << 32U) |
         destination.value();
}

[[nodiscard]] hipStream_t as_stream(const void* const handle) noexcept {
  return reinterpret_cast<hipStream_t>(const_cast<void*>(handle));
}

// Deliberately distinct, stable kernel symbols make the measured replay range
// recognizable in rocprof traces without treating host timestamps as GPU
// evidence. Keep these kernels empty and launch them on the campaign stream.
__global__ void bfnew_shootout_profile_range_begin_marker_kernel() {}
__global__ void bfnew_shootout_profile_range_end_marker_kernel() {}

void check_marker_launch(const char* const name) {
  throw_if_hip_error(
      static_cast<std::int32_t>(hipGetLastError()), name);
}

[[nodiscard]] bool is_kernel_legality_error(
    const std::runtime_error& error) noexcept {
  const std::string_view message{error.what()};
  return message.find("zero occupancy") != std::string_view::npos ||
         message.find("zero reported occupancy") != std::string_view::npos;
}

}  // namespace

ShootoutTilePairIndex::ShootoutTilePairIndex(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs) {
  if (validate_tile_run_layout(graph, tile_runs).ok() == false) {
    throw std::invalid_argument{
        "shootout tile-pair index requires valid CSR/CSC tile runs"};
  }
  const std::size_t tiles = graph.tile_coordinates().size();
  vertex_counts_.resize(tiles, 0U);
  outgoing_.resize(tiles);
  const auto tile_vertex_offsets = graph.tile_vertex_offsets();
  for (std::size_t tile = 0U; tile < tiles; ++tile) {
    vertex_counts_[tile] =
        tile_vertex_offsets[tile + 1U] - tile_vertex_offsets[tile];
  }

  std::unordered_map<std::uint64_t, std::uint64_t> counts;
  const std::size_t reserve_hint =
      tiles > std::numeric_limits<std::size_t>::max() / 8U
          ? tile_runs.csr_run_destination_tiles.size()
          : std::min(
                tile_runs.csr_run_destination_tiles.size(), tiles * 8U);
  counts.reserve(reserve_hint);
  const auto owner_tiles = graph.owner_tiles();
  for (std::size_t vertex = 0U; vertex < graph.vertex_count(); ++vertex) {
    const TileId source_tile = owner_tiles[vertex];
    const std::size_t begin = static_cast<std::size_t>(
        tile_runs.csr_row_run_offsets[vertex]);
    const std::size_t end = static_cast<std::size_t>(
        tile_runs.csr_row_run_offsets[vertex + 1U]);
    for (std::size_t run = begin; run < end; ++run) {
      const TileId destination_tile =
          tile_runs.csr_run_destination_tiles[run];
      const std::uint64_t edges =
          tile_runs.csr_run_edge_offsets[run + 1U] -
          tile_runs.csr_run_edge_offsets[run];
      std::uint64_t& count = counts[tile_pair_key(source_tile, destination_tile)];
      if (edges > std::numeric_limits<std::uint64_t>::max() - count) {
        throw std::overflow_error{"shootout tile-pair edge count overflow"};
      }
      count += edges;
    }
  }
  for (const auto& [key, edges] : counts) {
    const TileId source{static_cast<std::uint32_t>(key >> 32U)};
    const TileId destination{static_cast<std::uint32_t>(key)};
    outgoing_[source.value()].push_back(DestinationCount{destination, edges});
  }
  for (auto& row : outgoing_) {
    std::sort(
        row.begin(),
        row.end(),
        [](const DestinationCount& left, const DestinationCount& right) {
          return left.destination < right.destination;
        });
  }
}

std::uint64_t ShootoutTilePairIndex::selected_vertex_count(
    const std::span<const TileId> selected_tiles) const {
  std::uint64_t result = 0U;
  TileId previous{};
  bool has_previous = false;
  for (const TileId tile : selected_tiles) {
    if (tile.value() >= vertex_counts_.size()) {
      throw std::out_of_range{"shootout selected tile is outside the graph"};
    }
    if (has_previous && !(previous < tile)) {
      throw std::invalid_argument{
          "shootout selected tiles must be sorted and unique"};
    }
    const std::uint64_t vertices = vertex_counts_[tile.value()];
    if (vertices > std::numeric_limits<std::uint64_t>::max() - result) {
      throw std::overflow_error{"shootout selected vertex count overflow"};
    }
    result += vertices;
    previous = tile;
    has_previous = true;
  }
  return result;
}

std::uint64_t ShootoutTilePairIndex::selected_edge_count(
    const std::span<const TileId> selected_tiles) const {
  static_cast<void>(selected_vertex_count(selected_tiles));
  std::uint64_t result = 0U;
  for (const TileId source : selected_tiles) {
    for (const DestinationCount& destination : outgoing_[source.value()]) {
      if (!std::binary_search(
              selected_tiles.begin(),
              selected_tiles.end(),
              destination.destination)) {
        continue;
      }
      if (destination.edges >
          std::numeric_limits<std::uint64_t>::max() - result) {
        throw std::overflow_error{"shootout selected edge count overflow"};
      }
      result += destination.edges;
    }
  }
  return result;
}

std::uint32_t ShootoutTilePairIndex::tile_count() const noexcept {
  return static_cast<std::uint32_t>(vertex_counts_.size());
}

class EngineShootoutExecutor::Impl final {
 public:
  Impl(
      const WeightedGraph& graph_value,
      const TileRunLayout64& runs_value,
      const ResidentDeviceGraph& resident_value,
      ReusableDeviceWorkspace& workspace_value,
      const HipStream& stream_value)
      : graph{graph_value},
        runs{runs_value},
        resident{resident_value},
        workspace{workspace_value},
        stream{stream_value},
        jacobi{graph, runs, resident, workspace, stream},
        dense{graph, runs, resident, workspace, stream},
        frontier{graph, runs, resident, workspace, stream} {}

  [[nodiscard]] NormalizedOutput run_engine(
      const RouteQuery& query,
      const GpuRunOptions& options,
      const bool download_selected_distances) {
    switch (options.engine) {
      case EngineKind::jacobi_pull:
        return download_selected_distances
                   ? normalize_output(
                         jacobi.run_with_selected_distances(query, options))
                   : normalize_output(jacobi.run_status_only(query, options));
      case EngineKind::dense_chaotic_push:
        return download_selected_distances
                   ? normalize_output(
                         dense.run_with_selected_distances(query, options))
                   : normalize_output(dense.run_status_only(query, options));
      case EngineKind::frontier_push:
        return download_selected_distances
                   ? normalize_output(
                         frontier.run_with_selected_distances(query, options))
                   : normalize_output(frontier.run_status_only(query, options));
    }
    throw std::invalid_argument{"shootout requested an unknown engine"};
  }

  const WeightedGraph& graph;
  const TileRunLayout64& runs;
  const ResidentDeviceGraph& resident;
  ReusableDeviceWorkspace& workspace;
  const HipStream& stream;
  JacobiPullEngine jacobi;
  DenseChaoticPushEngine dense;
  FrontierPushEngine frontier;

  bool stage_active{};
  ShootoutManifest manifest;
  ShootoutConfigurationCatalog catalog;
  ShootoutRunKind run_kind{ShootoutRunKind::correctness};
  InstrumentationLevel instrumentation{InstrumentationLevel::none};
  std::map<std::uint32_t, ShootoutTuning> tunings;
  std::set<std::uint32_t> query_ids;
};

EngineShootoutExecutor::EngineShootoutExecutor(
    const WeightedGraph& host_graph,
    const TileRunLayout64& tile_runs,
    const ResidentDeviceGraph& resident_graph,
    ReusableDeviceWorkspace& workspace,
    const HipStream& stream)
    : impl_{new Impl{
          host_graph, tile_runs, resident_graph, workspace, stream}} {}

EngineShootoutExecutor::~EngineShootoutExecutor() { delete impl_; }

std::vector<ShootoutKernelLimit> EngineShootoutExecutor::probe_kernel_limits(
    const RouteQuery& query,
    const InstrumentationLevel instrumentation) {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot use a moved-from shootout executor"};
  }
  require(
      validate_route_query(impl_->graph, query).ok(),
      "shootout occupancy probe requires a valid route query");
  require(
      instrumentation == InstrumentationLevel::none ||
          instrumentation == InstrumentationLevel::debug,
      "shootout occupancy probe supports only None or Debug kernels");
  std::vector<ShootoutKernelLimit> result;
  result.reserve(9U);
  for (const EngineKind engine : {
           EngineKind::jacobi_pull,
           EngineKind::dense_chaotic_push,
           EngineKind::frontier_push}) {
    for (const std::uint32_t block_size : required_block_sizes) {
      const auto make_options = [&](const ControlMode control_mode) {
        GpuRunOptions options;
        options.engine = engine;
        options.control_mode = control_mode;
        options.rounds_per_chunk = 1U;
        options.block_size = block_size;
        options.grid_policy = GridPolicy::occupancy_derived;
        options.blocks_per_wgp = 0U;
        options.instrumentation = instrumentation;
        options.maximum_rounds = 1U;
        options.enable_per_lane_convergence = 1U;
        return options;
      };

      bool ordinary_legal = false;
      try {
        const NormalizedOutput output =
            impl_->run_engine(
                query, make_options(ControlMode::per_round_host_poll), false);
        if (!output.distances.empty()) {
          throw std::runtime_error{
              "ordinary shootout probe returned graph-sized output"};
        }
        ordinary_legal = true;
      } catch (const std::invalid_argument&) {
        ordinary_legal = false;
      } catch (const std::runtime_error& error) {
        if (!is_kernel_legality_error(error)) {
          throw;
        }
      }

      bool persistent_legal = false;
      std::uint32_t persistent_active_blocks_per_wgp = 0U;
      try {
        const NormalizedOutput output = impl_->run_engine(
            query,
            make_options(ControlMode::persistent_cooperative),
            false);
        if (!output.distances.empty() ||
            output.cooperative_active_blocks_per_wgp == 0U) {
          throw std::runtime_error{
              "persistent shootout probe returned invalid status-only output"};
        }
        persistent_legal = true;
        persistent_active_blocks_per_wgp =
            output.cooperative_active_blocks_per_wgp;
      } catch (const std::invalid_argument&) {
        persistent_legal = false;
      } catch (const std::runtime_error& error) {
        if (!is_kernel_legality_error(error)) {
          throw;
        }
      }
      result.push_back(ShootoutKernelLimit{
          engine,
          block_size,
          persistent_active_blocks_per_wgp,
          ordinary_legal,
          persistent_legal,
      });
    }
  }
  return result;
}

void EngineShootoutExecutor::emit_profile_range_begin_marker() {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot use a moved-from shootout executor"};
  }
  hipLaunchKernelGGL(
      bfnew_shootout_profile_range_begin_marker_kernel,
      dim3(1U),
      dim3(1U),
      0U,
      as_stream(impl_->stream.native_handle()));
  check_marker_launch("bfnew_shootout_profile_range_begin_marker_kernel");
}

void EngineShootoutExecutor::emit_profile_range_end_marker() {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot use a moved-from shootout executor"};
  }
  hipLaunchKernelGGL(
      bfnew_shootout_profile_range_end_marker_kernel,
      dim3(1U),
      dim3(1U),
      0U,
      as_stream(impl_->stream.native_handle()));
  check_marker_launch("bfnew_shootout_profile_range_end_marker_kernel");
  impl_->stream.synchronize();
}

std::uint64_t EngineShootoutExecutor::run_jacobi_pilot(
    const RouteQuery& query,
    const std::uint64_t maximum_rounds,
    const std::uint32_t block_size) {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot use a moved-from shootout executor"};
  }
  GpuRunOptions options;
  options.engine = EngineKind::jacobi_pull;
  options.control_mode = ControlMode::per_round_host_poll;
  options.rounds_per_chunk = 1U;
  options.block_size = block_size;
  options.grid_policy = GridPolicy::occupancy_derived;
  options.blocks_per_wgp = 0U;
  options.instrumentation = InstrumentationLevel::none;
  options.maximum_rounds = maximum_rounds;
  options.enable_per_lane_convergence = 1U;
  if (validate_gpu_run_options(options) != GpuRunOptionsError::none) {
    throw std::invalid_argument{"shootout Jacobi pilot options are invalid"};
  }
  const NormalizedOutput output = impl_->run_engine(query, options, false);
  if (!output.distances.empty() || output.result.status.converged == 0U ||
      output.result.status.stop_reason !=
          static_cast<std::uint32_t>(DeviceStopReason::converged) ||
      output.result.status.rounds_completed == 0U) {
    throw std::runtime_error{
        "shootout Jacobi pilot did not converge with status-only output"};
  }
  return output.result.status.rounds_completed;
}

void EngineShootoutExecutor::begin_stage(
    const ShootoutManifest& manifest,
    const ShootoutConfigurationCatalog& catalog,
    const std::span<const ShootoutSample> correctness_samples,
    const ShootoutRunKind run_kind,
    const InstrumentationLevel instrumentation) {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot use a moved-from shootout executor"};
  }
  validate_shootout_configuration_catalog(catalog);
  // Serialization is also the public deep manifest validator.
  static_cast<void>(serialize_shootout_manifest_tsv(manifest));
  require(
      catalog.fingerprint == manifest.fingerprint,
      "shootout catalog fingerprint does not match its manifest");
  require(
      same_workload(catalog.workload, manifest.workload),
      "shootout catalog workload does not match its manifest");
  require(
      valid_stage_instrumentation(run_kind, instrumentation),
      "shootout stage instrumentation is invalid");

  if (run_kind == ShootoutRunKind::correctness) {
    require(
        correctness_samples.empty(),
        "shootout correctness stage cannot consume prior correctness rows");
  } else {
    require_complete_correctness_gate(
        manifest, catalog.tunings, correctness_samples, {});
  }

  impl_->manifest = manifest;
  impl_->catalog = catalog;
  impl_->run_kind = run_kind;
  impl_->instrumentation = instrumentation;
  impl_->tunings.clear();
  for (const ShootoutTuning& tuning : catalog.tunings) {
    impl_->tunings.emplace(tuning.configuration_id, tuning);
  }
  impl_->query_ids.clear();
  for (const ShootoutManifestEntry& entry : manifest.entries) {
    impl_->query_ids.insert(entry.features.query_id.value());
  }
  impl_->stage_active = true;
}

ShootoutEngineExecution EngineShootoutExecutor::execute(
    const RouteQuery& query,
    const ShootoutScheduleEntry& schedule,
    const std::span<const float> expected_distances,
    const ShootoutDistanceComparison comparison) {
  if (impl_ == nullptr || !impl_->stage_active) {
    throw std::logic_error{"shootout executor has no active evidence stage"};
  }
  require(
      validate_route_query(impl_->graph, query).ok(),
      "shootout execution requires a valid route query");
  require(
      schedule.run_kind == impl_->run_kind &&
          same_workload(schedule.workload, impl_->manifest.workload),
      "shootout schedule entry does not match the active stage");
  require(
      schedule.query_id == query.query_id &&
          impl_->query_ids.contains(query.query_id.value()),
      "shootout schedule query is absent or mismatched");
  const auto tuning = impl_->tunings.find(schedule.configuration_id);
  require(
      tuning != impl_->tunings.end(),
      "shootout schedule configuration is absent from its catalog");

  const bool correctness = impl_->run_kind == ShootoutRunKind::correctness;
  if (correctness) {
    const auto tile_offsets = impl_->graph.tile_vertex_offsets();
    std::size_t selected_vertices = 0U;
    for (const TileId tile : query.selected_tiles) {
      selected_vertices += static_cast<std::size_t>(
          tile_offsets[tile.value() + 1U] - tile_offsets[tile.value()]);
    }
    require(
        expected_distances.size() == selected_vertices,
        "shootout correctness requires one expected label per selected vertex");
  } else {
    require(
        expected_distances.empty(),
        "shootout status-only stage cannot consume expected distance labels");
  }

  const GpuRunOptions options =
      options_for(tuning->second, impl_->instrumentation);
  if (validate_gpu_run_options(options) != GpuRunOptionsError::none) {
    throw std::invalid_argument{"shootout catalog produced invalid run options"};
  }
  const std::uint64_t allocations_before = impl_->workspace.allocation_events();
  NormalizedOutput output =
      impl_->run_engine(query, options, correctness);
  const std::uint64_t allocations_after = impl_->workspace.allocation_events();
  if (impl_->run_kind == ShootoutRunKind::timing &&
      allocations_after != allocations_before) {
    throw std::runtime_error{
        "shootout timing rejected retained-workspace allocation growth"};
  }
  if (!correctness && !output.distances.empty()) {
    throw std::runtime_error{
        "shootout status-only stage downloaded distance labels"};
  }

  ShootoutEngineExecution execution;
  execution.sample.fingerprint = impl_->manifest.fingerprint;
  execution.sample.workload = impl_->manifest.workload;
  execution.sample.run_kind = impl_->run_kind;
  execution.sample.execution_ordinal = schedule.execution_ordinal;
  execution.sample.repetition = schedule.repetition;
  execution.sample.query_id = query.query_id;
  execution.sample.configuration_id = schedule.configuration_id;
  execution.sample.instrumentation = impl_->instrumentation;
  execution.sample.distances_downloaded = correctness;
  execution.sample.result = output.result;
  execution.sample.cooperative_grid_blocks =
      output.cooperative_grid_blocks;
  execution.sample.cooperative_active_blocks_per_wgp =
      output.cooperative_active_blocks_per_wgp;

  if (correctness) {
    execution.sample.correctness_passed =
        output.result.status.converged != 0U &&
        output.result.status.stop_reason ==
            static_cast<std::uint32_t>(DeviceStopReason::converged) &&
        distances_agree(output.distances, expected_distances, comparison);
    execution.distances = std::move(output.distances);
  } else {
    // Gated evidence rows name the already-passed correctness matrix. Warmup
    // is deliberately non-evidence and keeps this field clear.
    execution.sample.correctness_passed =
        impl_->run_kind != ShootoutRunKind::warmup;
  }

  if (impl_->run_kind == ShootoutRunKind::timing) {
    require_valid_metric(
        output.preparation_gpu_milliseconds, "preparation GPU time");
    require_valid_metric(
        output.sssp_device_timeline_milliseconds, "SSSP device timeline");
    require_valid_metric(
        output.result_transfer_gpu_milliseconds, "result-transfer GPU time");
    require_valid_metric(output.end_to_end_wall_milliseconds, "wall time");
    execution.sample.timing.preparation_gpu_milliseconds = {
        ShootoutEvidenceState::measured,
        output.preparation_gpu_milliseconds,
    };
    execution.sample.timing.sssp_device_timeline_milliseconds = {
        ShootoutEvidenceState::measured,
        output.sssp_device_timeline_milliseconds,
    };
    execution.sample.timing.result_transfer_gpu_milliseconds = {
        ShootoutEvidenceState::measured,
        output.result_transfer_gpu_milliseconds,
    };
    execution.sample.timing.end_to_end_wall_milliseconds = {
        ShootoutEvidenceState::measured,
        output.end_to_end_wall_milliseconds,
    };
  }
  return execution;
}

std::uint64_t EngineShootoutExecutor::workspace_allocation_events() const
    noexcept {
  return impl_ == nullptr ? 0U : impl_->workspace.allocation_events();
}

}  // namespace bfnew::hip
