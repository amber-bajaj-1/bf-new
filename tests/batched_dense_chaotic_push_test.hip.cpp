#include "bfnew/hip/batched_dense_chaotic_push.hpp"

#include "bfnew/hip/dense_chaotic_push.hpp"
#include "bfnew/sssp.hpp"
#include "batched_dense_fixture_suite.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

inline constexpr std::uint32_t positive_infinity_bits = 0x7f800000U;

[[noreturn]] void fail(
    const std::string_view fixture,
    const std::string_view message) {
  std::cerr << "batched dense chaotic push HIP test failed [" << fixture
            << "]: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void expect(
    const bool condition,
    const std::string_view fixture,
    const std::string_view message) {
  if (!condition) {
    fail(fixture, message);
  }
}

template <typename Exception, typename Function>
void expect_throws(
    Function&& function,
    const std::string_view fixture,
    const std::string_view message) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  fail(fixture, message);
}

[[nodiscard]] bool bits_equal(const float left, const float right) noexcept {
  return std::bit_cast<std::uint32_t>(left) ==
         std::bit_cast<std::uint32_t>(right);
}

[[nodiscard]] const bfnew::RouteQuery& lane_query(
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchDeviceDescription& description,
    const std::size_t lane) {
  for (const bfnew::RouteQuery& query : queries) {
    if (query.query_id.value() == description.query_ids_by_lane[lane]) {
      return query;
    }
  }
  throw std::logic_error{"HIP batched dense lane query is missing"};
}

[[nodiscard]] std::vector<float> bounded_oracle(
    const bfnew::WeightedGraph& graph,
    const bfnew::RouteQuery& query) {
  const bfnew::InducedQueryGraph induced =
      bfnew::build_induced_query_graph(graph, query);
  const bfnew::SsspResult result =
      bfnew::dijkstra_oracle(induced.graph, induced.sources);
  std::vector<float> global(
      graph.vertex_count(), std::numeric_limits<float>::infinity());
  for (std::size_t local = 0U; local < induced.local_to_global.size();
       ++local) {
    global[induced.local_to_global[local].value()] = result.distances[local];
  }
  return global;
}

[[nodiscard]] bool selected_vertex(
    const bfnew::WeightedGraph& graph,
    const bfnew::RouteQuery& query,
    const std::size_t vertex) {
  return std::binary_search(
      query.selected_tiles.begin(),
      query.selected_tiles.end(),
      graph.owner_tiles()[vertex]);
}

[[nodiscard]] bfnew::GpuRunOptions options_for(
    const bfnew::ControlMode control,
    const bool per_lane,
    const bfnew::InstrumentationLevel instrumentation =
        bfnew::InstrumentationLevel::debug) {
  bfnew::GpuRunOptions options;
  options.engine = bfnew::EngineKind::dense_chaotic_push;
  options.control_mode = control;
  options.rounds_per_chunk = 3U;
  options.block_size = 128U;
  options.instrumentation = instrumentation;
  options.maximum_rounds = 64U;
  options.enable_per_lane_convergence = per_lane ? 1U : 0U;
  return options;
}

[[nodiscard]] std::vector<bfnew::GpuRunOptions> control_matrix(
    const bool per_lane,
    const std::uint64_t maximum_rounds = 64U,
    const bfnew::InstrumentationLevel instrumentation =
        bfnew::InstrumentationLevel::debug) {
  std::vector<bfnew::GpuRunOptions> controls;
  bfnew::GpuRunOptions persistent = options_for(
      bfnew::ControlMode::persistent_cooperative,
      per_lane,
      instrumentation);
  persistent.maximum_rounds = maximum_rounds;
  controls.push_back(persistent);
  bfnew::GpuRunOptions per_round = options_for(
      bfnew::ControlMode::per_round_host_poll,
      per_lane,
      instrumentation);
  per_round.maximum_rounds = maximum_rounds;
  controls.push_back(per_round);
  for (const std::uint32_t rounds_per_chunk :
       std::array<std::uint32_t, 5U>{2U, 4U, 8U, 16U, 32U}) {
    bfnew::GpuRunOptions chunked = options_for(
        bfnew::ControlMode::chunked_host_poll,
        per_lane,
        instrumentation);
    chunked.maximum_rounds = maximum_rounds;
    chunked.rounds_per_chunk = rounds_per_chunk;
    controls.push_back(chunked);
  }
  return controls;
}

[[nodiscard]] std::size_t distance_index(
    const std::size_t vertex,
    const std::size_t lane,
    const std::uint32_t lane_width) {
  return static_cast<std::size_t>(bfnew::batched_dense_distance_index(
      static_cast<std::uint32_t>(vertex),
      static_cast<std::uint32_t>(lane),
      lane_width));
}

[[nodiscard]] std::uint64_t selected_lane_vertex_count(
    const bfnew::BatchDeviceDescription& description) {
  std::uint64_t count = 0U;
  for (const bfnew::BatchVertexRange range :
       description.selected_vertex_ranges) {
    count += static_cast<std::uint64_t>(range.end - range.begin) *
             static_cast<std::uint64_t>(std::popcount(range.lane_mask));
  }
  return count;
}

void expect_control_accounting(
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::BatchedDenseRunOutput& output,
    const std::string_view name) {
  const bfnew::DeviceWorkStatistics& work = output.result.work;
  expect(
      work.host_checks == output.metrics.convergence_host_checks &&
          work.controller_copies ==
              output.metrics.convergence_host_checks + 2U &&
          work.host_synchronizations ==
              output.metrics.convergence_host_checks + 1U,
      name,
      "physical host-control accounting is inconsistent");

  if (options.control_mode == bfnew::ControlMode::persistent_cooperative) {
    expect(
        output.metrics.convergence_host_checks == 0U &&
            output.metrics.engine_round_dispatches == 0U &&
            output.metrics.controller_advance_dispatches == 0U &&
            output.metrics.cooperative_grid_blocks != 0U &&
            output.metrics.cooperative_active_blocks_per_wgp != 0U &&
            output.metrics.ordinary_active_blocks_per_wgp == 0U &&
            work.kernel_dispatches == 1U,
        name,
        "persistent control was not one cooperative convergence launch");
    return;
  }

  expect(
      output.metrics.ordinary_active_blocks_per_wgp != 0U &&
          output.metrics.cooperative_grid_blocks == 0U &&
          output.metrics.cooperative_active_blocks_per_wgp == 0U &&
          output.metrics.engine_round_dispatches ==
              output.metrics.controller_advance_dispatches &&
          output.metrics.convergence_host_checks != 0U &&
          work.kernel_dispatches ==
              2U + 2U * output.metrics.engine_round_dispatches,
      name,
      "ordinary control violated initialization/round/finalize accounting");

  if (options.control_mode == bfnew::ControlMode::per_round_host_poll) {
    expect(
        output.metrics.engine_round_dispatches ==
                output.result.status.rounds_completed &&
            output.metrics.convergence_host_checks ==
                output.result.status.rounds_completed,
        name,
        "per-round control did not poll after every completed scan");
    return;
  }

  const std::uint64_t chunks =
      (output.result.status.rounds_completed + options.rounds_per_chunk - 1U) /
      options.rounds_per_chunk;
  expect(
      output.metrics.convergence_host_checks == chunks &&
          output.metrics.engine_round_dispatches ==
              chunks * options.rounds_per_chunk,
      name,
      "chunked control did not retain exact queued K-round pairs");
}

void expect_instrumentation_gating(
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::BatchedDenseRunOutput& output,
    const std::string_view name) {
  const bfnew::DeviceWorkStatistics& common = output.result.work;
  const bfnew::BatchedDenseWorkStatistics& dense = output.batch_work;

  expect(
      common.queue_claims == 0U &&
          common.duplicate_suppressions == 0U &&
          common.maximum_queue_size == 0U && common.expansion_count == 0U &&
          common.high_contention_destinations == 0U &&
          common.empty_frontier_rounds == 0U &&
          common.small_frontier_rounds == 0U &&
          common.overflow_events == 0U,
      name,
      "batched dense exposed frontier or unavailable standalone counters");

  if (options.instrumentation == bfnew::InstrumentationLevel::none) {
    expect(
        common.edges_examined == 0U &&
            common.successful_decreases == 0U &&
            common.active_vertices == 0U &&
            common.active_lane_rounds == 0U &&
            common.atomic_attempts == 0U &&
            common.successful_atomic_updates == 0U &&
            common.mask_operations == 0U &&
            common.changed_flag_updates == 0U &&
            common.full_edge_rounds == 0U &&
            dense.successful_atomic_updates == 0U &&
            dense.changed_round_publications == 0U,
        name,
        "None instrumentation published an algorithmic device counter");
    return;
  }

  expect(
      common.edges_examined == dense.lane_edge_relaxations &&
          common.successful_decreases ==
              dense.successful_atomic_updates &&
          common.successful_decreases != 0U &&
          common.active_vertices ==
              dense.active_source_lane_evaluations &&
          common.active_lane_rounds == dense.active_lane_rounds &&
          common.full_edge_rounds == dense.full_edge_rounds,
      name,
      "Light aggregate counters disagree with exact semantic accounting");

  if (options.instrumentation == bfnew::InstrumentationLevel::light) {
    expect(
        common.atomic_attempts == 0U &&
            common.successful_atomic_updates == 0U &&
            common.mask_operations == 0U &&
            common.changed_flag_updates == 0U &&
            dense.changed_round_publications == 0U,
        name,
        "Light instrumentation published Debug atomic or mask counters");
    return;
  }

  expect(
      common.atomic_attempts == dense.atomic_min_attempts &&
          common.atomic_attempts == dense.lane_edge_relaxations &&
          common.successful_atomic_updates ==
              dense.successful_atomic_updates &&
          common.successful_atomic_updates != 0U &&
          common.mask_operations == dense.csr_runs_considered &&
          common.changed_flag_updates ==
              dense.changed_round_publications &&
          common.changed_flag_updates != 0U,
      name,
      "Debug instrumentation omitted or miscounted atomic diagnostics");
}

void validate_output(
    const bfnew::test::BatchedDenseFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchDeviceDescription& description,
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::BatchedDenseRunOutput& output,
    const std::string& name) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::size_t elements =
      static_cast<std::size_t>(graph.vertex_count()) *
      description.lane_width;

  expect(
      output.result.engine_kind == static_cast<std::uint32_t>(
                                       bfnew::EngineKind::dense_chaotic_push) &&
          output.result.control_mode ==
              static_cast<std::uint32_t>(options.control_mode),
      name,
      "result lost the selected engine or control identity");
  expect(
      bfnew::validate_device_run_status(output.result.status) ==
              bfnew::DeviceRunStatusError::none &&
          output.result.status.stop_reason ==
              static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::converged) &&
          output.result.status.converged == 1U &&
          output.result.status.final_distance_slot == 0U &&
          output.result.status.valid_lane_mask ==
              description.valid_lane_mask &&
          output.result.status.active_lane_mask == 0U &&
          output.result.status.converged_lane_mask ==
              description.valid_lane_mask &&
          output.result.status.rounds_completed != 0U &&
          output.result.status.rounds_completed <= options.maximum_rounds,
      name,
      "terminal status does not describe normal one-slot convergence");
  expect(
      output.distances_downloaded && output.distances.size() == elements &&
          output.distance_bits.size() == elements,
      name,
      "distance download is not exact V-by-W atomic storage");
  expect(
      output.lane_convergence_rounds.size() == description.lane_width &&
          output.lane_executed_rounds.size() == description.lane_width &&
          output.lane_tail_rounds.size() == description.lane_width,
      name,
      "per-lane convergence records do not have width entries");

  for (std::size_t element = 0U; element < elements; ++element) {
    expect(
        std::bit_cast<std::uint32_t>(output.distances[element]) ==
            output.distance_bits[element],
        name,
        "float projection differs from the downloaded atomic word");
  }

  bfnew::LaneMask reached = 0U;
  bfnew::LaneMask missed = 0U;
  std::uint64_t active_lane_rounds = 0U;
  std::uint64_t tail_lane_rounds = 0U;
  std::uint64_t maximum_convergence_round = 0U;
  for (std::size_t lane = 0U; lane < description.lane_width; ++lane) {
    const bfnew::LaneMask bit = bfnew::LaneMask{1U} << lane;
    if ((description.valid_lane_mask & bit) == 0U) {
      expect(
          output.lane_convergence_rounds[lane] == 0U &&
              output.lane_executed_rounds[lane] == 0U &&
              output.lane_tail_rounds[lane] == 0U &&
              description.source_offsets[lane] ==
                  description.source_offsets[lane + 1U] &&
              description.target_offsets[lane] ==
                  description.target_offsets[lane + 1U],
          name,
          "padded lane retained terminals or performed semantic scan work");
      for (const bfnew::BatchVertexRange range :
           description.selected_vertex_ranges) {
        expect(
            (range.lane_mask & bit) == 0U,
            name,
            "padded lane entered a selected vertex range");
      }
      for (const bfnew::LaneMask run_mask :
           description.csr_run_lane_masks) {
        expect(
            (run_mask & bit) == 0U,
            name,
            "padded lane entered a retained CSR run mask");
      }
      for (std::size_t vertex = 0U; vertex < graph.vertex_count(); ++vertex) {
        const std::size_t index =
            distance_index(vertex, lane, description.lane_width);
        expect(
            output.distance_bits[index] == positive_infinity_bits &&
                std::isinf(output.distances[index]),
            name,
            "padded lane published a source, update, or stale scratch word");
      }
      continue;
    }

    const bfnew::RouteQuery& query = lane_query(queries, description, lane);
    const std::vector<float> oracle = bounded_oracle(graph, query);
    for (std::size_t vertex = 0U; vertex < graph.vertex_count(); ++vertex) {
      const std::size_t index =
          distance_index(vertex, lane, description.lane_width);
      if (selected_vertex(graph, query, vertex)) {
        expect(
            bits_equal(output.distances[index], oracle[vertex]) &&
                output.distance_bits[index] ==
                    std::bit_cast<std::uint32_t>(oracle[vertex]),
            name,
            "one lane differs from its independent bounded Dijkstra oracle");
      } else {
        expect(
            output.distance_bits[index] == positive_infinity_bits &&
                std::isinf(output.distances[index]),
            name,
            "nonselected output word was not normalized to positive infinity");
      }
    }
    for (const bfnew::VertexId source : query.sources) {
      const std::size_t index =
          distance_index(source.value(), lane, description.lane_width);
      expect(
          output.distance_bits[index] == 0U,
          name,
          "lane did not retain its own canonical positive-zero source");
    }

    bool all_targets = true;
    for (const bfnew::VertexId target : query.targets) {
      all_targets = all_targets && std::isfinite(oracle[target.value()]);
    }
    if (all_targets) {
      reached |= bit;
    } else {
      missed |= bit;
    }

    const std::uint64_t convergence =
        output.lane_convergence_rounds[lane];
    expect(
        convergence != 0U &&
            convergence <= output.result.status.rounds_completed,
        name,
        "valid lane omitted its one-based proven no-change scan");
    maximum_convergence_round =
        std::max(maximum_convergence_round, convergence);
    const std::uint64_t expected_executed =
        options.enable_per_lane_convergence != 0U
            ? convergence
            : output.result.status.rounds_completed;
    expect(
        output.lane_executed_rounds[lane] == expected_executed &&
            output.lane_tail_rounds[lane] ==
                output.result.status.rounds_completed - convergence,
        name,
        "per-lane freeze or tail accounting is inconsistent");
    active_lane_rounds += output.lane_executed_rounds[lane];
    tail_lane_rounds += output.lane_tail_rounds[lane];
  }

  expect(
      maximum_convergence_round == output.result.status.rounds_completed,
      name,
      "batch duration is not its longest proven convergence scan");
  expect(
      output.result.status.reached_target_mask == reached &&
          output.result.status.bounding_box_miss_mask == missed &&
          (reached | missed) == description.valid_lane_mask,
      name,
      "device reached/miss masks disagree with bounded Dijkstra");

  const bfnew::BatchedDenseWorkStatistics& dense = output.batch_work;
  expect(
      dense.csr_runs_considered ==
              dense.csr_runs_visited + dense.csr_runs_skipped &&
          dense.csr_runs_visited != 0U &&
          dense.csr_runs_skipped != 0U &&
          dense.active_lanes_over_visited_runs >= dense.csr_runs_visited &&
          dense.csr_edge_loads != 0U &&
          dense.csr_edge_loads <= dense.lane_edge_relaxations &&
          dense.lane_edge_relaxations == dense.atomic_source_loads &&
          dense.atomic_source_loads == dense.atomic_min_attempts &&
          dense.successful_atomic_updates <= dense.atomic_min_attempts,
      name,
      "shared CSR loads or per-lane atomic accounting is inconsistent");
  expect(
      dense.full_edge_rounds == output.result.status.rounds_completed &&
          dense.active_lane_rounds == active_lane_rounds &&
          dense.tail_lane_rounds == tail_lane_rounds &&
          dense.valid_lane_round_capacity ==
              output.result.status.rounds_completed *
                  static_cast<std::uint64_t>(
                      std::popcount(description.valid_lane_mask)) &&
          dense.lane_width_round_capacity ==
              output.result.status.rounds_completed *
                  description.lane_width &&
          dense.wave32_lane_round_capacity ==
              output.result.status.rounds_completed *
                  bfnew::maximum_batch_lanes &&
          dense.wave32_lane_round_capacity ==
              dense.lane_width_round_capacity +
                  dense.unused_wave_lane_round_capacity &&
          dense.valid_lane_round_capacity ==
              dense.active_lane_rounds +
                  dense.inactive_valid_lane_rounds &&
          dense.lane_width_round_capacity ==
              dense.valid_lane_round_capacity +
                  dense.padded_lane_round_capacity,
      name,
      "valid/configured/wave32 lane-round capacities do not balance");
  expect(
      dense.edge_wave_lane_capacity ==
              dense.csr_edge_loads * bfnew::maximum_batch_lanes &&
          dense.edge_wave_lane_capacity ==
              dense.lane_edge_relaxations +
                  dense.unused_edge_wave_lane_capacity &&
          dense.padded_lane_semantic_work == 0U &&
          dense.frontier_semantic_work == 0U,
      name,
      "edge-wave capacity contains padding or frontier semantic work");
  if (options.enable_per_lane_convergence != 0U) {
    expect(
        dense.tail_lane_rounds_avoided == dense.tail_lane_rounds &&
            dense.tail_lane_rounds_executed == 0U,
        name,
        "enabled convergence did not classify every tail scan as avoided");
  } else {
    expect(
        dense.tail_lane_rounds_avoided == 0U &&
            dense.tail_lane_rounds_executed == dense.tail_lane_rounds &&
            dense.lane_edge_relaxations_avoided_by_early_convergence == 0U,
        name,
        "disabled convergence reported avoided scan or edge work");
  }

  const std::uint64_t valid_lanes = static_cast<std::uint64_t>(
      std::popcount(description.valid_lane_mask));
  if (valid_lanes > 1U) {
    expect(
        dense.csr_edge_loads < dense.lane_edge_relaxations &&
            dense.active_lanes_over_visited_runs > dense.csr_runs_visited,
        name,
        "overlapping batch did not expose strict CSR edge/run lane reuse");
  }
  std::uint64_t selected_tile_lane_positions = 0U;
  for (const std::uint32_t tile : description.union_tiles) {
    selected_tile_lane_positions += static_cast<std::uint64_t>(
        std::popcount(description.tile_lane_masks[tile]));
  }
  expect(
      dense.distance_reset_bytes ==
              selected_lane_vertex_count(description) *
                  sizeof(std::uint32_t) &&
          dense.source_seed_write_bytes ==
              description.sources.size() * sizeof(std::uint32_t) &&
          dense.union_tile_lane_positions ==
              description.union_tiles.size() * valid_lanes &&
          dense.selected_tile_lane_positions ==
              selected_tile_lane_positions &&
          dense.union_tile_lane_positions >=
              dense.selected_tile_lane_positions,
      name,
      "selected-only reset, seed, or tile-union accounting is wrong");
  expect(
      output.metrics.kernel_registers_per_thread != 0U &&
          output.metrics.lane_width == description.lane_width &&
          output.metrics.valid_lane_count == valid_lanes &&
          output.metrics.unused_wave_lane_rounds ==
              dense.unused_wave_lane_round_capacity &&
          output.metrics.edge_record_read_bytes_requested ==
              dense.csr_edge_loads *
                  (sizeof(std::uint32_t) + sizeof(float)) &&
          output.metrics.atomic_source_read_bytes_requested ==
              dense.lane_edge_relaxations * sizeof(std::uint32_t) &&
          output.metrics.atomic_destination_access_bytes_requested ==
              dense.lane_edge_relaxations * sizeof(std::uint32_t) &&
          output.metrics.average_active_lanes_per_nonzero_run > 0.0 &&
          output.metrics.lane_round_utilization > 0.0 &&
          output.metrics.lane_round_utilization <= 1.0,
      name,
      "selected-kernel or requested-byte metrics are inconsistent");
  expect(
      output.metrics.hardware_counters.available == 0U &&
          output.metrics.hardware_counters.l2_read_bytes == 0U &&
          output.metrics.hardware_counters.l2_write_bytes == 0U &&
          output.metrics.hardware_counters.atomic_stall_cycles == 0U &&
          output.metrics.hardware_counters.write_stall_cycles == 0U &&
          output.metrics.hardware_counters.waves == 0U,
      name,
      "unavailable profiler counters were synthesized");

  expect_control_accounting(options, output, name);
  expect_instrumentation_gating(options, output, name);
}

void expect_mixed_duration_evidence(
    const bfnew::BatchDeviceDescription& description,
    const bfnew::hip::BatchedDenseRunOutput& output,
    const std::string_view name) {
  const auto lane_for = [&](const bfnew::QueryId query_id) {
    for (std::size_t lane = 0U; lane < description.lane_width; ++lane) {
      if ((description.valid_lane_mask & (bfnew::LaneMask{1U} << lane)) !=
              0U &&
          description.query_ids_by_lane[lane] == query_id.value()) {
        return lane;
      }
    }
    throw std::logic_error{"mixed-duration query lane is missing"};
  };

  const std::size_t no_change_lane = lane_for(bfnew::QueryId{1400U});
  const std::size_t unreachable_lane = lane_for(bfnew::QueryId{1403U});
  expect(
      output.lane_convergence_rounds[no_change_lane] == 1U &&
          output.lane_convergence_rounds[unreachable_lane] == 1U,
      name,
      "no-first-decrease or unreachable lane missed its first no-change scan");

  bool has_longer_lane = false;
  for (std::size_t lane = 0U; lane < description.lane_width; ++lane) {
    if ((description.valid_lane_mask & (bfnew::LaneMask{1U} << lane)) != 0U) {
      has_longer_lane = has_longer_lane ||
                        output.lane_convergence_rounds[lane] > 1U;
    }
  }
  expect(
      has_longer_lane && output.result.status.rounds_completed > 1U &&
          output.lane_tail_rounds[no_change_lane] > 0U &&
          output.lane_tail_rounds[unreachable_lane] > 0U,
      name,
      "mixed fixture did not retain a longer lane and early-lane tail");
}

void expect_cross_lane_isolation(
    const bfnew::test::BatchedDenseFixture& fixture,
    const bfnew::BatchDeviceDescription& description,
    const bfnew::hip::BatchedDenseRunOutput& output,
    const std::string_view name) {
  const auto lane_for = [&](const bfnew::QueryId query_id) {
    for (std::size_t lane = 0U; lane < description.lane_width; ++lane) {
      if ((description.valid_lane_mask & (bfnew::LaneMask{1U} << lane)) !=
              0U &&
          description.query_ids_by_lane[lane] == query_id.value()) {
        return lane;
      }
    }
    throw std::logic_error{"cross-lane query lane is missing"};
  };
  const std::size_t unreachable_lane = lane_for(bfnew::QueryId{1403U});
  const std::size_t reaching_lane = lane_for(bfnew::QueryId{1402U});
  const std::uint32_t shared_vertex =
      lane_query(fixture.queries, description, unreachable_lane)
          .targets.front()
          .value();
  const std::size_t unreachable_index = distance_index(
      shared_vertex, unreachable_lane, description.lane_width);
  const std::size_t reaching_index =
      distance_index(shared_vertex, reaching_lane, description.lane_width);
  expect(
      std::isinf(output.distances[unreachable_index]) &&
          std::isfinite(output.distances[reaching_index]),
      name,
      "one query lane seeded or updated another lane's atomic word");
}

void validate_standalone_output(
    const bfnew::test::BatchedDenseFixture& fixture,
    const bfnew::RouteQuery& query,
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::DenseRunOutput& output) {
  expect(
      output.distances_downloaded &&
          output.distances.size() == fixture.partitioned.graph.vertex_count(),
      "width-one-standalone",
      "standalone dense did not download one word per vertex");
  expect(
      output.result.engine_kind == static_cast<std::uint32_t>(
                                       bfnew::EngineKind::dense_chaotic_push) &&
          output.result.control_mode ==
              static_cast<std::uint32_t>(options.control_mode) &&
          bfnew::validate_device_run_status(output.result.status) ==
              bfnew::DeviceRunStatusError::none &&
          output.result.status.stop_reason ==
              static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::converged) &&
          output.result.status.final_distance_slot == 0U,
      "width-one-standalone",
      "standalone dense terminal identity or status is invalid");
  const std::vector<float> oracle =
      bounded_oracle(fixture.partitioned.graph, query);
  for (std::size_t vertex = 0U;
       vertex < fixture.partitioned.graph.vertex_count();
       ++vertex) {
    if (selected_vertex(fixture.partitioned.graph, query, vertex)) {
      expect(
          bits_equal(output.distances[vertex], oracle[vertex]),
          "width-one-standalone",
          "standalone dense differs from bounded Dijkstra");
    }
  }
}

void test_mixed_width_control_matrix() {
  const bfnew::test::BatchedDenseFixture fixture =
      bfnew::test::make_mixed_duration_batched_dense_fixture();
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedDenseWorkspace workspace;
  bfnew::hip::BatchedDenseChaoticPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};

  for (const std::uint32_t width : {8U, 16U, 32U}) {
    const bfnew::test::PreparedBatchedDenseFixture prepared =
        bfnew::test::prepare_batched_dense_fixture(
            fixture,
            fixture.queries,
            width,
            bfnew::BatchRunRepresentation::retained_per_run_masks);
    expect(
        prepared.description.valid_lane_mask == 0x1fU &&
            prepared.description.lane_width == width,
        "mixed-width",
        "fixture does not expose five low-prefix valid lanes plus padding");

    std::vector<std::uint32_t> width_baseline;
    for (const bfnew::GpuRunOptions& enabled_control :
         control_matrix(true)) {
      std::vector<std::uint32_t> toggle_baseline;
      for (const bool per_lane : {true, false}) {
        bfnew::GpuRunOptions options = enabled_control;
        options.enable_per_lane_convergence = per_lane ? 1U : 0U;
        std::vector<std::uint32_t> strategy_baseline;
        for (const bfnew::hip::BatchedDenseLoadStrategy strategy : {
                 bfnew::hip::BatchedDenseLoadStrategy::compiler_uniform,
                 bfnew::hip::BatchedDenseLoadStrategy::
                     explicit_wave_broadcast}) {
          const bfnew::hip::BatchedDenseRunOutput output =
              engine.run_with_distances(
                  prepared.description, options, strategy);
          const std::string name =
              "width-" + std::to_string(width) + "-control-" +
              std::to_string(
                  static_cast<std::uint32_t>(options.control_mode)) +
              "-K-" + std::to_string(options.rounds_per_chunk) +
              (per_lane ? "-enabled" : "-disabled") + "-strategy-" +
              std::to_string(static_cast<std::uint32_t>(strategy));
          validate_output(
              fixture,
              fixture.queries,
              prepared.description,
              options,
              output,
              name);
          expect_mixed_duration_evidence(
              prepared.description, output, name);
          expect_cross_lane_isolation(
              fixture, prepared.description, output, name);
          expect(
              output.metrics.load_strategy == strategy,
              name,
              "result did not retain the selected load strategy");
          if (per_lane) {
            expect(
                output.batch_work.tail_lane_rounds_avoided != 0U &&
                    output.batch_work
                            .lane_edge_relaxations_avoided_by_early_convergence !=
                        0U,
                name,
                "enabled convergence exposed no mixed-duration work saving");
          }

          if (strategy_baseline.empty()) {
            strategy_baseline = output.distance_bits;
          } else {
            expect(
                strategy_baseline == output.distance_bits,
                name,
                "explicit wave broadcast changed a final atomic word");
          }
          if (toggle_baseline.empty()) {
            toggle_baseline = output.distance_bits;
          } else {
            expect(
                toggle_baseline == output.distance_bits,
                name,
                "per-lane convergence toggle changed a final atomic word");
          }
          if (width_baseline.empty()) {
            width_baseline = output.distance_bits;
          } else {
            expect(
                width_baseline == output.distance_bits,
                name,
                "control mode changed a final atomic word");
          }
        }
      }
    }
  }
}

void test_width_one_matches_standalone() {
  const bfnew::test::BatchedDenseFixture fixture =
      bfnew::test::make_mixed_duration_batched_dense_fixture();
  const std::array<bfnew::RouteQuery, 1U> singleton{fixture.queries[2U]};
  const bfnew::test::PreparedBatchedDenseFixture prepared =
      bfnew::test::prepare_batched_dense_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedDenseWorkspace batch_workspace;
  bfnew::hip::ReusableDeviceWorkspace standalone_workspace;
  bfnew::hip::BatchedDenseChaoticPushEngine batched{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      batch_workspace,
      stream};
  bfnew::hip::DenseChaoticPushEngine standalone{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      standalone_workspace,
      stream};

  std::vector<std::uint32_t> control_baseline;
  for (const bfnew::GpuRunOptions& enabled_control : control_matrix(true)) {
    for (const bool per_lane : {true, false}) {
      bfnew::GpuRunOptions options = enabled_control;
      options.enable_per_lane_convergence = per_lane ? 1U : 0U;
      const bfnew::hip::DenseRunOutput standalone_output =
          standalone.run_with_distances(singleton.front(), options);
      validate_standalone_output(
          fixture, singleton.front(), options, standalone_output);

      std::vector<std::uint32_t> strategy_baseline;
      for (const bfnew::hip::BatchedDenseLoadStrategy strategy : {
               bfnew::hip::BatchedDenseLoadStrategy::compiler_uniform,
               bfnew::hip::BatchedDenseLoadStrategy::
                   explicit_wave_broadcast}) {
        const bfnew::hip::BatchedDenseRunOutput batch_output =
            batched.run_with_distances(
                prepared.description, options, strategy);
        validate_output(
            fixture,
            singleton,
            prepared.description,
            options,
            batch_output,
            "width-one");
        expect(
            batch_output.result.status.reached_target_mask ==
                    standalone_output.result.status.reached_target_mask &&
                batch_output.result.status.bounding_box_miss_mask ==
                    standalone_output.result.status.bounding_box_miss_mask,
            "width-one",
            "batched and standalone dense disagree on terminal status");
        for (std::size_t vertex = 0U;
             vertex < fixture.partitioned.graph.vertex_count();
             ++vertex) {
          if (!selected_vertex(
                  fixture.partitioned.graph, singleton.front(), vertex)) {
            continue;
          }
          expect(
              batch_output.distance_bits[vertex] ==
                  std::bit_cast<std::uint32_t>(
                      standalone_output.distances[vertex]),
              "width-one",
              "width one differs bitwise from standalone HIP dense push");
        }
        if (strategy_baseline.empty()) {
          strategy_baseline = batch_output.distance_bits;
        } else {
          expect(
              strategy_baseline == batch_output.distance_bits,
              "width-one",
              "width-one load strategy changed final atomic words");
        }
        if (control_baseline.empty()) {
          control_baseline = batch_output.distance_bits;
        } else {
          expect(
              control_baseline == batch_output.distance_bits,
              "width-one",
              "width-one control or convergence setting changed final bits");
        }
      }
    }
  }
}

void test_instrumentation_levels() {
  const bfnew::test::BatchedDenseFixture fixture =
      bfnew::test::make_mixed_duration_batched_dense_fixture();
  const bfnew::test::PreparedBatchedDenseFixture prepared =
      bfnew::test::prepare_batched_dense_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedDenseWorkspace workspace;
  bfnew::hip::BatchedDenseChaoticPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};

  std::vector<std::uint32_t> baseline;
  for (const bfnew::hip::BatchedDenseLoadStrategy strategy : {
           bfnew::hip::BatchedDenseLoadStrategy::compiler_uniform,
           bfnew::hip::BatchedDenseLoadStrategy::explicit_wave_broadcast}) {
    for (const bfnew::InstrumentationLevel instrumentation : {
             bfnew::InstrumentationLevel::none,
             bfnew::InstrumentationLevel::light,
             bfnew::InstrumentationLevel::debug}) {
      const bfnew::GpuRunOptions options = options_for(
          bfnew::ControlMode::per_round_host_poll,
          true,
          instrumentation);
      const bfnew::hip::BatchedDenseRunOutput output =
          engine.run_with_distances(
              prepared.description, options, strategy);
      const std::string name =
          "instrumentation-" +
          std::to_string(static_cast<std::uint32_t>(instrumentation)) +
          "-strategy-" +
          std::to_string(static_cast<std::uint32_t>(strategy));
      validate_output(
          fixture,
          fixture.queries,
          prepared.description,
          options,
          output,
          name);
      expect_mixed_duration_evidence(prepared.description, output, name);
      if (baseline.empty()) {
        baseline = output.distance_bits;
      } else {
        expect(
            baseline == output.distance_bits,
            name,
            "None/Light/Debug or load strategy changed final distance bits");
      }
    }
  }
}

void test_maximum_rounds_and_guards() {
  const bfnew::test::BatchedDenseFixture fixture =
      bfnew::test::make_mixed_duration_batched_dense_fixture();
  const std::array<bfnew::RouteQuery, 1U> singleton{fixture.queries[2U]};
  const bfnew::test::PreparedBatchedDenseFixture prepared =
      bfnew::test::prepare_batched_dense_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedDenseWorkspace workspace;
  bfnew::hip::BatchedDenseChaoticPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};

  for (const bool per_lane : {true, false}) {
    for (const bfnew::GpuRunOptions& options :
         control_matrix(per_lane, 1U)) {
      for (const bfnew::hip::BatchedDenseLoadStrategy strategy : {
               bfnew::hip::BatchedDenseLoadStrategy::compiler_uniform,
               bfnew::hip::BatchedDenseLoadStrategy::
                   explicit_wave_broadcast}) {
        const bfnew::hip::BatchedDenseRunOutput exhausted =
            engine.run_status_only(
                prepared.description, options, strategy);
        const std::string name =
            "maximum-rounds-control-" +
            std::to_string(
                static_cast<std::uint32_t>(options.control_mode)) +
            "-K-" + std::to_string(options.rounds_per_chunk) +
            (per_lane ? "-enabled" : "-disabled") + "-strategy-" +
            std::to_string(static_cast<std::uint32_t>(strategy));
        expect(
            !exhausted.distances_downloaded &&
                exhausted.distances.empty() &&
                exhausted.distance_bits.empty(),
            name,
            "status-only maximum-round run downloaded distance words");
        expect(
            bfnew::validate_device_run_status(exhausted.result.status) ==
                    bfnew::DeviceRunStatusError::none &&
                exhausted.result.status.stop_reason ==
                    static_cast<std::uint32_t>(
                        bfnew::DeviceStopReason::maximum_rounds) &&
                exhausted.result.status.converged == 0U &&
                exhausted.result.status.rounds_completed == 1U &&
                exhausted.result.status.final_distance_slot == 0U &&
                exhausted.result.status.valid_lane_mask == 1U &&
                exhausted.result.status.active_lane_mask == 1U &&
                exhausted.result.status.converged_lane_mask == 0U &&
                exhausted.result.status.reached_target_mask == 0U &&
                exhausted.result.status.bounding_box_miss_mask == 0U &&
                exhausted.result.status.error_bits == bfnew::device_error::none,
            name,
            "round exhaustion was misreported as convergence, miss, or error");
        expect(
            exhausted.lane_convergence_rounds ==
                    std::vector<std::uint64_t>{0U} &&
                exhausted.lane_executed_rounds ==
                    std::vector<std::uint64_t>{1U} &&
                exhausted.lane_tail_rounds ==
                    std::vector<std::uint64_t>{0U},
            name,
            "unfinished lane published invalid convergence evidence");
        expect(
            exhausted.metrics.load_strategy == strategy &&
                exhausted.result.engine_kind ==
                    static_cast<std::uint32_t>(
                        bfnew::EngineKind::dense_chaotic_push) &&
                exhausted.result.control_mode ==
                    static_cast<std::uint32_t>(options.control_mode),
            name,
            "maximum-round output lost its execution identity");
        expect_control_accounting(options, exhausted, name);
        expect_instrumentation_gating(options, exhausted, name);
      }
    }
  }

  bfnew::GpuRunOptions limited = options_for(
      bfnew::ControlMode::persistent_cooperative, true);
  limited.maximum_rounds = 1U;

  const bfnew::test::PreparedBatchedDenseFixture compact =
      bfnew::test::prepare_batched_dense_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::compact_nonzero_descriptors);
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            engine.run_status_only(compact.description, limited));
      },
      "guards",
      "HIP batched dense accepted compact descriptors instead of retained masks");

  bfnew::BatchDeviceDescription corrupt_mask = prepared.description;
  const std::uint32_t touched_run = corrupt_mask.touched_csr_runs.front();
  corrupt_mask.csr_run_lane_masks[touched_run] = 0U;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(engine.run_status_only(corrupt_mask, limited));
      },
      "guards",
      "HIP batched dense accepted a non-endpoint-exact retained CSR mask");

  bfnew::GpuRunOptions wrong_engine = limited;
  wrong_engine.engine = bfnew::EngineKind::jacobi_pull;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            engine.run_status_only(prepared.description, wrong_engine));
      },
      "guards",
      "HIP batched dense accepted another engine identity");

  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(engine.run_status_only(
            prepared.description,
            limited,
            static_cast<bfnew::hip::BatchedDenseLoadStrategy>(255U)));
      },
      "guards",
      "HIP batched dense accepted an unknown load strategy");

  bfnew::GpuRunOptions invalid_block = limited;
  invalid_block.block_size = 127U;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            engine.run_status_only(prepared.description, invalid_block));
      },
      "guards",
      "wave-per-row engine accepted a non-wave32 block size");
}

}  // namespace

int main() {
  expect(
      bfnew::supported_batched_dense_width(1U) &&
          bfnew::supported_batched_dense_width(8U) &&
          bfnew::supported_batched_dense_width(16U) &&
          bfnew::supported_batched_dense_width(32U) &&
          !bfnew::supported_batched_dense_width(2U) &&
          !bfnew::supported_batched_dense_width(64U),
      "supported-widths",
      "device API does not expose exactly widths 1, 8, 16, and 32");
  test_mixed_width_control_matrix();
  test_width_one_matches_standalone();
  test_instrumentation_levels();
  test_maximum_rounds_and_guards();
  std::cout << "batched dense chaotic push HIP tests passed\n";
  return 0;
}
