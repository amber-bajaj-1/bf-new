#include "bfnew/hip/batched_jacobi.hpp"

#include "bfnew/hip/jacobi.hpp"
#include "bfnew/sssp.hpp"
#include "batched_jacobi_fixture_suite.hpp"

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
#include <vector>

namespace {

[[noreturn]] void fail(
    const std::string_view fixture,
    const std::string_view message) {
  std::cerr << "batched Jacobi HIP test failed [" << fixture
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
  throw std::logic_error{"HIP batch lane query is missing"};
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
  for (std::size_t local = 0U; local < induced.local_to_global.size(); ++local) {
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
    const bool per_lane) {
  bfnew::GpuRunOptions options;
  options.engine = bfnew::EngineKind::jacobi_pull;
  options.control_mode = control;
  options.rounds_per_chunk = 3U;
  options.block_size = 128U;
  options.instrumentation = bfnew::InstrumentationLevel::debug;
  options.maximum_rounds = 64U;
  options.enable_per_lane_convergence = per_lane ? 1U : 0U;
  return options;
}

[[nodiscard]] std::vector<bfnew::GpuRunOptions> control_matrix(
    const bool per_lane,
    const std::uint64_t maximum_rounds = 64U) {
  std::vector<bfnew::GpuRunOptions> controls;
  bfnew::GpuRunOptions persistent = options_for(
      bfnew::ControlMode::persistent_cooperative, per_lane);
  persistent.maximum_rounds = maximum_rounds;
  controls.push_back(persistent);
  bfnew::GpuRunOptions per_round = options_for(
      bfnew::ControlMode::per_round_host_poll, per_lane);
  per_round.maximum_rounds = maximum_rounds;
  controls.push_back(per_round);
  for (const std::uint32_t rounds_per_chunk :
       std::array<std::uint32_t, 5U>{2U, 4U, 8U, 16U, 32U}) {
    bfnew::GpuRunOptions chunked = options_for(
        bfnew::ControlMode::chunked_host_poll, per_lane);
    chunked.maximum_rounds = maximum_rounds;
    chunked.rounds_per_chunk = rounds_per_chunk;
    controls.push_back(chunked);
  }
  return controls;
}

void validate_output(
    const bfnew::test::BatchedJacobiFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchDeviceDescription& description,
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::BatchedJacobiRunOutput& output,
    const std::string& name) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  expect(
      bfnew::validate_device_run_status(output.result.status) ==
              bfnew::DeviceRunStatusError::none &&
          output.result.status.stop_reason ==
              static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::converged) &&
          output.result.status.valid_lane_mask ==
              description.valid_lane_mask &&
          output.result.status.converged_lane_mask ==
              description.valid_lane_mask,
      name,
      "terminal status does not describe normal all-lane convergence");
  expect(
      output.distances_downloaded &&
          output.distances.size() ==
              static_cast<std::size_t>(graph.vertex_count()) *
                  description.lane_width &&
          output.converged_slots_bitwise_identical_mask ==
              description.valid_lane_mask,
      name,
      "distance shape or two-column equality invariant is wrong");
  expect(
      output.lane_convergence_rounds.size() == description.lane_width &&
          output.lane_executed_rounds.size() == description.lane_width &&
          output.lane_tail_rounds.size() == description.lane_width,
      name,
      "per-lane convergence records do not have width entries");

  bfnew::LaneMask reached = 0U;
  bfnew::LaneMask missed = 0U;
  for (std::size_t lane = 0U; lane < description.lane_width; ++lane) {
    const bfnew::LaneMask bit = bfnew::LaneMask{1U} << lane;
    if ((description.valid_lane_mask & bit) == 0U) {
      expect(
          output.lane_convergence_rounds[lane] == 0U &&
              output.lane_executed_rounds[lane] == 0U &&
              output.lane_tail_rounds[lane] == 0U,
          name,
          "padded lane performed semantic work");
      continue;
    }
    const bfnew::RouteQuery& query = lane_query(queries, description, lane);
    const std::vector<float> oracle = bounded_oracle(graph, query);
    for (std::size_t vertex = 0U; vertex < graph.vertex_count(); ++vertex) {
      if (!selected_vertex(graph, query, vertex)) {
        continue;
      }
      const std::size_t index = vertex * description.lane_width + lane;
      expect(
          bits_equal(output.distances[index], oracle[vertex]),
          name,
          "one lane differs from its independent bounded Dijkstra oracle");
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
    expect(
        output.lane_convergence_rounds[lane] != 0U &&
            output.lane_convergence_rounds[lane] <=
                output.result.status.rounds_completed,
        name,
        "valid lane omitted its proven no-change round");
    if (options.enable_per_lane_convergence != 0U) {
      expect(
          output.lane_executed_rounds[lane] ==
              output.lane_convergence_rounds[lane],
          name,
          "enabled convergence did not freeze the lane immediately");
    } else {
      expect(
          output.lane_executed_rounds[lane] ==
              output.result.status.rounds_completed,
          name,
          "disabled convergence removed a lane early");
    }
  }
  expect(
      output.result.status.reached_target_mask == reached &&
          output.result.status.bounding_box_miss_mask == missed,
      name,
      "GPU reached/miss masks disagree with bounded Dijkstra");
  expect(
      output.metrics.cooperative_grid_blocks != 0U ||
          output.metrics.ordinary_active_blocks_per_wgp != 0U,
      name,
      "actual selected-kernel occupancy was not recorded");
  expect(
      output.metrics.kernel_registers_per_thread != 0U &&
          output.metrics.hardware_counters.available == 0U,
      name,
      "register evidence or explicitly unavailable profiler counters are wrong");
  expect(
      output.metrics.admitted_lane_edge_pairs >=
              output.metrics.csc_edge_records_loaded &&
          output.metrics.csc_runs_considered ==
              output.metrics.csc_runs_visited +
                  output.metrics.csc_runs_skipped &&
          output.metrics.csc_runs_considered != 0U &&
          output.metrics.csc_runs_visited != 0U &&
          output.metrics.padded_distance_reset_bytes == 0U &&
          output.metrics.source_seed_write_bytes <=
              output.metrics.distance_reset_bytes,
      name,
      "edge reuse or selected-only reset accounting is invalid");
  expect(
      output.result.work.host_checks ==
              output.metrics.convergence_host_checks &&
          output.result.work.controller_copies ==
              output.metrics.convergence_host_checks + 1U &&
          output.result.work.host_synchronizations ==
              output.metrics.convergence_host_checks + 1U,
      name,
      "physical host-control accounting is inconsistent");

  if (options.control_mode == bfnew::ControlMode::persistent_cooperative) {
    expect(
        output.metrics.convergence_host_checks == 0U &&
            output.metrics.engine_round_dispatches == 0U &&
            output.metrics.controller_advance_dispatches == 0U &&
            output.result.work.kernel_dispatches == 1U,
        name,
        "persistent execution polled the host or launched round pairs");
  } else {
    expect(
        output.metrics.engine_round_dispatches ==
                output.metrics.controller_advance_dispatches &&
            output.metrics.convergence_host_checks != 0U,
        name,
        "ordinary execution violated round/advance host-poll control");
  }
}

void test_mixed_width_control_matrix() {
  const bfnew::test::BatchedJacobiFixture fixture =
      bfnew::test::make_mixed_duration_batched_jacobi_fixture();
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedJacobiWorkspace workspace;
  bfnew::hip::BatchedJacobiPullEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};

  for (const std::uint32_t width : {8U, 16U, 32U}) {
    const bfnew::test::PreparedBatchedJacobiFixture prepared =
        bfnew::test::prepare_batched_fixture(
            fixture,
            fixture.queries,
            width,
            bfnew::BatchRunRepresentation::device_materialized_run_masks);
    std::vector<float> first_control;
    for (const bfnew::GpuRunOptions& enabled_control :
         control_matrix(true)) {
      std::vector<float> toggle_baseline;
      std::uint64_t enabled_lane_edges = 0U;
      std::uint64_t enabled_avoided_lane_edges = 0U;
      std::uint64_t enabled_avoided_lane_rounds = 0U;
      for (const bool per_lane : {true, false}) {
        bfnew::GpuRunOptions options = enabled_control;
        options.enable_per_lane_convergence = per_lane ? 1U : 0U;
        const bfnew::hip::BatchedJacobiRunOutput uniform =
            engine.run_with_distances(
                prepared.description,
                options,
                bfnew::hip::BatchedJacobiLoadStrategy::compiler_uniform);
        const std::string name =
            "width-" + std::to_string(width) + "-control-" +
            std::to_string(
                static_cast<std::uint32_t>(options.control_mode)) +
            "-K-" + std::to_string(options.rounds_per_chunk) +
            (per_lane ? "-enabled" : "-disabled");
        validate_output(
            fixture,
            fixture.queries,
            prepared.description,
            options,
            uniform,
            name + "-uniform");
        const bfnew::hip::BatchedJacobiRunOutput broadcast =
            engine.run_with_distances(
                prepared.description,
                options,
                bfnew::hip::BatchedJacobiLoadStrategy::
                    explicit_wave_broadcast);
        validate_output(
            fixture,
            fixture.queries,
            prepared.description,
            options,
            broadcast,
            name + "-broadcast");
        expect(
            uniform.distances == broadcast.distances &&
                uniform.lane_convergence_rounds ==
                    broadcast.lane_convergence_rounds &&
                uniform.result.status.reached_target_mask ==
                    broadcast.result.status.reached_target_mask &&
                uniform.result.status.bounding_box_miss_mask ==
                    broadcast.result.status.bounding_box_miss_mask,
            name,
            "explicit broadcast experiment changed semantic evidence");

        const std::array<std::pair<bfnew::QueryId, std::uint64_t>, 5U>
            expected_convergence{{
                {bfnew::QueryId{1400U}, 1U},
                {bfnew::QueryId{1401U}, 2U},
                {bfnew::QueryId{1402U}, 6U},
                {bfnew::QueryId{1403U}, 1U},
                {bfnew::QueryId{1404U}, 4U},
            }};
        for (const auto& [query_id, expected_round] :
             expected_convergence) {
          const std::size_t lane = bfnew::test::lane_for_query(
              prepared.plan.batches.front(), query_id);
          expect(
              uniform.lane_convergence_rounds[lane] == expected_round &&
                  uniform.lane_tail_rounds[lane] ==
                      uniform.result.status.rounds_completed - expected_round,
              name,
              "mixed-duration lane has the wrong convergence/tail round");
        }
        if (toggle_baseline.empty()) {
          toggle_baseline = uniform.distances;
        } else {
          expect(
              toggle_baseline == uniform.distances,
              name,
              "convergence toggle changed a final distance bit");
        }
        if (first_control.empty()) {
          first_control = uniform.distances;
        } else {
          expect(
              first_control == uniform.distances,
              name,
              "control mode changed a final distance bit");
        }
        if (per_lane) {
          enabled_lane_edges = uniform.metrics.admitted_lane_edge_pairs;
          enabled_avoided_lane_edges =
              uniform.metrics
                  .lane_edge_relaxations_avoided_by_early_convergence;
          enabled_avoided_lane_rounds =
              uniform.metrics
                  .lane_rounds_avoided_by_early_convergence;
          expect(
              enabled_avoided_lane_edges != 0U &&
                  enabled_avoided_lane_rounds != 0U,
              name,
              "mixed batch did not expose early-convergence work savings");
        } else {
          expect(
              uniform.metrics.admitted_lane_edge_pairs >=
                      enabled_lane_edges &&
                  uniform.metrics.admitted_lane_edge_pairs -
                          enabled_lane_edges ==
                      enabled_avoided_lane_edges &&
                  uniform.metrics
                          .lane_edge_relaxations_avoided_by_early_convergence ==
                      0U &&
                  uniform.metrics
                          .lane_rounds_avoided_by_early_convergence ==
                      0U,
              name,
              "enabled/disabled work delta is not the exact avoided work");
        }
      }
    }
  }
}

void test_width_one_matches_standalone() {
  const bfnew::test::BatchedJacobiFixture fixture =
      bfnew::test::make_mixed_duration_batched_jacobi_fixture();
  const std::array<bfnew::RouteQuery, 1U> singleton{fixture.queries[2U]};
  const bfnew::test::PreparedBatchedJacobiFixture prepared =
      bfnew::test::prepare_batched_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::device_materialized_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedJacobiWorkspace batch_workspace;
  bfnew::hip::ReusableDeviceWorkspace standalone_workspace;
  bfnew::hip::BatchedJacobiPullEngine batched{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      batch_workspace,
      stream};
  bfnew::hip::JacobiPullEngine standalone{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      standalone_workspace,
      stream};
  for (const bfnew::GpuRunOptions& enabled_control : control_matrix(true)) {
    for (const bool per_lane : {true, false}) {
      bfnew::GpuRunOptions options = enabled_control;
      options.enable_per_lane_convergence = per_lane ? 1U : 0U;
      const bfnew::hip::JacobiRunOutput standalone_output =
          standalone.run_with_distances(singleton.front(), options);
      std::vector<float> strategy_baseline;
      for (const bfnew::hip::BatchedJacobiLoadStrategy strategy : {
               bfnew::hip::BatchedJacobiLoadStrategy::compiler_uniform,
               bfnew::hip::BatchedJacobiLoadStrategy::
                   explicit_wave_broadcast}) {
        const bfnew::hip::BatchedJacobiRunOutput batch_output =
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
            batch_output.result.status.rounds_completed ==
                standalone_output.result.status.rounds_completed,
            "width-one",
            "width one and standalone completed different round counts");
        for (std::size_t vertex = 0U;
             vertex < fixture.partitioned.graph.vertex_count();
             ++vertex) {
          if (selected_vertex(
                  fixture.partitioned.graph, singleton.front(), vertex)) {
            expect(
                bits_equal(batch_output.distances[vertex],
                           standalone_output.distances[vertex]),
                "width-one",
                "width one differs bitwise from standalone Jacobi");
          }
        }
        if (strategy_baseline.empty()) {
          strategy_baseline = batch_output.distances;
        } else {
          expect(
              strategy_baseline == batch_output.distances,
              "width-one",
              "width-one load strategy changed distance bits");
        }
      }
    }
  }
}

void test_guards_and_maximum_rounds() {
  const bfnew::test::BatchedJacobiFixture fixture =
      bfnew::test::make_mixed_duration_batched_jacobi_fixture();
  const std::array<bfnew::RouteQuery, 1U> singleton{fixture.queries[2U]};
  bfnew::test::PreparedBatchedJacobiFixture prepared =
      bfnew::test::prepare_batched_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedJacobiWorkspace workspace;
  bfnew::hip::BatchedJacobiPullEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};
  for (const bfnew::GpuRunOptions& exhausted_options :
       control_matrix(true, 2U)) {
    const bfnew::hip::BatchedJacobiRunOutput exhausted =
        engine.run_status_only(prepared.description, exhausted_options);
    expect(
        exhausted.result.status.stop_reason ==
                static_cast<std::uint32_t>(
                    bfnew::DeviceStopReason::maximum_rounds) &&
            exhausted.result.status.rounds_completed == 2U &&
            exhausted.result.status.reached_target_mask == 0U &&
            exhausted.result.status.bounding_box_miss_mask == 0U,
        "maximum-rounds",
        "one control misreported exact round exhaustion as convergence/miss");
  }

  bfnew::GpuRunOptions limited =
      options_for(bfnew::ControlMode::persistent_cooperative, true);
  limited.maximum_rounds = 2U;

  bfnew::BatchDeviceDescription descriptors = prepared.description;
  descriptors.run_representation =
      bfnew::BatchRunRepresentation::compact_nonzero_descriptors;
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(engine.run_status_only(descriptors, limited)); },
      "guards",
      "HIP engine accepted the unselected descriptor representation");
  bfnew::BatchDeviceDescription corrupt_mask = prepared.description;
  const std::uint32_t touched = corrupt_mask.touched_csc_runs.front();
  corrupt_mask.csc_run_lane_masks[touched] = 0U;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(engine.run_status_only(corrupt_mask, limited));
      },
      "guards",
      "HIP engine accepted a retained mask inconsistent with endpoint tiles");
  bfnew::BatchDeviceDescription corrupt_union = prepared.description;
  bool corrupted_outside_union = false;
  for (std::size_t tile = 0U;
       tile < corrupt_union.tile_lane_masks.size();
       ++tile) {
    if (std::find(
            corrupt_union.union_tiles.begin(),
            corrupt_union.union_tiles.end(),
            static_cast<std::uint32_t>(tile)) ==
        corrupt_union.union_tiles.end()) {
      corrupt_union.tile_lane_masks[tile] = bfnew::LaneMask{1U};
      corrupted_outside_union = true;
      break;
    }
  }
  expect(corrupted_outside_union, "guards", "fixture lacks an outside tile");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(engine.run_status_only(corrupt_union, limited));
      },
      "guards",
      "HIP engine accepted a nonzero tile mask outside the union ranges");

  bfnew::test::PreparedBatchedJacobiFixture materialized =
      bfnew::test::prepare_batched_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::device_materialized_run_masks);
  ++materialized.description.selected_edge_estimates_by_lane.front();
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            engine.run_status_only(materialized.description, limited));
      },
      "guards",
      "full/evidence Jacobi accepted stale device-materialized edge estimates");
  --materialized.description.selected_edge_estimates_by_lane.front();
  ++materialized.description.selected_vertex_counts_by_lane.front();
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            engine.run_status_only(materialized.description, limited));
      },
      "guards",
      "Jacobi accepted selected vertex counts inconsistent with its ranges");

  bfnew::BatchDeviceDescription with_empty_tile = prepared.description;
  const auto tile_offsets = fixture.partitioned.graph.tile_vertex_offsets();
  bool inserted_empty_tile = false;
  for (std::size_t tile = 0U; tile + 1U < tile_offsets.size(); ++tile) {
    if (tile_offsets[tile] != tile_offsets[tile + 1U] ||
        std::binary_search(
            with_empty_tile.union_tiles.begin(),
            with_empty_tile.union_tiles.end(),
            static_cast<std::uint32_t>(tile))) {
      continue;
    }
    const auto position = std::lower_bound(
        with_empty_tile.union_tiles.begin(),
        with_empty_tile.union_tiles.end(),
        static_cast<std::uint32_t>(tile));
    const std::size_t index = static_cast<std::size_t>(
        position - with_empty_tile.union_tiles.begin());
    with_empty_tile.union_tiles.insert(
        position, static_cast<std::uint32_t>(tile));
    with_empty_tile.selected_vertex_ranges.insert(
        with_empty_tile.selected_vertex_ranges.begin() +
            static_cast<std::ptrdiff_t>(index),
        bfnew::BatchVertexRange{
            static_cast<std::uint32_t>(tile_offsets[tile]),
            static_cast<std::uint32_t>(tile_offsets[tile + 1U]),
            bfnew::LaneMask{1U}});
    with_empty_tile.tile_lane_masks[tile] = bfnew::LaneMask{1U};
    inserted_empty_tile = true;
    break;
  }
  expect(
      inserted_empty_tile,
      "empty-selected-tile",
      "fixture did not expose its empty spill tile");
  const bfnew::hip::BatchedJacobiRunOutput empty_tile_output =
      engine.run_status_only(with_empty_tile, limited);
  expect(
      empty_tile_output.result.status.rounds_completed == 2U &&
          empty_tile_output.result.status.stop_reason ==
              static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::maximum_rounds),
      "empty-selected-tile",
      "legal empty selected tile changed bounded execution");
  bfnew::GpuRunOptions invalid_block = limited;
  invalid_block.block_size = 127U;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            engine.run_status_only(prepared.description, invalid_block));
      },
      "guards",
      "wave-per-destination engine accepted a non-wave32 block size");
}

}  // namespace

int main() {
  test_mixed_width_control_matrix();
  test_width_one_matches_standalone();
  test_guards_and_maximum_rounds();
  std::cout << "batched Jacobi HIP tests passed\n";
  return 0;
}
