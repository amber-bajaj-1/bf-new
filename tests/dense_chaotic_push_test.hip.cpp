#include "bfnew/hip/dense_chaotic_push.hpp"
#include "bfnew/sssp.hpp"
#include "dense_fixture_suite.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void fail(
    const std::string_view fixture,
    const std::string_view message) {
  std::cerr << "Dense HIP test failed [" << fixture << "]: " << message
            << '\n';
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

[[nodiscard]] bool bitwise_equal(
    const std::span<const float> left,
    const std::span<const float> right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (std::bit_cast<std::uint32_t>(left[index]) !=
        std::bit_cast<std::uint32_t>(right[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::vector<float> bounded_oracle(
    const bfnew::WeightedGraph& graph,
    const bfnew::RouteQuery& query) {
  const bfnew::InducedQueryGraph induced =
      bfnew::build_induced_query_graph(graph, query);
  const bfnew::SsspResult local =
      bfnew::dijkstra_oracle(induced.graph, induced.sources);
  std::vector<float> global(
      graph.vertex_count(), std::numeric_limits<float>::infinity());
  for (std::size_t local_vertex = 0U;
       local_vertex < induced.local_to_global.size();
       ++local_vertex) {
    global[induced.local_to_global[local_vertex].value()] =
        local.distances[local_vertex];
  }
  return global;
}

[[nodiscard]] bool matches_oracle(
    const bfnew::test::JacobiFixtureCase& fixture,
    const std::span<const float> actual,
    const std::span<const float> expected) {
  if (fixture.comparison == bfnew::test::JacobiComparisonPolicy::bitwise) {
    return bitwise_equal(actual, expected);
  }
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t vertex = 0U; vertex < actual.size(); ++vertex) {
    if (!bfnew::nonnegative_distance_within_ulps(
            actual[vertex], expected[vertex], 4U)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bfnew::GpuRunOptions base_options(
    const bfnew::ControlMode mode,
    const std::uint32_t rounds_per_chunk = 1U) {
  bfnew::GpuRunOptions options;
  options.engine = bfnew::EngineKind::dense_chaotic_push;
  options.control_mode = mode;
  options.rounds_per_chunk = rounds_per_chunk;
  options.block_size = 128U;
  options.maximum_rounds = 128U;
  options.instrumentation = bfnew::InstrumentationLevel::debug;
  return options;
}

void validate_run(
    const bfnew::test::JacobiFixtureCase& fixture,
    const std::span<const float> oracle,
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::DenseRunOutput& output) {
  expect(
      output.result.engine_kind == static_cast<std::uint32_t>(
                                       bfnew::EngineKind::dense_chaotic_push) &&
          output.result.control_mode ==
              static_cast<std::uint32_t>(options.control_mode),
      fixture.name,
      "result lost independently selected engine/control identity");
  expect(
      bfnew::validate_device_run_status(output.result.status) ==
              bfnew::DeviceRunStatusError::none &&
          output.result.status.stop_reason == static_cast<std::uint32_t>(
                                                   bfnew::DeviceStopReason::converged) &&
          output.result.status.final_distance_slot == 0U,
      fixture.name,
      "dense terminal status is invalid or reports a second distance slot");
  expect(
      output.result.status.reached_target_mask ==
              fixture.expected_reached_mask &&
          output.result.status.bounding_box_miss_mask ==
              fixture.expected_miss_mask,
      fixture.name,
      "reached/miss status disagrees with the bounded fixture");
  expect(
      matches_oracle(fixture, output.distances, oracle),
      fixture.name,
      "dense labels disagree with bounded CPU Dijkstra");
  expect(
      output.result.work.queue_claims == 0U &&
          output.result.work.duplicate_suppressions == 0U &&
          output.result.work.maximum_queue_size == 0U,
      fixture.name,
      "dense execution exposed frontier/worklist accounting");
  if (options.instrumentation == bfnew::InstrumentationLevel::debug) {
    expect(
        output.result.work.atomic_attempts ==
                output.result.work.edges_examined &&
            output.result.work.successful_atomic_updates ==
                output.result.work.successful_decreases,
        fixture.name,
        "Debug atomic instrumentation is inconsistent");
  } else {
    expect(
        output.result.work.atomic_attempts == 0U &&
            output.result.work.successful_atomic_updates == 0U,
        fixture.name,
        "non-Debug mode recorded expensive atomic counters");
  }
  expect(
      output.result.work.full_edge_rounds ==
          (options.instrumentation == bfnew::InstrumentationLevel::none
               ? 0U
               : output.result.status.rounds_completed),
      fixture.name,
      "complete-edge-round instrumentation is inconsistent");
  expect(
      output.result.work.host_checks ==
              output.metrics.convergence_host_checks &&
          output.result.work.controller_copies ==
              output.metrics.convergence_host_checks + 2U &&
          output.result.work.host_synchronizations ==
              output.metrics.convergence_host_checks + 1U,
      fixture.name,
      "physical host-control accounting is inconsistent");
  if (options.control_mode == bfnew::ControlMode::persistent_cooperative) {
    expect(
        output.result.work.kernel_dispatches == 1U &&
            output.metrics.convergence_host_checks == 0U &&
            output.metrics.cooperative_grid_blocks != 0U,
        fixture.name,
        "persistent mode was not one cooperative query kernel");
  } else if (options.control_mode ==
             bfnew::ControlMode::per_round_host_poll) {
    expect(
        output.metrics.engine_round_dispatches ==
                output.result.status.rounds_completed &&
            output.metrics.controller_advance_dispatches ==
                output.metrics.engine_round_dispatches &&
            output.metrics.convergence_host_checks ==
                output.result.status.rounds_completed,
        fixture.name,
        "per-round mode did not submit exact ordered pairs");
  } else {
    expect(
        output.metrics.engine_round_dispatches % options.rounds_per_chunk ==
                0U &&
            output.metrics.controller_advance_dispatches ==
                output.metrics.engine_round_dispatches &&
            output.metrics.convergence_host_checks * options.rounds_per_chunk ==
                output.metrics.engine_round_dispatches,
        fixture.name,
        "chunked mode synchronized within a K-round chunk");
  }
}

void run_fixture(const bfnew::test::JacobiFixtureCase& fixture) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const bfnew::TileRunLayout64 runs = bfnew::build_tile_run_layout(graph);
  const bfnew::DeviceGraphLayout32 layout =
      bfnew::build_device_graph_layout32(graph, runs);
  const std::vector<float> oracle = bounded_oracle(graph, fixture.query);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(layout), stream);
  bfnew::hip::ReusableDeviceWorkspace workspace;
  bfnew::hip::DenseChaoticPushEngine engine{
      graph, runs, resident, workspace, stream};

  std::vector<float> control_baseline;
  const auto exercise = [&](const bfnew::GpuRunOptions& options) {
    const bfnew::hip::DenseRunOutput output =
        engine.run_with_distances(fixture.query, options);
    validate_run(fixture, oracle, options, output);
    if (control_baseline.empty()) {
      control_baseline = output.distances;
    } else {
      expect(
          bitwise_equal(control_baseline, output.distances),
          fixture.name,
          "control mode changed final dense distance bits");
    }
    return output;
  };

  const bfnew::hip::DenseRunOutput per_round = exercise(
      base_options(bfnew::ControlMode::per_round_host_poll));
  if (fixture.name == "dense_high_fan_in_contention") {
    expect(
        per_round.result.work.high_contention_destinations >= 1U &&
            per_round.result.work.changed_flag_updates >= 1U,
        fixture.name,
        "Debug omitted contention or changed-block instrumentation");
  }
  for (const std::uint32_t k :
       std::array<std::uint32_t, 5U>{2U, 4U, 8U, 16U, 32U}) {
    static_cast<void>(exercise(base_options(
        bfnew::ControlMode::chunked_host_poll, k)));
  }
  const bfnew::hip::DenseRunOutput persistent = exercise(
      base_options(bfnew::ControlMode::persistent_cooperative));

  if (fixture.name == "Phase 5 core") {
    expect(
        persistent.metrics.cooperative_active_blocks_per_wgp >= 2U,
        fixture.name,
        "dense occupancy did not expose two legal cooperative grids");
    std::uint32_t one_block_grid = 0U;
    for (const std::uint32_t blocks_per_wgp : {1U, 2U}) {
      bfnew::GpuRunOptions fixed =
          base_options(bfnew::ControlMode::persistent_cooperative);
      fixed.grid_policy = bfnew::GridPolicy::fixed_blocks_per_wgp;
      fixed.blocks_per_wgp = blocks_per_wgp;
      const bfnew::hip::DenseRunOutput fixed_output = exercise(fixed);
      if (blocks_per_wgp == 1U) {
        one_block_grid = fixed_output.metrics.cooperative_grid_blocks;
      } else {
        expect(
            fixed_output.metrics.cooperative_grid_blocks ==
                one_block_grid * 2U,
            fixture.name,
            "runtime-derived cooperative grid did not scale by residency");
      }
    }

    bfnew::GpuRunOptions none =
        base_options(bfnew::ControlMode::per_round_host_poll);
    none.instrumentation = bfnew::InstrumentationLevel::none;
    const bfnew::hip::DenseRunOutput none_output = exercise(none);
    bfnew::GpuRunOptions light = none;
    light.instrumentation = bfnew::InstrumentationLevel::light;
    const bfnew::hip::DenseRunOutput light_output = exercise(light);
    expect(
        none_output.result.work.edges_examined == 0U &&
            none_output.result.work.full_edge_rounds == 0U &&
            light_output.result.work.edges_examined != 0U &&
            light_output.result.work.atomic_attempts == 0U &&
            per_round.result.work.atomic_attempts ==
                per_round.result.work.edges_examined,
        fixture.name,
        "None/Light/Debug instrumentation levels are not isolated");
  }

  const bfnew::hip::DenseRunOutput repeated = engine.run_with_distances(
      fixture.query,
      base_options(bfnew::ControlMode::per_round_host_poll));
  expect(
      bitwise_equal(repeated.distances, per_round.distances),
      fixture.name,
      "repeated dense execution changed final distance bits");
}

void test_maximum_rounds() {
  const bfnew::test::JacobiFixtureCase fixture =
      bfnew::test::make_long_chain_jacobi_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const bfnew::TileRunLayout64 runs = bfnew::build_tile_run_layout(graph);
  const bfnew::DeviceGraphLayout32 layout =
      bfnew::build_device_graph_layout32(graph, runs);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(layout), stream);
  bfnew::hip::ReusableDeviceWorkspace workspace;
  bfnew::hip::DenseChaoticPushEngine engine{
      graph, runs, resident, workspace, stream};
  for (const bfnew::ControlMode mode : {
           bfnew::ControlMode::persistent_cooperative,
           bfnew::ControlMode::per_round_host_poll,
           bfnew::ControlMode::chunked_host_poll,
       }) {
    bfnew::GpuRunOptions options = base_options(mode, 32U);
    options.maximum_rounds = 1U;
    const bfnew::hip::DenseRunOutput output =
        engine.run_with_distances(fixture.query, options);
    expect(
        output.result.status.stop_reason == static_cast<std::uint32_t>(
                                                bfnew::DeviceStopReason::maximum_rounds) &&
            output.result.status.rounds_completed == 1U &&
            output.result.status.final_distance_slot == 0U &&
            output.result.status.bounding_box_miss_mask == 0U,
        "maximum-round exhaustion",
        "a dense control mode misreported the terminal condition");
    if (mode == bfnew::ControlMode::chunked_host_poll) {
      expect(
          output.metrics.engine_round_dispatches == 32U &&
              output.result.work.kernel_dispatches == 66U,
          "maximum-round exhaustion",
          "K=32 did not retain 31 already-queued no-op pairs");
    }
  }
}

}  // namespace

int main() {
  for (const bfnew::test::JacobiFixtureCase& fixture :
       bfnew::test::make_dense_fixture_suite()) {
    run_fixture(fixture);
  }
  test_maximum_rounds();
  return 0;
}
