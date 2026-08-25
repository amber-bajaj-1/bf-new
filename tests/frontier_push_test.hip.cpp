#include "bfnew/device_layout.hpp"
#include "bfnew/hip/frontier_push.hpp"
#include "bfnew/sssp.hpp"
#include "frontier_fixture_suite.hpp"

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
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void fail(
    const std::string_view fixture,
    const std::string_view message) {
  std::cerr << "Frontier HIP test failed [" << fixture << "]: " << message
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

template <typename Exception, typename Function>
void expect_throws(
    Function&& function,
    const std::string_view fixture,
    const std::string_view message) {
  try {
    function();
  } catch (const Exception&) {
    return;
  } catch (...) {
  }
  fail(fixture, message);
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
  options.engine = bfnew::EngineKind::frontier_push;
  options.control_mode = mode;
  options.rounds_per_chunk = rounds_per_chunk;
  options.block_size = 128U;
  options.maximum_rounds = 128U;
  options.instrumentation = bfnew::InstrumentationLevel::debug;
  return options;
}

void validate_physical_control_accounting(
    const std::string_view fixture,
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::FrontierRunOutput& output) {
  const bfnew::DeviceWorkStatistics& work = output.result.work;
  expect(
      work.host_checks == output.metrics.convergence_host_checks &&
          work.controller_copies ==
              output.metrics.convergence_host_checks + 2U &&
          work.host_synchronizations ==
              output.metrics.convergence_host_checks + 1U,
      fixture,
      "physical host checks, controller copies, or synchronizations disagree");

  if (options.control_mode == bfnew::ControlMode::persistent_cooperative) {
    expect(
        output.metrics.engine_round_dispatches == 0U &&
            output.metrics.controller_advance_dispatches == 0U &&
            output.metrics.convergence_host_checks == 0U &&
            work.kernel_dispatches == 1U &&
            output.metrics.cooperative_grid_blocks != 0U &&
            output.metrics.cooperative_active_blocks_per_wgp != 0U,
        fixture,
        "persistent convergence was not one GPU-controlled query kernel");
    return;
  }

  expect(
      output.metrics.engine_round_dispatches ==
              output.metrics.controller_advance_dispatches &&
          output.metrics.engine_round_dispatches >=
              output.result.status.rounds_completed &&
          work.kernel_dispatches ==
              2U + 2U * output.metrics.engine_round_dispatches,
      fixture,
      "ordinary execution violated initialization/round/advance/finalize accounting");
  if (options.control_mode == bfnew::ControlMode::per_round_host_poll) {
    expect(
        output.metrics.engine_round_dispatches ==
                output.result.status.rounds_completed &&
            output.metrics.convergence_host_checks ==
                output.result.status.rounds_completed,
        fixture,
        "per-round mode did not submit exactly one pair per host check");
  } else {
    expect(
        output.metrics.engine_round_dispatches % options.rounds_per_chunk ==
                0U &&
            output.metrics.convergence_host_checks *
                    options.rounds_per_chunk ==
                output.metrics.engine_round_dispatches,
        fixture,
        "chunked mode synchronized or copied controller state inside a chunk");
  }
}

void validate_algorithm_counters(
    const bfnew::test::JacobiFixtureCase& fixture,
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::FrontierRunOutput& output) {
  const bfnew::DeviceWorkStatistics& work = output.result.work;
  expect(
      work.full_edge_rounds == 0U && work.changed_flag_updates == 0U &&
          work.high_contention_destinations == 0U,
      fixture.name,
      "frontier execution leaked dense-chaotic counters");

  if (options.instrumentation == bfnew::InstrumentationLevel::none) {
    expect(
        work.edges_examined == 0U && work.successful_decreases == 0U &&
            work.active_vertices == 0U && work.active_lane_rounds == 0U &&
            work.maximum_queue_size == 0U && work.atomic_attempts == 0U &&
            work.successful_atomic_updates == 0U &&
            work.queue_claims == 0U &&
            work.duplicate_suppressions == 0U &&
            work.mask_operations == 0U && work.overflow_events == 0U &&
            work.empty_frontier_rounds == 0U &&
            work.small_frontier_rounds == 0U,
        fixture.name,
        "None instrumentation recorded frontier algorithm counters");
    return;
  }

  expect(
      work.active_lane_rounds == output.result.status.rounds_completed &&
          work.maximum_queue_size >= fixture.query.sources.size() &&
          work.maximum_queue_size <= fixture.partitioned.graph.vertex_count() &&
          work.empty_frontier_rounds == 1U &&
          work.small_frontier_rounds <= work.active_lane_rounds &&
          work.overflow_events == 0U,
      fixture.name,
      "frontier round, queue-size, or terminal-empty accounting is inconsistent");

  if (options.instrumentation == bfnew::InstrumentationLevel::light) {
    expect(
        work.atomic_attempts == 0U &&
            work.successful_atomic_updates == 0U &&
            work.queue_claims == 0U &&
            work.duplicate_suppressions == 0U &&
            work.mask_operations == 0U,
        fixture.name,
        "Light instrumentation recorded Debug-only atomic or mask counters");
    return;
  }

  expect(
      work.atomic_attempts == work.edges_examined &&
          work.successful_atomic_updates == work.successful_decreases &&
          work.queue_claims + work.duplicate_suppressions ==
              work.successful_atomic_updates &&
          (work.edges_examined == 0U || work.mask_operations != 0U),
      fixture.name,
      "Debug atomic, queue-claim, duplicate, or mask counters are inconsistent");
}

void validate_converged_run(
    const bfnew::test::JacobiFixtureCase& fixture,
    const std::span<const float> oracle,
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::FrontierRunOutput& output) {
  const bfnew::DeviceController& controller = output.final_controller;
  const bfnew::DeviceRunStatus& status = output.result.status;
  expect(
      output.result.engine_kind == static_cast<std::uint32_t>(
                                       bfnew::EngineKind::frontier_push) &&
          output.result.control_mode ==
              static_cast<std::uint32_t>(options.control_mode),
      fixture.name,
      "result lost independently selected frontier/control identity");
  expect(
      bfnew::validate_device_controller(controller) ==
              bfnew::DeviceControllerError::none &&
          bfnew::validate_device_run_status(status) ==
              bfnew::DeviceRunStatusError::none,
      fixture.name,
      "terminal frontier controller or status is invalid");
  expect(
      controller.done == 1U && controller.execute_lane_mask == 0U &&
          controller.stop_reason == static_cast<std::uint32_t>(
                                        bfnew::DeviceStopReason::converged) &&
          controller.error_bits == bfnew::device_error::none &&
          status.converged == 1U &&
          status.stop_reason == controller.stop_reason &&
          status.error_bits == controller.error_bits &&
          status.rounds_completed == controller.rounds_completed &&
          status.active_lane_mask == controller.active_lane_mask &&
          status.converged_lane_mask == controller.converged_lane_mask,
      fixture.name,
      "controller/status terminal state disagrees");
  expect(
      status.final_distance_slot == 0U &&
          controller.distance_read_slot == 0U &&
          controller.distance_write_slot == 0U &&
          controller.frontier_read_slot ==
              controller.rounds_completed % 2U &&
          controller.frontier_write_slot ==
              1U - controller.frontier_read_slot &&
          controller.frontier_size[controller.frontier_read_slot] == 0U,
      fixture.name,
      "final distance bit storage or frontier-slot identity is inconsistent");
  expect(
      fixture.expected_rounds == 0U ||
          status.rounds_completed >= fixture.expected_rounds,
      fixture.name,
      "target reachability stopped frontier convergence early");
  expect(
      status.reached_target_mask == fixture.expected_reached_mask &&
          status.bounding_box_miss_mask == fixture.expected_miss_mask,
      fixture.name,
      "bounded reached/miss masks disagree with the fixture contract");
  expect(
      matches_oracle(fixture, output.distances, oracle),
      fixture.name,
      "frontier labels disagree with bounded CPU Dijkstra");
  expect(
      output.queue_capacity == fixture.partitioned.graph.vertex_count(),
      fixture.name,
      "default frontier capacity is not one unique entry per vertex");
  expect(
      output.metrics.gpu_milliseconds >= 0.0F &&
          output.metrics.wall_milliseconds >= 0.0,
      fixture.name,
      "frontier timing fields are invalid");
  validate_algorithm_counters(fixture, options, output);
  validate_physical_control_accounting(fixture.name, options, output);
}

void run_fixture_matrix(const bfnew::test::JacobiFixtureCase& fixture) {
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
  bfnew::hip::FrontierPushEngine engine{
      graph, runs, resident, workspace, stream};

  std::vector<float> control_baseline;
  const auto exercise = [&](const bfnew::GpuRunOptions& options) {
    const bfnew::hip::FrontierRunOutput output =
        engine.run_with_distances(fixture.query, options);
    validate_converged_run(fixture, oracle, options, output);
    if (control_baseline.empty()) {
      control_baseline = output.distances;
    } else {
      expect(
          bitwise_equal(control_baseline, output.distances),
          fixture.name,
          "control mode or K changed final frontier distance bits");
    }
    return output;
  };

  static_cast<void>(
      exercise(base_options(bfnew::ControlMode::per_round_host_poll)));
  for (const std::uint32_t rounds_per_chunk :
       std::array<std::uint32_t, 5U>{2U, 4U, 8U, 16U, 32U}) {
    static_cast<void>(exercise(base_options(
        bfnew::ControlMode::chunked_host_poll, rounds_per_chunk)));
  }
  static_cast<void>(
      exercise(base_options(bfnew::ControlMode::persistent_cooperative)));
}

void test_instrumentation_levels() {
  const bfnew::test::JacobiFixtureCase fixture =
      bfnew::test::make_phase5_core_jacobi_fixture();
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
  bfnew::hip::FrontierPushEngine engine{
      graph, runs, resident, workspace, stream};

  const auto run_at = [&](const bfnew::InstrumentationLevel level) {
    bfnew::GpuRunOptions options =
        base_options(bfnew::ControlMode::per_round_host_poll);
    options.instrumentation = level;
    const bfnew::hip::FrontierRunOutput output =
        engine.run_with_distances(fixture.query, options);
    validate_converged_run(fixture, oracle, options, output);
    return output;
  };

  const bfnew::hip::FrontierRunOutput none =
      run_at(bfnew::InstrumentationLevel::none);
  const bfnew::hip::FrontierRunOutput light =
      run_at(bfnew::InstrumentationLevel::light);
  const bfnew::hip::FrontierRunOutput debug =
      run_at(bfnew::InstrumentationLevel::debug);
  expect(
      bitwise_equal(none.distances, light.distances) &&
          bitwise_equal(light.distances, debug.distances),
      fixture.name,
      "instrumentation level changed final distance bits");
  expect(
      light.result.work.edges_examined != 0U &&
          light.result.work.active_vertices != 0U &&
          light.result.work.maximum_queue_size != 0U &&
          light.result.work.atomic_attempts == 0U &&
          debug.result.work.edges_examined ==
              light.result.work.edges_examined &&
          debug.result.work.active_vertices ==
              light.result.work.active_vertices &&
          debug.result.work.atomic_attempts ==
              debug.result.work.edges_examined &&
          debug.result.work.queue_claims != 0U &&
          debug.result.work.mask_operations != 0U,
      fixture.name,
      "None/Light/Debug frontier instrumentation levels are not isolated");
}

void test_cooperative_grid_policies() {
  const bfnew::test::JacobiFixtureCase fixture =
      bfnew::test::make_phase5_core_jacobi_fixture();
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
  bfnew::hip::FrontierPushEngine engine{
      graph, runs, resident, workspace, stream};

  bfnew::GpuRunOptions occupancy =
      base_options(bfnew::ControlMode::persistent_cooperative);
  const bfnew::hip::FrontierRunOutput occupancy_output =
      engine.run_with_distances(fixture.query, occupancy);
  validate_converged_run(fixture, oracle, occupancy, occupancy_output);
  expect(
      occupancy_output.metrics.cooperative_active_blocks_per_wgp >= 2U,
      fixture.name,
      "real frontier-kernel occupancy does not expose two fixed legal grids");

  std::vector<float> baseline = occupancy_output.distances;
  std::uint32_t one_block_grid = 0U;
  for (const std::uint32_t blocks_per_wgp :
       std::array<std::uint32_t, 2U>{1U, 2U}) {
    bfnew::GpuRunOptions fixed = occupancy;
    fixed.grid_policy = bfnew::GridPolicy::fixed_blocks_per_wgp;
    fixed.blocks_per_wgp = blocks_per_wgp;
    const bfnew::hip::FrontierRunOutput output =
        engine.run_with_distances(fixture.query, fixed);
    validate_converged_run(fixture, oracle, fixed, output);
    expect(
        output.metrics.cooperative_active_blocks_per_wgp ==
                occupancy_output.metrics.cooperative_active_blocks_per_wgp &&
            bitwise_equal(output.distances, baseline),
        fixture.name,
        "fixed cooperative residency changed occupancy evidence or labels");
    if (blocks_per_wgp == 1U) {
      one_block_grid = output.metrics.cooperative_grid_blocks;
      expect(
          one_block_grid != 0U,
          fixture.name,
          "one-block-per-WGP policy produced an empty cooperative grid");
    } else {
      expect(
          output.metrics.cooperative_grid_blocks == one_block_grid * 2U,
          fixture.name,
          "two-block fixed residency did not double the legal grid");
    }
  }
  expect(
      occupancy_output.metrics.cooperative_grid_blocks ==
          one_block_grid *
              occupancy_output.metrics.cooperative_active_blocks_per_wgp,
      fixture.name,
      "occupancy-derived frontier grid did not use real-kernel residency");
}

void test_maximum_round_exhaustion() {
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
  bfnew::hip::FrontierPushEngine engine{
      graph, runs, resident, workspace, stream};

  std::vector<float> control_baseline;
  for (const bfnew::ControlMode mode : {
           bfnew::ControlMode::persistent_cooperative,
           bfnew::ControlMode::per_round_host_poll,
           bfnew::ControlMode::chunked_host_poll,
       }) {
    bfnew::GpuRunOptions options = base_options(mode, 32U);
    options.maximum_rounds = 1U;
    const bfnew::hip::FrontierRunOutput output =
        engine.run_with_distances(fixture.query, options);
    const bfnew::DeviceController& controller = output.final_controller;
    const bfnew::DeviceRunStatus& status = output.result.status;
    expect(
        bfnew::validate_device_controller(controller) ==
                bfnew::DeviceControllerError::none &&
            bfnew::validate_device_run_status(status) ==
                bfnew::DeviceRunStatusError::none &&
            controller.done == 1U && controller.execute_lane_mask == 0U &&
            status.stop_reason == static_cast<std::uint32_t>(
                                      bfnew::DeviceStopReason::maximum_rounds) &&
            status.error_bits == bfnew::device_error::none &&
            status.rounds_completed == 1U && status.converged == 0U &&
            status.reached_target_mask == 0U &&
            status.bounding_box_miss_mask == 0U &&
            controller.frontier_read_slot == 1U &&
            controller.frontier_write_slot == 0U &&
            controller.frontier_size[controller.frontier_read_slot] != 0U,
        "maximum-round exhaustion",
        "a frontier control mode lost its partial-work terminal state");
    validate_physical_control_accounting(
        "maximum-round exhaustion", options, output);
    if (control_baseline.empty()) {
      control_baseline = output.distances;
    } else {
      expect(
          bitwise_equal(control_baseline, output.distances),
          "maximum-round exhaustion",
          "post-terminal no-op pairs changed partial distance bits");
    }
    if (mode == bfnew::ControlMode::chunked_host_poll) {
      expect(
          output.metrics.engine_round_dispatches == 32U &&
              output.metrics.controller_advance_dispatches == 32U &&
              output.metrics.convergence_host_checks == 1U &&
              output.result.work.kernel_dispatches == 66U,
          "maximum-round exhaustion",
          "K=32 did not retain its already-queued no-op pairs");
    }
  }
}

void test_queue_overflow_seam() {
  const bfnew::test::JacobiFixtureCase fixture =
      bfnew::test::make_expanding_grid_frontier_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const bfnew::TileRunLayout64 runs = bfnew::build_tile_run_layout(graph);
  const bfnew::DeviceGraphLayout32 layout =
      bfnew::build_device_graph_layout32(graph, runs);

  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(layout), stream);
  bfnew::hip::ReusableDeviceWorkspace workspace;
  bfnew::hip::FrontierPushEngine engine{
      graph, runs, resident, workspace, stream, 1U};

  for (const bfnew::ControlMode mode : {
           bfnew::ControlMode::persistent_cooperative,
           bfnew::ControlMode::per_round_host_poll,
           bfnew::ControlMode::chunked_host_poll,
       }) {
    bfnew::GpuRunOptions options = base_options(mode, 32U);
    const bfnew::hip::FrontierRunOutput output =
        engine.run_with_distances(fixture.query, options);
    const bfnew::DeviceController& controller = output.final_controller;
    const bfnew::DeviceRunStatus& status = output.result.status;
    expect(
        bfnew::validate_device_controller(controller) ==
                bfnew::DeviceControllerError::none &&
            bfnew::validate_device_run_status(status) ==
                bfnew::DeviceRunStatusError::none &&
            output.queue_capacity == 1U && controller.done == 1U &&
            status.stop_reason == static_cast<std::uint32_t>(
                                      bfnew::DeviceStopReason::queue_overflow) &&
            status.error_bits == bfnew::device_error::queue_overflow &&
            status.rounds_completed == 1U && status.converged == 0U &&
            status.reached_target_mask == 0U &&
            status.bounding_box_miss_mask == 0U &&
            output.result.work.overflow_events == 1U &&
            output.result.work.maximum_queue_size > output.queue_capacity,
        "queue-overflow seam",
        "frontier overflow was silent, misclassified, or out of the tested seam");
    validate_physical_control_accounting("queue-overflow seam", options, output);
    if (mode == bfnew::ControlMode::chunked_host_poll) {
      expect(
          output.metrics.engine_round_dispatches == 32U &&
              output.metrics.controller_advance_dispatches == 32U &&
              output.metrics.convergence_host_checks == 1U &&
              output.result.work.kernel_dispatches == 66U,
          "queue-overflow seam",
          "K=32 overflow did not retain already-submitted no-op pairs");
    }
  }
}

void test_empty_initial_frontier_is_invalid() {
  const bfnew::test::JacobiFixtureCase fixture =
      bfnew::test::make_phase5_core_jacobi_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const bfnew::TileRunLayout64 runs = bfnew::build_tile_run_layout(graph);
  const bfnew::DeviceGraphLayout32 layout =
      bfnew::build_device_graph_layout32(graph, runs);

  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(layout), stream);
  bfnew::hip::ReusableDeviceWorkspace workspace;
  bfnew::hip::FrontierPushEngine engine{
      graph, runs, resident, workspace, stream};

  bfnew::RouteQuery invalid = fixture.query;
  invalid.sources.clear();
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(engine.run_with_distances(
            invalid,
            base_options(bfnew::ControlMode::persistent_cooperative)));
      },
      "empty initial frontier",
      "frontier engine accepted a query without a canonical source");
}

}  // namespace

int main() {
  for (const bfnew::test::JacobiFixtureCase& fixture :
       bfnew::test::make_frontier_fixture_suite()) {
    run_fixture_matrix(fixture);
  }
  test_instrumentation_levels();
  test_cooperative_grid_policies();
  test_maximum_round_exhaustion();
  test_queue_overflow_seam();
  test_empty_initial_frontier_is_invalid();
  return 0;
}
