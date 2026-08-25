#include "bfnew/device_layout.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/jacobi_pull.hpp"
#include "bfnew/query.hpp"
#include "bfnew/sssp.hpp"
#include "jacobi_fixture_suite.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

int failures = 0;
std::uint32_t maximum_observed_random_ulps = 0U;

void expect(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAILED: " << description << '\n';
    ++failures;
  }
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, const std::string_view description) {
  try {
    function();
    expect(false, description);
  } catch (const Exception&) {
  } catch (...) {
    expect(false, description);
  }
}

[[nodiscard]] bool bitwise_equal(const float left, const float right) noexcept {
  return std::bit_cast<std::uint32_t>(left) ==
         std::bit_cast<std::uint32_t>(right);
}

[[nodiscard]] bool vectors_bitwise_equal(
    const std::span<const float> left,
    const std::span<const float> right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (!bitwise_equal(left[index], right[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::uint32_t nonnegative_ulp_distance(
    const float left,
    const float right) noexcept {
  if (left == right) {
    return 0U;
  }
  const std::uint32_t left_bits = std::bit_cast<std::uint32_t>(left);
  const std::uint32_t right_bits = std::bit_cast<std::uint32_t>(right);
  return left_bits > right_bits ? left_bits - right_bits
                                : right_bits - left_bits;
}

struct BoundedOracle {
  std::vector<float> global_distances;
};

[[nodiscard]] BoundedOracle bounded_oracle(
    const bfnew::WeightedGraph& graph,
    const bfnew::RouteQuery& query) {
  const bfnew::InducedQueryGraph induced =
      bfnew::build_induced_query_graph(graph, query);
  const bfnew::SsspResult result =
      bfnew::dijkstra_oracle(induced.graph, induced.sources);
  BoundedOracle oracle;
  oracle.global_distances.assign(
      graph.vertex_count(), std::numeric_limits<float>::infinity());
  for (std::size_t local = 0U; local < induced.local_to_global.size(); ++local) {
    oracle.global_distances[induced.local_to_global[local].value()] =
        result.distances[local];
  }
  return oracle;
}

struct PreparedFixture {
  bfnew::TileRunLayout64 runs;
  bfnew::DeviceGraphLayout32 layout;
  std::vector<bfnew::LaneMask> tile_masks;
  bfnew::TileRunLaneMasks run_masks;
};

[[nodiscard]] PreparedFixture prepare_fixture(
    const bfnew::test::JacobiFixtureCase& fixture) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  PreparedFixture prepared;
  prepared.runs = bfnew::build_tile_run_layout(graph);
  prepared.layout = bfnew::build_device_graph_layout32(graph, prepared.runs);
  prepared.tile_masks.assign(graph.tile_coordinates().size(), 0U);
  for (const bfnew::TileId tile : fixture.query.selected_tiles) {
    prepared.tile_masks[tile.value()] = 1U;
  }
  bfnew::compute_tile_run_lane_masks(
      graph, prepared.runs, prepared.tile_masks, prepared.run_masks);
  expect(
      bfnew::prove_run_admission_equivalence(
          graph, prepared.runs, prepared.tile_masks, prepared.run_masks)
          .ok(),
      fixture.name + ": CSC run masks equal endpoint admission");
  return prepared;
}

[[nodiscard]] std::vector<bfnew::GpuRunOptions> control_matrix() {
  std::vector<bfnew::GpuRunOptions> controls;
  controls.reserve(7U);

  bfnew::GpuRunOptions persistent;
  persistent.engine = bfnew::EngineKind::jacobi_pull;
  persistent.control_mode = bfnew::ControlMode::persistent_cooperative;
  persistent.maximum_rounds = 64U;
  persistent.instrumentation = bfnew::InstrumentationLevel::debug;
  controls.push_back(persistent);

  bfnew::GpuRunOptions per_round = persistent;
  per_round.control_mode = bfnew::ControlMode::per_round_host_poll;
  controls.push_back(per_round);

  for (const std::uint32_t k : {2U, 4U, 8U, 16U, 32U}) {
    bfnew::GpuRunOptions chunked = persistent;
    chunked.control_mode = bfnew::ControlMode::chunked_host_poll;
    chunked.rounds_per_chunk = k;
    controls.push_back(chunked);
  }
  return controls;
}

void compare_with_oracle(
    const bfnew::test::JacobiFixtureCase& fixture,
    const std::span<const float> actual,
    const std::span<const float> expected) {
  expect(actual.size() == expected.size(), fixture.name + ": distance shape");
  if (actual.size() != expected.size()) {
    return;
  }
  if (fixture.comparison == bfnew::test::JacobiComparisonPolicy::bitwise) {
    expect(
        vectors_bitwise_equal(actual, expected),
        fixture.name + ": bitwise agreement with bounded Dijkstra");
    return;
  }
  bool within_policy = true;
  for (std::size_t vertex = 0U; vertex < actual.size(); ++vertex) {
    if (!bfnew::nonnegative_distance_within_ulps(
            actual[vertex], expected[vertex], 4U)) {
      within_policy = false;
      break;
    }
    if (std::isfinite(actual[vertex]) && std::isfinite(expected[vertex])) {
      maximum_observed_random_ulps = std::max(
          maximum_observed_random_ulps,
          nonnegative_ulp_distance(actual[vertex], expected[vertex]));
    }
  }
  expect(
      within_policy,
      fixture.name + ": general weights agree with bounded Dijkstra within 4 ULP");
}

void check_control_accounting(
    const bfnew::test::JacobiFixtureCase& fixture,
    const bfnew::GpuRunOptions& options,
    const bfnew::HostJacobiRunResult& run) {
  const std::uint64_t rounds = run.controller.rounds_completed;
  const bfnew::DeviceWorkStatistics& work = run.result.work;
  if (options.control_mode == bfnew::ControlMode::persistent_cooperative) {
    expect(work.kernel_dispatches == 1U,
           fixture.name + ": persistent uses one cooperative dispatch");
    expect(work.controller_copies == 1U && work.host_synchronizations == 1U &&
               work.host_checks == 1U,
           fixture.name + ": persistent interacts with host only after convergence");
    expect(run.queued_round_pairs == 0U && run.completed_host_chunks == 0U,
           fixture.name + ": persistent has no ordinary polling chunks");
    return;
  }

  if (options.control_mode == bfnew::ControlMode::per_round_host_poll) {
    expect(run.queued_round_pairs == rounds,
           fixture.name + ": per-round queues one pair per completed round");
    expect(run.completed_host_chunks == rounds &&
               work.controller_copies == rounds &&
               work.host_synchronizations == rounds && work.host_checks == rounds,
           fixture.name + ": per-round performs one control interaction per round");
    expect(work.kernel_dispatches == 1U + 2U * rounds,
           fixture.name + ": per-round counts init and every round/advance dispatch");
    return;
  }

  const std::uint64_t k = options.rounds_per_chunk;
  const std::uint64_t chunks = (rounds + k - 1U) / k;
  const std::uint64_t queued_pairs = chunks * k;
  expect(run.completed_host_chunks == chunks,
         fixture.name + ": chunk count follows completed rounds");
  expect(run.queued_round_pairs == queued_pairs,
         fixture.name + ": final chunk retains its queued no-op pairs");
  expect(work.controller_copies == chunks &&
             work.host_synchronizations == chunks && work.host_checks == chunks,
         fixture.name + ": one control interaction occurs per chunk");
  expect(work.kernel_dispatches == 1U + 2U * queued_pairs,
         fixture.name + ": chunked counts init and every queued dispatch");
}

void run_fixture_matrix(const bfnew::test::JacobiFixtureCase& fixture) {
  const PreparedFixture prepared = prepare_fixture(fixture);
  const BoundedOracle oracle =
      bounded_oracle(fixture.partitioned.graph, fixture.query);
  const std::uint64_t selected_vertices = bfnew::estimate_selected_vertex_count(
      fixture.partitioned.graph, fixture.query.selected_tiles);
  const std::uint64_t selected_edges = bfnew::estimate_selected_edge_count(
      fixture.partitioned.graph, fixture.query.selected_tiles);

  std::vector<float> first_control_distances;
  for (const bfnew::GpuRunOptions& options : control_matrix()) {
    const bfnew::HostJacobiRunResult run = bfnew::run_host_jacobi_pull(
        prepared.layout,
        fixture.query,
        prepared.tile_masks,
        prepared.run_masks.csc_run_masks,
        options);
    expect(
        run.result.engine_kind ==
                static_cast<std::uint32_t>(bfnew::EngineKind::jacobi_pull) &&
            run.result.control_mode ==
                static_cast<std::uint32_t>(options.control_mode),
        fixture.name + ": result retains exact engine/control identity");
    expect(
        bfnew::validate_device_controller(run.controller) ==
                bfnew::DeviceControllerError::none &&
            bfnew::validate_device_run_status(run.result.status) ==
                bfnew::DeviceRunStatusError::none,
        fixture.name + ": controller and status pass terminal validation");
    expect(
        run.result.status.reached_target_mask == fixture.expected_reached_mask &&
            run.result.status.bounding_box_miss_mask == fixture.expected_miss_mask,
        fixture.name + ": reached/miss masks match bounded semantics");
    expect(
        run.result.status.stop_reason ==
            static_cast<std::uint32_t>(bfnew::DeviceStopReason::converged),
        fixture.name + ": normal fixture converges without an algorithm error");
    expect(
        run.result.status.final_distance_slot ==
            run.controller.distance_read_slot,
        fixture.name + ": final slot comes from the controller");
    if (fixture.expected_rounds != 0U) {
      expect(
          run.controller.rounds_completed == fixture.expected_rounds,
          fixture.name + ": expected complete-round count");
    }
    expect(
        run.controller.distance_read_slot ==
            static_cast<std::uint32_t>(run.controller.rounds_completed & 1U),
        fixture.name + ": final parity follows executed rounds, not requested K");
    expect(
        vectors_bitwise_equal(run.distance_slots[0], run.distance_slots[1]),
        fixture.name + ": convergence leaves complete Jacobi columns identical");
    compare_with_oracle(fixture, run.distances, oracle.global_distances);

    if (first_control_distances.empty()) {
      first_control_distances = run.distances;
    } else {
      expect(
          vectors_bitwise_equal(run.distances, first_control_distances),
          fixture.name + ": all control modes are bitwise identical");
    }
    for (const bfnew::VertexId source : fixture.query.sources) {
      expect(
          bitwise_equal(run.distances[source.value()], 0.0F),
          fixture.name + ": every canonical source remains exact zero");
    }
    expect(
        run.result.work.edges_examined ==
            selected_edges * run.controller.rounds_completed,
        fixture.name + ": only admitted CSC edges are examined each real round");
    expect(
        run.result.work.active_vertices ==
            selected_vertices * run.controller.rounds_completed,
        fixture.name + ": only admitted destinations are processed each real round");
    check_control_accounting(fixture, options, run);
  }

  const bfnew::VertexId primary_target = fixture.query.targets.front();
  if (fixture.expected_bounded_primary_target) {
    expect(
        bitwise_equal(
            oracle.global_distances[primary_target.value()],
            *fixture.expected_bounded_primary_target),
        fixture.name + ": documented bounded target cost");
  }
  if (fixture.expected_unbounded_primary_target) {
    const bfnew::SsspResult unbounded = bfnew::dijkstra_oracle(
        fixture.partitioned.graph, fixture.query.sources);
    expect(
        bitwise_equal(
            unbounded.distances[primary_target.value()],
            *fixture.expected_unbounded_primary_target),
        fixture.name + ": documented unbounded comparison cost");
  }
}

void test_scratch_and_shared_controller_transition() {
  const bfnew::JacobiScratchLayout layout =
      bfnew::make_jacobi_scratch_layout(17U);
  expect(layout.vertex_count == 17U && layout.distance_slot_bytes == 68U &&
             layout.distance_slot_offsets[0] == 0U &&
             layout.distance_slot_offsets[1] == 68U && layout.total_bytes == 136U,
         "Jacobi scratch planner exposes exactly two adjacent float columns");
  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(bfnew::make_jacobi_scratch_layout(0U)); },
      "Jacobi scratch planner rejects an empty distance space");
  expect_throws<std::overflow_error>(
      [] {
        static_cast<void>(bfnew::make_jacobi_scratch_layout(
            std::numeric_limits<std::size_t>::max()));
      },
      "Jacobi scratch planner rejects an unrepresentable device vertex count");

  bfnew::GpuRunOptions options;
  options.engine = bfnew::EngineKind::jacobi_pull;
  options.maximum_rounds = 2U;
  bfnew::DeviceController controller =
      bfnew::initialize_device_controller(options, 1U);
  std::array<float, 34U> scratch{};
  const bfnew::JacobiDistanceView initial = bfnew::bind_jacobi_distance_view(
      scratch.data(), sizeof(scratch), 17U, controller);
  expect(bfnew::valid_jacobi_distance_view(initial) &&
             initial.read_distances == scratch.data() &&
             initial.write_distances == scratch.data() + 17U,
         "typed Jacobi view binds distinct const-read and mutable-write columns");
  expect(
      !bfnew::valid_jacobi_distance_view(bfnew::bind_jacobi_distance_view(
          scratch.data(), sizeof(float) * 33U, 17U, controller)),
      "typed Jacobi view rejects an undersized scratch allocation");

  controller.changed_lane_mask = 1U;
  expect(
      bfnew::advance_jacobi_controller(controller) ==
              bfnew::JacobiAdvanceResult::continue_execution &&
          controller.rounds_completed == 1U &&
          controller.distance_read_slot == 1U &&
          controller.distance_write_slot == 0U,
      "shared controller transition swaps after a changing round");
  controller.changed_lane_mask = 0U;
  expect(
      bfnew::advance_jacobi_controller(controller) ==
              bfnew::JacobiAdvanceResult::converged &&
          controller.rounds_completed == 2U && controller.done == 1U &&
          controller.stop_reason ==
              static_cast<std::uint32_t>(bfnew::DeviceStopReason::converged),
      "no-change convergence takes precedence at the exact maximum round");
  const bfnew::DeviceController terminal = controller;
  expect(
      bfnew::advance_jacobi_controller(controller) ==
              bfnew::JacobiAdvanceResult::no_op &&
          std::memcmp(&terminal, &controller, sizeof(controller)) == 0,
      "advance is a byte-preserving no-op after done");

  bfnew::DeviceController invalid =
      bfnew::initialize_device_controller(options, 1U);
  invalid.changed_lane_mask = 2U;
  expect(
      bfnew::advance_jacobi_controller(invalid) ==
              bfnew::JacobiAdvanceResult::invalid_controller_state &&
          invalid.error_bits == bfnew::device_error::invalid_controller_state &&
          bfnew::validate_device_controller(invalid) ==
              bfnew::DeviceControllerError::none,
      "shared transition converts invalid pre-round state into an explicit device error");
  const bfnew::DeviceRunStatus error_status =
      bfnew::make_jacobi_run_status(invalid, 1U, 1U);
  expect(error_status.reached_target_mask == 0U &&
             error_status.bounding_box_miss_mask == 0U,
         "controller errors cannot be reported as bounding-box misses");

  std::vector<bfnew::DeviceController> corrupt_controllers;
  bfnew::DeviceController corrupt_slot =
      bfnew::initialize_device_controller(options, 1U);
  corrupt_slot.distance_read_slot = 2U;
  corrupt_controllers.push_back(corrupt_slot);
  bfnew::DeviceController corrupt_limit =
      bfnew::initialize_device_controller(options, 1U);
  corrupt_limit.maximum_rounds = 0U;
  corrupt_controllers.push_back(corrupt_limit);
  bfnew::DeviceController corrupt_engine =
      bfnew::initialize_device_controller(options, 1U);
  corrupt_engine.engine_kind = 99U;
  corrupt_controllers.push_back(corrupt_engine);
  bfnew::DeviceController corrupt_lanes =
      bfnew::initialize_device_controller(options, 1U);
  corrupt_lanes.valid_lane_mask = 0U;
  corrupt_controllers.push_back(corrupt_lanes);
  for (bfnew::DeviceController& corrupt : corrupt_controllers) {
    expect(
        bfnew::advance_jacobi_controller(corrupt) ==
                bfnew::JacobiAdvanceResult::invalid_controller_state &&
            bfnew::validate_device_controller(corrupt) ==
                bfnew::DeviceControllerError::none,
        "every malformed controller becomes a validator-clean terminal error");
    const bfnew::DeviceRunStatus status =
        bfnew::make_jacobi_run_status(corrupt, 1U, 1U);
    expect(
        bfnew::validate_device_run_status(status) ==
                bfnew::DeviceRunStatusError::none &&
            status.reached_target_mask == 0U &&
            status.bounding_box_miss_mask == 0U,
        "malformed controller errors produce safe non-miss status");
  }
}

void test_queued_no_ops_and_maximum_rounds() {
  const bfnew::test::JacobiFixtureCase short_chain =
      bfnew::test::make_parallel_jacobi_fixture();
  const PreparedFixture short_prepared = prepare_fixture(short_chain);
  bfnew::GpuRunOptions chunked;
  chunked.engine = bfnew::EngineKind::jacobi_pull;
  chunked.control_mode = bfnew::ControlMode::chunked_host_poll;
  chunked.rounds_per_chunk = 32U;
  chunked.maximum_rounds = 64U;
  const bfnew::HostJacobiRunResult no_ops = bfnew::run_host_jacobi_pull(
      short_prepared.layout,
      short_chain.query,
      short_prepared.tile_masks,
      short_prepared.run_masks.csc_run_masks,
      chunked);
  expect(no_ops.controller.rounds_completed == 3U &&
             no_ops.queued_round_pairs == 32U &&
             no_ops.result.work.kernel_dispatches == 65U &&
             no_ops.result.status.final_distance_slot == 1U,
         "K=32 queues 29 cheap no-op pairs and preserves actual-round parity");

  const bfnew::test::JacobiFixtureCase long_chain =
      bfnew::test::make_long_chain_jacobi_fixture();
  const PreparedFixture long_prepared = prepare_fixture(long_chain);
  for (const bfnew::ControlMode mode : {
           bfnew::ControlMode::persistent_cooperative,
           bfnew::ControlMode::per_round_host_poll,
           bfnew::ControlMode::chunked_host_poll,
       }) {
    bfnew::GpuRunOptions limited;
    limited.engine = bfnew::EngineKind::jacobi_pull;
    limited.control_mode = mode;
    limited.rounds_per_chunk = 32U;
    limited.maximum_rounds = 2U;
    const bfnew::HostJacobiRunResult exhausted = bfnew::run_host_jacobi_pull(
        long_prepared.layout,
        long_chain.query,
        long_prepared.tile_masks,
        long_prepared.run_masks.csc_run_masks,
        limited);
    expect(exhausted.controller.rounds_completed == 2U &&
               exhausted.result.status.converged == 0U &&
               exhausted.result.status.stop_reason ==
                   static_cast<std::uint32_t>(
                       bfnew::DeviceStopReason::maximum_rounds) &&
               exhausted.result.status.final_distance_slot == 0U &&
               exhausted.result.status.reached_target_mask == 0U &&
               exhausted.result.status.bounding_box_miss_mask == 0U,
           "every control mode reports exact round exhaustion without a box miss");
    if (mode == bfnew::ControlMode::chunked_host_poll) {
      expect(
          exhausted.queued_round_pairs == 32U &&
              exhausted.completed_host_chunks == 1U &&
              exhausted.result.work.kernel_dispatches == 65U,
          "maximum-round chunk queues all K pairs and makes 30 post-stop no-ops");
    }
  }
}

void test_instrumentation_levels() {
  const bfnew::test::JacobiFixtureCase fixture =
      bfnew::test::make_parallel_jacobi_fixture();
  const PreparedFixture prepared = prepare_fixture(fixture);
  const auto run_at = [&](const bfnew::InstrumentationLevel level) {
    bfnew::GpuRunOptions options;
    options.engine = bfnew::EngineKind::jacobi_pull;
    options.control_mode = bfnew::ControlMode::per_round_host_poll;
    options.maximum_rounds = 8U;
    options.instrumentation = level;
    return bfnew::run_host_jacobi_pull(
        prepared.layout,
        fixture.query,
        prepared.tile_masks,
        prepared.run_masks.csc_run_masks,
        options);
  };

  const bfnew::HostJacobiRunResult none =
      run_at(bfnew::InstrumentationLevel::none);
  const bfnew::HostJacobiRunResult light =
      run_at(bfnew::InstrumentationLevel::light);
  const bfnew::HostJacobiRunResult debug =
      run_at(bfnew::InstrumentationLevel::debug);
  expect(
      vectors_bitwise_equal(none.distances, light.distances) &&
          vectors_bitwise_equal(light.distances, debug.distances),
      "instrumentation level never changes Jacobi distances");
  expect(
      none.result.work.edges_examined == 0U &&
          none.result.work.successful_decreases == 0U &&
          none.result.work.active_vertices == 0U &&
          none.result.work.active_lane_rounds == 0U &&
          none.result.work.mask_operations == 0U,
      "None suppresses device-style algorithm counters");
  expect(
      light.result.work.edges_examined != 0U &&
          light.result.work.successful_decreases != 0U &&
          light.result.work.active_vertices != 0U &&
          light.result.work.active_lane_rounds != 0U &&
          light.result.work.mask_operations == 0U,
      "Light records basic work without debug mask operations");
  expect(
      debug.result.work.edges_examined == light.result.work.edges_examined &&
          debug.result.work.successful_decreases ==
              light.result.work.successful_decreases &&
          debug.result.work.active_vertices == light.result.work.active_vertices &&
          debug.result.work.active_lane_rounds ==
              light.result.work.active_lane_rounds &&
          debug.result.work.mask_operations != 0U,
      "Debug adds mask operations without changing Light counters");
}

void test_hip_source_contract_if_configured() {
#if defined(BFNEW_PHASE9_HIP_SOURCE_PATH)
  std::ifstream input{BFNEW_PHASE9_HIP_SOURCE_PATH};
  expect(input.good(), "configured Phase 9 HIP source opens for structural audit");
  const std::string source{
      std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  const auto contains = [&source](const std::string_view marker) {
    return source.find(marker) != std::string::npos;
  };
  for (const std::string_view marker : {
           "graph.csc",
           "column_run_offsets",
           "run_edge_offsets",
           "run_source_tiles",
           "read_distances",
           "write_distances",
           "atomicOr",
           "this_grid",
           "grid.sync",
           "hipLaunchCooperativeKernel",
           "hipOccupancyMaxActiveBlocksPerMultiprocessor",
       }) {
    expect(contains(marker), std::string{"HIP structural marker missing: "} +
                                 std::string{marker});
  }
  expect(!contains("atomicMin"),
         "Jacobi HIP source contains no atomic distance minimum");
  expect(!contains("atomicCAS"),
         "Jacobi HIP source contains no CAS-based distance update");
  const std::size_t round_begin =
      source.find("__device__ void perform_jacobi_round");
  const std::size_t round_end =
      source.find("__device__ void initialize_jacobi_state", round_begin);
  const std::string round_source =
      round_begin != std::string::npos && round_end != std::string::npos
          ? source.substr(round_begin, round_end - round_begin)
          : std::string{};
  expect(
      round_begin != std::string::npos && round_end != std::string::npos &&
          round_source.find("graph.csr") == std::string::npos,
      "Jacobi round kernel structurally avoids outgoing CSR");
  expect(
      round_source.find("target") == std::string::npos,
      "Jacobi round kernel has no target-based early-stop path");
  expect(
      round_source.find("destination = global_thread") != std::string::npos &&
          round_source.find("destination += grid_threads") != std::string::npos,
      "Jacobi round structurally assigns grid-stride destination owners");
  expect(
      round_source.find("__syncthreads") != std::string::npos &&
          round_source.find("threadIdx.x == 0U") != std::string::npos &&
          round_source.find("atomicOr") != std::string::npos,
      "Jacobi changed state is structurally reduced once per block");

  const std::size_t initialization_begin = source.find(
      "__device__ void initialize_jacobi_state");
  const std::size_t initialization_end =
      source.find("__global__ void initialize_jacobi_query", initialization_begin);
  const std::string initialization_source =
      initialization_begin != std::string::npos &&
              initialization_end != std::string::npos
          ? source.substr(
                initialization_begin,
                initialization_end - initialization_begin)
          : std::string{};
  expect(
      initialization_source.find("graph.owner_tiles") != std::string::npos &&
          initialization_source.find("column_run_offsets") !=
              std::string::npos &&
          initialization_source.find("run_source_tiles") != std::string::npos &&
          initialization_source.find("destination_mask &") != std::string::npos,
      "Jacobi initialization structurally materializes exact endpoint run masks");

  const std::size_t persistent_begin = source.find(
      "if (options.control_mode == ControlMode::persistent_cooperative)");
  const std::size_t persistent_end = source.find("} else {", persistent_begin);
  const std::string persistent_driver =
      persistent_begin != std::string::npos && persistent_end != std::string::npos
          ? source.substr(persistent_begin, persistent_end - persistent_begin)
          : std::string{};
  expect(
      persistent_driver.find("launch_persistent") != std::string::npos &&
          persistent_driver.find("launch_initialize") == std::string::npos &&
          persistent_driver.find("hipMemcpy") == std::string::npos &&
          persistent_driver.find("synchronize") == std::string::npos,
      "persistent driver has one cooperative launch and no convergence polling");
  expect(!contains("hipDeviceSynchronize"),
         "Jacobi engine does not introduce a device-wide synchronization");
  expect(
      contains("options.control_mode != ControlMode::persistent_cooperative") &&
          contains("launch_initialize") && contains("launch_persistent"),
      "persistent source structurally excludes the ordinary initialization launch");
#endif
}

}  // namespace

int main() {
  test_scratch_and_shared_controller_transition();
  for (const bfnew::test::JacobiFixtureCase& fixture :
       bfnew::test::make_jacobi_fixture_suite()) {
    run_fixture_matrix(fixture);
  }
  test_queued_no_ops_and_maximum_rounds();
  test_instrumentation_levels();
  test_hip_source_contract_if_configured();

  std::cout << "Phase 9 seeded-random maximum observed difference: "
            << maximum_observed_random_ulps << " ULP\n";
  if (failures != 0) {
    std::cerr << failures << " Jacobi pull assertion(s) failed\n";
    return 1;
  }
  return 0;
}
