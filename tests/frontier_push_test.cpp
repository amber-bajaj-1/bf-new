#include "bfnew/device_layout.hpp"
#include "bfnew/frontier_push.hpp"
#include "bfnew/query.hpp"
#include "bfnew/sssp.hpp"
#include "frontier_fixture_suite.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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
      fixture.name + ": CSR run masks equal endpoint admission");
  return prepared;
}

[[nodiscard]] std::vector<bfnew::GpuRunOptions> control_matrix() {
  std::vector<bfnew::GpuRunOptions> controls;
  controls.reserve(7U);
  bfnew::GpuRunOptions persistent;
  persistent.engine = bfnew::EngineKind::frontier_push;
  persistent.control_mode = bfnew::ControlMode::persistent_cooperative;
  persistent.maximum_rounds = 128U;
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
      fixture.name + ": general weights agree with Dijkstra within 4 ULP");
}

void check_control_accounting(
    const bfnew::test::JacobiFixtureCase& fixture,
    const bfnew::GpuRunOptions& options,
    const bfnew::HostFrontierRunResult& run) {
  const std::uint64_t rounds = run.controller.rounds_completed;
  const bfnew::DeviceWorkStatistics& work = run.result.work;
  expect(
      work.active_lane_rounds == rounds &&
          run.current_frontier_sizes.size() == rounds &&
          work.active_vertices == std::accumulate(
                                      run.current_frontier_sizes.begin(),
                                      run.current_frontier_sizes.end(),
                                      std::uint64_t{0U}),
      fixture.name + ": executed frontier-round accounting is inconsistent");
  expect(
      work.empty_frontier_rounds == 1U &&
          work.small_frontier_rounds == static_cast<std::uint64_t>(
              std::ranges::count_if(
                  run.current_frontier_sizes,
                  [](const std::uint64_t size) {
                    return size != 0U && size < 32U;
                  })) &&
          work.maximum_queue_size <= fixture.partitioned.graph.vertex_count(),
      fixture.name + ": converged frontier did not end with one empty next queue");
  expect(
      work.full_edge_rounds == 0U && work.changed_flag_updates == 0U,
      fixture.name + ": frontier result leaked dense-scan counters");
  if (options.control_mode == bfnew::ControlMode::persistent_cooperative) {
    expect(
        work.kernel_dispatches == 1U && work.controller_copies == 1U &&
            work.host_synchronizations == 1U && work.host_checks == 1U &&
            run.queued_round_pairs == 0U,
        fixture.name + ": persistent semantic model polled frontier size");
    return;
  }
  if (options.control_mode == bfnew::ControlMode::per_round_host_poll) {
    expect(
        run.queued_round_pairs == rounds &&
            run.completed_host_chunks == rounds &&
            work.controller_copies == rounds &&
            work.host_synchronizations == rounds && work.host_checks == rounds &&
            work.kernel_dispatches == 1U + 2U * rounds,
        fixture.name + ": per-round frontier control accounting is wrong");
    return;
  }
  const std::uint64_t k = options.rounds_per_chunk;
  const std::uint64_t chunks = (rounds + k - 1U) / k;
  expect(
      run.queued_round_pairs == chunks * k &&
          run.completed_host_chunks == chunks &&
          work.controller_copies == chunks &&
          work.host_synchronizations == chunks && work.host_checks == chunks &&
          work.kernel_dispatches == 1U + 2U * chunks * k,
      fixture.name + ": chunked frontier control did not retain full K chunks");
}

void run_fixture_matrix(const bfnew::test::JacobiFixtureCase& fixture) {
  const PreparedFixture prepared = prepare_fixture(fixture);
  const std::vector<float> oracle =
      bounded_oracle(fixture.partitioned.graph, fixture.query);
  std::vector<float> control_baseline;
  for (const bfnew::GpuRunOptions& options : control_matrix()) {
    const bfnew::HostFrontierRunResult run = bfnew::run_host_frontier_push(
        prepared.layout,
        fixture.query,
        prepared.tile_masks,
        prepared.run_masks.csr_run_masks,
        options);
    expect(
        run.result.engine_kind ==
                static_cast<std::uint32_t>(bfnew::EngineKind::frontier_push) &&
            run.result.control_mode ==
                static_cast<std::uint32_t>(options.control_mode),
        fixture.name + ": result lost frontier engine/control identity");
    expect(
        bfnew::validate_device_controller(run.controller) ==
                bfnew::DeviceControllerError::none &&
            bfnew::validate_device_run_status(run.result.status) ==
                bfnew::DeviceRunStatusError::none,
        fixture.name + ": terminal frontier controller/status is invalid");
    expect(
        run.result.status.stop_reason == static_cast<std::uint32_t>(
                                             bfnew::DeviceStopReason::converged) &&
            run.result.status.final_distance_slot == 0U &&
            run.result.status.reached_target_mask ==
                fixture.expected_reached_mask &&
            run.result.status.bounding_box_miss_mask ==
                fixture.expected_miss_mask,
        fixture.name + ": frontier completion/reachability status is wrong");
    expect(
        run.controller.frontier_read_slot ==
                run.controller.rounds_completed % 2U &&
            run.controller.frontier_write_slot ==
                1U - run.controller.frontier_read_slot &&
            run.controller.frontier_size[run.controller.frontier_read_slot] ==
                0U,
        fixture.name + ": controller lost actual frontier-slot parity");
    expect(
        !run.current_frontier_sizes.empty() &&
            run.current_frontier_sizes.front() == fixture.query.sources.size(),
        fixture.name + ": initial frontier is not the canonical source set");
    expect(
        run.result.work.atomic_attempts == run.result.work.edges_examined &&
            run.result.work.successful_atomic_updates ==
                run.result.work.successful_decreases &&
            run.result.work.queue_claims +
                    run.result.work.duplicate_suppressions ==
                run.result.work.successful_atomic_updates &&
            (run.result.work.edges_examined == 0U ||
             run.result.work.mask_operations != 0U) &&
            run.result.work.overflow_events == 0U,
        fixture.name + ": atomic/overflow instrumentation is inconsistent");
    compare_with_oracle(fixture, run.distances, oracle);
    if (control_baseline.empty()) {
      control_baseline = run.distances;
    } else {
      expect(
          vectors_bitwise_equal(control_baseline, run.distances),
          fixture.name + ": control mode changed final frontier distance bits");
    }
    check_control_accounting(fixture, options, run);
  }
}

void test_scratch_layout() {
  const bfnew::FrontierScratchLayout layout =
      bfnew::make_frontier_scratch_layout(5U, 3U);
  expect(
      layout.vertex_count == 5U && layout.queue_capacity == 3U &&
          layout.distance_bits_offset == 0U &&
          layout.distance_bits_bytes == 20U &&
          layout.frontier_offsets[0] == 20U &&
          layout.frontier_offsets[1] == 32U &&
          layout.frontier_bytes_each == 12U &&
          layout.enqueue_generation_offset == 48U &&
          layout.enqueue_generation_bytes == 40U &&
          layout.total_bytes == 88U,
      "frontier scratch component offsets are exact and aligned");
  std::vector<std::uint64_t> storage((layout.total_bytes + 7U) / 8U);
  const bfnew::FrontierScratchView view = bfnew::bind_frontier_scratch(
      storage.data(), layout.total_bytes, 5U, 3U);
  expect(
      bfnew::valid_frontier_scratch_view(view) &&
          view.queue_capacity == 3U &&
          reinterpret_cast<std::uintptr_t>(view.enqueue_generation) %
                  alignof(std::uint64_t) ==
              0U,
      "frontier scratch binds disjoint aligned queue/generation views");
  expect(
      !bfnew::valid_frontier_scratch_view(bfnew::bind_frontier_scratch(
          storage.data(), layout.total_bytes - 1U, 5U, 3U)),
      "frontier scratch rejects an undersized allocation");
  expect(
      !bfnew::valid_frontier_scratch_view(bfnew::bind_frontier_scratch(
          storage.data(), layout.total_bytes, 5U, 6U)),
      "frontier scratch binder rejects capacity above the vertex bound");
  auto* const misaligned =
      reinterpret_cast<std::uint8_t*>(storage.data()) + 1U;
  expect(
      !bfnew::valid_frontier_scratch_view(bfnew::bind_frontier_scratch(
          misaligned, layout.total_bytes - 1U, 5U, 3U)),
      "frontier scratch binder rejects a misaligned base pointer");
  const bfnew::FrontierScratchLayout default_capacity =
      bfnew::make_frontier_scratch_layout(5U);
  expect(
      default_capacity.queue_capacity == 5U,
      "zero requested queue capacity selects one slot per vertex");
  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(bfnew::make_frontier_scratch_layout(0U)); },
      "frontier scratch rejects an empty graph");
  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(bfnew::make_frontier_scratch_layout(4U, 5U)); },
      "frontier scratch rejects capacity above the unique-vertex bound");
}

void test_frontier_controller_transition() {
  bfnew::GpuRunOptions options;
  options.engine = bfnew::EngineKind::frontier_push;
  options.maximum_rounds = 4U;
  bfnew::DeviceController controller =
      bfnew::initialize_device_controller(options, 1U, 1U);
  controller.frontier_size[1] = 2U;
  controller.next_frontier_lane_mask = 1U;
  expect(
      bfnew::advance_frontier_controller(controller) ==
              bfnew::FrontierAdvanceResult::continue_execution &&
          controller.rounds_completed == 1U &&
          controller.frontier_read_slot == 1U &&
          controller.frontier_write_slot == 0U &&
          controller.frontier_size[0] == 0U,
      "frontier controller swaps queues and clears the recycled size");
  expect(
      bfnew::advance_frontier_controller(controller) ==
              bfnew::FrontierAdvanceResult::converged &&
          controller.rounds_completed == 2U && controller.done == 1U,
      "an executed round with no next frontier converges");
  const bfnew::DeviceController terminal = controller;
  expect(
      bfnew::advance_frontier_controller(controller) ==
              bfnew::FrontierAdvanceResult::no_op &&
          std::memcmp(&terminal, &controller, sizeof(controller)) == 0,
      "terminal frontier advances are byte-preserving no-ops");

  bfnew::DeviceController overflow =
      bfnew::initialize_device_controller(options, 1U, 1U);
  overflow.frontier_size[1] = 3U;
  overflow.next_frontier_lane_mask = 1U;
  overflow.error_bits = bfnew::device_error::queue_overflow;
  expect(
      bfnew::advance_frontier_controller(overflow) ==
              bfnew::FrontierAdvanceResult::queue_overflow &&
          overflow.rounds_completed == 1U &&
          overflow.stop_reason == static_cast<std::uint32_t>(
                                      bfnew::DeviceStopReason::queue_overflow) &&
          bfnew::validate_device_controller(overflow) ==
              bfnew::DeviceControllerError::none,
      "queue overflow is explicit, terminal, and validator-clean");

  bfnew::DeviceController malformed =
      bfnew::initialize_device_controller(options, 1U, 1U);
  malformed.frontier_write_slot = 0U;
  expect(
      bfnew::advance_frontier_controller(malformed) ==
              bfnew::FrontierAdvanceResult::invalid_controller_state &&
          bfnew::validate_device_controller(malformed) ==
              bfnew::DeviceControllerError::none,
      "malformed frontier state becomes a canonical terminal error");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            bfnew::initialize_device_controller(options, 1U, 0U));
      },
      "frontier controller initialization rejects an empty initial queue");
  bfnew::DeviceController empty_live =
      bfnew::initialize_device_controller(options, 1U, 1U);
  empty_live.frontier_size[empty_live.frontier_read_slot] = 0U;
  expect(
      bfnew::validate_device_controller(empty_live) ==
          bfnew::DeviceControllerError::invalid_frontier_slots,
      "the common validator rejects a live frontier without current work");
}

void test_frontier_shapes_and_deduplication() {
  const bfnew::test::JacobiFixtureCase chain =
      bfnew::test::make_long_chain_jacobi_fixture();
  const PreparedFixture chain_prepared = prepare_fixture(chain);
  bfnew::GpuRunOptions options;
  options.engine = bfnew::EngineKind::frontier_push;
  options.control_mode = bfnew::ControlMode::per_round_host_poll;
  options.maximum_rounds = 128U;
  options.instrumentation = bfnew::InstrumentationLevel::debug;
  const bfnew::HostFrontierRunResult chain_run =
      bfnew::run_host_frontier_push(
          chain_prepared.layout,
          chain.query,
          chain_prepared.tile_masks,
          chain_prepared.run_masks.csr_run_masks,
          options);
  expect(
      std::ranges::all_of(
          chain_run.current_frontier_sizes,
          [](const std::uint64_t size) { return size == 1U; }),
      "sparse wavefront chain keeps exactly one active vertex per round");

  const bfnew::test::JacobiFixtureCase grid =
      bfnew::test::make_expanding_grid_frontier_fixture();
  const PreparedFixture grid_prepared = prepare_fixture(grid);
  const bfnew::HostFrontierRunResult grid_run = bfnew::run_host_frontier_push(
      grid_prepared.layout,
      grid.query,
      grid_prepared.tile_masks,
      grid_prepared.run_masks.csr_run_masks,
      options);
  expect(
      !grid_run.current_frontier_sizes.empty() &&
          *std::max_element(
              grid_run.current_frontier_sizes.begin(),
              grid_run.current_frontier_sizes.end()) >= 5U,
      "grid fixture expands beyond a sparse one-vertex wavefront");

  const bfnew::test::JacobiFixtureCase repeated =
      bfnew::test::make_repeated_improvement_frontier_fixture();
  const PreparedFixture repeated_prepared = prepare_fixture(repeated);
  const bfnew::HostFrontierRunResult repeated_run =
      bfnew::run_host_frontier_push(
          repeated_prepared.layout,
          repeated.query,
          repeated_prepared.tile_masks,
          repeated_prepared.run_masks.csr_run_masks,
          options);
  expect(
      repeated_run.result.work.duplicate_suppressions == 2U &&
          repeated_run.result.work.queue_claims <
              repeated_run.result.work.successful_atomic_updates,
      "later improvements update distance without duplicating the queued vertex");
  const bfnew::HostFrontierRunResult repeated_again =
      bfnew::run_host_frontier_push(
          repeated_prepared.layout,
          repeated.query,
          repeated_prepared.tile_masks,
          repeated_prepared.run_masks.csr_run_masks,
          options);
  expect(
      repeated_again.distance_bits == repeated_run.distance_bits &&
          repeated_again.current_frontier_sizes ==
              repeated_run.current_frontier_sizes,
      "repeated deterministic portable frontier runs reproduce bits and shape");
}

void test_instrumentation_overflow_and_limits() {
  const bfnew::test::JacobiFixtureCase fixture =
      bfnew::test::make_repeated_improvement_frontier_fixture();
  const PreparedFixture prepared = prepare_fixture(fixture);
  const auto run_at = [&](const bfnew::InstrumentationLevel level) {
    bfnew::GpuRunOptions options;
    options.engine = bfnew::EngineKind::frontier_push;
    options.control_mode = bfnew::ControlMode::per_round_host_poll;
    options.maximum_rounds = 64U;
    options.instrumentation = level;
    return bfnew::run_host_frontier_push(
        prepared.layout,
        fixture.query,
        prepared.tile_masks,
        prepared.run_masks.csr_run_masks,
        options);
  };
  const bfnew::HostFrontierRunResult none =
      run_at(bfnew::InstrumentationLevel::none);
  const bfnew::HostFrontierRunResult light =
      run_at(bfnew::InstrumentationLevel::light);
  const bfnew::HostFrontierRunResult debug =
      run_at(bfnew::InstrumentationLevel::debug);
  expect(
      none.distance_bits == light.distance_bits &&
          light.distance_bits == debug.distance_bits,
      "frontier instrumentation does not alter final labels");
  expect(
      none.result.work.edges_examined == 0U &&
          none.result.work.active_vertices == 0U &&
          none.result.work.maximum_queue_size == 0U,
      "None suppresses frontier algorithm counters");
  expect(
      light.result.work.edges_examined != 0U &&
          light.result.work.active_vertices != 0U &&
          light.result.work.empty_frontier_rounds == 1U &&
          light.result.work.atomic_attempts == 0U &&
          light.result.work.queue_claims == 0U,
      "Light records frontier shape/work without expensive atomic counters");
  expect(
      debug.result.work.edges_examined == light.result.work.edges_examined &&
          debug.result.work.atomic_attempts == debug.result.work.edges_examined &&
          debug.result.work.successful_atomic_updates ==
              debug.result.work.successful_decreases &&
          debug.result.work.queue_claims != 0U &&
          debug.result.work.duplicate_suppressions == 2U,
      "Debug extends Light with atomic, claim, duplicate, and mask counters");

  const bfnew::test::JacobiFixtureCase grid =
      bfnew::test::make_expanding_grid_frontier_fixture();
  const PreparedFixture grid_prepared = prepare_fixture(grid);
  for (const bfnew::ControlMode mode : {
           bfnew::ControlMode::persistent_cooperative,
           bfnew::ControlMode::per_round_host_poll,
           bfnew::ControlMode::chunked_host_poll,
       }) {
    bfnew::GpuRunOptions overflow_options;
    overflow_options.engine = bfnew::EngineKind::frontier_push;
    overflow_options.control_mode = mode;
    overflow_options.rounds_per_chunk = 32U;
    overflow_options.maximum_rounds = 64U;
    overflow_options.instrumentation = bfnew::InstrumentationLevel::debug;
    const bfnew::HostFrontierRunResult overflow =
        bfnew::run_host_frontier_push(
            grid_prepared.layout,
            grid.query,
            grid_prepared.tile_masks,
            grid_prepared.run_masks.csr_run_masks,
            overflow_options,
            1U);
    expect(
        overflow.result.status.stop_reason == static_cast<std::uint32_t>(
                                                  bfnew::DeviceStopReason::queue_overflow) &&
            overflow.result.status.bounding_box_miss_mask == 0U &&
            overflow.result.status.rounds_completed == 1U &&
            overflow.result.work.overflow_events == 1U &&
            overflow.result.work.maximum_queue_size > 1U,
        "queue overflow is detected without an out-of-bounds write or miss");
    if (mode == bfnew::ControlMode::chunked_host_poll) {
      expect(
          overflow.queued_round_pairs == 32U &&
              overflow.result.work.kernel_dispatches == 65U,
          "K=32 overflow retains its already-queued no-op pairs");
    }
  }

  const bfnew::test::JacobiFixtureCase chain =
      bfnew::test::make_long_chain_jacobi_fixture();
  const PreparedFixture chain_prepared = prepare_fixture(chain);
  for (const bfnew::ControlMode mode : {
           bfnew::ControlMode::persistent_cooperative,
           bfnew::ControlMode::per_round_host_poll,
           bfnew::ControlMode::chunked_host_poll,
       }) {
    bfnew::GpuRunOptions limited;
    limited.engine = bfnew::EngineKind::frontier_push;
    limited.control_mode = mode;
    limited.rounds_per_chunk = 32U;
    limited.maximum_rounds = 1U;
    const bfnew::HostFrontierRunResult run = bfnew::run_host_frontier_push(
        chain_prepared.layout,
        chain.query,
        chain_prepared.tile_masks,
        chain_prepared.run_masks.csr_run_masks,
        limited);
    expect(
        run.result.status.stop_reason == static_cast<std::uint32_t>(
                                             bfnew::DeviceStopReason::maximum_rounds) &&
            run.result.status.rounds_completed == 1U &&
            run.result.status.bounding_box_miss_mask == 0U,
        "all frontier controls distinguish maximum rounds from a box miss");
  }

  bfnew::RouteQuery invalid = fixture.query;
  invalid.sources.clear();
  bfnew::GpuRunOptions invalid_options;
  invalid_options.engine = bfnew::EngineKind::frontier_push;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_frontier_push(
            prepared.layout,
            invalid,
            prepared.tile_masks,
            prepared.run_masks.csr_run_masks,
            invalid_options));
      },
      "an empty initial frontier is rejected as invalid input");
  bfnew::RouteQuery duplicated = fixture.query;
  duplicated.sources.push_back(duplicated.sources.front());
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_frontier_push(
            prepared.layout,
            duplicated,
            prepared.tile_masks,
            prepared.run_masks.csr_run_masks,
            invalid_options));
      },
      "duplicate initial sources are rejected instead of duplicated in queue");
}

void test_hip_source_contract_if_configured() {
#if defined(BFNEW_PHASE11_HIP_SOURCE_PATH)
  std::ifstream input{BFNEW_PHASE11_HIP_SOURCE_PATH};
  expect(input.good(), "configured frontier HIP source opens for structural audit");
  const std::string source{
      std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  const auto contains = [&source](const std::string_view marker) {
    return source.find(marker) != std::string::npos;
  };
  for (const std::string_view marker : {
           "graph.csr",
           "row_run_offsets",
           "run_edge_offsets",
           "run_destination_tiles",
           "atomic_load_nonnegative_float_bits",
           "atomic_min_nonnegative_float_bits",
           "atomicCAS",
           "atomicExch",
           "atomicAdd",
           "queue_capacity",
           "duplicate_suppressions",
           "queue_overflow",
           "hipLaunchCooperativeKernel",
           "hipOccupancyMaxActiveBlocksPerMultiprocessor",
           "grid.sync",
       }) {
    expect(
        contains(marker),
        std::string{"frontier HIP structural marker missing: "} +
            std::string{marker});
  }
  const std::size_t round_begin =
      source.find("__device__ void perform_frontier_round");
  const std::size_t round_end =
      source.find("__device__ void initialize_frontier_state", round_begin);
  const std::string round_source =
      round_begin != std::string::npos && round_end != std::string::npos
          ? source.substr(round_begin, round_end - round_begin)
          : std::string{};
  expect(
      round_source.find("graph.csc") == std::string::npos &&
          round_source.find("perform_dense_round") == std::string::npos &&
          round_source.find("prefix") == std::string::npos &&
          round_source.find("virtual_warp") == std::string::npos,
      "frontier round is a simple CSR queue consumer without deferred schedulers");
  expect(
      round_source.find("atomic_load_nonnegative_float_bits") !=
              std::string::npos &&
          round_source.find("atomic_min_nonnegative_float_bits") !=
              std::string::npos &&
          round_source.find("distance_bits[") == std::string::npos,
      "frontier round never mixes ordinary distance loads with atomic writes");
  expect(
      !contains("hipDeviceSynchronize"),
      "frontier engine introduces no device-wide synchronization");
#endif
}

}  // namespace

int main() {
  test_scratch_layout();
  test_frontier_controller_transition();
  for (const bfnew::test::JacobiFixtureCase& fixture :
       bfnew::test::make_frontier_fixture_suite()) {
    run_fixture_matrix(fixture);
  }
  test_frontier_shapes_and_deduplication();
  test_instrumentation_overflow_and_limits();
  test_hip_source_contract_if_configured();

  std::cout << "Phase 11 seeded-random maximum observed difference: "
            << maximum_observed_random_ulps << " ULP\n";
  if (failures != 0) {
    std::cerr << failures << " frontier-push assertion(s) failed\n";
    return 1;
  }
  return 0;
}
