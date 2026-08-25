#include "bfnew/dense_chaotic_push.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/query.hpp"
#include "bfnew/sssp.hpp"
#include "dense_fixture_suite.hpp"

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
  persistent.engine = bfnew::EngineKind::dense_chaotic_push;
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
    const bfnew::HostDenseRunResult& run) {
  const std::uint64_t rounds = run.controller.rounds_completed;
  const bfnew::DeviceWorkStatistics& work = run.result.work;
  expect(
      work.full_edge_rounds == rounds && work.active_lane_rounds == rounds,
      fixture.name + ": every executed round is one complete edge scan");
  expect(
      work.queue_claims == 0U && work.duplicate_suppressions == 0U &&
          work.maximum_queue_size == 0U,
      fixture.name + ": dense execution allocated no frontier semantics");
  if (options.control_mode == bfnew::ControlMode::persistent_cooperative) {
    expect(
        work.kernel_dispatches == 1U && work.controller_copies == 1U &&
            work.host_synchronizations == 1U && work.host_checks == 1U &&
            run.queued_round_pairs == 0U && run.completed_host_chunks == 0U,
        fixture.name + ": persistent semantic model has one query dispatch");
    return;
  }
  if (options.control_mode == bfnew::ControlMode::per_round_host_poll) {
    expect(
        run.queued_round_pairs == rounds &&
            run.completed_host_chunks == rounds &&
            work.controller_copies == rounds &&
            work.host_synchronizations == rounds && work.host_checks == rounds &&
            work.kernel_dispatches == 1U + 2U * rounds,
        fixture.name + ": per-round control accounts for every pair and poll");
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
      fixture.name + ": chunked control retains complete K-pair chunks");
}

void run_fixture_matrix(const bfnew::test::JacobiFixtureCase& fixture) {
  const PreparedFixture prepared = prepare_fixture(fixture);
  const std::vector<float> oracle =
      bounded_oracle(fixture.partitioned.graph, fixture.query);
  const std::uint64_t selected_edges = bfnew::estimate_selected_edge_count(
      fixture.partitioned.graph, fixture.query.selected_tiles);
  std::vector<float> first_control_distances;
  for (const bfnew::GpuRunOptions& options : control_matrix()) {
    const bfnew::HostDenseRunResult run =
        bfnew::run_host_dense_chaotic_push(
            prepared.layout,
            fixture.query,
            prepared.tile_masks,
            prepared.run_masks.csr_run_masks,
            options);
    expect(
        run.result.engine_kind ==
                static_cast<std::uint32_t>(
                    bfnew::EngineKind::dense_chaotic_push) &&
            run.result.control_mode ==
                static_cast<std::uint32_t>(options.control_mode),
        fixture.name + ": result retains dense engine/control identity");
    expect(
        bfnew::validate_device_controller(run.controller) ==
                bfnew::DeviceControllerError::none &&
            bfnew::validate_device_run_status(run.result.status) ==
                bfnew::DeviceRunStatusError::none,
        fixture.name + ": terminal controller/status validates");
    expect(
        run.result.status.stop_reason ==
                static_cast<std::uint32_t>(bfnew::DeviceStopReason::converged) &&
            run.result.status.final_distance_slot == 0U &&
            run.result.status.reached_target_mask ==
                fixture.expected_reached_mask &&
            run.result.status.bounding_box_miss_mask == fixture.expected_miss_mask,
        fixture.name + ": dense completion/reachability status is exact");
    compare_with_oracle(fixture, run.distances, oracle);
    if (first_control_distances.empty()) {
      first_control_distances = run.distances;
    } else {
      expect(
          vectors_bitwise_equal(first_control_distances, run.distances),
          fixture.name + ": controls changed final dense distance bits");
    }
    for (const bfnew::VertexId source : fixture.query.sources) {
      expect(
          run.distance_bits[source.value()] == 0U,
          fixture.name + ": source lost its canonical atomic zero");
    }
    expect(
        run.result.work.edges_examined ==
                selected_edges * run.controller.rounds_completed &&
            run.result.work.atomic_attempts ==
                selected_edges * run.controller.rounds_completed &&
            run.result.work.successful_atomic_updates ==
                run.result.work.successful_decreases,
        fixture.name + ": complete scans and atomic instrumentation disagree");
    check_control_accounting(fixture, options, run);
  }
}

void test_atomic_float_bit_domain() {
  const bfnew::DenseScratchLayout layout = bfnew::make_dense_scratch_layout(17U);
  expect(
      layout.vertex_count == 17U && layout.distance_bits_bytes == 68U &&
          layout.total_bytes == 68U,
      "dense scratch contains exactly one 32-bit word per vertex");
  std::vector<std::uint32_t> scratch(17U);
  const bfnew::DenseDistanceView view = bfnew::bind_dense_distance_view(
      scratch.data(), scratch.size() * sizeof(std::uint32_t), 17U);
  expect(
      bfnew::valid_dense_distance_view(view) &&
          view.distance_bits == scratch.data(),
      "dense scratch binds one in-place atomic-compatible view");
  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(bfnew::make_dense_scratch_layout(0U)); },
      "dense scratch rejects an empty vertex space");

  const std::vector<std::uint32_t> ordered_bits{
      0x00000000U,
      0x00000001U,
      0x007fffffU,
      0x00800000U,
      0x3f000000U,
      0x3f800000U,
      0x40000000U,
      0x7f7fffffU,
      0x7f800000U,
  };
  for (std::size_t index = 0U; index < ordered_bits.size(); ++index) {
    const float value = bfnew::dense_atomic_bits_float(ordered_bits[index]);
    expect(
        bfnew::is_dense_atomic_float(value) &&
            bfnew::dense_atomic_float_bits(value) == ordered_bits[index],
        "supported nonnegative float bits round trip exactly");
    if (index != 0U) {
      const float preceding =
          bfnew::dense_atomic_bits_float(ordered_bits[index - 1U]);
      expect(
          preceding < value && ordered_bits[index - 1U] < ordered_bits[index],
          "unsigned bit order matches float order across IEEE boundaries");
    }
  }
  for (std::uint32_t exponent = 0U; exponent <= 254U; ++exponent) {
    for (const std::uint32_t mantissa :
         {0U, 1U, 0x003fffffU, 0x007fffffU}) {
      const std::uint32_t bits = (exponent << 23U) | mantissa;
      const float value = bfnew::dense_atomic_bits_float(bits);
      expect(
          bfnew::dense_atomic_float_bits(value) == bits,
          "sampled sign-zero IEEE encodings preserve unsigned order identity");
    }
  }
  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(bfnew::dense_atomic_float_bits(-0.0F)); },
      "dense atomic domain rejects negative zero");
  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(bfnew::dense_atomic_float_bits(-1.0F)); },
      "dense atomic domain rejects negative values");
  expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(bfnew::dense_atomic_float_bits(
            std::numeric_limits<float>::quiet_NaN()));
      },
      "dense atomic domain rejects NaN");
  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(bfnew::dense_atomic_bits_float(0x7f800001U)); },
      "dense atomic domain rejects NaN bit patterns");

  const bfnew::DenseAtomicMinResult decrease = bfnew::dense_atomic_min_bits(
      0x7f800000U, bfnew::dense_atomic_float_bits(0.5F));
  const bfnew::DenseAtomicMinResult unchanged = bfnew::dense_atomic_min_bits(
      decrease.final_bits, bfnew::dense_atomic_float_bits(1.0F));
  expect(
      decrease.decreased && bitwise_equal(
                                bfnew::dense_atomic_bits_float(decrease.final_bits),
                                0.5F) &&
          !unchanged.decreased && unchanged.final_bits == decrease.final_bits,
      "portable unsigned atomic-min model is monotonic and strict");
}

void test_dense_controller_transition() {
  bfnew::GpuRunOptions options;
  options.engine = bfnew::EngineKind::dense_chaotic_push;
  options.maximum_rounds = 2U;
  bfnew::DeviceController controller =
      bfnew::initialize_device_controller(options, 1U);
  expect(
      controller.distance_read_slot == 0U &&
          controller.distance_write_slot == 0U,
      "dense controller selects one in-place distance slot");
  controller.changed_lane_mask = 1U;
  expect(
      bfnew::advance_dense_controller(controller) ==
              bfnew::DenseAdvanceResult::continue_execution &&
          controller.rounds_completed == 1U &&
          controller.distance_read_slot == 0U,
      "dense controller advances a changing full scan without swapping");
  controller.changed_lane_mask = 0U;
  expect(
      bfnew::advance_dense_controller(controller) ==
              bfnew::DenseAdvanceResult::converged &&
          controller.rounds_completed == 2U && controller.done == 1U,
      "dense no-change scan converges at the exact round limit");
  const bfnew::DeviceController terminal = controller;
  expect(
      bfnew::advance_dense_controller(controller) ==
              bfnew::DenseAdvanceResult::no_op &&
          std::memcmp(&terminal, &controller, sizeof(controller)) == 0,
      "dense controller is byte-preserving after done");

  bfnew::DeviceController corrupt =
      bfnew::initialize_device_controller(options, 1U);
  corrupt.distance_write_slot = 1U;
  expect(
      bfnew::advance_dense_controller(corrupt) ==
              bfnew::DenseAdvanceResult::invalid_controller_state &&
          bfnew::validate_device_controller(corrupt) ==
              bfnew::DeviceControllerError::none,
      "dense transition canonicalizes malformed state into a valid error");

  for (const std::uint32_t corruption : {0U, 1U, 2U}) {
    bfnew::DeviceController malformed =
        bfnew::initialize_device_controller(options, 1U);
    if (corruption == 0U) {
      malformed.valid_lane_mask = 0U;
    } else if (corruption == 1U) {
      malformed.maximum_rounds = 0U;
    } else {
      malformed.engine_kind =
          static_cast<std::uint32_t>(bfnew::EngineKind::jacobi_pull);
    }
    expect(
        bfnew::advance_dense_controller(malformed) ==
                bfnew::DenseAdvanceResult::invalid_controller_state &&
            bfnew::validate_device_controller(malformed) ==
                bfnew::DeviceControllerError::none &&
            malformed.stop_reason == static_cast<std::uint32_t>(
                                         bfnew::DeviceStopReason::invalid_controller_state),
        "dense transition canonicalizes lane, limit, and engine corruption");
  }
}

void test_schedule_stress_and_repetition() {
  const bfnew::test::JacobiFixtureCase fixture =
      bfnew::test::make_long_chain_jacobi_fixture();
  const PreparedFixture prepared = prepare_fixture(fixture);
  const std::vector<float> oracle =
      bounded_oracle(fixture.partitioned.graph, fixture.query);
  bfnew::GpuRunOptions options;
  options.engine = bfnew::EngineKind::dense_chaotic_push;
  options.control_mode = bfnew::ControlMode::per_round_host_poll;
  options.maximum_rounds = 128U;
  options.instrumentation = bfnew::InstrumentationLevel::debug;

  std::vector<bfnew::HostDenseRunResult> schedules;
  for (const bfnew::DenseHostSchedule schedule : {
           bfnew::DenseHostSchedule::csr_forward,
           bfnew::DenseHostSchedule::csr_reverse,
           bfnew::DenseHostSchedule::alternating,
       }) {
    schedules.push_back(bfnew::run_host_dense_chaotic_push(
        prepared.layout,
        fixture.query,
        prepared.tile_masks,
        prepared.run_masks.csr_run_masks,
        options,
        schedule));
    expect(
        vectors_bitwise_equal(schedules.back().distances, oracle),
        "adversarial complete-scan schedules converge to bounded Dijkstra");
  }
  expect(
      schedules[1U].controller.rounds_completed >
              schedules[0U].controller.rounds_completed &&
          vectors_bitwise_equal(schedules[0U].distances, schedules[1U].distances) &&
          vectors_bitwise_equal(schedules[1U].distances, schedules[2U].distances),
      "chaotic schedule may change progression but not final distance bits");
  const bfnew::HostDenseRunResult repeated =
      bfnew::run_host_dense_chaotic_push(
          prepared.layout,
          fixture.query,
          prepared.tile_masks,
          prepared.run_masks.csr_run_masks,
          options,
          bfnew::DenseHostSchedule::csr_reverse);
  expect(
      repeated.distance_bits == schedules[1U].distance_bits &&
          repeated.controller.rounds_completed ==
              schedules[1U].controller.rounds_completed &&
          repeated.result.work.atomic_attempts ==
              schedules[1U].result.work.atomic_attempts,
      "repeated deterministic host schedules reproduce bits and counters");
}

void test_instrumentation_and_maximum_rounds() {
  const bfnew::test::JacobiFixtureCase contention =
      bfnew::test::make_high_fan_in_dense_fixture();
  const PreparedFixture prepared = prepare_fixture(contention);
  const auto run_at = [&](const bfnew::InstrumentationLevel level) {
    bfnew::GpuRunOptions options;
    options.engine = bfnew::EngineKind::dense_chaotic_push;
    options.control_mode = bfnew::ControlMode::per_round_host_poll;
    options.maximum_rounds = 64U;
    options.instrumentation = level;
    return bfnew::run_host_dense_chaotic_push(
        prepared.layout,
        contention.query,
        prepared.tile_masks,
        prepared.run_masks.csr_run_masks,
        options);
  };
  const bfnew::HostDenseRunResult none =
      run_at(bfnew::InstrumentationLevel::none);
  const bfnew::HostDenseRunResult light =
      run_at(bfnew::InstrumentationLevel::light);
  const bfnew::HostDenseRunResult debug =
      run_at(bfnew::InstrumentationLevel::debug);
  expect(
      vectors_bitwise_equal(none.distances, light.distances) &&
          vectors_bitwise_equal(light.distances, debug.distances),
      "dense instrumentation never changes final labels");
  expect(
      none.result.work.edges_examined == 0U &&
          none.result.work.successful_decreases == 0U &&
          none.result.work.atomic_attempts == 0U &&
          none.result.work.full_edge_rounds == 0U,
      "None suppresses dense algorithm counters");
  expect(
      light.result.work.edges_examined != 0U &&
          light.result.work.successful_decreases != 0U &&
          light.result.work.full_edge_rounds != 0U &&
          light.result.work.atomic_attempts == 0U &&
          light.result.work.high_contention_destinations == 0U,
      "Light records scan work without expensive atomic/contention counters");
  expect(
      debug.result.work.edges_examined == light.result.work.edges_examined &&
          debug.result.work.successful_decreases ==
              light.result.work.successful_decreases &&
          debug.result.work.atomic_attempts == debug.result.work.edges_examined &&
          debug.result.work.successful_atomic_updates ==
              debug.result.work.successful_decreases &&
          debug.result.work.high_contention_destinations >= 1U &&
          debug.result.work.changed_flag_updates >= 1U,
      "Debug adds atomic, contention, mask, and changed-publication counters");

  const bfnew::test::JacobiFixtureCase chain =
      bfnew::test::make_long_chain_jacobi_fixture();
  const PreparedFixture chain_prepared = prepare_fixture(chain);
  for (const bfnew::ControlMode mode : {
           bfnew::ControlMode::persistent_cooperative,
           bfnew::ControlMode::per_round_host_poll,
           bfnew::ControlMode::chunked_host_poll,
       }) {
    bfnew::GpuRunOptions limited;
    limited.engine = bfnew::EngineKind::dense_chaotic_push;
    limited.control_mode = mode;
    limited.rounds_per_chunk = 32U;
    limited.maximum_rounds = 1U;
    const bfnew::HostDenseRunResult run =
        bfnew::run_host_dense_chaotic_push(
            chain_prepared.layout,
            chain.query,
            chain_prepared.tile_masks,
            chain_prepared.run_masks.csr_run_masks,
            limited);
    expect(
        run.result.status.stop_reason ==
                static_cast<std::uint32_t>(
                    bfnew::DeviceStopReason::maximum_rounds) &&
            run.result.status.rounds_completed == 1U &&
            run.result.status.final_distance_slot == 0U &&
            run.result.status.bounding_box_miss_mask == 0U,
        "all dense controls distinguish maximum-round exhaustion from a miss");
    if (mode == bfnew::ControlMode::chunked_host_poll) {
      expect(
          run.queued_round_pairs == 32U &&
              run.result.work.kernel_dispatches == 65U,
          "dense K=32 maximum stop retains 31 queued no-op pairs");
    }
  }
}

void test_hip_source_contract_if_configured() {
#if defined(BFNEW_PHASE10_HIP_SOURCE_PATH)
  std::ifstream input{BFNEW_PHASE10_HIP_SOURCE_PATH};
  expect(input.good(), "configured dense HIP source opens for structural audit");
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
           "atomicCAS",
           "atomic_load_nonnegative_float_bits",
           "atomic_min_nonnegative_float_bits",
           "hipLaunchCooperativeKernel",
           "hipOccupancyMaxActiveBlocksPerMultiprocessor",
           "grid.sync",
           "high_contention_destinations",
           "changed_flag_updates",
           "full_edge_rounds",
       }) {
    expect(
        contains(marker),
        std::string{"dense HIP structural marker missing: "} +
            std::string{marker});
  }
  const std::size_t round_begin =
      source.find("__device__ void perform_dense_round");
  const std::size_t round_end =
      source.find("__device__ void initialize_dense_state", round_begin);
  const std::string round_source =
      round_begin != std::string::npos && round_end != std::string::npos
          ? source.substr(round_begin, round_end - round_begin)
          : std::string{};
  expect(
      round_source.find("graph.csc") == std::string::npos &&
          round_source.find("frontier") == std::string::npos &&
          round_source.find("worklist") == std::string::npos,
      "dense round uses CSR full scans without frontier or CSC traversal");
  expect(
      round_source.find("atomic_load_nonnegative_float_bits") !=
              std::string::npos &&
          round_source.find("atomic_min_nonnegative_float_bits") !=
              std::string::npos &&
          round_source.find("distance_bits[") == std::string::npos,
      "dense round never mixes ordinary distance loads with atomic writes");
  expect(
      source.find("Gauss") == std::string::npos &&
          source.find("gauss") == std::string::npos,
      "dense source consistently describes chaotic rather than ordered execution");
  expect(
      !contains("hipDeviceSynchronize"),
      "dense engine introduces no device-wide synchronization");
#endif
}

}  // namespace

int main() {
  test_atomic_float_bit_domain();
  test_dense_controller_transition();
  for (const bfnew::test::JacobiFixtureCase& fixture :
       bfnew::test::make_dense_fixture_suite()) {
    run_fixture_matrix(fixture);
  }
  test_schedule_stress_and_repetition();
  test_instrumentation_and_maximum_rounds();
  test_hip_source_contract_if_configured();

  std::cout << "Phase 10 seeded-random maximum observed difference: "
            << maximum_observed_random_ulps << " ULP\n";
  if (failures != 0) {
    std::cerr << failures << " dense chaotic push assertion(s) failed\n";
    return 1;
  }
  return 0;
}
