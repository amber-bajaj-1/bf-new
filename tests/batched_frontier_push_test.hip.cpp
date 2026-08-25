#include "bfnew/hip/batched_frontier_push.hpp"

#include "bfnew/hip/frontier_push.hpp"
#include "bfnew/sssp.hpp"
#include "batched_frontier_fixture_suite.hpp"
#include "frontier_fixture_suite.hpp"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

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
  std::cerr << "batched frontier push HIP test failed [" << fixture
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

#if defined(__HIPCC__)
__global__ void advance_controller_probe(
    bfnew::DeviceController* const controllers,
    std::uint32_t* const transitions) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 2U) {
    transitions[index] = static_cast<std::uint32_t>(
        bfnew::advance_batched_frontier_controller(controllers[index]));
  }
}

void check_test_hip(
    const hipError_t status,
    const std::string_view expression) {
  bfnew::hip::throw_if_hip_error(
      static_cast<std::int32_t>(status), expression);
}
#endif

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
  throw std::logic_error{"HIP batched frontier lane query is missing"};
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
  options.engine = bfnew::EngineKind::frontier_push;
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
  return static_cast<std::size_t>(bfnew::batched_frontier_distance_index(
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

[[nodiscard]] std::uint64_t union_vertex_count(
    const bfnew::BatchDeviceDescription& description) {
  std::uint64_t count = 0U;
  for (const bfnew::BatchVertexRange range :
       description.selected_vertex_ranges) {
    count += range.end - range.begin;
  }
  return count;
}

[[nodiscard]] std::uint64_t distinct_source_count(
    const bfnew::BatchDeviceDescription& description) {
  std::vector<std::uint32_t> sources = description.sources;
  std::ranges::sort(sources);
  sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
  return sources.size();
}

void expect_control_accounting(
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::BatchedFrontierRunOutput& output,
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
        output.metrics.engine_round_dispatches == 0U &&
            output.metrics.controller_advance_dispatches == 0U &&
            output.metrics.convergence_host_checks == 0U &&
            output.metrics.ordinary_grid_blocks == 0U &&
            output.metrics.cooperative_grid_blocks != 0U &&
            output.metrics.cooperative_active_blocks_per_wgp != 0U &&
            output.metrics.ordinary_active_blocks_per_wgp == 0U &&
            work.kernel_dispatches == 1U,
        name,
        "persistent control was not one cooperative queue-to-convergence launch");
    return;
  }

  expect(
      output.metrics.ordinary_grid_blocks != 0U &&
          output.metrics.ordinary_grid_blocks <=
              std::max<std::uint64_t>(
                  1U,
                  (static_cast<std::uint64_t>(
                       output.metrics.queue_capacity) +
                   options.block_size - 1U) /
                      options.block_size) &&
          output.metrics.ordinary_active_blocks_per_wgp != 0U &&
          output.metrics.cooperative_grid_blocks == 0U &&
          output.metrics.cooperative_active_blocks_per_wgp == 0U &&
          output.metrics.engine_round_dispatches ==
              output.metrics.controller_advance_dispatches &&
          output.metrics.convergence_host_checks != 0U &&
          work.kernel_dispatches ==
              4U + 2U * output.metrics.engine_round_dispatches,
      name,
      "ordinary control violated storage/seed/finish/round/finalize accounting");

  const std::uint64_t recorded_rounds =
      output.result.status.rounds_completed;
  if (options.control_mode == bfnew::ControlMode::per_round_host_poll) {
    const std::uint64_t expected_dispatches =
        std::max<std::uint64_t>(1U, recorded_rounds);
    expect(
        output.metrics.engine_round_dispatches == expected_dispatches &&
            output.metrics.convergence_host_checks == expected_dispatches,
        name,
        "per-round control did not poll after each submitted pair");
    return;
  }

  const std::uint64_t chunks = std::max<std::uint64_t>(
      1U,
      (recorded_rounds + options.rounds_per_chunk - 1U) /
          options.rounds_per_chunk);
  expect(
      output.metrics.convergence_host_checks == chunks &&
          output.metrics.engine_round_dispatches ==
              chunks * options.rounds_per_chunk,
      name,
      "chunked control did not retain complete queued K-round pairs");
}

void expect_instrumentation_gating(
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::BatchedFrontierRunOutput& output,
    const std::string_view name) {
  const bfnew::DeviceWorkStatistics& common = output.result.work;
  const bfnew::BatchedFrontierWorkStatistics& frontier = output.batch_work;
  expect(
      common.full_edge_rounds == 0U &&
          common.changed_flag_updates == 0U &&
          common.high_contention_destinations == 0U &&
          common.expansion_count == 0U,
      name,
      "frontier execution leaked dense or expansion counters");

  if (options.instrumentation == bfnew::InstrumentationLevel::none) {
    expect(
        common.edges_examined == 0U &&
            common.successful_decreases == 0U &&
            common.active_vertices == 0U &&
            common.active_lane_rounds == 0U &&
            common.maximum_queue_size == 0U &&
            common.atomic_attempts == 0U &&
            common.successful_atomic_updates == 0U &&
            common.queue_claims == 0U &&
            common.duplicate_suppressions == 0U &&
            common.mask_operations == 0U && common.overflow_events == 0U &&
            common.empty_frontier_rounds == 0U &&
            common.small_frontier_rounds == 0U &&
            frontier.frontier_vertex_entries == 0U &&
            frontier.active_vertex_lane_pairs == 0U &&
            frontier.csr_runs_considered == 0U &&
            frontier.csr_edge_loads == 0U &&
            frontier.lane_edge_relaxations == 0U &&
            frontier.successful_distance_atomic_updates == 0U &&
            frontier.maximum_queue_size == 0U &&
            frontier.small_frontier_rounds == 0U &&
            frontier.current_mask_atomic_exchanges == 0U &&
            frontier.next_mask_atomic_ors == 0U &&
            frontier.controller_mask_atomic_ors == 0U &&
            frontier.queue_claims == 0U &&
            frontier.duplicate_suppressions == 0U &&
            frontier.overflow_events == 0U,
        name,
        "None instrumentation published an algorithmic device counter");
    return;
  }

  expect(
      common.edges_examined == frontier.lane_edge_relaxations &&
          common.successful_decreases ==
              frontier.successful_distance_atomic_updates &&
          common.active_vertices == frontier.frontier_vertex_entries &&
          common.active_lane_rounds == frontier.active_lane_rounds &&
          common.maximum_queue_size == frontier.maximum_queue_size &&
          common.empty_frontier_rounds == frontier.empty_frontier_rounds &&
          common.small_frontier_rounds == frontier.small_frontier_rounds,
      name,
      "Light aggregate counters disagree with frontier accounting");

  if (options.instrumentation == bfnew::InstrumentationLevel::light) {
    expect(
        common.atomic_attempts == 0U &&
            common.successful_atomic_updates == 0U &&
            common.queue_claims == 0U &&
            common.duplicate_suppressions == 0U &&
            common.mask_operations == 0U && common.overflow_events == 0U &&
            frontier.current_mask_atomic_exchanges == 0U &&
            frontier.next_mask_atomic_ors == 0U &&
            frontier.controller_mask_atomic_ors == 0U &&
            frontier.unique_next_vertex_lane_activations == 0U &&
            frontier.queue_claims == 0U &&
            frontier.queue_entries_saved_by_lane_merging == 0U &&
            frontier.same_lane_duplicate_suppressions == 0U &&
            frontier.duplicate_suppressions == 0U &&
            frontier.overflow_events == 0U,
        name,
        "Light instrumentation published Debug mask or queue counters");
    return;
  }

  expect(
      common.atomic_attempts == frontier.distance_atomic_attempts &&
          common.successful_atomic_updates ==
              frontier.successful_distance_atomic_updates &&
          common.queue_claims == frontier.queue_claims &&
          common.duplicate_suppressions ==
              frontier.duplicate_suppressions &&
          common.overflow_events == frontier.overflow_events &&
          frontier.current_mask_atomic_exchanges ==
              frontier.frontier_vertex_entries &&
          frontier.next_mask_atomic_ors ==
              frontier.controller_mask_atomic_ors &&
          frontier.successful_distance_atomic_updates >=
              frontier.unique_next_vertex_lane_activations &&
          frontier.unique_next_vertex_lane_activations >=
              frontier.queue_claims &&
          frontier.next_mask_atomic_ors >= frontier.queue_claims &&
          frontier.queue_entries_saved_by_lane_merging ==
              frontier.unique_next_vertex_lane_activations -
                  frontier.queue_claims &&
          frontier.same_lane_duplicate_suppressions ==
              frontier.successful_distance_atomic_updates -
                  frontier.unique_next_vertex_lane_activations &&
          frontier.duplicate_suppressions ==
              frontier.successful_distance_atomic_updates -
                  frontier.queue_claims &&
          common.mask_operations ==
              frontier.csr_runs_considered +
                  frontier.current_mask_atomic_exchanges +
                  frontier.next_mask_atomic_ors +
                  frontier.controller_mask_atomic_ors,
      name,
      "Debug mask, distance-atomic, or merged-queue identities disagree");
}

void validate_normal_output(
    const bfnew::test::BatchedFrontierFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchDeviceDescription& description,
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::BatchedFrontierRunOutput& output,
    const std::string& name) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const bfnew::DeviceController& controller = output.final_controller;
  const bfnew::DeviceRunStatus& status = output.result.status;
  const std::size_t elements =
      static_cast<std::size_t>(graph.vertex_count()) *
      description.lane_width;

  expect(
      output.result.engine_kind ==
              static_cast<std::uint32_t>(bfnew::EngineKind::frontier_push) &&
          output.result.control_mode ==
              static_cast<std::uint32_t>(options.control_mode),
      name,
      "result lost the selected frontier or control identity");
  expect(
      bfnew::validate_device_controller(controller) ==
              bfnew::DeviceControllerError::none &&
          bfnew::validate_device_run_status(status) ==
              bfnew::DeviceRunStatusError::none,
      name,
      "terminal frontier controller or status is invalid");
  expect(
      controller.done == 1U && controller.execute_lane_mask == 0U &&
          controller.stop_reason ==
              static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::converged) &&
          controller.error_bits == bfnew::device_error::none &&
          controller.valid_lane_mask == description.valid_lane_mask &&
          controller.active_lane_mask == 0U &&
          controller.converged_lane_mask == description.valid_lane_mask &&
          status.converged == 1U && status.stop_reason == controller.stop_reason &&
          status.error_bits == controller.error_bits &&
          status.rounds_completed == controller.rounds_completed &&
          status.active_lane_mask == controller.active_lane_mask &&
          status.converged_lane_mask == controller.converged_lane_mask &&
          status.valid_lane_mask == description.valid_lane_mask &&
          status.rounds_completed != 0U &&
          status.rounds_completed <= options.maximum_rounds,
      name,
      "normal controller/status convergence state disagrees");
  expect(
      status.final_distance_slot == 0U &&
          controller.distance_read_slot == 0U &&
          controller.distance_write_slot == 0U &&
          controller.frontier_read_slot ==
              controller.rounds_completed % 2U &&
          controller.frontier_write_slot ==
              1U - controller.frontier_read_slot &&
          controller.frontier_size[controller.frontier_read_slot] == 0U,
      name,
      "one-slot distance or final queue parity is inconsistent");
  expect(
      output.distances_downloaded && output.distances.size() == elements &&
          output.distance_bits.size() == elements,
      name,
      "full readback is not exact V-by-W atomic storage");
  expect(
      output.frontier_rounds_by_lane.size() == description.lane_width &&
          output.lane_convergence_rounds.size() == description.lane_width &&
          output.lane_tail_rounds.size() == description.lane_width,
      name,
      "per-lane frontier evidence does not have width entries");

  for (std::size_t element = 0U; element < elements; ++element) {
    expect(
        std::bit_cast<std::uint32_t>(output.distances[element]) ==
            output.distance_bits[element],
        name,
        "float projection differs from its atomic distance word");
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
          output.frontier_rounds_by_lane[lane] == 0U &&
              output.lane_convergence_rounds[lane] == 0U &&
              output.lane_tail_rounds[lane] == 0U &&
              description.source_offsets[lane] ==
                  description.source_offsets[lane + 1U] &&
              description.target_offsets[lane] ==
                  description.target_offsets[lane + 1U],
          name,
          "padded lane retained terminals or frontier work");
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
            "padded lane published a source, update, or stale word");
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
        convergence != 0U && convergence <= status.rounds_completed &&
            output.frontier_rounds_by_lane[lane] == convergence &&
            output.lane_tail_rounds[lane] ==
                status.rounds_completed - convergence,
        name,
        "lane frontier, one-based convergence, or tail evidence is wrong");
    maximum_convergence_round =
        std::max(maximum_convergence_round, convergence);
    active_lane_rounds += output.frontier_rounds_by_lane[lane];
    tail_lane_rounds += output.lane_tail_rounds[lane];
  }

  expect(
      maximum_convergence_round == status.rounds_completed,
      name,
      "batch duration is not its longest lane's empty-next-frontier round");
  expect(
      status.reached_target_mask == reached &&
          status.bounding_box_miss_mask == missed &&
          (reached | missed) == description.valid_lane_mask,
      name,
      "device reached/miss masks disagree with bounded Dijkstra");

  const bfnew::BatchedFrontierWorkStatistics& frontier = output.batch_work;
  const std::uint64_t valid_lanes = static_cast<std::uint64_t>(
      std::popcount(description.valid_lane_mask));
  expect(
      frontier.initial_source_lane_activations ==
              description.sources.size() &&
          frontier.initial_queue_entries ==
              distinct_source_count(description) &&
          frontier.initial_source_lane_activations ==
              frontier.initial_queue_entries +
                  frontier.initial_queue_entries_saved_by_lane_merging &&
          frontier.frontier_rounds == status.rounds_completed &&
          frontier.empty_frontier_rounds == 1U &&
          frontier.active_lane_rounds == active_lane_rounds &&
          frontier.tail_lane_rounds == tail_lane_rounds &&
          frontier.tail_lane_rounds_without_frontier_work ==
              frontier.tail_lane_rounds &&
          frontier.semantic_lane_edge_work_avoided_by_per_lane_convergence ==
              0U,
      name,
      "source merging or per-lane frontier/tail accounting is inconsistent");
  expect(
      frontier.valid_lane_round_capacity ==
              status.rounds_completed * valid_lanes &&
          frontier.lane_width_round_capacity ==
              status.rounds_completed * description.lane_width &&
          frontier.wave32_lane_round_capacity ==
              status.rounds_completed * bfnew::maximum_batch_lanes &&
          frontier.wave32_lane_round_capacity ==
              frontier.lane_width_round_capacity +
                  frontier.unused_wave_lane_round_capacity &&
          frontier.valid_lane_round_capacity ==
              frontier.active_lane_rounds +
                  frontier.inactive_valid_lane_rounds &&
          frontier.lane_width_round_capacity ==
              frontier.valid_lane_round_capacity +
                  frontier.padded_lane_round_capacity &&
          frontier.padded_lane_semantic_work == 0U,
      name,
      "valid/configured/wave32 lane-round capacities do not balance");

  std::uint64_t selected_tile_lane_positions = 0U;
  for (const std::uint32_t tile : description.union_tiles) {
    selected_tile_lane_positions += static_cast<std::uint64_t>(
        std::popcount(description.tile_lane_masks[tile]));
  }
  expect(
      frontier.distance_reset_bytes ==
              selected_lane_vertex_count(description) *
                  sizeof(std::uint32_t) &&
          frontier.activity_mask_reset_bytes ==
              union_vertex_count(description) * 2U *
                  sizeof(bfnew::LaneMask) &&
          frontier.source_seed_write_bytes ==
              description.sources.size() * sizeof(std::uint32_t) &&
          frontier.frontier_queue_storage_bytes ==
              graph.vertex_count() * 2U * sizeof(std::uint32_t) &&
          frontier.union_tile_lane_positions ==
              description.union_tiles.size() * valid_lanes &&
          frontier.selected_tile_lane_positions ==
              selected_tile_lane_positions &&
          frontier.union_tile_lane_positions >=
              frontier.selected_tile_lane_positions,
      name,
      "selected reset, queue storage, seed, or tile accounting is wrong");

  if (options.instrumentation != bfnew::InstrumentationLevel::none) {
    expect(
        frontier.frontier_vertex_entries != 0U &&
            frontier.active_vertex_lane_pairs >=
                frontier.frontier_vertex_entries &&
            frontier.shared_vertex_entries_saved ==
                frontier.active_vertex_lane_pairs -
                    frontier.frontier_vertex_entries &&
            frontier.csr_runs_considered ==
                frontier.csr_runs_visited + frontier.csr_runs_skipped &&
            frontier.active_lanes_over_visited_runs >=
                frontier.csr_runs_visited &&
            frontier.lane_edge_relaxations >= frontier.csr_edge_loads &&
            frontier.shared_edge_lane_work_saved ==
                frontier.lane_edge_relaxations -
                    frontier.csr_edge_loads &&
            frontier.distance_atomic_source_loads ==
                frontier.lane_edge_relaxations &&
            frontier.distance_atomic_attempts ==
                frontier.lane_edge_relaxations &&
            frontier.successful_distance_atomic_updates <=
                frontier.distance_atomic_attempts &&
            frontier.maximum_queue_size >=
                frontier.initial_queue_entries &&
            frontier.maximum_queue_size <= graph.vertex_count() &&
            frontier.overflow_events == 0U,
        name,
        "frontier sharing, atomic, or queue high-water counters are wrong");
  }

  const std::uint64_t mask_atomics =
      frontier.current_mask_atomic_exchanges +
      frontier.next_mask_atomic_ors +
      frontier.controller_mask_atomic_ors;
  expect(
      output.metrics.kernel_registers_per_thread != 0U &&
          output.metrics.lane_width == description.lane_width &&
          output.metrics.valid_lane_count == valid_lanes &&
          output.metrics.queue_capacity == graph.vertex_count() &&
          output.metrics.edge_record_read_bytes_requested ==
              frontier.csr_edge_loads *
                  (sizeof(std::uint32_t) + sizeof(float)) &&
          output.metrics.distance_atomic_read_bytes_requested ==
              frontier.distance_atomic_source_loads *
                  sizeof(std::uint32_t) &&
          output.metrics.distance_atomic_min_bytes_requested ==
              frontier.distance_atomic_attempts * sizeof(std::uint32_t) &&
          output.metrics.activity_mask_atomic_bytes_requested ==
              mask_atomics * sizeof(bfnew::LaneMask),
      name,
      "selected-kernel, queue-capacity, or requested-byte metrics are wrong");
  if (options.instrumentation != bfnew::InstrumentationLevel::none) {
    expect(
        output.metrics.average_active_lanes_per_frontier_vertex >= 1.0 &&
            output.metrics.average_active_lanes_per_nonzero_run >= 1.0 &&
            output.metrics.frontier_lane_utilization > 0.0 &&
            output.metrics.frontier_lane_utilization <= 1.0 &&
            output.metrics.lane_round_utilization > 0.0 &&
            output.metrics.lane_round_utilization <= 1.0,
        name,
        "frontier/run/lane arithmetic utilization is invalid");
  }
  expect(
      output.metrics.hardware_counters.available == 0U &&
          output.metrics.hardware_counters.l2_read_bytes == 0U &&
          output.metrics.hardware_counters.l2_write_bytes == 0U &&
          output.metrics.hardware_counters.atomic_stall_cycles == 0U &&
          output.metrics.hardware_counters.waves == 0U,
      name,
      "unavailable profiler counters were synthesized");

  expect_control_accounting(options, output, name);
  expect_instrumentation_gating(options, output, name);
}

void expect_mixed_frontier_evidence(
    const bfnew::test::BatchedFrontierFixture& fixture,
    const bfnew::BatchDeviceDescription& description,
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::BatchedFrontierRunOutput& output,
    const std::string_view name) {
  const auto lane_for = [&](const bfnew::QueryId query_id) {
    for (std::size_t lane = 0U; lane < description.lane_width; ++lane) {
      if ((description.valid_lane_mask & (bfnew::LaneMask{1U} << lane)) !=
              0U &&
          description.query_ids_by_lane[lane] == query_id.value()) {
        return lane;
      }
    }
    throw std::logic_error{"mixed frontier query lane is missing"};
  };

  const std::size_t no_next_lane = lane_for(bfnew::QueryId{1400U});
  const std::size_t unreachable_lane = lane_for(bfnew::QueryId{1403U});
  const std::size_t long_lane = lane_for(bfnew::QueryId{1402U});
  const std::size_t shared_source_lane = lane_for(bfnew::QueryId{1404U});
  expect(
      output.lane_convergence_rounds[no_next_lane] == 1U &&
          output.lane_convergence_rounds[unreachable_lane] == 1U &&
          output.lane_convergence_rounds[long_lane] > 1U &&
          output.result.status.rounds_completed > 1U &&
          output.lane_tail_rounds[no_next_lane] > 0U &&
          output.lane_tail_rounds[unreachable_lane] > 0U,
      name,
      "immediate, unreachable, longer, or tail frontier evidence is missing");

  const bfnew::RouteQuery& long_query =
      lane_query(fixture.queries, description, long_lane);
  const bfnew::RouteQuery& shared_query =
      lane_query(fixture.queries, description, shared_source_lane);
  std::uint32_t shared_source = std::numeric_limits<std::uint32_t>::max();
  for (const bfnew::VertexId left : long_query.sources) {
    if (std::find(
            shared_query.sources.begin(),
            shared_query.sources.end(),
            left) != shared_query.sources.end()) {
      shared_source = left.value();
      break;
    }
  }
  expect(
      shared_source != std::numeric_limits<std::uint32_t>::max(),
      name,
      "fixture did not retain its source shared across two lanes");
  expect(
      output.distance_bits[distance_index(
          shared_source, long_lane, description.lane_width)] == 0U &&
          output.distance_bits[distance_index(
              shared_source,
              shared_source_lane,
              description.lane_width)] == 0U &&
          output.batch_work.initial_source_lane_activations >
              output.batch_work.initial_queue_entries &&
          output.batch_work.initial_queue_entries_saved_by_lane_merging != 0U,
      name,
      "shared source did not preserve two zero labels in one queue entry");

  const std::uint32_t cross_lane_vertex =
      lane_query(fixture.queries, description, unreachable_lane)
          .targets.front()
          .value();
  expect(
      std::isinf(output.distances[distance_index(
          cross_lane_vertex, unreachable_lane, description.lane_width)]) &&
          std::isfinite(output.distances[distance_index(
              cross_lane_vertex, long_lane, description.lane_width)]),
      name,
      "one frontier lane seeded or updated another lane's distance word");

  if (options.instrumentation == bfnew::InstrumentationLevel::debug) {
    expect(
        output.batch_work.multi_lane_frontier_vertex_entries != 0U &&
            output.batch_work.shared_vertex_entries_saved != 0U &&
            output.batch_work.multi_lane_csr_edge_loads != 0U &&
            output.batch_work.shared_edge_lane_work_saved != 0U &&
            output.batch_work.queue_entries_saved_by_lane_merging != 0U,
        name,
        "Debug counters did not expose shared vertices, edges, and queue entries");
  }
}

void validate_standalone_output(
    const bfnew::test::BatchedFrontierFixture& fixture,
    const bfnew::RouteQuery& query,
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::FrontierRunOutput& output) {
  expect(
      output.distances_downloaded &&
          output.distances.size() == fixture.partitioned.graph.vertex_count(),
      "width-one-standalone",
      "standalone frontier did not download one word per vertex");
  expect(
      output.result.engine_kind ==
              static_cast<std::uint32_t>(bfnew::EngineKind::frontier_push) &&
          output.result.control_mode ==
              static_cast<std::uint32_t>(options.control_mode) &&
          bfnew::validate_device_controller(output.final_controller) ==
              bfnew::DeviceControllerError::none &&
          bfnew::validate_device_run_status(output.result.status) ==
              bfnew::DeviceRunStatusError::none &&
          output.result.status.stop_reason ==
              static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::converged) &&
          output.result.status.final_distance_slot == 0U,
      "width-one-standalone",
      "standalone frontier terminal identity or status is invalid");
  const std::vector<float> oracle =
      bounded_oracle(fixture.partitioned.graph, query);
  for (std::size_t vertex = 0U;
       vertex < fixture.partitioned.graph.vertex_count();
       ++vertex) {
    if (selected_vertex(fixture.partitioned.graph, query, vertex)) {
      expect(
          bits_equal(output.distances[vertex], oracle[vertex]),
          "width-one-standalone",
          "standalone frontier differs from bounded Dijkstra");
    }
  }
}

void test_device_controller_error_canonicalization() {
#if defined(__HIPCC__)
  const bfnew::GpuRunOptions options = options_for(
      bfnew::ControlMode::per_round_host_poll,
      true,
      bfnew::InstrumentationLevel::debug);
  std::array<bfnew::DeviceController, 2U> controllers{
      bfnew::initialize_device_controller(options, 1U, 1U),
      bfnew::initialize_device_controller(options, 1U, 1U),
  };
  controllers[0U].distance_read_slot = 1U;
  controllers[1U].error_bits = bfnew::device_error::device_failure;
  std::array<std::uint32_t, 2U> transitions{};

  bfnew::hip::HipStream stream;
  bfnew::hip::DeviceBuffer controller_buffer;
  bfnew::hip::DeviceBuffer transition_buffer;
  expect(
      controller_buffer.reserve(
          sizeof(controllers), bfnew::hip::BufferGrowth::exact) &&
          transition_buffer.reserve(
              sizeof(transitions), bfnew::hip::BufferGrowth::exact),
      "device-controller-errors",
      "controller probe did not allocate fresh bounded device buffers");
  controller_buffer.copy_from_host_async(
      controllers.data(), sizeof(controllers), stream);
  transition_buffer.clear_async(sizeof(transitions), stream);
  hipLaunchKernelGGL(
      advance_controller_probe,
      dim3(1U),
      dim3(2U),
      0U,
      reinterpret_cast<hipStream_t>(stream.native_handle()),
      static_cast<bfnew::DeviceController*>(controller_buffer.data()),
      static_cast<std::uint32_t*>(transition_buffer.data()));
  check_test_hip(
      hipGetLastError(), "advance_controller_probe launch");
  controller_buffer.copy_to_host_async(
      controllers.data(), sizeof(controllers), stream);
  transition_buffer.copy_to_host_async(
      transitions.data(), sizeof(transitions), stream);
  stream.synchronize();

  expect(
      transitions[0U] == static_cast<std::uint32_t>(
                              bfnew::BatchedFrontierAdvanceResult::
                                  invalid_controller_state) &&
          bfnew::validate_device_controller(controllers[0U]) ==
              bfnew::DeviceControllerError::none &&
          controllers[0U].done == 1U &&
          controllers[0U].stop_reason ==
              static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::invalid_controller_state) &&
          controllers[0U].error_bits ==
              bfnew::device_error::invalid_controller_state &&
          controllers[0U].active_lane_mask == 0U &&
          controllers[0U].execute_lane_mask == 0U,
      "device-controller-errors",
      "device advance did not canonicalize malformed controller state");
  expect(
      transitions[1U] == static_cast<std::uint32_t>(
                              bfnew::BatchedFrontierAdvanceResult::
                                  device_failure) &&
          bfnew::validate_device_controller(controllers[1U]) ==
              bfnew::DeviceControllerError::none &&
          controllers[1U].done == 1U &&
          controllers[1U].rounds_completed == 1U &&
          controllers[1U].stop_reason ==
              static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::device_failure) &&
          controllers[1U].error_bits ==
              bfnew::device_error::device_failure &&
          controllers[1U].active_lane_mask == 0U &&
          controllers[1U].execute_lane_mask == 0U,
      "device-controller-errors",
      "device advance did not canonicalize a recognized device failure");
#endif
}

void test_scratch_layout() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  const std::size_t vertices = fixture.partitioned.graph.vertex_count();
  for (const std::uint32_t width : {1U, 8U, 16U, 32U}) {
    for (const std::size_t requested_queue : {std::size_t{0U},
                                              std::size_t{1U}}) {
      const bfnew::BatchedFrontierScratchLayout layout =
          bfnew::make_batched_frontier_scratch_layout(
              vertices, width, requested_queue);
      const std::size_t queue = requested_queue == 0U ? vertices
                                                       : requested_queue;
      const std::uint64_t distance_bytes =
          vertices * static_cast<std::uint64_t>(width) *
          sizeof(std::uint32_t);
      const std::uint64_t mask_bytes =
          vertices * sizeof(bfnew::LaneMask);
      const std::uint64_t queue_bytes =
          queue * sizeof(std::uint32_t);
      expect(
          layout.vertex_count == vertices &&
              layout.lane_width == width &&
              layout.queue_capacity == queue &&
              layout.distance_bits_offset == 0U &&
              layout.distance_bits_bytes == distance_bytes &&
              layout.activity_mask_offsets[0U] == distance_bytes &&
              layout.activity_mask_offsets[1U] ==
                  distance_bytes + mask_bytes &&
              layout.activity_mask_bytes_each == mask_bytes &&
              layout.frontier_offsets[0U] ==
                  distance_bytes + 2U * mask_bytes &&
              layout.frontier_offsets[1U] ==
                  distance_bytes + 2U * mask_bytes + queue_bytes &&
              layout.frontier_bytes_each == queue_bytes &&
              layout.total_bytes ==
                  distance_bytes + 2U * mask_bytes + 2U * queue_bytes &&
              bfnew::hip::batched_frontier_scratch_bytes(
                  static_cast<std::uint32_t>(vertices),
                  width,
                  static_cast<std::uint32_t>(requested_queue)) ==
                  layout.total_bytes,
          "scratch-layout",
          "public planner and HIP scratch-byte API disagree");
    }
  }
}

void test_mixed_width_control_matrix() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedFrontierWorkspace workspace;
  bfnew::hip::BatchedFrontierPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};

  for (const std::uint32_t width : {8U, 16U, 32U}) {
    const bfnew::test::PreparedBatchedFrontierFixture prepared =
        bfnew::test::prepare_batched_frontier_fixture(
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
    bfnew::LaneMask expected_reached = 0U;
    bfnew::LaneMask expected_miss = 0U;
    for (const bfnew::GpuRunOptions& enabled_control :
         control_matrix(true)) {
      std::vector<std::uint32_t> toggle_baseline;
      for (const bool per_lane : {true, false}) {
        bfnew::GpuRunOptions options = enabled_control;
        options.enable_per_lane_convergence = per_lane ? 1U : 0U;
        const bfnew::hip::BatchedFrontierRunOutput output =
            engine.run_with_distances(prepared.description, options);
        const std::string name =
            "width-" + std::to_string(width) + "-control-" +
            std::to_string(
                static_cast<std::uint32_t>(options.control_mode)) +
            "-K-" + std::to_string(options.rounds_per_chunk) +
            (per_lane ? "-enabled" : "-disabled");
        validate_normal_output(
            fixture,
            fixture.queries,
            prepared.description,
            options,
            output,
            name);
        expect_mixed_frontier_evidence(
            fixture, prepared.description, options, output, name);

        if (toggle_baseline.empty()) {
          toggle_baseline = output.distance_bits;
          expected_reached = output.result.status.reached_target_mask;
          expected_miss = output.result.status.bounding_box_miss_mask;
        } else {
          expect(
              toggle_baseline == output.distance_bits &&
                  expected_reached ==
                      output.result.status.reached_target_mask &&
                  expected_miss ==
                      output.result.status.bounding_box_miss_mask,
              name,
              "convergence toggle changed final bits or terminal masks");
        }
        if (width_baseline.empty()) {
          width_baseline = output.distance_bits;
        } else {
          expect(
              width_baseline == output.distance_bits,
              name,
              "control mode or K changed a final atomic distance word");
        }
      }
    }
  }
}

void test_width_one_matches_standalone() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  const std::array<bfnew::RouteQuery, 1U> singleton{fixture.queries[2U]};
  const bfnew::test::PreparedBatchedFrontierFixture prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedFrontierWorkspace batch_workspace;
  bfnew::hip::ReusableDeviceWorkspace standalone_workspace;
  bfnew::hip::BatchedFrontierPushEngine batched{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      batch_workspace,
      stream};
  bfnew::hip::FrontierPushEngine standalone{
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
      const bfnew::hip::FrontierRunOutput standalone_output =
          standalone.run_with_distances(singleton.front(), options);
      validate_standalone_output(
          fixture, singleton.front(), options, standalone_output);
      const bfnew::hip::BatchedFrontierRunOutput batch_output =
          batched.run_with_distances(prepared.description, options);
      validate_normal_output(
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
          "batched and standalone frontier disagree on terminal classification");
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
            "width one differs bitwise from standalone HIP frontier push");
      }
      if (control_baseline.empty()) {
        control_baseline = batch_output.distance_bits;
      } else {
        expect(
            control_baseline == batch_output.distance_bits,
            "width-one",
            "width-one control or convergence toggle changed final bits");
      }
    }
  }
}

enum class FocusedFrontierCase : std::uint8_t {
  repeated_improvement,
  high_fan_in,
  zero_weight_cycle,
};

void test_schedule_robust_contention_and_zero_cycle() {
  const auto run_fixture = [](
                               bfnew::test::BatchedFrontierFixture fixture,
                               const FocusedFrontierCase kind,
                               const std::string_view fixture_name) {
    const bfnew::test::PreparedBatchedFrontierFixture prepared =
        bfnew::test::prepare_batched_frontier_fixture(
            fixture,
            fixture.queries,
            1U,
            bfnew::BatchRunRepresentation::retained_per_run_masks);
    bfnew::hip::HipStream stream;
    bfnew::hip::ResidentDeviceGraph resident;
    resident.upload_once_async(
        bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
    bfnew::hip::ReusableBatchedFrontierWorkspace workspace;
    bfnew::hip::BatchedFrontierPushEngine engine{
        fixture.partitioned.graph,
        fixture.tile_runs,
        resident,
        workspace,
        stream};

    std::vector<std::uint32_t> baseline_bits;
    for (const bool per_lane : {true, false}) {
      bfnew::GpuRunOptions options = options_for(
          bfnew::ControlMode::per_round_host_poll,
          per_lane,
          bfnew::InstrumentationLevel::debug);
      options.maximum_rounds = 32U;
      const bfnew::hip::BatchedFrontierRunOutput output =
          engine.run_with_distances(prepared.description, options);
      const std::string name =
          std::string{fixture_name} +
          (per_lane ? "-enabled" : "-disabled");
      validate_normal_output(
          fixture,
          fixture.queries,
          prepared.description,
          options,
          output,
          name);
      expect(
          output.result.status.rounds_completed < options.maximum_rounds,
          name,
          "focused frontier fixture did not terminate below its safety bound");

      const bfnew::BatchedFrontierWorkStatistics& work = output.batch_work;
      expect(
          work.unique_next_vertex_lane_activations == work.queue_claims &&
              work.queue_entries_saved_by_lane_merging == 0U &&
              work.same_lane_duplicate_suppressions ==
                  work.successful_distance_atomic_updates -
                      work.queue_claims &&
              work.duplicate_suppressions ==
                  work.same_lane_duplicate_suppressions,
          name,
          "width-one contention counters violate single-claim identities");

      switch (kind) {
        case FocusedFrontierCase::repeated_improvement:
          // The three candidates for the hub may win in any order. The
          // physical queue claims and attempts are fixed; successful strict
          // improvements range from best-first (five) to worst-first (seven).
          expect(
              work.distance_atomic_attempts == 7U &&
                  work.queue_claims == 5U &&
                  work.unique_next_vertex_lane_activations == 5U &&
                  work.maximum_queue_size == 3U &&
                  work.successful_distance_atomic_updates >= 5U &&
                  work.successful_distance_atomic_updates <= 7U,
              name,
              "repeated fan-in did not preserve one physical hub claim");
          break;
        case FocusedFrontierCase::high_fan_in:
          // Twelve intermediate vertices race with only two distinct hub
          // candidates. The winning order can add at most one extra success.
          expect(
              work.distance_atomic_attempts == 24U &&
                  work.queue_claims == 13U &&
                  work.unique_next_vertex_lane_activations == 13U &&
                  work.maximum_queue_size == 12U &&
                  work.successful_distance_atomic_updates >= 13U &&
                  work.successful_distance_atomic_updates <= 14U,
              name,
              "high fan-in did not collapse the hub to one queue claim");
          break;
        case FocusedFrontierCase::zero_weight_cycle:
          expect(
              work.distance_atomic_attempts == 4U &&
                  work.successful_distance_atomic_updates == 3U &&
                  work.queue_claims == 3U &&
                  work.unique_next_vertex_lane_activations == 3U &&
                  work.same_lane_duplicate_suppressions == 0U &&
                  work.maximum_queue_size == 2U,
              name,
              "equal zero-cycle candidate reactivated a settled lane");
          break;
      }

      if (baseline_bits.empty()) {
        baseline_bits = output.distance_bits;
      } else {
        expect(
            baseline_bits == output.distance_bits,
            name,
            "convergence publication changed focused final distance bits");
      }
    }
  };

  run_fixture(
      bfnew::test::make_single_query_batched_frontier_fixture(
          bfnew::test::make_repeated_improvement_frontier_fixture()),
      FocusedFrontierCase::repeated_improvement,
      "repeated-improvement");
  run_fixture(
      bfnew::test::make_single_query_batched_frontier_fixture(
          bfnew::test::make_high_fan_in_dense_fixture()),
      FocusedFrontierCase::high_fan_in,
      "high-fan-in");
  run_fixture(
      bfnew::test::make_single_query_batched_frontier_fixture(
          bfnew::test::make_phase5_zero_cycle_jacobi_fixture()),
      FocusedFrontierCase::zero_weight_cycle,
      "zero-weight-cycle");
}

[[nodiscard]] bfnew::test::BatchedFrontierFixture
make_tiny_width32_reuse_fixture() {
  bfnew::InputGraph input{
      bfnew::test::located_vertices(3U),
      {
          bfnew::test::jacobi_edge(0U, 1U, 1.0F, 16000U),
          bfnew::test::jacobi_edge(1U, 2U, 1.0F, 16001U),
      },
  };
  const bfnew::UniformGridPartitioner partitioner{
      bfnew::SpatialOrderConfig{0, 0, 1U, 1U}};
  bfnew::PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[0U]};
  const std::array targets{partitioned.graph.old_to_new()[2U]};
  std::vector<bfnew::RouteQuery> queries;
  queries.reserve(bfnew::maximum_batch_lanes);
  for (std::uint32_t lane = 0U; lane < bfnew::maximum_batch_lanes; ++lane) {
    queries.push_back(bfnew::test::make_full_region_query(
        bfnew::QueryId{1600U + lane},
        partitioned.graph,
        sources,
        targets));
  }
  bfnew::TileRunLayout64 tile_runs =
      bfnew::build_tile_run_layout(partitioned.graph);
  bfnew::DeviceGraphLayout32 device_graph =
      bfnew::build_device_graph_layout32(partitioned.graph, tile_runs);
  return bfnew::test::BatchedFrontierFixture{
      std::move(partitioned),
      std::move(tile_runs),
      std::move(device_graph),
      std::move(queries),
  };
}

void test_width32_reused_workspace_convergence_reset() {
  const bfnew::test::BatchedFrontierFixture fixture =
      make_tiny_width32_reuse_fixture();
  const bfnew::test::PreparedBatchedFrontierFixture priming_prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          fixture.queries,
          32U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  std::vector<bfnew::RouteQuery> tiny_queries;
  tiny_queries.reserve(bfnew::maximum_batch_lanes);
  const std::array tiny_sources{fixture.partitioned.graph.old_to_new()[0U]};
  const std::array tiny_targets{fixture.partitioned.graph.old_to_new()[0U]};
  const bfnew::TileId tiny_tile =
      fixture.partitioned.graph.owner_tiles()[tiny_sources.front().value()];
  for (std::uint32_t lane = 0U; lane < bfnew::maximum_batch_lanes; ++lane) {
    tiny_queries.push_back(bfnew::test::with_exact_selected_tiles(
        bfnew::make_route_query(
            bfnew::QueryId{1700U + lane},
            fixture.partitioned.graph,
            tiny_sources,
            tiny_targets),
        {tiny_tile},
        fixture.partitioned.graph));
  }
  const bfnew::test::PreparedBatchedFrontierFixture tiny_prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          tiny_queries,
          32U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  expect(
      priming_prepared.description.valid_lane_mask ==
              std::numeric_limits<bfnew::LaneMask>::max() &&
          tiny_prepared.description.valid_lane_mask ==
              std::numeric_limits<bfnew::LaneMask>::max() &&
          union_vertex_count(tiny_prepared.description) == 1U,
      "width32-reuse-reset",
      "reset regression fixture is not a full-width one-vertex union");

  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedFrontierWorkspace workspace;
  bfnew::hip::BatchedFrontierPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};

  bfnew::GpuRunOptions priming_options = options_for(
      bfnew::ControlMode::per_round_host_poll,
      true,
      bfnew::InstrumentationLevel::none);
  priming_options.block_size = 1U;
  priming_options.maximum_rounds = 8U;
  const bfnew::hip::BatchedFrontierRunOutput priming =
      engine.run_with_distances(
          priming_prepared.description, priming_options);
  validate_normal_output(
      fixture,
      fixture.queries,
      priming_prepared.description,
      priming_options,
      priming,
      "width32-reuse-prime");
  expect(
      std::ranges::all_of(
          priming.lane_convergence_rounds,
          [](const std::uint64_t round) { return round > 1U; }),
      "width32-reuse-prime",
      "priming run did not leave multi-round convergence records");

  for (std::uint32_t repetition = 0U; repetition < 2U; ++repetition) {
    const bfnew::hip::BatchedFrontierRunOutput output =
        engine.run_with_distances(
            tiny_prepared.description, priming_options);
    const std::string name =
        "width32-reuse-single-vertex-" + std::to_string(repetition);
    validate_normal_output(
        fixture,
        tiny_queries,
        tiny_prepared.description,
        priming_options,
        output,
        name);
    expect(
        std::ranges::all_of(
            output.lane_convergence_rounds,
            [](const std::uint64_t round) { return round == 1U; }) &&
            std::ranges::all_of(
                output.frontier_rounds_by_lane,
                [](const std::uint64_t rounds) { return rounds == 1U; }) &&
            std::ranges::all_of(
                output.lane_tail_rounds,
                [](const std::uint64_t rounds) { return rounds == 0U; }),
        name,
        "one-vertex reused lease retained multi-round convergence records");
  }
}

void test_instrumentation_levels() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  const bfnew::test::PreparedBatchedFrontierFixture prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedFrontierWorkspace workspace;
  bfnew::hip::BatchedFrontierPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};

  std::vector<std::uint32_t> baseline;
  for (const bfnew::InstrumentationLevel instrumentation : {
           bfnew::InstrumentationLevel::none,
           bfnew::InstrumentationLevel::light,
           bfnew::InstrumentationLevel::debug}) {
    const bfnew::GpuRunOptions options = options_for(
        bfnew::ControlMode::per_round_host_poll,
        true,
        instrumentation);
    const bfnew::hip::BatchedFrontierRunOutput output =
        engine.run_with_distances(prepared.description, options);
    const std::string name =
        "instrumentation-" +
        std::to_string(static_cast<std::uint32_t>(instrumentation));
    validate_normal_output(
        fixture,
        fixture.queries,
        prepared.description,
        options,
        output,
        name);
    expect_mixed_frontier_evidence(
        fixture, prepared.description, options, output, name);
    if (baseline.empty()) {
      baseline = output.distance_bits;
    } else {
      expect(
          baseline == output.distance_bits,
          name,
          "None/Light/Debug changed final frontier distance bits");
    }
  }
}

void test_status_only_parity() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  const bfnew::test::PreparedBatchedFrontierFixture prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedFrontierWorkspace workspace;
  bfnew::hip::BatchedFrontierPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};

  for (const bfnew::ControlMode control : {
           bfnew::ControlMode::persistent_cooperative,
           bfnew::ControlMode::per_round_host_poll,
           bfnew::ControlMode::chunked_host_poll}) {
    bfnew::GpuRunOptions options = options_for(
        control, true, bfnew::InstrumentationLevel::none);
    if (control == bfnew::ControlMode::chunked_host_poll) {
      options.rounds_per_chunk = 4U;
    }
    const bfnew::hip::BatchedFrontierRunOutput full =
        engine.run_with_distances(prepared.description, options);
    validate_normal_output(
        fixture,
        fixture.queries,
        prepared.description,
        options,
        full,
        "full-readback");
    const bfnew::hip::BatchedFrontierRunOutput status_only =
        engine.run_status_only(prepared.description, options);
    expect(
        !status_only.distances_downloaded &&
            status_only.distances.empty() &&
            status_only.distance_bits.empty(),
        "status-only",
        "status-only entry point downloaded distance storage");
    expect(
        bfnew::validate_device_controller(status_only.final_controller) ==
                bfnew::DeviceControllerError::none &&
            bfnew::validate_device_run_status(status_only.result.status) ==
                bfnew::DeviceRunStatusError::none &&
            status_only.result.status.stop_reason ==
                full.result.status.stop_reason &&
            status_only.result.status.rounds_completed ==
                full.result.status.rounds_completed &&
            status_only.result.status.reached_target_mask ==
                full.result.status.reached_target_mask &&
            status_only.result.status.bounding_box_miss_mask ==
                full.result.status.bounding_box_miss_mask &&
            status_only.result.status.converged_lane_mask ==
                full.result.status.converged_lane_mask &&
            status_only.final_controller.frontier_read_slot ==
                full.final_controller.frontier_read_slot &&
            status_only.frontier_rounds_by_lane ==
                full.frontier_rounds_by_lane &&
            status_only.lane_convergence_rounds ==
                full.lane_convergence_rounds &&
            status_only.lane_tail_rounds == full.lane_tail_rounds &&
            status_only.batch_work == full.batch_work,
        "status-only",
        "status-only and full readback changed terminal semantics");
    expect_control_accounting(options, status_only, "status-only");
    expect_instrumentation_gating(options, status_only, "status-only");
  }
}

void test_legal_occupancy_grids() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  const bfnew::test::PreparedBatchedFrontierFixture prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedFrontierWorkspace workspace;
  bfnew::hip::BatchedFrontierPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};

  bfnew::GpuRunOptions occupancy =
      options_for(bfnew::ControlMode::persistent_cooperative, true);
  const bfnew::hip::BatchedFrontierRunOutput occupancy_output =
      engine.run_with_distances(prepared.description, occupancy);
  validate_normal_output(
      fixture,
      fixture.queries,
      prepared.description,
      occupancy,
      occupancy_output,
      "occupancy-derived-persistent");
  const std::uint32_t active =
      occupancy_output.metrics.cooperative_active_blocks_per_wgp;

  bfnew::GpuRunOptions fixed_one = occupancy;
  fixed_one.grid_policy = bfnew::GridPolicy::fixed_blocks_per_wgp;
  fixed_one.blocks_per_wgp = 1U;
  const bfnew::hip::BatchedFrontierRunOutput one_output =
      engine.run_with_distances(prepared.description, fixed_one);
  validate_normal_output(
      fixture,
      fixture.queries,
      prepared.description,
      fixed_one,
      one_output,
      "fixed-one-persistent");
  expect(
      one_output.metrics.cooperative_active_blocks_per_wgp == active &&
          occupancy_output.metrics.cooperative_grid_blocks ==
              one_output.metrics.cooperative_grid_blocks * active &&
          occupancy_output.distance_bits == one_output.distance_bits,
      "occupancy",
      "one-block legal grid changed occupancy evidence or final bits");

  bfnew::GpuRunOptions fixed_limit = occupancy;
  fixed_limit.grid_policy = bfnew::GridPolicy::fixed_blocks_per_wgp;
  fixed_limit.blocks_per_wgp = active;
  const bfnew::hip::BatchedFrontierRunOutput limit_output =
      engine.run_with_distances(prepared.description, fixed_limit);
  validate_normal_output(
      fixture,
      fixture.queries,
      prepared.description,
      fixed_limit,
      limit_output,
      "fixed-limit-persistent");
  expect(
      limit_output.metrics.cooperative_grid_blocks ==
              occupancy_output.metrics.cooperative_grid_blocks &&
          limit_output.distance_bits == occupancy_output.distance_bits,
      "occupancy",
      "maximum legal fixed cooperative grid changed output");

  bfnew::GpuRunOptions ordinary =
      options_for(bfnew::ControlMode::per_round_host_poll, true);
  const bfnew::hip::BatchedFrontierRunOutput ordinary_output =
      engine.run_with_distances(prepared.description, ordinary);
  validate_normal_output(
      fixture,
      fixture.queries,
      prepared.description,
      ordinary,
      ordinary_output,
      "occupancy-derived-ordinary");
  bfnew::GpuRunOptions ordinary_fixed = ordinary;
  ordinary_fixed.grid_policy = bfnew::GridPolicy::fixed_blocks_per_wgp;
  ordinary_fixed.blocks_per_wgp = 1U;
  const bfnew::hip::BatchedFrontierRunOutput ordinary_fixed_output =
      engine.run_with_distances(prepared.description, ordinary_fixed);
  validate_normal_output(
      fixture,
      fixture.queries,
      prepared.description,
      ordinary_fixed,
      ordinary_fixed_output,
      "fixed-one-ordinary");
  expect(
      ordinary_fixed_output.metrics.ordinary_active_blocks_per_wgp ==
              ordinary_output.metrics.ordinary_active_blocks_per_wgp &&
          ordinary_fixed_output.distance_bits == ordinary_output.distance_bits,
      "occupancy",
      "legal fixed ordinary residency changed occupancy or final bits");
}

void test_maximum_rounds() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  const std::array<bfnew::RouteQuery, 1U> singleton{fixture.queries[2U]};
  const bfnew::test::PreparedBatchedFrontierFixture prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedFrontierWorkspace workspace;
  bfnew::hip::BatchedFrontierPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};

  for (const bool per_lane : {true, false}) {
    for (const bfnew::GpuRunOptions& options :
         control_matrix(per_lane, 1U)) {
      const bfnew::hip::BatchedFrontierRunOutput exhausted =
          engine.run_status_only(prepared.description, options);
      const std::string name =
          "maximum-rounds-control-" +
          std::to_string(
              static_cast<std::uint32_t>(options.control_mode)) +
          "-K-" + std::to_string(options.rounds_per_chunk) +
          (per_lane ? "-enabled" : "-disabled");
      expect(
          !exhausted.distances_downloaded && exhausted.distances.empty() &&
              exhausted.distance_bits.empty(),
          name,
          "maximum-round status-only run downloaded distance storage");
      expect(
          bfnew::validate_device_controller(exhausted.final_controller) ==
                  bfnew::DeviceControllerError::none &&
              bfnew::validate_device_run_status(exhausted.result.status) ==
                  bfnew::DeviceRunStatusError::none &&
              exhausted.result.status.stop_reason ==
                  static_cast<std::uint32_t>(
                      bfnew::DeviceStopReason::maximum_rounds) &&
              exhausted.result.status.error_bits ==
                  bfnew::device_error::none &&
              exhausted.result.status.converged == 0U &&
              exhausted.result.status.rounds_completed == 1U &&
              exhausted.result.status.valid_lane_mask == 1U &&
              exhausted.result.status.active_lane_mask == 1U &&
              exhausted.result.status.converged_lane_mask == 0U &&
              exhausted.result.status.reached_target_mask == 0U &&
              exhausted.result.status.bounding_box_miss_mask == 0U &&
              exhausted.final_controller.frontier_read_slot == 1U &&
              exhausted.final_controller.frontier_write_slot == 0U &&
              exhausted.final_controller.frontier_size[1U] != 0U,
          name,
          "round exhaustion lost its live next frontier or terminal identity");
      expect(
          exhausted.frontier_rounds_by_lane ==
                  std::vector<std::uint64_t>{1U} &&
              exhausted.lane_convergence_rounds ==
                  std::vector<std::uint64_t>{0U} &&
              exhausted.lane_tail_rounds ==
                  std::vector<std::uint64_t>{0U} &&
              exhausted.batch_work.frontier_rounds == 1U &&
              exhausted.batch_work.empty_frontier_rounds == 0U &&
              exhausted.batch_work.active_lane_rounds == 1U &&
              exhausted.batch_work.overflow_events == 0U,
          name,
          "unfinished lane published convergence, tail, or overflow evidence");
      expect_control_accounting(options, exhausted, name);
      expect_instrumentation_gating(options, exhausted, name);
    }
  }

  const std::array<bfnew::RouteQuery, 2U> mixed_queries{
      fixture.queries[0U], fixture.queries[2U]};
  const bfnew::test::PreparedBatchedFrontierFixture mixed_prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          mixed_queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  const bfnew::BatchPlanEntry& mixed_batch =
      mixed_prepared.plan.batches.front();
  const std::size_t immediate_lane =
      bfnew::test::frontier_lane_for_query(
          mixed_batch, bfnew::QueryId{1400U});
  const std::size_t live_lane = bfnew::test::frontier_lane_for_query(
      mixed_batch, bfnew::QueryId{1402U});
  const bfnew::LaneMask immediate_bit =
      bfnew::LaneMask{1U} << immediate_lane;
  const bfnew::LaneMask live_bit = bfnew::LaneMask{1U} << live_lane;

  for (const bool per_lane : {true, false}) {
    for (const bfnew::GpuRunOptions& options :
         control_matrix(per_lane, 1U)) {
      const bfnew::hip::BatchedFrontierRunOutput exhausted =
          engine.run_status_only(mixed_prepared.description, options);
      const std::string name =
          "mixed-maximum-rounds-control-" +
          std::to_string(
              static_cast<std::uint32_t>(options.control_mode)) +
          "-K-" + std::to_string(options.rounds_per_chunk) +
          (per_lane ? "-enabled" : "-disabled");
      expect(
          !exhausted.distances_downloaded && exhausted.distances.empty() &&
              exhausted.distance_bits.empty() &&
              bfnew::validate_device_controller(
                  exhausted.final_controller) ==
                  bfnew::DeviceControllerError::none &&
              bfnew::validate_device_run_status(exhausted.result.status) ==
                  bfnew::DeviceRunStatusError::none,
          name,
          "mixed maximum-round status-only result is malformed");
      expect(
          exhausted.result.status.stop_reason ==
                  static_cast<std::uint32_t>(
                      bfnew::DeviceStopReason::maximum_rounds) &&
              exhausted.result.status.error_bits ==
                  bfnew::device_error::none &&
              exhausted.result.status.converged == 0U &&
              exhausted.result.status.rounds_completed == 1U &&
              exhausted.result.status.valid_lane_mask ==
                  (immediate_bit | live_bit) &&
              exhausted.result.status.active_lane_mask == live_bit &&
              exhausted.result.status.converged_lane_mask ==
                  (per_lane ? immediate_bit : 0U) &&
              exhausted.result.status.reached_target_mask == 0U &&
              exhausted.result.status.bounding_box_miss_mask == 0U &&
              exhausted.final_controller.frontier_size
                      [exhausted.final_controller.frontier_read_slot] != 0U,
          name,
          "clean mixed exhaustion lost its live lane or invented a result");
      expect(
          exhausted.lane_convergence_rounds[immediate_lane] == 1U &&
              exhausted.lane_convergence_rounds[live_lane] == 0U &&
              exhausted.frontier_rounds_by_lane[immediate_lane] == 1U &&
              exhausted.frontier_rounds_by_lane[live_lane] == 1U &&
              exhausted.lane_tail_rounds[immediate_lane] == 0U &&
              exhausted.lane_tail_rounds[live_lane] == 0U &&
              exhausted.batch_work.frontier_rounds == 1U &&
              exhausted.batch_work.empty_frontier_rounds == 0U &&
              exhausted.batch_work.active_lane_rounds == 2U &&
              exhausted.batch_work.tail_lane_rounds == 0U &&
              exhausted.batch_work.overflow_events == 0U,
          name,
          "clean maximum transition did not publish only the proven no-next lane");
      for (std::size_t lane = 0U;
           lane < mixed_prepared.description.lane_width;
           ++lane) {
        if (lane == immediate_lane || lane == live_lane) {
          continue;
        }
        expect(
            exhausted.lane_convergence_rounds[lane] == 0U &&
                exhausted.frontier_rounds_by_lane[lane] == 0U &&
                exhausted.lane_tail_rounds[lane] == 0U,
            name,
            "mixed maximum-round padding retained lane evidence");
      }
      expect_control_accounting(options, exhausted, name);
      expect_instrumentation_gating(options, exhausted, name);
    }
  }
}

void test_small_capacity_round_overflow() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_single_query_batched_frontier_fixture(
          bfnew::test::make_expanding_grid_frontier_fixture());
  const bfnew::test::PreparedBatchedFrontierFixture prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          fixture.queries,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedFrontierWorkspace workspace;
  bfnew::hip::BatchedFrontierPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream,
      1U};

  for (const bool per_lane : {true, false}) {
    for (const bfnew::GpuRunOptions& options : control_matrix(per_lane)) {
      const bfnew::hip::BatchedFrontierRunOutput overflow =
          engine.run_status_only(prepared.description, options);
      const std::string name =
          "round-overflow-control-" +
          std::to_string(
              static_cast<std::uint32_t>(options.control_mode)) +
          "-K-" + std::to_string(options.rounds_per_chunk) +
          (per_lane ? "-enabled" : "-disabled");
      expect(
          bfnew::validate_device_controller(overflow.final_controller) ==
                  bfnew::DeviceControllerError::none &&
              bfnew::validate_device_run_status(overflow.result.status) ==
                  bfnew::DeviceRunStatusError::none &&
              overflow.metrics.queue_capacity == 1U &&
              overflow.result.status.stop_reason ==
                  static_cast<std::uint32_t>(
                      bfnew::DeviceStopReason::queue_overflow) &&
              overflow.result.status.error_bits ==
                  bfnew::device_error::queue_overflow &&
              overflow.result.status.rounds_completed == 1U &&
              overflow.result.status.converged == 0U &&
              overflow.result.status.active_lane_mask == 0U &&
              overflow.result.status.converged_lane_mask == 0U &&
              overflow.result.status.reached_target_mask == 0U &&
              overflow.result.status.bounding_box_miss_mask == 0U &&
              overflow.final_controller.frontier_size
                      [overflow.final_controller.frontier_read_slot] > 1U,
          name,
          "capacity-one round overflow was silent or misclassified");
      expect(
          !overflow.distances_downloaded && overflow.distances.empty() &&
              overflow.distance_bits.empty() &&
              overflow.frontier_rounds_by_lane ==
                  std::vector<std::uint64_t>{1U} &&
              overflow.lane_convergence_rounds ==
                  std::vector<std::uint64_t>{0U} &&
              overflow.lane_tail_rounds ==
                  std::vector<std::uint64_t>{0U} &&
              overflow.batch_work.frontier_rounds == 1U &&
              overflow.batch_work.empty_frontier_rounds == 0U &&
              overflow.batch_work.maximum_queue_size > 1U &&
              overflow.batch_work.overflow_events == 1U &&
              overflow.result.work.maximum_queue_size > 1U &&
              overflow.result.work.overflow_events == 1U,
          name,
          "capacity-one overflow lost queue high-water or exactly-once error evidence");
      expect_control_accounting(options, overflow, name);
      expect_instrumentation_gating(options, overflow, name);
    }
  }
}

void test_initial_source_overflow_and_guards() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  const bfnew::test::PreparedBatchedFrontierFixture prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedFrontierWorkspace small_workspace;
  bfnew::hip::BatchedFrontierPushEngine small_engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      small_workspace,
      stream,
      1U};
  const bfnew::GpuRunOptions options =
      options_for(bfnew::ControlMode::persistent_cooperative, true);
  const bfnew::hip::BatchedFrontierRunOutput overflow =
      small_engine.run_status_only(prepared.description, options);
  expect(
      bfnew::validate_device_controller(overflow.final_controller) ==
              bfnew::DeviceControllerError::none &&
          bfnew::validate_device_run_status(overflow.result.status) ==
              bfnew::DeviceRunStatusError::none &&
          overflow.result.status.stop_reason ==
              static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::queue_overflow) &&
          overflow.result.status.error_bits ==
              bfnew::device_error::queue_overflow &&
          overflow.result.status.rounds_completed == 0U &&
          overflow.result.status.reached_target_mask == 0U &&
          overflow.result.status.bounding_box_miss_mask == 0U &&
          overflow.batch_work.initial_source_lane_activations ==
              prepared.description.sources.size() &&
          overflow.batch_work.initial_queue_entries ==
              distinct_source_count(prepared.description) &&
          overflow.batch_work.initial_queue_entries_saved_by_lane_merging !=
              0U &&
          overflow.batch_work.initial_queue_entries > 1U &&
          overflow.batch_work.frontier_rounds == 0U &&
          overflow.batch_work.overflow_events == 1U &&
          overflow.result.work.overflow_events == 1U,
      "initial-overflow",
      "capacity-one source seeding did not preserve merge accounting or overflow");
  expect_control_accounting(options, overflow, "initial-overflow");
  expect_instrumentation_gating(options, overflow, "initial-overflow");

  bfnew::hip::ReusableBatchedFrontierWorkspace workspace;
  bfnew::hip::BatchedFrontierPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};
  const bfnew::test::PreparedBatchedFrontierFixture compact =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::compact_nonzero_descriptors);
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            engine.run_status_only(compact.description, options));
      },
      "guards",
      "HIP batched frontier accepted compact descriptors instead of retained masks");
  bfnew::GpuRunOptions wrong_engine = options;
  wrong_engine.engine = bfnew::EngineKind::dense_chaotic_push;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            engine.run_status_only(prepared.description, wrong_engine));
      },
      "guards",
      "HIP batched frontier accepted another engine identity");
}

void test_malformed_retained_inputs_and_options() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  const bfnew::test::PreparedBatchedFrontierFixture prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  const std::array<bfnew::RouteQuery, 1U> singleton_queries{
      fixture.queries.front()};
  const bfnew::test::PreparedBatchedFrontierFixture singleton =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          singleton_queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);
  bfnew::hip::ReusableBatchedFrontierWorkspace workspace;
  bfnew::hip::BatchedFrontierPushEngine engine{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream};
  const bfnew::GpuRunOptions valid = options_for(
      bfnew::ControlMode::per_round_host_poll,
      true,
      bfnew::InstrumentationLevel::debug);

  const auto reject_description = [&engine, &valid](
                                      const bfnew::BatchDeviceDescription& bad,
                                      const std::string_view reason) {
    expect_throws<std::invalid_argument>(
        [&] { static_cast<void>(engine.run_status_only(bad, valid)); },
        "malformed-retained-input",
        reason);
  };
  const auto reject_mutation = [&prepared, &reject_description](
                                   auto&& mutate,
                                   const std::string_view reason) {
    bfnew::BatchDeviceDescription bad = prepared.description;
    mutate(bad);
    reject_description(bad, reason);
  };

  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) { bad.lane_width = 2U; },
      "accepted an unsupported lane width");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.valid_lane_mask = 0U;
      },
      "accepted an empty valid-lane mask");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.valid_lane_mask = 0x5U;
      },
      "accepted a non-prefix valid-lane mask");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.valid_lane_mask |= bfnew::LaneMask{1U} << bad.lane_width;
      },
      "accepted a valid-lane bit outside the configured width");

  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.query_ids_by_lane.pop_back();
      },
      "accepted a short query-id lane image");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.expansion_generations_by_lane.pop_back();
      },
      "accepted a short expansion-generation lane image");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.selected_vertex_counts_by_lane.pop_back();
      },
      "accepted a short selected-vertex lane image");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.selected_edge_estimates_by_lane.pop_back();
      },
      "accepted a short selected-edge lane image");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.query_ids_by_lane[1U] = bad.query_ids_by_lane[0U];
      },
      "accepted duplicate query IDs in valid lanes");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        const std::size_t padding_lane =
            static_cast<std::size_t>(std::popcount(bad.valid_lane_mask));
        bad.query_ids_by_lane[padding_lane] = 0U;
      },
      "accepted a noncanonical padded query ID");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        const std::size_t padding_lane =
            static_cast<std::size_t>(std::popcount(bad.valid_lane_mask));
        bad.expansion_generations_by_lane[padding_lane] = 1U;
      },
      "accepted a nonzero padded expansion generation");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        const std::size_t padding_lane =
            static_cast<std::size_t>(std::popcount(bad.valid_lane_mask));
        bad.selected_vertex_counts_by_lane[padding_lane] = 1U;
      },
      "accepted a nonzero padded selected-vertex count");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        const std::size_t padding_lane =
            static_cast<std::size_t>(std::popcount(bad.valid_lane_mask));
        bad.selected_edge_estimates_by_lane[padding_lane] = 1U;
      },
      "accepted a nonzero padded selected-edge estimate");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.source_offsets.pop_back();
      },
      "accepted malformed source offsets");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.target_offsets.pop_back();
      },
      "accepted malformed target offsets");

  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.reached_lane_mask = 1U;
      },
      "accepted a nonzero initial reached mask");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) { bad.miss_lane_mask = 1U; },
      "accepted a nonzero initial miss mask");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.tile_lane_masks.pop_back();
      },
      "accepted a short dense tile-mask image");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.run_representation =
            bfnew::BatchRunRepresentation::compact_nonzero_descriptors;
      },
      "accepted compact descriptors instead of retained masks");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.run_representation =
            static_cast<bfnew::BatchRunRepresentation>(255U);
      },
      "accepted an unknown retained-run representation");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.run_representation_initialized = false;
      },
      "accepted an uninitialized retained-run image");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.csr_run_lane_masks.pop_back();
      },
      "accepted a short retained CSR mask image");

  expect(
      prepared.description.union_tiles.size() > 1U &&
          !prepared.description.selected_vertex_ranges.empty(),
      "malformed-retained-input",
      "fixture does not expose canonical-union guard seams");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) { bad.union_tiles.clear(); },
      "accepted an empty selected union");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.selected_vertex_ranges.pop_back();
      },
      "accepted a union/range count mismatch");
  reject_mutation(
      [&fixture](bfnew::BatchDeviceDescription& bad) {
        bad.union_tiles.back() = static_cast<std::uint32_t>(
            fixture.partitioned.graph.tile_coordinates().size());
      },
      "accepted an out-of-range union tile");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.union_tiles[1U] = bad.union_tiles[0U];
      },
      "accepted noncanonical duplicate union tiles");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        ++bad.selected_vertex_ranges.front().begin;
      },
      "accepted a selected range with the wrong tile bounds");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        bad.selected_vertex_ranges.front().lane_mask = 0U;
      },
      "accepted a zero selected-range lane mask");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        const std::uint32_t tile = bad.union_tiles.front();
        const bfnew::LaneMask padded = bfnew::LaneMask{1U} << 7U;
        bad.tile_lane_masks[tile] |= padded;
        bad.selected_vertex_ranges.front().lane_mask |= padded;
      },
      "accepted a padded lane in tile and range masks");

  bfnew::BatchDeviceDescription outside_union = singleton.description;
  const auto outside_tile = std::ranges::find(
      outside_union.tile_lane_masks, bfnew::LaneMask{0U});
  expect(
      outside_tile != outside_union.tile_lane_masks.end(),
      "malformed-retained-input",
      "singleton fixture has no tile outside its union");
  outside_union.tile_lane_masks[static_cast<std::size_t>(
      std::distance(outside_union.tile_lane_masks.begin(), outside_tile))] =
      1U;
  reject_description(
      outside_union,
      "accepted a nonzero tile mask outside the selected union");

  std::size_t outside_source_run =
      std::numeric_limits<std::size_t>::max();
  for (std::size_t vertex = 0U;
       vertex < fixture.partitioned.graph.vertex_count();
       ++vertex) {
    const bool selected = std::ranges::any_of(
        singleton.description.selected_vertex_ranges,
        [vertex](const bfnew::BatchVertexRange range) {
          return range.begin <= vertex && vertex < range.end;
        });
    const std::size_t run_begin = static_cast<std::size_t>(
        fixture.tile_runs.csr_row_run_offsets[vertex]);
    const std::size_t run_end = static_cast<std::size_t>(
        fixture.tile_runs.csr_row_run_offsets[vertex + 1U]);
    if (!selected && run_begin < run_end) {
      outside_source_run = run_begin;
      break;
    }
  }
  expect(
      outside_source_run != std::numeric_limits<std::size_t>::max(),
      "malformed-retained-input",
      "singleton fixture has no CSR run outside selected source rows");
  bfnew::BatchDeviceDescription outside_source = singleton.description;
  outside_source.csr_run_lane_masks[outside_source_run] = 1U;
  reject_description(
      outside_source,
      "accepted a nonzero retained mask outside selected source rows");

  const auto active_run = std::ranges::find_if(
      prepared.description.csr_run_lane_masks,
      [](const bfnew::LaneMask mask) { return mask != 0U; });
  expect(
      active_run != prepared.description.csr_run_lane_masks.end(),
      "malformed-retained-input",
      "fixture has no active retained CSR run");
  const std::size_t active_run_index = static_cast<std::size_t>(
      std::distance(
          prepared.description.csr_run_lane_masks.begin(), active_run));
  reject_mutation(
      [active_run_index](bfnew::BatchDeviceDescription& bad) {
        bad.csr_run_lane_masks[active_run_index] = 0U;
      },
      "accepted a CSR mask that disagrees with endpoint admission");
  reject_mutation(
      [active_run_index](bfnew::BatchDeviceDescription& bad) {
        bad.csr_run_lane_masks[active_run_index] |=
            bfnew::LaneMask{1U} << 7U;
      },
      "accepted a padded lane in a retained CSR mask");

  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        const std::uint32_t removed =
            bad.source_offsets[1U] - bad.source_offsets[0U];
        bad.sources.erase(
            bad.sources.begin(), bad.sources.begin() + removed);
        for (std::size_t index = 1U; index < bad.source_offsets.size();
             ++index) {
          bad.source_offsets[index] -= removed;
        }
      },
      "accepted a valid lane without a source");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        const std::uint32_t removed =
            bad.target_offsets[1U] - bad.target_offsets[0U];
        bad.targets.erase(
            bad.targets.begin(), bad.targets.begin() + removed);
        for (std::size_t index = 1U; index < bad.target_offsets.size();
             ++index) {
          bad.target_offsets[index] -= removed;
        }
      },
      "accepted a valid lane without a target");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        const std::size_t padding_lane =
            static_cast<std::size_t>(std::popcount(bad.valid_lane_mask));
        bad.sources.push_back(bad.sources.front());
        for (std::size_t index = padding_lane + 1U;
             index < bad.source_offsets.size();
             ++index) {
          ++bad.source_offsets[index];
        }
      },
      "accepted a source payload in a padded lane");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        const std::size_t insert = bad.source_offsets[0U] + 1U;
        bad.sources.insert(
            bad.sources.begin() + static_cast<std::ptrdiff_t>(insert),
            bad.sources.front());
        for (std::size_t index = 1U; index < bad.source_offsets.size();
             ++index) {
          ++bad.source_offsets[index];
        }
      },
      "accepted duplicate sources within one lane slice");
  reject_mutation(
      [](bfnew::BatchDeviceDescription& bad) {
        const std::size_t insert = bad.target_offsets[0U] + 1U;
        bad.targets.insert(
            bad.targets.begin() + static_cast<std::ptrdiff_t>(insert),
            bad.targets.front());
        for (std::size_t index = 1U; index < bad.target_offsets.size();
             ++index) {
          ++bad.target_offsets[index];
        }
      },
      "accepted duplicate targets within one lane slice");
  reject_mutation(
      [&fixture](bfnew::BatchDeviceDescription& bad) {
        bad.sources.front() = static_cast<std::uint32_t>(
            fixture.partitioned.graph.vertex_count());
      },
      "accepted an out-of-range source terminal");
  reject_mutation(
      [&fixture](bfnew::BatchDeviceDescription& bad) {
        bad.targets.front() = static_cast<std::uint32_t>(
            fixture.partitioned.graph.vertex_count());
      },
      "accepted an out-of-range target terminal");

  std::uint32_t unadmitted_vertex = static_cast<std::uint32_t>(
      fixture.partitioned.graph.vertex_count());
  for (std::size_t vertex = 0U;
       vertex < fixture.partitioned.graph.vertex_count();
       ++vertex) {
    const std::uint32_t tile =
        fixture.partitioned.graph.owner_tiles()[vertex].value();
    if ((prepared.description.tile_lane_masks[tile] & 1U) == 0U) {
      unadmitted_vertex = static_cast<std::uint32_t>(vertex);
      break;
    }
  }
  expect(
      unadmitted_vertex < fixture.partitioned.graph.vertex_count(),
      "malformed-retained-input",
      "fixture has no terminal outside lane zero's admitted tiles");
  reject_mutation(
      [unadmitted_vertex](bfnew::BatchDeviceDescription& bad) {
        bad.sources.front() = unadmitted_vertex;
      },
      "accepted a terminal whose owner tile does not admit its lane");

  const auto reject_options = [&engine, &prepared](
                                  const bfnew::GpuRunOptions& bad,
                                  const std::string_view reason) {
    expect_throws<std::invalid_argument>(
        [&] {
          static_cast<void>(
              engine.run_status_only(prepared.description, bad));
        },
        "malformed-options",
        reason);
  };
  const auto reject_option_mutation = [&valid, &reject_options](
                                          auto&& mutate,
                                          const std::string_view reason) {
    bfnew::GpuRunOptions bad = valid;
    mutate(bad);
    reject_options(bad, reason);
  };
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) {
        bad.engine = static_cast<bfnew::EngineKind>(255U);
      },
      "accepted an unknown engine value");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) {
        bad.engine = bfnew::EngineKind::jacobi_pull;
      },
      "accepted a valid but different engine");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) {
        bad.control_mode = static_cast<bfnew::ControlMode>(255U);
      },
      "accepted an unknown control mode");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) { bad.rounds_per_chunk = 0U; },
      "accepted a zero chunk size");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) { bad.block_size = 0U; },
      "accepted a zero block size");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) {
        bad.grid_policy = static_cast<bfnew::GridPolicy>(255U);
      },
      "accepted an unknown grid policy");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) {
        bad.grid_policy = bfnew::GridPolicy::fixed_blocks_per_wgp;
        bad.blocks_per_wgp = 0U;
      },
      "accepted a zero fixed block count");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) { bad.blocks_per_wgp = 1U; },
      "accepted fixed blocks under the occupancy-derived policy");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) {
        bad.instrumentation =
            static_cast<bfnew::InstrumentationLevel>(255U);
      },
      "accepted an unknown instrumentation level");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) { bad.maximum_rounds = 0U; },
      "accepted a zero maximum-round bound");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) {
        bad.enable_per_lane_convergence = 2U;
      },
      "accepted a non-Boolean per-lane convergence flag");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) {
        bad.block_size = std::numeric_limits<std::uint32_t>::max();
      },
      "accepted a block size above the selected device limit");
  reject_option_mutation(
      [](bfnew::GpuRunOptions& bad) {
        bad.grid_policy = bfnew::GridPolicy::fixed_blocks_per_wgp;
        bad.blocks_per_wgp = std::numeric_limits<std::uint32_t>::max();
      },
      "accepted an ordinary fixed grid above kernel occupancy");

  bfnew::GpuRunOptions cooperative_oversubscription = valid;
  cooperative_oversubscription.control_mode =
      bfnew::ControlMode::persistent_cooperative;
  cooperative_oversubscription.grid_policy =
      bfnew::GridPolicy::fixed_blocks_per_wgp;
  cooperative_oversubscription.blocks_per_wgp =
      std::numeric_limits<std::uint32_t>::max();
  reject_options(
      cooperative_oversubscription,
      "accepted a cooperative fixed grid above kernel occupancy");

  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::hip::ReusableBatchedFrontierWorkspace oversized_workspace;
        bfnew::hip::BatchedFrontierPushEngine oversized{
            fixture.partitioned.graph,
            fixture.tile_runs,
            resident,
            oversized_workspace,
            stream,
            static_cast<std::uint32_t>(
                fixture.partitioned.graph.vertex_count() + 1U)};
        static_cast<void>(oversized);
      },
      "malformed-options",
      "accepted a queue capacity above the vertex count");

  expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(
            bfnew::hip::batched_frontier_scratch_bytes(0U, 8U));
      },
      "malformed-options",
      "accepted zero vertices in the HIP scratch planner");
  expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(
            bfnew::hip::batched_frontier_scratch_bytes(1U, 2U));
      },
      "malformed-options",
      "accepted an unsupported width in the HIP scratch planner");
  expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(
            bfnew::hip::batched_frontier_scratch_bytes(1U, 1U, 2U));
      },
      "malformed-options",
      "accepted an oversized queue in the HIP scratch planner");
}

}  // namespace

int main() {
  expect(
      bfnew::supported_batched_frontier_width(1U) &&
          bfnew::supported_batched_frontier_width(8U) &&
          bfnew::supported_batched_frontier_width(16U) &&
          bfnew::supported_batched_frontier_width(32U) &&
          !bfnew::supported_batched_frontier_width(2U) &&
          !bfnew::supported_batched_frontier_width(64U),
      "supported-widths",
      "device API does not expose exactly widths 1, 8, 16, and 32");
  test_device_controller_error_canonicalization();
  test_scratch_layout();
  test_mixed_width_control_matrix();
  test_width_one_matches_standalone();
  test_schedule_robust_contention_and_zero_cycle();
  test_width32_reused_workspace_convergence_reset();
  test_instrumentation_levels();
  test_status_only_parity();
  test_legal_occupancy_grids();
  test_maximum_rounds();
  test_small_capacity_round_overflow();
  test_initial_source_overflow_and_guards();
  test_malformed_retained_inputs_and_options();
  std::cout << "batched frontier push HIP tests passed\n";
  return 0;
}
