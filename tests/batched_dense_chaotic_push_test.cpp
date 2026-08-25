#include "bfnew/batched_dense_chaotic_push.hpp"
#include "bfnew/sssp.hpp"
#include "batched_dense_fixture_suite.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

[[nodiscard]] bool bits_equal(const float left, const float right) noexcept {
  return std::bit_cast<std::uint32_t>(left) ==
         std::bit_cast<std::uint32_t>(right);
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
  for (std::size_t vertex = 0U; vertex < induced.local_to_global.size();
       ++vertex) {
    global[induced.local_to_global[vertex].value()] = local.distances[vertex];
  }
  return global;
}

[[nodiscard]] bool query_selects_vertex(
    const bfnew::WeightedGraph& graph,
    const bfnew::RouteQuery& query,
    const std::size_t vertex) {
  return std::binary_search(
      query.selected_tiles.begin(),
      query.selected_tiles.end(),
      graph.owner_tiles()[vertex]);
}

[[nodiscard]] const bfnew::RouteQuery& query_for_lane(
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchPlanEntry& batch,
    const std::size_t lane) {
  for (const bfnew::RouteQuery& query : queries) {
    if (query.query_id == batch.query_ids_by_lane[lane]) {
      return query;
    }
  }
  throw std::logic_error{"batched dense test lane query is missing"};
}

[[nodiscard]] std::vector<bfnew::GpuRunOptions> control_matrix(
    const bool per_lane_convergence,
    const std::uint64_t maximum_rounds = 64U) {
  bfnew::GpuRunOptions base;
  base.engine = bfnew::EngineKind::dense_chaotic_push;
  base.instrumentation = bfnew::InstrumentationLevel::debug;
  base.maximum_rounds = maximum_rounds;
  base.enable_per_lane_convergence = per_lane_convergence ? 1U : 0U;

  std::vector<bfnew::GpuRunOptions> controls;
  bfnew::GpuRunOptions persistent = base;
  persistent.control_mode = bfnew::ControlMode::persistent_cooperative;
  controls.push_back(persistent);
  bfnew::GpuRunOptions per_round = base;
  per_round.control_mode = bfnew::ControlMode::per_round_host_poll;
  controls.push_back(per_round);
  for (const std::uint32_t k : {2U, 4U, 8U, 16U, 32U}) {
    bfnew::GpuRunOptions chunked = base;
    chunked.control_mode = bfnew::ControlMode::chunked_host_poll;
    chunked.rounds_per_chunk = k;
    controls.push_back(chunked);
  }
  return controls;
}

void expect_control_accounting(
    const bfnew::GpuRunOptions& options,
    const bfnew::HostBatchedDenseRunResult& run,
    const std::string& prefix) {
  const std::uint64_t rounds = run.controller.rounds_completed;
  const bfnew::DeviceWorkStatistics& work = run.result.work;
  if (options.control_mode == bfnew::ControlMode::persistent_cooperative) {
    expect(
        work.kernel_dispatches == 1U && work.controller_copies == 1U &&
            work.host_synchronizations == 1U && work.host_checks == 1U &&
            run.queued_round_pairs == 0U &&
            run.completed_host_chunks == 0U,
        prefix + ": persistent control performs one modeled dispatch");
    return;
  }
  if (options.control_mode == bfnew::ControlMode::per_round_host_poll) {
    expect(
        run.queued_round_pairs == rounds &&
            run.completed_host_chunks == rounds &&
            work.controller_copies == rounds &&
            work.host_synchronizations == rounds && work.host_checks == rounds &&
            work.kernel_dispatches == 1U + 2U * rounds,
        prefix + ": per-round control counts exact scan/advance pairs");
    return;
  }
  const std::uint64_t chunks =
      (rounds + options.rounds_per_chunk - 1U) / options.rounds_per_chunk;
  const std::uint64_t pairs = chunks * options.rounds_per_chunk;
  expect(
      run.completed_host_chunks == chunks &&
          run.queued_round_pairs == pairs &&
          work.controller_copies == chunks &&
          work.host_synchronizations == chunks && work.host_checks == chunks &&
          work.kernel_dispatches == 1U + 2U * pairs,
      prefix + ": chunked control retains queued terminal no-op pairs");
}

void validate_output(
    const bfnew::test::BatchedDenseFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchPlanEntry& batch,
    const bfnew::BatchDeviceDescription& description,
    const bfnew::GpuRunOptions& options,
    const bfnew::HostBatchedDenseRunResult& run,
    const std::string& prefix) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::size_t elements =
      static_cast<std::size_t>(graph.vertex_count()) * batch.lane_width;
  expect(
      run.distances.size() == elements && run.distance_bits.size() == elements,
      prefix + ": dense distances use exact V-by-W atomic storage");
  expect(
      run.rounds_executed_by_lane.size() == batch.lane_width &&
          run.convergence_round_by_lane.size() == batch.lane_width &&
          run.tail_rounds_by_lane.size() == batch.lane_width,
      prefix + ": convergence evidence has one entry per lane");
  expect(
      run.result.engine_kind == static_cast<std::uint32_t>(
                                    bfnew::EngineKind::dense_chaotic_push) &&
          run.result.control_mode ==
              static_cast<std::uint32_t>(options.control_mode),
      prefix + ": result retains dense and control identities");
  expect(
      bfnew::validate_device_controller(run.controller) ==
              bfnew::DeviceControllerError::none &&
          bfnew::validate_device_run_status(run.result.status) ==
              bfnew::DeviceRunStatusError::none,
      prefix + ": terminal controller and status validate");
  expect(
      run.result.status.stop_reason == static_cast<std::uint32_t>(
                                           bfnew::DeviceStopReason::converged) &&
          run.result.status.valid_lane_mask == batch.valid_lane_mask &&
          run.result.status.converged_lane_mask == batch.valid_lane_mask &&
          run.result.status.final_distance_slot == 0U,
      prefix + ": every lane converges in the single in-place slot");

  bfnew::LaneMask expected_reached = 0U;
  bfnew::LaneMask expected_miss = 0U;
  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    const bfnew::LaneMask bit = bfnew::LaneMask{1U} << lane;
    if ((batch.valid_lane_mask & bit) == 0U) {
      expect(
          run.rounds_executed_by_lane[lane] == 0U &&
              run.convergence_round_by_lane[lane] == 0U &&
              run.tail_rounds_by_lane[lane] == 0U,
          prefix + ": padded lanes have no semantic scan state");
      continue;
    }
    const bfnew::RouteQuery& query = query_for_lane(queries, batch, lane);
    const std::vector<float> oracle = bounded_oracle(graph, query);
    for (std::size_t vertex = 0U; vertex < graph.vertex_count(); ++vertex) {
      if (!query_selects_vertex(graph, query, vertex)) {
        continue;
      }
      const std::size_t index = static_cast<std::size_t>(
          bfnew::batched_dense_distance_index(
              static_cast<std::uint32_t>(vertex),
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      expect(
          bits_equal(run.distances[index], oracle[vertex]) &&
              run.distance_bits[index] ==
                  bfnew::dense_atomic_float_bits(run.distances[index]),
          prefix + ": every selected atomic label matches bounded Dijkstra");
      if (lane + 1U < batch.lane_width) {
        expect(
            bfnew::batched_dense_distance_index(
                static_cast<std::uint32_t>(vertex),
                static_cast<std::uint32_t>(lane + 1U),
                batch.lane_width) == index + 1U,
            prefix + ": query words are contiguous within a vertex");
      }
    }
    for (const bfnew::VertexId source : query.sources) {
      const std::size_t index = static_cast<std::size_t>(
          bfnew::batched_dense_distance_index(
              source.value(),
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      expect(
          run.distance_bits[index] == 0U,
          prefix + ": each lane retains only its own canonical zero sources");
    }
    bool reached = true;
    for (const bfnew::VertexId target : query.targets) {
      const std::size_t index = static_cast<std::size_t>(
          bfnew::batched_dense_distance_index(
              target.value(),
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      reached = reached && std::isfinite(run.distances[index]);
    }
    if (reached) {
      expected_reached |= bit;
    } else {
      expected_miss |= bit;
    }
    if (options.enable_per_lane_convergence != 0U) {
      expect(
          run.rounds_executed_by_lane[lane] ==
              run.convergence_round_by_lane[lane],
          prefix + ": enabled convergence freezes at the first no-change scan");
    } else {
      expect(
          run.rounds_executed_by_lane[lane] ==
              run.controller.rounds_completed,
          prefix + ": disabled convergence executes every batch scan");
    }
  }
  expect(
      run.result.status.reached_target_mask == expected_reached &&
          run.result.status.bounding_box_miss_mask == expected_miss,
      prefix + ": reached and bounded-miss masks are lane exact");

  const bfnew::BatchedDenseWorkStatistics& dense = run.batch_work;
  const bfnew::DeviceWorkStatistics& common = run.result.work;
  expect(
      dense.csr_runs_considered ==
              dense.csr_runs_visited + dense.csr_runs_skipped &&
          dense.csr_runs_visited != 0U &&
          dense.active_lanes_over_visited_runs >= dense.csr_runs_visited,
      prefix + ": considered/visited/skipped CSR vocabulary balances");
  expect(
      dense.csr_edge_loads <= dense.lane_edge_relaxations &&
          dense.lane_edge_relaxations == dense.atomic_source_loads &&
          dense.atomic_source_loads == dense.atomic_min_attempts &&
          dense.successful_atomic_updates <= dense.atomic_min_attempts,
      prefix + ": shared loads and per-lane atomic operations are exact");
  expect(
      common.edges_examined == dense.lane_edge_relaxations &&
          common.atomic_attempts == dense.atomic_min_attempts &&
          common.successful_decreases ==
              dense.successful_atomic_updates &&
          common.successful_atomic_updates ==
              dense.successful_atomic_updates &&
          common.active_vertices == dense.active_source_lane_evaluations &&
          common.active_lane_rounds == dense.active_lane_rounds &&
          common.mask_operations == dense.csr_runs_considered &&
          common.changed_flag_updates == dense.changed_round_publications &&
          common.full_edge_rounds == dense.full_edge_rounds,
      prefix + ": common debug counters match the batched dense vocabulary");
  expect(
      dense.full_edge_rounds == run.controller.rounds_completed &&
          dense.valid_lane_round_capacity ==
              dense.active_lane_rounds + dense.inactive_valid_lane_rounds &&
          dense.lane_width_round_capacity ==
              dense.valid_lane_round_capacity +
                  dense.padded_lane_round_capacity &&
          dense.wave32_lane_round_capacity ==
              run.controller.rounds_completed *
                  bfnew::maximum_batch_lanes &&
          dense.wave32_lane_round_capacity ==
              dense.lane_width_round_capacity +
                  dense.unused_wave_lane_round_capacity &&
          dense.edge_wave_lane_capacity ==
              dense.csr_edge_loads * bfnew::maximum_batch_lanes &&
          dense.edge_wave_lane_capacity ==
              dense.lane_edge_relaxations +
                  dense.unused_edge_wave_lane_capacity &&
          dense.padded_lane_semantic_work == 0U &&
          dense.frontier_semantic_work == 0U && common.queue_claims == 0U &&
          common.duplicate_suppressions == 0U &&
          common.maximum_queue_size == 0U,
      prefix + ": scan utilization contains no padding or frontier work");
  expect(
      dense.distance_reset_bytes ==
              batch.selected_lane_vertex_count * sizeof(std::uint32_t) &&
          dense.source_seed_write_bytes ==
              description.sources.size() * sizeof(std::uint32_t),
      prefix + ": reset and seed bytes cover admitted atomic words only");
  expect(
      dense.union_tile_lane_positions ==
              batch.union_tiles.size() *
                  static_cast<std::uint64_t>(
                      std::popcount(batch.valid_lane_mask)) &&
          dense.selected_tile_lane_positions != 0U &&
          dense.union_tile_lane_positions >=
              dense.selected_tile_lane_positions,
      prefix + ": union-tile inflation has exact integer terms");
  expect_control_accounting(options, run, prefix);
}

[[nodiscard]] bool selected_outputs_equal(
    const bfnew::test::BatchedDenseFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchPlanEntry& batch,
    const bfnew::HostBatchedDenseRunResult& left,
    const bfnew::HostBatchedDenseRunResult& right) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    if ((batch.valid_lane_mask & (bfnew::LaneMask{1U} << lane)) == 0U) {
      continue;
    }
    const bfnew::RouteQuery& query = query_for_lane(queries, batch, lane);
    for (std::size_t vertex = 0U; vertex < graph.vertex_count(); ++vertex) {
      if (!query_selects_vertex(graph, query, vertex)) {
        continue;
      }
      const std::size_t index = static_cast<std::size_t>(
          bfnew::batched_dense_distance_index(
              static_cast<std::uint32_t>(vertex),
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      if (left.distance_bits[index] != right.distance_bits[index]) {
        return false;
      }
    }
  }
  return true;
}

void test_mixed_duration_matrix() {
  const bfnew::test::BatchedDenseFixture fixture =
      bfnew::test::make_mixed_duration_batched_dense_fixture();
  for (const bfnew::BatchRunRepresentation representation : {
           bfnew::BatchRunRepresentation::retained_per_run_masks,
           bfnew::BatchRunRepresentation::compact_nonzero_descriptors,
       }) {
    for (const std::uint32_t width : {8U, 16U, 32U}) {
      const bfnew::test::PreparedBatchedDenseFixture prepared =
          bfnew::test::prepare_batched_dense_fixture(
              fixture, fixture.queries, width, representation);
      const bfnew::BatchPlanEntry& batch = prepared.plan.batches.front();
      expect(
          std::popcount(batch.valid_lane_mask) == 5 &&
              batch.lane_width == width,
          "mixed dense batch retains five valid lanes and exact width");
      const std::vector<bfnew::GpuRunOptions> enabled_controls =
          control_matrix(true);
      const std::vector<bfnew::GpuRunOptions> disabled_controls =
          control_matrix(false);
      std::vector<std::uint32_t> first_control_bits;
      for (std::size_t control = 0U; control < enabled_controls.size();
           ++control) {
        const bfnew::GpuRunOptions enabled_options = enabled_controls[control];
        const bfnew::GpuRunOptions disabled_options =
            disabled_controls[control];
        const bfnew::HostBatchedDenseRunResult enabled =
            bfnew::run_host_batched_dense_chaotic_push(
                fixture.device_graph,
                fixture.queries,
                batch,
                prepared.description,
                enabled_options,
                bfnew::DenseHostSchedule::csr_reverse);
        const bfnew::HostBatchedDenseRunResult disabled =
            bfnew::run_host_batched_dense_chaotic_push(
                fixture.device_graph,
                fixture.queries,
                batch,
                prepared.description,
                disabled_options,
                bfnew::DenseHostSchedule::csr_reverse);
        const std::string prefix =
            "dense-representation-" +
            std::to_string(static_cast<std::uint32_t>(representation)) +
            "-width-" + std::to_string(width) + "-control-" +
            std::to_string(control);
        validate_output(
            fixture,
            fixture.queries,
            batch,
            prepared.description,
            enabled_options,
            enabled,
            prefix + "-enabled");
        validate_output(
            fixture,
            fixture.queries,
            batch,
            prepared.description,
            disabled_options,
            disabled,
            prefix + "-disabled");
        expect(
            selected_outputs_equal(
                fixture, fixture.queries, batch, enabled, disabled),
            prefix + ": convergence toggle preserves every selected bit");
        expect(
            enabled.controller.rounds_completed == 6U &&
                disabled.controller.rounds_completed == 6U,
            prefix + ": longest lane takes five changes plus no-change scan");

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
          const std::size_t lane =
              bfnew::test::dense_lane_for_query(batch, query_id);
          expect(
              enabled.convergence_round_by_lane[lane] == expected_round &&
                  disabled.convergence_round_by_lane[lane] == expected_round &&
                  enabled.tail_rounds_by_lane[lane] == 6U - expected_round &&
                  disabled.tail_rounds_by_lane[lane] == 6U - expected_round,
              prefix + ": mixed lane has exact convergence and tail scans");
        }
        expect(
            enabled.batch_work.tail_lane_rounds == 16U &&
                enabled.batch_work.tail_lane_rounds_avoided == 16U &&
                enabled.batch_work.tail_lane_rounds_executed == 0U &&
                enabled.batch_work.inactive_valid_lane_rounds == 16U &&
                disabled.batch_work.tail_lane_rounds == 16U &&
                disabled.batch_work.tail_lane_rounds_avoided == 0U &&
                disabled.batch_work.tail_lane_rounds_executed == 16U &&
                disabled.batch_work.inactive_valid_lane_rounds == 0U,
            prefix + ": enabled/disabled tail accounting is exact");
        expect(
            enabled.batch_work.lane_edge_relaxations +
                    enabled.batch_work
                        .lane_edge_relaxations_avoided_by_early_convergence ==
                disabled.batch_work.lane_edge_relaxations &&
                disabled.batch_work
                        .lane_edge_relaxations_avoided_by_early_convergence ==
                    0U,
            prefix + ": logical edge delta equals early-convergence savings");
        expect(
            enabled.batch_work.csr_runs_skipped != 0U &&
                enabled.batch_work.csr_edge_loads <
                    enabled.batch_work.lane_edge_relaxations &&
                enabled.batch_work.active_lanes_over_visited_runs >
                    enabled.batch_work.csr_runs_visited &&
                enabled.batch_work.successful_atomic_updates != 0U,
            prefix +
                ": fixture exercises zero-mask skips, strict edge reuse, "
                "multi-lane runs, and successful updates");

        const std::size_t unreachable_lane =
            bfnew::test::dense_lane_for_query(batch, bfnew::QueryId{1403U});
        const std::size_t reaching_lane =
            bfnew::test::dense_lane_for_query(batch, bfnew::QueryId{1402U});
        const std::uint32_t shared_vertex =
            query_for_lane(fixture.queries, batch, unreachable_lane)
                .targets.front()
                .value();
        const std::size_t unreachable_index = static_cast<std::size_t>(
            bfnew::batched_dense_distance_index(
                shared_vertex,
                static_cast<std::uint32_t>(unreachable_lane),
                batch.lane_width));
        const std::size_t reaching_index = static_cast<std::size_t>(
            bfnew::batched_dense_distance_index(
                shared_vertex,
                static_cast<std::uint32_t>(reaching_lane),
                batch.lane_width));
        expect(
            std::isinf(enabled.distances[unreachable_index]) &&
                std::isfinite(enabled.distances[reaching_index]),
            prefix + ": one lane cannot seed or update another lane");

        if (first_control_bits.empty()) {
          first_control_bits = enabled.distance_bits;
        } else {
          expect(
              first_control_bits == enabled.distance_bits,
              prefix + ": control mode changes no atomic distance bit");
        }
      }
    }
  }
}

void test_width_one_matches_standalone() {
  const bfnew::test::BatchedDenseFixture fixture =
      bfnew::test::make_mixed_duration_batched_dense_fixture();
  const std::array<bfnew::RouteQuery, 1U> singleton{fixture.queries[2U]};
  const bfnew::test::PreparedBatchedDenseFixture retained =
      bfnew::test::prepare_batched_dense_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  for (const bfnew::BatchRunRepresentation representation : {
           bfnew::BatchRunRepresentation::retained_per_run_masks,
           bfnew::BatchRunRepresentation::compact_nonzero_descriptors,
       }) {
    const bfnew::test::PreparedBatchedDenseFixture prepared =
        bfnew::test::prepare_batched_dense_fixture(
            fixture, singleton, 1U, representation);
    for (const bfnew::DenseHostSchedule schedule : {
             bfnew::DenseHostSchedule::csr_forward,
             bfnew::DenseHostSchedule::csr_reverse,
             bfnew::DenseHostSchedule::alternating,
         }) {
      for (const bfnew::GpuRunOptions& enabled_control : control_matrix(true)) {
        for (const bool per_lane : {true, false}) {
          bfnew::GpuRunOptions options = enabled_control;
          options.enable_per_lane_convergence = per_lane ? 1U : 0U;
          const bfnew::HostDenseRunResult standalone =
              bfnew::run_host_dense_chaotic_push(
                  fixture.device_graph,
                  singleton.front(),
                  retained.description.tile_lane_masks,
                  retained.description.csr_run_lane_masks,
                  options,
                  schedule);
          const bfnew::HostBatchedDenseRunResult batched =
              bfnew::run_host_batched_dense_chaotic_push(
                  fixture.device_graph,
                  singleton,
                  prepared.plan.batches.front(),
                  prepared.description,
                  options,
                  schedule);
          expect(
              batched.controller.rounds_completed ==
                      standalone.controller.rounds_completed &&
                  batched.result.status.reached_target_mask ==
                      standalone.result.status.reached_target_mask &&
                  batched.result.status.bounding_box_miss_mask ==
                      standalone.result.status.bounding_box_miss_mask,
              "width one and standalone dense retain identical completion");
          for (std::size_t vertex = 0U;
               vertex < fixture.partitioned.graph.vertex_count();
               ++vertex) {
            if (!query_selects_vertex(
                    fixture.partitioned.graph, singleton.front(), vertex)) {
              continue;
            }
            expect(
                batched.distance_bits[vertex] ==
                    standalone.distance_bits[vertex],
                "width one is bitwise identical to standalone dense push");
          }
          expect(
              batched.batch_work.atomic_min_attempts ==
                  standalone.result.work.atomic_attempts,
              "width one and standalone perform the same atomic attempts");
        }
      }
    }
  }
}

void test_representation_parity() {
  const bfnew::test::BatchedDenseFixture fixture =
      bfnew::test::make_mixed_duration_batched_dense_fixture();
  const bfnew::test::PreparedBatchedDenseFixture retained =
      bfnew::test::prepare_batched_dense_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  const bfnew::test::PreparedBatchedDenseFixture descriptors =
      bfnew::test::prepare_batched_dense_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::compact_nonzero_descriptors);
  expect(
      retained.plan == descriptors.plan,
      "retained and descriptor dense fixtures use the exact same batch plan");
  for (const bool per_lane : {true, false}) {
    for (const bfnew::GpuRunOptions& options : control_matrix(per_lane)) {
      const bfnew::HostBatchedDenseRunResult retained_run =
          bfnew::run_host_batched_dense_chaotic_push(
              fixture.device_graph,
              fixture.queries,
              retained.plan.batches.front(),
              retained.description,
              options,
              bfnew::DenseHostSchedule::csr_reverse);
      const bfnew::HostBatchedDenseRunResult descriptor_run =
          bfnew::run_host_batched_dense_chaotic_push(
              fixture.device_graph,
              fixture.queries,
              descriptors.plan.batches.front(),
              descriptors.description,
              options,
              bfnew::DenseHostSchedule::csr_reverse);
      expect(
          selected_outputs_equal(
              fixture,
              fixture.queries,
              retained.plan.batches.front(),
              retained_run,
              descriptor_run) &&
              retained_run.result.status.stop_reason ==
                  descriptor_run.result.status.stop_reason &&
              retained_run.result.status.rounds_completed ==
                  descriptor_run.result.status.rounds_completed &&
              retained_run.result.status.reached_target_mask ==
                  descriptor_run.result.status.reached_target_mask &&
              retained_run.result.status.bounding_box_miss_mask ==
                  descriptor_run.result.status.bounding_box_miss_mask &&
              retained_run.batch_work.lane_edge_relaxations ==
                  descriptor_run.batch_work.lane_edge_relaxations &&
              retained_run.batch_work.atomic_min_attempts ==
                  descriptor_run.batch_work.atomic_min_attempts &&
              retained_run.batch_work.csr_edge_loads ==
                  descriptor_run.batch_work.csr_edge_loads &&
              retained_run.batch_work.csr_runs_considered >=
                  descriptor_run.batch_work.csr_runs_considered,
          "retained masks and compact descriptors preserve dense semantics and logical work");
    }
  }
}

void test_instrumentation_separation() {
  const bfnew::test::BatchedDenseFixture fixture =
      bfnew::test::make_mixed_duration_batched_dense_fixture();
  const bfnew::test::PreparedBatchedDenseFixture prepared =
      bfnew::test::prepare_batched_dense_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  const bfnew::BatchPlanEntry& batch = prepared.plan.batches.front();

  std::array<bfnew::HostBatchedDenseRunResult, 3U> runs;
  for (std::size_t level = 0U; level < runs.size(); ++level) {
    bfnew::GpuRunOptions options = control_matrix(true).front();
    options.instrumentation = static_cast<bfnew::InstrumentationLevel>(level);
    runs[level] = bfnew::run_host_batched_dense_chaotic_push(
        fixture.device_graph,
        fixture.queries,
        batch,
        prepared.description,
        options,
        bfnew::DenseHostSchedule::csr_reverse);
  }
  expect(
      selected_outputs_equal(
          fixture, fixture.queries, batch, runs[0U], runs[1U]) &&
          selected_outputs_equal(
              fixture, fixture.queries, batch, runs[0U], runs[2U]) &&
          runs[0U].batch_work == runs[1U].batch_work &&
          runs[0U].batch_work == runs[2U].batch_work,
      "None, Light, and Debug preserve labels and semantic batch accounting");

  const bfnew::DeviceWorkStatistics& none = runs[0U].result.work;
  const bfnew::DeviceWorkStatistics& light = runs[1U].result.work;
  const bfnew::DeviceWorkStatistics& debug = runs[2U].result.work;
  expect(
      none.edges_examined == 0U && none.successful_decreases == 0U &&
          none.active_vertices == 0U && none.active_lane_rounds == 0U &&
          none.atomic_attempts == 0U &&
          none.successful_atomic_updates == 0U &&
          none.mask_operations == 0U && none.changed_flag_updates == 0U &&
          none.full_edge_rounds == 0U,
      "None suppresses every algorithmic common device counter");
  expect(
      light.edges_examined != 0U && light.successful_decreases != 0U &&
          light.active_vertices != 0U && light.active_lane_rounds != 0U &&
          light.full_edge_rounds != 0U && light.atomic_attempts == 0U &&
          light.successful_atomic_updates == 0U &&
          light.mask_operations == 0U && light.changed_flag_updates == 0U,
      "Light retains aggregate work while suppressing Debug atomics and masks");
  expect(
      debug.edges_examined == runs[2U].batch_work.lane_edge_relaxations &&
          debug.atomic_attempts == runs[2U].batch_work.atomic_min_attempts &&
          debug.atomic_attempts != 0U &&
          debug.successful_atomic_updates != 0U &&
          debug.mask_operations != 0U && debug.changed_flag_updates != 0U,
      "Debug exposes atomic, mask, and changed-publication diagnostics");
}

void test_maximum_rounds_and_guards() {
  const bfnew::test::BatchedDenseFixture fixture =
      bfnew::test::make_mixed_duration_batched_dense_fixture();
  std::vector<bfnew::RouteQuery> singleton{fixture.queries[2U]};
  bfnew::test::PreparedBatchedDenseFixture prepared =
      bfnew::test::prepare_batched_dense_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  for (const bool per_lane : {true, false}) {
    for (const bfnew::GpuRunOptions& base : control_matrix(per_lane, 2U)) {
      const bfnew::HostBatchedDenseRunResult exhausted =
          bfnew::run_host_batched_dense_chaotic_push(
              fixture.device_graph,
              singleton,
              prepared.plan.batches.front(),
              prepared.description,
              base,
              bfnew::DenseHostSchedule::csr_reverse);
      expect(
          exhausted.result.status.stop_reason ==
                  static_cast<std::uint32_t>(
                      bfnew::DeviceStopReason::maximum_rounds) &&
              exhausted.result.status.rounds_completed == 2U &&
              exhausted.result.status.converged == 0U &&
              exhausted.result.status.reached_target_mask == 0U &&
              exhausted.result.status.bounding_box_miss_mask == 0U &&
              exhausted.convergence_round_by_lane[0U] == 0U,
          "all dense controls/toggles keep round exhaustion distinct from miss");
    }
  }

  const std::array<bfnew::RouteQuery, 2U> mixed_limit_queries{
      fixture.queries[0U], fixture.queries[2U]};
  const bfnew::test::PreparedBatchedDenseFixture mixed_limit =
      bfnew::test::prepare_batched_dense_fixture(
          fixture,
          mixed_limit_queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  const bfnew::BatchPlanEntry& mixed_batch = mixed_limit.plan.batches.front();
  const std::size_t immediate_lane = bfnew::test::dense_lane_for_query(
      mixed_batch, bfnew::QueryId{1400U});
  const std::size_t unfinished_lane = bfnew::test::dense_lane_for_query(
      mixed_batch, bfnew::QueryId{1402U});
  const bfnew::LaneMask immediate_bit = bfnew::LaneMask{1U} << immediate_lane;
  const bfnew::LaneMask unfinished_bit = bfnew::LaneMask{1U} << unfinished_lane;
  for (const bool per_lane : {true, false}) {
    for (const bfnew::GpuRunOptions& base : control_matrix(per_lane, 2U)) {
      const bfnew::HostBatchedDenseRunResult exhausted =
          bfnew::run_host_batched_dense_chaotic_push(
              fixture.device_graph,
              mixed_limit_queries,
              mixed_batch,
              mixed_limit.description,
              base,
              bfnew::DenseHostSchedule::csr_reverse);
      expect(
          exhausted.result.status.stop_reason ==
                  static_cast<std::uint32_t>(
                      bfnew::DeviceStopReason::maximum_rounds) &&
              exhausted.result.status.reached_target_mask == 0U &&
              exhausted.result.status.bounding_box_miss_mask == 0U &&
              exhausted.convergence_round_by_lane[immediate_lane] == 1U &&
              exhausted.convergence_round_by_lane[unfinished_lane] == 0U,
          "mixed maximum-round exit retains one proven and one unfinished lane");
      if (per_lane) {
        expect(
            exhausted.rounds_executed_by_lane[immediate_lane] == 1U &&
                exhausted.rounds_executed_by_lane[unfinished_lane] == 2U &&
                exhausted.controller.converged_lane_mask == immediate_bit &&
                exhausted.controller.active_lane_mask == unfinished_bit,
            "enabled convergence freezes the early lane before round limit");
      } else {
        expect(
            exhausted.rounds_executed_by_lane[immediate_lane] == 2U &&
                exhausted.rounds_executed_by_lane[unfinished_lane] == 2U &&
                exhausted.controller.converged_lane_mask == 0U &&
                exhausted.controller.active_lane_mask ==
                    mixed_batch.valid_lane_mask,
            "disabled convergence retains both lanes through round limit");
      }
    }
  }

  const bfnew::GpuRunOptions limited = control_matrix(true, 2U).front();
  bfnew::GpuRunOptions wrong_engine = limited;
  wrong_engine.engine = bfnew::EngineKind::jacobi_pull;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_dense_chaotic_push(
            fixture.device_graph,
            singleton,
            prepared.plan.batches.front(),
            prepared.description,
            wrong_engine));
      },
      "batched dense rejects another engine identity");

  bfnew::BatchPlanEntry corrupt_index = prepared.plan.batches.front();
  corrupt_index.query_indices_by_lane[0U] =
      static_cast<std::uint32_t>(singleton.size());
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_dense_chaotic_push(
            fixture.device_graph,
            singleton,
            corrupt_index,
            prepared.description,
            limited));
      },
      "batched dense rejects an out-of-range plan query index");

  bfnew::BatchDeviceDescription corrupt_mask = prepared.description;
  const std::uint32_t touched = corrupt_mask.touched_csr_runs.front();
  corrupt_mask.csr_run_lane_masks[touched] = 0U;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_dense_chaotic_push(
            fixture.device_graph,
            singleton,
            prepared.plan.batches.front(),
            corrupt_mask,
            limited));
      },
      "batched dense rejects a CSR mask not implied by endpoint tiles");

  bfnew::test::PreparedBatchedDenseFixture compact =
      bfnew::test::prepare_batched_dense_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::compact_nonzero_descriptors);
  compact.description.csr_run_descriptors.front().lane_mask = 0U;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_dense_chaotic_push(
            fixture.device_graph,
            singleton,
            compact.plan.batches.front(),
            compact.description,
            limited));
      },
      "batched dense rejects a noncanonical compact CSR descriptor");

  std::vector<bfnew::RouteQuery> duplicate_queries{
      singleton.front(), singleton.front()};
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_dense_chaotic_push(
            fixture.device_graph,
            duplicate_queries,
            prepared.plan.batches.front(),
            prepared.description,
            limited));
      },
      "batched dense rejects duplicate query identities");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_dense_chaotic_push(
            fixture.device_graph,
            singleton,
            prepared.plan.batches.front(),
            prepared.description,
            limited,
            static_cast<bfnew::DenseHostSchedule>(255U)));
      },
      "batched dense rejects an unknown portable schedule");
}

#if (defined(BFNEW_PHASE15_HIP_SOURCE_PATH) && \
     !defined(BFNEW_PHASE15_WORKSPACE_HIP_SOURCE_PATH)) || \
    (!defined(BFNEW_PHASE15_HIP_SOURCE_PATH) && \
     defined(BFNEW_PHASE15_WORKSPACE_HIP_SOURCE_PATH))
#error "Phase 15 structural source paths must be defined together"
#endif

#if defined(BFNEW_PHASE15_HIP_SOURCE_PATH) && \
    defined(BFNEW_PHASE15_WORKSPACE_HIP_SOURCE_PATH)
[[nodiscard]] std::string read_source_file(const char* const path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error{"cannot open Phase 15 structural source"};
  }
  return std::string{
      std::istreambuf_iterator<char>{input},
      std::istreambuf_iterator<char>{}};
}
#endif

void test_future_hip_source_structure() {
#if defined(BFNEW_PHASE15_HIP_SOURCE_PATH) && \
    defined(BFNEW_PHASE15_WORKSPACE_HIP_SOURCE_PATH)
  const std::string kernel = read_source_file(BFNEW_PHASE15_HIP_SOURCE_PATH);
  const std::string workspace =
      read_source_file(BFNEW_PHASE15_WORKSPACE_HIP_SOURCE_PATH);
  const std::size_t round_begin =
      kernel.find("perform_batched_dense_round");
  const std::size_t round_end =
      kernel.find("advance_batched_dense_controller", round_begin);
  expect(
      round_begin != std::string::npos && round_end != std::string::npos &&
          round_begin < round_end,
      "future HIP source exposes one complete batched dense scan");
  const std::string round =
      round_begin != std::string::npos && round_end != std::string::npos &&
              round_begin < round_end
          ? kernel.substr(round_begin, round_end - round_begin)
          : std::string{};
  const std::size_t completed_retire_begin = workspace.find(
      "void ReusableBatchedDenseWorkspace::retire_after_stream_completion");
  const std::size_t completed_retire_end =
      workspace.find("void ReusableBatchedDenseWorkspace::recover_noexcept",
                     completed_retire_begin);
  const std::string completed_retire =
      completed_retire_begin != std::string::npos &&
              completed_retire_end != std::string::npos &&
              completed_retire_begin < completed_retire_end
          ? workspace.substr(completed_retire_begin,
                             completed_retire_end - completed_retire_begin)
          : std::string{};
  expect(
      round.find("graph.csr.row_run_offsets") != std::string::npos &&
          round.find("graph.csr.run_edge_offsets") != std::string::npos &&
          round.find("graph.csr.destinations") != std::string::npos &&
          round.find("graph.csr.weights") != std::string::npos,
      "future HIP dense scan structurally traverses outgoing CSR tile runs");
  expect(
      round.find("prepared_mask & execute_lane_mask") != std::string::npos &&
          round.find("active_run_mask == 0U") != std::string::npos,
      "future HIP dense scan intersects admission once and skips zero runs");
  expect(
      round.find("atomic_load_nonnegative_float_bits") != std::string::npos &&
          round.find("atomic_min_nonnegative_float_bits") !=
              std::string::npos,
      "future HIP dense scan preserves atomic-compatible loads and updates");
  expect(
      round.find(
          "static_cast<std::uint64_t>(source) * batch.lane_width") !=
              std::string::npos &&
          round.find(
              "static_cast<std::uint64_t>(destination) * batch.lane_width") !=
              std::string::npos,
      "future HIP dense scan keeps vertex-major independent lane words");
  expect(
      round.find("distances[") == std::string::npos &&
          round.find("workspace.targets") == std::string::npos &&
          round.find("target_offsets") == std::string::npos &&
          round.find("graph.csc") == std::string::npos &&
          round.find("frontier") == std::string::npos,
      "future HIP dense relaxation has no ordinary distance load, target stop, CSC, or frontier path");
  expect(
      kernel.find("cg::this_grid") != std::string::npos &&
          kernel.find("grid.sync()") != std::string::npos &&
          kernel.find("hipLaunchCooperativeKernel") != std::string::npos &&
          kernel.find("hipOccupancyMaxActiveBlocksPerMultiprocessor") !=
              std::string::npos,
      "future HIP persistent control has barriers and actual-kernel occupancy");
  expect(
      kernel.find("completed_batched_dense_status") != std::string::npos &&
          kernel.find("target_offsets") != std::string::npos &&
          kernel.find("make_dense_run_status") != std::string::npos,
      "future HIP finalization computes per-lane reached and miss masks");
  expect(
      workspace.find("ReusableBatchedDenseWorkspace") != std::string::npos &&
          workspace.find("DeviceBuffer engine_scratch") != std::string::npos &&
          workspace.find("selected_ranges") != std::string::npos &&
          workspace.find("lane_convergence_rounds") != std::string::npos,
      "future HIP workspace owns batch metadata and convergence state");
  expect(
      workspace.find("engine_scratch.clear_async") == std::string::npos,
      "future HIP workspace preserves selected-only reset instead of clearing full scratch");
  expect(
      kernel.find("retire_after_stream_completion") != std::string::npos &&
          completed_retire.find("clear_lease()") != std::string::npos &&
          completed_retire.find("stream.synchronize()") == std::string::npos,
      "compact dense retirement has no redundant stream fence");
#endif
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
      "Phase 15 supports exactly widths 1, 8, 16, and 32");
  test_mixed_duration_matrix();
  test_representation_parity();
  test_width_one_matches_standalone();
  test_instrumentation_separation();
  test_maximum_rounds_and_guards();
  test_future_hip_source_structure();
  if (failures != 0) {
    std::cerr << failures << " batched dense assertion(s) failed\n";
    return 1;
  }
  std::cout << "batched dense chaotic push CPU tests passed\n";
  return 0;
}
