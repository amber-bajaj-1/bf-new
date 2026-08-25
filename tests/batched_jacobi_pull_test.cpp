#include "bfnew/batched_jacobi_pull.hpp"
#include "bfnew/jacobi_pull.hpp"
#include "bfnew/sssp.hpp"
#include "batched_jacobi_fixture_suite.hpp"

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

[[nodiscard]] bool bitwise_equal(const float left, const float right) noexcept {
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
  const bfnew::TileId owner = graph.owner_tiles()[vertex];
  return std::binary_search(
      query.selected_tiles.begin(), query.selected_tiles.end(), owner);
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
  throw std::logic_error{"batched Jacobi test lane query is missing"};
}

[[nodiscard]] std::vector<bfnew::GpuRunOptions> control_matrix(
    const bool per_lane_convergence,
    const std::uint64_t maximum_rounds = 64U) {
  bfnew::GpuRunOptions base;
  base.engine = bfnew::EngineKind::jacobi_pull;
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
  for (const std::uint32_t rounds_per_chunk : {2U, 4U, 8U, 16U, 32U}) {
    bfnew::GpuRunOptions chunked = base;
    chunked.control_mode = bfnew::ControlMode::chunked_host_poll;
    chunked.rounds_per_chunk = rounds_per_chunk;
    controls.push_back(chunked);
  }
  return controls;
}

void expect_control_accounting(
    const bfnew::GpuRunOptions& options,
    const bfnew::HostBatchedJacobiRunResult& run,
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
            work.host_synchronizations == rounds &&
            work.host_checks == rounds &&
            work.kernel_dispatches == 1U + 2U * rounds,
        prefix + ": per-round control counts exact round/advance pairs");
    return;
  }
  const std::uint64_t chunks =
      (rounds + options.rounds_per_chunk - 1U) /
      options.rounds_per_chunk;
  const std::uint64_t pairs = chunks * options.rounds_per_chunk;
  expect(
      run.completed_host_chunks == chunks &&
          run.queued_round_pairs == pairs &&
          work.controller_copies == chunks &&
          work.host_synchronizations == chunks &&
          work.host_checks == chunks &&
          work.kernel_dispatches == 1U + 2U * pairs,
      prefix + ": chunked control retains queued terminal no-op pairs");
}

void expect_lane_oracles_and_invariants(
    const bfnew::test::BatchedJacobiFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchPlanEntry& batch,
    const bfnew::BatchDeviceDescription& description,
    const bfnew::GpuRunOptions& options,
    const bfnew::HostBatchedJacobiRunResult& run,
    const std::string& prefix) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::size_t expected_elements =
      static_cast<std::size_t>(graph.vertex_count()) * batch.lane_width;
  expect(
      run.final_distances.size() == expected_elements &&
          run.distance_slots[0U].size() == expected_elements &&
          run.distance_slots[1U].size() == expected_elements,
      prefix + ": both distance slots use exact V-by-W storage");
  expect(
      run.rounds_executed_by_lane.size() == batch.lane_width &&
          run.convergence_round_by_lane.size() == batch.lane_width &&
          run.tail_rounds_by_lane.size() == batch.lane_width,
      prefix + ": per-lane convergence evidence has width entries");
  expect(
      run.result.engine_kind ==
              static_cast<std::uint32_t>(bfnew::EngineKind::jacobi_pull) &&
          run.result.control_mode ==
              static_cast<std::uint32_t>(options.control_mode),
      prefix + ": result retains Jacobi and control identities");
  expect(
      bfnew::validate_device_controller(run.controller) ==
              bfnew::DeviceControllerError::none &&
          bfnew::validate_device_run_status(run.result.status) ==
              bfnew::DeviceRunStatusError::none,
      prefix + ": terminal controller and status validate");
  expect(
      run.result.status.stop_reason ==
              static_cast<std::uint32_t>(bfnew::DeviceStopReason::converged) &&
          run.result.status.valid_lane_mask == batch.valid_lane_mask &&
          run.result.status.converged_lane_mask == batch.valid_lane_mask &&
          run.converged_slots_bitwise_identical_mask == batch.valid_lane_mask,
      prefix + ": every valid lane converges with equal selected columns");
  expect(
      run.result.status.final_distance_slot ==
              run.controller.distance_read_slot &&
          run.controller.distance_read_slot ==
              static_cast<std::uint32_t>(
                  run.controller.rounds_completed & 1U),
      prefix + ": final distance slot follows completed-round parity");

  bfnew::LaneMask expected_reached = 0U;
  bfnew::LaneMask expected_miss = 0U;
  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    const bfnew::LaneMask bit = bfnew::LaneMask{1U} << lane;
    if ((batch.valid_lane_mask & bit) == 0U) {
      expect(
          run.rounds_executed_by_lane[lane] == 0U &&
              run.convergence_round_by_lane[lane] == 0U &&
              run.tail_rounds_by_lane[lane] == 0U,
          prefix + ": padded lanes have no semantic round state");
      continue;
    }
    const bfnew::RouteQuery& query = query_for_lane(queries, batch, lane);
    const std::vector<float> oracle = bounded_oracle(graph, query);
    for (std::size_t vertex = 0U; vertex < graph.vertex_count(); ++vertex) {
      if (!query_selects_vertex(graph, query, vertex)) {
        continue;
      }
      const std::size_t index = static_cast<std::size_t>(
          bfnew::batched_jacobi_distance_index(
              static_cast<std::uint32_t>(vertex),
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      expect(
          bitwise_equal(run.final_distances[index], oracle[vertex]),
          prefix + ": every selected lane label matches bounded Dijkstra");
      expect(
          bitwise_equal(
              run.distance_slots[0U][index], run.distance_slots[1U][index]),
          prefix + ": both converged slots are bitwise equal on selected tiles");
      if (lane + 1U < batch.lane_width) {
        const std::uint64_t next_index =
            bfnew::batched_jacobi_distance_index(
                static_cast<std::uint32_t>(vertex),
                static_cast<std::uint32_t>(lane + 1U),
                batch.lane_width);
        expect(
            next_index == index + 1U,
            prefix + ": query lanes are contiguous within a vertex");
      }
    }
    for (const bfnew::VertexId source : query.sources) {
      const std::size_t index = static_cast<std::size_t>(
          bfnew::batched_jacobi_distance_index(
              source.value(),
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      expect(
          bitwise_equal(run.final_distances[index], 0.0F),
          prefix + ": each lane retains its own zero-valued sources");
    }
    bool targets_reached = true;
    for (const bfnew::VertexId target : query.targets) {
      const std::size_t index = static_cast<std::size_t>(
          bfnew::batched_jacobi_distance_index(
              target.value(),
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      targets_reached =
          targets_reached && std::isfinite(run.final_distances[index]);
    }
    if (targets_reached) {
      expected_reached |= bit;
    } else {
      expected_miss |= bit;
    }
  }
  expect(
      run.result.status.reached_target_mask == expected_reached &&
          run.result.status.bounding_box_miss_mask == expected_miss,
      prefix + ": reached and bounded-miss masks are lane exact");

  const bfnew::BatchedJacobiWorkStatistics& batch_work = run.batch_work;
  expect(
      batch_work.csc_runs_considered ==
              batch_work.csc_runs_visited + batch_work.csc_runs_skipped &&
          batch_work.active_lanes_over_visited_runs >=
              batch_work.csc_runs_visited,
      prefix + ": CSC run visit/skip and active-lane counts balance");
  expect(
      batch_work.csc_edge_loads <= batch_work.lane_edge_relaxations &&
          run.result.work.edges_examined ==
              batch_work.lane_edge_relaxations &&
          run.result.work.active_vertices ==
              batch_work.destination_lane_writes &&
          run.result.work.successful_decreases ==
              batch_work.successful_decreases &&
          run.result.work.active_lane_rounds ==
              batch_work.active_lane_rounds &&
          run.result.work.mask_operations ==
              batch_work.csc_runs_considered,
      prefix + ": common and batched work counters use exact vocabulary");
  expect(
      batch_work.valid_lane_round_capacity ==
              batch_work.active_lane_rounds +
                  batch_work.inactive_valid_lane_rounds &&
          batch_work.wave_lane_round_capacity ==
              batch_work.valid_lane_round_capacity +
                  batch_work.padded_lane_round_capacity &&
          batch_work.padded_lane_semantic_work == 0U,
      prefix + ": lane utilization accounts for inactive and padded capacity");
  expect(
      batch_work.distance_reset_bytes ==
              batch.selected_lane_vertex_count * 2U * sizeof(float) &&
          batch_work.source_seed_write_bytes ==
              description.sources.size() * 2U * sizeof(float),
      prefix + ": reset and source-seed traffic use admitted lanes only");
  expect(
      batch_work.union_tile_lane_positions ==
              batch.union_tiles.size() *
                  static_cast<std::uint64_t>(
                      std::popcount(batch.valid_lane_mask)) &&
          batch_work.selected_tile_lane_positions != 0U &&
          batch_work.union_tile_lane_positions >=
              batch_work.selected_tile_lane_positions,
      prefix + ": union-tile inflation has exact integer terms");
  expect_control_accounting(options, run, prefix);
}

[[nodiscard]] bool selected_outputs_equal(
    const bfnew::test::BatchedJacobiFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchPlanEntry& batch,
    const bfnew::HostBatchedJacobiRunResult& left,
    const bfnew::HostBatchedJacobiRunResult& right) {
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
          bfnew::batched_jacobi_distance_index(
              static_cast<std::uint32_t>(vertex),
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      if (!bitwise_equal(
              left.final_distances[index], right.final_distances[index])) {
        return false;
      }
    }
  }
  return true;
}

void test_mixed_duration_matrix() {
  const bfnew::test::BatchedJacobiFixture fixture =
      bfnew::test::make_mixed_duration_batched_jacobi_fixture();
  for (const std::uint32_t width : {8U, 16U, 32U}) {
    const bfnew::test::PreparedBatchedJacobiFixture prepared =
        bfnew::test::prepare_batched_fixture(
            fixture,
            fixture.queries,
            width,
            bfnew::BatchRunRepresentation::retained_per_run_masks);
    const bfnew::BatchPlanEntry& batch = prepared.plan.batches.front();
    expect(
        std::popcount(batch.valid_lane_mask) == 5 &&
            batch.lane_width == width,
        "mixed-duration batch retains five valid lanes and exact width");

    const std::vector<bfnew::GpuRunOptions> enabled_controls =
        control_matrix(true);
    const std::vector<bfnew::GpuRunOptions> disabled_controls =
        control_matrix(false);
    for (std::size_t control = 0U; control < enabled_controls.size(); ++control) {
      const bfnew::GpuRunOptions enabled_options = enabled_controls[control];
      const bfnew::GpuRunOptions disabled_options = disabled_controls[control];
      const bfnew::HostBatchedJacobiRunResult enabled =
          bfnew::run_host_batched_jacobi_pull(
              fixture.device_graph,
              fixture.queries,
              batch,
              prepared.description,
              enabled_options);
      const bfnew::HostBatchedJacobiRunResult disabled =
          bfnew::run_host_batched_jacobi_pull(
              fixture.device_graph,
              fixture.queries,
              batch,
              prepared.description,
              disabled_options);
      const std::string prefix =
          "mixed-width-" + std::to_string(width) + "-control-" +
          std::to_string(control);
      expect_lane_oracles_and_invariants(
          fixture,
          fixture.queries,
          batch,
          prepared.description,
          enabled_options,
          enabled,
          prefix + "-enabled");
      expect_lane_oracles_and_invariants(
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
          prefix + ": per-lane convergence toggle preserves every label bit");
      expect(
          enabled.controller.rounds_completed ==
                  disabled.controller.rounds_completed &&
              enabled.batch_work.tail_lane_rounds_avoided ==
                  enabled.batch_work.tail_lane_rounds &&
              enabled.batch_work.tail_lane_rounds_executed == 0U &&
              enabled.batch_work.inactive_valid_lane_rounds ==
                  enabled.batch_work.tail_lane_rounds &&
              disabled.batch_work.tail_lane_rounds_avoided == 0U &&
              disabled.batch_work.tail_lane_rounds_executed ==
                  disabled.batch_work.tail_lane_rounds &&
              disabled.batch_work.inactive_valid_lane_rounds == 0U,
          prefix + ": enabled convergence avoids exactly the early-lane tail");
      expect(
          disabled.batch_work.lane_edge_relaxations -
                  enabled.batch_work.lane_edge_relaxations ==
              enabled.batch_work
                  .lane_edge_relaxations_avoided_by_early_convergence,
          prefix + ": avoided logical lane-edge work is exact");

      const std::array<std::pair<bfnew::QueryId, std::uint64_t>, 5U>
          expected_convergence{{
              {bfnew::QueryId{1400U}, 1U},
              {bfnew::QueryId{1401U}, 2U},
              {bfnew::QueryId{1402U}, 6U},
              {bfnew::QueryId{1403U}, 1U},
              {bfnew::QueryId{1404U}, 4U},
          }};
      for (const auto& [query_id, expected_round] : expected_convergence) {
        const std::size_t lane =
            bfnew::test::lane_for_query(batch, query_id);
        expect(
            enabled.convergence_round_by_lane[lane] == expected_round &&
                disabled.convergence_round_by_lane[lane] == expected_round &&
                enabled.rounds_executed_by_lane[lane] == expected_round &&
                disabled.rounds_executed_by_lane[lane] ==
                    disabled.controller.rounds_completed,
            prefix + ": each mixed-duration lane has its exact convergence round");
      }
      const std::size_t miss_lane =
          bfnew::test::lane_for_query(batch, bfnew::QueryId{1403U});
      expect(
          (enabled.result.status.bounding_box_miss_mask &
           (bfnew::LaneMask{1U} << miss_lane)) != 0U,
          prefix + ": unreachable admitted target becomes one compact miss bit");
    }
  }
}

void test_descriptor_parity() {
  const bfnew::test::BatchedJacobiFixture fixture =
      bfnew::test::make_mixed_duration_batched_jacobi_fixture();
  const bfnew::test::PreparedBatchedJacobiFixture retained =
      bfnew::test::prepare_batched_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  const bfnew::test::PreparedBatchedJacobiFixture descriptors =
      bfnew::test::prepare_batched_fixture(
          fixture,
          fixture.queries,
          8U,
          bfnew::BatchRunRepresentation::compact_nonzero_descriptors);
  expect(
      retained.plan == descriptors.plan,
      "retained and descriptor fixtures share the exact plan");
  for (const bool per_lane : {false, true}) {
    for (const bfnew::GpuRunOptions& options : control_matrix(per_lane)) {
      const bfnew::HostBatchedJacobiRunResult retained_run =
          bfnew::run_host_batched_jacobi_pull(
              fixture.device_graph,
              fixture.queries,
              retained.plan.batches.front(),
              retained.description,
              options);
      const bfnew::HostBatchedJacobiRunResult descriptor_run =
          bfnew::run_host_batched_jacobi_pull(
              fixture.device_graph,
              fixture.queries,
              descriptors.plan.batches.front(),
              descriptors.description,
              options);
      expect(
          selected_outputs_equal(
              fixture,
              fixture.queries,
              retained.plan.batches.front(),
              retained_run,
              descriptor_run) &&
              retained_run.result.status.reached_target_mask ==
                  descriptor_run.result.status.reached_target_mask &&
              retained_run.result.status.bounding_box_miss_mask ==
                  descriptor_run.result.status.bounding_box_miss_mask,
          "retained masks and compact descriptors have identical semantics");
      expect(
          retained_run.batch_work.csc_runs_considered >=
                  descriptor_run.batch_work.csc_runs_considered &&
              retained_run.batch_work.lane_edge_relaxations ==
                  descriptor_run.batch_work.lane_edge_relaxations,
          "compact descriptors omit only prepared zero-mask runs");
    }
  }
}

void test_width_one_matches_standalone() {
  const bfnew::test::BatchedJacobiFixture fixture =
      bfnew::test::make_mixed_duration_batched_jacobi_fixture();
  std::vector<bfnew::RouteQuery> singleton{
      fixture.queries[2U],
  };
  const bfnew::test::PreparedBatchedJacobiFixture prepared =
      bfnew::test::prepare_batched_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  const bfnew::BatchPlanEntry& batch = prepared.plan.batches.front();
  expect(
      batch.lane_width == 1U && batch.valid_lane_mask == 1U,
      "explicit width-one plan has one valid, unpadded lane");
  for (const bool per_lane : {false, true}) {
    for (const bfnew::GpuRunOptions& options : control_matrix(per_lane)) {
      const bfnew::HostBatchedJacobiRunResult batched =
          bfnew::run_host_batched_jacobi_pull(
              fixture.device_graph,
              singleton,
              batch,
              prepared.description,
              options);
      const bfnew::HostJacobiRunResult standalone = bfnew::run_host_jacobi_pull(
          fixture.device_graph,
          singleton.front(),
          prepared.description.tile_lane_masks,
          prepared.description.csc_run_lane_masks,
          options);
      bool same =
          batched.controller.rounds_completed ==
              standalone.controller.rounds_completed &&
          batched.result.status.reached_target_mask ==
              standalone.result.status.reached_target_mask &&
          batched.result.status.bounding_box_miss_mask ==
              standalone.result.status.bounding_box_miss_mask;
      for (std::size_t vertex = 0U;
           same && vertex < fixture.partitioned.graph.vertex_count();
           ++vertex) {
        if (query_selects_vertex(
                fixture.partitioned.graph, singleton.front(), vertex)) {
          same = bitwise_equal(
              batched.final_distances[vertex],
              standalone.distances[vertex]);
        }
      }
      expect(same, "width one is bitwise identical to standalone Jacobi");
    }
  }
}

void test_maximum_rounds_and_input_guards() {
  const bfnew::test::BatchedJacobiFixture fixture =
      bfnew::test::make_mixed_duration_batched_jacobi_fixture();
  std::vector<bfnew::RouteQuery> singleton{fixture.queries[2U]};
  bfnew::test::PreparedBatchedJacobiFixture prepared =
      bfnew::test::prepare_batched_fixture(
          fixture,
          singleton,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  const std::vector<bfnew::GpuRunOptions> limited_controls =
      control_matrix(true, 2U);
  for (const bfnew::GpuRunOptions& limited : limited_controls) {
    const bfnew::HostBatchedJacobiRunResult exhausted =
        bfnew::run_host_batched_jacobi_pull(
            fixture.device_graph,
            singleton,
            prepared.plan.batches.front(),
            prepared.description,
            limited);
    expect(
        exhausted.result.status.stop_reason ==
                static_cast<std::uint32_t>(
                    bfnew::DeviceStopReason::maximum_rounds) &&
            exhausted.result.status.rounds_completed == 2U &&
            exhausted.result.status.converged == 0U &&
            exhausted.result.status.reached_target_mask == 0U &&
            exhausted.result.status.bounding_box_miss_mask == 0U,
        "maximum-round exhaustion is not reported as convergence or a box miss");
  }

  const bfnew::GpuRunOptions limited = limited_controls.front();

  bfnew::GpuRunOptions wrong_engine = limited;
  wrong_engine.engine = bfnew::EngineKind::dense_chaotic_push;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_jacobi_pull(
            fixture.device_graph,
            singleton,
            prepared.plan.batches.front(),
            prepared.description,
            wrong_engine));
      },
      "batched Jacobi rejects another engine identity");

  bfnew::BatchPlanEntry corrupt_index = prepared.plan.batches.front();
  corrupt_index.query_indices_by_lane[0U] =
      static_cast<std::uint32_t>(singleton.size());
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_jacobi_pull(
            fixture.device_graph,
            singleton,
            corrupt_index,
            prepared.description,
            limited));
      },
      "batched Jacobi rejects an out-of-range plan query index");

  const std::uint32_t touched =
      prepared.description.touched_csc_runs.front();
  prepared.description.csc_run_lane_masks[touched] = 0U;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_jacobi_pull(
            fixture.device_graph,
            singleton,
            prepared.plan.batches.front(),
            prepared.description,
            limited));
      },
      "batched Jacobi rejects a CSC run mask not implied by endpoint tiles");
}

#if (defined(BFNEW_PHASE14_HIP_SOURCE_PATH) && \
     !defined(BFNEW_PHASE14_WORKSPACE_HIP_SOURCE_PATH)) || \
    (!defined(BFNEW_PHASE14_HIP_SOURCE_PATH) && \
     defined(BFNEW_PHASE14_WORKSPACE_HIP_SOURCE_PATH))
#error "Phase 14 structural source paths must be defined together"
#endif

#if defined(BFNEW_PHASE14_HIP_SOURCE_PATH) && \
    defined(BFNEW_PHASE14_WORKSPACE_HIP_SOURCE_PATH)
[[nodiscard]] std::string read_source_file(const char* const path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error{"cannot open Phase 14 structural source"};
  }
  return std::string{
      std::istreambuf_iterator<char>{input},
      std::istreambuf_iterator<char>{}};
}
#endif

void test_hip_source_structure() {
#if defined(BFNEW_PHASE14_HIP_SOURCE_PATH) && \
    defined(BFNEW_PHASE14_WORKSPACE_HIP_SOURCE_PATH)
  const std::string kernel = read_source_file(BFNEW_PHASE14_HIP_SOURCE_PATH);
  const std::string workspace =
      read_source_file(BFNEW_PHASE14_WORKSPACE_HIP_SOURCE_PATH);
  const std::size_t round_begin =
      kernel.find("perform_batched_jacobi_round");
  const std::size_t round_end =
      kernel.find("advance_batched_jacobi_controller", round_begin);
  expect(
      round_begin != std::string::npos && round_end != std::string::npos &&
          round_begin < round_end,
      "HIP structure exposes a separate complete Jacobi round");
  const std::string round =
      round_begin != std::string::npos && round_end != std::string::npos &&
              round_begin < round_end
          ? kernel.substr(round_begin, round_end - round_begin)
          : std::string{};
  const std::size_t initialize_begin =
      kernel.find("initialize_batched_jacobi_state");
  const std::size_t initialize_end =
      kernel.find("perform_batched_jacobi_round", initialize_begin);
  const std::string initialize =
      initialize_begin != std::string::npos &&
              initialize_end != std::string::npos &&
              initialize_begin < initialize_end
          ? kernel.substr(initialize_begin, initialize_end - initialize_begin)
          : std::string{};
  const std::size_t completed_retire_begin = workspace.find(
      "void ReusableBatchedJacobiWorkspace::retire_after_stream_completion");
  const std::size_t completed_retire_end =
      workspace.find("void ReusableBatchedJacobiWorkspace::recover_noexcept",
                     completed_retire_begin);
  const std::string completed_retire =
      completed_retire_begin != std::string::npos &&
              completed_retire_end != std::string::npos &&
              completed_retire_begin < completed_retire_end
          ? workspace.substr(completed_retire_begin,
                             completed_retire_end - completed_retire_begin)
          : std::string{};
  expect(
      round.find("graph.csc.column_run_offsets") != std::string::npos &&
          round.find("graph.csc.run_edge_offsets") != std::string::npos &&
          round.find("graph.csc.sources") != std::string::npos &&
          round.find("graph.csc.weights") != std::string::npos,
      "HIP round structurally traverses incoming CSC source-tile runs");
  expect(
      round.find("prepared_mask & destination_execute_mask") !=
              std::string::npos &&
          round.find("active_run_mask == 0U") != std::string::npos,
      "HIP round intersects lane admission once per run and skips zero masks");
  expect(
      round.find("* batch.lane_width + wave_lane") != std::string::npos,
      "HIP round structurally indexes vertex-major contiguous query lanes");
  expect(
      kernel.find("cg::this_grid") != std::string::npos &&
          kernel.find("grid.sync()") != std::string::npos &&
          kernel.find("hipLaunchCooperativeKernel") != std::string::npos,
      "HIP persistent control contains real cooperative grid barriers");
  expect(
      kernel.find("hipOccupancyMaxActiveBlocksPerMultiprocessor") !=
          std::string::npos,
      "HIP launch sizing queries occupancy on the actual batched kernel");
  expect(
      kernel.find("finalize_batched_jacobi_status") != std::string::npos &&
          kernel.find("completed_batched_jacobi_status") !=
              std::string::npos &&
          kernel.find("target_offsets") != std::string::npos &&
          kernel.find("make_jacobi_run_status") != std::string::npos,
      "HIP computes reached and miss masks in a device finalization step");
  expect(
      kernel.find("ControlMode::persistent_cooperative") !=
              std::string::npos &&
          kernel.find("ControlMode::per_round_host_poll") !=
              std::string::npos &&
          kernel.find("launch_round_pair") != std::string::npos,
      "HIP source retains persistent, per-round, and chunked round-pair control");
  expect(
      workspace.find("ReusableBatchedJacobiWorkspace") != std::string::npos &&
          workspace.find("DeviceBuffer engine_scratch") != std::string::npos &&
          workspace.find("selected_ranges") != std::string::npos &&
          workspace.find("csc_run_lane_masks") != std::string::npos,
      "HIP uses a dedicated reusable batch workspace with selected-range state");
  expect(
      workspace.find("engine_scratch.clear_async") == std::string::npos &&
          workspace.find("hipMemset") == std::string::npos &&
          workspace.find("batch.csc_run_lane_masks.data()") ==
              std::string::npos,
      "HIP batch preparation clears neither distance scratch nor a graph-wide "
      "run image and does not upload host CSC masks");
  expect(
      workspace.find("DeviceBuffer selected_tiles") == std::string::npos &&
          workspace.find("controller.copy_from_host_async") ==
              std::string::npos &&
          workspace.find("status.clear_async") == std::string::npos,
      "Jacobi preparation does not upload unused union tiles or an overwritten "
      "controller and does not clear a fully overwritten status record");
  expect(
      initialize.find("graph.csc.run_source_tiles[run]") !=
              std::string::npos &&
          initialize.find("workspace.run_lane_masks[run]") !=
              std::string::npos &&
          initialize.find("run += phase14_wave_width") !=
              std::string::npos,
      "Jacobi initialization wave-cooperatively materializes selected CSC "
      "run masks on the device");
  expect(
      kernel.find("retire_after_stream_completion") != std::string::npos &&
          completed_retire.find("clear_lease()") != std::string::npos &&
          completed_retire.find("stream.synchronize()") == std::string::npos,
      "compact Jacobi retirement releases an already-complete lease without "
      "a redundant stream fence");
  expect(
      round.find("graph.csr") == std::string::npos &&
          round.find("target_offsets") == std::string::npos &&
          round.find("workspace.targets") == std::string::npos &&
          kernel.find("atomicMin") == std::string::npos &&
          kernel.find("atomicCAS") == std::string::npos &&
          kernel.find("dense_chaotic_push") == std::string::npos &&
          kernel.find("DenseChaoticPush") == std::string::npos,
      "Phase 14 HIP has no CSR relaxation, distance atomic, target early stop, or Phase 15 engine");
#endif
}

}  // namespace

int main() {
  expect(
      bfnew::supported_batched_jacobi_width(1U) &&
          bfnew::supported_batched_jacobi_width(8U) &&
          bfnew::supported_batched_jacobi_width(16U) &&
          bfnew::supported_batched_jacobi_width(32U) &&
          !bfnew::supported_batched_jacobi_width(2U) &&
          !bfnew::supported_batched_jacobi_width(64U),
      "Phase 14 supports exactly widths 1, 8, 16, and 32");
  test_mixed_duration_matrix();
  test_descriptor_parity();
  test_width_one_matches_standalone();
  test_maximum_rounds_and_input_guards();
  test_hip_source_structure();
  if (failures != 0) {
    std::cerr << failures << " batched Jacobi assertion(s) failed\n";
    return 1;
  }
  std::cout << "batched Jacobi CPU tests passed\n";
  return 0;
}
