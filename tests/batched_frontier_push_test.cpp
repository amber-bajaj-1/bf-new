#include "bfnew/batched_frontier_push.hpp"
#include "bfnew/sssp.hpp"
#include "batched_frontier_fixture_suite.hpp"

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
  for (std::size_t local_vertex = 0U;
       local_vertex < induced.local_to_global.size();
       ++local_vertex) {
    global[induced.local_to_global[local_vertex].value()] =
        local.distances[local_vertex];
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
  throw std::logic_error{"batched frontier test lane query is missing"};
}

[[nodiscard]] std::vector<bfnew::GpuRunOptions> control_matrix(
    const bool per_lane_convergence,
    const std::uint64_t maximum_rounds = 64U,
    const bfnew::InstrumentationLevel instrumentation =
        bfnew::InstrumentationLevel::debug) {
  bfnew::GpuRunOptions base;
  base.engine = bfnew::EngineKind::frontier_push;
  base.instrumentation = instrumentation;
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

[[nodiscard]] bfnew::test::BatchedFrontierFixture
make_mixed_error_round_overflow_fixture() {
  const bfnew::ResourceClassId resource{1U};
  bfnew::InputGraph input{
      {
          bfnew::VertexMetadata::located(0, 0, resource),
          bfnew::VertexMetadata::located(10, 0, resource),
          bfnew::VertexMetadata::located(11, 0, resource),
      },
      {
          bfnew::test::jacobi_edge(0U, 1U, 1.0F, 16000U),
          bfnew::test::jacobi_edge(0U, 2U, 2.0F, 16001U),
      },
  };
  const bfnew::UniformGridPartitioner partitioner{
      bfnew::SpatialOrderConfig{0, 0, 10U, 1U}};
  bfnew::PartitionedGraph partitioned = partitioner.partition(input);
  const bfnew::WeightedGraph& graph = partitioned.graph;
  const auto old_to_new = graph.old_to_new();
  const auto owners = graph.owner_tiles();
  const bfnew::TileId source_tile = owners[old_to_new[0U].value()];
  const bfnew::TileId destination_tile = owners[old_to_new[1U].value()];
  const std::array shared_source{old_to_new[0U]};
  const std::array immediate_target{old_to_new[0U]};
  const std::array expanding_target{old_to_new[2U]};
  std::vector<bfnew::RouteQuery> queries;
  queries.push_back(bfnew::test::with_exact_selected_tiles(
      bfnew::make_route_query(
          bfnew::QueryId{1600U},
          graph,
          shared_source,
          immediate_target),
      {source_tile},
      graph));
  queries.push_back(bfnew::test::with_exact_selected_tiles(
      bfnew::make_route_query(
          bfnew::QueryId{1601U},
          graph,
          shared_source,
          expanding_target),
      {source_tile, destination_tile},
      graph));
  bfnew::TileRunLayout64 tile_runs = bfnew::build_tile_run_layout(graph);
  bfnew::DeviceGraphLayout32 device_graph =
      bfnew::build_device_graph_layout32(graph, tile_runs);
  return bfnew::test::BatchedFrontierFixture{
      std::move(partitioned),
      std::move(tile_runs),
      std::move(device_graph),
      std::move(queries),
  };
}

void expect_control_accounting(
    const bfnew::GpuRunOptions& options,
    const bfnew::HostBatchedFrontierRunResult& run,
    const std::string& prefix) {
  const std::uint64_t rounds = run.controller.rounds_completed;
  const bfnew::DeviceWorkStatistics& work = run.result.work;
  if (options.control_mode == bfnew::ControlMode::persistent_cooperative) {
    expect(
        work.kernel_dispatches == 1U && work.controller_copies == 1U &&
            work.host_synchronizations == 1U && work.host_checks == 1U &&
            run.queued_round_pairs == 0U &&
            run.completed_host_chunks == 0U,
        prefix + ": persistent control has no frontier polling");
    return;
  }
  if (options.control_mode == bfnew::ControlMode::per_round_host_poll) {
    expect(
        run.queued_round_pairs == rounds &&
            run.completed_host_chunks == rounds &&
            work.controller_copies == rounds &&
            work.host_synchronizations == rounds && work.host_checks == rounds &&
            work.kernel_dispatches == 1U + 2U * rounds,
        prefix + ": per-round control counts exact round/advance pairs");
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
      prefix + ": chunked control retains terminal queued no-ops");
}

[[nodiscard]] bool selected_outputs_equal(
    const bfnew::test::BatchedFrontierFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchPlanEntry& batch,
    const bfnew::HostBatchedFrontierRunResult& left,
    const bfnew::HostBatchedFrontierRunResult& right) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    const bfnew::LaneMask bit = bfnew::LaneMask{1U} << lane;
    if ((batch.valid_lane_mask & bit) == 0U) {
      continue;
    }
    const bfnew::RouteQuery& query = query_for_lane(queries, batch, lane);
    for (std::size_t vertex = 0U; vertex < graph.vertex_count(); ++vertex) {
      if (!query_selects_vertex(graph, query, vertex)) {
        continue;
      }
      const std::size_t index = static_cast<std::size_t>(
          bfnew::batched_frontier_distance_index(
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

void validate_converged_output(
    const bfnew::test::BatchedFrontierFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchPlanEntry& batch,
    const bfnew::BatchDeviceDescription& description,
    const bfnew::GpuRunOptions& options,
    const bfnew::HostBatchedFrontierRunResult& run,
    const std::string& prefix) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::size_t elements =
      static_cast<std::size_t>(graph.vertex_count()) * batch.lane_width;
  expect(
      run.distances.size() == elements && run.distance_bits.size() == elements,
      prefix + ": distances use exact V-by-W vertex-major storage");
  expect(
      run.result.status.stop_reason == static_cast<std::uint32_t>(
                                           bfnew::DeviceStopReason::converged) &&
          run.result.status.valid_lane_mask == batch.valid_lane_mask &&
          run.result.status.converged_lane_mask == batch.valid_lane_mask &&
          run.result.status.active_lane_mask == 0U &&
          bfnew::validate_device_controller(run.controller) ==
              bfnew::DeviceControllerError::none &&
          bfnew::validate_device_run_status(run.result.status) ==
              bfnew::DeviceRunStatusError::none,
      prefix + ": terminal controller and status validate");
  expect(
      run.current_frontier_sizes.size() == run.controller.rounds_completed &&
          run.current_frontier_lane_unions.size() ==
              run.controller.rounds_completed &&
          run.batch_work.frontier_rounds == run.controller.rounds_completed &&
          run.controller.frontier_size[0U] == 0U &&
          run.controller.frontier_size[1U] == 0U,
      prefix + ": every executed round has one merged-frontier trace entry");
  expect(
      std::ranges::all_of(
          run.frontier_lane_masks[0U],
          [](const bfnew::LaneMask mask) { return mask == 0U; }) &&
          std::ranges::all_of(
              run.frontier_lane_masks[1U],
              [](const bfnew::LaneMask mask) { return mask == 0U; }),
      prefix + ": converged recycled activity-mask slots are clear");

  bfnew::LaneMask expected_reached = 0U;
  bfnew::LaneMask expected_miss = 0U;
  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    const bfnew::LaneMask bit = bfnew::LaneMask{1U} << lane;
    if ((batch.valid_lane_mask & bit) == 0U) {
      expect(
          run.frontier_rounds_by_lane[lane] == 0U &&
              run.convergence_round_by_lane[lane] == 0U &&
              run.tail_rounds_by_lane[lane] == 0U,
          prefix + ": padded lanes have no frontier state");
      for (std::size_t vertex = 0U; vertex < graph.vertex_count(); ++vertex) {
        const std::size_t index = static_cast<std::size_t>(
            bfnew::batched_frontier_distance_index(
                static_cast<std::uint32_t>(vertex),
                static_cast<std::uint32_t>(lane),
                batch.lane_width));
        expect(
            run.distance_bits[index] == 0x7f800000U &&
                std::isinf(run.distances[index]),
            prefix + ": every padded V-by-W cell remains positive infinity");
      }
      continue;
    }
    const bfnew::RouteQuery& query = query_for_lane(queries, batch, lane);
    const std::vector<float> oracle = bounded_oracle(graph, query);
    for (std::size_t vertex = 0U; vertex < graph.vertex_count(); ++vertex) {
      if (!query_selects_vertex(graph, query, vertex)) {
        const std::size_t index = static_cast<std::size_t>(
            bfnew::batched_frontier_distance_index(
                static_cast<std::uint32_t>(vertex),
                static_cast<std::uint32_t>(lane),
                batch.lane_width));
        expect(
            run.distance_bits[index] == 0x7f800000U &&
                std::isinf(run.distances[index]),
            prefix + ": nonselected valid-lane cells remain positive infinity");
        continue;
      }
      const std::size_t index = static_cast<std::size_t>(
          bfnew::batched_frontier_distance_index(
              static_cast<std::uint32_t>(vertex),
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      expect(
          bits_equal(run.distances[index], oracle[vertex]) &&
              run.distance_bits[index] ==
                  bfnew::dense_atomic_float_bits(run.distances[index]),
          prefix + ": every selected lane label matches bounded Dijkstra");
    }
    bool reached = true;
    for (const bfnew::VertexId target : query.targets) {
      const std::size_t index = static_cast<std::size_t>(
          bfnew::batched_frontier_distance_index(
              target.value(),
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      reached = reached && run.distance_bits[index] != 0x7f800000U;
    }
    if (reached) {
      expected_reached |= bit;
    } else {
      expected_miss |= bit;
    }
    expect(
        run.frontier_rounds_by_lane[lane] ==
                run.convergence_round_by_lane[lane] &&
            run.tail_rounds_by_lane[lane] ==
                run.controller.rounds_completed -
                    run.convergence_round_by_lane[lane],
        prefix + ": a lane executes no work after its frontier empties");
  }
  expect(
      run.result.status.reached_target_mask == expected_reached &&
          run.result.status.bounding_box_miss_mask == expected_miss,
      prefix + ": reached and bounded-miss masks are lane exact");

  const bfnew::BatchedFrontierWorkStatistics& batch_work = run.batch_work;
  const bfnew::DeviceWorkStatistics& common = run.result.work;
  expect(
      batch_work.csr_runs_considered ==
              batch_work.csr_runs_visited + batch_work.csr_runs_skipped &&
          batch_work.active_vertex_lane_pairs ==
              batch_work.frontier_vertex_entries +
                  batch_work.shared_vertex_entries_saved &&
          batch_work.lane_edge_relaxations ==
              batch_work.csr_edge_loads +
                  batch_work.shared_edge_lane_work_saved,
      prefix + ": vertex/run/edge sharing identities balance");
  expect(
      batch_work.distance_atomic_source_loads ==
              batch_work.distance_atomic_attempts &&
          batch_work.distance_atomic_attempts ==
              batch_work.lane_edge_relaxations &&
          batch_work.successful_distance_atomic_updates ==
              batch_work.unique_next_vertex_lane_activations +
                  batch_work.same_lane_duplicate_suppressions &&
          batch_work.unique_next_vertex_lane_activations ==
              batch_work.queue_claims +
                  batch_work.queue_entries_saved_by_lane_merging &&
          batch_work.duplicate_suppressions ==
              batch_work.queue_entries_saved_by_lane_merging +
                  batch_work.same_lane_duplicate_suppressions,
      prefix + ": atomic OR, claim, merge, and duplicate identities balance");
  expect(
      batch_work.valid_lane_round_capacity ==
              batch_work.active_lane_rounds +
                  batch_work.inactive_valid_lane_rounds &&
          batch_work.lane_width_round_capacity ==
              batch_work.valid_lane_round_capacity +
                  batch_work.padded_lane_round_capacity &&
          batch_work.wave32_lane_round_capacity ==
              batch_work.lane_width_round_capacity +
                  batch_work.unused_wave_lane_round_capacity &&
          batch_work.tail_lane_rounds ==
              batch_work.tail_lane_rounds_without_frontier_work &&
          batch_work.semantic_lane_edge_work_avoided_by_per_lane_convergence ==
              0U &&
          batch_work.padded_lane_semantic_work == 0U,
      prefix + ": lane capacities and no-work tails are honest");
  expect(
      batch_work.distance_reset_bytes ==
              batch.selected_lane_vertex_count * sizeof(std::uint32_t) &&
          batch_work.activity_mask_reset_bytes ==
              batch.union_vertex_count * 2U * sizeof(bfnew::LaneMask) &&
          batch_work.source_seed_write_bytes ==
              description.sources.size() * sizeof(std::uint32_t) &&
          batch_work.frontier_queue_storage_bytes ==
              run.queue_capacity * 2U * sizeof(std::uint32_t),
      prefix + ": reset and retained queue byte terms are exact");
  expect(
      common.edges_examined == batch_work.lane_edge_relaxations &&
          common.successful_decreases ==
              batch_work.successful_distance_atomic_updates &&
          common.active_vertices == batch_work.frontier_vertex_entries &&
          common.active_lane_rounds == batch_work.active_lane_rounds &&
          common.atomic_attempts == batch_work.distance_atomic_attempts &&
          common.successful_atomic_updates ==
              batch_work.successful_distance_atomic_updates &&
          common.queue_claims == batch_work.queue_claims &&
          common.duplicate_suppressions ==
              batch_work.duplicate_suppressions &&
          common.maximum_queue_size == batch_work.maximum_queue_size,
      prefix + ": common Debug counters match frontier semantic counters");
  expect_control_accounting(options, run, prefix);
}

void test_mixed_duration_matrix() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  for (const bfnew::BatchRunRepresentation representation : {
           bfnew::BatchRunRepresentation::retained_per_run_masks,
           bfnew::BatchRunRepresentation::compact_nonzero_descriptors,
       }) {
    for (const std::uint32_t width : {8U, 16U, 32U}) {
      const bfnew::test::PreparedBatchedFrontierFixture prepared =
          bfnew::test::prepare_batched_frontier_fixture(
              fixture, fixture.queries, width, representation);
      const bfnew::BatchPlanEntry& batch = prepared.plan.batches.front();
      for (std::size_t control = 0U; control < control_matrix(true).size();
           ++control) {
        const bfnew::GpuRunOptions enabled_options =
            control_matrix(true)[control];
        const bfnew::GpuRunOptions disabled_options =
            control_matrix(false)[control];
        const bfnew::HostBatchedFrontierRunResult enabled =
            bfnew::run_host_batched_frontier_push(
                fixture.device_graph,
                fixture.queries,
                batch,
                prepared.description,
                enabled_options);
        const bfnew::HostBatchedFrontierRunResult disabled =
            bfnew::run_host_batched_frontier_push(
                fixture.device_graph,
                fixture.queries,
                batch,
                prepared.description,
                disabled_options);
        const std::string prefix =
            "frontier-representation-" +
            std::to_string(static_cast<std::uint32_t>(representation)) +
            "-width-" + std::to_string(width) + "-control-" +
            std::to_string(control);
        validate_converged_output(
            fixture,
            fixture.queries,
            batch,
            prepared.description,
            enabled_options,
            enabled,
            prefix + "-enabled");
        validate_converged_output(
            fixture,
            fixture.queries,
            batch,
            prepared.description,
            disabled_options,
            disabled,
            prefix + "-disabled");
        expect(
            selected_outputs_equal(
                fixture, fixture.queries, batch, enabled, disabled) &&
                enabled.batch_work == disabled.batch_work,
            prefix + ": convergence publication preserves labels and work");
        expect(
            enabled.controller.rounds_completed == 6U &&
                disabled.controller.rounds_completed == 6U,
            prefix + ": longest frontier lane empties in round six");
        const std::array<std::pair<bfnew::QueryId, std::uint64_t>, 5U>
            expected_rounds{{
                {bfnew::QueryId{1400U}, 1U},
                {bfnew::QueryId{1401U}, 2U},
                {bfnew::QueryId{1402U}, 6U},
                {bfnew::QueryId{1403U}, 1U},
                {bfnew::QueryId{1404U}, 4U},
            }};
        for (const auto& [query_id, rounds] : expected_rounds) {
          const std::size_t lane =
              bfnew::test::frontier_lane_for_query(batch, query_id);
          expect(
              enabled.convergence_round_by_lane[lane] == rounds &&
                  disabled.convergence_round_by_lane[lane] == rounds,
              prefix + ": lane frontier duration is exact");
        }
        expect(
            enabled.batch_work.initial_queue_entries_saved_by_lane_merging >=
                    1U &&
                enabled.batch_work.shared_vertex_entries_saved >= 1U &&
                enabled.batch_work.queue_entries_saved_by_lane_merging >= 1U &&
                enabled.batch_work.multi_lane_csr_edge_loads >= 1U,
            prefix + ": shared sources, vertices, queues, and edges are measured");
      }
    }
  }
}

void test_representation_parity_matrix() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  for (const std::uint32_t width : {8U, 16U, 32U}) {
    const bfnew::test::PreparedBatchedFrontierFixture retained =
        bfnew::test::prepare_batched_frontier_fixture(
            fixture,
            fixture.queries,
            width,
            bfnew::BatchRunRepresentation::retained_per_run_masks);
    const bfnew::test::PreparedBatchedFrontierFixture descriptors =
        bfnew::test::prepare_batched_frontier_fixture(
            fixture,
            fixture.queries,
            width,
            bfnew::BatchRunRepresentation::compact_nonzero_descriptors);
    expect(
        retained.plan == descriptors.plan,
        "retained and descriptor frontier images share the exact batch plan");
    const bfnew::BatchPlanEntry& batch = retained.plan.batches.front();
    for (const bool per_lane : {true, false}) {
      const std::vector<bfnew::GpuRunOptions> controls =
          control_matrix(per_lane);
      for (std::size_t control = 0U; control < controls.size(); ++control) {
        const bfnew::HostBatchedFrontierRunResult retained_run =
            bfnew::run_host_batched_frontier_push(
                fixture.device_graph,
                fixture.queries,
                batch,
                retained.description,
                controls[control]);
        const bfnew::HostBatchedFrontierRunResult descriptor_run =
            bfnew::run_host_batched_frontier_push(
                fixture.device_graph,
                fixture.queries,
                descriptors.plan.batches.front(),
                descriptors.description,
                controls[control]);
        const std::string prefix =
            "frontier-representation-parity-width-" +
            std::to_string(width) + "-toggle-" +
            std::to_string(static_cast<unsigned int>(per_lane)) +
            "-control-" + std::to_string(control);
        expect(
            selected_outputs_equal(
                fixture,
                fixture.queries,
                batch,
                retained_run,
                descriptor_run) &&
                retained_run.distance_bits == descriptor_run.distance_bits &&
                retained_run.result.status.stop_reason ==
                    descriptor_run.result.status.stop_reason &&
                retained_run.result.status.rounds_completed ==
                    descriptor_run.result.status.rounds_completed &&
                retained_run.result.status.reached_target_mask ==
                    descriptor_run.result.status.reached_target_mask &&
                retained_run.result.status.bounding_box_miss_mask ==
                    descriptor_run.result.status.bounding_box_miss_mask &&
                retained_run.convergence_round_by_lane ==
                    descriptor_run.convergence_round_by_lane,
            prefix +
                ": retained and descriptor images preserve every V-by-W bit and terminal lane result");
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
  for (const bool per_lane : {true, false}) {
    for (const bfnew::GpuRunOptions& options : control_matrix(per_lane)) {
      const bfnew::HostFrontierRunResult standalone =
          bfnew::run_host_frontier_push(
              fixture.device_graph,
              singleton.front(),
              prepared.description.tile_lane_masks,
              prepared.description.csr_run_lane_masks,
              options);
      const bfnew::HostBatchedFrontierRunResult batched =
          bfnew::run_host_batched_frontier_push(
              fixture.device_graph,
              singleton,
              prepared.plan.batches.front(),
              prepared.description,
              options);
      expect(
          batched.distance_bits == standalone.distance_bits &&
              batched.controller.rounds_completed ==
                  standalone.controller.rounds_completed &&
              batched.result.status.reached_target_mask ==
                  standalone.result.status.reached_target_mask &&
              batched.result.status.bounding_box_miss_mask ==
                  standalone.result.status.bounding_box_miss_mask,
          "width one is bitwise identical to standalone frontier push");
    }
  }
}

void test_zero_weight_cycle() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_single_query_batched_frontier_fixture(
          bfnew::test::make_phase5_zero_cycle_jacobi_fixture());
  std::vector<std::uint32_t> retained_bits;
  for (const bfnew::BatchRunRepresentation representation : {
           bfnew::BatchRunRepresentation::retained_per_run_masks,
           bfnew::BatchRunRepresentation::compact_nonzero_descriptors,
       }) {
    const bfnew::test::PreparedBatchedFrontierFixture prepared =
        bfnew::test::prepare_batched_frontier_fixture(
            fixture, fixture.queries, 1U, representation);
    bfnew::GpuRunOptions options = control_matrix(true).front();
    options.maximum_rounds = 32U;
    const bfnew::HostBatchedFrontierRunResult run =
        bfnew::run_host_batched_frontier_push(
            fixture.device_graph,
            fixture.queries,
            prepared.plan.batches.front(),
            prepared.description,
            options);
    const std::string prefix =
        "zero-weight-cycle-representation-" +
        std::to_string(static_cast<std::uint32_t>(representation));
    validate_converged_output(
        fixture,
        fixture.queries,
        prepared.plan.batches.front(),
        prepared.description,
        options,
        run,
        prefix);
    expect(
        run.controller.rounds_completed > 0U &&
            run.controller.rounds_completed < options.maximum_rounds &&
            run.result.status.converged == 1U,
        prefix + ": strict decreases terminate across the zero-weight cycle");
    if (retained_bits.empty()) {
      retained_bits = run.distance_bits;
    } else {
      expect(
          run.distance_bits == retained_bits,
          prefix + ": retained and descriptor zero-cycle bits agree exactly");
    }
  }
}

void test_high_fan_in() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_single_query_batched_frontier_fixture(
          bfnew::test::make_high_fan_in_dense_fixture());
  std::vector<std::uint32_t> retained_bits;
  for (const bfnew::BatchRunRepresentation representation : {
           bfnew::BatchRunRepresentation::retained_per_run_masks,
           bfnew::BatchRunRepresentation::compact_nonzero_descriptors,
       }) {
    const bfnew::test::PreparedBatchedFrontierFixture prepared =
        bfnew::test::prepare_batched_frontier_fixture(
            fixture, fixture.queries, 1U, representation);
    bfnew::GpuRunOptions options = control_matrix(true).front();
    options.maximum_rounds = 32U;
    const bfnew::HostBatchedFrontierRunResult run =
        bfnew::run_host_batched_frontier_push(
            fixture.device_graph,
            fixture.queries,
            prepared.plan.batches.front(),
            prepared.description,
            options);
    const std::string prefix =
        "high-fan-in-representation-" +
        std::to_string(static_cast<std::uint32_t>(representation));
    validate_converged_output(
        fixture,
        fixture.queries,
        prepared.plan.batches.front(),
        prepared.description,
        options,
        run,
        prefix);
    expect(
        run.batch_work.distance_atomic_attempts == 24U &&
            run.batch_work.queue_claims == 13U &&
            run.batch_work.unique_next_vertex_lane_activations == 13U &&
            run.batch_work.maximum_queue_size == 12U &&
            run.batch_work.successful_distance_atomic_updates >= 13U &&
            run.batch_work.successful_distance_atomic_updates <= 14U &&
            run.batch_work.duplicate_suppressions ==
                run.batch_work.successful_distance_atomic_updates - 13U,
        prefix +
            ": twelve-way fan-in collapses the shared hub to one queue claim");
    if (retained_bits.empty()) {
      retained_bits = run.distance_bits;
    } else {
      expect(
          run.distance_bits == retained_bits,
          prefix + ": retained and descriptor fan-in bits agree exactly");
    }
  }
}

void test_repeated_improvement_and_instrumentation() {
  const bfnew::test::BatchedFrontierFixture fixture =
      bfnew::test::make_single_query_batched_frontier_fixture(
          bfnew::test::make_repeated_improvement_frontier_fixture());
  const bfnew::test::PreparedBatchedFrontierFixture prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          fixture,
          fixture.queries,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  std::array<bfnew::HostBatchedFrontierRunResult, 3U> runs;
  for (std::size_t level = 0U; level < runs.size(); ++level) {
    bfnew::GpuRunOptions options = control_matrix(true).front();
    options.instrumentation = static_cast<bfnew::InstrumentationLevel>(level);
    runs[level] = bfnew::run_host_batched_frontier_push(
        fixture.device_graph,
        fixture.queries,
        prepared.plan.batches.front(),
        prepared.description,
        options);
  }
  expect(
      runs[0U].distance_bits == runs[1U].distance_bits &&
          runs[0U].distance_bits == runs[2U].distance_bits &&
          runs[0U].batch_work == runs[1U].batch_work &&
          runs[0U].batch_work == runs[2U].batch_work,
      "None, Light, and Debug preserve frontier labels and semantic counters");
  expect(
      runs[2U].batch_work.same_lane_duplicate_suppressions == 2U &&
          runs[2U].batch_work.duplicate_suppressions == 2U &&
          runs[2U].batch_work.queue_entries_saved_by_lane_merging == 0U,
      "three descending fan-in improvements claim once and suppress twice");

  const bfnew::DeviceWorkStatistics& none = runs[0U].result.work;
  const bfnew::DeviceWorkStatistics& light = runs[1U].result.work;
  const bfnew::DeviceWorkStatistics& debug = runs[2U].result.work;
  expect(
      none.edges_examined == 0U && none.successful_decreases == 0U &&
          none.active_vertices == 0U && none.active_lane_rounds == 0U &&
          none.maximum_queue_size == 0U && none.atomic_attempts == 0U &&
          none.queue_claims == 0U && none.mask_operations == 0U,
      "None suppresses every algorithmic common counter");
  expect(
      light.edges_examined != 0U && light.successful_decreases != 0U &&
          light.active_vertices != 0U && light.maximum_queue_size != 0U &&
          light.atomic_attempts == 0U && light.queue_claims == 0U &&
          light.duplicate_suppressions == 0U && light.mask_operations == 0U,
      "Light retains aggregate frontier work but suppresses Debug atomics");
  expect(
      debug.atomic_attempts ==
              runs[2U].batch_work.distance_atomic_attempts &&
          debug.successful_atomic_updates ==
              runs[2U].batch_work.successful_distance_atomic_updates &&
          debug.queue_claims == runs[2U].batch_work.queue_claims &&
          debug.duplicate_suppressions == 2U && debug.mask_operations != 0U,
      "Debug exposes distance, mask, queue, and duplicate operations");
}

void test_overflow_maximum_rounds_and_guards() {
  const bfnew::test::BatchedFrontierFixture expanding =
      bfnew::test::make_single_query_batched_frontier_fixture(
          bfnew::test::make_expanding_grid_frontier_fixture());
  const bfnew::test::PreparedBatchedFrontierFixture expanding_prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          expanding,
          expanding.queries,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  for (const bool per_lane : {true, false}) {
    for (const bfnew::GpuRunOptions& options : control_matrix(per_lane)) {
      const bfnew::HostBatchedFrontierRunResult overflow =
          bfnew::run_host_batched_frontier_push(
              expanding.device_graph,
              expanding.queries,
              expanding_prepared.plan.batches.front(),
              expanding_prepared.description,
              options,
              1U);
      expect(
          overflow.result.status.stop_reason ==
                  static_cast<std::uint32_t>(
                      bfnew::DeviceStopReason::queue_overflow) &&
              overflow.controller.rounds_completed == 1U &&
              overflow.batch_work.overflow_events == 1U &&
              overflow.batch_work.maximum_queue_size >
                  overflow.queue_capacity &&
              overflow.result.status.bounding_box_miss_mask == 0U &&
              std::ranges::all_of(
                  overflow.convergence_round_by_lane,
                  [](const std::uint64_t round) { return round == 0U; }) &&
              std::ranges::all_of(
                  overflow.tail_rounds_by_lane,
                  [](const std::uint64_t tail) { return tail == 0U; }),
          "round queue overflow is explicit after one executed round");
    }
  }

  const bfnew::test::BatchedFrontierFixture mixed_error =
      make_mixed_error_round_overflow_fixture();
  for (const bfnew::BatchRunRepresentation representation : {
           bfnew::BatchRunRepresentation::retained_per_run_masks,
           bfnew::BatchRunRepresentation::compact_nonzero_descriptors,
       }) {
    const bfnew::test::PreparedBatchedFrontierFixture prepared =
        bfnew::test::prepare_batched_frontier_fixture(
            mixed_error, mixed_error.queries, 8U, representation);
    for (const bool per_lane : {true, false}) {
      for (const bfnew::GpuRunOptions& options : control_matrix(per_lane)) {
        const bfnew::HostBatchedFrontierRunResult overflow =
            bfnew::run_host_batched_frontier_push(
                mixed_error.device_graph,
                mixed_error.queries,
                prepared.plan.batches.front(),
                prepared.description,
                options,
                1U);
        expect(
            overflow.result.status.stop_reason ==
                    static_cast<std::uint32_t>(
                        bfnew::DeviceStopReason::queue_overflow) &&
                overflow.controller.rounds_completed == 1U &&
                overflow.batch_work.initial_queue_entries == 1U &&
                overflow.batch_work
                        .initial_queue_entries_saved_by_lane_merging == 1U &&
                overflow.batch_work.maximum_queue_size == 2U &&
                std::ranges::all_of(
                    overflow.convergence_round_by_lane,
                    [](const std::uint64_t round) { return round == 0U; }) &&
                std::ranges::all_of(
                    overflow.tail_rounds_by_lane,
                    [](const std::uint64_t tail) { return tail == 0U; }),
            "a no-next lane publishes no evidence when its shared round overflows");
      }
    }
  }

  const bfnew::test::BatchedFrontierFixture mixed =
      bfnew::test::make_mixed_duration_batched_frontier_fixture();
  const std::array<bfnew::RouteQuery, 1U> two_source{mixed.queries[4U]};
  const bfnew::test::PreparedBatchedFrontierFixture two_source_prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          mixed,
          two_source,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  const bfnew::HostBatchedFrontierRunResult initial_overflow =
      bfnew::run_host_batched_frontier_push(
          mixed.device_graph,
          two_source,
          two_source_prepared.plan.batches.front(),
          two_source_prepared.description,
          control_matrix(true).front(),
          1U);
  expect(
      initial_overflow.result.status.stop_reason ==
              static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::queue_overflow) &&
          initial_overflow.controller.rounds_completed == 0U &&
          initial_overflow.batch_work.initial_queue_entries == 2U &&
          initial_overflow.batch_work.overflow_events == 1U &&
          std::ranges::all_of(
              initial_overflow.convergence_round_by_lane,
              [](const std::uint64_t round) { return round == 0U; }) &&
          std::ranges::all_of(
              initial_overflow.tail_rounds_by_lane,
              [](const std::uint64_t tail) { return tail == 0U; }),
      "initial unique-source overflow is explicit before round one");

  const std::array<bfnew::RouteQuery, 1U> long_query{mixed.queries[2U]};
  const bfnew::test::PreparedBatchedFrontierFixture limited_prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          mixed,
          long_query,
          1U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  for (const bool per_lane : {true, false}) {
    for (const bfnew::GpuRunOptions& options : control_matrix(per_lane, 2U)) {
      const bfnew::HostBatchedFrontierRunResult limited =
          bfnew::run_host_batched_frontier_push(
              mixed.device_graph,
              long_query,
              limited_prepared.plan.batches.front(),
              limited_prepared.description,
              options);
      expect(
          limited.result.status.stop_reason ==
                  static_cast<std::uint32_t>(
                      bfnew::DeviceStopReason::maximum_rounds) &&
              limited.controller.rounds_completed == 2U &&
              limited.result.status.reached_target_mask == 0U &&
              limited.result.status.bounding_box_miss_mask == 0U,
          "maximum-round exhaustion remains distinct from miss");
    }
  }

  const std::array<bfnew::RouteQuery, 2U> mixed_limit_queries{
      mixed.queries[0U], mixed.queries[2U]};
  const bfnew::test::PreparedBatchedFrontierFixture mixed_limit_prepared =
      bfnew::test::prepare_batched_frontier_fixture(
          mixed,
          mixed_limit_queries,
          8U,
          bfnew::BatchRunRepresentation::retained_per_run_masks);
  const bfnew::BatchPlanEntry& mixed_limit_batch =
      mixed_limit_prepared.plan.batches.front();
  const std::size_t immediate_lane = bfnew::test::frontier_lane_for_query(
      mixed_limit_batch, bfnew::QueryId{1400U});
  const std::size_t live_lane = bfnew::test::frontier_lane_for_query(
      mixed_limit_batch, bfnew::QueryId{1402U});
  for (const bool per_lane : {true, false}) {
    for (const bfnew::GpuRunOptions& options : control_matrix(per_lane, 1U)) {
      const bfnew::HostBatchedFrontierRunResult limited =
          bfnew::run_host_batched_frontier_push(
              mixed.device_graph,
              mixed_limit_queries,
              mixed_limit_batch,
              mixed_limit_prepared.description,
              options);
      expect(
          limited.result.status.stop_reason ==
                  static_cast<std::uint32_t>(
                      bfnew::DeviceStopReason::maximum_rounds) &&
              limited.controller.rounds_completed == 1U &&
              limited.convergence_round_by_lane[immediate_lane] == 1U &&
              limited.convergence_round_by_lane[live_lane] == 0U &&
              limited.tail_rounds_by_lane[immediate_lane] == 0U &&
              limited.tail_rounds_by_lane[live_lane] == 0U,
          "a clean maximum-round transition publishes only its proven no-next lane");
    }
  }

  bfnew::GpuRunOptions wrong_engine = control_matrix(true).front();
  wrong_engine.engine = bfnew::EngineKind::dense_chaotic_push;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_frontier_push(
            mixed.device_graph,
            long_query,
            limited_prepared.plan.batches.front(),
            limited_prepared.description,
            wrong_engine));
      },
      "batched frontier rejects another engine identity");

  bfnew::BatchDeviceDescription corrupt = limited_prepared.description;
  corrupt.csr_run_lane_masks[corrupt.touched_csr_runs.front()] = 0U;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_frontier_push(
            mixed.device_graph,
            long_query,
            limited_prepared.plan.batches.front(),
            corrupt,
            control_matrix(true).front()));
      },
      "batched frontier rejects a non-endpoint-exact run mask");
}

void test_scratch_layout_and_controller_guards() {
  const bfnew::BatchedFrontierScratchLayout layout =
      bfnew::make_batched_frontier_scratch_layout(10U, 8U);
  expect(
      layout.queue_capacity == 10U && layout.distance_bits_offset == 0U &&
          layout.distance_bits_bytes == 10U * 8U * sizeof(std::uint32_t) &&
          layout.activity_mask_offsets[0U] == layout.distance_bits_bytes &&
          layout.activity_mask_offsets[1U] ==
              layout.activity_mask_offsets[0U] +
                  10U * sizeof(bfnew::LaneMask) &&
          layout.frontier_offsets[0U] ==
              layout.activity_mask_offsets[1U] +
                  10U * sizeof(bfnew::LaneMask) &&
          layout.frontier_offsets[1U] ==
              layout.frontier_offsets[0U] + 10U * sizeof(std::uint32_t) &&
          layout.total_bytes ==
              10U * 8U * sizeof(std::uint32_t) +
                  20U * sizeof(bfnew::LaneMask) +
                  20U * sizeof(std::uint32_t),
      "default-Q scratch layout is exact distance, masks, then queues");
  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(bfnew::make_batched_frontier_scratch_layout(0U, 8U)); },
      "scratch rejects an empty graph");
  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(bfnew::make_batched_frontier_scratch_layout(4U, 2U)); },
      "scratch rejects an unsupported width");
  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(bfnew::make_batched_frontier_scratch_layout(4U, 8U, 5U)); },
      "scratch rejects capacity above V");
  if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
    expect_throws<std::overflow_error>(
        [] {
          static_cast<void>(bfnew::make_batched_frontier_scratch_layout(
              static_cast<std::size_t>(
                  std::numeric_limits<std::uint32_t>::max()) +
                  1U,
              32U));
        },
        "scratch rejects a vertex count above the device ABI");
  }

  bfnew::GpuRunOptions options = control_matrix(true).front();
  bfnew::DeviceController controller =
      bfnew::initialize_device_controller(options, 0x3U, 1U);
  controller.next_frontier_lane_mask = 0x4U;
  controller.frontier_size[controller.frontier_write_slot] = 1U;
  expect(
      bfnew::advance_batched_frontier_controller(controller) ==
              bfnew::BatchedFrontierAdvanceResult::invalid_controller_state &&
          controller.stop_reason == static_cast<std::uint32_t>(
                                        bfnew::DeviceStopReason::invalid_controller_state),
      "advance rejects a next lane outside the just-executed mask");

  controller = bfnew::initialize_device_controller(options, 0x3U, 1U);
  controller.active_lane_mask = 0x1U;
  controller.execute_lane_mask = 0x1U;
  expect(
      bfnew::advance_batched_frontier_controller(controller) ==
          bfnew::BatchedFrontierAdvanceResult::invalid_controller_state,
      "enabled advance requires active and converged lanes to cover valid");

  options.enable_per_lane_convergence = 0U;
  controller = bfnew::initialize_device_controller(options, 0x3U, 1U);
  controller.converged_lane_mask = 0x2U;
  controller.active_lane_mask = 0x1U;
  controller.execute_lane_mask = 0x1U;
  expect(
      bfnew::advance_batched_frontier_controller(controller) ==
          bfnew::BatchedFrontierAdvanceResult::invalid_controller_state,
      "disabled advance forbids premature converged-lane publication");
}

#if (defined(BFNEW_PHASE16_HIP_SOURCE_PATH) && \
     !defined(BFNEW_PHASE16_WORKSPACE_HIP_SOURCE_PATH)) || \
    (!defined(BFNEW_PHASE16_HIP_SOURCE_PATH) && \
     defined(BFNEW_PHASE16_WORKSPACE_HIP_SOURCE_PATH))
#error "Phase 16 structural source paths must be defined together"
#endif

#if defined(BFNEW_PHASE16_HIP_SOURCE_PATH) && \
    defined(BFNEW_PHASE16_WORKSPACE_HIP_SOURCE_PATH)
[[nodiscard]] std::string read_source_file(const char* const path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error{"cannot open Phase 16 structural source"};
  }
  return std::string{
      std::istreambuf_iterator<char>{input},
      std::istreambuf_iterator<char>{}};
}
#endif

void test_future_hip_source_structure() {
#if defined(BFNEW_PHASE16_HIP_SOURCE_PATH) && \
    defined(BFNEW_PHASE16_WORKSPACE_HIP_SOURCE_PATH)
  const std::string kernel = read_source_file(BFNEW_PHASE16_HIP_SOURCE_PATH);
  const std::string workspace =
      read_source_file(BFNEW_PHASE16_WORKSPACE_HIP_SOURCE_PATH);
  const std::size_t round_begin =
      kernel.find("perform_batched_frontier_round");
  const std::size_t round_end =
      kernel.find("advance_batched_frontier_and_count", round_begin);
  const std::string round =
      round_begin != std::string::npos && round_end != std::string::npos &&
              round_begin < round_end
          ? kernel.substr(round_begin, round_end - round_begin)
          : std::string{};
  const std::size_t completed_retire_begin = workspace.find(
      "void ReusableBatchedFrontierWorkspace::retire_after_stream_completion");
  const std::size_t completed_retire_end = workspace.find(
      "void ReusableBatchedFrontierWorkspace::recover_noexcept",
      completed_retire_begin);
  const std::string completed_retire =
      completed_retire_begin != std::string::npos &&
              completed_retire_end != std::string::npos &&
              completed_retire_begin < completed_retire_end
          ? workspace.substr(completed_retire_begin,
                             completed_retire_end - completed_retire_begin)
          : std::string{};
  expect(
      !round.empty() &&
          round.find("graph.csr.row_run_offsets") != std::string::npos &&
          round.find("graph.csr.run_edge_offsets") != std::string::npos &&
          round.find("graph.csr.destinations") != std::string::npos &&
          round.find("graph.csr.weights") != std::string::npos,
      "future HIP frontier round traverses outgoing CSR tile runs");
  expect(
      round.find("atomicExch") != std::string::npos &&
          round.find("successful_lanes") != std::string::npos &&
          round.find("old_destination_mask == 0U") != std::string::npos &&
          round.find("atomicOr") != std::string::npos,
      "future HIP frontier round exchanges current masks and claims only zero-transition destinations");
  expect(
      kernel.find("advance_batched_frontier_controller") !=
              std::string::npos &&
          kernel.find("cg::this_grid") != std::string::npos &&
          kernel.find("grid.sync()") != std::string::npos &&
          kernel.find("hipLaunchCooperativeKernel") != std::string::npos &&
          kernel.find("hipOccupancyMaxActiveBlocksPerMultiprocessor") !=
              std::string::npos,
      "future HIP controls share the batched transition and real-kernel occupancy");
  expect(
      round.find("prefix_sum") == std::string::npos &&
          round.find("virtual_warp") == std::string::npos &&
          round.find("adaptive_frontier") == std::string::npos &&
          round.find("dense_chaotic") == std::string::npos,
      "future HIP round has no deferred balancing, aggregation, adaptive, or dense fallback");
  expect(
      workspace.find("ReusableBatchedFrontierWorkspace") !=
              std::string::npos &&
          workspace.find("DeviceBuffer engine_scratch") != std::string::npos &&
          workspace.find("selected_ranges") != std::string::npos &&
          workspace.find("lane_convergence_rounds") != std::string::npos &&
          workspace.find("engine_scratch.clear_async") == std::string::npos,
      "future HIP workspace retains selected-only frontier state without full scratch clear");
  expect(
      kernel.find("retire_after_stream_completion") != std::string::npos &&
          completed_retire.find("clear_lease()") != std::string::npos &&
          completed_retire.find("stream.synchronize()") == std::string::npos,
      "compact frontier retirement has no redundant stream fence");
#endif
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
      "Phase 16 supports exactly widths 1, 8, 16, and 32");
  test_mixed_duration_matrix();
  test_representation_parity_matrix();
  test_width_one_matches_standalone();
  test_zero_weight_cycle();
  test_high_fan_in();
  test_repeated_improvement_and_instrumentation();
  test_overflow_maximum_rounds_and_guards();
  test_scratch_layout_and_controller_guards();
  test_future_hip_source_structure();
  if (failures != 0) {
    std::cerr << failures << " batched frontier assertion(s) failed\n";
    return 1;
  }
  std::cout << "batched frontier push CPU tests passed\n";
  return 0;
}
