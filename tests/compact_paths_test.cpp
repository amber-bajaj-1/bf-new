#include "bfnew/compact_paths.hpp"
#include "bfnew/no_congestion_pipeline.hpp"
#include "bfnew/sssp.hpp"
#include "batched_expansion_fixture_suite.hpp"
#include "compact_paths_fixture_suite.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <ranges>
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

template <typename Function>
void expect_throws(Function&& function, const std::string_view description) {
  try {
    function();
    expect(false, description);
  } catch (const std::exception&) {
  } catch (...) {
    expect(false, description);
  }
}

[[nodiscard]] bool bitwise_equal(const float left, const float right) noexcept {
  return std::bit_cast<std::uint32_t>(left) ==
         std::bit_cast<std::uint32_t>(right);
}

[[nodiscard]] std::vector<float> induced_global_distances(
    const bfnew::WeightedGraph& graph,
    const bfnew::RouteQuery& query) {
  const bfnew::InducedQueryGraph induced =
      bfnew::build_induced_query_graph(graph, query);
  const bfnew::SsspResult oracle =
      bfnew::dijkstra_oracle(induced.graph, induced.sources);
  std::vector<float> result(
      graph.vertex_count(), std::numeric_limits<float>::infinity());
  for (std::size_t local = 0U; local < induced.local_to_global.size(); ++local) {
    result[induced.local_to_global[local].value()] = oracle.distances[local];
  }
  return result;
}

[[nodiscard]] const bfnew::CompactTargetPath& target_for(
    const bfnew::CompactQueryResult& result,
    const bfnew::VertexId target) {
  const auto position = std::lower_bound(
      result.targets.begin(),
      result.targets.end(),
      target,
      [](const bfnew::CompactTargetPath& path, const bfnew::VertexId id) {
        return path.summary.target < id;
      });
  if (position == result.targets.end() || position->summary.target != target) {
    throw std::logic_error{"compact target result is missing"};
  }
  return *position;
}

[[nodiscard]] const bfnew::CompactQueryResult& result_for(
    const std::span<const bfnew::CompactQueryResult> results,
    const bfnew::QueryId query_id) {
  const auto position = std::lower_bound(
      results.begin(),
      results.end(),
      query_id,
      [](const bfnew::CompactQueryResult& result, const bfnew::QueryId id) {
        return result.query_id < id;
      });
  if (position == results.end() || position->query_id != query_id) {
    throw std::logic_error{"compact query result is missing"};
  }
  return *position;
}

void expect_valid(
    const bfnew::WeightedGraph& graph,
    const bfnew::RouteQuery& query,
    const bfnew::CompactQueryResult& result,
    const std::span<const float> distances,
    const std::string& prefix) {
  expect(
      bfnew::validate_compact_query_result(graph, query, result).ok(),
      prefix + ": compact source/target/cost/region validation passes");
  expect(
      bfnew::validate_compact_query_result_against_distances(
          graph, query, result, distances)
          .ok(),
      prefix + ": sampled exact GPU-label tightness validation passes");
}

void test_phase5_semantics_and_compact_summaries() {
  const bfnew::test::CompactPathFixture fixture =
      bfnew::test::make_compact_path_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::span<const bfnew::VertexId> map = graph.old_to_new();

  const std::vector<float> cycle_distances =
      induced_global_distances(graph, fixture.cycle_query);
  const bfnew::CompactQueryResult cycle =
      bfnew::reconstruct_compact_query_paths(
          graph,
          fixture.cycle_query,
          cycle_distances,
          bfnew::ExpansionQueryDisposition::reached);
  expect_valid(
      graph, fixture.cycle_query, cycle, cycle_distances, "zero-cycle query");
  const bfnew::CompactTargetPath& cycle_path = cycle.targets.front();
  expect(
      cycle_path.summary.reached ==
              bfnew::CompactTargetReachStatus::reached &&
          cycle_path.summary.reconstruction ==
              bfnew::CompactPathStatus::complete &&
          cycle_path.summary.has_selected_source == 1U &&
          cycle_path.summary.selected_source == map[3U] &&
          cycle_path.summary.path_length == 2U &&
          cycle_path.vertices ==
              std::vector<bfnew::VertexId>{map[3U], map[1U], map[2U]} &&
          cycle_path.distance_labels ==
              std::vector<float>{0.0F, 1.0F, 2.0F} &&
          cycle_path.edge_ids.size() == 2U &&
          bitwise_equal(cycle_path.summary.distance, 2.0F),
      "stable incoming traversal backtracks out of a zero-weight cycle");

  const std::vector<float> tie_distances =
      induced_global_distances(graph, fixture.tie_query);
  const bfnew::CompactQueryResult tie =
      bfnew::reconstruct_compact_query_paths(
          graph,
          fixture.tie_query,
          tie_distances,
          bfnew::ExpansionQueryDisposition::reached);
  expect_valid(graph, fixture.tie_query, tie, tie_distances, "equal-path query");
  expect(
      tie.targets.front().vertices ==
              std::vector<bfnew::VertexId>{map[4U], map[5U], map[7U]} &&
          tie.targets.front().summary.path_length == 2U &&
          bitwise_equal(tie.targets.front().summary.distance, 1.0F),
      "equal-cost alternatives choose the smaller stable incoming EdgeId");
  const bfnew::OutgoingCsrView outgoing = graph.outgoing();
  std::vector<bfnew::EdgeId> parallel_ids;
  const bfnew::EdgeOffset parallel_begin =
      outgoing.row_offsets[map[5U].value()];
  const bfnew::EdgeOffset parallel_end =
      outgoing.row_offsets[map[5U].value() + 1U];
  for (bfnew::EdgeOffset position = parallel_begin;
       position < parallel_end;
       ++position) {
    if (outgoing.destinations[position] == map[7U] &&
        bitwise_equal(outgoing.weights[position], 0.5F)) {
      parallel_ids.push_back(outgoing.edge_ids[position]);
    }
  }
  expect(
      parallel_ids.size() == 2U && tie.targets.front().edge_ids.back() ==
                                         *std::ranges::min_element(parallel_ids),
      "equal-cost parallel edges choose the smaller stable EdgeId");

  const std::vector<float> bounded_distances =
      induced_global_distances(graph, fixture.bounded_query);
  const bfnew::CompactQueryResult bounded =
      bfnew::reconstruct_compact_query_paths(
          graph,
          fixture.bounded_query,
          bounded_distances,
          bfnew::ExpansionQueryDisposition::reached);
  expect_valid(
      graph,
      fixture.bounded_query,
      bounded,
      bounded_distances,
      "bounded-region query");
  const bfnew::SsspResult unbounded =
      bfnew::dijkstra_oracle(graph, fixture.bounded_query.sources);
  expect(
      bounded.targets.front().vertices ==
              std::vector<bfnew::VertexId>{map[8U], map[10U], map[9U]} &&
          bitwise_equal(bounded.targets.front().summary.distance, 4.0F) &&
          bitwise_equal(unbounded.distances[map[9U].value()], 1.0F),
      "bounded compact path remains valid when an excluded global path is cheaper");
}

void test_multisource_duplicate_terminals_and_failures() {
  const bfnew::test::CompactPathFixture fixture =
      bfnew::test::make_compact_path_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::span<const bfnew::VertexId> map = graph.old_to_new();

  const std::vector<float> distances =
      induced_global_distances(graph, fixture.multisource_query);
  const bfnew::CompactQueryResult result =
      bfnew::reconstruct_compact_query_paths(
          graph,
          fixture.multisource_query,
          distances,
          bfnew::ExpansionQueryDisposition::reached);
  expect_valid(
      graph,
      fixture.multisource_query,
      result,
      distances,
      "multi-source query");
  expect(
      result.target_terminal_to_target ==
          fixture.multisource_query.target_terminal_to_target &&
          result.targets.size() == fixture.multisource_query.targets.size(),
      "canonical compact targets preserve duplicate input-terminal mapping");

  const bfnew::CompactTargetPath& source_target = target_for(result, map[12U]);
  expect(
      source_target.summary.selected_source == map[12U] &&
          source_target.summary.has_selected_source == 1U &&
          source_target.summary.path_length == 0U &&
          source_target.vertices == std::vector<bfnew::VertexId>{map[12U]} &&
          source_target.distance_labels == std::vector<float>{0.0F} &&
          source_target.edge_ids.empty() &&
          bitwise_equal(source_target.summary.distance, 0.0F),
      "a target that is already a source returns a zero-edge compact path");

  const bfnew::CompactTargetPath& routed_target = target_for(result, map[14U]);
  expect(
      routed_target.summary.selected_source == map[13U] &&
          routed_target.summary.has_selected_source == 1U &&
          routed_target.summary.path_length == 1U &&
          routed_target.vertices ==
              std::vector<bfnew::VertexId>{map[13U], map[14U]} &&
          routed_target.distance_labels == std::vector<float>{0.0F, 1.0F} &&
          bitwise_equal(routed_target.summary.distance, 1.0F),
      "each compact target reports the source selected by its tight path");

  const std::vector<float> unreachable_distances =
      induced_global_distances(graph, fixture.unreachable_query);
  const bfnew::CompactQueryResult unreachable =
      bfnew::reconstruct_compact_query_paths(
          graph,
          fixture.unreachable_query,
          unreachable_distances,
          bfnew::ExpansionQueryDisposition::unreachable_in_full_region);
  expect_valid(
      graph,
      fixture.unreachable_query,
      unreachable,
      unreachable_distances,
      "unreachable query");
  expect(
      unreachable.targets.front().summary.reached ==
              bfnew::CompactTargetReachStatus::not_reached &&
          unreachable.targets.front().summary.reconstruction ==
              bfnew::CompactPathStatus::unreachable &&
          unreachable.targets.front().summary.has_selected_source == 0U &&
          !std::isfinite(unreachable.targets.front().summary.distance) &&
          unreachable.targets.front().vertices.empty() &&
          unreachable.targets.front().distance_labels.empty() &&
          unreachable.targets.front().edge_ids.empty(),
      "clean full-region miss publishes an explicit unreachable target summary");

  const bfnew::CompactQueryResult failed =
      bfnew::make_failed_compact_query_result(
          fixture.unreachable_query,
          bfnew::ExpansionQueryDisposition::engine_failure);
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.unreachable_query, failed)
          .ok() &&
          failed.targets.front().summary.reconstruction ==
              bfnew::CompactPathStatus::query_terminal_failure &&
          failed.targets.front().summary.has_selected_source == 0U,
      "engine failure has an explicit target result without invented labels");

  expect_throws(
      [&] {
        static_cast<void>(bfnew::make_failed_compact_query_result(
            fixture.tie_query,
            bfnew::ExpansionQueryDisposition::reached));
      },
      "a reached query cannot be converted to a label-free failure result");
  expect_throws(
      [&] {
        static_cast<void>(bfnew::reconstruct_compact_query_paths(
            graph,
            fixture.tie_query,
            std::span<const float>{distances.data(), 1U},
            bfnew::ExpansionQueryDisposition::reached));
      },
      "compact reconstruction rejects a malformed distance-image shape");
}

[[nodiscard]] bfnew::BatchedExpansionOptions expansion_options(
    const bfnew::EngineKind engine) {
  bfnew::BatchedExpansionOptions options;
  options.run_options.engine = engine;
  options.run_options.control_mode = bfnew::ControlMode::per_round_host_poll;
  options.run_options.rounds_per_chunk = 1U;
  options.run_options.block_size = 128U;
  options.run_options.maximum_rounds = 64U;
  options.run_options.enable_per_lane_convergence = 1U;
  options.planner_policy.lane_width = 8U;
  options.planner_policy.minimum_jaccard_numerator = 0U;
  options.planner_policy.minimum_jaccard_denominator = 1U;
  options.planner_policy.maximum_union_inflation_numerator = 32U;
  options.planner_policy.maximum_union_inflation_denominator = 1U;
  options.execution_configuration_fingerprint = 0x1800'0000'0000'0001ULL;
  options.schedule = bfnew::one_ring_expansion();
  options.maximum_expansions = 4U;
  options.terminal_policy =
      bfnew::ExpansionTerminalPolicy::full_region_fallback;
  options.enable_compact_paths = 1U;
  return options;
}

void test_expansion_integration_all_engines() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::array engines{
      bfnew::EngineKind::jacobi_pull,
      bfnew::EngineKind::dense_chaotic_push,
      bfnew::EngineKind::frontier_push,
  };
  const std::array<std::uint32_t, 4U> lane_widths{1U, 8U, 16U, 32U};
  std::vector<bfnew::CompactQueryResult> reference;

  for (const bfnew::EngineKind engine : engines) {
    for (const std::uint32_t lane_width : lane_widths) {
      bfnew::BatchedExpansionOptions options = expansion_options(engine);
      options.planner_policy.lane_width = lane_width;
      options.execution_configuration_fingerprint ^=
          static_cast<std::uint64_t>(lane_width) << 32U;
      const std::string prefix =
          "portable engine " +
          std::to_string(static_cast<std::uint32_t>(engine)) + " width " +
          std::to_string(lane_width);
      bfnew::BatchedExpansionRunResult run =
          bfnew::run_host_batched_expansion(
              graph,
              fixture.directory,
              fixture.tile_runs,
              fixture.device_graph,
              fixture.queries,
              options);
      if (lane_width > fixture.queries.size()) {
        const bfnew::LaneMask expected_valid =
            (bfnew::LaneMask{1U} << fixture.queries.size()) - 1U;
        expect(
            !run.trace.empty() &&
                run.trace.front().valid_lane_mask == expected_valid,
            prefix +
                ": padded lanes stay invalid while every valid lane is compacted");
      }
      std::vector<bfnew::CompactPathPayload> payloads;
      for (const bfnew::ExpansionQueryOutcome& outcome : run.queries) {
        if (outcome.reached()) {
          expect(
              outcome.compact_paths.has_value() &&
                  outcome.compact_paths->query_id ==
                      outcome.final_query.query_id &&
                  outcome.compact_paths->expansion_generation ==
                      outcome.final_query.expansion_generation,
              "compact payload is captured only from the reached terminal generation");
          if (outcome.compact_paths) {
            payloads.push_back(*outcome.compact_paths);
          }
        } else if (
            outcome.disposition ==
            bfnew::ExpansionQueryDisposition::unreachable_in_full_region) {
          expect(
              outcome.compact_paths.has_value() &&
                  std::ranges::any_of(
                      outcome.compact_paths->targets,
                      [](const bfnew::CompactTargetPath& target) {
                        return target.summary.reconstruction ==
                               bfnew::CompactPathStatus::unreachable;
                      }),
              "a terminal full-region miss retains its classified compact "
              "target summaries");
          if (outcome.compact_paths) {
            payloads.push_back(*outcome.compact_paths);
          }
        } else {
          expect(
              !outcome.compact_paths.has_value(),
              "non-reachability terminal failures publish no compact payload");
        }
      }

      const bfnew::CompactTransferAccounting transfer =
          bfnew::measure_compact_transfer(payloads);
      std::uint64_t target_count = 0U;
      std::uint64_t vertex_count = 0U;
      std::uint64_t edge_count = 0U;
      for (const bfnew::CompactPathPayload& payload : payloads) {
        target_count += payload.targets.size();
        for (const bfnew::CompactTargetPath& target : payload.targets) {
          vertex_count += target.vertices.size();
          edge_count += target.edge_ids.size();
        }
      }
      expect(
          transfer.summary_bytes ==
                  target_count * sizeof(bfnew::CompactTargetSummary) &&
              transfer.vertex_bytes == vertex_count * sizeof(bfnew::VertexId) &&
              transfer.distance_label_bytes == vertex_count * sizeof(float) &&
              transfer.edge_id_bytes == edge_count * sizeof(std::uint32_t) &&
              transfer.total_bytes == transfer.summary_bytes +
                                          transfer.vertex_bytes +
                                          transfer.distance_label_bytes +
                                          transfer.edge_id_bytes,
          "compact transfer accounting contains summaries and exact path arenas only");

      const std::vector<bfnew::CompactQueryResult> compact =
          bfnew::extract_host_compact_paths(graph, run.queries);
      expect(
          compact.size() == fixture.queries.size() &&
              std::ranges::all_of(
                run.queries,
                [](const bfnew::ExpansionQueryOutcome& outcome) {
                  return outcome.final_distances.empty() &&
                         outcome.final_distances.capacity() == 0U;
                }),
          prefix +
              ": extraction returns one compact result and releases every full image");

      for (std::size_t index = 0U; index < compact.size(); ++index) {
        const bfnew::ExpansionQueryOutcome& outcome = run.queries[index];
        expect(
            compact[index].query_id == outcome.final_query.query_id &&
                compact[index].expansion_generation ==
                    outcome.final_query.expansion_generation &&
                compact[index].disposition == outcome.disposition,
            prefix + ": terminal QueryId, generation, and disposition survive");
        expect(
            bfnew::validate_compact_query_result(
                graph, outcome.final_query, compact[index])
                .ok(),
            prefix + " query " +
                std::to_string(compact[index].query_id.value()) +
                ": compact-label validation passes without retaining a V-sized "
                "diagnostic image");
        if (outcome.reached()) {
          expect(
              std::ranges::all_of(
                  compact[index].targets,
                  [](const bfnew::CompactTargetPath& target) {
                    return target.summary.reached ==
                               bfnew::CompactTargetReachStatus::reached &&
                           target.summary.reconstruction ==
                               bfnew::CompactPathStatus::complete &&
                           target.vertices.size() ==
                               static_cast<std::size_t>(
                                   target.summary.path_length) +
                                   1U &&
                           target.distance_labels.size() ==
                               target.vertices.size() &&
                           target.edge_ids.size() ==
                               target.summary.path_length;
                  }),
              prefix + ": every reached query returns complete compact paths");
        }
      }

      const bfnew::CompactQueryResult& long_path = result_for(
          compact, bfnew::test::phase17_long_edge_query_id);
      const bfnew::CompactQueryResult& spill_path = result_for(
          compact, bfnew::test::phase17_spill_query_id);
      const bfnew::CompactQueryResult& multi_path = result_for(
          compact, bfnew::test::phase17_multi_source_query_id);
      const bfnew::CompactQueryResult& unreachable = result_for(
          compact, bfnew::test::phase17_unreachable_query_id);
      expect(
          long_path.targets.front().summary.path_length == 2U,
          prefix + ": long-edge path retains its two-edge reconstruction");
      expect(
          spill_path.targets.front().summary.path_length == 2U,
          prefix + ": spill path retains its two-edge reconstruction");
      expect(
          multi_path.targets.front().summary.path_length == 3U,
          prefix + ": multi-source path retains its three-edge reconstruction");
      expect(
          unreachable.disposition ==
                  bfnew::ExpansionQueryDisposition::unreachable_in_full_region &&
              unreachable.targets.front().summary.reconstruction ==
                  bfnew::CompactPathStatus::unreachable,
          prefix +
              ": full-region disposition and unreachable target summary remain explicit");

      if (reference.empty()) {
        reference = compact;
      } else {
        expect(
            compact == reference,
            "all engines and W=1/8/16/32 produce identical compact target paths");
      }
    }
  }
}

void test_reached_status_with_unreachable_payload_fails_closed() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  const bfnew::RouteQuery query = bfnew::test::expansion_query(
      fixture, bfnew::test::phase17_immediate_query_id);
  const bfnew::BatchedExpansionOptions options =
      expansion_options(bfnew::EngineKind::jacobi_pull);
  const bfnew::ExpansionBatchRunner malformed_runner =
      [&](const std::span<const bfnew::RouteQuery> current_queries,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        bfnew::ExpansionBatchExecution execution;
        execution.result.engine_kind =
            static_cast<std::uint32_t>(options.run_options.engine);
        execution.result.control_mode =
            static_cast<std::uint32_t>(options.run_options.control_mode);
        execution.result.status.valid_lane_mask = batch.valid_lane_mask;
        execution.result.status.converged_lane_mask = batch.valid_lane_mask;
        execution.result.status.reached_target_mask = batch.valid_lane_mask;
        execution.result.status.converged = 1U;
        execution.result.status.rounds_completed = 1U;
        execution.result.status.stop_reason = static_cast<std::uint32_t>(
            bfnew::DeviceStopReason::converged);
        execution.work_evidence = bfnew::ExpansionWorkEvidence::measured;
        execution.compact_execution.host_timing =
            bfnew::CompactStageTimingEvidence::measured;

        bfnew::CompactPathPayload unreachable_payload;
        unreachable_payload.query_id = current_queries.front().query_id;
        unreachable_payload.expansion_generation =
            current_queries.front().expansion_generation;
        unreachable_payload.target_terminal_to_target =
            current_queries.front().target_terminal_to_target;
        for (const bfnew::VertexId target : current_queries.front().targets) {
          bfnew::CompactTargetPath path;
          path.summary.target = target;
          path.summary.reconstruction = bfnew::CompactPathStatus::unreachable;
          unreachable_payload.targets.push_back(std::move(path));
        }
        execution.compact_paths.push_back(std::move(unreachable_payload));
        execution.compact_transfer =
            bfnew::measure_compact_transfer(execution.compact_paths);
        return execution;
      };

  const std::array queries{query};
  const bfnew::BatchedExpansionRunResult run =
      bfnew::run_batched_expansion(
          fixture.partitioned.graph,
          fixture.directory,
          fixture.tile_runs,
          queries,
          options,
          malformed_runner);
  expect(
      run.queries.size() == 1U &&
          run.queries.front().disposition ==
              bfnew::ExpansionQueryDisposition::engine_failure &&
          !run.queries.front().reached() &&
          run.metrics.engine_failure_queries == 1U &&
          !run.queries.front().compact_paths.has_value(),
      "a clean reached status cannot promote an unreachable-only compact "
      "payload and is normalized to a label-free explicit engine failure");
}

void test_miss_status_requires_consistent_target_classification() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const bfnew::RouteQuery query = bfnew::test::expansion_query(
      fixture, bfnew::test::phase17_immediate_query_id);
  const std::array queries{query};
  const bfnew::BatchedExpansionOptions options =
      expansion_options(bfnew::EngineKind::jacobi_pull);

  const auto run_case = [&](const bool explicit_reconstruction_failure) {
    const bfnew::ExpansionBatchRunner runner =
        [&](const std::span<const bfnew::RouteQuery> current_queries,
            const std::span<const bfnew::BatchQueryFeatures>,
            const bfnew::BatchPlanEntry& batch,
            const bfnew::ExpansionBatchContext&) {
          bfnew::ExpansionBatchExecution execution;
          execution.result.engine_kind =
              static_cast<std::uint32_t>(options.run_options.engine);
          execution.result.control_mode =
              static_cast<std::uint32_t>(options.run_options.control_mode);
          execution.result.status.valid_lane_mask = batch.valid_lane_mask;
          execution.result.status.converged_lane_mask = batch.valid_lane_mask;
          execution.result.status.bounding_box_miss_mask =
              batch.valid_lane_mask;
          execution.result.status.converged = 1U;
          execution.result.status.rounds_completed = 1U;
          execution.result.status.stop_reason = static_cast<std::uint32_t>(
              bfnew::DeviceStopReason::converged);
          execution.work_evidence = bfnew::ExpansionWorkEvidence::measured;
          execution.compact_execution.host_timing =
              bfnew::CompactStageTimingEvidence::measured;

          const std::vector<float> distances =
              induced_global_distances(graph, current_queries.front());
          bfnew::CompactPathPayload payload =
              bfnew::reconstruct_compact_path_payload(
                  graph, current_queries.front(), distances);
          if (explicit_reconstruction_failure) {
            bfnew::CompactTargetPath& target = payload.targets.front();
            target.summary.reconstruction =
                bfnew::CompactPathStatus::no_tight_path;
            target.summary.has_selected_source = 0U;
            target.summary.path_length = 0U;
            target.vertices.clear();
            target.distance_labels.clear();
            target.edge_ids.clear();
          }
          execution.compact_paths.push_back(std::move(payload));
          execution.compact_transfer =
              bfnew::measure_compact_transfer(execution.compact_paths);
          return execution;
        };
    return bfnew::run_batched_expansion(
        graph,
        fixture.directory,
        fixture.tile_runs,
        queries,
        options,
        runner);
  };

  bfnew::BatchedExpansionRunResult all_complete_miss = run_case(false);
  expect(
      all_complete_miss.queries.size() == 1U &&
          all_complete_miss.queries.front().disposition ==
              bfnew::ExpansionQueryDisposition::engine_failure &&
          !all_complete_miss.queries.front().compact_paths.has_value() &&
          all_complete_miss.metrics.engine_failure_queries == 1U,
      "a lane-level miss with only complete targets becomes an explicit "
      "label-free engine failure instead of being retried");

  bfnew::BatchedExpansionRunResult failed_reconstruction = run_case(true);
  expect(
      failed_reconstruction.queries.size() == 1U &&
          failed_reconstruction.queries.front().disposition ==
              bfnew::ExpansionQueryDisposition::engine_failure &&
          failed_reconstruction.queries.front().compact_paths.has_value() &&
          failed_reconstruction.metrics.engine_failure_queries == 1U,
      "a lane-level miss with explicit reconstruction failure is terminal and "
      "is never silently expanded");
  const std::vector<bfnew::CompactQueryResult> compact =
      bfnew::extract_host_compact_paths(
          graph, failed_reconstruction.queries);
  expect(
      compact.size() == 1U &&
          compact.front().disposition ==
              bfnew::ExpansionQueryDisposition::engine_failure &&
          compact.front().targets.front().summary.reconstruction ==
              bfnew::CompactPathStatus::no_tight_path,
      "explicit miss-lane reconstruction failure survives compaction as an "
      "engine-failure result");
}

enum class CompactLifecycleCorruption : std::uint8_t {
  none,
  missing_lane,
  wrong_generation,
  duplicate_lane,
  padded_lane,
  off_by_one_transfer,
  controller_poll_byte_mismatch,
  controller_poll_overall_mismatch,
  zero_host_poll_count,
};

void test_compact_payload_lifecycle_rejections() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const bfnew::RouteQuery query = bfnew::test::expansion_query(
      fixture, bfnew::test::phase17_immediate_query_id);
  const std::array queries{query};
  bfnew::BatchedExpansionOptions options =
      expansion_options(bfnew::EngineKind::jacobi_pull);
  options.planner_policy.lane_width = 8U;

  const auto runner_for =
      [&](const CompactLifecycleCorruption corruption) {
        return bfnew::ExpansionBatchRunner{
            [&, corruption](
                const std::span<const bfnew::RouteQuery> current_queries,
                const std::span<const bfnew::BatchQueryFeatures>,
                const bfnew::BatchPlanEntry& batch,
                const bfnew::ExpansionBatchContext&) {
              bfnew::ExpansionBatchExecution execution;
              execution.result.engine_kind =
                  static_cast<std::uint32_t>(options.run_options.engine);
              execution.result.control_mode =
                  static_cast<std::uint32_t>(options.run_options.control_mode);
              execution.result.status.valid_lane_mask = batch.valid_lane_mask;
              execution.result.status.converged_lane_mask =
                  batch.valid_lane_mask;
              execution.result.status.reached_target_mask =
                  batch.valid_lane_mask;
              execution.result.status.converged = 1U;
              execution.result.status.rounds_completed = 1U;
              execution.result.status.stop_reason = static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::converged);
              execution.work_evidence = bfnew::ExpansionWorkEvidence::measured;
              execution.compact_execution.host_timing =
                  bfnew::CompactStageTimingEvidence::measured;

              for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
                if ((batch.valid_lane_mask & (bfnew::LaneMask{1U} << lane)) ==
                    0U) {
                  continue;
                }
                const auto current = std::ranges::find_if(
                    current_queries,
                    [&](const bfnew::RouteQuery& candidate) {
                      return candidate.query_id ==
                             batch.query_ids_by_lane[lane];
                    });
                if (current == current_queries.end()) {
                  throw std::logic_error{
                      "lifecycle fixture lost a valid-lane query"};
                }
                const std::vector<float> distances =
                    induced_global_distances(graph, *current);
                execution.compact_paths.push_back(
                    bfnew::reconstruct_compact_path_payload(
                        graph, *current, distances));
              }

              std::ranges::sort(
                  execution.compact_paths,
                  {},
                  &bfnew::CompactPathPayload::query_id);
              switch (corruption) {
                case CompactLifecycleCorruption::none:
                  break;
                case CompactLifecycleCorruption::missing_lane:
                  execution.compact_paths.erase(
                      execution.compact_paths.begin());
                  break;
                case CompactLifecycleCorruption::wrong_generation:
                  ++execution.compact_paths.front().expansion_generation;
                  break;
                case CompactLifecycleCorruption::duplicate_lane:
                  execution.compact_paths.push_back(
                      execution.compact_paths.front());
                  break;
                case CompactLifecycleCorruption::padded_lane: {
                  expect(
                      batch.lane_width == 8U &&
                          std::popcount(batch.valid_lane_mask) == 1 &&
                          batch.query_ids_by_lane[1U] ==
                              bfnew::invalid_batch_query_id,
                      "lifecycle fixture exposes an explicit padded lane");
                  bfnew::CompactPathPayload padded =
                      execution.compact_paths.front();
                  padded.query_id = batch.query_ids_by_lane[1U];
                  padded.expansion_generation =
                      batch.expansion_generations_by_lane[1U];
                  execution.compact_paths.push_back(std::move(padded));
                  break;
                }
                case CompactLifecycleCorruption::off_by_one_transfer:
                case CompactLifecycleCorruption::controller_poll_byte_mismatch:
                case CompactLifecycleCorruption::controller_poll_overall_mismatch:
                case CompactLifecycleCorruption::zero_host_poll_count:
                  break;
              }
              std::ranges::sort(
                  execution.compact_paths,
                  {},
                  &bfnew::CompactPathPayload::query_id);
              execution.compact_transfer =
                  bfnew::measure_compact_transfer(execution.compact_paths);
              if (corruption ==
                  CompactLifecycleCorruption::off_by_one_transfer) {
                ++execution.compact_transfer.vertex_bytes;
                ++execution.compact_transfer.total_bytes;
              }
              execution.compact_status_bytes = sizeof(bfnew::DeviceRunStatus);
              execution.compact_error_bytes = 2U * sizeof(std::uint32_t);
              execution.compact_total_device_to_host_bytes =
                  execution.compact_transfer.total_bytes +
                  execution.compact_status_bytes +
                  execution.compact_error_bytes;
              execution.compact_controller_poll_count = 3U;
              execution.compact_controller_poll_bytes =
                  execution.compact_controller_poll_count *
                  sizeof(bfnew::DeviceController);
              execution.compact_overall_device_to_host_bytes =
                  execution.compact_total_device_to_host_bytes +
                  execution.compact_controller_poll_bytes;
              if (corruption ==
                  CompactLifecycleCorruption::controller_poll_byte_mismatch) {
                ++execution.compact_controller_poll_bytes;
              } else if (
                  corruption == CompactLifecycleCorruption::
                                    controller_poll_overall_mismatch) {
                ++execution.compact_overall_device_to_host_bytes;
              } else if (
                  corruption ==
                  CompactLifecycleCorruption::zero_host_poll_count) {
                execution.compact_controller_poll_count = 0U;
                execution.compact_controller_poll_bytes = 0U;
                execution.compact_overall_device_to_host_bytes =
                    execution.compact_total_device_to_host_bytes;
              }
              return execution;
            }};
      };

  {
    const bfnew::ExpansionBatchRunner runner =
        runner_for(CompactLifecycleCorruption::none);
    const bfnew::BatchedExpansionRunResult clean =
        bfnew::run_batched_expansion(
            graph,
            fixture.directory,
            fixture.tile_runs,
            queries,
            options,
            runner);
    expect(
        clean.queries.size() == 1U && clean.queries.front().reached() &&
            clean.queries.front().compact_paths.has_value() &&
            clean.metrics.compact_controller_poll_count == 3U &&
            clean.metrics.compact_controller_poll_bytes ==
                3U * sizeof(bfnew::DeviceController) &&
            clean.metrics.compact_overall_device_to_host_bytes ==
                clean.metrics.compact_total_device_to_host_bytes +
                    clean.metrics.compact_controller_poll_bytes,
        "lifecycle control runner publishes one valid generation-bound "
        "payload with an exact controller-poll transfer envelope");
  }

  const auto expect_rejected =
      [&](const CompactLifecycleCorruption corruption,
          const std::string_view description) {
        expect_throws(
            [&] {
              const bfnew::ExpansionBatchRunner runner =
                  runner_for(corruption);
              static_cast<void>(bfnew::run_batched_expansion(
                  graph,
                  fixture.directory,
                  fixture.tile_runs,
                  queries,
                  options,
                  runner));
            },
            description);
      };
  expect_rejected(
      CompactLifecycleCorruption::missing_lane,
      "Phase 17 orchestration rejects a missing valid-lane compact payload");
  expect_rejected(
      CompactLifecycleCorruption::wrong_generation,
      "Phase 17 orchestration rejects a compact payload from the wrong "
      "generation");
  expect_rejected(
      CompactLifecycleCorruption::duplicate_lane,
      "Phase 17 orchestration rejects duplicate valid-lane compact payloads");
  expect_rejected(
      CompactLifecycleCorruption::padded_lane,
      "Phase 17 orchestration rejects a payload attributed to a padded lane");
  expect_rejected(
      CompactLifecycleCorruption::off_by_one_transfer,
      "Phase 17 orchestration rejects internally consistent transfer bytes "
      "that are one byte above the exact compact arenas");
  expect_rejected(
      CompactLifecycleCorruption::controller_poll_byte_mismatch,
      "compact transport rejects controller bytes that disagree with the "
      "exact DeviceController poll count");
  expect_rejected(
      CompactLifecycleCorruption::controller_poll_overall_mismatch,
      "compact transport rejects an overall D2H total that disagrees with "
      "compact-result and controller-poll subtotals");
  expect_rejected(
      CompactLifecycleCorruption::zero_host_poll_count,
      "host-polled compact execution rejects a zero controller-poll count");
}

void test_terminal_full_region_retains_mixed_target_payload() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  const std::span<const bfnew::VertexId> map =
      fixture.partitioned.graph.old_to_new();
  const std::array sources{map[0U]};
  const std::array targets{map[2U], map[14U]};
  const bfnew::RouteQuery mixed = bfnew::make_route_query(
      bfnew::QueryId{1898U}, fixture.partitioned.graph, sources, targets);
  const std::array queries{mixed};
  bfnew::BatchedExpansionOptions options =
      expansion_options(bfnew::EngineKind::jacobi_pull);
  options.planner_policy.lane_width = 1U;

  bfnew::BatchedExpansionRunResult run = bfnew::run_host_batched_expansion(
      fixture.partitioned.graph,
      fixture.directory,
      fixture.tile_runs,
      fixture.device_graph,
      queries,
      options);
  const bfnew::ExpansionQueryOutcome& outcome = run.queries.front();
  std::size_t complete_count = 0U;
  std::size_t unreachable_count = 0U;
  if (outcome.compact_paths) {
    complete_count = static_cast<std::size_t>(std::ranges::count_if(
        outcome.compact_paths->targets,
        [](const bfnew::CompactTargetPath& target) {
          return target.summary.reconstruction ==
                 bfnew::CompactPathStatus::complete;
        }));
    unreachable_count = static_cast<std::size_t>(std::ranges::count_if(
        outcome.compact_paths->targets,
        [](const bfnew::CompactTargetPath& target) {
          return target.summary.reconstruction ==
                 bfnew::CompactPathStatus::unreachable;
        }));
  }
  expect(
      outcome.disposition ==
              bfnew::ExpansionQueryDisposition::unreachable_in_full_region &&
          outcome.compact_paths.has_value() && complete_count == 1U &&
          unreachable_count == 1U,
      "terminal full-region miss retains its mixed complete/unreachable target "
      "payload");

  const std::vector<bfnew::CompactQueryResult> compact =
      bfnew::extract_host_compact_paths(fixture.partitioned.graph, run.queries);
  expect(
      compact.size() == 1U &&
          bfnew::validate_compact_query_result(
              fixture.partitioned.graph,
              run.queries.front().final_query,
              compact.front())
              .ok(),
      "mixed terminal payload remains independently valid after extraction");
  const bfnew::CompactPathQualitySample quality =
      bfnew::sample_compact_path_quality(
          fixture.partitioned.graph,
          run.queries,
          compact,
          1U,
          0x1898'0000'0000'0001ULL);
  expect(
      quality.sampled_query_count == 1U &&
          quality.finite_target_pairs == 1U &&
          quality.observations.size() == 1U &&
          quality.observations.front().target == map[2U] &&
          quality.observations.front().cost_inflation_ratio == 1.0 &&
          quality.observations.front().path_length_inflation_ratio == 1.0,
      "quality sampling includes complete targets from a terminal mixed "
      "complete/unreachable full-region result");
}

void test_validation_fail_closed_and_determinism() {
  const bfnew::test::CompactPathFixture fixture =
      bfnew::test::make_compact_path_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::span<const bfnew::VertexId> map = graph.old_to_new();
  const std::vector<float> distances =
      induced_global_distances(graph, fixture.tie_query);
  const bfnew::CompactQueryResult valid =
      bfnew::reconstruct_compact_query_paths(
          graph,
          fixture.tie_query,
          distances,
          bfnew::ExpansionQueryDisposition::reached);
  const bfnew::CompactQueryResult repeated =
      bfnew::reconstruct_compact_query_paths(
          graph,
          fixture.tie_query,
          distances,
          bfnew::ExpansionQueryDisposition::reached);
  expect(valid == repeated, "compact reconstruction is byte-semantic deterministic");
  expect(
      bfnew::validate_compact_query_result(graph, fixture.tie_query, valid)
          .ok(),
      "disposition-matrix baseline accepts its complete reached payload");

  bfnew::CompactQueryResult corrupted = valid;
  corrupted.query_id = bfnew::QueryId{999U};
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code ==
          bfnew::CompactPathValidationErrorCode::query_identity_mismatch,
      "validator rejects a changed compact QueryId");

  corrupted = valid;
  ++corrupted.expansion_generation;
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code == bfnew::CompactPathValidationErrorCode::
                           expansion_generation_mismatch,
      "validator rejects a changed expansion generation");

  corrupted = valid;
  corrupted.target_terminal_to_target.front() ^= 1U;
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code ==
          bfnew::CompactPathValidationErrorCode::terminal_map_mismatch,
      "validator rejects a changed original-terminal map");

  corrupted = valid;
  corrupted.targets.front().summary.target = map[6U];
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code ==
          bfnew::CompactPathValidationErrorCode::target_identity_mismatch,
      "validator rejects a target identity outside canonical target order");

  const std::vector<float> multisource_distances =
      induced_global_distances(graph, fixture.multisource_query);
  bfnew::CompactQueryResult reordered_targets =
      bfnew::reconstruct_compact_query_paths(
          graph,
          fixture.multisource_query,
          multisource_distances,
          bfnew::ExpansionQueryDisposition::reached);
  std::swap(reordered_targets.targets[0U], reordered_targets.targets[1U]);
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.multisource_query, reordered_targets)
              .code ==
          bfnew::CompactPathValidationErrorCode::target_identity_mismatch,
      "validator rejects reordered canonical target results");

  corrupted = valid;
  corrupted.targets.front().summary.reached =
      static_cast<bfnew::CompactTargetReachStatus>(99U);
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code ==
          bfnew::CompactPathValidationErrorCode::invalid_reach_status,
      "validator rejects an unknown reach-status enum");

  corrupted = valid;
  corrupted.targets.front().summary.reconstruction =
      static_cast<bfnew::CompactPathStatus>(99U);
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code ==
          bfnew::CompactPathValidationErrorCode::invalid_reconstruction_status,
      "validator rejects an unknown reconstruction-status enum");

  corrupted = valid;
  corrupted.disposition = bfnew::ExpansionQueryDisposition::engine_failure;
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code == bfnew::CompactPathValidationErrorCode::
                           failed_query_reported_complete,
      "validator rejects a complete payload labeled as an engine failure");

  corrupted = valid;
  corrupted.disposition =
      bfnew::ExpansionQueryDisposition::unreachable_in_full_region;
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code == bfnew::CompactPathValidationErrorCode::
                           inconsistent_unreachable_result,
      "validator rejects a complete payload labeled as a full-region miss");

  corrupted = valid;
  corrupted.disposition = bfnew::ExpansionQueryDisposition::expansion_limit;
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code == bfnew::CompactPathValidationErrorCode::
                           inconsistent_failure_result,
      "validator rejects a complete payload labeled as an expansion limit");

  const std::vector<float> unreachable_distances =
      induced_global_distances(graph, fixture.unreachable_query);
  const bfnew::CompactQueryResult unreachable =
      bfnew::reconstruct_compact_query_paths(
          graph,
          fixture.unreachable_query,
          unreachable_distances,
          bfnew::ExpansionQueryDisposition::unreachable_in_full_region);
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.unreachable_query, unreachable)
          .ok(),
      "disposition-matrix baseline accepts its full-region unreachable "
      "payload");
  bfnew::CompactQueryResult mismatched_unreachable = unreachable;
  mismatched_unreachable.disposition =
      bfnew::ExpansionQueryDisposition::reached;
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.unreachable_query, mismatched_unreachable)
              .code == bfnew::CompactPathValidationErrorCode::
                           reached_query_incomplete,
      "validator rejects an unreachable target labeled as reached");

  const bfnew::CompactQueryResult terminal_failure =
      bfnew::make_failed_compact_query_result(
          fixture.unreachable_query,
          bfnew::ExpansionQueryDisposition::engine_failure);
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.unreachable_query, terminal_failure)
          .ok(),
      "disposition-matrix baseline accepts its explicit engine-failure "
      "payload");
  bfnew::CompactQueryResult mismatched_failure = terminal_failure;
  mismatched_failure.disposition = bfnew::ExpansionQueryDisposition::reached;
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.unreachable_query, mismatched_failure)
              .code == bfnew::CompactPathValidationErrorCode::
                           reached_query_incomplete,
      "validator rejects a terminal failure labeled as reached");

  mismatched_failure = terminal_failure;
  mismatched_failure.disposition =
      bfnew::ExpansionQueryDisposition::unreachable_in_full_region;
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.unreachable_query, mismatched_failure)
              .code == bfnew::CompactPathValidationErrorCode::
                           inconsistent_unreachable_result,
      "validator rejects a terminal failure labeled as a full-region miss");

  corrupted = valid;
  corrupted.disposition =
      static_cast<bfnew::ExpansionQueryDisposition>(99U);
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code == bfnew::CompactPathValidationErrorCode::
                           invalid_query_disposition,
      "validator rejects an unknown query-disposition enum");

  corrupted = valid;
  ++corrupted.targets.front().summary.path_length;
  expect(
      !bfnew::validate_compact_query_result(
           graph, fixture.tie_query, corrupted)
           .ok(),
      "validator rejects a path-length/arena shape disagreement");

  corrupted = valid;
  corrupted.targets.front().summary.selected_source =
      fixture.tie_query.targets.front();
  expect(
      !bfnew::validate_compact_query_result(
           graph, fixture.tie_query, corrupted)
           .ok(),
      "validator rejects a selected source outside the canonical source set");

  corrupted = valid;
  corrupted.targets.front().summary.has_selected_source = 2U;
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code ==
          bfnew::CompactPathValidationErrorCode::invalid_selected_source_flag,
      "validator rejects a non-Boolean selected-source validity word");

  corrupted = valid;
  corrupted.targets.front().vertices[1U] =
      corrupted.targets.front().vertices.front();
  expect(
      !bfnew::validate_compact_query_result(
           graph, fixture.tie_query, corrupted)
           .ok(),
      "validator rejects a repeated path vertex before accepting continuity");

  corrupted = valid;
  corrupted.targets.front().vertices[1U] = map[0U];
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code == bfnew::CompactPathValidationErrorCode::
                           vertex_outside_selected_region,
      "validator rejects a path vertex outside the final selected region");

  corrupted = valid;
  corrupted.targets.front().edge_ids.front() =
      bfnew::EdgeId{graph.edge_count()};
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code ==
          bfnew::CompactPathValidationErrorCode::edge_id_out_of_range,
      "validator rejects an out-of-range stable EdgeId");

  corrupted = valid;
  corrupted.targets.front().edge_ids.front() = bfnew::EdgeId{0U};
  expect(
      bfnew::validate_compact_query_result(
          graph, fixture.tie_query, corrupted)
              .code ==
          bfnew::CompactPathValidationErrorCode::edge_continuity_mismatch,
      "validator rejects an in-range EdgeId with the wrong endpoints");

  corrupted = valid;
  corrupted.targets.front().summary.distance = 2.0F;
  expect(
      !bfnew::validate_compact_query_result(
           graph, fixture.tie_query, corrupted)
           .ok(),
      "validator rejects an incorrect reported path cost");

  corrupted = valid;
  corrupted.targets.front().distance_labels[1U] = 0.75F;
  expect(
      !bfnew::validate_compact_query_result(
           graph, fixture.tie_query, corrupted)
           .ok(),
      "ordinary compact validation rejects a non-tight transferred path label");

  std::vector<float> corrupted_distances = distances;
  corrupted_distances[valid.targets.front().vertices[1U].value()] = 0.75F;
  expect(
      !bfnew::validate_compact_query_result_against_distances(
           graph, fixture.tie_query, valid, corrupted_distances)
           .ok(),
      "sampled diagnostic validator rejects a non-tight GPU label");

  bfnew::ExpansionQueryOutcome missing_image;
  missing_image.final_query = fixture.tie_query;
  missing_image.disposition = bfnew::ExpansionQueryDisposition::reached;
  expect_throws(
      [&] {
        static_cast<void>(bfnew::extract_host_compact_paths(
            graph,
            std::span<bfnew::ExpansionQueryOutcome>{&missing_image, 1U}));
      },
      "portable extraction rejects reached success without a final label image");
}

void test_no_congestion_stage_ledger() {
  using bfnew::PipelineStageTiming;
  using bfnew::PipelineTimingEvidence;

  PipelineStageTiming upload = bfnew::measured_host_stage(20U);
  upload.device_evidence = PipelineTimingEvidence::measured;
  upload.device_milliseconds = 0.25;
  PipelineStageTiming sssp = bfnew::measured_host_stage(40U);
  sssp.device_evidence = PipelineTimingEvidence::measured;
  sssp.device_milliseconds = 0.5;
  PipelineStageTiming reconstruction = bfnew::measured_host_stage(60U);
  reconstruction.device_evidence = PipelineTimingEvidence::measured;
  reconstruction.device_milliseconds = 0.75;
  PipelineStageTiming transfer = bfnew::measured_host_stage(70U);
  transfer.device_evidence = PipelineTimingEvidence::measured;
  transfer.device_milliseconds = 1.0;

  const bfnew::NoCongestionStageLedger ledger =
      bfnew::make_no_congestion_stage_ledger(
          bfnew::measured_host_stage(10U),
          upload,
          bfnew::measured_host_stage(30U),
          sssp,
          bfnew::measured_host_stage(50U),
          bfnew::measured_host_stage(55U),
          reconstruction,
          transfer,
          bfnew::measured_host_stage(305U),
          bfnew::measured_host_stage(400U));
  expect(
      bfnew::validate_no_congestion_stage_ledger(ledger) ==
              bfnew::NoCongestionStageLedgerError::none &&
          ledger.warm_all_query.host_evidence ==
              PipelineTimingEvidence::measured &&
          ledger.expansion.host_nanoseconds == 50U &&
          ledger.controller_orchestration.host_nanoseconds == 55U &&
          ledger.warm_all_query.host_nanoseconds == 305U &&
          ledger.cold_pipeline.host_evidence ==
              PipelineTimingEvidence::measured &&
          ledger.cold_execution.host_nanoseconds == 400U &&
          ledger.cold_pipeline.host_nanoseconds == 430U &&
          ledger.warm_all_query.device_evidence ==
              PipelineTimingEvidence::unavailable &&
          ledger.cold_pipeline.device_evidence ==
              PipelineTimingEvidence::unavailable,
      "stage ledger checks independent warm and capacity-growing cold host "
      "sums without inventing aggregate device time");

  const bfnew::NoCongestionStageLedger partially_unavailable =
      bfnew::make_no_congestion_stage_ledger(
          bfnew::measured_host_stage(1U),
          bfnew::measured_host_stage(2U),
          bfnew::measured_host_stage(3U),
          PipelineStageTiming{},
          bfnew::measured_host_stage(5U),
          bfnew::measured_host_stage(6U),
          bfnew::measured_host_stage(7U),
          bfnew::measured_host_stage(8U),
          bfnew::measured_host_stage(77U),
          bfnew::measured_host_stage(9U));
  expect(
      bfnew::validate_no_congestion_stage_ledger(partially_unavailable) ==
              bfnew::NoCongestionStageLedgerError::none &&
          partially_unavailable.warm_all_query.host_evidence ==
              PipelineTimingEvidence::measured &&
          partially_unavailable.warm_all_query.host_nanoseconds == 77U &&
          partially_unavailable.cold_pipeline.host_evidence ==
              PipelineTimingEvidence::measured &&
          partially_unavailable.cold_pipeline.host_nanoseconds == 12U,
      "an unavailable asynchronous host substage permits an independent "
      "enclosing warm observation and does not erase cold evidence");

  bfnew::NoCongestionStageLedger malformed_partial = partially_unavailable;
  malformed_partial.warm_all_query = bfnew::measured_host_stage(1U);
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed_partial) ==
          bfnew::NoCongestionStageLedgerError::warm_sum_mismatch,
      "measured warm components cannot exceed a partially observed enclosing "
      "warm interval");

  const bfnew::NoCongestionStageLedger warm_without_cold =
      bfnew::make_no_congestion_stage_ledger(
          bfnew::measured_host_stage(1U),
          bfnew::measured_host_stage(2U),
          bfnew::measured_host_stage(3U),
          bfnew::measured_host_stage(4U),
          bfnew::measured_host_stage(5U),
          bfnew::measured_host_stage(6U),
          bfnew::measured_host_stage(7U),
          bfnew::measured_host_stage(8U),
          bfnew::measured_host_stage(33U),
          PipelineStageTiming{});
  expect(
      warm_without_cold.warm_all_query.host_evidence ==
              PipelineTimingEvidence::measured &&
          warm_without_cold.cold_pipeline.host_evidence ==
              PipelineTimingEvidence::unavailable,
      "measured pre-grown warm stages do not fabricate a cold execution");

  bfnew::NoCongestionStageLedger malformed = ledger;
  malformed.batch_planning.host_evidence =
      static_cast<PipelineTimingEvidence>(99U);
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed) ==
          bfnew::NoCongestionStageLedgerError::invalid_evidence,
      "stage validator rejects an unknown timing-evidence enum");

  malformed = ledger;
  malformed.batch_planning.host_evidence = PipelineTimingEvidence::unavailable;
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed) ==
          bfnew::NoCongestionStageLedgerError::unavailable_host_has_value,
      "an unavailable host stage cannot carry a numeric duration");

  malformed = ledger;
  malformed.expansion.device_milliseconds = 0.125;
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed) ==
          bfnew::NoCongestionStageLedgerError::unavailable_device_has_value,
      "an unavailable device stage cannot carry a numeric duration");

  malformed = ledger;
  malformed.sssp.device_milliseconds =
      std::numeric_limits<double>::quiet_NaN();
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed) ==
          bfnew::NoCongestionStageLedgerError::invalid_device_duration,
      "measured device duration rejects NaN");

  malformed = ledger;
  malformed.batch_planning.device_evidence =
      PipelineTimingEvidence::measured;
  malformed.batch_planning.device_milliseconds = 0.125;
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed) ==
          bfnew::NoCongestionStageLedgerError::cpu_only_device_evidence,
      "CPU-only planning cannot publish an invented GPU-event duration");

  malformed = ledger;
  malformed.warm_all_query = {};
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed) ==
          bfnew::NoCongestionStageLedgerError::warm_evidence_mismatch,
      "measured warm components require a measured warm aggregate");

  malformed = ledger;
  malformed.cold_pipeline = {};
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed) ==
          bfnew::NoCongestionStageLedgerError::cold_evidence_mismatch,
      "measured load, upload, and cold execution require a measured cold "
      "aggregate");

  malformed = ledger;
  ++malformed.warm_all_query.host_nanoseconds;
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed) ==
          bfnew::NoCongestionStageLedgerError::warm_sum_mismatch,
      "warm total cannot disagree with its exact stage sum");

  malformed = ledger;
  malformed.cold_execution.host_evidence =
      PipelineTimingEvidence::unavailable;
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed) ==
          bfnew::NoCongestionStageLedgerError::unavailable_host_has_value,
      "an unavailable cold execution cannot carry a numeric duration");

  malformed = ledger;
  ++malformed.cold_pipeline.host_nanoseconds;
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed) ==
          bfnew::NoCongestionStageLedgerError::cold_sum_mismatch,
      "cold total cannot disagree with load, upload, and its distinct cold "
      "execution");

  malformed = ledger;
  malformed.batch_planning.host_nanoseconds =
      std::numeric_limits<std::uint64_t>::max();
  expect(
      bfnew::validate_no_congestion_stage_ledger(malformed) ==
          bfnew::NoCongestionStageLedgerError::warm_sum_overflow,
      "stage validator reports warm aggregate overflow without wrapping");
  expect_throws(
      [] {
        static_cast<void>(bfnew::make_no_congestion_stage_ledger(
            bfnew::measured_host_stage(1U),
            bfnew::measured_host_stage(1U),
            bfnew::measured_host_stage(
                std::numeric_limits<std::uint64_t>::max()),
            bfnew::measured_host_stage(1U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(0U)));
      },
      "stage-ledger construction fails closed on a warm timing overflow");
  expect_throws(
      [] {
        static_cast<void>(bfnew::make_no_congestion_stage_ledger(
            bfnew::measured_host_stage(
                std::numeric_limits<std::uint64_t>::max()),
            bfnew::measured_host_stage(1U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(0U),
            bfnew::measured_host_stage(1U)));
      },
      "stage-ledger construction fails closed on a cold timing overflow");
}

void test_no_congestion_transfer_accounting() {
  const bfnew::test::CompactPathFixture fixture =
      bfnew::test::make_compact_path_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::vector<float> multi_distances =
      induced_global_distances(graph, fixture.multisource_query);
  const std::vector<float> unreachable_distances =
      induced_global_distances(graph, fixture.unreachable_query);

  std::vector<bfnew::CompactQueryResult> results;
  results.push_back(bfnew::reconstruct_compact_query_paths(
      graph,
      fixture.multisource_query,
      multi_distances,
      bfnew::ExpansionQueryDisposition::reached));
  results.push_back(bfnew::reconstruct_compact_query_paths(
      graph,
      fixture.unreachable_query,
      unreachable_distances,
      bfnew::ExpansionQueryDisposition::unreachable_in_full_region));
  bfnew::RouteQuery failed_query = fixture.unreachable_query;
  failed_query.query_id = bfnew::QueryId{1805U};
  results.push_back(bfnew::make_failed_compact_query_result(
      failed_query,
      bfnew::ExpansionQueryDisposition::engine_failure));

  bfnew::CompactQueryResult reconstruction_failure;
  reconstruction_failure.query_id = bfnew::QueryId{1899U};
  reconstruction_failure.targets.resize(1U);
  reconstruction_failure.targets.front().summary.reached =
      bfnew::CompactTargetReachStatus::reached;
  reconstruction_failure.targets.front().summary.distance = 1.0F;
  reconstruction_failure.targets.front().summary.reconstruction =
      bfnew::CompactPathStatus::no_tight_path;
  results.push_back(std::move(reconstruction_failure));

  constexpr std::uint64_t modeled_status_records = 7U;
  const bfnew::NoCongestionResultAccounting accounting =
      bfnew::measure_no_congestion_result_transfer(
          results, modeled_status_records);
  const std::uint64_t expected_summary_bytes =
      5U * sizeof(bfnew::CompactTargetSummary);
  const std::uint64_t expected_vertex_bytes =
      3U * sizeof(std::uint32_t);
  const std::uint64_t expected_label_bytes = 3U * sizeof(float);
  const std::uint64_t expected_edge_bytes = sizeof(std::uint32_t);
  const std::uint64_t expected_compact_bytes = expected_summary_bytes +
                                               expected_vertex_bytes +
                                               expected_label_bytes +
                                               expected_edge_bytes;
  const std::uint64_t expected_status_bytes =
      modeled_status_records * sizeof(bfnew::DeviceRunStatus);
  expect(
      accounting.query_count == 4U &&
          accounting.target_summary_count == 5U &&
          accounting.complete_path_count == 2U &&
          accounting.unreachable_target_count == 1U &&
          accounting.terminal_failure_target_count == 1U &&
          accounting.reconstruction_failure_target_count == 1U &&
          accounting.modeled_batch_status_count == modeled_status_records &&
          accounting.modeled_batch_status_bytes == expected_status_bytes &&
          accounting.final_result_serialization.summary_bytes ==
              expected_summary_bytes &&
          accounting.final_result_serialization.vertex_bytes ==
              expected_vertex_bytes &&
          accounting.final_result_serialization.distance_label_bytes ==
              expected_label_bytes &&
          accounting.final_result_serialization.edge_id_bytes ==
              expected_edge_bytes &&
          accounting.final_result_serialization.total_bytes ==
              expected_compact_bytes &&
          accounting.modeled_final_transfer_bytes ==
              expected_status_bytes + expected_compact_bytes &&
          accounting.device_transfer_evidence ==
              bfnew::CompactStageTimingEvidence::unavailable &&
          accounting.actual_compact_device_transfer ==
              bfnew::CompactTransferAccounting{} &&
          accounting.actual_status_bytes == 0U &&
          accounting.actual_error_bytes == 0U &&
          accounting.actual_compact_total_device_to_host_bytes == 0U &&
          accounting.actual_controller_poll_count == 0U &&
          accounting.actual_controller_poll_bytes == 0U &&
          accounting.actual_overall_device_to_host_bytes == 0U,
      "no-congestion accounting separates the final serialization model "
      "from unavailable physical device-transfer evidence");

  bfnew::BatchedExpansionMetrics device_metrics;
  device_metrics.compact_transfer =
      accounting.final_result_serialization;
  device_metrics.compact_transfer.summary_bytes +=
      sizeof(bfnew::CompactTargetSummary);
  device_metrics.compact_transfer.total_bytes +=
      sizeof(bfnew::CompactTargetSummary);
  device_metrics.compact_status_bytes = expected_status_bytes;
  device_metrics.compact_error_bytes =
      modeled_status_records * 2U * sizeof(std::uint32_t);
  device_metrics.compact_total_device_to_host_bytes =
      device_metrics.compact_transfer.total_bytes +
      device_metrics.compact_status_bytes + device_metrics.compact_error_bytes;
  device_metrics.compact_controller_poll_count = 3U;
  device_metrics.compact_controller_poll_bytes =
      device_metrics.compact_controller_poll_count *
      sizeof(bfnew::DeviceController);
  device_metrics.compact_overall_device_to_host_bytes =
      device_metrics.compact_total_device_to_host_bytes +
      device_metrics.compact_controller_poll_bytes;
  const bfnew::NoCongestionResultAccounting measured =
      bfnew::measure_no_congestion_result_transfer(
          results, modeled_status_records, &device_metrics);
  expect(
      measured.device_transfer_evidence ==
              bfnew::CompactStageTimingEvidence::measured &&
          measured.actual_compact_device_transfer ==
              device_metrics.compact_transfer &&
          measured.actual_compact_device_transfer !=
              measured.final_result_serialization &&
          measured.actual_status_bytes == device_metrics.compact_status_bytes &&
          measured.actual_error_bytes == device_metrics.compact_error_bytes &&
          measured.actual_compact_total_device_to_host_bytes ==
              device_metrics.compact_total_device_to_host_bytes &&
          measured.actual_controller_poll_count ==
              device_metrics.compact_controller_poll_count &&
          measured.actual_controller_poll_bytes ==
              device_metrics.compact_controller_poll_bytes &&
          measured.actual_overall_device_to_host_bytes ==
              device_metrics.compact_overall_device_to_host_bytes,
      "measured device-transfer evidence retains retry-capable payload, "
      "status/error, controller polls, compact subtotal, and overall bytes");

  bfnew::BatchedExpansionMetrics persistent_device_metrics = device_metrics;
  persistent_device_metrics.compact_controller_poll_count = 0U;
  persistent_device_metrics.compact_controller_poll_bytes = 0U;
  persistent_device_metrics.compact_overall_device_to_host_bytes =
      persistent_device_metrics.compact_total_device_to_host_bytes;
  const bfnew::NoCongestionResultAccounting persistent_measured =
      bfnew::measure_no_congestion_result_transfer(
          results, modeled_status_records, &persistent_device_metrics);
  expect(
      persistent_measured.actual_controller_poll_count == 0U &&
          persistent_measured.actual_controller_poll_bytes == 0U &&
          persistent_measured.actual_overall_device_to_host_bytes ==
              persistent_measured
                  .actual_compact_total_device_to_host_bytes,
      "persistent compact accounting adds no controller-poll D2H traffic");

  bfnew::BatchedExpansionMetrics malformed_device_metrics = device_metrics;
  ++malformed_device_metrics.compact_total_device_to_host_bytes;
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            results, modeled_status_records, &malformed_device_metrics));
      },
      "result accounting rejects an inconsistent physical D2H aggregate");

  malformed_device_metrics = {};
  malformed_device_metrics.compact_status_bytes =
      sizeof(bfnew::DeviceRunStatus);
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            results,
            modeled_status_records,
            &malformed_device_metrics));
      },
      "unavailable device-transfer evidence cannot carry numeric status bytes");

  malformed_device_metrics = device_metrics;
  ++malformed_device_metrics.compact_controller_poll_bytes;
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            results, modeled_status_records, &malformed_device_metrics));
      },
      "result accounting rejects controller bytes that disagree with the "
      "reported poll count");

  malformed_device_metrics = device_metrics;
  ++malformed_device_metrics.compact_overall_device_to_host_bytes;
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            results, modeled_status_records, &malformed_device_metrics));
      },
      "result accounting rejects an inconsistent overall physical D2H total");

  std::vector<bfnew::CompactQueryResult> malformed{results.front()};
  malformed.front().targets.front().distance_labels.clear();
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            malformed, 0U));
      },
      "transfer accounting rejects a complete path with malformed arenas");

  malformed = {results[1U]};
  malformed.front().targets.front().summary.reached =
      bfnew::CompactTargetReachStatus::reached;
  malformed.front().targets.front().summary.distance = 1.0F;
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            malformed, 0U));
      },
      "final serialization rejects an unreachable summary marked reached and "
      "finite");

  malformed = {results[1U]};
  malformed.front().targets.front().summary.distance =
      std::numeric_limits<float>::quiet_NaN();
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            malformed, 0U));
      },
      "final serialization rejects NaN as an unreachable distance");

  malformed = {results[2U]};
  malformed.front().targets.front().summary.has_selected_source = 1U;
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            malformed, 0U));
      },
      "final serialization rejects a terminal-failure summary with a source "
      "validity flag");

  malformed = {results[2U]};
  malformed.front().targets.front().summary.distance =
      -std::numeric_limits<float>::infinity();
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            malformed, 0U));
      },
      "final serialization rejects negative infinity as a terminal-failure "
      "distance");

  malformed = {results[3U]};
  malformed.front().targets.front().summary.path_length = 1U;
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            malformed, 0U));
      },
      "final serialization rejects reconstruction failure with nonzero path "
      "length");

  malformed = {results[1U]};
  malformed.front().disposition = bfnew::ExpansionQueryDisposition::reached;
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            malformed, 0U));
      },
      "final serialization rejects a reached query with an unreachable "
      "target");

  malformed = {results.front()};
  malformed.front().disposition =
      static_cast<bfnew::ExpansionQueryDisposition>(99U);
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            malformed, 0U));
      },
      "final serialization rejects an unknown query disposition");

  malformed = {results.front()};
  malformed.front().targets.front().summary.reconstruction =
      static_cast<bfnew::CompactPathStatus>(99U);
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            malformed, 0U));
      },
      "transfer accounting rejects an unknown reconstruction status");

  expect_throws(
      [&] {
        constexpr std::uint64_t overflowing_status_count =
            std::numeric_limits<std::uint64_t>::max() /
                sizeof(bfnew::DeviceRunStatus) +
            1U;
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            {}, overflowing_status_count));
      },
      "status-transfer byte multiplication fails closed on overflow");

  std::vector<bfnew::CompactQueryResult> duplicate_results = results;
  duplicate_results.insert(
      duplicate_results.begin() + 1,
      duplicate_results.front());
  expect_throws(
      [&] {
        static_cast<void>(bfnew::measure_no_congestion_result_transfer(
            duplicate_results, modeled_status_records));
      },
      "result accounting rejects duplicate QueryIds");
}

void test_host_no_congestion_pipeline() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  bfnew::HostNoCongestionPipelineOptions pipeline_options;
  pipeline_options.cold_artifact_load = bfnew::measured_host_stage(11U);
  pipeline_options.graph_upload = bfnew::measured_host_stage(13U);
  pipeline_options.cold_execution = bfnew::measured_host_stage(17U);
  pipeline_options.quality_sample_query_count = fixture.queries.size();
  pipeline_options.quality_selection_seed = 0x1800'c01d'5eed'0001ULL;

  const bfnew::HostNoCongestionPipelineResult result =
      bfnew::run_host_no_congestion_pipeline(
          fixture.partitioned.graph,
          fixture.directory,
          fixture.tile_runs,
          fixture.device_graph,
          fixture.queries,
          expansion_options(bfnew::EngineKind::jacobi_pull),
          pipeline_options);

  expect(
      result.compact_results.size() == fixture.queries.size() &&
          result.expansion.queries.size() == fixture.queries.size() &&
          std::ranges::all_of(
              result.expansion.queries,
              [](const bfnew::ExpansionQueryOutcome& outcome) {
                return outcome.final_distances.empty() &&
                       outcome.final_distances.capacity() == 0U &&
                       !outcome.compact_paths.has_value();
              }),
      "host no-congestion pipeline returns one compact result per query and "
      "retains no graph-sized lane image or consumed payload");

  bool all_results_valid = true;
  for (std::size_t index = 0U; index < result.compact_results.size(); ++index) {
    all_results_valid =
        all_results_valid &&
        bfnew::validate_compact_query_result(
            fixture.partitioned.graph,
            result.expansion.queries[index].final_query,
            result.compact_results[index])
            .ok();
  }
  expect(
      all_results_valid,
      "host no-congestion pipeline publishes independently valid compact results");

  const bfnew::NoCongestionResultAccounting expected_accounting =
      bfnew::measure_no_congestion_result_transfer(
          result.compact_results,
          result.expansion.trace.size(),
          &result.expansion.metrics);
  expect(
      result.result_accounting == expected_accounting &&
          result.result_accounting.query_count == fixture.queries.size() &&
          result.result_accounting.modeled_batch_status_count ==
              result.expansion.trace.size(),
      "pipeline result accounting matches the final serialization model and "
      "preserves unavailable physical-transfer evidence");

  expect(
      result.expansion.metrics.compact_host_timing ==
              bfnew::CompactStageTimingEvidence::measured &&
          result.expansion.metrics.compact_device_timing ==
              bfnew::CompactStageTimingEvidence::unavailable &&
          bfnew::validate_no_congestion_stage_ledger(result.timing) ==
              bfnew::NoCongestionStageLedgerError::none &&
          result.timing.cold_artifact_load ==
              bfnew::measured_host_stage(11U) &&
          result.timing.graph_upload == bfnew::measured_host_stage(13U) &&
          result.timing.cold_execution ==
              bfnew::measured_host_stage(17U) &&
          result.timing.expansion.host_nanoseconds ==
              result.expansion.metrics.geometric_expansion_host_nanoseconds &&
          result.timing.warm_all_query.host_evidence ==
              bfnew::PipelineTimingEvidence::measured &&
          result.timing.cold_pipeline.host_evidence ==
              bfnew::PipelineTimingEvidence::measured &&
          result.timing.cold_pipeline.host_nanoseconds == 41U,
      "pipeline keeps geometric expansion separate from controller residual, "
      "keeps cold/warm executions distinct, and invents no device-event "
      "timing");

  const bfnew::CompactPathQualitySample expected_quality =
      bfnew::sample_compact_path_quality(
          fixture.partitioned.graph,
          result.expansion.queries,
          result.compact_results,
          fixture.queries.size(),
          pipeline_options.quality_selection_seed);
  expect(
      result.quality == expected_quality &&
          result.quality.population_query_count == fixture.queries.size() &&
          result.quality.sampled_query_count == fixture.queries.size() &&
          result.quality.finite_target_pairs == fixture.queries.size() - 1U,
      "pipeline quality sample is the deterministic QueryId-hash sample against "
      "the unbounded oracle");
}

void test_deterministic_path_quality_sample() {
  const bfnew::test::CompactPathFixture fixture =
      bfnew::test::make_compact_path_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::array<const bfnew::RouteQuery*, 5U> queries{
      &fixture.cycle_query,
      &fixture.tie_query,
      &fixture.bounded_query,
      &fixture.multisource_query,
      &fixture.unreachable_query,
  };
  std::vector<bfnew::ExpansionQueryOutcome> outcomes;
  std::vector<bfnew::CompactQueryResult> results;
  for (const bfnew::RouteQuery* query : queries) {
    bfnew::ExpansionQueryOutcome outcome;
    outcome.final_query = *query;
    outcome.disposition =
        query->query_id == bfnew::test::phase18_unreachable_query_id
            ? bfnew::ExpansionQueryDisposition::unreachable_in_full_region
            : bfnew::ExpansionQueryDisposition::reached;
    const std::vector<float> distances = induced_global_distances(graph, *query);
    results.push_back(bfnew::reconstruct_compact_query_paths(
        graph, *query, distances, outcome.disposition));
    outcomes.push_back(std::move(outcome));
  }

  constexpr std::uint64_t seed = 0x1800'5eed'1234'5678ULL;
  const bfnew::CompactPathQualitySample all =
      bfnew::sample_compact_path_quality(
          graph, outcomes, results, outcomes.size(), seed);
  expect(
      all.method == bfnew::PathQualitySamplingMethod::splitmix64_query_id &&
          all.selection_seed == seed && all.population_query_count == 5U &&
          all.requested_sample_query_count == 5U &&
          all.sampled_query_count == 5U &&
          all.sampled_query_ids.size() == 5U &&
          all.finite_target_pairs == 5U && all.observations.size() == 5U &&
          all.absolute_cost_inflation_p50 == 0.0 &&
          all.absolute_cost_inflation_p95 == 3.0 &&
          all.absolute_cost_inflation_p99 == 3.0 &&
          all.absolute_cost_inflation_max == 3.0 &&
          all.cost_ratio_p50 == 1.0 && all.cost_ratio_p95 == 4.0 &&
          all.cost_ratio_p99 == 4.0 && all.cost_ratio_max == 4.0 &&
          all.absolute_path_length_inflation_p50 == 0.0 &&
          all.absolute_path_length_inflation_p95 == 0.0 &&
          all.absolute_path_length_inflation_p99 == 0.0 &&
          all.absolute_path_length_inflation_max == 0.0 &&
          all.path_length_ratio_p50 == 1.0 &&
          all.path_length_ratio_p95 == 1.0 &&
          all.path_length_ratio_p99 == 1.0 &&
          all.path_length_ratio_max == 1.0,
      "quality sample reports deterministic nearest-rank bounded/unbounded "
      "cost and path-length inflation");

  const auto bounded = std::find_if(
      all.observations.begin(),
      all.observations.end(),
      [](const bfnew::CompactPathQualityObservation& observation) {
        return observation.query_id ==
            bfnew::test::phase18_bounded_query_id;
      });
  expect(
      bounded != all.observations.end() &&
          bitwise_equal(bounded->bounded_distance, 4.0F) &&
          bitwise_equal(bounded->unbounded_distance, 1.0F) &&
          bounded->bounded_path_length == 2U &&
          bounded->unbounded_path_length == 2U &&
          bounded->absolute_cost_inflation == 3.0 &&
          bounded->absolute_path_length_inflation == 0 &&
          bounded->cost_inflation_ratio == 4.0 &&
          bounded->path_length_inflation_ratio == 1.0,
      "excluded cheaper global route is reported as quality inflation, not a "
      "bounded correctness error");

  const auto source_target = std::find_if(
      all.observations.begin(),
      all.observations.end(),
      [](const bfnew::CompactPathQualityObservation& observation) {
        return observation.bounded_path_length == 0U;
      });
  expect(
      source_target != all.observations.end() &&
          source_target->bounded_distance == 0.0F &&
          source_target->unbounded_distance == 0.0F &&
          source_target->absolute_cost_inflation == 0.0 &&
          source_target->absolute_path_length_inflation == 0 &&
          source_target->cost_inflation_ratio == 1.0 &&
          source_target->path_length_inflation_ratio == 1.0,
      "target-is-source quality evidence defines zero absolute inflation and "
      "unit relative inflation without division by zero");

  const bfnew::ResourceClassId resource{1U};
  const bfnew::InputGraph zero_reference_input{
      {
          bfnew::VertexMetadata::located(0, 0, resource),
          bfnew::VertexMetadata::located(20, 0, resource),
          bfnew::VertexMetadata::located(10, 0, resource),
          bfnew::VertexMetadata::located(10, 20, resource),
      },
      {
          bfnew::test::phase18_edge(0U, 2U, 1.0F, 18'100U),
          bfnew::test::phase18_edge(2U, 1U, 1.0F, 18'101U),
          bfnew::test::phase18_edge(0U, 3U, 0.0F, 18'102U),
          bfnew::test::phase18_edge(3U, 1U, 0.0F, 18'103U),
      }};
  const bfnew::UniformGridPartitioner partitioner{
      bfnew::SpatialOrderConfig{0, 0, 10U, 10U}};
  const bfnew::PartitionedGraph zero_reference_partitioned =
      partitioner.partition(zero_reference_input);
  const bfnew::WeightedGraph& zero_reference_graph =
      zero_reference_partitioned.graph;
  const std::span<const bfnew::VertexId> zero_map =
      zero_reference_graph.old_to_new();
  const std::array zero_sources{zero_map[0U]};
  const std::array zero_targets{zero_map[1U]};
  const bfnew::RouteQuery zero_reference_query = bfnew::make_route_query(
      bfnew::QueryId{1897U},
      zero_reference_graph,
      zero_sources,
      zero_targets);
  const std::vector<float> zero_reference_distances =
      induced_global_distances(zero_reference_graph, zero_reference_query);
  const bfnew::CompactQueryResult zero_reference_result =
      bfnew::reconstruct_compact_query_paths(
          zero_reference_graph,
          zero_reference_query,
          zero_reference_distances,
          bfnew::ExpansionQueryDisposition::reached);
  bfnew::ExpansionQueryOutcome zero_reference_outcome;
  zero_reference_outcome.final_query = zero_reference_query;
  zero_reference_outcome.disposition =
      bfnew::ExpansionQueryDisposition::reached;
  const std::array zero_reference_outcomes{zero_reference_outcome};
  const std::array zero_reference_results{zero_reference_result};
  const bfnew::CompactPathQualitySample infinite_ratio =
      bfnew::sample_compact_path_quality(
          zero_reference_graph,
          zero_reference_outcomes,
          zero_reference_results,
          1U,
          0x1897'0000'0000'0001ULL);
  expect(
      infinite_ratio.observations.size() == 1U &&
          bitwise_equal(
              infinite_ratio.observations.front().bounded_distance, 2.0F) &&
          bitwise_equal(
              infinite_ratio.observations.front().unbounded_distance, 0.0F) &&
          infinite_ratio.observations.front().absolute_cost_inflation == 2.0 &&
          std::isinf(
              infinite_ratio.observations.front().cost_inflation_ratio) &&
          infinite_ratio.observations.front().cost_inflation_ratio > 0.0 &&
          infinite_ratio.observations.front().bounded_path_length == 2U &&
          infinite_ratio.observations.front().unbounded_path_length == 2U &&
          infinite_ratio.observations.front().absolute_path_length_inflation ==
              0 &&
          infinite_ratio.observations.front().path_length_inflation_ratio ==
              1.0 &&
          std::isinf(infinite_ratio.cost_ratio_p50) &&
          std::isinf(infinite_ratio.cost_ratio_p95) &&
          std::isinf(infinite_ratio.cost_ratio_p99) &&
          std::isinf(infinite_ratio.cost_ratio_max),
      "a positive bounded path with an excluded zero-cost global detour has "
      "finite absolute inflation and positive-infinite relative inflation");

  std::reverse(outcomes.begin(), outcomes.end());
  std::reverse(results.begin(), results.end());
  const bfnew::CompactPathQualitySample reordered =
      bfnew::sample_compact_path_quality(
          graph, outcomes, results, outcomes.size(), seed);
  expect(
      reordered == all,
      "splitmix64 QueryId sampling and observation order are input-order independent");

  const bfnew::CompactPathQualitySample subset_a =
      bfnew::sample_compact_path_quality(graph, outcomes, results, 2U, seed);
  const bfnew::CompactPathQualitySample subset_b =
      bfnew::sample_compact_path_quality(graph, outcomes, results, 2U, seed);
  expect(
      subset_a == subset_b && subset_a.sampled_query_count == 2U &&
          subset_a.sampled_query_ids.size() == 2U,
      "recorded seed selects the same predeclared QueryId sample repeatedly");

  const bfnew::CompactPathQualitySample empty =
      bfnew::sample_compact_path_quality(graph, outcomes, results, 0U, seed);
  expect(
      empty.sampled_query_count == 0U && empty.finite_target_pairs == 0U &&
          empty.sampled_query_ids.empty() && empty.observations.empty() &&
          empty.cost_ratio_max == 0.0 &&
          empty.path_length_ratio_max == 0.0,
      "zero requested quality sample is explicit and deterministic");

  std::vector<bfnew::CompactQueryResult> mismatched = results;
  mismatched.front().query_id = bfnew::QueryId{999U};
  expect_throws(
      [&] {
        static_cast<void>(bfnew::sample_compact_path_quality(
            graph, outcomes, mismatched, 1U, seed));
      },
      "quality sampling rejects an incomplete or mismatched result ledger");

  std::vector<bfnew::ExpansionQueryOutcome> duplicate_outcomes = outcomes;
  std::vector<bfnew::CompactQueryResult> duplicate_results = results;
  duplicate_outcomes.push_back(duplicate_outcomes.front());
  duplicate_results.push_back(duplicate_results.front());
  expect_throws(
      [&] {
        static_cast<void>(bfnew::sample_compact_path_quality(
            graph,
            duplicate_outcomes,
            duplicate_results,
            duplicate_outcomes.size(),
            seed));
      },
      "quality sampling rejects duplicate QueryIds");
}

}  // namespace

int main() {
  const auto run = [](const std::string_view name, const auto& test) {
    try {
      test();
    } catch (const std::exception& error) {
      std::cerr << "FAILED: " << name << " threw: " << error.what() << '\n';
      ++failures;
    }
  };
  run("Phase 5 semantics", test_phase5_semantics_and_compact_summaries);
  run(
      "multisource and failures",
      test_multisource_duplicate_terminals_and_failures);
  run("engine integration", test_expansion_integration_all_engines);
  run(
      "reached/malformed payload",
      test_reached_status_with_unreachable_payload_fails_closed);
  run(
      "miss/malformed payload",
      test_miss_status_requires_consistent_target_classification);
  run("payload lifecycle", test_compact_payload_lifecycle_rejections);
  run(
      "terminal mixed payload",
      test_terminal_full_region_retains_mixed_target_payload);
  run("fail-closed validation", test_validation_fail_closed_and_determinism);
  run("stage ledger", test_no_congestion_stage_ledger);
  run("transfer accounting", test_no_congestion_transfer_accounting);
  run("host pipeline", test_host_no_congestion_pipeline);
  run("quality sample", test_deterministic_path_quality_sample);

  if (failures != 0) {
    std::cerr << failures << " compact-path test assertion(s) failed\n";
    return 1;
  }
  return 0;
}
