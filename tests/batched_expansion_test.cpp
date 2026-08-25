#include "bfnew/batched_expansion.hpp"
#include "bfnew/sssp.hpp"
#include "batched_expansion_fixture_suite.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
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

[[nodiscard]] const bfnew::ExpansionQueryOutcome& outcome_for(
    const bfnew::BatchedExpansionRunResult& run,
    const bfnew::QueryId query_id) {
  const auto position = std::lower_bound(
      run.queries.begin(),
      run.queries.end(),
      query_id,
      [](const bfnew::ExpansionQueryOutcome& outcome, const bfnew::QueryId id) {
        return outcome.final_query.query_id < id;
      });
  if (position == run.queries.end() ||
      position->final_query.query_id != query_id) {
    throw std::logic_error{"Phase 17 outcome is missing"};
  }
  return *position;
}

[[nodiscard]] std::vector<bfnew::RouteQuery> query_subset(
    const bfnew::test::BatchedExpansionFixture& fixture,
    const std::span<const bfnew::QueryId> query_ids) {
  std::vector<bfnew::RouteQuery> result;
  result.reserve(query_ids.size());
  for (const bfnew::QueryId query_id : query_ids) {
    result.push_back(bfnew::test::expansion_query(fixture, query_id));
  }
  return result;
}

[[nodiscard]] bfnew::BatchedExpansionOptions expansion_options(
    const bfnew::EngineKind engine,
    const bfnew::ExpansionSchedulePolicy schedule,
    const std::uint32_t maximum_expansions = 4U,
    const bfnew::ExpansionTerminalPolicy terminal_policy =
        bfnew::ExpansionTerminalPolicy::full_region_fallback) {
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
  options.execution_configuration_fingerprint = 0x1700'0000'0000'0001ULL;
  options.schedule = schedule;
  options.maximum_expansions = maximum_expansions;
  options.terminal_policy = terminal_policy;
  return options;
}

[[nodiscard]] std::array<bfnew::ExpansionSchedulePolicy, 4U>
schedule_matrix() {
  return {
      bfnew::one_ring_expansion(),
      bfnew::fixed_ring_expansion(2U),
      bfnew::doubling_margin_expansion(),
      bfnew::hybrid_margin_expansion(2U),
  };
}

[[nodiscard]] std::uint32_t expected_expansions(
    const bfnew::ExpansionScheduleKind schedule,
    const bfnew::QueryId query_id) {
  if (query_id == bfnew::test::phase17_immediate_query_id) {
    return 0U;
  }
  if (query_id == bfnew::test::phase17_unreachable_query_id) {
    return 1U;  // Its first stalled ring enters the one final full fallback.
  }
  switch (schedule) {
    case bfnew::ExpansionScheduleKind::one_geometric_ring:
      if (query_id == bfnew::test::phase17_two_ring_query_id ||
          query_id == bfnew::test::phase17_spill_query_id) {
        return 2U;
      }
      if (query_id == bfnew::test::phase17_long_edge_query_id) {
        return 4U;
      }
      return 3U;
    case bfnew::ExpansionScheduleKind::fixed_larger_ring:
      if (query_id == bfnew::test::phase17_two_ring_query_id ||
          query_id == bfnew::test::phase17_spill_query_id) {
        return 1U;
      }
      return 2U;
    case bfnew::ExpansionScheduleKind::doubling_xy_margins:
    case bfnew::ExpansionScheduleKind::hybrid_small_then_doubling:
      if (query_id == bfnew::test::phase17_two_ring_query_id ||
          query_id == bfnew::test::phase17_spill_query_id) {
        return 2U;
      }
      return 3U;
    case bfnew::ExpansionScheduleKind::unspecified:
      break;
  }
  throw std::logic_error{"unexpected Phase 17 schedule/query pair"};
}

void expect_dijkstra_on_final_region(
    const bfnew::WeightedGraph& graph,
    const bfnew::ExpansionQueryOutcome& outcome,
    const std::string& prefix) {
  expect(
      outcome.final_distances.size() == graph.vertex_count(),
      prefix + ": terminal portable image has one full graph projection");
  if (outcome.final_distances.size() != graph.vertex_count()) {
    return;
  }
  const bfnew::InducedQueryGraph induced =
      bfnew::build_induced_query_graph(graph, outcome.final_query);
  const bfnew::SsspResult oracle =
      bfnew::dijkstra_oracle(induced.graph, induced.sources);
  for (std::size_t local = 0U; local < induced.local_to_global.size(); ++local) {
    const std::size_t global = induced.local_to_global[local].value();
    expect(
        bitwise_equal(outcome.final_distances[global], oracle.distances[local]),
        prefix + ": terminal distance bits agree with induced Dijkstra");
  }
  bool all_targets_finite = true;
  for (const bfnew::VertexId target : outcome.final_query.targets) {
    all_targets_finite =
        all_targets_finite &&
        std::isfinite(outcome.final_distances[target.value()]);
  }
  expect(
      all_targets_finite == outcome.reached(),
      prefix + ": no missed target is classified as success");
}

void expect_metric_identities(
    const bfnew::BatchedExpansionRunResult& run,
    const std::string& prefix) {
  const bfnew::BatchedExpansionMetrics& metrics = run.metrics;
  expect(
      metrics.input_queries == run.queries.size(),
      prefix + ": input count equals canonical outcomes");
  expect(
      metrics.reached_queries + metrics.unreachable_full_region_queries +
              metrics.expansion_limit_queries + metrics.stalled_region_queries +
              metrics.identity_or_count_overflow_queries +
              metrics.engine_failure_queries ==
          metrics.input_queries,
      prefix + ": terminal disposition counters partition input queries");
  expect(
      metrics.batches_executed ==
          metrics.initial_batches_executed + metrics.retry_batches_executed &&
          metrics.planning_passes != 0U,
      prefix + ": batch and planning-pass accounting is exact");
  expect(
      metrics.retry_valid_lane_observations <= metrics.retry_lane_capacity &&
          metrics.failed_lane_observations <=
              metrics.failed_origin_valid_lane_observations,
      prefix + ": failed-lane utilization numerators stay bounded");
  expect(
      metrics.device_work.expansion_count ==
          metrics.scheduled_expansions + metrics.full_region_fallbacks,
      prefix + ": device-facing expansion count includes every restart");

  std::uint64_t histogram_total = 0U;
  std::uint64_t scheduled_total = 0U;
  std::uint64_t fallback_total = 0U;
  std::uint64_t final_tiles = 0U;
  std::uint64_t final_vertices = 0U;
  std::uint64_t final_edges = 0U;
  for (const std::uint64_t count : metrics.expansion_count_histogram) {
    histogram_total += count;
  }
  for (const bfnew::ExpansionQueryOutcome& outcome : run.queries) {
    scheduled_total += outcome.scheduled_expansions;
    fallback_total += outcome.used_full_region_fallback ? 1U : 0U;
    final_tiles += outcome.final_query.selected_tiles.size();
    final_vertices += outcome.selected_vertex_count;
    final_edges += outcome.selected_edge_count;
    expect(
        outcome.attempts == outcome.total_expansions + 1U &&
            outcome.final_query.expansion_generation ==
                outcome.total_expansions,
        prefix + ": every expansion is followed by a clean source restart");
  }
  expect(
      histogram_total == metrics.input_queries &&
          scheduled_total == metrics.scheduled_expansions &&
          fallback_total == metrics.full_region_fallbacks,
      prefix + ": expansion distribution and totals agree");
  expect(
      final_tiles == metrics.final_selected_tile_count &&
          final_vertices == metrics.final_selected_vertex_count &&
          final_edges == metrics.final_selected_edge_count,
      prefix + ": final admitted-region totals agree with outcomes");
}

void test_all_schedules_and_engines() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::array all_query_ids{
      bfnew::test::phase17_immediate_query_id,
      bfnew::test::phase17_two_ring_query_id,
      bfnew::test::phase17_long_edge_query_id,
      bfnew::test::phase17_spill_query_id,
      bfnew::test::phase17_unreachable_query_id,
      bfnew::test::phase17_second_miss_query_id,
  };
  const std::vector<bfnew::RouteQuery> queries =
      query_subset(fixture, all_query_ids);
  const std::array engines{
      bfnew::EngineKind::jacobi_pull,
      bfnew::EngineKind::dense_chaotic_push,
      bfnew::EngineKind::frontier_push,
  };

  for (const bfnew::EngineKind engine : engines) {
    const std::array schedules = schedule_matrix();
    std::array<bfnew::ExpansionScheduleEvidence, 4U> evidence{};
    for (std::size_t schedule_index = 0U;
         schedule_index < schedules.size();
         ++schedule_index) {
      const bfnew::ExpansionSchedulePolicy schedule =
          schedules[schedule_index];
      const bfnew::BatchedExpansionOptions options =
          expansion_options(engine, schedule);
      const bfnew::BatchedExpansionRunResult run =
          bfnew::run_host_batched_expansion(
              graph,
              fixture.directory,
              fixture.tile_runs,
              fixture.device_graph,
              queries,
              options);
      const std::string prefix =
          "engine " + std::to_string(static_cast<std::uint32_t>(engine)) +
          " schedule " +
          std::to_string(static_cast<std::uint32_t>(schedule.kind));

      expect(
          run.queries.size() == queries.size() &&
              run.metrics.initial_reached_queries == 1U &&
              run.metrics.reached_queries == 5U &&
              run.metrics.unreachable_full_region_queries == 1U &&
              run.metrics.engine_failure_queries == 0U,
          prefix + ": multiple initial misses resolve without false success");
      expect(
          !run.trace.empty() && run.trace.front().valid_lane_mask == 0x3FU &&
              run.trace.front().reached_lane_mask == 0x1U &&
              run.trace.front().miss_lane_mask == 0x3EU,
          prefix + ": one reached lane and five missed lanes share first batch");
      expect(
          run.metrics.work_evidence == bfnew::ExpansionWorkEvidence::measured &&
              run.metrics.work_measured_batches == run.metrics.batches_executed &&
              run.metrics.logical_lane_edge_work >= run.metrics.shared_edge_work,
          prefix + ": portable work and retry utilization are measured");

      for (const bfnew::QueryId query_id : all_query_ids) {
        const bfnew::ExpansionQueryOutcome& outcome =
            outcome_for(run, query_id);
        const std::uint32_t expected =
            expected_expansions(schedule.kind, query_id);
        expect(
            outcome.total_expansions == expected &&
                outcome.attempts == expected + 1U,
            prefix + ": schedule has the expected bounded retry count");
        if (query_id == bfnew::test::phase17_unreachable_query_id) {
          expect(
              outcome.disposition ==
                      bfnew::ExpansionQueryDisposition::unreachable_in_full_region &&
                  outcome.used_full_region_fallback &&
                  outcome.scheduled_expansions == 0U,
              prefix + ": globally unreachable query terminates after fallback");
        } else {
          expect(
              outcome.reached() &&
                  outcome.used_full_region_fallback == false,
              prefix + ": routable query reaches without unnecessary fallback");
        }
        if (query_id == bfnew::test::phase17_immediate_query_id) {
          expect(
              outcome.final_query.expansion_generation == 0U &&
                  outcome.attempts == 1U,
              prefix + ": reached query is never expanded");
        }
        expect_dijkstra_on_final_region(graph, outcome, prefix);
      }
      expect_metric_identities(run, prefix);
      evidence[schedule_index] =
          bfnew::make_expansion_schedule_evidence(schedule, run);
    }
    const std::optional<bfnew::ExpansionSchedulePolicy> selected =
        bfnew::select_expansion_schedule_from_evidence(evidence);
    expect(
        selected.has_value() &&
            evidence.front().comparison_fingerprint != 0U &&
            std::ranges::all_of(
                evidence,
                [&](const bfnew::ExpansionScheduleEvidence& record) {
                  return record.comparison_fingerprint ==
                      evidence.front().comparison_fingerprint;
                }),
        "one complete comparable synthetic matrix exercises evidence-only selection");
  }
}

void test_fallback_failure_and_single_miss() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::array query_ids{
      bfnew::test::phase17_immediate_query_id,
      bfnew::test::phase17_long_edge_query_id,
  };
  const std::vector<bfnew::RouteQuery> queries =
      query_subset(fixture, query_ids);

  bfnew::BatchedExpansionOptions options = expansion_options(
      bfnew::EngineKind::frontier_push,
      bfnew::one_ring_expansion(),
      1U,
      bfnew::ExpansionTerminalPolicy::full_region_fallback);
  bfnew::BatchedExpansionRunResult run =
      bfnew::run_host_batched_expansion(
          graph,
          fixture.directory,
          fixture.tile_runs,
          fixture.device_graph,
          queries,
          options);
  const bfnew::ExpansionQueryOutcome& reached =
      outcome_for(run, bfnew::test::phase17_immediate_query_id);
  const bfnew::ExpansionQueryOutcome& fallback =
      outcome_for(run, bfnew::test::phase17_long_edge_query_id);
  expect(
      reached.reached() && reached.total_expansions == 0U &&
          fallback.reached() && fallback.scheduled_expansions == 1U &&
          fallback.total_expansions == 2U &&
          fallback.used_full_region_fallback &&
          run.metrics.initial_reached_queries == 1U &&
          run.metrics.full_region_fallbacks == 1U,
      "single missed lane alone restarts through the final full-region fallback");
  expect_dijkstra_on_final_region(graph, fallback, "successful full fallback");

  options.terminal_policy = bfnew::ExpansionTerminalPolicy::explicit_failure;
  run = bfnew::run_host_batched_expansion(
      graph,
      fixture.directory,
      fixture.tile_runs,
      fixture.device_graph,
      queries,
      options);
  const bfnew::ExpansionQueryOutcome& limited =
      outcome_for(run, bfnew::test::phase17_long_edge_query_id);
  expect(
      limited.disposition == bfnew::ExpansionQueryDisposition::expansion_limit &&
          limited.scheduled_expansions == 1U &&
          limited.total_expansions == 1U && !limited.used_full_region_fallback &&
          run.metrics.expansion_limit_queries == 1U &&
          run.metrics.full_region_fallbacks == 0U,
      "explicit terminal policy reports the configured expansion limit");

  const std::array stalled_id{bfnew::test::phase17_unreachable_query_id};
  const std::vector<bfnew::RouteQuery> stalled_query =
      query_subset(fixture, stalled_id);
  run = bfnew::run_host_batched_expansion(
      graph,
      fixture.directory,
      fixture.tile_runs,
      fixture.device_graph,
      stalled_query,
      expansion_options(
          bfnew::EngineKind::jacobi_pull,
          bfnew::one_ring_expansion(),
          4U,
          bfnew::ExpansionTerminalPolicy::explicit_failure));
  expect(
      run.queries.front().disposition ==
              bfnew::ExpansionQueryDisposition::region_stalled &&
          run.queries.front().total_expansions == 0U &&
          run.metrics.stalled_region_queries == 1U,
      "an empty geometric growth step is explicit, never an infinite retry");
}

void test_restart_identity_determinism_and_representation() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::array ordered_ids{
      bfnew::test::phase17_immediate_query_id,
      bfnew::test::phase17_two_ring_query_id,
      bfnew::test::phase17_second_miss_query_id,
  };
  std::vector<bfnew::RouteQuery> ordered = query_subset(fixture, ordered_ids);
  std::vector<bfnew::RouteQuery> shuffled{ordered[2U], ordered[0U], ordered[1U]};
  const std::vector<bfnew::RouteQuery> originals = ordered;
  const bfnew::BatchedExpansionOptions options = expansion_options(
      bfnew::EngineKind::frontier_push, bfnew::one_ring_expansion());

  bfnew::BatchedExpansionOptions unbound_options = options;
  unbound_options.execution_configuration_fingerprint = 0U;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::run_host_batched_expansion(
            graph,
            fixture.directory,
            fixture.tile_runs,
            fixture.device_graph,
            ordered,
            unbound_options));
      },
      "portable adapter rejects options without a runner identity");

  const bfnew::BatchedExpansionRunResult first =
      bfnew::run_host_batched_expansion(
          graph,
          fixture.directory,
          fixture.tile_runs,
          fixture.device_graph,
          ordered,
          options);
  const bfnew::BatchedExpansionRunResult second =
      bfnew::run_host_batched_expansion(
          graph,
          fixture.directory,
          fixture.tile_runs,
          fixture.device_graph,
          shuffled,
          options);
  expect(
      ordered == originals && first.trace == second.trace,
      "replanning is deterministic and leaves caller queries unchanged");
  for (std::size_t index = 0U; index < first.queries.size(); ++index) {
    const bfnew::ExpansionQueryOutcome& left = first.queries[index];
    const bfnew::ExpansionQueryOutcome& right = second.queries[index];
    expect(
        left.final_query == right.final_query &&
            left.disposition == right.disposition &&
            left.attempts == right.attempts &&
            left.scheduled_expansions == right.scheduled_expansions &&
            left.final_distances == right.final_distances,
        "canonical outcomes are independent of input order");
    const bfnew::RouteQuery& original =
        bfnew::test::expansion_query(fixture, left.final_query.query_id);
    expect(
        left.final_query.query_id == original.query_id &&
            left.final_query.source_terminals == original.source_terminals &&
            left.final_query.target_terminals == original.target_terminals &&
            left.final_query.sources == original.sources &&
            left.final_query.targets == original.targets &&
            left.final_query.source_terminal_to_source ==
                original.source_terminal_to_source &&
            left.final_query.target_terminal_to_target ==
                original.target_terminal_to_target,
        "identity and original terminal/source mappings survive every restart");
  }

  bfnew::HostBatchedExpansionOptions compact;
  compact.run_representation =
      bfnew::BatchRunRepresentation::compact_nonzero_descriptors;
  const bfnew::BatchedExpansionRunResult descriptor =
      bfnew::run_host_batched_expansion(
          graph,
          fixture.directory,
          fixture.tile_runs,
          fixture.device_graph,
          ordered,
          options,
          compact);
  expect(
      descriptor.trace == first.trace &&
          descriptor.metrics.schedule_comparison_fingerprint !=
              first.metrics.schedule_comparison_fingerprint,
      "retained masks and compact descriptors preserve results but retain "
      "distinct comparison identities");
  for (const bfnew::ExpansionQueryOutcome& retained : first.queries) {
    const bfnew::ExpansionQueryOutcome& compact_outcome =
        outcome_for(descriptor, retained.final_query.query_id);
    expect(
        retained.final_query == compact_outcome.final_query &&
            retained.disposition == compact_outcome.disposition &&
            retained.final_distances == compact_outcome.final_distances,
        "retained masks and compact descriptors have exact terminal parity");
  }

  const bfnew::ExpansionQueryOutcome& north =
      outcome_for(first, bfnew::test::phase17_two_ring_query_id);
  const bfnew::ExpansionQueryOutcome& south =
      outcome_for(first, bfnew::test::phase17_second_miss_query_id);
  expect(
      north.final_query.selected_tiles != south.final_query.selected_tiles &&
          north.final_query.expansion_generation != 0U &&
          south.final_query.expansion_generation != 0U,
      "failed lanes with dissimilar expanded regions are replanned by identity");
}

[[nodiscard]] bfnew::ExpansionBatchExecution synthetic_execution(
    const bfnew::BatchedExpansionOptions& options,
    const bfnew::BatchPlanEntry& batch,
    const bfnew::LaneMask reached,
    const bfnew::LaneMask missed,
    const bfnew::DeviceStopReason reason = bfnew::DeviceStopReason::converged) {
  bfnew::ExpansionBatchExecution execution;
  execution.result.engine_kind =
      static_cast<std::uint32_t>(options.run_options.engine);
  execution.result.control_mode =
      static_cast<std::uint32_t>(options.run_options.control_mode);
  bfnew::DeviceRunStatus& status = execution.result.status;
  status.valid_lane_mask = batch.valid_lane_mask;
  status.reached_target_mask = reached;
  status.bounding_box_miss_mask = missed;
  status.rounds_completed = 1U;
  status.stop_reason = static_cast<std::uint32_t>(reason);
  if (reason == bfnew::DeviceStopReason::converged) {
    status.converged = 1U;
    status.converged_lane_mask = batch.valid_lane_mask;
  } else {
    status.active_lane_mask = batch.valid_lane_mask;
  }
  if (reason == bfnew::DeviceStopReason::queue_overflow) {
    status.error_bits = bfnew::device_error::queue_overflow;
  } else if (reason == bfnew::DeviceStopReason::device_failure) {
    status.error_bits = bfnew::device_error::device_failure;
  } else if (reason == bfnew::DeviceStopReason::invalid_controller_state) {
    status.error_bits = bfnew::device_error::invalid_controller_state;
  }
  execution.work_evidence = bfnew::ExpansionWorkEvidence::measured;
  execution.shared_edge_work = 3U;
  execution.logical_lane_edge_work = 7U;
  execution.result.work.edges_examined = 7U;
  execution.result.work.kernel_dispatches = 1U;
  return execution;
}

void test_generic_status_errors_and_restart_contract() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::array query_ids{
      bfnew::test::phase17_immediate_query_id,
      bfnew::test::phase17_two_ring_query_id,
  };
  const std::vector<bfnew::RouteQuery> queries =
      query_subset(fixture, query_ids);
  const bfnew::BatchedExpansionOptions options = expansion_options(
      bfnew::EngineKind::frontier_push, bfnew::one_ring_expansion());

  std::uint32_t callbacks = 0U;
  const bfnew::ExpansionBatchRunner maximum_runner =
      [&](const std::span<const bfnew::RouteQuery> current,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        ++callbacks;
        for (const bfnew::RouteQuery& query : current) {
          expect(
              query.expansion_generation == 0U,
              "maximum-round error never receives an expanded query");
        }
        return synthetic_execution(
            options,
            batch,
            0U,
            0U,
            bfnew::DeviceStopReason::maximum_rounds);
      };
  bfnew::BatchedExpansionRunResult run = bfnew::run_batched_expansion(
      graph,
      fixture.directory,
      fixture.tile_runs,
      queries,
      options,
      maximum_runner);
  expect(
      callbacks == 1U && run.metrics.engine_failure_queries == queries.size() &&
          run.metrics.scheduled_expansions == 0U &&
          run.metrics.full_region_fallbacks == 0U,
      "maximum-round terminalizes every lane without converting it to a miss");
  for (const bfnew::ExpansionQueryOutcome& outcome : run.queries) {
    expect(
        outcome.disposition == bfnew::ExpansionQueryDisposition::engine_failure &&
            outcome.attempts == 1U && outcome.total_expansions == 0U &&
            outcome.final_distances.empty(),
        "engine error clears partial labels and preserves generation zero");
  }

  const bfnew::ExpansionBatchRunner overflow_runner =
      [&](const std::span<const bfnew::RouteQuery>,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        return synthetic_execution(
            options,
            batch,
            0U,
            0U,
            bfnew::DeviceStopReason::queue_overflow);
      };
  run = bfnew::run_batched_expansion(
      graph,
      fixture.directory,
      fixture.tile_runs,
      queries,
      options,
      overflow_runner);
  expect(
      run.metrics.engine_failure_queries == queries.size() &&
          run.metrics.device_work.expansion_count == 0U,
      "queue overflow is an engine failure, never reachability expansion");

  const bfnew::ExpansionBatchRunner incomplete_partition_runner =
      [&](const std::span<const bfnew::RouteQuery>,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        return synthetic_execution(options, batch, 0U, 0U);
      };
  expect_throws<std::logic_error>(
      [&] {
        static_cast<void>(bfnew::run_batched_expansion(
            graph,
            fixture.directory,
            fixture.tile_runs,
            queries,
            options,
            incomplete_partition_runner));
      },
      "clean convergence must partition every valid lane");

  const bfnew::ExpansionBatchRunner nonclean_classification_runner =
      [&](const std::span<const bfnew::RouteQuery>,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        return synthetic_execution(
            options,
            batch,
            batch.valid_lane_mask,
            0U,
            bfnew::DeviceStopReason::maximum_rounds);
      };
  expect_throws<std::logic_error>(
      [&] {
        static_cast<void>(bfnew::run_batched_expansion(
            graph,
            fixture.directory,
            fixture.tile_runs,
            queries,
            options,
            nonclean_classification_runner));
      },
      "non-clean status cannot carry reached or miss classification bits");

  const bfnew::ExpansionBatchRunner malformed_image_runner =
      [&](const std::span<const bfnew::RouteQuery>,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        bfnew::ExpansionBatchExecution execution = synthetic_execution(
            options, batch, batch.valid_lane_mask, 0U);
        execution.final_distances.push_back(0.0F);
        return execution;
      };
  expect_throws<std::logic_error>(
      [&] {
        static_cast<void>(bfnew::run_batched_expansion(
            graph,
            fixture.directory,
            fixture.tile_runs,
            queries,
            options,
            malformed_image_runner));
      },
      "runner distance image must use exact V-by-W shape");

  const bfnew::ExpansionBatchRunner invalid_evidence_runner =
      [&](const std::span<const bfnew::RouteQuery>,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        bfnew::ExpansionBatchExecution execution = synthetic_execution(
            options, batch, batch.valid_lane_mask, 0U);
        execution.work_evidence =
            static_cast<bfnew::ExpansionWorkEvidence>(255U);
        return execution;
      };
  expect_throws<std::logic_error>(
      [&] {
        static_cast<void>(bfnew::run_batched_expansion(
            graph,
            fixture.directory,
            fixture.tile_runs,
            queries,
            options,
            invalid_evidence_runner));
      },
      "runner rejects an unknown work-evidence state");

  const bfnew::ExpansionBatchRunner unavailable_numeric_runner =
      [&](const std::span<const bfnew::RouteQuery>,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        bfnew::ExpansionBatchExecution execution = synthetic_execution(
            options, batch, batch.valid_lane_mask, 0U);
        execution.work_evidence = bfnew::ExpansionWorkEvidence::unavailable;
        execution.shared_edge_work = 1U;
        execution.logical_lane_edge_work = 0U;
        return execution;
      };
  expect_throws<std::logic_error>(
      [&] {
        static_cast<void>(bfnew::run_batched_expansion(
            graph,
            fixture.directory,
            fixture.tile_runs,
            queries,
            options,
            unavailable_numeric_runner));
      },
      "unavailable work evidence cannot smuggle numeric work counters");

  const bfnew::ExpansionBatchRunner unavailable_device_work_runner =
      [&](const std::span<const bfnew::RouteQuery>,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        bfnew::ExpansionBatchExecution execution = synthetic_execution(
            options, batch, batch.valid_lane_mask, 0U);
        execution.work_evidence = bfnew::ExpansionWorkEvidence::unavailable;
        execution.shared_edge_work = 0U;
        execution.logical_lane_edge_work = 0U;
        execution.result.work.edges_examined = 1U;
        return execution;
      };
  expect_throws<std::logic_error>(
      [&] {
        static_cast<void>(bfnew::run_batched_expansion(
            graph,
            fixture.directory,
            fixture.tile_runs,
            queries,
            options,
            unavailable_device_work_runner));
      },
      "unavailable evidence cannot smuggle device work counters");

  const bfnew::ExpansionBatchRunner impossible_measured_work_runner =
      [&](const std::span<const bfnew::RouteQuery>,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        bfnew::ExpansionBatchExecution execution = synthetic_execution(
            options, batch, batch.valid_lane_mask, 0U);
        execution.shared_edge_work = 8U;
        execution.logical_lane_edge_work = 7U;
        return execution;
      };
  expect_throws<std::logic_error>(
      [&] {
        static_cast<void>(bfnew::run_batched_expansion(
            graph,
            fixture.directory,
            fixture.tile_runs,
            queries,
            options,
            impossible_measured_work_runner));
      },
      "measured logical lane work cannot be below shared edge work");

  const bfnew::ExpansionBatchRunner unavailable_runner =
      [&](const std::span<const bfnew::RouteQuery>,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        bfnew::ExpansionBatchExecution execution = synthetic_execution(
            options, batch, batch.valid_lane_mask, 0U);
        execution.work_evidence = bfnew::ExpansionWorkEvidence::unavailable;
        execution.shared_edge_work = 0U;
        execution.logical_lane_edge_work = 0U;
        execution.result.work = {};
        return execution;
      };
  run = bfnew::run_batched_expansion(
      graph,
      fixture.directory,
      fixture.tile_runs,
      queries,
      options,
      unavailable_runner);
  expect(
      run.metrics.work_evidence == bfnew::ExpansionWorkEvidence::unavailable &&
          run.metrics.work_measured_batches == 0U &&
          run.metrics.shared_edge_work == 0U &&
          run.metrics.logical_lane_edge_work == 0U,
      "unavailable work remains explicit rather than measured zero");

  std::uint32_t empty_callbacks = 0U;
  const bfnew::BatchedExpansionRunResult empty = bfnew::run_batched_expansion(
      graph,
      fixture.directory,
      fixture.tile_runs,
      {},
      options,
      [&](const std::span<const bfnew::RouteQuery>,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        ++empty_callbacks;
        return synthetic_execution(options, batch, 0U, 0U);
      });
  expect(
      empty_callbacks == 0U && empty.queries.empty() && empty.trace.empty() &&
          empty.metrics.input_queries == 0U &&
          empty.metrics.expansion_count_histogram == std::vector<std::uint64_t>{0U},
      "empty all-query execution is a deterministic no-op");
}

void test_multisource_generation_overflow_and_retry_work() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const bfnew::BatchedExpansionOptions options = expansion_options(
      bfnew::EngineKind::jacobi_pull, bfnew::one_ring_expansion());

  const std::array multi_id{bfnew::test::phase17_multi_source_query_id};
  const std::vector<bfnew::RouteQuery> multi = query_subset(fixture, multi_id);
  bfnew::BatchedExpansionRunResult run =
      bfnew::run_host_batched_expansion(
          graph,
          fixture.directory,
          fixture.tile_runs,
          fixture.device_graph,
          multi,
          options);
  const bfnew::ExpansionQueryOutcome& multi_outcome = run.queries.front();
  expect(
      multi_outcome.reached() && multi_outcome.attempts == 3U &&
          multi_outcome.total_expansions == 2U &&
          multi_outcome.final_query.sources == multi.front().sources &&
          multi_outcome.final_query.source_terminals ==
              multi.front().source_terminals &&
          multi_outcome.final_query.sources.size() == 2U,
      "multi-source query restarts from its complete original source set");
  for (const bfnew::VertexId source : multi_outcome.final_query.sources) {
    expect(
        bitwise_equal(multi_outcome.final_distances[source.value()], 0.0F),
        "each original source is reseeded to positive zero after restart");
  }
  expect_dijkstra_on_final_region(
      graph, multi_outcome, "multi-source restart");

  bfnew::RouteQuery high_identity = bfnew::test::expansion_query(
      fixture, bfnew::test::phase17_two_ring_query_id);
  high_identity.query_id = bfnew::QueryId{
      std::numeric_limits<std::uint32_t>::max()};
  high_identity.expansion_generation = 7U;
  expect(
      bfnew::validate_route_query(graph, high_identity).ok(),
      "UINT32_MAX QueryId with nonzero generation remains a valid identity");
  run = bfnew::run_host_batched_expansion(
      graph,
      fixture.directory,
      fixture.tile_runs,
      fixture.device_graph,
      std::span<const bfnew::RouteQuery>{&high_identity, 1U},
      options);
  expect(
      run.queries.front().reached() &&
          run.queries.front().final_query.query_id == high_identity.query_id &&
          run.queries.front().final_query.expansion_generation == 9U &&
          run.queries.front().total_expansions == 2U,
      "maximum QueryId and nonzero initial generation survive replanning");

  bfnew::RouteQuery generation_limit = high_identity;
  generation_limit.query_id = bfnew::QueryId{1710U};
  generation_limit.expansion_generation =
      std::numeric_limits<std::uint32_t>::max();
  const bfnew::ExpansionBatchRunner always_miss =
      [&](const std::span<const bfnew::RouteQuery>,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        return synthetic_execution(options, batch, 0U, batch.valid_lane_mask);
      };
  run = bfnew::run_batched_expansion(
      graph,
      fixture.directory,
      fixture.tile_runs,
      std::span<const bfnew::RouteQuery>{&generation_limit, 1U},
      options,
      always_miss);
  expect(
      run.queries.front().disposition ==
              bfnew::ExpansionQueryDisposition::identity_or_count_overflow &&
          run.queries.front().attempts == 1U &&
          run.queries.front().total_expansions == 0U &&
          run.queries.front().final_query.expansion_generation ==
              std::numeric_limits<std::uint32_t>::max() &&
          run.metrics.identity_or_count_overflow_queries == 1U &&
          run.metrics.scheduled_expansions == 0U &&
          run.metrics.full_region_fallbacks == 0U,
      "generation/count overflow terminalizes one query without wrap or abort");

  const std::array retry_id{bfnew::test::phase17_two_ring_query_id};
  const std::vector<bfnew::RouteQuery> retry_query =
      query_subset(fixture, retry_id);
  const bfnew::ExpansionBatchRunner measured_retry =
      [&](const std::span<const bfnew::RouteQuery> current,
          const std::span<const bfnew::BatchQueryFeatures>,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext&) {
        expect(
            current.size() == 1U &&
                current.front().sources == retry_query.front().sources,
            "retry callback receives original sources and one current generation");
        const bool reached = current.front().expansion_generation >= 2U;
        return synthetic_execution(
            options,
            batch,
            reached ? batch.valid_lane_mask : 0U,
            reached ? 0U : batch.valid_lane_mask);
      };
  run = bfnew::run_batched_expansion(
      graph,
      fixture.directory,
      fixture.tile_runs,
      retry_query,
      options,
      measured_retry);
  expect(
      run.metrics.batches_executed == 3U &&
          run.metrics.work_measured_batches == 3U &&
          run.metrics.shared_edge_work == 9U &&
          run.metrics.logical_lane_edge_work == 21U &&
          run.metrics.retry_work_measured_batches == 2U &&
          run.metrics.retry_shared_edge_work == 6U &&
          run.metrics.retry_logical_lane_edge_work == 14U &&
          run.metrics.failed_batch_work_measured_batches == 2U &&
          run.metrics.failed_batch_shared_edge_work == 6U &&
          run.metrics.failed_batch_logical_lane_edge_work == 14U &&
          run.metrics.device_work.edges_examined == 21U &&
          run.metrics.device_work.kernel_dispatches == 3U &&
          run.metrics.repeated_selected_edge_estimate <=
              run.metrics.attempted_selected_edge_estimate &&
          run.metrics.schedule_comparison_fingerprint != 0U &&
          run.metrics.total_nanoseconds != 0U &&
          run.metrics.all_query_throughput_milliqueries_per_second != 0U,
      "retry and failed-batch actual work plus bounded throughput are explicit");
  expect(
      run.trace.size() == 3U &&
          run.trace[0U].expansion_generations_by_lane[0U] == 0U &&
          run.trace[1U].expansion_generations_by_lane[0U] == 1U &&
          run.trace[2U].expansion_generations_by_lane[0U] == 2U,
      "retry trace retains exact identity and generation sequence");
}

void test_option_validation_and_schedule_selection() {
  bfnew::BatchedExpansionOptions options = expansion_options(
      bfnew::EngineKind::jacobi_pull, bfnew::one_ring_expansion());
  expect(
      bfnew::validate_batched_expansion_options(options) ==
          bfnew::BatchedExpansionOptionsError::none,
      "baseline Phase 17 options validate");

  options.schedule = {};
  expect(
      bfnew::validate_batched_expansion_options(options) ==
          bfnew::BatchedExpansionOptionsError::unspecified_schedule,
      "unspecified schedule is never a default");
  options = expansion_options(
      bfnew::EngineKind::jacobi_pull, bfnew::one_ring_expansion());
  options.schedule.kind = static_cast<bfnew::ExpansionScheduleKind>(255U);
  expect(
      bfnew::validate_batched_expansion_options(options) ==
          bfnew::BatchedExpansionOptionsError::invalid_schedule,
      "unknown schedule enums are rejected fail closed");
  options = expansion_options(
      bfnew::EngineKind::jacobi_pull, bfnew::one_ring_expansion());
  options.execution_configuration_fingerprint = 0U;
  expect(
      bfnew::validate_batched_expansion_options(options) ==
      bfnew::BatchedExpansionOptionsError::
              missing_execution_configuration_fingerprint,
      "generic runners must bind external execution settings into evidence");
  options = expansion_options(
      bfnew::EngineKind::jacobi_pull, bfnew::one_ring_expansion());
  options.schedule = bfnew::fixed_ring_expansion(1U);
  expect(
      bfnew::validate_batched_expansion_options(options) ==
          bfnew::BatchedExpansionOptionsError::invalid_fixed_ring_size,
      "fixed larger ring must actually exceed one ring");
  options.schedule = bfnew::one_ring_expansion();
  options.schedule.fixed_ring_size = 2U;
  expect(
      bfnew::validate_batched_expansion_options(options) ==
          bfnew::BatchedExpansionOptionsError::unexpected_fixed_ring_size,
      "one-ring schedule rejects fixed-ring metadata");
  options.schedule = bfnew::hybrid_margin_expansion(0U);
  expect(
      bfnew::validate_batched_expansion_options(options) ==
          bfnew::BatchedExpansionOptionsError::invalid_hybrid_small_expansion_count,
      "hybrid schedule requires a small-first stage");
  options = expansion_options(
      bfnew::EngineKind::jacobi_pull, bfnew::one_ring_expansion());
  options.run_options.maximum_rounds = 0U;
  expect(
      bfnew::validate_batched_expansion_options(options) ==
          bfnew::BatchedExpansionOptionsError::invalid_run_options,
      "invalid engine options are rejected before planning");
  options = expansion_options(
      bfnew::EngineKind::jacobi_pull, bfnew::one_ring_expansion());
  options.planner_policy.lane_width = 4U;
  expect(
      bfnew::validate_batched_expansion_options(options) ==
          bfnew::BatchedExpansionOptionsError::invalid_planner_policy,
      "invalid overlap planner policy is rejected");

  const std::array schedules = schedule_matrix();
  std::array<bfnew::ExpansionScheduleEvidence, 4U> evidence{};
  for (std::size_t index = 0U; index < evidence.size(); ++index) {
    evidence[index].schedule = schedules[index];
    evidence[index].input_queries = 10U;
    evidence[index].reached_queries = 10U;
    evidence[index].terminal_failures = 0U;
    evidence[index].batches_executed = 5U + index;
    evidence[index].total_expansions = 8U + index;
    evidence[index].attempted_selected_edge_estimate = 100U + index;
    evidence[index].work_evidence = bfnew::ExpansionWorkEvidence::measured;
    evidence[index].shared_edge_work = 80U - index;
    evidence[index].logical_lane_edge_work = 120U - index;
    evidence[index].comparison_fingerprint = 42U;
    evidence[index].campaign_valid = true;
  }
  const std::optional<bfnew::ExpansionSchedulePolicy> selected =
      bfnew::select_expansion_schedule_from_evidence(evidence);
  expect(
      selected.has_value() &&
          selected->kind ==
              bfnew::ExpansionScheduleKind::hybrid_small_then_doubling,
      "a schedule recommendation is selected from complete comparable evidence");
  expect(
      !bfnew::select_expansion_schedule_from_evidence(
           std::span<const bfnew::ExpansionScheduleEvidence>{evidence}.first(3U))
           .has_value(),
      "incomplete four-schedule evidence cannot create a default");
  evidence[3U].schedule = evidence[2U].schedule;
  expect(
      !bfnew::select_expansion_schedule_from_evidence(evidence).has_value(),
      "duplicate schedule evidence cannot create a default");

  evidence[3U].schedule = schedules[3U];
  evidence[3U].comparison_fingerprint = 43U;
  expect(
      !bfnew::select_expansion_schedule_from_evidence(evidence).has_value(),
      "mismatched workload/configuration fingerprints are not comparable");
  evidence[3U].comparison_fingerprint = 42U;
  evidence[3U].work_evidence =
      static_cast<bfnew::ExpansionWorkEvidence>(255U);
  expect(
      !bfnew::select_expansion_schedule_from_evidence(evidence).has_value(),
      "invalid schedule work evidence cannot create a default");
  evidence[3U].work_evidence = bfnew::ExpansionWorkEvidence::measured;

  evidence[3U].campaign_valid = false;
  expect(
      !bfnew::select_expansion_schedule_from_evidence(evidence).has_value(),
      "engine or identity failures cannot select a schedule");
  evidence[3U].campaign_valid = true;
  evidence[3U].batches_executed = 0U;
  expect(
      !bfnew::select_expansion_schedule_from_evidence(evidence).has_value(),
      "nonempty campaign evidence requires an executed batch");
  evidence[3U].batches_executed = 8U;
  evidence[3U].work_evidence = bfnew::ExpansionWorkEvidence::unavailable;
  evidence[3U].shared_edge_work = 1U;
  expect(
      !bfnew::select_expansion_schedule_from_evidence(evidence).has_value(),
      "unavailable schedule evidence cannot carry numeric work");
  evidence[3U].work_evidence = bfnew::ExpansionWorkEvidence::measured;
  evidence[3U].shared_edge_work = 9U;
  evidence[3U].logical_lane_edge_work = 8U;
  expect(
      !bfnew::select_expansion_schedule_from_evidence(evidence).has_value(),
      "impossible measured work cannot select a schedule");

  for (std::size_t index = 0U; index < evidence.size(); ++index) {
    evidence[index].schedule = schedules[index];
    evidence[index].input_queries = 10U;
    evidence[index].reached_queries = 10U;
    evidence[index].terminal_failures = 0U;
    evidence[index].batches_executed = 5U;
    evidence[index].total_expansions = 8U;
    evidence[index].attempted_selected_edge_estimate = 100U;
    evidence[index].work_evidence = bfnew::ExpansionWorkEvidence::measured;
    evidence[index].shared_edge_work = 80U;
    evidence[index].logical_lane_edge_work = 120U;
    evidence[index].comparison_fingerprint = 42U;
    evidence[index].campaign_valid = true;
  }
  expect(
      !bfnew::select_expansion_schedule_from_evidence(evidence).has_value(),
      "an exact evidence-score tie does not create an enum-order default");

  bfnew::BatchedExpansionRunResult overflowing_expansion_total;
  overflowing_expansion_total.metrics.input_queries = 1U;
  overflowing_expansion_total.metrics.reached_queries = 1U;
  overflowing_expansion_total.metrics.batches_executed = 1U;
  overflowing_expansion_total.metrics.scheduled_expansions =
      std::numeric_limits<std::uint64_t>::max();
  overflowing_expansion_total.metrics.full_region_fallbacks = 1U;
  overflowing_expansion_total.metrics.schedule_comparison_fingerprint = 42U;
  const bfnew::ExpansionScheduleEvidence overflowing_evidence =
      bfnew::make_expansion_schedule_evidence(
          schedules.front(), overflowing_expansion_total);
  expect(
      !overflowing_evidence.campaign_valid &&
          overflowing_evidence.total_expansions ==
              std::numeric_limits<std::uint64_t>::max(),
      "overflowing aggregate expansion evidence saturates and is unselectable");

  const bfnew::BatchedExpansionRunResult zero_query_run;
  for (std::size_t index = 0U; index < evidence.size(); ++index) {
    evidence[index] = bfnew::make_expansion_schedule_evidence(
        schedules[index], zero_query_run);
  }
  expect(
      !bfnew::select_expansion_schedule_from_evidence(evidence).has_value(),
      "zero-query evidence cannot select a production schedule");
}

void test_unselected_spill_enters_only_through_real_adjacency() {
  const bfnew::ResourceClassId resource{1U};
  const std::vector<bfnew::VertexMetadata> vertices{
      bfnew::VertexMetadata::located(0, 0, resource),
      bfnew::VertexMetadata::located(20, 0, resource),
      bfnew::VertexMetadata::located(0, 20, resource),
      bfnew::VertexMetadata::unlocated(resource),
      bfnew::VertexMetadata::located(0, 10, resource),
  };
  const bfnew::InputGraph input{
      vertices,
      {
          bfnew::test::jacobi_edge(0U, 2U, 1.0F, 17'100U),
          bfnew::test::jacobi_edge(2U, 3U, 1.0F, 17'101U),
          bfnew::test::jacobi_edge(3U, 1U, 1.0F, 17'102U),
      }};
  const bfnew::UniformGridPartitioner partitioner{
      bfnew::SpatialOrderConfig{0, 0, 10U, 10U}};
  const bfnew::PartitionedGraph partitioned = partitioner.partition(input);
  const bfnew::WeightedGraph& graph = partitioned.graph;
  const std::span<const bfnew::VertexId> old_to_new = graph.old_to_new();
  const std::array sources{old_to_new[0U]};
  const std::array targets{old_to_new[1U]};
  const bfnew::RouteQuery query =
      bfnew::make_route_query(bfnew::QueryId{1799U}, graph, sources, targets);
  const bfnew::TileDirectory directory = bfnew::build_tile_directory(graph);
  const bfnew::TileRunLayout64 tile_runs = bfnew::build_tile_run_layout(graph);
  const bfnew::DeviceGraphLayout32 device_graph =
      bfnew::build_device_graph_layout32(graph, tile_runs);
  const bfnew::TileId spill = directory.spill_tile();
  expect(
      !std::binary_search(
          query.selected_tiles.begin(), query.selected_tiles.end(), spill),
      "located terminals do not preselect the spill tile");

  const std::array queries{query};
  const bfnew::BatchedExpansionOptions options = expansion_options(
      bfnew::EngineKind::jacobi_pull,
      bfnew::one_ring_expansion(),
      2U,
      bfnew::ExpansionTerminalPolicy::explicit_failure);
  const bfnew::BatchedExpansionRunResult run =
      bfnew::run_host_batched_expansion(
          graph,
          directory,
          tile_runs,
          device_graph,
          queries,
          options);
  const bfnew::ExpansionQueryOutcome& outcome = run.queries.front();
  expect(
      outcome.reached() && outcome.scheduled_expansions == 2U &&
          !outcome.used_full_region_fallback &&
          std::binary_search(
              outcome.final_query.selected_tiles.begin(),
              outcome.final_query.selected_tiles.end(),
              spill) &&
          outcome.final_distances.size() == graph.vertex_count() &&
          bitwise_equal(
              outcome.final_distances[targets.front().value()], 3.0F),
      "an initially absent spill enters through the selected intermediate's "
      "real tile adjacency and enables the bounded route");
}

}  // namespace

int main() {
  test_all_schedules_and_engines();
  test_fallback_failure_and_single_miss();
  test_restart_identity_determinism_and_representation();
  test_generic_status_errors_and_restart_contract();
  test_multisource_generation_overflow_and_retry_work();
  test_option_validation_and_schedule_selection();
  test_unselected_spill_enters_only_through_real_adjacency();

  if (failures != 0) {
    std::cerr << failures << " Phase 17 batched expansion assertion(s) failed\n";
    return 1;
  }
  std::cout << "Phase 17 batched expansion assertions passed\n";
  return 0;
}
