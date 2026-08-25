#include "bfnew/hip/batched_expansion.hpp"

#include "batched_expansion_fixture_suite.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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

[[nodiscard]] bfnew::BatchedExpansionOptions options_for(
    const bfnew::EngineKind engine,
    const bfnew::ExpansionSchedulePolicy schedule,
    const bfnew::InstrumentationLevel instrumentation =
        bfnew::InstrumentationLevel::none,
    const std::uint64_t maximum_rounds = 64U,
    const std::uint32_t maximum_expansions = 4U,
    const bfnew::ControlMode control =
        bfnew::ControlMode::per_round_host_poll) {
  bfnew::BatchedExpansionOptions options;
  options.run_options.engine = engine;
  options.run_options.control_mode = control;
  options.run_options.rounds_per_chunk = 1U;
  options.run_options.block_size = 128U;
  options.run_options.instrumentation = instrumentation;
  options.run_options.maximum_rounds = maximum_rounds;
  options.run_options.enable_per_lane_convergence = 1U;
  options.planner_policy.lane_width = 8U;
  options.planner_policy.minimum_jaccard_numerator = 0U;
  options.planner_policy.minimum_jaccard_denominator = 1U;
  options.planner_policy.maximum_union_inflation_numerator = 32U;
  options.planner_policy.maximum_union_inflation_denominator = 1U;
  options.execution_configuration_fingerprint = 0x1700'0000'0000'0002ULL;
  options.schedule = schedule;
  options.maximum_expansions = maximum_expansions;
  options.terminal_policy =
      bfnew::ExpansionTerminalPolicy::full_region_fallback;
  return options;
}

[[nodiscard]] std::array<bfnew::ExpansionSchedulePolicy, 4U>
schedules() {
  return {
      bfnew::one_ring_expansion(),
      bfnew::fixed_ring_expansion(2U),
      bfnew::doubling_margin_expansion(),
      bfnew::hybrid_margin_expansion(2U),
  };
}

[[nodiscard]] const bfnew::ExpansionQueryOutcome& outcome_for(
    const bfnew::BatchedExpansionRunResult& run,
    const bfnew::QueryId query_id) {
  const auto position = std::lower_bound(
      run.queries.begin(),
      run.queries.end(),
      query_id,
      [](const bfnew::ExpansionQueryOutcome& outcome,
         const bfnew::QueryId id) {
        return outcome.final_query.query_id < id;
      });
  if (position == run.queries.end() ||
      position->final_query.query_id != query_id) {
    throw std::logic_error{"deferred HIP expansion outcome is missing"};
  }
  return *position;
}

[[nodiscard]] std::uint32_t expected_expansions(
    const bfnew::ExpansionScheduleKind schedule,
    const bfnew::QueryId query_id) {
  if (query_id == bfnew::test::phase17_immediate_query_id) {
    return 0U;
  }
  if (query_id == bfnew::test::phase17_unreachable_query_id) {
    return 1U;
  }
  const bool two_ring_shape =
      query_id == bfnew::test::phase17_two_ring_query_id ||
      query_id == bfnew::test::phase17_multi_source_query_id;
  switch (schedule) {
    case bfnew::ExpansionScheduleKind::one_geometric_ring:
      if (two_ring_shape ||
          query_id == bfnew::test::phase17_spill_query_id) {
        return 2U;
      }
      return query_id == bfnew::test::phase17_long_edge_query_id ? 4U : 3U;
    case bfnew::ExpansionScheduleKind::fixed_larger_ring:
      if (two_ring_shape ||
          query_id == bfnew::test::phase17_spill_query_id) {
        return 1U;
      }
      return 2U;
    case bfnew::ExpansionScheduleKind::doubling_xy_margins:
    case bfnew::ExpansionScheduleKind::hybrid_small_then_doubling:
      if (two_ring_shape ||
          query_id == bfnew::test::phase17_spill_query_id) {
        return 2U;
      }
      return 3U;
    case bfnew::ExpansionScheduleKind::unspecified:
      break;
  }
  throw std::logic_error{"unexpected deferred HIP expansion case"};
}

[[nodiscard]] bfnew::ExpansionBatchRunner bind_runner(
    bfnew::hip::BatchedExpansionExecutor& executor) {
  return [&executor](
             const std::span<const bfnew::RouteQuery> queries,
             const std::span<const bfnew::BatchQueryFeatures> features,
             const bfnew::BatchPlanEntry& batch,
             const bfnew::ExpansionBatchContext& context) {
    return executor(queries, features, batch, context);
  };
}

[[nodiscard]] bfnew::BatchedExpansionRunResult run_with(
    const bfnew::test::BatchedExpansionFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchedExpansionOptions& options,
    bfnew::hip::BatchedExpansionExecutor& executor) {
  const bfnew::ExpansionBatchRunner runner = bind_runner(executor);
  return bfnew::run_batched_expansion(
      fixture.partitioned.graph,
      fixture.directory,
      fixture.tile_runs,
      queries,
      options,
      runner);
}

void validate_compact_run(
    const bfnew::test::BatchedExpansionFixture& fixture,
    const bfnew::BatchedExpansionRunResult& run,
    const bfnew::ExpansionScheduleKind schedule,
    const std::string& prefix) {
  expect(
      run.queries.size() == fixture.queries.size() &&
          run.metrics.input_queries == fixture.queries.size() &&
          run.metrics.initial_reached_queries == 1U &&
          run.metrics.reached_queries == 6U &&
          run.metrics.unreachable_full_region_queries == 1U &&
          run.metrics.engine_failure_queries == 0U,
      prefix + ": compact all-query dispositions are exact");
  expect(
      run.metrics.work_evidence ==
              bfnew::ExpansionWorkEvidence::unavailable &&
          run.metrics.work_measured_batches == 0U &&
          run.metrics.shared_edge_work == 0U &&
          run.metrics.logical_lane_edge_work == 0U,
      prefix + ": compact execution does not manufacture work evidence");
  expect(
      !run.trace.empty() && run.trace.front().valid_lane_mask == 0x7fU &&
          run.trace.front().reached_lane_mask == 0x1U &&
          run.trace.front().miss_lane_mask == 0x7eU,
      prefix + ": GPU compact status partitions multiple initial misses");

  for (const bfnew::RouteQuery& original : fixture.queries) {
    const bfnew::ExpansionQueryOutcome& outcome =
        outcome_for(run, original.query_id);
    const std::uint32_t expansions =
        expected_expansions(schedule, original.query_id);
    expect(
        outcome.attempts == expansions + 1U &&
            outcome.total_expansions == expansions &&
            outcome.final_query.expansion_generation == expansions &&
            outcome.final_distances.empty(),
        prefix + ": retry generation and status-only result shape are exact");
    expect(
        outcome.final_query.query_id == original.query_id &&
            outcome.final_query.sources == original.sources &&
            outcome.final_query.targets == original.targets &&
            outcome.final_query.source_terminals == original.source_terminals &&
            outcome.final_query.target_terminals == original.target_terminals,
        prefix + ": query identity and original terminals survive replanning");
    if (original.query_id == bfnew::test::phase17_unreachable_query_id) {
      expect(
          outcome.disposition ==
                  bfnew::ExpansionQueryDisposition::unreachable_in_full_region &&
              outcome.used_full_region_fallback,
          prefix + ": unreachable query terminates after one final fallback");
    } else {
      expect(
          outcome.reached() && !outcome.used_full_region_fallback,
          prefix + ": routable query reaches without unnecessary fallback");
    }
  }

  const auto& north =
      outcome_for(run, bfnew::test::phase17_two_ring_query_id);
  const auto& south =
      outcome_for(run, bfnew::test::phase17_second_miss_query_id);
  expect(
      north.final_query.selected_tiles != south.final_query.selected_tiles,
      prefix + ": dissimilar expanded lanes retain distinct regions");
}

template <typename Workspace>
void exercise_engine(
    const bfnew::test::BatchedExpansionFixture& fixture,
    const bfnew::hip::ResidentDeviceGraph& resident,
    const bfnew::hip::HipStream& stream,
    Workspace& workspace,
    const bfnew::EngineKind engine,
    const bool exercise_single_miss) {
  bfnew::BatchedExpansionOptions compact_options =
      options_for(engine, bfnew::one_ring_expansion());
  bfnew::hip::BatchedExpansionExecutor compact_executor{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream,
      compact_options.run_options,
      bfnew::hip::BatchedExpansionTransferMode::compact_status};

  std::uint64_t allocation_events_after_first = 0U;
  for (std::size_t index = 0U; index < schedules().size(); ++index) {
    const bfnew::ExpansionSchedulePolicy schedule = schedules()[index];
    compact_options.schedule = schedule;
    const bfnew::BatchedExpansionRunResult run =
        run_with(fixture, fixture.queries, compact_options, compact_executor);
    const std::string prefix =
        "engine-" + std::to_string(static_cast<std::uint32_t>(engine)) +
        "-schedule-" +
        std::to_string(static_cast<std::uint32_t>(schedule.kind));
    validate_compact_run(fixture, run, schedule.kind, prefix);
    if (index == 0U) {
      allocation_events_after_first = workspace.capacity().allocation_events;
      expect(
          allocation_events_after_first != 0U,
          prefix + ": first run established reusable device capacity");
    }
  }
  expect(
      workspace.capacity().allocation_events == allocation_events_after_first,
      "all compact schedules reuse the warmed engine workspace");

  bfnew::BatchedExpansionOptions evidence_options = options_for(
      engine,
      bfnew::one_ring_expansion(),
      bfnew::InstrumentationLevel::debug);
  bfnew::hip::BatchedExpansionExecutor evidence_executor{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream,
      evidence_options.run_options,
      bfnew::hip::BatchedExpansionTransferMode::status_and_work_evidence};
  const bfnew::BatchedExpansionRunResult evidence =
      run_with(fixture, fixture.queries, evidence_options, evidence_executor);
  expect(
      evidence.metrics.work_evidence ==
              bfnew::ExpansionWorkEvidence::measured &&
          evidence.metrics.work_measured_batches ==
              evidence.metrics.batches_executed &&
          evidence.metrics.logical_lane_edge_work >=
              evidence.metrics.shared_edge_work &&
          evidence.metrics.logical_lane_edge_work != 0U,
      "evidence transfer reports measured shared/logical work");
  expect(
      workspace.capacity().allocation_events == allocation_events_after_first,
      "evidence replay reuses compact-warmed capacity");

  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::hip::BatchedExpansionExecutor rejected{
            fixture.partitioned.graph,
            fixture.tile_runs,
            resident,
            workspace,
            stream,
            evidence_options.run_options,
            bfnew::hip::BatchedExpansionTransferMode::compact_status};
        static_cast<void>(rejected);
      },
      "instrumented execution cannot masquerade as compact transfer");

  bfnew::BatchedExpansionOptions error_options = options_for(
      engine, bfnew::one_ring_expansion(), bfnew::InstrumentationLevel::none, 1U);
  bfnew::hip::BatchedExpansionExecutor error_executor{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream,
      error_options.run_options,
      bfnew::hip::BatchedExpansionTransferMode::compact_status};
  const std::array one_query{
      bfnew::test::expansion_query(
          fixture, bfnew::test::phase17_immediate_query_id)};
  const bfnew::BatchedExpansionRunResult error_run =
      run_with(fixture, one_query, error_options, error_executor);
  expect(
      error_run.metrics.engine_failure_queries == 1U &&
          error_run.metrics.scheduled_expansions == 0U &&
          error_run.metrics.full_region_fallbacks == 0U &&
          error_run.queries.front().attempts == 1U &&
          error_run.queries.front().total_expansions == 0U &&
          error_run.queries.front().disposition ==
              bfnew::ExpansionQueryDisposition::engine_failure,
      "maximum-round engine exit never becomes a bounding miss or expansion");

  bfnew::BatchedExpansionOptions persistent_options = options_for(
      engine,
      bfnew::one_ring_expansion(),
      bfnew::InstrumentationLevel::none,
      64U,
      4U,
      bfnew::ControlMode::persistent_cooperative);
  bfnew::hip::BatchedExpansionExecutor persistent_executor{
      fixture.partitioned.graph,
      fixture.tile_runs,
      resident,
      workspace,
      stream,
      persistent_options.run_options,
      bfnew::hip::BatchedExpansionTransferMode::compact_status};
  const bfnew::BatchedExpansionRunResult persistent =
      run_with(fixture, one_query, persistent_options, persistent_executor);
  expect(
      persistent.metrics.reached_queries == 1U &&
          persistent.metrics.batches_executed == 1U &&
          persistent.metrics.work_evidence ==
              bfnew::ExpansionWorkEvidence::unavailable,
      "persistent executor completes through one final compact status");

  if (exercise_single_miss) {
    const std::array two_queries{
        bfnew::test::expansion_query(
            fixture, bfnew::test::phase17_immediate_query_id),
        bfnew::test::expansion_query(
            fixture, bfnew::test::phase17_long_edge_query_id),
    };
    bfnew::BatchedExpansionOptions single_options =
        options_for(engine, bfnew::one_ring_expansion());
    single_options.maximum_expansions = 1U;
    const bfnew::BatchedExpansionRunResult single =
        run_with(fixture, two_queries, single_options, compact_executor);
    const auto& fallback =
        outcome_for(single, bfnew::test::phase17_long_edge_query_id);
    expect(
        single.metrics.initial_reached_queries == 1U &&
            single.metrics.full_region_fallbacks == 1U &&
            fallback.reached() && fallback.scheduled_expansions == 1U &&
            fallback.total_expansions == 2U &&
            std::ranges::any_of(
                single.trace,
                [](const bfnew::ExpansionBatchTrace& trace) {
                  return trace.context.retry_pass &&
                         trace.valid_lane_mask == 1U;
                }),
        "one failed lane is compacted, expanded, and finally restarted alone");
  }
}

void test_all_engines() {
  const bfnew::test::BatchedExpansionFixture fixture =
      bfnew::test::make_batched_expansion_fixture();
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(fixture.device_graph), stream);

  bfnew::hip::ReusableBatchedJacobiWorkspace jacobi_workspace;
  exercise_engine(
      fixture,
      resident,
      stream,
      jacobi_workspace,
      bfnew::EngineKind::jacobi_pull,
      false);

  bfnew::hip::ReusableBatchedDenseWorkspace dense_workspace;
  exercise_engine(
      fixture,
      resident,
      stream,
      dense_workspace,
      bfnew::EngineKind::dense_chaotic_push,
      false);

  bfnew::hip::ReusableBatchedFrontierWorkspace frontier_workspace;
  exercise_engine(
      fixture,
      resident,
      stream,
      frontier_workspace,
      bfnew::EngineKind::frontier_push,
      true);
}

}  // namespace

int main() {
  try {
    test_all_engines();
  } catch (const std::exception& error) {
    std::cerr << "deferred HIP batched expansion test threw: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
  if (failures != 0) {
    std::cerr << failures << " deferred HIP batched expansion checks failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "deferred HIP batched expansion checks passed\n";
  return EXIT_SUCCESS;
}
