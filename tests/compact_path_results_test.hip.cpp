#include "bfnew/compact_paths.hpp"
#include "bfnew/hip/batched_expansion.hpp"

#include "compact_paths_fixture_suite.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;
inline constexpr bfnew::QueryId mixed_target_query_id{1805U};

void expect(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAILED: " << description << '\n';
    ++failures;
  }
}

[[nodiscard]] bool same_status(
    const bfnew::DeviceRunStatus& left,
    const bfnew::DeviceRunStatus& right) noexcept {
  return left.final_distance_slot == right.final_distance_slot &&
         left.converged == right.converged &&
         left.rounds_completed == right.rounds_completed &&
         left.reached_target_mask == right.reached_target_mask &&
         left.bounding_box_miss_mask == right.bounding_box_miss_mask &&
         left.valid_lane_mask == right.valid_lane_mask &&
         left.active_lane_mask == right.active_lane_mask &&
         left.converged_lane_mask == right.converged_lane_mask &&
         left.stop_reason == right.stop_reason &&
         left.error_bits == right.error_bits;
}

[[nodiscard]] bool transport_is_consistent(
    const bfnew::hip::CompactPathBatchOutput& output) noexcept {
  const std::uint64_t maximum =
      std::numeric_limits<std::uint64_t>::max();
  if (output.transfer.total_bytes > maximum - output.transport.status_bytes) {
    return false;
  }
  const std::uint64_t payload_and_status =
      output.transfer.total_bytes + output.transport.status_bytes;
  if (payload_and_status > maximum - output.transport.error_bytes) {
    return false;
  }
  const std::uint64_t compact_total =
      payload_and_status + output.transport.error_bytes;
  if (compact_total != output.transport.total_device_to_host_bytes ||
      output.transport.controller_poll_count >
          maximum / sizeof(bfnew::DeviceController)) {
    return false;
  }
  const std::uint64_t poll_bytes =
      output.transport.controller_poll_count *
      sizeof(bfnew::DeviceController);
  return poll_bytes == output.transport.controller_poll_bytes &&
         compact_total <= maximum - poll_bytes &&
         compact_total + poll_bytes ==
             output.transport.overall_device_to_host_bytes;
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

[[nodiscard]] const bfnew::RouteQuery& query_for(
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::QueryId query_id) {
  const auto position = std::find_if(
      queries.begin(), queries.end(), [query_id](const bfnew::RouteQuery& query) {
        return query.query_id == query_id;
      });
  if (position == queries.end()) {
    throw std::logic_error{"Phase 18 query is missing"};
  }
  return *position;
}

[[nodiscard]] const bfnew::CompactPathPayload& payload_for(
    const std::span<const bfnew::CompactPathPayload> payloads,
    const bfnew::QueryId query_id) {
  const auto position = std::find_if(
      payloads.begin(),
      payloads.end(),
      [query_id](const bfnew::CompactPathPayload& payload) {
        return payload.query_id == query_id;
      });
  if (position == payloads.end()) {
    throw std::logic_error{"Phase 18 compact payload is missing"};
  }
  return *position;
}

[[nodiscard]] const bfnew::CompactTargetPath& target_for(
    const bfnew::CompactPathPayload& payload,
    const bfnew::VertexId target) {
  const auto position = std::find_if(
      payload.targets.begin(),
      payload.targets.end(),
      [target](const bfnew::CompactTargetPath& path) {
        return path.summary.target == target;
      });
  if (position == payload.targets.end()) {
    throw std::logic_error{"Phase 18 compact target is missing"};
  }
  return *position;
}

[[nodiscard]] std::vector<bfnew::CompactPathPayload> clean_payloads(
    const bfnew::hip::CompactPathBatchOutput& output,
    const bfnew::BatchDeviceDescription& description,
    const std::span<const bfnew::RouteQuery> queries) {
  if (output.targets.size() != description.targets.size()) {
    throw std::logic_error{"compact target flattening changed shape"};
  }
  std::vector<bfnew::CompactPathPayload> payloads;
  for (std::uint32_t lane = 0U; lane < description.lane_width; ++lane) {
    const bfnew::LaneMask bit = bfnew::LaneMask{1U} << lane;
    if ((description.valid_lane_mask & bit) == 0U) {
      continue;
    }
    const bfnew::QueryId query_id{description.query_ids_by_lane[lane]};
    const bfnew::RouteQuery& query = query_for(queries, query_id);
    const std::size_t begin = description.target_offsets[lane];
    const std::size_t end = description.target_offsets[lane + 1U];
    if (end < begin || end > output.targets.size()) {
      throw std::logic_error{"compact target slice is invalid"};
    }
    bfnew::CompactPathPayload payload;
    payload.query_id = query_id;
    payload.expansion_generation =
        description.expansion_generations_by_lane[lane];
    payload.target_terminal_to_target = query.target_terminal_to_target;
    payload.targets.assign(
        output.targets.begin() + static_cast<std::ptrdiff_t>(begin),
        output.targets.begin() + static_cast<std::ptrdiff_t>(end));
    payloads.push_back(std::move(payload));
  }
  std::sort(
      payloads.begin(),
      payloads.end(),
      [](const bfnew::CompactPathPayload& left,
         const bfnew::CompactPathPayload& right) {
        return left.query_id < right.query_id;
      });
  return payloads;
}

void validate_phase5_paths(
    const bfnew::test::CompactPathFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchDeviceDescription& description,
    const bfnew::hip::CompactPathBatchOutput& output,
    const std::string& prefix) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const std::span<const bfnew::VertexId> map = graph.old_to_new();
  const std::vector<bfnew::CompactPathPayload> payloads =
      clean_payloads(output, description, queries);
  expect(
      payloads.size() == 6U,
      prefix + ": every clean valid lane publishes target results");
  for (const bfnew::CompactPathPayload& payload : payloads) {
    const bfnew::RouteQuery& query = query_for(queries, payload.query_id);
    expect(
        bfnew::validate_compact_path_payload(graph, query, payload).ok(),
        prefix +
            ": source/target/continuity/tightness/cost/region validation passes");
    for (const bfnew::CompactTargetPath& target : payload.targets) {
      expect(
          target.summary.reconstruction != bfnew::CompactPathStatus::complete ||
              (target.vertices.size() == target.distance_labels.size() &&
               target.edge_ids.size() + 1U == target.vertices.size()),
          prefix + ": every compact edge has actual aligned GPU labels");
    }
  }

  const bfnew::CompactTargetPath& cycle =
      payload_for(payloads, bfnew::test::phase18_cycle_query_id)
          .targets.front();
  expect(
      cycle.vertices ==
              std::vector<bfnew::VertexId>{map[3U], map[1U], map[2U]} &&
          cycle.distance_labels == std::vector<float>{0.0F, 1.0F, 2.0F},
      prefix + ": stable traversal truly backtracks out of the zero cycle");

  const bfnew::CompactTargetPath& tie =
      payload_for(payloads, bfnew::test::phase18_tie_query_id).targets.front();
  std::vector<bfnew::EdgeId> parallel_ids;
  const bfnew::OutgoingCsrView outgoing = graph.outgoing();
  const bfnew::EdgeOffset parallel_begin =
      outgoing.row_offsets[map[5U].value()];
  const bfnew::EdgeOffset parallel_end =
      outgoing.row_offsets[map[5U].value() + 1U];
  for (bfnew::EdgeOffset position = parallel_begin;
       position < parallel_end;
       ++position) {
    if (outgoing.destinations[position] == map[7U] &&
        outgoing.weights[position] == 0.5F) {
      parallel_ids.push_back(outgoing.edge_ids[position]);
    }
  }
  expect(
      tie.vertices ==
              std::vector<bfnew::VertexId>{map[4U], map[5U], map[7U]} &&
          parallel_ids.size() == 2U && !tie.edge_ids.empty() &&
          tie.edge_ids.back() ==
              *std::min_element(parallel_ids.begin(), parallel_ids.end()),
      prefix + ": equal parallel edges choose the smaller stable EdgeId");

  const bfnew::CompactTargetPath& bounded =
      payload_for(payloads, bfnew::test::phase18_bounded_query_id)
          .targets.front();
  expect(
      bounded.vertices ==
          std::vector<bfnew::VertexId>{map[8U], map[10U], map[9U]},
      prefix + ": reconstruction remains inside the selected induced region");

  const bfnew::CompactPathPayload& multisource =
      payload_for(payloads, bfnew::test::phase18_multisource_query_id);
  const bfnew::CompactTargetPath& source_target =
      target_for(multisource, map[12U]);
  const bfnew::CompactTargetPath& routed_target =
      target_for(multisource, map[14U]);
  expect(
      source_target.summary.path_length == 0U &&
          source_target.summary.selected_source == map[12U] &&
          source_target.vertices == std::vector<bfnew::VertexId>{map[12U]} &&
          source_target.distance_labels == std::vector<float>{0.0F} &&
          source_target.edge_ids.empty(),
      prefix + ": a source target emits one label and zero edges");
  expect(
      routed_target.summary.selected_source == map[13U] &&
          routed_target.vertices ==
              std::vector<bfnew::VertexId>{map[13U], map[14U]} &&
          routed_target.distance_labels == std::vector<float>{0.0F, 1.0F},
      prefix + ": canonical multi-source selection is preserved");

  const bfnew::CompactPathPayload& mixed =
      payload_for(payloads, mixed_target_query_id);
  const bfnew::CompactTargetPath& mixed_complete =
      target_for(mixed, map[14U]);
  const bfnew::CompactTargetPath& mixed_unreachable =
      target_for(mixed, map[15U]);
  expect(
      mixed_complete.summary.reached ==
              bfnew::CompactTargetReachStatus::reached &&
          mixed_complete.summary.reconstruction ==
              bfnew::CompactPathStatus::complete &&
          mixed_complete.summary.selected_source == map[13U] &&
          mixed_complete.vertices ==
              std::vector<bfnew::VertexId>{map[13U], map[14U]} &&
          mixed_complete.distance_labels ==
              std::vector<float>{0.0F, 1.0F} &&
          mixed_unreachable.summary.reached ==
              bfnew::CompactTargetReachStatus::not_reached &&
          mixed_unreachable.summary.reconstruction ==
              bfnew::CompactPathStatus::unreachable &&
          mixed_unreachable.vertices.empty() &&
          mixed_unreachable.distance_labels.empty() &&
          mixed_unreachable.edge_ids.empty(),
      prefix + ": one miss lane preserves its finite and infinite targets");

  const auto unreachable_lane = std::find(
      description.query_ids_by_lane.begin(),
      description.query_ids_by_lane.end(),
      bfnew::test::phase18_unreachable_query_id.value());
  if (unreachable_lane == description.query_ids_by_lane.end()) {
    throw std::logic_error{"unreachable lane is missing"};
  }
  const std::size_t lane = static_cast<std::size_t>(
      unreachable_lane - description.query_ids_by_lane.begin());
  const bfnew::CompactTargetPath& unreachable =
      output.targets[description.target_offsets[lane]];
  expect(
      unreachable.summary.reached ==
              bfnew::CompactTargetReachStatus::not_reached &&
          unreachable.summary.reconstruction ==
              bfnew::CompactPathStatus::unreachable &&
          unreachable.summary.has_selected_source == 0U &&
          unreachable.vertices.empty() && unreachable.distance_labels.empty() &&
          unreachable.edge_ids.empty(),
      prefix + ": miss has an explicit summary and no invented path arena");

  const bfnew::CompactTransferAccounting published =
      bfnew::measure_compact_transfer(payloads);
  expect(
      output.transfer.summary_bytes ==
              description.targets.size() *
                  sizeof(bfnew::CompactTargetSummary) &&
      output.transfer.summary_bytes == published.summary_bytes &&
          output.transfer.vertex_bytes == published.vertex_bytes &&
          output.transfer.distance_label_bytes ==
              published.distance_label_bytes &&
          output.transfer.edge_id_bytes == published.edge_id_bytes &&
          output.transport.status_bytes == sizeof(bfnew::DeviceRunStatus) &&
          output.transport.error_bytes == 2U * sizeof(std::uint32_t) &&
          transport_is_consistent(output) &&
          output.transport.controller_poll_count != 0U &&
          output.transport.controller_poll_bytes ==
              output.transport.controller_poll_count *
                  sizeof(bfnew::DeviceController),
      prefix + ": payload bytes and exact total D2H envelope stay distinct");
  expect(
      output.metrics.sssp_device_milliseconds >= 0.0F &&
          output.metrics.reconstruction_device_milliseconds >= 0.0F &&
          output.metrics.result_transfer_device_milliseconds >= 0.0F &&
          output.metrics.end_to_end_wall_milliseconds >= 0.0,
      prefix + ": all required stage timings are reported");
}

[[nodiscard]] bfnew::GpuRunOptions gpu_options(
    const bfnew::EngineKind engine,
    const std::uint64_t maximum_rounds = 64U) {
  bfnew::GpuRunOptions options;
  options.engine = engine;
  options.control_mode = bfnew::ControlMode::per_round_host_poll;
  options.rounds_per_chunk = 1U;
  options.block_size = 128U;
  options.instrumentation = bfnew::InstrumentationLevel::none;
  options.maximum_rounds = maximum_rounds;
  options.enable_per_lane_convergence = 1U;
  return options;
}

[[nodiscard]] std::vector<bfnew::BatchDeviceDescription>
make_descriptions(
    const bfnew::WeightedGraph& graph,
    const bfnew::TileRunLayout64& tile_runs,
    const std::span<const bfnew::RouteQuery> queries,
    const std::uint32_t lane_width) {
  const bfnew::SelectedRegionIndex selected_regions{graph, tile_runs};
  const std::vector<bfnew::BatchQueryFeatures> features =
      bfnew::make_batch_query_features(graph, selected_regions, queries);
  bfnew::BatchPlannerPolicy policy;
  policy.lane_width = lane_width;
  policy.minimum_jaccard_numerator = 0U;
  policy.minimum_jaccard_denominator = 1U;
  policy.maximum_union_inflation_numerator = 64U;
  policy.maximum_union_inflation_denominator = 1U;
  const bfnew::BatchPlan plan =
      bfnew::make_overlapping_batch_plan(selected_regions, features, policy);
  std::vector<bfnew::BatchDeviceDescription> descriptions;
  descriptions.reserve(plan.batches.size());
  for (const bfnew::BatchPlanEntry& batch : plan.batches) {
    bfnew::BatchDeviceDescription description;
    bfnew::prepare_batch_device_description(
        graph,
        tile_runs,
        queries,
        features,
        batch,
        bfnew::BatchRunRepresentation::retained_per_run_masks,
        description);
    descriptions.push_back(std::move(description));
  }
  return descriptions;
}

[[nodiscard]] std::vector<bfnew::CompactPathPayload> validate_width_output(
    const bfnew::WeightedGraph& graph,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchDeviceDescription& description,
    const bfnew::hip::CompactPathBatchOutput& output,
    const std::string& prefix) {
  const bfnew::LaneMask classified = output.status.reached_target_mask |
                                     output.status.bounding_box_miss_mask;
  expect(
      output.status.stop_reason ==
              static_cast<std::uint32_t>(
                  bfnew::DeviceStopReason::converged) &&
          output.status.error_bits == bfnew::device_error::none &&
          output.status.valid_lane_mask == description.valid_lane_mask &&
          classified == description.valid_lane_mask &&
          (output.status.reached_target_mask &
           output.status.bounding_box_miss_mask) == 0U,
      prefix + ": compact status partitions every valid lane");
  std::vector<bfnew::CompactPathPayload> payloads =
      clean_payloads(output, description, queries);
  for (const bfnew::CompactPathPayload& payload : payloads) {
    expect(
        bfnew::validate_compact_path_payload(
            graph, query_for(queries, payload.query_id), payload)
            .ok(),
        prefix + ": every width preserves exact compact path semantics");
  }
  const bfnew::CompactTransferAccounting measured =
      bfnew::measure_compact_transfer(payloads);
  expect(
      output.transfer == measured &&
          output.transport.status_bytes == sizeof(bfnew::DeviceRunStatus) &&
          transport_is_consistent(output) &&
          output.transport.controller_poll_count != 0U,
      prefix + ": only summaries and compact arenas accompany control D2H");
  return payloads;
}

template <typename Engine>
void exercise_width_matrix(
    const bfnew::WeightedGraph& graph,
    const bfnew::TileRunLayout64& tile_runs,
    const std::span<const bfnew::RouteQuery> queries,
    Engine& engine,
    bfnew::hip::ReusableCompactPathWorkspace& compact_workspace,
    const bfnew::EngineKind engine_kind) {
  const std::array<std::uint32_t, 4U> widths{1U, 8U, 16U, 32U};
  std::vector<bfnew::CompactPathPayload> reference;
  for (const std::uint32_t width : widths) {
    const std::vector<bfnew::BatchDeviceDescription> descriptions =
        make_descriptions(graph, tile_runs, queries, width);
    expect(
        width == 1U ? descriptions.size() == queries.size()
                    : descriptions.size() == 1U,
        "W1 splits bounded queries while W8/W16/W32 retain padded batching");
    std::vector<bfnew::CompactPathPayload> observed;
    for (const bfnew::BatchDeviceDescription& description : descriptions) {
      const std::uint32_t valid_count = static_cast<std::uint32_t>(
          std::popcount(description.valid_lane_mask));
      const std::uint32_t expected_valid_count =
          width == 1U ? 1U : static_cast<std::uint32_t>(queries.size());
      const bfnew::LaneMask expected_mask =
          expected_valid_count == 32U
              ? std::numeric_limits<bfnew::LaneMask>::max()
              : (bfnew::LaneMask{1U} << expected_valid_count) - 1U;
      const bfnew::LaneMask width_mask =
          width == 32U
              ? std::numeric_limits<bfnew::LaneMask>::max()
              : (bfnew::LaneMask{1U} << width) - 1U;
      expect(
          description.lane_width == width &&
              valid_count == expected_valid_count &&
              description.valid_lane_mask == expected_mask &&
              (description.valid_lane_mask & ~width_mask) == 0U,
          "valid queries occupy the low lanes and padding remains invalid");
      const bfnew::hip::CompactPathBatchOutput output =
          engine.run_compact_paths(
              description, gpu_options(engine_kind), compact_workspace);
      std::vector<bfnew::CompactPathPayload> batch_payloads =
          validate_width_output(
              graph,
              queries,
              description,
              output,
              "engine-" +
                  std::to_string(static_cast<std::uint32_t>(engine_kind)) +
                  "-W" + std::to_string(width));
      observed.insert(
          observed.end(),
          std::make_move_iterator(batch_payloads.begin()),
          std::make_move_iterator(batch_payloads.end()));
    }
    std::sort(
        observed.begin(),
        observed.end(),
        [](const bfnew::CompactPathPayload& left,
           const bfnew::CompactPathPayload& right) {
          return left.query_id < right.query_id;
        });
    if (reference.empty()) {
      reference = observed;
    } else {
      expect(
          observed == reference,
          "W1/W8/W16/W32 produce identical canonical compact payloads");
    }
  }
}

template <typename Engine>
void exercise_representative_controls(
    const bfnew::BatchDeviceDescription& description,
    Engine& engine,
    bfnew::hip::ReusableCompactPathWorkspace& compact_workspace,
    const bfnew::EngineKind engine_kind,
    const bfnew::hip::CompactPathBatchOutput& per_round) {
  bfnew::GpuRunOptions chunked = gpu_options(engine_kind);
  chunked.control_mode = bfnew::ControlMode::chunked_host_poll;
  chunked.rounds_per_chunk = 2U;
  const bfnew::hip::CompactPathBatchOutput chunked_output =
      engine.run_compact_paths(description, chunked, compact_workspace);

  bfnew::GpuRunOptions persistent = gpu_options(engine_kind);
  persistent.control_mode = bfnew::ControlMode::persistent_cooperative;
  const bfnew::hip::CompactPathBatchOutput persistent_output =
      engine.run_compact_paths(description, persistent, compact_workspace);
  expect(
      same_status(per_round.status, chunked_output.status) &&
          same_status(per_round.status, persistent_output.status) &&
          per_round.targets == chunked_output.targets &&
          per_round.targets == persistent_output.targets &&
          per_round.transfer == chunked_output.transfer &&
          per_round.transfer == persistent_output.transfer &&
          per_round.transport.status_bytes ==
              chunked_output.transport.status_bytes &&
          per_round.transport.status_bytes ==
              persistent_output.transport.status_bytes &&
          per_round.transport.error_bytes ==
              chunked_output.transport.error_bytes &&
          per_round.transport.error_bytes ==
              persistent_output.transport.error_bytes &&
          per_round.transport.total_device_to_host_bytes ==
              chunked_output.transport.total_device_to_host_bytes &&
          per_round.transport.total_device_to_host_bytes ==
              persistent_output.transport.total_device_to_host_bytes &&
          transport_is_consistent(per_round) &&
          transport_is_consistent(chunked_output) &&
          transport_is_consistent(persistent_output) &&
          per_round.transport.controller_poll_count != 0U &&
          chunked_output.transport.controller_poll_count != 0U &&
          persistent_output.transport.controller_poll_count == 0U &&
          persistent_output.transport.controller_poll_bytes == 0U &&
          persistent_output.transport.overall_device_to_host_bytes ==
              persistent_output.transport.total_device_to_host_bytes,
      "controls preserve compact results while exposing exact poll traffic");
}

[[nodiscard]] bfnew::BatchedExpansionOptions expansion_options(
    const bfnew::EngineKind engine,
    const bfnew::ExpansionSchedulePolicy schedule,
    const std::uint64_t maximum_rounds = 64U) {
  bfnew::BatchedExpansionOptions options;
  options.run_options = gpu_options(engine, maximum_rounds);
  options.planner_policy.lane_width = 8U;
  options.planner_policy.minimum_jaccard_numerator = 0U;
  options.planner_policy.minimum_jaccard_denominator = 1U;
  options.planner_policy.maximum_union_inflation_numerator = 64U;
  options.planner_policy.maximum_union_inflation_denominator = 1U;
  options.execution_configuration_fingerprint = 0x1800'0000'0000'0018ULL;
  options.schedule = schedule;
  options.maximum_expansions = 4U;
  options.terminal_policy =
      bfnew::ExpansionTerminalPolicy::full_region_fallback;
  options.enable_compact_paths = 1U;
  return options;
}

[[nodiscard]] std::array<bfnew::ExpansionSchedulePolicy, 4U> schedules() {
  return {
      bfnew::one_ring_expansion(),
      bfnew::fixed_ring_expansion(2U),
      bfnew::doubling_margin_expansion(),
      bfnew::hybrid_margin_expansion(2U),
  };
}

[[nodiscard]] bfnew::BatchedExpansionRunResult run_expansion(
    const bfnew::test::CompactPathFixture& fixture,
    const bfnew::TileRunLayout64& tile_runs,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchedExpansionOptions& options,
    bfnew::hip::BatchedExpansionExecutor& executor) {
  const bfnew::ExpansionBatchRunner runner =
      [&executor](
          const std::span<const bfnew::RouteQuery> retry_queries,
          const std::span<const bfnew::BatchQueryFeatures> features,
          const bfnew::BatchPlanEntry& batch,
          const bfnew::ExpansionBatchContext& context) {
        return executor(retry_queries, features, batch, context);
      };
  return bfnew::run_batched_expansion(
      fixture.partitioned.graph,
      fixture.partitioned.tiles,
      tile_runs,
      queries,
      options,
      runner);
}

void validate_expansion_run(
    const bfnew::test::CompactPathFixture& fixture,
    const std::span<const bfnew::RouteQuery> queries,
    const bfnew::BatchedExpansionRunResult& run,
    const std::string& prefix) {
  expect(
      run.metrics.reached_queries == 4U &&
          run.metrics.unreachable_full_region_queries == 2U &&
          run.metrics.engine_failure_queries == 0U &&
          run.metrics.compact_device_timing ==
              bfnew::CompactStageTimingEvidence::measured &&
          run.metrics.compact_host_timing ==
              bfnew::CompactStageTimingEvidence::measured &&
          run.metrics.compact_device_timing_measured_batches ==
              run.metrics.batches_executed &&
          run.metrics.compact_host_timing_measured_batches ==
              run.metrics.batches_executed &&
          run.metrics.compact_transfer.total_bytes != 0U &&
          run.metrics.compact_controller_poll_count != 0U &&
          run.metrics.compact_controller_poll_bytes ==
              run.metrics.compact_controller_poll_count *
                  sizeof(bfnew::DeviceController) &&
          run.metrics.compact_overall_device_to_host_bytes ==
              run.metrics.compact_total_device_to_host_bytes +
                  run.metrics.compact_controller_poll_bytes,
      prefix + ": expansion aggregates compact timing and payload accounting");
  for (const bfnew::ExpansionQueryOutcome& outcome : run.queries) {
    const bfnew::RouteQuery& original =
        query_for(queries, outcome.final_query.query_id);
    if (original.query_id == bfnew::test::phase18_unreachable_query_id ||
        original.query_id == mixed_target_query_id) {
      expect(
          outcome.disposition ==
                  bfnew::ExpansionQueryDisposition::unreachable_in_full_region &&
              outcome.used_full_region_fallback &&
              outcome.compact_paths.has_value() &&
              bfnew::validate_compact_path_payload(
                  fixture.partitioned.graph,
                  outcome.final_query,
                  *outcome.compact_paths)
                  .ok(),
          prefix + ": final full-region miss retains explicit target results");
      if (original.query_id == mixed_target_query_id &&
          outcome.compact_paths.has_value()) {
        const std::span<const bfnew::VertexId> map =
            fixture.partitioned.graph.old_to_new();
        const bfnew::CompactTargetPath& complete =
            target_for(*outcome.compact_paths, map[14U]);
        const bfnew::CompactTargetPath& unreachable =
            target_for(*outcome.compact_paths, map[15U]);
        expect(
            complete.summary.reconstruction ==
                    bfnew::CompactPathStatus::complete &&
                unreachable.summary.reconstruction ==
                    bfnew::CompactPathStatus::unreachable,
            prefix +
                ": terminal executor retains mixed complete/unreachable targets");
      }
      continue;
    }
    expect(
        outcome.reached() && outcome.compact_paths.has_value(),
        prefix + ": reached generation retains its payload before reuse");
    if (outcome.compact_paths) {
      expect(
          outcome.compact_paths->query_id == outcome.final_query.query_id &&
              outcome.compact_paths->expansion_generation ==
                  outcome.final_query.expansion_generation &&
              bfnew::validate_compact_path_payload(
                  fixture.partitioned.graph,
                  outcome.final_query,
                  *outcome.compact_paths)
                  .ok(),
          prefix + ": identity/generation and compact validation are exact");
    }
  }
}

template <typename Workspace>
void exercise_executor(
    const bfnew::test::CompactPathFixture& fixture,
    const bfnew::TileRunLayout64& tile_runs,
    const bfnew::hip::ResidentDeviceGraph& resident,
    const bfnew::hip::HipStream& stream,
    const std::span<const bfnew::RouteQuery> queries,
    Workspace& workspace,
    const bfnew::EngineKind engine) {
  bfnew::BatchedExpansionOptions options =
      expansion_options(engine, schedules().front());
  bfnew::hip::BatchedExpansionExecutor executor{
      fixture.partitioned.graph,
      tile_runs,
      resident,
      workspace,
      stream,
      options.run_options,
      bfnew::hip::BatchedExpansionTransferMode::compact_paths};
  std::uint64_t warmed_allocations = 0U;
  for (std::size_t schedule = 0U; schedule < schedules().size(); ++schedule) {
    options.schedule = schedules()[schedule];
    const bfnew::BatchedExpansionRunResult run =
        run_expansion(fixture, tile_runs, queries, options, executor);
    validate_expansion_run(
        fixture,
        queries,
        run,
        "engine-" +
            std::to_string(static_cast<std::uint32_t>(engine)) +
            "-schedule-" + std::to_string(schedule));
    if (schedule == 0U) {
      warmed_allocations = workspace.capacity().allocation_events;
    }
  }
  expect(
      warmed_allocations != 0U &&
          workspace.capacity().allocation_events == warmed_allocations,
      "all Phase 18 schedules reuse the warmed engine workspace");

  bfnew::GpuRunOptions instrumented = options.run_options;
  instrumented.instrumentation = bfnew::InstrumentationLevel::debug;
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::hip::BatchedExpansionExecutor rejected{
            fixture.partitioned.graph,
            tile_runs,
            resident,
            workspace,
            stream,
            instrumented,
            bfnew::hip::BatchedExpansionTransferMode::compact_paths};
        static_cast<void>(rejected);
      },
      "instrumented evidence cannot masquerade as production compact paths");

  const std::array one_query{
      query_for(queries, bfnew::test::phase18_bounded_query_id)};
  bfnew::BatchedExpansionOptions failure_options = expansion_options(
      engine, bfnew::one_ring_expansion(), 1U);
  bfnew::hip::BatchedExpansionExecutor failure_executor{
      fixture.partitioned.graph,
      tile_runs,
      resident,
      workspace,
      stream,
      failure_options.run_options,
      bfnew::hip::BatchedExpansionTransferMode::compact_paths};
  const bfnew::BatchedExpansionRunResult failure = run_expansion(
      fixture, tile_runs, one_query, failure_options, failure_executor);
  expect(
      failure.metrics.engine_failure_queries == 1U &&
          failure.metrics.scheduled_expansions == 0U &&
          failure.queries.front().disposition ==
              bfnew::ExpansionQueryDisposition::engine_failure &&
          !failure.queries.front().compact_paths.has_value(),
      "round-limit failure neither expands nor publishes a reached payload");
}

void test_compact_device_paths() {
  const bfnew::test::CompactPathFixture fixture =
      bfnew::test::make_compact_path_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const bfnew::TileRunLayout64 tile_runs =
      bfnew::build_tile_run_layout(graph);
  const bfnew::DeviceGraphLayout32 device_graph =
      bfnew::build_device_graph_layout32(graph, tile_runs);
  const std::span<const bfnew::VertexId> map = graph.old_to_new();
  const std::array mixed_sources{map[13U]};
  const std::array mixed_targets{map[14U], map[15U]};
  const bfnew::RouteQuery mixed_query = bfnew::make_route_query(
      mixed_target_query_id, graph, mixed_sources, mixed_targets);
  const std::vector<bfnew::RouteQuery> queries{
      fixture.cycle_query,
      fixture.tie_query,
      fixture.bounded_query,
      fixture.multisource_query,
      fixture.unreachable_query,
      mixed_query,
  };
  const bfnew::SelectedRegionIndex selected_regions{graph, tile_runs};
  const std::vector<bfnew::BatchQueryFeatures> features =
      bfnew::make_batch_query_features(graph, selected_regions, queries);
  bfnew::BatchPlannerPolicy policy;
  policy.lane_width = 8U;
  policy.minimum_jaccard_numerator = 0U;
  policy.minimum_jaccard_denominator = 1U;
  policy.maximum_union_inflation_numerator = 64U;
  policy.maximum_union_inflation_denominator = 1U;
  const bfnew::BatchPlan plan =
      bfnew::make_overlapping_batch_plan(selected_regions, features, policy);
  expect(plan.batches.size() == 1U, "Phase 18 fixture forms one bounded batch");
  bfnew::BatchDeviceDescription description;
  bfnew::prepare_batch_device_description(
      graph,
      tile_runs,
      queries,
      features,
      plan.batches.front(),
      bfnew::BatchRunRepresentation::retained_per_run_masks,
      description);

  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(device_graph), stream);

  {
    bfnew::hip::ReusableBatchedJacobiWorkspace workspace;
    bfnew::hip::ReusableCompactPathWorkspace compact_workspace;
    bfnew::hip::BatchedJacobiPullEngine engine{
        graph, tile_runs, resident, workspace, stream};
    const bfnew::hip::CompactPathBatchOutput first =
        engine.run_compact_paths(
            description,
            gpu_options(bfnew::EngineKind::jacobi_pull),
            compact_workspace);
    validate_phase5_paths(fixture, queries, description, first, "Jacobi");
    const bfnew::hip::CompactPathWorkspaceCapacity capacity =
        compact_workspace.capacity();
    const bfnew::hip::CompactPathBatchOutput second =
        engine.run_compact_paths(
            description,
            gpu_options(bfnew::EngineKind::jacobi_pull),
            compact_workspace);
    expect(
        same_status(first.status, second.status) &&
            first.targets == second.targets &&
            compact_workspace.capacity().allocation_events ==
                capacity.allocation_events &&
            capacity.dfs_vertex_capacity == graph.vertex_count(),
        "Jacobi reuses one O(V), not O(V*targets), DFS stack");
    exercise_representative_controls(
        description,
        engine,
        compact_workspace,
        bfnew::EngineKind::jacobi_pull,
        first);
    exercise_width_matrix(
        graph,
        tile_runs,
        queries,
        engine,
        compact_workspace,
        bfnew::EngineKind::jacobi_pull);
    exercise_executor(
        fixture,
        tile_runs,
        resident,
        stream,
        queries,
        workspace,
        bfnew::EngineKind::jacobi_pull);
  }
  {
    bfnew::hip::ReusableBatchedDenseWorkspace workspace;
    bfnew::hip::ReusableCompactPathWorkspace compact_workspace;
    bfnew::hip::BatchedDenseChaoticPushEngine engine{
        graph, tile_runs, resident, workspace, stream};
    const bfnew::hip::CompactPathBatchOutput first =
        engine.run_compact_paths(
            description,
            gpu_options(bfnew::EngineKind::dense_chaotic_push),
            compact_workspace);
    validate_phase5_paths(fixture, queries, description, first, "dense");
    const bfnew::hip::CompactPathWorkspaceCapacity capacity =
        compact_workspace.capacity();
    const bfnew::hip::CompactPathBatchOutput second =
        engine.run_compact_paths(
            description,
            gpu_options(bfnew::EngineKind::dense_chaotic_push),
            compact_workspace);
    expect(
        same_status(first.status, second.status) &&
            first.targets == second.targets &&
            compact_workspace.capacity().allocation_events ==
                capacity.allocation_events &&
            capacity.dfs_vertex_capacity == graph.vertex_count(),
        "dense push reuses one O(V), not O(V*targets), DFS stack");
    exercise_representative_controls(
        description,
        engine,
        compact_workspace,
        bfnew::EngineKind::dense_chaotic_push,
        first);
    exercise_width_matrix(
        graph,
        tile_runs,
        queries,
        engine,
        compact_workspace,
        bfnew::EngineKind::dense_chaotic_push);
    exercise_executor(
        fixture,
        tile_runs,
        resident,
        stream,
        queries,
        workspace,
        bfnew::EngineKind::dense_chaotic_push);
  }
  {
    bfnew::hip::ReusableBatchedFrontierWorkspace workspace;
    bfnew::hip::ReusableCompactPathWorkspace compact_workspace;
    bfnew::hip::BatchedFrontierPushEngine engine{
        graph, tile_runs, resident, workspace, stream};
    const bfnew::hip::CompactPathBatchOutput first =
        engine.run_compact_paths(
            description,
            gpu_options(bfnew::EngineKind::frontier_push),
            compact_workspace);
    validate_phase5_paths(fixture, queries, description, first, "frontier");
    const bfnew::hip::CompactPathWorkspaceCapacity capacity =
        compact_workspace.capacity();
    const bfnew::hip::CompactPathBatchOutput second =
        engine.run_compact_paths(
            description,
            gpu_options(bfnew::EngineKind::frontier_push),
            compact_workspace);
    expect(
        same_status(first.status, second.status) &&
            first.targets == second.targets &&
            compact_workspace.capacity().allocation_events ==
                capacity.allocation_events &&
            capacity.dfs_vertex_capacity == graph.vertex_count(),
        "frontier push reuses one O(V), not O(V*targets), DFS stack");
    exercise_representative_controls(
        description,
        engine,
        compact_workspace,
        bfnew::EngineKind::frontier_push,
        first);
    exercise_width_matrix(
        graph,
        tile_runs,
        queries,
        engine,
        compact_workspace,
        bfnew::EngineKind::frontier_push);
    exercise_executor(
        fixture,
        tile_runs,
        resident,
        stream,
        queries,
        workspace,
        bfnew::EngineKind::frontier_push);
  }
}

}  // namespace

int main() {
  try {
    test_compact_device_paths();
  } catch (const std::exception& error) {
    std::cerr << "deferred HIP compact-path test threw: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
  if (failures != 0) {
    std::cerr << failures << " deferred HIP compact-path checks failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "deferred HIP compact-path checks passed\n";
  return EXIT_SUCCESS;
}
