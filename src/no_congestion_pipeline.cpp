#include "bfnew/no_congestion_pipeline.hpp"

#include "bfnew/gpu_api.hpp"
#include "bfnew/sssp.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

[[nodiscard]] bool valid_evidence(
    const PipelineTimingEvidence evidence) noexcept {
  return evidence == PipelineTimingEvidence::unavailable ||
         evidence == PipelineTimingEvidence::measured;
}

[[nodiscard]] bool try_add(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t& result) noexcept {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

void checked_add(
    std::uint64_t& destination,
    const std::uint64_t value,
    const std::string_view what) {
  std::uint64_t result = 0U;
  if (!try_add(destination, value, result)) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  destination = result;
}

[[nodiscard]] std::uint64_t checked_multiply(
    const std::uint64_t count,
    const std::uint64_t width,
    const std::string_view what) {
  if (width != 0U &&
      count > std::numeric_limits<std::uint64_t>::max() / width) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  return count * width;
}

[[nodiscard]] NoCongestionStageLedgerError validate_stage(
    const PipelineStageTiming& stage) noexcept {
  if (!valid_evidence(stage.host_evidence) ||
      !valid_evidence(stage.device_evidence)) {
    return NoCongestionStageLedgerError::invalid_evidence;
  }
  if (stage.host_evidence == PipelineTimingEvidence::unavailable &&
      stage.host_nanoseconds != 0U) {
    return NoCongestionStageLedgerError::unavailable_host_has_value;
  }
  if (stage.device_evidence == PipelineTimingEvidence::unavailable &&
      stage.device_milliseconds != 0.0) {
    return NoCongestionStageLedgerError::unavailable_device_has_value;
  }
  if (stage.device_evidence == PipelineTimingEvidence::measured &&
      (!std::isfinite(stage.device_milliseconds) ||
       stage.device_milliseconds < 0.0)) {
    return NoCongestionStageLedgerError::invalid_device_duration;
  }
  return NoCongestionStageLedgerError::none;
}

[[nodiscard]] bool all_host_measured(
    const std::initializer_list<const PipelineStageTiming*> stages) noexcept {
  return std::all_of(
      stages.begin(),
      stages.end(),
      [](const PipelineStageTiming* stage) {
        return stage->host_evidence == PipelineTimingEvidence::measured;
      });
}

[[nodiscard]] bool sum_host_stages(
    const std::initializer_list<const PipelineStageTiming*> stages,
    std::uint64_t& total) noexcept {
  total = 0U;
  for (const PipelineStageTiming* stage : stages) {
    if (!try_add(total, stage->host_nanoseconds, total)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::uint64_t size64(
    const std::size_t value,
    const std::string_view what) {
  if (value > std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  return static_cast<std::uint64_t>(value);
}

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] double ratio(
    const double bounded,
    const double unbounded) {
  if (!std::isfinite(bounded) || !std::isfinite(unbounded) || bounded < 0.0 ||
      unbounded < 0.0) {
    throw std::invalid_argument{
        "path-quality ratios require finite nonnegative values"};
  }
  if (unbounded == 0.0) {
    return bounded == 0.0 ? 1.0 : std::numeric_limits<double>::infinity();
  }
  return bounded / unbounded;
}

[[nodiscard]] double percentile(
    std::vector<double> values,
    const std::uint32_t percent) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::uint64_t count = static_cast<std::uint64_t>(values.size());
  const std::uint64_t rank =
      (static_cast<std::uint64_t>(percent) * count + 99U) / 100U;
  const std::size_t index = static_cast<std::size_t>(rank - 1U);
  return values[index];
}

}  // namespace

NoCongestionStageLedger make_no_congestion_stage_ledger(
    PipelineStageTiming cold_artifact_load,
    PipelineStageTiming graph_upload,
    PipelineStageTiming batch_planning,
    PipelineStageTiming sssp,
    PipelineStageTiming expansion,
    PipelineStageTiming controller_orchestration,
    PipelineStageTiming reconstruction,
    PipelineStageTiming result_transfer,
    PipelineStageTiming warm_all_query,
    PipelineStageTiming cold_execution) {
  NoCongestionStageLedger ledger{
      cold_artifact_load,
      graph_upload,
      batch_planning,
      sssp,
      expansion,
      controller_orchestration,
      reconstruction,
      result_transfer,
      warm_all_query,
      cold_execution,
      {}};
  const std::initializer_list<const PipelineStageTiming*> cold_stages{
      &ledger.cold_artifact_load,
      &ledger.graph_upload,
      &ledger.cold_execution};
  if (all_host_measured(cold_stages)) {
    ledger.cold_pipeline.host_evidence = PipelineTimingEvidence::measured;
    if (!sum_host_stages(
            cold_stages, ledger.cold_pipeline.host_nanoseconds)) {
      throw std::overflow_error{"cold pipeline timing sum overflow"};
    }
  }
  const NoCongestionStageLedgerError validation =
      validate_no_congestion_stage_ledger(ledger);
  if (validation != NoCongestionStageLedgerError::none) {
    throw std::invalid_argument{"invalid no-congestion stage timing input"};
  }
  return ledger;
}

NoCongestionStageLedgerError validate_no_congestion_stage_ledger(
    const NoCongestionStageLedger& ledger) noexcept {
  const std::array<const PipelineStageTiming*, 11U> all_stages{
      &ledger.cold_artifact_load,
      &ledger.graph_upload,
      &ledger.batch_planning,
      &ledger.sssp,
      &ledger.expansion,
      &ledger.controller_orchestration,
      &ledger.reconstruction,
      &ledger.result_transfer,
      &ledger.warm_all_query,
      &ledger.cold_execution,
      &ledger.cold_pipeline};
  for (const PipelineStageTiming* stage : all_stages) {
    const NoCongestionStageLedgerError error = validate_stage(*stage);
    if (error != NoCongestionStageLedgerError::none) {
      return error;
    }
  }

  const std::array<const PipelineStageTiming*, 7U> host_only_stages{
      &ledger.cold_artifact_load,
      &ledger.batch_planning,
      &ledger.expansion,
      &ledger.controller_orchestration,
      &ledger.warm_all_query,
      &ledger.cold_execution,
      &ledger.cold_pipeline};
  if (std::ranges::any_of(
          host_only_stages,
          [](const PipelineStageTiming* stage) {
            return stage->device_evidence !=
                   PipelineTimingEvidence::unavailable;
          })) {
    return NoCongestionStageLedgerError::cpu_only_device_evidence;
  }

  const std::initializer_list<const PipelineStageTiming*> warm_stages{
      &ledger.batch_planning,
      &ledger.sssp,
      &ledger.expansion,
      &ledger.controller_orchestration,
      &ledger.reconstruction,
      &ledger.result_transfer};
  const bool warm_components_measured = all_host_measured(warm_stages);
  std::uint64_t known_warm_total = 0U;
  if (!sum_host_stages(warm_stages, known_warm_total)) {
    return NoCongestionStageLedgerError::warm_sum_overflow;
  }
  if (warm_components_measured) {
    if (ledger.warm_all_query.host_evidence !=
        PipelineTimingEvidence::measured) {
      return NoCongestionStageLedgerError::warm_evidence_mismatch;
    }
    if (ledger.warm_all_query.host_nanoseconds != known_warm_total) {
      return NoCongestionStageLedgerError::warm_sum_mismatch;
    }
  } else if (ledger.warm_all_query.host_evidence ==
                 PipelineTimingEvidence::measured &&
             known_warm_total > ledger.warm_all_query.host_nanoseconds) {
    return NoCongestionStageLedgerError::warm_sum_mismatch;
  }

  const std::initializer_list<const PipelineStageTiming*> cold_stages{
      &ledger.cold_artifact_load,
      &ledger.graph_upload,
      &ledger.cold_execution};
  const bool cold_components_measured = all_host_measured(cold_stages);
  if ((ledger.cold_pipeline.host_evidence ==
       PipelineTimingEvidence::measured) != cold_components_measured) {
    return NoCongestionStageLedgerError::cold_evidence_mismatch;
  }
  if (cold_components_measured) {
    std::uint64_t expected = 0U;
    if (!sum_host_stages(cold_stages, expected)) {
      return NoCongestionStageLedgerError::cold_sum_overflow;
    }
    if (ledger.cold_pipeline.host_nanoseconds != expected) {
      return NoCongestionStageLedgerError::cold_sum_mismatch;
    }
  }
  return NoCongestionStageLedgerError::none;
}

NoCongestionResultAccounting measure_no_congestion_result_transfer(
    const std::span<const CompactQueryResult> results,
    const std::uint64_t modeled_batch_status_count,
    const BatchedExpansionMetrics* const device_metrics) {
  NoCongestionResultAccounting accounting;
  accounting.query_count = size64(results.size(), "compact query count");
  accounting.modeled_batch_status_count = modeled_batch_status_count;
  accounting.modeled_batch_status_bytes = checked_multiply(
      modeled_batch_status_count,
      sizeof(DeviceRunStatus),
      "compact batch-status transfer bytes");

  std::uint64_t vertex_count = 0U;
  std::uint64_t edge_count = 0U;
  for (std::size_t query_index = 0U; query_index < results.size(); ++query_index) {
    const CompactQueryResult& query = results[query_index];
    if (query_index != 0U &&
        !(results[query_index - 1U].query_id < query.query_id)) {
      throw std::invalid_argument{
          "compact result accounting requires canonical unique query IDs"};
    }
    checked_add(
        accounting.target_summary_count,
        size64(query.targets.size(), "compact target count"),
        "compact target count");
    if (query.targets.empty()) {
      throw std::invalid_argument{
          "compact result accounting requires at least one target"};
    }
    bool all_complete = true;
    bool any_unreachable = false;
    bool all_complete_or_unreachable = true;
    bool any_reconstruction_failure = false;
    bool all_terminal_failure = true;
    for (const CompactTargetPath& target : query.targets) {
      const bool arenas_empty = target.vertices.empty() &&
                                target.distance_labels.empty() &&
                                target.edge_ids.empty();
      switch (target.summary.reconstruction) {
        case CompactPathStatus::complete:
          all_terminal_failure = false;
          checked_add(
              accounting.complete_path_count, 1U, "complete path count");
          if (target.summary.reached != CompactTargetReachStatus::reached ||
              target.summary.has_selected_source != 1U ||
              !std::isfinite(target.summary.distance) ||
              target.summary.distance < 0.0F ||
              target.vertices.size() != target.distance_labels.size() ||
              target.vertices.size() != target.edge_ids.size() + 1U ||
              target.edge_ids.size() != target.summary.path_length) {
            throw std::invalid_argument{
                "complete compact path has inconsistent arena shape"};
          }
          checked_add(
              vertex_count,
              size64(target.vertices.size(), "compact path vertex count"),
              "compact path vertex count");
          checked_add(
              edge_count,
              size64(target.edge_ids.size(), "compact path edge count"),
              "compact path edge count");
          break;
        case CompactPathStatus::unreachable:
          all_complete = false;
          any_unreachable = true;
          all_terminal_failure = false;
          if (target.summary.reached !=
                  CompactTargetReachStatus::not_reached ||
              !std::isinf(target.summary.distance) ||
              target.summary.distance < 0.0F ||
              target.summary.has_selected_source != 0U ||
              target.summary.path_length != 0U || !arenas_empty) {
            throw std::invalid_argument{
                "unreachable compact summary has inconsistent fields"};
          }
          checked_add(
              accounting.unreachable_target_count,
              1U,
              "unreachable target count");
          break;
        case CompactPathStatus::query_terminal_failure:
          all_complete = false;
          all_complete_or_unreachable = false;
          any_reconstruction_failure = true;
          if (target.summary.reached !=
                  CompactTargetReachStatus::not_reached ||
              !std::isinf(target.summary.distance) ||
              target.summary.distance < 0.0F ||
              target.summary.has_selected_source != 0U ||
              target.summary.path_length != 0U || !arenas_empty) {
            throw std::invalid_argument{
                "terminal-failure compact summary has inconsistent fields"};
          }
          checked_add(
              accounting.terminal_failure_target_count,
              1U,
              "terminal-failure target count");
          break;
        case CompactPathStatus::no_tight_path:
        case CompactPathStatus::path_length_overflow:
          all_complete = false;
          all_complete_or_unreachable = false;
          any_reconstruction_failure = true;
          all_terminal_failure = false;
          if (target.summary.reached != CompactTargetReachStatus::reached ||
              !std::isfinite(target.summary.distance) ||
              target.summary.distance < 0.0F ||
              target.summary.has_selected_source != 0U ||
              target.summary.path_length != 0U || !arenas_empty) {
            throw std::invalid_argument{
                "reconstruction-failure compact summary has inconsistent fields"};
          }
          checked_add(
              accounting.reconstruction_failure_target_count,
              1U,
              "reconstruction-failure target count");
          break;
        default:
          throw std::invalid_argument{
              "compact target has an unknown reconstruction status"};
      }
    }
    switch (query.disposition) {
      case ExpansionQueryDisposition::reached:
        if (!all_complete) {
          throw std::invalid_argument{
              "reached compact result contains an incomplete target"};
        }
        break;
      case ExpansionQueryDisposition::unreachable_in_full_region:
        if (!any_unreachable || !all_complete_or_unreachable) {
          throw std::invalid_argument{
              "full-region-unreachable result has inconsistent targets"};
        }
        break;
      case ExpansionQueryDisposition::engine_failure:
        if (!any_reconstruction_failure) {
          throw std::invalid_argument{
              "engine-failure result has no failed target"};
        }
        break;
      case ExpansionQueryDisposition::expansion_limit:
      case ExpansionQueryDisposition::region_stalled:
      case ExpansionQueryDisposition::identity_or_count_overflow:
        if (!all_terminal_failure) {
          throw std::invalid_argument{
              "terminal query failure contains a nonfailure target"};
        }
        break;
      default:
        throw std::invalid_argument{
            "compact result has an unknown query disposition"};
    }
  }

  accounting.final_result_serialization.summary_bytes = checked_multiply(
      accounting.target_summary_count,
      sizeof(CompactTargetSummary),
      "compact target-summary bytes");
  accounting.final_result_serialization.vertex_bytes = checked_multiply(
      vertex_count, sizeof(std::uint32_t), "compact path-vertex bytes");
  accounting.final_result_serialization.distance_label_bytes = checked_multiply(
      vertex_count, sizeof(float), "compact path-label bytes");
  accounting.final_result_serialization.edge_id_bytes = checked_multiply(
      edge_count, sizeof(std::uint32_t), "compact path-edge-ID bytes");
  accounting.final_result_serialization.total_bytes =
      accounting.final_result_serialization.summary_bytes;
  checked_add(
      accounting.final_result_serialization.total_bytes,
      accounting.final_result_serialization.vertex_bytes,
      "compact result bytes");
  checked_add(
      accounting.final_result_serialization.total_bytes,
      accounting.final_result_serialization.distance_label_bytes,
      "compact result bytes");
  checked_add(
      accounting.final_result_serialization.total_bytes,
      accounting.final_result_serialization.edge_id_bytes,
      "compact result bytes");
  accounting.modeled_final_transfer_bytes =
      accounting.modeled_batch_status_bytes;
  checked_add(
      accounting.modeled_final_transfer_bytes,
      accounting.final_result_serialization.total_bytes,
      "total compact result transfer bytes");

  if (device_metrics != nullptr &&
      device_metrics->compact_total_device_to_host_bytes != 0U) {
    std::uint64_t component_total = 0U;
    checked_add(
        component_total,
        device_metrics->compact_transfer.summary_bytes,
        "actual compact payload bytes");
    checked_add(
        component_total,
        device_metrics->compact_transfer.vertex_bytes,
        "actual compact payload bytes");
    checked_add(
        component_total,
        device_metrics->compact_transfer.distance_label_bytes,
        "actual compact payload bytes");
    checked_add(
        component_total,
        device_metrics->compact_transfer.edge_id_bytes,
        "actual compact payload bytes");
    if (component_total != device_metrics->compact_transfer.total_bytes) {
      throw std::invalid_argument{
          "actual compact payload byte components are inconsistent"};
    }
    const std::uint64_t expected_status_bytes = checked_multiply(
        modeled_batch_status_count,
        sizeof(DeviceRunStatus),
        "actual compact status bytes");
    const std::uint64_t minimum_error_bytes = checked_multiply(
        modeled_batch_status_count,
        sizeof(std::uint32_t),
        "actual compact error bytes");
    const std::uint64_t maximum_error_bytes = checked_multiply(
        minimum_error_bytes, 2U, "actual compact error bytes");
    if (device_metrics->compact_status_bytes != expected_status_bytes ||
        device_metrics->compact_error_bytes < minimum_error_bytes ||
        device_metrics->compact_error_bytes > maximum_error_bytes ||
        device_metrics->compact_error_bytes % sizeof(std::uint32_t) != 0U) {
      throw std::invalid_argument{
          "actual compact control-transfer bytes disagree with batch count"};
    }
    std::uint64_t expected_total = device_metrics->compact_transfer.total_bytes;
    checked_add(
        expected_total,
        device_metrics->compact_status_bytes,
        "actual compact status bytes");
    checked_add(
        expected_total,
        device_metrics->compact_error_bytes,
        "actual compact error bytes");
    if (expected_total !=
        device_metrics->compact_total_device_to_host_bytes) {
      throw std::invalid_argument{
          "actual compact device-transfer metrics are inconsistent"};
    }
    if (device_metrics->compact_controller_poll_count >
        std::numeric_limits<std::uint64_t>::max() /
            sizeof(DeviceController)) {
      throw std::invalid_argument{
          "actual compact controller-poll byte count overflows"};
    }
    const std::uint64_t expected_controller_poll_bytes =
        device_metrics->compact_controller_poll_count *
        sizeof(DeviceController);
    if (expected_controller_poll_bytes !=
        device_metrics->compact_controller_poll_bytes) {
      throw std::invalid_argument{
          "actual compact controller-poll bytes disagree with poll count"};
    }
    checked_add(
        expected_total,
        device_metrics->compact_controller_poll_bytes,
        "actual compact overall device-transfer bytes");
    if (expected_total !=
        device_metrics->compact_overall_device_to_host_bytes) {
      throw std::invalid_argument{
          "actual compact overall device-transfer metrics are inconsistent"};
    }
    accounting.device_transfer_evidence =
        CompactStageTimingEvidence::measured;
    accounting.actual_compact_device_transfer =
        device_metrics->compact_transfer;
    accounting.actual_status_bytes = device_metrics->compact_status_bytes;
    accounting.actual_error_bytes = device_metrics->compact_error_bytes;
    accounting.actual_compact_total_device_to_host_bytes =
        device_metrics->compact_total_device_to_host_bytes;
    accounting.actual_controller_poll_count =
        device_metrics->compact_controller_poll_count;
    accounting.actual_controller_poll_bytes =
        device_metrics->compact_controller_poll_bytes;
    accounting.actual_overall_device_to_host_bytes =
        device_metrics->compact_overall_device_to_host_bytes;
  } else if (device_metrics != nullptr &&
             (device_metrics->compact_status_bytes != 0U ||
              device_metrics->compact_error_bytes != 0U ||
              device_metrics->compact_controller_poll_count != 0U ||
              device_metrics->compact_controller_poll_bytes != 0U ||
              device_metrics->compact_overall_device_to_host_bytes != 0U)) {
    throw std::invalid_argument{
        "unavailable compact device transfer carries numeric control bytes"};
  }
  return accounting;
}

CompactPathQualitySample sample_compact_path_quality(
    const WeightedGraph& graph,
    const std::span<const ExpansionQueryOutcome> outcomes,
    const std::span<const CompactQueryResult> results,
    const std::size_t sample_query_count,
    const std::uint64_t selection_seed) {
  if (!validate_weighted_graph(graph).ok() || outcomes.size() != results.size()) {
    throw std::invalid_argument{
        "path-quality sampling requires a valid graph and complete result ledger"};
  }
  struct Candidate {
    std::uint64_t key{};
    QueryId query_id{};
    std::size_t index{};
  };
  std::vector<Candidate> candidates;
  candidates.reserve(outcomes.size());
  for (std::size_t index = 0U; index < outcomes.size(); ++index) {
    const ExpansionQueryOutcome& outcome = outcomes[index];
    const CompactQueryResult& result = results[index];
    if (outcome.final_query.query_id != result.query_id ||
        outcome.final_query.expansion_generation !=
            result.expansion_generation ||
        outcome.disposition != result.disposition ||
        !validate_compact_query_result(
             graph, outcome.final_query, result)
             .ok()) {
      throw std::invalid_argument{
          "path-quality sampling received a mismatched compact result"};
    }
    candidates.push_back(Candidate{
        splitmix64(
            selection_seed ^
            static_cast<std::uint64_t>(result.query_id.value())),
        result.query_id,
        index});
  }
  std::vector<QueryId> unique_query_ids;
  unique_query_ids.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    unique_query_ids.push_back(candidate.query_id);
  }
  std::sort(unique_query_ids.begin(), unique_query_ids.end());
  if (std::adjacent_find(
          unique_query_ids.begin(), unique_query_ids.end()) !=
      unique_query_ids.end()) {
    throw std::invalid_argument{
        "path-quality sampling requires unique query identities"};
  }
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const Candidate& left, const Candidate& right) {
        if (left.key != right.key) {
          return left.key < right.key;
        }
        return left.query_id < right.query_id;
      });

  CompactPathQualitySample sample;
  sample.selection_seed = selection_seed;
  sample.population_query_count =
      size64(outcomes.size(), "quality population query count");
  sample.requested_sample_query_count =
      size64(sample_query_count, "requested quality sample count");
  const std::size_t selected_count =
      std::min(sample_query_count, candidates.size());
  sample.sampled_query_count =
      size64(selected_count, "quality sampled query count");
  sample.sampled_query_ids.reserve(selected_count);

  std::vector<double> cost_ratios;
  std::vector<double> length_ratios;
  std::vector<double> absolute_cost_inflations;
  std::vector<double> absolute_length_inflations;
  for (std::size_t selection = 0U; selection < selected_count; ++selection) {
    const Candidate& candidate = candidates[selection];
    const ExpansionQueryOutcome& outcome = outcomes[candidate.index];
    const CompactQueryResult& result = results[candidate.index];
    sample.sampled_query_ids.push_back(candidate.query_id);
    const SsspResult unbounded =
        dijkstra_oracle(graph, outcome.final_query.sources);
    for (const CompactTargetPath& target : result.targets) {
      if (target.summary.reconstruction != CompactPathStatus::complete) {
        continue;
      }
      const std::size_t target_index = target.summary.target.value();
      if (target_index >= unbounded.distances.size() ||
          !std::isfinite(unbounded.distances[target_index])) {
        throw std::logic_error{
            "bounded reachable target is unreachable in the full graph"};
      }
      const auto unbounded_path = reconstruct_path_from_distances(
          graph, unbounded, target.summary.target);
      if (!unbounded_path ||
          unbounded_path->edge_ids.size() >
              std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error{
            "unbounded quality reference did not yield a representable path"};
      }
      const std::uint32_t unbounded_length =
          static_cast<std::uint32_t>(unbounded_path->edge_ids.size());
      const double cost_ratio = ratio(
          static_cast<double>(target.summary.distance),
          static_cast<double>(unbounded.distances[target_index]));
      const double length_ratio = ratio(
          static_cast<double>(target.summary.path_length),
          static_cast<double>(unbounded_length));
      const double absolute_cost_inflation =
          static_cast<double>(target.summary.distance) -
          static_cast<double>(unbounded.distances[target_index]);
      const std::int64_t absolute_path_length_inflation =
          static_cast<std::int64_t>(target.summary.path_length) -
          static_cast<std::int64_t>(unbounded_length);
      sample.observations.push_back(CompactPathQualityObservation{
          candidate.query_id,
          target.summary.target,
          target.summary.distance,
          unbounded.distances[target_index],
          target.summary.path_length,
          unbounded_length,
          absolute_cost_inflation,
          absolute_path_length_inflation,
          cost_ratio,
          length_ratio});
      cost_ratios.push_back(cost_ratio);
      length_ratios.push_back(length_ratio);
      absolute_cost_inflations.push_back(absolute_cost_inflation);
      absolute_length_inflations.push_back(
          static_cast<double>(absolute_path_length_inflation));
    }
  }
  sample.finite_target_pairs =
      size64(sample.observations.size(), "quality finite target-pair count");
  sample.absolute_cost_inflation_p50 =
      percentile(absolute_cost_inflations, 50U);
  sample.absolute_cost_inflation_p95 =
      percentile(absolute_cost_inflations, 95U);
  sample.absolute_cost_inflation_p99 =
      percentile(absolute_cost_inflations, 99U);
  sample.absolute_cost_inflation_max =
      absolute_cost_inflations.empty()
          ? 0.0
          : *std::max_element(
                absolute_cost_inflations.begin(),
                absolute_cost_inflations.end());
  sample.cost_ratio_p50 = percentile(cost_ratios, 50U);
  sample.cost_ratio_p95 = percentile(cost_ratios, 95U);
  sample.cost_ratio_p99 = percentile(cost_ratios, 99U);
  sample.cost_ratio_max =
      cost_ratios.empty()
          ? 0.0
          : *std::max_element(cost_ratios.begin(), cost_ratios.end());
  sample.absolute_path_length_inflation_p50 =
      percentile(absolute_length_inflations, 50U);
  sample.absolute_path_length_inflation_p95 =
      percentile(absolute_length_inflations, 95U);
  sample.absolute_path_length_inflation_p99 =
      percentile(absolute_length_inflations, 99U);
  sample.absolute_path_length_inflation_max =
      absolute_length_inflations.empty()
          ? 0.0
          : *std::max_element(
                absolute_length_inflations.begin(),
                absolute_length_inflations.end());
  sample.path_length_ratio_p50 = percentile(length_ratios, 50U);
  sample.path_length_ratio_p95 = percentile(length_ratios, 95U);
  sample.path_length_ratio_p99 = percentile(length_ratios, 99U);
  sample.path_length_ratio_max =
      length_ratios.empty()
          ? 0.0
          : *std::max_element(length_ratios.begin(), length_ratios.end());
  return sample;
}

HostNoCongestionPipelineResult run_host_no_congestion_pipeline(
    const WeightedGraph& graph,
    const TileDirectory& directory,
    const TileRunLayout64& tile_runs,
    const DeviceGraphLayout32& device_graph,
    const std::span<const RouteQuery> queries,
    const BatchedExpansionOptions& expansion_options,
    const HostNoCongestionPipelineOptions& pipeline_options) {
  if (validate_batched_expansion_options(expansion_options) !=
      BatchedExpansionOptionsError::none) {
    throw std::invalid_argument{
        "host no-congestion pipeline requires valid expansion options"};
  }

  // Exercise the same generation-bound result handoff as device production.
  // The host adapter reports its internal SSSP/reconstruction/result-building
  // intervals explicitly and clears its diagnostic lane matrix before the
  // callback returns.
  BatchedExpansionOptions host_expansion_options = expansion_options;
  host_expansion_options.enable_compact_paths = 1U;
  HostNoCongestionPipelineResult output;
  output.expansion = run_host_batched_expansion(
      graph,
      directory,
      tile_runs,
      device_graph,
      queries,
      host_expansion_options,
      pipeline_options.engine);

  const auto extraction_begin = std::chrono::steady_clock::now();
  output.compact_results =
      extract_host_compact_paths(graph, output.expansion.queries);
  const auto extraction_end = std::chrono::steady_clock::now();
  const auto extraction_duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          extraction_end - extraction_begin);
  if (extraction_duration.count() < 0) {
    throw std::logic_error{"steady compact extraction timer moved backward"};
  }
  const std::uint64_t extraction_nanoseconds =
      static_cast<std::uint64_t>(extraction_duration.count());

  output.result_accounting = measure_no_congestion_result_transfer(
      output.compact_results,
      size64(output.expansion.trace.size(), "batch-status transfer count"),
      &output.expansion.metrics);
  output.quality = sample_compact_path_quality(
      graph,
      output.expansion.queries,
      output.compact_results,
      pipeline_options.quality_sample_query_count,
      pipeline_options.quality_selection_seed);

  std::uint64_t planning_nanoseconds =
      output.expansion.metrics.initial_planning_nanoseconds;
  checked_add(
      planning_nanoseconds,
      output.expansion.metrics.replanning_nanoseconds,
      "portable planning time");
  if (output.expansion.metrics.compact_host_timing !=
      CompactStageTimingEvidence::measured) {
    throw std::logic_error{
        "portable compact pipeline did not measure every compact stage"};
  }
  const std::uint64_t result_transfer_nanoseconds =
      output.expansion.metrics.result_transfer_host_nanoseconds;
  std::uint64_t named_stage_nanoseconds = planning_nanoseconds;
  checked_add(
      named_stage_nanoseconds,
      output.expansion.metrics.sssp_host_nanoseconds,
      "portable named stage time");
  checked_add(
      named_stage_nanoseconds,
      output.expansion.metrics.reconstruction_host_nanoseconds,
      "portable named stage time");
  checked_add(
      named_stage_nanoseconds,
      output.expansion.metrics.result_transfer_host_nanoseconds,
      "portable named stage time");
  checked_add(
      named_stage_nanoseconds,
      output.expansion.metrics.geometric_expansion_host_nanoseconds,
      "portable named stage time");
  if (named_stage_nanoseconds > output.expansion.metrics.total_nanoseconds) {
    throw std::logic_error{
        "portable expansion timing components exceed their enclosing interval"};
  }
  std::uint64_t controller_orchestration_nanoseconds =
      output.expansion.metrics.total_nanoseconds - named_stage_nanoseconds;
  checked_add(
      controller_orchestration_nanoseconds,
      extraction_nanoseconds,
      "portable compact extraction/orchestration time");
  std::uint64_t warm_all_query_nanoseconds =
      output.expansion.metrics.total_nanoseconds;
  checked_add(
      warm_all_query_nanoseconds,
      extraction_nanoseconds,
      "portable warm all-query time");
  output.timing = make_no_congestion_stage_ledger(
      pipeline_options.cold_artifact_load,
      pipeline_options.graph_upload,
      measured_host_stage(planning_nanoseconds),
      measured_host_stage(output.expansion.metrics.sssp_host_nanoseconds),
      measured_host_stage(
          output.expansion.metrics.geometric_expansion_host_nanoseconds),
      measured_host_stage(controller_orchestration_nanoseconds),
      measured_host_stage(
          output.expansion.metrics.reconstruction_host_nanoseconds),
      measured_host_stage(result_transfer_nanoseconds),
      measured_host_stage(warm_all_query_nanoseconds),
      pipeline_options.cold_execution);
  return output;
}

}  // namespace bfnew
