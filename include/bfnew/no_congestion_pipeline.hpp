#pragma once

#include "bfnew/batched_expansion.hpp"
#include "bfnew/compact_paths.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace bfnew {

enum class PipelineTimingEvidence : std::uint8_t {
  unavailable,
  measured,
};

// Host-wall and device-event observations are kept independent. A numeric
// zero is meaningful only when the corresponding evidence field is measured.
struct PipelineStageTiming {
  PipelineTimingEvidence host_evidence{PipelineTimingEvidence::unavailable};
  std::uint64_t host_nanoseconds{};
  PipelineTimingEvidence device_evidence{PipelineTimingEvidence::unavailable};
  double device_milliseconds{};

  constexpr bool operator==(const PipelineStageTiming&) const noexcept =
      default;
};

[[nodiscard]] constexpr PipelineStageTiming measured_host_stage(
    const std::uint64_t nanoseconds) noexcept {
  return PipelineStageTiming{
      PipelineTimingEvidence::measured,
      nanoseconds,
      PipelineTimingEvidence::unavailable,
      0.0};
}

// The warm stage intervals are nonoverlapping and are recorded only after a
// capacity-growing run. When every host stage is measured, warm_all_query must
// equal their exact checked sum. An asynchronous device implementation may
// instead leave unpartitionable named host stages unavailable and provide a
// separately measured enclosing warm_all_query observation; device events
// remain attached only to their applicable named stages.
// cold_execution is a distinct first all-query interval after upload; it may
// include workspace construction and capacity growth. cold_pipeline is the
// exact checked sum of artifact load, graph upload, and cold_execution. The
// cold and warm execution observations are deliberately not equated.
struct NoCongestionStageLedger {
  PipelineStageTiming cold_artifact_load;
  PipelineStageTiming graph_upload;
  PipelineStageTiming batch_planning;
  PipelineStageTiming sssp;
  PipelineStageTiming expansion;
  // Validation, batch-description preparation, controller polling, retry
  // collection, and terminal-ledger assembly that are not geometric region
  // growth. Keeping this separate prevents those costs from being mislabeled
  // as expansion.
  PipelineStageTiming controller_orchestration;
  PipelineStageTiming reconstruction;
  PipelineStageTiming result_transfer;
  PipelineStageTiming warm_all_query;
  PipelineStageTiming cold_execution;
  PipelineStageTiming cold_pipeline;

  constexpr bool operator==(const NoCongestionStageLedger&) const noexcept =
      default;
};

enum class NoCongestionStageLedgerError : std::uint8_t {
  none,
  invalid_evidence,
  unavailable_host_has_value,
  unavailable_device_has_value,
  invalid_device_duration,
  cpu_only_device_evidence,
  warm_evidence_mismatch,
  warm_sum_overflow,
  warm_sum_mismatch,
  cold_evidence_mismatch,
  cold_sum_overflow,
  cold_sum_mismatch,
};

[[nodiscard]] NoCongestionStageLedger make_no_congestion_stage_ledger(
    PipelineStageTiming cold_artifact_load,
    PipelineStageTiming graph_upload,
    PipelineStageTiming batch_planning,
    PipelineStageTiming sssp,
    PipelineStageTiming expansion,
    PipelineStageTiming controller_orchestration,
    PipelineStageTiming reconstruction,
    PipelineStageTiming result_transfer,
    PipelineStageTiming warm_all_query,
    PipelineStageTiming cold_execution);

[[nodiscard]] NoCongestionStageLedgerError
validate_no_congestion_stage_ledger(
    const NoCongestionStageLedger& ledger) noexcept;

struct NoCongestionResultAccounting {
  std::uint64_t query_count{};
  std::uint64_t target_summary_count{};
  std::uint64_t complete_path_count{};
  std::uint64_t unreachable_target_count{};
  std::uint64_t terminal_failure_target_count{};
  std::uint64_t reconstruction_failure_target_count{};
  // Canonical final-result serialization model. It is useful for sizing, but
  // is not physical D2H evidence in the portable pipeline and deliberately
  // excludes compact payloads discarded by retries.
  std::uint64_t modeled_batch_status_count{};
  std::uint64_t modeled_batch_status_bytes{};
  CompactTransferAccounting final_result_serialization{};
  std::uint64_t modeled_final_transfer_bytes{};

  // Exact cumulative device transfers for the selected HIP repetition. The
  // compact subtotal includes retry/miss summaries plus fixed status/error
  // words. Controller polls remain explicit; overall is their checked sum.
  // Portable execution leaves the evidence unavailable and all fields zero.
  CompactStageTimingEvidence device_transfer_evidence{
      CompactStageTimingEvidence::unavailable};
  CompactTransferAccounting actual_compact_device_transfer{};
  std::uint64_t actual_status_bytes{};
  std::uint64_t actual_error_bytes{};
  std::uint64_t actual_compact_total_device_to_host_bytes{};
  std::uint64_t actual_controller_poll_count{};
  std::uint64_t actual_controller_poll_bytes{};
  std::uint64_t actual_overall_device_to_host_bytes{};

  constexpr bool operator==(const NoCongestionResultAccounting&) const noexcept =
      default;
};

// Device stable EdgeIds are checked uint32 values and are widened to EdgeId on
// the host. Accounting therefore charges four transfer bytes per path edge,
// not sizeof(EdgeId).
[[nodiscard]] NoCongestionResultAccounting
measure_no_congestion_result_transfer(
    std::span<const CompactQueryResult> results,
    std::uint64_t modeled_batch_status_count,
    const BatchedExpansionMetrics* device_metrics = nullptr);

enum class PathQualitySamplingMethod : std::uint8_t {
  splitmix64_query_id,
};

struct CompactPathQualityObservation {
  QueryId query_id{};
  VertexId target{};
  float bounded_distance{};
  float unbounded_distance{};
  std::uint32_t bounded_path_length{};
  std::uint32_t unbounded_path_length{};
  double absolute_cost_inflation{};
  std::int64_t absolute_path_length_inflation{};
  double cost_inflation_ratio{};
  double path_length_inflation_ratio{};

  constexpr bool operator==(
      const CompactPathQualityObservation&) const noexcept = default;
};

struct CompactPathQualitySample {
  PathQualitySamplingMethod method{
      PathQualitySamplingMethod::splitmix64_query_id};
  std::uint64_t selection_seed{};
  std::uint64_t population_query_count{};
  std::uint64_t requested_sample_query_count{};
  std::uint64_t sampled_query_count{};
  std::uint64_t finite_target_pairs{};
  std::vector<QueryId> sampled_query_ids;
  std::vector<CompactPathQualityObservation> observations;
  double absolute_cost_inflation_p50{};
  double absolute_cost_inflation_p95{};
  double absolute_cost_inflation_p99{};
  double absolute_cost_inflation_max{};
  double cost_ratio_p50{};
  double cost_ratio_p95{};
  double cost_ratio_p99{};
  double cost_ratio_max{};
  double absolute_path_length_inflation_p50{};
  double absolute_path_length_inflation_p95{};
  double absolute_path_length_inflation_p99{};
  double absolute_path_length_inflation_max{};
  double path_length_ratio_p50{};
  double path_length_ratio_p95{};
  double path_length_ratio_p99{};
  double path_length_ratio_max{};

  bool operator==(const CompactPathQualitySample&) const = default;
};

// The sample is selected independently of input order by hashing QueryId with
// the recorded seed. The reference is unbounded multi-source Dijkstra on the
// immutable graph. Quality inflation is evidence, not bounded correctness.
[[nodiscard]] CompactPathQualitySample sample_compact_path_quality(
    const WeightedGraph& graph,
    std::span<const ExpansionQueryOutcome> outcomes,
    std::span<const CompactQueryResult> results,
    std::size_t sample_query_count,
    std::uint64_t selection_seed);

struct HostNoCongestionPipelineOptions {
  HostBatchedExpansionOptions engine{};
  PipelineStageTiming cold_artifact_load{};
  PipelineStageTiming graph_upload{};
  // Optional enclosing observation from a separate first, capacity-growing
  // execution. The portable reference run below supplies the detailed warm
  // stage model and never relabels that same run as cold evidence.
  PipelineStageTiming cold_execution{};
  std::size_t quality_sample_query_count{};
  std::uint64_t quality_selection_seed{};
};

struct HostNoCongestionPipelineResult {
  BatchedExpansionRunResult expansion;
  std::vector<CompactQueryResult> compact_results;
  NoCongestionResultAccounting result_accounting{};
  NoCongestionStageLedger timing{};
  CompactPathQualitySample quality{};
};

// Portable correctness/reference pipeline. It enables the same
// generation-bound payload handoff as device production, while the portable
// engine may still use its host label image internally to construct that
// payload. The returned pipeline never retains a batch lane matrix.
[[nodiscard]] HostNoCongestionPipelineResult run_host_no_congestion_pipeline(
    const WeightedGraph& graph,
    const TileDirectory& directory,
    const TileRunLayout64& tile_runs,
    const DeviceGraphLayout32& device_graph,
    std::span<const RouteQuery> queries,
    const BatchedExpansionOptions& expansion_options,
    const HostNoCongestionPipelineOptions& pipeline_options = {});

}  // namespace bfnew
