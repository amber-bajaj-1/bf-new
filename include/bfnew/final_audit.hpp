#pragma once

#include "bfnew/batch_workspace.hpp"
#include "bfnew/batched_expansion.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace bfnew {

inline constexpr std::uint32_t phase19_audit_schema_version = 1U;

struct Phase19Fingerprint {
  std::uint64_t words[2]{};

  constexpr bool operator==(const Phase19Fingerprint&) const noexcept = default;
};

enum class Phase19FeatureClassification : std::uint8_t {
  implemented_correctness_tested = 0U,
  implemented_performance_profiled = 1U,
  implemented_not_representative = 2U,
  designed_deferred = 3U,
};

// This is the fixed Phase 19 audit inventory. Engine/control rows cover both
// the named implementation and its bounded semantic contract; the finding on
// each row states the remaining device-evidence boundary.
enum class Phase19FeatureId : std::uint16_t {
  portable_standalone_jacobi_pull = 0U,
  portable_standalone_dense_chaotic_push,
  portable_standalone_frontier_push,
  portable_standalone_jacobi_persistent_control,
  portable_standalone_jacobi_chunked_control,
  portable_standalone_jacobi_per_round_control,
  portable_standalone_dense_persistent_control,
  portable_standalone_dense_chunked_control,
  portable_standalone_dense_per_round_control,
  portable_standalone_frontier_persistent_control,
  portable_standalone_frontier_chunked_control,
  portable_standalone_frontier_per_round_control,
  hip_standalone_jacobi_pull,
  hip_standalone_dense_chaotic_push,
  hip_standalone_frontier_push,
  portable_batched_jacobi_pull,
  portable_batched_dense_chaotic_push,
  portable_batched_frontier_push,
  portable_batched_jacobi_persistent_control,
  portable_batched_jacobi_chunked_control,
  portable_batched_jacobi_per_round_control,
  portable_batched_dense_persistent_control,
  portable_batched_dense_chunked_control,
  portable_batched_dense_per_round_control,
  portable_batched_frontier_persistent_control,
  portable_batched_frontier_chunked_control,
  portable_batched_frontier_per_round_control,
  hip_batched_jacobi_pull,
  hip_batched_dense_chaotic_push,
  hip_batched_frontier_push,
  batching_width_1,
  batching_width_8,
  batching_width_16,
  batching_width_32,
  overlap_greedy_planner,
  provisional_workspace_strategy,
  portable_batched_expansion,
  hip_batched_expansion_adapter,
  portable_compact_reconstruction,
  hip_compact_reconstruction,
  no_congestion_pipeline,
  compact_transfer_accounting,
  hip_fpgaif_campaign_report_surface,
  portable_algorithm_counter_instrumentation,
  hip_algorithm_counter_instrumentation,
  dependency_trace_collection,
  profiler_l2_group,
  profiler_occupancy_memory_group,
  profiler_instruction_wave_group,
  correctness_evidence_inventory,
  phase12_shootout_evidence_contract,
  representative_logicnets_campaign,
  representative_profiler_campaign,
  performance_claim_inventory,
  production_default_selection,
  hybrid_experiment_decision,
  count,
};

struct Phase19FeatureAudit {
  Phase19FeatureId id{Phase19FeatureId::portable_standalone_jacobi_pull};
  Phase19FeatureClassification classification{
      Phase19FeatureClassification::designed_deferred};
  Phase19Fingerprint evidence_fingerprint{};
  std::string finding;

  bool operator==(const Phase19FeatureAudit&) const = default;
};

enum class Phase19ConclusionState : std::uint8_t {
  insufficient_evidence = 0U,
  measured = 1U,
};

enum class Phase19MetricUnit : std::uint8_t {
  none = 0U,
  nanoseconds,
  queries_per_second,
  ratio,
  percentage,
  bytes,
  count,
};

// Selection rows name a selected configuration fingerprint when measured.
// Metric rows additionally use the fixed unit associated with their ID.
enum class Phase19ComparisonId : std::uint16_t {
  best_engine = 0U,
  best_control_jacobi,
  best_control_dense,
  best_control_frontier,
  best_ordinary_k_jacobi,
  best_ordinary_k_dense,
  best_ordinary_k_frontier,
  best_persistent_grid_jacobi,
  best_persistent_grid_dense,
  best_persistent_grid_frontier,
  best_batching_width,
  best_overlap_planner_thresholds,
  best_expansion_schedule,
  latency_p50,
  latency_p95,
  latency_p99,
  all_query_throughput,
  warm_all_query_time,
  cold_pipeline_time,
  box_miss_rate,
  expansion_rate,
  fallback_rate,
  path_quality_summary,
  memory_footprint_summary,
  synchronization_count,
  copy_count,
  profiler_supported_bottleneck,
  count,
};

struct Phase19Comparison {
  Phase19ComparisonId id{Phase19ComparisonId::best_engine};
  Phase19ConclusionState state{
      Phase19ConclusionState::insufficient_evidence};
  std::uint64_t selected_configuration_fingerprint{};
  Phase19MetricUnit metric_unit{Phase19MetricUnit::none};
  double metric_value{};
  Phase19Fingerprint evidence_fingerprint{};
  std::string finding;

  bool operator==(const Phase19Comparison&) const = default;
};

enum class Phase19ProfilerMetricId : std::uint8_t {
  gpu_active_time = 0U,
  l2_hit_percentage,
  l2_read_bytes,
  l2_write_bytes,
  occupancy_percentage,
  memory_unit_busy_percentage,
  wave_count,
  vector_instruction_count,
  scalar_instruction_count,
  memory_instruction_count,
  count,
};

struct Phase19ProfilerMetric {
  Phase19ProfilerMetricId id{Phase19ProfilerMetricId::gpu_active_time};
  Phase19ConclusionState state{
      Phase19ConclusionState::insufficient_evidence};
  std::uint64_t configuration_fingerprint{};
  Phase19MetricUnit metric_unit{Phase19MetricUnit::none};
  double metric_value{};
  Phase19Fingerprint evidence_fingerprint{};
  std::string finding;

  bool operator==(const Phase19ProfilerMetric&) const = default;
};

enum class Phase19QuestionId : std::uint8_t {
  jacobi_gpu_benefit = 0U,
  atomic_elimination_vs_dense_scans,
  chaotic_round_reduction_vs_atomics,
  frontier_work_reduction_vs_queue_overhead,
  cooperative_persistence_vs_chunking,
  overlap_batching_for_utilization,
  dominant_all_query_tail_class,
  count,
};

struct Phase19QuestionAnswer {
  Phase19QuestionId id{Phase19QuestionId::jacobi_gpu_benefit};
  Phase19ConclusionState state{
      Phase19ConclusionState::insufficient_evidence};
  Phase19Fingerprint evidence_fingerprint{};
  std::vector<std::uint64_t> supporting_configuration_fingerprints;
  std::string finding;

  bool operator==(const Phase19QuestionAnswer&) const = default;
};

struct Phase19EvidenceSnapshot {
  // These fields are provenance-bound attestations supplied by a normalized
  // external campaign. The Phase 19 audit validates their consistency and
  // gates conclusions; it does not reconstruct or rank raw benchmark rows.
  std::uint64_t corpus_query_count{};
  std::uint64_t evaluated_query_count{};
  bool representative_query_artifact{};
  bool hip_compiler_validation{};
  bool gpu_device_validation{};
  bool representative_correctness{};
  bool representative_timing{};
  bool dependency_trace{};
  bool compatible_pmc{};
  bool complete_candidate_matrix{};
  bool all_query_evidence{};
  bool tail_attribution_evidence{};
  bool unique_winner{};
  bool comparable_cpu_baseline{};
  bool memory_evidence{};
  bool synchronization_copy_evidence{};
  bool path_quality_evidence{};
  Phase19Fingerprint source_fingerprint{};
  Phase19Fingerprint build_fingerprint{};
  Phase19Fingerprint graph_fingerprint{};
  Phase19Fingerprint query_fingerprint{};
  Phase19Fingerprint workload_fingerprint{};
  Phase19Fingerprint device_fingerprint{};
  Phase19Fingerprint runtime_fingerprint{};
  Phase19Fingerprint protocol_fingerprint{};
  Phase19Fingerprint tail_attribution_fingerprint{};
  Phase19Fingerprint candidate_catalog_fingerprint{};
  Phase19Fingerprint profiler_fingerprint{};

  constexpr bool operator==(const Phase19EvidenceSnapshot&) const noexcept =
      default;
};

enum class Phase19RecommendationBlocker : std::uint8_t {
  missing_representative_query_artifact = 0U,
  missing_hip_compiler_validation,
  missing_gpu_device_validation,
  missing_comparable_cpu_baseline,
  missing_representative_correctness,
  missing_representative_timing,
  missing_dependency_trace,
  missing_compatible_pmc,
  incomplete_candidate_matrix,
  missing_all_query_evidence,
  missing_tail_attribution_evidence,
  no_unique_winner,
  missing_memory_evidence,
  missing_synchronization_copy_evidence,
  missing_path_quality_evidence,
};

enum class Phase19HybridExperimentDisposition : std::uint8_t {
  deferred_until_standalone_results = 0U,
  not_indicated,
  recommended_as_future_experiment,
};

enum class Phase19LoadStrategy : std::uint8_t {
  compiler_uniform = 0U,
  explicit_wave_broadcast,
};

enum class Phase19TransferMode : std::uint8_t {
  compact_paths = 0U,
};

enum class Phase19ReconstructionMode : std::uint8_t {
  incoming_csc_backtracking = 0U,
};

struct Phase19ProductionConfiguration {
  std::uint64_t configuration_fingerprint{};
  BatchedExpansionOptions expansion{};
  BatchVertexStorageStrategy workspace_vertex_storage{
      BatchVertexStorageStrategy::full_graph_vertex_major};
  BatchRunRepresentation run_representation{
      BatchRunRepresentation::retained_per_run_masks};
  std::uint32_t tile_width{};
  std::uint32_t tile_height{};
  std::uint32_t frontier_queue_capacity{};
  Phase19LoadStrategy jacobi_load_strategy{
      Phase19LoadStrategy::compiler_uniform};
  Phase19LoadStrategy dense_load_strategy{
      Phase19LoadStrategy::compiler_uniform};
  Phase19TransferMode transfer_mode{Phase19TransferMode::compact_paths};
  Phase19ReconstructionMode reconstruction_mode{
      Phase19ReconstructionMode::incoming_csc_backtracking};

  constexpr bool operator==(
      const Phase19ProductionConfiguration&) const noexcept = default;
};

struct Phase19ProductionRecommendation {
  Phase19ProductionConfiguration configuration{};
  Phase19Fingerprint evidence_fingerprint{};
  std::string rationale;

  bool operator==(const Phase19ProductionRecommendation&) const = default;
};

struct Phase19AuditReport {
  std::uint32_t schema_version{phase19_audit_schema_version};
  Phase19EvidenceSnapshot evidence{};
  std::vector<Phase19RecommendationBlocker> blockers;
  std::vector<Phase19FeatureAudit> features;
  std::vector<Phase19Comparison> comparisons;
  std::vector<Phase19ProfilerMetric> profiler_metrics;
  std::vector<Phase19QuestionAnswer> questions;
  Phase19HybridExperimentDisposition hybrid_experiment{
      Phase19HybridExperimentDisposition::deferred_until_standalone_results};
  std::string hybrid_finding;
  std::optional<Phase19ProductionRecommendation> production_recommendation;

  bool operator==(const Phase19AuditReport&) const = default;
};

enum class Phase19AuditError : std::uint8_t {
  none = 0U,
  invalid_schema,
  invalid_evidence_snapshot,
  blocker_mismatch,
  invalid_feature_inventory,
  invalid_feature_record,
  premature_performance_profiled_feature,
  invalid_comparison_inventory,
  invalid_comparison_record,
  premature_measured_comparison,
  invalid_profiler_metric_inventory,
  invalid_profiler_metric_record,
  premature_measured_profiler_metric,
  invalid_question_inventory,
  invalid_question_record,
  premature_measured_question,
  invalid_hybrid_disposition,
  premature_hybrid_decision,
  premature_recommendation,
  missing_recommendation,
  invalid_recommendation,
};

struct Phase19AuditValidationResult {
  static constexpr std::size_t no_position = static_cast<std::size_t>(-1);

  Phase19AuditError code{Phase19AuditError::none};
  std::size_t position{no_position};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == Phase19AuditError::none;
  }
};

[[nodiscard]] Phase19AuditReport make_local_phase19_audit();

[[nodiscard]] Phase19AuditValidationResult validate_phase19_audit(
    const Phase19AuditReport& report) noexcept;

[[nodiscard]] std::uint64_t fingerprint_phase19_configuration(
    const Phase19ProductionConfiguration& configuration) noexcept;

// This is the canonical identity required by every measured conclusion and
// recommendation in a report. It binds all snapshot flags, counts, and input,
// workload, device, runtime, protocol, catalog, and profiler fingerprints.
[[nodiscard]] Phase19Fingerprint fingerprint_phase19_evidence_snapshot(
    const Phase19EvidenceSnapshot& evidence) noexcept;

[[nodiscard]] std::string serialize_phase19_audit_tsv(
    const Phase19AuditReport& report);

[[nodiscard]] Phase19AuditReport deserialize_phase19_audit_tsv(
    std::string_view text);

}  // namespace bfnew
