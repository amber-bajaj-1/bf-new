#pragma once

#include "bfnew/gpu_api.hpp"
#include "bfnew/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace bfnew {

inline constexpr std::uint32_t shootout_schema_version = 2U;
inline constexpr std::uint32_t minimum_logicnets_shootout_queries = 1000U;
inline constexpr std::size_t shootout_stratification_dimensions = 5U;
inline constexpr std::size_t shootout_performance_question_count = 7U;

enum class ShootoutWorkloadKind : std::uint8_t {
  logicnets_jscl = 0U,
  synthetic = 1U,
};

// A manifest contains exactly one graph/workload case. Synthetic adversaries
// use separate identities and manifests and never count toward the required
// logicnets_jscl sample.
struct ShootoutWorkloadIdentity {
  ShootoutWorkloadKind kind{ShootoutWorkloadKind::logicnets_jscl};
  std::uint64_t case_id{};
  std::string case_name;

  bool operator==(const ShootoutWorkloadIdentity&) const noexcept = default;
};

enum class ShootoutRunKind : std::uint8_t {
  warmup = 0U,
  correctness = 1U,
  timing = 2U,
  algorithm_counters = 3U,
  trace = 4U,
  pmc = 5U,
};

enum class ShootoutEvidenceState : std::uint8_t {
  unavailable = 0U,
  not_applicable = 1U,
  measured = 2U,
};

struct ShootoutEvidenceValue {
  ShootoutEvidenceState state{ShootoutEvidenceState::unavailable};
  double value{};

  constexpr bool operator==(const ShootoutEvidenceValue&) const noexcept =
      default;
};

struct ShootoutInputFingerprint {
  std::uint64_t graph_words[2]{};
  std::uint64_t query_words[2]{};
  std::uint64_t corpus_query_count{};
  std::uint32_t schema_version{shootout_schema_version};

  constexpr bool operator==(const ShootoutInputFingerprint&) const noexcept =
      default;
};

struct ShootoutQueryFeatures {
  QueryId query_id{};
  std::uint64_t selected_vertices{};
  std::uint64_t selected_edges{};
  std::uint32_t fanout{};
  std::uint32_t source_count{};
  // Completed rounds from the deterministic Jacobi pilot. Pilot execution is
  // selection metadata and never contributes timing evidence.
  std::uint64_t expected_rounds{};

  constexpr bool operator==(const ShootoutQueryFeatures&) const noexcept =
      default;
};

struct ShootoutManifestEntry {
  ShootoutQueryFeatures features{};
  std::array<std::uint8_t, shootout_stratification_dimensions> quantile_bins{};

  constexpr bool operator==(const ShootoutManifestEntry&) const noexcept =
      default;
};

struct ShootoutManifest {
  ShootoutInputFingerprint fingerprint{};
  ShootoutWorkloadIdentity workload{};
  std::uint64_t selection_seed{};
  std::uint64_t order_seed{};
  std::uint32_t warmup_repetitions{};
  std::uint32_t timing_repetitions{};
  std::uint32_t requested_query_count{};
  std::vector<ShootoutManifestEntry> entries;

  bool operator==(const ShootoutManifest&) const noexcept = default;
};

// Instrumentation is a stage property rather than part of the tuning key.
// Timing and counter evidence can therefore join on a stable configuration ID
// without pretending that the two executions are the same sample.
struct ShootoutTuning {
  std::uint32_t configuration_id{};
  EngineKind engine{EngineKind::jacobi_pull};
  ControlMode control_mode{ControlMode::persistent_cooperative};
  std::uint32_t rounds_per_chunk{1U};
  std::uint32_t block_size{128U};
  GridPolicy grid_policy{GridPolicy::occupancy_derived};
  std::uint32_t blocks_per_wgp{};
  std::uint64_t maximum_rounds{1U};

  constexpr bool operator==(const ShootoutTuning&) const noexcept = default;
};

struct ShootoutConfigurationCatalog {
  ShootoutInputFingerprint fingerprint{};
  ShootoutWorkloadIdentity workload{};
  std::vector<ShootoutTuning> tunings;

  bool operator==(const ShootoutConfigurationCatalog&) const noexcept = default;
};

struct ShootoutKernelLimit {
  EngineKind engine{EngineKind::jacobi_pull};
  std::uint32_t block_size{};
  std::uint32_t persistent_active_blocks_per_wgp{};
  bool ordinary_block_size_legal{};
  bool persistent_block_size_legal{};

  constexpr bool operator==(const ShootoutKernelLimit&) const noexcept =
      default;
};

enum class ShootoutConfigurationRejection : std::uint8_t {
  none = 0U,
  invalid_tuning = 1U,
  missing_kernel_limit = 2U,
  illegal_ordinary_block_size = 3U,
  illegal_persistent_block_size = 4U,
  illegal_persistent_residency = 5U,
};

struct ShootoutConfigurationDecision {
  ShootoutTuning tuning{};
  ShootoutConfigurationRejection rejection{
      ShootoutConfigurationRejection::none};
  std::uint32_t measured_persistent_active_blocks_per_wgp{};

  constexpr bool operator==(
      const ShootoutConfigurationDecision&) const noexcept = default;
};

struct ShootoutScheduleEntry {
  ShootoutWorkloadIdentity workload{};
  ShootoutRunKind run_kind{ShootoutRunKind::correctness};
  std::uint64_t execution_ordinal{};
  std::uint32_t repetition{};
  QueryId query_id{};
  std::uint32_t configuration_id{};

  bool operator==(const ShootoutScheduleEntry&) const noexcept = default;
};

struct ShootoutTimingRecord {
  ShootoutEvidenceValue preparation_gpu_milliseconds{};
  // In host-poll modes this event span includes host-idle gaps. It is not
  // active GPU time and is reported separately from the trace-derived value.
  ShootoutEvidenceValue sssp_device_timeline_milliseconds{};
  ShootoutEvidenceValue result_transfer_gpu_milliseconds{};
  ShootoutEvidenceValue end_to_end_wall_milliseconds{};

  constexpr bool operator==(const ShootoutTimingRecord&) const noexcept =
      default;
};

struct ShootoutProfilerProvenance {
  // Stable IDs from the profiler campaign ledger. A PMC counter-set ID names
  // one compatibility-checked pass and may not be synthesized by joining
  // incompatible passes.
  std::uint64_t pass_id{};
  std::uint64_t counter_set_id{};
  bool compatible{};

  constexpr bool operator==(
      const ShootoutProfilerProvenance&) const noexcept = default;
};

struct ShootoutProfilerRecord {
  // Trace-derived sum of active SSSP/control/finalization kernel durations.
  // Profiler-instrumented time never enters ordinary timing distributions.
  ShootoutEvidenceValue gpu_active_milliseconds{};
  ShootoutEvidenceValue l2_hit_percent{};
  ShootoutEvidenceValue l2_read_bytes{};
  ShootoutEvidenceValue l2_write_bytes{};
  ShootoutEvidenceValue occupancy_percent{};
  ShootoutEvidenceValue memory_unit_busy_percent{};
  ShootoutEvidenceValue waves{};
  ShootoutEvidenceValue vector_instructions{};
  ShootoutEvidenceValue scalar_instructions{};
  ShootoutEvidenceValue memory_instructions{};

  constexpr bool operator==(const ShootoutProfilerRecord&) const noexcept =
      default;
};

struct ShootoutSample {
  ShootoutInputFingerprint fingerprint{};
  ShootoutWorkloadIdentity workload{};
  ShootoutRunKind run_kind{ShootoutRunKind::correctness};
  std::uint64_t execution_ordinal{};
  std::uint32_t repetition{};
  QueryId query_id{};
  std::uint32_t configuration_id{};
  InstrumentationLevel instrumentation{InstrumentationLevel::none};
  bool correctness_passed{};
  bool distances_downloaded{};
  GpuSsspResult result{};
  ShootoutTimingRecord timing{};
  ShootoutProfilerProvenance profiler_provenance{};
  ShootoutProfilerRecord profiler{};
  std::uint32_t cooperative_grid_blocks{};
  std::uint32_t cooperative_active_blocks_per_wgp{};
};

struct ShootoutDistribution {
  std::uint64_t count{};
  double minimum{};
  double p50{};
  double p95{};
  double p99{};
  double maximum{};
  double mean{};

  constexpr bool operator==(const ShootoutDistribution&) const noexcept =
      default;
};

enum class ShootoutTailMetric : std::uint8_t {
  wall = 0U,
  gpu_active = 1U,
};

struct ShootoutLongTailRecord {
  ShootoutWorkloadIdentity workload{};
  ShootoutTailMetric metric{ShootoutTailMetric::wall};
  QueryId query_id{};
  std::uint32_t configuration_id{};
  std::uint32_t repetition{};
  double milliseconds{};
  ShootoutQueryFeatures features{};

  bool operator==(const ShootoutLongTailRecord&) const noexcept = default;
};

struct ShootoutProfileMetricSummary {
  ShootoutEvidenceValue mean{};
  std::uint64_t counter_set_id{};
  std::vector<std::uint64_t> pass_ids;

  bool operator==(const ShootoutProfileMetricSummary&) const noexcept = default;
};

struct ShootoutProfilerSummary {
  ShootoutDistribution gpu_active_milliseconds{};
  std::vector<std::uint64_t> gpu_active_pass_ids;
  ShootoutProfileMetricSummary l2_hit_percent{};
  ShootoutProfileMetricSummary l2_read_bytes{};
  ShootoutProfileMetricSummary l2_write_bytes{};
  ShootoutProfileMetricSummary occupancy_percent{};
  ShootoutProfileMetricSummary memory_unit_busy_percent{};
  ShootoutProfileMetricSummary waves{};
  ShootoutProfileMetricSummary vector_instructions{};
  ShootoutProfileMetricSummary scalar_instructions{};
  ShootoutProfileMetricSummary memory_instructions{};

  bool operator==(const ShootoutProfilerSummary&) const noexcept = default;
};

struct ShootoutTuningSummary {
  ShootoutTuning tuning{};
  std::uint32_t cooperative_grid_blocks{};
  std::uint32_t cooperative_active_blocks_per_wgp{};
  ShootoutDistribution wall_milliseconds{};
  ShootoutDistribution device_timeline_milliseconds{};
  ShootoutEvidenceValue throughput_queries_per_second{};
  ShootoutEvidenceState algorithm_counters_state{
      ShootoutEvidenceState::unavailable};
  std::uint64_t total_rounds{};
  std::uint64_t total_edges_examined{};
  std::uint64_t total_successful_decreases{};
  ShootoutEvidenceValue useful_decrease_ratio{};
  std::uint64_t total_active_vertices{};
  std::uint64_t total_active_lane_rounds{};
  std::uint64_t maximum_queue_size{};
  std::uint64_t host_checks{};
  std::uint64_t atomic_attempts{};
  std::uint64_t successful_atomic_updates{};
  std::uint64_t queue_claims{};
  std::uint64_t duplicate_suppressions{};
  std::uint64_t expansion_count{};
  std::uint64_t mask_operations{};
  std::uint64_t overflow_events{};
  std::uint64_t high_contention_destinations{};
  std::uint64_t changed_flag_updates{};
  std::uint64_t full_edge_rounds{};
  std::uint64_t empty_frontier_rounds{};
  std::uint64_t small_frontier_rounds{};
  std::uint64_t kernel_dispatches{};
  std::uint64_t host_synchronizations{};
  std::uint64_t controller_copies{};
  ShootoutProfilerSummary profiler{};
  std::vector<ShootoutLongTailRecord> long_tail;
};

struct ShootoutEvidenceReference {
  ShootoutRunKind run_kind{ShootoutRunKind::timing};
  std::uint64_t execution_ordinal{};
  std::uint64_t profiler_pass_id{};
  std::uint64_t profiler_counter_set_id{};
  QueryId query_id{};
  std::uint32_t configuration_id{};

  constexpr bool operator==(const ShootoutEvidenceReference&) const noexcept =
      default;
};

enum class ShootoutConclusionState : std::uint8_t {
  pending = 0U,
  insufficient_evidence = 1U,
  measured = 2U,
};

struct ShootoutQuestionAnswer {
  std::uint32_t question_id{};
  ShootoutConclusionState state{ShootoutConclusionState::pending};
  ShootoutWorkloadIdentity workload{};
  std::vector<std::uint32_t> configuration_ids;
  std::vector<ShootoutEvidenceReference> evidence;
  std::string conclusion;

  bool operator==(const ShootoutQuestionAnswer&) const noexcept = default;
};

struct ShootoutRecommendation {
  ShootoutConclusionState state{ShootoutConclusionState::pending};
  ShootoutWorkloadIdentity workload{};
  std::vector<std::uint32_t> configuration_ids;
  std::vector<ShootoutEvidenceReference> evidence;
  std::string rationale;
  bool toggles_remain_configurable{true};

  bool operator==(const ShootoutRecommendation&) const noexcept = default;
};

struct ShootoutCampaignReport {
  ShootoutInputFingerprint fingerprint{};
  ShootoutWorkloadIdentity workload{};
  std::uint64_t selection_seed{};
  std::uint64_t order_seed{};
  std::uint32_t warmup_repetitions{};
  std::uint32_t timing_repetitions{};
  std::uint32_t selected_query_count{};
  std::vector<ShootoutManifestEntry> queries;
  std::vector<ShootoutTuningSummary> summaries;
  std::vector<ShootoutEvidenceReference> supplied_evidence;
  std::array<ShootoutQuestionAnswer, shootout_performance_question_count>
      answers;
  std::vector<ShootoutRecommendation> recommendations;
};

[[nodiscard]] std::vector<ShootoutTuning> make_shootout_tunings(
    std::uint64_t maximum_rounds,
    std::span<const std::uint32_t> fixed_persistent_blocks_per_wgp);

[[nodiscard]] std::vector<ShootoutConfigurationDecision>
resolve_shootout_configurations(
    std::span<const ShootoutTuning> tunings,
    std::span<const ShootoutKernelLimit> limits);

[[nodiscard]] ShootoutManifest select_logicnets_shootout_manifest(
    const ShootoutInputFingerprint& fingerprint,
    const ShootoutWorkloadIdentity& workload,
    std::span<const ShootoutQueryFeatures> candidates,
    std::uint32_t requested_query_count,
    std::uint64_t selection_seed,
    std::uint64_t order_seed,
    std::uint32_t warmup_repetitions,
    std::uint32_t timing_repetitions);

[[nodiscard]] ShootoutManifest make_synthetic_shootout_manifest(
    const ShootoutInputFingerprint& fingerprint,
    const ShootoutWorkloadIdentity& workload,
    std::span<const ShootoutQueryFeatures> cases,
    std::uint64_t order_seed,
    std::uint32_t warmup_repetitions,
    std::uint32_t timing_repetitions);

[[nodiscard]] std::vector<ShootoutScheduleEntry>
make_interleaved_shootout_schedule(
    const ShootoutManifest& manifest,
    std::span<const ShootoutTuning> tunings,
    ShootoutRunKind run_kind,
    std::uint32_t repetitions,
    std::uint64_t order_seed);

[[nodiscard]] std::vector<ShootoutScheduleEntry>
make_grouped_shootout_correctness_schedule(
    const ShootoutManifest& manifest,
    std::span<const ShootoutTuning> tunings);

void validate_shootout_configuration_catalog(
    const ShootoutConfigurationCatalog& catalog);

void validate_shootout_samples(
    const ShootoutManifest& manifest,
    std::span<const ShootoutTuning> tunings,
    std::span<const ShootoutSample> samples,
    ShootoutRunKind expected_kind);

void require_complete_correctness_gate(
    const ShootoutManifest& manifest,
    std::span<const ShootoutTuning> tunings,
    std::span<const ShootoutSample> correctness_samples,
    std::span<const ShootoutSample> gated_samples);

[[nodiscard]] ShootoutDistribution shootout_distribution(
    std::span<const double> values);

[[nodiscard]] ShootoutCampaignReport summarize_shootout_campaign(
    const ShootoutManifest& manifest,
    std::span<const ShootoutTuning> tunings,
    std::span<const ShootoutSample> correctness_samples,
    std::span<const ShootoutSample> timing_samples,
    std::span<const ShootoutSample> counter_samples,
    std::span<const ShootoutSample> trace_samples,
    std::span<const ShootoutSample> pmc_samples);

void validate_shootout_report(const ShootoutCampaignReport& report);

[[nodiscard]] std::string serialize_shootout_manifest_tsv(
    const ShootoutManifest& manifest);
[[nodiscard]] ShootoutManifest deserialize_shootout_manifest_tsv(
    std::string_view text);

[[nodiscard]] std::string serialize_shootout_catalog_tsv(
    const ShootoutConfigurationCatalog& catalog);
[[nodiscard]] ShootoutConfigurationCatalog deserialize_shootout_catalog_tsv(
    std::string_view text);

[[nodiscard]] std::string serialize_shootout_samples_tsv(
    const ShootoutManifest& manifest,
    std::span<const ShootoutTuning> tunings,
    std::span<const ShootoutSample> samples,
    ShootoutRunKind run_kind);
[[nodiscard]] std::vector<ShootoutSample> deserialize_shootout_samples_tsv(
    const ShootoutManifest& manifest,
    std::span<const ShootoutTuning> tunings,
    std::string_view text,
    ShootoutRunKind run_kind);

[[nodiscard]] std::string serialize_shootout_report_json(
    const ShootoutCampaignReport& report);

}  // namespace bfnew
