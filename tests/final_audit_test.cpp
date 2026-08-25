#include "bfnew/final_audit.hpp"
#include "bfnew/engine_shootout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

int failures = 0;

void expect(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

[[nodiscard]] constexpr bool zero_fingerprint(
    const bfnew::Phase19Fingerprint& value) noexcept {
  return value.words[0] == 0U && value.words[1] == 0U;
}

[[nodiscard]] constexpr bfnew::Phase19Fingerprint fingerprint(
    const std::uint64_t value) noexcept {
  return bfnew::Phase19Fingerprint{{value, value + 1000U}};
}

void expect_error(
    const bfnew::Phase19AuditReport& report,
    const bfnew::Phase19AuditError expected,
    const std::string_view message) {
  const bfnew::Phase19AuditValidationResult result =
      bfnew::validate_phase19_audit(report);
  if (result.code != expected) {
    std::cerr << "FAIL: " << message << " (expected error "
              << static_cast<std::uint32_t>(expected) << ", got "
              << static_cast<std::uint32_t>(result.code) << ")\n";
    ++failures;
  }
}

template <class Callable>
void expect_invalid_argument(
    Callable&& callable,
    const std::string_view message) {
  bool threw = false;
  try {
    std::forward<Callable>(callable)();
  } catch (const std::invalid_argument&) {
    threw = true;
  } catch (...) {
  }
  expect(threw, message);
}

void replace_once(
    std::string& text,
    const std::string_view from,
    const std::string_view to) {
  const std::size_t position = text.find(from);
  if (position == std::string::npos) {
    throw std::logic_error{"test mutation pattern is absent"};
  }
  text.replace(position, from.size(), to);
}

[[nodiscard]] std::size_t count_rows(
    const std::string_view text,
    const std::string_view prefix) {
  std::size_t count = 0U;
  std::size_t begin = 0U;
  while (begin < text.size()) {
    const std::size_t end = text.find('\n', begin);
    const std::string_view line = text.substr(begin, end - begin);
    if (line.starts_with(prefix)) {
      ++count;
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  return count;
}

void erase_blocker(
    bfnew::Phase19AuditReport& report,
    const bfnew::Phase19RecommendationBlocker blocker) {
  const auto position =
      std::find(report.blockers.begin(), report.blockers.end(), blocker);
  if (position == report.blockers.end()) {
    throw std::logic_error{"test blocker is absent"};
  }
  report.blockers.erase(position);
}

[[nodiscard]] bfnew::Phase19ProductionConfiguration
make_valid_production_configuration() {
  bfnew::Phase19ProductionConfiguration configuration;
  configuration.expansion.run_options.engine = bfnew::EngineKind::jacobi_pull;
  configuration.expansion.run_options.control_mode =
      bfnew::ControlMode::chunked_host_poll;
  configuration.expansion.run_options.rounds_per_chunk = 8U;
  configuration.expansion.run_options.block_size = 256U;
  configuration.expansion.run_options.grid_policy =
      bfnew::GridPolicy::occupancy_derived;
  configuration.expansion.run_options.blocks_per_wgp = 0U;
  configuration.expansion.run_options.instrumentation =
      bfnew::InstrumentationLevel::none;
  configuration.expansion.run_options.maximum_rounds = 100U;
  configuration.expansion.run_options.enable_per_lane_convergence = 1U;
  configuration.expansion.planner_policy.lane_width = 32U;
  configuration.expansion.execution_configuration_fingerprint = 0x1234U;
  configuration.expansion.schedule = bfnew::one_ring_expansion();
  configuration.expansion.maximum_expansions = 4U;
  configuration.expansion.terminal_policy =
      bfnew::ExpansionTerminalPolicy::full_region_fallback;
  configuration.expansion.enable_compact_paths = 1U;
  configuration.workspace_vertex_storage =
      bfnew::BatchVertexStorageStrategy::full_graph_vertex_major;
  configuration.run_representation =
      bfnew::BatchRunRepresentation::device_materialized_run_masks;
  configuration.tile_width = 8U;
  configuration.tile_height = 8U;
  configuration.frontier_queue_capacity = 0U;
  configuration.configuration_fingerprint =
      bfnew::fingerprint_phase19_configuration(configuration);
  return configuration;
}

[[nodiscard]] bfnew::Phase19EvidenceSnapshot make_complete_snapshot() {
  bfnew::Phase19EvidenceSnapshot evidence;
  evidence.corpus_query_count = bfnew::minimum_logicnets_shootout_queries;
  evidence.evaluated_query_count = evidence.corpus_query_count;
  evidence.representative_query_artifact = true;
  evidence.hip_compiler_validation = true;
  evidence.gpu_device_validation = true;
  evidence.representative_correctness = true;
  evidence.representative_timing = true;
  evidence.dependency_trace = true;
  evidence.compatible_pmc = true;
  evidence.complete_candidate_matrix = true;
  evidence.all_query_evidence = true;
  evidence.tail_attribution_evidence = true;
  evidence.unique_winner = true;
  evidence.comparable_cpu_baseline = true;
  evidence.memory_evidence = true;
  evidence.synchronization_copy_evidence = true;
  evidence.path_quality_evidence = true;
  evidence.source_fingerprint = fingerprint(1U);
  evidence.build_fingerprint = fingerprint(2U);
  evidence.graph_fingerprint = fingerprint(3U);
  evidence.query_fingerprint = fingerprint(4U);
  evidence.workload_fingerprint = fingerprint(5U);
  evidence.device_fingerprint = fingerprint(6U);
  evidence.runtime_fingerprint = fingerprint(7U);
  evidence.protocol_fingerprint = fingerprint(8U);
  evidence.tail_attribution_fingerprint = fingerprint(9U);
  evidence.candidate_catalog_fingerprint = fingerprint(10U);
  evidence.profiler_fingerprint = fingerprint(11U);
  return evidence;
}

void set_measured_metric_values(bfnew::Phase19AuditReport& report) {
  for (bfnew::Phase19Comparison& comparison : report.comparisons) {
    switch (comparison.id) {
      case bfnew::Phase19ComparisonId::latency_p50:
        comparison.metric_value = 10.0;
        break;
      case bfnew::Phase19ComparisonId::latency_p95:
        comparison.metric_value = 20.0;
        break;
      case bfnew::Phase19ComparisonId::latency_p99:
        comparison.metric_value = 30.0;
        break;
      case bfnew::Phase19ComparisonId::box_miss_rate:
        comparison.metric_value = 0.25;
        break;
      case bfnew::Phase19ComparisonId::expansion_rate:
        comparison.metric_value = 0.5;
        break;
      case bfnew::Phase19ComparisonId::fallback_rate:
        comparison.metric_value = 0.125;
        break;
      default:
        comparison.metric_value =
            comparison.metric_unit == bfnew::Phase19MetricUnit::none ? 0.0
                                                                     : 42.0;
        break;
    }
  }
}

[[nodiscard]] bfnew::Phase19AuditReport make_complete_report() {
  bfnew::Phase19AuditReport report = bfnew::make_local_phase19_audit();
  report.evidence = make_complete_snapshot();
  report.blockers.clear();
  const bfnew::Phase19Fingerprint evidence_identity =
      bfnew::fingerprint_phase19_evidence_snapshot(report.evidence);

  for (bfnew::Phase19FeatureAudit& feature : report.features) {
    if (feature.classification !=
        bfnew::Phase19FeatureClassification::implemented_correctness_tested) {
      feature.classification = bfnew::Phase19FeatureClassification::
          implemented_performance_profiled;
      feature.evidence_fingerprint = evidence_identity;
    }
  }

  const bfnew::Phase19ProductionConfiguration configuration =
      make_valid_production_configuration();
  for (bfnew::Phase19Comparison& comparison : report.comparisons) {
    comparison.state = bfnew::Phase19ConclusionState::measured;
    comparison.selected_configuration_fingerprint =
        configuration.configuration_fingerprint;
    comparison.evidence_fingerprint = evidence_identity;
    comparison.finding = "measured representative comparison";
  }
  set_measured_metric_values(report);

  for (bfnew::Phase19ProfilerMetric& metric : report.profiler_metrics) {
    metric.state = bfnew::Phase19ConclusionState::measured;
    metric.configuration_fingerprint =
        configuration.configuration_fingerprint;
    metric.metric_value =
        metric.metric_unit == bfnew::Phase19MetricUnit::percentage ? 50.0
                                                                   : 42.0;
    metric.evidence_fingerprint = evidence_identity;
    metric.finding = "measured representative profiler metric";
  }

  for (bfnew::Phase19QuestionAnswer& answer : report.questions) {
    answer.state = bfnew::Phase19ConclusionState::measured;
    answer.evidence_fingerprint = evidence_identity;
    answer.supporting_configuration_fingerprints = {
        configuration.configuration_fingerprint};
    answer.finding = "measured representative answer";
  }
  const std::uint64_t selected = configuration.configuration_fingerprint;
  const std::uint64_t additional =
      selected == std::numeric_limits<std::uint64_t>::max() ? selected - 1U
                                                            : selected + 1U;
  report.questions.front().supporting_configuration_fingerprints = {
      std::min(selected, additional), std::max(selected, additional)};
  report.hybrid_experiment =
      bfnew::Phase19HybridExperimentDisposition::not_indicated;
  report.hybrid_finding = "standalone evidence does not indicate a hybrid";
  report.production_recommendation = bfnew::Phase19ProductionRecommendation{
      configuration, evidence_identity, "measured configurable default"};
  return report;
}

void test_local_report_inventory() {
  static_assert(
      static_cast<std::uint8_t>(
          bfnew::Phase19FeatureClassification::implemented_correctness_tested) ==
      0U);
  static_assert(
      static_cast<std::uint8_t>(
          bfnew::Phase19FeatureClassification::implemented_performance_profiled) ==
      1U);
  static_assert(
      static_cast<std::uint8_t>(
          bfnew::Phase19FeatureClassification::implemented_not_representative) ==
      2U);
  static_assert(
      static_cast<std::uint8_t>(
          bfnew::Phase19FeatureClassification::designed_deferred) == 3U);

  const bfnew::Phase19AuditReport report = bfnew::make_local_phase19_audit();
  expect(
      bfnew::validate_phase19_audit(report).ok(),
      "canonical local audit passes deep validation");
  expect(
      report.schema_version == bfnew::phase19_audit_schema_version,
      "canonical local audit publishes the exact schema version");
  expect(
      report.evidence == bfnew::Phase19EvidenceSnapshot{},
      "local evidence snapshot contains no invented representative evidence");

  const std::array<bfnew::Phase19RecommendationBlocker, 15U> blockers{
      bfnew::Phase19RecommendationBlocker::
          missing_representative_query_artifact,
      bfnew::Phase19RecommendationBlocker::missing_hip_compiler_validation,
      bfnew::Phase19RecommendationBlocker::missing_gpu_device_validation,
      bfnew::Phase19RecommendationBlocker::missing_comparable_cpu_baseline,
      bfnew::Phase19RecommendationBlocker::missing_representative_correctness,
      bfnew::Phase19RecommendationBlocker::missing_representative_timing,
      bfnew::Phase19RecommendationBlocker::missing_dependency_trace,
      bfnew::Phase19RecommendationBlocker::missing_compatible_pmc,
      bfnew::Phase19RecommendationBlocker::incomplete_candidate_matrix,
      bfnew::Phase19RecommendationBlocker::missing_all_query_evidence,
      bfnew::Phase19RecommendationBlocker::missing_tail_attribution_evidence,
      bfnew::Phase19RecommendationBlocker::no_unique_winner,
      bfnew::Phase19RecommendationBlocker::missing_memory_evidence,
      bfnew::Phase19RecommendationBlocker::
          missing_synchronization_copy_evidence,
      bfnew::Phase19RecommendationBlocker::missing_path_quality_evidence,
  };
  expect(
      std::equal(
          report.blockers.begin(),
          report.blockers.end(),
          blockers.begin(),
          blockers.end()),
      "local blocker inventory is exhaustive and canonically ordered");

  const std::size_t feature_count =
      static_cast<std::size_t>(bfnew::Phase19FeatureId::count);
  expect(
      report.features.size() == feature_count,
      "local feature inventory is exhaustive");
  std::array<std::size_t, 4U> classifications{};
  for (std::size_t index = 0U; index < report.features.size(); ++index) {
    const bfnew::Phase19FeatureAudit& feature = report.features[index];
    expect(
        static_cast<std::size_t>(feature.id) == index,
        "feature IDs are unique and canonical");
    const std::size_t classification =
        static_cast<std::size_t>(feature.classification);
    expect(classification < classifications.size(), "feature class is known");
    if (classification < classifications.size()) {
      ++classifications[classification];
    }
    expect(
        zero_fingerprint(feature.evidence_fingerprint),
        "local feature has no representative evidence fingerprint");
    expect(!feature.finding.empty(), "local feature has an explicit finding");
  }
  expect(
      classifications == std::array<std::size_t, 4U>{36U, 0U, 12U, 8U},
      "local feature classes are exactly 36 correctness, 0 profiled, 12 "
      "not-representative, and 8 deferred");
  expect(
      report.features[static_cast<std::size_t>(
          bfnew::Phase19FeatureId::portable_standalone_jacobi_pull)]
                  .classification ==
              bfnew::Phase19FeatureClassification::
                  implemented_correctness_tested &&
          report.features[static_cast<std::size_t>(
              bfnew::Phase19FeatureId::hip_standalone_jacobi_pull)]
                  .classification ==
              bfnew::Phase19FeatureClassification::
                  implemented_not_representative,
      "portable correctness and HIP device evidence remain separate scopes");

  const std::array<bfnew::Phase19ComparisonId, 27U> comparison_ids{
      bfnew::Phase19ComparisonId::best_engine,
      bfnew::Phase19ComparisonId::best_control_jacobi,
      bfnew::Phase19ComparisonId::best_control_dense,
      bfnew::Phase19ComparisonId::best_control_frontier,
      bfnew::Phase19ComparisonId::best_ordinary_k_jacobi,
      bfnew::Phase19ComparisonId::best_ordinary_k_dense,
      bfnew::Phase19ComparisonId::best_ordinary_k_frontier,
      bfnew::Phase19ComparisonId::best_persistent_grid_jacobi,
      bfnew::Phase19ComparisonId::best_persistent_grid_dense,
      bfnew::Phase19ComparisonId::best_persistent_grid_frontier,
      bfnew::Phase19ComparisonId::best_batching_width,
      bfnew::Phase19ComparisonId::best_overlap_planner_thresholds,
      bfnew::Phase19ComparisonId::best_expansion_schedule,
      bfnew::Phase19ComparisonId::latency_p50,
      bfnew::Phase19ComparisonId::latency_p95,
      bfnew::Phase19ComparisonId::latency_p99,
      bfnew::Phase19ComparisonId::all_query_throughput,
      bfnew::Phase19ComparisonId::warm_all_query_time,
      bfnew::Phase19ComparisonId::cold_pipeline_time,
      bfnew::Phase19ComparisonId::box_miss_rate,
      bfnew::Phase19ComparisonId::expansion_rate,
      bfnew::Phase19ComparisonId::fallback_rate,
      bfnew::Phase19ComparisonId::path_quality_summary,
      bfnew::Phase19ComparisonId::memory_footprint_summary,
      bfnew::Phase19ComparisonId::synchronization_count,
      bfnew::Phase19ComparisonId::copy_count,
      bfnew::Phase19ComparisonId::profiler_supported_bottleneck,
  };
  const std::array<bfnew::Phase19MetricUnit, 27U> units{
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::none,
      bfnew::Phase19MetricUnit::nanoseconds,
      bfnew::Phase19MetricUnit::nanoseconds,
      bfnew::Phase19MetricUnit::nanoseconds,
      bfnew::Phase19MetricUnit::queries_per_second,
      bfnew::Phase19MetricUnit::nanoseconds,
      bfnew::Phase19MetricUnit::nanoseconds,
      bfnew::Phase19MetricUnit::ratio,
      bfnew::Phase19MetricUnit::ratio,
      bfnew::Phase19MetricUnit::ratio,
      bfnew::Phase19MetricUnit::ratio,
      bfnew::Phase19MetricUnit::bytes,
      bfnew::Phase19MetricUnit::count,
      bfnew::Phase19MetricUnit::count,
      bfnew::Phase19MetricUnit::none,
  };
  expect(
      report.comparisons.size() == comparison_ids.size() &&
          report.comparisons.size() == units.size(),
      "all 27 required comparison rows are present");
  for (std::size_t index = 0U; index < report.comparisons.size(); ++index) {
    const bfnew::Phase19Comparison& comparison = report.comparisons[index];
    expect(
        comparison.id == comparison_ids[index] &&
            static_cast<std::size_t>(comparison.id) == index,
        "comparison IDs are unique and canonical");
    expect(
        comparison.state ==
                bfnew::Phase19ConclusionState::insufficient_evidence &&
            comparison.metric_unit == units[index] &&
            comparison.metric_value == 0.0 &&
            comparison.selected_configuration_fingerprint == 0U &&
            zero_fingerprint(comparison.evidence_fingerprint),
        "local comparison is unavailable without value, provenance, or "
        "configuration");
  }

  const std::array<bfnew::Phase19ProfilerMetricId, 10U> profiler_metric_ids{
      bfnew::Phase19ProfilerMetricId::gpu_active_time,
      bfnew::Phase19ProfilerMetricId::l2_hit_percentage,
      bfnew::Phase19ProfilerMetricId::l2_read_bytes,
      bfnew::Phase19ProfilerMetricId::l2_write_bytes,
      bfnew::Phase19ProfilerMetricId::occupancy_percentage,
      bfnew::Phase19ProfilerMetricId::memory_unit_busy_percentage,
      bfnew::Phase19ProfilerMetricId::wave_count,
      bfnew::Phase19ProfilerMetricId::vector_instruction_count,
      bfnew::Phase19ProfilerMetricId::scalar_instruction_count,
      bfnew::Phase19ProfilerMetricId::memory_instruction_count,
  };
  const std::array<bfnew::Phase19MetricUnit, 10U> profiler_metric_units{
      bfnew::Phase19MetricUnit::nanoseconds,
      bfnew::Phase19MetricUnit::percentage,
      bfnew::Phase19MetricUnit::bytes,
      bfnew::Phase19MetricUnit::bytes,
      bfnew::Phase19MetricUnit::percentage,
      bfnew::Phase19MetricUnit::percentage,
      bfnew::Phase19MetricUnit::count,
      bfnew::Phase19MetricUnit::count,
      bfnew::Phase19MetricUnit::count,
      bfnew::Phase19MetricUnit::count,
  };
  expect(
      report.profiler_metrics.size() == profiler_metric_ids.size(),
      "all ten trace/PMC profiler metric rows are present");
  for (std::size_t index = 0U; index < report.profiler_metrics.size(); ++index) {
    const bfnew::Phase19ProfilerMetric& metric = report.profiler_metrics[index];
    expect(
        metric.id == profiler_metric_ids[index] &&
            static_cast<std::size_t>(metric.id) == index,
        "profiler metric IDs are unique and canonical");
    expect(
        metric.state ==
                bfnew::Phase19ConclusionState::insufficient_evidence &&
            metric.configuration_fingerprint == 0U &&
            metric.metric_unit == profiler_metric_units[index] &&
            metric.metric_value == 0.0 &&
            zero_fingerprint(metric.evidence_fingerprint) &&
            !metric.finding.empty(),
        "local profiler metric is unavailable without value, provenance, or "
        "configuration");
  }

  const std::array<bfnew::Phase19QuestionId, 7U> question_ids{
      bfnew::Phase19QuestionId::jacobi_gpu_benefit,
      bfnew::Phase19QuestionId::atomic_elimination_vs_dense_scans,
      bfnew::Phase19QuestionId::chaotic_round_reduction_vs_atomics,
      bfnew::Phase19QuestionId::frontier_work_reduction_vs_queue_overhead,
      bfnew::Phase19QuestionId::cooperative_persistence_vs_chunking,
      bfnew::Phase19QuestionId::overlap_batching_for_utilization,
      bfnew::Phase19QuestionId::dominant_all_query_tail_class,
  };
  expect(
      report.questions.size() == question_ids.size() &&
          report.questions.size() ==
              static_cast<std::size_t>(bfnew::Phase19QuestionId::count),
      "all seven Phase 19 questions are present");
  for (std::size_t index = 0U; index < report.questions.size(); ++index) {
    const bfnew::Phase19QuestionAnswer& answer = report.questions[index];
    expect(
        answer.id == question_ids[index] &&
            static_cast<std::size_t>(answer.id) == index,
        "question IDs are unique and canonical");
    expect(
        answer.state ==
                bfnew::Phase19ConclusionState::insufficient_evidence &&
            zero_fingerprint(answer.evidence_fingerprint) &&
            answer.supporting_configuration_fingerprints.empty() &&
            !answer.finding.empty(),
        "local question is explicitly insufficient without invented evidence");
  }
  expect(
      report.hybrid_experiment == bfnew::Phase19HybridExperimentDisposition::
                                      deferred_until_standalone_results &&
          !report.hybrid_finding.empty() &&
          !report.production_recommendation.has_value(),
      "local audit defers the hybrid decision and recommends no configuration");
}

void test_local_validation_failures() {
  const bfnew::Phase19AuditReport local = bfnew::make_local_phase19_audit();

  {
    bfnew::Phase19AuditReport malformed = local;
    ++malformed.schema_version;
    expect_error(
        malformed, bfnew::Phase19AuditError::invalid_schema, "schema mismatch");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.evidence.corpus_query_count = 1U;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_evidence_snapshot,
        "missing artifact cannot carry a corpus count");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.evidence.gpu_device_validation = true;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_evidence_snapshot,
        "device validation cannot appear without artifact and HIP compiler");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.evidence.graph_fingerprint = fingerprint(1U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_evidence_snapshot,
        "absent representative artifact cannot carry a graph identity");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.blockers.pop_back();
    expect_error(
        malformed,
        bfnew::Phase19AuditError::blocker_mismatch,
        "missing blocker is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    std::swap(malformed.blockers[0U], malformed.blockers[1U]);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::blocker_mismatch,
        "reordered blockers are rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.blockers.push_back(malformed.blockers.back());
    expect_error(
        malformed,
        bfnew::Phase19AuditError::blocker_mismatch,
        "duplicate blocker is rejected");
  }

  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.features.pop_back();
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_feature_inventory,
        "missing feature is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.features[1U].id = malformed.features[0U].id;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_feature_inventory,
        "duplicate feature ID is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    std::swap(malformed.features[0U], malformed.features[1U]);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_feature_inventory,
        "reordered features are rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.features[0U].id = bfnew::Phase19FeatureId::count;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_feature_inventory,
        "unknown feature ID is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.features[0U].classification =
        static_cast<bfnew::Phase19FeatureClassification>(255U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_feature_record,
        "unknown feature classification is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.features[0U].finding.clear();
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_feature_record,
        "empty feature finding is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.features[0U].classification =
        bfnew::Phase19FeatureClassification::implemented_performance_profiled;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::premature_performance_profiled_feature,
        "class 2 is forbidden before every representative gate passes");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.features[0U].classification =
        bfnew::Phase19FeatureClassification::implemented_not_representative;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_feature_record,
        "local host correctness cannot be relabeled as HIP source evidence");
  }

  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons.pop_back();
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_inventory,
        "missing comparison is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons[1U].id = malformed.comparisons[0U].id;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_inventory,
        "duplicate comparison ID is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    std::swap(malformed.comparisons[0U], malformed.comparisons[1U]);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_inventory,
        "reordered comparisons are rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons[0U].id = bfnew::Phase19ComparisonId::count;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_inventory,
        "unknown comparison ID is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons[0U].state =
        static_cast<bfnew::Phase19ConclusionState>(255U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "unknown comparison state is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons[static_cast<std::size_t>(
        bfnew::Phase19ComparisonId::latency_p50)]
        .metric_unit = bfnew::Phase19MetricUnit::none;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "comparison unit mismatch is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons[0U].metric_unit =
        static_cast<bfnew::Phase19MetricUnit>(255U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "unknown comparison unit is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons[0U].selected_configuration_fingerprint = 1U;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "unavailable comparison cannot select a configuration");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons[static_cast<std::size_t>(
        bfnew::Phase19ComparisonId::latency_p50)]
        .metric_value = 1.0;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "unavailable comparison cannot carry a numeric value");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons[0U].evidence_fingerprint = fingerprint(1U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "unavailable comparison cannot carry provenance");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons[0U].state =
        bfnew::Phase19ConclusionState::measured;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::premature_measured_comparison,
        "measured comparison is forbidden before representative evidence");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons[0U].metric_value =
        std::numeric_limits<double>::quiet_NaN();
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "nonfinite comparison value is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.comparisons[0U].metric_value = -0.0;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "signed negative zero is rejected from canonical comparisons");
  }

  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.profiler_metrics.pop_back();
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_inventory,
        "missing profiler metric is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.profiler_metrics[1U].id = malformed.profiler_metrics[0U].id;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_inventory,
        "duplicate profiler metric ID is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    std::swap(malformed.profiler_metrics[0U], malformed.profiler_metrics[1U]);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_inventory,
        "reordered profiler metrics are rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.profiler_metrics[0U].id = bfnew::Phase19ProfilerMetricId::count;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_inventory,
        "unknown profiler metric ID is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.profiler_metrics[0U].state =
        static_cast<bfnew::Phase19ConclusionState>(255U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_record,
        "unknown profiler metric state is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.profiler_metrics[0U].metric_unit =
        bfnew::Phase19MetricUnit::count;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_record,
        "profiler metric unit mismatch is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.profiler_metrics[0U].configuration_fingerprint = 1U;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_record,
        "unavailable profiler metric cannot select a configuration");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.profiler_metrics[0U].metric_value = 1.0;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_record,
        "unavailable profiler metric cannot carry a numeric value");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.profiler_metrics[0U].metric_value = -0.0;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_record,
        "signed negative zero is rejected from canonical profiler metrics");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.profiler_metrics[0U].evidence_fingerprint = fingerprint(1U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_record,
        "unavailable profiler metric cannot carry provenance");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.profiler_metrics[0U].state =
        bfnew::Phase19ConclusionState::measured;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::premature_measured_profiler_metric,
        "measured profiler metric is forbidden before profiler evidence");
  }

  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.questions.pop_back();
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_question_inventory,
        "missing question is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.questions[1U].id = malformed.questions[0U].id;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_question_inventory,
        "duplicate question ID is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    std::swap(malformed.questions[0U], malformed.questions[1U]);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_question_inventory,
        "reordered questions are rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.questions[0U].id = bfnew::Phase19QuestionId::count;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_question_inventory,
        "unknown question ID is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.questions[0U].state =
        static_cast<bfnew::Phase19ConclusionState>(255U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_question_record,
        "unknown question state is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.questions[0U].evidence_fingerprint = fingerprint(1U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_question_record,
        "unavailable question cannot carry provenance");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.questions[0U].supporting_configuration_fingerprints = {1U};
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_question_record,
        "unavailable question cannot cite a configuration");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.questions[0U].state = bfnew::Phase19ConclusionState::measured;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::premature_measured_question,
        "measured answer is forbidden before representative evidence");
  }

  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.hybrid_experiment =
        static_cast<bfnew::Phase19HybridExperimentDisposition>(255U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_hybrid_disposition,
        "unknown hybrid disposition is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.hybrid_experiment =
        bfnew::Phase19HybridExperimentDisposition::not_indicated;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::premature_hybrid_decision,
        "hybrid decision is forbidden before standalone evidence is clear");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.hybrid_finding.clear();
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_hybrid_disposition,
        "empty hybrid finding is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = local;
    malformed.production_recommendation =
        bfnew::Phase19ProductionRecommendation{};
    expect_error(
        malformed,
        bfnew::Phase19AuditError::premature_recommendation,
        "production recommendation is forbidden before every gate passes");
  }
}

void test_incremental_evidence_and_tail_gate() {
  {
    bfnew::Phase19AuditReport malformed = bfnew::make_local_phase19_audit();
    malformed.evidence = make_complete_snapshot();
    ++malformed.evidence.evaluated_query_count;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_evidence_snapshot,
        "evaluated query count cannot exceed the corpus count");
  }
  {
    bfnew::Phase19AuditReport malformed = bfnew::make_local_phase19_audit();
    malformed.evidence = make_complete_snapshot();
    malformed.evidence.candidate_catalog_fingerprint = {};
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_evidence_snapshot,
        "complete candidate matrix requires its catalog identity");
  }
  {
    bfnew::Phase19AuditReport malformed = bfnew::make_local_phase19_audit();
    malformed.evidence = make_complete_snapshot();
    malformed.evidence.profiler_fingerprint = {};
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_evidence_snapshot,
        "trace and PMC attestations require their profiler identity");
  }
  {
    bfnew::Phase19AuditReport compiler_only =
        bfnew::make_local_phase19_audit();
    compiler_only.evidence.hip_compiler_validation = true;
    compiler_only.evidence.source_fingerprint = fingerprint(1U);
    compiler_only.evidence.build_fingerprint = fingerprint(2U);
    compiler_only.evidence.runtime_fingerprint = fingerprint(3U);
    erase_blocker(
        compiler_only,
        bfnew::Phase19RecommendationBlocker::
            missing_hip_compiler_validation);
    expect(
        bfnew::validate_phase19_audit(compiler_only).ok(),
        "HIP compiler evidence advances independently of the query artifact");
  }
  {
    bfnew::Phase19AuditReport artifact_only =
        bfnew::make_local_phase19_audit();
    artifact_only.evidence.corpus_query_count =
        bfnew::minimum_logicnets_shootout_queries;
    artifact_only.evidence.representative_query_artifact = true;
    artifact_only.evidence.graph_fingerprint = fingerprint(1U);
    artifact_only.evidence.query_fingerprint = fingerprint(2U);
    artifact_only.evidence.workload_fingerprint = fingerprint(3U);
    erase_blocker(
        artifact_only,
        bfnew::Phase19RecommendationBlocker::
            missing_representative_query_artifact);
    expect(
        bfnew::validate_phase19_audit(artifact_only).ok(),
        "query-artifact evidence advances without GPU/device/runtime evidence");
  }
  {
    bfnew::Phase19AuditReport malformed = bfnew::make_local_phase19_audit();
    malformed.evidence = make_complete_snapshot();
    malformed.evidence.tail_attribution_fingerprint = {};
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_evidence_snapshot,
        "tail-attribution attestation requires normalized provenance");
  }
  {
    bfnew::Phase19AuditReport malformed = bfnew::make_local_phase19_audit();
    malformed.evidence.tail_attribution_fingerprint = fingerprint(1U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_evidence_snapshot,
        "tail-attribution provenance cannot appear without its attestation");
  }
  {
    bfnew::Phase19AuditReport tail_missing =
        bfnew::make_local_phase19_audit();
    tail_missing.evidence = make_complete_snapshot();
    tail_missing.evidence.tail_attribution_evidence = false;
    tail_missing.evidence.tail_attribution_fingerprint = {};
    tail_missing.blockers = {
        bfnew::Phase19RecommendationBlocker::
            missing_tail_attribution_evidence};
    bfnew::Phase19QuestionAnswer& answer =
        tail_missing.questions[static_cast<std::size_t>(
            bfnew::Phase19QuestionId::dominant_all_query_tail_class)];
    answer.state = bfnew::Phase19ConclusionState::measured;
    answer.evidence_fingerprint =
        bfnew::fingerprint_phase19_evidence_snapshot(tail_missing.evidence);
    answer.supporting_configuration_fingerprints = {1U};
    answer.finding = "unsupported dominant tail answer";
    expect_error(
        tail_missing,
        bfnew::Phase19AuditError::premature_measured_question,
        "dominant-tail answer requires explicit normalized tail attribution");
  }
}

void test_complete_report_and_fail_closed_edges() {
  const bfnew::Phase19AuditReport complete = make_complete_report();
  expect(
      bfnew::validate_phase19_audit(complete).ok(),
      "fully measured fixture passes the complete-evidence branch");
  std::array<std::size_t, 4U> complete_classifications{};
  for (const bfnew::Phase19FeatureAudit& feature : complete.features) {
    ++complete_classifications[static_cast<std::size_t>(
        feature.classification)];
  }
  expect(
      complete_classifications ==
          std::array<std::size_t, 4U>{36U, 20U, 0U, 0U},
      "complete evidence promotes every non-correctness feature coherently");

  {
    bfnew::Phase19AuditReport malformed = complete;
    const std::size_t index = static_cast<std::size_t>(
        bfnew::Phase19FeatureId::performance_claim_inventory);
    malformed.features[index].evidence_fingerprint = fingerprint(99U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_feature_record,
        "class-2 feature must use the exact snapshot identity");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    const std::size_t index = static_cast<std::size_t>(
        bfnew::Phase19FeatureId::performance_claim_inventory);
    malformed.features[index].classification =
        bfnew::Phase19FeatureClassification::designed_deferred;
    malformed.features[index].evidence_fingerprint = {};
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_feature_record,
        "complete evidence cannot retain a contradictory deferred feature");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.comparisons[0U].evidence_fingerprint = fingerprint(99U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "measured comparison must use the exact snapshot identity");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.profiler_metrics[0U].evidence_fingerprint = fingerprint(99U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_record,
        "measured profiler metric must use the exact snapshot identity");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.profiler_metrics[0U].configuration_fingerprint = 1U;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_record,
        "profiler metric must match the profiler-bottleneck configuration");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.profiler_metrics[static_cast<std::size_t>(
        bfnew::Phase19ProfilerMetricId::occupancy_percentage)]
        .metric_value = 100.01;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_profiler_metric_record,
        "profiler percentage above one hundred is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.questions[0U].evidence_fingerprint = fingerprint(99U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_question_record,
        "measured question must use the exact snapshot identity");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    const std::uint64_t value =
        malformed.questions[0U].supporting_configuration_fingerprints.front();
    malformed.questions[0U].supporting_configuration_fingerprints.push_back(
        value);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_question_record,
        "duplicate supporting evidence configuration is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.questions[0U].supporting_configuration_fingerprints = {2U, 1U};
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_question_record,
        "supporting configuration references must be strictly canonical");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.questions[0U].supporting_configuration_fingerprints = {0U};
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_question_record,
        "zero supporting configuration reference is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.comparisons[static_cast<std::size_t>(
        bfnew::Phase19ComparisonId::box_miss_rate)]
        .metric_value = 1.01;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "rate above one is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.comparisons[static_cast<std::size_t>(
        bfnew::Phase19ComparisonId::profiler_supported_bottleneck)]
        .selected_configuration_fingerprint = 1U;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "profiler bottleneck must match the best-engine configuration");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.comparisons[static_cast<std::size_t>(
        bfnew::Phase19ComparisonId::latency_p50)]
        .metric_value = 21.0;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_comparison_record,
        "nonmonotonic P50/P95/P99 is rejected");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.production_recommendation.reset();
    expect_error(
        malformed,
        bfnew::Phase19AuditError::missing_recommendation,
        "complete evidence requires an explicit recommendation");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.hybrid_experiment = bfnew::Phase19HybridExperimentDisposition::
        deferred_until_standalone_results;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_hybrid_disposition,
        "complete evidence must resolve the hybrid experiment decision");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.production_recommendation->evidence_fingerprint =
        fingerprint(99U);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation must use the exact snapshot identity");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    bfnew::Phase19ProductionConfiguration& configuration =
        malformed.production_recommendation->configuration;
    configuration.expansion.run_options.rounds_per_chunk = 3U;
    configuration.configuration_fingerprint =
        bfnew::fingerprint_phase19_configuration(configuration);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation rejects a noncatalog chunk size");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    bfnew::Phase19ProductionConfiguration& configuration =
        malformed.production_recommendation->configuration;
    configuration.expansion.run_options.block_size = 64U;
    configuration.configuration_fingerprint =
        bfnew::fingerprint_phase19_configuration(configuration);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation rejects a noncatalog block size");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    bfnew::Phase19ProductionConfiguration& configuration =
        malformed.production_recommendation->configuration;
    configuration.tile_width = 0U;
    configuration.configuration_fingerprint =
        bfnew::fingerprint_phase19_configuration(configuration);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation rejects a zero tile dimension");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    bfnew::Phase19ProductionConfiguration& configuration =
        malformed.production_recommendation->configuration;
    configuration.workspace_vertex_storage =
        bfnew::BatchVertexStorageStrategy::compact_union_tiles;
    configuration.configuration_fingerprint =
        bfnew::fingerprint_phase19_configuration(configuration);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation rejects an unimplemented compact GPU label layout");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    bfnew::Phase19ProductionConfiguration& configuration =
        malformed.production_recommendation->configuration;
    configuration.run_representation =
        bfnew::BatchRunRepresentation::retained_per_run_masks;
    configuration.configuration_fingerprint =
        bfnew::fingerprint_phase19_configuration(configuration);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation rejects a run representation unsupported by its engine");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    bfnew::Phase19ProductionConfiguration& configuration =
        malformed.production_recommendation->configuration;
    configuration.expansion.run_options.engine =
        bfnew::EngineKind::frontier_push;
    configuration.run_representation =
        bfnew::BatchRunRepresentation::retained_per_run_masks;
    configuration.frontier_queue_capacity = 0U;
    configuration.configuration_fingerprint =
        bfnew::fingerprint_phase19_configuration(configuration);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation rejects a zero frontier queue capacity");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    bfnew::Phase19ProductionConfiguration& configuration =
        malformed.production_recommendation->configuration;
    configuration.jacobi_load_strategy =
        static_cast<bfnew::Phase19LoadStrategy>(255U);
    configuration.configuration_fingerprint =
        bfnew::fingerprint_phase19_configuration(configuration);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation rejects an unknown load strategy");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    bfnew::Phase19ProductionConfiguration& configuration =
        malformed.production_recommendation->configuration;
    configuration.transfer_mode =
        static_cast<bfnew::Phase19TransferMode>(255U);
    configuration.configuration_fingerprint =
        bfnew::fingerprint_phase19_configuration(configuration);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation rejects an unknown transfer mode");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    bfnew::Phase19ProductionConfiguration& configuration =
        malformed.production_recommendation->configuration;
    configuration.reconstruction_mode =
        static_cast<bfnew::Phase19ReconstructionMode>(255U);
    configuration.configuration_fingerprint =
        bfnew::fingerprint_phase19_configuration(configuration);
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation rejects an unknown reconstruction mode");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    malformed.production_recommendation->configuration.configuration_fingerprint =
        1U;
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation rejects a stale configuration fingerprint");
  }
  {
    bfnew::Phase19AuditReport malformed = complete;
    std::uint64_t& selected =
        malformed.comparisons[static_cast<std::size_t>(
            bfnew::Phase19ComparisonId::best_engine)]
            .selected_configuration_fingerprint;
    selected = selected == 1U ? 2U : 1U;
    malformed.comparisons[static_cast<std::size_t>(
        bfnew::Phase19ComparisonId::profiler_supported_bottleneck)]
        .selected_configuration_fingerprint = selected;
    for (bfnew::Phase19ProfilerMetric& metric : malformed.profiler_metrics) {
      metric.configuration_fingerprint = selected;
    }
    expect_error(
        malformed,
        bfnew::Phase19AuditError::invalid_recommendation,
        "recommendation must match the attested best-engine selection");
  }
  {
    bfnew::Phase19AuditReport tied = complete;
    tied.evidence.unique_winner = false;
    tied.blockers = {bfnew::Phase19RecommendationBlocker::no_unique_winner};
    tied.features = bfnew::make_local_phase19_audit().features;
    expect_error(
        tied,
        bfnew::Phase19AuditError::premature_measured_comparison,
        "a tied campaign cannot retain measured selection rows");
  }

  const bfnew::Phase19ProductionConfiguration configuration =
      make_valid_production_configuration();
  bfnew::Phase19ProductionConfiguration changed = configuration;
  ++changed.tile_width;
  expect(
      bfnew::fingerprint_phase19_configuration(configuration) ==
              configuration.configuration_fingerprint &&
          bfnew::fingerprint_phase19_configuration(changed) !=
              configuration.configuration_fingerprint,
      "production configuration fingerprint is deterministic and sensitive");
  bfnew::Phase19EvidenceSnapshot evidence = make_complete_snapshot();
  const bfnew::Phase19Fingerprint first =
      bfnew::fingerprint_phase19_evidence_snapshot(evidence);
  ++evidence.tail_attribution_fingerprint.words[0];
  const bfnew::Phase19Fingerprint second =
      bfnew::fingerprint_phase19_evidence_snapshot(evidence);
  expect(
      !zero_fingerprint(first) && first != second,
      "evidence fingerprint binds the complete snapshot");
}

void test_serialization_and_corruption() {
  const bfnew::Phase19AuditReport local = bfnew::make_local_phase19_audit();
  const std::string serialized = bfnew::serialize_phase19_audit_tsv(local);
  expect(
      serialized == bfnew::serialize_phase19_audit_tsv(local),
      "local audit serialization is deterministic");
  expect(
      serialized.starts_with("schema\tbfnew.phase19-final-audit.v1\n"),
      "serialized audit carries the fixed schema identity");
  expect(
      count_rows(serialized, "blocker\t") == 15U &&
          count_rows(serialized, "feature\t") ==
              static_cast<std::size_t>(bfnew::Phase19FeatureId::count) &&
          count_rows(serialized, "comparison\t") == 27U &&
          count_rows(serialized, "profiler_metric\t") == 10U &&
          count_rows(serialized, "question\t") == 7U,
      "serialized audit preserves every canonical inventory row");
  const bfnew::Phase19AuditReport round_trip =
      bfnew::deserialize_phase19_audit_tsv(serialized);
  expect(round_trip == local, "serialized local audit round-trips exactly");
  expect(
      bfnew::serialize_phase19_audit_tsv(round_trip) == serialized,
      "audit round-trip is byte-identical");

  const bfnew::Phase19AuditReport complete = make_complete_report();
  const std::string complete_serialized =
      bfnew::serialize_phase19_audit_tsv(complete);
  const bfnew::Phase19AuditReport complete_round_trip =
      bfnew::deserialize_phase19_audit_tsv(complete_serialized);
  expect(
      complete_round_trip == complete &&
          bfnew::serialize_phase19_audit_tsv(complete_round_trip) ==
              complete_serialized,
      "measured recommendation row also round-trips byte-identically");

  expect_invalid_argument(
      [] { static_cast<void>(bfnew::deserialize_phase19_audit_tsv("")); },
      "empty TSV is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        malformed.pop_back();
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "TSV without final newline is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(
            malformed,
            "bfnew.phase19-final-audit.v1",
            "bfnew.phase19-final-audit.v2");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "unknown TSV schema is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(malformed, "snapshot\t0\t0\t0", "snapshot\t0\t0\t2");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "non-Boolean snapshot flag is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(malformed, "fingerprint\tgraph\t", "fingerprint\twrong\t");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "reordered or unknown fingerprint identity is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(malformed, "blocker\t0\n", "blocker\t255\n");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "unknown serialized blocker is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(malformed, "feature\t0\t0\t", "feature\t999\t0\t");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "unknown serialized feature ID is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(malformed, "feature\t0\t0\t", "feature\t0\t9\t");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "unknown serialized classification is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(
            malformed,
            "comparison\t0\t0\t0\t0\t0\t0\t0\t",
            "comparison\t0\t0\t0\t0\t-0\t0\t0\t");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "serialized signed negative zero is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(malformed, "feature\t1\t0\t", "feature\t0\t0\t");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "duplicate serialized feature ID is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(malformed, "feature\t0\t0\t", "feature\t999\t0\t");
        replace_once(malformed, "feature\t1\t0\t", "feature\t0\t0\t");
        replace_once(malformed, "feature\t999\t0\t", "feature\t1\t0\t");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "reordered serialized feature rows are rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(
            malformed, "comparison\t1\t0\t", "comparison\t0\t0\t");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "duplicate serialized comparison ID is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(
            malformed,
            "profiler_metric\t1\t0\t",
            "profiler_metric\t0\t0\t");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "duplicate serialized profiler metric ID is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(malformed, "question\t1\t0\t", "question\t0\t0\t");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "duplicate serialized question ID is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        replace_once(malformed, "feature\t0\t0\t", "unknown\t0\t0\t");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "unknown serialized row is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        malformed += "unexpected\trow\n";
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "trailing serialized row is rejected");
  expect_invalid_argument(
      [&] {
        std::string malformed = serialized;
        const std::size_t first_feature = malformed.find("feature\t0\t");
        malformed.insert(first_feature, "blocker\t0\n");
        static_cast<void>(bfnew::deserialize_phase19_audit_tsv(malformed));
      },
      "duplicate serialized blocker is rejected");
}

}  // namespace

int main() {
  test_local_report_inventory();
  test_local_validation_failures();
  test_incremental_evidence_and_tail_gate();
  test_complete_report_and_fail_closed_edges();
  test_serialization_and_corruption();
  if (failures != 0) {
    std::cerr << failures << " Phase 19 final-audit assertion(s) failed\n";
    return 1;
  }
  std::cout << "Phase 19 final audit tests passed\n";
  return 0;
}
