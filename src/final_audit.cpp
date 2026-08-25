#include "bfnew/final_audit.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

constexpr std::size_t feature_count =
    static_cast<std::size_t>(Phase19FeatureId::count);
constexpr std::size_t comparison_count =
    static_cast<std::size_t>(Phase19ComparisonId::count);
constexpr std::size_t profiler_metric_count =
    static_cast<std::size_t>(Phase19ProfilerMetricId::count);
constexpr std::size_t question_count =
    static_cast<std::size_t>(Phase19QuestionId::count);
constexpr std::size_t blocker_count = 15U;
constexpr std::uint64_t minimum_representative_query_count = 1000U;
constexpr std::string_view audit_schema = "bfnew.phase19-final-audit.v1";

[[nodiscard]] constexpr bool zero_fingerprint(
    const Phase19Fingerprint& value) noexcept {
  return value.words[0] == 0U && value.words[1] == 0U;
}

[[nodiscard]] bool valid_text(const std::string& value) noexcept {
  return !value.empty() && value.find_first_of("\t\r\n") == std::string::npos;
}

template <class Enum>
[[nodiscard]] constexpr bool enum_below(
    const Enum value,
    const Enum limit) noexcept {
  using Raw = std::underlying_type_t<Enum>;
  return static_cast<Raw>(value) < static_cast<Raw>(limit);
}

[[nodiscard]] constexpr bool valid_classification(
    const Phase19FeatureClassification value) noexcept {
  return static_cast<std::uint8_t>(value) <=
         static_cast<std::uint8_t>(
             Phase19FeatureClassification::designed_deferred);
}

[[nodiscard]] constexpr bool valid_conclusion_state(
    const Phase19ConclusionState value) noexcept {
  return static_cast<std::uint8_t>(value) <=
         static_cast<std::uint8_t>(Phase19ConclusionState::measured);
}

[[nodiscard]] constexpr bool valid_metric_unit(
    const Phase19MetricUnit value) noexcept {
  return static_cast<std::uint8_t>(value) <=
         static_cast<std::uint8_t>(Phase19MetricUnit::count);
}

[[nodiscard]] constexpr bool valid_hybrid_disposition(
    const Phase19HybridExperimentDisposition value) noexcept {
  return static_cast<std::uint8_t>(value) <=
         static_cast<std::uint8_t>(
             Phase19HybridExperimentDisposition::recommended_as_future_experiment);
}

[[nodiscard]] constexpr Phase19MetricUnit expected_metric_unit(
    const Phase19ComparisonId id) noexcept {
  switch (id) {
    case Phase19ComparisonId::latency_p50:
    case Phase19ComparisonId::latency_p95:
    case Phase19ComparisonId::latency_p99:
    case Phase19ComparisonId::warm_all_query_time:
    case Phase19ComparisonId::cold_pipeline_time:
      return Phase19MetricUnit::nanoseconds;
    case Phase19ComparisonId::all_query_throughput:
      return Phase19MetricUnit::queries_per_second;
    case Phase19ComparisonId::box_miss_rate:
    case Phase19ComparisonId::expansion_rate:
    case Phase19ComparisonId::fallback_rate:
    case Phase19ComparisonId::path_quality_summary:
      return Phase19MetricUnit::ratio;
    case Phase19ComparisonId::memory_footprint_summary:
      return Phase19MetricUnit::bytes;
    case Phase19ComparisonId::synchronization_count:
    case Phase19ComparisonId::copy_count:
      return Phase19MetricUnit::count;
    case Phase19ComparisonId::best_engine:
    case Phase19ComparisonId::best_control_jacobi:
    case Phase19ComparisonId::best_control_dense:
    case Phase19ComparisonId::best_control_frontier:
    case Phase19ComparisonId::best_ordinary_k_jacobi:
    case Phase19ComparisonId::best_ordinary_k_dense:
    case Phase19ComparisonId::best_ordinary_k_frontier:
    case Phase19ComparisonId::best_persistent_grid_jacobi:
    case Phase19ComparisonId::best_persistent_grid_dense:
    case Phase19ComparisonId::best_persistent_grid_frontier:
    case Phase19ComparisonId::best_batching_width:
    case Phase19ComparisonId::best_overlap_planner_thresholds:
    case Phase19ComparisonId::best_expansion_schedule:
    case Phase19ComparisonId::profiler_supported_bottleneck:
    case Phase19ComparisonId::count:
      return Phase19MetricUnit::none;
  }
  return Phase19MetricUnit::none;
}

[[nodiscard]] constexpr Phase19MetricUnit expected_profiler_metric_unit(
    const Phase19ProfilerMetricId id) noexcept {
  switch (id) {
    case Phase19ProfilerMetricId::gpu_active_time:
      return Phase19MetricUnit::nanoseconds;
    case Phase19ProfilerMetricId::l2_hit_percentage:
    case Phase19ProfilerMetricId::occupancy_percentage:
    case Phase19ProfilerMetricId::memory_unit_busy_percentage:
      return Phase19MetricUnit::percentage;
    case Phase19ProfilerMetricId::l2_read_bytes:
    case Phase19ProfilerMetricId::l2_write_bytes:
      return Phase19MetricUnit::bytes;
    case Phase19ProfilerMetricId::wave_count:
    case Phase19ProfilerMetricId::vector_instruction_count:
    case Phase19ProfilerMetricId::scalar_instruction_count:
    case Phase19ProfilerMetricId::memory_instruction_count:
      return Phase19MetricUnit::count;
    case Phase19ProfilerMetricId::count:
      return Phase19MetricUnit::none;
  }
  return Phase19MetricUnit::none;
}

[[nodiscard]] constexpr bool blocker_is_present(
    const Phase19EvidenceSnapshot& evidence,
    const Phase19RecommendationBlocker blocker) noexcept {
  switch (blocker) {
    case Phase19RecommendationBlocker::missing_representative_query_artifact:
      return !evidence.representative_query_artifact;
    case Phase19RecommendationBlocker::missing_hip_compiler_validation:
      return !evidence.hip_compiler_validation;
    case Phase19RecommendationBlocker::missing_gpu_device_validation:
      return !evidence.gpu_device_validation;
    case Phase19RecommendationBlocker::missing_comparable_cpu_baseline:
      return !evidence.comparable_cpu_baseline;
    case Phase19RecommendationBlocker::missing_representative_correctness:
      return !evidence.representative_correctness;
    case Phase19RecommendationBlocker::missing_representative_timing:
      return !evidence.representative_timing;
    case Phase19RecommendationBlocker::missing_dependency_trace:
      return !evidence.dependency_trace;
    case Phase19RecommendationBlocker::missing_compatible_pmc:
      return !evidence.compatible_pmc;
    case Phase19RecommendationBlocker::incomplete_candidate_matrix:
      return !evidence.complete_candidate_matrix;
    case Phase19RecommendationBlocker::missing_all_query_evidence:
      return !evidence.all_query_evidence;
    case Phase19RecommendationBlocker::missing_tail_attribution_evidence:
      return !evidence.tail_attribution_evidence;
    case Phase19RecommendationBlocker::no_unique_winner:
      return !evidence.unique_winner;
    case Phase19RecommendationBlocker::missing_memory_evidence:
      return !evidence.memory_evidence;
    case Phase19RecommendationBlocker::missing_synchronization_copy_evidence:
      return !evidence.synchronization_copy_evidence;
    case Phase19RecommendationBlocker::missing_path_quality_evidence:
      return !evidence.path_quality_evidence;
  }
  return true;
}

[[nodiscard]] constexpr Phase19RecommendationBlocker blocker_at(
    const std::size_t index) noexcept {
  return static_cast<Phase19RecommendationBlocker>(index);
}

[[nodiscard]] bool complete_representative_evidence(
    const Phase19EvidenceSnapshot& evidence) noexcept {
  return evidence.representative_query_artifact &&
         evidence.hip_compiler_validation &&
         evidence.gpu_device_validation &&
         evidence.representative_correctness &&
         evidence.representative_timing && evidence.dependency_trace &&
         evidence.compatible_pmc && evidence.complete_candidate_matrix &&
         evidence.all_query_evidence && evidence.tail_attribution_evidence &&
         evidence.unique_winner &&
         evidence.comparable_cpu_baseline && evidence.memory_evidence &&
         evidence.synchronization_copy_evidence &&
         evidence.path_quality_evidence;
}

[[nodiscard]] bool valid_snapshot(
    const Phase19EvidenceSnapshot& evidence) noexcept {
  if (evidence.representative_query_artifact) {
    if (evidence.corpus_query_count < minimum_representative_query_count ||
        evidence.evaluated_query_count > evidence.corpus_query_count ||
        zero_fingerprint(evidence.graph_fingerprint) ||
        zero_fingerprint(evidence.query_fingerprint) ||
        zero_fingerprint(evidence.workload_fingerprint)) {
      return false;
    }
  } else if (evidence.corpus_query_count != 0U ||
             evidence.evaluated_query_count != 0U ||
             !zero_fingerprint(evidence.graph_fingerprint) ||
             !zero_fingerprint(evidence.query_fingerprint) ||
             !zero_fingerprint(evidence.workload_fingerprint)) {
    return false;
  }
  if (evidence.hip_compiler_validation) {
    if (zero_fingerprint(evidence.source_fingerprint) ||
        zero_fingerprint(evidence.build_fingerprint) ||
        zero_fingerprint(evidence.runtime_fingerprint)) {
      return false;
    }
  } else if (!zero_fingerprint(evidence.source_fingerprint) ||
             !zero_fingerprint(evidence.build_fingerprint) ||
             !zero_fingerprint(evidence.runtime_fingerprint)) {
    return false;
  }
  if (evidence.gpu_device_validation) {
    if (!evidence.hip_compiler_validation ||
        zero_fingerprint(evidence.device_fingerprint)) {
      return false;
    }
  } else if (!zero_fingerprint(evidence.device_fingerprint)) {
    return false;
  }
  if (evidence.representative_correctness &&
      (!evidence.representative_query_artifact ||
       !evidence.gpu_device_validation ||
       evidence.evaluated_query_count == 0U ||
       zero_fingerprint(evidence.protocol_fingerprint))) {
    return false;
  }
  if (evidence.comparable_cpu_baseline &&
      (!evidence.representative_query_artifact ||
       evidence.evaluated_query_count == 0U ||
       zero_fingerprint(evidence.protocol_fingerprint))) {
    return false;
  }
  if (!evidence.representative_correctness &&
      !evidence.comparable_cpu_baseline &&
      !zero_fingerprint(evidence.protocol_fingerprint)) {
    return false;
  }
  if (evidence.representative_timing &&
      !evidence.representative_correctness) {
    return false;
  }
  if ((evidence.dependency_trace || evidence.compatible_pmc) &&
      (!evidence.representative_correctness ||
       zero_fingerprint(evidence.profiler_fingerprint))) {
    return false;
  }
  if (!evidence.dependency_trace && !evidence.compatible_pmc &&
      !zero_fingerprint(evidence.profiler_fingerprint)) {
    return false;
  }
  if (evidence.complete_candidate_matrix) {
    if (!evidence.representative_query_artifact ||
        zero_fingerprint(evidence.candidate_catalog_fingerprint)) {
      return false;
    }
  } else if (!zero_fingerprint(evidence.candidate_catalog_fingerprint)) {
    return false;
  }
  if (evidence.unique_winner &&
      (!evidence.complete_candidate_matrix ||
       !evidence.representative_timing)) {
    return false;
  }
  if (evidence.all_query_evidence &&
      (!evidence.representative_correctness ||
       !evidence.representative_timing ||
       !evidence.complete_candidate_matrix ||
       evidence.evaluated_query_count != evidence.corpus_query_count)) {
    return false;
  }
  if (evidence.tail_attribution_evidence) {
    if (!evidence.all_query_evidence ||
        zero_fingerprint(evidence.tail_attribution_fingerprint)) {
      return false;
    }
  } else if (!zero_fingerprint(evidence.tail_attribution_fingerprint)) {
    return false;
  }
  if ((evidence.memory_evidence || evidence.synchronization_copy_evidence ||
       evidence.path_quality_evidence) &&
      !evidence.all_query_evidence) {
    return false;
  }
  return true;
}

[[nodiscard]] constexpr Phase19FeatureClassification local_classification(
    const Phase19FeatureId id) noexcept {
  switch (id) {
    case Phase19FeatureId::hip_standalone_jacobi_pull:
    case Phase19FeatureId::hip_standalone_dense_chaotic_push:
    case Phase19FeatureId::hip_standalone_frontier_push:
    case Phase19FeatureId::hip_batched_jacobi_pull:
    case Phase19FeatureId::hip_batched_dense_chaotic_push:
    case Phase19FeatureId::hip_batched_frontier_push:
    case Phase19FeatureId::provisional_workspace_strategy:
    case Phase19FeatureId::hip_batched_expansion_adapter:
    case Phase19FeatureId::hip_compact_reconstruction:
    case Phase19FeatureId::hip_fpgaif_campaign_report_surface:
    case Phase19FeatureId::hip_algorithm_counter_instrumentation:
    case Phase19FeatureId::dependency_trace_collection:
      return Phase19FeatureClassification::implemented_not_representative;
    case Phase19FeatureId::profiler_l2_group:
    case Phase19FeatureId::profiler_occupancy_memory_group:
    case Phase19FeatureId::profiler_instruction_wave_group:
    case Phase19FeatureId::representative_logicnets_campaign:
    case Phase19FeatureId::representative_profiler_campaign:
    case Phase19FeatureId::performance_claim_inventory:
    case Phase19FeatureId::production_default_selection:
    case Phase19FeatureId::hybrid_experiment_decision:
      return Phase19FeatureClassification::designed_deferred;
    case Phase19FeatureId::portable_standalone_jacobi_pull:
    case Phase19FeatureId::portable_standalone_dense_chaotic_push:
    case Phase19FeatureId::portable_standalone_frontier_push:
    case Phase19FeatureId::portable_standalone_jacobi_persistent_control:
    case Phase19FeatureId::portable_standalone_jacobi_chunked_control:
    case Phase19FeatureId::portable_standalone_jacobi_per_round_control:
    case Phase19FeatureId::portable_standalone_dense_persistent_control:
    case Phase19FeatureId::portable_standalone_dense_chunked_control:
    case Phase19FeatureId::portable_standalone_dense_per_round_control:
    case Phase19FeatureId::portable_standalone_frontier_persistent_control:
    case Phase19FeatureId::portable_standalone_frontier_chunked_control:
    case Phase19FeatureId::portable_standalone_frontier_per_round_control:
    case Phase19FeatureId::portable_batched_jacobi_pull:
    case Phase19FeatureId::portable_batched_dense_chaotic_push:
    case Phase19FeatureId::portable_batched_frontier_push:
    case Phase19FeatureId::portable_batched_jacobi_persistent_control:
    case Phase19FeatureId::portable_batched_jacobi_chunked_control:
    case Phase19FeatureId::portable_batched_jacobi_per_round_control:
    case Phase19FeatureId::portable_batched_dense_persistent_control:
    case Phase19FeatureId::portable_batched_dense_chunked_control:
    case Phase19FeatureId::portable_batched_dense_per_round_control:
    case Phase19FeatureId::portable_batched_frontier_persistent_control:
    case Phase19FeatureId::portable_batched_frontier_chunked_control:
    case Phase19FeatureId::portable_batched_frontier_per_round_control:
    case Phase19FeatureId::batching_width_1:
    case Phase19FeatureId::batching_width_8:
    case Phase19FeatureId::batching_width_16:
    case Phase19FeatureId::batching_width_32:
    case Phase19FeatureId::overlap_greedy_planner:
    case Phase19FeatureId::portable_batched_expansion:
    case Phase19FeatureId::portable_compact_reconstruction:
    case Phase19FeatureId::no_congestion_pipeline:
    case Phase19FeatureId::compact_transfer_accounting:
    case Phase19FeatureId::portable_algorithm_counter_instrumentation:
    case Phase19FeatureId::correctness_evidence_inventory:
    case Phase19FeatureId::phase12_shootout_evidence_contract:
      return Phase19FeatureClassification::implemented_correctness_tested;
    case Phase19FeatureId::count:
      return Phase19FeatureClassification::designed_deferred;
  }
  return Phase19FeatureClassification::designed_deferred;
}

[[nodiscard]] const char* local_feature_finding(
    const Phase19FeatureId id) noexcept {
  switch (id) {
    case Phase19FeatureId::hip_standalone_jacobi_pull:
    case Phase19FeatureId::hip_standalone_dense_chaotic_push:
    case Phase19FeatureId::hip_standalone_frontier_push:
    case Phase19FeatureId::hip_batched_jacobi_pull:
    case Phase19FeatureId::hip_batched_dense_chaotic_push:
    case Phase19FeatureId::hip_batched_frontier_push:
    case Phase19FeatureId::hip_batched_expansion_adapter:
    case Phase19FeatureId::hip_compact_reconstruction:
    case Phase19FeatureId::hip_fpgaif_campaign_report_surface:
    case Phase19FeatureId::hip_algorithm_counter_instrumentation:
      return "implemented HIP source surface; no HIP compiler or GPU has validated it";
    case Phase19FeatureId::provisional_workspace_strategy:
      return "implemented provisional full-vertex retained-mask strategy; representative production choice is unavailable";
    case Phase19FeatureId::dependency_trace_collection:
      return "implemented evidence path; no representative dependency trace has been collected";
    case Phase19FeatureId::profiler_l2_group:
    case Phase19FeatureId::profiler_occupancy_memory_group:
    case Phase19FeatureId::profiler_instruction_wave_group:
      return "profiler group is designed and provenance-typed; representative compatible counters are deferred";
    case Phase19FeatureId::performance_claim_inventory:
      return "no representative GPU performance claim is currently supported";
    case Phase19FeatureId::representative_logicnets_campaign:
      return "representative logicnets query artifact and campaign are deferred";
    case Phase19FeatureId::representative_profiler_campaign:
      return "representative compatible profiler campaign is deferred";
    case Phase19FeatureId::production_default_selection:
      return "production selection is deferred until complete comparable evidence has a unique winner";
    case Phase19FeatureId::hybrid_experiment_decision:
      return "hybrid experiment decision is deferred until standalone results are clear";
    case Phase19FeatureId::phase12_shootout_evidence_contract:
      return "implemented and host-tested Phase 12 shootout evidence contract; no representative GPU campaign exists";
    default:
      return "implemented and bounded correctness-tested; representative GPU performance evidence is absent";
  }
}

[[nodiscard]] const char* local_question_finding(
    const Phase19QuestionId id) noexcept {
  switch (id) {
    case Phase19QuestionId::jacobi_gpu_benefit:
      return "insufficient evidence: no comparable representative CPU and GPU baseline exists";
    case Phase19QuestionId::atomic_elimination_vs_dense_scans:
      return "insufficient evidence: dense scans, atomics, timing, and profiler counters were not measured comparably";
    case Phase19QuestionId::chaotic_round_reduction_vs_atomics:
      return "insufficient evidence: representative round reduction and atomic cost were not measured";
    case Phase19QuestionId::frontier_work_reduction_vs_queue_overhead:
      return "insufficient evidence: representative frontier work and queue overhead were not measured";
    case Phase19QuestionId::cooperative_persistence_vs_chunking:
      return "insufficient evidence: no complete warmed persistent-versus-chunked matrix exists";
    case Phase19QuestionId::overlap_batching_for_utilization:
      return "insufficient evidence: widths and planner thresholds were not measured on one representative campaign";
    case Phase19QuestionId::dominant_all_query_tail_class:
      return "insufficient evidence: no representative all-query latency attribution exists";
    case Phase19QuestionId::count:
      return "invalid Phase 19 question";
  }
  return "invalid Phase 19 question";
}

void fingerprint_u64(std::uint64_t& hash, const std::uint64_t value) noexcept {
  constexpr std::uint64_t prime = 1099511628211ULL;
  for (std::uint32_t byte = 0U; byte < 8U; ++byte) {
    hash ^= (value >> (byte * 8U)) & 0xffU;
    hash *= prime;
  }
}

[[nodiscard]] bool valid_workspace_strategy(
    const BatchVertexStorageStrategy value) noexcept {
  return value == BatchVertexStorageStrategy::full_graph_vertex_major ||
         value == BatchVertexStorageStrategy::compact_union_tiles;
}

[[nodiscard]] bool valid_run_representation(
    const BatchRunRepresentation value) noexcept {
  return value == BatchRunRepresentation::retained_per_run_masks ||
         value == BatchRunRepresentation::compact_nonzero_descriptors ||
         value == BatchRunRepresentation::device_materialized_run_masks;
}

[[nodiscard]] bool production_run_representation_matches_engine(
    const Phase19ProductionConfiguration& configuration) noexcept {
  switch (configuration.expansion.run_options.engine) {
    case EngineKind::jacobi_pull:
      return configuration.run_representation ==
             BatchRunRepresentation::device_materialized_run_masks;
    case EngineKind::dense_chaotic_push:
    case EngineKind::frontier_push:
      return configuration.run_representation ==
             BatchRunRepresentation::retained_per_run_masks;
  }
  return false;
}

[[nodiscard]] bool valid_load_strategy(
    const Phase19LoadStrategy value) noexcept {
  return value == Phase19LoadStrategy::compiler_uniform ||
         value == Phase19LoadStrategy::explicit_wave_broadcast;
}

[[nodiscard]] constexpr bool allowed_production_block_size(
    const std::uint32_t value) noexcept {
  return value == 128U || value == 256U || value == 512U;
}

[[nodiscard]] constexpr bool allowed_production_chunk_size(
    const std::uint32_t value) noexcept {
  return value == 2U || value == 4U || value == 8U || value == 16U ||
         value == 32U;
}

[[nodiscard]] constexpr bool valid_production_run_shape(
    const GpuRunOptions& options) noexcept {
  if (!allowed_production_block_size(options.block_size)) {
    return false;
  }
  switch (options.control_mode) {
    case ControlMode::persistent_cooperative:
      return options.rounds_per_chunk == 1U;
    case ControlMode::chunked_host_poll:
      return allowed_production_chunk_size(options.rounds_per_chunk) &&
             options.grid_policy == GridPolicy::occupancy_derived &&
             options.blocks_per_wgp == 0U;
    case ControlMode::per_round_host_poll:
      return options.rounds_per_chunk == 1U &&
             options.grid_policy == GridPolicy::occupancy_derived &&
             options.blocks_per_wgp == 0U;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_frontier_queue_capacity(
    const Phase19ProductionConfiguration& configuration) noexcept {
  if (configuration.expansion.run_options.engine == EngineKind::frontier_push) {
    return configuration.frontier_queue_capacity != 0U;
  }
  return configuration.frontier_queue_capacity == 0U;
}

[[nodiscard]] bool valid_production_configuration(
    const Phase19ProductionConfiguration& configuration) noexcept {
  return configuration.configuration_fingerprint != 0U &&
         configuration.configuration_fingerprint ==
             fingerprint_phase19_configuration(configuration) &&
         validate_batched_expansion_options(configuration.expansion) ==
             BatchedExpansionOptionsError::none &&
         valid_production_run_shape(configuration.expansion.run_options) &&
         configuration.expansion.run_options.instrumentation ==
             InstrumentationLevel::none &&
         configuration.expansion.enable_compact_paths == 1U &&
         valid_workspace_strategy(configuration.workspace_vertex_storage) &&
         configuration.workspace_vertex_storage ==
             BatchVertexStorageStrategy::full_graph_vertex_major &&
         valid_run_representation(configuration.run_representation) &&
         production_run_representation_matches_engine(configuration) &&
         configuration.tile_width != 0U && configuration.tile_height != 0U &&
         valid_frontier_queue_capacity(configuration) &&
         valid_load_strategy(configuration.jacobi_load_strategy) &&
         valid_load_strategy(configuration.dense_load_strategy) &&
         configuration.transfer_mode == Phase19TransferMode::compact_paths &&
         configuration.reconstruction_mode ==
             Phase19ReconstructionMode::incoming_csc_backtracking;
}

[[nodiscard]] std::vector<Phase19RecommendationBlocker> local_blockers(
    const Phase19EvidenceSnapshot& evidence) {
  std::vector<Phase19RecommendationBlocker> result;
  result.reserve(blocker_count);
  for (std::size_t index = 0U; index < blocker_count; ++index) {
    const Phase19RecommendationBlocker blocker = blocker_at(index);
    if (blocker_is_present(evidence, blocker)) {
      result.push_back(blocker);
    }
  }
  return result;
}

[[nodiscard]] std::vector<std::string_view> split_fields(
    const std::string_view line) {
  std::vector<std::string_view> result;
  std::size_t begin = 0U;
  while (true) {
    const std::size_t end = line.find('\t', begin);
    result.push_back(line.substr(begin, end - begin));
    if (end == std::string_view::npos) {
      return result;
    }
    begin = end + 1U;
  }
}

[[nodiscard]] std::vector<std::string_view> split_lines(
    const std::string_view text) {
  if (text.empty() || !text.ends_with('\n')) {
    throw std::invalid_argument{"Phase 19 TSV must end with a newline"};
  }
  std::vector<std::string_view> lines;
  std::size_t begin = 0U;
  while (begin < text.size()) {
    const std::size_t end = text.find('\n', begin);
    const std::string_view line = text.substr(begin, end - begin);
    if (line.empty() || line.ends_with('\r')) {
      throw std::invalid_argument{"Phase 19 TSV has an empty or CR-terminated row"};
    }
    lines.push_back(line);
    begin = end + 1U;
  }
  return lines;
}

template <class Integer>
[[nodiscard]] Integer parse_unsigned(
    const std::string_view text,
    const char* const name) {
  static_assert(std::is_unsigned_v<Integer>);
  Integer value{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size()) {
    throw std::invalid_argument{std::string{"invalid Phase 19 "} + name};
  }
  return value;
}

[[nodiscard]] bool parse_bool(
    const std::string_view text,
    const char* const name) {
  const std::uint32_t value = parse_unsigned<std::uint32_t>(text, name);
  if (value > 1U) {
    throw std::invalid_argument{std::string{"invalid Phase 19 "} + name};
  }
  return value != 0U;
}

[[nodiscard]] double parse_double(
    const std::string_view text,
    const char* const name) {
  std::istringstream input{std::string{text}};
  input.imbue(std::locale::classic());
  double value{};
  input >> std::noskipws >> value;
  if (text.empty() || !input || !input.eof() || !std::isfinite(value)) {
    throw std::invalid_argument{std::string{"invalid Phase 19 "} + name};
  }
  return value;
}

template <class Enum>
[[nodiscard]] Enum parse_enum(
    const std::string_view text,
    const char* const name) {
  using Raw = std::underlying_type_t<Enum>;
  return static_cast<Enum>(parse_unsigned<Raw>(text, name));
}

void require_fields(
    const std::vector<std::string_view>& fields,
    const std::size_t count,
    const std::string_view row) {
  if (fields.size() != count || fields.front() != row) {
    throw std::invalid_argument{"Phase 19 TSV row is not canonical"};
  }
}

[[nodiscard]] std::vector<std::uint64_t> parse_configuration_list(
    const std::string_view text) {
  std::vector<std::uint64_t> result;
  if (text.empty()) {
    return result;
  }
  std::size_t begin = 0U;
  while (true) {
    const std::size_t end = text.find(',', begin);
    result.push_back(parse_unsigned<std::uint64_t>(
        text.substr(begin, end - begin), "supporting configuration"));
    if (end == std::string_view::npos) {
      return result;
    }
    begin = end + 1U;
  }
}

void write_fingerprint(
    std::ostream& output,
    const std::string_view name,
    const Phase19Fingerprint& fingerprint) {
  output << "fingerprint\t" << name << '\t' << fingerprint.words[0] << '\t'
         << fingerprint.words[1] << '\n';
}

[[nodiscard]] Phase19Fingerprint parse_fingerprint_row(
    const std::string_view line,
    const std::string_view expected_name) {
  const std::vector<std::string_view> fields = split_fields(line);
  require_fields(fields, 4U, "fingerprint");
  if (fields[1] != expected_name) {
    throw std::invalid_argument{"Phase 19 fingerprint order is not canonical"};
  }
  return Phase19Fingerprint{
      {parse_unsigned<std::uint64_t>(fields[2], "fingerprint word"),
       parse_unsigned<std::uint64_t>(fields[3], "fingerprint word")}};
}

}  // namespace

std::uint64_t fingerprint_phase19_configuration(
    const Phase19ProductionConfiguration& configuration) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  fingerprint_u64(hash, 0x62666e6577313901ULL);
  const BatchedExpansionOptions& options = configuration.expansion;
  fingerprint_u64(hash, static_cast<std::uint32_t>(options.run_options.engine));
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(options.run_options.control_mode));
  fingerprint_u64(hash, options.run_options.rounds_per_chunk);
  fingerprint_u64(hash, options.run_options.block_size);
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(options.run_options.grid_policy));
  fingerprint_u64(hash, options.run_options.blocks_per_wgp);
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(options.run_options.instrumentation));
  fingerprint_u64(hash, options.run_options.maximum_rounds);
  fingerprint_u64(hash, options.run_options.enable_per_lane_convergence);
  fingerprint_u64(hash, options.planner_policy.lane_width);
  fingerprint_u64(hash, options.planner_policy.minimum_jaccard_numerator);
  fingerprint_u64(hash, options.planner_policy.minimum_jaccard_denominator);
  fingerprint_u64(
      hash, options.planner_policy.maximum_union_inflation_numerator);
  fingerprint_u64(
      hash, options.planner_policy.maximum_union_inflation_denominator);
  fingerprint_u64(hash, options.execution_configuration_fingerprint);
  fingerprint_u64(hash, static_cast<std::uint32_t>(options.schedule.kind));
  fingerprint_u64(hash, options.schedule.fixed_ring_size);
  fingerprint_u64(hash, options.schedule.hybrid_small_expansion_count);
  fingerprint_u64(hash, options.maximum_expansions);
  fingerprint_u64(hash, static_cast<std::uint32_t>(options.terminal_policy));
  fingerprint_u64(hash, options.enable_compact_paths);
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(configuration.workspace_vertex_storage));
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(configuration.run_representation));
  fingerprint_u64(hash, configuration.tile_width);
  fingerprint_u64(hash, configuration.tile_height);
  fingerprint_u64(hash, configuration.frontier_queue_capacity);
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(configuration.jacobi_load_strategy));
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(configuration.dense_load_strategy));
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(configuration.transfer_mode));
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(configuration.reconstruction_mode));
  return hash == 0U ? 1U : hash;
}

Phase19Fingerprint fingerprint_phase19_evidence_snapshot(
    const Phase19EvidenceSnapshot& evidence) noexcept {
  std::uint64_t first = 1469598103934665603ULL;
  std::uint64_t second = 7809847782465536322ULL;
  const auto append = [&](const std::uint64_t value) {
    fingerprint_u64(first, value);
    fingerprint_u64(second, value);
  };
  append(0x62666e6577313902ULL);
  append(phase19_audit_schema_version);
  append(evidence.corpus_query_count);
  append(evidence.evaluated_query_count);
  append(static_cast<std::uint64_t>(evidence.representative_query_artifact));
  append(static_cast<std::uint64_t>(evidence.hip_compiler_validation));
  append(static_cast<std::uint64_t>(evidence.gpu_device_validation));
  append(static_cast<std::uint64_t>(evidence.representative_correctness));
  append(static_cast<std::uint64_t>(evidence.representative_timing));
  append(static_cast<std::uint64_t>(evidence.dependency_trace));
  append(static_cast<std::uint64_t>(evidence.compatible_pmc));
  append(static_cast<std::uint64_t>(evidence.complete_candidate_matrix));
  append(static_cast<std::uint64_t>(evidence.all_query_evidence));
  append(static_cast<std::uint64_t>(evidence.tail_attribution_evidence));
  append(static_cast<std::uint64_t>(evidence.unique_winner));
  append(static_cast<std::uint64_t>(evidence.comparable_cpu_baseline));
  append(static_cast<std::uint64_t>(evidence.memory_evidence));
  append(static_cast<std::uint64_t>(
      evidence.synchronization_copy_evidence));
  append(static_cast<std::uint64_t>(evidence.path_quality_evidence));
  const auto append_fingerprint = [&](const Phase19Fingerprint& fingerprint) {
    append(fingerprint.words[0]);
    append(fingerprint.words[1]);
  };
  append_fingerprint(evidence.source_fingerprint);
  append_fingerprint(evidence.build_fingerprint);
  append_fingerprint(evidence.graph_fingerprint);
  append_fingerprint(evidence.query_fingerprint);
  append_fingerprint(evidence.workload_fingerprint);
  append_fingerprint(evidence.device_fingerprint);
  append_fingerprint(evidence.runtime_fingerprint);
  append_fingerprint(evidence.protocol_fingerprint);
  append_fingerprint(evidence.tail_attribution_fingerprint);
  append_fingerprint(evidence.candidate_catalog_fingerprint);
  append_fingerprint(evidence.profiler_fingerprint);
  if (first == 0U) {
    first = 1U;
  }
  if (second == 0U) {
    second = 1U;
  }
  return Phase19Fingerprint{{first, second}};
}

Phase19AuditReport make_local_phase19_audit() {
  Phase19AuditReport report;
  report.blockers = local_blockers(report.evidence);
  report.features.reserve(feature_count);
  for (std::size_t index = 0U; index < feature_count; ++index) {
    const Phase19FeatureId id = static_cast<Phase19FeatureId>(index);
    report.features.push_back(Phase19FeatureAudit{
        id, local_classification(id), {}, local_feature_finding(id)});
  }
  report.comparisons.reserve(comparison_count);
  for (std::size_t index = 0U; index < comparison_count; ++index) {
    const Phase19ComparisonId id = static_cast<Phase19ComparisonId>(index);
    report.comparisons.push_back(Phase19Comparison{
        id,
        Phase19ConclusionState::insufficient_evidence,
        0U,
        expected_metric_unit(id),
        0.0,
        {},
        "insufficient evidence: representative comparable GPU campaign is absent"});
  }
  report.profiler_metrics.reserve(profiler_metric_count);
  for (std::size_t index = 0U; index < profiler_metric_count; ++index) {
    const Phase19ProfilerMetricId id =
        static_cast<Phase19ProfilerMetricId>(index);
    report.profiler_metrics.push_back(Phase19ProfilerMetric{
        id,
        Phase19ConclusionState::insufficient_evidence,
        0U,
        expected_profiler_metric_unit(id),
        0.0,
        {},
        "insufficient evidence: no accepted representative trace/PMC sample"});
  }
  report.questions.reserve(question_count);
  for (std::size_t index = 0U; index < question_count; ++index) {
    const Phase19QuestionId id = static_cast<Phase19QuestionId>(index);
    report.questions.push_back(Phase19QuestionAnswer{
        id,
        Phase19ConclusionState::insufficient_evidence,
        {},
        {},
        local_question_finding(id)});
  }
  report.hybrid_finding =
      "future hybrid experimentation is deferred until standalone representative results are clear";
  const Phase19AuditValidationResult validation =
      validate_phase19_audit(report);
  if (!validation.ok()) {
    throw std::logic_error{"internal local Phase 19 audit is invalid"};
  }
  return report;
}

Phase19AuditValidationResult validate_phase19_audit(
    const Phase19AuditReport& report) noexcept {
  const auto error = [](const Phase19AuditError code,
                        const std::size_t position =
                            Phase19AuditValidationResult::no_position) {
    return Phase19AuditValidationResult{code, position};
  };
  if (report.schema_version != phase19_audit_schema_version) {
    return error(Phase19AuditError::invalid_schema);
  }
  if (!valid_snapshot(report.evidence)) {
    return error(Phase19AuditError::invalid_evidence_snapshot);
  }
  std::size_t blocker_position = 0U;
  for (std::size_t index = 0U; index < blocker_count; ++index) {
    const Phase19RecommendationBlocker blocker = blocker_at(index);
    if (!blocker_is_present(report.evidence, blocker)) {
      continue;
    }
    if (blocker_position >= report.blockers.size() ||
        report.blockers[blocker_position] != blocker) {
      return error(Phase19AuditError::blocker_mismatch, blocker_position);
    }
    ++blocker_position;
  }
  if (blocker_position != report.blockers.size()) {
    return error(Phase19AuditError::blocker_mismatch, blocker_position);
  }

  const bool complete = complete_representative_evidence(report.evidence);
  const Phase19Fingerprint evidence_identity =
      fingerprint_phase19_evidence_snapshot(report.evidence);
  if (report.features.size() != feature_count) {
    return error(Phase19AuditError::invalid_feature_inventory);
  }
  for (std::size_t index = 0U; index < report.features.size(); ++index) {
    const Phase19FeatureAudit& feature = report.features[index];
    if (!enum_below(feature.id, Phase19FeatureId::count) ||
        static_cast<std::size_t>(feature.id) != index) {
      return error(Phase19AuditError::invalid_feature_inventory, index);
    }
    if (!valid_classification(feature.classification) ||
        !valid_text(feature.finding)) {
      return error(Phase19AuditError::invalid_feature_record, index);
    }
    if (!complete &&
        feature.classification ==
            Phase19FeatureClassification::implemented_performance_profiled) {
      return error(
          Phase19AuditError::premature_performance_profiled_feature, index);
    }
    const Phase19FeatureClassification local = local_classification(feature.id);
    const Phase19FeatureClassification expected =
        complete &&
                local != Phase19FeatureClassification::implemented_correctness_tested
            ? Phase19FeatureClassification::implemented_performance_profiled
            : local;
    const Phase19Fingerprint expected_fingerprint =
        expected ==
                Phase19FeatureClassification::implemented_performance_profiled
            ? evidence_identity
            : Phase19Fingerprint{};
    if (feature.classification != expected ||
        feature.evidence_fingerprint != expected_fingerprint) {
      return error(Phase19AuditError::invalid_feature_record, index);
    }
  }

  if (report.comparisons.size() != comparison_count) {
    return error(Phase19AuditError::invalid_comparison_inventory);
  }
  for (std::size_t index = 0U; index < report.comparisons.size(); ++index) {
    const Phase19Comparison& comparison = report.comparisons[index];
    if (!enum_below(comparison.id, Phase19ComparisonId::count) ||
        static_cast<std::size_t>(comparison.id) != index) {
      return error(Phase19AuditError::invalid_comparison_inventory, index);
    }
    if (!valid_conclusion_state(comparison.state) ||
        !valid_metric_unit(comparison.metric_unit) ||
        comparison.metric_unit != expected_metric_unit(comparison.id) ||
        !std::isfinite(comparison.metric_value) ||
        comparison.metric_value < 0.0 || std::signbit(comparison.metric_value) ||
        !valid_text(comparison.finding)) {
      return error(Phase19AuditError::invalid_comparison_record, index);
    }
    if (comparison.state == Phase19ConclusionState::insufficient_evidence) {
      if (comparison.selected_configuration_fingerprint != 0U ||
          comparison.metric_value != 0.0 ||
          !zero_fingerprint(comparison.evidence_fingerprint)) {
        return error(Phase19AuditError::invalid_comparison_record, index);
      }
      if (complete) {
        return error(Phase19AuditError::invalid_comparison_record, index);
      }
    } else {
      if (!complete) {
        return error(Phase19AuditError::premature_measured_comparison, index);
      }
      if (comparison.selected_configuration_fingerprint == 0U ||
          comparison.evidence_fingerprint != evidence_identity ||
          (comparison.metric_unit == Phase19MetricUnit::none &&
           comparison.metric_value != 0.0)) {
        return error(Phase19AuditError::invalid_comparison_record, index);
      }
    }
  }
  for (const Phase19ComparisonId id : {
           Phase19ComparisonId::box_miss_rate,
           Phase19ComparisonId::expansion_rate,
           Phase19ComparisonId::fallback_rate}) {
    const std::size_t index = static_cast<std::size_t>(id);
    if (report.comparisons[index].metric_value > 1.0) {
      return error(Phase19AuditError::invalid_comparison_record, index);
    }
  }
  const std::size_t p50_index =
      static_cast<std::size_t>(Phase19ComparisonId::latency_p50);
  const std::size_t p95_index =
      static_cast<std::size_t>(Phase19ComparisonId::latency_p95);
  const std::size_t p99_index =
      static_cast<std::size_t>(Phase19ComparisonId::latency_p99);
  if (report.comparisons[p50_index].metric_value >
          report.comparisons[p95_index].metric_value ||
      report.comparisons[p95_index].metric_value >
          report.comparisons[p99_index].metric_value) {
    return error(Phase19AuditError::invalid_comparison_record, p50_index);
  }
  const std::size_t best_engine_index =
      static_cast<std::size_t>(Phase19ComparisonId::best_engine);
  const std::size_t bottleneck_index = static_cast<std::size_t>(
      Phase19ComparisonId::profiler_supported_bottleneck);
  if (complete &&
      report.comparisons[bottleneck_index]
              .selected_configuration_fingerprint !=
          report.comparisons[best_engine_index]
              .selected_configuration_fingerprint) {
    return error(
        Phase19AuditError::invalid_comparison_record, bottleneck_index);
  }

  if (report.profiler_metrics.size() != profiler_metric_count) {
    return error(Phase19AuditError::invalid_profiler_metric_inventory);
  }
  for (std::size_t index = 0U; index < report.profiler_metrics.size(); ++index) {
    const Phase19ProfilerMetric& metric = report.profiler_metrics[index];
    if (!enum_below(metric.id, Phase19ProfilerMetricId::count) ||
        static_cast<std::size_t>(metric.id) != index) {
      return error(
          Phase19AuditError::invalid_profiler_metric_inventory, index);
    }
    if (!valid_conclusion_state(metric.state) ||
        !valid_metric_unit(metric.metric_unit) ||
        metric.metric_unit != expected_profiler_metric_unit(metric.id) ||
        !std::isfinite(metric.metric_value) || metric.metric_value < 0.0 ||
        std::signbit(metric.metric_value) ||
        !valid_text(metric.finding) ||
        (metric.metric_unit == Phase19MetricUnit::percentage &&
         metric.metric_value > 100.0)) {
      return error(Phase19AuditError::invalid_profiler_metric_record, index);
    }
    if (metric.state == Phase19ConclusionState::insufficient_evidence) {
      if (metric.configuration_fingerprint != 0U ||
          metric.metric_value != 0.0 ||
          !zero_fingerprint(metric.evidence_fingerprint)) {
        return error(Phase19AuditError::invalid_profiler_metric_record, index);
      }
      if (complete) {
        return error(Phase19AuditError::invalid_profiler_metric_record, index);
      }
    } else {
      if (!complete) {
        return error(
            Phase19AuditError::premature_measured_profiler_metric, index);
      }
      if (metric.configuration_fingerprint == 0U ||
          metric.configuration_fingerprint !=
              report.comparisons[bottleneck_index]
                  .selected_configuration_fingerprint ||
          metric.evidence_fingerprint != evidence_identity) {
        return error(Phase19AuditError::invalid_profiler_metric_record, index);
      }
    }
  }

  if (report.questions.size() != question_count) {
    return error(Phase19AuditError::invalid_question_inventory);
  }
  for (std::size_t index = 0U; index < report.questions.size(); ++index) {
    const Phase19QuestionAnswer& answer = report.questions[index];
    if (!enum_below(answer.id, Phase19QuestionId::count) ||
        static_cast<std::size_t>(answer.id) != index) {
      return error(Phase19AuditError::invalid_question_inventory, index);
    }
    if (!valid_conclusion_state(answer.state) || !valid_text(answer.finding)) {
      return error(Phase19AuditError::invalid_question_record, index);
    }
    if (answer.state == Phase19ConclusionState::insufficient_evidence) {
      if (!zero_fingerprint(answer.evidence_fingerprint) ||
          !answer.supporting_configuration_fingerprints.empty()) {
        return error(Phase19AuditError::invalid_question_record, index);
      }
      if (complete) {
        return error(Phase19AuditError::invalid_question_record, index);
      }
    } else {
      if (!complete) {
        return error(Phase19AuditError::premature_measured_question, index);
      }
      if (answer.evidence_fingerprint != evidence_identity ||
          answer.supporting_configuration_fingerprints.empty()) {
        return error(Phase19AuditError::invalid_question_record, index);
      }
      for (std::size_t left = 0U;
           left < answer.supporting_configuration_fingerprints.size(); ++left) {
        const std::uint64_t value =
            answer.supporting_configuration_fingerprints[left];
        if (value == 0U ||
            (left != 0U &&
             answer.supporting_configuration_fingerprints[left - 1U] >=
                 value)) {
          return error(Phase19AuditError::invalid_question_record, index);
        }
      }
    }
  }

  if (!valid_hybrid_disposition(report.hybrid_experiment) ||
      !valid_text(report.hybrid_finding)) {
    return error(Phase19AuditError::invalid_hybrid_disposition);
  }
  if (!complete &&
      report.hybrid_experiment !=
          Phase19HybridExperimentDisposition::deferred_until_standalone_results) {
    return error(Phase19AuditError::premature_hybrid_decision);
  }
  if (complete &&
      report.hybrid_experiment ==
          Phase19HybridExperimentDisposition::deferred_until_standalone_results) {
    return error(Phase19AuditError::invalid_hybrid_disposition);
  }

  if (!complete) {
    if (report.production_recommendation.has_value()) {
      return error(Phase19AuditError::premature_recommendation);
    }
    return {};
  }
  if (!report.production_recommendation.has_value()) {
    return error(Phase19AuditError::missing_recommendation);
  }
  const Phase19ProductionRecommendation& recommendation =
      *report.production_recommendation;
  if (!valid_production_configuration(recommendation.configuration) ||
      recommendation.evidence_fingerprint != evidence_identity ||
      !valid_text(recommendation.rationale) ||
      recommendation.configuration.configuration_fingerprint !=
          report.comparisons[static_cast<std::size_t>(
              Phase19ComparisonId::best_engine)]
              .selected_configuration_fingerprint) {
    return error(Phase19AuditError::invalid_recommendation);
  }
  return {};
}

std::string serialize_phase19_audit_tsv(const Phase19AuditReport& report) {
  if (!validate_phase19_audit(report).ok()) {
    throw std::invalid_argument{"cannot serialize an invalid Phase 19 audit"};
  }
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << "schema\t" << audit_schema << '\n';
  const Phase19EvidenceSnapshot& evidence = report.evidence;
  output << "snapshot\t" << evidence.corpus_query_count << '\t'
         << evidence.evaluated_query_count << '\t'
         << static_cast<std::uint32_t>(evidence.representative_query_artifact)
         << '\t'
         << static_cast<std::uint32_t>(evidence.hip_compiler_validation)
         << '\t'
         << static_cast<std::uint32_t>(evidence.gpu_device_validation) << '\t'
         << static_cast<std::uint32_t>(evidence.representative_correctness)
         << '\t'
         << static_cast<std::uint32_t>(evidence.representative_timing) << '\t'
         << static_cast<std::uint32_t>(evidence.dependency_trace) << '\t'
         << static_cast<std::uint32_t>(evidence.compatible_pmc) << '\t'
         << static_cast<std::uint32_t>(evidence.complete_candidate_matrix)
         << '\t' << static_cast<std::uint32_t>(evidence.all_query_evidence)
         << '\t'
         << static_cast<std::uint32_t>(evidence.tail_attribution_evidence)
         << '\t' << static_cast<std::uint32_t>(evidence.unique_winner) << '\t'
         << static_cast<std::uint32_t>(evidence.comparable_cpu_baseline) << '\t'
         << static_cast<std::uint32_t>(evidence.memory_evidence) << '\t'
         << static_cast<std::uint32_t>(
                evidence.synchronization_copy_evidence)
         << '\t' << static_cast<std::uint32_t>(evidence.path_quality_evidence)
         << '\n';
  write_fingerprint(output, "source", evidence.source_fingerprint);
  write_fingerprint(output, "build", evidence.build_fingerprint);
  write_fingerprint(output, "graph", evidence.graph_fingerprint);
  write_fingerprint(output, "query", evidence.query_fingerprint);
  write_fingerprint(output, "workload", evidence.workload_fingerprint);
  write_fingerprint(output, "device", evidence.device_fingerprint);
  write_fingerprint(output, "runtime", evidence.runtime_fingerprint);
  write_fingerprint(output, "protocol", evidence.protocol_fingerprint);
  write_fingerprint(
      output, "tail_attribution", evidence.tail_attribution_fingerprint);
  write_fingerprint(
      output, "candidate_catalog", evidence.candidate_catalog_fingerprint);
  write_fingerprint(output, "profiler", evidence.profiler_fingerprint);
  for (const Phase19RecommendationBlocker blocker : report.blockers) {
    output << "blocker\t" << static_cast<std::uint32_t>(blocker) << '\n';
  }
  for (const Phase19FeatureAudit& feature : report.features) {
    output << "feature\t" << static_cast<std::uint32_t>(feature.id) << '\t'
           << static_cast<std::uint32_t>(feature.classification) << '\t'
           << feature.evidence_fingerprint.words[0] << '\t'
           << feature.evidence_fingerprint.words[1] << '\t' << feature.finding
           << '\n';
  }
  output << std::setprecision(17);
  for (const Phase19Comparison& comparison : report.comparisons) {
    output << "comparison\t" << static_cast<std::uint32_t>(comparison.id)
           << '\t' << static_cast<std::uint32_t>(comparison.state) << '\t'
           << comparison.selected_configuration_fingerprint << '\t'
           << static_cast<std::uint32_t>(comparison.metric_unit) << '\t'
           << comparison.metric_value << '\t'
           << comparison.evidence_fingerprint.words[0] << '\t'
           << comparison.evidence_fingerprint.words[1] << '\t'
           << comparison.finding << '\n';
  }
  for (const Phase19ProfilerMetric& metric : report.profiler_metrics) {
    output << "profiler_metric\t" << static_cast<std::uint32_t>(metric.id)
           << '\t' << static_cast<std::uint32_t>(metric.state) << '\t'
           << metric.configuration_fingerprint << '\t'
           << static_cast<std::uint32_t>(metric.metric_unit) << '\t'
           << metric.metric_value << '\t' << metric.evidence_fingerprint.words[0]
           << '\t' << metric.evidence_fingerprint.words[1] << '\t'
           << metric.finding << '\n';
  }
  for (const Phase19QuestionAnswer& answer : report.questions) {
    output << "question\t" << static_cast<std::uint32_t>(answer.id) << '\t'
           << static_cast<std::uint32_t>(answer.state) << '\t'
           << answer.evidence_fingerprint.words[0] << '\t'
           << answer.evidence_fingerprint.words[1] << '\t';
    for (std::size_t index = 0U;
         index < answer.supporting_configuration_fingerprints.size(); ++index) {
      if (index != 0U) {
        output << ',';
      }
      output << answer.supporting_configuration_fingerprints[index];
    }
    output << '\t' << answer.finding << '\n';
  }
  output << "hybrid\t" << static_cast<std::uint32_t>(report.hybrid_experiment)
         << '\t' << report.hybrid_finding << '\n';
  output << "recommendation_state\t"
         << static_cast<std::uint32_t>(
                report.production_recommendation.has_value())
         << '\n';
  if (report.production_recommendation.has_value()) {
    const Phase19ProductionRecommendation& recommendation =
        *report.production_recommendation;
    const Phase19ProductionConfiguration& configuration =
        recommendation.configuration;
    const BatchedExpansionOptions& options = configuration.expansion;
    output << "recommendation\t" << configuration.configuration_fingerprint
           << '\t' << recommendation.evidence_fingerprint.words[0] << '\t'
           << recommendation.evidence_fingerprint.words[1] << '\t'
           << static_cast<std::uint32_t>(options.run_options.engine) << '\t'
           << static_cast<std::uint32_t>(options.run_options.control_mode)
           << '\t' << options.run_options.rounds_per_chunk << '\t'
           << options.run_options.block_size << '\t'
           << static_cast<std::uint32_t>(options.run_options.grid_policy) << '\t'
           << options.run_options.blocks_per_wgp << '\t'
           << static_cast<std::uint32_t>(options.run_options.instrumentation)
           << '\t' << options.run_options.maximum_rounds << '\t'
           << options.run_options.enable_per_lane_convergence << '\t'
           << options.planner_policy.lane_width << '\t'
           << options.planner_policy.minimum_jaccard_numerator << '\t'
           << options.planner_policy.minimum_jaccard_denominator << '\t'
           << options.planner_policy.maximum_union_inflation_numerator << '\t'
           << options.planner_policy.maximum_union_inflation_denominator << '\t'
           << options.execution_configuration_fingerprint << '\t'
           << static_cast<std::uint32_t>(options.schedule.kind) << '\t'
           << options.schedule.fixed_ring_size << '\t'
           << options.schedule.hybrid_small_expansion_count << '\t'
           << options.maximum_expansions << '\t'
           << static_cast<std::uint32_t>(options.terminal_policy) << '\t'
           << options.enable_compact_paths << '\t'
           << static_cast<std::uint32_t>(
                  configuration.workspace_vertex_storage)
           << '\t'
           << static_cast<std::uint32_t>(configuration.run_representation)
           << '\t' << configuration.tile_width << '\t'
           << configuration.tile_height << '\t'
           << configuration.frontier_queue_capacity << '\t'
           << static_cast<std::uint32_t>(configuration.jacobi_load_strategy)
           << '\t'
           << static_cast<std::uint32_t>(configuration.dense_load_strategy)
           << '\t' << static_cast<std::uint32_t>(configuration.transfer_mode)
           << '\t'
           << static_cast<std::uint32_t>(configuration.reconstruction_mode)
           << '\t' << recommendation.rationale << '\n';
  }
  return output.str();
}

Phase19AuditReport deserialize_phase19_audit_tsv(const std::string_view text) {
  const std::vector<std::string_view> lines = split_lines(text);
  std::size_t row = 0U;
  const auto next = [&]() -> std::string_view {
    if (row >= lines.size()) {
      throw std::invalid_argument{"Phase 19 TSV ended early"};
    }
    return lines[row++];
  };
  {
    const std::vector<std::string_view> fields = split_fields(next());
    require_fields(fields, 2U, "schema");
    if (fields[1] != audit_schema) {
      throw std::invalid_argument{"unsupported Phase 19 TSV schema"};
    }
  }

  Phase19AuditReport report;
  {
    const std::vector<std::string_view> fields = split_fields(next());
    require_fields(fields, 18U, "snapshot");
    Phase19EvidenceSnapshot& evidence = report.evidence;
    evidence.corpus_query_count =
        parse_unsigned<std::uint64_t>(fields[1], "corpus query count");
    evidence.evaluated_query_count =
        parse_unsigned<std::uint64_t>(fields[2], "evaluated query count");
    evidence.representative_query_artifact =
        parse_bool(fields[3], "query artifact flag");
    evidence.hip_compiler_validation =
        parse_bool(fields[4], "HIP compiler flag");
    evidence.gpu_device_validation = parse_bool(fields[5], "GPU device flag");
    evidence.representative_correctness =
        parse_bool(fields[6], "correctness flag");
    evidence.representative_timing = parse_bool(fields[7], "timing flag");
    evidence.dependency_trace = parse_bool(fields[8], "trace flag");
    evidence.compatible_pmc = parse_bool(fields[9], "PMC flag");
    evidence.complete_candidate_matrix =
        parse_bool(fields[10], "candidate matrix flag");
    evidence.all_query_evidence = parse_bool(fields[11], "all-query flag");
    evidence.tail_attribution_evidence =
        parse_bool(fields[12], "tail attribution flag");
    evidence.unique_winner = parse_bool(fields[13], "unique winner flag");
    evidence.comparable_cpu_baseline =
        parse_bool(fields[14], "CPU baseline flag");
    evidence.memory_evidence = parse_bool(fields[15], "memory flag");
    evidence.synchronization_copy_evidence =
        parse_bool(fields[16], "synchronization/copy flag");
    evidence.path_quality_evidence =
        parse_bool(fields[17], "path quality flag");
    evidence.source_fingerprint = parse_fingerprint_row(next(), "source");
    evidence.build_fingerprint = parse_fingerprint_row(next(), "build");
    evidence.graph_fingerprint = parse_fingerprint_row(next(), "graph");
    evidence.query_fingerprint = parse_fingerprint_row(next(), "query");
    evidence.workload_fingerprint = parse_fingerprint_row(next(), "workload");
    evidence.device_fingerprint = parse_fingerprint_row(next(), "device");
    evidence.runtime_fingerprint = parse_fingerprint_row(next(), "runtime");
    evidence.protocol_fingerprint = parse_fingerprint_row(next(), "protocol");
    evidence.tail_attribution_fingerprint =
        parse_fingerprint_row(next(), "tail_attribution");
    evidence.candidate_catalog_fingerprint =
        parse_fingerprint_row(next(), "candidate_catalog");
    evidence.profiler_fingerprint = parse_fingerprint_row(next(), "profiler");
  }

  while (row < lines.size()) {
    const std::vector<std::string_view> fields = split_fields(lines[row]);
    if (fields.front() != "blocker") {
      break;
    }
    require_fields(fields, 2U, "blocker");
    report.blockers.push_back(parse_enum<Phase19RecommendationBlocker>(
        fields[1], "recommendation blocker"));
    ++row;
  }

  report.features.reserve(feature_count);
  for (std::size_t index = 0U; index < feature_count; ++index) {
    const std::vector<std::string_view> fields = split_fields(next());
    require_fields(fields, 6U, "feature");
    report.features.push_back(Phase19FeatureAudit{
        parse_enum<Phase19FeatureId>(fields[1], "feature ID"),
        parse_enum<Phase19FeatureClassification>(
            fields[2], "feature classification"),
        Phase19Fingerprint{
            {parse_unsigned<std::uint64_t>(fields[3], "evidence word"),
             parse_unsigned<std::uint64_t>(fields[4], "evidence word")}},
        std::string{fields[5]}});
  }

  report.comparisons.reserve(comparison_count);
  for (std::size_t index = 0U; index < comparison_count; ++index) {
    const std::vector<std::string_view> fields = split_fields(next());
    require_fields(fields, 9U, "comparison");
    report.comparisons.push_back(Phase19Comparison{
        parse_enum<Phase19ComparisonId>(fields[1], "comparison ID"),
        parse_enum<Phase19ConclusionState>(fields[2], "comparison state"),
        parse_unsigned<std::uint64_t>(fields[3], "selected configuration"),
        parse_enum<Phase19MetricUnit>(fields[4], "metric unit"),
        parse_double(fields[5], "metric value"),
        Phase19Fingerprint{
            {parse_unsigned<std::uint64_t>(fields[6], "evidence word"),
             parse_unsigned<std::uint64_t>(fields[7], "evidence word")}},
        std::string{fields[8]}});
  }

  report.profiler_metrics.reserve(profiler_metric_count);
  for (std::size_t index = 0U; index < profiler_metric_count; ++index) {
    const std::vector<std::string_view> fields = split_fields(next());
    require_fields(fields, 9U, "profiler_metric");
    report.profiler_metrics.push_back(Phase19ProfilerMetric{
        parse_enum<Phase19ProfilerMetricId>(fields[1], "profiler metric ID"),
        parse_enum<Phase19ConclusionState>(fields[2], "profiler metric state"),
        parse_unsigned<std::uint64_t>(fields[3], "profiler configuration"),
        parse_enum<Phase19MetricUnit>(fields[4], "profiler metric unit"),
        parse_double(fields[5], "profiler metric value"),
        Phase19Fingerprint{
            {parse_unsigned<std::uint64_t>(fields[6], "evidence word"),
             parse_unsigned<std::uint64_t>(fields[7], "evidence word")}},
        std::string{fields[8]}});
  }

  report.questions.reserve(question_count);
  for (std::size_t index = 0U; index < question_count; ++index) {
    const std::vector<std::string_view> fields = split_fields(next());
    require_fields(fields, 7U, "question");
    report.questions.push_back(Phase19QuestionAnswer{
        parse_enum<Phase19QuestionId>(fields[1], "question ID"),
        parse_enum<Phase19ConclusionState>(fields[2], "question state"),
        Phase19Fingerprint{
            {parse_unsigned<std::uint64_t>(fields[3], "evidence word"),
             parse_unsigned<std::uint64_t>(fields[4], "evidence word")}},
        parse_configuration_list(fields[5]),
        std::string{fields[6]}});
  }
  {
    const std::vector<std::string_view> fields = split_fields(next());
    require_fields(fields, 3U, "hybrid");
    report.hybrid_experiment =
        parse_enum<Phase19HybridExperimentDisposition>(
            fields[1], "hybrid disposition");
    report.hybrid_finding = std::string{fields[2]};
  }
  bool has_recommendation = false;
  {
    const std::vector<std::string_view> fields = split_fields(next());
    require_fields(fields, 2U, "recommendation_state");
    has_recommendation = parse_bool(fields[1], "recommendation state");
  }
  if (has_recommendation) {
    const std::vector<std::string_view> fields = split_fields(next());
    require_fields(fields, 35U, "recommendation");
    Phase19ProductionRecommendation recommendation;
    Phase19ProductionConfiguration& configuration =
        recommendation.configuration;
    BatchedExpansionOptions& options = configuration.expansion;
    configuration.configuration_fingerprint =
        parse_unsigned<std::uint64_t>(fields[1], "configuration fingerprint");
    recommendation.evidence_fingerprint = Phase19Fingerprint{
        {parse_unsigned<std::uint64_t>(fields[2], "evidence word"),
         parse_unsigned<std::uint64_t>(fields[3], "evidence word")}};
    options.run_options.engine =
        parse_enum<EngineKind>(fields[4], "engine");
    options.run_options.control_mode =
        parse_enum<ControlMode>(fields[5], "control mode");
    options.run_options.rounds_per_chunk =
        parse_unsigned<std::uint32_t>(fields[6], "rounds per chunk");
    options.run_options.block_size =
        parse_unsigned<std::uint32_t>(fields[7], "block size");
    options.run_options.grid_policy =
        parse_enum<GridPolicy>(fields[8], "grid policy");
    options.run_options.blocks_per_wgp =
        parse_unsigned<std::uint32_t>(fields[9], "blocks per WGP");
    options.run_options.instrumentation =
        parse_enum<InstrumentationLevel>(fields[10], "instrumentation");
    options.run_options.maximum_rounds =
        parse_unsigned<std::uint64_t>(fields[11], "maximum rounds");
    options.run_options.enable_per_lane_convergence =
        parse_unsigned<std::uint32_t>(fields[12], "per-lane convergence");
    options.planner_policy.lane_width =
        parse_unsigned<std::uint32_t>(fields[13], "lane width");
    options.planner_policy.minimum_jaccard_numerator =
        parse_unsigned<std::uint32_t>(fields[14], "Jaccard numerator");
    options.planner_policy.minimum_jaccard_denominator =
        parse_unsigned<std::uint32_t>(fields[15], "Jaccard denominator");
    options.planner_policy.maximum_union_inflation_numerator =
        parse_unsigned<std::uint32_t>(fields[16], "inflation numerator");
    options.planner_policy.maximum_union_inflation_denominator =
        parse_unsigned<std::uint32_t>(fields[17], "inflation denominator");
    options.execution_configuration_fingerprint =
        parse_unsigned<std::uint64_t>(fields[18], "execution fingerprint");
    options.schedule.kind =
        parse_enum<ExpansionScheduleKind>(fields[19], "expansion schedule");
    options.schedule.fixed_ring_size =
        parse_unsigned<std::uint32_t>(fields[20], "fixed ring size");
    options.schedule.hybrid_small_expansion_count =
        parse_unsigned<std::uint32_t>(fields[21], "hybrid expansion count");
    options.maximum_expansions =
        parse_unsigned<std::uint32_t>(fields[22], "maximum expansions");
    options.terminal_policy =
        parse_enum<ExpansionTerminalPolicy>(fields[23], "terminal policy");
    options.enable_compact_paths =
        parse_unsigned<std::uint32_t>(fields[24], "compact paths flag");
    configuration.workspace_vertex_storage =
        parse_enum<BatchVertexStorageStrategy>(fields[25], "workspace strategy");
    configuration.run_representation =
        parse_enum<BatchRunRepresentation>(fields[26], "run representation");
    configuration.tile_width =
        parse_unsigned<std::uint32_t>(fields[27], "tile width");
    configuration.tile_height =
        parse_unsigned<std::uint32_t>(fields[28], "tile height");
    configuration.frontier_queue_capacity =
        parse_unsigned<std::uint32_t>(fields[29], "frontier queue capacity");
    configuration.jacobi_load_strategy =
        parse_enum<Phase19LoadStrategy>(fields[30], "Jacobi load strategy");
    configuration.dense_load_strategy =
        parse_enum<Phase19LoadStrategy>(fields[31], "dense load strategy");
    configuration.transfer_mode =
        parse_enum<Phase19TransferMode>(fields[32], "transfer mode");
    configuration.reconstruction_mode = parse_enum<Phase19ReconstructionMode>(
        fields[33], "reconstruction mode");
    recommendation.rationale = std::string{fields[34]};
    report.production_recommendation = std::move(recommendation);
  }
  if (row != lines.size()) {
    throw std::invalid_argument{"Phase 19 TSV has unexpected trailing rows"};
  }
  const Phase19AuditValidationResult validation =
      validate_phase19_audit(report);
  if (!validation.ok()) {
    throw std::invalid_argument{"deserialized Phase 19 audit is invalid"};
  }
  const std::string canonical = serialize_phase19_audit_tsv(report);
  if (std::string_view{canonical} != text) {
    throw std::invalid_argument{"Phase 19 TSV is not canonical"};
  }
  return report;
}

}  // namespace bfnew
