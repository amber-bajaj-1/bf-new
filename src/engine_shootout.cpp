#include "bfnew/engine_shootout.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

constexpr std::array<EngineKind, 3U> engines{
    EngineKind::jacobi_pull,
    EngineKind::dense_chaotic_push,
    EngineKind::frontier_push,
};
constexpr std::array<std::uint32_t, 3U> block_sizes{128U, 256U, 512U};
constexpr std::array<std::uint32_t, 5U> chunk_sizes{2U, 4U, 8U, 16U, 32U};

using Stratum =
    std::array<std::uint8_t, shootout_stratification_dimensions>;

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] std::uint64_t keyed_hash(
    const std::uint64_t seed,
    const std::uint64_t first,
    const std::uint64_t second = 0U) noexcept {
  return mix64(seed ^ mix64(first) ^ (mix64(second) << 1U));
}

void require(const bool condition, const char* const message) {
  if (!condition) {
    throw std::invalid_argument{message};
  }
}

[[nodiscard]] bool valid_workload_kind(
    const ShootoutWorkloadKind kind) noexcept {
  switch (kind) {
    case ShootoutWorkloadKind::logicnets_jscl:
    case ShootoutWorkloadKind::synthetic:
      return true;
  }
  return false;
}

[[nodiscard]] bool valid_run_kind(const ShootoutRunKind kind) noexcept {
  switch (kind) {
    case ShootoutRunKind::warmup:
    case ShootoutRunKind::correctness:
    case ShootoutRunKind::timing:
    case ShootoutRunKind::algorithm_counters:
    case ShootoutRunKind::trace:
    case ShootoutRunKind::pmc:
      return true;
  }
  return false;
}

void validate_workload(const ShootoutWorkloadIdentity& workload) {
  require(valid_workload_kind(workload.kind), "invalid shootout workload kind");
  require(workload.case_id != 0U, "shootout workload requires a case ID");
  require(!workload.case_name.empty(), "shootout workload requires a case name");
  require(
      workload.case_name.find_first_of("\t\r\n") == std::string::npos,
      "shootout workload case name is not TSV-safe");
}

void validate_fingerprint(const ShootoutInputFingerprint& fingerprint) {
  require(
      fingerprint.schema_version == shootout_schema_version,
      "shootout fingerprint schema version is unsupported");
  require(
      fingerprint.corpus_query_count != 0U,
      "shootout fingerprint requires a nonempty corpus");
}

[[nodiscard]] bool allowed_block_size(const std::uint32_t block_size) noexcept {
  return std::ranges::find(block_sizes, block_size) != block_sizes.end();
}

[[nodiscard]] bool allowed_chunk_size(const std::uint32_t chunk_size) noexcept {
  return std::ranges::find(chunk_sizes, chunk_size) != chunk_sizes.end();
}

[[nodiscard]] GpuRunOptions options_for_tuning(
    const ShootoutTuning& tuning,
    const InstrumentationLevel instrumentation) noexcept {
  GpuRunOptions options;
  options.engine = tuning.engine;
  options.control_mode = tuning.control_mode;
  options.rounds_per_chunk = tuning.rounds_per_chunk;
  options.block_size = tuning.block_size;
  options.grid_policy = tuning.grid_policy;
  options.blocks_per_wgp = tuning.blocks_per_wgp;
  options.instrumentation = instrumentation;
  options.maximum_rounds = tuning.maximum_rounds;
  return options;
}

[[nodiscard]] bool valid_tuning(const ShootoutTuning& tuning) noexcept {
  if (!allowed_block_size(tuning.block_size) || tuning.maximum_rounds == 0U ||
      validate_gpu_run_options(
          options_for_tuning(tuning, InstrumentationLevel::none)) !=
          GpuRunOptionsError::none) {
    return false;
  }
  switch (tuning.control_mode) {
    case ControlMode::persistent_cooperative:
      return tuning.rounds_per_chunk == 1U;
    case ControlMode::per_round_host_poll:
      return tuning.rounds_per_chunk == 1U &&
             tuning.grid_policy == GridPolicy::occupancy_derived &&
             tuning.blocks_per_wgp == 0U;
    case ControlMode::chunked_host_poll:
      return allowed_chunk_size(tuning.rounds_per_chunk) &&
             tuning.grid_policy == GridPolicy::occupancy_derived &&
             tuning.blocks_per_wgp == 0U;
  }
  return false;
}

[[nodiscard]] std::map<std::uint32_t, const ShootoutTuning*> tuning_index(
    const std::span<const ShootoutTuning> tunings) {
  require(!tunings.empty(), "shootout requires at least one tuning");
  std::map<std::uint32_t, const ShootoutTuning*> result;
  for (const ShootoutTuning& tuning : tunings) {
    require(valid_tuning(tuning), "shootout tuning is invalid");
    require(
        result.emplace(tuning.configuration_id, &tuning).second,
        "shootout tuning IDs must be unique");
  }
  return result;
}

[[nodiscard]] std::uint64_t feature_value(
    const ShootoutQueryFeatures& features,
    const std::size_t dimension) noexcept {
  switch (dimension) {
    case 0U:
      return features.selected_vertices;
    case 1U:
      return features.selected_edges;
    case 2U:
      return features.fanout;
    case 3U:
      return features.source_count;
    case 4U:
      return features.expected_rounds;
    default:
      return 0U;
  }
}

void validate_features(const ShootoutQueryFeatures& features) {
  require(
      features.selected_vertices != 0U,
      "shootout query features require selected vertices");
  require(features.fanout != 0U, "shootout query features require fanout");
  require(
      features.source_count != 0U,
      "shootout query features require at least one source");
  require(
      features.expected_rounds != 0U,
      "shootout query features require positive pilot rounds");
}

[[nodiscard]] std::array<std::uint64_t, 3U> quantile_thresholds(
    const std::span<const ShootoutQueryFeatures> features,
    const std::size_t dimension) {
  require(!features.empty(), "shootout quantiles require candidates");
  std::vector<std::uint64_t> values;
  values.reserve(features.size());
  for (const ShootoutQueryFeatures& value : features) {
    values.push_back(feature_value(value, dimension));
  }
  std::sort(values.begin(), values.end());
  const auto threshold = [&values](const std::size_t numerator) {
    const std::size_t rank = (values.size() * numerator + 3U) / 4U;
    return values[std::min(values.size() - 1U, rank == 0U ? 0U : rank - 1U)];
  };
  return {threshold(1U), threshold(2U), threshold(3U)};
}

[[nodiscard]] std::uint8_t quantile_bin(
    const std::uint64_t value,
    const std::array<std::uint64_t, 3U>& thresholds) noexcept {
  return static_cast<std::uint8_t>(
      std::lower_bound(thresholds.begin(), thresholds.end(), value) -
      thresholds.begin());
}

[[nodiscard]] std::vector<Stratum> feature_bins(
    const std::span<const ShootoutQueryFeatures> candidates) {
  std::array<std::array<std::uint64_t, 3U>,
             shootout_stratification_dimensions>
      thresholds{};
  for (std::size_t dimension = 0U;
       dimension < shootout_stratification_dimensions;
       ++dimension) {
    thresholds[dimension] = quantile_thresholds(candidates, dimension);
  }
  std::vector<Stratum> bins(candidates.size());
  for (std::size_t index = 0U; index < candidates.size(); ++index) {
    for (std::size_t dimension = 0U;
         dimension < shootout_stratification_dimensions;
         ++dimension) {
      bins[index][dimension] = quantile_bin(
          feature_value(candidates[index], dimension), thresholds[dimension]);
    }
  }
  return bins;
}

void validate_manifest(const ShootoutManifest& manifest) {
  validate_fingerprint(manifest.fingerprint);
  validate_workload(manifest.workload);
  require(
      manifest.warmup_repetitions != 0U &&
          manifest.timing_repetitions != 0U,
      "shootout manifest requires positive warmup/timing repetitions");
  require(
      manifest.requested_query_count != 0U &&
          manifest.entries.size() == manifest.requested_query_count,
      "shootout manifest query count is inconsistent");
  require(
      manifest.fingerprint.corpus_query_count >= manifest.entries.size(),
      "shootout manifest exceeds its fingerprinted corpus");
  if (manifest.workload.kind == ShootoutWorkloadKind::logicnets_jscl) {
    require(
        manifest.requested_query_count >= minimum_logicnets_shootout_queries,
        "logicnets shootout manifest contains fewer than 1000 queries");
  }
  std::set<std::uint32_t> query_ids;
  for (const ShootoutManifestEntry& entry : manifest.entries) {
    validate_features(entry.features);
    require(
        query_ids.insert(entry.features.query_id.value()).second,
        "shootout manifest contains a duplicate query ID");
    require(
        std::ranges::all_of(entry.quantile_bins, [](const std::uint8_t value) {
          return value <= 3U;
        }),
        "shootout manifest contains an invalid quantile bin");
  }
}

[[nodiscard]] std::map<std::uint32_t, ShootoutQueryFeatures> feature_index(
    const ShootoutManifest& manifest) {
  std::map<std::uint32_t, ShootoutQueryFeatures> result;
  for (const ShootoutManifestEntry& entry : manifest.entries) {
    result.emplace(entry.features.query_id.value(), entry.features);
  }
  return result;
}

[[nodiscard]] bool valid_evidence_state(
    const ShootoutEvidenceState state) noexcept {
  switch (state) {
    case ShootoutEvidenceState::unavailable:
    case ShootoutEvidenceState::not_applicable:
    case ShootoutEvidenceState::measured:
      return true;
  }
  return false;
}

void validate_evidence_value(const ShootoutEvidenceValue& value) {
  require(valid_evidence_state(value.state), "invalid shootout evidence state");
  if (value.state == ShootoutEvidenceState::measured) {
    require(
        std::isfinite(value.value) && value.value >= 0.0,
        "measured shootout evidence must be finite and nonnegative");
  } else {
    require(
        value.value == 0.0,
        "unavailable or nonapplicable evidence may not encode a number");
  }
}

template <typename Function>
void for_each_timing_value(
    const ShootoutTimingRecord& timing,
    Function&& function) {
  function(timing.preparation_gpu_milliseconds);
  function(timing.sssp_device_timeline_milliseconds);
  function(timing.result_transfer_gpu_milliseconds);
  function(timing.end_to_end_wall_milliseconds);
}

template <typename Function>
void for_each_profiler_value(
    const ShootoutProfilerRecord& profiler,
    Function&& function) {
  function(profiler.gpu_active_milliseconds);
  function(profiler.l2_hit_percent);
  function(profiler.l2_read_bytes);
  function(profiler.l2_write_bytes);
  function(profiler.occupancy_percent);
  function(profiler.memory_unit_busy_percent);
  function(profiler.waves);
  function(profiler.vector_instructions);
  function(profiler.scalar_instructions);
  function(profiler.memory_instructions);
}

[[nodiscard]] bool timing_is_unavailable(const ShootoutTimingRecord& timing) {
  bool unavailable = true;
  for_each_timing_value(timing, [&unavailable](const ShootoutEvidenceValue& value) {
    unavailable = unavailable &&
                  value.state == ShootoutEvidenceState::unavailable;
  });
  return unavailable;
}

[[nodiscard]] bool profiler_is_unavailable(
    const ShootoutProfilerRecord& profiler) {
  bool unavailable = true;
  for_each_profiler_value(
      profiler, [&unavailable](const ShootoutEvidenceValue& value) {
        unavailable = unavailable &&
                      value.state == ShootoutEvidenceState::unavailable;
      });
  return unavailable;
}

[[nodiscard]] bool hardware_profiler_is_unavailable(
    const ShootoutProfilerRecord& profiler) {
  bool unavailable = true;
  const auto check = [&unavailable](const ShootoutEvidenceValue& value) {
    unavailable = unavailable &&
                  value.state == ShootoutEvidenceState::unavailable;
  };
  check(profiler.l2_hit_percent);
  check(profiler.l2_read_bytes);
  check(profiler.l2_write_bytes);
  check(profiler.occupancy_percent);
  check(profiler.memory_unit_busy_percent);
  check(profiler.waves);
  check(profiler.vector_instructions);
  check(profiler.scalar_instructions);
  check(profiler.memory_instructions);
  return unavailable;
}

[[nodiscard]] bool any_hardware_profiler_measurement(
    const ShootoutProfilerRecord& profiler) {
  bool measured = false;
  const auto check = [&measured](const ShootoutEvidenceValue& value) {
    measured = measured || value.state == ShootoutEvidenceState::measured;
  };
  check(profiler.l2_hit_percent);
  check(profiler.l2_read_bytes);
  check(profiler.l2_write_bytes);
  check(profiler.occupancy_percent);
  check(profiler.memory_unit_busy_percent);
  check(profiler.waves);
  check(profiler.vector_instructions);
  check(profiler.scalar_instructions);
  check(profiler.memory_instructions);
  return measured;
}

void validate_percent(const ShootoutEvidenceValue& value) {
  if (value.state == ShootoutEvidenceState::measured) {
    require(value.value <= 100.0, "shootout percentage exceeds 100");
  }
}

[[nodiscard]] bool successful_status(const DeviceRunStatus& status) noexcept {
  return status.converged == 1U &&
         status.stop_reason ==
             static_cast<std::uint32_t>(DeviceStopReason::converged) &&
         status.error_bits == device_error::none;
}

using BasicSampleIdentity =
    std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
using ProfileSampleIdentity = std::tuple<
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint64_t,
    std::uint64_t>;
using ProfileExecutionIdentity =
    std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>;

[[nodiscard]] BasicSampleIdentity basic_sample_identity(
    const ShootoutSample& sample) noexcept {
  return {
      sample.query_id.value(), sample.configuration_id, sample.repetition};
}

[[nodiscard]] ProfileSampleIdentity profile_sample_identity(
    const ShootoutSample& sample) noexcept {
  return {
      sample.query_id.value(),
      sample.configuration_id,
      sample.repetition,
      sample.profiler_provenance.pass_id,
      sample.profiler_provenance.counter_set_id,
  };
}

void validate_rectangular_stage(
    const ShootoutManifest& manifest,
    const std::span<const ShootoutTuning> tunings,
    const std::span<const ShootoutSample> samples,
    const ShootoutRunKind kind) {
  if (samples.empty()) {
    require(
        kind != ShootoutRunKind::warmup,
        "shootout warmup stage cannot be empty");
    return;
  }
  if (kind == ShootoutRunKind::trace || kind == ShootoutRunKind::pmc) {
    return;
  }
  std::uint32_t maximum_repetition = 0U;
  for (const ShootoutSample& sample : samples) {
    maximum_repetition = std::max(maximum_repetition, sample.repetition);
  }
  if (kind == ShootoutRunKind::correctness) {
    require(
        maximum_repetition == 0U,
        "shootout correctness gate must have exactly one repetition");
  }
  if (kind == ShootoutRunKind::algorithm_counters) {
    require(
        maximum_repetition == 0U,
        "shootout counter stage must have exactly one repetition");
  }
  if (kind == ShootoutRunKind::warmup) {
    require(
        maximum_repetition + 1U == manifest.warmup_repetitions,
        "shootout warmup count differs from its persisted campaign");
  }
  if (kind == ShootoutRunKind::timing) {
    require(
        maximum_repetition + 1U == manifest.timing_repetitions,
        "shootout timing count differs from its persisted campaign");
  }
  std::size_t configuration_count = tunings.size();
  if (kind == ShootoutRunKind::algorithm_counters) {
    std::set<std::uint32_t> included;
    for (const ShootoutSample& sample : samples) {
      included.insert(sample.configuration_id);
    }
    require(
        !included.empty(),
        "shootout counter stage requires at least one supported tuning");
    configuration_count = included.size();
  }
  const std::uint64_t cells =
      static_cast<std::uint64_t>(manifest.entries.size()) *
      configuration_count;
  const std::uint64_t repetitions =
      static_cast<std::uint64_t>(maximum_repetition) + 1U;
  require(
      cells == 0U || repetitions <=
                           std::numeric_limits<std::uint64_t>::max() / cells,
      "shootout stage cardinality overflow");
  require(
      samples.size() == cells * repetitions,
      "shootout stage is not a complete query/configuration rectangle");
}

void validate_schedule_binding(
    const ShootoutManifest& manifest,
    const std::span<const ShootoutTuning> tunings,
    const std::span<const ShootoutSample> samples,
    const ShootoutRunKind kind) {
  if (samples.empty()) {
    return;
  }
  std::uint32_t repetitions = 1U;
  for (const ShootoutSample& sample : samples) {
    repetitions = std::max(repetitions, sample.repetition + 1U);
  }
  std::vector<ShootoutTuning> stage_tunings;
  if (kind == ShootoutRunKind::algorithm_counters) {
    std::set<std::uint32_t> included;
    for (const ShootoutSample& sample : samples) {
      included.insert(sample.configuration_id);
    }
    for (const ShootoutTuning& tuning : tunings) {
      if (included.contains(tuning.configuration_id)) {
        stage_tunings.push_back(tuning);
      }
    }
  } else {
    stage_tunings.assign(tunings.begin(), tunings.end());
  }
  const std::vector<ShootoutScheduleEntry> schedule =
      kind == ShootoutRunKind::correctness
          ? make_grouped_shootout_correctness_schedule(
                manifest, stage_tunings)
          : make_interleaved_shootout_schedule(
                manifest,
                stage_tunings,
                kind,
                repetitions,
                manifest.order_seed);
  std::map<BasicSampleIdentity, std::uint64_t> expected_ordinals;
  for (const ShootoutScheduleEntry& entry : schedule) {
    expected_ordinals.emplace(
        BasicSampleIdentity{
            entry.query_id.value(),
            entry.configuration_id,
            entry.repetition},
        entry.execution_ordinal);
  }
  for (const ShootoutSample& sample : samples) {
    const auto found = expected_ordinals.find(basic_sample_identity(sample));
    require(
        found != expected_ordinals.end() &&
            found->second == sample.execution_ordinal,
        "shootout sample execution ordinal/order seed is inconsistent");
  }
}

void checked_add(std::uint64_t& total, const std::uint64_t value) {
  if (total > std::numeric_limits<std::uint64_t>::max() - value) {
    throw std::overflow_error{"shootout aggregate counter overflow"};
  }
  total += value;
}

[[nodiscard]] ShootoutEvidenceValue evidence_ratio(
    const std::uint64_t numerator,
    const std::uint64_t denominator,
    const bool available) {
  if (!available) {
    return {};
  }
  if (denominator == 0U) {
    return {ShootoutEvidenceState::not_applicable, 0.0};
  }
  return {
      ShootoutEvidenceState::measured,
      static_cast<double>(numerator) / static_cast<double>(denominator),
  };
}

[[nodiscard]] ShootoutProfileMetricSummary summarize_profile_metric(
    const std::span<const ShootoutSample> samples,
    ShootoutEvidenceValue ShootoutProfilerRecord::* const member) {
  ShootoutProfileMetricSummary result;
  double total = 0.0;
  std::uint64_t count = 0U;
  bool saw_not_applicable = false;
  std::set<std::uint64_t> counter_sets;
  std::set<std::uint64_t> passes;
  for (const ShootoutSample& sample : samples) {
    const ShootoutEvidenceValue& value = sample.profiler.*member;
    if (value.state == ShootoutEvidenceState::measured) {
      total += value.value;
      ++count;
      counter_sets.insert(sample.profiler_provenance.counter_set_id);
      passes.insert(sample.profiler_provenance.pass_id);
    } else if (value.state == ShootoutEvidenceState::not_applicable) {
      saw_not_applicable = true;
    }
  }
  require(
      counter_sets.size() <= 1U,
      "shootout summary would combine incompatible PMC counter sets");
  if (count != 0U) {
    result.mean = {
        ShootoutEvidenceState::measured,
        total / static_cast<double>(count),
    };
    result.counter_set_id = *counter_sets.begin();
    result.pass_ids.assign(passes.begin(), passes.end());
  } else if (saw_not_applicable) {
    result.mean = {ShootoutEvidenceState::not_applicable, 0.0};
  }
  return result;
}

[[nodiscard]] std::string json_escape(const std::string& value) {
  std::string result;
  result.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        result.push_back(character);
        break;
    }
  }
  return result;
}

[[nodiscard]] bool valid_conclusion_state(
    const ShootoutConclusionState state) noexcept {
  switch (state) {
    case ShootoutConclusionState::pending:
    case ShootoutConclusionState::insufficient_evidence:
    case ShootoutConclusionState::measured:
      return true;
  }
  return false;
}

void validate_evidence_reference(const ShootoutEvidenceReference& reference) {
  require(valid_run_kind(reference.run_kind), "invalid evidence reference kind");
  if (reference.run_kind == ShootoutRunKind::trace) {
    require(
        reference.profiler_pass_id != 0U &&
            reference.profiler_counter_set_id == 0U,
        "trace evidence reference has invalid provenance");
  } else if (reference.run_kind == ShootoutRunKind::pmc) {
    require(
        reference.profiler_pass_id != 0U &&
            reference.profiler_counter_set_id != 0U,
        "PMC evidence reference has invalid provenance");
  } else {
    require(
        reference.profiler_pass_id == 0U &&
            reference.profiler_counter_set_id == 0U,
        "non-profiler evidence reference carries profiler provenance");
  }
}

[[nodiscard]] ShootoutEvidenceReference evidence_reference(
    const ShootoutSample& sample) noexcept {
  return ShootoutEvidenceReference{
      sample.run_kind,
      sample.execution_ordinal,
      sample.profiler_provenance.pass_id,
      sample.profiler_provenance.counter_set_id,
      sample.query_id,
      sample.configuration_id,
  };
}

void validate_measured_evidence_mix(
    const std::span<const ShootoutEvidenceReference> evidence) {
  bool timing = false;
  bool counters = false;
  for (const ShootoutEvidenceReference& reference : evidence) {
    validate_evidence_reference(reference);
    timing = timing || reference.run_kind == ShootoutRunKind::timing;
    counters = counters ||
               reference.run_kind == ShootoutRunKind::algorithm_counters ||
               reference.run_kind == ShootoutRunKind::trace ||
               reference.run_kind == ShootoutRunKind::pmc;
  }
  require(
      timing && counters,
      "measured shootout conclusion requires timing and counter evidence");
}

// TSV parsing helpers intentionally reject quoting and implicit defaults. Case
// names are validated as tab/newline-free, so every persisted row has an exact
// field count and deterministic representation.
[[nodiscard]] std::vector<std::string_view> split_fields(
    const std::string_view line) {
  std::vector<std::string_view> fields;
  std::size_t begin = 0U;
  while (true) {
    const std::size_t end = line.find('\t', begin);
    fields.push_back(line.substr(begin, end - begin));
    if (end == std::string_view::npos) {
      return fields;
    }
    begin = end + 1U;
  }
}

[[nodiscard]] std::vector<std::string_view> data_lines(
    const std::string_view text,
    const std::string_view header) {
  require(!text.empty(), "shootout TSV is empty");
  const std::size_t first_end = text.find('\n');
  require(first_end != std::string_view::npos, "shootout TSV has no data rows");
  require(text.substr(0U, first_end) == header, "shootout TSV header mismatch");
  std::vector<std::string_view> lines;
  std::size_t begin = first_end + 1U;
  while (begin < text.size()) {
    const std::size_t end = text.find('\n', begin);
    const std::string_view line = text.substr(begin, end - begin);
    require(!line.empty(), "shootout TSV contains an empty data row");
    require(line.back() != '\r', "shootout TSV must use LF line endings");
    lines.push_back(line);
    if (end == std::string_view::npos) {
      begin = text.size();
    } else {
      begin = end + 1U;
    }
  }
  require(!lines.empty(), "shootout TSV has no data rows");
  return lines;
}

template <typename Integer>
[[nodiscard]] Integer parse_unsigned(
    const std::string_view field,
    const char* const message) {
  static_assert(std::is_unsigned_v<Integer>);
  Integer result{};
  const auto parsed = std::from_chars(
      field.data(), field.data() + field.size(), result, 10);
  require(
      parsed.ec == std::errc{} && parsed.ptr == field.data() + field.size(),
      message);
  return result;
}

[[nodiscard]] bool parse_bool(
    const std::string_view field,
    const char* const message) {
  const std::uint32_t value = parse_unsigned<std::uint32_t>(field, message);
  require(value <= 1U, message);
  return value != 0U;
}

[[nodiscard]] double parse_double(
    const std::string_view field,
    const char* const message) {
  std::istringstream input{std::string{field}};
  input.imbue(std::locale::classic());
  double value{};
  input >> value;
  require(input && input.peek() == std::char_traits<char>::eof(), message);
  return value;
}

void append_evidence(
    std::ostream& output,
    const ShootoutEvidenceValue& value) {
  output << '\t' << static_cast<std::uint32_t>(value.state) << '\t'
         << std::setprecision(std::numeric_limits<double>::max_digits10)
         << value.value;
}

[[nodiscard]] ShootoutEvidenceValue parse_evidence(
    const std::vector<std::string_view>& fields,
    std::size_t& position) {
  require(position + 1U < fields.size(), "truncated shootout evidence value");
  ShootoutEvidenceValue value;
  value.state = static_cast<ShootoutEvidenceState>(
      parse_unsigned<std::uint32_t>(fields[position++], "invalid evidence state"));
  value.value = parse_double(fields[position++], "invalid evidence number");
  validate_evidence_value(value);
  return value;
}

}  // namespace

std::vector<ShootoutTuning> make_shootout_tunings(
    const std::uint64_t maximum_rounds,
    const std::span<const std::uint32_t> fixed_persistent_blocks_per_wgp) {
  require(maximum_rounds != 0U, "shootout maximum rounds must be positive");
  std::vector<std::uint32_t> fixed(
      fixed_persistent_blocks_per_wgp.begin(),
      fixed_persistent_blocks_per_wgp.end());
  require(
      std::ranges::none_of(fixed, [](const std::uint32_t value) {
        return value == 0U;
      }),
      "fixed persistent blocks per WGP must be positive");
  std::sort(fixed.begin(), fixed.end());
  fixed.erase(std::unique(fixed.begin(), fixed.end()), fixed.end());

  std::vector<ShootoutTuning> result;
  std::uint32_t configuration_id = 0U;
  const auto append = [&](const EngineKind engine,
                          const ControlMode control,
                          const std::uint32_t chunk,
                          const std::uint32_t block,
                          const GridPolicy grid,
                          const std::uint32_t residency) {
    require(
        configuration_id != std::numeric_limits<std::uint32_t>::max(),
        "shootout configuration ID overflow");
    result.push_back(ShootoutTuning{
        configuration_id++,
        engine,
        control,
        chunk,
        block,
        grid,
        residency,
        maximum_rounds,
    });
  };

  for (const EngineKind engine : engines) {
    for (const std::uint32_t block : block_sizes) {
      append(
          engine,
          ControlMode::persistent_cooperative,
          1U,
          block,
          GridPolicy::occupancy_derived,
          0U);
      for (const std::uint32_t residency : fixed) {
        append(
            engine,
            ControlMode::persistent_cooperative,
            1U,
            block,
            GridPolicy::fixed_blocks_per_wgp,
            residency);
      }
      append(
          engine,
          ControlMode::per_round_host_poll,
          1U,
          block,
          GridPolicy::occupancy_derived,
          0U);
      for (const std::uint32_t chunk : chunk_sizes) {
        append(
            engine,
            ControlMode::chunked_host_poll,
            chunk,
            block,
            GridPolicy::occupancy_derived,
            0U);
      }
    }
  }
  return result;
}

std::vector<ShootoutConfigurationDecision> resolve_shootout_configurations(
    const std::span<const ShootoutTuning> tunings,
    const std::span<const ShootoutKernelLimit> limits) {
  static_cast<void>(tuning_index(tunings));
  std::map<std::pair<EngineKind, std::uint32_t>, ShootoutKernelLimit> indexed;
  for (const ShootoutKernelLimit& limit : limits) {
    require(
        std::ranges::find(engines, limit.engine) != engines.end(),
        "shootout kernel limit has an invalid engine");
    require(
        allowed_block_size(limit.block_size),
        "shootout kernel limit has an unsupported block size");
    require(
        indexed.emplace(std::pair{limit.engine, limit.block_size}, limit).second,
        "shootout kernel limits contain a duplicate engine/block row");
  }

  std::vector<ShootoutConfigurationDecision> decisions;
  decisions.reserve(tunings.size());
  for (const ShootoutTuning& tuning : tunings) {
    ShootoutConfigurationDecision decision;
    decision.tuning = tuning;
    if (!valid_tuning(tuning)) {
      decision.rejection = ShootoutConfigurationRejection::invalid_tuning;
      decisions.push_back(decision);
      continue;
    }
    const auto found = indexed.find(std::pair{tuning.engine, tuning.block_size});
    if (found == indexed.end()) {
      decision.rejection =
          ShootoutConfigurationRejection::missing_kernel_limit;
      decisions.push_back(decision);
      continue;
    }
    decision.measured_persistent_active_blocks_per_wgp =
        found->second.persistent_active_blocks_per_wgp;
    if (tuning.control_mode != ControlMode::persistent_cooperative &&
        !found->second.ordinary_block_size_legal) {
      decision.rejection =
          ShootoutConfigurationRejection::illegal_ordinary_block_size;
    } else if (
        tuning.control_mode == ControlMode::persistent_cooperative &&
        (!found->second.persistent_block_size_legal ||
         found->second.persistent_active_blocks_per_wgp == 0U)) {
      decision.rejection =
          ShootoutConfigurationRejection::illegal_persistent_block_size;
    } else if (
        tuning.control_mode == ControlMode::persistent_cooperative &&
        tuning.grid_policy == GridPolicy::fixed_blocks_per_wgp &&
        tuning.blocks_per_wgp >
            found->second.persistent_active_blocks_per_wgp) {
      decision.rejection =
          ShootoutConfigurationRejection::illegal_persistent_residency;
    }
    decisions.push_back(decision);
  }
  return decisions;
}

ShootoutManifest select_logicnets_shootout_manifest(
    const ShootoutInputFingerprint& fingerprint,
    const ShootoutWorkloadIdentity& workload,
    const std::span<const ShootoutQueryFeatures> candidates,
    const std::uint32_t requested_query_count,
    const std::uint64_t selection_seed,
    const std::uint64_t order_seed,
    const std::uint32_t warmup_repetitions,
    const std::uint32_t timing_repetitions) {
  validate_fingerprint(fingerprint);
  validate_workload(workload);
  require(
      workload.kind == ShootoutWorkloadKind::logicnets_jscl,
      "logicnets selector requires a logicnets workload identity");
  require(
      requested_query_count >= minimum_logicnets_shootout_queries,
      "logicnets shootout requires at least 1000 queries");
  require(
      requested_query_count <= candidates.size(),
      "logicnets corpus contains fewer eligible queries than requested");
  require(
      fingerprint.corpus_query_count >= candidates.size(),
      "shootout fingerprint corpus count is smaller than its candidates");

  std::set<std::uint32_t> query_ids;
  for (const ShootoutQueryFeatures& candidate : candidates) {
    validate_features(candidate);
    require(
        query_ids.insert(candidate.query_id.value()).second,
        "shootout candidates contain a duplicate query ID");
  }

  const std::vector<Stratum> bins = feature_bins(candidates);
  struct Group {
    Stratum stratum{};
    std::vector<std::size_t> indexes;
    std::size_t allocation{};
    std::uint64_t remainder{};
  };
  std::map<Stratum, std::vector<std::size_t>> grouped;
  for (std::size_t index = 0U; index < candidates.size(); ++index) {
    grouped[bins[index]].push_back(index);
  }
  std::vector<Group> groups;
  groups.reserve(grouped.size());
  std::size_t allocated = 0U;
  for (auto& [stratum, indexes] : grouped) {
    std::sort(
        indexes.begin(),
        indexes.end(),
        [&](const std::size_t left, const std::size_t right) {
          const std::uint64_t left_hash =
              keyed_hash(selection_seed, candidates[left].query_id.value());
          const std::uint64_t right_hash =
              keyed_hash(selection_seed, candidates[right].query_id.value());
          return std::tuple{left_hash, candidates[left].query_id.value()} <
                 std::tuple{right_hash, candidates[right].query_id.value()};
        });
    require(
        indexes.size() <=
            std::numeric_limits<std::uint64_t>::max() /
                requested_query_count,
        "shootout proportional allocation overflow");
    const std::uint64_t product =
        static_cast<std::uint64_t>(indexes.size()) * requested_query_count;
    Group group;
    group.stratum = stratum;
    group.indexes = std::move(indexes);
    group.allocation =
        static_cast<std::size_t>(product / candidates.size());
    group.remainder = product % candidates.size();
    allocated += group.allocation;
    groups.push_back(std::move(group));
  }

  require(
      allocated <= requested_query_count,
      "shootout proportional allocation exceeded its request");
  std::vector<std::size_t> remainder_order(groups.size());
  std::iota(remainder_order.begin(), remainder_order.end(), 0U);
  std::sort(
      remainder_order.begin(),
      remainder_order.end(),
      [&](const std::size_t left, const std::size_t right) {
        if (groups[left].remainder != groups[right].remainder) {
          return groups[left].remainder > groups[right].remainder;
        }
        return groups[left].stratum < groups[right].stratum;
      });
  std::size_t remaining = requested_query_count - allocated;
  for (const std::size_t group_index : remainder_order) {
    if (remaining == 0U) {
      break;
    }
    Group& group = groups[group_index];
    if (group.allocation < group.indexes.size()) {
      ++group.allocation;
      --remaining;
    }
  }
  require(remaining == 0U, "shootout largest-remainder allocation failed");

  ShootoutManifest manifest;
  manifest.fingerprint = fingerprint;
  manifest.workload = workload;
  manifest.selection_seed = selection_seed;
  manifest.order_seed = order_seed;
  manifest.warmup_repetitions = warmup_repetitions;
  manifest.timing_repetitions = timing_repetitions;
  manifest.requested_query_count = requested_query_count;
  manifest.entries.reserve(requested_query_count);
  for (const Group& group : groups) {
    for (std::size_t selected = 0U; selected < group.allocation; ++selected) {
      const std::size_t candidate_index = group.indexes[selected];
      manifest.entries.push_back(ShootoutManifestEntry{
          candidates[candidate_index], bins[candidate_index]});
    }
  }
  validate_manifest(manifest);
  return manifest;
}

ShootoutManifest make_synthetic_shootout_manifest(
    const ShootoutInputFingerprint& fingerprint,
    const ShootoutWorkloadIdentity& workload,
    const std::span<const ShootoutQueryFeatures> cases,
    const std::uint64_t order_seed,
    const std::uint32_t warmup_repetitions,
    const std::uint32_t timing_repetitions) {
  validate_fingerprint(fingerprint);
  validate_workload(workload);
  require(
      workload.kind == ShootoutWorkloadKind::synthetic,
      "synthetic manifest requires a synthetic workload identity");
  require(!cases.empty(), "synthetic shootout requires at least one case");
  require(
      cases.size() <= std::numeric_limits<std::uint32_t>::max(),
      "synthetic shootout case count exceeds 32-bit manifest storage");
  require(
      fingerprint.corpus_query_count >= cases.size(),
      "synthetic fingerprint corpus count is smaller than its cases");
  std::set<std::uint32_t> query_ids;
  for (const ShootoutQueryFeatures& features : cases) {
    validate_features(features);
    require(
        query_ids.insert(features.query_id.value()).second,
        "synthetic shootout contains a duplicate query ID");
  }
  const std::vector<Stratum> bins = feature_bins(cases);
  std::vector<std::size_t> order(cases.size());
  std::iota(order.begin(), order.end(), 0U);
  std::sort(
      order.begin(),
      order.end(),
      [&](const std::size_t left, const std::size_t right) {
        return cases[left].query_id < cases[right].query_id;
      });
  ShootoutManifest manifest;
  manifest.fingerprint = fingerprint;
  manifest.workload = workload;
  manifest.order_seed = order_seed;
  manifest.warmup_repetitions = warmup_repetitions;
  manifest.timing_repetitions = timing_repetitions;
  manifest.requested_query_count = static_cast<std::uint32_t>(cases.size());
  manifest.entries.reserve(cases.size());
  for (const std::size_t index : order) {
    manifest.entries.push_back(ShootoutManifestEntry{cases[index], bins[index]});
  }
  validate_manifest(manifest);
  return manifest;
}

std::vector<ShootoutScheduleEntry> make_interleaved_shootout_schedule(
    const ShootoutManifest& manifest,
    const std::span<const ShootoutTuning> tunings,
    const ShootoutRunKind run_kind,
    const std::uint32_t repetitions,
    const std::uint64_t order_seed) {
  validate_manifest(manifest);
  static_cast<void>(tuning_index(tunings));
  require(valid_run_kind(run_kind), "invalid shootout schedule run kind");
  require(repetitions != 0U, "shootout schedule requires repetitions");

  using CommonKey = std::tuple<
      ControlMode,
      std::uint32_t,
      std::uint32_t,
      GridPolicy,
      std::uint32_t,
      std::uint64_t>;
  std::map<CommonKey, std::vector<const ShootoutTuning*>> tuning_groups;
  for (const ShootoutTuning& tuning : tunings) {
    tuning_groups[CommonKey{
        tuning.control_mode,
        tuning.rounds_per_chunk,
        tuning.block_size,
        tuning.grid_policy,
        tuning.blocks_per_wgp,
        tuning.maximum_rounds}]
        .push_back(&tuning);
  }
  for (auto& [key, group] : tuning_groups) {
    static_cast<void>(key);
    std::sort(
        group.begin(),
        group.end(),
        [](const ShootoutTuning* const left,
           const ShootoutTuning* const right) {
          return left->engine < right->engine;
        });
    require(
        std::adjacent_find(
            group.begin(),
            group.end(),
            [](const ShootoutTuning* const left,
               const ShootoutTuning* const right) {
              return left->engine == right->engine;
            }) == group.end(),
        "shootout has duplicate engine tunings in one comparison cell");
  }

  struct Cell {
    std::size_t query{};
    std::size_t group{};
  };
  std::vector<const std::vector<const ShootoutTuning*>*> groups;
  for (const auto& [key, group] : tuning_groups) {
    static_cast<void>(key);
    groups.push_back(&group);
  }
  const std::uint64_t entries_per_repetition =
      static_cast<std::uint64_t>(manifest.entries.size()) * tunings.size();
  require(
      entries_per_repetition == 0U ||
          repetitions <=
              std::numeric_limits<std::size_t>::max() /
                  entries_per_repetition,
      "shootout schedule size overflow");
  std::vector<ShootoutScheduleEntry> schedule;
  schedule.reserve(
      static_cast<std::size_t>(entries_per_repetition * repetitions));
  std::uint64_t ordinal = 0U;
  for (std::uint32_t repetition = 0U; repetition < repetitions; ++repetition) {
    std::vector<Cell> cells;
    cells.reserve(manifest.entries.size() * groups.size());
    for (std::size_t query = 0U; query < manifest.entries.size(); ++query) {
      for (std::size_t group = 0U; group < groups.size(); ++group) {
        cells.push_back(Cell{query, group});
      }
    }
    std::sort(
        cells.begin(),
        cells.end(),
        [&](const Cell& left, const Cell& right) {
          const std::uint32_t left_query =
              manifest.entries[left.query].features.query_id.value();
          const std::uint32_t right_query =
              manifest.entries[right.query].features.query_id.value();
          return std::tuple{
                     keyed_hash(
                         order_seed,
                         static_cast<std::uint64_t>(repetition) << 32U |
                             left_query,
                         left.group),
                     left_query,
                     left.group} <
                 std::tuple{
                     keyed_hash(
                         order_seed,
                         static_cast<std::uint64_t>(repetition) << 32U |
                             right_query,
                         right.group),
                     right_query,
                     right.group};
        });
    for (const Cell& cell : cells) {
      const auto& group = *groups[cell.group];
      std::vector<std::size_t> engine_order;
      engine_order.reserve(group.size());
      if (group.size() == 3U) {
        constexpr std::array<std::array<std::size_t, 3U>, 6U> permutations{
            std::array<std::size_t, 3U>{0U, 1U, 2U},
            std::array<std::size_t, 3U>{0U, 2U, 1U},
            std::array<std::size_t, 3U>{1U, 0U, 2U},
            std::array<std::size_t, 3U>{1U, 2U, 0U},
            std::array<std::size_t, 3U>{2U, 0U, 1U},
            std::array<std::size_t, 3U>{2U, 1U, 0U},
        };
        const std::uint32_t query_id =
            manifest.entries[cell.query].features.query_id.value();
        const std::size_t phase = static_cast<std::size_t>(
            keyed_hash(order_seed, query_id, cell.group) % permutations.size());
        engine_order.assign(
            permutations[(phase + repetition) % permutations.size()].begin(),
            permutations[(phase + repetition) % permutations.size()].end());
      } else if (group.size() == 2U) {
        const std::uint32_t query_id =
            manifest.entries[cell.query].features.query_id.value();
        const bool reverse =
            ((keyed_hash(order_seed, query_id, cell.group) + repetition) & 1U) !=
            0U;
        engine_order = reverse ? std::vector<std::size_t>{1U, 0U}
                               : std::vector<std::size_t>{0U, 1U};
      } else {
        require(
            group.size() == 1U,
            "shootout comparison group has more than three engines");
        engine_order.push_back(0U);
      }
      for (std::size_t slot = 0U; slot < group.size(); ++slot) {
        const ShootoutTuning& tuning = *group[engine_order[slot]];
        schedule.push_back(ShootoutScheduleEntry{
            manifest.workload,
            run_kind,
            ordinal++,
            repetition,
            manifest.entries[cell.query].features.query_id,
            tuning.configuration_id,
        });
      }
    }
  }
  require(
      schedule.size() == entries_per_repetition * repetitions,
      "shootout interleaver omitted a tuning");
  return schedule;
}

std::vector<ShootoutScheduleEntry>
make_grouped_shootout_correctness_schedule(
    const ShootoutManifest& manifest,
    const std::span<const ShootoutTuning> tunings) {
  std::vector<ShootoutScheduleEntry> schedule =
      make_interleaved_shootout_schedule(
          manifest,
          tunings,
          ShootoutRunKind::correctness,
          1U,
          manifest.order_seed);
  std::sort(
      schedule.begin(),
      schedule.end(),
      [](const ShootoutScheduleEntry& left,
         const ShootoutScheduleEntry& right) {
        return std::tuple{left.query_id.value(), left.execution_ordinal} <
               std::tuple{right.query_id.value(), right.execution_ordinal};
      });
  for (std::size_t index = 0U; index < schedule.size(); ++index) {
    schedule[index].execution_ordinal = static_cast<std::uint64_t>(index);
  }
  return schedule;
}

void validate_shootout_configuration_catalog(
    const ShootoutConfigurationCatalog& catalog) {
  validate_fingerprint(catalog.fingerprint);
  validate_workload(catalog.workload);
  static_cast<void>(tuning_index(catalog.tunings));
}

void validate_shootout_samples(
    const ShootoutManifest& manifest,
    const std::span<const ShootoutTuning> tunings,
    const std::span<const ShootoutSample> samples,
    const ShootoutRunKind expected_kind) {
  validate_manifest(manifest);
  require(valid_run_kind(expected_kind), "invalid expected shootout run kind");
  const auto tuning_by_id = tuning_index(tunings);
  const auto features = feature_index(manifest);
  std::set<BasicSampleIdentity> basic_identities;
  std::set<ProfileSampleIdentity> profile_identities;
  std::set<std::uint64_t> ordinals;
  std::set<ProfileExecutionIdentity> profile_ordinals;
  std::map<std::uint64_t, std::uint64_t> pass_counter_sets;
  using MetricKey =
      std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::size_t>;
  std::map<MetricKey, std::uint64_t> metric_counter_sets;

  for (const ShootoutSample& sample : samples) {
    require(
        sample.fingerprint == manifest.fingerprint,
        "shootout sample fingerprint does not match its manifest");
    require(
        sample.workload == manifest.workload,
        "shootout sample workload does not match its manifest");
    require(
        sample.run_kind == expected_kind,
        "shootout stage mixes incompatible run kinds");
    require(
        features.contains(sample.query_id.value()),
        "shootout sample query is absent from its manifest");
    const auto tuning = tuning_by_id.find(sample.configuration_id);
    require(
        tuning != tuning_by_id.end(),
        "shootout sample configuration is absent from its catalog");
    if (expected_kind == ShootoutRunKind::trace ||
        expected_kind == ShootoutRunKind::pmc) {
      require(
          profile_identities.insert(profile_sample_identity(sample)).second,
          "shootout profiler stage contains a duplicate sample/pass");
      require(
          profile_ordinals
              .emplace(
                  sample.execution_ordinal,
                  sample.profiler_provenance.pass_id,
                  sample.profiler_provenance.counter_set_id)
              .second,
          "shootout profiler pass contains a duplicate execution ordinal");
    } else {
      require(
          basic_identities.insert(basic_sample_identity(sample)).second,
          "shootout stage contains a duplicate query/configuration/repetition");
      require(
          ordinals.insert(sample.execution_ordinal).second,
          "shootout stage contains a duplicate execution ordinal");
    }
    require(
        sample.result.engine_kind ==
                static_cast<std::uint32_t>(tuning->second->engine) &&
            sample.result.control_mode ==
                static_cast<std::uint32_t>(tuning->second->control_mode),
        "shootout result engine/control identity disagrees with its tuning");
    require(
        validate_device_run_status(sample.result.status) ==
            DeviceRunStatusError::none,
        "shootout sample contains an invalid device status");
    if (tuning->second->control_mode ==
        ControlMode::persistent_cooperative) {
      require(
          sample.cooperative_grid_blocks != 0U &&
              sample.cooperative_active_blocks_per_wgp != 0U,
          "persistent sample lacks actual cooperative launch dimensions");
    } else {
      require(
          sample.cooperative_grid_blocks == 0U &&
              sample.cooperative_active_blocks_per_wgp == 0U,
          "ordinary sample carries cooperative launch dimensions");
    }
    for_each_timing_value(sample.timing, validate_evidence_value);
    for_each_profiler_value(sample.profiler, validate_evidence_value);
    validate_percent(sample.profiler.l2_hit_percent);
    validate_percent(sample.profiler.occupancy_percent);
    validate_percent(sample.profiler.memory_unit_busy_percent);

    const bool no_provenance =
        sample.profiler_provenance.pass_id == 0U &&
        sample.profiler_provenance.counter_set_id == 0U &&
        !sample.profiler_provenance.compatible;
    switch (expected_kind) {
      case ShootoutRunKind::correctness:
        require(
            sample.repetition == 0U &&
                sample.instrumentation == InstrumentationLevel::none &&
                sample.distances_downloaded &&
                timing_is_unavailable(sample.timing) &&
                profiler_is_unavailable(sample.profiler) && no_provenance,
            "shootout correctness must download labels outside timing");
        if (sample.correctness_passed) {
          require(
              successful_status(sample.result.status),
              "passed correctness sample has a nonconverged status");
        }
        break;
      case ShootoutRunKind::timing:
        require(
            sample.instrumentation == InstrumentationLevel::none &&
                sample.correctness_passed && !sample.distances_downloaded &&
                sample.timing.preparation_gpu_milliseconds.state ==
                    ShootoutEvidenceState::measured &&
                sample.timing.sssp_device_timeline_milliseconds.state ==
                    ShootoutEvidenceState::measured &&
                sample.timing.result_transfer_gpu_milliseconds.state ==
                    ShootoutEvidenceState::measured &&
                sample.timing.end_to_end_wall_milliseconds.state ==
                    ShootoutEvidenceState::measured &&
                profiler_is_unavailable(sample.profiler) && no_provenance &&
                successful_status(sample.result.status),
            "shootout timing requires status-only None execution");
        break;
      case ShootoutRunKind::algorithm_counters:
        require(
            sample.instrumentation == InstrumentationLevel::debug &&
                sample.correctness_passed && !sample.distances_downloaded &&
                timing_is_unavailable(sample.timing) &&
                profiler_is_unavailable(sample.profiler) && no_provenance &&
                successful_status(sample.result.status),
            "shootout counter stage requires separate Debug execution");
        break;
      case ShootoutRunKind::warmup:
        require(
            sample.instrumentation == InstrumentationLevel::none &&
                !sample.correctness_passed && !sample.distances_downloaded &&
                timing_is_unavailable(sample.timing) &&
                profiler_is_unavailable(sample.profiler) && no_provenance &&
                successful_status(sample.result.status),
            "shootout warmup requires status-only None execution");
        break;
      case ShootoutRunKind::trace:
        require(
            sample.repetition == 0U &&
                sample.instrumentation == InstrumentationLevel::none &&
                sample.correctness_passed && !sample.distances_downloaded &&
                timing_is_unavailable(sample.timing) &&
                sample.profiler_provenance.pass_id != 0U &&
                sample.profiler_provenance.counter_set_id == 0U &&
                sample.profiler_provenance.compatible &&
                (sample.profiler.gpu_active_milliseconds.state ==
                     ShootoutEvidenceState::measured ||
                 sample.profiler.gpu_active_milliseconds.state ==
                     ShootoutEvidenceState::unavailable) &&
                hardware_profiler_is_unavailable(sample.profiler) &&
                successful_status(sample.result.status),
            "shootout trace sample has invalid stage/provenance evidence");
        break;
      case ShootoutRunKind::pmc:
        require(
            sample.repetition == 0U &&
                sample.instrumentation == InstrumentationLevel::none &&
                sample.correctness_passed && !sample.distances_downloaded &&
                timing_is_unavailable(sample.timing) &&
                sample.profiler_provenance.pass_id != 0U &&
                sample.profiler_provenance.counter_set_id != 0U &&
                sample.profiler_provenance.compatible &&
                sample.profiler.gpu_active_milliseconds.state ==
                    ShootoutEvidenceState::unavailable &&
                (any_hardware_profiler_measurement(sample.profiler) ||
                 profiler_is_unavailable(sample.profiler)) &&
                successful_status(sample.result.status),
            "shootout PMC sample has invalid stage/provenance evidence");
        {
          const auto [found, inserted] = pass_counter_sets.emplace(
              sample.profiler_provenance.pass_id,
              sample.profiler_provenance.counter_set_id);
          require(
              inserted ||
                  found->second == sample.profiler_provenance.counter_set_id,
              "one profiler pass ID names multiple counter sets");
          std::array<ShootoutEvidenceValue, 9U> metrics{
              sample.profiler.l2_hit_percent,
              sample.profiler.l2_read_bytes,
              sample.profiler.l2_write_bytes,
              sample.profiler.occupancy_percent,
              sample.profiler.memory_unit_busy_percent,
              sample.profiler.waves,
              sample.profiler.vector_instructions,
              sample.profiler.scalar_instructions,
              sample.profiler.memory_instructions,
          };
          for (std::size_t metric = 0U; metric < metrics.size(); ++metric) {
            if (metrics[metric].state != ShootoutEvidenceState::measured) {
              continue;
            }
            const MetricKey key{
                sample.query_id.value(),
                sample.configuration_id,
                sample.repetition,
                metric,
            };
            const auto [metric_found, metric_inserted] =
                metric_counter_sets.emplace(
                    key, sample.profiler_provenance.counter_set_id);
            require(
                metric_inserted ||
                    metric_found->second ==
                        sample.profiler_provenance.counter_set_id,
                "one profiler metric was supplied by incompatible counter sets");
          }
        }
        break;
    }
  }
  validate_rectangular_stage(manifest, tunings, samples, expected_kind);
  validate_schedule_binding(manifest, tunings, samples, expected_kind);
}

void require_complete_correctness_gate(
    const ShootoutManifest& manifest,
    const std::span<const ShootoutTuning> tunings,
    const std::span<const ShootoutSample> correctness_samples,
    const std::span<const ShootoutSample> gated_samples) {
  validate_shootout_samples(
      manifest,
      tunings,
      correctness_samples,
      ShootoutRunKind::correctness);
  const std::uint64_t required =
      static_cast<std::uint64_t>(manifest.entries.size()) * tunings.size();
  require(
      correctness_samples.size() == required,
      "shootout correctness gate is incomplete");
  std::set<std::pair<std::uint32_t, std::uint32_t>> passed;
  for (const ShootoutSample& sample : correctness_samples) {
    require(
        sample.correctness_passed,
        "shootout correctness failed before a gated stage");
    passed.emplace(sample.query_id.value(), sample.configuration_id);
  }
  require(passed.size() == required, "shootout correctness pairs are incomplete");

  if (gated_samples.empty()) {
    return;
  }
  const ShootoutRunKind kind = gated_samples.front().run_kind;
  require(
      kind != ShootoutRunKind::correctness && kind != ShootoutRunKind::warmup,
      "correctness/warmup cannot be used as a gated evidence stage");
  validate_shootout_samples(manifest, tunings, gated_samples, kind);
  for (const ShootoutSample& sample : gated_samples) {
    require(
        sample.correctness_passed && passed.contains(
            std::pair{sample.query_id.value(), sample.configuration_id}),
        "shootout sample has no matching passed correctness record");
  }
}

ShootoutDistribution shootout_distribution(
    const std::span<const double> values) {
  ShootoutDistribution result;
  result.count = values.size();
  if (values.empty()) {
    return result;
  }
  std::vector<double> sorted(values.begin(), values.end());
  for (const double value : sorted) {
    require(
        std::isfinite(value) && value >= 0.0,
        "shootout distributions require finite nonnegative values");
  }
  std::sort(sorted.begin(), sorted.end());
  const auto percentile = [&sorted](const double fraction) {
    const std::size_t rank = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(sorted.size())));
    return sorted[std::max<std::size_t>(1U, rank) - 1U];
  };
  result.minimum = sorted.front();
  result.p50 = percentile(0.50);
  result.p95 = percentile(0.95);
  result.p99 = percentile(0.99);
  result.maximum = sorted.back();
  result.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
                static_cast<double>(sorted.size());
  return result;
}

ShootoutCampaignReport summarize_shootout_campaign(
    const ShootoutManifest& manifest,
    const std::span<const ShootoutTuning> tunings,
    const std::span<const ShootoutSample> correctness_samples,
    const std::span<const ShootoutSample> timing_samples,
    const std::span<const ShootoutSample> counter_samples,
    const std::span<const ShootoutSample> trace_samples,
    const std::span<const ShootoutSample> pmc_samples) {
  validate_shootout_samples(
      manifest, tunings, timing_samples, ShootoutRunKind::timing);
  validate_shootout_samples(
      manifest,
      tunings,
      counter_samples,
      ShootoutRunKind::algorithm_counters);
  validate_shootout_samples(
      manifest, tunings, trace_samples, ShootoutRunKind::trace);
  validate_shootout_samples(
      manifest, tunings, pmc_samples, ShootoutRunKind::pmc);
  require(!timing_samples.empty(), "shootout report requires timing samples");
  require(!counter_samples.empty(), "shootout report requires counter samples");
  require_complete_correctness_gate(
      manifest, tunings, correctness_samples, timing_samples);
  require_complete_correctness_gate(
      manifest, tunings, correctness_samples, counter_samples);
  require_complete_correctness_gate(
      manifest, tunings, correctness_samples, trace_samples);
  require_complete_correctness_gate(
      manifest, tunings, correctness_samples, pmc_samples);

  const auto features = feature_index(manifest);
  ShootoutCampaignReport report;
  report.fingerprint = manifest.fingerprint;
  report.workload = manifest.workload;
  report.selection_seed = manifest.selection_seed;
  report.order_seed = manifest.order_seed;
  report.warmup_repetitions = manifest.warmup_repetitions;
  report.timing_repetitions = manifest.timing_repetitions;
  report.selected_query_count = manifest.requested_query_count;
  report.queries = manifest.entries;
  const auto append_evidence = [&report](
                                   const std::span<const ShootoutSample> stage) {
    for (const ShootoutSample& sample : stage) {
      report.supplied_evidence.push_back(evidence_reference(sample));
    }
  };
  append_evidence(correctness_samples);
  append_evidence(timing_samples);
  append_evidence(counter_samples);
  append_evidence(trace_samples);
  append_evidence(pmc_samples);
  report.summaries.reserve(tunings.size());
  for (const ShootoutTuning& tuning : tunings) {
    std::vector<const ShootoutSample*> timing;
    std::vector<const ShootoutSample*> counters;
    std::vector<ShootoutSample> trace;
    std::vector<ShootoutSample> pmc;
    std::vector<double> wall;
    std::vector<double> device;
    std::vector<double> gpu_active;
    for (const ShootoutSample& sample : timing_samples) {
      if (sample.configuration_id == tuning.configuration_id) {
        timing.push_back(&sample);
        wall.push_back(sample.timing.end_to_end_wall_milliseconds.value);
        device.push_back(
            sample.timing.sssp_device_timeline_milliseconds.value);
      }
    }
    for (const ShootoutSample& sample : counter_samples) {
      if (sample.configuration_id == tuning.configuration_id) {
        counters.push_back(&sample);
      }
    }
    for (const ShootoutSample& sample : trace_samples) {
      if (sample.configuration_id == tuning.configuration_id) {
        trace.push_back(sample);
        if (sample.profiler.gpu_active_milliseconds.state ==
            ShootoutEvidenceState::measured) {
          gpu_active.push_back(sample.profiler.gpu_active_milliseconds.value);
        }
      }
    }
    for (const ShootoutSample& sample : pmc_samples) {
      if (sample.configuration_id == tuning.configuration_id) {
        pmc.push_back(sample);
      }
    }

    ShootoutTuningSummary summary;
    summary.tuning = tuning;
    bool launch_seen = false;
    const auto collect_launch = [&](
                                    const std::span<const ShootoutSample> stage) {
      for (const ShootoutSample& sample : stage) {
        if (sample.configuration_id != tuning.configuration_id) {
          continue;
        }
        if (!launch_seen) {
          summary.cooperative_grid_blocks = sample.cooperative_grid_blocks;
          summary.cooperative_active_blocks_per_wgp =
              sample.cooperative_active_blocks_per_wgp;
          launch_seen = true;
        } else {
          require(
              summary.cooperative_grid_blocks ==
                      sample.cooperative_grid_blocks &&
                  summary.cooperative_active_blocks_per_wgp ==
                      sample.cooperative_active_blocks_per_wgp,
              "shootout stages disagree on cooperative launch dimensions");
        }
      }
    };
    collect_launch(correctness_samples);
    collect_launch(timing_samples);
    collect_launch(trace_samples);
    collect_launch(pmc_samples);
    require(launch_seen, "shootout summary tuning has no supplied samples");

    // Debug counter kernels may use more registers or dynamic shared memory
    // than their None counterparts. Their legal occupancy-derived grid can
    // therefore differ without changing the stable tuning identity used by
    // correctness and timing. Validate Debug rows against one another, but
    // never overwrite or compare the reported None launch dimensions with
    // Debug occupancy.
    bool counter_launch_seen = false;
    std::uint32_t counter_grid_blocks = 0U;
    std::uint32_t counter_active_blocks_per_wgp = 0U;
    for (const ShootoutSample& sample : counter_samples) {
      if (sample.configuration_id != tuning.configuration_id) {
        continue;
      }
      if (!counter_launch_seen) {
        counter_grid_blocks = sample.cooperative_grid_blocks;
        counter_active_blocks_per_wgp =
            sample.cooperative_active_blocks_per_wgp;
        counter_launch_seen = true;
      } else {
        require(
            counter_grid_blocks == sample.cooperative_grid_blocks &&
                counter_active_blocks_per_wgp ==
                    sample.cooperative_active_blocks_per_wgp,
            "Debug counter rows disagree on cooperative launch dimensions");
      }
    }
    summary.wall_milliseconds = shootout_distribution(wall);
    summary.device_timeline_milliseconds = shootout_distribution(device);
    summary.profiler.gpu_active_milliseconds =
        shootout_distribution(gpu_active);
    {
      std::set<std::uint64_t> passes;
      for (const ShootoutSample& sample : trace) {
        if (sample.profiler.gpu_active_milliseconds.state ==
            ShootoutEvidenceState::measured) {
          passes.insert(sample.profiler_provenance.pass_id);
        }
      }
      summary.profiler.gpu_active_pass_ids.assign(
          passes.begin(), passes.end());
    }
    const double total_wall = std::accumulate(wall.begin(), wall.end(), 0.0);
    if (!wall.empty() && total_wall > 0.0) {
      summary.throughput_queries_per_second = {
          ShootoutEvidenceState::measured,
          static_cast<double>(wall.size()) * 1000.0 / total_wall,
      };
    } else if (!wall.empty()) {
      summary.throughput_queries_per_second = {
          ShootoutEvidenceState::not_applicable, 0.0};
    }

    if (!counters.empty()) {
      summary.algorithm_counters_state = ShootoutEvidenceState::measured;
    }

    for (const ShootoutSample* const sample : counters) {
      const DeviceWorkStatistics& work = sample->result.work;
      checked_add(summary.total_rounds, sample->result.status.rounds_completed);
      checked_add(summary.total_edges_examined, work.edges_examined);
      checked_add(
          summary.total_successful_decreases, work.successful_decreases);
      checked_add(summary.total_active_vertices, work.active_vertices);
      checked_add(summary.total_active_lane_rounds, work.active_lane_rounds);
      summary.maximum_queue_size =
          std::max(summary.maximum_queue_size, work.maximum_queue_size);
      checked_add(summary.host_checks, work.host_checks);
      checked_add(summary.atomic_attempts, work.atomic_attempts);
      checked_add(
          summary.successful_atomic_updates,
          work.successful_atomic_updates);
      checked_add(summary.queue_claims, work.queue_claims);
      checked_add(
          summary.duplicate_suppressions, work.duplicate_suppressions);
      checked_add(summary.expansion_count, work.expansion_count);
      checked_add(summary.mask_operations, work.mask_operations);
      checked_add(summary.overflow_events, work.overflow_events);
      checked_add(
          summary.high_contention_destinations,
          work.high_contention_destinations);
      checked_add(summary.changed_flag_updates, work.changed_flag_updates);
      checked_add(summary.full_edge_rounds, work.full_edge_rounds);
      checked_add(summary.empty_frontier_rounds, work.empty_frontier_rounds);
      checked_add(summary.small_frontier_rounds, work.small_frontier_rounds);
      checked_add(summary.kernel_dispatches, work.kernel_dispatches);
      checked_add(
          summary.host_synchronizations, work.host_synchronizations);
      checked_add(summary.controller_copies, work.controller_copies);
    }
    summary.useful_decrease_ratio = evidence_ratio(
        summary.total_successful_decreases,
        summary.total_edges_examined,
        !counters.empty());

    summary.profiler.l2_hit_percent = summarize_profile_metric(
        pmc, &ShootoutProfilerRecord::l2_hit_percent);
    summary.profiler.l2_read_bytes = summarize_profile_metric(
        pmc, &ShootoutProfilerRecord::l2_read_bytes);
    summary.profiler.l2_write_bytes = summarize_profile_metric(
        pmc, &ShootoutProfilerRecord::l2_write_bytes);
    summary.profiler.occupancy_percent = summarize_profile_metric(
        pmc, &ShootoutProfilerRecord::occupancy_percent);
    summary.profiler.memory_unit_busy_percent = summarize_profile_metric(
        pmc, &ShootoutProfilerRecord::memory_unit_busy_percent);
    summary.profiler.waves =
        summarize_profile_metric(pmc, &ShootoutProfilerRecord::waves);
    summary.profiler.vector_instructions = summarize_profile_metric(
        pmc, &ShootoutProfilerRecord::vector_instructions);
    summary.profiler.scalar_instructions = summarize_profile_metric(
        pmc, &ShootoutProfilerRecord::scalar_instructions);
    summary.profiler.memory_instructions = summarize_profile_metric(
        pmc, &ShootoutProfilerRecord::memory_instructions);

    if (!timing.empty()) {
      const double threshold = summary.wall_milliseconds.p99;
      for (const ShootoutSample* const sample : timing) {
        const double value = sample->timing.end_to_end_wall_milliseconds.value;
        if (value >= threshold) {
          summary.long_tail.push_back(ShootoutLongTailRecord{
              manifest.workload,
              ShootoutTailMetric::wall,
              sample->query_id,
              sample->configuration_id,
              sample->repetition,
              value,
              features.at(sample->query_id.value()),
          });
        }
      }
    }
    if (!gpu_active.empty()) {
      const double threshold = summary.profiler.gpu_active_milliseconds.p99;
      for (const ShootoutSample& sample : trace) {
        if (sample.profiler.gpu_active_milliseconds.state !=
            ShootoutEvidenceState::measured) {
          continue;
        }
        const double value = sample.profiler.gpu_active_milliseconds.value;
        if (value >= threshold) {
          summary.long_tail.push_back(ShootoutLongTailRecord{
              manifest.workload,
              ShootoutTailMetric::gpu_active,
              sample.query_id,
              sample.configuration_id,
              sample.repetition,
              value,
              features.at(sample.query_id.value()),
          });
        }
      }
    }
    std::sort(
        summary.long_tail.begin(),
        summary.long_tail.end(),
        [](const ShootoutLongTailRecord& left,
           const ShootoutLongTailRecord& right) {
          return std::tuple{
                     left.metric,
                     -left.milliseconds,
                     left.query_id.value(),
                     left.repetition} <
                 std::tuple{
                     right.metric,
                     -right.milliseconds,
                     right.query_id.value(),
                     right.repetition};
        });
    if (summary.long_tail.size() > 50U) {
      summary.long_tail.resize(50U);
    }
    report.summaries.push_back(std::move(summary));
  }

  constexpr std::array<const char*, shootout_performance_question_count>
      pending_text{
          "pending measurement: persistent versus chunked",
          "pending measurement: best K per ordinary engine",
          "pending measurement: cooperative occupancy sensitivity",
          "pending measurement: dense pull versus sparse frontier regions",
          "pending measurement: dense chaotic push utility",
          "pending measurement: long-tail correlates",
          "pending measurement: eliminated per-round polling time",
      };
  for (std::size_t index = 0U; index < report.answers.size(); ++index) {
    report.answers[index].question_id = static_cast<std::uint32_t>(index + 1U);
    report.answers[index].state = ShootoutConclusionState::pending;
    report.answers[index].workload = manifest.workload;
    report.answers[index].conclusion = pending_text[index];
  }
  report.recommendations.push_back(ShootoutRecommendation{
      ShootoutConclusionState::pending,
      manifest.workload,
      {},
      {},
      "pending measured GPU campaign; every engine/control/K/grid/block toggle "
      "remains configurable",
      true,
  });
  validate_shootout_report(report);
  return report;
}

void validate_shootout_report(const ShootoutCampaignReport& report) {
  validate_fingerprint(report.fingerprint);
  validate_workload(report.workload);
  require(
      report.warmup_repetitions != 0U && report.timing_repetitions != 0U &&
          report.selected_query_count != 0U &&
          report.queries.size() == report.selected_query_count,
      "shootout report has invalid campaign/query metadata");
  if (report.workload.kind == ShootoutWorkloadKind::logicnets_jscl) {
    require(
        report.selected_query_count >= minimum_logicnets_shootout_queries,
        "logicnets report contains fewer than 1000 selected queries");
  }
  std::map<std::uint32_t, ShootoutManifestEntry> report_queries;
  for (const ShootoutManifestEntry& entry : report.queries) {
    validate_features(entry.features);
    require(
        std::ranges::all_of(entry.quantile_bins, [](const std::uint8_t bin) {
          return bin <= 3U;
        }) &&
            report_queries
                .emplace(entry.features.query_id.value(), entry)
                .second,
        "shootout report has invalid or duplicate query strata");
  }
  using EvidenceKey = std::tuple<
      ShootoutRunKind,
      std::uint64_t,
      std::uint64_t,
      std::uint64_t,
      std::uint32_t,
      std::uint32_t>;
  const auto evidence_key = [](const ShootoutEvidenceReference& reference) {
    return EvidenceKey{
        reference.run_kind,
        reference.execution_ordinal,
        reference.profiler_pass_id,
        reference.profiler_counter_set_id,
        reference.query_id.value(),
        reference.configuration_id};
  };
  std::set<EvidenceKey> supplied_evidence;
  for (const ShootoutEvidenceReference& reference : report.supplied_evidence) {
    validate_evidence_reference(reference);
    require(
        report_queries.contains(reference.query_id.value()),
        "shootout supplied evidence names an unknown query");
    require(
        supplied_evidence.insert(evidence_key(reference)).second,
        "shootout report contains duplicate supplied evidence IDs");
  }
  const auto require_supplied = [&](
                                    const std::span<const ShootoutEvidenceReference>
                                        references) {
    for (const ShootoutEvidenceReference& reference : references) {
      validate_evidence_reference(reference);
      require(
          supplied_evidence.contains(evidence_key(reference)),
          "shootout conclusion references evidence absent from its campaign");
    }
  };
  std::set<std::uint32_t> configurations;
  for (const ShootoutTuningSummary& summary : report.summaries) {
    require(valid_tuning(summary.tuning), "shootout report has invalid tuning");
    require(
        configurations.insert(summary.tuning.configuration_id).second,
        "shootout report has duplicate tuning summaries");
    if (summary.tuning.control_mode ==
        ControlMode::persistent_cooperative) {
      require(
          summary.cooperative_grid_blocks != 0U &&
              summary.cooperative_active_blocks_per_wgp != 0U,
          "persistent report summary lacks cooperative launch dimensions");
    } else {
      require(
          summary.cooperative_grid_blocks == 0U &&
              summary.cooperative_active_blocks_per_wgp == 0U,
          "ordinary report summary carries cooperative launch dimensions");
    }
    validate_evidence_value(summary.throughput_queries_per_second);
    validate_evidence_value(summary.useful_decrease_ratio);
    require(
        summary.algorithm_counters_state == ShootoutEvidenceState::measured ||
            summary.algorithm_counters_state ==
                ShootoutEvidenceState::unavailable,
        "shootout counter summary has an invalid availability state");
    if (summary.algorithm_counters_state ==
        ShootoutEvidenceState::unavailable) {
      require(
          summary.total_rounds == 0U &&
              summary.total_edges_examined == 0U &&
              summary.total_successful_decreases == 0U &&
              summary.useful_decrease_ratio.state ==
                  ShootoutEvidenceState::unavailable &&
              summary.total_active_vertices == 0U &&
              summary.total_active_lane_rounds == 0U &&
              summary.maximum_queue_size == 0U && summary.host_checks == 0U &&
              summary.atomic_attempts == 0U &&
              summary.successful_atomic_updates == 0U &&
              summary.queue_claims == 0U &&
              summary.duplicate_suppressions == 0U &&
              summary.expansion_count == 0U &&
              summary.mask_operations == 0U &&
              summary.overflow_events == 0U &&
              summary.high_contention_destinations == 0U &&
              summary.changed_flag_updates == 0U &&
              summary.full_edge_rounds == 0U &&
              summary.empty_frontier_rounds == 0U &&
              summary.small_frontier_rounds == 0U &&
              summary.kernel_dispatches == 0U &&
              summary.host_synchronizations == 0U &&
              summary.controller_copies == 0U,
          "unavailable counter summary contains manufactured values");
    }
    const std::array<const ShootoutProfileMetricSummary*, 9U> metrics{
        &summary.profiler.l2_hit_percent,
        &summary.profiler.l2_read_bytes,
        &summary.profiler.l2_write_bytes,
        &summary.profiler.occupancy_percent,
        &summary.profiler.memory_unit_busy_percent,
        &summary.profiler.waves,
        &summary.profiler.vector_instructions,
        &summary.profiler.scalar_instructions,
        &summary.profiler.memory_instructions,
    };
    for (const ShootoutProfileMetricSummary* const metric : metrics) {
      validate_evidence_value(metric->mean);
      if (metric->mean.state == ShootoutEvidenceState::measured) {
        require(
            metric->counter_set_id != 0U && !metric->pass_ids.empty(),
            "measured profiler summary lacks pass/counter-set provenance");
      } else {
        require(
            metric->counter_set_id == 0U && metric->pass_ids.empty(),
            "unmeasured profiler summary carries provenance");
      }
    }
    for (const ShootoutLongTailRecord& tail : summary.long_tail) {
      const auto query = report_queries.find(tail.query_id.value());
      require(
          tail.workload == report.workload &&
              tail.configuration_id == summary.tuning.configuration_id &&
              std::isfinite(tail.milliseconds) && tail.milliseconds >= 0.0 &&
              query != report_queries.end() &&
              tail.features == query->second.features,
          "shootout long-tail record is inconsistent");
    }
  }

  for (const ShootoutEvidenceReference& reference : report.supplied_evidence) {
    require(
        configurations.contains(reference.configuration_id),
        "shootout supplied evidence names an unknown configuration");
  }

  std::set<std::uint32_t> question_ids;
  for (const ShootoutQuestionAnswer& answer : report.answers) {
    require(
        answer.question_id >= 1U &&
            answer.question_id <= shootout_performance_question_count &&
            question_ids.insert(answer.question_id).second,
        "shootout report has an invalid or duplicate question ID");
    require(
        valid_conclusion_state(answer.state) &&
            answer.workload == report.workload && !answer.conclusion.empty(),
        "shootout answer is incomplete or cross-workload");
    for (const std::uint32_t configuration : answer.configuration_ids) {
      require(
          configurations.contains(configuration),
          "shootout answer names an unknown configuration");
    }
    for (const ShootoutEvidenceReference& reference : answer.evidence) {
      require(
          std::ranges::find(
              answer.configuration_ids, reference.configuration_id) !=
              answer.configuration_ids.end(),
          "shootout answer cites evidence from an unnamed configuration");
    }
    if (answer.state == ShootoutConclusionState::measured) {
      require(
          !answer.configuration_ids.empty() && !answer.evidence.empty(),
          "measured shootout answer lacks configurations/evidence");
      validate_measured_evidence_mix(answer.evidence);
      require_supplied(answer.evidence);
    } else if (answer.state == ShootoutConclusionState::pending) {
      require(
          answer.configuration_ids.empty() && answer.evidence.empty(),
          "pending shootout answer carries premature evidence");
    } else {
      require_supplied(answer.evidence);
    }
  }
  require(
      question_ids.size() == shootout_performance_question_count,
      "shootout report does not contain all seven questions");

  require(
      !report.recommendations.empty(),
      "shootout report requires a configurable recommendation record");
  for (const ShootoutRecommendation& recommendation : report.recommendations) {
    require(
        valid_conclusion_state(recommendation.state) &&
            recommendation.workload == report.workload &&
            !recommendation.rationale.empty() &&
            recommendation.toggles_remain_configurable,
        "shootout recommendation is incomplete or disables a toggle");
    for (const std::uint32_t configuration :
         recommendation.configuration_ids) {
      require(
          configurations.contains(configuration),
          "shootout recommendation names an unknown configuration");
    }
    for (const ShootoutEvidenceReference& reference :
         recommendation.evidence) {
      require(
          std::ranges::find(
              recommendation.configuration_ids,
              reference.configuration_id) !=
              recommendation.configuration_ids.end(),
          "recommendation cites evidence from an unnamed configuration");
    }
    if (recommendation.state == ShootoutConclusionState::measured) {
      require(
          !recommendation.configuration_ids.empty() &&
              !recommendation.evidence.empty(),
          "measured recommendation lacks configurations/evidence");
      validate_measured_evidence_mix(recommendation.evidence);
      require_supplied(recommendation.evidence);
    } else if (
        recommendation.state == ShootoutConclusionState::pending) {
      require(
          recommendation.configuration_ids.empty() &&
              recommendation.evidence.empty(),
          "pending recommendation carries premature evidence");
    } else {
      require_supplied(recommendation.evidence);
    }
  }
}

namespace {

constexpr std::string_view manifest_tsv_header =
    "schema\tworkload_kind\tcase_id\tcase_name\tgraph0\tgraph1\tquery0\t"
    "query1\tcorpus_queries\tselection_seed\torder_seed\twarmup_repetitions\t"
    "timing_repetitions\trequested\tselection_ordinal\tquery_id\t"
    "selected_vertices\tselected_edges\tfanout\tsource_count\texpected_rounds\t"
    "vertex_bin\tedge_bin\tfanout_bin\tsource_bin\tround_bin";

constexpr std::string_view catalog_tsv_header =
    "schema\tworkload_kind\tcase_id\tcase_name\tgraph0\tgraph1\tquery0\t"
    "query1\tcorpus_queries\tconfiguration_id\tengine\tcontrol\t"
    "rounds_per_chunk\tblock_size\tgrid_policy\tblocks_per_wgp\tmaximum_rounds";

[[nodiscard]] ShootoutInputFingerprint parse_fingerprint_prefix(
    const std::vector<std::string_view>& fields) {
  require(fields.size() >= 9U, "truncated shootout TSV metadata");
  ShootoutInputFingerprint fingerprint;
  fingerprint.schema_version =
      parse_unsigned<std::uint32_t>(fields[0], "invalid shootout schema");
  fingerprint.graph_words[0] =
      parse_unsigned<std::uint64_t>(fields[4], "invalid graph fingerprint");
  fingerprint.graph_words[1] =
      parse_unsigned<std::uint64_t>(fields[5], "invalid graph fingerprint");
  fingerprint.query_words[0] =
      parse_unsigned<std::uint64_t>(fields[6], "invalid query fingerprint");
  fingerprint.query_words[1] =
      parse_unsigned<std::uint64_t>(fields[7], "invalid query fingerprint");
  fingerprint.corpus_query_count =
      parse_unsigned<std::uint64_t>(fields[8], "invalid corpus count");
  validate_fingerprint(fingerprint);
  return fingerprint;
}

[[nodiscard]] ShootoutWorkloadIdentity parse_workload_prefix(
    const std::vector<std::string_view>& fields) {
  require(fields.size() >= 4U, "truncated shootout workload metadata");
  ShootoutWorkloadIdentity workload;
  workload.kind = static_cast<ShootoutWorkloadKind>(
      parse_unsigned<std::uint32_t>(fields[1], "invalid workload kind"));
  workload.case_id =
      parse_unsigned<std::uint64_t>(fields[2], "invalid workload case ID");
  workload.case_name = std::string{fields[3]};
  validate_workload(workload);
  return workload;
}

void write_metadata_prefix(
    std::ostream& output,
    const ShootoutInputFingerprint& fingerprint,
    const ShootoutWorkloadIdentity& workload) {
  output << fingerprint.schema_version << '\t'
         << static_cast<std::uint32_t>(workload.kind) << '\t'
         << workload.case_id << '\t' << workload.case_name << '\t'
         << fingerprint.graph_words[0] << '\t' << fingerprint.graph_words[1]
         << '\t' << fingerprint.query_words[0] << '\t'
         << fingerprint.query_words[1] << '\t'
         << fingerprint.corpus_query_count;
}

}  // namespace

std::string serialize_shootout_manifest_tsv(const ShootoutManifest& manifest) {
  validate_manifest(manifest);
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << manifest_tsv_header << '\n';
  for (std::size_t index = 0U; index < manifest.entries.size(); ++index) {
    const ShootoutManifestEntry& entry = manifest.entries[index];
    write_metadata_prefix(output, manifest.fingerprint, manifest.workload);
    output << '\t' << manifest.selection_seed << '\t' << manifest.order_seed
           << '\t' << manifest.warmup_repetitions << '\t'
           << manifest.timing_repetitions << '\t'
           << manifest.requested_query_count << '\t' << index << '\t'
           << entry.features.query_id.value() << '\t'
           << entry.features.selected_vertices << '\t'
           << entry.features.selected_edges << '\t' << entry.features.fanout
           << '\t' << entry.features.source_count << '\t'
           << entry.features.expected_rounds;
    for (const std::uint8_t bin : entry.quantile_bins) {
      output << '\t' << static_cast<std::uint32_t>(bin);
    }
    output << '\n';
  }
  return output.str();
}

ShootoutManifest deserialize_shootout_manifest_tsv(
    const std::string_view text) {
  const std::vector<std::string_view> lines =
      data_lines(text, manifest_tsv_header);
  ShootoutManifest manifest;
  for (std::size_t line_index = 0U; line_index < lines.size(); ++line_index) {
    const std::vector<std::string_view> fields = split_fields(lines[line_index]);
    require(fields.size() == 26U, "manifest TSV row has wrong field count");
    const ShootoutInputFingerprint fingerprint =
        parse_fingerprint_prefix(fields);
    const ShootoutWorkloadIdentity workload = parse_workload_prefix(fields);
    const std::uint64_t seed =
        parse_unsigned<std::uint64_t>(fields[9], "invalid manifest seed");
    const std::uint64_t order_seed =
        parse_unsigned<std::uint64_t>(fields[10], "invalid manifest order seed");
    const std::uint32_t warmups = parse_unsigned<std::uint32_t>(
        fields[11], "invalid manifest warmup count");
    const std::uint32_t timing_repetitions = parse_unsigned<std::uint32_t>(
        fields[12], "invalid manifest timing count");
    const std::uint32_t requested =
        parse_unsigned<std::uint32_t>(fields[13], "invalid requested count");
    const std::size_t ordinal =
        parse_unsigned<std::size_t>(fields[14], "invalid manifest ordinal");
    require(ordinal == line_index, "manifest TSV ordinal is not canonical");
    if (line_index == 0U) {
      manifest.fingerprint = fingerprint;
      manifest.workload = workload;
      manifest.selection_seed = seed;
      manifest.order_seed = order_seed;
      manifest.warmup_repetitions = warmups;
      manifest.timing_repetitions = timing_repetitions;
      manifest.requested_query_count = requested;
      manifest.entries.reserve(requested);
    } else {
      require(
          fingerprint == manifest.fingerprint && workload == manifest.workload &&
              seed == manifest.selection_seed &&
              order_seed == manifest.order_seed &&
              warmups == manifest.warmup_repetitions &&
              timing_repetitions == manifest.timing_repetitions &&
              requested == manifest.requested_query_count,
          "manifest TSV mixes metadata or workloads");
    }
    ShootoutManifestEntry entry;
    entry.features.query_id = QueryId{parse_unsigned<std::uint32_t>(
        fields[15], "invalid manifest query ID")};
    entry.features.selected_vertices = parse_unsigned<std::uint64_t>(
        fields[16], "invalid selected vertex count");
    entry.features.selected_edges = parse_unsigned<std::uint64_t>(
        fields[17], "invalid selected edge count");
    entry.features.fanout =
        parse_unsigned<std::uint32_t>(fields[18], "invalid fanout");
    entry.features.source_count =
        parse_unsigned<std::uint32_t>(fields[19], "invalid source count");
    entry.features.expected_rounds =
        parse_unsigned<std::uint64_t>(fields[20], "invalid expected rounds");
    for (std::size_t bin = 0U; bin < entry.quantile_bins.size(); ++bin) {
      entry.quantile_bins[bin] = static_cast<std::uint8_t>(
          parse_unsigned<std::uint32_t>(fields[21U + bin], "invalid quantile bin"));
    }
    manifest.entries.push_back(entry);
  }
  validate_manifest(manifest);
  return manifest;
}

std::string serialize_shootout_catalog_tsv(
    const ShootoutConfigurationCatalog& catalog) {
  validate_shootout_configuration_catalog(catalog);
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << catalog_tsv_header << '\n';
  std::vector<const ShootoutTuning*> ordered;
  ordered.reserve(catalog.tunings.size());
  for (const ShootoutTuning& tuning : catalog.tunings) {
    ordered.push_back(&tuning);
  }
  std::sort(
      ordered.begin(),
      ordered.end(),
      [](const ShootoutTuning* const left,
         const ShootoutTuning* const right) {
        return left->configuration_id < right->configuration_id;
      });
  for (const ShootoutTuning* const tuning : ordered) {
    write_metadata_prefix(output, catalog.fingerprint, catalog.workload);
    output << '\t' << tuning->configuration_id << '\t'
           << static_cast<std::uint32_t>(tuning->engine) << '\t'
           << static_cast<std::uint32_t>(tuning->control_mode) << '\t'
           << tuning->rounds_per_chunk << '\t' << tuning->block_size << '\t'
           << static_cast<std::uint32_t>(tuning->grid_policy) << '\t'
           << tuning->blocks_per_wgp << '\t' << tuning->maximum_rounds << '\n';
  }
  return output.str();
}

ShootoutConfigurationCatalog deserialize_shootout_catalog_tsv(
    const std::string_view text) {
  const std::vector<std::string_view> lines =
      data_lines(text, catalog_tsv_header);
  ShootoutConfigurationCatalog catalog;
  for (std::size_t index = 0U; index < lines.size(); ++index) {
    const std::vector<std::string_view> fields = split_fields(lines[index]);
    require(fields.size() == 17U, "catalog TSV row has wrong field count");
    const ShootoutInputFingerprint fingerprint =
        parse_fingerprint_prefix(fields);
    const ShootoutWorkloadIdentity workload = parse_workload_prefix(fields);
    if (index == 0U) {
      catalog.fingerprint = fingerprint;
      catalog.workload = workload;
      catalog.tunings.reserve(lines.size());
    } else {
      require(
          fingerprint == catalog.fingerprint && workload == catalog.workload,
          "catalog TSV mixes metadata or workloads");
    }
    ShootoutTuning tuning;
    tuning.configuration_id =
        parse_unsigned<std::uint32_t>(fields[9], "invalid configuration ID");
    tuning.engine = static_cast<EngineKind>(
        parse_unsigned<std::uint32_t>(fields[10], "invalid engine"));
    tuning.control_mode = static_cast<ControlMode>(
        parse_unsigned<std::uint32_t>(fields[11], "invalid control mode"));
    tuning.rounds_per_chunk = parse_unsigned<std::uint32_t>(
        fields[12], "invalid rounds per chunk");
    tuning.block_size =
        parse_unsigned<std::uint32_t>(fields[13], "invalid block size");
    tuning.grid_policy = static_cast<GridPolicy>(
        parse_unsigned<std::uint32_t>(fields[14], "invalid grid policy"));
    tuning.blocks_per_wgp =
        parse_unsigned<std::uint32_t>(fields[15], "invalid blocks per WGP");
    tuning.maximum_rounds =
        parse_unsigned<std::uint64_t>(fields[16], "invalid maximum rounds");
    catalog.tunings.push_back(tuning);
  }
  validate_shootout_configuration_catalog(catalog);
  return catalog;
}

namespace {

constexpr std::string_view samples_tsv_header =
    "schema\tworkload_kind\tcase_id\tcase_name\tgraph0\tgraph1\tquery0\t"
    "query1\tcorpus_queries\tselection_seed\torder_seed\twarmup_repetitions\t"
    "timing_repetitions\trun_kind\texecution_ordinal\trepetition\tquery_id\t"
    "configuration_id\tinstrumentation\tcorrectness_passed\tdistances_downloaded\t"
    "engine\tcontrol\tfinal_distance_slot\tconverged\trounds_completed\t"
    "reached_target_mask\tbounding_box_miss_mask\tvalid_lane_mask\tactive_lane_mask\t"
    "converged_lane_mask\tstop_reason\terror_bits\tedges_examined\t"
    "successful_decreases\tactive_vertices\tactive_lane_rounds\tmaximum_queue_size\t"
    "host_checks\thost_synchronizations\tcontroller_copies\tkernel_dispatches\t"
    "expansion_count\tatomic_attempts\tsuccessful_atomic_updates\tqueue_claims\t"
    "duplicate_suppressions\tmask_operations\toverflow_events\t"
    "high_contention_destinations\tchanged_flag_updates\tfull_edge_rounds\t"
    "empty_frontier_rounds\tsmall_frontier_rounds\tprep_state\tprep_value\t"
    "timeline_state\ttimeline_value\ttransfer_state\ttransfer_value\twall_state\t"
    "wall_value\tprofiler_pass_id\tcounter_set_id\tprofiler_compatible\t"
    "gpu_active_state\tgpu_active_value\tl2_hit_state\tl2_hit_value\t"
    "l2_read_state\tl2_read_value\tl2_write_state\tl2_write_value\t"
    "occupancy_state\toccupancy_value\tmemory_busy_state\tmemory_busy_value\t"
    "waves_state\twaves_value\tvector_state\tvector_value\tscalar_state\t"
    "scalar_value\tmemory_inst_state\tmemory_inst_value\tcooperative_grid_blocks\t"
    "cooperative_active_blocks_per_wgp";

void write_status(std::ostream& output, const DeviceRunStatus& status) {
  output << '\t' << status.final_distance_slot << '\t' << status.converged
         << '\t' << status.rounds_completed << '\t'
         << status.reached_target_mask << '\t' << status.bounding_box_miss_mask
         << '\t' << status.valid_lane_mask << '\t' << status.active_lane_mask
         << '\t' << status.converged_lane_mask << '\t' << status.stop_reason
         << '\t' << status.error_bits;
}

void write_work(std::ostream& output, const DeviceWorkStatistics& work) {
  output << '\t' << work.edges_examined << '\t' << work.successful_decreases
         << '\t' << work.active_vertices << '\t' << work.active_lane_rounds
         << '\t' << work.maximum_queue_size << '\t' << work.host_checks
         << '\t' << work.host_synchronizations << '\t' << work.controller_copies
         << '\t' << work.kernel_dispatches << '\t' << work.expansion_count
         << '\t' << work.atomic_attempts << '\t'
         << work.successful_atomic_updates << '\t' << work.queue_claims << '\t'
         << work.duplicate_suppressions << '\t' << work.mask_operations << '\t'
         << work.overflow_events << '\t' << work.high_contention_destinations
         << '\t' << work.changed_flag_updates << '\t' << work.full_edge_rounds
         << '\t' << work.empty_frontier_rounds << '\t'
         << work.small_frontier_rounds;
}

[[nodiscard]] DeviceRunStatus parse_status(
    const std::vector<std::string_view>& fields,
    std::size_t& position) {
  require(position + 9U < fields.size(), "truncated device status in samples TSV");
  DeviceRunStatus status;
  status.final_distance_slot = parse_unsigned<std::uint32_t>(
      fields[position++], "invalid final distance slot");
  status.converged =
      parse_unsigned<std::uint32_t>(fields[position++], "invalid converged flag");
  status.rounds_completed = parse_unsigned<std::uint64_t>(
      fields[position++], "invalid completed rounds");
  status.reached_target_mask =
      parse_unsigned<LaneMask>(fields[position++], "invalid reached mask");
  status.bounding_box_miss_mask =
      parse_unsigned<LaneMask>(fields[position++], "invalid miss mask");
  status.valid_lane_mask =
      parse_unsigned<LaneMask>(fields[position++], "invalid valid mask");
  status.active_lane_mask =
      parse_unsigned<LaneMask>(fields[position++], "invalid active mask");
  status.converged_lane_mask =
      parse_unsigned<LaneMask>(fields[position++], "invalid converged mask");
  status.stop_reason =
      parse_unsigned<std::uint32_t>(fields[position++], "invalid stop reason");
  status.error_bits =
      parse_unsigned<std::uint32_t>(fields[position++], "invalid error bits");
  return status;
}

[[nodiscard]] DeviceWorkStatistics parse_work(
    const std::vector<std::string_view>& fields,
    std::size_t& position) {
  require(position + 20U < fields.size(), "truncated work record in samples TSV");
  DeviceWorkStatistics work;
  const auto next = [&fields, &position](const char* const message) {
    return parse_unsigned<std::uint64_t>(fields[position++], message);
  };
  work.edges_examined = next("invalid edges examined");
  work.successful_decreases = next("invalid successful decreases");
  work.active_vertices = next("invalid active vertices");
  work.active_lane_rounds = next("invalid active lane rounds");
  work.maximum_queue_size = next("invalid maximum queue size");
  work.host_checks = next("invalid host checks");
  work.host_synchronizations = next("invalid host synchronizations");
  work.controller_copies = next("invalid controller copies");
  work.kernel_dispatches = next("invalid kernel dispatches");
  work.expansion_count = next("invalid expansion count");
  work.atomic_attempts = next("invalid atomic attempts");
  work.successful_atomic_updates = next("invalid successful atomic updates");
  work.queue_claims = next("invalid queue claims");
  work.duplicate_suppressions = next("invalid duplicate suppressions");
  work.mask_operations = next("invalid mask operations");
  work.overflow_events = next("invalid overflow events");
  work.high_contention_destinations = next("invalid contention destinations");
  work.changed_flag_updates = next("invalid changed flag updates");
  work.full_edge_rounds = next("invalid full-edge rounds");
  work.empty_frontier_rounds = next("invalid empty frontier rounds");
  work.small_frontier_rounds = next("invalid small frontier rounds");
  return work;
}

}  // namespace

std::string serialize_shootout_samples_tsv(
    const ShootoutManifest& manifest,
    const std::span<const ShootoutTuning> tunings,
    const std::span<const ShootoutSample> samples,
    const ShootoutRunKind run_kind) {
  validate_shootout_samples(manifest, tunings, samples, run_kind);
  std::vector<const ShootoutSample*> ordered;
  ordered.reserve(samples.size());
  for (const ShootoutSample& sample : samples) {
    ordered.push_back(&sample);
  }
  std::sort(
      ordered.begin(),
      ordered.end(),
      [](const ShootoutSample* const left,
         const ShootoutSample* const right) {
        return std::tuple{
                   left->execution_ordinal,
                   left->query_id.value(),
                   left->configuration_id,
                   left->repetition,
                   left->profiler_provenance.pass_id,
                   left->profiler_provenance.counter_set_id} <
               std::tuple{
                   right->execution_ordinal,
                   right->query_id.value(),
                   right->configuration_id,
                   right->repetition,
                   right->profiler_provenance.pass_id,
                   right->profiler_provenance.counter_set_id};
      });
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << samples_tsv_header << '\n';
  for (const ShootoutSample* const sample : ordered) {
    write_metadata_prefix(output, sample->fingerprint, sample->workload);
    output << '\t' << manifest.selection_seed << '\t' << manifest.order_seed
           << '\t' << manifest.warmup_repetitions << '\t'
           << manifest.timing_repetitions << '\t'
           << static_cast<std::uint32_t>(sample->run_kind) << '\t'
           << sample->execution_ordinal << '\t' << sample->repetition << '\t'
           << sample->query_id.value() << '\t' << sample->configuration_id
           << '\t' << static_cast<std::uint32_t>(sample->instrumentation)
           << '\t' << (sample->correctness_passed ? 1U : 0U) << '\t'
           << (sample->distances_downloaded ? 1U : 0U) << '\t'
           << sample->result.engine_kind << '\t' << sample->result.control_mode;
    write_status(output, sample->result.status);
    write_work(output, sample->result.work);
    append_evidence(output, sample->timing.preparation_gpu_milliseconds);
    append_evidence(output, sample->timing.sssp_device_timeline_milliseconds);
    append_evidence(output, sample->timing.result_transfer_gpu_milliseconds);
    append_evidence(output, sample->timing.end_to_end_wall_milliseconds);
    output << '\t' << sample->profiler_provenance.pass_id << '\t'
           << sample->profiler_provenance.counter_set_id << '\t'
           << (sample->profiler_provenance.compatible ? 1U : 0U);
    append_evidence(output, sample->profiler.gpu_active_milliseconds);
    append_evidence(output, sample->profiler.l2_hit_percent);
    append_evidence(output, sample->profiler.l2_read_bytes);
    append_evidence(output, sample->profiler.l2_write_bytes);
    append_evidence(output, sample->profiler.occupancy_percent);
    append_evidence(output, sample->profiler.memory_unit_busy_percent);
    append_evidence(output, sample->profiler.waves);
    append_evidence(output, sample->profiler.vector_instructions);
    append_evidence(output, sample->profiler.scalar_instructions);
    append_evidence(output, sample->profiler.memory_instructions);
    output << '\t' << sample->cooperative_grid_blocks << '\t'
           << sample->cooperative_active_blocks_per_wgp << '\n';
  }
  return output.str();
}

std::vector<ShootoutSample> deserialize_shootout_samples_tsv(
    const ShootoutManifest& manifest,
    const std::span<const ShootoutTuning> tunings,
    const std::string_view text,
    const ShootoutRunKind run_kind) {
  validate_manifest(manifest);
  static_cast<void>(tuning_index(tunings));
  const std::vector<std::string_view> lines =
      data_lines(text, samples_tsv_header);
  std::vector<ShootoutSample> samples;
  samples.reserve(lines.size());
  for (const std::string_view line : lines) {
    const std::vector<std::string_view> fields = split_fields(line);
    require(fields.size() == 87U, "samples TSV row has wrong field count");
    ShootoutSample sample;
    sample.fingerprint = parse_fingerprint_prefix(fields);
    sample.workload = parse_workload_prefix(fields);
    std::size_t position = 9U;
    const std::uint64_t selection_seed = parse_unsigned<std::uint64_t>(
        fields[position++], "invalid sample selection seed");
    const std::uint64_t order_seed = parse_unsigned<std::uint64_t>(
        fields[position++], "invalid sample order seed");
    const std::uint32_t warmups = parse_unsigned<std::uint32_t>(
        fields[position++], "invalid sample warmup count");
    const std::uint32_t timing_repetitions = parse_unsigned<std::uint32_t>(
        fields[position++], "invalid sample timing count");
    require(
        selection_seed == manifest.selection_seed &&
            order_seed == manifest.order_seed &&
            warmups == manifest.warmup_repetitions &&
            timing_repetitions == manifest.timing_repetitions,
        "samples TSV campaign seeds/warmups do not match its manifest");
    sample.run_kind = static_cast<ShootoutRunKind>(
        parse_unsigned<std::uint32_t>(fields[position++], "invalid run kind"));
    sample.execution_ordinal = parse_unsigned<std::uint64_t>(
        fields[position++], "invalid execution ordinal");
    sample.repetition =
        parse_unsigned<std::uint32_t>(fields[position++], "invalid repetition");
    sample.query_id = QueryId{parse_unsigned<std::uint32_t>(
        fields[position++], "invalid sample query ID")};
    sample.configuration_id = parse_unsigned<std::uint32_t>(
        fields[position++], "invalid sample configuration ID");
    sample.instrumentation = static_cast<InstrumentationLevel>(
        parse_unsigned<std::uint32_t>(fields[position++], "invalid instrumentation"));
    sample.correctness_passed =
        parse_bool(fields[position++], "invalid correctness flag");
    sample.distances_downloaded =
        parse_bool(fields[position++], "invalid distances flag");
    sample.result.engine_kind =
        parse_unsigned<std::uint32_t>(fields[position++], "invalid result engine");
    sample.result.control_mode =
        parse_unsigned<std::uint32_t>(fields[position++], "invalid result control");
    sample.result.status = parse_status(fields, position);
    sample.result.work = parse_work(fields, position);
    sample.timing.preparation_gpu_milliseconds =
        parse_evidence(fields, position);
    sample.timing.sssp_device_timeline_milliseconds =
        parse_evidence(fields, position);
    sample.timing.result_transfer_gpu_milliseconds =
        parse_evidence(fields, position);
    sample.timing.end_to_end_wall_milliseconds =
        parse_evidence(fields, position);
    sample.profiler_provenance.pass_id = parse_unsigned<std::uint64_t>(
        fields[position++], "invalid profiler pass ID");
    sample.profiler_provenance.counter_set_id = parse_unsigned<std::uint64_t>(
        fields[position++], "invalid profiler counter-set ID");
    sample.profiler_provenance.compatible =
        parse_bool(fields[position++], "invalid profiler compatibility flag");
    sample.profiler.gpu_active_milliseconds = parse_evidence(fields, position);
    sample.profiler.l2_hit_percent = parse_evidence(fields, position);
    sample.profiler.l2_read_bytes = parse_evidence(fields, position);
    sample.profiler.l2_write_bytes = parse_evidence(fields, position);
    sample.profiler.occupancy_percent = parse_evidence(fields, position);
    sample.profiler.memory_unit_busy_percent = parse_evidence(fields, position);
    sample.profiler.waves = parse_evidence(fields, position);
    sample.profiler.vector_instructions = parse_evidence(fields, position);
    sample.profiler.scalar_instructions = parse_evidence(fields, position);
    sample.profiler.memory_instructions = parse_evidence(fields, position);
    sample.cooperative_grid_blocks = parse_unsigned<std::uint32_t>(
        fields[position++], "invalid cooperative grid blocks");
    sample.cooperative_active_blocks_per_wgp = parse_unsigned<std::uint32_t>(
        fields[position++], "invalid cooperative active blocks");
    require(position == fields.size(), "samples TSV parser left extra fields");
    samples.push_back(std::move(sample));
  }
  validate_shootout_samples(manifest, tunings, samples, run_kind);
  return samples;
}

namespace {

void write_json_distribution(
    std::ostream& output,
    const ShootoutDistribution& distribution) {
  output << "{\"count\":" << distribution.count
         << ",\"minimum\":" << distribution.minimum
         << ",\"p50\":" << distribution.p50
         << ",\"p95\":" << distribution.p95
         << ",\"p99\":" << distribution.p99
         << ",\"maximum\":" << distribution.maximum
         << ",\"mean\":" << distribution.mean << '}';
}

void write_json_evidence(
    std::ostream& output,
    const ShootoutEvidenceValue& value) {
  output << "{\"state\":" << static_cast<std::uint32_t>(value.state)
         << ",\"value\":" << value.value << '}';
}

template <typename Integer>
void write_json_integer_array(
    std::ostream& output,
    const std::vector<Integer>& values) {
  output << '[';
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << values[index];
  }
  output << ']';
}

void write_json_profile_metric(
    std::ostream& output,
    const ShootoutProfileMetricSummary& metric) {
  output << "{\"mean\":";
  write_json_evidence(output, metric.mean);
  output << ",\"counter_set_id\":" << metric.counter_set_id
         << ",\"pass_ids\":";
  write_json_integer_array(output, metric.pass_ids);
  output << '}';
}

void write_json_evidence_references(
    std::ostream& output,
    const std::span<const ShootoutEvidenceReference> references) {
  output << '[';
  for (std::size_t index = 0U; index < references.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    const ShootoutEvidenceReference& reference = references[index];
    output << "{\"run_kind\":"
           << static_cast<std::uint32_t>(reference.run_kind)
           << ",\"execution_ordinal\":" << reference.execution_ordinal
           << ",\"profiler_pass_id\":" << reference.profiler_pass_id
           << ",\"profiler_counter_set_id\":"
           << reference.profiler_counter_set_id
           << ",\"query_id\":" << reference.query_id.value()
           << ",\"configuration_id\":" << reference.configuration_id
           << '}';
  }
  output << ']';
}

}  // namespace

std::string serialize_shootout_report_json(
    const ShootoutCampaignReport& report) {
  validate_shootout_report(report);
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output << "{\n\"schema\":" << report.fingerprint.schema_version
         << ",\n\"fingerprint\":{\"graph\":["
         << report.fingerprint.graph_words[0] << ','
         << report.fingerprint.graph_words[1] << "],\"queries\":["
         << report.fingerprint.query_words[0] << ','
         << report.fingerprint.query_words[1] << "],\"corpus_query_count\":"
         << report.fingerprint.corpus_query_count << "},\n\"workload\":{\"kind\":"
         << static_cast<std::uint32_t>(report.workload.kind)
         << ",\"case_id\":" << report.workload.case_id
         << ",\"case_name\":\"" << json_escape(report.workload.case_name)
         << "\"},\n\"selection_seed\":" << report.selection_seed
         << ",\n\"order_seed\":" << report.order_seed
         << ",\n\"warmup_repetitions\":" << report.warmup_repetitions
         << ",\n\"timing_repetitions\":" << report.timing_repetitions
         << ",\n\"selected_query_count\":" << report.selected_query_count
         << ",\n\"queries\":[";
  for (std::size_t index = 0U; index < report.queries.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    const ShootoutManifestEntry& entry = report.queries[index];
    output << "\n{\"query_id\":" << entry.features.query_id.value()
           << ",\"selected_vertices\":" << entry.features.selected_vertices
           << ",\"selected_edges\":" << entry.features.selected_edges
           << ",\"fanout\":" << entry.features.fanout
           << ",\"source_count\":" << entry.features.source_count
           << ",\"expected_rounds\":" << entry.features.expected_rounds
           << ",\"quantile_bins\":[";
    for (std::size_t bin = 0U; bin < entry.quantile_bins.size(); ++bin) {
      if (bin != 0U) {
        output << ',';
      }
      output << static_cast<std::uint32_t>(entry.quantile_bins[bin]);
    }
    output << "]}";
  }
  output << "\n],\n\"summaries\":[";
  for (std::size_t index = 0U; index < report.summaries.size(); ++index) {
    const ShootoutTuningSummary& summary = report.summaries[index];
    if (index != 0U) {
      output << ',';
    }
    output << "\n{\"tuning\":{\"configuration_id\":"
           << summary.tuning.configuration_id << ",\"engine\":"
           << static_cast<std::uint32_t>(summary.tuning.engine)
           << ",\"control_mode\":"
           << static_cast<std::uint32_t>(summary.tuning.control_mode)
           << ",\"rounds_per_chunk\":" << summary.tuning.rounds_per_chunk
           << ",\"block_size\":" << summary.tuning.block_size
           << ",\"grid_policy\":"
           << static_cast<std::uint32_t>(summary.tuning.grid_policy)
           << ",\"blocks_per_wgp\":" << summary.tuning.blocks_per_wgp
           << ",\"maximum_rounds\":" << summary.tuning.maximum_rounds
           << "},\"cooperative_grid_blocks\":"
           << summary.cooperative_grid_blocks
           << ",\"cooperative_active_blocks_per_wgp\":"
           << summary.cooperative_active_blocks_per_wgp
           << ",\"wall_milliseconds\":";
    write_json_distribution(output, summary.wall_milliseconds);
    output << ",\"device_timeline_milliseconds\":";
    write_json_distribution(output, summary.device_timeline_milliseconds);
    output << ",\"throughput_queries_per_second\":";
    write_json_evidence(output, summary.throughput_queries_per_second);
    output << ",\"algorithm_work\":{\"state\":"
           << static_cast<std::uint32_t>(summary.algorithm_counters_state)
           << ",\"rounds\":" << summary.total_rounds
           << ",\"edges_examined\":" << summary.total_edges_examined
           << ",\"successful_decreases\":"
           << summary.total_successful_decreases
           << ",\"useful_decrease_ratio\":";
    write_json_evidence(output, summary.useful_decrease_ratio);
    output << ",\"active_vertices\":" << summary.total_active_vertices
           << ",\"active_lane_rounds\":"
           << summary.total_active_lane_rounds
           << ",\"maximum_queue_size\":" << summary.maximum_queue_size
           << ",\"host_checks\":" << summary.host_checks
           << ",\"atomic_attempts\":" << summary.atomic_attempts
           << ",\"successful_atomic_updates\":"
           << summary.successful_atomic_updates
           << ",\"queue_claims\":" << summary.queue_claims
           << ",\"duplicate_suppressions\":"
           << summary.duplicate_suppressions
           << ",\"expansion_count\":" << summary.expansion_count
           << ",\"mask_operations\":" << summary.mask_operations
           << ",\"overflow_events\":" << summary.overflow_events
           << ",\"high_contention_destinations\":"
           << summary.high_contention_destinations
           << ",\"changed_flag_updates\":" << summary.changed_flag_updates
           << ",\"full_edge_rounds\":" << summary.full_edge_rounds
           << ",\"empty_frontier_rounds\":"
           << summary.empty_frontier_rounds
           << ",\"small_frontier_rounds\":"
           << summary.small_frontier_rounds
           << ",\"kernel_dispatches\":" << summary.kernel_dispatches
           << ",\"host_synchronizations\":"
           << summary.host_synchronizations
           << ",\"controller_copies\":" << summary.controller_copies
           << "},\"profiler\":{\"gpu_active_milliseconds\":";
    write_json_distribution(output, summary.profiler.gpu_active_milliseconds);
    output << ",\"gpu_active_pass_ids\":";
    write_json_integer_array(output, summary.profiler.gpu_active_pass_ids);
    output << ",\"l2_hit_percent\":";
    write_json_profile_metric(output, summary.profiler.l2_hit_percent);
    output << ",\"l2_read_bytes\":";
    write_json_profile_metric(output, summary.profiler.l2_read_bytes);
    output << ",\"l2_write_bytes\":";
    write_json_profile_metric(output, summary.profiler.l2_write_bytes);
    output << ",\"occupancy_percent\":";
    write_json_profile_metric(output, summary.profiler.occupancy_percent);
    output << ",\"memory_unit_busy_percent\":";
    write_json_profile_metric(
        output, summary.profiler.memory_unit_busy_percent);
    output << ",\"waves\":";
    write_json_profile_metric(output, summary.profiler.waves);
    output << ",\"vector_instructions\":";
    write_json_profile_metric(output, summary.profiler.vector_instructions);
    output << ",\"scalar_instructions\":";
    write_json_profile_metric(output, summary.profiler.scalar_instructions);
    output << ",\"memory_instructions\":";
    write_json_profile_metric(output, summary.profiler.memory_instructions);
    output << "},\"long_tail\":[";
    for (std::size_t tail_index = 0U;
         tail_index < summary.long_tail.size();
         ++tail_index) {
      if (tail_index != 0U) {
        output << ',';
      }
      const ShootoutLongTailRecord& tail = summary.long_tail[tail_index];
      output << "{\"metric\":" << static_cast<std::uint32_t>(tail.metric)
             << ",\"query_id\":" << tail.query_id.value()
             << ",\"configuration_id\":" << tail.configuration_id
             << ",\"repetition\":" << tail.repetition
             << ",\"milliseconds\":" << tail.milliseconds
             << ",\"features\":{\"selected_vertices\":"
             << tail.features.selected_vertices << ",\"selected_edges\":"
             << tail.features.selected_edges << ",\"fanout\":"
             << tail.features.fanout << ",\"source_count\":"
             << tail.features.source_count << ",\"expected_rounds\":"
             << tail.features.expected_rounds << "}}";
    }
    output << "]}";
  }
  output << "\n],\n\"supplied_evidence\":";
  write_json_evidence_references(output, report.supplied_evidence);
  output << ",\n\"answers\":[";
  for (std::size_t index = 0U; index < report.answers.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    const ShootoutQuestionAnswer& answer = report.answers[index];
    output << "\n{\"question_id\":" << answer.question_id
           << ",\"state\":" << static_cast<std::uint32_t>(answer.state)
           << ",\"workload_case_id\":" << answer.workload.case_id
           << ",\"configuration_ids\":";
    write_json_integer_array(output, answer.configuration_ids);
    output << ",\"evidence\":";
    write_json_evidence_references(output, answer.evidence);
    output << ",\"conclusion\":\"" << json_escape(answer.conclusion)
           << "\"}";
  }
  output << "\n],\n\"recommendations\":[";
  for (std::size_t index = 0U; index < report.recommendations.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    const ShootoutRecommendation& recommendation = report.recommendations[index];
    output << "\n{\"state\":"
           << static_cast<std::uint32_t>(recommendation.state)
           << ",\"workload_case_id\":" << recommendation.workload.case_id
           << ",\"configuration_ids\":";
    write_json_integer_array(output, recommendation.configuration_ids);
    output << ",\"evidence\":";
    write_json_evidence_references(output, recommendation.evidence);
    output << ",\"rationale\":\"" << json_escape(recommendation.rationale)
           << "\",\"toggles_remain_configurable\":"
           << (recommendation.toggles_remain_configurable ? "true" : "false")
           << '}';
  }
  output << "\n]\n}\n";
  return output.str();
}

}  // namespace bfnew
