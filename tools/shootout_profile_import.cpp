#include "bfnew/engine_shootout.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  std::filesystem::path catalog;
  std::filesystem::path manifest;
  std::filesystem::path input;
  std::filesystem::path output;
  bfnew::ShootoutRunKind kind{bfnew::ShootoutRunKind::trace};
  bool has_kind{};
  std::map<std::string, std::string> ungrouped_metrics;
  std::map<std::uint64_t, std::map<std::string, std::string>> case_metrics;
  std::optional<std::uint64_t> current_case_id;
};

[[nodiscard]] std::uint64_t parse_u64(
    const std::string& text,
    const char* const name) {
  std::uint64_t value{};
  const auto parsed = std::from_chars(
      text.data(), text.data() + text.size(), value, 10);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size()) {
    throw std::invalid_argument{std::string{name} +
                                " must be an unsigned integer"};
  }
  return value;
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&](const char* const name) -> std::string {
      if (index + 1 >= argc) {
        throw std::invalid_argument{std::string{name} + " requires a value"};
      }
      return argv[++index];
    };
    if (argument == "--catalog") {
      options.catalog = value("--catalog");
    } else if (argument == "--manifest") {
      options.manifest = value("--manifest");
    } else if (argument == "--input") {
      options.input = value("--input");
    } else if (argument == "--output") {
      options.output = value("--output");
    } else if (argument == "--kind") {
      const std::string kind = value("--kind");
      if (kind == "trace") {
        options.kind = bfnew::ShootoutRunKind::trace;
      } else if (kind == "pmc") {
        options.kind = bfnew::ShootoutRunKind::pmc;
      } else {
        throw std::invalid_argument{"--kind must be trace or pmc"};
      }
      options.has_kind = true;
    } else if (argument == "--case-id") {
      if (!options.ungrouped_metrics.empty()) {
        throw std::invalid_argument{
            "--case-id must precede every grouped --metric"};
      }
      const std::uint64_t case_id =
          parse_u64(value("--case-id"), "case ID");
      if (!options.case_metrics
               .emplace(case_id, std::map<std::string, std::string>{})
               .second) {
        throw std::invalid_argument{"duplicate profile case ID"};
      }
      options.current_case_id = case_id;
    } else if (argument == "--metric") {
      const std::string assignment = value("--metric");
      const std::size_t separator = assignment.find('=');
      if (separator == std::string::npos || separator == 0U ||
          separator + 1U == assignment.size()) {
        throw std::invalid_argument{"--metric requires NAME=VALUE"};
      }
      auto& metrics = options.current_case_id
                          ? options.case_metrics.at(*options.current_case_id)
                          : options.ungrouped_metrics;
      if (!metrics
               .emplace(
                   assignment.substr(0U, separator),
                   assignment.substr(separator + 1U))
               .second) {
        throw std::invalid_argument{"duplicate profiler metric"};
      }
    } else if (argument == "--help") {
      std::cout
          << "Usage: bfnew_shootout_profile_import --kind trace|pmc "
             "--manifest FILE --catalog FILE --input STAGING.tsv "
             "[--case-id ID --metric NAME=VALUE ...] ... "
             "--output MEASURED.tsv\n"
             "A one-row input may omit --case-id. Multi-row input requires "
             "one unique --case-id group per execution ordinal.\n"
             "Metrics: gpu_active_ms, l2_hit_percent, l2_read_bytes, "
             "l2_write_bytes, occupancy_percent, memory_unit_busy_percent, "
             "waves, vector_instructions, scalar_instructions, "
             "memory_instructions. VALUE may be n/a or unavailable.\n";
      std::exit(0);
    } else {
      throw std::invalid_argument{"unknown option: " + argument};
    }
  }
  if (!options.has_kind || options.catalog.empty() || options.manifest.empty() ||
      options.input.empty() || options.output.empty() ||
      (options.ungrouped_metrics.empty() && options.case_metrics.empty())) {
    throw std::invalid_argument{
        "kind, catalog, manifest, input, metrics, and output are required"};
  }
  for (const auto& [case_id, metrics] : options.case_metrics) {
    static_cast<void>(case_id);
    if (metrics.empty()) {
      throw std::invalid_argument{
          "every --case-id must be followed by at least one --metric"};
    }
  }
  return options;
}

[[nodiscard]] std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error{"cannot open profile import input: " + path.string()};
  }
  return std::string{
      std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

void write_text(
    const std::filesystem::path& path,
    const std::string_view contents) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary);
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output) {
    throw std::runtime_error{"cannot write measured profiler TSV"};
  }
}

[[nodiscard]] bfnew::ShootoutEvidenceValue parse_evidence(
    const std::string& text) {
  if (text == "n/a") {
    return {bfnew::ShootoutEvidenceState::not_applicable, 0.0};
  }
  if (text == "unavailable") {
    return {bfnew::ShootoutEvidenceState::unavailable, 0.0};
  }
  std::size_t consumed = 0U;
  const double value = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument{"profiler metric must be finite and nonnegative"};
  }
  return {bfnew::ShootoutEvidenceState::measured, value};
}

void assign_metric(
    bfnew::ShootoutProfilerRecord& profiler,
    const std::string& name,
    const bfnew::ShootoutEvidenceValue value) {
  if (name == "gpu_active_ms") {
    profiler.gpu_active_milliseconds = value;
  } else if (name == "l2_hit_percent") {
    profiler.l2_hit_percent = value;
  } else if (name == "l2_read_bytes") {
    profiler.l2_read_bytes = value;
  } else if (name == "l2_write_bytes") {
    profiler.l2_write_bytes = value;
  } else if (name == "occupancy_percent") {
    profiler.occupancy_percent = value;
  } else if (name == "memory_unit_busy_percent") {
    profiler.memory_unit_busy_percent = value;
  } else if (name == "waves") {
    profiler.waves = value;
  } else if (name == "vector_instructions") {
    profiler.vector_instructions = value;
  } else if (name == "scalar_instructions") {
    profiler.scalar_instructions = value;
  } else if (name == "memory_instructions") {
    profiler.memory_instructions = value;
  } else {
    throw std::invalid_argument{"unknown normalized profiler metric: " + name};
  }
}

[[nodiscard]] bool any_hardware_measurement(
    const bfnew::ShootoutProfilerRecord& profiler) noexcept {
  return profiler.l2_hit_percent.state == bfnew::ShootoutEvidenceState::measured ||
         profiler.l2_read_bytes.state == bfnew::ShootoutEvidenceState::measured ||
         profiler.l2_write_bytes.state == bfnew::ShootoutEvidenceState::measured ||
         profiler.occupancy_percent.state == bfnew::ShootoutEvidenceState::measured ||
         profiler.memory_unit_busy_percent.state ==
             bfnew::ShootoutEvidenceState::measured ||
         profiler.waves.state == bfnew::ShootoutEvidenceState::measured ||
         profiler.vector_instructions.state ==
             bfnew::ShootoutEvidenceState::measured ||
         profiler.scalar_instructions.state ==
             bfnew::ShootoutEvidenceState::measured ||
         profiler.memory_instructions.state ==
             bfnew::ShootoutEvidenceState::measured;
}

void validate_imported_metrics(
    const bfnew::ShootoutRunKind kind,
    const bfnew::ShootoutProfilerRecord& profiler) {
  if (kind == bfnew::ShootoutRunKind::trace) {
    if (profiler.gpu_active_milliseconds.state !=
            bfnew::ShootoutEvidenceState::measured ||
        any_hardware_measurement(profiler)) {
      throw std::invalid_argument{
          "trace import requires gpu_active_ms only"};
    }
  } else if (!any_hardware_measurement(profiler) ||
             profiler.gpu_active_milliseconds.state ==
                 bfnew::ShootoutEvidenceState::measured) {
    throw std::invalid_argument{
        "PMC import requires hardware counters and no GPU-active time"};
  }
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const bfnew::ShootoutManifest manifest =
        bfnew::deserialize_shootout_manifest_tsv(read_text(options.manifest));
    const bfnew::ShootoutConfigurationCatalog catalog =
        bfnew::deserialize_shootout_catalog_tsv(read_text(options.catalog));
    if (manifest.fingerprint != catalog.fingerprint ||
        manifest.workload != catalog.workload) {
      throw std::invalid_argument{"profile import catalog/manifest mismatch"};
    }
    std::vector<bfnew::ShootoutSample> samples =
        bfnew::deserialize_shootout_samples_tsv(
            manifest,
            catalog.tunings,
            read_text(options.input),
            options.kind);
    if (!options.ungrouped_metrics.empty()) {
      if (samples.size() != 1U) {
        throw std::invalid_argument{
            "multi-row profile import requires --case-id metric groups"};
      }
      for (const auto& [name, text] : options.ungrouped_metrics) {
        assign_metric(samples.front().profiler, name, parse_evidence(text));
      }
      validate_imported_metrics(options.kind, samples.front().profiler);
    } else {
      if (samples.size() != options.case_metrics.size()) {
        throw std::invalid_argument{
            "profile metric groups do not form a one-to-one staging-row map"};
      }
      std::set<std::uint64_t> staging_case_ids;
      for (bfnew::ShootoutSample& sample : samples) {
        if (!staging_case_ids.insert(sample.execution_ordinal).second) {
          throw std::invalid_argument{
              "profile staging input has a duplicate execution ordinal"};
        }
        const auto metrics = options.case_metrics.find(sample.execution_ordinal);
        if (metrics == options.case_metrics.end()) {
          throw std::invalid_argument{
              "profile staging row has no matching --case-id metric group"};
        }
        for (const auto& [name, text] : metrics->second) {
          assign_metric(sample.profiler, name, parse_evidence(text));
        }
        validate_imported_metrics(options.kind, sample.profiler);
      }
      for (const auto& [case_id, metrics] : options.case_metrics) {
        static_cast<void>(metrics);
        if (!staging_case_ids.contains(case_id)) {
          throw std::invalid_argument{
              "--case-id metric group has no matching staging row"};
        }
      }
    }
    if (samples.empty()) {
      throw std::invalid_argument{
          "profile import requires at least one staging row"};
    }
    write_text(
        options.output,
        bfnew::serialize_shootout_samples_tsv(
            manifest, catalog.tunings, samples, options.kind));
    std::cout << "Validated normalized profiler evidence and wrote "
              << options.output.string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "bfnew_shootout_profile_import: " << error.what() << '\n';
    return 1;
  }
}
