#include "bfnew/engine_shootout.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::filesystem::path catalog;
  std::filesystem::path manifest;
  std::filesystem::path correctness;
  std::filesystem::path timing;
  std::filesystem::path counters;
  std::vector<std::filesystem::path> trace;
  std::vector<std::filesystem::path> pmc;
  std::filesystem::path output;
  std::filesystem::path profile_plan;
  std::filesystem::path conclusions;
  std::optional<std::uint64_t> order_seed;
};

inline constexpr std::size_t maximum_profile_cases = 48U;
inline constexpr std::uint32_t profile_plan_schema_version = 1U;
inline constexpr std::uint32_t conclusion_schema_version = 1U;

[[nodiscard]] std::uint64_t parse_u64(
    const std::string& text,
    const char* const name) {
  std::size_t consumed = 0U;
  const unsigned long long value = std::stoull(text, &consumed, 10);
  if (consumed != text.size()) {
    throw std::invalid_argument{std::string{name} + " must be an unsigned integer"};
  }
  return static_cast<std::uint64_t>(value);
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&](const char* const name) -> std::filesystem::path {
      if (index + 1 >= argc) {
        throw std::invalid_argument{std::string{name} + " requires a path"};
      }
      return argv[++index];
    };
    if (argument == "--catalog") {
      options.catalog = value("--catalog");
    } else if (argument == "--manifest") {
      options.manifest = value("--manifest");
    } else if (argument == "--correctness") {
      options.correctness = value("--correctness");
    } else if (argument == "--timing") {
      options.timing = value("--timing");
    } else if (argument == "--counters") {
      options.counters = value("--counters");
    } else if (argument == "--trace") {
      options.trace.push_back(value("--trace"));
    } else if (argument == "--pmc") {
      options.pmc.push_back(value("--pmc"));
    } else if (argument == "--output") {
      options.output = value("--output");
    } else if (argument == "--profile-plan") {
      options.profile_plan = value("--profile-plan");
    } else if (argument == "--conclusions") {
      options.conclusions = value("--conclusions");
    } else if (argument == "--order-seed") {
      const std::filesystem::path parsed = value("--order-seed");
      options.order_seed = parse_u64(parsed.string(), "order seed");
    } else if (argument == "--help") {
      std::cout
          << "Usage: bfnew_shootout_report --catalog FILE --manifest FILE "
             "--correctness FILE --timing FILE --counters FILE "
             "[--trace FILE ...] [--pmc FILE ...] --output FILE "
             "[--profile-plan FILE] [--conclusions FILE] "
             "[--order-seed N]\n"
             "The replay order seed defaults to the manifest seed; an "
             "explicit mismatch is rejected.\n";
      std::exit(0);
    } else {
      throw std::invalid_argument{"unknown option: " + argument};
    }
  }
  if (options.catalog.empty() || options.manifest.empty() ||
      options.correctness.empty() || options.timing.empty() ||
      options.counters.empty() || options.output.empty()) {
    throw std::invalid_argument{
        "catalog, manifest, correctness, timing, counters, and output are required"};
  }
  return options;
}

[[nodiscard]] std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error{"cannot open shootout input: " + path.string()};
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
  if (!output) {
    throw std::runtime_error{"cannot create shootout report: " + path.string()};
  }
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output) {
    throw std::runtime_error{"failed while writing shootout report"};
  }
}

void append_samples(
    std::vector<bfnew::ShootoutSample>& destination,
    std::vector<bfnew::ShootoutSample> source) {
  destination.insert(
      destination.end(),
      std::make_move_iterator(source.begin()),
      std::make_move_iterator(source.end()));
}

[[nodiscard]] std::vector<std::string_view> split(
    const std::string_view text,
    const char delimiter) {
  std::vector<std::string_view> fields;
  std::size_t begin = 0U;
  while (true) {
    const std::size_t end = text.find(delimiter, begin);
    fields.push_back(text.substr(begin, end - begin));
    if (end == std::string_view::npos) {
      return fields;
    }
    begin = end + 1U;
  }
}

template <class Integer>
[[nodiscard]] Integer parse_unsigned_field(
    const std::string_view text,
    const char* const name) {
  static_assert(std::is_unsigned_v<Integer>);
  Integer result{};
  const auto parsed = std::from_chars(
      text.data(), text.data() + text.size(), result, 10);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size()) {
    throw std::invalid_argument{std::string{"invalid "} + name};
  }
  return result;
}

[[nodiscard]] bfnew::ShootoutConclusionState parse_conclusion_state(
    const std::string_view text) {
  if (text == "measured") {
    return bfnew::ShootoutConclusionState::measured;
  }
  if (text == "insufficient_evidence") {
    return bfnew::ShootoutConclusionState::insufficient_evidence;
  }
  throw std::invalid_argument{
      "conclusion state must be measured or insufficient_evidence"};
}

[[nodiscard]] bfnew::ShootoutWorkloadKind parse_workload_kind(
    const std::string_view text) {
  if (text == "logicnets_jscl") {
    return bfnew::ShootoutWorkloadKind::logicnets_jscl;
  }
  if (text == "synthetic") {
    return bfnew::ShootoutWorkloadKind::synthetic;
  }
  throw std::invalid_argument{
      "conclusion workload must be logicnets_jscl or synthetic"};
}

[[nodiscard]] bfnew::ShootoutRunKind parse_evidence_kind(
    const std::string_view text) {
  if (text == "correctness") {
    return bfnew::ShootoutRunKind::correctness;
  }
  if (text == "timing") {
    return bfnew::ShootoutRunKind::timing;
  }
  if (text == "algorithm_counters") {
    return bfnew::ShootoutRunKind::algorithm_counters;
  }
  if (text == "trace") {
    return bfnew::ShootoutRunKind::trace;
  }
  if (text == "pmc") {
    return bfnew::ShootoutRunKind::pmc;
  }
  throw std::invalid_argument{"invalid conclusion evidence kind"};
}

[[nodiscard]] std::vector<std::uint32_t> parse_configuration_ids(
    const std::string_view text) {
  if (text.empty()) {
    return {};
  }
  std::vector<std::uint32_t> result;
  std::set<std::uint32_t> unique;
  for (const std::string_view field : split(text, ',')) {
    const std::uint32_t id =
        parse_unsigned_field<std::uint32_t>(field, "configuration ID");
    if (!unique.insert(id).second) {
      throw std::invalid_argument{
          "conclusion contains a duplicate configuration ID"};
    }
    result.push_back(id);
  }
  return result;
}

[[nodiscard]] std::vector<bfnew::ShootoutEvidenceReference>
parse_evidence_references(const std::string_view text) {
  if (text.empty()) {
    return {};
  }
  using EvidenceKey = std::tuple<
      bfnew::ShootoutRunKind,
      std::uint64_t,
      std::uint64_t,
      std::uint64_t,
      std::uint32_t,
      std::uint32_t>;
  std::set<EvidenceKey> unique;
  std::vector<bfnew::ShootoutEvidenceReference> result;
  for (const std::string_view encoded : split(text, ';')) {
    const std::vector<std::string_view> fields = split(encoded, ':');
    if (fields.size() != 6U) {
      throw std::invalid_argument{
          "evidence reference must be "
          "kind:ordinal:pass:counter_set:query_id:configuration_id"};
    }
    bfnew::ShootoutEvidenceReference reference{
        parse_evidence_kind(fields[0]),
        parse_unsigned_field<std::uint64_t>(
            fields[1], "evidence execution ordinal"),
        parse_unsigned_field<std::uint64_t>(
            fields[2], "evidence profiler pass ID"),
        parse_unsigned_field<std::uint64_t>(
            fields[3], "evidence profiler counter-set ID"),
        bfnew::QueryId{parse_unsigned_field<std::uint32_t>(
            fields[4], "evidence query ID")},
        parse_unsigned_field<std::uint32_t>(
            fields[5], "evidence configuration ID"),
    };
    const EvidenceKey key{
        reference.run_kind,
        reference.execution_ordinal,
        reference.profiler_pass_id,
        reference.profiler_counter_set_id,
        reference.query_id.value(),
        reference.configuration_id};
    if (!unique.insert(key).second) {
      throw std::invalid_argument{
          "conclusion contains a duplicate evidence reference"};
    }
    result.push_back(reference);
  }
  return result;
}

void apply_conclusions(
    bfnew::ShootoutCampaignReport& report,
    const std::string_view text) {
  constexpr std::string_view header =
      "schema\trow_type\trecord_id\tstate\tworkload_kind\tworkload_case_id\t"
      "workload_case_name\tconfiguration_ids\tevidence_refs\t"
      "toggles_remain_configurable\ttext";
  std::array<bfnew::ShootoutQuestionAnswer,
             bfnew::shootout_performance_question_count>
      questions;
  std::array<bool, bfnew::shootout_performance_question_count> have_question{};
  std::vector<std::pair<std::uint32_t, bfnew::ShootoutRecommendation>>
      recommendations;
  std::set<std::uint32_t> recommendation_ids;

  std::size_t begin = 0U;
  std::size_t line_number = 0U;
  while (begin < text.size()) {
    const std::size_t end = text.find('\n', begin);
    const std::string_view line = text.substr(begin, end - begin);
    ++line_number;
    if (line.ends_with('\r') || line.empty()) {
      throw std::invalid_argument{
          "conclusions TSV contains an empty or CR-terminated line"};
    }
    if (line_number == 1U) {
      if (line != header) {
        throw std::invalid_argument{"conclusions TSV header is not canonical"};
      }
    } else {
      const std::vector<std::string_view> fields = split(line, '\t');
      if (fields.size() != 11U) {
        throw std::invalid_argument{
            "conclusions TSV row must contain exactly 11 fields"};
      }
      if (parse_unsigned_field<std::uint32_t>(fields[0], "conclusion schema") !=
          conclusion_schema_version) {
        throw std::invalid_argument{"unsupported conclusions TSV schema"};
      }
      const std::uint32_t record_id =
          parse_unsigned_field<std::uint32_t>(fields[2], "conclusion record ID");
      const bfnew::ShootoutConclusionState state =
          parse_conclusion_state(fields[3]);
      const bfnew::ShootoutWorkloadIdentity workload{
          parse_workload_kind(fields[4]),
          parse_unsigned_field<std::uint64_t>(
              fields[5], "conclusion workload case ID"),
          std::string{fields[6]},
      };
      if (workload != report.workload) {
        throw std::invalid_argument{
            "conclusion row identifies a different workload"};
      }
      std::vector<std::uint32_t> configurations =
          parse_configuration_ids(fields[7]);
      std::vector<bfnew::ShootoutEvidenceReference> evidence =
          parse_evidence_references(fields[8]);
      if (configurations.empty() || evidence.empty()) {
        throw std::invalid_argument{
            "conclusion rows require cited configurations and evidence"};
      }
      if (fields[10].empty()) {
        throw std::invalid_argument{"conclusion text must not be empty"};
      }

      if (fields[1] == "question") {
        if (record_id == 0U || record_id > questions.size() ||
            have_question[record_id - 1U]) {
          throw std::invalid_argument{
              "conclusions TSV has an invalid or duplicate question ID"};
        }
        if (fields[9] != "n/a") {
          throw std::invalid_argument{
              "question rows require n/a in the toggle field"};
        }
        questions[record_id - 1U] = bfnew::ShootoutQuestionAnswer{
            record_id,
            state,
            workload,
            std::move(configurations),
            std::move(evidence),
            std::string{fields[10]},
        };
        have_question[record_id - 1U] = true;
      } else if (fields[1] == "recommendation") {
        if (record_id == 0U || !recommendation_ids.insert(record_id).second) {
          throw std::invalid_argument{
              "conclusions TSV has an invalid or duplicate recommendation ID"};
        }
        if (fields[9] != "1") {
          throw std::invalid_argument{
              "recommendations must keep every toggle configurable"};
        }
        recommendations.emplace_back(
            record_id,
            bfnew::ShootoutRecommendation{
                state,
                workload,
                std::move(configurations),
                std::move(evidence),
                std::string{fields[10]},
                true,
            });
      } else {
        throw std::invalid_argument{
            "conclusion row type must be question or recommendation"};
      }
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  if (line_number == 0U ||
      !std::ranges::all_of(have_question, [](const bool value) {
        return value;
      }) ||
      recommendations.empty()) {
    throw std::invalid_argument{
        "conclusions TSV requires questions 1..7 and a recommendation"};
  }
  std::sort(
      recommendations.begin(),
      recommendations.end(),
      [](const auto& left, const auto& right) {
        return left.first < right.first;
      });
  report.answers = std::move(questions);
  report.recommendations.clear();
  report.recommendations.reserve(recommendations.size());
  for (auto& [id, recommendation] : recommendations) {
    static_cast<void>(id);
    report.recommendations.push_back(std::move(recommendation));
  }
}

[[nodiscard]] std::string make_profile_plan(
    const bfnew::ShootoutManifest& manifest,
    const std::span<const bfnew::ShootoutTuning> tunings,
    const bfnew::ShootoutCampaignReport& report,
    const std::uint64_t order_seed) {
  const std::vector<bfnew::ShootoutScheduleEntry> schedule =
      bfnew::make_interleaved_shootout_schedule(
          manifest,
          tunings,
          bfnew::ShootoutRunKind::trace,
          1U,
          order_seed);
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint64_t> case_ids;
  for (const bfnew::ShootoutScheduleEntry& entry : schedule) {
    if (!case_ids
             .emplace(
                 std::pair{entry.query_id.value(), entry.configuration_id},
                 entry.execution_ordinal)
             .second) {
      throw std::logic_error{"profile schedule contains a duplicate case"};
    }
  }

  std::map<std::uint32_t, const bfnew::ShootoutTuning*> tuning_by_id;
  for (const bfnew::ShootoutTuning& tuning : tunings) {
    if (!tuning_by_id.emplace(tuning.configuration_id, &tuning).second) {
      throw std::logic_error{"profile plan received duplicate tuning IDs"};
    }
  }
  struct Candidate {
    const bfnew::ShootoutLongTailRecord* tail{};
    const bfnew::ShootoutTuning* tuning{};
  };
  std::vector<Candidate> tails;
  for (const bfnew::ShootoutTuningSummary& summary : report.summaries) {
    for (const bfnew::ShootoutLongTailRecord& tail : summary.long_tail) {
      const auto tuning = tuning_by_id.find(tail.configuration_id);
      if (tuning == tuning_by_id.end()) {
        throw std::logic_error{"long-tail record names an unknown tuning"};
      }
      tails.push_back(Candidate{&tail, tuning->second});
    }
  }
  std::sort(
      tails.begin(),
      tails.end(),
      [](const Candidate& left, const Candidate& right) {
        if (left.tail->milliseconds != right.tail->milliseconds) {
          return left.tail->milliseconds > right.tail->milliseconds;
        }
        return std::tuple{
                   left.tail->metric,
                   left.tuning->engine,
                   left.tuning->control_mode,
                   left.tail->configuration_id,
                   left.tail->query_id.value(),
                   left.tail->repetition} <
               std::tuple{
                   right.tail->metric,
                   right.tuning->engine,
                   right.tuning->control_mode,
                   right.tail->configuration_id,
                   right.tail->query_id.value(),
                   right.tail->repetition};
      });

  using CandidateKey = std::tuple<
      bfnew::ShootoutTailMetric,
      std::uint32_t,
      std::uint32_t>;
  std::set<CandidateKey> deduplicated;
  std::vector<Candidate> candidates;
  candidates.reserve(tails.size());
  for (const Candidate& candidate : tails) {
    if (!deduplicated
             .emplace(
                 candidate.tail->metric,
                 candidate.tail->query_id.value(),
                 candidate.tail->configuration_id)
             .second) {
      continue;
    }
    candidates.push_back(candidate);
  }

  struct Selection {
    Candidate candidate;
    const char* reason{};
    std::set<bfnew::ShootoutTailMetric> covered_metrics;
  };
  std::vector<Selection> selected;
  selected.reserve(std::min(maximum_profile_cases, candidates.size()));
  std::set<std::tuple<
      bfnew::ShootoutTailMetric,
      bfnew::EngineKind,
      bfnew::ControlMode>>
      covered_strata;
  using CaseKey = std::pair<std::uint32_t, std::uint32_t>;
  std::map<CaseKey, std::size_t> selected_cases;
  for (const Candidate& candidate : candidates) {
    const auto stratum = std::tuple{
        candidate.tail->metric,
        candidate.tuning->engine,
        candidate.tuning->control_mode};
    if (covered_strata.insert(stratum).second) {
      const CaseKey key{
          candidate.tail->query_id.value(),
          candidate.tail->configuration_id};
      const auto existing = selected_cases.find(key);
      if (existing != selected_cases.end()) {
        selected[existing->second].covered_metrics.insert(
            candidate.tail->metric);
      } else {
        const std::size_t index = selected.size();
        selected.push_back(Selection{
            candidate, "stratum_coverage", {candidate.tail->metric}});
        selected_cases.emplace(key, index);
      }
    }
  }
  for (const Candidate& candidate : candidates) {
    if (selected.size() == maximum_profile_cases) {
      break;
    }
    const CaseKey key{
        candidate.tail->query_id.value(),
        candidate.tail->configuration_id};
    if (!selected_cases.contains(key)) {
      const std::size_t index = selected.size();
      selected.push_back(Selection{
          candidate, "highest_remaining", {candidate.tail->metric}});
      selected_cases.emplace(key, index);
    }
  }
  if (selected.size() > maximum_profile_cases) {
    throw std::logic_error{"profile-plan stratum coverage exceeds its cap"};
  }

  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output
      << "profile_case_id\tplan_schema\tplan_cap\tplan_selected_count\t"
         "selection_rank\tselection_reason\tranking_metric\tcovered_metrics\t"
         "workload_kind\tworkload_case_id\tworkload_case_name\tinput_schema\tgraph0\t"
         "graph1\tquery0\tquery1\tcorpus_queries\tselection_seed\t"
         "manifest_order_seed\treplay_order_seed\twarmup_repetitions\t"
         "timing_repetitions\tschedule_policy\tquery_id\tconfiguration_id\t"
         "engine\tcontrol\trounds_per_chunk\tblock_size\tgrid_policy\t"
         "blocks_per_wgp\tmaximum_rounds\tsource_repetition\tmilliseconds\t"
         "selected_vertices\tselected_edges\tfanout\tsource_count\t"
         "expected_rounds\n";
  for (std::size_t rank = 0U; rank < selected.size(); ++rank) {
    const Selection& selection = selected[rank];
    const bfnew::ShootoutLongTailRecord& tail = *selection.candidate.tail;
    const bfnew::ShootoutTuning& tuning = *selection.candidate.tuning;
    const auto found = case_ids.find(
        std::pair{tail.query_id.value(), tail.configuration_id});
    if (found == case_ids.end()) {
      throw std::logic_error{"long-tail record is absent from profile schedule"};
    }
    output << found->second << '\t' << profile_plan_schema_version << '\t'
           << maximum_profile_cases << '\t' << selected.size() << '\t' << rank
           << '\t' << selection.reason << '\t'
           << static_cast<std::uint32_t>(tail.metric) << '\t';
    bool first_metric = true;
    for (const bfnew::ShootoutTailMetric metric :
         selection.covered_metrics) {
      if (!first_metric) {
        output << ',';
      }
      output << static_cast<std::uint32_t>(metric);
      first_metric = false;
    }
    output << '\t'
           << static_cast<std::uint32_t>(manifest.workload.kind) << '\t'
           << manifest.workload.case_id << '\t' << manifest.workload.case_name
           << '\t' << manifest.fingerprint.schema_version << '\t'
           << manifest.fingerprint.graph_words[0] << '\t'
           << manifest.fingerprint.graph_words[1] << '\t'
           << manifest.fingerprint.query_words[0] << '\t'
           << manifest.fingerprint.query_words[1] << '\t'
           << manifest.fingerprint.corpus_query_count << '\t'
           << manifest.selection_seed << '\t' << manifest.order_seed << '\t'
           << order_seed << '\t' << manifest.warmup_repetitions << '\t'
           << manifest.timing_repetitions
           << "\tinterleaved_one_repetition_v1\t"
           << tail.query_id.value() << '\t' << tail.configuration_id << '\t'
           << static_cast<std::uint32_t>(tuning.engine) << '\t'
           << static_cast<std::uint32_t>(tuning.control_mode) << '\t'
           << tuning.rounds_per_chunk << '\t' << tuning.block_size << '\t'
           << static_cast<std::uint32_t>(tuning.grid_policy) << '\t'
           << tuning.blocks_per_wgp << '\t' << tuning.maximum_rounds << '\t'
           << tail.repetition << '\t' << tail.milliseconds << '\t'
           << tail.features.selected_vertices << '\t'
           << tail.features.selected_edges << '\t' << tail.features.fanout
           << '\t' << tail.features.source_count << '\t'
           << tail.features.expected_rounds << '\n';
  }
  return output.str();
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const bfnew::ShootoutManifest manifest =
        bfnew::deserialize_shootout_manifest_tsv(read_text(options.manifest));
    const bfnew::ShootoutConfigurationCatalog catalog =
        bfnew::deserialize_shootout_catalog_tsv(read_text(options.catalog));
    bfnew::validate_shootout_configuration_catalog(catalog);
    if (catalog.fingerprint != manifest.fingerprint ||
        catalog.workload != manifest.workload) {
      throw std::invalid_argument{
          "shootout catalog and manifest identify different inputs"};
    }
    if (options.order_seed && *options.order_seed != manifest.order_seed) {
      throw std::invalid_argument{
          "explicit replay order seed differs from the manifest order seed"};
    }
    const std::uint64_t replay_order_seed =
        options.order_seed.value_or(manifest.order_seed);

    const auto read_stage = [&](const std::filesystem::path& path,
                                const bfnew::ShootoutRunKind kind) {
      return bfnew::deserialize_shootout_samples_tsv(
          manifest, catalog.tunings, read_text(path), kind);
    };
    const std::vector<bfnew::ShootoutSample> correctness =
        read_stage(options.correctness, bfnew::ShootoutRunKind::correctness);
    const std::vector<bfnew::ShootoutSample> timing =
        read_stage(options.timing, bfnew::ShootoutRunKind::timing);
    const std::vector<bfnew::ShootoutSample> counters = read_stage(
        options.counters, bfnew::ShootoutRunKind::algorithm_counters);
    std::vector<bfnew::ShootoutSample> trace;
    for (const std::filesystem::path& path : options.trace) {
      append_samples(trace, read_stage(path, bfnew::ShootoutRunKind::trace));
    }
    std::vector<bfnew::ShootoutSample> pmc;
    for (const std::filesystem::path& path : options.pmc) {
      append_samples(pmc, read_stage(path, bfnew::ShootoutRunKind::pmc));
    }

    bfnew::ShootoutCampaignReport report =
        bfnew::summarize_shootout_campaign(
            manifest,
            catalog.tunings,
            correctness,
            timing,
            counters,
            trace,
            pmc);
    if (!options.conclusions.empty()) {
      apply_conclusions(report, read_text(options.conclusions));
    }
    bfnew::validate_shootout_report(report);
    write_text(options.output, bfnew::serialize_shootout_report_json(report));
    if (!options.profile_plan.empty()) {
      write_text(
          options.profile_plan,
          make_profile_plan(
              manifest, catalog.tunings, report, replay_order_seed));
    }
    std::cout << "Validated Phase 12 evidence and wrote "
              << options.output.string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "bfnew_shootout_report: " << error.what() << '\n';
    return 1;
  }
}
