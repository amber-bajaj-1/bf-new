#include "bfnew/dense_chaotic_push.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/engine_shootout.hpp"
#include "bfnew/frontier_push.hpp"
#include "bfnew/jacobi_pull.hpp"
#include "bfnew/query.hpp"
#include "bfnew/sssp.hpp"
#include "jacobi_fixture_suite.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

int failures = 0;
constexpr std::uint64_t shootout_order_seed = 0x5eed1234U;
constexpr std::uint32_t shootout_warmups = 2U;
constexpr std::uint32_t shootout_timing_repetitions = 4U;

void expect(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAILED: " << description << '\n';
    ++failures;
  }
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

[[nodiscard]] bfnew::ShootoutInputFingerprint fingerprint(
    const std::uint64_t query_count) {
  return bfnew::ShootoutInputFingerprint{
      {0x123456789abcdef0ULL, 0x0fedcba987654321ULL},
      {0x1020304050607080ULL, 0x8877665544332211ULL},
      query_count,
      bfnew::shootout_schema_version,
  };
}

[[nodiscard]] bfnew::ShootoutQueryFeatures features(
    const std::uint32_t query,
    const std::uint64_t salt = 0U) {
  const std::uint64_t value = static_cast<std::uint64_t>(query) + salt;
  return bfnew::ShootoutQueryFeatures{
      bfnew::QueryId{query},
      8U + value % 509U,
      12U + (value * 17U) % 4001U,
      1U + static_cast<std::uint32_t>((value * 7U) % 64U),
      1U + static_cast<std::uint32_t>(value % 2U),
      1U + (value * 13U) % 97U,
  };
}

[[nodiscard]] bfnew::ShootoutManifest make_small_manifest() {
  const std::array cases{
      features(10U, 1U),
      features(11U, 101U),
      features(12U, 1001U),
  };
  return bfnew::make_synthetic_shootout_manifest(
      fingerprint(cases.size()),
      bfnew::ShootoutWorkloadIdentity{
          bfnew::ShootoutWorkloadKind::synthetic,
          0x53594e5448455449ULL,
          "phase12-sparse-and-dense-adversaries"},
      cases,
      shootout_order_seed,
      shootout_warmups,
      shootout_timing_repetitions);
}

[[nodiscard]] std::vector<bfnew::ShootoutTuning> comparison_tunings() {
  const std::array<std::uint32_t, 1U> fixed{1U};
  const std::vector<bfnew::ShootoutTuning> all =
      bfnew::make_shootout_tunings(256U, fixed);
  std::vector<bfnew::ShootoutTuning> selected;
  for (const bfnew::ShootoutTuning& tuning : all) {
    if (tuning.block_size == 128U &&
        tuning.control_mode ==
            bfnew::ControlMode::persistent_cooperative &&
        tuning.grid_policy == bfnew::GridPolicy::occupancy_derived) {
      selected.push_back(tuning);
    }
  }
  expect(selected.size() == 3U, "comparison cell has exactly three engines");
  return selected;
}

[[nodiscard]] const bfnew::ShootoutTuning& find_tuning(
    const std::span<const bfnew::ShootoutTuning> tunings,
    const std::uint32_t configuration_id) {
  const auto found = std::ranges::find_if(
      tunings,
      [configuration_id](const bfnew::ShootoutTuning& tuning) {
        return tuning.configuration_id == configuration_id;
      });
  if (found == tunings.end()) {
    throw std::logic_error{"test schedule references an unknown tuning"};
  }
  return *found;
}

[[nodiscard]] bfnew::DeviceRunStatus successful_status(
    const std::uint64_t rounds = 3U) {
  return bfnew::DeviceRunStatus{
      0U,
      1U,
      rounds,
      1U,
      0U,
      1U,
      0U,
      1U,
      static_cast<std::uint32_t>(bfnew::DeviceStopReason::converged),
      bfnew::device_error::none,
  };
}

[[nodiscard]] bfnew::ShootoutEvidenceValue measured(const double value) {
  return {bfnew::ShootoutEvidenceState::measured, value};
}

[[nodiscard]] bfnew::ShootoutSample make_sample(
    const bfnew::ShootoutManifest& manifest,
    const std::span<const bfnew::ShootoutTuning> tunings,
    const bfnew::ShootoutScheduleEntry& entry) {
  const bfnew::ShootoutTuning& tuning =
      find_tuning(tunings, entry.configuration_id);
  bfnew::ShootoutSample sample;
  sample.fingerprint = manifest.fingerprint;
  sample.workload = manifest.workload;
  sample.run_kind = entry.run_kind;
  sample.execution_ordinal = entry.execution_ordinal;
  sample.repetition = entry.repetition;
  sample.query_id = entry.query_id;
  sample.configuration_id = entry.configuration_id;
  sample.result.engine_kind = static_cast<std::uint32_t>(tuning.engine);
  sample.result.control_mode =
      static_cast<std::uint32_t>(tuning.control_mode);
  sample.result.status = successful_status(3U + entry.query_id.value() % 4U);
  if (tuning.control_mode == bfnew::ControlMode::persistent_cooperative) {
    sample.cooperative_grid_blocks = 8U;
    sample.cooperative_active_blocks_per_wgp = 2U;
  }
  sample.correctness_passed =
      entry.run_kind != bfnew::ShootoutRunKind::warmup;

  switch (entry.run_kind) {
    case bfnew::ShootoutRunKind::correctness:
      sample.instrumentation = bfnew::InstrumentationLevel::none;
      sample.distances_downloaded = true;
      break;
    case bfnew::ShootoutRunKind::timing: {
      sample.instrumentation = bfnew::InstrumentationLevel::none;
      const double base =
          1.0 + static_cast<double>(entry.query_id.value() - 10U) +
          static_cast<double>(entry.repetition) * 0.25 +
          static_cast<double>(static_cast<std::uint32_t>(tuning.engine)) * 0.1;
      sample.timing.preparation_gpu_milliseconds = measured(base * 0.1);
      sample.timing.sssp_device_timeline_milliseconds = measured(base * 0.7);
      sample.timing.result_transfer_gpu_milliseconds = measured(base * 0.05);
      sample.timing.end_to_end_wall_milliseconds = measured(base);
      break;
    }
    case bfnew::ShootoutRunKind::algorithm_counters:
      sample.instrumentation = bfnew::InstrumentationLevel::debug;
      sample.result.work.edges_examined = 100U + entry.query_id.value();
      sample.result.work.successful_decreases = 20U;
      sample.result.work.active_vertices = 30U;
      sample.result.work.maximum_queue_size = 9U;
      sample.result.work.atomic_attempts =
          tuning.engine == bfnew::EngineKind::jacobi_pull ? 0U : 100U;
      sample.result.work.successful_atomic_updates =
          tuning.engine == bfnew::EngineKind::jacobi_pull ? 0U : 20U;
      sample.result.work.kernel_dispatches = 7U;
      sample.result.work.host_synchronizations = 4U;
      sample.result.work.controller_copies = 3U;
      break;
    case bfnew::ShootoutRunKind::warmup:
      sample.instrumentation = bfnew::InstrumentationLevel::none;
      break;
    case bfnew::ShootoutRunKind::trace:
      sample.instrumentation = bfnew::InstrumentationLevel::none;
      sample.profiler_provenance = {
          1000U + entry.execution_ordinal, 0U, true};
      sample.profiler.gpu_active_milliseconds = measured(
          0.5 + static_cast<double>(entry.query_id.value() - 10U));
      break;
    case bfnew::ShootoutRunKind::pmc:
      sample.instrumentation = bfnew::InstrumentationLevel::none;
      sample.profiler_provenance = {
          2000U + entry.execution_ordinal, 77U, true};
      sample.profiler.l2_hit_percent = measured(72.0);
      sample.profiler.l2_read_bytes = measured(4096.0);
      sample.profiler.occupancy_percent = measured(50.0);
      sample.profiler.waves = measured(12.0);
      break;
  }
  return sample;
}

[[nodiscard]] std::vector<bfnew::ShootoutSample> make_stage(
    const bfnew::ShootoutManifest& manifest,
    const std::span<const bfnew::ShootoutTuning> tunings,
    const bfnew::ShootoutRunKind kind,
    const std::uint32_t repetitions,
    const std::uint64_t seed) {
  const std::vector<bfnew::ShootoutScheduleEntry> schedule =
      kind == bfnew::ShootoutRunKind::correctness
          ? bfnew::make_grouped_shootout_correctness_schedule(
                manifest, tunings)
          : bfnew::make_interleaved_shootout_schedule(
                manifest, tunings, kind, repetitions, seed);
  std::vector<bfnew::ShootoutSample> samples;
  samples.reserve(schedule.size());
  for (const bfnew::ShootoutScheduleEntry& entry : schedule) {
    samples.push_back(make_sample(manifest, tunings, entry));
  }
  return samples;
}

void test_matrix_and_configuration_resolution() {
  const std::array<std::uint32_t, 3U> fixed{1U, 2U, 4U};
  const std::vector<bfnew::ShootoutTuning> tunings =
      bfnew::make_shootout_tunings(512U, fixed);
  expect(tunings.size() == 90U, "canonical Phase 12 tuning matrix size");
  std::set<std::uint32_t> ids;
  std::map<bfnew::EngineKind, std::size_t> engine_counts;
  for (const bfnew::ShootoutTuning& tuning : tunings) {
    expect(ids.insert(tuning.configuration_id).second, "unique tuning ID");
    ++engine_counts[tuning.engine];
  }
  expect(
      engine_counts[bfnew::EngineKind::jacobi_pull] == 30U &&
          engine_counts[bfnew::EngineKind::dense_chaotic_push] == 30U &&
          engine_counts[bfnew::EngineKind::frontier_push] == 30U,
      "all three engines retain the same configuration surface");

  std::vector<bfnew::ShootoutKernelLimit> limits;
  for (const bfnew::EngineKind engine : {
           bfnew::EngineKind::jacobi_pull,
           bfnew::EngineKind::dense_chaotic_push,
           bfnew::EngineKind::frontier_push}) {
    for (const std::uint32_t block : {128U, 256U, 512U}) {
      limits.push_back({engine, block, 2U, true, true});
    }
  }
  const std::vector<bfnew::ShootoutConfigurationDecision> decisions =
      bfnew::resolve_shootout_configurations(tunings, limits);
  expect(decisions.size() == tunings.size(), "one legality row per tuning");
  expect(
      std::ranges::count_if(
          decisions,
          [](const bfnew::ShootoutConfigurationDecision& decision) {
            return decision.rejection ==
                   bfnew::ShootoutConfigurationRejection::
                       illegal_persistent_residency;
          }) == 9,
      "fixed four-block residency is retained as explicitly unsupported");
  expect(
      std::ranges::count_if(
          decisions,
          [](const bfnew::ShootoutConfigurationDecision& decision) {
            return decision.rejection ==
                   bfnew::ShootoutConfigurationRejection::none;
          }) == 81,
      "all legal configurations remain runnable");

  const std::array asymmetric_tunings{
      bfnew::ShootoutTuning{
          1001U,
          bfnew::EngineKind::jacobi_pull,
          bfnew::ControlMode::per_round_host_poll,
          1U,
          128U,
          bfnew::GridPolicy::occupancy_derived,
          0U,
          64U},
      bfnew::ShootoutTuning{
          1002U,
          bfnew::EngineKind::jacobi_pull,
          bfnew::ControlMode::persistent_cooperative,
          1U,
          128U,
          bfnew::GridPolicy::occupancy_derived,
          0U,
          64U}};
  const std::array ordinary_only_limit{bfnew::ShootoutKernelLimit{
      bfnew::EngineKind::jacobi_pull, 128U, 0U, true, false}};
  const auto ordinary_only = bfnew::resolve_shootout_configurations(
      asymmetric_tunings, ordinary_only_limit);
  expect(
      ordinary_only[0].rejection ==
              bfnew::ShootoutConfigurationRejection::none &&
          ordinary_only[1].rejection ==
              bfnew::ShootoutConfigurationRejection::
                  illegal_persistent_block_size,
      "ordinary and persistent kernel legality are independent");
  const std::array persistent_only_limit{bfnew::ShootoutKernelLimit{
      bfnew::EngineKind::jacobi_pull, 128U, 2U, false, true}};
  const auto persistent_only = bfnew::resolve_shootout_configurations(
      asymmetric_tunings, persistent_only_limit);
  expect(
      persistent_only[0].rejection ==
              bfnew::ShootoutConfigurationRejection::
                  illegal_ordinary_block_size &&
          persistent_only[1].rejection ==
              bfnew::ShootoutConfigurationRejection::none,
      "persistent legality does not imply ordinary-kernel legality");
}

void test_logic_manifest_selection() {
  std::vector<bfnew::ShootoutQueryFeatures> candidates;
  candidates.reserve(1200U);
  for (std::uint32_t query = 0U; query < 1200U; ++query) {
    candidates.push_back(features(query, 31U));
  }
  const bfnew::ShootoutWorkloadIdentity workload{
      bfnew::ShootoutWorkloadKind::logicnets_jscl,
      0x4c4f4749434e4554ULL,
      "logicnets_jscl-padding1"};
  const bfnew::ShootoutManifest first =
      bfnew::select_logicnets_shootout_manifest(
          fingerprint(candidates.size()),
          workload,
          candidates,
          1000U,
          20260824U,
          shootout_order_seed,
          shootout_warmups,
          shootout_timing_repetitions);
  const bfnew::ShootoutManifest repeat =
      bfnew::select_logicnets_shootout_manifest(
          fingerprint(candidates.size()),
          workload,
          candidates,
          1000U,
          20260824U,
          shootout_order_seed,
          shootout_warmups,
          shootout_timing_repetitions);
  expect(first == repeat, "logicnets stratification is deterministic");
  expect(first.entries.size() == 1000U, "exactly 1000 logic queries selected");
  std::array<std::set<std::uint8_t>,
             bfnew::shootout_stratification_dimensions>
      observed_bins;
  for (const bfnew::ShootoutManifestEntry& entry : first.entries) {
    for (std::size_t dimension = 0U;
         dimension < entry.quantile_bins.size();
         ++dimension) {
      observed_bins[dimension].insert(entry.quantile_bins[dimension]);
    }
  }
  expect(
      std::ranges::all_of(observed_bins, [](const auto& bins) {
        return bins.size() >= 2U;
      }),
      "selection retains variation in all five stratification dimensions");

  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::select_logicnets_shootout_manifest(
            fingerprint(candidates.size()),
            workload,
            candidates,
            999U,
            20260824U,
            shootout_order_seed,
            shootout_warmups,
            shootout_timing_repetitions));
      },
      "logicnets count below 1000 is rejected");
  candidates.resize(999U);
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::select_logicnets_shootout_manifest(
            fingerprint(candidates.size()),
            workload,
            candidates,
            1000U,
            20260824U,
            shootout_order_seed,
            shootout_warmups,
            shootout_timing_repetitions));
      },
      "corpus with fewer than 1000 eligible queries is rejected");
}

void test_interleaving_balance() {
  const bfnew::ShootoutManifest manifest = make_small_manifest();
  const std::vector<bfnew::ShootoutTuning> tunings = comparison_tunings();
  const auto first = bfnew::make_interleaved_shootout_schedule(
      manifest,
      tunings,
      bfnew::ShootoutRunKind::timing,
      6U,
      0xabcdefU);
  const auto repeat = bfnew::make_interleaved_shootout_schedule(
      manifest,
      tunings,
      bfnew::ShootoutRunKind::timing,
      6U,
      0xabcdefU);
  expect(first == repeat, "interleaved execution order is deterministic");
  expect(
      first.size() == manifest.entries.size() * tunings.size() * 6U,
      "interleaver emits every query/configuration/repetition once");

  std::map<std::pair<std::uint32_t, bfnew::EngineKind>,
           std::array<std::uint32_t, 3U>>
      positions;
  for (std::size_t offset = 0U; offset < first.size(); offset += 3U) {
    const bfnew::QueryId query = first[offset].query_id;
    for (std::size_t slot = 0U; slot < 3U; ++slot) {
      expect(
          first[offset + slot].query_id == query,
          "three-engine comparison cell remains contiguous");
      const auto engine =
          find_tuning(tunings, first[offset + slot].configuration_id).engine;
      ++positions[{query.value(), engine}][slot];
    }
  }
  for (const auto& [identity, counts] : positions) {
    static_cast<void>(identity);
    expect(
        counts == std::array<std::uint32_t, 3U>{2U, 2U, 2U},
        "each engine occupies each order slot twice over six repetitions");
  }
}

void test_evidence_gates_and_round_trips() {
  const bfnew::ShootoutManifest manifest = make_small_manifest();
  const std::vector<bfnew::ShootoutTuning> tunings = comparison_tunings();
  const bfnew::ShootoutConfigurationCatalog catalog{
      manifest.fingerprint, manifest.workload, tunings};
  bfnew::validate_shootout_configuration_catalog(catalog);

  const auto correctness = make_stage(
      manifest,
      tunings,
      bfnew::ShootoutRunKind::correctness,
      1U,
      shootout_order_seed);
  const auto timing =
      make_stage(
          manifest,
          tunings,
          bfnew::ShootoutRunKind::timing,
          4U,
          shootout_order_seed);
  auto counters = make_stage(
      manifest,
      tunings,
      bfnew::ShootoutRunKind::algorithm_counters,
      1U,
      shootout_order_seed);
  const auto trace =
      make_stage(
          manifest,
          tunings,
          bfnew::ShootoutRunKind::trace,
          1U,
          shootout_order_seed);
  const auto pmc =
      make_stage(
          manifest,
          tunings,
          bfnew::ShootoutRunKind::pmc,
          1U,
          shootout_order_seed);
  const auto warmups = make_stage(
      manifest,
      tunings,
      bfnew::ShootoutRunKind::warmup,
      shootout_warmups,
      shootout_order_seed);
  bfnew::validate_shootout_samples(
      manifest, tunings, warmups, bfnew::ShootoutRunKind::warmup);
  const auto too_few_warmups = make_stage(
      manifest,
      tunings,
      bfnew::ShootoutRunKind::warmup,
      shootout_warmups - 1U,
      shootout_order_seed);
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            too_few_warmups,
            bfnew::ShootoutRunKind::warmup);
      },
      "a complete but truncated warmup rectangle is rejected");

  bfnew::require_complete_correctness_gate(
      manifest, tunings, correctness, timing);
  bfnew::require_complete_correctness_gate(
      manifest, tunings, correctness, counters);
  bfnew::require_complete_correctness_gate(
      manifest, tunings, correctness, trace);
  bfnew::require_complete_correctness_gate(
      manifest, tunings, correctness, pmc);

  const std::string manifest_text =
      bfnew::serialize_shootout_manifest_tsv(manifest);
  expect(
      bfnew::deserialize_shootout_manifest_tsv(manifest_text) == manifest,
      "manifest TSV round trip is exact");
  const std::string catalog_text = bfnew::serialize_shootout_catalog_tsv(catalog);
  expect(
      bfnew::deserialize_shootout_catalog_tsv(catalog_text) == catalog,
      "configuration catalog TSV round trip is exact");
  const std::string timing_text = bfnew::serialize_shootout_samples_tsv(
      manifest, tunings, timing, bfnew::ShootoutRunKind::timing);
  const std::vector<bfnew::ShootoutSample> timing_round_trip =
      bfnew::deserialize_shootout_samples_tsv(
          manifest,
          tunings,
          timing_text,
          bfnew::ShootoutRunKind::timing);
  expect(
      bfnew::serialize_shootout_samples_tsv(
          manifest,
          tunings,
          timing_round_trip,
          bfnew::ShootoutRunKind::timing) == timing_text,
      "sample TSV round trip preserves every fixed-width field and evidence state");

  for (bfnew::ShootoutSample& sample : counters) {
    const bfnew::ShootoutTuning& tuning =
        find_tuning(tunings, sample.configuration_id);
    if (tuning.control_mode ==
            bfnew::ControlMode::persistent_cooperative &&
        tuning.grid_policy == bfnew::GridPolicy::occupancy_derived) {
      sample.cooperative_grid_blocks = 4U;
      sample.cooperative_active_blocks_per_wgp = 1U;
    }
  }

  const bfnew::ShootoutCampaignReport report =
      bfnew::summarize_shootout_campaign(
          manifest, tunings, correctness, timing, counters, trace, pmc);
  bfnew::validate_shootout_report(report);
  expect(report.summaries.size() == tunings.size(), "one summary per tuning");
  expect(
      std::ranges::all_of(
          report.summaries,
          [](const bfnew::ShootoutTuningSummary& summary) {
            return summary.tuning.control_mode !=
                       bfnew::ControlMode::persistent_cooperative ||
                   (summary.cooperative_grid_blocks == 8U &&
                    summary.cooperative_active_blocks_per_wgp == 2U);
          }),
      "None launch dimensions remain authoritative when Debug occupancy differs");
  expect(
      std::ranges::all_of(report.answers, [](const auto& answer) {
        return answer.state == bfnew::ShootoutConclusionState::pending &&
               answer.evidence.empty();
      }),
      "bounded CPU metadata never manufactures measured performance answers");
  expect(
      report.recommendations.size() == 1U &&
          report.recommendations.front().state ==
              bfnew::ShootoutConclusionState::pending &&
          report.recommendations.front().toggles_remain_configurable,
      "recommended defaults remain pending and every toggle is retained");
  const std::string report_json = bfnew::serialize_shootout_report_json(report);
  expect(
      report_json.find("pending measured GPU campaign") != std::string::npos,
      "report labels recommendations as pending measurement");
  for (const std::string_view required_key : {
           "\"algorithm_work\"",
           "\"atomic_attempts\"",
           "\"kernel_dispatches\"",
           "\"gpu_active_milliseconds\"",
           "\"memory_unit_busy_percent\"",
           "\"selection_seed\"",
           "\"order_seed\"",
           "\"timing_repetitions\"",
           "\"quantile_bins\"",
           "\"cooperative_grid_blocks\"",
           "\"host_checks\"",
           "\"overflow_events\"",
           "\"empty_frontier_rounds\"",
           "\"small_frontier_rounds\"",
           "\"supplied_evidence\"",
           "\"long_tail\"",
           "\"features\""}) {
    expect(
        report_json.find(required_key) != std::string::npos,
        "serialized report retains every required metric and tail feature");
  }
  bfnew::ShootoutCampaignReport resolved_report = report;
  resolved_report.answers[0].state = bfnew::ShootoutConclusionState::measured;
  resolved_report.answers[0].configuration_ids = {
      timing.front().configuration_id};
  if (counters.front().configuration_id != timing.front().configuration_id) {
    resolved_report.answers[0].configuration_ids.push_back(
        counters.front().configuration_id);
  }
  resolved_report.answers[0].evidence = {
      {bfnew::ShootoutRunKind::timing,
       timing.front().execution_ordinal,
       0U,
       0U,
       timing.front().query_id,
       timing.front().configuration_id},
      {bfnew::ShootoutRunKind::algorithm_counters,
       counters.front().execution_ordinal,
       0U,
       0U,
       counters.front().query_id,
       counters.front().configuration_id}};
  resolved_report.answers[0].conclusion = "bounded evidence-resolution test";
  bfnew::validate_shootout_report(resolved_report);
  const auto other_tuning = std::ranges::find_if(
      tunings,
      [&](const bfnew::ShootoutTuning& tuning) {
        return std::ranges::find(
                   resolved_report.answers[0].configuration_ids,
                   tuning.configuration_id) ==
               resolved_report.answers[0].configuration_ids.end();
      });
  expect(other_tuning != tunings.end(), "evidence test has an alternate tuning");
  if (other_tuning != tunings.end()) {
    const auto other_timing = std::ranges::find_if(
        timing,
        [&](const bfnew::ShootoutSample& sample) {
          return sample.configuration_id == other_tuning->configuration_id;
        });
    const auto other_counter = std::ranges::find_if(
        counters,
        [&](const bfnew::ShootoutSample& sample) {
          return sample.configuration_id == other_tuning->configuration_id;
        });
    bfnew::ShootoutCampaignReport cross_configuration = resolved_report;
    cross_configuration.answers[0].evidence = {
        {bfnew::ShootoutRunKind::timing,
         other_timing->execution_ordinal,
         0U,
         0U,
         other_timing->query_id,
         other_timing->configuration_id},
        {bfnew::ShootoutRunKind::algorithm_counters,
         other_counter->execution_ordinal,
         0U,
         0U,
         other_counter->query_id,
         other_counter->configuration_id}};
    expect_throws<std::invalid_argument>(
        [&] { bfnew::validate_shootout_report(cross_configuration); },
        "a conclusion cannot cite evidence from an unnamed configuration");
  }
  resolved_report.answers[0].evidence.back().execution_ordinal += 100000U;
  expect_throws<std::invalid_argument>(
      [&] { bfnew::validate_shootout_report(resolved_report); },
      "structured conclusions cannot cite evidence absent from the campaign");

  const std::array counter_subset_tunings{tunings.front()};
  const auto counter_subset = make_stage(
      manifest,
      counter_subset_tunings,
      bfnew::ShootoutRunKind::algorithm_counters,
      1U,
      shootout_order_seed);
  bfnew::validate_shootout_samples(
      manifest,
      tunings,
      counter_subset,
      bfnew::ShootoutRunKind::algorithm_counters);
  const bfnew::ShootoutCampaignReport subset_report =
      bfnew::summarize_shootout_campaign(
          manifest,
          tunings,
          correctness,
          timing,
          counter_subset,
          trace,
          pmc);
  expect(
      subset_report.summaries.front().algorithm_counters_state ==
              bfnew::ShootoutEvidenceState::measured &&
          std::ranges::all_of(
              std::span{subset_report.summaries}.subspan(1U),
              [](const bfnew::ShootoutTuningSummary& summary) {
                return summary.algorithm_counters_state ==
                           bfnew::ShootoutEvidenceState::unavailable &&
                       summary.useful_decrease_ratio.state ==
                           bfnew::ShootoutEvidenceState::unavailable;
              }),
      "Debug-unsupported catalog tunings retain timing with unavailable counters");
  std::vector<bfnew::ShootoutSample> partial_counter_subset = counter_subset;
  partial_counter_subset.pop_back();
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            partial_counter_subset,
            bfnew::ShootoutRunKind::algorithm_counters);
      },
      "counter subset rejects a partial query/configuration rectangle");

  std::vector<bfnew::ShootoutSample> incomplete = correctness;
  incomplete.pop_back();
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::require_complete_correctness_gate(
            manifest, tunings, incomplete, timing);
      },
      "timing is rejected before complete correctness");
  std::vector<bfnew::ShootoutSample> polluted_timing = timing;
  polluted_timing.front().instrumentation = bfnew::InstrumentationLevel::debug;
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            polluted_timing,
            bfnew::ShootoutRunKind::timing);
      },
      "final timing rejects instrumentation");
  polluted_timing = timing;
  polluted_timing.front().distances_downloaded = true;
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            polluted_timing,
            bfnew::ShootoutRunKind::timing);
      },
      "final timing rejects graph-sized distance downloads");
  polluted_timing = timing;
  polluted_timing.front().profiler.l2_hit_percent = measured(50.0);
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            polluted_timing,
            bfnew::ShootoutRunKind::timing);
      },
      "profiler evidence cannot enter ordinary timing samples");
  polluted_timing = timing;
  polluted_timing.front().timing.preparation_gpu_milliseconds = {};
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            polluted_timing,
            bfnew::ShootoutRunKind::timing);
      },
      "timing requires measured preparation time");
  polluted_timing = timing;
  polluted_timing.front().timing.result_transfer_gpu_milliseconds = {};
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            polluted_timing,
            bfnew::ShootoutRunKind::timing);
      },
      "timing requires measured result-transfer time");
  polluted_timing = timing;
  polluted_timing[1U].execution_ordinal =
      polluted_timing.front().execution_ordinal;
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            polluted_timing,
            bfnew::ShootoutRunKind::timing);
      },
      "ordinary stages retain globally unique execution ordinals");
  polluted_timing = timing;
  polluted_timing.front().execution_ordinal =
      timing.back().execution_ordinal + 1000U;
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            polluted_timing,
            bfnew::ShootoutRunKind::timing);
      },
      "ordinary samples must match the persisted order seed schedule");

  std::vector<bfnew::ShootoutSample> truncated_timing;
  std::ranges::copy_if(
      timing,
      std::back_inserter(truncated_timing),
      [](const bfnew::ShootoutSample& sample) {
        return sample.repetition + 1U < shootout_timing_repetitions;
      });
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            truncated_timing,
            bfnew::ShootoutRunKind::timing);
      },
      "a complete but truncated timing rectangle is rejected");

  std::vector<bfnew::ShootoutSample> trace_replays{trace.front()};
  bfnew::ShootoutSample second_trace_pass = trace.front();
  second_trace_pass.profiler_provenance.pass_id += 100000U;
  trace_replays.push_back(second_trace_pass);
  bfnew::validate_shootout_samples(
      manifest,
      tunings,
      trace_replays,
      bfnew::ShootoutRunKind::trace);
  const std::string trace_replay_text = bfnew::serialize_shootout_samples_tsv(
      manifest,
      tunings,
      trace_replays,
      bfnew::ShootoutRunKind::trace);
  expect(
      bfnew::serialize_shootout_samples_tsv(
          manifest,
          tunings,
          bfnew::deserialize_shootout_samples_tsv(
              manifest,
              tunings,
              trace_replay_text,
              bfnew::ShootoutRunKind::trace),
          bfnew::ShootoutRunKind::trace) == trace_replay_text,
      "trace replays preserve one stable case ordinal across distinct passes");

  std::vector<bfnew::ShootoutSample> duplicate_trace = trace_replays;
  duplicate_trace.push_back(trace_replays.front());
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            duplicate_trace,
            bfnew::ShootoutRunKind::trace);
      },
      "an exact profiler replay duplicate is rejected");

  bfnew::ShootoutSample first_counter_set = pmc.front();
  first_counter_set.profiler = {};
  first_counter_set.profiler_provenance = {4001U, 71U, true};
  first_counter_set.profiler.l2_hit_percent = measured(72.0);
  bfnew::ShootoutSample second_counter_set = first_counter_set;
  second_counter_set.profiler = {};
  second_counter_set.profiler_provenance = {4002U, 72U, true};
  second_counter_set.profiler.waves = measured(12.0);
  const std::array distinct_counter_sets{
      first_counter_set, second_counter_set};
  bfnew::validate_shootout_samples(
      manifest,
      tunings,
      distinct_counter_sets,
      bfnew::ShootoutRunKind::pmc);

  std::vector<bfnew::ShootoutSample> incompatible = pmc;
  bfnew::ShootoutSample conflicting = incompatible.front();
  conflicting.profiler_provenance.pass_id += 100000U;
  conflicting.profiler_provenance.counter_set_id = 88U;
  incompatible.push_back(conflicting);
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::validate_shootout_samples(
            manifest,
            tunings,
            incompatible,
            bfnew::ShootoutRunKind::pmc);
      },
      "one profiler metric cannot be joined across incompatible counter sets");
}

void test_distribution() {
  std::vector<double> values;
  values.reserve(100U);
  for (std::uint32_t value = 1U; value <= 100U; ++value) {
    values.push_back(static_cast<double>(value));
  }
  const bfnew::ShootoutDistribution distribution =
      bfnew::shootout_distribution(values);
  expect(
      distribution.count == 100U && distribution.minimum == 1.0 &&
          distribution.p50 == 50.0 && distribution.p95 == 95.0 &&
          distribution.p99 == 99.0 && distribution.maximum == 100.0 &&
          distribution.mean == 50.5,
      "nearest-rank P50/P95/P99 distribution is deterministic");
  expect_throws<std::invalid_argument>(
      [] {
        const std::array invalid{
            1.0, std::numeric_limits<double>::quiet_NaN()};
        static_cast<void>(bfnew::shootout_distribution(invalid));
      },
      "nonfinite timing samples are rejected");
}

[[nodiscard]] bool bitwise_equal(
    const std::span<const float> left,
    const std::span<const float> right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (std::bit_cast<std::uint32_t>(left[index]) !=
        std::bit_cast<std::uint32_t>(right[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::vector<float> bounded_oracle(
    const bfnew::WeightedGraph& graph,
    const bfnew::RouteQuery& query) {
  const bfnew::InducedQueryGraph induced =
      bfnew::build_induced_query_graph(graph, query);
  const bfnew::SsspResult local =
      bfnew::dijkstra_oracle(induced.graph, induced.sources);
  std::vector<float> global(
      graph.vertex_count(), std::numeric_limits<float>::infinity());
  for (std::size_t local_vertex = 0U;
       local_vertex < induced.local_to_global.size();
       ++local_vertex) {
    global[induced.local_to_global[local_vertex].value()] =
        local.distances[local_vertex];
  }
  return global;
}

void test_portable_three_engine_agreement() {
  const bfnew::test::JacobiFixtureCase fixture =
      bfnew::test::make_phase5_core_jacobi_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const bfnew::TileRunLayout64 runs = bfnew::build_tile_run_layout(graph);
  const bfnew::DeviceGraphLayout32 layout =
      bfnew::build_device_graph_layout32(graph, runs);
  std::vector<bfnew::LaneMask> tile_masks(graph.tile_coordinates().size(), 0U);
  for (const bfnew::TileId tile : fixture.query.selected_tiles) {
    tile_masks[tile.value()] = 1U;
  }
  bfnew::TileRunLaneMasks run_masks;
  bfnew::compute_tile_run_lane_masks(graph, runs, tile_masks, run_masks);

  bfnew::GpuRunOptions options;
  options.control_mode = bfnew::ControlMode::chunked_host_poll;
  options.rounds_per_chunk = 4U;
  options.block_size = 128U;
  options.maximum_rounds = 128U;
  options.instrumentation = bfnew::InstrumentationLevel::none;
  options.engine = bfnew::EngineKind::jacobi_pull;
  const bfnew::HostJacobiRunResult jacobi = bfnew::run_host_jacobi_pull(
      layout, fixture.query, tile_masks, run_masks.csc_run_masks, options);
  options.engine = bfnew::EngineKind::dense_chaotic_push;
  const bfnew::HostDenseRunResult dense = bfnew::run_host_dense_chaotic_push(
      layout, fixture.query, tile_masks, run_masks.csr_run_masks, options);
  options.engine = bfnew::EngineKind::frontier_push;
  const bfnew::HostFrontierRunResult frontier = bfnew::run_host_frontier_push(
      layout, fixture.query, tile_masks, run_masks.csr_run_masks, options);
  const std::vector<float> oracle = bounded_oracle(graph, fixture.query);
  expect(
      bitwise_equal(jacobi.distances, oracle) &&
          bitwise_equal(dense.distances, oracle) &&
          bitwise_equal(frontier.distances, oracle),
      "all three portable standalone engines agree bitwise with bounded Dijkstra");
}

[[nodiscard]] std::string read_file(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error{"cannot read HIP source contract: " + path};
  }
  return std::string{
      std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

void test_status_only_source_contract() {
#if defined(BFNEW_PHASE12_JACOBI_HIP_SOURCE_PATH) && \
    defined(BFNEW_PHASE12_DENSE_HIP_SOURCE_PATH) && \
    defined(BFNEW_PHASE12_FRONTIER_HIP_SOURCE_PATH)
  for (const std::string& path : {
           std::string{BFNEW_PHASE12_JACOBI_HIP_SOURCE_PATH},
           std::string{BFNEW_PHASE12_DENSE_HIP_SOURCE_PATH},
           std::string{BFNEW_PHASE12_FRONTIER_HIP_SOURCE_PATH}}) {
    const std::string source = read_file(path);
    expect(
        source.find("run_status_only") != std::string::npos,
        "each HIP engine exposes a measured status-only path");
    expect(
        source.find("return run_with_distances(query, options).result;") ==
            std::string::npos,
        "common run path does not download graph-sized labels");
  }
#endif
#if defined(BFNEW_PHASE12_EXECUTOR_HIP_SOURCE_PATH) && \
    defined(BFNEW_PHASE12_RUNNER_SOURCE_PATH)
  const std::string executor =
      read_file(BFNEW_PHASE12_EXECUTOR_HIP_SOURCE_PATH);
  const std::string runner = read_file(BFNEW_PHASE12_RUNNER_SOURCE_PATH);
  expect(
      executor.find("ControlMode::per_round_host_poll") != std::string::npos &&
          executor.find("ControlMode::persistent_cooperative") !=
              std::string::npos &&
          executor.find("run_with_selected_distances") != std::string::npos,
      "shootout probes ordinary/persistent kernels and correctness reads only "
      "selected ranges");
  expect(
      executor.find("bfnew_shootout_profile_range_begin_marker_kernel") !=
              std::string::npos &&
          executor.find("bfnew_shootout_profile_range_end_marker_kernel") !=
              std::string::npos,
      "profile replay exposes recognizable begin/end marker kernels");
  expect(
      runner.find("prescreen_seed_domain") != std::string::npos &&
          runner.find("make_grouped_shootout_correctness_schedule") !=
              std::string::npos &&
          runner.find("counter-configuration-decisions.v1.tsv") !=
              std::string::npos &&
          runner.find("profile-range-ledger.v1.tsv") != std::string::npos &&
          runner.find("build_induced_query_graph") == std::string::npos,
      "runner bounds pilot/oracle work and records counter/profile legality");
#endif
}

}  // namespace

int main() {
  test_matrix_and_configuration_resolution();
  test_logic_manifest_selection();
  test_interleaving_balance();
  test_evidence_gates_and_round_trips();
  test_distribution();
  test_portable_three_engine_agreement();
  test_status_only_source_contract();
  if (failures != 0) {
    std::cerr << failures << " Phase 12 check(s) failed\n";
    return 1;
  }
  std::cout
      << "Phase 12 bounded shootout planning/reporting checks passed; "
         "no GPU timing evidence was collected.\n";
  return 0;
}
