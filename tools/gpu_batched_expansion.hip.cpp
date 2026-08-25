#include "bfnew/batched_expansion.hpp"
#include "bfnew/compact_paths.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/fpga_interchange.hpp"
#include "bfnew/hip/batched_expansion.hpp"
#include "bfnew/hip/runtime.hpp"
#include "bfnew/no_congestion_pipeline.hpp"
#include "bfnew/spatial.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <ranges>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::filesystem::path graph_path;
  std::filesystem::path queries_path;
  std::filesystem::path output_path;
  bool has_engine{};
  bfnew::EngineKind engine{bfnew::EngineKind::jacobi_pull};
  bool has_schedule{};
  bfnew::ExpansionScheduleKind schedule{
      bfnew::ExpansionScheduleKind::unspecified};
  std::uint32_t tile_width{8U};
  std::uint32_t tile_height{8U};
  std::uint32_t lane_width{32U};
  std::uint32_t fixed_ring_size{};
  std::uint32_t hybrid_small_expansions{};
  std::uint32_t maximum_expansions{4U};
  bfnew::ExpansionTerminalPolicy terminal_policy{
      bfnew::ExpansionTerminalPolicy::full_region_fallback};
  bool has_terminal_policy{};
  bool has_maximum_expansions{};
  bool has_planner{};
  bool has_control{};
  bfnew::ControlMode control{bfnew::ControlMode::chunked_host_poll};
  std::uint32_t rounds_per_chunk{8U};
  std::uint32_t block_size{256U};
  bfnew::GridPolicy grid_policy{bfnew::GridPolicy::occupancy_derived};
  std::uint32_t blocks_per_wgp{};
  std::uint64_t maximum_rounds{};
  std::uint32_t enable_per_lane_convergence{1U};
  bfnew::hip::BatchedExpansionTransferMode transfer_mode{
      bfnew::hip::BatchedExpansionTransferMode::compact_status};
  bool has_transfer{};
  bool has_reconstruction{};
  bfnew::InstrumentationLevel instrumentation{
      bfnew::InstrumentationLevel::none};
  std::uint32_t warmups{};
  std::uint32_t repetitions{};
  std::uint32_t quality_sample_count{};
  std::uint64_t quality_seed{};
  bool has_warmups{};
  bool has_repetitions{};
  bool has_quality_sample_count{};
  bool has_quality_seed{};
};

[[nodiscard]] std::uint32_t parse_u32(
    const std::string& text,
    const char* const name) {
  std::size_t consumed = 0U;
  const unsigned long value = std::stoul(text, &consumed, 10);
  if (consumed != text.size() || !std::in_range<std::uint32_t>(value)) {
    throw std::invalid_argument{
        std::string{name} + " must be a 32-bit unsigned integer"};
  }
  return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::uint64_t parse_u64(
    const std::string& text,
    const char* const name) {
  if (text.empty() || text.front() == '-') {
    throw std::invalid_argument{
        std::string{name} + " must be a 64-bit unsigned integer"};
  }
  std::size_t consumed = 0U;
  const unsigned long long value = std::stoull(text, &consumed, 10);
  if (consumed != text.size()) {
    throw std::invalid_argument{
        std::string{name} + " must be a 64-bit unsigned integer"};
  }
  return static_cast<std::uint64_t>(value);
}

[[nodiscard]] bfnew::EngineKind parse_engine(const std::string& text) {
  if (text == "jacobi") {
    return bfnew::EngineKind::jacobi_pull;
  }
  if (text == "dense") {
    return bfnew::EngineKind::dense_chaotic_push;
  }
  if (text == "frontier") {
    return bfnew::EngineKind::frontier_push;
  }
  throw std::invalid_argument{
      "--engine must be jacobi, dense, or frontier"};
}

[[nodiscard]] bfnew::ExpansionScheduleKind parse_schedule(
    const std::string& text) {
  if (text == "one-ring") {
    return bfnew::ExpansionScheduleKind::one_geometric_ring;
  }
  if (text == "fixed-ring") {
    return bfnew::ExpansionScheduleKind::fixed_larger_ring;
  }
  if (text == "doubling") {
    return bfnew::ExpansionScheduleKind::doubling_xy_margins;
  }
  if (text == "hybrid") {
    return bfnew::ExpansionScheduleKind::hybrid_small_then_doubling;
  }
  throw std::invalid_argument{
      "--schedule must be one-ring, fixed-ring, doubling, or hybrid"};
}

[[nodiscard]] bfnew::ControlMode parse_control(const std::string& text) {
  if (text == "persistent") {
    return bfnew::ControlMode::persistent_cooperative;
  }
  if (text == "chunked") {
    return bfnew::ControlMode::chunked_host_poll;
  }
  if (text == "per-round") {
    return bfnew::ControlMode::per_round_host_poll;
  }
  throw std::invalid_argument{
      "--control must be persistent, chunked, or per-round"};
}

[[nodiscard]] bfnew::GridPolicy parse_grid_policy(const std::string& text) {
  if (text == "occupancy") {
    return bfnew::GridPolicy::occupancy_derived;
  }
  if (text == "fixed") {
    return bfnew::GridPolicy::fixed_blocks_per_wgp;
  }
  throw std::invalid_argument{"--grid must be occupancy or fixed"};
}

[[nodiscard]] bfnew::InstrumentationLevel parse_instrumentation(
    const std::string& text) {
  if (text == "none") {
    return bfnew::InstrumentationLevel::none;
  }
  if (text == "light") {
    return bfnew::InstrumentationLevel::light;
  }
  if (text == "debug") {
    return bfnew::InstrumentationLevel::debug;
  }
  throw std::invalid_argument{
      "--instrumentation must be none, light, or debug"};
}

void print_help() {
  std::cout
      << "Usage: bfnew_gpu_batched_expansion --graph FILE --queries FILE "
         "--output FILE --engine ENGINE --schedule SCHEDULE [options]\n"
         "Required explicit choices:\n"
         "  --engine jacobi|dense|frontier\n"
         "  --schedule one-ring|fixed-ring|doubling|hybrid\n"
         "  --control persistent|chunked|per-round\n"
         "  --planner overlap-greedy\n"
         "Input/planning:\n"
         "  --tile-width N --tile-height N --lane-width 1|8|16|32\n"
         "  --fixed-ring-size N              (fixed-ring only, N >= 2)\n"
         "  --hybrid-small-expansions N      (hybrid only, N >= 1)\n"
         "  --maximum-expansions N\n"
         "  --terminal-policy fallback|failure\n"
         "Engine control:\n"
         "  --control persistent|chunked|per-round\n"
         "  --rounds-per-chunk N --block-size N\n"
         "  --grid occupancy|fixed --blocks-per-wgp N\n"
         "  --maximum-rounds N               (0 means V + 1)\n"
         "  --per-lane-convergence on|off\n"
         "Transfer/evidence:\n"
         "  --transfer compact|paths|evidence\n"
         "  --instrumentation none|light|debug\n"
         "Phase 18 repeated path evidence (required with --transfer paths):\n"
         "  --warmups N                       (N >= 1; first grows capacity)\n"
         "  --repetitions N                   (N >= 1; pre-grown samples)\n"
         "  --quality-sample-count N          (N >= 1)\n"
         "  --quality-seed N\n"
         "  --reconstruction backtracking\n"
         "Compact/paths require none. Evidence requires light or debug. "
         "Paths enables the Phase 18 no-congestion pipeline and exact compact "
         "path/stage-ledger rows. The output "
         "path must not already exist. Graph upload is completed before the "
         "cold execution; first-workspace growth stays in cold_execution, "
         "and repetitions begin only after the requested warmups.\n";
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&](const char* const option) -> std::string {
      if (index + 1 >= argc) {
        throw std::invalid_argument{std::string{option} + " requires a value"};
      }
      return argv[++index];
    };
    if (argument == "--help") {
      print_help();
      std::exit(EXIT_SUCCESS);
    } else if (argument == "--graph") {
      options.graph_path = value("--graph");
    } else if (argument == "--queries") {
      options.queries_path = value("--queries");
    } else if (argument == "--output") {
      options.output_path = value("--output");
    } else if (argument == "--engine") {
      options.engine = parse_engine(value("--engine"));
      options.has_engine = true;
    } else if (argument == "--schedule") {
      options.schedule = parse_schedule(value("--schedule"));
      options.has_schedule = true;
    } else if (argument == "--planner") {
      if (value("--planner") != "overlap-greedy") {
        throw std::invalid_argument{
            "--planner must be overlap-greedy"};
      }
      options.has_planner = true;
    } else if (argument == "--tile-width") {
      options.tile_width = parse_u32(value("--tile-width"), "tile width");
    } else if (argument == "--tile-height") {
      options.tile_height = parse_u32(value("--tile-height"), "tile height");
    } else if (argument == "--lane-width") {
      options.lane_width = parse_u32(value("--lane-width"), "lane width");
    } else if (argument == "--fixed-ring-size") {
      options.fixed_ring_size =
          parse_u32(value("--fixed-ring-size"), "fixed ring size");
    } else if (argument == "--hybrid-small-expansions") {
      options.hybrid_small_expansions = parse_u32(
          value("--hybrid-small-expansions"),
          "hybrid small-expansion count");
    } else if (argument == "--maximum-expansions") {
      options.maximum_expansions = parse_u32(
          value("--maximum-expansions"), "maximum expansions");
      options.has_maximum_expansions = true;
    } else if (argument == "--terminal-policy") {
      const std::string policy = value("--terminal-policy");
      if (policy == "fallback") {
        options.terminal_policy =
            bfnew::ExpansionTerminalPolicy::full_region_fallback;
      } else if (policy == "failure") {
        options.terminal_policy =
            bfnew::ExpansionTerminalPolicy::explicit_failure;
      } else {
        throw std::invalid_argument{
            "--terminal-policy must be fallback or failure"};
      }
      options.has_terminal_policy = true;
    } else if (argument == "--control") {
      options.control = parse_control(value("--control"));
      options.has_control = true;
    } else if (argument == "--rounds-per-chunk") {
      options.rounds_per_chunk = parse_u32(
          value("--rounds-per-chunk"), "rounds per chunk");
    } else if (argument == "--block-size") {
      options.block_size = parse_u32(value("--block-size"), "block size");
    } else if (argument == "--grid") {
      options.grid_policy = parse_grid_policy(value("--grid"));
    } else if (argument == "--blocks-per-wgp") {
      options.blocks_per_wgp = parse_u32(
          value("--blocks-per-wgp"), "blocks per WGP");
    } else if (argument == "--maximum-rounds") {
      options.maximum_rounds =
          parse_u64(value("--maximum-rounds"), "maximum rounds");
    } else if (argument == "--per-lane-convergence") {
      const std::string enabled = value("--per-lane-convergence");
      if (enabled == "on") {
        options.enable_per_lane_convergence = 1U;
      } else if (enabled == "off") {
        options.enable_per_lane_convergence = 0U;
      } else {
        throw std::invalid_argument{
            "--per-lane-convergence must be on or off"};
      }
    } else if (argument == "--transfer") {
      const std::string transfer = value("--transfer");
      if (transfer == "compact") {
        options.transfer_mode =
            bfnew::hip::BatchedExpansionTransferMode::compact_status;
      } else if (transfer == "paths") {
        options.transfer_mode =
            bfnew::hip::BatchedExpansionTransferMode::compact_paths;
      } else if (transfer == "evidence") {
        options.transfer_mode = bfnew::hip::BatchedExpansionTransferMode::
            status_and_work_evidence;
      } else {
        throw std::invalid_argument{
            "--transfer must be compact, paths, or evidence"};
      }
      options.has_transfer = true;
    } else if (argument == "--instrumentation") {
      options.instrumentation =
          parse_instrumentation(value("--instrumentation"));
    } else if (argument == "--warmups") {
      options.warmups = parse_u32(value("--warmups"), "warmup count");
      options.has_warmups = true;
    } else if (argument == "--repetitions") {
      options.repetitions =
          parse_u32(value("--repetitions"), "repetition count");
      options.has_repetitions = true;
    } else if (argument == "--quality-sample-count") {
      options.quality_sample_count = parse_u32(
          value("--quality-sample-count"), "quality sample count");
      options.has_quality_sample_count = true;
    } else if (argument == "--quality-seed") {
      options.quality_seed =
          parse_u64(value("--quality-seed"), "quality seed");
      options.has_quality_seed = true;
    } else if (argument == "--reconstruction") {
      if (value("--reconstruction") != "backtracking") {
        throw std::invalid_argument{
            "--reconstruction must be backtracking"};
      }
      options.has_reconstruction = true;
    } else {
      throw std::invalid_argument{"unknown option: " + argument};
    }
  }

  if (options.graph_path.empty() || options.queries_path.empty() ||
      options.output_path.empty() || !options.has_engine ||
      !options.has_schedule || !options.has_control ||
      !options.has_planner || !options.has_transfer ||
      !options.has_maximum_expansions || !options.has_terminal_policy) {
    throw std::invalid_argument{
        "--graph, --queries, --output, --engine, --schedule, --control, "
        "--planner, --transfer, --maximum-expansions, and --terminal-policy "
        "are required"};
  }
  if (options.tile_width == 0U || options.tile_height == 0U) {
    throw std::invalid_argument{"tile dimensions must be positive"};
  }
  if (options.lane_width != 1U && options.lane_width != 8U &&
      options.lane_width != 16U && options.lane_width != 32U) {
    throw std::invalid_argument{"--lane-width must be 1, 8, 16, or 32"};
  }
  if (options.grid_policy == bfnew::GridPolicy::occupancy_derived &&
      options.blocks_per_wgp != 0U) {
    throw std::invalid_argument{
        "occupancy grid cannot specify --blocks-per-wgp"};
  }
  if (options.grid_policy == bfnew::GridPolicy::fixed_blocks_per_wgp &&
      options.blocks_per_wgp == 0U) {
    throw std::invalid_argument{
        "fixed grid requires a positive --blocks-per-wgp"};
  }
  if ((options.transfer_mode ==
           bfnew::hip::BatchedExpansionTransferMode::compact_status ||
       options.transfer_mode ==
           bfnew::hip::BatchedExpansionTransferMode::compact_paths) &&
      options.instrumentation != bfnew::InstrumentationLevel::none) {
    throw std::invalid_argument{
        "compact/path transfer requires --instrumentation none"};
  }
  if (options.transfer_mode == bfnew::hip::BatchedExpansionTransferMode::
                                   status_and_work_evidence &&
      options.instrumentation == bfnew::InstrumentationLevel::none) {
    throw std::invalid_argument{
        "evidence transfer requires light or debug instrumentation"};
  }
  if (options.transfer_mode ==
      bfnew::hip::BatchedExpansionTransferMode::compact_paths) {
    if (!options.has_warmups || !options.has_repetitions ||
        !options.has_quality_sample_count || !options.has_quality_seed ||
        !options.has_reconstruction ||
        options.warmups == 0U || options.repetitions == 0U ||
        options.quality_sample_count == 0U) {
      throw std::invalid_argument{
          "path transfer requires explicit positive --warmups, --repetitions, "
          "and --quality-sample-count plus explicit --quality-seed and "
          "--reconstruction backtracking"};
    }
  } else if (options.has_warmups || options.has_repetitions ||
             options.has_quality_sample_count || options.has_quality_seed ||
             options.has_reconstruction) {
    throw std::invalid_argument{
        "warm repetition and quality options require --transfer paths"};
  }
  return options;
}

[[nodiscard]] std::array<std::uint64_t, 2U> fingerprint_file(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error{"cannot fingerprint input: " + path.string()};
  }
  std::uint64_t first = 1469598103934665603ULL;
  std::uint64_t second = 0x9e3779b97f4a7c15ULL;
  std::array<char, 64U * 1024U> buffer{};
  std::uint64_t position = 0U;
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    for (std::streamsize index = 0; index < count; ++index) {
      const auto byte = static_cast<unsigned char>(
          buffer[static_cast<std::size_t>(index)]);
      first = (first ^ byte) * 1099511628211ULL;
      second ^= static_cast<std::uint64_t>(byte) +
                0x9e3779b97f4a7c15ULL + (second << 6U) +
                (second >> 2U) + position;
      ++position;
    }
  }
  if (!input.eof()) {
    throw std::runtime_error{
        "failed while fingerprinting input: " + path.string()};
  }
  return {first, second};
}

struct LoadedInputs {
  bfnew::PartitionedGraph partitioned;
  bfnew::TileRunLayout64 tile_runs;
  bfnew::DeviceGraphLayout32 device_graph;
  bfnew::DeviceGraphFingerprint graph_fingerprint{};
  std::array<std::uint64_t, 2U> graph_file_fingerprint{};
  std::array<std::uint64_t, 2U> query_file_fingerprint{};
  std::vector<bfnew::MappedRouteQuery> mapped_queries;
  std::vector<bfnew::RouteQuery> queries;
};

struct WarmRepetitionSample {
  std::uint64_t host_nanoseconds{};
  std::uint64_t sssp_device_nanoseconds{};
  std::uint64_t reconstruction_device_nanoseconds{};
  std::uint64_t result_transfer_device_nanoseconds{};
  std::uint64_t compact_payload_bytes{};
  std::uint64_t status_bytes{};
  std::uint64_t error_bytes{};
  std::uint64_t total_device_to_host_bytes{};
  std::uint64_t controller_poll_count{};
  std::uint64_t controller_poll_bytes{};
  std::uint64_t overall_device_to_host_bytes{};
};

struct DriverStageLedger {
  std::uint64_t artifact_load_host_nanoseconds{};
  std::uint64_t resident_upload_host_nanoseconds{};
  double resident_upload_device_milliseconds{};
  std::uint64_t cold_execution_host_nanoseconds{};
  std::uint64_t controller_host_nanoseconds{};
  std::uint64_t pipeline_host_nanoseconds{};
  std::vector<WarmRepetitionSample> warm_repetitions;
  std::uint64_t warm_p50_host_nanoseconds{};
  std::uint64_t warm_p95_host_nanoseconds{};
  std::uint64_t warm_p99_host_nanoseconds{};
  std::uint64_t process_execution_count{};
  bfnew::CompactTransferAccounting process_compact_transfer{};
  std::uint64_t process_status_bytes{};
  std::uint64_t process_error_bytes{};
  std::uint64_t process_total_device_to_host_bytes{};
  std::uint64_t process_controller_poll_count{};
  std::uint64_t process_controller_poll_bytes{};
  std::uint64_t process_overall_device_to_host_bytes{};
  bfnew::NoCongestionStageLedger no_congestion{};
};

[[nodiscard]] std::uint64_t elapsed_nanoseconds(
    const std::chrono::steady_clock::time_point begin,
    const std::chrono::steady_clock::time_point end) {
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  if (elapsed < 0) {
    throw std::runtime_error{"steady clock moved backward"};
  }
  return static_cast<std::uint64_t>(elapsed);
}

[[nodiscard]] std::uint64_t checked_add_u64(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* const what) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  return left + right;
}

[[nodiscard]] std::uint64_t nearest_rank_u64(
    std::vector<std::uint64_t> values,
    const std::uint32_t percentile) {
  if (values.empty()) {
    return 0U;
  }
  std::sort(values.begin(), values.end());
  const std::uint64_t count = static_cast<std::uint64_t>(values.size());
  const std::uint64_t rank =
      (static_cast<std::uint64_t>(percentile) * count + 99U) / 100U;
  return values[static_cast<std::size_t>(rank - 1U)];
}

[[nodiscard]] bfnew::PipelineStageTiming device_only_stage(
    const bfnew::CompactStageTimingEvidence device_evidence,
    const std::uint64_t device_nanoseconds) {
  bfnew::PipelineStageTiming stage;
  if (device_evidence == bfnew::CompactStageTimingEvidence::measured) {
    stage.device_evidence = bfnew::PipelineTimingEvidence::measured;
    stage.device_milliseconds =
        static_cast<double>(device_nanoseconds) / 1'000'000.0;
  }
  return stage;
}

void accumulate_process_transfer(
    DriverStageLedger& driver,
    const bfnew::BatchedExpansionMetrics& metrics) {
  driver.process_execution_count = checked_add_u64(
      driver.process_execution_count, 1U, "process execution count");
#define BFNEW_ACCUMULATE_TRANSFER(field)                                      \
  driver.process_compact_transfer.field = checked_add_u64(                   \
      driver.process_compact_transfer.field,                                 \
      metrics.compact_transfer.field,                                        \
      "process compact transfer")
  BFNEW_ACCUMULATE_TRANSFER(summary_bytes);
  BFNEW_ACCUMULATE_TRANSFER(vertex_bytes);
  BFNEW_ACCUMULATE_TRANSFER(distance_label_bytes);
  BFNEW_ACCUMULATE_TRANSFER(edge_id_bytes);
  BFNEW_ACCUMULATE_TRANSFER(total_bytes);
#undef BFNEW_ACCUMULATE_TRANSFER
  driver.process_status_bytes = checked_add_u64(
      driver.process_status_bytes,
      metrics.compact_status_bytes,
      "process compact status bytes");
  driver.process_error_bytes = checked_add_u64(
      driver.process_error_bytes,
      metrics.compact_error_bytes,
      "process compact error bytes");
  driver.process_total_device_to_host_bytes = checked_add_u64(
      driver.process_total_device_to_host_bytes,
      metrics.compact_total_device_to_host_bytes,
      "process device-to-host bytes");
  driver.process_controller_poll_count = checked_add_u64(
      driver.process_controller_poll_count,
      metrics.compact_controller_poll_count,
      "process controller-poll count");
  driver.process_controller_poll_bytes = checked_add_u64(
      driver.process_controller_poll_bytes,
      metrics.compact_controller_poll_bytes,
      "process controller-poll bytes");
  driver.process_overall_device_to_host_bytes = checked_add_u64(
      driver.process_overall_device_to_host_bytes,
      metrics.compact_overall_device_to_host_bytes,
      "process overall device-to-host bytes");
}

[[nodiscard]] LoadedInputs load_inputs(const Options& options) {
  bfnew::InputGraph input = bfnew::read_graph_artifact(options.graph_path);
  const bfnew::UniformGridPartitioner partitioner{
      bfnew::SpatialOrderConfig{
          0, 0, options.tile_width, options.tile_height}};
  bfnew::PartitionedGraph partitioned = partitioner.partition(input);
  if (!bfnew::validate_weighted_graph(partitioned.graph).ok() ||
      !bfnew::validate_tile_directory(
           partitioned.graph, partitioned.tiles).ok()) {
    throw std::runtime_error{
        "batched expansion graph failed weighted/tile-directory validation"};
  }
  bfnew::TileRunLayout64 tile_runs =
      bfnew::build_tile_run_layout(partitioned.graph);
  if (!bfnew::validate_tile_run_layout(partitioned.graph, tile_runs).ok()) {
    throw std::runtime_error{
        "batched expansion graph failed tile-run validation"};
  }
  bfnew::DeviceGraphLayout32 device_graph =
      bfnew::build_device_graph_layout32(partitioned.graph, tile_runs);
  if (!bfnew::validate_device_graph_layout32(
           partitioned.graph, tile_runs, device_graph).ok()) {
    throw std::runtime_error{
        "batched expansion graph failed device-layout validation"};
  }

  std::vector<bfnew::MappedRouteQuery> mapped =
      bfnew::read_query_artifact(options.queries_path, partitioned.graph);
  if (mapped.empty()) {
    throw std::runtime_error{"query artifact contains no route queries"};
  }
  std::set<std::uint32_t> query_ids;
  std::vector<bfnew::RouteQuery> queries;
  queries.reserve(mapped.size());
  for (const bfnew::MappedRouteQuery& record : mapped) {
    if (!bfnew::validate_route_query(partitioned.graph, record.query).ok()) {
      throw std::runtime_error{
          "query artifact contains an invalid route query"};
    }
    if (!query_ids.insert(record.query.query_id.value()).second) {
      throw std::runtime_error{"query artifact contains duplicate query IDs"};
    }
    queries.push_back(record.query);
  }

  const bfnew::DeviceGraphFingerprint graph_fingerprint =
      bfnew::fingerprint_device_graph_layout32(device_graph);
  const std::array<std::uint64_t, 2U> graph_file_fingerprint =
      fingerprint_file(options.graph_path);
  const std::array<std::uint64_t, 2U> query_file_fingerprint =
      fingerprint_file(options.queries_path);
  return LoadedInputs{
      std::move(partitioned),
      std::move(tile_runs),
      std::move(device_graph),
      graph_fingerprint,
      graph_file_fingerprint,
      query_file_fingerprint,
      std::move(mapped),
      std::move(queries)};
}

[[nodiscard]] bfnew::ExpansionSchedulePolicy schedule_policy(
    const Options& options) noexcept {
  return bfnew::ExpansionSchedulePolicy{
      options.schedule,
      options.fixed_ring_size,
      options.hybrid_small_expansions};
}

[[nodiscard]] bfnew::BatchedExpansionOptions expansion_options(
    const Options& options,
    const bfnew::WeightedGraph& graph) {
  bfnew::BatchedExpansionOptions result;
  result.run_options.engine = options.engine;
  result.run_options.control_mode = options.control;
  result.run_options.rounds_per_chunk = options.rounds_per_chunk;
  result.run_options.block_size = options.block_size;
  result.run_options.grid_policy = options.grid_policy;
  result.run_options.blocks_per_wgp = options.blocks_per_wgp;
  result.run_options.instrumentation = options.instrumentation;
  result.run_options.maximum_rounds =
      options.maximum_rounds == 0U
          ? static_cast<std::uint64_t>(graph.vertex_count()) + 1U
          : options.maximum_rounds;
  result.run_options.enable_per_lane_convergence =
      options.enable_per_lane_convergence;
  result.planner_policy.lane_width = options.lane_width;
  // The engine kind/control/tuning are already part of the shared comparison
  // identity. This extra runner identity binds the otherwise external compact
  // versus evidence transfer choice; the current CLI fixes load strategy and
  // frontier queue capacity to their documented defaults.
  std::uint64_t execution_identity = 0x1700'4750'5500'0001ULL;
  const auto mix_execution_identity = [&](const std::uint64_t value) {
    execution_identity ^=
        value + 0x9e37'79b9'7f4a'7c15ULL + (execution_identity << 6U) +
        (execution_identity >> 2U);
  };
  mix_execution_identity(static_cast<std::uint64_t>(options.transfer_mode));
  mix_execution_identity(static_cast<std::uint64_t>(
      options.engine == bfnew::EngineKind::jacobi_pull
          ? bfnew::BatchRunRepresentation::device_materialized_run_masks
          : bfnew::BatchRunRepresentation::retained_per_run_masks));
  mix_execution_identity(static_cast<std::uint64_t>(
      bfnew::hip::BatchedJacobiLoadStrategy::compiler_uniform));
  mix_execution_identity(static_cast<std::uint64_t>(
      bfnew::hip::BatchedDenseLoadStrategy::compiler_uniform));
  mix_execution_identity(graph.vertex_count());
  result.execution_configuration_fingerprint =
      execution_identity == 0U ? 1U : execution_identity;
  result.schedule = schedule_policy(options);
  result.maximum_expansions = options.maximum_expansions;
  result.terminal_policy = options.terminal_policy;
  result.enable_compact_paths =
      options.transfer_mode ==
              bfnew::hip::BatchedExpansionTransferMode::compact_paths
          ? 1U
          : 0U;
  if (bfnew::validate_batched_expansion_options(result) !=
      bfnew::BatchedExpansionOptionsError::none) {
    throw std::invalid_argument{
        "combined engine/planner/expansion options are invalid"};
  }
  return result;
}

[[nodiscard]] std::string engine_name(const bfnew::EngineKind engine) {
  switch (engine) {
    case bfnew::EngineKind::jacobi_pull:
      return "jacobi";
    case bfnew::EngineKind::dense_chaotic_push:
      return "dense";
    case bfnew::EngineKind::frontier_push:
      return "frontier";
  }
  throw std::logic_error{"unknown engine in report"};
}

[[nodiscard]] std::string schedule_name(
    const bfnew::ExpansionScheduleKind schedule) {
  switch (schedule) {
    case bfnew::ExpansionScheduleKind::one_geometric_ring:
      return "one-ring";
    case bfnew::ExpansionScheduleKind::fixed_larger_ring:
      return "fixed-ring";
    case bfnew::ExpansionScheduleKind::doubling_xy_margins:
      return "doubling";
    case bfnew::ExpansionScheduleKind::hybrid_small_then_doubling:
      return "hybrid";
    case bfnew::ExpansionScheduleKind::unspecified:
      return "unspecified";
  }
  throw std::logic_error{"unknown schedule in report"};
}

[[nodiscard]] std::string transfer_name(
    const bfnew::hip::BatchedExpansionTransferMode transfer) {
  switch (transfer) {
    case bfnew::hip::BatchedExpansionTransferMode::compact_status:
      return "compact";
    case bfnew::hip::BatchedExpansionTransferMode::status_and_work_evidence:
      return "evidence";
    case bfnew::hip::BatchedExpansionTransferMode::compact_paths:
      return "paths";
  }
  throw std::logic_error{"unknown transfer mode in report"};
}

[[nodiscard]] std::string disposition_name(
    const bfnew::ExpansionQueryDisposition disposition) {
  switch (disposition) {
    case bfnew::ExpansionQueryDisposition::reached:
      return "reached";
    case bfnew::ExpansionQueryDisposition::unreachable_in_full_region:
      return "unreachable-in-full-region";
    case bfnew::ExpansionQueryDisposition::expansion_limit:
      return "expansion-limit";
    case bfnew::ExpansionQueryDisposition::region_stalled:
      return "region-stalled";
    case bfnew::ExpansionQueryDisposition::identity_or_count_overflow:
      return "identity-or-count-overflow";
    case bfnew::ExpansionQueryDisposition::engine_failure:
      return "engine-failure";
  }
  throw std::logic_error{"unknown disposition in report"};
}

[[nodiscard]] std::string escape_tsv(const std::string_view text) {
  std::string output;
  output.reserve(text.size());
  for (const char character : text) {
    switch (character) {
      case '\\':
        output += "\\\\";
        break;
      case '\t':
        output += "\\t";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      default:
        output.push_back(character);
        break;
    }
  }
  return output;
}

template <typename Value>
void write_metric(
    std::ostringstream& output,
    const std::string_view name,
    const Value value) {
  output << "metric\t" << name << '\t' << value << '\n';
}

[[nodiscard]] std::string valid_lane_identities(
    const bfnew::ExpansionBatchTrace& trace) {
  std::ostringstream output;
  for (std::uint32_t lane = 0U; lane < trace.lane_width; ++lane) {
    if ((trace.valid_lane_mask & (bfnew::LaneMask{1U} << lane)) == 0U) {
      continue;
    }
    if (output.tellp() != std::streampos{0}) {
      output << ',';
    }
    output << trace.query_ids_by_lane[lane].value() << ':'
           << trace.expansion_generations_by_lane[lane];
  }
  return output.str();
}

[[nodiscard]] std::string selected_tile_list(
    const std::span<const bfnew::TileId> tiles) {
  std::ostringstream output;
  for (std::size_t index = 0U; index < tiles.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << tiles[index].value();
  }
  return output.str();
}

[[nodiscard]] std::string vertex_list(
    const std::span<const bfnew::VertexId> vertices) {
  std::ostringstream output;
  for (std::size_t index = 0U; index < vertices.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << vertices[index].value();
  }
  return output.str();
}

[[nodiscard]] std::string edge_list(
    const std::span<const bfnew::EdgeId> edges) {
  std::ostringstream output;
  for (std::size_t index = 0U; index < edges.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << edges[index].value();
  }
  return output.str();
}

[[nodiscard]] std::string float_bit_list(
    const std::span<const float> values) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << "0x" << std::setw(8)
           << std::bit_cast<std::uint32_t>(values[index]);
  }
  return output.str();
}

[[nodiscard]] std::string u32_list(
    const std::span<const std::uint32_t> values) {
  std::ostringstream output;
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << values[index];
  }
  return output.str();
}

[[nodiscard]] std::string make_report(
    const Options& cli,
    const LoadedInputs& inputs,
    const bfnew::BatchedExpansionOptions& options,
    const bfnew::BatchedExpansionRunResult& run,
    const std::span<const bfnew::CompactQueryResult> compact_results,
    const bfnew::CompactPathQualitySample* const quality,
    const DriverStageLedger& stages,
    const bool campaign_ok) {
  std::map<std::uint32_t, std::string> names;
  for (const bfnew::MappedRouteQuery& mapped : inputs.mapped_queries) {
    names.emplace(mapped.query.query_id.value(), mapped.net_name);
  }

  const bfnew::BatchedExpansionMetrics& metrics = run.metrics;
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << "schema\t"
         << (options.enable_compact_paths != 0U
                 ? "bfnew.no-congestion-paths.v1"
                 : "bfnew.batched-expansion.v1")
         << '\n';
  output << "campaign\tstatus\t"
         << (campaign_ok ? "complete" : "failed-engine-or-identity") << '\n';
  output << "input\tgraph\t" << escape_tsv(cli.graph_path.string()) << '\n';
  output << "input\tqueries\t" << escape_tsv(cli.queries_path.string())
         << '\n';
  output << "identity\tdevice_graph\t" << inputs.graph_fingerprint.first
         << '\t' << inputs.graph_fingerprint.second << '\n';
  output << "identity\tgraph_file\t" << inputs.graph_file_fingerprint[0U]
         << '\t' << inputs.graph_file_fingerprint[1U] << '\n';
  output << "identity\tquery_file\t" << inputs.query_file_fingerprint[0U]
         << '\t' << inputs.query_file_fingerprint[1U] << '\n';
  output << "identity\tschedule_comparison\t"
         << metrics.schedule_comparison_fingerprint << '\n';
  output << "config\tengine\t" << engine_name(options.run_options.engine)
         << '\n';
  output << "config\tschedule\t" << schedule_name(options.schedule.kind)
         << '\n';
  output << "config\ttile_width\t" << cli.tile_width << '\n';
  output << "config\ttile_height\t" << cli.tile_height << '\n';
  output << "config\tlane_width\t" << options.planner_policy.lane_width
         << '\n';
  output << "config\texecution_configuration_fingerprint\t"
         << options.execution_configuration_fingerprint << '\n';
  output << "config\trun_representation\t"
         << (options.run_options.engine == bfnew::EngineKind::jacobi_pull
                 ? "device-materialized-run-masks"
                 : "retained-per-run-masks")
         << '\n';
  output << "config\tjacobi_load_strategy\tcompiler-uniform\n";
  output << "config\tdense_load_strategy\tcompiler-uniform\n";
  output << "config\tfrontier_queue_capacity\t"
         << inputs.partitioned.graph.vertex_count() << '\n';
  output << "config\tresident_graph_upload\tcompleted-before-controller\n";
  output << "config\tworkspace_initial_state\t"
         << (options.enable_compact_paths != 0U
                 ? "separate-cold-growth-then-pre-grown-warm"
                 : "cold-reused-within-campaign")
         << '\n';
  output << "config\ttiming_boundary\t"
         << (options.enable_compact_paths != 0U
                 ? "cold-and-pre-grown-warm-reported-separately"
                 : "controller-including-first-workspace-growth")
         << '\n';
  output << "config\tminimum_jaccard_numerator\t"
         << options.planner_policy.minimum_jaccard_numerator << '\n';
  output << "config\tminimum_jaccard_denominator\t"
         << options.planner_policy.minimum_jaccard_denominator << '\n';
  output << "config\tmaximum_union_inflation_numerator\t"
         << options.planner_policy.maximum_union_inflation_numerator << '\n';
  output << "config\tmaximum_union_inflation_denominator\t"
         << options.planner_policy.maximum_union_inflation_denominator << '\n';
  output << "config\tfixed_ring_size\t" << options.schedule.fixed_ring_size
         << '\n';
  output << "config\thybrid_small_expansions\t"
         << options.schedule.hybrid_small_expansion_count << '\n';
  output << "config\tmaximum_expansions\t" << options.maximum_expansions
         << '\n';
  output << "config\tterminal_policy\t"
         << (options.terminal_policy ==
                     bfnew::ExpansionTerminalPolicy::full_region_fallback
                 ? "fallback"
                 : "failure")
         << '\n';
  output << "config\tcontrol_mode\t"
         << static_cast<std::uint32_t>(options.run_options.control_mode)
         << '\n';
  output << "config\trounds_per_chunk\t"
         << options.run_options.rounds_per_chunk << '\n';
  output << "config\tblock_size\t" << options.run_options.block_size << '\n';
  output << "config\tgrid_policy\t"
         << static_cast<std::uint32_t>(options.run_options.grid_policy)
         << '\n';
  output << "config\tblocks_per_wgp\t" << options.run_options.blocks_per_wgp
         << '\n';
  output << "config\tmaximum_rounds\t" << options.run_options.maximum_rounds
         << '\n';
  output << "config\tper_lane_convergence\t"
         << options.run_options.enable_per_lane_convergence << '\n';
  output << "config\ttransfer\t" << transfer_name(cli.transfer_mode) << '\n';
  output << "config\tinstrumentation\t"
         << static_cast<std::uint32_t>(options.run_options.instrumentation)
         << '\n';
  output << "config\tplanner\toverlap-greedy\n";
  output << "config\twarmups\t" << cli.warmups << '\n';
  output << "config\trepetitions\t" << cli.repetitions << '\n';
  output << "config\tquality_sample_count\t" << cli.quality_sample_count
         << '\n';
  output << "config\tquality_seed\t" << cli.quality_seed << '\n';
  output << "config\treconstruction\t"
         << (cli.has_reconstruction ? "backtracking" : "none") << '\n';
  output << "config\trepetition_order\tsequential-after-warmups\n";
  if (options.enable_compact_paths != 0U) {
    output << "config\trepresentative_repetition\t"
           << (cli.repetitions - 1U) << '\n';
    output << "config\tdetailed_result_scope\t"
              "representative-final-warm-repetition\n";
    output << "config\tprocess_transfer_scope\t"
              "cold-plus-all-warmups-plus-all-repetitions\n";
  }

  write_metric(output, "input_queries", metrics.input_queries);
  write_metric(output, "initial_reached_queries", metrics.initial_reached_queries);
  write_metric(output, "reached_queries", metrics.reached_queries);
  write_metric(
      output,
      "unreachable_full_region_queries",
      metrics.unreachable_full_region_queries);
  write_metric(output, "expansion_limit_queries", metrics.expansion_limit_queries);
  write_metric(output, "stalled_region_queries", metrics.stalled_region_queries);
  write_metric(
      output,
      "identity_or_count_overflow_queries",
      metrics.identity_or_count_overflow_queries);
  write_metric(output, "engine_failure_queries", metrics.engine_failure_queries);
  write_metric(output, "planning_passes", metrics.planning_passes);
  write_metric(output, "batches_executed", metrics.batches_executed);
  write_metric(
      output, "initial_batches_executed", metrics.initial_batches_executed);
  write_metric(output, "retry_batches_executed", metrics.retry_batches_executed);
  write_metric(output, "failed_lane_observations", metrics.failed_lane_observations);
  write_metric(
      output,
      "failed_origin_valid_lane_observations",
      metrics.failed_origin_valid_lane_observations);
  write_metric(
      output,
      "retry_valid_lane_observations",
      metrics.retry_valid_lane_observations);
  write_metric(output, "retry_lane_capacity", metrics.retry_lane_capacity);
  write_metric(output, "scheduled_expansions", metrics.scheduled_expansions);
  write_metric(output, "full_region_fallbacks", metrics.full_region_fallbacks);
  write_metric(
      output,
      "repeated_selected_edge_estimate",
      metrics.repeated_selected_edge_estimate);
  write_metric(
      output,
      "attempted_selected_vertex_count",
      metrics.attempted_selected_vertex_count);
  write_metric(
      output,
      "attempted_selected_edge_estimate",
      metrics.attempted_selected_edge_estimate);
  write_metric(
      output, "final_selected_tile_count", metrics.final_selected_tile_count);
  write_metric(
      output,
      "final_selected_vertex_count",
      metrics.final_selected_vertex_count);
  write_metric(
      output, "final_selected_edge_count", metrics.final_selected_edge_count);
  write_metric(
      output,
      "work_evidence",
      static_cast<std::uint32_t>(metrics.work_evidence));
  write_metric(output, "work_measured_batches", metrics.work_measured_batches);
  write_metric(output, "shared_edge_work", metrics.shared_edge_work);
  write_metric(
      output, "logical_lane_edge_work", metrics.logical_lane_edge_work);
  write_metric(
      output,
      "retry_work_measured_batches",
      metrics.retry_work_measured_batches);
  write_metric(output, "retry_shared_edge_work", metrics.retry_shared_edge_work);
  write_metric(
      output,
      "retry_logical_lane_edge_work",
      metrics.retry_logical_lane_edge_work);
  write_metric(
      output,
      "failed_batch_work_measured_batches",
      metrics.failed_batch_work_measured_batches);
  write_metric(
      output,
      "failed_batch_shared_edge_work",
      metrics.failed_batch_shared_edge_work);
  write_metric(
      output,
      "failed_batch_logical_lane_edge_work",
      metrics.failed_batch_logical_lane_edge_work);
  write_metric(
      output,
      "compact_device_timing",
      static_cast<std::uint32_t>(metrics.compact_device_timing));
  write_metric(
      output,
      "compact_end_to_end_host_timing_evidence",
      static_cast<std::uint32_t>(metrics.compact_host_timing));
  write_metric(
      output,
      "compact_device_timing_measured_batches",
      metrics.compact_device_timing_measured_batches);
  write_metric(
      output,
      "compact_host_timing_measured_batches",
      metrics.compact_host_timing_measured_batches);
  write_metric(
      output, "sssp_device_nanoseconds", metrics.sssp_device_nanoseconds);
  write_metric(
      output,
      "reconstruction_device_nanoseconds",
      metrics.reconstruction_device_nanoseconds);
  write_metric(
      output,
      "result_transfer_device_nanoseconds",
      metrics.result_transfer_device_nanoseconds);
  write_metric(
      output,
      "compact_end_to_end_host_nanoseconds",
      metrics.compact_end_to_end_host_nanoseconds);
  write_metric(
      output,
      "geometric_expansion_host_nanoseconds",
      metrics.geometric_expansion_host_nanoseconds);
  write_metric(
      output,
      "compact_transfer.summary_bytes",
      metrics.compact_transfer.summary_bytes);
  write_metric(
      output,
      "compact_transfer.vertex_bytes",
      metrics.compact_transfer.vertex_bytes);
  write_metric(
      output,
      "compact_transfer.distance_label_bytes",
      metrics.compact_transfer.distance_label_bytes);
  write_metric(
      output,
      "compact_transfer.edge_id_bytes",
      metrics.compact_transfer.edge_id_bytes);
  write_metric(
      output,
      "compact_transfer.payload_total_bytes",
      metrics.compact_transfer.total_bytes);
  write_metric(
      output, "compact_status_bytes", metrics.compact_status_bytes);
  write_metric(output, "compact_error_bytes", metrics.compact_error_bytes);
  write_metric(
      output,
      "compact_total_device_to_host_bytes",
      metrics.compact_total_device_to_host_bytes);
  write_metric(
      output,
      "compact_controller_poll_count",
      metrics.compact_controller_poll_count);
  write_metric(
      output,
      "compact_controller_poll_bytes",
      metrics.compact_controller_poll_bytes);
  write_metric(
      output,
      "compact_overall_device_to_host_bytes",
      metrics.compact_overall_device_to_host_bytes);
  write_metric(
      output,
      "stage.artifact_load_host_nanoseconds",
      stages.artifact_load_host_nanoseconds);
  write_metric(
      output,
      "stage.resident_upload_host_nanoseconds",
      stages.resident_upload_host_nanoseconds);
  write_metric(
      output,
      "stage.cold_execution_host_nanoseconds",
      stages.cold_execution_host_nanoseconds);
  write_metric(
      output,
      "stage.controller_host_nanoseconds",
      stages.controller_host_nanoseconds);
  write_metric(
      output,
      "stage.pipeline_host_nanoseconds",
      stages.pipeline_host_nanoseconds);
  if (options.enable_compact_paths != 0U) {
    write_metric(
        output,
        "stage.warm_all_query_host_nanoseconds",
        stages.no_congestion.warm_all_query.host_nanoseconds);
    write_metric(
        output,
        "stage.cold_pipeline_host_nanoseconds",
        stages.no_congestion.cold_pipeline.host_nanoseconds);
    write_metric(
        output,
        "stage.controller_orchestration_host_nanoseconds",
        stages.no_congestion.controller_orchestration.host_nanoseconds);
    write_metric(
        output,
        "warm_host_nanoseconds_p50",
        stages.warm_p50_host_nanoseconds);
    write_metric(
        output,
        "warm_host_nanoseconds_p95",
        stages.warm_p95_host_nanoseconds);
    write_metric(
        output,
        "warm_host_nanoseconds_p99",
        stages.warm_p99_host_nanoseconds);
    output << "stage_header\tname\thost_evidence\thost_nanoseconds\t"
              "device_evidence\tdevice_milliseconds\n";
    const auto write_stage =
        [&](const std::string_view name,
            const bfnew::PipelineStageTiming& stage) {
          output << "stage\t" << name << '\t'
                 << static_cast<std::uint32_t>(stage.host_evidence) << '\t'
                 << stage.host_nanoseconds << '\t'
                 << static_cast<std::uint32_t>(stage.device_evidence) << '\t'
                 << std::setprecision(17) << stage.device_milliseconds
                 << '\n';
        };
    write_stage(
        "cold_artifact_load", stages.no_congestion.cold_artifact_load);
    write_stage("graph_upload", stages.no_congestion.graph_upload);
    write_stage("batch_planning", stages.no_congestion.batch_planning);
    write_stage("sssp", stages.no_congestion.sssp);
    write_stage("expansion", stages.no_congestion.expansion);
    write_stage(
        "controller_orchestration",
        stages.no_congestion.controller_orchestration);
    write_stage("reconstruction", stages.no_congestion.reconstruction);
    write_stage("result_transfer", stages.no_congestion.result_transfer);
    write_stage("warm_all_query", stages.no_congestion.warm_all_query);
    write_stage("cold_execution", stages.no_congestion.cold_execution);
    write_stage("cold_pipeline", stages.no_congestion.cold_pipeline);
    std::uint64_t complete_path_targets = 0U;
    for (const bfnew::CompactQueryResult& query : compact_results) {
      for (const bfnew::CompactTargetPath& target : query.targets) {
        if (target.summary.reconstruction ==
            bfnew::CompactPathStatus::complete) {
          complete_path_targets = checked_add_u64(
              complete_path_targets, 1U, "complete path target count");
        }
      }
    }
    output << "warm_sample_header\trepetition\thost_nanoseconds\t"
              "milliqueries_per_second\tcomplete_path_targets_per_second\t"
              "sssp_device_nanoseconds\t"
              "reconstruction_device_nanoseconds\t"
              "result_transfer_device_nanoseconds\tpayload_bytes\t"
              "status_bytes\terror_bytes\t"
              "compact_total_device_to_host_bytes\tcontroller_poll_count\t"
              "controller_poll_bytes\toverall_device_to_host_bytes\n";
    for (std::size_t sample = 0U;
         sample < stages.warm_repetitions.size();
         ++sample) {
      const WarmRepetitionSample& evidence = stages.warm_repetitions[sample];
      const std::uint64_t duration = evidence.host_nanoseconds;
      if (metrics.input_queries >
          std::numeric_limits<std::uint64_t>::max() /
              1'000'000'000'000ULL) {
        throw std::overflow_error{"warm throughput numerator overflow"};
      }
      const std::uint64_t throughput =
          duration == 0U
              ? 0U
              : (metrics.input_queries * 1'000'000'000'000ULL) / duration;
      if (complete_path_targets >
          std::numeric_limits<std::uint64_t>::max() / 1'000'000'000ULL) {
        throw std::overflow_error{"path-target throughput numerator overflow"};
      }
      const std::uint64_t path_target_throughput =
          duration == 0U
              ? 0U
              : (complete_path_targets * 1'000'000'000ULL) / duration;
      output << "warm_sample\t" << sample << '\t' << duration << '\t'
             << throughput << '\t' << path_target_throughput << '\t'
             << evidence.sssp_device_nanoseconds << '\t'
             << evidence.reconstruction_device_nanoseconds << '\t'
             << evidence.result_transfer_device_nanoseconds << '\t'
             << evidence.compact_payload_bytes << '\t'
             << evidence.status_bytes << '\t' << evidence.error_bytes << '\t'
             << evidence.total_device_to_host_bytes << '\t'
             << evidence.controller_poll_count << '\t'
             << evidence.controller_poll_bytes << '\t'
             << evidence.overall_device_to_host_bytes << '\n';
    }
  }
  write_metric(
      output,
      "initial_planning_nanoseconds",
      metrics.initial_planning_nanoseconds);
  write_metric(output, "replanning_nanoseconds", metrics.replanning_nanoseconds);
  write_metric(output, "execution_nanoseconds", metrics.execution_nanoseconds);
  write_metric(output, "total_nanoseconds", metrics.total_nanoseconds);
  write_metric(
      output,
      "all_query_throughput_milliqueries_per_second",
      metrics.all_query_throughput_milliqueries_per_second);

#define BFNEW_REPORT_WORK(field) \
  write_metric(output, "device_work." #field, metrics.device_work.field)
  BFNEW_REPORT_WORK(edges_examined);
  BFNEW_REPORT_WORK(successful_decreases);
  BFNEW_REPORT_WORK(active_vertices);
  BFNEW_REPORT_WORK(active_lane_rounds);
  BFNEW_REPORT_WORK(maximum_queue_size);
  BFNEW_REPORT_WORK(host_checks);
  BFNEW_REPORT_WORK(host_synchronizations);
  BFNEW_REPORT_WORK(controller_copies);
  BFNEW_REPORT_WORK(kernel_dispatches);
  BFNEW_REPORT_WORK(expansion_count);
  BFNEW_REPORT_WORK(atomic_attempts);
  BFNEW_REPORT_WORK(successful_atomic_updates);
  BFNEW_REPORT_WORK(queue_claims);
  BFNEW_REPORT_WORK(duplicate_suppressions);
  BFNEW_REPORT_WORK(mask_operations);
  BFNEW_REPORT_WORK(overflow_events);
  BFNEW_REPORT_WORK(high_contention_destinations);
  BFNEW_REPORT_WORK(changed_flag_updates);
  BFNEW_REPORT_WORK(full_edge_rounds);
  BFNEW_REPORT_WORK(empty_frontier_rounds);
  BFNEW_REPORT_WORK(small_frontier_rounds);
#undef BFNEW_REPORT_WORK

  output << "histogram_header\ttotal_expansions\tquery_count\n";
  for (std::size_t expansion = 0U;
       expansion < metrics.expansion_count_histogram.size();
       ++expansion) {
    output << "histogram\t" << expansion << '\t'
           << metrics.expansion_count_histogram[expansion] << '\n';
  }

  output << "query_header\tquery_id\tnet_name\tdisposition\tattempts\t"
            "scheduled_expansions\ttotal_expansions\tfallback\tgeneration\t"
            "selected_tiles\tselected_vertices\tselected_edges\tstop_reason\t"
            "error_bits\ttile_ids\n";
  for (const bfnew::ExpansionQueryOutcome& query : run.queries) {
    const std::uint32_t id = query.final_query.query_id.value();
    const auto name = names.find(id);
    if (name == names.end()) {
      throw std::logic_error{"terminal query has no artifact identity"};
    }
    output << "query\t" << id << '\t' << escape_tsv(name->second) << '\t'
           << disposition_name(query.disposition) << '\t' << query.attempts
           << '\t' << query.scheduled_expansions << '\t'
           << query.total_expansions << '\t'
           << (query.used_full_region_fallback ? 1U : 0U) << '\t'
           << query.final_query.expansion_generation << '\t'
           << query.final_query.selected_tiles.size() << '\t'
           << query.selected_vertex_count << '\t' << query.selected_edge_count
           << '\t' << query.terminal_stop_reason << '\t'
           << query.terminal_error_bits << '\t'
           << selected_tile_list(query.final_query.selected_tiles) << '\n';
  }

  if (options.enable_compact_paths != 0U) {
    if (compact_results.size() != run.queries.size()) {
      throw std::logic_error{
          "compact report results do not cover every terminal query"};
    }
    output << "compact_query_header\tquery_id\tdisposition\tgeneration\t"
              "target_terminal_to_target\n";
    output << "compact_target_header\tquery_id\ttarget_index\ttarget\t"
              "reached\treconstruction\tdistance_bits\tpath_length\t"
              "has_selected_source\tselected_source\tvertices\t"
              "distance_label_bits\tedge_ids\n";
    for (std::size_t query_index = 0U;
         query_index < run.queries.size();
         ++query_index) {
      const bfnew::ExpansionQueryOutcome& outcome = run.queries[query_index];
      const bfnew::CompactQueryResult& compact = compact_results[query_index];
      if (!bfnew::validate_compact_query_result(
               inputs.partitioned.graph, outcome.final_query, compact)
               .ok()) {
        throw std::runtime_error{
            "terminal compact target results failed independent validation"};
      }
      output << "compact_query\t" << compact.query_id.value() << '\t'
             << disposition_name(compact.disposition) << '\t'
             << compact.expansion_generation << '\t'
             << u32_list(compact.target_terminal_to_target) << '\n';
      for (std::size_t target_index = 0U;
           target_index < compact.targets.size();
           ++target_index) {
        const bfnew::CompactTargetPath& target = compact.targets[target_index];
        output << "compact_target\t" << compact.query_id.value() << '\t'
               << target_index << '\t' << target.summary.target.value() << '\t'
               << static_cast<std::uint32_t>(target.summary.reached) << '\t'
               << static_cast<std::uint32_t>(target.summary.reconstruction)
               << "\t0x" << std::hex << std::setw(8) << std::setfill('0')
               << std::bit_cast<std::uint32_t>(target.summary.distance)
               << std::dec << std::setfill(' ') << '\t'
               << target.summary.path_length << '\t'
               << target.summary.has_selected_source << '\t';
        if (target.summary.has_selected_source != 0U) {
          output << target.summary.selected_source.value();
        }
        output << '\t' << vertex_list(target.vertices) << '\t'
               << float_bit_list(target.distance_labels) << '\t'
               << edge_list(target.edge_ids) << '\n';
      }
    }
    const bfnew::NoCongestionResultAccounting accounting =
        bfnew::measure_no_congestion_result_transfer(
            compact_results,
            static_cast<std::uint64_t>(run.trace.size()),
            &metrics);
    write_metric(
        output, "result.query_count", accounting.query_count);
    write_metric(
        output,
        "result.target_summary_count",
        accounting.target_summary_count);
    write_metric(
        output, "result.complete_path_count", accounting.complete_path_count);
    write_metric(
        output,
        "result.unreachable_target_count",
        accounting.unreachable_target_count);
    write_metric(
        output,
        "result.terminal_failure_target_count",
        accounting.terminal_failure_target_count);
    write_metric(
        output,
        "result.reconstruction_failure_target_count",
        accounting.reconstruction_failure_target_count);
    write_metric(
        output,
        "result.modeled_batch_status_bytes",
        accounting.modeled_batch_status_bytes);
    write_metric(
        output,
        "result.final_serialization_summary_bytes",
        accounting.final_result_serialization.summary_bytes);
    write_metric(
        output,
        "result.final_serialization_vertex_bytes",
        accounting.final_result_serialization.vertex_bytes);
    write_metric(
        output,
        "result.final_serialization_distance_label_bytes",
        accounting.final_result_serialization.distance_label_bytes);
    write_metric(
        output,
        "result.final_serialization_edge_id_bytes",
        accounting.final_result_serialization.edge_id_bytes);
    write_metric(
        output,
        "result.final_serialization_total_bytes",
        accounting.final_result_serialization.total_bytes);
    write_metric(
        output,
        "result.modeled_final_transfer_bytes",
        accounting.modeled_final_transfer_bytes);
    write_metric(
        output,
        "result.device_transfer_evidence",
        static_cast<std::uint32_t>(accounting.device_transfer_evidence));
    write_metric(
        output,
        "result.actual_payload_total_bytes",
        accounting.actual_compact_device_transfer.total_bytes);
    write_metric(
        output,
        "result.actual_status_bytes",
        accounting.actual_status_bytes);
    write_metric(
        output,
        "result.actual_error_bytes",
        accounting.actual_error_bytes);
    write_metric(
        output,
        "result.actual_compact_total_device_to_host_bytes",
        accounting.actual_compact_total_device_to_host_bytes);
    write_metric(
        output,
        "result.actual_controller_poll_count",
        accounting.actual_controller_poll_count);
    write_metric(
        output,
        "result.actual_controller_poll_bytes",
        accounting.actual_controller_poll_bytes);
    write_metric(
        output,
        "result.actual_overall_device_to_host_bytes",
        accounting.actual_overall_device_to_host_bytes);
    write_metric(
        output,
        "process.execution_count",
        stages.process_execution_count);
    write_metric(
        output,
        "process.compact_payload_bytes",
        stages.process_compact_transfer.total_bytes);
    write_metric(
        output, "process.status_bytes", stages.process_status_bytes);
    write_metric(
        output, "process.error_bytes", stages.process_error_bytes);
    write_metric(
        output,
        "process.compact_total_device_to_host_bytes",
        stages.process_total_device_to_host_bytes);
    write_metric(
        output,
        "process.controller_poll_count",
        stages.process_controller_poll_count);
    write_metric(
        output,
        "process.controller_poll_bytes",
        stages.process_controller_poll_bytes);
    write_metric(
        output,
        "process.overall_device_to_host_bytes",
        stages.process_overall_device_to_host_bytes);
    if (quality == nullptr) {
      throw std::logic_error{"compact report has no quality sample"};
    }
    output << "quality_header\tmethod\tseed\tpopulation_queries\t"
              "requested_queries\tsampled_queries\tfinite_target_pairs\n";
    output << "quality\t"
           << static_cast<std::uint32_t>(quality->method) << '\t'
           << quality->selection_seed << '\t'
           << quality->population_query_count << '\t'
           << quality->requested_sample_query_count << '\t'
           << quality->sampled_query_count << '\t'
           << quality->finite_target_pairs << '\n';
    output << "quality_sampled_query_header\tquery_id\t"
              "finite_observation_count\n";
    std::uint64_t zero_observation_queries = 0U;
    for (const bfnew::QueryId query_id : quality->sampled_query_ids) {
      const std::uint64_t observation_count =
          static_cast<std::uint64_t>(std::ranges::count_if(
              quality->observations,
              [query_id](const bfnew::CompactPathQualityObservation& value) {
                return value.query_id == query_id;
              }));
      if (observation_count == 0U) {
        ++zero_observation_queries;
      }
      output << "quality_sampled_query\t" << query_id.value() << '\t'
             << observation_count << '\n';
    }
    write_metric(
        output,
        "quality_sampled_zero_observation_queries",
        zero_observation_queries);
    output << "quality_metric_header\tname\tvalue\n";
    output << std::setprecision(17);
#define BFNEW_REPORT_QUALITY(field) \
    output << "quality_metric\t" #field "\t" << quality->field << '\n'
    BFNEW_REPORT_QUALITY(absolute_cost_inflation_p50);
    BFNEW_REPORT_QUALITY(absolute_cost_inflation_p95);
    BFNEW_REPORT_QUALITY(absolute_cost_inflation_p99);
    BFNEW_REPORT_QUALITY(absolute_cost_inflation_max);
    BFNEW_REPORT_QUALITY(cost_ratio_p50);
    BFNEW_REPORT_QUALITY(cost_ratio_p95);
    BFNEW_REPORT_QUALITY(cost_ratio_p99);
    BFNEW_REPORT_QUALITY(cost_ratio_max);
    BFNEW_REPORT_QUALITY(absolute_path_length_inflation_p50);
    BFNEW_REPORT_QUALITY(absolute_path_length_inflation_p95);
    BFNEW_REPORT_QUALITY(absolute_path_length_inflation_p99);
    BFNEW_REPORT_QUALITY(absolute_path_length_inflation_max);
    BFNEW_REPORT_QUALITY(path_length_ratio_p50);
    BFNEW_REPORT_QUALITY(path_length_ratio_p95);
    BFNEW_REPORT_QUALITY(path_length_ratio_p99);
    BFNEW_REPORT_QUALITY(path_length_ratio_max);
#undef BFNEW_REPORT_QUALITY
    output << "quality_observation_header\tquery_id\ttarget\t"
              "bounded_distance\tunbounded_distance\tbounded_edges\t"
              "unbounded_edges\tabsolute_cost_inflation\t"
              "absolute_edge_inflation\tcost_ratio\tedge_ratio\n";
    for (const bfnew::CompactPathQualityObservation& observation :
         quality->observations) {
      output << "quality_observation\t" << observation.query_id.value()
             << '\t' << observation.target.value() << '\t'
             << observation.bounded_distance << '\t'
             << observation.unbounded_distance << '\t'
             << observation.bounded_path_length << '\t'
             << observation.unbounded_path_length << '\t'
             << observation.absolute_cost_inflation << '\t'
             << observation.absolute_path_length_inflation << '\t'
             << observation.cost_inflation_ratio << '\t'
             << observation.path_length_inflation_ratio << '\n';
    }
  }

  output << "trace_header\texecution_ordinal\tplanning_pass\tbatch_index\t"
            "retry_pass\tlane_width\tvalid_mask\treached_mask\tmiss_mask\t"
            "union_vertices\tunion_edges\tselected_lane_vertices\t"
            "selected_lane_edges\twork_evidence\tshared_edge_work\t"
            "logical_lane_edge_work\tstop_reason\terror_bits\tlane_identities\n";
  for (const bfnew::ExpansionBatchTrace& trace : run.trace) {
    output << "trace\t" << trace.context.execution_ordinal << '\t'
           << trace.context.planning_pass << '\t' << trace.context.batch_index
           << '\t' << (trace.context.retry_pass ? 1U : 0U) << '\t'
           << trace.lane_width << '\t' << trace.valid_lane_mask << '\t'
           << trace.reached_lane_mask << '\t' << trace.miss_lane_mask << '\t'
           << trace.union_vertex_count << '\t' << trace.union_edge_estimate
           << '\t' << trace.selected_lane_vertex_count << '\t'
           << trace.selected_lane_edge_estimate << '\t'
           << static_cast<std::uint32_t>(trace.work_evidence) << '\t'
           << trace.shared_edge_work << '\t' << trace.logical_lane_edge_work
           << '\t' << trace.stop_reason << '\t' << trace.error_bits << '\t'
           << valid_lane_identities(trace) << '\n';
  }
  return output.str();
}

void write_report_fail_closed(
    const std::filesystem::path& path,
    const std::string_view report) {
  if (std::filesystem::exists(path)) {
    throw std::runtime_error{
        "refusing to overwrite existing output: " + path.string()};
  }
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  if (std::filesystem::exists(temporary)) {
    throw std::runtime_error{
        "refusing to overwrite stale temporary output: " +
        temporary.string()};
  }
  try {
    {
      std::ofstream output(temporary, std::ios::binary);
      if (!output) {
        throw std::runtime_error{
            "cannot open temporary output: " + temporary.string()};
      }
      output.write(report.data(), static_cast<std::streamsize>(report.size()));
      output.flush();
      if (!output) {
        throw std::runtime_error{
            "cannot write complete temporary output: " + temporary.string()};
      }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
      throw std::runtime_error{
          "cannot publish report: " + error.message()};
    }
  } catch (...) {
    std::error_code ignored;
    static_cast<void>(std::filesystem::remove(temporary, ignored));
    throw;
  }
}

[[nodiscard]] bfnew::ExpansionBatchRunner bind_runner(
    bfnew::hip::BatchedExpansionExecutor& executor) {
  return [&executor](
             const std::span<const bfnew::RouteQuery> queries,
             const std::span<const bfnew::BatchQueryFeatures> features,
             const bfnew::BatchPlanEntry& batch,
             const bfnew::ExpansionBatchContext& context) {
    return executor(queries, features, batch, context);
  };
}

void validate_complete_run(
    const LoadedInputs& inputs,
    const bfnew::BatchedExpansionRunResult& run) {
  const std::uint64_t expected_count =
      static_cast<std::uint64_t>(inputs.queries.size());
  if (run.queries.size() != inputs.queries.size() ||
      run.metrics.input_queries != expected_count ||
      run.metrics.batches_executed != run.trace.size()) {
    throw std::runtime_error{
        "all-query execution returned an incomplete outcome ledger"};
  }

  std::map<std::uint32_t, const bfnew::RouteQuery*> originals;
  for (const bfnew::RouteQuery& query : inputs.queries) {
    originals.emplace(query.query_id.value(), &query);
  }
  bool first = true;
  std::uint32_t previous_id = 0U;
  for (const bfnew::ExpansionQueryOutcome& outcome : run.queries) {
    const std::uint32_t id = outcome.final_query.query_id.value();
    const auto original = originals.find(id);
    if (original == originals.end() || (!first && id <= previous_id)) {
      throw std::runtime_error{
          "all-query execution returned a duplicate, unknown, or "
          "noncanonical query identity"};
    }
    const bfnew::RouteQuery& expected = *original->second;
    const bfnew::RouteQuery& actual = outcome.final_query;
    if (actual.source_terminals != expected.source_terminals ||
        actual.target_terminals != expected.target_terminals ||
        actual.sources != expected.sources ||
        actual.targets != expected.targets ||
        actual.source_terminal_to_source !=
            expected.source_terminal_to_source ||
        actual.target_terminal_to_target !=
            expected.target_terminal_to_target) {
      throw std::runtime_error{
          "all-query execution changed immutable terminal identity"};
    }
    first = false;
    previous_id = id;
  }

  const std::array terminal_counts{
      run.metrics.reached_queries,
      run.metrics.unreachable_full_region_queries,
      run.metrics.expansion_limit_queries,
      run.metrics.stalled_region_queries,
      run.metrics.identity_or_count_overflow_queries,
      run.metrics.engine_failure_queries};
  std::uint64_t terminal_total = 0U;
  for (const std::uint64_t count : terminal_counts) {
    if (count > expected_count - terminal_total) {
      throw std::runtime_error{
          "all-query terminal aggregate exceeds the input query count"};
    }
    terminal_total += count;
  }
  if (terminal_total != expected_count ||
      run.metrics.initial_reached_queries > run.metrics.reached_queries) {
    throw std::runtime_error{
        "all-query terminal aggregates do not partition the input queries"};
  }

  for (std::size_t index = 0U; index < run.trace.size(); ++index) {
    const bfnew::ExpansionBatchTrace& trace = run.trace[index];
    if (trace.context.execution_ordinal != index ||
        trace.valid_lane_mask == 0U ||
        trace.query_ids_by_lane.size() != trace.lane_width ||
        trace.expansion_generations_by_lane.size() != trace.lane_width) {
      throw std::runtime_error{
          "all-query execution returned a malformed batch trace"};
    }
  }
}

[[nodiscard]] bool successful_campaign(
    const bfnew::BatchedExpansionRunResult& run) noexcept {
  return run.metrics.engine_failure_queries == 0U &&
         run.metrics.identity_or_count_overflow_queries == 0U;
}

[[nodiscard]] std::vector<bfnew::CompactQueryResult> make_compact_results(
    const LoadedInputs& inputs,
    const bfnew::BatchedExpansionRunResult& run,
    const bool enabled) {
  if (!enabled) {
    return {};
  }
  std::vector<bfnew::CompactQueryResult> results;
  results.reserve(run.queries.size());
  for (const bfnew::ExpansionQueryOutcome& outcome : run.queries) {
    if ((outcome.reached() ||
         outcome.disposition ==
             bfnew::ExpansionQueryDisposition::unreachable_in_full_region) &&
        !outcome.compact_paths.has_value()) {
      throw std::runtime_error{
          "successful/final-unreachable query lost compact target results"};
    }
    bfnew::CompactQueryResult compact =
        outcome.compact_paths.has_value()
            ? bfnew::make_compact_query_result(
                  *outcome.compact_paths, outcome.disposition)
            : bfnew::make_failed_compact_query_result(
                  outcome.final_query, outcome.disposition);
    if (!bfnew::validate_compact_query_result(
             inputs.partitioned.graph, outcome.final_query, compact)
             .ok()) {
      throw std::runtime_error{
          "terminal compact target results failed independent validation"};
    }
    results.push_back(std::move(compact));
  }
  return results;
}

struct ValidatedExecution {
  bfnew::BatchedExpansionRunResult run;
  std::vector<bfnew::CompactQueryResult> compact_results;
  std::uint64_t host_nanoseconds{};
};

[[nodiscard]] ValidatedExecution execute_once(
    const LoadedInputs& inputs,
    const bfnew::BatchedExpansionOptions& options,
    const bfnew::ExpansionBatchRunner& runner) {
  const auto begin = std::chrono::steady_clock::now();
  ValidatedExecution execution;
  execution.run = bfnew::run_batched_expansion(
      inputs.partitioned.graph,
      inputs.partitioned.tiles,
      inputs.tile_runs,
      inputs.queries,
      options,
      runner);
  validate_complete_run(inputs, execution.run);
  if (options.enable_compact_paths != 0U &&
      (execution.run.metrics.batches_executed == 0U ||
       execution.run.metrics.compact_device_timing !=
           bfnew::CompactStageTimingEvidence::measured ||
       execution.run.metrics.compact_device_timing_measured_batches !=
           execution.run.metrics.batches_executed)) {
    throw std::runtime_error{
        "path execution lacks complete measured HIP event timing"};
  }
  execution.compact_results = make_compact_results(
      inputs, execution.run, options.enable_compact_paths != 0U);
  execution.host_nanoseconds =
      elapsed_nanoseconds(begin, std::chrono::steady_clock::now());
  return execution;
}

[[nodiscard]] bfnew::NoCongestionStageLedger make_path_stage_ledger(
    const DriverStageLedger& driver,
    const bfnew::BatchedExpansionMetrics& metrics,
    const std::uint64_t warm_host_nanoseconds) {
  if (metrics.compact_host_timing !=
      bfnew::CompactStageTimingEvidence::measured) {
    throw std::runtime_error{
        "path campaign lacks measured host stage timing"};
  }
  const std::uint64_t planning = checked_add_u64(
      metrics.initial_planning_nanoseconds,
      metrics.replanning_nanoseconds,
      "planning time");
  const std::uint64_t named = checked_add_u64(
      planning,
      metrics.geometric_expansion_host_nanoseconds,
      "measured host stage time");
  if (named > warm_host_nanoseconds) {
    throw std::runtime_error{
        "named path stages exceed their enclosing warm execution"};
  }
  const bfnew::CompactStageTimingEvidence device_evidence =
      metrics.compact_device_timing;
  return bfnew::make_no_congestion_stage_ledger(
      bfnew::measured_host_stage(driver.artifact_load_host_nanoseconds),
      bfnew::PipelineStageTiming{
          bfnew::PipelineTimingEvidence::measured,
          driver.resident_upload_host_nanoseconds,
          bfnew::PipelineTimingEvidence::measured,
          driver.resident_upload_device_milliseconds},
      bfnew::measured_host_stage(planning),
      device_only_stage(device_evidence, metrics.sssp_device_nanoseconds),
      bfnew::measured_host_stage(
          metrics.geometric_expansion_host_nanoseconds),
      bfnew::PipelineStageTiming{},
      device_only_stage(
          device_evidence, metrics.reconstruction_device_nanoseconds),
      device_only_stage(
          device_evidence, metrics.result_transfer_device_nanoseconds),
      bfnew::measured_host_stage(warm_host_nanoseconds),
      bfnew::measured_host_stage(driver.cold_execution_host_nanoseconds));
}

[[nodiscard]] int run(const Options& cli) {
  DriverStageLedger stages;
  std::filesystem::path temporary_output = cli.output_path;
  temporary_output += ".tmp";
  if (std::filesystem::exists(cli.output_path) ||
      std::filesystem::exists(temporary_output)) {
    throw std::runtime_error{
        "refusing to start a campaign whose output or temporary output "
        "already exists"};
  }

  const auto artifact_begin = std::chrono::steady_clock::now();
  LoadedInputs inputs = load_inputs(cli);
  const bfnew::BatchedExpansionOptions options =
      expansion_options(cli, inputs.partitioned.graph);
  const auto artifact_end = std::chrono::steady_clock::now();
  stages.artifact_load_host_nanoseconds =
      elapsed_nanoseconds(artifact_begin, artifact_end);

  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  bfnew::hip::HipEventTimer upload_device_timer;
  const auto upload_begin = artifact_end;
  upload_device_timer.start(stream);
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(std::move(inputs.device_graph)),
      stream);
  upload_device_timer.stop(stream);
  // Complete the immutable graph upload and release its staging image before
  // allocating the first large engine workspace.
  resident.synchronize_upload();
  const auto upload_end = std::chrono::steady_clock::now();
  stages.resident_upload_host_nanoseconds =
      elapsed_nanoseconds(upload_begin, upload_end);
  stages.resident_upload_device_milliseconds =
      upload_device_timer.elapsed_milliseconds_after_stream_synchronization();

  // This first interval intentionally includes engine/workspace construction
  // and capacity growth. It is cold evidence, never a warm throughput sample.
  const auto cold_execution_begin = upload_end;
  std::unique_ptr<bfnew::hip::ReusableBatchedJacobiWorkspace>
      jacobi_workspace;
  std::unique_ptr<bfnew::hip::ReusableBatchedDenseWorkspace> dense_workspace;
  std::unique_ptr<bfnew::hip::ReusableBatchedFrontierWorkspace>
      frontier_workspace;
  std::unique_ptr<bfnew::hip::BatchedExpansionExecutor> executor;
  switch (options.run_options.engine) {
    case bfnew::EngineKind::jacobi_pull:
      jacobi_workspace =
          std::make_unique<bfnew::hip::ReusableBatchedJacobiWorkspace>();
      executor = std::make_unique<bfnew::hip::BatchedExpansionExecutor>(
          inputs.partitioned.graph,
          inputs.tile_runs,
          resident,
          *jacobi_workspace,
          stream,
          options.run_options,
          cli.transfer_mode,
          bfnew::hip::BatchedJacobiLoadStrategy::compiler_uniform);
      break;
    case bfnew::EngineKind::dense_chaotic_push:
      dense_workspace =
          std::make_unique<bfnew::hip::ReusableBatchedDenseWorkspace>();
      executor = std::make_unique<bfnew::hip::BatchedExpansionExecutor>(
          inputs.partitioned.graph,
          inputs.tile_runs,
          resident,
          *dense_workspace,
          stream,
          options.run_options,
          cli.transfer_mode,
          bfnew::hip::BatchedDenseLoadStrategy::compiler_uniform);
      break;
    case bfnew::EngineKind::frontier_push:
      frontier_workspace =
          std::make_unique<bfnew::hip::ReusableBatchedFrontierWorkspace>();
      executor = std::make_unique<bfnew::hip::BatchedExpansionExecutor>(
          inputs.partitioned.graph,
          inputs.tile_runs,
          resident,
          *frontier_workspace,
          stream,
          options.run_options,
          cli.transfer_mode,
          0U);
      break;
  }
  if (executor == nullptr) {
    throw std::logic_error{"no batched expansion engine was selected"};
  }

  const bfnew::ExpansionBatchRunner runner = bind_runner(*executor);
  ValidatedExecution cold = execute_once(inputs, options, runner);
  accumulate_process_transfer(stages, cold.run.metrics);
  stages.cold_execution_host_nanoseconds = elapsed_nanoseconds(
      cold_execution_begin, std::chrono::steady_clock::now());
  bool campaign_ok = successful_campaign(cold.run);

  ValidatedExecution selected = std::move(cold);
  bfnew::CompactPathQualitySample quality;
  const bfnew::CompactPathQualitySample* quality_ptr = nullptr;
  if (options.enable_compact_paths != 0U) {
    const std::vector<bfnew::CompactQueryResult> canonical_results =
        selected.compact_results;
    for (std::uint32_t warmup = 1U; warmup < cli.warmups; ++warmup) {
      ValidatedExecution execution = execute_once(inputs, options, runner);
      if (execution.compact_results != canonical_results) {
        throw std::runtime_error{
            "warmup changed canonical compact target/path results"};
      }
      accumulate_process_transfer(stages, execution.run.metrics);
      campaign_ok = campaign_ok && successful_campaign(execution.run);
    }

    for (std::uint32_t repetition = 0U;
         repetition < cli.repetitions;
         ++repetition) {
      ValidatedExecution execution = execute_once(inputs, options, runner);
      if (execution.compact_results != canonical_results) {
        throw std::runtime_error{
            "measured repetition changed canonical compact target/path results"};
      }
      accumulate_process_transfer(stages, execution.run.metrics);
      campaign_ok = campaign_ok && successful_campaign(execution.run);
      stages.warm_repetitions.push_back(WarmRepetitionSample{
          execution.host_nanoseconds,
          execution.run.metrics.sssp_device_nanoseconds,
          execution.run.metrics.reconstruction_device_nanoseconds,
          execution.run.metrics.result_transfer_device_nanoseconds,
          execution.run.metrics.compact_transfer.total_bytes,
          execution.run.metrics.compact_status_bytes,
          execution.run.metrics.compact_error_bytes,
          execution.run.metrics.compact_total_device_to_host_bytes,
          execution.run.metrics.compact_controller_poll_count,
          execution.run.metrics.compact_controller_poll_bytes,
          execution.run.metrics.compact_overall_device_to_host_bytes});
      selected = std::move(execution);
    }
    std::vector<std::uint64_t> warm_durations;
    warm_durations.reserve(stages.warm_repetitions.size());
    for (const WarmRepetitionSample& sample : stages.warm_repetitions) {
      warm_durations.push_back(sample.host_nanoseconds);
    }
    stages.warm_p50_host_nanoseconds = nearest_rank_u64(
        warm_durations, 50U);
    stages.warm_p95_host_nanoseconds = nearest_rank_u64(
        warm_durations, 95U);
    stages.warm_p99_host_nanoseconds = nearest_rank_u64(
        warm_durations, 99U);
    stages.no_congestion = make_path_stage_ledger(
        stages, selected.run.metrics, selected.host_nanoseconds);
    stages.controller_host_nanoseconds =
        stages.no_congestion.controller_orchestration.host_nanoseconds;
    stages.pipeline_host_nanoseconds =
        stages.no_congestion.cold_pipeline.host_nanoseconds;
    quality = bfnew::sample_compact_path_quality(
        inputs.partitioned.graph,
        selected.run.queries,
        selected.compact_results,
        cli.quality_sample_count,
        cli.quality_seed);
    quality_ptr = &quality;
  } else {
    stages.controller_host_nanoseconds = selected.host_nanoseconds;
    stages.pipeline_host_nanoseconds = checked_add_u64(
        checked_add_u64(
            stages.artifact_load_host_nanoseconds,
            stages.resident_upload_host_nanoseconds,
            "cold pipeline time"),
        stages.cold_execution_host_nanoseconds,
        "cold pipeline time");
  }

  std::uint64_t process_payload_components =
      stages.process_compact_transfer.summary_bytes;
  process_payload_components = checked_add_u64(
      process_payload_components,
      stages.process_compact_transfer.vertex_bytes,
      "process compact payload bytes");
  process_payload_components = checked_add_u64(
      process_payload_components,
      stages.process_compact_transfer.distance_label_bytes,
      "process compact payload bytes");
  process_payload_components = checked_add_u64(
      process_payload_components,
      stages.process_compact_transfer.edge_id_bytes,
      "process compact payload bytes");
  std::uint64_t process_total = stages.process_compact_transfer.total_bytes;
  process_total = checked_add_u64(
      process_total, stages.process_status_bytes, "process transfer bytes");
  process_total = checked_add_u64(
      process_total, stages.process_error_bytes, "process transfer bytes");
  if (stages.process_controller_poll_count >
      std::numeric_limits<std::uint64_t>::max() /
          sizeof(bfnew::DeviceController)) {
    throw std::overflow_error{"process controller-poll bytes overflow"};
  }
  const std::uint64_t expected_controller_poll_bytes =
      stages.process_controller_poll_count * sizeof(bfnew::DeviceController);
  const std::uint64_t process_overall = checked_add_u64(
      process_total,
      stages.process_controller_poll_bytes,
      "process overall transfer bytes");
  if (process_payload_components !=
          stages.process_compact_transfer.total_bytes ||
      process_total != stages.process_total_device_to_host_bytes ||
      expected_controller_poll_bytes !=
          stages.process_controller_poll_bytes ||
      process_overall != stages.process_overall_device_to_host_bytes) {
    throw std::runtime_error{
        "process compact transfer accounting is internally inconsistent"};
  }

  const std::string report = make_report(
      cli,
      inputs,
      options,
      selected.run,
      selected.compact_results,
      quality_ptr,
      stages,
      campaign_ok);
  write_report_fail_closed(cli.output_path, report);
  std::cout << "wrote " << selected.run.queries.size()
            << " canonical query outcomes and " << selected.run.trace.size()
            << " batch trace rows to " << cli.output_path << '\n';
  if (!campaign_ok) {
    std::cerr << "campaign completed with engine or identity/count failures; "
                 "the report is diagnostic and must not be accepted as "
                 "successful all-query evidence\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "bfnew GPU batched expansion failed: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
}
