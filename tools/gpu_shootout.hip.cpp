#include "bfnew/device_layout.hpp"
#include "bfnew/engine_shootout.hpp"
#include "bfnew/fpga_interchange.hpp"
#include "bfnew/hip/engine_shootout.hpp"
#include "bfnew/hip/runtime.hpp"
#include "bfnew/spatial.hpp"
#include "bfnew/sssp.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

enum class Stage {
  pilot,
  correctness,
  counters,
  timing,
  profile_case,
};

enum class WorkloadMode {
  logicnets_jscl,
  synthetic,
};

inline constexpr std::uint64_t prescreen_seed_domain =
    0x7072657363726565ULL;

struct Options {
  Stage stage{Stage::pilot};
  bool has_stage{};
  WorkloadMode workload{WorkloadMode::logicnets_jscl};
  std::filesystem::path graph_path;
  std::filesystem::path queries_path;
  std::filesystem::path manifest_path;
  std::filesystem::path catalog_path;
  std::filesystem::path correctness_path;
  std::filesystem::path output_directory;
  std::uint32_t tile_width{8U};
  std::uint32_t tile_height{8U};
  std::uint32_t query_count{bfnew::minimum_logicnets_shootout_queries};
  bool has_query_count{};
  std::string synthetic_case_name;
  std::uint64_t selection_seed{20260824U};
  bool has_selection_seed{};
  std::uint64_t order_seed{20260824U};
  bool has_order_seed{};
  std::uint64_t maximum_rounds{};
  std::uint32_t warmups{5U};
  bool has_warmups{};
  std::uint32_t repetitions{30U};
  bool has_repetitions{};
  bfnew::InstrumentationLevel counter_instrumentation{
      bfnew::InstrumentationLevel::debug};
  bool has_instrumentation{};
  bfnew::ShootoutRunKind profile_kind{bfnew::ShootoutRunKind::trace};
  bool has_profile_kind{};
  std::vector<std::uint64_t> profile_case_ids;
  std::uint64_t profiler_pass_id{};
  std::uint64_t profiler_counter_set_id{};
};

[[nodiscard]] std::uint32_t parse_u32(
    const std::string& text,
    const char* const name) {
  std::size_t consumed = 0U;
  const unsigned long value = std::stoul(text, &consumed, 10);
  if (consumed != text.size() || !std::in_range<std::uint32_t>(value)) {
    throw std::invalid_argument{std::string{name} +
                                " must be a 32-bit unsigned integer"};
  }
  return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::uint64_t parse_u64(
    const std::string& text,
    const char* const name) {
  std::size_t consumed = 0U;
  const unsigned long long value = std::stoull(text, &consumed, 10);
  if (consumed != text.size()) {
    throw std::invalid_argument{std::string{name} +
                                " must be a 64-bit unsigned integer"};
  }
  return static_cast<std::uint64_t>(value);
}

[[nodiscard]] Stage parse_stage(const std::string& text) {
  if (text == "pilot") {
    return Stage::pilot;
  }
  if (text == "correctness") {
    return Stage::correctness;
  }
  if (text == "counters") {
    return Stage::counters;
  }
  if (text == "timing") {
    return Stage::timing;
  }
  if (text == "profile-case") {
    return Stage::profile_case;
  }
  throw std::invalid_argument{
      "--stage must be pilot, correctness, counters, timing, or profile-case"};
}

[[nodiscard]] WorkloadMode parse_workload(const std::string& text) {
  if (text == "logicnets_jscl") {
    return WorkloadMode::logicnets_jscl;
  }
  if (text == "synthetic") {
    return WorkloadMode::synthetic;
  }
  throw std::invalid_argument{
      "--workload must be logicnets_jscl or synthetic"};
}

void print_help() {
  std::cout
      << "Usage: bfnew_gpu_shootout --stage STAGE --workload WORKLOAD "
         "--output DIR [options]\n"
         "Stages:\n"
         "  pilot       build pilot.v1.tsv, manifest.v1.tsv, catalog.v1.tsv\n"
         "  correctness require --manifest and --catalog\n"
         "  counters    require --manifest, --catalog, and --correctness\n"
         "  timing      require --manifest, --catalog, and --correctness\n"
         "  profile-case require the same plus --profile-kind trace|pmc,\n"
         "               --profile-case-id N, --profiler-pass-id N, and for\n"
         "               PMC --profiler-counter-set-id N\n"
         "Common options:\n"
         "  --tile-width N --tile-height N --maximum-rounds N\n"
         "  --selection-seed N --order-seed N --query-count N\n"
         "  --warmups N --repetitions N\n"
         "  profile-case accepts repeated --profile-case-id N options\n"
         "  logicnets_jscl: --graph FILE --queries FILE (>=1000 queries)\n"
         "  synthetic: --case-name sparse-wavefront|dense-frontier\n"
         "  --instrumentation debug  (counters only)\n";
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
      std::exit(0);
    } else if (argument == "--stage") {
      options.stage = parse_stage(value("--stage"));
      options.has_stage = true;
    } else if (argument == "--workload") {
      options.workload = parse_workload(value("--workload"));
    } else if (argument == "--graph") {
      options.graph_path = value("--graph");
    } else if (argument == "--queries") {
      options.queries_path = value("--queries");
    } else if (argument == "--manifest") {
      options.manifest_path = value("--manifest");
    } else if (argument == "--catalog") {
      options.catalog_path = value("--catalog");
    } else if (argument == "--correctness") {
      options.correctness_path = value("--correctness");
    } else if (argument == "--output") {
      options.output_directory = value("--output");
    } else if (argument == "--tile-width") {
      options.tile_width = parse_u32(value("--tile-width"), "tile width");
    } else if (argument == "--tile-height") {
      options.tile_height = parse_u32(value("--tile-height"), "tile height");
    } else if (argument == "--query-count") {
      options.query_count = parse_u32(value("--query-count"), "query count");
      options.has_query_count = true;
    } else if (argument == "--case-name") {
      options.synthetic_case_name = value("--case-name");
    } else if (argument == "--selection-seed") {
      options.selection_seed =
          parse_u64(value("--selection-seed"), "selection seed");
      options.has_selection_seed = true;
    } else if (argument == "--order-seed") {
      options.order_seed = parse_u64(value("--order-seed"), "order seed");
      options.has_order_seed = true;
    } else if (argument == "--maximum-rounds") {
      options.maximum_rounds =
          parse_u64(value("--maximum-rounds"), "maximum rounds");
    } else if (argument == "--warmups") {
      options.warmups = parse_u32(value("--warmups"), "warmups");
      options.has_warmups = true;
    } else if (argument == "--repetitions") {
      options.repetitions =
          parse_u32(value("--repetitions"), "repetitions");
      options.has_repetitions = true;
    } else if (argument == "--instrumentation") {
      const std::string level = value("--instrumentation");
      if (level == "debug") {
        options.counter_instrumentation =
            bfnew::InstrumentationLevel::debug;
        options.has_instrumentation = true;
      } else {
        throw std::invalid_argument{
            "--instrumentation must be debug"};
      }
    } else if (argument == "--profile-kind") {
      const std::string kind = value("--profile-kind");
      if (kind == "trace") {
        options.profile_kind = bfnew::ShootoutRunKind::trace;
      } else if (kind == "pmc") {
        options.profile_kind = bfnew::ShootoutRunKind::pmc;
      } else {
        throw std::invalid_argument{"--profile-kind must be trace or pmc"};
      }
      options.has_profile_kind = true;
    } else if (argument == "--profile-case-id") {
      options.profile_case_ids.push_back(
          parse_u64(value("--profile-case-id"), "profile case ID"));
    } else if (argument == "--profiler-pass-id") {
      options.profiler_pass_id =
          parse_u64(value("--profiler-pass-id"), "profiler pass ID");
    } else if (argument == "--profiler-counter-set-id") {
      options.profiler_counter_set_id = parse_u64(
          value("--profiler-counter-set-id"), "profiler counter-set ID");
    } else {
      throw std::invalid_argument{"unknown option: " + argument};
    }
  }

  if (!options.has_stage || options.output_directory.empty()) {
    throw std::invalid_argument{
        "--stage and --output are required"};
  }
  if (options.tile_width == 0U || options.tile_height == 0U) {
    throw std::invalid_argument{"tile dimensions must be positive"};
  }
  if (options.workload == WorkloadMode::logicnets_jscl) {
    if (options.graph_path.empty() || options.queries_path.empty()) {
      throw std::invalid_argument{
          "logicnets_jscl requires --graph and --queries"};
    }
    if (options.query_count < bfnew::minimum_logicnets_shootout_queries) {
      throw std::invalid_argument{
          "the logicnets shootout requires at least 1000 queries"};
    }
    if (!options.synthetic_case_name.empty()) {
      throw std::invalid_argument{
          "logicnets_jscl does not accept --case-name"};
    }
  } else {
    if (!options.graph_path.empty() || !options.queries_path.empty() ||
        options.has_query_count) {
      throw std::invalid_argument{
          "built-in synthetic workloads do not accept graph, query, or "
          "query-count options"};
    }
    if (options.synthetic_case_name != "sparse-wavefront" &&
        options.synthetic_case_name != "dense-frontier") {
      throw std::invalid_argument{
          "synthetic --case-name must be sparse-wavefront or dense-frontier"};
    }
  }
  if (options.stage != Stage::pilot &&
      (options.manifest_path.empty() || options.catalog_path.empty())) {
    throw std::invalid_argument{
        "non-pilot stages require --manifest and --catalog"};
  }
  if ((options.stage == Stage::counters || options.stage == Stage::timing ||
       options.stage == Stage::profile_case) &&
      options.correctness_path.empty()) {
    throw std::invalid_argument{
        "post-correctness stages require --correctness"};
  }
  if (options.warmups == 0U || options.repetitions == 0U) {
    throw std::invalid_argument{
        "shootout warmup and timing repetition policies must be positive"};
  }
  if (options.has_instrumentation && options.stage != Stage::counters) {
    throw std::invalid_argument{
        "--instrumentation is accepted only by the counters stage"};
  }
  if (options.stage == Stage::profile_case) {
    if (!options.has_profile_kind || options.profile_case_ids.empty()) {
      throw std::invalid_argument{
          "profile-case requires --profile-kind and --profile-case-id"};
    }
    std::set<std::uint64_t> unique_profile_cases;
    for (const std::uint64_t case_id : options.profile_case_ids) {
      if (!unique_profile_cases.insert(case_id).second) {
        throw std::invalid_argument{
            "profile-case IDs must be unique within one replay process"};
      }
    }
    if (options.profiler_pass_id == 0U) {
      throw std::invalid_argument{"profile-case requires a profiler pass ID"};
    }
    if (options.profile_kind == bfnew::ShootoutRunKind::pmc &&
        options.profiler_counter_set_id == 0U) {
      throw std::invalid_argument{"PMC profile-case requires a counter-set ID"};
    }
    if (options.profile_kind == bfnew::ShootoutRunKind::trace &&
        options.profiler_counter_set_id != 0U) {
      throw std::invalid_argument{"trace profile-case cannot name a counter set"};
    }
  }
  return options;
}

[[nodiscard]] std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error{"cannot open shootout input: " + path.string()};
  }
  input.seekg(0, std::ios::end);
  const std::streamoff end = input.tellg();
  if (end < 0) {
    throw std::runtime_error{"cannot size shootout input: " + path.string()};
  }
  input.seekg(0, std::ios::beg);
  std::string text(static_cast<std::size_t>(end), '\0');
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (input.gcount() != static_cast<std::streamsize>(text.size())) {
    throw std::runtime_error{"cannot read complete shootout input: " +
                             path.string()};
  }
  return text;
}

void write_text(
    const std::filesystem::path& path,
    const std::string_view text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!output) {
    throw std::runtime_error{"cannot write shootout output: " + path.string()};
  }
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
      const std::uint64_t byte = static_cast<unsigned char>(buffer[static_cast<std::size_t>(index)]);
      first = (first ^ byte) * 1099511628211ULL;
      second ^= byte + 0x9e3779b97f4a7c15ULL + (second << 6U) +
                (second >> 2U) + position;
      ++position;
    }
  }
  if (!input.eof()) {
    throw std::runtime_error{"failed while fingerprinting input: " +
                             path.string()};
  }
  return {first, second};
}

[[nodiscard]] bfnew::ShootoutInputFingerprint make_input_fingerprint(
    const bfnew::DeviceGraphFingerprint& graph,
    const std::filesystem::path& query_path,
    const std::size_t query_count) {
  if (!std::in_range<std::uint64_t>(query_count)) {
    throw std::overflow_error{"shootout query count exceeds 64 bits"};
  }
  const auto query = fingerprint_file(query_path);
  bfnew::ShootoutInputFingerprint result;
  result.graph_words[0] = graph.first;
  result.graph_words[1] = graph.second;
  result.query_words[0] = query[0];
  result.query_words[1] = query[1];
  result.corpus_query_count = static_cast<std::uint64_t>(query_count);
  return result;
}

struct SyntheticFixture {
  bfnew::InputGraph graph;
  std::vector<bfnew::VertexId> sources;
  std::vector<bfnew::VertexId> targets;
};

[[nodiscard]] bfnew::PhysicalProvenance synthetic_provenance(
    const std::uint64_t source_record) noexcept {
  return bfnew::PhysicalProvenance{
      bfnew::provenance_domain::synthetic,
      bfnew::provenance_kind::synthetic_edge,
      source_record,
  };
}

[[nodiscard]] SyntheticFixture make_synthetic_fixture(
    const std::string_view case_name) {
  const bfnew::ResourceClassId routing_class{1U};
  if (case_name == "sparse-wavefront") {
    constexpr std::uint32_t vertex_count = 64U;
    std::vector<bfnew::VertexMetadata> vertices;
    std::vector<bfnew::EdgeInputRecord> edges;
    vertices.reserve(vertex_count);
    edges.reserve(vertex_count - 1U);
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex) {
      vertices.push_back(bfnew::VertexMetadata::located(
          static_cast<std::int32_t>(vertex), 0, routing_class));
      if (vertex != 0U) {
        edges.push_back(bfnew::EdgeInputRecord{
            bfnew::VertexId{vertex - 1U},
            bfnew::VertexId{vertex},
            1.0F,
            synthetic_provenance(vertex - 1U),
        });
      }
    }
    return SyntheticFixture{
        bfnew::InputGraph{std::move(vertices), std::move(edges)},
        {bfnew::VertexId{0U}},
        {bfnew::VertexId{vertex_count - 1U}},
    };
  }

  if (case_name != "dense-frontier") {
    throw std::invalid_argument{"unknown built-in synthetic case"};
  }
  constexpr std::uint32_t source = 0U;
  constexpr std::uint32_t first_begin = 1U;
  constexpr std::uint32_t first_end = 17U;
  constexpr std::uint32_t second_begin = first_end;
  constexpr std::uint32_t second_end = 49U;
  constexpr std::uint32_t third_begin = second_end;
  constexpr std::uint32_t third_end = 65U;
  std::vector<bfnew::VertexMetadata> vertices;
  vertices.reserve(third_end);
  for (std::uint32_t vertex = 0U; vertex < third_end; ++vertex) {
    vertices.push_back(bfnew::VertexMetadata::located(
        static_cast<std::int32_t>(vertex % 8U),
        static_cast<std::int32_t>(vertex / 8U),
        routing_class));
  }
  std::vector<bfnew::EdgeInputRecord> edges;
  edges.reserve(
      (first_end - first_begin) +
      (first_end - first_begin) * (second_end - second_begin) +
      (second_end - second_begin) * (third_end - third_begin));
  std::uint64_t record = 0U;
  for (std::uint32_t first = first_begin; first < first_end; ++first) {
    edges.push_back(bfnew::EdgeInputRecord{
        bfnew::VertexId{source},
        bfnew::VertexId{first},
        1.0F,
        synthetic_provenance(record++),
    });
  }
  for (std::uint32_t first = first_begin; first < first_end; ++first) {
    for (std::uint32_t second = second_begin; second < second_end; ++second) {
      edges.push_back(bfnew::EdgeInputRecord{
          bfnew::VertexId{first},
          bfnew::VertexId{second},
          1.0F + static_cast<float>((first + second) % 3U) * 0.125F,
          synthetic_provenance(record++),
      });
    }
  }
  for (std::uint32_t second = second_begin; second < second_end; ++second) {
    for (std::uint32_t third = third_begin; third < third_end; ++third) {
      edges.push_back(bfnew::EdgeInputRecord{
          bfnew::VertexId{second},
          bfnew::VertexId{third},
          1.0F + static_cast<float>((second + third) % 2U) * 0.25F,
          synthetic_provenance(record++),
      });
    }
  }
  std::vector<bfnew::VertexId> targets;
  targets.reserve(third_end - third_begin);
  for (std::uint32_t third = third_begin; third < third_end; ++third) {
    targets.push_back(bfnew::VertexId{third});
  }
  return SyntheticFixture{
      bfnew::InputGraph{std::move(vertices), std::move(edges)},
      {bfnew::VertexId{source}},
      std::move(targets),
  };
}

void fingerprint_word(
    std::array<std::uint64_t, 2U>& state,
    const std::uint64_t word) noexcept {
  state[0] = (state[0] ^ word) * 1099511628211ULL;
  state[1] ^= word + 0x9e3779b97f4a7c15ULL + (state[1] << 6U) +
              (state[1] >> 2U);
}

[[nodiscard]] bfnew::ShootoutInputFingerprint make_synthetic_fingerprint(
    const bfnew::DeviceGraphFingerprint& graph,
    const std::string_view case_name,
    const std::span<const bfnew::MappedRouteQuery> queries) {
  std::array<std::uint64_t, 2U> query_words{
      1469598103934665603ULL, 0x9e3779b97f4a7c15ULL};
  for (const char character : case_name) {
    fingerprint_word(
        query_words, static_cast<unsigned char>(character));
  }
  fingerprint_word(query_words, static_cast<std::uint64_t>(queries.size()));
  for (const bfnew::MappedRouteQuery& mapped : queries) {
    fingerprint_word(query_words, mapped.query.query_id.value());
    fingerprint_word(
        query_words,
        static_cast<std::uint64_t>(mapped.query.sources.size()));
    for (const bfnew::VertexId source : mapped.query.sources) {
      fingerprint_word(query_words, source.value());
    }
    fingerprint_word(
        query_words,
        static_cast<std::uint64_t>(mapped.query.targets.size()));
    for (const bfnew::VertexId target : mapped.query.targets) {
      fingerprint_word(query_words, target.value());
    }
    fingerprint_word(
        query_words,
        static_cast<std::uint64_t>(mapped.query.selected_tiles.size()));
    for (const bfnew::TileId tile : mapped.query.selected_tiles) {
      fingerprint_word(query_words, tile.value());
    }
    fingerprint_word(query_words, mapped.query.expansion_generation);
  }
  return bfnew::ShootoutInputFingerprint{
      {graph.first, graph.second},
      {query_words[0], query_words[1]},
      static_cast<std::uint64_t>(queries.size()),
      bfnew::shootout_schema_version,
  };
}

[[nodiscard]] bfnew::ShootoutWorkloadIdentity workload_identity(
    const Options& options,
    const bfnew::ShootoutInputFingerprint& fingerprint) {
  std::uint64_t case_id =
      fingerprint.graph_words[0] ^ (fingerprint.graph_words[1] << 1U) ^
      fingerprint.query_words[0] ^ (fingerprint.query_words[1] << 7U);
  if (case_id == 0U) {
    case_id = 1U;
  }
  return bfnew::ShootoutWorkloadIdentity{
      options.workload == WorkloadMode::logicnets_jscl
          ? bfnew::ShootoutWorkloadKind::logicnets_jscl
          : bfnew::ShootoutWorkloadKind::synthetic,
      case_id,
      options.workload == WorkloadMode::logicnets_jscl
          ? std::string{"logicnets_jscl"}
          : options.synthetic_case_name,
  };
}

struct LoadedCampaign {
  bfnew::PartitionedGraph partitioned;
  bfnew::TileRunLayout64 runs;
  bfnew::DeviceGraphFingerprint graph_fingerprint{};
  std::vector<bfnew::MappedRouteQuery> mapped_queries;
  bfnew::ShootoutInputFingerprint fingerprint{};
  bfnew::ShootoutWorkloadIdentity workload;
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  bfnew::hip::ReusableDeviceWorkspace workspace;

  LoadedCampaign(
      bfnew::PartitionedGraph graph,
      bfnew::TileRunLayout64 tile_runs,
      const bfnew::DeviceGraphFingerprint graph_identity,
      std::vector<bfnew::MappedRouteQuery> queries,
      const bfnew::ShootoutInputFingerprint input_identity,
      bfnew::ShootoutWorkloadIdentity workload_identity)
      : partitioned{std::move(graph)},
        runs{std::move(tile_runs)},
        graph_fingerprint{graph_identity},
        mapped_queries{std::move(queries)},
        fingerprint{input_identity},
        workload{std::move(workload_identity)} {}
};

[[nodiscard]] LoadedCampaign load_campaign(const Options& options) {
  std::optional<SyntheticFixture> synthetic;
  bfnew::InputGraph input = [&]() {
    if (options.workload == WorkloadMode::logicnets_jscl) {
      return bfnew::read_graph_artifact(options.graph_path);
    }
    synthetic.emplace(make_synthetic_fixture(options.synthetic_case_name));
    return std::move(synthetic->graph);
  }();
  const bfnew::UniformGridPartitioner partitioner(bfnew::SpatialOrderConfig{
      0, 0, options.tile_width, options.tile_height});
  bfnew::PartitionedGraph partitioned = partitioner.partition(input);
  if (!bfnew::validate_weighted_graph(partitioned.graph).ok() ||
      !bfnew::validate_tile_directory(
           partitioned.graph, partitioned.tiles).ok()) {
    throw std::runtime_error{"shootout real graph failed deep validation"};
  }
  bfnew::TileRunLayout64 runs =
      bfnew::build_tile_run_layout(partitioned.graph);
  // Immutable run metadata is deep-validated once at campaign load. Per-query
  // feature/oracle paths thereafter traverse only selected tile ranges/runs.
  if (!bfnew::validate_tile_run_layout(partitioned.graph, runs).ok()) {
    throw std::runtime_error{
        "shootout campaign tile-run layout failed one-time validation"};
  }
  bfnew::DeviceGraphLayout32 layout =
      bfnew::build_device_graph_layout32(partitioned.graph, runs);
  const bfnew::DeviceGraphFingerprint graph_fingerprint =
      bfnew::fingerprint_device_graph_layout32(layout);
  std::vector<bfnew::MappedRouteQuery> mapped;
  if (options.workload == WorkloadMode::logicnets_jscl) {
    mapped = bfnew::read_query_artifact(
        options.queries_path, partitioned.graph);
  } else {
    const auto remap = [&](const std::span<const bfnew::VertexId> original) {
      std::vector<bfnew::VertexId> result;
      result.reserve(original.size());
      for (const bfnew::VertexId vertex : original) {
        result.push_back(partitioned.graph.old_to_new()[vertex.value()]);
      }
      return result;
    };
    const std::vector<bfnew::VertexId> sources = remap(synthetic->sources);
    const std::vector<bfnew::VertexId> targets = remap(synthetic->targets);
    mapped.push_back(bfnew::MappedRouteQuery{
        options.synthetic_case_name,
        bfnew::make_route_query(
            bfnew::QueryId{0U},
            partitioned.graph,
            sources,
            targets,
            0U),
        {},
        {},
    });
  }
  if (options.workload == WorkloadMode::logicnets_jscl &&
      mapped.size() < bfnew::minimum_logicnets_shootout_queries) {
    throw std::runtime_error{
        "logicnets query artifact contains fewer than 1000 queries"};
  }
  const bfnew::ShootoutInputFingerprint fingerprint =
      options.workload == WorkloadMode::logicnets_jscl
          ? make_input_fingerprint(
                graph_fingerprint, options.queries_path, mapped.size())
          : make_synthetic_fingerprint(
                graph_fingerprint, options.synthetic_case_name, mapped);
  LoadedCampaign campaign{
      std::move(partitioned),
      std::move(runs),
      graph_fingerprint,
      std::move(mapped),
      fingerprint,
      workload_identity(options, fingerprint),
  };
  campaign.resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(std::move(layout)),
      campaign.stream);
  return campaign;
}

[[nodiscard]] std::uint64_t maximum_rounds(
    const Options& options,
    const bfnew::WeightedGraph& graph) {
  if (options.maximum_rounds != 0U) {
    return options.maximum_rounds;
  }
  return static_cast<std::uint64_t>(graph.vertex_count()) + 1U;
}

[[nodiscard]] std::map<std::uint32_t, const bfnew::RouteQuery*> query_index(
    const std::vector<bfnew::MappedRouteQuery>& queries) {
  std::map<std::uint32_t, const bfnew::RouteQuery*> result;
  for (const bfnew::MappedRouteQuery& mapped : queries) {
    if (!result.emplace(mapped.query.query_id.value(), &mapped.query).second) {
      throw std::runtime_error{"query artifact contains duplicate query IDs"};
    }
  }
  return result;
}

[[nodiscard]] std::vector<float> bounded_oracle(
    const bfnew::WeightedGraph& graph,
    const bfnew::TileRunLayout64& runs,
    const bfnew::RouteQuery& query) {
  if (!bfnew::validate_route_query(graph, query).ok()) {
    throw std::invalid_argument{
        "shootout oracle requires a valid route query"};
  }
  const auto tile_offsets = graph.tile_vertex_offsets();
  std::map<bfnew::TileId, std::uint32_t> local_bases;
  std::vector<bfnew::VertexMetadata> vertices;
  for (const bfnew::TileId tile : query.selected_tiles) {
    if (!std::in_range<std::uint32_t>(vertices.size())) {
      throw std::overflow_error{
          "shootout selected oracle vertex count exceeds 32 bits"};
    }
    local_bases.emplace(
        tile, static_cast<std::uint32_t>(vertices.size()));
    const std::size_t begin =
        static_cast<std::size_t>(tile_offsets[tile.value()]);
    const std::size_t end = static_cast<std::size_t>(
        tile_offsets[tile.value() + 1U]);
    vertices.insert(
        vertices.end(),
        graph.vertices().begin() + static_cast<std::ptrdiff_t>(begin),
        graph.vertices().begin() + static_cast<std::ptrdiff_t>(end));
  }

  const auto local_vertex = [&](const bfnew::VertexId global) {
    const bfnew::TileId tile = graph.owner_tiles()[global.value()];
    const auto base = local_bases.find(tile);
    if (base == local_bases.end()) {
      throw std::logic_error{
          "shootout oracle encountered a vertex outside selected tiles"};
    }
    const std::uint64_t local =
        static_cast<std::uint64_t>(base->second) + global.value() -
        tile_offsets[tile.value()];
    if (!std::in_range<std::uint32_t>(local)) {
      throw std::overflow_error{"shootout oracle local vertex overflow"};
    }
    return bfnew::VertexId{static_cast<std::uint32_t>(local)};
  };

  std::vector<bfnew::EdgeInputRecord> edges;
  const bfnew::OutgoingCsrView outgoing = graph.outgoing();
  for (const bfnew::TileId source_tile : query.selected_tiles) {
    const std::size_t source_begin =
        static_cast<std::size_t>(tile_offsets[source_tile.value()]);
    const std::size_t source_end = static_cast<std::size_t>(
        tile_offsets[source_tile.value() + 1U]);
    for (std::size_t source = source_begin; source < source_end; ++source) {
      const std::size_t run_begin =
          static_cast<std::size_t>(runs.csr_row_run_offsets[source]);
      const std::size_t run_end =
          static_cast<std::size_t>(runs.csr_row_run_offsets[source + 1U]);
      for (std::size_t run = run_begin; run < run_end; ++run) {
        if (!local_bases.contains(runs.csr_run_destination_tiles[run])) {
          continue;
        }
        const std::size_t edge_begin =
            static_cast<std::size_t>(runs.csr_run_edge_offsets[run]);
        const std::size_t edge_end =
            static_cast<std::size_t>(runs.csr_run_edge_offsets[run + 1U]);
        for (std::size_t edge = edge_begin; edge < edge_end; ++edge) {
          const bfnew::EdgeId edge_id = outgoing.edge_ids[edge];
          edges.push_back(bfnew::EdgeInputRecord{
              local_vertex(bfnew::VertexId{static_cast<std::uint32_t>(source)}),
              local_vertex(outgoing.destinations[edge]),
              outgoing.weights[edge],
              graph.edge_provenance()[edge_id.value()],
          });
        }
      }
    }
  }
  std::vector<bfnew::VertexId> local_sources;
  local_sources.reserve(query.sources.size());
  for (const bfnew::VertexId source : query.sources) {
    local_sources.push_back(local_vertex(source));
  }
  const bfnew::WeightedGraph induced = bfnew::build_weighted_graph(
      bfnew::InputGraph{std::move(vertices), std::move(edges)});
  return bfnew::dijkstra_oracle(induced, local_sources).distances;
}

void require_current_identity(
    const LoadedCampaign& campaign,
    const bfnew::ShootoutManifest& manifest,
    const bfnew::ShootoutConfigurationCatalog& catalog) {
  if (manifest.fingerprint != campaign.fingerprint ||
      catalog.fingerprint != campaign.fingerprint ||
      manifest.workload != campaign.workload ||
      catalog.workload != campaign.workload) {
    throw std::runtime_error{
        "shootout persisted input identity does not match current artifacts"};
  }
}

[[nodiscard]] bfnew::ShootoutManifest load_manifest(
    const std::filesystem::path& path) {
  return bfnew::deserialize_shootout_manifest_tsv(read_text(path));
}

[[nodiscard]] bfnew::ShootoutConfigurationCatalog load_catalog(
    const std::filesystem::path& path) {
  return bfnew::deserialize_shootout_catalog_tsv(read_text(path));
}

void require_replay_policy(
    const Options& options,
    const bfnew::ShootoutManifest& manifest) {
  if (options.has_selection_seed &&
      options.selection_seed != manifest.selection_seed) {
    throw std::invalid_argument{
        "--selection-seed does not match the persisted manifest policy"};
  }
  if (options.has_order_seed && options.order_seed != manifest.order_seed) {
    throw std::invalid_argument{
        "--order-seed does not match the persisted manifest policy"};
  }
  if (options.has_warmups &&
      options.warmups != manifest.warmup_repetitions) {
    throw std::invalid_argument{
        "--warmups does not match the persisted manifest policy"};
  }
  if (options.has_repetitions &&
      options.repetitions != manifest.timing_repetitions) {
    throw std::invalid_argument{
        "--repetitions does not match the persisted manifest policy"};
  }
  if (options.has_query_count &&
      options.query_count != manifest.requested_query_count) {
    throw std::invalid_argument{
        "--query-count does not match the persisted manifest policy"};
  }
}

[[nodiscard]] std::vector<bfnew::ShootoutSample> load_correctness(
    const Options& options,
    const bfnew::ShootoutManifest& manifest,
    const bfnew::ShootoutConfigurationCatalog& catalog) {
  return bfnew::deserialize_shootout_samples_tsv(
      manifest,
      catalog.tunings,
      read_text(options.correctness_path),
      bfnew::ShootoutRunKind::correctness);
}

void write_samples(
    const std::filesystem::path& path,
    const bfnew::ShootoutManifest& manifest,
    const bfnew::ShootoutConfigurationCatalog& catalog,
    const std::span<const bfnew::ShootoutSample> samples,
    const bfnew::ShootoutRunKind kind) {
  write_text(
      path,
      bfnew::serialize_shootout_samples_tsv(
      manifest, catalog.tunings, samples, kind));
}

void write_configuration_decisions(
    const std::filesystem::path& path,
    const bfnew::InstrumentationLevel instrumentation,
    const std::span<const bfnew::ShootoutConfigurationDecision> decisions) {
  std::string output =
      "configuration_id\tengine\tcontrol\tblock_size\tchunk_size\t"
      "grid_policy\tblocks_per_wgp\tinstrumentation\t"
      "persistent_active_blocks_per_wgp\trejection\n";
  for (const bfnew::ShootoutConfigurationDecision& decision : decisions) {
    output += std::to_string(decision.tuning.configuration_id) + '\t' +
              std::to_string(static_cast<std::uint32_t>(
                  decision.tuning.engine)) + '\t' +
              std::to_string(static_cast<std::uint32_t>(
                  decision.tuning.control_mode)) + '\t' +
              std::to_string(decision.tuning.block_size) + '\t' +
              std::to_string(decision.tuning.rounds_per_chunk) + '\t' +
              std::to_string(static_cast<std::uint32_t>(
                  decision.tuning.grid_policy)) + '\t' +
              std::to_string(decision.tuning.blocks_per_wgp) + '\t' +
              std::to_string(static_cast<std::uint32_t>(instrumentation)) +
              '\t' +
              std::to_string(
                  decision.measured_persistent_active_blocks_per_wgp) +
              '\t' +
              std::to_string(static_cast<std::uint32_t>(
                  decision.rejection)) + '\n';
  }
  write_text(path, output);
}

void run_pilot(
    const Options& options,
    LoadedCampaign& campaign,
    bfnew::hip::EngineShootoutExecutor& executor) {
  if (options.workload == WorkloadMode::logicnets_jscl &&
      options.query_count > campaign.mapped_queries.size()) {
    throw std::runtime_error{
        "requested shootout manifest exceeds the logicnets corpus"};
  }
  bfnew::hip::ShootoutTilePairIndex tile_pairs{
      campaign.partitioned.graph, campaign.runs};
  const std::uint64_t round_limit =
      maximum_rounds(options, campaign.partitioned.graph);
  std::vector<bfnew::ShootoutQueryFeatures> four_feature_candidates;
  four_feature_candidates.reserve(campaign.mapped_queries.size());
  for (const bfnew::MappedRouteQuery& mapped : campaign.mapped_queries) {
    if (!std::in_range<std::uint32_t>(mapped.query.target_terminals.size()) ||
        !std::in_range<std::uint32_t>(mapped.query.sources.size())) {
      throw std::overflow_error{"shootout query feature exceeds 32 bits"};
    }
    four_feature_candidates.push_back(bfnew::ShootoutQueryFeatures{
        mapped.query.query_id,
        tile_pairs.selected_vertex_count(mapped.query.selected_tiles),
        tile_pairs.selected_edge_count(mapped.query.selected_tiles),
        static_cast<std::uint32_t>(mapped.query.target_terminals.size()),
        static_cast<std::uint32_t>(mapped.query.sources.size()),
        1U,
    });
  }
  std::vector<bfnew::ShootoutQueryFeatures> candidates;
  if (options.workload == WorkloadMode::logicnets_jscl) {
    const std::uint64_t target =
        static_cast<std::uint64_t>(options.query_count) * 4U;
    const std::size_t oversample_size = std::min<std::size_t>(
        campaign.mapped_queries.size(),
        static_cast<std::size_t>(std::min<std::uint64_t>(
            target, std::numeric_limits<std::uint32_t>::max())));
    const bfnew::ShootoutManifest oversample =
        bfnew::select_logicnets_shootout_manifest(
            campaign.fingerprint,
            campaign.workload,
            four_feature_candidates,
            static_cast<std::uint32_t>(oversample_size),
            options.selection_seed ^ prescreen_seed_domain,
            options.order_seed,
            options.warmups,
            options.repetitions);
    const auto queries = query_index(campaign.mapped_queries);
    candidates.reserve(oversample.entries.size());
    for (const bfnew::ShootoutManifestEntry& entry : oversample.entries) {
      bfnew::ShootoutQueryFeatures feature = entry.features;
      feature.expected_rounds = executor.run_jacobi_pilot(
          *queries.at(feature.query_id.value()), round_limit);
      candidates.push_back(feature);
    }
  } else {
    candidates = four_feature_candidates;
    const auto queries = query_index(campaign.mapped_queries);
    for (bfnew::ShootoutQueryFeatures& feature : candidates) {
      feature.expected_rounds = executor.run_jacobi_pilot(
          *queries.at(feature.query_id.value()), round_limit);
    }
  }
  const bfnew::ShootoutManifest manifest =
      options.workload == WorkloadMode::logicnets_jscl
          ? bfnew::select_logicnets_shootout_manifest(
                campaign.fingerprint,
                campaign.workload,
                candidates,
                options.query_count,
                options.selection_seed,
                options.order_seed,
                options.warmups,
                options.repetitions)
          : bfnew::make_synthetic_shootout_manifest(
                campaign.fingerprint,
                campaign.workload,
                candidates,
                options.order_seed,
                options.warmups,
                options.repetitions);
  const auto queries = query_index(campaign.mapped_queries);
  const bfnew::RouteQuery& occupancy_query =
      *queries.at(manifest.entries.front().features.query_id.value());
  const std::vector<bfnew::ShootoutKernelLimit> limits =
      executor.probe_kernel_limits(
          occupancy_query, bfnew::InstrumentationLevel::none);
  std::uint32_t maximum_residency = 0U;
  for (const bfnew::ShootoutKernelLimit& limit : limits) {
    maximum_residency =
        std::max(
            maximum_residency,
            limit.persistent_active_blocks_per_wgp);
  }
  std::vector<std::uint32_t> fixed_residencies;
  for (std::uint32_t value = 1U; value <= maximum_residency; ++value) {
    fixed_residencies.push_back(value);
  }
  const std::vector<bfnew::ShootoutTuning> requested =
      bfnew::make_shootout_tunings(round_limit, fixed_residencies);
  const std::vector<bfnew::ShootoutConfigurationDecision> decisions =
      bfnew::resolve_shootout_configurations(requested, limits);
  write_configuration_decisions(
      options.output_directory / "configuration-decisions.v1.tsv",
      bfnew::InstrumentationLevel::none,
      decisions);
  bfnew::ShootoutConfigurationCatalog catalog;
  catalog.fingerprint = campaign.fingerprint;
  catalog.workload = campaign.workload;
  for (const bfnew::ShootoutConfigurationDecision& decision : decisions) {
    if (decision.rejection ==
        bfnew::ShootoutConfigurationRejection::none) {
      catalog.tunings.push_back(decision.tuning);
    }
  }
  bfnew::validate_shootout_configuration_catalog(catalog);
  for (const bfnew::EngineKind engine : {
           bfnew::EngineKind::jacobi_pull,
           bfnew::EngineKind::dense_chaotic_push,
           bfnew::EngineKind::frontier_push}) {
    if (std::ranges::none_of(
            catalog.tunings,
            [engine](const bfnew::ShootoutTuning& tuning) {
              return tuning.engine == engine;
            })) {
      throw std::runtime_error{
          "shootout legality discovery produced no configuration for one of "
          "the three required engines"};
    }
  }

  write_text(
      options.output_directory / "manifest.v1.tsv",
      bfnew::serialize_shootout_manifest_tsv(manifest));
  write_text(
      options.output_directory / "catalog.v1.tsv",
      bfnew::serialize_shootout_catalog_tsv(catalog));

  std::string pilot =
      "query_id\tselected_vertices\tselected_edges\tfanout\tsource_count\t"
      "jacobi_expected_rounds\n";
  std::ranges::sort(candidates, {}, &bfnew::ShootoutQueryFeatures::query_id);
  for (const bfnew::ShootoutQueryFeatures& feature : candidates) {
    pilot += std::to_string(feature.query_id.value()) + '\t' +
             std::to_string(feature.selected_vertices) + '\t' +
             std::to_string(feature.selected_edges) + '\t' +
             std::to_string(feature.fanout) + '\t' +
             std::to_string(feature.source_count) + '\t' +
             std::to_string(feature.expected_rounds) + '\n';
  }
  write_text(options.output_directory / "pilot.v1.tsv", pilot);

  std::string metadata;
  metadata += "schema=" +
              std::to_string(bfnew::shootout_schema_version) + '\n';
  metadata += "workload=" + campaign.workload.case_name + '\n';
  metadata += "graph_path=" +
              (options.workload == WorkloadMode::logicnets_jscl
                   ? options.graph_path.string()
                   : std::string{"<built-in>"}) +
              '\n';
  metadata += "query_path=" +
              (options.workload == WorkloadMode::logicnets_jscl
                   ? options.queries_path.string()
                   : std::string{"<built-in>"}) +
              '\n';
  metadata += "graph_fingerprint_0=" +
              std::to_string(campaign.fingerprint.graph_words[0]) + '\n';
  metadata += "graph_fingerprint_1=" +
              std::to_string(campaign.fingerprint.graph_words[1]) + '\n';
  metadata += "query_fingerprint_0=" +
              std::to_string(campaign.fingerprint.query_words[0]) + '\n';
  metadata += "query_fingerprint_1=" +
              std::to_string(campaign.fingerprint.query_words[1]) + '\n';
  metadata += "corpus_queries=" +
              std::to_string(campaign.fingerprint.corpus_query_count) + '\n';
  metadata += "selected_queries=" +
              std::to_string(manifest.entries.size()) + '\n';
  metadata += "jacobi_pilot_queries=" +
              std::to_string(candidates.size()) + '\n';
  metadata += "selection_seed=" +
              std::to_string(manifest.selection_seed) + '\n';
  metadata += "prescreen_seed=" +
              (options.workload == WorkloadMode::logicnets_jscl
                   ? std::to_string(
                         options.selection_seed ^ prescreen_seed_domain)
                   : std::string{"not_applicable"}) +
              '\n';
  metadata += "order_seed=" + std::to_string(options.order_seed) + '\n';
  metadata += "warmup_repetitions=" +
              std::to_string(manifest.warmup_repetitions) + '\n';
  metadata += "timing_repetitions=" +
              std::to_string(manifest.timing_repetitions) + '\n';
  metadata += "tile_width=" + std::to_string(options.tile_width) + '\n';
  metadata += "tile_height=" + std::to_string(options.tile_height) + '\n';
  metadata += "maximum_rounds=" + std::to_string(round_limit) + '\n';
  metadata += "configuration_count=" +
              std::to_string(catalog.tunings.size()) + '\n';
  metadata += "gpu_measurements=pilot_only_not_timing_evidence\n";
  write_text(options.output_directory / "metadata.v1.txt", metadata);
}

[[nodiscard]] std::vector<bfnew::ShootoutSample> execute_schedule(
    bfnew::hip::EngineShootoutExecutor& executor,
    const bfnew::ShootoutManifest& manifest,
    const bfnew::ShootoutConfigurationCatalog& catalog,
    const std::map<std::uint32_t, const bfnew::RouteQuery*>& queries,
    const std::vector<bfnew::ShootoutScheduleEntry>& schedule,
    const bool correctness,
    const bfnew::WeightedGraph& graph,
    const bfnew::TileRunLayout64& runs) {
  std::vector<bfnew::ShootoutSample> samples;
  samples.reserve(schedule.size());
  std::optional<bfnew::QueryId> oracle_query;
  std::vector<float> oracle;
  std::set<std::uint32_t> oracle_queries;
  for (const bfnew::ShootoutScheduleEntry& entry : schedule) {
    const auto found = queries.find(entry.query_id.value());
    if (found == queries.end()) {
      throw std::runtime_error{"manifest query is absent from query artifact"};
    }
    if (correctness && (!oracle_query || *oracle_query != entry.query_id)) {
      if (!oracle_queries.insert(entry.query_id.value()).second) {
        throw std::logic_error{
            "correctness schedule would rebuild one query oracle"};
      }
      oracle = bounded_oracle(graph, runs, *found->second);
      oracle_query = entry.query_id;
    }
    bfnew::hip::ShootoutEngineExecution execution = executor.execute(
        *found->second,
        entry,
        correctness ? std::span<const float>{oracle}
                    : std::span<const float>{},
        correctness && manifest.workload.kind ==
                           bfnew::ShootoutWorkloadKind::synthetic
            ? bfnew::hip::ShootoutDistanceComparison::bitwise
            : bfnew::hip::ShootoutDistanceComparison::within_four_ulps);
    samples.push_back(std::move(execution.sample));
  }
  bfnew::validate_shootout_samples(
      manifest,
      catalog.tunings,
      samples,
      schedule.empty() ? bfnew::ShootoutRunKind::correctness
                       : schedule.front().run_kind);
  if (correctness && oracle_queries.size() != manifest.entries.size()) {
    throw std::logic_error{
        "correctness schedule did not build exactly one oracle per query"};
  }
  return samples;
}

void run_correctness(
    const Options& options,
    LoadedCampaign& campaign,
    bfnew::hip::EngineShootoutExecutor& executor,
    const bfnew::ShootoutManifest& manifest,
    const bfnew::ShootoutConfigurationCatalog& catalog) {
  const auto queries = query_index(campaign.mapped_queries);
  const std::vector<bfnew::ShootoutScheduleEntry> schedule =
      bfnew::make_grouped_shootout_correctness_schedule(
          manifest, catalog.tunings);
  executor.begin_stage(
      manifest,
      catalog,
      {},
      bfnew::ShootoutRunKind::correctness,
      bfnew::InstrumentationLevel::none);
  const std::vector<bfnew::ShootoutSample> samples = execute_schedule(
      executor,
      manifest,
      catalog,
      queries,
      schedule,
      true,
      campaign.partitioned.graph,
      campaign.runs);
  write_samples(
      options.output_directory / "correctness.v1.tsv",
      manifest,
      catalog,
      samples,
      bfnew::ShootoutRunKind::correctness);
  if (std::ranges::any_of(samples, [](const bfnew::ShootoutSample& sample) {
        return !sample.correctness_passed;
      })) {
    throw std::runtime_error{
        "shootout correctness failed; later stages remain blocked"};
  }
}

void run_counters(
    const Options& options,
    LoadedCampaign& campaign,
    bfnew::hip::EngineShootoutExecutor& executor,
    const bfnew::ShootoutManifest& manifest,
    const bfnew::ShootoutConfigurationCatalog& catalog,
    const std::vector<bfnew::ShootoutSample>& correctness) {
  const auto queries = query_index(campaign.mapped_queries);
  const bfnew::RouteQuery& discovery_query = *queries.at(
      manifest.entries.front().features.query_id.value());
  const std::vector<bfnew::ShootoutKernelLimit> debug_limits =
      executor.probe_kernel_limits(
          discovery_query, bfnew::InstrumentationLevel::debug);
  const std::vector<bfnew::ShootoutConfigurationDecision> decisions =
      bfnew::resolve_shootout_configurations(
          catalog.tunings, debug_limits);
  write_configuration_decisions(
      options.output_directory / "counter-configuration-decisions.v1.tsv",
      bfnew::InstrumentationLevel::debug,
      decisions);
  std::vector<bfnew::ShootoutTuning> counter_tunings;
  for (const bfnew::ShootoutConfigurationDecision& decision : decisions) {
    if (decision.rejection ==
        bfnew::ShootoutConfigurationRejection::none) {
      counter_tunings.push_back(decision.tuning);
    }
  }
  if (counter_tunings.empty()) {
    throw std::runtime_error{
        "Debug kernel discovery produced no legal counter configuration"};
  }
  const std::vector<bfnew::ShootoutScheduleEntry> schedule =
      bfnew::make_interleaved_shootout_schedule(
          manifest,
          counter_tunings,
          bfnew::ShootoutRunKind::algorithm_counters,
          1U,
          manifest.order_seed);
  executor.begin_stage(
      manifest,
      catalog,
      correctness,
      bfnew::ShootoutRunKind::algorithm_counters,
      options.counter_instrumentation);
  const std::vector<bfnew::ShootoutSample> samples = execute_schedule(
      executor,
      manifest,
      catalog,
      queries,
      schedule,
      false,
      campaign.partitioned.graph,
      campaign.runs);
  write_samples(
      options.output_directory / "counters.v1.tsv",
      manifest,
      catalog,
      samples,
      bfnew::ShootoutRunKind::algorithm_counters);
}

void run_timing(
    const Options& options,
    LoadedCampaign& campaign,
    bfnew::hip::EngineShootoutExecutor& executor,
    const bfnew::ShootoutManifest& manifest,
    const bfnew::ShootoutConfigurationCatalog& catalog,
    const std::vector<bfnew::ShootoutSample>& correctness) {
  const auto queries = query_index(campaign.mapped_queries);
  const auto warmup_schedule = bfnew::make_interleaved_shootout_schedule(
      manifest,
      catalog.tunings,
      bfnew::ShootoutRunKind::warmup,
      manifest.warmup_repetitions,
      manifest.order_seed);
  executor.begin_stage(
      manifest,
      catalog,
      correctness,
      bfnew::ShootoutRunKind::warmup,
      bfnew::InstrumentationLevel::none);
  static_cast<void>(execute_schedule(
      executor,
      manifest,
      catalog,
      queries,
      warmup_schedule,
      false,
      campaign.partitioned.graph,
      campaign.runs));

  const auto timing_schedule = bfnew::make_interleaved_shootout_schedule(
      manifest,
      catalog.tunings,
      bfnew::ShootoutRunKind::timing,
      manifest.timing_repetitions,
      manifest.order_seed);
  executor.begin_stage(
      manifest,
      catalog,
      correctness,
      bfnew::ShootoutRunKind::timing,
      bfnew::InstrumentationLevel::none);
  const std::vector<bfnew::ShootoutSample> samples = execute_schedule(
      executor,
      manifest,
      catalog,
      queries,
      timing_schedule,
      false,
      campaign.partitioned.graph,
      campaign.runs);
  write_samples(
      options.output_directory / "timing.v1.tsv",
      manifest,
      catalog,
      samples,
      bfnew::ShootoutRunKind::timing);
}

void run_profile_case(
    const Options& options,
    LoadedCampaign& campaign,
    bfnew::hip::EngineShootoutExecutor& executor,
    const bfnew::ShootoutManifest& manifest,
    const bfnew::ShootoutConfigurationCatalog& catalog,
    const std::vector<bfnew::ShootoutSample>& correctness) {
  const auto queries = query_index(campaign.mapped_queries);
  const auto schedule = bfnew::make_interleaved_shootout_schedule(
      manifest,
      catalog.tunings,
      options.profile_kind,
      1U,
      manifest.order_seed);
  std::vector<const bfnew::ShootoutScheduleEntry*> selected_cases;
  selected_cases.reserve(options.profile_case_ids.size());
  for (const std::uint64_t case_id : options.profile_case_ids) {
    const auto selected = std::ranges::find(
        schedule,
        case_id,
        &bfnew::ShootoutScheduleEntry::execution_ordinal);
    if (selected == schedule.end()) {
      throw std::out_of_range{
          "profile case ID is outside the stable manifest schedule"};
    }
    selected_cases.push_back(&*selected);
  }

  std::vector<bfnew::ShootoutSample> samples;
  samples.reserve(selected_cases.size());
  std::string ledger =
      "range_index\tprofile_case_id\tquery_id\tconfiguration_id\t"
      "begin_marker\tend_marker\n";
  for (std::size_t range_index = 0U;
       range_index < selected_cases.size();
       ++range_index) {
    const bfnew::ShootoutScheduleEntry& selected =
        *selected_cases[range_index];
    const bfnew::RouteQuery& query =
        *queries.at(selected.query_id.value());

    bfnew::ShootoutScheduleEntry warmup = selected;
    warmup.run_kind = bfnew::ShootoutRunKind::warmup;
    warmup.execution_ordinal = 0U;
    warmup.repetition = 0U;
    executor.begin_stage(
        manifest,
        catalog,
        correctness,
        bfnew::ShootoutRunKind::warmup,
        bfnew::InstrumentationLevel::none);
    static_cast<void>(executor.execute(query, warmup));

    executor.begin_stage(
        manifest,
        catalog,
        correctness,
        options.profile_kind,
        bfnew::InstrumentationLevel::none);
    executor.emit_profile_range_begin_marker();
    bfnew::hip::ShootoutEngineExecution execution =
        executor.execute(query, selected);
    executor.emit_profile_range_end_marker();
    execution.sample.profiler_provenance = bfnew::ShootoutProfilerProvenance{
        options.profiler_pass_id,
        options.profiler_counter_set_id,
        true,
    };
    samples.push_back(std::move(execution.sample));
    ledger += std::to_string(range_index) + '\t' +
              std::to_string(selected.execution_ordinal) + '\t' +
              std::to_string(selected.query_id.value()) + '\t' +
              std::to_string(selected.configuration_id) + '\t' +
              "bfnew_shootout_profile_range_begin_marker_kernel\t" +
              "bfnew_shootout_profile_range_end_marker_kernel\n";
  }
  write_samples(
      options.output_directory /
          (options.profile_kind == bfnew::ShootoutRunKind::trace
               ? "trace-case.v1.tsv"
               : "pmc-case.v1.tsv"),
      manifest,
      catalog,
      samples,
      options.profile_kind);
  write_text(
      options.output_directory / "profile-range-ledger.v1.tsv", ledger);
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::filesystem::create_directories(options.output_directory);

    if (options.stage == Stage::pilot) {
      LoadedCampaign campaign = load_campaign(options);
      bfnew::hip::EngineShootoutExecutor executor{
          campaign.partitioned.graph,
          campaign.runs,
          campaign.resident,
          campaign.workspace,
          campaign.stream};
      run_pilot(options, campaign, executor);
      return 0;
    }

    const bfnew::ShootoutManifest manifest =
        load_manifest(options.manifest_path);
    const bfnew::ShootoutConfigurationCatalog catalog =
        load_catalog(options.catalog_path);
    require_replay_policy(options, manifest);
    LoadedCampaign campaign = load_campaign(options);
    bfnew::hip::EngineShootoutExecutor executor{
        campaign.partitioned.graph,
        campaign.runs,
        campaign.resident,
        campaign.workspace,
        campaign.stream};
    require_current_identity(campaign, manifest, catalog);

    if (options.stage == Stage::correctness) {
      run_correctness(options, campaign, executor, manifest, catalog);
      return 0;
    }
    const std::vector<bfnew::ShootoutSample> correctness =
        load_correctness(options, manifest, catalog);
    if (options.stage == Stage::counters) {
      run_counters(
          options, campaign, executor, manifest, catalog, correctness);
    } else if (options.stage == Stage::timing) {
      run_timing(
          options, campaign, executor, manifest, catalog, correctness);
    } else {
      run_profile_case(
          options, campaign, executor, manifest, catalog, correctness);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "bfnew_gpu_shootout: " << error.what() << '\n';
    return 1;
  }
}
