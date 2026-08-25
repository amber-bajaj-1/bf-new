#include "bfnew/fpga_interchange.hpp"

#include "bfnew/graph.hpp"
#include "bfnew/query.hpp"
#include "bfnew/spatial.hpp"
#include "bfnew/sssp.hpp"

#include <kj/exception.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::filesystem::path data_root;
  std::filesystem::path output_root;
  std::string medium_name{"vtr_mcml_unrouted.phys"};
  std::uint32_t tile_width{8U};
  std::uint32_t tile_height{8U};
  std::uint32_t artifact_padding{1U};
};

struct Distribution {
  double minimum{};
  double median{};
  double p90{};
  double p99{};
  double maximum{};
  double mean{};
  std::size_t count{};
};

struct ScanRow {
  std::string file;
  bfnew::PhysicalScanStatistics statistics;
};

struct InputFingerprint {
  std::uintmax_t size{};
  std::filesystem::file_time_type modified{};
};

[[nodiscard]] std::uint32_t parse_u32(const std::string& text, const char* name) {
  std::size_t consumed = 0U;
  const unsigned long value = std::stoul(text, &consumed, 10);
  if (consumed != text.size() || !std::in_range<std::uint32_t>(value)) {
    throw std::invalid_argument{std::string{name} + " must be a 32-bit unsigned integer"};
  }
  return static_cast<std::uint32_t>(value);
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto require_value = [&](const char* option) -> std::string {
      if (index + 1 >= argc) {
        throw std::invalid_argument{std::string{option} + " requires a value"};
      }
      return argv[++index];
    };
    if (argument == "--data-root") {
      options.data_root = require_value("--data-root");
    } else if (argument == "--output-root") {
      options.output_root = require_value("--output-root");
    } else if (argument == "--medium") {
      options.medium_name = require_value("--medium");
    } else if (argument == "--tile-width") {
      options.tile_width = parse_u32(require_value("--tile-width"), "tile width");
    } else if (argument == "--tile-height") {
      options.tile_height = parse_u32(require_value("--tile-height"), "tile height");
    } else if (argument == "--padding") {
      options.artifact_padding = parse_u32(require_value("--padding"), "padding");
    } else if (argument == "--help") {
      std::cout
          << "Usage: bfnew_build_fpga_workload --data-root PATH --output-root PATH "
             "[--medium FILE] [--tile-width N] [--tile-height N] [--padding N]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument{"unknown option: " + argument};
    }
  }
  if (options.data_root.empty() || options.output_root.empty()) {
    throw std::invalid_argument{"--data-root and --output-root are required"};
  }
  if (options.tile_width == 0U || options.tile_height == 0U) {
    throw std::invalid_argument{"tile dimensions must be positive"};
  }
  return options;
}

[[nodiscard]] Distribution distribution(std::vector<double> values) {
  Distribution result;
  result.count = values.size();
  if (values.empty()) {
    return result;
  }
  std::sort(values.begin(), values.end());
  const auto percentile = [&values](const double fraction) {
    const double scaled = fraction * static_cast<double>(values.size() - 1U);
    return values[static_cast<std::size_t>(std::floor(scaled))];
  };
  result.minimum = values.front();
  result.median = percentile(0.50);
  result.p90 = percentile(0.90);
  result.p99 = percentile(0.99);
  result.maximum = values.back();
  result.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                static_cast<double>(values.size());
  return result;
}

void write_distribution(
    std::ostream& output,
    const std::string& label,
    const Distribution& values) {
  output << label << ": count=" << values.count << " min=" << values.minimum
         << " p50=" << values.median << " p90=" << values.p90
         << " p99=" << values.p99 << " max=" << values.maximum
         << " mean=" << values.mean << '\n';
}

[[nodiscard]] bool equal_graph_inputs(
    const bfnew::InputGraph& left,
    const bfnew::InputGraph& right) {
  return left.vertices().size() == right.vertices().size() &&
         left.edges().size() == right.edges().size() &&
         std::equal(
             left.vertices().begin(),
             left.vertices().end(),
             right.vertices().begin()) &&
         std::equal(
             left.edges().begin(),
             left.edges().end(),
             right.edges().begin());
}

[[nodiscard]] std::vector<std::filesystem::path> physical_inputs(
    const std::filesystem::path& root) {
  std::vector<std::filesystem::path> paths;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (entry.is_regular_file() && entry.path().extension() == ".phys") {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  if (paths.size() != 13U) {
    throw std::runtime_error{"Phase 7 requires exactly 13 .phys benchmark inputs"};
  }
  return paths;
}

[[nodiscard]] std::map<std::filesystem::path, InputFingerprint> fingerprint_inputs(
    const std::filesystem::path& device,
    const std::vector<std::filesystem::path>& physical) {
  std::map<std::filesystem::path, InputFingerprint> fingerprints;
  const auto add = [&fingerprints](const std::filesystem::path& path) {
    fingerprints.emplace(
        path,
        InputFingerprint{
            std::filesystem::file_size(path), std::filesystem::last_write_time(path)});
  };
  add(device);
  for (const auto& path : physical) {
    add(path);
  }
  return fingerprints;
}

void require_unchanged_inputs(
    const std::map<std::filesystem::path, InputFingerprint>& before) {
  for (const auto& [path, fingerprint] : before) {
    if (std::filesystem::file_size(path) != fingerprint.size ||
        std::filesystem::last_write_time(path) != fingerprint.modified) {
      throw std::runtime_error{"read-only input changed during bridge execution: " +
                               path.string()};
    }
  }
}

using TileEdgeCounts =
    std::vector<std::vector<std::pair<bfnew::TileId, std::uint64_t>>>;

[[nodiscard]] TileEdgeCounts build_tile_edge_counts(const bfnew::WeightedGraph& graph) {
  std::vector<std::map<bfnew::TileId, std::uint64_t>> counts(
      graph.tile_coordinates().size());
  const bfnew::OutgoingCsrView outgoing = graph.outgoing();
  for (std::size_t source = 0U; source < graph.vertex_count(); ++source) {
    const bfnew::TileId source_tile = graph.owner_tiles()[source];
    const auto begin = static_cast<std::size_t>(outgoing.row_offsets[source]);
    const auto end = static_cast<std::size_t>(outgoing.row_offsets[source + 1U]);
    for (std::size_t position = begin; position < end; ++position) {
      const bfnew::TileId destination_tile =
          graph.owner_tiles()[outgoing.destinations[position].value()];
      ++counts[source_tile.value()][destination_tile];
    }
  }

  TileEdgeCounts compact(counts.size());
  for (std::size_t tile = 0U; tile < counts.size(); ++tile) {
    compact[tile].assign(counts[tile].begin(), counts[tile].end());
  }
  return compact;
}

[[nodiscard]] std::uint64_t selected_edge_estimate(
    const TileEdgeCounts& counts,
    const std::vector<bfnew::TileId>& selected_tiles) {
  std::uint64_t edges = 0U;
  for (const bfnew::TileId source_tile : selected_tiles) {
    for (const auto& [destination_tile, count] : counts[source_tile.value()]) {
      if (std::binary_search(
              selected_tiles.begin(), selected_tiles.end(), destination_tile)) {
        edges += count;
      }
    }
  }
  return edges;
}

[[nodiscard]] std::size_t intersection_size(
    const std::vector<bfnew::TileId>& left,
    const std::vector<bfnew::TileId>& right) {
  std::size_t count = 0U;
  auto left_position = left.begin();
  auto right_position = right.begin();
  while (left_position != left.end() && right_position != right.end()) {
    if (*left_position < *right_position) {
      ++left_position;
    } else if (*right_position < *left_position) {
      ++right_position;
    } else {
      ++count;
      ++left_position;
      ++right_position;
    }
  }
  return count;
}

[[nodiscard]] std::pair<Distribution, Distribution> overlap_distributions(
    const std::vector<bfnew::MappedRouteQuery>& queries) {
  constexpr std::size_t maximum_pairs = 100000U;
  constexpr std::array<std::size_t, 3U> strides{1U, 17U, 257U};
  std::vector<double> overlap;
  std::vector<double> jaccard;
  overlap.reserve(std::min(maximum_pairs, queries.size() * strides.size()));
  jaccard.reserve(overlap.capacity());
  for (const std::size_t stride : strides) {
    for (std::size_t left = 0U;
         left + stride < queries.size() && overlap.size() < maximum_pairs;
         ++left) {
      const auto& left_tiles = queries[left].query.selected_tiles;
      const auto& right_tiles = queries[left + stride].query.selected_tiles;
      const std::size_t intersection = intersection_size(left_tiles, right_tiles);
      const std::size_t union_size =
          left_tiles.size() + right_tiles.size() - intersection;
      const std::size_t smaller = std::min(left_tiles.size(), right_tiles.size());
      overlap.push_back(
          smaller == 0U ? 0.0
                        : static_cast<double>(intersection) /
                              static_cast<double>(smaller));
      jaccard.push_back(
          union_size == 0U ? 0.0
                           : static_cast<double>(intersection) /
                                 static_cast<double>(union_size));
    }
  }
  return {distribution(std::move(overlap)), distribution(std::move(jaccard))};
}

[[nodiscard]] std::vector<double> selected_tile_counts(
    const std::vector<bfnew::MappedRouteQuery>& queries) {
  std::vector<double> values;
  values.reserve(queries.size());
  for (const auto& query : queries) {
    values.push_back(static_cast<double>(query.query.selected_tiles.size()));
  }
  return values;
}

[[nodiscard]] std::vector<double> fanouts(const bfnew::PhysicalCorpus& corpus) {
  std::vector<double> values;
  values.reserve(corpus.queries.size());
  for (const auto& query : corpus.queries) {
    values.push_back(static_cast<double>(query.targets.size()));
  }
  return values;
}

struct DijkstraEvidence {
  std::uint64_t sample_count{};
  std::uint64_t reached_targets{};
  std::uint64_t unreachable_targets{};
  std::uint64_t reconstructed_paths{};
};

[[nodiscard]] DijkstraEvidence run_bounded_dijkstra_samples(
    const bfnew::WeightedGraph& graph,
    const std::vector<bfnew::MappedRouteQuery>& queries) {
  struct Candidate {
    std::uint64_t vertices;
    std::size_t query;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(queries.size());
  for (std::size_t index = 0U; index < queries.size(); ++index) {
    const std::uint64_t vertices = bfnew::estimate_selected_vertex_count(
        graph, queries[index].query.selected_tiles);
    if (vertices != 0U && vertices <= 250000U) {
      candidates.push_back(Candidate{vertices, index});
    }
  }
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const Candidate& left, const Candidate& right) {
        return std::tie(left.vertices, left.query) <
               std::tie(right.vertices, right.query);
      });
  if (candidates.empty()) {
    throw std::runtime_error{"no bounded real query was available for Dijkstra"};
  }

  DijkstraEvidence evidence;
  const std::size_t sample_count = std::min<std::size_t>(5U, candidates.size());
  for (std::size_t sample = 0U; sample < sample_count; ++sample) {
    const bfnew::MappedRouteQuery& mapped = queries[candidates[sample].query];
    const bfnew::InducedQueryGraph bounded =
        bfnew::build_induced_query_graph(graph, mapped.query);
    const bfnew::SsspResult result =
        bfnew::dijkstra_oracle(bounded.graph, bounded.sources);
    for (const bfnew::VertexId source : bounded.sources) {
      if (result.distances[source.value()] != 0.0F) {
        throw std::runtime_error{"bounded Dijkstra did not preserve a source label"};
      }
    }
    for (const bfnew::VertexId target : bounded.targets) {
      if (std::isfinite(result.distances[target.value()])) {
        ++evidence.reached_targets;
        const auto path =
            bfnew::reconstruct_path_from_distances(bounded.graph, result, target);
        if (!path ||
            !bfnew::validate_reconstructed_path(bounded.graph, result, target, *path)) {
          throw std::runtime_error{"bounded Dijkstra path reconstruction failed"};
        }
        ++evidence.reconstructed_paths;
      } else {
        ++evidence.unreachable_targets;
      }
    }
    ++evidence.sample_count;
  }
  return evidence;
}

void write_scan_report(
    const std::filesystem::path& path,
    const std::vector<ScanRow>& rows) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error{"cannot create scan metadata report"};
  }
  output
      << "file\ttotal\taccepted\tone_source\ttwo_source\tsinks\tglobal_usednet"
         "\tstatic\tdriverless\tno_targets\tpartial_shape\tsource_count"
         "\tmissing_site_mapping\n";
  for (const ScanRow& row : rows) {
    const auto& stats = row.statistics;
    output << row.file << '\t' << stats.total_nets << '\t' << stats.accepted_queries
           << '\t' << stats.one_source_queries << '\t' << stats.two_source_queries
           << '\t' << stats.retained_sink_terminals << '\t'
           << stats.excluded_global_usednet << '\t' << stats.excluded_static << '\t'
           << stats.excluded_driverless << '\t' << stats.excluded_no_targets << '\t'
           << stats.excluded_partial_shape << '\t' << stats.excluded_source_count
           << '\t' << stats.excluded_missing_site_mapping << '\n';
  }
}

void write_main_report(
    const std::filesystem::path& path,
    const Options& options,
    const bfnew::DeviceImportStatistics& device,
    const bfnew::WeightedGraph& graph,
    const bfnew::PhysicalCorpus& logic,
    const bfnew::PhysicalCorpus& medium,
    const std::map<std::uint32_t, Distribution>& padding_tiles,
    const Distribution& fanout,
    const Distribution& vertex_estimates,
    const Distribution& edge_estimates,
    const Distribution& overlap,
    const Distribution& jaccard,
    const std::uint64_t missing_terminals,
    const std::uint64_t terminal_count,
    const std::uint64_t spill_queries,
    const DijkstraEvidence& dijkstra,
    const std::vector<bfnew::MappedRouteQuery>& queries) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error{"cannot create Phase 7 report"};
  }
  output << std::fixed << std::setprecision(6);
  output << "bfnew Phase 7 FPGA workload report\n"
         << "bridge_version=" << bfnew::fpga_interchange_bridge_version << '\n'
         << "artifact_version=" << bfnew::workload_artifact_version << '\n'
         << "device=xcvu3p.device\n"
         << "tile_shape=" << options.tile_width << 'x' << options.tile_height << '\n'
         << "artifact_padding=" << options.artifact_padding << '\n'
         << "device_tiles=" << device.tile_count << '\n'
         << "device_wires=" << device.wire_count << '\n'
         << "graph_vertices=" << graph.vertex_count() << '\n'
         << "graph_edges=" << graph.edge_count() << '\n'
         << "physical_pips=" << device.pip_count << '\n'
         << "bidirectional_pips=" << device.bidirectional_pip_count << '\n'
         << "directed_pip_edges=" << device.directed_pip_count << '\n'
         << "skipped_unmapped_pips=" << device.skipped_unmapped_pips << '\n'
         << "fallback_weight_edges=" << device.fallback_weight_edges << '\n'
         << "missing_coordinate_nodes=" << device.missing_coordinate_nodes << '\n'
         << "site_pin_mappings=" << device.site_pin_mappings << '\n'
         << "skipped_unmapped_site_pins=" << device.skipped_unmapped_site_pins
         << '\n'
         << "tile_directory_tiles=" << graph.tile_coordinates().size() << '\n'
         << "graph_validator=pass\n"
         << "tile_directory_validator=pass\n"
         << "graph_artifact_deterministic=pass\n"
         << "logic_query_artifact_deterministic=pass\n"
         << "medium_query_artifact_deterministic=pass\n"
         << "logic_benchmark=logicnets_jscl_unrouted.phys\n"
         << "logic_queries=" << logic.statistics.accepted_queries << '\n'
         << "logic_one_source=" << logic.statistics.one_source_queries << '\n'
         << "logic_two_source=" << logic.statistics.two_source_queries << '\n'
         << "logic_sink_terminals=" << logic.statistics.retained_sink_terminals << '\n'
         << "medium_benchmark=" << options.medium_name << '\n'
         << "medium_queries=" << medium.statistics.accepted_queries << '\n';
  write_distribution(output, "fanout", fanout);
  for (const auto& [padding, values] : padding_tiles) {
    write_distribution(
        output, "initial_bounding_tiles_padding_" + std::to_string(padding), values);
  }
  write_distribution(output, "selected_vertices", vertex_estimates);
  write_distribution(output, "selected_edges", edge_estimates);
  write_distribution(output, "sampled_box_overlap_coefficient", overlap);
  write_distribution(output, "sampled_tile_set_jaccard", jaccard);
  output << "overlap_sampling=canonical query pairs at strides 1,17,257 capped at "
            "100000 pairs\n"
         << "terminal_count=" << terminal_count << '\n'
         << "missing_coordinate_terminals=" << missing_terminals << '\n'
         << "missing_coordinate_rate="
         << (terminal_count == 0U
                 ? 0.0
                 : static_cast<double>(missing_terminals) /
                       static_cast<double>(terminal_count))
         << '\n'
         << "potential_spill_tile_admissions=" << spill_queries << '\n';
  for (const std::uint32_t width : {8U, 16U, 32U}) {
    const std::uint64_t batches =
        (queries.size() + static_cast<std::size_t>(width) - 1U) / width;
    const double occupancy = batches == 0U
                                 ? 0.0
                                 : static_cast<double>(queries.size()) /
                                       static_cast<double>(batches * width);
    output << "predicted_lane_occupancy_width_" << width << '=' << occupancy
           << " batches=" << batches << '\n';
  }
  output << "bounded_dijkstra_samples=" << dijkstra.sample_count << '\n'
         << "bounded_dijkstra_reached_targets=" << dijkstra.reached_targets << '\n'
         << "bounded_dijkstra_unreachable_targets=" << dijkstra.unreachable_targets
         << '\n'
         << "bounded_dijkstra_valid_reconstructed_paths="
         << dijkstra.reconstructed_paths << '\n'
         << "input_files_unchanged=pass\n"
         << "gpu_tests=deferred_by_maintainer_policy\n"
         << "endpoint_samples:\n";
  const std::size_t samples = std::min<std::size_t>(5U, queries.size());
  for (std::size_t sample = 0U; sample < samples; ++sample) {
    const auto& query = queries[sample];
    output << "  net=" << query.net_name << '\n';
    for (const auto& source : query.source_terminals) {
      output << "    source=" << source.site << '/' << source.pin
             << " original_node=" << source.original_vertex.value()
             << " vertex=" << source.vertex.value() << " xy=" << source.x << ','
             << source.y << " tile=" << source.owner_tile.value() << '\n';
    }
    const std::size_t targets = std::min<std::size_t>(3U, query.target_terminals.size());
    for (std::size_t target = 0U; target < targets; ++target) {
      const auto& terminal = query.target_terminals[target];
      output << "    target=" << terminal.site << '/' << terminal.pin
             << " original_node=" << terminal.original_vertex.value()
             << " vertex=" << terminal.vertex.value() << " xy=" << terminal.x << ','
             << terminal.y << " tile=" << terminal.owner_tile.value() << '\n';
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::filesystem::create_directories(options.output_root);
    const std::filesystem::path device_path = options.data_root / "xcvu3p.device";
    const std::vector<std::filesystem::path> phys_paths =
        physical_inputs(options.data_root);
    const auto input_fingerprints = fingerprint_inputs(device_path, phys_paths);

    std::cout << "Loading xcvu3p routing resources...\n";
    bfnew::DeviceImportResult imported =
        bfnew::load_fpga_interchange_device(device_path);
    const bfnew::DeviceImportStatistics device_statistics = imported.statistics;

    const auto graph_path = options.output_root / "xcvu3p.v1.bfgraph";
    const auto graph_repeat_path = options.output_root / "xcvu3p.v1.repeat.bfgraph";
    bfnew::write_graph_artifact(graph_path, *imported.graph);
    bfnew::write_graph_artifact(graph_repeat_path, *imported.graph);
    if (!bfnew::files_are_byte_identical(graph_path, graph_repeat_path)) {
      throw std::runtime_error{"repeated graph artifacts differ"};
    }
    bfnew::InputGraph graph_round_trip = bfnew::read_graph_artifact(graph_path);
    if (!equal_graph_inputs(*imported.graph, graph_round_trip)) {
      throw std::runtime_error{"graph artifact round trip differs from import"};
    }
    imported.graph.reset();
    std::filesystem::remove(graph_repeat_path);

    std::cout << "Building and deeply validating CSR, CSC, and tile metadata...\n";
    const bfnew::UniformGridPartitioner partitioner(bfnew::SpatialOrderConfig{
        0, 0, options.tile_width, options.tile_height});
    const bfnew::PartitionedGraph partitioned = partitioner.partition(graph_round_trip);
    if (!bfnew::validate_weighted_graph(partitioned.graph).ok()) {
      throw std::runtime_error{"deep weighted-graph validation failed"};
    }
    if (!bfnew::validate_tile_directory(partitioned.graph, partitioned.tiles).ok()) {
      throw std::runtime_error{"deep tile-directory validation failed"};
    }

    std::cout << "Scanning all 13 physical-netlist inputs...\n";
    std::vector<ScanRow> scan_rows;
    std::optional<bfnew::PhysicalCorpus> logic_corpus;
    std::optional<bfnew::PhysicalCorpus> medium_corpus;
    for (const auto& phys_path : phys_paths) {
      const std::string filename = phys_path.filename().string();
      const bool retain = filename == "logicnets_jscl_unrouted.phys" ||
                          filename == options.medium_name;
      bfnew::PhysicalCorpus corpus = bfnew::scan_fpga_interchange_physical_netlist(
          phys_path, imported.site_pin_mappings, retain);
      scan_rows.push_back(ScanRow{filename, corpus.statistics});
      if (filename == "logicnets_jscl_unrouted.phys") {
        logic_corpus = std::move(corpus);
      } else if (filename == options.medium_name) {
        medium_corpus = std::move(corpus);
      }
    }
    if (!logic_corpus || !medium_corpus) {
      throw std::runtime_error{"logic or requested medium benchmark was not found"};
    }
    if (logic_corpus->queries.size() != logic_corpus->statistics.accepted_queries ||
        medium_corpus->queries.size() != medium_corpus->statistics.accepted_queries) {
      throw std::runtime_error{"retained query corpus count mismatch"};
    }
    write_scan_report(options.output_root / "all_inputs.v1.tsv", scan_rows);

    std::cout << "Mapping complete logicnets_jscl query corpus...\n";
    std::vector<bfnew::MappedRouteQuery> logic_queries = bfnew::map_route_queries(
        *logic_corpus, partitioned.graph, options.artifact_padding);
    for (const auto& mapped : logic_queries) {
      if (!bfnew::validate_route_query(partitioned.graph, mapped.query).ok()) {
        throw std::runtime_error{"mapped logic query failed deep validation"};
      }
    }
    const auto logic_path = options.output_root /
                            ("logicnets_jscl.padding" +
                             std::to_string(options.artifact_padding) +
                             ".v1.bfqueries");
    const auto logic_repeat_path = options.output_root / "logicnets_jscl.v1.repeat.bfqueries";
    bfnew::write_query_artifact(logic_path, logic_queries);
    bfnew::write_query_artifact(logic_repeat_path, logic_queries);
    if (!bfnew::files_are_byte_identical(logic_path, logic_repeat_path)) {
      throw std::runtime_error{"repeated logic query artifacts differ"};
    }
    if (bfnew::read_query_artifact(logic_path, partitioned.graph) != logic_queries) {
      throw std::runtime_error{"logic query artifact round trip differs"};
    }
    std::filesystem::remove(logic_repeat_path);

    std::cout << "Mapping and validating medium benchmark " << options.medium_name
              << "...\n";
    std::vector<bfnew::MappedRouteQuery> medium_queries = bfnew::map_route_queries(
        *medium_corpus, partitioned.graph, options.artifact_padding);
    const std::string medium_stem =
        std::filesystem::path(options.medium_name).stem().string();
    const auto medium_path = options.output_root /
                             (medium_stem + ".padding" +
                              std::to_string(options.artifact_padding) +
                              ".v1.bfqueries");
    const auto medium_repeat_path = options.output_root / (medium_stem + ".v1.repeat");
    bfnew::write_query_artifact(medium_path, medium_queries);
    bfnew::write_query_artifact(medium_repeat_path, medium_queries);
    if (!bfnew::files_are_byte_identical(medium_path, medium_repeat_path) ||
        bfnew::read_query_artifact(medium_path, partitioned.graph) != medium_queries) {
      throw std::runtime_error{"medium query artifact determinism or round trip failed"};
    }
    std::filesystem::remove(medium_repeat_path);

    std::cout << "Computing Phase 7 workload statistics and bounded CPU samples...\n";
    std::map<std::uint32_t, Distribution> padding_tiles;
    for (const std::uint32_t padding : {0U, 1U, 2U, 4U}) {
      const auto mapped =
          bfnew::map_route_queries(*logic_corpus, partitioned.graph, padding);
      padding_tiles.emplace(padding, distribution(selected_tile_counts(mapped)));
    }
    const TileEdgeCounts tile_edges = build_tile_edge_counts(partitioned.graph);
    std::vector<double> vertex_estimates;
    std::vector<double> edge_estimates;
    vertex_estimates.reserve(logic_queries.size());
    edge_estimates.reserve(logic_queries.size());
    std::uint64_t missing_terminals = 0U;
    std::uint64_t terminal_count = 0U;
    std::uint64_t spill_queries = 0U;
    for (const auto& mapped : logic_queries) {
      vertex_estimates.push_back(static_cast<double>(
          bfnew::estimate_selected_vertex_count(
              partitioned.graph, mapped.query.selected_tiles)));
      edge_estimates.push_back(static_cast<double>(
          selected_edge_estimate(tile_edges, mapped.query.selected_tiles)));
      const bool has_spill = std::binary_search(
          mapped.query.selected_tiles.begin(),
          mapped.query.selected_tiles.end(),
          partitioned.tiles.spill_tile());
      spill_queries += has_spill ? 1U : 0U;
      for (const auto& terminal : mapped.source_terminals) {
        ++terminal_count;
        missing_terminals += terminal.has_location ? 0U : 1U;
      }
      for (const auto& terminal : mapped.target_terminals) {
        ++terminal_count;
        missing_terminals += terminal.has_location ? 0U : 1U;
      }
    }
    const auto [overlap, jaccard] = overlap_distributions(logic_queries);
    const DijkstraEvidence dijkstra =
        run_bounded_dijkstra_samples(partitioned.graph, logic_queries);

    require_unchanged_inputs(input_fingerprints);
    write_main_report(
        options.output_root / "phase7_report.v1.txt",
        options,
        device_statistics,
        partitioned.graph,
        *logic_corpus,
        *medium_corpus,
        padding_tiles,
        distribution(fanouts(*logic_corpus)),
        distribution(std::move(vertex_estimates)),
        distribution(std::move(edge_estimates)),
        overlap,
        jaccard,
        missing_terminals,
        terminal_count,
        spill_queries,
        dijkstra,
        logic_queries);

    std::cout << "Phase 7 CPU workload bridge completed. Artifacts: "
              << options.output_root << '\n';
    return 0;
  } catch (const kj::Exception& error) {
    std::cerr << "error: " << error.getDescription().cStr() << std::endl;
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "error: unknown non-standard exception" << std::endl;
    return 1;
  }
}
