#pragma once

#include "bfnew/graph.hpp"
#include "bfnew/query.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace bfnew {

inline constexpr std::uint32_t fpga_interchange_bridge_version = 1U;
inline constexpr std::uint32_t workload_artifact_version = 1U;

struct SitePinMapping {
  std::string site;
  std::string site_type;
  std::string pin;
  VertexId original_vertex{};

  constexpr bool operator==(const SitePinMapping&) const noexcept = default;
};

struct DeviceImportStatistics {
  std::uint64_t tile_count{};
  std::uint64_t wire_count{};
  std::uint64_t node_count{};
  std::uint64_t pip_count{};
  std::uint64_t directed_pip_count{};
  std::uint64_t bidirectional_pip_count{};
  std::uint64_t skipped_unmapped_pips{};
  std::uint64_t missing_coordinate_nodes{};
  std::uint64_t fallback_weight_edges{};
  std::uint64_t site_pin_mappings{};
  std::uint64_t skipped_unmapped_site_pins{};
};

struct DeviceImportResult {
  std::unique_ptr<InputGraph> graph;
  std::vector<SitePinMapping> site_pin_mappings;
  DeviceImportStatistics statistics;
};

struct PhysicalEndpoint {
  std::string site;
  std::string pin;
  VertexId original_vertex{};

  constexpr bool operator==(const PhysicalEndpoint&) const noexcept = default;
};

struct PhysicalQueryRecord {
  std::string net_name;
  std::vector<PhysicalEndpoint> sources;
  std::vector<PhysicalEndpoint> targets;

  constexpr bool operator==(const PhysicalQueryRecord&) const noexcept = default;
};

struct PhysicalScanStatistics {
  std::uint64_t total_nets{};
  std::uint64_t accepted_queries{};
  std::uint64_t one_source_queries{};
  std::uint64_t two_source_queries{};
  std::uint64_t excluded_global_usednet{};
  std::uint64_t excluded_static{};
  std::uint64_t excluded_driverless{};
  std::uint64_t excluded_no_targets{};
  std::uint64_t excluded_partial_shape{};
  std::uint64_t excluded_source_count{};
  std::uint64_t excluded_missing_site_mapping{};
  std::uint64_t retained_sink_terminals{};
};

struct PhysicalCorpus {
  std::string part;
  std::vector<PhysicalQueryRecord> queries;
  PhysicalScanStatistics statistics;
};

struct MappedTerminal {
  std::string site;
  std::string pin;
  VertexId original_vertex{};
  VertexId vertex{};
  std::int32_t x{};
  std::int32_t y{};
  bool has_location{};
  TileId owner_tile{};

  constexpr bool operator==(const MappedTerminal&) const noexcept = default;
};

struct MappedRouteQuery {
  std::string net_name;
  RouteQuery query;
  std::vector<MappedTerminal> source_terminals;
  std::vector<MappedTerminal> target_terminals;

  constexpr bool operator==(const MappedRouteQuery&) const noexcept = default;
};

[[nodiscard]] DeviceImportResult load_fpga_interchange_device(
    const std::filesystem::path& gzip_device_path);

[[nodiscard]] PhysicalCorpus scan_fpga_interchange_physical_netlist(
    const std::filesystem::path& gzip_phys_path,
    const std::vector<SitePinMapping>& site_pin_mappings,
    bool retain_queries = true);

[[nodiscard]] std::vector<MappedRouteQuery> map_route_queries(
    const PhysicalCorpus& corpus,
    const WeightedGraph& spatial_graph,
    std::uint32_t tile_padding);

void write_graph_artifact(
    const std::filesystem::path& path,
    const InputGraph& graph);

[[nodiscard]] InputGraph read_graph_artifact(const std::filesystem::path& path);

void write_query_artifact(
    const std::filesystem::path& path,
    const std::vector<MappedRouteQuery>& queries);

[[nodiscard]] std::vector<MappedRouteQuery> read_query_artifact(
    const std::filesystem::path& path,
    const WeightedGraph& spatial_graph);

[[nodiscard]] bool files_are_byte_identical(
    const std::filesystem::path& left,
    const std::filesystem::path& right);

}  // namespace bfnew
