#pragma once

#include "bfnew/graph.hpp"
#include "bfnew/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace bfnew {

struct RouteQuery {
  QueryId query_id{};

  // Terminals retain their input order. The canonical sets are sorted and
  // deduplicated for solver-facing use; the maps preserve terminal identity.
  std::vector<VertexId> source_terminals;
  std::vector<VertexId> target_terminals;
  std::vector<VertexId> sources;
  std::vector<VertexId> targets;
  std::vector<std::uint32_t> source_terminal_to_source;
  std::vector<std::uint32_t> target_terminal_to_target;

  std::vector<TileId> selected_tiles;
  std::uint32_t expansion_generation{};

  constexpr bool operator==(const RouteQuery&) const noexcept = default;
};

enum class RouteQueryValidationErrorCode : std::uint8_t {
  none,
  graph_is_not_spatially_ordered,
  empty_sources,
  empty_targets,
  source_out_of_range,
  target_out_of_range,
  noncanonical_sources,
  noncanonical_targets,
  source_map_size_mismatch,
  target_map_size_mismatch,
  source_map_mismatch,
  target_map_mismatch,
  empty_selected_tiles,
  selected_tile_out_of_range,
  noncanonical_selected_tiles,
  terminal_owner_tile_not_selected,
};

struct RouteQueryValidationResult {
  static constexpr std::size_t no_position = static_cast<std::size_t>(-1);

  RouteQueryValidationErrorCode code{RouteQueryValidationErrorCode::none};
  std::size_t position{no_position};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == RouteQueryValidationErrorCode::none;
  }
};

[[nodiscard]] RouteQuery make_route_query(
    QueryId query_id,
    const WeightedGraph& graph,
    std::span<const VertexId> source_terminals,
    std::span<const VertexId> target_terminals,
    std::uint32_t tile_padding = 0U,
    std::uint32_t expansion_generation = 0U);

[[nodiscard]] RouteQueryValidationResult validate_route_query(
    const WeightedGraph& graph,
    const RouteQuery& query) noexcept;

[[nodiscard]] std::uint64_t estimate_selected_vertex_count(
    const WeightedGraph& graph,
    std::span<const TileId> selected_tiles);

[[nodiscard]] std::uint64_t estimate_selected_edge_count(
    const WeightedGraph& graph,
    std::span<const TileId> selected_tiles);

struct InducedQueryGraph {
  WeightedGraph graph;
  std::vector<VertexId> local_to_global;
  std::vector<VertexId> global_to_local;
  std::vector<VertexId> sources;
  std::vector<VertexId> targets;
};

[[nodiscard]] InducedQueryGraph build_induced_query_graph(
    const WeightedGraph& graph,
    const RouteQuery& query);

}  // namespace bfnew
