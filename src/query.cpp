#include "bfnew/query.hpp"

#include "bfnew/spatial.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

constexpr std::uint32_t unmapped_vertex = std::numeric_limits<std::uint32_t>::max();

template <typename Id>
[[nodiscard]] bool canonical_ids(const std::span<const Id> ids) noexcept {
  return std::adjacent_find(
             ids.begin(), ids.end(), [](const Id left, const Id right) {
               return !(left < right);
             }) == ids.end();
}

[[nodiscard]] std::pair<std::vector<VertexId>, std::vector<std::uint32_t>>
canonicalize_terminals(const std::span<const VertexId> terminals) {
  std::vector<VertexId> canonical(terminals.begin(), terminals.end());
  std::sort(canonical.begin(), canonical.end());
  canonical.erase(std::unique(canonical.begin(), canonical.end()), canonical.end());

  std::vector<std::uint32_t> mapping;
  mapping.reserve(terminals.size());
  for (const VertexId terminal : terminals) {
    const auto position = std::lower_bound(canonical.begin(), canonical.end(), terminal);
    mapping.push_back(checked_id<VertexId>(position - canonical.begin()).value());
  }
  return {std::move(canonical), std::move(mapping)};
}

[[nodiscard]] std::int64_t saturating_subtract(
    const std::int64_t value,
    const std::uint32_t amount) noexcept {
  const auto wide_amount = static_cast<std::uint64_t>(amount);
  if (value < 0 && wide_amount >
          static_cast<std::uint64_t>(value - std::numeric_limits<std::int64_t>::min())) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return value - static_cast<std::int64_t>(amount);
}

[[nodiscard]] std::int64_t saturating_add(
    const std::int64_t value,
    const std::uint32_t amount) noexcept {
  const auto wide_amount = static_cast<std::uint64_t>(amount);
  if (value >= 0 && wide_amount >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() - value)) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return value + static_cast<std::int64_t>(amount);
}

[[nodiscard]] std::vector<TileId> select_tiles(
    const WeightedGraph& graph,
    const std::span<const VertexId> source_terminals,
    const std::span<const VertexId> target_terminals,
    const std::uint32_t padding) {
  bool has_located_terminal = false;
  bool needs_spill = false;
  std::int64_t minimum_x = std::numeric_limits<std::int64_t>::max();
  std::int64_t minimum_y = std::numeric_limits<std::int64_t>::max();
  std::int64_t maximum_x = std::numeric_limits<std::int64_t>::min();
  std::int64_t maximum_y = std::numeric_limits<std::int64_t>::min();

  const auto visit_terminal = [&](const VertexId terminal) {
    if (!is_valid_vertex_id(terminal, graph.vertex_count())) {
      throw std::out_of_range{"query terminal is outside the graph"};
    }
    const TileId owner = graph.owner_tiles()[terminal.value()];
    const TileCoordinate coordinate = graph.tile_coordinates()[owner.value()];
    if (!coordinate.has_location) {
      needs_spill = true;
      return;
    }
    has_located_terminal = true;
    minimum_x = std::min(minimum_x, coordinate.tile_x);
    minimum_y = std::min(minimum_y, coordinate.tile_y);
    maximum_x = std::max(maximum_x, coordinate.tile_x);
    maximum_y = std::max(maximum_y, coordinate.tile_y);
  };
  for (const VertexId terminal : source_terminals) {
    visit_terminal(terminal);
  }
  for (const VertexId terminal : target_terminals) {
    visit_terminal(terminal);
  }

  std::vector<TileId> selected;
  if (has_located_terminal) {
    const std::int64_t lower_x = saturating_subtract(minimum_x, padding);
    const std::int64_t lower_y = saturating_subtract(minimum_y, padding);
    const std::int64_t upper_x = saturating_add(maximum_x, padding);
    const std::int64_t upper_y = saturating_add(maximum_y, padding);
    for (std::size_t tile = 0U; tile < graph.tile_coordinates().size(); ++tile) {
      const TileCoordinate coordinate = graph.tile_coordinates()[tile];
      if (coordinate.has_location && coordinate.tile_x >= lower_x &&
          coordinate.tile_x <= upper_x && coordinate.tile_y >= lower_y &&
          coordinate.tile_y <= upper_y) {
        selected.push_back(checked_id<TileId>(tile));
      }
    }
  }
  if (needs_spill) {
    selected.push_back(checked_id<TileId>(graph.tile_coordinates().size() - 1U));
  }
  return selected;
}

[[nodiscard]] RouteQueryValidationResult query_error(
    const RouteQueryValidationErrorCode code,
    const std::size_t position = RouteQueryValidationResult::no_position) noexcept {
  return RouteQueryValidationResult{code, position};
}

}  // namespace

RouteQuery make_route_query(
    const QueryId query_id,
    const WeightedGraph& graph,
    const std::span<const VertexId> source_terminals,
    const std::span<const VertexId> target_terminals,
    const std::uint32_t tile_padding,
    const std::uint32_t expansion_generation) {
  if (!graph.has_spatial_ordering()) {
    throw std::invalid_argument{"route queries require a spatially ordered graph"};
  }
  if (source_terminals.empty() || target_terminals.empty()) {
    throw std::invalid_argument{"route queries require sources and targets"};
  }

  RouteQuery query;
  query.query_id = query_id;
  query.source_terminals.assign(source_terminals.begin(), source_terminals.end());
  query.target_terminals.assign(target_terminals.begin(), target_terminals.end());
  std::tie(query.sources, query.source_terminal_to_source) =
      canonicalize_terminals(source_terminals);
  std::tie(query.targets, query.target_terminal_to_target) =
      canonicalize_terminals(target_terminals);
  query.selected_tiles = select_tiles(
      graph, query.source_terminals, query.target_terminals, tile_padding);
  query.expansion_generation = expansion_generation;

  const RouteQueryValidationResult validation = validate_route_query(graph, query);
  if (!validation.ok()) {
    throw std::logic_error{"constructed route query failed deep validation"};
  }
  return query;
}

RouteQueryValidationResult validate_route_query(
    const WeightedGraph& graph,
    const RouteQuery& query) noexcept {
  if (!graph.has_spatial_ordering()) {
    return query_error(RouteQueryValidationErrorCode::graph_is_not_spatially_ordered);
  }
  if (query.sources.empty() || query.source_terminals.empty()) {
    return query_error(RouteQueryValidationErrorCode::empty_sources);
  }
  if (query.targets.empty() || query.target_terminals.empty()) {
    return query_error(RouteQueryValidationErrorCode::empty_targets);
  }
  for (std::size_t index = 0U; index < query.sources.size(); ++index) {
    if (!is_valid_vertex_id(query.sources[index], graph.vertex_count())) {
      return query_error(RouteQueryValidationErrorCode::source_out_of_range, index);
    }
  }
  for (std::size_t index = 0U; index < query.targets.size(); ++index) {
    if (!is_valid_vertex_id(query.targets[index], graph.vertex_count())) {
      return query_error(RouteQueryValidationErrorCode::target_out_of_range, index);
    }
  }
  if (!canonical_ids<VertexId>(query.sources)) {
    return query_error(RouteQueryValidationErrorCode::noncanonical_sources);
  }
  if (!canonical_ids<VertexId>(query.targets)) {
    return query_error(RouteQueryValidationErrorCode::noncanonical_targets);
  }
  if (query.source_terminal_to_source.size() != query.source_terminals.size()) {
    return query_error(RouteQueryValidationErrorCode::source_map_size_mismatch);
  }
  if (query.target_terminal_to_target.size() != query.target_terminals.size()) {
    return query_error(RouteQueryValidationErrorCode::target_map_size_mismatch);
  }
  for (std::size_t index = 0U; index < query.source_terminals.size(); ++index) {
    const std::uint32_t mapped = query.source_terminal_to_source[index];
    if (mapped >= query.sources.size() ||
        query.sources[mapped] != query.source_terminals[index]) {
      return query_error(RouteQueryValidationErrorCode::source_map_mismatch, index);
    }
  }
  for (std::size_t index = 0U; index < query.target_terminals.size(); ++index) {
    const std::uint32_t mapped = query.target_terminal_to_target[index];
    if (mapped >= query.targets.size() ||
        query.targets[mapped] != query.target_terminals[index]) {
      return query_error(RouteQueryValidationErrorCode::target_map_mismatch, index);
    }
  }
  if (query.selected_tiles.empty()) {
    return query_error(RouteQueryValidationErrorCode::empty_selected_tiles);
  }
  if (!canonical_ids<TileId>(query.selected_tiles)) {
    return query_error(RouteQueryValidationErrorCode::noncanonical_selected_tiles);
  }
  for (std::size_t index = 0U; index < query.selected_tiles.size(); ++index) {
    if (query.selected_tiles[index].value() >= graph.tile_coordinates().size()) {
      return query_error(
          RouteQueryValidationErrorCode::selected_tile_out_of_range, index);
    }
  }
  const auto owns_selected_tile = [&](const VertexId terminal) {
    if (!is_valid_vertex_id(terminal, graph.vertex_count())) {
      return false;
    }
    const TileId owner = graph.owner_tiles()[terminal.value()];
    return std::binary_search(
        query.selected_tiles.begin(), query.selected_tiles.end(), owner);
  };
  for (std::size_t index = 0U; index < query.source_terminals.size(); ++index) {
    if (!owns_selected_tile(query.source_terminals[index])) {
      return query_error(
          RouteQueryValidationErrorCode::terminal_owner_tile_not_selected, index);
    }
  }
  for (std::size_t index = 0U; index < query.target_terminals.size(); ++index) {
    if (!owns_selected_tile(query.target_terminals[index])) {
      return query_error(
          RouteQueryValidationErrorCode::terminal_owner_tile_not_selected, index);
    }
  }
  return {};
}

std::uint64_t estimate_selected_vertex_count(
    const WeightedGraph& graph,
    const std::span<const TileId> selected_tiles) {
  std::uint64_t count = 0U;
  for (const TileId tile : selected_tiles) {
    if (tile.value() >= graph.tile_coordinates().size()) {
      throw std::out_of_range{"selected tile is outside the graph"};
    }
    const auto offsets = graph.tile_vertex_offsets();
    count += offsets[tile.value() + 1U] - offsets[tile.value()];
  }
  return count;
}

std::uint64_t estimate_selected_edge_count(
    const WeightedGraph& graph,
    const std::span<const TileId> selected_tiles) {
  std::vector<bool> selected(graph.tile_coordinates().size(), false);
  for (const TileId tile : selected_tiles) {
    if (tile.value() >= selected.size()) {
      throw std::out_of_range{"selected tile is outside the graph"};
    }
    selected[tile.value()] = true;
  }

  std::uint64_t count = 0U;
  const OutgoingCsrView outgoing = graph.outgoing();
  for (std::size_t source = 0U; source < graph.vertex_count(); ++source) {
    if (!selected[graph.owner_tiles()[source].value()]) {
      continue;
    }
    const auto begin = static_cast<std::size_t>(outgoing.row_offsets[source]);
    const auto end = static_cast<std::size_t>(outgoing.row_offsets[source + 1U]);
    for (std::size_t position = begin; position < end; ++position) {
      if (selected[graph.owner_tiles()[outgoing.destinations[position].value()].value()]) {
        ++count;
      }
    }
  }
  return count;
}

InducedQueryGraph build_induced_query_graph(
    const WeightedGraph& graph,
    const RouteQuery& query) {
  const RouteQueryValidationResult validation = validate_route_query(graph, query);
  if (!validation.ok()) {
    throw std::invalid_argument{"induced graph requires a deeply valid route query"};
  }

  std::vector<bool> selected(graph.tile_coordinates().size(), false);
  for (const TileId tile : query.selected_tiles) {
    selected[tile.value()] = true;
  }

  std::vector<VertexId> local_to_global;
  std::vector<VertexId> global_to_local(
      graph.vertex_count(), VertexId{unmapped_vertex});
  std::vector<VertexMetadata> vertices;
  for (std::size_t global = 0U; global < graph.vertex_count(); ++global) {
    if (!selected[graph.owner_tiles()[global].value()]) {
      continue;
    }
    const VertexId local = checked_id<VertexId>(local_to_global.size());
    global_to_local[global] = local;
    local_to_global.push_back(checked_id<VertexId>(global));
    vertices.push_back(graph.vertices()[global]);
  }

  std::vector<EdgeInputRecord> edges;
  const OutgoingCsrView outgoing = graph.outgoing();
  for (std::size_t global_source = 0U; global_source < graph.vertex_count();
       ++global_source) {
    const VertexId local_source = global_to_local[global_source];
    if (local_source.value() == unmapped_vertex) {
      continue;
    }
    const auto begin = static_cast<std::size_t>(outgoing.row_offsets[global_source]);
    const auto end = static_cast<std::size_t>(outgoing.row_offsets[global_source + 1U]);
    for (std::size_t position = begin; position < end; ++position) {
      const VertexId global_destination = outgoing.destinations[position];
      const VertexId local_destination = global_to_local[global_destination.value()];
      if (local_destination.value() == unmapped_vertex) {
        continue;
      }
      const EdgeId edge_id = outgoing.edge_ids[position];
      edges.push_back(EdgeInputRecord{
          local_source,
          local_destination,
          outgoing.weights[position],
          graph.edge_provenance()[edge_id.value()],
      });
    }
  }

  const auto remap = [&global_to_local](const std::vector<VertexId>& globals) {
    std::vector<VertexId> locals;
    locals.reserve(globals.size());
    for (const VertexId global : globals) {
      const VertexId local = global_to_local[global.value()];
      if (local.value() == unmapped_vertex) {
        throw std::logic_error{"query terminal was omitted from its induced graph"};
      }
      locals.push_back(local);
    }
    return locals;
  };

  std::vector<VertexId> local_sources = remap(query.sources);
  std::vector<VertexId> local_targets = remap(query.targets);
  InputGraph input(std::move(vertices), std::move(edges));
  return InducedQueryGraph{
      build_weighted_graph(input),
      std::move(local_to_global),
      std::move(global_to_local),
      std::move(local_sources),
      std::move(local_targets),
  };
}

}  // namespace bfnew
