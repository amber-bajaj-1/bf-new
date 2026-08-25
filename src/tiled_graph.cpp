#include "bfnew/spatial.hpp"

#include "bfnew/graph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

template <typename T>
struct FlattenedGroups {
  std::vector<EdgeOffset> offsets;
  std::vector<T> values;
};

template <typename T>
[[nodiscard]] FlattenedGroups<T> flatten_groups(
    const std::vector<std::vector<T>>& groups) {
  FlattenedGroups<T> flattened;
  flattened.offsets.reserve(groups.size() + 1U);
  flattened.offsets.push_back(0U);
  for (const std::vector<T>& group : groups) {
    if (!std::in_range<EdgeOffset>(group.size()) ||
        flattened.offsets.back() >
            std::numeric_limits<EdgeOffset>::max() -
                static_cast<EdgeOffset>(group.size())) {
      throw std::length_error{"tile metadata exceeds the 64-bit offset representation"};
    }
    flattened.values.insert(flattened.values.end(), group.begin(), group.end());
    flattened.offsets.push_back(
        flattened.offsets.back() + static_cast<EdgeOffset>(group.size()));
  }
  return flattened;
}

struct GraphEdgeIndex {
  std::vector<VertexId> sources;
  std::vector<VertexId> destinations;
  std::vector<std::size_t> csr_ranks;
  std::vector<std::size_t> csc_ranks;
};

[[nodiscard]] GraphEdgeIndex index_graph_edges(const WeightedGraph& graph) {
  const std::size_t edge_count = static_cast<std::size_t>(graph.edge_count());
  GraphEdgeIndex index{
      std::vector<VertexId>(edge_count),
      std::vector<VertexId>(edge_count),
      std::vector<std::size_t>(edge_count),
      std::vector<std::size_t>(edge_count),
  };

  const OutgoingCsrView outgoing = graph.outgoing();
  for (std::size_t source = 0U; source < graph.vertex_count(); ++source) {
    const std::size_t row_begin = static_cast<std::size_t>(outgoing.row_offsets[source]);
    const std::size_t row_end =
        static_cast<std::size_t>(outgoing.row_offsets[source + 1U]);
    for (std::size_t position = row_begin; position < row_end; ++position) {
      const std::size_t edge_id =
          static_cast<std::size_t>(outgoing.edge_ids[position].value());
      index.sources[edge_id] = VertexId{static_cast<std::uint32_t>(source)};
      index.destinations[edge_id] = outgoing.destinations[position];
      index.csr_ranks[edge_id] = position;
    }
  }

  const IncomingCscView incoming = graph.incoming();
  for (std::size_t position = 0U; position < incoming.edge_ids.size(); ++position) {
    const std::size_t edge_id =
        static_cast<std::size_t>(incoming.edge_ids[position].value());
    index.csc_ranks[edge_id] = position;
  }
  return index;
}

[[nodiscard]] std::optional<std::int64_t> add_neighbor_delta(
    const std::int64_t coordinate,
    const int delta) noexcept {
  if (delta > 0 && coordinate == std::numeric_limits<std::int64_t>::max()) {
    return std::nullopt;
  }
  if (delta < 0 && coordinate == std::numeric_limits<std::int64_t>::min()) {
    return std::nullopt;
  }
  return coordinate + delta;
}

template <typename T, typename Less>
void sort_and_deduplicate(std::vector<T>& values, Less less) {
  std::sort(values.begin(), values.end(), less);
  values.erase(
      std::unique(
          values.begin(),
          values.end(),
          [&less](const T& left, const T& right) {
            return !less(left, right) && !less(right, left);
          }),
      values.end());
}

[[nodiscard]] std::vector<std::vector<TileId>> expected_neighbor_groups(
    const WeightedGraph& graph,
    const GraphEdgeIndex& edges) {
  const std::size_t tile_count = graph.tile_coordinates().size();
  const TileId spill_tile = checked_id<TileId>(tile_count - 1U);
  std::vector<std::vector<TileId>> neighbors(tile_count);

  std::map<std::pair<std::int64_t, std::int64_t>, TileId> located_tile_ids;
  for (std::size_t tile = 0U; tile + 1U < tile_count; ++tile) {
    const TileCoordinate coordinate = graph.tile_coordinates()[tile];
    located_tile_ids.emplace(
        std::pair{coordinate.tile_x, coordinate.tile_y}, checked_id<TileId>(tile));
  }

  for (std::size_t tile = 0U; tile + 1U < tile_count; ++tile) {
    const TileCoordinate coordinate = graph.tile_coordinates()[tile];
    for (int delta_y = -1; delta_y <= 1; ++delta_y) {
      for (int delta_x = -1; delta_x <= 1; ++delta_x) {
        if (delta_x == 0 && delta_y == 0) {
          continue;
        }
        const auto neighbor_x = add_neighbor_delta(coordinate.tile_x, delta_x);
        const auto neighbor_y = add_neighbor_delta(coordinate.tile_y, delta_y);
        if (!neighbor_x || !neighbor_y) {
          continue;
        }
        const auto found = located_tile_ids.find(std::pair{*neighbor_x, *neighbor_y});
        if (found != located_tile_ids.end()) {
          neighbors[tile].push_back(found->second);
        }
      }
    }
  }

  for (std::size_t edge_id = 0U; edge_id < edges.sources.size(); ++edge_id) {
    const TileId source_tile = graph.owner_tiles()[edges.sources[edge_id].value()];
    const TileId destination_tile =
        graph.owner_tiles()[edges.destinations[edge_id].value()];
    if (source_tile == destination_tile) {
      continue;
    }
    if (source_tile == spill_tile || destination_tile == spill_tile) {
      neighbors[source_tile.value()].push_back(destination_tile);
      neighbors[destination_tile.value()].push_back(source_tile);
    }
  }

  for (std::vector<TileId>& group : neighbors) {
    sort_and_deduplicate(
        group, [](const TileId left, const TileId right) { return left < right; });
  }
  return neighbors;
}

[[nodiscard]] std::vector<std::vector<VertexId>> expected_halo_groups(
    const WeightedGraph& graph,
    const GraphEdgeIndex& edges) {
  std::vector<std::vector<VertexId>> halos(graph.tile_coordinates().size());
  for (std::size_t edge_id = 0U; edge_id < edges.sources.size(); ++edge_id) {
    const VertexId source = edges.sources[edge_id];
    const VertexId destination = edges.destinations[edge_id];
    const TileId source_tile = graph.owner_tiles()[source.value()];
    const TileId destination_tile = graph.owner_tiles()[destination.value()];
    if (source_tile == destination_tile) {
      continue;
    }
    halos[source_tile.value()].push_back(destination);
    halos[destination_tile.value()].push_back(source);
  }

  const auto halo_less = [&graph](const VertexId left, const VertexId right) {
    return std::tuple{graph.owner_tiles()[left.value()].value(), left.value()} <
           std::tuple{graph.owner_tiles()[right.value()].value(), right.value()};
  };
  for (std::vector<VertexId>& group : halos) {
    sort_and_deduplicate(group, halo_less);
  }
  return halos;
}

[[nodiscard]] bool valid_group_offsets(
    const std::span<const EdgeOffset> offsets,
    const std::size_t group_count,
    const std::size_t value_count) noexcept {
  if (offsets.size() != group_count + 1U || offsets.empty() || offsets.front() != 0U ||
      offsets.back() != value_count) {
    return false;
  }
  for (std::size_t index = 1U; index < offsets.size(); ++index) {
    if (offsets[index] < offsets[index - 1U] || offsets[index] > value_count) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] TileDirectoryValidationResult directory_error(
    const TileDirectoryValidationErrorCode code,
    const EdgeOffset position = TileDirectoryValidationResult::no_position) noexcept {
  return TileDirectoryValidationResult{code, position};
}

}  // namespace

TileDirectory build_tile_directory(const WeightedGraph& graph) {
  const WeightedGraphValidationResult graph_validation = validate_weighted_graph(graph);
  if (!graph_validation.ok()) {
    throw std::invalid_argument{"tile directory requires a deeply valid weighted graph"};
  }
  if (!graph.has_spatial_ordering()) {
    throw std::invalid_argument{"tile directory requires a spatially ordered graph"};
  }

  const std::size_t tile_count = graph.tile_coordinates().size();
  const GraphEdgeIndex edges = index_graph_edges(graph);
  std::vector<std::vector<EdgeId>> internal_edges(tile_count);
  std::vector<std::vector<EdgeId>> outgoing_cross_edges(tile_count);
  std::vector<std::vector<EdgeId>> incoming_cross_edges(tile_count);

  const OutgoingCsrView outgoing = graph.outgoing();
  for (std::size_t source = 0U; source < graph.vertex_count(); ++source) {
    const TileId source_tile = graph.owner_tiles()[source];
    const std::size_t row_begin = static_cast<std::size_t>(outgoing.row_offsets[source]);
    const std::size_t row_end =
        static_cast<std::size_t>(outgoing.row_offsets[source + 1U]);
    for (std::size_t position = row_begin; position < row_end; ++position) {
      const VertexId destination = outgoing.destinations[position];
      const TileId destination_tile = graph.owner_tiles()[destination.value()];
      if (source_tile == destination_tile) {
        internal_edges[source_tile.value()].push_back(outgoing.edge_ids[position]);
      } else {
        outgoing_cross_edges[source_tile.value()].push_back(outgoing.edge_ids[position]);
      }
    }
  }

  const IncomingCscView incoming = graph.incoming();
  for (std::size_t destination = 0U; destination < graph.vertex_count(); ++destination) {
    const TileId destination_tile = graph.owner_tiles()[destination];
    const std::size_t column_begin =
        static_cast<std::size_t>(incoming.column_offsets[destination]);
    const std::size_t column_end =
        static_cast<std::size_t>(incoming.column_offsets[destination + 1U]);
    for (std::size_t position = column_begin; position < column_end; ++position) {
      const VertexId source = incoming.sources[position];
      if (graph.owner_tiles()[source.value()] != destination_tile) {
        incoming_cross_edges[destination_tile.value()].push_back(
            incoming.edge_ids[position]);
      }
    }
  }

  const auto neighbor_groups = expected_neighbor_groups(graph, edges);
  const auto halo_groups = expected_halo_groups(graph, edges);
  FlattenedGroups<TileId> flattened_neighbors = flatten_groups(neighbor_groups);
  FlattenedGroups<EdgeId> flattened_internal = flatten_groups(internal_edges);
  FlattenedGroups<EdgeId> flattened_outgoing = flatten_groups(outgoing_cross_edges);
  FlattenedGroups<EdgeId> flattened_incoming = flatten_groups(incoming_cross_edges);
  FlattenedGroups<VertexId> flattened_halos = flatten_groups(halo_groups);

  TileDirectory directory;
  directory.spill_tile_ = checked_id<TileId>(tile_count - 1U);
  directory.neighbor_tile_offsets_ = std::move(flattened_neighbors.offsets);
  directory.neighbor_tiles_ = std::move(flattened_neighbors.values);
  directory.internal_edge_offsets_ = std::move(flattened_internal.offsets);
  directory.internal_edge_ids_ = std::move(flattened_internal.values);
  directory.outgoing_cross_edge_offsets_ = std::move(flattened_outgoing.offsets);
  directory.outgoing_cross_edge_ids_ = std::move(flattened_outgoing.values);
  directory.incoming_cross_edge_offsets_ = std::move(flattened_incoming.offsets);
  directory.incoming_cross_edge_ids_ = std::move(flattened_incoming.values);
  directory.halo_vertex_offsets_ = std::move(flattened_halos.offsets);
  directory.halo_vertices_ = std::move(flattened_halos.values);

  const TileDirectoryValidationResult validation = validate_tile_directory(graph, directory);
  if (!validation.ok()) {
    throw std::logic_error{
        "constructed tile directory failed validation with code " +
        std::to_string(static_cast<unsigned int>(validation.code)) + " at position " +
        std::to_string(validation.position)};
  }
  return directory;
}

TileDirectoryValidationResult validate_tile_directory(
    const WeightedGraph& graph,
    const TileDirectory& directory) {
  if (!validate_weighted_graph(graph).ok()) {
    return directory_error(TileDirectoryValidationErrorCode::graph_validation_failed);
  }
  if (!graph.has_spatial_ordering()) {
    return directory_error(
        TileDirectoryValidationErrorCode::graph_is_not_spatially_ordered);
  }

  const std::size_t tile_count = graph.tile_coordinates().size();
  if (directory.tile_count() != tile_count) {
    return directory_error(TileDirectoryValidationErrorCode::tile_count_mismatch);
  }
  if (directory.spill_tile() != checked_id<TileId>(tile_count - 1U)) {
    return directory_error(TileDirectoryValidationErrorCode::spill_tile_mismatch);
  }
  if (!valid_group_offsets(
          directory.neighbor_tile_offsets(), tile_count, directory.neighbor_tiles().size())) {
    return directory_error(TileDirectoryValidationErrorCode::invalid_neighbor_offsets);
  }
  if (!valid_group_offsets(
          directory.internal_edge_offsets(), tile_count, directory.internal_edge_ids().size())) {
    return directory_error(
        TileDirectoryValidationErrorCode::invalid_internal_edge_offsets);
  }
  if (!valid_group_offsets(
          directory.outgoing_cross_edge_offsets(),
          tile_count,
          directory.outgoing_cross_edge_ids().size())) {
    return directory_error(
        TileDirectoryValidationErrorCode::invalid_outgoing_cross_edge_offsets);
  }
  if (!valid_group_offsets(
          directory.incoming_cross_edge_offsets(),
          tile_count,
          directory.incoming_cross_edge_ids().size())) {
    return directory_error(
        TileDirectoryValidationErrorCode::invalid_incoming_cross_edge_offsets);
  }
  if (!valid_group_offsets(
          directory.halo_vertex_offsets(), tile_count, directory.halo_vertices().size())) {
    return directory_error(TileDirectoryValidationErrorCode::invalid_halo_offsets);
  }

  const GraphEdgeIndex edges = index_graph_edges(graph);
  const auto expected_neighbors = expected_neighbor_groups(graph, edges);
  for (std::size_t tile = 0U; tile < tile_count; ++tile) {
    const std::size_t begin =
        static_cast<std::size_t>(directory.neighbor_tile_offsets()[tile]);
    const std::size_t end =
        static_cast<std::size_t>(directory.neighbor_tile_offsets()[tile + 1U]);
    for (std::size_t position = begin; position < end; ++position) {
      const TileId neighbor = directory.neighbor_tiles()[position];
      if (neighbor.value() >= tile_count || neighbor == checked_id<TileId>(tile)) {
        return directory_error(
            TileDirectoryValidationErrorCode::neighbor_out_of_range, position);
      }
      if (position > begin && !(directory.neighbor_tiles()[position - 1U] < neighbor)) {
        return directory_error(
            TileDirectoryValidationErrorCode::neighbor_order_violation, position);
      }
      const std::size_t reverse_begin = static_cast<std::size_t>(
          directory.neighbor_tile_offsets()[neighbor.value()]);
      const std::size_t reverse_end = static_cast<std::size_t>(
          directory.neighbor_tile_offsets()[neighbor.value() + 1U]);
      if (!std::binary_search(
              directory.neighbor_tiles().begin() +
                  static_cast<std::ptrdiff_t>(reverse_begin),
              directory.neighbor_tiles().begin() +
                  static_cast<std::ptrdiff_t>(reverse_end),
              checked_id<TileId>(tile))) {
        return directory_error(
            TileDirectoryValidationErrorCode::neighbor_asymmetry, position);
      }
    }
    if (!std::equal(
            directory.neighbor_tiles().begin() + static_cast<std::ptrdiff_t>(begin),
            directory.neighbor_tiles().begin() + static_cast<std::ptrdiff_t>(end),
            expected_neighbors[tile].begin(),
            expected_neighbors[tile].end())) {
      return directory_error(TileDirectoryValidationErrorCode::unexpected_neighbor, tile);
    }
  }

  const std::size_t edge_count = static_cast<std::size_t>(graph.edge_count());
  std::vector<bool> source_classified(edge_count, false);
  std::vector<bool> incoming_cross_seen(edge_count, false);
  for (std::size_t tile = 0U; tile < tile_count; ++tile) {
    const TileId tile_id = checked_id<TileId>(tile);
    const auto validate_source_list = [&](const std::span<const EdgeOffset> offsets,
                                          const std::span<const EdgeId> edge_ids,
                                          const bool expect_internal)
        -> TileDirectoryValidationResult {
      const std::size_t begin = static_cast<std::size_t>(offsets[tile]);
      const std::size_t end = static_cast<std::size_t>(offsets[tile + 1U]);
      std::size_t previous_rank = 0U;
      bool has_previous = false;
      for (std::size_t position = begin; position < end; ++position) {
        const EdgeId edge_id = edge_ids[position];
        if (edge_id.value() >= graph.edge_count()) {
          return directory_error(
              TileDirectoryValidationErrorCode::edge_id_out_of_range, position);
        }
        const std::size_t logical_id = static_cast<std::size_t>(edge_id.value());
        const TileId source_tile = graph.owner_tiles()[edges.sources[logical_id].value()];
        const TileId destination_tile =
            graph.owner_tiles()[edges.destinations[logical_id].value()];
        const bool is_internal = source_tile == destination_tile;
        if (source_tile != tile_id || is_internal != expect_internal) {
          return directory_error(
              TileDirectoryValidationErrorCode::edge_classification_mismatch, position);
        }
        if (source_classified[logical_id]) {
          return directory_error(
              TileDirectoryValidationErrorCode::duplicate_source_edge_classification,
              position);
        }
        if (has_previous && edges.csr_ranks[logical_id] <= previous_rank) {
          return directory_error(
              TileDirectoryValidationErrorCode::edge_metadata_order_violation, position);
        }
        source_classified[logical_id] = true;
        previous_rank = edges.csr_ranks[logical_id];
        has_previous = true;
      }
      return {};
    };

    const TileDirectoryValidationResult internal_validation = validate_source_list(
        directory.internal_edge_offsets(), directory.internal_edge_ids(), true);
    if (!internal_validation.ok()) {
      return internal_validation;
    }
    const TileDirectoryValidationResult outgoing_validation = validate_source_list(
        directory.outgoing_cross_edge_offsets(),
        directory.outgoing_cross_edge_ids(),
        false);
    if (!outgoing_validation.ok()) {
      return outgoing_validation;
    }

    const std::size_t incoming_begin =
        static_cast<std::size_t>(directory.incoming_cross_edge_offsets()[tile]);
    const std::size_t incoming_end =
        static_cast<std::size_t>(directory.incoming_cross_edge_offsets()[tile + 1U]);
    std::size_t previous_csc_rank = 0U;
    bool has_previous_csc_rank = false;
    for (std::size_t position = incoming_begin; position < incoming_end; ++position) {
      const EdgeId edge_id = directory.incoming_cross_edge_ids()[position];
      if (edge_id.value() >= graph.edge_count()) {
        return directory_error(
            TileDirectoryValidationErrorCode::edge_id_out_of_range, position);
      }
      const std::size_t logical_id = static_cast<std::size_t>(edge_id.value());
      const TileId source_tile = graph.owner_tiles()[edges.sources[logical_id].value()];
      const TileId destination_tile =
          graph.owner_tiles()[edges.destinations[logical_id].value()];
      if (destination_tile != tile_id || source_tile == destination_tile) {
        return directory_error(
            TileDirectoryValidationErrorCode::edge_classification_mismatch, position);
      }
      if (incoming_cross_seen[logical_id]) {
        return directory_error(
            TileDirectoryValidationErrorCode::duplicate_incoming_cross_edge, position);
      }
      if (has_previous_csc_rank &&
          edges.csc_ranks[logical_id] <= previous_csc_rank) {
        return directory_error(
            TileDirectoryValidationErrorCode::edge_metadata_order_violation, position);
      }
      incoming_cross_seen[logical_id] = true;
      previous_csc_rank = edges.csc_ranks[logical_id];
      has_previous_csc_rank = true;
    }
  }

  for (std::size_t edge_id = 0U; edge_id < edge_count; ++edge_id) {
    if (!source_classified[edge_id]) {
      return directory_error(
          TileDirectoryValidationErrorCode::missing_source_edge_classification, edge_id);
    }
    const TileId source_tile = graph.owner_tiles()[edges.sources[edge_id].value()];
    const TileId destination_tile =
        graph.owner_tiles()[edges.destinations[edge_id].value()];
    if (source_tile != destination_tile && !incoming_cross_seen[edge_id]) {
      return directory_error(
          TileDirectoryValidationErrorCode::missing_incoming_cross_edge, edge_id);
    }
    if (source_tile == destination_tile && incoming_cross_seen[edge_id]) {
      return directory_error(
          TileDirectoryValidationErrorCode::edge_classification_mismatch, edge_id);
    }
  }

  const auto expected_halos = expected_halo_groups(graph, edges);
  const auto halo_less = [&graph](const VertexId left, const VertexId right) {
    return std::tuple{graph.owner_tiles()[left.value()].value(), left.value()} <
           std::tuple{graph.owner_tiles()[right.value()].value(), right.value()};
  };
  for (std::size_t tile = 0U; tile < tile_count; ++tile) {
    const std::size_t begin =
        static_cast<std::size_t>(directory.halo_vertex_offsets()[tile]);
    const std::size_t end =
        static_cast<std::size_t>(directory.halo_vertex_offsets()[tile + 1U]);
    for (std::size_t position = begin; position < end; ++position) {
      const VertexId halo = directory.halo_vertices()[position];
      if (!is_valid_vertex_id(halo, graph.vertex_count())) {
        return directory_error(
            TileDirectoryValidationErrorCode::halo_vertex_out_of_range, position);
      }
      if (graph.owner_tiles()[halo.value()] == checked_id<TileId>(tile)) {
        return directory_error(
            TileDirectoryValidationErrorCode::halo_owner_mismatch, position);
      }
      if (position > begin &&
          !halo_less(directory.halo_vertices()[position - 1U], halo)) {
        return directory_error(
            TileDirectoryValidationErrorCode::halo_order_violation, position);
      }
    }
    if (!std::equal(
            directory.halo_vertices().begin() + static_cast<std::ptrdiff_t>(begin),
            directory.halo_vertices().begin() + static_cast<std::ptrdiff_t>(end),
            expected_halos[tile].begin(),
            expected_halos[tile].end())) {
      return directory_error(TileDirectoryValidationErrorCode::halo_set_mismatch, tile);
    }
  }

  return {};
}

UniformGridPartitioner::UniformGridPartitioner(
    const SpatialOrderConfig config,
    LocalityKeyPolicy locality_policy)
    : config_{config}, locality_policy_{std::move(locality_policy)} {
  if (config_.tile_width == 0U || config_.tile_height == 0U) {
    throw std::invalid_argument{"tile width and height must both be positive"};
  }
  if (!locality_policy_) {
    throw std::invalid_argument{"locality-key policy must be callable"};
  }
}

PartitionedGraph UniformGridPartitioner::partition(const InputGraph& input) const {
  WeightedGraph graph = build_spatially_ordered_graph(input, config_, locality_policy_);
  TileDirectory directory = build_tile_directory(graph);
  return PartitionedGraph{std::move(graph), std::move(directory)};
}

}  // namespace bfnew
