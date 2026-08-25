#include "bfnew/spatial.hpp"

#include "bfnew/graph.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

[[nodiscard]] std::int64_t checked_add(
    const std::int64_t left,
    const std::int64_t right) {
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
    throw std::overflow_error{"signed coordinate addition overflow"};
  }
  return left + right;
}

[[nodiscard]] std::int64_t checked_subtract(
    const std::int64_t left,
    const std::int64_t right) {
  if ((right > 0 && left < std::numeric_limits<std::int64_t>::min() + right) ||
      (right < 0 && left > std::numeric_limits<std::int64_t>::max() + right)) {
    throw std::overflow_error{"signed coordinate subtraction overflow"};
  }
  return left - right;
}

[[nodiscard]] std::int64_t checked_multiply_positive(
    const std::int64_t value,
    const std::uint32_t positive_factor) {
  if (positive_factor == 0U) {
    throw std::invalid_argument{"tile dimension must be positive"};
  }
  const std::int64_t factor = static_cast<std::int64_t>(positive_factor);
  if (value > std::numeric_limits<std::int64_t>::max() / factor ||
      value < std::numeric_limits<std::int64_t>::min() / factor) {
    throw std::overflow_error{"signed tile-origin multiplication overflow"};
  }
  return value * factor;
}

[[nodiscard]] std::int64_t floor_divide_positive(
    const std::int64_t numerator,
    const std::uint32_t positive_denominator) {
  if (positive_denominator == 0U) {
    throw std::invalid_argument{"tile dimension must be positive"};
  }
  const std::int64_t denominator =
      static_cast<std::int64_t>(positive_denominator);
  std::int64_t quotient = numerator / denominator;
  const std::int64_t remainder = numerator % denominator;
  if (remainder < 0) {
    --quotient;
  }
  return quotient;
}

struct DerivedLocation {
  TileCoordinate tile;
  std::uint32_t local_x;
  std::uint32_t local_y;
};

[[nodiscard]] DerivedLocation derive_location(
    const VertexMetadata& metadata,
    const SpatialOrderConfig& config) {
  const std::int64_t coordinate_x = static_cast<std::int64_t>(metadata.x);
  const std::int64_t coordinate_y = static_cast<std::int64_t>(metadata.y);
  const std::int64_t origin_x = static_cast<std::int64_t>(config.origin_x);
  const std::int64_t origin_y = static_cast<std::int64_t>(config.origin_y);
  const std::int64_t delta_x = checked_subtract(coordinate_x, origin_x);
  const std::int64_t delta_y = checked_subtract(coordinate_y, origin_y);
  const std::int64_t tile_x = floor_divide_positive(delta_x, config.tile_width);
  const std::int64_t tile_y = floor_divide_positive(delta_y, config.tile_height);

  const std::int64_t lower_x = checked_add(
      origin_x, checked_multiply_positive(tile_x, config.tile_width));
  const std::int64_t lower_y = checked_add(
      origin_y, checked_multiply_positive(tile_y, config.tile_height));
  const std::int64_t local_x = checked_subtract(coordinate_x, lower_x);
  const std::int64_t local_y = checked_subtract(coordinate_y, lower_y);
  if (local_x < 0 || local_x >= static_cast<std::int64_t>(config.tile_width) ||
      local_y < 0 || local_y >= static_cast<std::int64_t>(config.tile_height)) {
    throw std::logic_error{"derived local coordinate is outside its selected tile"};
  }

  return DerivedLocation{
      TileCoordinate::located(tile_x, tile_y),
      static_cast<std::uint32_t>(local_x),
      static_cast<std::uint32_t>(local_y),
  };
}

[[nodiscard]] bool tile_coordinate_less(
    const TileCoordinate& left,
    const TileCoordinate& right) noexcept {
  return std::tuple{left.tile_y, left.tile_x} <
         std::tuple{right.tile_y, right.tile_x};
}

[[nodiscard]] bool same_tile_coordinate(
    const TileCoordinate& left,
    const TileCoordinate& right) noexcept {
  return left.tile_x == right.tile_x && left.tile_y == right.tile_y;
}

struct VertexOrderRecord {
  VertexId original_vertex;
  bool has_location;
  TileCoordinate tile;
  TileId owner_tile;
  ResourceClassId resource_class;
  std::uint64_t locality_key;
};

struct LayoutEdge {
  VertexId source;
  VertexId destination;
  float weight;
  EdgeId edge_id;
};

[[nodiscard]] std::uint64_t spread_morton_bits(const std::uint32_t input) noexcept {
  std::uint64_t value = input;
  value = (value | (value << 16U)) & 0x0000FFFF0000FFFFULL;
  value = (value | (value << 8U)) & 0x00FF00FF00FF00FFULL;
  value = (value | (value << 4U)) & 0x0F0F0F0F0F0F0F0FULL;
  value = (value | (value << 2U)) & 0x3333333333333333ULL;
  value = (value | (value << 1U)) & 0x5555555555555555ULL;
  return value;
}

}  // namespace

std::uint64_t MortonLocalityPolicy::operator()(
    const std::uint32_t local_x,
    const std::uint32_t local_y) const noexcept {
  return spread_morton_bits(local_x) | (spread_morton_bits(local_y) << 1U);
}

WeightedGraph build_spatially_ordered_graph(
    const InputGraph& input,
    const SpatialOrderConfig& config,
    const LocalityKeyPolicy& locality_policy) {
  if (config.tile_width == 0U || config.tile_height == 0U) {
    throw std::invalid_argument{"tile width and height must both be positive"};
  }
  if (!locality_policy) {
    throw std::invalid_argument{"locality-key policy must be callable"};
  }

  const WeightedGraph original_graph = build_weighted_graph(input);
  std::vector<VertexOrderRecord> order_records;
  order_records.reserve(input.vertex_count());
  std::vector<TileCoordinate> located_tiles;
  located_tiles.reserve(input.vertex_count());

  for (std::size_t old_index = 0U; old_index < input.vertices().size(); ++old_index) {
    const VertexMetadata& metadata = input.vertices()[old_index];
    const VertexId original_vertex = checked_id<VertexId>(old_index);
    if (!metadata.has_location) {
      order_records.push_back(VertexOrderRecord{
          original_vertex,
          false,
          TileCoordinate::spill(),
          TileId{},
          metadata.resource_class,
          0U,
      });
      continue;
    }

    const DerivedLocation location = derive_location(metadata, config);
    located_tiles.push_back(location.tile);
    order_records.push_back(VertexOrderRecord{
        original_vertex,
        true,
        location.tile,
        TileId{},
        metadata.resource_class,
        locality_policy(location.local_x, location.local_y),
    });
  }

  std::sort(located_tiles.begin(), located_tiles.end(), tile_coordinate_less);
  located_tiles.erase(
      std::unique(located_tiles.begin(), located_tiles.end(), same_tile_coordinate),
      located_tiles.end());
  const TileId spill_tile = checked_id<TileId>(located_tiles.size());

  for (VertexOrderRecord& record : order_records) {
    if (!record.has_location) {
      record.owner_tile = spill_tile;
      continue;
    }
    const auto tile_position = std::lower_bound(
        located_tiles.begin(), located_tiles.end(), record.tile, tile_coordinate_less);
    if (tile_position == located_tiles.end() ||
        !same_tile_coordinate(*tile_position, record.tile)) {
      throw std::logic_error{"located vertex tile is absent from the dense tile table"};
    }
    record.owner_tile = checked_id<TileId>(
        static_cast<std::size_t>(tile_position - located_tiles.begin()));
  }

  std::sort(
      order_records.begin(),
      order_records.end(),
      [](const VertexOrderRecord& left, const VertexOrderRecord& right) {
        if (left.has_location != right.has_location) {
          return left.has_location;
        }
        if (!left.has_location) {
          return std::tuple{left.resource_class.value(), left.original_vertex.value()} <
                 std::tuple{right.resource_class.value(), right.original_vertex.value()};
        }
        return std::tuple{
                   left.tile.tile_y,
                   left.tile.tile_x,
                   left.resource_class.value(),
                   left.locality_key,
                   left.original_vertex.value()} <
               std::tuple{
                   right.tile.tile_y,
                   right.tile.tile_x,
                   right.resource_class.value(),
                   right.locality_key,
                   right.original_vertex.value()};
      });

  WeightedGraph graph;
  graph.vertex_count_ = input.vertex_count();
  graph.edge_count_ = original_graph.edge_count();
  graph.edge_provenance_.assign(
      original_graph.edge_provenance().begin(), original_graph.edge_provenance().end());
  graph.has_spatial_ordering_ = true;
  graph.spatial_order_config_ = config;
  graph.tile_coordinates_ = located_tiles;
  graph.tile_coordinates_.push_back(TileCoordinate::spill());

  graph.vertices_.reserve(order_records.size());
  graph.old_to_new_.resize(order_records.size());
  graph.new_to_old_.reserve(order_records.size());
  graph.original_vertex_ids_.reserve(order_records.size());
  graph.owner_tiles_.reserve(order_records.size());
  for (std::size_t new_index = 0U; new_index < order_records.size(); ++new_index) {
    const VertexOrderRecord& record = order_records[new_index];
    const VertexId new_vertex = checked_id<VertexId>(new_index);
    graph.old_to_new_[record.original_vertex.value()] = new_vertex;
    graph.new_to_old_.push_back(record.original_vertex);
    graph.original_vertex_ids_.push_back(record.original_vertex);
    graph.owner_tiles_.push_back(record.owner_tile);
    graph.vertices_.push_back(input.vertices()[record.original_vertex.value()]);
  }

  graph.tile_vertex_offsets_.assign(graph.tile_coordinates_.size() + 1U, 0U);
  for (const TileId owner_tile : graph.owner_tiles_) {
    const std::size_t next_tile = static_cast<std::size_t>(owner_tile.value()) + 1U;
    ++graph.tile_vertex_offsets_[next_tile];
  }
  for (std::size_t tile = 1U; tile < graph.tile_vertex_offsets_.size(); ++tile) {
    graph.tile_vertex_offsets_[tile] += graph.tile_vertex_offsets_[tile - 1U];
  }

  std::vector<LayoutEdge> layout_edges;
  layout_edges.reserve(static_cast<std::size_t>(original_graph.edge_count()));
  const OutgoingCsrView original_outgoing = original_graph.outgoing();
  for (std::size_t old_source = 0U; old_source < input.vertex_count(); ++old_source) {
    const std::size_t row_begin =
        static_cast<std::size_t>(original_outgoing.row_offsets[old_source]);
    const std::size_t row_end =
        static_cast<std::size_t>(original_outgoing.row_offsets[old_source + 1U]);
    for (std::size_t position = row_begin; position < row_end; ++position) {
      layout_edges.push_back(LayoutEdge{
          graph.old_to_new_[old_source],
          graph.old_to_new_[original_outgoing.destinations[position].value()],
          original_outgoing.weights[position],
          original_outgoing.edge_ids[position],
      });
    }
  }

  std::vector<LayoutEdge> csr_edges = layout_edges;
  std::sort(
      csr_edges.begin(),
      csr_edges.end(),
      [&graph](const LayoutEdge& left, const LayoutEdge& right) {
        return std::tuple{
                   left.source.value(),
                   graph.owner_tiles_[left.destination.value()].value(),
                   left.destination.value(),
                   left.edge_id.value()} <
               std::tuple{
                   right.source.value(),
                   graph.owner_tiles_[right.destination.value()].value(),
                   right.destination.value(),
                   right.edge_id.value()};
      });

  graph.outgoing_row_offsets_.assign(
      static_cast<std::size_t>(graph.vertex_count_) + 1U, 0U);
  for (const LayoutEdge& edge : csr_edges) {
    ++graph.outgoing_row_offsets_[static_cast<std::size_t>(edge.source.value()) + 1U];
  }
  for (std::size_t row = 1U; row < graph.outgoing_row_offsets_.size(); ++row) {
    graph.outgoing_row_offsets_[row] += graph.outgoing_row_offsets_[row - 1U];
  }
  graph.outgoing_destinations_.reserve(csr_edges.size());
  graph.outgoing_weights_.reserve(csr_edges.size());
  graph.outgoing_edge_ids_.reserve(csr_edges.size());
  for (const LayoutEdge& edge : csr_edges) {
    graph.outgoing_destinations_.push_back(edge.destination);
    graph.outgoing_weights_.push_back(edge.weight);
    graph.outgoing_edge_ids_.push_back(edge.edge_id);
  }

  std::vector<LayoutEdge> csc_edges = std::move(layout_edges);
  std::sort(
      csc_edges.begin(),
      csc_edges.end(),
      [&graph](const LayoutEdge& left, const LayoutEdge& right) {
        return std::tuple{
                   left.destination.value(),
                   graph.owner_tiles_[left.source.value()].value(),
                   left.source.value(),
                   left.edge_id.value()} <
               std::tuple{
                   right.destination.value(),
                   graph.owner_tiles_[right.source.value()].value(),
                   right.source.value(),
                   right.edge_id.value()};
      });

  graph.incoming_column_offsets_.assign(
      static_cast<std::size_t>(graph.vertex_count_) + 1U, 0U);
  for (const LayoutEdge& edge : csc_edges) {
    ++graph.incoming_column_offsets_[
        static_cast<std::size_t>(edge.destination.value()) + 1U];
  }
  for (std::size_t column = 1U; column < graph.incoming_column_offsets_.size(); ++column) {
    graph.incoming_column_offsets_[column] +=
        graph.incoming_column_offsets_[column - 1U];
  }
  graph.incoming_sources_.reserve(csc_edges.size());
  graph.incoming_weights_.reserve(csc_edges.size());
  graph.incoming_edge_ids_.reserve(csc_edges.size());
  for (const LayoutEdge& edge : csc_edges) {
    graph.incoming_sources_.push_back(edge.source);
    graph.incoming_weights_.push_back(edge.weight);
    graph.incoming_edge_ids_.push_back(edge.edge_id);
  }

  const WeightedGraphValidationResult validation = validate_weighted_graph(graph);
  if (!validation.ok()) {
    throw std::logic_error{
        "spatially reordered graph failed deep validation with code " +
        std::to_string(static_cast<unsigned int>(validation.code)) + " at position " +
        std::to_string(validation.position)};
  }
  return graph;
}

}  // namespace bfnew
