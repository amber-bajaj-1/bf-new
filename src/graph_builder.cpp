#include "bfnew/graph.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

[[noreturn]] void throw_validation_error(const GraphValidationResult result) {
  const std::string edge_suffix =
      result.edge_index == GraphValidationResult::no_edge
          ? std::string{}
          : std::string{" at edge input index "} + std::to_string(result.edge_index);

  switch (result.code) {
    case GraphValidationErrorCode::vertex_count_overflow:
      throw std::length_error{"vertex count exceeds the 32-bit representation"};
    case GraphValidationErrorCode::edge_count_overflow:
      throw std::length_error{"edge count exceeds the 64-bit representation"};
    case GraphValidationErrorCode::nonfinite_weight:
      throw std::invalid_argument{"edge weight must be finite" + edge_suffix};
    case GraphValidationErrorCode::negative_weight:
      throw std::invalid_argument{"edge weight must be nonnegative" + edge_suffix};
    case GraphValidationErrorCode::source_out_of_range:
      throw std::out_of_range{"edge source is outside the vertex range" + edge_suffix};
    case GraphValidationErrorCode::destination_out_of_range:
      throw std::out_of_range{"edge destination is outside the vertex range" + edge_suffix};
    case GraphValidationErrorCode::none:
      break;
  }

  throw std::logic_error{"attempted to throw a successful graph validation result"};
}

struct LogicalEdgeRecord {
  VertexId source;
  VertexId destination;
  float weight;
  PhysicalProvenance provenance;
  EdgeId edge_id;
};

[[nodiscard]] auto canonical_edge_key(const EdgeInputRecord& edge) noexcept {
  return std::tuple{
      edge.source.value(),
      edge.destination.value(),
      canonical_weight_bits(edge.weight),
      edge.provenance.domain,
      edge.provenance.kind_and_flags,
      edge.provenance.source_record,
  };
}

[[nodiscard]] auto canonical_logical_edge_key(
    const VertexId source,
    const VertexId destination,
    const float weight,
    const PhysicalProvenance provenance) noexcept {
  return std::tuple{
      source.value(),
      destination.value(),
      canonical_weight_bits(weight),
      provenance.domain,
      provenance.kind_and_flags,
      provenance.source_record,
  };
}

struct EdgeIdentity {
  VertexId source{};
  VertexId destination{};
  std::uint32_t weight_bits{};

  constexpr bool operator==(const EdgeIdentity&) const noexcept = default;
};

[[nodiscard]] WeightedGraphValidationResult weighted_error(
    const WeightedGraphValidationErrorCode code,
    const EdgeOffset position = WeightedGraphValidationResult::no_position) noexcept {
  return WeightedGraphValidationResult{code, position};
}

}  // namespace

float canonicalize_weight(const float weight) noexcept {
  return weight == 0.0F ? 0.0F : weight;
}

bool is_valid_weight(const float weight) noexcept {
  return std::isfinite(weight) && !(weight < 0.0F);
}

std::uint32_t canonical_weight_bits(const float weight) noexcept {
  return std::bit_cast<std::uint32_t>(canonicalize_weight(weight));
}

bool is_valid_vertex_id(const VertexId id, const std::uint32_t vertex_count) noexcept {
  return id.value() < vertex_count;
}

GraphValidationResult validate_graph_input(
    const std::size_t vertex_count,
    const std::span<const EdgeInputRecord> edges) noexcept {
  if (!std::in_range<std::uint32_t>(vertex_count)) {
    return {GraphValidationErrorCode::vertex_count_overflow,
            GraphValidationResult::no_edge};
  }
  if (!std::in_range<std::uint64_t>(edges.size())) {
    return {GraphValidationErrorCode::edge_count_overflow,
            GraphValidationResult::no_edge};
  }

  const auto checked_vertex_count = static_cast<std::uint32_t>(vertex_count);
  for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    const EdgeInputRecord& edge = edges[edge_index];
    if (!std::isfinite(edge.weight)) {
      return {GraphValidationErrorCode::nonfinite_weight, edge_index};
    }
    if (edge.weight < 0.0F) {
      return {GraphValidationErrorCode::negative_weight, edge_index};
    }
    if (!is_valid_vertex_id(edge.source, checked_vertex_count)) {
      return {GraphValidationErrorCode::source_out_of_range, edge_index};
    }
    if (!is_valid_vertex_id(edge.destination, checked_vertex_count)) {
      return {GraphValidationErrorCode::destination_out_of_range, edge_index};
    }
  }

  return {};
}

bool are_valid_offsets(
    const std::span<const EdgeOffset> offsets,
    const std::uint32_t bucket_count,
    const EdgeCount entry_count) noexcept {
  const std::uint64_t expected_size = static_cast<std::uint64_t>(bucket_count) + 1U;
  if (!std::in_range<std::size_t>(expected_size) ||
      offsets.size() != static_cast<std::size_t>(expected_size)) {
    return false;
  }
  if (offsets.front() != 0U || offsets.back() != entry_count) {
    return false;
  }
  for (std::size_t index = 1U; index < offsets.size(); ++index) {
    if (offsets[index] < offsets[index - 1U] || offsets[index] > entry_count) {
      return false;
    }
  }
  return true;
}

InputGraph::InputGraph(
    std::vector<VertexMetadata> vertices,
    std::vector<EdgeInputRecord> edges)
    : vertices_{std::move(vertices)}, edges_{std::move(edges)} {
  for (VertexMetadata& metadata : vertices_) {
    if (!metadata.has_location) {
      metadata.x = 0;
      metadata.y = 0;
    }
  }
  for (EdgeInputRecord& edge : edges_) {
    edge.weight = canonicalize_weight(edge.weight);
  }

  const GraphValidationResult validation = validate_graph_input(vertices_.size(), edges_);
  if (!validation.ok()) {
    throw_validation_error(validation);
  }

  vertex_count_ = static_cast<std::uint32_t>(vertices_.size());
  edge_count_ = static_cast<std::uint64_t>(edges_.size());
}

WeightedGraph build_weighted_graph(const InputGraph& input) {
  WeightedGraph graph;
  graph.vertex_count_ = input.vertex_count();
  graph.edge_count_ = input.edge_count();
  graph.vertices_.assign(input.vertices().begin(), input.vertices().end());
  graph.old_to_new_.reserve(graph.vertex_count_);
  graph.new_to_old_.reserve(graph.vertex_count_);
  graph.original_vertex_ids_.reserve(graph.vertex_count_);
  for (std::uint32_t vertex = 0U; vertex < graph.vertex_count_; ++vertex) {
    const VertexId vertex_id{vertex};
    graph.old_to_new_.push_back(vertex_id);
    graph.new_to_old_.push_back(vertex_id);
    graph.original_vertex_ids_.push_back(vertex_id);
  }

  std::vector<EdgeInputRecord> canonical_edges(input.edges().begin(), input.edges().end());
  std::sort(
      canonical_edges.begin(),
      canonical_edges.end(),
      [](const EdgeInputRecord& left, const EdgeInputRecord& right) {
        return canonical_edge_key(left) < canonical_edge_key(right);
      });

  std::vector<LogicalEdgeRecord> logical_edges;
  logical_edges.reserve(canonical_edges.size());
  graph.edge_provenance_.reserve(canonical_edges.size());
  for (std::size_t index = 0U; index < canonical_edges.size(); ++index) {
    const EdgeInputRecord& edge = canonical_edges[index];
    const EdgeId edge_id = checked_id<EdgeId>(index);
    logical_edges.push_back(
        LogicalEdgeRecord{edge.source, edge.destination, edge.weight, edge.provenance, edge_id});
    graph.edge_provenance_.push_back(edge.provenance);
  }

  std::vector<LogicalEdgeRecord> csr_edges = logical_edges;
  std::sort(
      csr_edges.begin(),
      csr_edges.end(),
      [](const LogicalEdgeRecord& left, const LogicalEdgeRecord& right) {
        return std::tuple{left.source.value(), left.destination.value(), left.edge_id.value()} <
               std::tuple{right.source.value(), right.destination.value(), right.edge_id.value()};
      });

  const std::size_t offset_count = static_cast<std::size_t>(graph.vertex_count_) + 1U;
  graph.outgoing_row_offsets_.assign(offset_count, 0U);
  for (const LogicalEdgeRecord& edge : csr_edges) {
    const std::size_t next_row = static_cast<std::size_t>(edge.source.value()) + 1U;
    ++graph.outgoing_row_offsets_[next_row];
  }
  for (std::size_t row = 1U; row < graph.outgoing_row_offsets_.size(); ++row) {
    graph.outgoing_row_offsets_[row] += graph.outgoing_row_offsets_[row - 1U];
  }

  graph.outgoing_destinations_.reserve(csr_edges.size());
  graph.outgoing_weights_.reserve(csr_edges.size());
  graph.outgoing_edge_ids_.reserve(csr_edges.size());
  for (const LogicalEdgeRecord& edge : csr_edges) {
    graph.outgoing_destinations_.push_back(edge.destination);
    graph.outgoing_weights_.push_back(edge.weight);
    graph.outgoing_edge_ids_.push_back(edge.edge_id);
  }

  std::vector<LogicalEdgeRecord> csc_edges = std::move(logical_edges);
  std::sort(
      csc_edges.begin(),
      csc_edges.end(),
      [](const LogicalEdgeRecord& left, const LogicalEdgeRecord& right) {
        return std::tuple{
                   left.destination.value(), left.source.value(), left.edge_id.value()} <
               std::tuple{
                   right.destination.value(), right.source.value(), right.edge_id.value()};
      });

  graph.incoming_column_offsets_.assign(offset_count, 0U);
  for (const LogicalEdgeRecord& edge : csc_edges) {
    const std::size_t next_column =
        static_cast<std::size_t>(edge.destination.value()) + 1U;
    ++graph.incoming_column_offsets_[next_column];
  }
  for (std::size_t column = 1U; column < graph.incoming_column_offsets_.size(); ++column) {
    graph.incoming_column_offsets_[column] +=
        graph.incoming_column_offsets_[column - 1U];
  }

  graph.incoming_sources_.reserve(csc_edges.size());
  graph.incoming_weights_.reserve(csc_edges.size());
  graph.incoming_edge_ids_.reserve(csc_edges.size());
  for (const LogicalEdgeRecord& edge : csc_edges) {
    graph.incoming_sources_.push_back(edge.source);
    graph.incoming_weights_.push_back(edge.weight);
    graph.incoming_edge_ids_.push_back(edge.edge_id);
  }

  const WeightedGraphValidationResult validation = validate_weighted_graph(graph);
  if (!validation.ok()) {
    throw std::logic_error{
        "constructed weighted graph failed deep validation with code " +
        std::to_string(static_cast<unsigned int>(validation.code)) + " at position " +
        std::to_string(validation.position)};
  }
  return graph;
}

WeightedGraphValidationResult validate_weighted_graph(const WeightedGraph& graph) {
  const EdgeCount edge_count = graph.edge_count();
  if (graph.vertices().size() != graph.vertex_count()) {
    return weighted_error(
        WeightedGraphValidationErrorCode::vertex_metadata_size_mismatch);
  }
  const std::size_t vertex_count = static_cast<std::size_t>(graph.vertex_count());
  if (graph.old_to_new().size() != vertex_count ||
      graph.new_to_old().size() != vertex_count ||
      graph.original_vertex_ids().size() != vertex_count) {
    return weighted_error(WeightedGraphValidationErrorCode::permutation_size_mismatch);
  }
  std::vector<bool> new_vertex_seen(vertex_count, false);
  for (std::size_t old_index = 0U; old_index < vertex_count; ++old_index) {
    const VertexId new_id = graph.old_to_new()[old_index];
    if (!is_valid_vertex_id(new_id, graph.vertex_count())) {
      return weighted_error(
          WeightedGraphValidationErrorCode::permutation_out_of_range, old_index);
    }
    const std::size_t new_index = static_cast<std::size_t>(new_id.value());
    if (new_vertex_seen[new_index] ||
        graph.new_to_old()[new_index] != VertexId{static_cast<std::uint32_t>(old_index)}) {
      return weighted_error(
          WeightedGraphValidationErrorCode::permutation_round_trip_mismatch, old_index);
    }
    new_vertex_seen[new_index] = true;
  }
  for (std::size_t new_index = 0U; new_index < vertex_count; ++new_index) {
    if (graph.original_vertex_ids()[new_index] != graph.new_to_old()[new_index]) {
      return weighted_error(
          WeightedGraphValidationErrorCode::original_vertex_id_mismatch, new_index);
    }
  }

  if (!graph.has_spatial_ordering()) {
    if (!graph.owner_tiles().empty() || !graph.tile_coordinates().empty() ||
        !graph.tile_vertex_offsets().empty()) {
      return weighted_error(WeightedGraphValidationErrorCode::incomplete_spatial_metadata);
    }
  } else {
    const std::size_t tile_count = graph.tile_coordinates().size();
    if (tile_count == 0U || graph.owner_tiles().size() != vertex_count ||
        graph.tile_vertex_offsets().size() != tile_count + 1U) {
      return weighted_error(WeightedGraphValidationErrorCode::incomplete_spatial_metadata);
    }
    for (std::size_t tile_index = 0U; tile_index < tile_count; ++tile_index) {
      const TileCoordinate coordinate = graph.tile_coordinates()[tile_index];
      const bool is_spill = tile_index + 1U == tile_count;
      if (coordinate.has_location == is_spill) {
        return weighted_error(
            WeightedGraphValidationErrorCode::invalid_tile_coordinate_table, tile_index);
      }
      if (tile_index > 0U && !is_spill) {
        const TileCoordinate previous = graph.tile_coordinates()[tile_index - 1U];
        const auto previous_key = std::tuple{previous.tile_y, previous.tile_x};
        const auto current_key = std::tuple{coordinate.tile_y, coordinate.tile_x};
        if (!(previous_key < current_key)) {
          return weighted_error(
              WeightedGraphValidationErrorCode::invalid_tile_coordinate_table, tile_index);
        }
      }
    }

    const auto tile_offsets = graph.tile_vertex_offsets();
    if (tile_offsets.front() != 0U || tile_offsets.back() != graph.vertex_count()) {
      return weighted_error(WeightedGraphValidationErrorCode::invalid_tile_vertex_offsets);
    }
    for (std::size_t tile_index = 0U; tile_index < tile_count; ++tile_index) {
      if (tile_offsets[tile_index] > tile_offsets[tile_index + 1U] ||
          tile_offsets[tile_index + 1U] > graph.vertex_count()) {
        return weighted_error(
            WeightedGraphValidationErrorCode::invalid_tile_vertex_offsets, tile_index);
      }
      const TileId expected_owner = checked_id<TileId>(tile_index);
      const std::size_t span_begin = static_cast<std::size_t>(tile_offsets[tile_index]);
      const std::size_t span_end =
          static_cast<std::size_t>(tile_offsets[tile_index + 1U]);
      for (std::size_t vertex_index = span_begin; vertex_index < span_end; ++vertex_index) {
        if (graph.owner_tiles()[vertex_index] != expected_owner) {
          return weighted_error(
              WeightedGraphValidationErrorCode::noncontiguous_tile_span, vertex_index);
        }
      }
    }
    for (std::size_t vertex_index = 0U; vertex_index < vertex_count; ++vertex_index) {
      if (graph.owner_tiles()[vertex_index].value() >= tile_count) {
        return weighted_error(
            WeightedGraphValidationErrorCode::owner_tile_out_of_range, vertex_index);
      }
    }
  }
  if (!std::in_range<std::size_t>(edge_count) ||
      graph.edge_provenance().size() != static_cast<std::size_t>(edge_count)) {
    return weighted_error(
        WeightedGraphValidationErrorCode::logical_provenance_size_mismatch);
  }

  const OutgoingCsrView outgoing = graph.outgoing();
  const IncomingCscView incoming = graph.incoming();
  if (!are_valid_offsets(outgoing.row_offsets, graph.vertex_count(), edge_count)) {
    return weighted_error(WeightedGraphValidationErrorCode::invalid_csr_offsets);
  }
  if (!are_valid_offsets(incoming.column_offsets, graph.vertex_count(), edge_count)) {
    return weighted_error(WeightedGraphValidationErrorCode::invalid_csc_offsets);
  }

  const std::size_t expected_edges = static_cast<std::size_t>(edge_count);
  if (outgoing.destinations.size() != expected_edges ||
      outgoing.weights.size() != expected_edges || outgoing.edge_ids.size() != expected_edges) {
    return weighted_error(WeightedGraphValidationErrorCode::csr_field_size_mismatch);
  }
  if (incoming.sources.size() != expected_edges || incoming.weights.size() != expected_edges ||
      incoming.edge_ids.size() != expected_edges) {
    return weighted_error(WeightedGraphValidationErrorCode::csc_field_size_mismatch);
  }

  std::vector<EdgeIdentity> csr_identity_by_id(expected_edges);
  std::vector<bool> csr_id_seen(expected_edges, false);
  for (std::size_t source_index = 0U; source_index < graph.vertex_count(); ++source_index) {
    const std::size_t row_begin =
        static_cast<std::size_t>(outgoing.row_offsets[source_index]);
    const std::size_t row_end =
        static_cast<std::size_t>(outgoing.row_offsets[source_index + 1U]);
    const VertexId source{static_cast<std::uint32_t>(source_index)};

    for (std::size_t position = row_begin; position < row_end; ++position) {
      const VertexId destination = outgoing.destinations[position];
      const float weight = outgoing.weights[position];
      const EdgeId edge_id = outgoing.edge_ids[position];
      if (!is_valid_vertex_id(destination, graph.vertex_count())) {
        return weighted_error(
            WeightedGraphValidationErrorCode::csr_destination_out_of_range, position);
      }
      if (!is_valid_weight(weight)) {
        return weighted_error(
            WeightedGraphValidationErrorCode::invalid_stored_weight, position);
      }
      if (weight == 0.0F && std::bit_cast<std::uint32_t>(weight) != 0U) {
        return weighted_error(
            WeightedGraphValidationErrorCode::noncanonical_stored_zero, position);
      }
      if (edge_id.value() >= edge_count) {
        return weighted_error(
            WeightedGraphValidationErrorCode::edge_id_out_of_range, position);
      }
      const std::size_t logical_index = static_cast<std::size_t>(edge_id.value());
      if (csr_id_seen[logical_index]) {
        return weighted_error(
            WeightedGraphValidationErrorCode::duplicate_csr_edge_id, position);
      }
      if (position > row_begin) {
        const VertexId previous_destination = outgoing.destinations[position - 1U];
        const EdgeId previous_edge_id = outgoing.edge_ids[position - 1U];
        const TileId destination_tile = graph.has_spatial_ordering()
                                            ? graph.owner_tiles()[destination.value()]
                                            : TileId{};
        const TileId previous_destination_tile =
            graph.has_spatial_ordering()
                ? graph.owner_tiles()[previous_destination.value()]
                : TileId{};
        if (destination_tile < previous_destination_tile ||
            (destination_tile == previous_destination_tile &&
             (destination < previous_destination ||
              (destination == previous_destination && edge_id < previous_edge_id)))) {
          return weighted_error(
              WeightedGraphValidationErrorCode::csr_row_order_violation, position);
        }
      }

      csr_id_seen[logical_index] = true;
      csr_identity_by_id[logical_index] =
          EdgeIdentity{source, destination, canonical_weight_bits(weight)};
    }
  }

  std::vector<bool> csc_id_seen(expected_edges, false);
  for (std::size_t destination_index = 0U;
       destination_index < graph.vertex_count();
       ++destination_index) {
    const std::size_t column_begin =
        static_cast<std::size_t>(incoming.column_offsets[destination_index]);
    const std::size_t column_end =
        static_cast<std::size_t>(incoming.column_offsets[destination_index + 1U]);
    const VertexId destination{static_cast<std::uint32_t>(destination_index)};

    for (std::size_t position = column_begin; position < column_end; ++position) {
      const VertexId source = incoming.sources[position];
      const float weight = incoming.weights[position];
      const EdgeId edge_id = incoming.edge_ids[position];
      if (!is_valid_vertex_id(source, graph.vertex_count())) {
        return weighted_error(
            WeightedGraphValidationErrorCode::csc_source_out_of_range, position);
      }
      if (!is_valid_weight(weight)) {
        return weighted_error(
            WeightedGraphValidationErrorCode::invalid_stored_weight, position);
      }
      if (weight == 0.0F && std::bit_cast<std::uint32_t>(weight) != 0U) {
        return weighted_error(
            WeightedGraphValidationErrorCode::noncanonical_stored_zero, position);
      }
      if (edge_id.value() >= edge_count) {
        return weighted_error(
            WeightedGraphValidationErrorCode::edge_id_out_of_range, position);
      }
      const std::size_t logical_index = static_cast<std::size_t>(edge_id.value());
      if (csc_id_seen[logical_index]) {
        return weighted_error(
            WeightedGraphValidationErrorCode::duplicate_csc_edge_id, position);
      }
      if (position > column_begin) {
        const VertexId previous_source = incoming.sources[position - 1U];
        const EdgeId previous_edge_id = incoming.edge_ids[position - 1U];
        const TileId source_tile =
            graph.has_spatial_ordering() ? graph.owner_tiles()[source.value()] : TileId{};
        const TileId previous_source_tile =
            graph.has_spatial_ordering()
                ? graph.owner_tiles()[previous_source.value()]
                : TileId{};
        if (source_tile < previous_source_tile ||
            (source_tile == previous_source_tile &&
             (source < previous_source ||
              (source == previous_source && edge_id < previous_edge_id)))) {
          return weighted_error(
              WeightedGraphValidationErrorCode::csc_column_order_violation, position);
        }
      }

      const EdgeIdentity csc_identity{
          source,
          destination,
          canonical_weight_bits(weight),
      };
      if (!(csr_identity_by_id[logical_index] == csc_identity)) {
        return weighted_error(
            WeightedGraphValidationErrorCode::transpose_mismatch, position);
      }
      csc_id_seen[logical_index] = true;
    }
  }

  for (std::size_t logical_index = 1U; logical_index < expected_edges; ++logical_index) {
    const EdgeIdentity& previous = csr_identity_by_id[logical_index - 1U];
    const EdgeIdentity& current = csr_identity_by_id[logical_index];
    const VertexId previous_original_source =
        graph.original_vertex_ids()[previous.source.value()];
    const VertexId previous_original_destination =
        graph.original_vertex_ids()[previous.destination.value()];
    const VertexId current_original_source =
        graph.original_vertex_ids()[current.source.value()];
    const VertexId current_original_destination =
        graph.original_vertex_ids()[current.destination.value()];
    const auto previous_key = canonical_logical_edge_key(
        previous_original_source,
        previous_original_destination,
        std::bit_cast<float>(previous.weight_bits),
        graph.edge_provenance()[logical_index - 1U]);
    const auto current_key = canonical_logical_edge_key(
        current_original_source,
        current_original_destination,
        std::bit_cast<float>(current.weight_bits),
        graph.edge_provenance()[logical_index]);
    if (current_key < previous_key) {
      return weighted_error(
          WeightedGraphValidationErrorCode::canonical_edge_id_order_violation,
          logical_index);
    }
  }

  return {};
}

}  // namespace bfnew
