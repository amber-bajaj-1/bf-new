#pragma once

#include "bfnew/spatial.hpp"
#include "bfnew/types.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace bfnew {

static_assert(sizeof(float) == sizeof(std::uint32_t));
static_assert(std::numeric_limits<float>::is_iec559);

struct VertexMetadata {
  std::int32_t x{};
  std::int32_t y{};
  bool has_location{};
  ResourceClassId resource_class{};

  [[nodiscard]] static constexpr VertexMetadata located(
      std::int32_t x_value,
      std::int32_t y_value,
      ResourceClassId class_id) noexcept {
    return VertexMetadata{x_value, y_value, true, class_id};
  }

  [[nodiscard]] static constexpr VertexMetadata unlocated(
      ResourceClassId class_id) noexcept {
    return VertexMetadata{0, 0, false, class_id};
  }

  constexpr bool operator==(const VertexMetadata&) const noexcept = default;
};

struct EdgeInputRecord {
  VertexId source{};
  VertexId destination{};
  float weight{};
  PhysicalProvenance provenance{};

  constexpr bool operator==(const EdgeInputRecord&) const noexcept = default;
};

enum class GraphValidationErrorCode : std::uint8_t {
  none,
  vertex_count_overflow,
  edge_count_overflow,
  nonfinite_weight,
  negative_weight,
  source_out_of_range,
  destination_out_of_range,
};

struct GraphValidationResult {
  static constexpr std::size_t no_edge = std::numeric_limits<std::size_t>::max();

  GraphValidationErrorCode code{GraphValidationErrorCode::none};
  std::size_t edge_index{no_edge};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == GraphValidationErrorCode::none;
  }
};

struct OutgoingCsrView {
  std::span<const EdgeOffset> row_offsets;
  std::span<const VertexId> destinations;
  std::span<const float> weights;
  std::span<const EdgeId> edge_ids;
};

struct IncomingCscView {
  std::span<const EdgeOffset> column_offsets;
  std::span<const VertexId> sources;
  std::span<const float> weights;
  std::span<const EdgeId> edge_ids;
};

enum class WeightedGraphValidationErrorCode : std::uint8_t {
  none,
  vertex_metadata_size_mismatch,
  logical_provenance_size_mismatch,
  invalid_csr_offsets,
  invalid_csc_offsets,
  csr_field_size_mismatch,
  csc_field_size_mismatch,
  csr_destination_out_of_range,
  csc_source_out_of_range,
  invalid_stored_weight,
  noncanonical_stored_zero,
  edge_id_out_of_range,
  duplicate_csr_edge_id,
  duplicate_csc_edge_id,
  csr_row_order_violation,
  csc_column_order_violation,
  transpose_mismatch,
  canonical_edge_id_order_violation,
  permutation_size_mismatch,
  permutation_out_of_range,
  permutation_round_trip_mismatch,
  original_vertex_id_mismatch,
  incomplete_spatial_metadata,
  invalid_tile_coordinate_table,
  invalid_tile_vertex_offsets,
  owner_tile_out_of_range,
  noncontiguous_tile_span,
};

struct WeightedGraphValidationResult {
  static constexpr EdgeOffset no_position = std::numeric_limits<EdgeOffset>::max();

  WeightedGraphValidationErrorCode code{WeightedGraphValidationErrorCode::none};
  EdgeOffset position{no_position};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == WeightedGraphValidationErrorCode::none;
  }
};

[[nodiscard]] float canonicalize_weight(float weight) noexcept;
[[nodiscard]] bool is_valid_weight(float weight) noexcept;
[[nodiscard]] std::uint32_t canonical_weight_bits(float weight) noexcept;

[[nodiscard]] bool is_valid_vertex_id(VertexId id, std::uint32_t vertex_count) noexcept;

[[nodiscard]] GraphValidationResult validate_graph_input(
    std::size_t vertex_count,
    std::span<const EdgeInputRecord> edges) noexcept;

[[nodiscard]] bool are_valid_offsets(
    std::span<const EdgeOffset> offsets,
    std::uint32_t bucket_count,
    EdgeCount entry_count) noexcept;

class InputGraph {
 public:
  InputGraph(std::vector<VertexMetadata> vertices, std::vector<EdgeInputRecord> edges);

  [[nodiscard]] std::uint32_t vertex_count() const noexcept { return vertex_count_; }
  [[nodiscard]] std::uint64_t edge_count() const noexcept { return edge_count_; }

  [[nodiscard]] std::span<const VertexMetadata> vertices() const noexcept {
    return vertices_;
  }

  [[nodiscard]] std::span<const EdgeInputRecord> edges() const noexcept { return edges_; }

 private:
  std::uint32_t vertex_count_{};
  std::uint64_t edge_count_{};
  std::vector<VertexMetadata> vertices_;
  std::vector<EdgeInputRecord> edges_;
};

class WeightedGraph {
 public:
  [[nodiscard]] std::uint32_t vertex_count() const noexcept { return vertex_count_; }
  [[nodiscard]] EdgeCount edge_count() const noexcept { return edge_count_; }

  [[nodiscard]] std::span<const VertexMetadata> vertices() const noexcept {
    return vertices_;
  }

  [[nodiscard]] std::span<const PhysicalProvenance> edge_provenance() const noexcept {
    return edge_provenance_;
  }

  [[nodiscard]] OutgoingCsrView outgoing() const noexcept {
    return OutgoingCsrView{
        outgoing_row_offsets_,
        outgoing_destinations_,
        outgoing_weights_,
        outgoing_edge_ids_,
    };
  }

  [[nodiscard]] IncomingCscView incoming() const noexcept {
    return IncomingCscView{
        incoming_column_offsets_,
        incoming_sources_,
        incoming_weights_,
        incoming_edge_ids_,
    };
  }

  [[nodiscard]] std::span<const VertexId> old_to_new() const noexcept {
    return old_to_new_;
  }

  [[nodiscard]] std::span<const VertexId> new_to_old() const noexcept {
    return new_to_old_;
  }

  [[nodiscard]] std::span<const VertexId> original_vertex_ids() const noexcept {
    return original_vertex_ids_;
  }

  [[nodiscard]] bool has_spatial_ordering() const noexcept {
    return has_spatial_ordering_;
  }

  [[nodiscard]] std::span<const TileId> owner_tiles() const noexcept {
    return owner_tiles_;
  }

  [[nodiscard]] std::span<const TileCoordinate> tile_coordinates() const noexcept {
    return tile_coordinates_;
  }

  [[nodiscard]] std::span<const EdgeOffset> tile_vertex_offsets() const noexcept {
    return tile_vertex_offsets_;
  }

  [[nodiscard]] const SpatialOrderConfig& spatial_order_config() const noexcept {
    return spatial_order_config_;
  }

 private:
  WeightedGraph() = default;

  std::uint32_t vertex_count_{};
  EdgeCount edge_count_{};
  std::vector<VertexMetadata> vertices_;
  std::vector<PhysicalProvenance> edge_provenance_;

  std::vector<EdgeOffset> outgoing_row_offsets_;
  std::vector<VertexId> outgoing_destinations_;
  std::vector<float> outgoing_weights_;
  std::vector<EdgeId> outgoing_edge_ids_;

  std::vector<EdgeOffset> incoming_column_offsets_;
  std::vector<VertexId> incoming_sources_;
  std::vector<float> incoming_weights_;
  std::vector<EdgeId> incoming_edge_ids_;

  std::vector<VertexId> old_to_new_;
  std::vector<VertexId> new_to_old_;
  std::vector<VertexId> original_vertex_ids_;

  bool has_spatial_ordering_{};
  SpatialOrderConfig spatial_order_config_{};
  std::vector<TileId> owner_tiles_;
  std::vector<TileCoordinate> tile_coordinates_;
  std::vector<EdgeOffset> tile_vertex_offsets_;

  friend WeightedGraph build_weighted_graph(const InputGraph& input);
  friend WeightedGraph build_spatially_ordered_graph(
      const InputGraph& input,
      const SpatialOrderConfig& config,
      const LocalityKeyPolicy& locality_policy);
};

struct PartitionedGraph {
  WeightedGraph graph;
  TileDirectory tiles;
};

[[nodiscard]] WeightedGraph build_weighted_graph(const InputGraph& input);

[[nodiscard]] WeightedGraphValidationResult validate_weighted_graph(
    const WeightedGraph& graph);

}  // namespace bfnew
