#pragma once

#include "bfnew/types.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace bfnew {

class InputGraph;
class WeightedGraph;
struct PartitionedGraph;

struct SpatialOrderConfig {
  std::int32_t origin_x{};
  std::int32_t origin_y{};
  std::uint32_t tile_width{};
  std::uint32_t tile_height{};

  constexpr bool operator==(const SpatialOrderConfig&) const noexcept = default;
};

struct TileCoordinate {
  std::int64_t tile_x{};
  std::int64_t tile_y{};
  bool has_location{};

  [[nodiscard]] static constexpr TileCoordinate located(
      const std::int64_t x,
      const std::int64_t y) noexcept {
    return TileCoordinate{x, y, true};
  }

  [[nodiscard]] static constexpr TileCoordinate spill() noexcept {
    return TileCoordinate{0, 0, false};
  }

  constexpr bool operator==(const TileCoordinate&) const noexcept = default;
};

using LocalityKeyPolicy =
    std::function<std::uint64_t(std::uint32_t local_x, std::uint32_t local_y)>;

struct MortonLocalityPolicy {
  [[nodiscard]] std::uint64_t operator()(
      std::uint32_t local_x,
      std::uint32_t local_y) const noexcept;
};

[[nodiscard]] WeightedGraph build_spatially_ordered_graph(
    const InputGraph& input,
    const SpatialOrderConfig& config,
    const LocalityKeyPolicy& locality_policy = MortonLocalityPolicy{});

class TileDirectory {
 public:
  TileDirectory() = default;

  [[nodiscard]] std::size_t tile_count() const noexcept {
    return neighbor_tile_offsets_.empty() ? 0U : neighbor_tile_offsets_.size() - 1U;
  }

  [[nodiscard]] TileId spill_tile() const noexcept { return spill_tile_; }

  [[nodiscard]] std::span<const EdgeOffset> neighbor_tile_offsets() const noexcept {
    return neighbor_tile_offsets_;
  }

  [[nodiscard]] std::span<const TileId> neighbor_tiles() const noexcept {
    return neighbor_tiles_;
  }

  [[nodiscard]] std::span<const EdgeOffset> internal_edge_offsets() const noexcept {
    return internal_edge_offsets_;
  }

  [[nodiscard]] std::span<const EdgeId> internal_edge_ids() const noexcept {
    return internal_edge_ids_;
  }

  [[nodiscard]] std::span<const EdgeOffset> outgoing_cross_edge_offsets() const noexcept {
    return outgoing_cross_edge_offsets_;
  }

  [[nodiscard]] std::span<const EdgeId> outgoing_cross_edge_ids() const noexcept {
    return outgoing_cross_edge_ids_;
  }

  [[nodiscard]] std::span<const EdgeOffset> incoming_cross_edge_offsets() const noexcept {
    return incoming_cross_edge_offsets_;
  }

  [[nodiscard]] std::span<const EdgeId> incoming_cross_edge_ids() const noexcept {
    return incoming_cross_edge_ids_;
  }

  [[nodiscard]] std::span<const EdgeOffset> halo_vertex_offsets() const noexcept {
    return halo_vertex_offsets_;
  }

  [[nodiscard]] std::span<const VertexId> halo_vertices() const noexcept {
    return halo_vertices_;
  }

 private:
  TileId spill_tile_{};
  std::vector<EdgeOffset> neighbor_tile_offsets_;
  std::vector<TileId> neighbor_tiles_;
  std::vector<EdgeOffset> internal_edge_offsets_;
  std::vector<EdgeId> internal_edge_ids_;
  std::vector<EdgeOffset> outgoing_cross_edge_offsets_;
  std::vector<EdgeId> outgoing_cross_edge_ids_;
  std::vector<EdgeOffset> incoming_cross_edge_offsets_;
  std::vector<EdgeId> incoming_cross_edge_ids_;
  std::vector<EdgeOffset> halo_vertex_offsets_;
  std::vector<VertexId> halo_vertices_;

  friend TileDirectory build_tile_directory(const WeightedGraph& graph);
};

enum class TileDirectoryValidationErrorCode : std::uint8_t {
  none,
  graph_validation_failed,
  graph_is_not_spatially_ordered,
  tile_count_mismatch,
  spill_tile_mismatch,
  invalid_neighbor_offsets,
  invalid_internal_edge_offsets,
  invalid_outgoing_cross_edge_offsets,
  invalid_incoming_cross_edge_offsets,
  invalid_halo_offsets,
  neighbor_out_of_range,
  neighbor_order_violation,
  neighbor_asymmetry,
  unexpected_neighbor,
  edge_id_out_of_range,
  edge_classification_mismatch,
  duplicate_source_edge_classification,
  missing_source_edge_classification,
  duplicate_incoming_cross_edge,
  missing_incoming_cross_edge,
  edge_metadata_order_violation,
  halo_vertex_out_of_range,
  halo_owner_mismatch,
  halo_order_violation,
  halo_set_mismatch,
};

struct TileDirectoryValidationResult {
  static constexpr EdgeOffset no_position = std::numeric_limits<EdgeOffset>::max();

  TileDirectoryValidationErrorCode code{TileDirectoryValidationErrorCode::none};
  EdgeOffset position{no_position};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == TileDirectoryValidationErrorCode::none;
  }
};

[[nodiscard]] TileDirectory build_tile_directory(const WeightedGraph& graph);

[[nodiscard]] TileDirectoryValidationResult validate_tile_directory(
    const WeightedGraph& graph,
    const TileDirectory& directory);

class SpatialPartitioner {
 public:
  virtual ~SpatialPartitioner() = default;

  [[nodiscard]] virtual PartitionedGraph partition(const InputGraph& input) const = 0;
};

class UniformGridPartitioner final : public SpatialPartitioner {
 public:
  explicit UniformGridPartitioner(
      SpatialOrderConfig config,
      LocalityKeyPolicy locality_policy = MortonLocalityPolicy{});

  [[nodiscard]] const SpatialOrderConfig& config() const noexcept { return config_; }

  [[nodiscard]] PartitionedGraph partition(const InputGraph& input) const override;

 private:
  SpatialOrderConfig config_;
  LocalityKeyPolicy locality_policy_;
};

}  // namespace bfnew
