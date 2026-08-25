#pragma once

#include "bfnew/device_layout.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace bfnew {

// One immutable O(tile-run-count) index supports exact selected-region and
// union estimates without rescanning every graph vertex/edge for every query.
class SelectedRegionIndex final {
 public:
  SelectedRegionIndex(
      const WeightedGraph& graph,
      const TileRunLayout64& tile_runs);

  [[nodiscard]] std::uint64_t selected_vertex_count(
      std::span<const TileId> selected_tiles) const;
  [[nodiscard]] std::uint64_t selected_edge_count(
      std::span<const TileId> selected_tiles) const;
  [[nodiscard]] std::uint32_t tile_count() const noexcept;
  [[nodiscard]] bool matches_graph(const WeightedGraph& graph) const noexcept;

 private:
  struct DestinationCount {
    TileId destination{};
    std::uint64_t edges{};

    constexpr bool operator==(const DestinationCount&) const noexcept = default;
  };

  const WeightedGraph* graph_{};
  std::vector<std::uint64_t> vertex_counts_;
  std::vector<std::vector<DestinationCount>> outgoing_;
};

}  // namespace bfnew
