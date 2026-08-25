#include "bfnew/selected_region_index.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace bfnew {
namespace {

[[nodiscard]] std::uint64_t checked_add(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* const description) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error{description};
  }
  return left + right;
}

void require_canonical_tiles(
    const std::span<const TileId> tiles,
    const std::size_t tile_count) {
  TileId previous{};
  bool has_previous = false;
  for (const TileId tile : tiles) {
    if (tile.value() >= tile_count) {
      throw std::out_of_range{"selected tile is outside the indexed graph"};
    }
    if (has_previous && !(previous < tile)) {
      throw std::invalid_argument{
          "selected tiles must be strictly increasing and unique"};
    }
    previous = tile;
    has_previous = true;
  }
}

}  // namespace

SelectedRegionIndex::SelectedRegionIndex(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs) {
  // The tile-run validator includes the graph deep validation. Keep that
  // expensive proof here, once, rather than repeating it for every query.
  if (!validate_tile_run_layout(graph, tile_runs).ok()) {
    throw std::invalid_argument{
        "selected-region index requires a valid graph and tile-run layout"};
  }
  graph_ = &graph;

  const std::size_t tile_count_value = graph.tile_coordinates().size();
  if (tile_count_value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{
        "selected-region index tile count exceeds the TileId representation"};
  }

  vertex_counts_.resize(tile_count_value);
  outgoing_.resize(tile_count_value);

  const std::span<const EdgeOffset> tile_offsets = graph.tile_vertex_offsets();
  for (std::size_t tile = 0U; tile < tile_count_value; ++tile) {
    vertex_counts_[tile] = tile_offsets[tile + 1U] - tile_offsets[tile];
  }

  const std::span<const TileId> owners = graph.owner_tiles();
  for (std::size_t row = 0U; row < graph.vertex_count(); ++row) {
    const std::size_t source_tile = owners[row].value();
    const std::size_t run_begin =
        static_cast<std::size_t>(tile_runs.csr_row_run_offsets[row]);
    const std::size_t run_end =
        static_cast<std::size_t>(tile_runs.csr_row_run_offsets[row + 1U]);
    auto& destination_counts = outgoing_[source_tile];
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const EdgeOffset edge_begin = tile_runs.csr_run_edge_offsets[run];
      const EdgeOffset edge_end = tile_runs.csr_run_edge_offsets[run + 1U];
      destination_counts.push_back(DestinationCount{
          tile_runs.csr_run_destination_tiles[run], edge_end - edge_begin});
    }
  }

  // Many vertex rows may contribute a run to the same ordered tile pair.
  // Coalescing once leaves every query estimate proportional to the selected
  // tile-pair index rather than to graph vertices or edges.
  for (auto& destination_counts : outgoing_) {
    std::sort(
        destination_counts.begin(),
        destination_counts.end(),
        [](const DestinationCount& left, const DestinationCount& right) {
          return left.destination < right.destination;
        });

    std::size_t write = 0U;
    for (const DestinationCount& value : destination_counts) {
      if (write != 0U &&
          destination_counts[write - 1U].destination == value.destination) {
        destination_counts[write - 1U].edges = checked_add(
            destination_counts[write - 1U].edges,
            value.edges,
            "tile-pair edge count overflow");
      } else {
        destination_counts[write] = value;
        ++write;
      }
    }
    destination_counts.resize(write);
    destination_counts.shrink_to_fit();
  }
}

std::uint64_t SelectedRegionIndex::selected_vertex_count(
    const std::span<const TileId> selected_tiles) const {
  require_canonical_tiles(selected_tiles, vertex_counts_.size());
  std::uint64_t result = 0U;
  for (const TileId tile : selected_tiles) {
    result = checked_add(
        result,
        vertex_counts_[tile.value()],
        "selected vertex count overflow");
  }
  return result;
}

std::uint64_t SelectedRegionIndex::selected_edge_count(
    const std::span<const TileId> selected_tiles) const {
  require_canonical_tiles(selected_tiles, outgoing_.size());
  std::uint64_t result = 0U;
  for (const TileId source : selected_tiles) {
    for (const DestinationCount& destination : outgoing_[source.value()]) {
      if (std::binary_search(
              selected_tiles.begin(),
              selected_tiles.end(),
              destination.destination)) {
        result = checked_add(
            result,
            destination.edges,
            "selected edge count overflow");
      }
    }
  }
  return result;
}

std::uint32_t SelectedRegionIndex::tile_count() const noexcept {
  return static_cast<std::uint32_t>(vertex_counts_.size());
}

bool SelectedRegionIndex::matches_graph(const WeightedGraph& graph) const noexcept {
  return graph_ == &graph;
}

}  // namespace bfnew
