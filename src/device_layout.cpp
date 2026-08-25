#include "bfnew/device_layout.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>

namespace bfnew {
namespace {

[[nodiscard]] TileRunValidationResult tile_run_error(
    const TileRunValidationErrorCode code,
    const EdgeOffset position = TileRunValidationResult::no_position) noexcept {
  return TileRunValidationResult{code, position};
}

[[nodiscard]] DeviceGraphLayoutValidationResult device_layout_error(
    const DeviceGraphLayoutValidationErrorCode code,
    const EdgeOffset position = DeviceGraphLayoutValidationResult::no_position) noexcept {
  return DeviceGraphLayoutValidationResult{code, position};
}

[[nodiscard]] RunAdmissionProofResult admission_error(
    const RunAdmissionProofErrorCode code,
    const EdgeOffset position = RunAdmissionProofResult::no_position) noexcept {
  return RunAdmissionProofResult{code, position};
}

[[nodiscard]] constexpr EdgeOffset as_edge_offset(const std::size_t value) noexcept {
  return static_cast<EdgeOffset>(value);
}

[[nodiscard]] bool valid_offsets(
    const std::span<const EdgeOffset> offsets,
    const std::size_t bucket_count,
    const EdgeOffset entry_count) noexcept {
  if (offsets.size() != bucket_count + 1U || offsets.empty() ||
      offsets.front() != 0U || offsets.back() != entry_count) {
    return false;
  }
  for (std::size_t index = 1U; index < offsets.size(); ++index) {
    if (offsets[index] < offsets[index - 1U] || offsets[index] > entry_count) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool graph_is_device_representable(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs) noexcept {
  constexpr EdgeOffset maximum = std::numeric_limits<std::uint32_t>::max();
  return graph.vertex_count() <= maximum && graph.edge_count() <= maximum &&
         graph.tile_coordinates().size() <= maximum &&
         tile_runs.csr_run_destination_tiles.size() <= maximum &&
         tile_runs.csc_run_source_tiles.size() <= maximum;
}

template <typename T>
[[nodiscard]] std::uint64_t vector_bytes(const std::vector<T>& values) noexcept {
  return static_cast<std::uint64_t>(values.size()) * static_cast<std::uint64_t>(sizeof(T));
}

class FingerprintBuilder final {
 public:
  void begin_component(
      const std::uint32_t tag,
      const std::uint64_t element_count) noexcept {
    append_u32(tag);
    append_u64(element_count);
  }

  void append_u32(const std::uint32_t value) noexcept {
    for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
      append_byte(static_cast<std::uint8_t>(value >> shift));
    }
  }

  [[nodiscard]] DeviceGraphFingerprint finish() const noexcept {
    return DeviceGraphFingerprint{first_, second_};
  }

 private:
  void append_u64(const std::uint64_t value) noexcept {
    for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
      append_byte(static_cast<std::uint8_t>(value >> shift));
    }
  }

  void append_byte(const std::uint8_t value) noexcept {
    first_ = (first_ ^ value) * 1099511628211ULL;
    second_ ^= static_cast<std::uint64_t>(value) + 0x9E3779B97F4A7C15ULL +
               (second_ << 6U) + (second_ >> 2U);
    second_ *= 0xD6E8FEB86659FD93ULL;
  }

  std::uint64_t first_{1469598103934665603ULL};
  std::uint64_t second_{0xA0761D6478BD642FULL};
};

template <typename Range, typename Convert>
void fingerprint_component(
    FingerprintBuilder& builder,
    const std::uint32_t tag,
    const Range& values,
    Convert&& convert) noexcept {
  builder.begin_component(tag, static_cast<std::uint64_t>(values.size()));
  for (const auto& value : values) {
    builder.append_u32(convert(value));
  }
}

void fingerprint_scalars(
    FingerprintBuilder& builder,
    const std::uint32_t vertex_count,
    const std::uint32_t edge_count,
    const std::uint32_t tile_count) noexcept {
  builder.begin_component(0U, 3U);
  builder.append_u32(vertex_count);
  builder.append_u32(edge_count);
  builder.append_u32(tile_count);
}

[[nodiscard]] bool equal_float_bits(
    const std::span<const float> left,
    const std::span<const float> right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (std::bit_cast<std::uint32_t>(left[index]) !=
        std::bit_cast<std::uint32_t>(right[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool lane_mask_shape_is_valid(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const std::span<const std::uint32_t> tile_lane_masks) noexcept {
  const std::size_t vertex_count = graph.vertex_count();
  const std::size_t tile_count = graph.tile_coordinates().size();
  const std::size_t csr_run_count = tile_runs.csr_run_destination_tiles.size();
  const std::size_t csc_run_count = tile_runs.csc_run_source_tiles.size();

  if (!graph.has_spatial_ordering() || graph.owner_tiles().size() != vertex_count ||
      tile_lane_masks.size() != tile_count ||
      tile_runs.csr_row_run_offsets.size() != vertex_count + 1U ||
      tile_runs.csc_column_run_offsets.size() != vertex_count + 1U ||
      tile_runs.csr_row_run_offsets.empty() ||
      tile_runs.csc_column_run_offsets.empty() ||
      tile_runs.csr_row_run_offsets.front() != 0U ||
      tile_runs.csc_column_run_offsets.front() != 0U ||
      tile_runs.csr_row_run_offsets.back() != as_edge_offset(csr_run_count) ||
      tile_runs.csc_column_run_offsets.back() != as_edge_offset(csc_run_count)) {
    return false;
  }

  for (std::size_t vertex = 0U; vertex < vertex_count; ++vertex) {
    if (graph.owner_tiles()[vertex].value() >= tile_count ||
        tile_runs.csr_row_run_offsets[vertex] >
            tile_runs.csr_row_run_offsets[vertex + 1U] ||
        tile_runs.csr_row_run_offsets[vertex + 1U] > as_edge_offset(csr_run_count) ||
        tile_runs.csc_column_run_offsets[vertex] >
            tile_runs.csc_column_run_offsets[vertex + 1U] ||
        tile_runs.csc_column_run_offsets[vertex + 1U] > as_edge_offset(csc_run_count)) {
      return false;
    }
  }
  return std::ranges::all_of(
             tile_runs.csr_run_destination_tiles,
             [tile_count](const TileId tile) { return tile.value() < tile_count; }) &&
         std::ranges::all_of(
             tile_runs.csc_run_source_tiles,
             [tile_count](const TileId tile) { return tile.value() < tile_count; });
}

}  // namespace

TileRunLayout64 build_tile_run_layout(const WeightedGraph& graph) {
  if (!validate_weighted_graph(graph).ok()) {
    throw std::invalid_argument{"tile runs require a deeply valid weighted graph"};
  }
  if (!graph.has_spatial_ordering()) {
    throw std::invalid_argument{"tile runs require a spatially ordered graph"};
  }

  TileRunLayout64 layout;
  const std::size_t vertex_count = graph.vertex_count();
  const OutgoingCsrView outgoing = graph.outgoing();
  const IncomingCscView incoming = graph.incoming();
  const std::span<const TileId> owner_tiles = graph.owner_tiles();

  layout.csr_row_run_offsets.reserve(vertex_count + 1U);
  layout.csr_run_edge_offsets.push_back(0U);
  layout.csr_row_run_offsets.push_back(0U);
  for (std::size_t row = 0U; row < vertex_count; ++row) {
    std::size_t position = static_cast<std::size_t>(outgoing.row_offsets[row]);
    const std::size_t row_end =
        static_cast<std::size_t>(outgoing.row_offsets[row + 1U]);
    while (position < row_end) {
      const TileId destination_tile =
          owner_tiles[outgoing.destinations[position].value()];
      ++position;
      while (position < row_end &&
             owner_tiles[outgoing.destinations[position].value()] == destination_tile) {
        ++position;
      }
      layout.csr_run_destination_tiles.push_back(destination_tile);
      layout.csr_run_edge_offsets.push_back(as_edge_offset(position));
    }
    layout.csr_row_run_offsets.push_back(
        as_edge_offset(layout.csr_run_destination_tiles.size()));
  }

  layout.csc_column_run_offsets.reserve(vertex_count + 1U);
  layout.csc_run_edge_offsets.push_back(0U);
  layout.csc_column_run_offsets.push_back(0U);
  for (std::size_t column = 0U; column < vertex_count; ++column) {
    std::size_t position = static_cast<std::size_t>(incoming.column_offsets[column]);
    const std::size_t column_end =
        static_cast<std::size_t>(incoming.column_offsets[column + 1U]);
    while (position < column_end) {
      const TileId source_tile = owner_tiles[incoming.sources[position].value()];
      ++position;
      while (position < column_end &&
             owner_tiles[incoming.sources[position].value()] == source_tile) {
        ++position;
      }
      layout.csc_run_source_tiles.push_back(source_tile);
      layout.csc_run_edge_offsets.push_back(as_edge_offset(position));
    }
    layout.csc_column_run_offsets.push_back(
        as_edge_offset(layout.csc_run_source_tiles.size()));
  }

  const TileRunValidationResult validation = validate_tile_run_layout(graph, layout);
  if (!validation.ok()) {
    throw std::logic_error{
        "constructed tile-run layout failed deep validation with code " +
        std::to_string(static_cast<unsigned int>(validation.code)) + " at position " +
        std::to_string(validation.position)};
  }
  return layout;
}

TileRunValidationResult validate_tile_run_layout(
    const WeightedGraph& graph,
    const TileRunLayout64& layout) {
  if (!validate_weighted_graph(graph).ok()) {
    return tile_run_error(TileRunValidationErrorCode::graph_validation_failed);
  }
  if (!graph.has_spatial_ordering()) {
    return tile_run_error(TileRunValidationErrorCode::graph_is_not_spatially_ordered);
  }

  const std::size_t vertex_count = graph.vertex_count();
  const EdgeOffset edge_count = graph.edge_count();
  const std::size_t tile_count = graph.tile_coordinates().size();
  const std::size_t csr_run_count = layout.csr_run_destination_tiles.size();
  if (!valid_offsets(
          layout.csr_row_run_offsets, vertex_count, as_edge_offset(csr_run_count))) {
    return tile_run_error(TileRunValidationErrorCode::invalid_csr_row_run_offsets);
  }
  if (layout.csr_run_edge_offsets.size() != csr_run_count + 1U) {
    return tile_run_error(TileRunValidationErrorCode::csr_run_tile_count_mismatch);
  }
  if (!valid_offsets(layout.csr_run_edge_offsets, csr_run_count, edge_count)) {
    return tile_run_error(TileRunValidationErrorCode::invalid_csr_run_edge_offsets);
  }

  const OutgoingCsrView outgoing = graph.outgoing();
  const std::span<const TileId> owner_tiles = graph.owner_tiles();
  for (std::size_t row = 0U; row < vertex_count; ++row) {
    const EdgeOffset row_edge_begin = outgoing.row_offsets[row];
    const EdgeOffset row_edge_end = outgoing.row_offsets[row + 1U];
    const std::size_t row_run_begin =
        static_cast<std::size_t>(layout.csr_row_run_offsets[row]);
    const std::size_t row_run_end =
        static_cast<std::size_t>(layout.csr_row_run_offsets[row + 1U]);
    EdgeOffset cursor = row_edge_begin;
    TileId preceding_tile{};
    bool has_preceding_tile = false;
    for (std::size_t run = row_run_begin; run < row_run_end; ++run) {
      const EdgeOffset run_begin = layout.csr_run_edge_offsets[run];
      const EdgeOffset run_end = layout.csr_run_edge_offsets[run + 1U];
      if (run_begin != cursor) {
        return tile_run_error(
            TileRunValidationErrorCode::csr_edge_coverage_mismatch, run_begin);
      }
      if (run_end <= run_begin) {
        return tile_run_error(TileRunValidationErrorCode::csr_empty_run, run);
      }
      if (run_begin < row_edge_begin || run_end > row_edge_end) {
        return tile_run_error(TileRunValidationErrorCode::csr_run_crosses_row, run);
      }
      const TileId run_tile = layout.csr_run_destination_tiles[run];
      if (run_tile.value() >= tile_count) {
        return tile_run_error(TileRunValidationErrorCode::csr_run_tile_out_of_range, run);
      }
      if (has_preceding_tile && preceding_tile == run_tile) {
        return tile_run_error(TileRunValidationErrorCode::csr_nonmaximal_runs, run);
      }
      for (EdgeOffset edge = run_begin; edge < run_end; ++edge) {
        const std::size_t position = static_cast<std::size_t>(edge);
        if (owner_tiles[outgoing.destinations[position].value()] != run_tile) {
          return tile_run_error(TileRunValidationErrorCode::csr_run_tile_mismatch, edge);
        }
      }
      preceding_tile = run_tile;
      has_preceding_tile = true;
      cursor = run_end;
    }
    if (cursor != row_edge_end) {
      return tile_run_error(
          TileRunValidationErrorCode::csr_edge_coverage_mismatch, cursor);
    }
  }

  const std::size_t csc_run_count = layout.csc_run_source_tiles.size();
  if (!valid_offsets(
          layout.csc_column_run_offsets, vertex_count, as_edge_offset(csc_run_count))) {
    return tile_run_error(TileRunValidationErrorCode::invalid_csc_column_run_offsets);
  }
  if (layout.csc_run_edge_offsets.size() != csc_run_count + 1U) {
    return tile_run_error(TileRunValidationErrorCode::csc_run_tile_count_mismatch);
  }
  if (!valid_offsets(layout.csc_run_edge_offsets, csc_run_count, edge_count)) {
    return tile_run_error(TileRunValidationErrorCode::invalid_csc_run_edge_offsets);
  }

  const IncomingCscView incoming = graph.incoming();
  for (std::size_t column = 0U; column < vertex_count; ++column) {
    const EdgeOffset column_edge_begin = incoming.column_offsets[column];
    const EdgeOffset column_edge_end = incoming.column_offsets[column + 1U];
    const std::size_t column_run_begin =
        static_cast<std::size_t>(layout.csc_column_run_offsets[column]);
    const std::size_t column_run_end =
        static_cast<std::size_t>(layout.csc_column_run_offsets[column + 1U]);
    EdgeOffset cursor = column_edge_begin;
    TileId preceding_tile{};
    bool has_preceding_tile = false;
    for (std::size_t run = column_run_begin; run < column_run_end; ++run) {
      const EdgeOffset run_begin = layout.csc_run_edge_offsets[run];
      const EdgeOffset run_end = layout.csc_run_edge_offsets[run + 1U];
      if (run_begin != cursor) {
        return tile_run_error(
            TileRunValidationErrorCode::csc_edge_coverage_mismatch, run_begin);
      }
      if (run_end <= run_begin) {
        return tile_run_error(TileRunValidationErrorCode::csc_empty_run, run);
      }
      if (run_begin < column_edge_begin || run_end > column_edge_end) {
        return tile_run_error(TileRunValidationErrorCode::csc_run_crosses_column, run);
      }
      const TileId run_tile = layout.csc_run_source_tiles[run];
      if (run_tile.value() >= tile_count) {
        return tile_run_error(TileRunValidationErrorCode::csc_run_tile_out_of_range, run);
      }
      if (has_preceding_tile && preceding_tile == run_tile) {
        return tile_run_error(TileRunValidationErrorCode::csc_nonmaximal_runs, run);
      }
      for (EdgeOffset edge = run_begin; edge < run_end; ++edge) {
        const std::size_t position = static_cast<std::size_t>(edge);
        if (owner_tiles[incoming.sources[position].value()] != run_tile) {
          return tile_run_error(TileRunValidationErrorCode::csc_run_tile_mismatch, edge);
        }
      }
      preceding_tile = run_tile;
      has_preceding_tile = true;
      cursor = run_end;
    }
    if (cursor != column_edge_end) {
      return tile_run_error(
          TileRunValidationErrorCode::csc_edge_coverage_mismatch, cursor);
    }
  }

  return {};
}

std::uint32_t checked_device_offset32(const EdgeOffset offset) {
  if (offset > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{"64-bit graph offset does not fit the device layout"};
  }
  return static_cast<std::uint32_t>(offset);
}

DeviceGraphLayout32 build_device_graph_layout32(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs) {
  const TileRunValidationResult run_validation =
      validate_tile_run_layout(graph, tile_runs);
  if (!run_validation.ok()) {
    throw std::invalid_argument{"device layout requires valid tile-run metadata"};
  }
  if (!graph_is_device_representable(graph, tile_runs)) {
    throw std::overflow_error{"graph counts do not fit the 32-bit device layout"};
  }

  DeviceGraphLayout32 layout;
  layout.vertex_count = checked_device_offset32(graph.vertex_count());
  layout.edge_count = checked_device_offset32(graph.edge_count());
  layout.tile_count = checked_device_offset32(graph.tile_coordinates().size());

  layout.owner_tiles.reserve(graph.owner_tiles().size());
  for (const TileId tile : graph.owner_tiles()) {
    layout.owner_tiles.push_back(tile.value());
  }

  const OutgoingCsrView outgoing = graph.outgoing();
  layout.csr_row_offsets.reserve(outgoing.row_offsets.size());
  for (const EdgeOffset offset : outgoing.row_offsets) {
    layout.csr_row_offsets.push_back(checked_device_offset32(offset));
  }
  layout.csr_destinations.reserve(outgoing.destinations.size());
  for (const VertexId destination : outgoing.destinations) {
    layout.csr_destinations.push_back(destination.value());
  }
  layout.csr_weights.assign(outgoing.weights.begin(), outgoing.weights.end());
  layout.csr_row_run_offsets.reserve(tile_runs.csr_row_run_offsets.size());
  for (const EdgeOffset offset : tile_runs.csr_row_run_offsets) {
    layout.csr_row_run_offsets.push_back(checked_device_offset32(offset));
  }
  layout.csr_run_edge_offsets.reserve(tile_runs.csr_run_edge_offsets.size());
  for (const EdgeOffset offset : tile_runs.csr_run_edge_offsets) {
    layout.csr_run_edge_offsets.push_back(checked_device_offset32(offset));
  }
  layout.csr_run_destination_tiles.reserve(
      tile_runs.csr_run_destination_tiles.size());
  for (const TileId tile : tile_runs.csr_run_destination_tiles) {
    layout.csr_run_destination_tiles.push_back(tile.value());
  }

  const IncomingCscView incoming = graph.incoming();
  layout.csc_column_offsets.reserve(incoming.column_offsets.size());
  for (const EdgeOffset offset : incoming.column_offsets) {
    layout.csc_column_offsets.push_back(checked_device_offset32(offset));
  }
  layout.csc_sources.reserve(incoming.sources.size());
  for (const VertexId source : incoming.sources) {
    layout.csc_sources.push_back(source.value());
  }
  layout.csc_weights.assign(incoming.weights.begin(), incoming.weights.end());
  layout.csc_edge_ids.reserve(incoming.edge_ids.size());
  for (const EdgeId edge_id : incoming.edge_ids) {
    layout.csc_edge_ids.push_back(checked_device_offset32(edge_id.value()));
  }
  layout.csc_column_run_offsets.reserve(tile_runs.csc_column_run_offsets.size());
  for (const EdgeOffset offset : tile_runs.csc_column_run_offsets) {
    layout.csc_column_run_offsets.push_back(checked_device_offset32(offset));
  }
  layout.csc_run_edge_offsets.reserve(tile_runs.csc_run_edge_offsets.size());
  for (const EdgeOffset offset : tile_runs.csc_run_edge_offsets) {
    layout.csc_run_edge_offsets.push_back(checked_device_offset32(offset));
  }
  layout.csc_run_source_tiles.reserve(tile_runs.csc_run_source_tiles.size());
  for (const TileId tile : tile_runs.csc_run_source_tiles) {
    layout.csc_run_source_tiles.push_back(tile.value());
  }

  const DeviceGraphLayoutValidationResult validation =
      validate_device_graph_layout32(graph, tile_runs, layout);
  if (!validation.ok()) {
    throw std::logic_error{
        "constructed device graph layout failed deep validation with code " +
        std::to_string(static_cast<unsigned int>(validation.code)) + " at position " +
        std::to_string(validation.position)};
  }
  return layout;
}

DeviceGraphLayoutValidationResult validate_device_graph_layout32(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const DeviceGraphLayout32& layout) {
  if (!validate_tile_run_layout(graph, tile_runs).ok()) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::tile_run_layout_invalid);
  }
  if (!graph_is_device_representable(graph, tile_runs)) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::graph_not_representable);
  }
  if (layout.vertex_count != graph.vertex_count() ||
      layout.edge_count != static_cast<std::uint32_t>(graph.edge_count()) ||
      layout.tile_count != static_cast<std::uint32_t>(graph.tile_coordinates().size())) {
    return device_layout_error(DeviceGraphLayoutValidationErrorCode::scalar_mismatch);
  }

  const auto compare_u32 = [](
                               const std::span<const std::uint32_t> actual,
                               const auto& expected,
                               const auto convert) -> EdgeOffset {
    if (actual.size() != expected.size()) {
      return DeviceGraphLayoutValidationResult::no_position;
    }
    for (std::size_t index = 0U; index < actual.size(); ++index) {
      if (actual[index] != convert(expected[index])) {
        return as_edge_offset(index);
      }
    }
    return as_edge_offset(actual.size());
  };

  const EdgeOffset owner_match = compare_u32(
      layout.owner_tiles, graph.owner_tiles(), [](const TileId tile) {
        return tile.value();
      });
  if (owner_match != as_edge_offset(layout.owner_tiles.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::owner_tiles_mismatch, owner_match);
  }

  const OutgoingCsrView outgoing = graph.outgoing();
  const EdgeOffset csr_row_match = compare_u32(
      layout.csr_row_offsets, outgoing.row_offsets, [](const EdgeOffset offset) {
        return static_cast<std::uint32_t>(offset);
      });
  if (csr_row_match != as_edge_offset(layout.csr_row_offsets.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csr_row_offsets_mismatch,
        csr_row_match);
  }
  const EdgeOffset csr_destination_match = compare_u32(
      layout.csr_destinations, outgoing.destinations, [](const VertexId vertex) {
        return vertex.value();
      });
  if (csr_destination_match != as_edge_offset(layout.csr_destinations.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csr_destinations_mismatch,
        csr_destination_match);
  }
  if (!equal_float_bits(layout.csr_weights, outgoing.weights)) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csr_weights_mismatch);
  }
  const EdgeOffset csr_row_run_match = compare_u32(
      layout.csr_row_run_offsets,
      tile_runs.csr_row_run_offsets,
      [](const EdgeOffset offset) { return static_cast<std::uint32_t>(offset); });
  if (csr_row_run_match != as_edge_offset(layout.csr_row_run_offsets.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csr_row_run_offsets_mismatch,
        csr_row_run_match);
  }
  const EdgeOffset csr_run_edge_match = compare_u32(
      layout.csr_run_edge_offsets,
      tile_runs.csr_run_edge_offsets,
      [](const EdgeOffset offset) { return static_cast<std::uint32_t>(offset); });
  if (csr_run_edge_match != as_edge_offset(layout.csr_run_edge_offsets.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csr_run_edge_offsets_mismatch,
        csr_run_edge_match);
  }
  const EdgeOffset csr_run_tile_match = compare_u32(
      layout.csr_run_destination_tiles,
      tile_runs.csr_run_destination_tiles,
      [](const TileId tile) { return tile.value(); });
  if (csr_run_tile_match != as_edge_offset(layout.csr_run_destination_tiles.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csr_run_destination_tiles_mismatch,
        csr_run_tile_match);
  }

  const IncomingCscView incoming = graph.incoming();
  const EdgeOffset csc_column_match = compare_u32(
      layout.csc_column_offsets,
      incoming.column_offsets,
      [](const EdgeOffset offset) { return static_cast<std::uint32_t>(offset); });
  if (csc_column_match != as_edge_offset(layout.csc_column_offsets.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csc_column_offsets_mismatch,
        csc_column_match);
  }
  const EdgeOffset csc_source_match = compare_u32(
      layout.csc_sources, incoming.sources, [](const VertexId vertex) {
        return vertex.value();
      });
  if (csc_source_match != as_edge_offset(layout.csc_sources.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csc_sources_mismatch, csc_source_match);
  }
  if (!equal_float_bits(layout.csc_weights, incoming.weights)) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csc_weights_mismatch);
  }
  const EdgeOffset csc_edge_id_match = compare_u32(
      layout.csc_edge_ids, incoming.edge_ids, [](const EdgeId edge_id) {
        return static_cast<std::uint32_t>(edge_id.value());
      });
  if (csc_edge_id_match != as_edge_offset(layout.csc_edge_ids.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csc_edge_ids_mismatch,
        csc_edge_id_match);
  }
  const EdgeOffset csc_column_run_match = compare_u32(
      layout.csc_column_run_offsets,
      tile_runs.csc_column_run_offsets,
      [](const EdgeOffset offset) { return static_cast<std::uint32_t>(offset); });
  if (csc_column_run_match != as_edge_offset(layout.csc_column_run_offsets.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csc_column_run_offsets_mismatch,
        csc_column_run_match);
  }
  const EdgeOffset csc_run_edge_match = compare_u32(
      layout.csc_run_edge_offsets,
      tile_runs.csc_run_edge_offsets,
      [](const EdgeOffset offset) { return static_cast<std::uint32_t>(offset); });
  if (csc_run_edge_match != as_edge_offset(layout.csc_run_edge_offsets.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csc_run_edge_offsets_mismatch,
        csc_run_edge_match);
  }
  const EdgeOffset csc_run_tile_match = compare_u32(
      layout.csc_run_source_tiles,
      tile_runs.csc_run_source_tiles,
      [](const TileId tile) { return tile.value(); });
  if (csc_run_tile_match != as_edge_offset(layout.csc_run_source_tiles.size())) {
    return device_layout_error(
        DeviceGraphLayoutValidationErrorCode::csc_run_source_tiles_mismatch,
        csc_run_tile_match);
  }
  return {};
}

bool device_graph_layouts_deep_equal(
    const DeviceGraphLayout32& left,
    const DeviceGraphLayout32& right) {
  return left.vertex_count == right.vertex_count && left.edge_count == right.edge_count &&
         left.tile_count == right.tile_count && left.owner_tiles == right.owner_tiles &&
         left.csr_row_offsets == right.csr_row_offsets &&
         left.csr_destinations == right.csr_destinations &&
         equal_float_bits(left.csr_weights, right.csr_weights) &&
         left.csr_row_run_offsets == right.csr_row_run_offsets &&
         left.csr_run_edge_offsets == right.csr_run_edge_offsets &&
         left.csr_run_destination_tiles == right.csr_run_destination_tiles &&
         left.csc_column_offsets == right.csc_column_offsets &&
         left.csc_sources == right.csc_sources &&
         equal_float_bits(left.csc_weights, right.csc_weights) &&
         left.csc_edge_ids == right.csc_edge_ids &&
         left.csc_column_run_offsets == right.csc_column_run_offsets &&
         left.csc_run_edge_offsets == right.csc_run_edge_offsets &&
         left.csc_run_source_tiles == right.csc_run_source_tiles;
}

DeviceGraphMemoryReport report_device_graph_memory(
    const DeviceGraphLayout32& layout) noexcept {
  DeviceGraphMemoryReport report;
  report.owner_tiles_bytes = vector_bytes(layout.owner_tiles);
  report.csr_row_offsets_bytes = vector_bytes(layout.csr_row_offsets);
  report.csr_destinations_bytes = vector_bytes(layout.csr_destinations);
  report.csr_weights_bytes = vector_bytes(layout.csr_weights);
  report.csr_row_run_offsets_bytes = vector_bytes(layout.csr_row_run_offsets);
  report.csr_run_edge_offsets_bytes = vector_bytes(layout.csr_run_edge_offsets);
  report.csr_run_destination_tiles_bytes =
      vector_bytes(layout.csr_run_destination_tiles);
  report.csc_column_offsets_bytes = vector_bytes(layout.csc_column_offsets);
  report.csc_sources_bytes = vector_bytes(layout.csc_sources);
  report.csc_weights_bytes = vector_bytes(layout.csc_weights);
  report.csc_edge_ids_bytes = vector_bytes(layout.csc_edge_ids);
  report.csc_column_run_offsets_bytes = vector_bytes(layout.csc_column_run_offsets);
  report.csc_run_edge_offsets_bytes = vector_bytes(layout.csc_run_edge_offsets);
  report.csc_run_source_tiles_bytes = vector_bytes(layout.csc_run_source_tiles);
  report.total_bytes = report.owner_tiles_bytes + report.csr_row_offsets_bytes +
                       report.csr_destinations_bytes + report.csr_weights_bytes +
                       report.csr_row_run_offsets_bytes +
                       report.csr_run_edge_offsets_bytes +
                       report.csr_run_destination_tiles_bytes +
                       report.csc_column_offsets_bytes + report.csc_sources_bytes +
                       report.csc_weights_bytes + report.csc_edge_ids_bytes +
                       report.csc_column_run_offsets_bytes +
                       report.csc_run_edge_offsets_bytes +
                       report.csc_run_source_tiles_bytes;
  return report;
}

DeviceGraphFingerprint fingerprint_device_graph_layout32(
    const DeviceGraphLayout32& layout) noexcept {
  FingerprintBuilder builder;
  fingerprint_scalars(
      builder, layout.vertex_count, layout.edge_count, layout.tile_count);
  const auto identity = [](const std::uint32_t value) { return value; };
  const auto float_bits = [](const float value) {
    return std::bit_cast<std::uint32_t>(value);
  };
  fingerprint_component(builder, 1U, layout.owner_tiles, identity);
  fingerprint_component(builder, 2U, layout.csr_row_offsets, identity);
  fingerprint_component(builder, 3U, layout.csr_destinations, identity);
  fingerprint_component(builder, 4U, layout.csr_weights, float_bits);
  fingerprint_component(builder, 5U, layout.csr_row_run_offsets, identity);
  fingerprint_component(builder, 6U, layout.csr_run_edge_offsets, identity);
  fingerprint_component(
      builder, 7U, layout.csr_run_destination_tiles, identity);
  fingerprint_component(builder, 8U, layout.csc_column_offsets, identity);
  fingerprint_component(builder, 9U, layout.csc_sources, identity);
  fingerprint_component(builder, 10U, layout.csc_weights, float_bits);
  fingerprint_component(builder, 11U, layout.csc_column_run_offsets, identity);
  fingerprint_component(builder, 12U, layout.csc_run_edge_offsets, identity);
  fingerprint_component(builder, 13U, layout.csc_run_source_tiles, identity);
  fingerprint_component(builder, 14U, layout.csc_edge_ids, identity);
  return builder.finish();
}

DeviceGraphFingerprint fingerprint_device_graph_source32(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs) {
  if (!validate_tile_run_layout(graph, tile_runs).ok()) {
    throw std::invalid_argument{
        "device graph fingerprint requires valid tile-run metadata"};
  }
  if (!graph_is_device_representable(graph, tile_runs)) {
    throw std::overflow_error{
        "device graph fingerprint source exceeds the 32-bit layout"};
  }

  FingerprintBuilder builder;
  fingerprint_scalars(
      builder,
      static_cast<std::uint32_t>(graph.vertex_count()),
      static_cast<std::uint32_t>(graph.edge_count()),
      static_cast<std::uint32_t>(graph.tile_coordinates().size()));
  const auto edge_offset = [](const EdgeOffset value) {
    return static_cast<std::uint32_t>(value);
  };
  const auto vertex_id = [](const VertexId value) { return value.value(); };
  const auto tile_id = [](const TileId value) { return value.value(); };
  const auto float_bits = [](const float value) {
    return std::bit_cast<std::uint32_t>(value);
  };

  fingerprint_component(builder, 1U, graph.owner_tiles(), tile_id);
  const OutgoingCsrView outgoing = graph.outgoing();
  fingerprint_component(builder, 2U, outgoing.row_offsets, edge_offset);
  fingerprint_component(builder, 3U, outgoing.destinations, vertex_id);
  fingerprint_component(builder, 4U, outgoing.weights, float_bits);
  fingerprint_component(builder, 5U, tile_runs.csr_row_run_offsets, edge_offset);
  fingerprint_component(builder, 6U, tile_runs.csr_run_edge_offsets, edge_offset);
  fingerprint_component(
      builder, 7U, tile_runs.csr_run_destination_tiles, tile_id);
  const IncomingCscView incoming = graph.incoming();
  fingerprint_component(builder, 8U, incoming.column_offsets, edge_offset);
  fingerprint_component(builder, 9U, incoming.sources, vertex_id);
  fingerprint_component(builder, 10U, incoming.weights, float_bits);
  fingerprint_component(
      builder, 11U, tile_runs.csc_column_run_offsets, edge_offset);
  fingerprint_component(builder, 12U, tile_runs.csc_run_edge_offsets, edge_offset);
  fingerprint_component(builder, 13U, tile_runs.csc_run_source_tiles, tile_id);
  const auto edge_id = [](const EdgeId value) {
    return static_cast<std::uint32_t>(value.value());
  };
  fingerprint_component(builder, 14U, incoming.edge_ids, edge_id);
  return builder.finish();
}

void compute_tile_run_lane_masks(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const std::span<const std::uint32_t> tile_lane_masks,
    TileRunLaneMasks& output) {
  if (!lane_mask_shape_is_valid(graph, tile_runs, tile_lane_masks)) {
    throw std::invalid_argument{"tile lane masks or tile-run shape are invalid"};
  }

  output.csr_run_masks.assign(tile_runs.csr_run_destination_tiles.size(), 0U);
  output.csc_run_masks.assign(tile_runs.csc_run_source_tiles.size(), 0U);
  const std::span<const TileId> owner_tiles = graph.owner_tiles();
  for (std::size_t row = 0U; row < graph.vertex_count(); ++row) {
    const std::uint32_t owner_mask = tile_lane_masks[owner_tiles[row].value()];
    const std::size_t run_begin =
        static_cast<std::size_t>(tile_runs.csr_row_run_offsets[row]);
    const std::size_t run_end =
        static_cast<std::size_t>(tile_runs.csr_row_run_offsets[row + 1U]);
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const TileId remote_tile = tile_runs.csr_run_destination_tiles[run];
      output.csr_run_masks[run] = owner_mask & tile_lane_masks[remote_tile.value()];
    }
  }
  for (std::size_t column = 0U; column < graph.vertex_count(); ++column) {
    const std::uint32_t owner_mask = tile_lane_masks[owner_tiles[column].value()];
    const std::size_t run_begin =
        static_cast<std::size_t>(tile_runs.csc_column_run_offsets[column]);
    const std::size_t run_end =
        static_cast<std::size_t>(tile_runs.csc_column_run_offsets[column + 1U]);
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const TileId remote_tile = tile_runs.csc_run_source_tiles[run];
      output.csc_run_masks[run] = owner_mask & tile_lane_masks[remote_tile.value()];
    }
  }
}

RunAdmissionProofResult prove_run_admission_equivalence(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const std::span<const std::uint32_t> tile_lane_masks,
    const TileRunLaneMasks& run_lane_masks) {
  if (!validate_tile_run_layout(graph, tile_runs).ok()) {
    return admission_error(RunAdmissionProofErrorCode::tile_run_layout_invalid);
  }
  if (tile_lane_masks.size() != graph.tile_coordinates().size()) {
    return admission_error(RunAdmissionProofErrorCode::tile_lane_mask_size_mismatch);
  }
  if (run_lane_masks.csr_run_masks.size() !=
          tile_runs.csr_run_destination_tiles.size() ||
      run_lane_masks.csc_run_masks.size() != tile_runs.csc_run_source_tiles.size()) {
    return admission_error(RunAdmissionProofErrorCode::run_lane_mask_size_mismatch);
  }

  const std::span<const TileId> owner_tiles = graph.owner_tiles();
  const OutgoingCsrView outgoing = graph.outgoing();
  for (std::size_t row = 0U; row < graph.vertex_count(); ++row) {
    const TileId source_tile = owner_tiles[row];
    const std::size_t run_begin =
        static_cast<std::size_t>(tile_runs.csr_row_run_offsets[row]);
    const std::size_t run_end =
        static_cast<std::size_t>(tile_runs.csr_row_run_offsets[row + 1U]);
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const EdgeOffset edge_begin = tile_runs.csr_run_edge_offsets[run];
      const EdgeOffset edge_end = tile_runs.csr_run_edge_offsets[run + 1U];
      for (EdgeOffset edge = edge_begin; edge < edge_end; ++edge) {
        const VertexId destination =
            outgoing.destinations[static_cast<std::size_t>(edge)];
        const std::uint32_t endpoint_mask =
            tile_lane_masks[source_tile.value()] &
            tile_lane_masks[owner_tiles[destination.value()].value()];
        if (run_lane_masks.csr_run_masks[run] != endpoint_mask) {
          return admission_error(
              RunAdmissionProofErrorCode::csr_endpoint_mismatch, edge);
        }
      }
    }
  }

  const IncomingCscView incoming = graph.incoming();
  for (std::size_t column = 0U; column < graph.vertex_count(); ++column) {
    const TileId destination_tile = owner_tiles[column];
    const std::size_t run_begin =
        static_cast<std::size_t>(tile_runs.csc_column_run_offsets[column]);
    const std::size_t run_end =
        static_cast<std::size_t>(tile_runs.csc_column_run_offsets[column + 1U]);
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const EdgeOffset edge_begin = tile_runs.csc_run_edge_offsets[run];
      const EdgeOffset edge_end = tile_runs.csc_run_edge_offsets[run + 1U];
      for (EdgeOffset edge = edge_begin; edge < edge_end; ++edge) {
        const VertexId source = incoming.sources[static_cast<std::size_t>(edge)];
        const std::uint32_t endpoint_mask =
            tile_lane_masks[destination_tile.value()] &
            tile_lane_masks[owner_tiles[source.value()].value()];
        if (run_lane_masks.csc_run_masks[run] != endpoint_mask) {
          return admission_error(
              RunAdmissionProofErrorCode::csc_endpoint_mismatch, edge);
        }
      }
    }
  }
  return {};
}

}  // namespace bfnew
