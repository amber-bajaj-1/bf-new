#include "bfnew/batch_layout.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bfnew {
namespace {

[[nodiscard]] constexpr BatchLayoutValidationResult layout_error(
    const BatchLayoutValidationErrorCode code,
    const std::size_t position =
        BatchLayoutValidationResult::no_position) noexcept {
  return BatchLayoutValidationResult{code, position};
}

[[nodiscard]] constexpr bool supported_lane_width(
    const std::uint32_t lane_width) noexcept {
  return lane_width == 1U || lane_width == 8U || lane_width == 16U ||
         lane_width == 32U;
}

[[nodiscard]] constexpr LaneMask lane_bit(const std::size_t lane) noexcept {
  return LaneMask{1U} << static_cast<std::uint32_t>(lane);
}

[[nodiscard]] constexpr LaneMask width_mask(
    const std::uint32_t lane_width) noexcept {
  return lane_width == 32U
             ? std::numeric_limits<LaneMask>::max()
             : (LaneMask{1U} << lane_width) - LaneMask{1U};
}

[[nodiscard]] constexpr bool is_low_lane_mask(
    const LaneMask mask) noexcept {
  return mask != 0U && (mask & (mask + LaneMask{1U})) == 0U;
}

[[nodiscard]] bool add_u64(
    std::uint64_t& value,
    const std::uint64_t increment) noexcept {
  if (value > std::numeric_limits<std::uint64_t>::max() - increment) {
    return false;
  }
  value += increment;
  return true;
}

void checked_add_u64(
    std::uint64_t& value,
    const std::uint64_t increment,
    const char* const message) {
  if (!add_u64(value, increment)) {
    throw std::overflow_error{message};
  }
}

[[nodiscard]] bool multiply_u64(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t& product) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  product = left * right;
  return true;
}

[[nodiscard]] std::uint32_t checked_size32(
    const std::size_t value,
    const char* const message) {
  if (!std::in_range<std::uint32_t>(value)) {
    throw std::overflow_error{message};
  }
  return static_cast<std::uint32_t>(value);
}

[[nodiscard]] bool canonical_tiles(
    const std::span<const TileId> tiles,
    const std::size_t tile_count) noexcept {
  for (std::size_t index = 0U; index < tiles.size(); ++index) {
    if (tiles[index].value() >= tile_count ||
        (index != 0U && !(tiles[index - 1U] < tiles[index]))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool contains_tile(
    const std::span<const TileId> tiles,
    const TileId tile) noexcept {
  return std::binary_search(tiles.begin(), tiles.end(), tile);
}

[[nodiscard]] bool terminal_tiles_match(
    const WeightedGraph& graph,
    const std::span<const VertexId> terminals,
    const std::span<const TileId> expected) noexcept {
  std::size_t expected_position = 0U;
  TileId preceding{};
  bool has_preceding = false;
  for (const VertexId terminal : terminals) {
    if (terminal.value() >= graph.vertex_count()) {
      return false;
    }
    const TileId owner = graph.owner_tiles()[terminal.value()];
    if (has_preceding && owner == preceding) {
      continue;
    }
    if (expected_position >= expected.size() ||
        expected[expected_position] != owner) {
      return false;
    }
    preceding = owner;
    has_preceding = true;
    ++expected_position;
  }
  return expected_position == expected.size();
}

[[nodiscard]] bool preparation_shapes_are_safe(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs) noexcept {
  const std::size_t vertex_count = graph.vertex_count();
  const std::size_t tile_count = graph.tile_coordinates().size();
  const std::size_t csr_run_count =
      tile_runs.csr_run_destination_tiles.size();
  const std::size_t csc_run_count = tile_runs.csc_run_source_tiles.size();
  if (!graph.has_spatial_ordering() || tile_count == 0U ||
      tile_count == std::numeric_limits<std::size_t>::max() ||
      vertex_count == std::numeric_limits<std::size_t>::max() ||
      csr_run_count == std::numeric_limits<std::size_t>::max() ||
      csc_run_count == std::numeric_limits<std::size_t>::max() ||
      graph.owner_tiles().size() != vertex_count ||
      graph.tile_vertex_offsets().size() != tile_count + 1U ||
      graph.tile_vertex_offsets().empty() ||
      graph.tile_vertex_offsets().front() != 0U ||
      graph.tile_vertex_offsets().back() != graph.vertex_count() ||
      !std::in_range<EdgeOffset>(csr_run_count) ||
      !std::in_range<EdgeOffset>(csc_run_count) ||
      tile_runs.csr_row_run_offsets.size() != vertex_count + 1U ||
      tile_runs.csc_column_run_offsets.size() != vertex_count + 1U ||
      tile_runs.csr_row_run_offsets.empty() ||
      tile_runs.csc_column_run_offsets.empty() ||
      tile_runs.csr_row_run_offsets.front() != 0U ||
      tile_runs.csc_column_run_offsets.front() != 0U ||
      tile_runs.csr_row_run_offsets.back() !=
          static_cast<EdgeOffset>(csr_run_count) ||
      tile_runs.csc_column_run_offsets.back() !=
          static_cast<EdgeOffset>(csc_run_count) ||
      tile_runs.csr_run_edge_offsets.size() != csr_run_count + 1U ||
      tile_runs.csc_run_edge_offsets.size() != csc_run_count + 1U ||
      tile_runs.csr_run_edge_offsets.empty() ||
      tile_runs.csc_run_edge_offsets.empty() ||
      tile_runs.csr_run_edge_offsets.front() != 0U ||
      tile_runs.csc_run_edge_offsets.front() != 0U ||
      tile_runs.csr_run_edge_offsets.back() != graph.edge_count() ||
      tile_runs.csc_run_edge_offsets.back() != graph.edge_count()) {
    return false;
  }
  return true;
}

struct ResolvedBatchInputs {
  BatchLayoutValidationResult validation{};
  std::array<const RouteQuery*, maximum_batch_lanes> queries_by_lane{};
};

[[nodiscard]] const RouteQuery* find_unique_query(
    const std::span<const RouteQuery> queries,
    const QueryId query_id) noexcept {
  const RouteQuery* match = nullptr;
  for (const RouteQuery& query : queries) {
    if (query.query_id != query_id) {
      continue;
    }
    if (match != nullptr) {
      return nullptr;
    }
    match = &query;
  }
  return match;
}

[[nodiscard]] LaneMask expected_tile_lane_mask(
    const std::span<const BatchQueryFeatures> features,
    const BatchPlanEntry& batch,
    const TileId tile) noexcept {
  LaneMask mask = 0U;
  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    const LaneMask bit = lane_bit(lane);
    if ((batch.valid_lane_mask & bit) == 0U) {
      continue;
    }
    const std::size_t query_index = batch.query_indices_by_lane[lane];
    if (query_index < features.size() &&
        contains_tile(features[query_index].selected_tiles, tile)) {
      mask |= bit;
    }
  }
  return mask;
}

[[nodiscard]] ResolvedBatchInputs validate_batch_inputs(
    const WeightedGraph& graph,
    const std::span<const RouteQuery> queries,
    const std::span<const BatchQueryFeatures> features,
    const BatchPlanEntry& batch) noexcept {
  ResolvedBatchInputs result;
  const std::size_t lane_width = batch.lane_width;
  if (!supported_lane_width(batch.lane_width) ||
      batch.query_indices_by_lane.size() != lane_width ||
      batch.query_ids_by_lane.size() != lane_width ||
      batch.expansion_generations_by_lane.size() != lane_width ||
      batch.union_tiles.size() != batch.union_tile_lane_masks.size() ||
      batch.union_tiles.empty() || batch.valid_lane_mask == 0U ||
      (batch.valid_lane_mask & ~width_mask(batch.lane_width)) != 0U ||
      !is_low_lane_mask(batch.valid_lane_mask)) {
    result.validation =
        layout_error(BatchLayoutValidationErrorCode::invalid_batch_shape);
    return result;
  }

  std::uint64_t selected_lane_vertices = 0U;
  std::uint64_t selected_lane_edges = 0U;
  for (std::size_t lane = 0U; lane < lane_width; ++lane) {
    const LaneMask bit = lane_bit(lane);
    const bool valid = (batch.valid_lane_mask & bit) != 0U;
    const std::uint32_t query_index = batch.query_indices_by_lane[lane];
    if (!valid) {
      if (query_index != invalid_batch_query_index ||
          batch.query_ids_by_lane[lane] != invalid_batch_query_id ||
          batch.expansion_generations_by_lane[lane] != 0U) {
        result.validation = layout_error(
            BatchLayoutValidationErrorCode::invalid_lane_identity, lane);
        return result;
      }
      continue;
    }
    if (query_index == invalid_batch_query_index || query_index >= features.size()) {
      result.validation = layout_error(
          BatchLayoutValidationErrorCode::invalid_lane_identity, lane);
      return result;
    }
    for (std::size_t preceding_lane = 0U; preceding_lane < lane;
         ++preceding_lane) {
      if ((batch.valid_lane_mask & lane_bit(preceding_lane)) != 0U &&
          (batch.query_indices_by_lane[preceding_lane] == query_index ||
           batch.query_ids_by_lane[preceding_lane] ==
               batch.query_ids_by_lane[lane])) {
        result.validation = layout_error(
            BatchLayoutValidationErrorCode::invalid_lane_identity, lane);
        return result;
      }
    }

    const BatchQueryFeatures& feature = features[query_index];
    const RouteQuery* const query = find_unique_query(queries, feature.query_id);
    if (query == nullptr || batch.query_ids_by_lane[lane] != feature.query_id ||
        batch.expansion_generations_by_lane[lane] !=
            feature.expansion_generation ||
        query->query_id != feature.query_id ||
        query->expansion_generation != feature.expansion_generation ||
        !validate_route_query(graph, *query).ok() ||
        !std::in_range<std::uint32_t>(query->sources.size()) ||
        !std::in_range<std::uint32_t>(query->targets.size()) ||
        feature.source_count != query->sources.size() ||
        feature.target_count != query->targets.size() ||
        feature.selected_tiles != query->selected_tiles ||
        !canonical_tiles(feature.source_tiles, graph.tile_coordinates().size()) ||
        !canonical_tiles(feature.target_tiles, graph.tile_coordinates().size()) ||
        !canonical_tiles(feature.selected_tiles, graph.tile_coordinates().size()) ||
        !terminal_tiles_match(graph, query->sources, feature.source_tiles) ||
        !terminal_tiles_match(graph, query->targets, feature.target_tiles)) {
      result.validation = layout_error(
          BatchLayoutValidationErrorCode::invalid_lane_identity, lane);
      return result;
    }

    std::uint64_t expected_vertices = 0U;
    const auto tile_offsets = graph.tile_vertex_offsets();
    for (const TileId tile : feature.selected_tiles) {
      const EdgeOffset begin = tile_offsets[tile.value()];
      const EdgeOffset end = tile_offsets[tile.value() + 1U];
      if (begin > end || end > graph.vertex_count()) {
        result.validation = layout_error(
            BatchLayoutValidationErrorCode::invalid_estimates, lane);
        return result;
      }
      if (!add_u64(
              expected_vertices,
              end - begin)) {
        result.validation = layout_error(
            BatchLayoutValidationErrorCode::invalid_estimates, lane);
        return result;
      }
    }
    if (feature.selected_vertex_count != expected_vertices ||
        !add_u64(selected_lane_vertices, feature.selected_vertex_count) ||
        !add_u64(selected_lane_edges, feature.selected_edge_estimate)) {
      result.validation = layout_error(
          BatchLayoutValidationErrorCode::invalid_estimates, lane);
      return result;
    }
    result.queries_by_lane[lane] = query;
  }

  const std::size_t tile_count = graph.tile_coordinates().size();
  std::uint64_t union_vertices = 0U;
  for (std::size_t position = 0U; position < batch.union_tiles.size();
       ++position) {
    const TileId tile = batch.union_tiles[position];
    if (tile.value() >= tile_count ||
        (position != 0U && !(batch.union_tiles[position - 1U] < tile))) {
      result.validation = layout_error(
          BatchLayoutValidationErrorCode::invalid_union_tiles, position);
      return result;
    }
    const LaneMask expected_mask =
        expected_tile_lane_mask(features, batch, tile);
    if (expected_mask == 0U ||
        batch.union_tile_lane_masks[position] != expected_mask) {
      result.validation = layout_error(
          BatchLayoutValidationErrorCode::invalid_tile_lane_masks, position);
      return result;
    }
    const auto offsets = graph.tile_vertex_offsets();
    const EdgeOffset begin = offsets[tile.value()];
    const EdgeOffset end = offsets[tile.value() + 1U];
    if (begin > end || end > graph.vertex_count()) {
      result.validation = layout_error(
          BatchLayoutValidationErrorCode::invalid_estimates, position);
      return result;
    }
    if (!add_u64(
            union_vertices,
            end - begin)) {
      result.validation = layout_error(
          BatchLayoutValidationErrorCode::invalid_estimates, position);
      return result;
    }
  }
  for (std::size_t lane = 0U; lane < lane_width; ++lane) {
    if ((batch.valid_lane_mask & lane_bit(lane)) == 0U) {
      continue;
    }
    const BatchQueryFeatures& feature =
        features[batch.query_indices_by_lane[lane]];
    for (const TileId tile : feature.selected_tiles) {
      if (!contains_tile(batch.union_tiles, tile)) {
        result.validation = layout_error(
            BatchLayoutValidationErrorCode::invalid_union_tiles, lane);
        return result;
      }
    }
  }
  if (batch.union_vertex_count != union_vertices ||
      batch.selected_lane_vertex_count != selected_lane_vertices ||
      batch.selected_lane_edge_estimate != selected_lane_edges) {
    result.validation =
        layout_error(BatchLayoutValidationErrorCode::invalid_estimates);
  }
  return result;
}

void clear_prior_touched(
    std::vector<LaneMask>& masks,
    std::vector<std::uint32_t>& touched,
    std::uint64_t& cleared,
    std::uint64_t& orientation_cleared) {
  std::uint32_t preceding = 0U;
  bool has_preceding = false;
  for (const std::uint32_t run : touched) {
    if (run >= masks.size() || (has_preceding && run <= preceding) ||
        masks[run] == 0U) {
      throw std::invalid_argument{
          "reused retained run-mask ledger is internally inconsistent"};
    }
    masks[run] = 0U;
    checked_add_u64(
        cleared, 1U, "retained run-mask clear count overflowed");
    checked_add_u64(
        orientation_cleared,
        1U,
        "orientation retained run-mask clear count overflowed");
    preceding = run;
    has_preceding = true;
  }
  touched.clear();
}

struct RunCoverage {
  std::uint64_t visited{};
  std::uint64_t active{};
  std::uint64_t lane_edge_pairs{};
  std::uint64_t union_edges{};
  std::array<std::uint64_t, maximum_batch_lanes> edges_by_lane{};
};

void accumulate_run_coverage(
    RunCoverage& coverage,
    const LaneMask mask,
    const bool remote_tile_is_in_union,
    const EdgeOffset edge_count) {
  checked_add_u64(coverage.visited, 1U, "run visit count overflowed");
  if (remote_tile_is_in_union) {
    checked_add_u64(
        coverage.union_edges,
        edge_count,
        "union edge estimate overflowed");
  }
  if (mask == 0U) {
    return;
  }
  checked_add_u64(coverage.active, 1U, "active run count overflowed");
  std::uint64_t pairs = 0U;
  if (!multiply_u64(
          edge_count,
          static_cast<std::uint64_t>(std::popcount(mask)),
          pairs)) {
    throw std::overflow_error{"lane-edge pair count overflowed"};
  }
  checked_add_u64(
      coverage.lane_edge_pairs, pairs, "lane-edge pair count overflowed");
  for (std::size_t lane = 0U; lane < maximum_batch_lanes; ++lane) {
    if ((mask & lane_bit(lane)) != 0U) {
      checked_add_u64(
          coverage.edges_by_lane[lane],
          edge_count,
          "per-lane admitted edge count overflowed");
    }
  }
}

void record_run(
    const bool csr,
    const std::size_t run,
    const LaneMask mask,
    BatchDeviceDescription& output) {
  if (output.run_representation ==
      BatchRunRepresentation::retained_per_run_masks) {
    std::vector<LaneMask>& masks =
        csr ? output.csr_run_lane_masks : output.csc_run_lane_masks;
    std::vector<std::uint32_t>& touched =
        csr ? output.touched_csr_runs : output.touched_csc_runs;
    if (mask != 0U) {
      masks[run] = mask;
      touched.push_back(checked_size32(run, "tile-run ID exceeds 32 bits"));
      checked_add_u64(
          output.run_report.retained_entries_written,
          1U,
          "retained run-mask write count overflowed");
      std::uint64_t& orientation_written =
          csr ? output.run_report.csr_retained_entries_written
              : output.run_report.csc_retained_entries_written;
      checked_add_u64(
          orientation_written,
          1U,
          "orientation retained run-mask write count overflowed");
    }
    return;
  }
  if (output.run_representation ==
      BatchRunRepresentation::device_materialized_run_masks) {
    return;
  }
  if (mask != 0U) {
    std::vector<RunLaneMaskDescriptor>& descriptors =
        csr ? output.csr_run_descriptors : output.csc_run_descriptors;
    descriptors.push_back(RunLaneMaskDescriptor{
        checked_size32(run, "tile-run ID exceeds 32 bits"), mask});
    checked_add_u64(
        output.run_report.descriptor_entries_written,
        1U,
        "run descriptor write count overflowed");
    std::uint64_t& orientation_written =
        csr ? output.run_report.csr_descriptor_entries_written
            : output.run_report.csc_descriptor_entries_written;
    checked_add_u64(
        orientation_written,
        1U,
        "orientation run descriptor write count overflowed");
  }
}

[[nodiscard]] std::pair<RunCoverage, RunCoverage> build_run_storage(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    BatchDeviceDescription& output) {
  RunCoverage csr;
  RunCoverage csc;
  std::size_t compact_vertex = 0U;
  for (const BatchVertexRange& range : output.selected_vertex_ranges) {
    if (range.begin > range.end || range.end > graph.vertex_count()) {
      throw std::invalid_argument{
          "selected batch vertex range is outside the graph"};
    }
    for (std::size_t vertex = range.begin; vertex < range.end; ++vertex) {
      if (output.run_representation ==
          BatchRunRepresentation::compact_nonzero_descriptors) {
        if (output.csr_descriptor_offsets_by_union_vertex.size() !=
                compact_vertex + 1U ||
            output.csc_descriptor_offsets_by_union_vertex.size() !=
                compact_vertex + 1U) {
          throw std::invalid_argument{
              "descriptor offset construction state is inconsistent"};
        }
      }
      const EdgeOffset csr_begin_offset =
          tile_runs.csr_row_run_offsets[vertex];
      const EdgeOffset csr_end_offset =
          tile_runs.csr_row_run_offsets[vertex + 1U];
      if (csr_begin_offset > csr_end_offset ||
          csr_end_offset > tile_runs.csr_run_destination_tiles.size() ||
          !std::in_range<std::size_t>(csr_begin_offset) ||
          !std::in_range<std::size_t>(csr_end_offset)) {
        throw std::invalid_argument{
            "selected CSR row has invalid tile-run offsets"};
      }
      const std::size_t csr_begin = static_cast<std::size_t>(csr_begin_offset);
      const std::size_t csr_end = static_cast<std::size_t>(csr_end_offset);
      for (std::size_t run = csr_begin; run < csr_end; ++run) {
        const TileId remote = tile_runs.csr_run_destination_tiles[run];
        if (remote.value() >= output.tile_lane_masks.size()) {
          throw std::invalid_argument{
              "selected CSR run references an invalid remote tile"};
        }
        const LaneMask remote_mask = output.tile_lane_masks[remote.value()];
        const LaneMask mask =
            range.lane_mask & remote_mask;
        const EdgeOffset edge_begin = tile_runs.csr_run_edge_offsets[run];
        const EdgeOffset edge_end = tile_runs.csr_run_edge_offsets[run + 1U];
        if (edge_begin >= edge_end || edge_end > graph.edge_count()) {
          throw std::invalid_argument{
              "selected CSR run has invalid edge offsets"};
        }
        const EdgeOffset edge_count = edge_end - edge_begin;
        accumulate_run_coverage(csr, mask, remote_mask != 0U, edge_count);
        record_run(true, run, mask, output);
      }
      if (output.run_representation ==
          BatchRunRepresentation::compact_nonzero_descriptors) {
        output.csr_descriptor_offsets_by_union_vertex.push_back(checked_size32(
            output.csr_run_descriptors.size(),
            "CSR descriptor position exceeds 32 bits"));
      }

      const EdgeOffset csc_begin_offset =
          tile_runs.csc_column_run_offsets[vertex];
      const EdgeOffset csc_end_offset =
          tile_runs.csc_column_run_offsets[vertex + 1U];
      if (csc_begin_offset > csc_end_offset ||
          csc_end_offset > tile_runs.csc_run_source_tiles.size() ||
          !std::in_range<std::size_t>(csc_begin_offset) ||
          !std::in_range<std::size_t>(csc_end_offset)) {
        throw std::invalid_argument{
            "selected CSC column has invalid tile-run offsets"};
      }
      const std::size_t csc_begin = static_cast<std::size_t>(csc_begin_offset);
      const std::size_t csc_end = static_cast<std::size_t>(csc_end_offset);
      for (std::size_t run = csc_begin; run < csc_end; ++run) {
        const TileId remote = tile_runs.csc_run_source_tiles[run];
        if (remote.value() >= output.tile_lane_masks.size()) {
          throw std::invalid_argument{
              "selected CSC run references an invalid remote tile"};
        }
        const LaneMask remote_mask = output.tile_lane_masks[remote.value()];
        const LaneMask mask =
            range.lane_mask & remote_mask;
        const EdgeOffset edge_begin = tile_runs.csc_run_edge_offsets[run];
        const EdgeOffset edge_end = tile_runs.csc_run_edge_offsets[run + 1U];
        if (edge_begin >= edge_end || edge_end > graph.edge_count()) {
          throw std::invalid_argument{
              "selected CSC run has invalid edge offsets"};
        }
        const EdgeOffset edge_count = edge_end - edge_begin;
        accumulate_run_coverage(csc, mask, remote_mask != 0U, edge_count);
        record_run(false, run, mask, output);
      }
      if (output.run_representation ==
          BatchRunRepresentation::compact_nonzero_descriptors) {
        output.csc_descriptor_offsets_by_union_vertex.push_back(checked_size32(
            output.csc_run_descriptors.size(),
            "CSC descriptor position exceeds 32 bits"));
      }
      ++compact_vertex;
    }
  }
  output.run_report.csr_runs_visited = csr.visited;
  output.run_report.csc_runs_visited = csc.visited;
  output.run_report.active_csr_runs = csr.active;
  output.run_report.active_csc_runs = csc.active;
  output.run_report.csr_lane_edge_pairs = csr.lane_edge_pairs;
  output.run_report.csc_lane_edge_pairs = csc.lane_edge_pairs;
  return {csr, csc};
}

[[nodiscard]] bool valid_flat_offsets(
    const std::span<const std::uint32_t> offsets,
    const std::size_t lane_width,
    const std::size_t payload_size,
    const LaneMask valid_lane_mask) noexcept {
  if (offsets.size() != lane_width + 1U || offsets.empty() ||
      offsets.front() != 0U || !std::in_range<std::uint32_t>(payload_size) ||
      offsets.back() != payload_size) {
    return false;
  }
  for (std::size_t lane = 0U; lane < lane_width; ++lane) {
    if (offsets[lane] > offsets[lane + 1U] ||
        offsets[lane + 1U] > payload_size ||
        ((valid_lane_mask & lane_bit(lane)) == 0U &&
         offsets[lane] != offsets[lane + 1U])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] BatchLayoutValidationResult validate_retained_shape(
    const BatchDeviceDescription& description,
    const std::size_t csr_run_count,
    const std::size_t csc_run_count) noexcept {
  if (description.csr_run_lane_masks.size() != csr_run_count ||
      description.csc_run_lane_masks.size() != csc_run_count ||
      !description.csr_run_descriptors.empty() ||
      !description.csc_run_descriptors.empty() ||
      !description.csr_descriptor_offsets_by_union_vertex.empty() ||
      !description.csc_descriptor_offsets_by_union_vertex.empty()) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_run_storage_shape);
  }
  const auto validate_ledger = [](
                                   const std::span<const LaneMask> masks,
                                   const std::span<const std::uint32_t> touched)
      noexcept -> bool {
    std::size_t ledger_position = 0U;
    for (std::size_t run = 0U; run < masks.size(); ++run) {
      if (masks[run] == 0U) {
        continue;
      }
      if (ledger_position >= touched.size() ||
          touched[ledger_position] != run) {
        return false;
      }
      ++ledger_position;
    }
    return ledger_position == touched.size();
  };
  if (!validate_ledger(
          description.csr_run_lane_masks, description.touched_csr_runs) ||
      !validate_ledger(
          description.csc_run_lane_masks, description.touched_csc_runs)) {
    return layout_error(BatchLayoutValidationErrorCode::invalid_run_mask);
  }
  return {};
}

[[nodiscard]] BatchLayoutValidationResult validate_descriptor_shape(
    const BatchDeviceDescription& description,
    const std::size_t csr_run_count,
    const std::size_t csc_run_count,
    const std::uint64_t union_vertex_count) noexcept {
  if (!description.csr_run_lane_masks.empty() ||
      !description.csc_run_lane_masks.empty() ||
      !description.touched_csr_runs.empty() ||
      !description.touched_csc_runs.empty()) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_run_storage_shape);
  }
  const auto validate_descriptors = [](
                                        const std::span<const RunLaneMaskDescriptor>
                                            descriptors,
                                        const std::size_t run_count)
      noexcept -> BatchLayoutValidationResult {
    for (std::size_t position = 0U; position < descriptors.size(); ++position) {
      const RunLaneMaskDescriptor descriptor = descriptors[position];
      if (position != 0U &&
          descriptor.run_id <= descriptors[position - 1U].run_id) {
        return layout_error(
            BatchLayoutValidationErrorCode::invalid_descriptor_order,
            position);
      }
      if (descriptor.run_id >= run_count || descriptor.lane_mask == 0U) {
        return layout_error(
            BatchLayoutValidationErrorCode::invalid_run_mask, position);
      }
    }
    return {};
  };
  BatchLayoutValidationResult validation = validate_descriptors(
      description.csr_run_descriptors, csr_run_count);
  if (!validation.ok()) {
    return validation;
  }
  validation =
      validate_descriptors(description.csc_run_descriptors, csc_run_count);
  if (!validation.ok()) {
    return validation;
  }
  if (union_vertex_count == std::numeric_limits<std::uint64_t>::max() ||
      !std::in_range<std::size_t>(union_vertex_count + 1U)) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_descriptor_offsets);
  }
  const std::size_t expected_offset_count =
      static_cast<std::size_t>(union_vertex_count + 1U);
  const auto validate_offsets = [](
                                    const std::span<const std::uint32_t> offsets,
                                    const std::size_t expected_count,
                                    const std::size_t descriptor_count) noexcept {
    if (offsets.size() != expected_count || offsets.empty() ||
        offsets.front() != 0U ||
        !std::in_range<std::uint32_t>(descriptor_count) ||
        offsets.back() != descriptor_count) {
      return false;
    }
    for (std::size_t position = 1U; position < offsets.size(); ++position) {
      if (offsets[position - 1U] > offsets[position] ||
          offsets[position] > descriptor_count) {
        return false;
      }
    }
    return true;
  };
  if (!validate_offsets(
          description.csr_descriptor_offsets_by_union_vertex,
          expected_offset_count,
          description.csr_run_descriptors.size()) ||
      !validate_offsets(
          description.csc_descriptor_offsets_by_union_vertex,
          expected_offset_count,
          description.csc_run_descriptors.size())) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_descriptor_offsets);
  }
  return {};
}

[[nodiscard]] BatchLayoutValidationResult
validate_device_materialized_shape(
    const BatchDeviceDescription& description) noexcept {
  if (!description.csr_run_lane_masks.empty() ||
      !description.csc_run_lane_masks.empty() ||
      !description.touched_csr_runs.empty() ||
      !description.touched_csc_runs.empty() ||
      !description.csr_run_descriptors.empty() ||
      !description.csc_run_descriptors.empty() ||
      !description.csr_descriptor_offsets_by_union_vertex.empty() ||
      !description.csc_descriptor_offsets_by_union_vertex.empty() ||
      description.run_report != BatchRunPreparationReport{}) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_run_storage_shape);
  }
  return {};
}

struct RunValidationState {
  RunCoverage coverage{};
  std::size_t stored_position{};
};

[[nodiscard]] bool validate_expected_run(
    const bool csr,
    const std::size_t run,
    const LaneMask expected_mask,
    const bool remote_tile_is_in_union,
    const EdgeOffset edge_count,
    const BatchDeviceDescription& description,
    RunValidationState& state) noexcept {
  if (!add_u64(state.coverage.visited, 1U)) {
    return false;
  }
  if (remote_tile_is_in_union &&
      !add_u64(state.coverage.union_edges, edge_count)) {
    return false;
  }
  if (description.run_representation ==
      BatchRunRepresentation::retained_per_run_masks) {
    const std::span<const LaneMask> masks = csr
        ? std::span<const LaneMask>{description.csr_run_lane_masks}
        : std::span<const LaneMask>{description.csc_run_lane_masks};
    if (masks[run] != expected_mask) {
      return false;
    }
    if (expected_mask != 0U) {
      const std::span<const std::uint32_t> touched = csr
          ? std::span<const std::uint32_t>{description.touched_csr_runs}
          : std::span<const std::uint32_t>{description.touched_csc_runs};
      if (state.stored_position >= touched.size() ||
          touched[state.stored_position] != run) {
        return false;
      }
      ++state.stored_position;
    }
  } else if (description.run_representation ==
                 BatchRunRepresentation::compact_nonzero_descriptors &&
             expected_mask != 0U) {
    const std::span<const RunLaneMaskDescriptor> descriptors = csr
        ? std::span<const RunLaneMaskDescriptor>{description.csr_run_descriptors}
        : std::span<const RunLaneMaskDescriptor>{description.csc_run_descriptors};
    if (state.stored_position >= descriptors.size() ||
        descriptors[state.stored_position] !=
            RunLaneMaskDescriptor{static_cast<std::uint32_t>(run), expected_mask}) {
      return false;
    }
    ++state.stored_position;
  }

  if (expected_mask == 0U) {
    return true;
  }
  if (!add_u64(state.coverage.active, 1U)) {
    return false;
  }
  std::uint64_t pairs = 0U;
  if (!multiply_u64(
          edge_count,
          static_cast<std::uint64_t>(std::popcount(expected_mask)),
          pairs) ||
      !add_u64(state.coverage.lane_edge_pairs, pairs)) {
    return false;
  }
  for (std::size_t lane = 0U; lane < maximum_batch_lanes; ++lane) {
    if ((expected_mask & lane_bit(lane)) != 0U &&
        !add_u64(state.coverage.edges_by_lane[lane], edge_count)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] BatchLayoutValidationResult validate_runs_and_coverage(
    const TileRunLayout64& tile_runs,
    const std::span<const BatchQueryFeatures> features,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description) noexcept {
  RunValidationState csr;
  RunValidationState csc;
  std::size_t compact_vertex = 0U;
  for (const BatchVertexRange& range : description.selected_vertex_ranges) {
    for (std::size_t vertex = range.begin; vertex < range.end; ++vertex) {
      const bool descriptors = description.run_representation ==
                               BatchRunRepresentation::compact_nonzero_descriptors;
      if (descriptors &&
          (description.csr_descriptor_offsets_by_union_vertex[compact_vertex] !=
               csr.stored_position ||
           description.csc_descriptor_offsets_by_union_vertex[compact_vertex] !=
               csc.stored_position)) {
        return layout_error(
            BatchLayoutValidationErrorCode::invalid_descriptor_offsets,
            compact_vertex);
      }
      const std::size_t csr_begin =
          static_cast<std::size_t>(tile_runs.csr_row_run_offsets[vertex]);
      const std::size_t csr_end =
          static_cast<std::size_t>(tile_runs.csr_row_run_offsets[vertex + 1U]);
      for (std::size_t run = csr_begin; run < csr_end; ++run) {
        const TileId remote = tile_runs.csr_run_destination_tiles[run];
        const LaneMask remote_mask =
            description.tile_lane_masks[remote.value()];
        const LaneMask mask =
            range.lane_mask & remote_mask;
        const EdgeOffset edge_count =
            tile_runs.csr_run_edge_offsets[run + 1U] -
            tile_runs.csr_run_edge_offsets[run];
        if (!validate_expected_run(
                true,
                run,
                mask,
                remote_mask != 0U,
                edge_count,
                description,
                csr)) {
          return layout_error(
              BatchLayoutValidationErrorCode::invalid_run_mask, run);
        }
      }
      if (descriptors &&
          description.csr_descriptor_offsets_by_union_vertex
                  [compact_vertex + 1U] != csr.stored_position) {
        return layout_error(
            BatchLayoutValidationErrorCode::invalid_descriptor_offsets,
            compact_vertex + 1U);
      }

      const std::size_t csc_begin =
          static_cast<std::size_t>(tile_runs.csc_column_run_offsets[vertex]);
      const std::size_t csc_end =
          static_cast<std::size_t>(tile_runs.csc_column_run_offsets[vertex + 1U]);
      for (std::size_t run = csc_begin; run < csc_end; ++run) {
        const TileId remote = tile_runs.csc_run_source_tiles[run];
        const LaneMask remote_mask =
            description.tile_lane_masks[remote.value()];
        const LaneMask mask =
            range.lane_mask & remote_mask;
        const EdgeOffset edge_count =
            tile_runs.csc_run_edge_offsets[run + 1U] -
            tile_runs.csc_run_edge_offsets[run];
        if (!validate_expected_run(
                false,
                run,
                mask,
                remote_mask != 0U,
                edge_count,
                description,
                csc)) {
          return layout_error(
              BatchLayoutValidationErrorCode::invalid_run_mask, run);
        }
      }
      if (descriptors &&
          description.csc_descriptor_offsets_by_union_vertex
                  [compact_vertex + 1U] != csc.stored_position) {
        return layout_error(
            BatchLayoutValidationErrorCode::invalid_descriptor_offsets,
            compact_vertex + 1U);
      }
      ++compact_vertex;
    }
  }

  const bool retained = description.run_representation ==
                        BatchRunRepresentation::retained_per_run_masks;
  const bool descriptors = description.run_representation ==
                           BatchRunRepresentation::compact_nonzero_descriptors;
  const std::size_t expected_csr_stored =
      retained ? description.touched_csr_runs.size()
               : (descriptors ? description.csr_run_descriptors.size() : 0U);
  const std::size_t expected_csc_stored =
      retained ? description.touched_csc_runs.size()
               : (descriptors ? description.csc_run_descriptors.size() : 0U);
  if (csr.stored_position != expected_csr_stored ||
      csc.stored_position != expected_csc_stored) {
    return layout_error(BatchLayoutValidationErrorCode::invalid_run_mask);
  }

  const BatchRunPreparationReport& report = description.run_report;
  if (description.run_representation ==
      BatchRunRepresentation::device_materialized_run_masks) {
    if (report != BatchRunPreparationReport{}) {
      return layout_error(BatchLayoutValidationErrorCode::invalid_run_mask);
    }
    if (csr.coverage.union_edges != batch.union_edge_estimate ||
        csc.coverage.union_edges != batch.union_edge_estimate) {
      return layout_error(
          BatchLayoutValidationErrorCode::admitted_edge_coverage_mismatch);
    }
    for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
      const std::uint64_t expected =
          (batch.valid_lane_mask & lane_bit(lane)) == 0U
              ? 0U
              : features[batch.query_indices_by_lane[lane]]
                    .selected_edge_estimate;
      if (csr.coverage.edges_by_lane[lane] != expected ||
          csc.coverage.edges_by_lane[lane] != expected ||
          description.selected_edge_estimates_by_lane[lane] != expected) {
        return layout_error(
            BatchLayoutValidationErrorCode::admitted_edge_coverage_mismatch,
            lane);
      }
    }
    return {};
  }
  std::uint64_t active_total = csr.coverage.active;
  if (!add_u64(active_total, csc.coverage.active)) {
    return layout_error(BatchLayoutValidationErrorCode::invalid_run_mask);
  }
  const std::uint64_t maximum_clear_count =
      static_cast<std::uint64_t>(tile_runs.csr_run_destination_tiles.size()) +
      static_cast<std::uint64_t>(tile_runs.csc_run_source_tiles.size());
  std::uint64_t orientation_initializations =
      report.csr_retained_entries_initialized;
  std::uint64_t orientation_clears = report.csr_retained_entries_cleared;
  std::uint64_t orientation_retained_writes =
      report.csr_retained_entries_written;
  std::uint64_t orientation_descriptor_writes =
      report.csr_descriptor_entries_written;
  if (!add_u64(
          orientation_initializations,
          report.csc_retained_entries_initialized) ||
      !add_u64(
          orientation_clears, report.csc_retained_entries_cleared) ||
      !add_u64(
          orientation_retained_writes,
          report.csc_retained_entries_written) ||
      !add_u64(
          orientation_descriptor_writes,
          report.csc_descriptor_entries_written)) {
    return layout_error(BatchLayoutValidationErrorCode::invalid_run_mask);
  }
  if (report.csr_runs_visited != csr.coverage.visited ||
      report.csc_runs_visited != csc.coverage.visited ||
      report.active_csr_runs != csr.coverage.active ||
      report.active_csc_runs != csc.coverage.active ||
      report.csr_lane_edge_pairs != csr.coverage.lane_edge_pairs ||
      report.csc_lane_edge_pairs != csc.coverage.lane_edge_pairs ||
      report.retained_entries_initialized > maximum_clear_count ||
      report.retained_entries_cleared > maximum_clear_count ||
      report.csr_retained_entries_initialized >
          tile_runs.csr_run_destination_tiles.size() ||
      report.csc_retained_entries_initialized >
          tile_runs.csc_run_source_tiles.size() ||
      report.csr_retained_entries_cleared >
          tile_runs.csr_run_destination_tiles.size() ||
      report.csc_retained_entries_cleared >
          tile_runs.csc_run_source_tiles.size() ||
      orientation_initializations != report.retained_entries_initialized ||
      orientation_clears != report.retained_entries_cleared ||
      orientation_retained_writes != report.retained_entries_written ||
      orientation_descriptor_writes != report.descriptor_entries_written ||
      report.retained_entries_written != (retained ? active_total : 0U) ||
      report.descriptor_entries_written != (retained ? 0U : active_total) ||
      (!retained && report.retained_entries_initialized != 0U) ||
      report.csr_retained_entries_written !=
          (retained ? csr.coverage.active : 0U) ||
      report.csc_retained_entries_written !=
          (retained ? csc.coverage.active : 0U) ||
      report.csr_descriptor_entries_written !=
          (retained ? 0U : csr.coverage.active) ||
      report.csc_descriptor_entries_written !=
          (retained ? 0U : csc.coverage.active)) {
    return layout_error(BatchLayoutValidationErrorCode::invalid_run_mask);
  }

  if (csr.coverage.union_edges != batch.union_edge_estimate ||
      csc.coverage.union_edges != batch.union_edge_estimate) {
    return layout_error(
        BatchLayoutValidationErrorCode::admitted_edge_coverage_mismatch);
  }
  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    const std::uint64_t expected =
        (batch.valid_lane_mask & lane_bit(lane)) == 0U
            ? 0U
            : features[batch.query_indices_by_lane[lane]]
                  .selected_edge_estimate;
    if (csr.coverage.edges_by_lane[lane] != expected ||
        csc.coverage.edges_by_lane[lane] != expected ||
        description.selected_edge_estimates_by_lane[lane] != expected) {
      return layout_error(
          BatchLayoutValidationErrorCode::admitted_edge_coverage_mismatch,
          lane);
    }
  }
  return {};
}

}  // namespace

void prepare_batch_device_description(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const std::span<const RouteQuery> queries,
    const std::span<const BatchQueryFeatures> features,
    const BatchPlanEntry& batch,
    const BatchRunRepresentation run_representation,
    BatchDeviceDescription& output) {
  if (!preparation_shapes_are_safe(graph, tile_runs)) {
    throw std::invalid_argument{
        "batch preparation requires a valid graph/tile-run top-level shape"};
  }
  const ResolvedBatchInputs inputs =
      validate_batch_inputs(graph, queries, features, batch);
  if (!inputs.validation.ok()) {
    throw std::invalid_argument{
        "batch preparation received an invalid batch plan entry"};
  }
  if (run_representation !=
          BatchRunRepresentation::retained_per_run_masks &&
      run_representation !=
          BatchRunRepresentation::compact_nonzero_descriptors &&
      run_representation !=
          BatchRunRepresentation::device_materialized_run_masks) {
    throw std::invalid_argument{"unknown batch run representation"};
  }
  const bool run_storage_was_initialized =
      output.run_representation_initialized;
  if (run_storage_was_initialized &&
      output.run_representation != run_representation) {
    throw std::invalid_argument{
        "a reusable batch description cannot switch run representations"};
  }
  static_cast<void>(checked_size32(
      tile_runs.csr_run_destination_tiles.size(),
      "CSR tile-run count exceeds 32 bits"));
  static_cast<void>(checked_size32(
      tile_runs.csc_run_source_tiles.size(),
      "CSC tile-run count exceeds 32 bits"));

  std::uint64_t flattened_source_count = 0U;
  std::uint64_t flattened_target_count = 0U;
  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    const RouteQuery* const query = inputs.queries_by_lane[lane];
    if (query == nullptr) {
      continue;
    }
    checked_add_u64(
        flattened_source_count,
        static_cast<std::uint64_t>(query->sources.size()),
        "flattened source count overflowed");
    checked_add_u64(
        flattened_target_count,
        static_cast<std::uint64_t>(query->targets.size()),
        "flattened target count overflowed");
  }
  if (flattened_source_count > std::numeric_limits<std::uint32_t>::max() ||
      flattened_target_count > std::numeric_limits<std::uint32_t>::max() ||
      !std::in_range<std::size_t>(flattened_source_count) ||
      !std::in_range<std::size_t>(flattened_target_count)) {
    throw std::overflow_error{
        "flattened batch terminal payload exceeds 32-bit offsets"};
  }

  output.lane_width = batch.lane_width;
  output.valid_lane_mask = batch.valid_lane_mask;
  output.reached_lane_mask = 0U;
  output.miss_lane_mask = 0U;
  // Preserve the reusable dense allocation and its old touched ledger. The
  // mapping is stale as soon as a new batch image starts being prepared and
  // is rebuilt explicitly (and timed separately) on compact-storage paths.
  output.compact_vertex_mapping_valid = false;
  output.compact_mapping_report = {};
  output.query_ids_by_lane.resize(batch.lane_width);
  output.expansion_generations_by_lane.resize(batch.lane_width);
  output.selected_vertex_counts_by_lane.resize(batch.lane_width);
  output.selected_edge_estimates_by_lane.resize(batch.lane_width);
  output.source_offsets.resize(static_cast<std::size_t>(batch.lane_width) + 1U);
  output.target_offsets.resize(static_cast<std::size_t>(batch.lane_width) + 1U);
  output.sources.clear();
  output.targets.clear();
  output.sources.reserve(static_cast<std::size_t>(flattened_source_count));
  output.targets.reserve(static_cast<std::size_t>(flattened_target_count));
  output.source_offsets[0U] = 0U;
  output.target_offsets[0U] = 0U;

  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    output.query_ids_by_lane[lane] = batch.query_ids_by_lane[lane].value();
    output.expansion_generations_by_lane[lane] =
        batch.expansion_generations_by_lane[lane];
    const RouteQuery* const query = inputs.queries_by_lane[lane];
    if (query == nullptr) {
      output.selected_vertex_counts_by_lane[lane] = 0U;
      output.selected_edge_estimates_by_lane[lane] = 0U;
    } else {
      const BatchQueryFeatures& feature =
          features[batch.query_indices_by_lane[lane]];
      output.selected_vertex_counts_by_lane[lane] =
          feature.selected_vertex_count;
      output.selected_edge_estimates_by_lane[lane] =
          feature.selected_edge_estimate;
      for (const VertexId source : query->sources) {
        output.sources.push_back(source.value());
      }
      for (const VertexId target : query->targets) {
        output.targets.push_back(target.value());
      }
    }
    output.source_offsets[lane + 1U] = checked_size32(
        output.sources.size(), "flattened source count exceeds 32 bits");
    output.target_offsets[lane + 1U] = checked_size32(
        output.targets.size(), "flattened target count exceeds 32 bits");
  }

  if (output.tile_lane_masks.size() != graph.tile_coordinates().size()) {
    output.tile_lane_masks.assign(graph.tile_coordinates().size(), 0U);
  } else {
    std::uint32_t preceding_tile = 0U;
    bool has_preceding_tile = false;
    for (const std::uint32_t tile : output.union_tiles) {
      if (tile >= output.tile_lane_masks.size() ||
          (has_preceding_tile && tile <= preceding_tile)) {
        throw std::invalid_argument{
            "reused union-tile mask ledger is internally inconsistent"};
      }
      output.tile_lane_masks[tile] = 0U;
      preceding_tile = tile;
      has_preceding_tile = true;
    }
  }
  output.union_tiles.clear();
  output.union_tiles.reserve(batch.union_tiles.size());
  output.selected_vertex_ranges.clear();
  output.selected_vertex_ranges.reserve(batch.union_tiles.size());
  const auto tile_offsets = graph.tile_vertex_offsets();
  for (std::size_t position = 0U; position < batch.union_tiles.size();
       ++position) {
    const TileId tile = batch.union_tiles[position];
    const LaneMask mask = batch.union_tile_lane_masks[position];
    output.union_tiles.push_back(tile.value());
    output.tile_lane_masks[tile.value()] = mask;
    output.selected_vertex_ranges.push_back(BatchVertexRange{
        checked_device_offset32(tile_offsets[tile.value()]),
        checked_device_offset32(tile_offsets[tile.value() + 1U]),
        mask,
    });
  }

  BatchRunPreparationReport report{};
  clear_prior_touched(
      output.csr_run_lane_masks,
      output.touched_csr_runs,
      report.retained_entries_cleared,
      report.csr_retained_entries_cleared);
  clear_prior_touched(
      output.csc_run_lane_masks,
      output.touched_csc_runs,
      report.retained_entries_cleared,
      report.csc_retained_entries_cleared);
  output.run_representation = run_representation;
  output.run_report = report;
  output.csr_run_descriptors.clear();
  output.csc_run_descriptors.clear();
  if (run_representation ==
      BatchRunRepresentation::retained_per_run_masks) {
    output.csr_descriptor_offsets_by_union_vertex.clear();
    output.csc_descriptor_offsets_by_union_vertex.clear();
    if (!run_storage_was_initialized ||
        output.csr_run_lane_masks.size() !=
        tile_runs.csr_run_destination_tiles.size()) {
      output.csr_run_lane_masks.assign(
          tile_runs.csr_run_destination_tiles.size(), 0U);
      output.run_report.csr_retained_entries_initialized =
          tile_runs.csr_run_destination_tiles.size();
    }
    if (!run_storage_was_initialized ||
        output.csc_run_lane_masks.size() !=
        tile_runs.csc_run_source_tiles.size()) {
      output.csc_run_lane_masks.assign(
          tile_runs.csc_run_source_tiles.size(), 0U);
      output.run_report.csc_retained_entries_initialized =
          tile_runs.csc_run_source_tiles.size();
    }
    output.run_report.retained_entries_initialized =
        output.run_report.csr_retained_entries_initialized;
    checked_add_u64(
        output.run_report.retained_entries_initialized,
        output.run_report.csc_retained_entries_initialized,
        "retained run-mask initialization count overflowed");
  } else if (run_representation ==
             BatchRunRepresentation::compact_nonzero_descriptors) {
    output.csr_run_lane_masks.clear();
    output.csc_run_lane_masks.clear();
    output.touched_csr_runs.clear();
    output.touched_csc_runs.clear();
    if (batch.union_vertex_count == std::numeric_limits<std::uint64_t>::max() ||
        !std::in_range<std::size_t>(batch.union_vertex_count + 1U)) {
      throw std::overflow_error{
          "compact descriptor vertex-offset count exceeds host size"};
    }
    const std::size_t descriptor_offset_count =
        static_cast<std::size_t>(batch.union_vertex_count + 1U);
    output.csr_descriptor_offsets_by_union_vertex.clear();
    output.csc_descriptor_offsets_by_union_vertex.clear();
    output.csr_descriptor_offsets_by_union_vertex.reserve(
        descriptor_offset_count);
    output.csc_descriptor_offsets_by_union_vertex.reserve(
        descriptor_offset_count);
    output.csr_descriptor_offsets_by_union_vertex.push_back(0U);
    output.csc_descriptor_offsets_by_union_vertex.push_back(0U);
  } else {
    output.csr_run_lane_masks.clear();
    output.csc_run_lane_masks.clear();
    output.touched_csr_runs.clear();
    output.touched_csc_runs.clear();
    output.csr_run_descriptors.clear();
    output.csc_run_descriptors.clear();
    output.csr_descriptor_offsets_by_union_vertex.clear();
    output.csc_descriptor_offsets_by_union_vertex.clear();
  }

  if (run_representation !=
      BatchRunRepresentation::device_materialized_run_masks) {
    const auto [csr, csc] = build_run_storage(graph, tile_runs, output);
    if (csr.union_edges != batch.union_edge_estimate ||
        csc.union_edges != batch.union_edge_estimate) {
      throw std::invalid_argument{
          "batch union edge estimate does not match admitted tile runs"};
    }
    for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
      const std::uint64_t expected =
          output.selected_edge_estimates_by_lane[lane];
      if (csr.edges_by_lane[lane] != expected ||
          csc.edges_by_lane[lane] != expected) {
        throw std::invalid_argument{
            "batch per-lane edge estimate does not match admitted tile runs"};
      }
    }
  }
  output.run_representation_initialized = true;
}

void prepare_compact_vertex_mapping(
    const WeightedGraph& graph,
    const BatchPlanEntry& batch,
    BatchDeviceDescription& output) {
  output.compact_vertex_mapping_valid = false;
  output.compact_mapping_report = {};
  const std::size_t tile_count = graph.tile_coordinates().size();
  const auto tile_offsets = graph.tile_vertex_offsets();
  if (!graph.has_spatial_ordering() || tile_count == 0U ||
      tile_count == std::numeric_limits<std::size_t>::max() ||
      tile_offsets.size() != tile_count + 1U || tile_offsets.empty() ||
      tile_offsets.front() != 0U ||
      tile_offsets.back() != graph.vertex_count() ||
      output.lane_width != batch.lane_width ||
      output.union_tiles.size() != batch.union_tiles.size() ||
      output.selected_vertex_ranges.size() != batch.union_tiles.size() ||
      batch.union_tiles.empty()) {
    throw std::invalid_argument{
        "compact vertex mapping requires a matching prepared batch"};
  }

  std::uint64_t packed_begin = 0U;
  for (std::size_t position = 0U; position < batch.union_tiles.size();
       ++position) {
    const TileId tile = batch.union_tiles[position];
    if (tile.value() >= tile_count ||
        (position != 0U && !(batch.union_tiles[position - 1U] < tile)) ||
        output.union_tiles[position] != tile.value()) {
      throw std::invalid_argument{
          "compact vertex mapping requires canonical matching union tiles"};
    }
    const EdgeOffset global_begin = tile_offsets[tile.value()];
    const EdgeOffset global_end = tile_offsets[tile.value() + 1U];
    if (global_begin > global_end || global_end > graph.vertex_count() ||
        global_begin < packed_begin ||
        !std::in_range<std::uint32_t>(global_begin) ||
        !std::in_range<std::uint32_t>(global_end) ||
        output.selected_vertex_ranges[position].begin != global_begin ||
        output.selected_vertex_ranges[position].end != global_end) {
      throw std::invalid_argument{
          "compact vertex mapping disagrees with immutable tile ranges"};
    }
    checked_add_u64(
        packed_begin,
        global_end - global_begin,
        "compact union vertex count overflowed");
  }
  if (packed_begin != batch.union_vertex_count ||
      !std::in_range<std::uint32_t>(packed_begin)) {
    throw std::invalid_argument{
        "compact vertex mapping disagrees with the union vertex count"};
  }

  if (output.compact_vertex_biases_by_tile.size() != tile_count) {
    output.compact_vertex_biases_by_tile.assign(tile_count, 0U);
    output.touched_compact_tiles.clear();
    output.compact_mapping_report.entries_initialized = tile_count;
  } else {
    std::uint32_t preceding = 0U;
    bool has_preceding = false;
    for (const std::uint32_t tile : output.touched_compact_tiles) {
      if (tile >= tile_count || (has_preceding && tile <= preceding)) {
        throw std::invalid_argument{
            "reused compact tile-mapping ledger is internally inconsistent"};
      }
      output.compact_vertex_biases_by_tile[tile] = 0U;
      checked_add_u64(
          output.compact_mapping_report.entries_cleared,
          1U,
          "compact tile-mapping clear count overflowed");
      preceding = tile;
      has_preceding = true;
    }
    output.touched_compact_tiles.clear();
  }

  output.touched_compact_tiles.reserve(batch.union_tiles.size());
  packed_begin = 0U;
  for (const TileId tile : batch.union_tiles) {
    const EdgeOffset global_begin = tile_offsets[tile.value()];
    const EdgeOffset global_end = tile_offsets[tile.value() + 1U];
    const std::uint64_t bias = global_begin - packed_begin;
    output.compact_vertex_biases_by_tile[tile.value()] =
        static_cast<std::uint32_t>(bias);
    output.touched_compact_tiles.push_back(tile.value());
    checked_add_u64(
        output.compact_mapping_report.entries_written,
        1U,
        "compact tile-mapping write count overflowed");
    packed_begin += global_end - global_begin;
  }
  output.compact_vertex_mapping_valid = true;
}

BatchLayoutValidationResult validate_compact_vertex_mapping(
    const WeightedGraph& graph,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description) noexcept {
  const std::size_t tile_count = graph.tile_coordinates().size();
  const auto tile_offsets = graph.tile_vertex_offsets();
  if (!description.compact_vertex_mapping_valid ||
      !graph.has_spatial_ordering() || tile_count == 0U ||
      tile_count == std::numeric_limits<std::size_t>::max() ||
      tile_offsets.size() != tile_count + 1U || tile_offsets.empty() ||
      tile_offsets.front() != 0U ||
      tile_offsets.back() != graph.vertex_count() ||
      description.compact_vertex_biases_by_tile.size() != tile_count ||
      description.touched_compact_tiles.size() != batch.union_tiles.size() ||
      description.union_tiles.size() != batch.union_tiles.size() ||
      description.selected_vertex_ranges.size() != batch.union_tiles.size() ||
      description.compact_mapping_report.entries_written !=
          batch.union_tiles.size() ||
      description.compact_mapping_report.entries_initialized > tile_count ||
      description.compact_mapping_report.entries_cleared > tile_count) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_compact_vertex_mapping);
  }

  std::uint64_t packed_begin = 0U;
  for (std::size_t position = 0U; position < batch.union_tiles.size();
       ++position) {
    const TileId tile = batch.union_tiles[position];
    if (tile.value() >= tile_count ||
        (position != 0U && !(batch.union_tiles[position - 1U] < tile)) ||
        description.union_tiles[position] != tile.value() ||
        description.touched_compact_tiles[position] != tile.value()) {
      return layout_error(
          BatchLayoutValidationErrorCode::invalid_compact_vertex_mapping,
          position);
    }
    const EdgeOffset global_begin = tile_offsets[tile.value()];
    const EdgeOffset global_end = tile_offsets[tile.value() + 1U];
    if (global_begin > global_end || global_end > graph.vertex_count() ||
        global_begin < packed_begin ||
        !std::in_range<std::uint32_t>(global_begin) ||
        !std::in_range<std::uint32_t>(global_end) ||
        description.selected_vertex_ranges[position].begin != global_begin ||
        description.selected_vertex_ranges[position].end != global_end ||
        description.compact_vertex_biases_by_tile[tile.value()] !=
            global_begin - packed_begin) {
      return layout_error(
          BatchLayoutValidationErrorCode::invalid_compact_vertex_mapping,
          position);
    }
    if (!add_u64(packed_begin, global_end - global_begin)) {
      return layout_error(
          BatchLayoutValidationErrorCode::invalid_compact_vertex_mapping,
          position);
    }
  }
  if (packed_begin != batch.union_vertex_count ||
      !std::in_range<std::uint32_t>(packed_begin)) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_compact_vertex_mapping);
  }

  std::size_t touched_position = 0U;
  for (std::size_t tile = 0U; tile < tile_count; ++tile) {
    if (touched_position < description.touched_compact_tiles.size() &&
        description.touched_compact_tiles[touched_position] == tile) {
      ++touched_position;
    } else if (description.compact_vertex_biases_by_tile[tile] != 0U) {
      return layout_error(
          BatchLayoutValidationErrorCode::invalid_compact_vertex_mapping,
          tile);
    }
  }
  if (touched_position != description.touched_compact_tiles.size()) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_compact_vertex_mapping);
  }
  return {};
}

BatchLayoutValidationResult validate_batch_device_description(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const std::span<const RouteQuery> queries,
    const std::span<const BatchQueryFeatures> features,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description) noexcept {
  try {
    if (!validate_tile_run_layout(graph, tile_runs).ok()) {
      return layout_error(BatchLayoutValidationErrorCode::invalid_tile_runs);
    }
  } catch (...) {
    return layout_error(BatchLayoutValidationErrorCode::invalid_tile_runs);
  }

  const ResolvedBatchInputs inputs =
      validate_batch_inputs(graph, queries, features, batch);
  if (!inputs.validation.ok()) {
    return inputs.validation;
  }
  const std::size_t lane_width = batch.lane_width;
  if (!std::in_range<std::uint32_t>(
          tile_runs.csr_run_destination_tiles.size()) ||
      !std::in_range<std::uint32_t>(
          tile_runs.csc_run_source_tiles.size())) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_run_storage_shape);
  }
  if (description.lane_width != batch.lane_width ||
      description.valid_lane_mask != batch.valid_lane_mask ||
      !description.run_representation_initialized ||
      description.query_ids_by_lane.size() != lane_width ||
      description.expansion_generations_by_lane.size() != lane_width ||
      description.selected_vertex_counts_by_lane.size() != lane_width ||
      description.selected_edge_estimates_by_lane.size() != lane_width) {
    return layout_error(BatchLayoutValidationErrorCode::invalid_batch_shape);
  }
  for (std::size_t lane = 0U; lane < lane_width; ++lane) {
    if (description.query_ids_by_lane[lane] !=
            batch.query_ids_by_lane[lane].value() ||
        description.expansion_generations_by_lane[lane] !=
            batch.expansion_generations_by_lane[lane]) {
      return layout_error(
          BatchLayoutValidationErrorCode::invalid_lane_identity, lane);
    }
  }

  if (!valid_flat_offsets(
          description.source_offsets,
          lane_width,
          description.sources.size(),
          description.valid_lane_mask)) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_source_offsets);
  }
  if (!valid_flat_offsets(
          description.target_offsets,
          lane_width,
          description.targets.size(),
          description.valid_lane_mask)) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_target_offsets);
  }
  for (std::size_t lane = 0U; lane < lane_width; ++lane) {
    const RouteQuery* const query = inputs.queries_by_lane[lane];
    const std::size_t source_begin = description.source_offsets[lane];
    const std::size_t source_end = description.source_offsets[lane + 1U];
    const std::size_t target_begin = description.target_offsets[lane];
    const std::size_t target_end = description.target_offsets[lane + 1U];
    if (query == nullptr) {
      continue;
    }
    if (source_end - source_begin != query->sources.size() ||
        target_end - target_begin != query->targets.size()) {
      return layout_error(
          BatchLayoutValidationErrorCode::invalid_terminal_payload, lane);
    }
    for (std::size_t position = 0U; position < query->sources.size();
         ++position) {
      if (description.sources[source_begin + position] !=
          query->sources[position].value()) {
        return layout_error(
            BatchLayoutValidationErrorCode::invalid_terminal_payload, lane);
      }
    }
    for (std::size_t position = 0U; position < query->targets.size();
         ++position) {
      if (description.targets[target_begin + position] !=
          query->targets[position].value()) {
        return layout_error(
            BatchLayoutValidationErrorCode::invalid_terminal_payload, lane);
      }
    }
  }

  for (std::size_t lane = 0U; lane < lane_width; ++lane) {
    const std::uint64_t expected_vertices =
        (batch.valid_lane_mask & lane_bit(lane)) == 0U
            ? 0U
            : features[batch.query_indices_by_lane[lane]]
                  .selected_vertex_count;
    const std::uint64_t expected_edges =
        (batch.valid_lane_mask & lane_bit(lane)) == 0U
            ? 0U
            : features[batch.query_indices_by_lane[lane]]
                  .selected_edge_estimate;
    if (description.selected_vertex_counts_by_lane[lane] !=
            expected_vertices ||
        description.selected_edge_estimates_by_lane[lane] != expected_edges) {
      return layout_error(
          BatchLayoutValidationErrorCode::invalid_estimates, lane);
    }
  }

  if (description.union_tiles.size() != batch.union_tiles.size()) {
    return layout_error(BatchLayoutValidationErrorCode::invalid_union_tiles);
  }
  for (std::size_t position = 0U; position < batch.union_tiles.size();
       ++position) {
    if (description.union_tiles[position] !=
        batch.union_tiles[position].value()) {
      return layout_error(
          BatchLayoutValidationErrorCode::invalid_union_tiles, position);
    }
  }
  if (description.tile_lane_masks.size() !=
      graph.tile_coordinates().size()) {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_tile_lane_masks);
  }
  std::size_t union_position = 0U;
  for (std::size_t tile = 0U; tile < description.tile_lane_masks.size();
       ++tile) {
    LaneMask expected = 0U;
    if (union_position < batch.union_tiles.size() &&
        batch.union_tiles[union_position].value() == tile) {
      expected = batch.union_tile_lane_masks[union_position];
      ++union_position;
    }
    if (description.tile_lane_masks[tile] != expected) {
      return layout_error(
          BatchLayoutValidationErrorCode::invalid_tile_lane_masks, tile);
    }
  }

  if (description.selected_vertex_ranges.size() != batch.union_tiles.size()) {
    return layout_error(BatchLayoutValidationErrorCode::invalid_vertex_ranges);
  }
  const auto tile_offsets = graph.tile_vertex_offsets();
  for (std::size_t position = 0U; position < batch.union_tiles.size();
       ++position) {
    const TileId tile = batch.union_tiles[position];
    if (!std::in_range<std::uint32_t>(tile_offsets[tile.value()]) ||
        !std::in_range<std::uint32_t>(
            tile_offsets[tile.value() + 1U])) {
      return layout_error(
          BatchLayoutValidationErrorCode::invalid_vertex_ranges, position);
    }
    const BatchVertexRange expected{
        static_cast<std::uint32_t>(tile_offsets[tile.value()]),
        static_cast<std::uint32_t>(tile_offsets[tile.value() + 1U]),
        batch.union_tile_lane_masks[position],
    };
    if (description.selected_vertex_ranges[position] != expected) {
      return layout_error(
          BatchLayoutValidationErrorCode::invalid_vertex_ranges, position);
    }
  }
  if (description.compact_vertex_mapping_valid) {
    const BatchLayoutValidationResult mapping_validation =
        validate_compact_vertex_mapping(graph, batch, description);
    if (!mapping_validation.ok()) {
      return mapping_validation;
    }
  }

  BatchLayoutValidationResult storage_validation;
  if (description.run_representation ==
      BatchRunRepresentation::retained_per_run_masks) {
    storage_validation = validate_retained_shape(
        description,
        tile_runs.csr_run_destination_tiles.size(),
        tile_runs.csc_run_source_tiles.size());
  } else if (description.run_representation ==
             BatchRunRepresentation::compact_nonzero_descriptors) {
    storage_validation = validate_descriptor_shape(
        description,
        tile_runs.csr_run_destination_tiles.size(),
        tile_runs.csc_run_source_tiles.size(),
        batch.union_vertex_count);
  } else if (description.run_representation ==
             BatchRunRepresentation::device_materialized_run_masks) {
    storage_validation = validate_device_materialized_shape(description);
  } else {
    return layout_error(
        BatchLayoutValidationErrorCode::invalid_run_storage_shape);
  }
  if (!storage_validation.ok()) {
    return storage_validation;
  }
  const BatchLayoutValidationResult run_validation =
      validate_runs_and_coverage(
          tile_runs, features, batch, description);
  if (!run_validation.ok()) {
    return run_validation;
  }
  if (description.reached_lane_mask != 0U ||
      description.miss_lane_mask != 0U) {
    return layout_error(
        BatchLayoutValidationErrorCode::nonzero_initial_result_mask);
  }
  return {};
}

RunAdmissionProofResult prove_batch_endpoint_admission(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const BatchDeviceDescription& description) {
  TileRunLaneMasks expanded;
  if (description.run_representation ==
      BatchRunRepresentation::retained_per_run_masks) {
    expanded.csr_run_masks.assign(
        description.csr_run_lane_masks.begin(),
        description.csr_run_lane_masks.end());
    expanded.csc_run_masks.assign(
        description.csc_run_lane_masks.begin(),
        description.csc_run_lane_masks.end());
  } else if (description.run_representation ==
             BatchRunRepresentation::compact_nonzero_descriptors) {
    expanded.csr_run_masks.assign(
        tile_runs.csr_run_destination_tiles.size(), 0U);
    expanded.csc_run_masks.assign(tile_runs.csc_run_source_tiles.size(), 0U);
    std::uint32_t preceding = 0U;
    bool has_preceding = false;
    for (const RunLaneMaskDescriptor descriptor :
         description.csr_run_descriptors) {
      if (descriptor.run_id >= expanded.csr_run_masks.size() ||
          descriptor.lane_mask == 0U ||
          (has_preceding && descriptor.run_id <= preceding)) {
        return RunAdmissionProofResult{
            RunAdmissionProofErrorCode::run_lane_mask_size_mismatch,
            RunAdmissionProofResult::no_position};
      }
      expanded.csr_run_masks[descriptor.run_id] = descriptor.lane_mask;
      preceding = descriptor.run_id;
      has_preceding = true;
    }
    preceding = 0U;
    has_preceding = false;
    for (const RunLaneMaskDescriptor descriptor :
         description.csc_run_descriptors) {
      if (descriptor.run_id >= expanded.csc_run_masks.size() ||
          descriptor.lane_mask == 0U ||
          (has_preceding && descriptor.run_id <= preceding)) {
        return RunAdmissionProofResult{
            RunAdmissionProofErrorCode::run_lane_mask_size_mismatch,
            RunAdmissionProofResult::no_position};
      }
      expanded.csc_run_masks[descriptor.run_id] = descriptor.lane_mask;
      preceding = descriptor.run_id;
      has_preceding = true;
    }
  } else if (description.run_representation ==
             BatchRunRepresentation::device_materialized_run_masks) {
    compute_tile_run_lane_masks(
        graph, tile_runs, description.tile_lane_masks, expanded);
  } else {
    return RunAdmissionProofResult{
        RunAdmissionProofErrorCode::run_lane_mask_size_mismatch,
        RunAdmissionProofResult::no_position};
  }
  return prove_run_admission_equivalence(
      graph, tile_runs, description.tile_lane_masks, expanded);
}

}  // namespace bfnew
