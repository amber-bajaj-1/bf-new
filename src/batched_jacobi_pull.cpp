#include "bfnew/batched_jacobi_pull.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bfnew {
namespace {

[[nodiscard]] constexpr LaneMask lane_bit(const std::size_t lane) noexcept {
  return LaneMask{1U} << static_cast<std::uint32_t>(lane);
}

[[nodiscard]] constexpr LaneMask width_mask(
    const std::uint32_t width) noexcept {
  return width == 32U ? std::numeric_limits<LaneMask>::max()
                      : (LaneMask{1U} << width) - LaneMask{1U};
}

[[nodiscard]] constexpr bool is_low_lane_mask(const LaneMask mask) noexcept {
  return mask != 0U && (mask & (mask + LaneMask{1U})) == 0U;
}

void require(const bool condition, const char* const message) {
  if (!condition) {
    throw std::invalid_argument{message};
  }
}

void checked_add(
    std::uint64_t& value,
    const std::uint64_t increment,
    const char* const message) {
  if (value > std::numeric_limits<std::uint64_t>::max() - increment) {
    throw std::overflow_error{message};
  }
  value += increment;
}

[[nodiscard]] std::uint64_t checked_product(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* const message) {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error{message};
  }
  return left * right;
}

[[nodiscard]] bool valid_offsets(
    const std::span<const std::uint32_t> offsets,
    const std::size_t bucket_count,
    const std::size_t entry_count) noexcept {
  return offsets.size() == bucket_count + 1U && !offsets.empty() &&
         offsets.front() == 0U && offsets.back() == entry_count &&
         std::is_sorted(offsets.begin(), offsets.end());
}

[[nodiscard]] bool canonical_vertices(
    const std::span<const VertexId> vertices,
    const std::uint32_t vertex_count) noexcept {
  if (vertices.empty()) {
    return false;
  }
  for (std::size_t position = 0U; position < vertices.size(); ++position) {
    if (vertices[position].value() >= vertex_count ||
        (position != 0U && !(vertices[position - 1U] < vertices[position]))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool canonical_tiles(
    const std::span<const TileId> tiles,
    const std::uint32_t tile_count) noexcept {
  if (tiles.empty()) {
    return false;
  }
  for (std::size_t position = 0U; position < tiles.size(); ++position) {
    if (tiles[position].value() >= tile_count ||
        (position != 0U && !(tiles[position - 1U] < tiles[position]))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool terminal_payload_matches(
    const std::span<const std::uint32_t> payload,
    const std::uint32_t begin,
    const std::uint32_t end,
    const std::span<const VertexId> expected) noexcept {
  if (begin > end || end > payload.size() || end - begin != expected.size()) {
    return false;
  }
  for (std::size_t position = 0U; position < expected.size(); ++position) {
    if (payload[static_cast<std::size_t>(begin) + position] !=
        expected[position].value()) {
      return false;
    }
  }
  return true;
}

struct ValidatedHostBatch {
  std::array<const RouteQuery*, maximum_batch_lanes> queries_by_lane{};
  std::array<std::uint64_t, maximum_batch_lanes>
      selected_vertices_by_lane{};
  std::array<std::uint64_t, maximum_batch_lanes> selected_edges_by_lane{};
  std::vector<std::size_t> compact_index_by_vertex;
  std::uint64_t selected_lane_vertices{};
  std::uint64_t selected_lane_edges{};
  std::uint64_t selected_tile_lane_positions{};
};

[[nodiscard]] ValidatedHostBatch validate_host_batch(
    const DeviceGraphLayout32& graph,
    const std::span<const RouteQuery> queries,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description,
    const GpuRunOptions& options) {
  require(
      validate_gpu_run_options(options) == GpuRunOptionsError::none,
      "batched host Jacobi requires valid run options");
  require(
      options.engine == EngineKind::jacobi_pull,
      "batched host Jacobi rejects another engine kind");
  require(
      supported_batched_jacobi_width(batch.lane_width),
      "batched host Jacobi supports widths 1, 8, 16, and 32 only");

  const std::size_t width = batch.lane_width;
  require(
      batch.query_indices_by_lane.size() == width &&
          batch.query_ids_by_lane.size() == width &&
          batch.expansion_generations_by_lane.size() == width &&
          batch.union_tiles.size() == batch.union_tile_lane_masks.size() &&
          !batch.union_tiles.empty() && batch.valid_lane_mask != 0U &&
          is_low_lane_mask(batch.valid_lane_mask) &&
          (batch.valid_lane_mask & ~width_mask(batch.lane_width)) == 0U,
      "batched host Jacobi received an invalid plan entry shape");
  require(
      description.lane_width == batch.lane_width &&
          description.valid_lane_mask == batch.valid_lane_mask &&
          description.reached_lane_mask == 0U &&
          description.miss_lane_mask == 0U &&
          description.query_ids_by_lane.size() == width &&
          description.expansion_generations_by_lane.size() == width &&
          description.selected_vertex_counts_by_lane.size() == width &&
          description.selected_edge_estimates_by_lane.size() == width &&
          valid_offsets(
              description.source_offsets, width, description.sources.size()) &&
          valid_offsets(
              description.target_offsets, width, description.targets.size()) &&
          description.run_representation_initialized,
      "batched host Jacobi received an invalid device description shape");

  const std::size_t vertex_count = graph.vertex_count;
  const std::size_t edge_count = graph.edge_count;
  const std::size_t tile_count = graph.tile_count;
  const std::size_t run_count = graph.csc_run_source_tiles.size();
  require(
      vertex_count != 0U && tile_count != 0U &&
          graph.owner_tiles.size() == vertex_count &&
          graph.csc_sources.size() == edge_count &&
          graph.csc_weights.size() == edge_count &&
          valid_offsets(graph.csc_column_offsets, vertex_count, edge_count) &&
          valid_offsets(graph.csc_column_run_offsets, vertex_count, run_count) &&
          graph.csc_run_edge_offsets.size() == run_count + 1U &&
          !graph.csc_run_edge_offsets.empty() &&
          graph.csc_run_edge_offsets.front() == 0U &&
          graph.csc_run_edge_offsets.back() == edge_count &&
          std::is_sorted(
              graph.csc_run_edge_offsets.begin(),
              graph.csc_run_edge_offsets.end()) &&
          description.tile_lane_masks.size() == tile_count,
      "batched host Jacobi received an invalid CSC device graph");

  std::vector<std::uint32_t> tile_begins(tile_count, 0U);
  std::vector<std::uint32_t> tile_ends(tile_count, 0U);
  std::size_t owner_cursor = 0U;
  for (std::size_t tile = 0U; tile < tile_count; ++tile) {
    tile_begins[tile] = static_cast<std::uint32_t>(owner_cursor);
    while (owner_cursor < vertex_count &&
           graph.owner_tiles[owner_cursor] == tile) {
      ++owner_cursor;
    }
    tile_ends[tile] = static_cast<std::uint32_t>(owner_cursor);
    if (owner_cursor < vertex_count) {
      require(
          graph.owner_tiles[owner_cursor] < tile_count &&
              graph.owner_tiles[owner_cursor] > tile,
          "batched host Jacobi requires contiguous spatial tile ownership");
    }
  }
  require(
      owner_cursor == vertex_count,
      "batched host Jacobi owner tile is out of range");

  for (std::size_t destination = 0U; destination < vertex_count;
       ++destination) {
    const std::size_t edge_begin = graph.csc_column_offsets[destination];
    const std::size_t edge_end = graph.csc_column_offsets[destination + 1U];
    const std::size_t run_begin =
        graph.csc_column_run_offsets[destination];
    const std::size_t run_end =
        graph.csc_column_run_offsets[destination + 1U];
    std::size_t cursor = edge_begin;
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const std::size_t begin = graph.csc_run_edge_offsets[run];
      const std::size_t end = graph.csc_run_edge_offsets[run + 1U];
      require(
          begin == cursor && begin < end && end <= edge_end,
          "batched host Jacobi CSC runs do not cover a column exactly");
      const std::uint32_t source_tile = graph.csc_run_source_tiles[run];
      require(
          source_tile < tile_count,
          "batched host Jacobi CSC run source tile is out of range");
      for (std::size_t edge = begin; edge < end; ++edge) {
        const std::uint32_t source = graph.csc_sources[edge];
        require(
            source < vertex_count && graph.owner_tiles[source] == source_tile &&
                std::isfinite(graph.csc_weights[edge]) &&
                graph.csc_weights[edge] >= 0.0F,
            "batched host Jacobi CSC edge disagrees with its source-tile run");
      }
      cursor = end;
    }
    require(
        cursor == edge_end,
        "batched host Jacobi CSC runs omit part of a destination column");
  }

  ValidatedHostBatch validated;
  validated.compact_index_by_vertex.assign(
      vertex_count, std::numeric_limits<std::size_t>::max());
  std::vector<const RouteQuery*> queries_by_plan_index;
  queries_by_plan_index.reserve(queries.size());
  for (const RouteQuery& query : queries) {
    queries_by_plan_index.push_back(&query);
  }
  std::sort(
      queries_by_plan_index.begin(),
      queries_by_plan_index.end(),
      [](const RouteQuery* const left, const RouteQuery* const right) {
        return left->query_id < right->query_id;
      });
  require(
      std::adjacent_find(
          queries_by_plan_index.begin(),
          queries_by_plan_index.end(),
          [](const RouteQuery* const left, const RouteQuery* const right) {
            return left->query_id == right->query_id;
          }) == queries_by_plan_index.end(),
      "batched host Jacobi query IDs are not unique");
  std::vector<LaneMask> expected_tile_masks(tile_count, 0U);
  LaneMask resolved_query_ids = 0U;
  for (std::size_t lane = 0U; lane < width; ++lane) {
    const LaneMask bit = lane_bit(lane);
    const bool valid = (batch.valid_lane_mask & bit) != 0U;
    const std::uint32_t source_begin = description.source_offsets[lane];
    const std::uint32_t source_end = description.source_offsets[lane + 1U];
    const std::uint32_t target_begin = description.target_offsets[lane];
    const std::uint32_t target_end = description.target_offsets[lane + 1U];
    if (!valid) {
      require(
          batch.query_indices_by_lane[lane] == invalid_batch_query_index &&
              batch.query_ids_by_lane[lane] == invalid_batch_query_id &&
              batch.expansion_generations_by_lane[lane] == 0U &&
              description.query_ids_by_lane[lane] ==
                  invalid_batch_query_id.value() &&
              description.expansion_generations_by_lane[lane] == 0U &&
              description.selected_vertex_counts_by_lane[lane] == 0U &&
              description.selected_edge_estimates_by_lane[lane] == 0U &&
              source_begin == source_end && target_begin == target_end,
          "batched host Jacobi padded lane contains semantic payload");
      continue;
    }

    const std::uint32_t query_index = batch.query_indices_by_lane[lane];
    require(
        query_index != invalid_batch_query_index &&
            query_index < queries_by_plan_index.size(),
        "batched host Jacobi valid lane has an invalid query index");
    const QueryId query_id = batch.query_ids_by_lane[lane];
    const RouteQuery* const query = queries_by_plan_index[query_index];
    require(
        query->query_id == query_id &&
            description.query_ids_by_lane[lane] == query_id.value() &&
            description.expansion_generations_by_lane[lane] ==
                query->expansion_generation &&
            batch.expansion_generations_by_lane[lane] ==
                query->expansion_generation &&
            canonical_vertices(query->sources, graph.vertex_count) &&
            canonical_vertices(query->targets, graph.vertex_count) &&
            canonical_tiles(query->selected_tiles, graph.tile_count) &&
            terminal_payload_matches(
                description.sources,
                source_begin,
                source_end,
                query->sources) &&
            terminal_payload_matches(
                description.targets,
                target_begin,
                target_end,
                query->targets),
        "batched host Jacobi lane identity or terminal payload is invalid");
    for (std::size_t earlier_lane = 0U; earlier_lane < lane; ++earlier_lane) {
      if ((resolved_query_ids & lane_bit(earlier_lane)) != 0U) {
        require(
            batch.query_ids_by_lane[earlier_lane] != query_id,
            "batched host Jacobi assigns one query to multiple lanes");
      }
    }
    resolved_query_ids |= bit;
    validated.queries_by_lane[lane] = query;
    for (const TileId tile : query->selected_tiles) {
      expected_tile_masks[tile.value()] |= bit;
    }
  }

  std::size_t union_position = 0U;
  std::uint64_t union_vertex_count = 0U;
  for (std::size_t tile = 0U; tile < tile_count; ++tile) {
    const LaneMask expected_mask = expected_tile_masks[tile];
    require(
        description.tile_lane_masks[tile] == expected_mask,
        "batched host Jacobi tile mask differs from lane query tiles");
    if (expected_mask == 0U) {
      continue;
    }
    require(
        union_position < batch.union_tiles.size() &&
            union_position < description.union_tiles.size() &&
            union_position < description.selected_vertex_ranges.size() &&
            batch.union_tiles[union_position].value() == tile &&
            batch.union_tile_lane_masks[union_position] == expected_mask &&
            description.union_tiles[union_position] == tile,
        "batched host Jacobi union-tile ledger is not canonical");
    const BatchVertexRange range =
        description.selected_vertex_ranges[union_position];
    require(
        range.begin == tile_begins[tile] && range.end == tile_ends[tile] &&
            range.lane_mask == expected_mask,
        "batched host Jacobi selected vertex range differs from its tile");
    const std::uint64_t tile_vertices = range.end - range.begin;
    checked_add(
        union_vertex_count,
        tile_vertices,
        "batched host Jacobi union vertex count overflowed");
    checked_add(
        validated.selected_tile_lane_positions,
        static_cast<std::uint64_t>(std::popcount(expected_mask)),
        "batched host Jacobi selected tile/lane count overflowed");
    for (std::size_t lane = 0U; lane < width; ++lane) {
      if ((expected_mask & lane_bit(lane)) != 0U) {
        checked_add(
            validated.selected_vertices_by_lane[lane],
            tile_vertices,
            "batched host Jacobi per-lane vertex count overflowed");
      }
    }
    ++union_position;
  }
  require(
      union_position == batch.union_tiles.size() &&
          union_position == description.union_tiles.size() &&
          union_position == description.selected_vertex_ranges.size() &&
          batch.union_vertex_count == union_vertex_count,
      "batched host Jacobi union vertex count or ledger shape is invalid");

  for (std::size_t lane = 0U; lane < width; ++lane) {
    const LaneMask bit = lane_bit(lane);
    if ((batch.valid_lane_mask & bit) == 0U) {
      continue;
    }
    const RouteQuery& query = *validated.queries_by_lane[lane];
    for (const VertexId source : query.sources) {
      require(
          (expected_tile_masks[graph.owner_tiles[source.value()]] & bit) != 0U,
          "batched host Jacobi source is outside its lane tiles");
    }
    for (const VertexId target : query.targets) {
      require(
          (expected_tile_masks[graph.owner_tiles[target.value()]] & bit) != 0U,
          "batched host Jacobi target is outside its lane tiles");
    }
    require(
        description.selected_vertex_counts_by_lane[lane] ==
            validated.selected_vertices_by_lane[lane],
        "batched host Jacobi per-lane vertex estimate is inconsistent");
    checked_add(
        validated.selected_lane_vertices,
        validated.selected_vertices_by_lane[lane],
        "batched host Jacobi selected lane vertex count overflowed");
  }
  require(
      batch.selected_lane_vertex_count == validated.selected_lane_vertices,
      "batched host Jacobi plan selected lane vertex count is inconsistent");

  std::vector<RunLaneMaskDescriptor> expected_descriptors;
  std::vector<std::uint32_t> expected_descriptor_offsets{0U};
  std::vector<std::uint32_t> expected_touched_runs;
  std::uint64_t expected_runs_considered = 0U;
  std::uint64_t expected_active_runs = 0U;
  std::uint64_t expected_lane_edge_pairs = 0U;
  std::uint64_t expected_union_edges = 0U;
  std::size_t compact_vertex = 0U;
  for (std::size_t destination = 0U; destination < vertex_count;
       ++destination) {
    const std::uint32_t destination_tile = graph.owner_tiles[destination];
    const LaneMask destination_mask = expected_tile_masks[destination_tile];
    const std::size_t run_begin =
        graph.csc_column_run_offsets[destination];
    const std::size_t run_end =
        graph.csc_column_run_offsets[destination + 1U];
    if (destination_mask != 0U) {
      validated.compact_index_by_vertex[destination] = compact_vertex;
      ++compact_vertex;
      checked_add(
          expected_runs_considered,
          run_end - run_begin,
          "batched host Jacobi CSC run count overflowed");
    }
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const std::uint32_t source_tile = graph.csc_run_source_tiles[run];
      const LaneMask expected_mask =
          destination_mask & expected_tile_masks[source_tile];
      const std::uint64_t edge_count_in_run =
          static_cast<std::uint64_t>(graph.csc_run_edge_offsets[run + 1U]) -
          graph.csc_run_edge_offsets[run];
      if (destination_mask != 0U && expected_tile_masks[source_tile] != 0U) {
        checked_add(
            expected_union_edges,
            edge_count_in_run,
            "batched host Jacobi union edge count overflowed");
      }
      if (expected_mask == 0U) {
        continue;
      }
      ++expected_active_runs;
      expected_touched_runs.push_back(static_cast<std::uint32_t>(run));
      expected_descriptors.push_back(RunLaneMaskDescriptor{
          static_cast<std::uint32_t>(run), expected_mask});
      const std::uint64_t lane_pairs = checked_product(
          edge_count_in_run,
          static_cast<std::uint64_t>(std::popcount(expected_mask)),
          "batched host Jacobi lane-edge count overflowed");
      checked_add(
          expected_lane_edge_pairs,
          lane_pairs,
          "batched host Jacobi lane-edge count overflowed");
      for (std::size_t lane = 0U; lane < width; ++lane) {
        if ((expected_mask & lane_bit(lane)) != 0U) {
          checked_add(
              validated.selected_edges_by_lane[lane],
              edge_count_in_run,
              "batched host Jacobi per-lane edge count overflowed");
        }
      }
    }
    if (destination_mask != 0U) {
      require(
          expected_descriptors.size() <=
              std::numeric_limits<std::uint32_t>::max(),
          "batched host Jacobi descriptor count exceeds 32 bits");
      expected_descriptor_offsets.push_back(
          static_cast<std::uint32_t>(expected_descriptors.size()));
    }
  }
  require(
      compact_vertex == union_vertex_count &&
          batch.union_edge_estimate == expected_union_edges,
      "batched host Jacobi compact vertex or union edge estimate is invalid");

  if (description.run_representation ==
      BatchRunRepresentation::retained_per_run_masks) {
    require(
        description.csc_run_lane_masks.size() == run_count &&
            description.csc_run_descriptors.empty() &&
            description.csc_descriptor_offsets_by_union_vertex.empty(),
        "batched host Jacobi retained CSC storage has an invalid shape");
    for (std::size_t destination = 0U; destination < vertex_count;
         ++destination) {
      const LaneMask destination_mask =
          expected_tile_masks[graph.owner_tiles[destination]];
      const std::size_t begin = graph.csc_column_run_offsets[destination];
      const std::size_t end = graph.csc_column_run_offsets[destination + 1U];
      for (std::size_t run = begin; run < end; ++run) {
        const LaneMask expected_mask =
            destination_mask &
            expected_tile_masks[graph.csc_run_source_tiles[run]];
        require(
            description.csc_run_lane_masks[run] == expected_mask,
            "batched host Jacobi retained CSC mask is not endpoint-exact");
      }
    }
    require(
        description.touched_csc_runs == expected_touched_runs,
        "batched host Jacobi retained CSC touched ledger is inconsistent");
  } else if (
      description.run_representation ==
      BatchRunRepresentation::compact_nonzero_descriptors) {
    require(
        description.csc_run_lane_masks.empty() &&
            description.touched_csc_runs.empty() &&
            description.csc_run_descriptors == expected_descriptors &&
            description.csc_descriptor_offsets_by_union_vertex ==
                expected_descriptor_offsets,
        "batched host Jacobi compact CSC descriptors are not endpoint-exact");
  } else {
    throw std::invalid_argument{
        "batched host Jacobi received an unknown run representation"};
  }

  for (std::size_t lane = 0U; lane < width; ++lane) {
    const LaneMask bit = lane_bit(lane);
    if ((batch.valid_lane_mask & bit) == 0U) {
      continue;
    }
    require(
        description.selected_edge_estimates_by_lane[lane] ==
            validated.selected_edges_by_lane[lane],
        "batched host Jacobi per-lane edge estimate is inconsistent");
    checked_add(
        validated.selected_lane_edges,
        validated.selected_edges_by_lane[lane],
        "batched host Jacobi selected lane edge count overflowed");
  }
  require(
      batch.selected_lane_edge_estimate == validated.selected_lane_edges &&
          description.run_report.csc_runs_visited == expected_runs_considered &&
          description.run_report.active_csc_runs == expected_active_runs &&
          description.run_report.csc_lane_edge_pairs == expected_lane_edge_pairs,
      "batched host Jacobi CSC preparation report is inconsistent");
  return validated;
}

[[nodiscard]] bool selected_slots_are_bitwise_identical(
    const DeviceGraphLayout32& graph,
    const BatchDeviceDescription& description,
    const std::array<std::vector<float>, 2U>& slots,
    const std::size_t lane) noexcept {
  const LaneMask bit = lane_bit(lane);
  for (std::size_t vertex = 0U; vertex < graph.vertex_count; ++vertex) {
    if ((description.tile_lane_masks[graph.owner_tiles[vertex]] & bit) == 0U) {
      continue;
    }
    const std::size_t index = static_cast<std::size_t>(
        batched_jacobi_distance_index(
            static_cast<std::uint32_t>(vertex),
            static_cast<std::uint32_t>(lane),
            description.lane_width));
    if (std::bit_cast<std::uint32_t>(slots[0U][index]) !=
        std::bit_cast<std::uint32_t>(slots[1U][index])) {
      return false;
    }
  }
  return true;
}

}  // namespace

HostBatchedJacobiRunResult run_host_batched_jacobi_pull(
    const DeviceGraphLayout32& graph,
    const std::span<const RouteQuery> queries,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description,
    const GpuRunOptions& options) {
  const ValidatedHostBatch validated =
      validate_host_batch(graph, queries, batch, description, options);

  const std::uint64_t element_count64 = checked_product(
      graph.vertex_count,
      batch.lane_width,
      "batched host Jacobi distance element count overflowed");
  require(
      element_count64 <= std::numeric_limits<std::size_t>::max(),
      "batched host Jacobi distance allocation exceeds host size");
  const std::size_t element_count = static_cast<std::size_t>(element_count64);
  const float infinity = std::numeric_limits<float>::infinity();

  HostBatchedJacobiRunResult output;
  output.distance_slots[0U].assign(element_count, infinity);
  output.distance_slots[1U].assign(element_count, infinity);
  output.rounds_executed_by_lane.assign(batch.lane_width, 0U);
  output.convergence_round_by_lane.assign(batch.lane_width, 0U);
  output.tail_rounds_by_lane.assign(batch.lane_width, 0U);
  output.controller =
      initialize_device_controller(options, batch.valid_lane_mask);

  output.batch_work.selected_tile_lane_positions =
      validated.selected_tile_lane_positions;
  output.batch_work.union_tile_lane_positions = checked_product(
      batch.union_tiles.size(),
      static_cast<std::uint64_t>(
          std::popcount(batch.valid_lane_mask)),
      "batched host Jacobi union tile/lane count overflowed");
  output.batch_work.distance_reset_bytes = checked_product(
      checked_product(
          validated.selected_lane_vertices,
          2U,
          "batched host Jacobi reset element count overflowed"),
      sizeof(float),
      "batched host Jacobi reset byte count overflowed");
  output.batch_work.source_seed_write_bytes = checked_product(
      checked_product(
          description.sources.size(),
          2U,
          "batched host Jacobi source seed count overflowed"),
      sizeof(float),
      "batched host Jacobi source seed byte count overflowed");

  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    if ((batch.valid_lane_mask & lane_bit(lane)) == 0U) {
      continue;
    }
    const std::size_t begin = description.source_offsets[lane];
    const std::size_t end = description.source_offsets[lane + 1U];
    for (std::size_t position = begin; position < end; ++position) {
      const std::uint32_t source = description.sources[position];
      const std::size_t index = static_cast<std::size_t>(
          batched_jacobi_distance_index(
              source,
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      output.distance_slots[0U][index] = 0.0F;
      output.distance_slots[1U][index] = 0.0F;
    }
  }

  DeviceWorkStatistics work;
  const bool collect_light =
      options.instrumentation != InstrumentationLevel::none;
  const bool collect_debug =
      options.instrumentation == InstrumentationLevel::debug;

  const auto execute_round = [&]() {
    if (output.controller.done != 0U ||
        output.controller.execute_lane_mask == 0U) {
      return;
    }
    const LaneMask execute = output.controller.execute_lane_mask;
    const std::uint64_t round_number = output.controller.rounds_completed + 1U;
    const std::size_t read_slot = output.controller.distance_read_slot;
    const std::size_t write_slot = output.controller.distance_write_slot;
    if (read_slot > 1U || write_slot > 1U || read_slot == write_slot) {
      detail::fail_jacobi_controller(output.controller);
      return;
    }
    const std::vector<float>& old_distances = output.distance_slots[read_slot];
    std::vector<float>& next_distances = output.distance_slots[write_slot];
    const std::uint64_t executing_lanes =
        static_cast<std::uint64_t>(std::popcount(execute));
    checked_add(
        output.batch_work.active_lane_rounds,
        executing_lanes,
        "batched host Jacobi active lane-round count overflowed");
    if (collect_light) {
      checked_add(
          work.active_lane_rounds,
          executing_lanes,
          "batched host Jacobi common active lane-round count overflowed");
    }
    for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
      if ((execute & lane_bit(lane)) != 0U) {
        checked_add(
            output.rounds_executed_by_lane[lane],
            1U,
            "batched host Jacobi per-lane round count overflowed");
      }
    }

    LaneMask changed = 0U;
    std::size_t compact_vertex = 0U;
    for (const BatchVertexRange& range : description.selected_vertex_ranges) {
      for (std::size_t destination = range.begin; destination < range.end;
           ++destination) {
        const LaneMask destination_lanes = range.lane_mask & execute;
        if (destination_lanes == 0U) {
          ++compact_vertex;
          continue;
        }
        std::array<float, maximum_batch_lanes> best{};
        for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
          if ((destination_lanes & lane_bit(lane)) == 0U) {
            continue;
          }
          const std::size_t index = static_cast<std::size_t>(
              batched_jacobi_distance_index(
                  static_cast<std::uint32_t>(destination),
                  static_cast<std::uint32_t>(lane),
                  batch.lane_width));
          best[lane] = old_distances[index];
        }

        const auto process_run = [&](
                                     const std::size_t run,
                                     const LaneMask prepared_mask) {
          checked_add(
              output.batch_work.csc_runs_considered,
              1U,
              "batched host Jacobi considered run count overflowed");
          if (collect_debug) {
            checked_add(
                work.mask_operations,
                1U,
                "batched host Jacobi common mask count overflowed");
          }
          const LaneMask active_mask = prepared_mask & execute;
          if (active_mask == 0U) {
            checked_add(
                output.batch_work.csc_runs_skipped,
                1U,
                "batched host Jacobi skipped run count overflowed");
            return;
          }
          checked_add(
              output.batch_work.csc_runs_visited,
              1U,
              "batched host Jacobi visited run count overflowed");
          checked_add(
              output.batch_work.active_lanes_over_visited_runs,
              static_cast<std::uint64_t>(std::popcount(active_mask)),
              "batched host Jacobi visited-run lane count overflowed");
          const std::size_t edge_begin = graph.csc_run_edge_offsets[run];
          const std::size_t edge_end = graph.csc_run_edge_offsets[run + 1U];
          for (std::size_t edge = edge_begin; edge < edge_end; ++edge) {
            checked_add(
                output.batch_work.csc_edge_loads,
                1U,
                "batched host Jacobi shared edge-load count overflowed");
            const std::uint32_t source = graph.csc_sources[edge];
            const float weight = graph.csc_weights[edge];
            for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
              if ((active_mask & lane_bit(lane)) == 0U) {
                continue;
              }
              checked_add(
                  output.batch_work.lane_edge_relaxations,
                  1U,
                  "batched host Jacobi lane-edge count overflowed");
              if (collect_light) {
                checked_add(
                    work.edges_examined,
                    1U,
                    "batched host Jacobi common edge count overflowed");
              }
              const std::size_t source_index = static_cast<std::size_t>(
                  batched_jacobi_distance_index(
                      source,
                      static_cast<std::uint32_t>(lane),
                      batch.lane_width));
              const float candidate = old_distances[source_index] + weight;
              if (candidate < best[lane]) {
                best[lane] = candidate;
              }
            }
          }
        };

        if (description.run_representation ==
            BatchRunRepresentation::retained_per_run_masks) {
          const std::size_t run_begin =
              graph.csc_column_run_offsets[destination];
          const std::size_t run_end =
              graph.csc_column_run_offsets[destination + 1U];
          for (std::size_t run = run_begin; run < run_end; ++run) {
            process_run(run, description.csc_run_lane_masks[run]);
          }
        } else {
          const std::size_t descriptor_begin =
              description.csc_descriptor_offsets_by_union_vertex
                  [compact_vertex];
          const std::size_t descriptor_end =
              description.csc_descriptor_offsets_by_union_vertex
                  [compact_vertex + 1U];
          for (std::size_t position = descriptor_begin;
               position < descriptor_end;
               ++position) {
            const RunLaneMaskDescriptor descriptor =
                description.csc_run_descriptors[position];
            process_run(descriptor.run_id, descriptor.lane_mask);
          }
        }

        for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
          if ((destination_lanes & lane_bit(lane)) == 0U) {
            continue;
          }
          const std::size_t index = static_cast<std::size_t>(
              batched_jacobi_distance_index(
                  static_cast<std::uint32_t>(destination),
                  static_cast<std::uint32_t>(lane),
                  batch.lane_width));
          next_distances[index] = best[lane];
          checked_add(
              output.batch_work.destination_lane_writes,
              1U,
              "batched host Jacobi destination write count overflowed");
          if (collect_light) {
            checked_add(
                work.active_vertices,
                1U,
                "batched host Jacobi common active vertex count overflowed");
          }
          if (best[lane] < old_distances[index]) {
            changed |= lane_bit(lane);
            checked_add(
                output.batch_work.successful_decreases,
                1U,
                "batched host Jacobi decrease count overflowed");
            if (collect_light) {
              checked_add(
                  work.successful_decreases,
                  1U,
                  "batched host Jacobi common decrease count overflowed");
            }
          }
        }
        ++compact_vertex;
      }
    }
    if (compact_vertex != batch.union_vertex_count) {
      throw std::logic_error{
          "batched host Jacobi compact destination traversal drifted"};
    }
    output.controller.changed_lane_mask |= changed;
    const LaneMask newly_observed_converged = execute & ~changed;
    for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
      if ((newly_observed_converged & lane_bit(lane)) != 0U &&
          output.convergence_round_by_lane[lane] == 0U) {
        output.convergence_round_by_lane[lane] = round_number;
      }
    }
  };

  const auto execute_pair = [&]() {
    execute_round();
    static_cast<void>(advance_jacobi_controller(output.controller));
  };

  if (options.control_mode == ControlMode::persistent_cooperative) {
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      execute_pair();
    }
    work.controller_copies = 1U;
    work.host_synchronizations = 1U;
    work.host_checks = 1U;
  } else if (options.control_mode == ControlMode::per_round_host_poll) {
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      checked_add(
          work.kernel_dispatches,
          2U,
          "batched host Jacobi dispatch count overflowed");
      ++output.queued_round_pairs;
      execute_pair();
      ++work.controller_copies;
      ++work.host_synchronizations;
      ++work.host_checks;
      ++output.completed_host_chunks;
    }
  } else {
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      for (std::uint64_t pair = 0U; pair < options.rounds_per_chunk; ++pair) {
        checked_add(
            work.kernel_dispatches,
            2U,
            "batched host Jacobi dispatch count overflowed");
        ++output.queued_round_pairs;
        execute_pair();
      }
      ++work.controller_copies;
      ++work.host_synchronizations;
      ++work.host_checks;
      ++output.completed_host_chunks;
    }
  }

  const std::uint64_t completed_rounds = output.controller.rounds_completed;
  const std::uint64_t valid_lanes =
      static_cast<std::uint64_t>(std::popcount(batch.valid_lane_mask));
  output.batch_work.valid_lane_round_capacity = checked_product(
      completed_rounds,
      valid_lanes,
      "batched host Jacobi valid lane-round capacity overflowed");
  output.batch_work.wave_lane_round_capacity = checked_product(
      completed_rounds,
      batch.lane_width,
      "batched host Jacobi wave lane-round capacity overflowed");
  output.batch_work.padded_lane_round_capacity = checked_product(
      completed_rounds,
      static_cast<std::uint64_t>(batch.lane_width) - valid_lanes,
      "batched host Jacobi padding capacity overflowed");
  output.batch_work.inactive_valid_lane_rounds =
      output.batch_work.valid_lane_round_capacity -
      output.batch_work.active_lane_rounds;

  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    const LaneMask bit = lane_bit(lane);
    if ((batch.valid_lane_mask & bit) == 0U) {
      continue;
    }
    const std::uint64_t convergence_round =
        output.convergence_round_by_lane[lane];
    if (convergence_round != 0U && convergence_round <= completed_rounds) {
      const std::uint64_t tail = completed_rounds - convergence_round;
      output.tail_rounds_by_lane[lane] = tail;
      checked_add(
          output.batch_work.tail_lane_rounds,
          tail,
          "batched host Jacobi tail lane-round count overflowed");
      if (options.enable_per_lane_convergence != 0U) {
        checked_add(
            output.batch_work.tail_lane_rounds_avoided,
            tail,
            "batched host Jacobi avoided tail count overflowed");
        const std::uint64_t avoided_edges = checked_product(
            tail,
            validated.selected_edges_by_lane[lane],
            "batched host Jacobi avoided lane-edge count overflowed");
        checked_add(
            output.batch_work
                .lane_edge_relaxations_avoided_by_early_convergence,
            avoided_edges,
            "batched host Jacobi avoided lane-edge count overflowed");
      } else {
        checked_add(
            output.batch_work.tail_lane_rounds_executed,
            tail,
            "batched host Jacobi executed tail count overflowed");
      }
      if (selected_slots_are_bitwise_identical(
              graph, description, output.distance_slots, lane)) {
        output.converged_slots_bitwise_identical_mask |= bit;
      }
    }
  }

  const std::size_t final_slot = output.controller.distance_read_slot;
  output.final_distances = output.distance_slots[final_slot];
  LaneMask reached = 0U;
  LaneMask miss = 0U;
  if (output.controller.stop_reason ==
      static_cast<std::uint32_t>(DeviceStopReason::converged)) {
    for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
      const LaneMask bit = lane_bit(lane);
      if ((batch.valid_lane_mask & bit) == 0U) {
        continue;
      }
      bool all_reached = true;
      const std::size_t begin = description.target_offsets[lane];
      const std::size_t end = description.target_offsets[lane + 1U];
      for (std::size_t position = begin; position < end; ++position) {
        const std::size_t index = static_cast<std::size_t>(
            batched_jacobi_distance_index(
                description.targets[position],
                static_cast<std::uint32_t>(lane),
                batch.lane_width));
        if (!std::isfinite(output.final_distances[index])) {
          all_reached = false;
          break;
        }
      }
      if (all_reached) {
        reached |= bit;
      } else {
        miss |= bit;
      }
    }
  }

  output.result.engine_kind =
      static_cast<std::uint32_t>(EngineKind::jacobi_pull);
  output.result.control_mode = static_cast<std::uint32_t>(options.control_mode);
  output.result.status =
      make_jacobi_run_status(output.controller, reached, miss);
  output.result.work = work;
  if (validate_device_controller(output.controller) !=
          DeviceControllerError::none ||
      validate_device_run_status(output.result.status) !=
          DeviceRunStatusError::none) {
    throw std::logic_error{
        "portable batched Jacobi produced invalid terminal controller state"};
  }
  if (output.result.status.converged != 0U &&
      output.converged_slots_bitwise_identical_mask !=
          batch.valid_lane_mask) {
    throw std::logic_error{
        "portable batched Jacobi violated converged-slot equality"};
  }
  return output;
}

}  // namespace bfnew
