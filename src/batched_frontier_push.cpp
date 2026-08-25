#include "bfnew/batched_frontier_push.hpp"

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

constexpr std::uint32_t positive_infinity_bits = 0x7f800000U;

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

[[nodiscard]] std::uint64_t checked_sum(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* const message) {
  std::uint64_t value = left;
  checked_add(value, right, message);
  return value;
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
  std::uint64_t union_vertex_count{};
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
      "batched host frontier push requires valid run options");
  require(
      options.engine == EngineKind::frontier_push,
      "batched host frontier push rejects another engine kind");
  require(
      supported_batched_frontier_width(batch.lane_width),
      "batched host frontier push supports widths 1, 8, 16, and 32 only");

  const std::size_t width = batch.lane_width;
  require(
      batch.query_indices_by_lane.size() == width &&
          batch.query_ids_by_lane.size() == width &&
          batch.expansion_generations_by_lane.size() == width &&
          batch.union_tiles.size() == batch.union_tile_lane_masks.size() &&
          !batch.union_tiles.empty() && batch.valid_lane_mask != 0U &&
          is_low_lane_mask(batch.valid_lane_mask) &&
          (batch.valid_lane_mask & ~width_mask(batch.lane_width)) == 0U,
      "batched host frontier push received an invalid plan entry shape");
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
      "batched host frontier push received an invalid device description");

  const std::size_t vertex_count = graph.vertex_count;
  const std::size_t edge_count = graph.edge_count;
  const std::size_t tile_count = graph.tile_count;
  const std::size_t run_count = graph.csr_run_destination_tiles.size();
  require(
      vertex_count != 0U && tile_count != 0U &&
          graph.owner_tiles.size() == vertex_count &&
          graph.csr_destinations.size() == edge_count &&
          graph.csr_weights.size() == edge_count &&
          valid_offsets(graph.csr_row_offsets, vertex_count, edge_count) &&
          valid_offsets(graph.csr_row_run_offsets, vertex_count, run_count) &&
          valid_offsets(graph.csr_run_edge_offsets, run_count, edge_count) &&
          description.tile_lane_masks.size() == tile_count,
      "batched host frontier push received an invalid CSR device graph");

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
          "batched host frontier push requires contiguous tile ownership");
    }
  }
  require(
      owner_cursor == vertex_count,
      "batched host frontier push owner tile is out of range");

  for (std::size_t source = 0U; source < vertex_count; ++source) {
    const std::size_t edge_begin = graph.csr_row_offsets[source];
    const std::size_t edge_end = graph.csr_row_offsets[source + 1U];
    const std::size_t run_begin = graph.csr_row_run_offsets[source];
    const std::size_t run_end = graph.csr_row_run_offsets[source + 1U];
    std::size_t cursor = edge_begin;
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const std::size_t begin = graph.csr_run_edge_offsets[run];
      const std::size_t end = graph.csr_run_edge_offsets[run + 1U];
      require(
          begin == cursor && begin < end && end <= edge_end,
          "batched host frontier push CSR runs do not cover a row exactly");
      const std::uint32_t destination_tile =
          graph.csr_run_destination_tiles[run];
      require(
          destination_tile < tile_count,
          "batched host frontier push CSR run tile is out of range");
      for (std::size_t edge = begin; edge < end; ++edge) {
        const std::uint32_t destination = graph.csr_destinations[edge];
        const float weight = graph.csr_weights[edge];
        require(
            destination < vertex_count &&
                graph.owner_tiles[destination] == destination_tile &&
                std::isfinite(weight) && weight >= 0.0F &&
                !(weight == 0.0F && std::signbit(weight)),
            "batched host frontier push edge disagrees with its tile run");
      }
      cursor = end;
    }
    require(
        cursor == edge_end,
        "batched host frontier push CSR runs omit part of a row");
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
      "batched host frontier push query IDs are not unique");

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
          "batched host frontier push padded lane has semantic payload");
      continue;
    }

    const std::uint32_t query_index = batch.query_indices_by_lane[lane];
    require(
        query_index != invalid_batch_query_index &&
            query_index < queries_by_plan_index.size(),
        "batched host frontier push lane has an invalid query index");
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
                description.sources, source_begin, source_end, query->sources) &&
            terminal_payload_matches(
                description.targets, target_begin, target_end, query->targets),
        "batched host frontier push lane identity or terminals are invalid");
    for (std::size_t earlier = 0U; earlier < lane; ++earlier) {
      if ((resolved_query_ids & lane_bit(earlier)) != 0U) {
        require(
            batch.query_ids_by_lane[earlier] != query_id,
            "batched host frontier push duplicates a query across lanes");
      }
    }
    resolved_query_ids |= bit;
    validated.queries_by_lane[lane] = query;
    for (const TileId tile : query->selected_tiles) {
      expected_tile_masks[tile.value()] |= bit;
    }
  }

  std::size_t union_position = 0U;
  for (std::size_t tile = 0U; tile < tile_count; ++tile) {
    const LaneMask expected_mask = expected_tile_masks[tile];
    require(
        description.tile_lane_masks[tile] == expected_mask,
        "batched host frontier push tile masks differ from query tiles");
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
        "batched host frontier push union-tile ledger is not canonical");
    const BatchVertexRange range =
        description.selected_vertex_ranges[union_position];
    require(
        range.begin == tile_begins[tile] && range.end == tile_ends[tile] &&
            range.begin <= range.end && range.lane_mask == expected_mask,
        "batched host frontier push selected range differs from its tile");
    const std::uint64_t tile_vertices = range.end - range.begin;
    checked_add(
        validated.union_vertex_count,
        tile_vertices,
        "batched host frontier push union vertex count overflowed");
    checked_add(
        validated.selected_tile_lane_positions,
        static_cast<std::uint64_t>(std::popcount(expected_mask)),
        "batched host frontier push selected tile/lane count overflowed");
    for (std::size_t lane = 0U; lane < width; ++lane) {
      if ((expected_mask & lane_bit(lane)) != 0U) {
        checked_add(
            validated.selected_vertices_by_lane[lane],
            tile_vertices,
            "batched host frontier push selected lane vertices overflowed");
      }
    }
    ++union_position;
  }
  require(
      union_position == batch.union_tiles.size() &&
          union_position == description.union_tiles.size() &&
          union_position == description.selected_vertex_ranges.size() &&
          batch.union_vertex_count == validated.union_vertex_count,
      "batched host frontier push union vertex ledger is inconsistent");

  for (std::size_t lane = 0U; lane < width; ++lane) {
    const LaneMask bit = lane_bit(lane);
    if ((batch.valid_lane_mask & bit) == 0U) {
      continue;
    }
    const RouteQuery& query = *validated.queries_by_lane[lane];
    for (const VertexId source : query.sources) {
      require(
          (expected_tile_masks[graph.owner_tiles[source.value()]] & bit) != 0U,
          "batched host frontier push source is outside its lane tiles");
    }
    for (const VertexId target : query.targets) {
      require(
          (expected_tile_masks[graph.owner_tiles[target.value()]] & bit) != 0U,
          "batched host frontier push target is outside its lane tiles");
    }
    require(
        description.selected_vertex_counts_by_lane[lane] ==
            validated.selected_vertices_by_lane[lane],
        "batched host frontier push selected vertex estimate is inconsistent");
    checked_add(
        validated.selected_lane_vertices,
        validated.selected_vertices_by_lane[lane],
        "batched host frontier push selected lane vertices overflowed");
  }
  require(
      batch.selected_lane_vertex_count == validated.selected_lane_vertices,
      "batched host frontier push plan vertex estimate is inconsistent");

  std::vector<RunLaneMaskDescriptor> expected_descriptors;
  std::vector<std::uint32_t> expected_descriptor_offsets{0U};
  std::vector<std::uint32_t> expected_touched_runs;
  std::uint64_t expected_runs_considered = 0U;
  std::uint64_t expected_active_runs = 0U;
  std::uint64_t expected_lane_edge_pairs = 0U;
  std::uint64_t expected_union_edges = 0U;
  std::size_t compact_vertex = 0U;
  for (std::size_t source = 0U; source < vertex_count; ++source) {
    const LaneMask source_mask =
        expected_tile_masks[graph.owner_tiles[source]];
    const std::size_t run_begin = graph.csr_row_run_offsets[source];
    const std::size_t run_end = graph.csr_row_run_offsets[source + 1U];
    if (source_mask != 0U) {
      validated.compact_index_by_vertex[source] = compact_vertex;
      ++compact_vertex;
      checked_add(
          expected_runs_considered,
          run_end - run_begin,
          "batched host frontier push CSR run count overflowed");
    }
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const LaneMask expected_mask =
          source_mask &
          expected_tile_masks[graph.csr_run_destination_tiles[run]];
      const std::uint64_t edges =
          static_cast<std::uint64_t>(
              graph.csr_run_edge_offsets[run + 1U]) -
          graph.csr_run_edge_offsets[run];
      if (source_mask != 0U &&
          expected_tile_masks[graph.csr_run_destination_tiles[run]] != 0U) {
        checked_add(
            expected_union_edges,
            edges,
            "batched host frontier push union edge count overflowed");
      }
      if (expected_mask == 0U) {
        continue;
      }
      checked_add(
          expected_active_runs,
          1U,
          "batched host frontier push active run count overflowed");
      expected_touched_runs.push_back(static_cast<std::uint32_t>(run));
      expected_descriptors.push_back(RunLaneMaskDescriptor{
          static_cast<std::uint32_t>(run), expected_mask});
      checked_add(
          expected_lane_edge_pairs,
          checked_product(
              edges,
              static_cast<std::uint64_t>(std::popcount(expected_mask)),
              "batched host frontier push lane-edge count overflowed"),
          "batched host frontier push lane-edge count overflowed");
      for (std::size_t lane = 0U; lane < width; ++lane) {
        if ((expected_mask & lane_bit(lane)) != 0U) {
          checked_add(
              validated.selected_edges_by_lane[lane],
              edges,
              "batched host frontier push selected lane edges overflowed");
        }
      }
    }
    if (source_mask != 0U) {
      require(
          expected_descriptors.size() <=
              std::numeric_limits<std::uint32_t>::max(),
          "batched host frontier push descriptor count exceeds 32 bits");
      expected_descriptor_offsets.push_back(
          static_cast<std::uint32_t>(expected_descriptors.size()));
    }
  }
  require(
      compact_vertex == validated.union_vertex_count &&
          batch.union_edge_estimate == expected_union_edges,
      "batched host frontier push compact vertex or union edge estimate is invalid");

  if (description.run_representation ==
      BatchRunRepresentation::retained_per_run_masks) {
    require(
        description.csr_run_lane_masks.size() == run_count &&
            description.csr_run_descriptors.empty() &&
            description.csr_descriptor_offsets_by_union_vertex.empty(),
        "batched host frontier push retained CSR storage is malformed");
    for (std::size_t source = 0U; source < vertex_count; ++source) {
      const LaneMask source_mask =
          expected_tile_masks[graph.owner_tiles[source]];
      const std::size_t begin = graph.csr_row_run_offsets[source];
      const std::size_t end = graph.csr_row_run_offsets[source + 1U];
      for (std::size_t run = begin; run < end; ++run) {
        const LaneMask expected_mask =
            source_mask &
            expected_tile_masks[graph.csr_run_destination_tiles[run]];
        require(
            description.csr_run_lane_masks[run] == expected_mask,
            "batched host frontier push retained CSR mask is not endpoint-exact");
      }
    }
    require(
        description.touched_csr_runs == expected_touched_runs,
        "batched host frontier push retained touched ledger is inconsistent");
  } else if (
      description.run_representation ==
      BatchRunRepresentation::compact_nonzero_descriptors) {
    require(
        description.csr_run_lane_masks.empty() &&
            description.touched_csr_runs.empty() &&
            description.csr_run_descriptors == expected_descriptors &&
            description.csr_descriptor_offsets_by_union_vertex ==
                expected_descriptor_offsets,
        "batched host frontier push compact CSR descriptors are not exact");
  } else {
    throw std::invalid_argument{
        "batched host frontier push received an unknown run representation"};
  }

  for (std::size_t lane = 0U; lane < width; ++lane) {
    if ((batch.valid_lane_mask & lane_bit(lane)) == 0U) {
      continue;
    }
    require(
        description.selected_edge_estimates_by_lane[lane] ==
            validated.selected_edges_by_lane[lane],
        "batched host frontier push selected edge estimate is inconsistent");
    checked_add(
        validated.selected_lane_edges,
        validated.selected_edges_by_lane[lane],
        "batched host frontier push selected lane edges overflowed");
  }
  require(
      batch.selected_lane_edge_estimate == validated.selected_lane_edges &&
          description.run_report.csr_runs_visited ==
              expected_runs_considered &&
          description.run_report.active_csr_runs == expected_active_runs &&
          description.run_report.csr_lane_edge_pairs ==
              expected_lane_edge_pairs,
      "batched host frontier push preparation report is inconsistent");
  return validated;
}

struct HostFrontierState {
  std::vector<std::uint32_t> distances;
  std::array<std::vector<std::uint32_t>, 2U> queues;
  std::array<std::vector<LaneMask>, 2U> activity_masks;
  std::size_t queue_capacity{};
};

[[nodiscard]] bool all_targets_reached(
    const BatchDeviceDescription& description,
    const std::vector<std::uint32_t>& distance_bits,
    const std::size_t lane) {
  const std::size_t begin = description.target_offsets[lane];
  const std::size_t end = description.target_offsets[lane + 1U];
  for (std::size_t position = begin; position < end; ++position) {
    const std::size_t index = static_cast<std::size_t>(
        batched_frontier_distance_index(
            description.targets[position],
            static_cast<std::uint32_t>(lane),
            description.lane_width));
    if (distance_bits[index] == positive_infinity_bits) {
      return false;
    }
  }
  return true;
}

}  // namespace

BatchedFrontierScratchLayout make_batched_frontier_scratch_layout(
    const std::size_t vertex_count,
    const std::uint32_t lane_width,
    const std::size_t requested_queue_capacity) {
  if (vertex_count == 0U) {
    throw std::invalid_argument{
        "batched frontier scratch requires at least one vertex"};
  }
  if (vertex_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{
        "batched frontier vertex count exceeds the device ABI"};
  }
  if (!supported_batched_frontier_width(lane_width)) {
    throw std::invalid_argument{
        "batched frontier scratch width must be 1, 8, 16, or 32"};
  }
  const std::size_t queue_capacity =
      requested_queue_capacity == 0U ? vertex_count
                                     : requested_queue_capacity;
  if (queue_capacity == 0U || queue_capacity > vertex_count ||
      queue_capacity > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument{
        "batched frontier queue capacity must be in [1, vertex_count]"};
  }

  const std::uint64_t distance_bytes = checked_product(
      checked_product(
          vertex_count,
          lane_width,
          "batched frontier distance element count overflowed"),
      sizeof(std::uint32_t),
      "batched frontier distance byte count overflowed");
  const std::uint64_t activity_bytes = checked_product(
      vertex_count,
      sizeof(LaneMask),
      "batched frontier activity byte count overflowed");
  const std::uint64_t queue_bytes = checked_product(
      queue_capacity,
      sizeof(std::uint32_t),
      "batched frontier queue byte count overflowed");
  const std::uint64_t activity_zero_offset = distance_bytes;
  const std::uint64_t activity_one_offset = checked_sum(
      activity_zero_offset,
      activity_bytes,
      "batched frontier activity-one offset overflowed");
  const std::uint64_t queue_zero_offset = checked_sum(
      activity_one_offset,
      activity_bytes,
      "batched frontier queue-zero offset overflowed");
  const std::uint64_t queue_one_offset = checked_sum(
      queue_zero_offset,
      queue_bytes,
      "batched frontier queue-one offset overflowed");
  const std::uint64_t total_bytes = checked_sum(
      queue_one_offset,
      queue_bytes,
      "batched frontier scratch total byte count overflowed");
  return BatchedFrontierScratchLayout{
      vertex_count,
      lane_width,
      0U,
      queue_capacity,
      0U,
      distance_bytes,
      {activity_zero_offset, activity_one_offset},
      activity_bytes,
      {queue_zero_offset, queue_one_offset},
      queue_bytes,
      total_bytes,
  };
}

HostBatchedFrontierRunResult run_host_batched_frontier_push(
    const DeviceGraphLayout32& graph,
    const std::span<const RouteQuery> queries,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description,
    const GpuRunOptions& options,
    const std::size_t requested_queue_capacity) {
  const ValidatedHostBatch validated =
      validate_host_batch(graph, queries, batch, description, options);
  const BatchedFrontierScratchLayout layout =
      make_batched_frontier_scratch_layout(
          graph.vertex_count, batch.lane_width, requested_queue_capacity);
  const std::size_t queue_capacity =
      static_cast<std::size_t>(layout.queue_capacity);
  const std::uint64_t element_count64 = checked_product(
      graph.vertex_count,
      batch.lane_width,
      "batched host frontier push distance count overflowed");
  require(
      element_count64 <= std::numeric_limits<std::size_t>::max(),
      "batched host frontier push distance allocation exceeds host size");

  HostBatchedFrontierRunResult output;
  output.queue_capacity = queue_capacity;
  output.frontier_rounds_by_lane.assign(batch.lane_width, 0U);
  output.convergence_round_by_lane.assign(batch.lane_width, 0U);
  output.tail_rounds_by_lane.assign(batch.lane_width, 0U);

  HostFrontierState state;
  state.queue_capacity = queue_capacity;
  state.distances.assign(
      static_cast<std::size_t>(element_count64), positive_infinity_bits);
  for (auto& queue : state.queues) {
    queue.assign(queue_capacity, 0U);
  }
  for (auto& masks : state.activity_masks) {
    masks.assign(graph.vertex_count, 0U);
  }

  output.batch_work.selected_tile_lane_positions =
      validated.selected_tile_lane_positions;
  output.batch_work.union_tile_lane_positions = checked_product(
      batch.union_tiles.size(),
      static_cast<std::uint64_t>(std::popcount(batch.valid_lane_mask)),
      "batched host frontier push union tile/lane count overflowed");
  output.batch_work.distance_reset_bytes = checked_product(
      validated.selected_lane_vertices,
      sizeof(std::uint32_t),
      "batched host frontier push distance reset bytes overflowed");
  output.batch_work.activity_mask_reset_bytes = checked_product(
      checked_product(
          validated.union_vertex_count,
          2U,
          "batched host frontier push activity reset count overflowed"),
      sizeof(LaneMask),
      "batched host frontier push activity reset bytes overflowed");
  output.batch_work.source_seed_write_bytes = checked_product(
      description.sources.size(),
      sizeof(std::uint32_t),
      "batched host frontier push source seed bytes overflowed");
  output.batch_work.frontier_queue_storage_bytes = checked_product(
      checked_product(
          queue_capacity,
          2U,
          "batched host frontier push queue storage count overflowed"),
      sizeof(std::uint32_t),
      "batched host frontier push queue storage bytes overflowed");

  std::uint64_t initial_queue_size = 0U;
  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    const LaneMask bit = lane_bit(lane);
    if ((batch.valid_lane_mask & bit) == 0U) {
      continue;
    }
    const std::size_t begin = description.source_offsets[lane];
    const std::size_t end = description.source_offsets[lane + 1U];
    for (std::size_t position = begin; position < end; ++position) {
      const std::uint32_t source = description.sources[position];
      const std::size_t distance_index = static_cast<std::size_t>(
          batched_frontier_distance_index(
              source,
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      state.distances[distance_index] = 0U;
      checked_add(
          output.batch_work.initial_source_lane_activations,
          1U,
          "batched host frontier push source activations overflowed");
      const LaneMask old_mask = state.activity_masks[0U][source];
      state.activity_masks[0U][source] = old_mask | bit;
      if (old_mask != 0U) {
        checked_add(
            output.batch_work.initial_queue_entries_saved_by_lane_merging,
            1U,
            "batched host frontier push initial merge savings overflowed");
        continue;
      }
      const std::uint64_t claim = initial_queue_size++;
      if (claim < queue_capacity) {
        state.queues[0U][static_cast<std::size_t>(claim)] = source;
      }
    }
  }
  output.batch_work.initial_queue_entries = initial_queue_size;
  output.batch_work.maximum_queue_size = initial_queue_size;
  require(
      output.batch_work.initial_source_lane_activations ==
          output.batch_work.initial_queue_entries +
              output.batch_work.initial_queue_entries_saved_by_lane_merging,
      "batched host frontier push initial queue merge identity failed");

  output.controller = initialize_device_controller(
      options, batch.valid_lane_mask, initial_queue_size);
  DeviceWorkStatistics work;
  const bool collect_light =
      options.instrumentation != InstrumentationLevel::none;
  const bool collect_debug =
      options.instrumentation == InstrumentationLevel::debug;
  if (collect_light) {
    work.maximum_queue_size = initial_queue_size;
  }
  if (initial_queue_size > queue_capacity) {
    output.batch_work.overflow_events = 1U;
    if (collect_debug) {
      work.overflow_events = 1U;
    }
    detail::canonicalize_frontier_error(
        output.controller,
        DeviceStopReason::queue_overflow,
        device_error::queue_overflow);
  }

  // A complete round proposes no-next evidence, but only its controller
  // transition can commit it. Error/no-op transitions never publish evidence
  // from the failed or nonexistent round; a clean maximum-round transition
  // may still publish lanes whose frontier exhaustion was fully observed.
  LaneMask pending_no_next_lanes = 0U;
  std::uint64_t pending_convergence_round = 0U;
  bool pending_convergence_evidence = false;
  const auto execute_round = [&]() {
    pending_no_next_lanes = 0U;
    pending_convergence_round = 0U;
    pending_convergence_evidence = false;
    if (output.controller.done != 0U ||
        output.controller.execute_lane_mask == 0U) {
      return;
    }
    const std::uint32_t read_slot = output.controller.frontier_read_slot;
    const std::uint32_t write_slot = output.controller.frontier_write_slot;
    const std::uint64_t current_size =
        output.controller.frontier_size[read_slot];
    if (current_size == 0U || current_size > state.queue_capacity ||
        output.controller.frontier_size[write_slot] != 0U ||
        output.controller.distance_read_slot != 0U ||
        output.controller.distance_write_slot != 0U) {
      detail::canonicalize_frontier_error(
          output.controller,
          DeviceStopReason::invalid_controller_state,
          device_error::invalid_controller_state);
      return;
    }
    for (const LaneMask mask : state.activity_masks[write_slot]) {
      if (mask != 0U) {
        detail::canonicalize_frontier_error(
            output.controller,
            DeviceStopReason::invalid_controller_state,
            device_error::invalid_controller_state);
        return;
      }
    }

    LaneMask current_lane_union = 0U;
    std::vector<bool> queued(graph.vertex_count, false);
    for (std::uint64_t entry = 0U; entry < current_size; ++entry) {
      const std::uint32_t vertex =
          state.queues[read_slot][static_cast<std::size_t>(entry)];
      if (vertex >= graph.vertex_count || queued[vertex] ||
          state.activity_masks[read_slot][vertex] == 0U ||
          (state.activity_masks[read_slot][vertex] &
           ~output.controller.execute_lane_mask) != 0U) {
        detail::canonicalize_frontier_error(
            output.controller,
            DeviceStopReason::invalid_controller_state,
            device_error::invalid_controller_state);
        return;
      }
      queued[vertex] = true;
      current_lane_union |= state.activity_masks[read_slot][vertex];
    }
    for (std::size_t vertex = 0U; vertex < graph.vertex_count; ++vertex) {
      if ((state.activity_masks[read_slot][vertex] != 0U) != queued[vertex]) {
        detail::canonicalize_frontier_error(
            output.controller,
            DeviceStopReason::invalid_controller_state,
            device_error::invalid_controller_state);
        return;
      }
    }
    if (current_lane_union != output.controller.execute_lane_mask) {
      detail::canonicalize_frontier_error(
          output.controller,
          DeviceStopReason::invalid_controller_state,
          device_error::invalid_controller_state);
      return;
    }

    output.current_frontier_sizes.push_back(current_size);
    output.current_frontier_lane_unions.push_back(current_lane_union);
    checked_add(
        output.batch_work.frontier_rounds,
        1U,
        "batched host frontier push round count overflowed");
    checked_add(
        output.batch_work.frontier_vertex_entries,
        current_size,
        "batched host frontier push entry count overflowed");
    const std::uint64_t round_lanes =
        static_cast<std::uint64_t>(std::popcount(current_lane_union));
    checked_add(
        output.batch_work.active_lane_rounds,
        round_lanes,
        "batched host frontier push active lane-round count overflowed");
    if (current_size < maximum_batch_lanes) {
      checked_add(
          output.batch_work.small_frontier_rounds,
          1U,
          "batched host frontier push small round count overflowed");
    }
    if (collect_light) {
      checked_add(
          work.active_vertices,
          current_size,
          "batched host frontier push common active vertices overflowed");
      checked_add(
          work.active_lane_rounds,
          round_lanes,
          "batched host frontier push common active lanes overflowed");
      if (current_size < maximum_batch_lanes) {
        checked_add(
            work.small_frontier_rounds,
            1U,
            "batched host frontier push common small rounds overflowed");
      }
    }
    for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
      if ((current_lane_union & lane_bit(lane)) != 0U) {
        checked_add(
            output.frontier_rounds_by_lane[lane],
            1U,
            "batched host frontier push per-lane rounds overflowed");
      }
    }

    const std::uint64_t round_number =
        output.controller.rounds_completed + 1U;
    for (std::uint64_t entry = 0U; entry < current_size; ++entry) {
      const std::uint32_t source =
          state.queues[read_slot][static_cast<std::size_t>(entry)];
      const LaneMask active_vertex_mask =
          state.activity_masks[read_slot][source];
      state.activity_masks[read_slot][source] = 0U;
      checked_add(
          output.batch_work.current_mask_atomic_exchanges,
          1U,
          "batched host frontier push current-mask exchanges overflowed");
      if (collect_debug) {
        checked_add(
            work.mask_operations,
            1U,
            "batched host frontier push common mask operations overflowed");
      }
      const std::uint64_t vertex_lanes = static_cast<std::uint64_t>(
          std::popcount(active_vertex_mask));
      checked_add(
          output.batch_work.active_vertex_lane_pairs,
          vertex_lanes,
          "batched host frontier push vertex/lane pairs overflowed");
      if (vertex_lanes > 1U) {
        checked_add(
            output.batch_work.multi_lane_frontier_vertex_entries,
            1U,
            "batched host frontier push shared entries overflowed");
      }

      const auto process_run = [&](const std::size_t run,
                                   const LaneMask prepared_mask) {
        checked_add(
            output.batch_work.csr_runs_considered,
            1U,
            "batched host frontier push considered runs overflowed");
        if (collect_debug) {
          checked_add(
              work.mask_operations,
              1U,
              "batched host frontier push common mask operations overflowed");
        }
        const LaneMask active_run_mask =
            prepared_mask & active_vertex_mask &
            output.controller.execute_lane_mask;
        if (active_run_mask == 0U) {
          checked_add(
              output.batch_work.csr_runs_skipped,
              1U,
              "batched host frontier push skipped runs overflowed");
          return;
        }
        checked_add(
            output.batch_work.csr_runs_visited,
            1U,
            "batched host frontier push visited runs overflowed");
        const std::uint64_t run_lanes =
            static_cast<std::uint64_t>(std::popcount(active_run_mask));
        checked_add(
            output.batch_work.active_lanes_over_visited_runs,
            run_lanes,
            "batched host frontier push run lane sum overflowed");

        const std::size_t edge_begin = graph.csr_run_edge_offsets[run];
        const std::size_t edge_end = graph.csr_run_edge_offsets[run + 1U];
        for (std::size_t edge = edge_begin; edge < edge_end; ++edge) {
          checked_add(
              output.batch_work.csr_edge_loads,
              1U,
              "batched host frontier push edge loads overflowed");
          checked_add(
              output.batch_work.lane_edge_relaxations,
              run_lanes,
              "batched host frontier push lane-edge work overflowed");
          if (run_lanes > 1U) {
            checked_add(
                output.batch_work.multi_lane_csr_edge_loads,
                1U,
                "batched host frontier push shared edge loads overflowed");
          }
          if (collect_light) {
            checked_add(
                work.edges_examined,
                run_lanes,
                "batched host frontier push common edges overflowed");
          }

          const std::uint32_t destination = graph.csr_destinations[edge];
          const float weight = graph.csr_weights[edge];
          LaneMask successful_lanes = 0U;
          for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
            const LaneMask bit = lane_bit(lane);
            if ((active_run_mask & bit) == 0U) {
              continue;
            }
            checked_add(
                output.batch_work.distance_atomic_source_loads,
                1U,
                "batched host frontier push source loads overflowed");
            checked_add(
                output.batch_work.distance_atomic_attempts,
                1U,
                "batched host frontier push atomic attempts overflowed");
            if (collect_debug) {
              checked_add(
                  work.atomic_attempts,
                  1U,
                  "batched host frontier push common attempts overflowed");
            }
            const std::size_t source_index = static_cast<std::size_t>(
                batched_frontier_distance_index(
                    source,
                    static_cast<std::uint32_t>(lane),
                    batch.lane_width));
            const float source_distance =
                dense_atomic_bits_float(state.distances[source_index]);
            if (!std::isfinite(source_distance)) {
              throw std::logic_error{
                  "batched host frontier push active lane has infinite source"};
            }
            const std::uint32_t candidate_bits = dense_atomic_float_bits(
                source_distance + weight);
            const std::size_t destination_index = static_cast<std::size_t>(
                batched_frontier_distance_index(
                    destination,
                    static_cast<std::uint32_t>(lane),
                    batch.lane_width));
            const DenseAtomicMinResult update = dense_atomic_min_bits(
                state.distances[destination_index], candidate_bits);
            if (!update.decreased) {
              continue;
            }
            state.distances[destination_index] = update.final_bits;
            successful_lanes |= bit;
            checked_add(
                output.batch_work.successful_distance_atomic_updates,
                1U,
                "batched host frontier push successful updates overflowed");
            if (collect_light) {
              checked_add(
                  work.successful_decreases,
                  1U,
                  "batched host frontier push common decreases overflowed");
            }
            if (collect_debug) {
              checked_add(
                  work.successful_atomic_updates,
                  1U,
                  "batched host frontier push common successes overflowed");
            }
          }
          if (successful_lanes == 0U) {
            continue;
          }

          const LaneMask old_mask =
              state.activity_masks[write_slot][destination];
          const LaneMask unique_lanes = successful_lanes & ~old_mask;
          state.activity_masks[write_slot][destination] =
              old_mask | successful_lanes;
          output.controller.next_frontier_lane_mask |= successful_lanes;
          checked_add(
              output.batch_work.next_mask_atomic_ors,
              1U,
              "batched host frontier push next-mask ORs overflowed");
          checked_add(
              output.batch_work.controller_mask_atomic_ors,
              1U,
              "batched host frontier push controller-mask ORs overflowed");
          if (collect_debug) {
            checked_add(
                work.mask_operations,
                2U,
                "batched host frontier push common mask operations overflowed");
          }
          const std::uint64_t unique_count =
              static_cast<std::uint64_t>(std::popcount(unique_lanes));
          const std::uint64_t successful_count =
              static_cast<std::uint64_t>(std::popcount(successful_lanes));
          checked_add(
              output.batch_work.unique_next_vertex_lane_activations,
              unique_count,
              "batched host frontier push unique activations overflowed");
          checked_add(
              output.batch_work.same_lane_duplicate_suppressions,
              successful_count - unique_count,
              "batched host frontier push same-lane duplicates overflowed");

          std::uint64_t claimed = 0U;
          if (old_mask == 0U) {
            claimed = 1U;
            const std::uint64_t claim =
                output.controller.frontier_size[write_slot]++;
            checked_add(
                output.batch_work.queue_claims,
                1U,
                "batched host frontier push queue claims overflowed");
            if (collect_debug) {
              checked_add(
                  work.queue_claims,
                  1U,
                  "batched host frontier push common queue claims overflowed");
            }
            const std::uint64_t logical_size = claim + 1U;
            output.batch_work.maximum_queue_size = std::max(
                output.batch_work.maximum_queue_size, logical_size);
            if (collect_light) {
              work.maximum_queue_size =
                  std::max(work.maximum_queue_size, logical_size);
            }
            if (claim < state.queue_capacity) {
              state.queues[write_slot][static_cast<std::size_t>(claim)] =
                  destination;
            } else {
              if (output.controller.error_bits == device_error::none) {
                checked_add(
                    output.batch_work.overflow_events,
                    1U,
                    "batched host frontier push overflow events overflowed");
                if (collect_debug) {
                  checked_add(
                      work.overflow_events,
                      1U,
                      "batched host frontier push common overflow events overflowed");
                }
              }
              output.controller.error_bits = device_error::queue_overflow;
            }
          }
          require(
              unique_count >= claimed,
              "batched host frontier push claim lacks a new lane activation");
          checked_add(
              output.batch_work.queue_entries_saved_by_lane_merging,
              unique_count - claimed,
              "batched host frontier push merge savings overflowed");
          checked_add(
              output.batch_work.duplicate_suppressions,
              successful_count - claimed,
              "batched host frontier push duplicate suppressions overflowed");
          if (collect_debug) {
            checked_add(
                work.duplicate_suppressions,
                successful_count - claimed,
                "batched host frontier push common duplicates overflowed");
          }
        }
      };

      if (description.run_representation ==
          BatchRunRepresentation::retained_per_run_masks) {
        const std::size_t run_begin = graph.csr_row_run_offsets[source];
        const std::size_t run_end = graph.csr_row_run_offsets[source + 1U];
        for (std::size_t run = run_begin; run < run_end; ++run) {
          process_run(run, description.csr_run_lane_masks[run]);
        }
      } else {
        const std::size_t compact = validated.compact_index_by_vertex[source];
        if (compact == std::numeric_limits<std::size_t>::max()) {
          throw std::logic_error{
              "batched host frontier push compact source mapping drifted"};
        }
        const std::size_t descriptor_begin =
            description.csr_descriptor_offsets_by_union_vertex[compact];
        const std::size_t descriptor_end =
            description.csr_descriptor_offsets_by_union_vertex[compact + 1U];
        for (std::size_t position = descriptor_begin;
             position < descriptor_end;
             ++position) {
          const RunLaneMaskDescriptor descriptor =
              description.csr_run_descriptors[position];
          process_run(descriptor.run_id, descriptor.lane_mask);
        }
      }
    }

    for (const LaneMask mask : state.activity_masks[read_slot]) {
      if (mask != 0U) {
        detail::canonicalize_frontier_error(
            output.controller,
            DeviceStopReason::invalid_controller_state,
            device_error::invalid_controller_state);
        return;
      }
    }
    if (output.controller.error_bits == device_error::none) {
      const std::uint64_t next_size =
          output.controller.frontier_size[write_slot];
      if (next_size > state.queue_capacity) {
        detail::canonicalize_frontier_error(
            output.controller,
            DeviceStopReason::invalid_controller_state,
            device_error::invalid_controller_state);
        return;
      }
      std::vector<bool> next_queued(graph.vertex_count, false);
      LaneMask verified_next_lanes = 0U;
      for (std::uint64_t entry = 0U; entry < next_size; ++entry) {
        const std::uint32_t vertex =
            state.queues[write_slot][static_cast<std::size_t>(entry)];
        if (vertex >= graph.vertex_count || next_queued[vertex] ||
            state.activity_masks[write_slot][vertex] == 0U) {
          detail::canonicalize_frontier_error(
              output.controller,
              DeviceStopReason::invalid_controller_state,
              device_error::invalid_controller_state);
          return;
        }
        next_queued[vertex] = true;
        verified_next_lanes |= state.activity_masks[write_slot][vertex];
      }
      for (std::size_t vertex = 0U; vertex < graph.vertex_count; ++vertex) {
        if ((state.activity_masks[write_slot][vertex] != 0U) !=
            next_queued[vertex]) {
          detail::canonicalize_frontier_error(
              output.controller,
              DeviceStopReason::invalid_controller_state,
              device_error::invalid_controller_state);
          return;
        }
      }
      if (verified_next_lanes !=
          output.controller.next_frontier_lane_mask) {
        detail::canonicalize_frontier_error(
            output.controller,
            DeviceStopReason::invalid_controller_state,
            device_error::invalid_controller_state);
        return;
      }
    }

    const LaneMask next_lanes = output.controller.next_frontier_lane_mask;
    pending_no_next_lanes = current_lane_union & ~next_lanes;
    pending_convergence_round = round_number;
    pending_convergence_evidence = true;
    if (output.controller.frontier_size[write_slot] == 0U) {
      checked_add(
          output.batch_work.empty_frontier_rounds,
          1U,
          "batched host frontier push empty rounds overflowed");
      if (collect_light) {
        checked_add(
            work.empty_frontier_rounds,
            1U,
            "batched host frontier push common empty rounds overflowed");
      }
    }
  };

  const auto execute_pair = [&]() {
    execute_round();
    const std::uint32_t pre_transition_error_bits =
        output.controller.error_bits;
    const BatchedFrontierAdvanceResult transition =
        advance_batched_frontier_controller(output.controller);
    const bool publish_clean_round =
        pending_convergence_evidence &&
        pre_transition_error_bits == device_error::none &&
        (transition == BatchedFrontierAdvanceResult::continue_execution ||
         transition == BatchedFrontierAdvanceResult::converged ||
         transition == BatchedFrontierAdvanceResult::maximum_rounds);
    if (publish_clean_round) {
      for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
        const LaneMask bit = lane_bit(lane);
        if ((pending_no_next_lanes & bit) != 0U &&
            output.convergence_round_by_lane[lane] == 0U) {
          output.convergence_round_by_lane[lane] =
              pending_convergence_round;
        }
      }
    }
    pending_no_next_lanes = 0U;
    pending_convergence_round = 0U;
    pending_convergence_evidence = false;
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
          "batched host frontier push dispatches overflowed");
      checked_add(
          output.queued_round_pairs,
          1U,
          "batched host frontier push queued pairs overflowed");
      execute_pair();
      checked_add(
          work.controller_copies,
          1U,
          "batched host frontier push controller copies overflowed");
      checked_add(
          work.host_synchronizations,
          1U,
          "batched host frontier push synchronizations overflowed");
      checked_add(
          work.host_checks,
          1U,
          "batched host frontier push host checks overflowed");
      checked_add(
          output.completed_host_chunks,
          1U,
          "batched host frontier push host chunks overflowed");
    }
  } else {
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      for (std::uint32_t pair = 0U; pair < options.rounds_per_chunk; ++pair) {
        checked_add(
            work.kernel_dispatches,
            2U,
            "batched host frontier push dispatches overflowed");
        checked_add(
            output.queued_round_pairs,
            1U,
            "batched host frontier push queued pairs overflowed");
        execute_pair();
      }
      checked_add(
          work.controller_copies,
          1U,
          "batched host frontier push controller copies overflowed");
      checked_add(
          work.host_synchronizations,
          1U,
          "batched host frontier push synchronizations overflowed");
      checked_add(
          work.host_checks,
          1U,
          "batched host frontier push host checks overflowed");
      checked_add(
          output.completed_host_chunks,
          1U,
          "batched host frontier push host chunks overflowed");
    }
  }

  require(
      output.batch_work.active_vertex_lane_pairs >=
              output.batch_work.frontier_vertex_entries &&
          output.batch_work.lane_edge_relaxations >=
              output.batch_work.csr_edge_loads,
      "batched host frontier push sharing counters underflowed");
  output.batch_work.shared_vertex_entries_saved =
      output.batch_work.active_vertex_lane_pairs -
      output.batch_work.frontier_vertex_entries;
  output.batch_work.shared_edge_lane_work_saved =
      output.batch_work.lane_edge_relaxations -
      output.batch_work.csr_edge_loads;
  require(
      output.batch_work.csr_runs_considered ==
              output.batch_work.csr_runs_visited +
                  output.batch_work.csr_runs_skipped &&
          output.batch_work.distance_atomic_source_loads ==
              output.batch_work.distance_atomic_attempts &&
          output.batch_work.distance_atomic_attempts ==
              output.batch_work.lane_edge_relaxations &&
          output.batch_work.successful_distance_atomic_updates ==
              output.batch_work.unique_next_vertex_lane_activations +
                  output.batch_work.same_lane_duplicate_suppressions &&
          output.batch_work.unique_next_vertex_lane_activations ==
              output.batch_work.queue_claims +
                  output.batch_work.queue_entries_saved_by_lane_merging &&
          output.batch_work.duplicate_suppressions ==
              output.batch_work.queue_entries_saved_by_lane_merging +
                  output.batch_work.same_lane_duplicate_suppressions,
      "batched host frontier push semantic counter identity failed");

  const std::uint64_t completed_rounds = output.controller.rounds_completed;
  const std::uint64_t valid_lanes =
      static_cast<std::uint64_t>(std::popcount(batch.valid_lane_mask));
  output.batch_work.valid_lane_round_capacity = checked_product(
      completed_rounds,
      valid_lanes,
      "batched host frontier push valid lane capacity overflowed");
  output.batch_work.lane_width_round_capacity = checked_product(
      completed_rounds,
      batch.lane_width,
      "batched host frontier push lane-width capacity overflowed");
  output.batch_work.wave32_lane_round_capacity = checked_product(
      completed_rounds,
      maximum_batch_lanes,
      "batched host frontier push wave32 capacity overflowed");
  require(
      output.batch_work.active_lane_rounds <=
              output.batch_work.valid_lane_round_capacity &&
          output.batch_work.lane_width_round_capacity <=
              output.batch_work.wave32_lane_round_capacity,
      "batched host frontier push lane capacity invariant failed");
  output.batch_work.inactive_valid_lane_rounds =
      output.batch_work.valid_lane_round_capacity -
      output.batch_work.active_lane_rounds;
  output.batch_work.padded_lane_round_capacity = checked_product(
      completed_rounds,
      static_cast<std::uint64_t>(batch.lane_width) - valid_lanes,
      "batched host frontier push padding capacity overflowed");
  output.batch_work.unused_wave_lane_round_capacity =
      output.batch_work.wave32_lane_round_capacity -
      output.batch_work.lane_width_round_capacity;

  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    if ((batch.valid_lane_mask & lane_bit(lane)) == 0U) {
      continue;
    }
    const std::uint64_t convergence_round =
        output.convergence_round_by_lane[lane];
    if (convergence_round == 0U) {
      continue;
    }
    require(
        convergence_round <= completed_rounds,
        "batched host frontier push convergence round exceeds batch rounds");
    const std::uint64_t tail = completed_rounds - convergence_round;
    output.tail_rounds_by_lane[lane] = tail;
    checked_add(
        output.batch_work.tail_lane_rounds,
        tail,
        "batched host frontier push tail lane rounds overflowed");
    checked_add(
        output.batch_work.tail_lane_rounds_without_frontier_work,
        tail,
        "batched host frontier push no-work tail rounds overflowed");
  }

  output.distance_bits = std::move(state.distances);
  output.distances.reserve(output.distance_bits.size());
  for (const std::uint32_t bits : output.distance_bits) {
    output.distances.push_back(dense_atomic_bits_float(bits));
  }
  output.frontier_queues = std::move(state.queues);
  output.frontier_lane_masks = std::move(state.activity_masks);

  LaneMask reached = 0U;
  LaneMask miss = 0U;
  if (output.controller.stop_reason ==
      static_cast<std::uint32_t>(DeviceStopReason::converged)) {
    for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
      const LaneMask bit = lane_bit(lane);
      if ((batch.valid_lane_mask & bit) == 0U) {
        continue;
      }
      if (all_targets_reached(description, output.distance_bits, lane)) {
        reached |= bit;
      } else {
        miss |= bit;
      }
    }
  }

  output.result.engine_kind =
      static_cast<std::uint32_t>(EngineKind::frontier_push);
  output.result.control_mode = static_cast<std::uint32_t>(options.control_mode);
  output.result.status =
      make_frontier_run_status(output.controller, reached, miss);
  output.result.work = work;
  if (validate_device_controller(output.controller) !=
          DeviceControllerError::none ||
      validate_device_run_status(output.result.status) !=
          DeviceRunStatusError::none) {
    throw std::logic_error{
        "portable batched frontier push produced invalid terminal state"};
  }
  return output;
}

}  // namespace bfnew
