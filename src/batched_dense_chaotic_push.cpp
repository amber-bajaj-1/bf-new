#include "bfnew/batched_dense_chaotic_push.hpp"

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
      "batched host dense push requires valid run options");
  require(
      options.engine == EngineKind::dense_chaotic_push,
      "batched host dense push rejects another engine kind");
  require(
      supported_batched_dense_width(batch.lane_width),
      "batched host dense push supports widths 1, 8, 16, and 32 only");
  require(
      options.enable_per_lane_convergence <= 1U,
      "batched host dense push requires a Boolean convergence toggle");

  const std::size_t width = batch.lane_width;
  require(
      batch.query_indices_by_lane.size() == width &&
          batch.query_ids_by_lane.size() == width &&
          batch.expansion_generations_by_lane.size() == width &&
          batch.union_tiles.size() == batch.union_tile_lane_masks.size() &&
          !batch.union_tiles.empty() && batch.valid_lane_mask != 0U &&
          is_low_lane_mask(batch.valid_lane_mask) &&
          (batch.valid_lane_mask & ~width_mask(batch.lane_width)) == 0U,
      "batched host dense push received an invalid plan entry shape");
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
      "batched host dense push received an invalid device description shape");

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
      "batched host dense push received an invalid CSR device graph");

  std::vector<std::uint32_t> tile_begins(tile_count, 0U);
  std::vector<std::uint32_t> tile_ends(tile_count, 0U);
  std::size_t owner_cursor = 0U;
  for (std::size_t tile = 0U; tile < tile_count; ++tile) {
    tile_begins[tile] = static_cast<std::uint32_t>(owner_cursor);
    while (owner_cursor < vertex_count && graph.owner_tiles[owner_cursor] == tile) {
      ++owner_cursor;
    }
    tile_ends[tile] = static_cast<std::uint32_t>(owner_cursor);
    if (owner_cursor < vertex_count) {
      require(
          graph.owner_tiles[owner_cursor] < tile_count &&
              graph.owner_tiles[owner_cursor] > tile,
          "batched host dense push requires contiguous spatial tile ownership");
    }
  }
  require(
      owner_cursor == vertex_count,
      "batched host dense push owner tile is out of range");

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
          "batched host dense push CSR runs do not cover a row exactly");
      const std::uint32_t destination_tile =
          graph.csr_run_destination_tiles[run];
      require(
          destination_tile < tile_count,
          "batched host dense push CSR run destination tile is out of range");
      for (std::size_t edge = begin; edge < end; ++edge) {
        const std::uint32_t destination = graph.csr_destinations[edge];
        const float weight = graph.csr_weights[edge];
        require(
            destination < vertex_count &&
                graph.owner_tiles[destination] == destination_tile &&
                std::isfinite(weight) && weight >= 0.0F &&
                !(weight == 0.0F && std::signbit(weight)),
            "batched host dense push CSR edge disagrees with its tile run");
      }
      cursor = end;
    }
    require(
        cursor == edge_end,
        "batched host dense push CSR runs omit part of a source row");
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
      "batched host dense push query IDs are not unique");

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
          "batched host dense push padded lane contains semantic payload");
      continue;
    }

    const std::uint32_t query_index = batch.query_indices_by_lane[lane];
    require(
        query_index != invalid_batch_query_index &&
            query_index < queries_by_plan_index.size(),
        "batched host dense push valid lane has an invalid query index");
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
        "batched host dense push lane identity or terminal payload is invalid");
    for (std::size_t earlier_lane = 0U; earlier_lane < lane; ++earlier_lane) {
      if ((resolved_query_ids & lane_bit(earlier_lane)) != 0U) {
        require(
            batch.query_ids_by_lane[earlier_lane] != query_id,
            "batched host dense push assigns one query to multiple lanes");
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
        "batched host dense push tile mask differs from lane query tiles");
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
        "batched host dense push union-tile ledger is not canonical");
    const BatchVertexRange range =
        description.selected_vertex_ranges[union_position];
    require(
        range.begin == tile_begins[tile] && range.end == tile_ends[tile] &&
            range.begin <= range.end && range.lane_mask == expected_mask,
        "batched host dense push selected range differs from its tile");
    const std::uint64_t tile_vertices = range.end - range.begin;
    checked_add(
        union_vertex_count,
        tile_vertices,
        "batched host dense push union vertex count overflowed");
    checked_add(
        validated.selected_tile_lane_positions,
        static_cast<std::uint64_t>(std::popcount(expected_mask)),
        "batched host dense push selected tile/lane count overflowed");
    for (std::size_t lane = 0U; lane < width; ++lane) {
      if ((expected_mask & lane_bit(lane)) != 0U) {
        checked_add(
            validated.selected_vertices_by_lane[lane],
            tile_vertices,
            "batched host dense push per-lane vertex count overflowed");
      }
    }
    ++union_position;
  }
  require(
      union_position == batch.union_tiles.size() &&
          union_position == description.union_tiles.size() &&
          union_position == description.selected_vertex_ranges.size() &&
          batch.union_vertex_count == union_vertex_count,
      "batched host dense push union vertex count or ledger is invalid");

  for (std::size_t lane = 0U; lane < width; ++lane) {
    const LaneMask bit = lane_bit(lane);
    if ((batch.valid_lane_mask & bit) == 0U) {
      continue;
    }
    const RouteQuery& query = *validated.queries_by_lane[lane];
    for (const VertexId source : query.sources) {
      require(
          (expected_tile_masks[graph.owner_tiles[source.value()]] & bit) != 0U,
          "batched host dense push source is outside its lane tiles");
    }
    for (const VertexId target : query.targets) {
      require(
          (expected_tile_masks[graph.owner_tiles[target.value()]] & bit) != 0U,
          "batched host dense push target is outside its lane tiles");
    }
    require(
        description.selected_vertex_counts_by_lane[lane] ==
            validated.selected_vertices_by_lane[lane],
        "batched host dense push per-lane vertex estimate is inconsistent");
    checked_add(
        validated.selected_lane_vertices,
        validated.selected_vertices_by_lane[lane],
        "batched host dense push selected lane vertex count overflowed");
  }
  require(
      batch.selected_lane_vertex_count == validated.selected_lane_vertices,
      "batched host dense push plan selected lane vertices are inconsistent");

  std::vector<RunLaneMaskDescriptor> expected_descriptors;
  std::vector<std::uint32_t> expected_descriptor_offsets{0U};
  std::vector<std::uint32_t> expected_touched_runs;
  std::uint64_t expected_runs_considered = 0U;
  std::uint64_t expected_active_runs = 0U;
  std::uint64_t expected_lane_edge_pairs = 0U;
  std::uint64_t expected_union_edges = 0U;
  std::size_t compact_vertex = 0U;
  for (std::size_t source = 0U; source < vertex_count; ++source) {
    const std::uint32_t source_tile = graph.owner_tiles[source];
    const LaneMask source_mask = expected_tile_masks[source_tile];
    const std::size_t run_begin = graph.csr_row_run_offsets[source];
    const std::size_t run_end = graph.csr_row_run_offsets[source + 1U];
    if (source_mask != 0U) {
      validated.compact_index_by_vertex[source] = compact_vertex;
      ++compact_vertex;
      checked_add(
          expected_runs_considered,
          run_end - run_begin,
          "batched host dense push CSR run count overflowed");
    }
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const std::uint32_t destination_tile =
          graph.csr_run_destination_tiles[run];
      const LaneMask expected_mask =
          source_mask & expected_tile_masks[destination_tile];
      const std::uint64_t edges =
          static_cast<std::uint64_t>(graph.csr_run_edge_offsets[run + 1U]) -
          graph.csr_run_edge_offsets[run];
      if (source_mask != 0U && expected_tile_masks[destination_tile] != 0U) {
        checked_add(
            expected_union_edges,
            edges,
            "batched host dense push union edge count overflowed");
      }
      if (expected_mask == 0U) {
        continue;
      }
      checked_add(
          expected_active_runs,
          1U,
          "batched host dense push active run count overflowed");
      expected_touched_runs.push_back(static_cast<std::uint32_t>(run));
      expected_descriptors.push_back(RunLaneMaskDescriptor{
          static_cast<std::uint32_t>(run), expected_mask});
      const std::uint64_t lane_pairs = checked_product(
          edges,
          static_cast<std::uint64_t>(std::popcount(expected_mask)),
          "batched host dense push lane-edge count overflowed");
      checked_add(
          expected_lane_edge_pairs,
          lane_pairs,
          "batched host dense push lane-edge count overflowed");
      for (std::size_t lane = 0U; lane < width; ++lane) {
        if ((expected_mask & lane_bit(lane)) != 0U) {
          checked_add(
              validated.selected_edges_by_lane[lane],
              edges,
              "batched host dense push per-lane edge count overflowed");
        }
      }
    }
    if (source_mask != 0U) {
      require(
          expected_descriptors.size() <=
              std::numeric_limits<std::uint32_t>::max(),
          "batched host dense push descriptor count exceeds 32 bits");
      expected_descriptor_offsets.push_back(
          static_cast<std::uint32_t>(expected_descriptors.size()));
    }
  }
  require(
      compact_vertex == union_vertex_count &&
          batch.union_edge_estimate == expected_union_edges,
      "batched host dense push compact vertex or union edge estimate is invalid");

  if (description.run_representation ==
      BatchRunRepresentation::retained_per_run_masks) {
    require(
        description.csr_run_lane_masks.size() == run_count &&
            description.csr_run_descriptors.empty() &&
            description.csr_descriptor_offsets_by_union_vertex.empty(),
        "batched host dense push retained CSR storage has an invalid shape");
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
            "batched host dense push retained CSR mask is not endpoint-exact");
      }
    }
    require(
        description.touched_csr_runs == expected_touched_runs,
        "batched host dense push retained CSR touched ledger is inconsistent");
  } else if (
      description.run_representation ==
      BatchRunRepresentation::compact_nonzero_descriptors) {
    require(
        description.csr_run_lane_masks.empty() &&
            description.touched_csr_runs.empty() &&
            description.csr_run_descriptors == expected_descriptors &&
            description.csr_descriptor_offsets_by_union_vertex ==
                expected_descriptor_offsets,
        "batched host dense push compact CSR descriptors are not endpoint-exact");
  } else {
    throw std::invalid_argument{
        "batched host dense push received an unknown run representation"};
  }

  for (std::size_t lane = 0U; lane < width; ++lane) {
    const LaneMask bit = lane_bit(lane);
    if ((batch.valid_lane_mask & bit) == 0U) {
      continue;
    }
    require(
        description.selected_edge_estimates_by_lane[lane] ==
            validated.selected_edges_by_lane[lane],
        "batched host dense push per-lane edge estimate is inconsistent");
    checked_add(
        validated.selected_lane_edges,
        validated.selected_edges_by_lane[lane],
        "batched host dense push selected lane edge count overflowed");
  }
  require(
      batch.selected_lane_edge_estimate == validated.selected_lane_edges &&
          description.run_report.csr_runs_visited ==
              expected_runs_considered &&
          description.run_report.active_csr_runs == expected_active_runs &&
          description.run_report.csr_lane_edge_pairs ==
              expected_lane_edge_pairs,
      "batched host dense push CSR preparation report is inconsistent");
  return validated;
}

[[nodiscard]] bool all_targets_reached(
    const BatchDeviceDescription& description,
    const std::vector<std::uint32_t>& distance_bits,
    const std::size_t lane) {
  const std::size_t begin = description.target_offsets[lane];
  const std::size_t end = description.target_offsets[lane + 1U];
  for (std::size_t position = begin; position < end; ++position) {
    const std::size_t index = static_cast<std::size_t>(
        batched_dense_distance_index(
            description.targets[position],
            static_cast<std::uint32_t>(lane),
            description.lane_width));
    if (!std::isfinite(dense_atomic_bits_float(distance_bits[index]))) {
      return false;
    }
  }
  return true;
}

}  // namespace

HostBatchedDenseRunResult run_host_batched_dense_chaotic_push(
    const DeviceGraphLayout32& graph,
    const std::span<const RouteQuery> queries,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description,
    const GpuRunOptions& options,
    const DenseHostSchedule schedule) {
  const ValidatedHostBatch validated =
      validate_host_batch(graph, queries, batch, description, options);
  require(
      schedule == DenseHostSchedule::csr_forward ||
          schedule == DenseHostSchedule::csr_reverse ||
          schedule == DenseHostSchedule::alternating,
      "batched host dense push received an unknown schedule");

  const std::uint64_t element_count64 = checked_product(
      graph.vertex_count,
      batch.lane_width,
      "batched host dense push distance element count overflowed");
  require(
      element_count64 <= std::numeric_limits<std::size_t>::max(),
      "batched host dense push distance allocation exceeds host size");
  const std::size_t element_count = static_cast<std::size_t>(element_count64);

  HostBatchedDenseRunResult output;
  output.distance_bits.assign(element_count, positive_infinity_bits);
  output.rounds_executed_by_lane.assign(batch.lane_width, 0U);
  output.convergence_round_by_lane.assign(batch.lane_width, 0U);
  output.tail_rounds_by_lane.assign(batch.lane_width, 0U);
  output.controller =
      initialize_device_controller(options, batch.valid_lane_mask);

  output.batch_work.selected_tile_lane_positions =
      validated.selected_tile_lane_positions;
  output.batch_work.union_tile_lane_positions = checked_product(
      batch.union_tiles.size(),
      static_cast<std::uint64_t>(std::popcount(batch.valid_lane_mask)),
      "batched host dense push union tile/lane count overflowed");
  output.batch_work.distance_reset_bytes = checked_product(
      validated.selected_lane_vertices,
      sizeof(std::uint32_t),
      "batched host dense push reset byte count overflowed");
  output.batch_work.source_seed_write_bytes = checked_product(
      description.sources.size(),
      sizeof(std::uint32_t),
      "batched host dense push source seed byte count overflowed");

  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    if ((batch.valid_lane_mask & lane_bit(lane)) == 0U) {
      continue;
    }
    const std::size_t begin = description.source_offsets[lane];
    const std::size_t end = description.source_offsets[lane + 1U];
    for (std::size_t position = begin; position < end; ++position) {
      const std::size_t index = static_cast<std::size_t>(
          batched_dense_distance_index(
              description.sources[position],
              static_cast<std::uint32_t>(lane),
              batch.lane_width));
      output.distance_bits[index] = 0U;
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
    if (output.controller.distance_read_slot != 0U ||
        output.controller.distance_write_slot != 0U) {
      detail::fail_dense_controller(output.controller);
      return;
    }

    const LaneMask execute = output.controller.execute_lane_mask;
    const std::uint64_t round_number = output.controller.rounds_completed + 1U;
    const std::uint64_t executing_lanes =
        static_cast<std::uint64_t>(std::popcount(execute));
    checked_add(
        output.batch_work.active_lane_rounds,
        executing_lanes,
        "batched host dense push active lane-round count overflowed");
    checked_add(
        output.batch_work.full_edge_rounds,
        1U,
        "batched host dense push full scan count overflowed");
    if (collect_light) {
      checked_add(
          work.active_lane_rounds,
          executing_lanes,
          "batched host dense push common active lane-round count overflowed");
      checked_add(
          work.full_edge_rounds,
          1U,
          "batched host dense push common full scan count overflowed");
    }
    for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
      if ((execute & lane_bit(lane)) != 0U) {
        checked_add(
            output.rounds_executed_by_lane[lane],
            1U,
            "batched host dense push per-lane round count overflowed");
      }
    }

    const bool reverse = schedule == DenseHostSchedule::csr_reverse ||
                         (schedule == DenseHostSchedule::alternating &&
                          (output.controller.rounds_completed & 1U) != 0U);
    LaneMask changed = 0U;
    const std::size_t vertex_count = graph.vertex_count;
    for (std::size_t visit = 0U; visit < vertex_count; ++visit) {
      const std::size_t source = reverse ? vertex_count - 1U - visit : visit;
      const LaneMask source_lanes =
          description.tile_lane_masks[graph.owner_tiles[source]] & execute;
      if (source_lanes == 0U) {
        continue;
      }
      const std::uint64_t source_lane_count =
          static_cast<std::uint64_t>(std::popcount(source_lanes));
      checked_add(
          output.batch_work.active_source_lane_evaluations,
          source_lane_count,
          "batched host dense push source/lane count overflowed");
      if (collect_light) {
        checked_add(
            work.active_vertices,
            source_lane_count,
            "batched host dense push common source/lane count overflowed");
      }

      const auto process_run = [&](const std::size_t run,
                                   const LaneMask prepared_mask) {
        checked_add(
            output.batch_work.csr_runs_considered,
            1U,
            "batched host dense push considered run count overflowed");
        if (collect_debug) {
          checked_add(
              work.mask_operations,
              1U,
              "batched host dense push common mask count overflowed");
        }
        const LaneMask active_mask = prepared_mask & execute;
        if (active_mask == 0U) {
          checked_add(
              output.batch_work.csr_runs_skipped,
              1U,
              "batched host dense push skipped run count overflowed");
          return;
        }
        checked_add(
            output.batch_work.csr_runs_visited,
            1U,
            "batched host dense push visited run count overflowed");
        checked_add(
            output.batch_work.active_lanes_over_visited_runs,
            static_cast<std::uint64_t>(std::popcount(active_mask)),
            "batched host dense push visited-run lane count overflowed");

        const std::size_t edge_begin = graph.csr_run_edge_offsets[run];
        const std::size_t edge_end = graph.csr_run_edge_offsets[run + 1U];
        const std::size_t edge_count = edge_end - edge_begin;
        for (std::size_t edge_visit = 0U; edge_visit < edge_count;
             ++edge_visit) {
          const std::size_t edge =
              reverse ? edge_end - 1U - edge_visit : edge_begin + edge_visit;
          checked_add(
              output.batch_work.csr_edge_loads,
              1U,
              "batched host dense push shared edge-load count overflowed");
          const std::uint32_t destination = graph.csr_destinations[edge];
          const float weight = graph.csr_weights[edge];
          for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
            const LaneMask bit = lane_bit(lane);
            if ((active_mask & bit) == 0U) {
              continue;
            }
            checked_add(
                output.batch_work.lane_edge_relaxations,
                1U,
                "batched host dense push lane-edge count overflowed");
            checked_add(
                output.batch_work.atomic_source_loads,
                1U,
                "batched host dense push atomic source-load count overflowed");
            checked_add(
                output.batch_work.atomic_min_attempts,
                1U,
                "batched host dense push atomic-min count overflowed");
            if (collect_light) {
              checked_add(
                  work.edges_examined,
                  1U,
                  "batched host dense push common edge count overflowed");
            }
            if (collect_debug) {
              checked_add(
                  work.atomic_attempts,
                  1U,
                  "batched host dense push common atomic count overflowed");
            }

            const std::size_t source_index = static_cast<std::size_t>(
                batched_dense_distance_index(
                    static_cast<std::uint32_t>(source),
                    static_cast<std::uint32_t>(lane),
                    batch.lane_width));
            const float source_distance =
                dense_atomic_bits_float(output.distance_bits[source_index]);
            const std::uint32_t candidate_bits = dense_atomic_float_bits(
                source_distance + weight);
            const std::size_t destination_index = static_cast<std::size_t>(
                batched_dense_distance_index(
                    destination,
                    static_cast<std::uint32_t>(lane),
                    batch.lane_width));
            const DenseAtomicMinResult update = dense_atomic_min_bits(
                output.distance_bits[destination_index], candidate_bits);
            if (!update.decreased) {
              continue;
            }
            output.distance_bits[destination_index] = update.final_bits;
            changed |= bit;
            checked_add(
                output.batch_work.successful_atomic_updates,
                1U,
                "batched host dense push successful update count overflowed");
            if (collect_light) {
              checked_add(
                  work.successful_decreases,
                  1U,
                  "batched host dense push common decrease count overflowed");
            }
            if (collect_debug) {
              checked_add(
                  work.successful_atomic_updates,
                  1U,
                  "batched host dense push common update count overflowed");
            }
          }
        }
      };

      if (description.run_representation ==
          BatchRunRepresentation::retained_per_run_masks) {
        const std::size_t run_begin = graph.csr_row_run_offsets[source];
        const std::size_t run_end = graph.csr_row_run_offsets[source + 1U];
        const std::size_t run_count = run_end - run_begin;
        for (std::size_t run_visit = 0U; run_visit < run_count; ++run_visit) {
          const std::size_t run =
              reverse ? run_end - 1U - run_visit : run_begin + run_visit;
          process_run(run, description.csr_run_lane_masks[run]);
        }
      } else {
        const std::size_t compact =
            validated.compact_index_by_vertex[source];
        if (compact == std::numeric_limits<std::size_t>::max()) {
          throw std::logic_error{
              "batched host dense push compact source mapping drifted"};
        }
        const std::size_t descriptor_begin =
            description.csr_descriptor_offsets_by_union_vertex[compact];
        const std::size_t descriptor_end =
            description.csr_descriptor_offsets_by_union_vertex[compact + 1U];
        const std::size_t descriptor_count =
            descriptor_end - descriptor_begin;
        for (std::size_t descriptor_visit = 0U;
             descriptor_visit < descriptor_count;
             ++descriptor_visit) {
          const std::size_t position =
              reverse ? descriptor_end - 1U - descriptor_visit
                      : descriptor_begin + descriptor_visit;
          const RunLaneMaskDescriptor descriptor =
              description.csr_run_descriptors[position];
          process_run(descriptor.run_id, descriptor.lane_mask);
        }
      }
    }

    output.controller.changed_lane_mask |= changed;
    const LaneMask newly_observed_converged = execute & ~changed;
    for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
      if ((newly_observed_converged & lane_bit(lane)) != 0U &&
          output.convergence_round_by_lane[lane] == 0U) {
        output.convergence_round_by_lane[lane] = round_number;
      }
    }
    if (changed != 0U) {
      checked_add(
          output.batch_work.changed_round_publications,
          1U,
          "batched host dense push changed publication count overflowed");
      if (collect_debug) {
        checked_add(
            work.changed_flag_updates,
            1U,
            "batched host dense push common changed count overflowed");
      }
    }
  };

  const auto execute_pair = [&]() {
    execute_round();
    static_cast<void>(advance_dense_controller(output.controller));
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
          "batched host dense push dispatch count overflowed");
      checked_add(
          output.queued_round_pairs,
          1U,
          "batched host dense push queued pair count overflowed");
      execute_pair();
      checked_add(
          work.controller_copies,
          1U,
          "batched host dense push controller copy count overflowed");
      checked_add(
          work.host_synchronizations,
          1U,
          "batched host dense push synchronization count overflowed");
      checked_add(
          work.host_checks,
          1U,
          "batched host dense push host check count overflowed");
      checked_add(
          output.completed_host_chunks,
          1U,
          "batched host dense push completed chunk count overflowed");
    }
  } else {
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      for (std::uint32_t pair = 0U; pair < options.rounds_per_chunk; ++pair) {
        checked_add(
            work.kernel_dispatches,
            2U,
            "batched host dense push dispatch count overflowed");
        checked_add(
            output.queued_round_pairs,
            1U,
            "batched host dense push queued pair count overflowed");
        execute_pair();
      }
      checked_add(
          work.controller_copies,
          1U,
          "batched host dense push controller copy count overflowed");
      checked_add(
          work.host_synchronizations,
          1U,
          "batched host dense push synchronization count overflowed");
      checked_add(
          work.host_checks,
          1U,
          "batched host dense push host check count overflowed");
      checked_add(
          output.completed_host_chunks,
          1U,
          "batched host dense push completed chunk count overflowed");
    }
  }

  const std::uint64_t completed_rounds = output.controller.rounds_completed;
  const std::uint64_t valid_lanes =
      static_cast<std::uint64_t>(std::popcount(batch.valid_lane_mask));
  output.batch_work.valid_lane_round_capacity = checked_product(
      completed_rounds,
      valid_lanes,
      "batched host dense push valid lane-round capacity overflowed");
  output.batch_work.lane_width_round_capacity = checked_product(
      completed_rounds,
      batch.lane_width,
      "batched host dense push width lane-round capacity overflowed");
  output.batch_work.wave32_lane_round_capacity = checked_product(
      completed_rounds,
      maximum_batch_lanes,
      "batched host dense push wave32 lane-round capacity overflowed");
  require(
      output.batch_work.lane_width_round_capacity <=
          output.batch_work.wave32_lane_round_capacity,
      "batched host dense push width exceeds wave32 capacity");
  output.batch_work.unused_wave_lane_round_capacity =
      output.batch_work.wave32_lane_round_capacity -
      output.batch_work.lane_width_round_capacity;
  output.batch_work.padded_lane_round_capacity = checked_product(
      completed_rounds,
      static_cast<std::uint64_t>(batch.lane_width) - valid_lanes,
      "batched host dense push padding capacity overflowed");
  require(
      output.batch_work.active_lane_rounds <=
          output.batch_work.valid_lane_round_capacity,
      "batched host dense push active lane rounds exceed capacity");
  output.batch_work.inactive_valid_lane_rounds =
      output.batch_work.valid_lane_round_capacity -
      output.batch_work.active_lane_rounds;
  output.batch_work.edge_wave_lane_capacity = checked_product(
      output.batch_work.csr_edge_loads,
      maximum_batch_lanes,
      "batched host dense push edge-wave capacity overflowed");
  require(
      output.batch_work.lane_edge_relaxations <=
          output.batch_work.edge_wave_lane_capacity,
      "batched host dense push lane-edge work exceeds wave32 capacity");
  output.batch_work.unused_edge_wave_lane_capacity =
      output.batch_work.edge_wave_lane_capacity -
      output.batch_work.lane_edge_relaxations;

  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    const LaneMask bit = lane_bit(lane);
    if ((batch.valid_lane_mask & bit) == 0U) {
      continue;
    }
    const std::uint64_t convergence_round =
        output.convergence_round_by_lane[lane];
    if (convergence_round == 0U) {
      continue;
    }
    require(
        convergence_round <= completed_rounds,
        "batched host dense push convergence round exceeds batch rounds");
    const std::uint64_t tail = completed_rounds - convergence_round;
    output.tail_rounds_by_lane[lane] = tail;
    checked_add(
        output.batch_work.tail_lane_rounds,
        tail,
        "batched host dense push tail lane-round count overflowed");
    if (options.enable_per_lane_convergence != 0U) {
      checked_add(
          output.batch_work.tail_lane_rounds_avoided,
          tail,
          "batched host dense push avoided tail count overflowed");
      checked_add(
          output.batch_work
              .lane_edge_relaxations_avoided_by_early_convergence,
          checked_product(
              tail,
              validated.selected_edges_by_lane[lane],
              "batched host dense push avoided lane-edge count overflowed"),
          "batched host dense push avoided lane-edge count overflowed");
    } else {
      checked_add(
          output.batch_work.tail_lane_rounds_executed,
          tail,
          "batched host dense push executed tail count overflowed");
    }
  }

  output.distances.reserve(output.distance_bits.size());
  for (const std::uint32_t bits : output.distance_bits) {
    output.distances.push_back(dense_atomic_bits_float(bits));
  }
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
      static_cast<std::uint32_t>(EngineKind::dense_chaotic_push);
  output.result.control_mode = static_cast<std::uint32_t>(options.control_mode);
  output.result.status = make_dense_run_status(output.controller, reached, miss);
  output.result.work = work;
  if (validate_device_controller(output.controller) !=
          DeviceControllerError::none ||
      validate_device_run_status(output.result.status) !=
          DeviceRunStatusError::none) {
    throw std::logic_error{
        "portable batched dense push produced invalid terminal state"};
  }
  return output;
}

}  // namespace bfnew
