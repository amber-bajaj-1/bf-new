#include "bfnew/batched_expansion.hpp"

#include "bfnew/batched_dense_chaotic_push.hpp"
#include "bfnew/batched_frontier_push.hpp"
#include "bfnew/batched_jacobi_pull.hpp"
#include "bfnew/compact_paths.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

[[nodiscard]] std::uint64_t size64(
    const std::size_t value,
    const char* const description) {
  if (value > std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{description};
  }
  return static_cast<std::uint64_t>(value);
}

[[nodiscard]] std::uint32_t size32(
    const std::size_t value,
    const char* const description) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{description};
  }
  return static_cast<std::uint32_t>(value);
}

void checked_add(
    std::uint64_t& destination,
    const std::uint64_t value,
    const char* const description) {
  if (value > std::numeric_limits<std::uint64_t>::max() - destination) {
    throw std::overflow_error{description};
  }
  destination += value;
}

[[nodiscard]] bool try_add_u64(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t& output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

[[nodiscard]] bool try_double_u64(
    const std::uint64_t value,
    std::uint64_t& output) noexcept {
  if (value > std::numeric_limits<std::uint64_t>::max() / 2U) {
    return false;
  }
  output = value * 2U;
  return true;
}

[[nodiscard]] std::uint64_t elapsed_nanoseconds(
    const std::chrono::steady_clock::time_point begin,
    const std::chrono::steady_clock::time_point end) noexcept {
  const auto count =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  if (count <= 0) {
    return 0U;
  }
  using Count = decltype(count);
  if constexpr (sizeof(Count) > sizeof(std::uint64_t)) {
    if (count > static_cast<Count>(std::numeric_limits<std::uint64_t>::max())) {
      return std::numeric_limits<std::uint64_t>::max();
    }
  }
  return static_cast<std::uint64_t>(count);
}

void fingerprint_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
  constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
  for (std::uint32_t byte = 0U; byte < 8U; ++byte) {
    hash ^= value & 0xffU;
    hash *= fnv_prime;
    value >>= 8U;
  }
}

template <typename Id>
void fingerprint_ids(
    std::uint64_t& hash,
    const std::span<const Id> values) noexcept {
  fingerprint_u64(hash, static_cast<std::uint64_t>(values.size()));
  for (const Id value : values) {
    fingerprint_u64(hash, value.value());
  }
}

void fingerprint_u32s(
    std::uint64_t& hash,
    const std::span<const std::uint32_t> values) noexcept {
  fingerprint_u64(hash, static_cast<std::uint64_t>(values.size()));
  for (const std::uint32_t value : values) {
    fingerprint_u64(hash, value);
  }
}

[[nodiscard]] std::uint64_t throughput_milliqueries_per_second(
    const std::uint64_t query_count,
    const std::uint64_t elapsed_ns) noexcept {
  if (query_count == 0U || elapsed_ns == 0U) {
    return 0U;
  }
  constexpr long double scale = 1'000'000'000'000.0L;
  const long double rate =
      static_cast<long double>(query_count) * scale /
      static_cast<long double>(elapsed_ns);
  if (rate >=
      static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return static_cast<std::uint64_t>(rate);
}

[[nodiscard]] bool valid_schedule_policy(
    const ExpansionSchedulePolicy& schedule) noexcept {
  switch (schedule.kind) {
    case ExpansionScheduleKind::one_geometric_ring:
    case ExpansionScheduleKind::doubling_xy_margins:
      return schedule.fixed_ring_size == 0U &&
             schedule.hybrid_small_expansion_count == 0U;
    case ExpansionScheduleKind::fixed_larger_ring:
      return schedule.fixed_ring_size >= 2U &&
             schedule.hybrid_small_expansion_count == 0U;
    case ExpansionScheduleKind::hybrid_small_then_doubling:
      return schedule.fixed_ring_size == 0U &&
             schedule.hybrid_small_expansion_count != 0U;
    case ExpansionScheduleKind::unspecified:
      return false;
  }
  return false;
}

struct LocatedBounds {
  bool present{};
  std::int64_t minimum_x{};
  std::int64_t minimum_y{};
  std::int64_t maximum_x{};
  std::int64_t maximum_y{};
};

[[nodiscard]] LocatedBounds located_bounds(
    const WeightedGraph& graph,
    const std::span<const TileId> selected_tiles) noexcept {
  LocatedBounds bounds;
  for (const TileId tile : selected_tiles) {
    const TileCoordinate coordinate = graph.tile_coordinates()[tile.value()];
    if (!coordinate.has_location) {
      continue;
    }
    if (!bounds.present) {
      bounds = LocatedBounds{
          true,
          coordinate.tile_x,
          coordinate.tile_y,
          coordinate.tile_x,
          coordinate.tile_y};
      continue;
    }
    bounds.minimum_x = std::min(bounds.minimum_x, coordinate.tile_x);
    bounds.minimum_y = std::min(bounds.minimum_y, coordinate.tile_y);
    bounds.maximum_x = std::max(bounds.maximum_x, coordinate.tile_x);
    bounds.maximum_y = std::max(bounds.maximum_y, coordinate.tile_y);
  }
  return bounds;
}

[[nodiscard]] std::int64_t saturating_subtract(
    const std::int64_t value,
    const std::uint64_t amount) noexcept {
  if (amount > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return std::numeric_limits<std::int64_t>::min();
  }
  const std::int64_t narrow = static_cast<std::int64_t>(amount);
  if (value < std::numeric_limits<std::int64_t>::min() + narrow) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return value - narrow;
}

[[nodiscard]] std::int64_t saturating_add(
    const std::int64_t value,
    const std::uint64_t amount) noexcept {
  if (amount > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return std::numeric_limits<std::int64_t>::max();
  }
  const std::int64_t narrow = static_cast<std::int64_t>(amount);
  if (value > std::numeric_limits<std::int64_t>::max() - narrow) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return value + narrow;
}

[[nodiscard]] std::span<const TileId> tile_neighbors(
    const TileDirectory& directory,
    const TileId tile) noexcept {
  const auto offsets = directory.neighbor_tile_offsets();
  const auto neighbors = directory.neighbor_tiles();
  const auto begin = static_cast<std::size_t>(offsets[tile.value()]);
  const auto end = static_cast<std::size_t>(offsets[tile.value() + 1U]);
  return neighbors.subspan(begin, end - begin);
}

struct QueryState {
  RouteQuery query;
  LocatedBounds anchor{};
  std::uint64_t intended_x_margin{};
  std::uint64_t intended_y_margin{};
  std::uint32_t attempts{};
  std::uint32_t scheduled_expansions{};
  std::uint32_t total_expansions{};
  bool fallback_used{};
  bool terminal{};
  ExpansionQueryDisposition disposition{
      ExpansionQueryDisposition::engine_failure};
  std::uint32_t terminal_stop_reason{};
  std::uint32_t terminal_error_bits{};
  std::vector<float> latest_distances;
  std::optional<CompactPathPayload> latest_compact_paths;
};

[[nodiscard]] std::uint64_t comparison_fingerprint(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const std::span<const QueryState> states,
    const BatchedExpansionOptions& options) {
  constexpr std::uint64_t fnv_offset = 14'695'981'039'346'656'037ULL;
  std::uint64_t hash = fnv_offset;
  const DeviceGraphFingerprint graph_fingerprint =
      fingerprint_device_graph_source32(graph, tile_runs);
  fingerprint_u64(hash, graph_fingerprint.first);
  fingerprint_u64(hash, graph_fingerprint.second);
  fingerprint_u64(hash, static_cast<std::uint32_t>(options.run_options.engine));
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(options.run_options.control_mode));
  fingerprint_u64(hash, options.run_options.rounds_per_chunk);
  fingerprint_u64(hash, options.run_options.block_size);
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(options.run_options.grid_policy));
  fingerprint_u64(hash, options.run_options.blocks_per_wgp);
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(options.run_options.instrumentation));
  fingerprint_u64(hash, options.run_options.maximum_rounds);
  fingerprint_u64(hash, options.run_options.enable_per_lane_convergence);
  fingerprint_u64(hash, options.planner_policy.lane_width);
  fingerprint_u64(hash, options.planner_policy.minimum_jaccard_numerator);
  fingerprint_u64(hash, options.planner_policy.minimum_jaccard_denominator);
  fingerprint_u64(
      hash, options.planner_policy.maximum_union_inflation_numerator);
  fingerprint_u64(
      hash, options.planner_policy.maximum_union_inflation_denominator);
  fingerprint_u64(hash, options.execution_configuration_fingerprint);
  fingerprint_u64(hash, options.maximum_expansions);
  fingerprint_u64(hash, static_cast<std::uint32_t>(options.terminal_policy));
  fingerprint_u64(hash, options.enable_compact_paths);
  fingerprint_u64(hash, static_cast<std::uint64_t>(states.size()));
  for (const QueryState& state : states) {
    const RouteQuery& query = state.query;
    fingerprint_u64(hash, query.query_id.value());
    fingerprint_u64(hash, query.expansion_generation);
    fingerprint_ids<VertexId>(hash, query.source_terminals);
    fingerprint_ids<VertexId>(hash, query.target_terminals);
    fingerprint_ids<VertexId>(hash, query.sources);
    fingerprint_ids<VertexId>(hash, query.targets);
    fingerprint_u32s(hash, query.source_terminal_to_source);
    fingerprint_u32s(hash, query.target_terminal_to_target);
    fingerprint_ids<TileId>(hash, query.selected_tiles);
  }
  return hash == 0U ? 1U : hash;
}

[[nodiscard]] std::uint64_t host_execution_configuration_fingerprint(
    const std::uint64_t caller_fingerprint,
    const HostBatchedExpansionOptions& host_options) noexcept {
  constexpr std::uint64_t fnv_offset = 14'695'981'039'346'656'037ULL;
  std::uint64_t hash = fnv_offset;
  fingerprint_u64(hash, caller_fingerprint);
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(host_options.run_representation));
  fingerprint_u64(
      hash, static_cast<std::uint32_t>(host_options.dense_schedule));
  fingerprint_u64(hash, static_cast<std::uint64_t>(
                            host_options.frontier_queue_capacity));
  return hash == 0U ? 1U : hash;
}

[[nodiscard]] bool advance_margin_cursor(
    QueryState& state,
    const ExpansionSchedulePolicy& schedule) noexcept {
  std::uint64_t next_x = state.intended_x_margin;
  std::uint64_t next_y = state.intended_y_margin;
  switch (schedule.kind) {
    case ExpansionScheduleKind::one_geometric_ring:
      if (!try_add_u64(next_x, 1U, next_x) ||
          !try_add_u64(next_y, 1U, next_y)) {
        return false;
      }
      break;
    case ExpansionScheduleKind::fixed_larger_ring:
      if (!try_add_u64(next_x, schedule.fixed_ring_size, next_x) ||
          !try_add_u64(next_y, schedule.fixed_ring_size, next_y)) {
        return false;
      }
      break;
    case ExpansionScheduleKind::doubling_xy_margins:
      if (next_x == 0U) {
        next_x = 1U;
      } else if (!try_double_u64(next_x, next_x)) {
        return false;
      }
      if (next_y == 0U) {
        next_y = 1U;
      } else if (!try_double_u64(next_y, next_y)) {
        return false;
      }
      break;
    case ExpansionScheduleKind::hybrid_small_then_doubling:
      if (state.scheduled_expansions <
          schedule.hybrid_small_expansion_count) {
        if (!try_add_u64(next_x, 1U, next_x) ||
            !try_add_u64(next_y, 1U, next_y)) {
          return false;
        }
      } else {
        if (next_x == 0U) {
          next_x = 1U;
        } else if (!try_double_u64(next_x, next_x)) {
          return false;
        }
        if (next_y == 0U) {
          next_y = 1U;
        } else if (!try_double_u64(next_y, next_y)) {
          return false;
        }
      }
      break;
    case ExpansionScheduleKind::unspecified:
      return false;
  }
  state.intended_x_margin = next_x;
  state.intended_y_margin = next_y;
  return true;
}

[[nodiscard]] std::vector<TileId> expanded_tiles(
    const WeightedGraph& graph,
    const TileDirectory& directory,
    QueryState& state) {
  std::vector<TileId> selected = state.query.selected_tiles;
  const TileId spill = directory.spill_tile();

  if (!state.anchor.present) {
    // A spill-only initial query has no geometric box.  Its first expansion
    // follows the Phase 4 actual-edge spill adjacency to establish one.
    if (std::binary_search(selected.begin(), selected.end(), spill)) {
      for (const TileId neighbor : tile_neighbors(directory, spill)) {
        if (graph.tile_coordinates()[neighbor.value()].has_location) {
          selected.push_back(neighbor);
        }
      }
      std::sort(selected.begin(), selected.end());
      selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
      state.anchor = located_bounds(graph, selected);
      state.intended_x_margin = 0U;
      state.intended_y_margin = 0U;
    }
    return selected;
  }

  const std::int64_t lower_x =
      saturating_subtract(state.anchor.minimum_x, state.intended_x_margin);
  const std::int64_t lower_y =
      saturating_subtract(state.anchor.minimum_y, state.intended_y_margin);
  const std::int64_t upper_x =
      saturating_add(state.anchor.maximum_x, state.intended_x_margin);
  const std::int64_t upper_y =
      saturating_add(state.anchor.maximum_y, state.intended_y_margin);

  for (std::size_t tile = 0U; tile < graph.tile_coordinates().size(); ++tile) {
    const TileCoordinate coordinate = graph.tile_coordinates()[tile];
    if (coordinate.has_location && coordinate.tile_x >= lower_x &&
        coordinate.tile_x <= upper_x && coordinate.tile_y >= lower_y &&
        coordinate.tile_y <= upper_y) {
      selected.push_back(checked_id<TileId>(tile));
    }
  }
  std::sort(selected.begin(), selected.end());
  selected.erase(std::unique(selected.begin(), selected.end()), selected.end());

  // Spill has no synthetic geometry.  It enters only through an actual
  // directory adjacency from a selected located tile and is then preserved.
  if (!std::binary_search(selected.begin(), selected.end(), spill)) {
    for (const TileId tile : selected) {
      if (!graph.tile_coordinates()[tile.value()].has_location) {
        continue;
      }
      const auto neighbors = tile_neighbors(directory, tile);
      if (std::binary_search(neighbors.begin(), neighbors.end(), spill)) {
        selected.push_back(spill);
        std::sort(selected.begin(), selected.end());
        break;
      }
    }
  }
  return selected;
}

[[nodiscard]] std::vector<TileId> full_region_tiles(
    const WeightedGraph& graph) {
  std::vector<TileId> tiles;
  tiles.reserve(graph.tile_coordinates().size());
  for (std::size_t tile = 0U; tile < graph.tile_coordinates().size(); ++tile) {
    tiles.push_back(checked_id<TileId>(tile));
  }
  return tiles;
}

[[nodiscard]] QueryState& find_state(
    std::vector<QueryState>& states,
    const QueryId query_id) {
  const auto position = std::lower_bound(
      states.begin(),
      states.end(),
      query_id,
      [](const QueryState& state, const QueryId id) {
        return state.query.query_id < id;
      });
  if (position == states.end() || position->query.query_id != query_id) {
    throw std::logic_error{"retry-local batch identity was not found by QueryId"};
  }
  return *position;
}

[[nodiscard]] const RouteQuery& find_query(
    const std::span<const RouteQuery> queries,
    const QueryId query_id) {
  const auto position = std::find_if(
      queries.begin(), queries.end(), [&](const RouteQuery& query) {
        return query.query_id == query_id;
      });
  if (position == queries.end()) {
    throw std::logic_error{"batch query identity was not found by QueryId"};
  }
  return *position;
}

void copy_lane_distances(
    const std::span<const float> image,
    const std::uint32_t vertex_count,
    const std::uint32_t lane_width,
    const std::uint32_t lane,
    std::vector<float>& output) {
  output.resize(vertex_count);
  for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(vertex) * lane_width + lane;
    output[vertex] = image[static_cast<std::size_t>(index)];
  }
}

void add_device_work(
    DeviceWorkStatistics& destination,
    const DeviceWorkStatistics& source) {
#define BFNEW_ADD_WORK_FIELD(field) \
  checked_add(destination.field, source.field, "device work counter overflow")
  BFNEW_ADD_WORK_FIELD(edges_examined);
  BFNEW_ADD_WORK_FIELD(successful_decreases);
  BFNEW_ADD_WORK_FIELD(active_vertices);
  BFNEW_ADD_WORK_FIELD(active_lane_rounds);
  destination.maximum_queue_size =
      std::max(destination.maximum_queue_size, source.maximum_queue_size);
  BFNEW_ADD_WORK_FIELD(host_checks);
  BFNEW_ADD_WORK_FIELD(host_synchronizations);
  BFNEW_ADD_WORK_FIELD(controller_copies);
  BFNEW_ADD_WORK_FIELD(kernel_dispatches);
  BFNEW_ADD_WORK_FIELD(expansion_count);
  BFNEW_ADD_WORK_FIELD(atomic_attempts);
  BFNEW_ADD_WORK_FIELD(successful_atomic_updates);
  BFNEW_ADD_WORK_FIELD(queue_claims);
  BFNEW_ADD_WORK_FIELD(duplicate_suppressions);
  BFNEW_ADD_WORK_FIELD(mask_operations);
  BFNEW_ADD_WORK_FIELD(overflow_events);
  BFNEW_ADD_WORK_FIELD(high_contention_destinations);
  BFNEW_ADD_WORK_FIELD(changed_flag_updates);
  BFNEW_ADD_WORK_FIELD(full_edge_rounds);
  BFNEW_ADD_WORK_FIELD(empty_frontier_rounds);
  BFNEW_ADD_WORK_FIELD(small_frontier_rounds);
#undef BFNEW_ADD_WORK_FIELD
}

[[nodiscard]] bool device_work_is_zero(
    const DeviceWorkStatistics& work) noexcept {
  return work.edges_examined == 0U && work.successful_decreases == 0U &&
         work.active_vertices == 0U && work.active_lane_rounds == 0U &&
         work.maximum_queue_size == 0U && work.host_checks == 0U &&
         work.host_synchronizations == 0U && work.controller_copies == 0U &&
         work.kernel_dispatches == 0U && work.expansion_count == 0U &&
         work.atomic_attempts == 0U &&
         work.successful_atomic_updates == 0U && work.queue_claims == 0U &&
         work.duplicate_suppressions == 0U && work.mask_operations == 0U &&
         work.overflow_events == 0U &&
         work.high_contention_destinations == 0U &&
         work.changed_flag_updates == 0U && work.full_edge_rounds == 0U &&
         work.empty_frontier_rounds == 0U &&
         work.small_frontier_rounds == 0U;
}

[[nodiscard]] bool transfer_accounting_is_zero(
    const CompactTransferAccounting& accounting) noexcept {
  return accounting.summary_bytes == 0U && accounting.vertex_bytes == 0U &&
         accounting.distance_label_bytes == 0U &&
         accounting.edge_id_bytes == 0U && accounting.total_bytes == 0U;
}

[[nodiscard]] bool transfer_accounting_is_consistent(
    const CompactTransferAccounting& accounting) noexcept {
  std::uint64_t total = 0U;
  return try_add_u64(total, accounting.summary_bytes, total) &&
         try_add_u64(total, accounting.vertex_bytes, total) &&
         try_add_u64(total, accounting.distance_label_bytes, total) &&
         try_add_u64(total, accounting.edge_id_bytes, total) &&
         total == accounting.total_bytes;
}

[[nodiscard]] bool compact_transport_is_consistent(
    const ExpansionBatchExecution& execution) noexcept {
  if (execution.compact_total_device_to_host_bytes == 0U) {
    return execution.compact_status_bytes == 0U &&
           execution.compact_error_bytes == 0U &&
           execution.compact_controller_poll_count == 0U &&
           execution.compact_controller_poll_bytes == 0U &&
           execution.compact_overall_device_to_host_bytes == 0U;
  }
  std::uint64_t compact_expected = 0U;
  if (!try_add_u64(
          compact_expected,
          execution.compact_transfer.total_bytes,
          compact_expected) ||
      !try_add_u64(
          compact_expected,
          execution.compact_status_bytes,
          compact_expected) ||
      !try_add_u64(
          compact_expected,
          execution.compact_error_bytes,
          compact_expected) ||
      compact_expected != execution.compact_total_device_to_host_bytes ||
      execution.compact_controller_poll_count >
          std::numeric_limits<std::uint64_t>::max() /
              sizeof(DeviceController)) {
    return false;
  }
  const std::uint64_t expected_poll_bytes =
      execution.compact_controller_poll_count * sizeof(DeviceController);
  if (execution.compact_controller_poll_bytes != expected_poll_bytes) {
    return false;
  }
  const ControlMode control =
      static_cast<ControlMode>(execution.result.control_mode);
  if ((control == ControlMode::persistent_cooperative &&
       execution.compact_controller_poll_count != 0U) ||
      ((control == ControlMode::chunked_host_poll ||
        control == ControlMode::per_round_host_poll) &&
       execution.compact_controller_poll_count == 0U)) {
    return false;
  }
  std::uint64_t overall_expected = 0U;
  return try_add_u64(
             compact_expected,
             execution.compact_controller_poll_bytes,
             overall_expected) &&
         overall_expected ==
             execution.compact_overall_device_to_host_bytes;
}

[[nodiscard]] bool valid_stage_evidence(
    const CompactStageTimingEvidence evidence) noexcept {
  return evidence == CompactStageTimingEvidence::unavailable ||
         evidence == CompactStageTimingEvidence::measured;
}

[[nodiscard]] bool compact_execution_is_valid(
    const CompactPathExecutionMetrics& metrics) noexcept {
  if (!valid_stage_evidence(metrics.device_timing) ||
      !valid_stage_evidence(metrics.host_timing)) {
    return false;
  }
  if (metrics.device_timing == CompactStageTimingEvidence::unavailable &&
      (metrics.sssp_device_nanoseconds != 0U ||
       metrics.reconstruction_device_nanoseconds != 0U ||
       metrics.result_transfer_device_nanoseconds != 0U)) {
    return false;
  }
  if (metrics.host_timing == CompactStageTimingEvidence::unavailable) {
    return metrics.sssp_host_nanoseconds == 0U &&
           metrics.reconstruction_host_nanoseconds == 0U &&
           metrics.result_transfer_host_nanoseconds == 0U &&
           metrics.end_to_end_host_nanoseconds == 0U;
  }
  std::uint64_t named_host_stages = 0U;
  return try_add_u64(
             named_host_stages,
             metrics.sssp_host_nanoseconds,
             named_host_stages) &&
         try_add_u64(
             named_host_stages,
             metrics.reconstruction_host_nanoseconds,
             named_host_stages) &&
         try_add_u64(
             named_host_stages,
             metrics.result_transfer_host_nanoseconds,
             named_host_stages) &&
         named_host_stages <= metrics.end_to_end_host_nanoseconds;
}

void add_transfer_accounting(
    CompactTransferAccounting& destination,
    const CompactTransferAccounting& source) {
  checked_add(
      destination.summary_bytes,
      source.summary_bytes,
      "compact summary-transfer bytes overflowed");
  checked_add(
      destination.vertex_bytes,
      source.vertex_bytes,
      "compact vertex-transfer bytes overflowed");
  checked_add(
      destination.distance_label_bytes,
      source.distance_label_bytes,
      "compact distance-label-transfer bytes overflowed");
  checked_add(
      destination.edge_id_bytes,
      source.edge_id_bytes,
      "compact edge-ID-transfer bytes overflowed");
  checked_add(
      destination.total_bytes,
      source.total_bytes,
      "compact result-transfer bytes overflowed");
}

[[nodiscard]] bool clean_convergence(
    const DeviceRunStatus& status) noexcept {
  return status.converged == 1U && status.error_bits == device_error::none &&
         status.stop_reason ==
             static_cast<std::uint32_t>(DeviceStopReason::converged);
}

void set_terminal(
    QueryState& state,
    const ExpansionQueryDisposition disposition,
    const DeviceRunStatus& status) {
  state.terminal = true;
  state.disposition = disposition;
  state.terminal_stop_reason = status.stop_reason;
  state.terminal_error_bits = status.error_bits;
}

enum class RetryDecision : std::uint8_t {
  retry,
  terminal,
};

[[nodiscard]] RetryDecision advance_failed_query(
    const WeightedGraph& graph,
    const TileDirectory& directory,
    const SelectedRegionIndex& selected_regions,
    const std::span<const TileId> full_tiles,
    const BatchedExpansionOptions& options,
    const DeviceRunStatus& status,
    QueryState& state,
    BatchedExpansionMetrics& metrics) {
  if (state.fallback_used ||
      std::equal(
          state.query.selected_tiles.begin(),
          state.query.selected_tiles.end(),
          full_tiles.begin(),
          full_tiles.end())) {
    set_terminal(
        state, ExpansionQueryDisposition::unreachable_in_full_region, status);
    return RetryDecision::terminal;
  }

  const auto install_restart = [&](std::vector<TileId> next_tiles,
                                   const bool fallback) {
    if (state.query.expansion_generation ==
            std::numeric_limits<std::uint32_t>::max() ||
        state.total_expansions == std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    const std::uint64_t discarded_edges =
        selected_regions.selected_edge_count(state.query.selected_tiles);
    std::uint64_t next_repeated_edges =
        metrics.repeated_selected_edge_estimate;
    std::uint64_t next_fallback_count = metrics.full_region_fallbacks;
    checked_add(
        next_repeated_edges,
        discarded_edges,
        "repeated selected-edge aggregate overflow");
    if (fallback) {
      checked_add(
          next_fallback_count, 1U, "full-region fallback aggregate overflow");
    }
    metrics.repeated_selected_edge_estimate = next_repeated_edges;
    ++state.query.expansion_generation;
    ++state.total_expansions;
    state.query.selected_tiles = std::move(next_tiles);
    state.latest_distances.clear();
    state.latest_compact_paths.reset();
    if (fallback) {
      state.fallback_used = true;
      metrics.full_region_fallbacks = next_fallback_count;
    }
    return true;
  };

  if (state.scheduled_expansions < options.maximum_expansions) {
    if (!advance_margin_cursor(state, options.schedule)) {
      set_terminal(
          state,
          ExpansionQueryDisposition::identity_or_count_overflow,
          status);
      return RetryDecision::terminal;
    }
    std::vector<TileId> next_tiles = expanded_tiles(graph, directory, state);
    if (next_tiles != state.query.selected_tiles) {
      if (state.scheduled_expansions ==
              std::numeric_limits<std::uint32_t>::max() ||
          !install_restart(std::move(next_tiles), false)) {
        set_terminal(
            state,
            ExpansionQueryDisposition::identity_or_count_overflow,
            status);
        return RetryDecision::terminal;
      }
      ++state.scheduled_expansions;
      checked_add(
          metrics.scheduled_expansions,
          1U,
          "scheduled-expansion aggregate overflow");
      return RetryDecision::retry;
    }
    if (options.terminal_policy ==
        ExpansionTerminalPolicy::explicit_failure) {
      set_terminal(state, ExpansionQueryDisposition::region_stalled, status);
      return RetryDecision::terminal;
    }
  } else if (options.terminal_policy ==
             ExpansionTerminalPolicy::explicit_failure) {
    set_terminal(state, ExpansionQueryDisposition::expansion_limit, status);
    return RetryDecision::terminal;
  }

  if (options.terminal_policy ==
      ExpansionTerminalPolicy::full_region_fallback) {
    if (!install_restart(
            std::vector<TileId>(full_tiles.begin(), full_tiles.end()), true)) {
      set_terminal(
          state,
          ExpansionQueryDisposition::identity_or_count_overflow,
          status);
      return RetryDecision::terminal;
    }
    return RetryDecision::retry;
  }

  // This is reachable only after a stalled scheduled expansion under a
  // malformed future enum extension; keep failure explicit rather than retry.
  set_terminal(state, ExpansionQueryDisposition::region_stalled, status);
  return RetryDecision::terminal;
}

[[nodiscard]] std::uint64_t expected_distance_image_size(
    const WeightedGraph& graph,
    const BatchPlanEntry& batch) {
  const std::uint64_t vertices = graph.vertex_count();
  if (vertices != 0U &&
      batch.lane_width > std::numeric_limits<std::uint64_t>::max() / vertices) {
    throw std::overflow_error{"batched expansion distance image size overflow"};
  }
  return vertices * batch.lane_width;
}

[[nodiscard]] bool evidence_precedes(
    const ExpansionScheduleEvidence& left,
    const ExpansionScheduleEvidence& right,
    const bool all_work_measured) noexcept {
  if (left.reached_queries != right.reached_queries) {
    return left.reached_queries > right.reached_queries;
  }
  if (left.terminal_failures != right.terminal_failures) {
    return left.terminal_failures < right.terminal_failures;
  }
  if (all_work_measured &&
      left.logical_lane_edge_work != right.logical_lane_edge_work) {
    return left.logical_lane_edge_work < right.logical_lane_edge_work;
  }
  if (all_work_measured && left.shared_edge_work != right.shared_edge_work) {
    return left.shared_edge_work < right.shared_edge_work;
  }
  return std::tie(
             left.attempted_selected_edge_estimate,
             left.total_expansions,
             left.batches_executed) <
         std::tie(
             right.attempted_selected_edge_estimate,
             right.total_expansions,
             right.batches_executed);
}

}  // namespace

BatchedExpansionOptionsError validate_batched_expansion_options(
    const BatchedExpansionOptions& options) noexcept {
  if (validate_gpu_run_options(options.run_options) != GpuRunOptionsError::none) {
    return BatchedExpansionOptionsError::invalid_run_options;
  }
  if (validate_batch_planner_policy(options.planner_policy) !=
      BatchPlannerPolicyError::none) {
    return BatchedExpansionOptionsError::invalid_planner_policy;
  }
  if (options.execution_configuration_fingerprint == 0U) {
    return BatchedExpansionOptionsError::
        missing_execution_configuration_fingerprint;
  }
  switch (options.schedule.kind) {
    case ExpansionScheduleKind::unspecified:
      return BatchedExpansionOptionsError::unspecified_schedule;
    case ExpansionScheduleKind::one_geometric_ring:
    case ExpansionScheduleKind::doubling_xy_margins:
      if (options.schedule.fixed_ring_size != 0U) {
        return BatchedExpansionOptionsError::unexpected_fixed_ring_size;
      }
      if (options.schedule.hybrid_small_expansion_count != 0U) {
        return BatchedExpansionOptionsError::unexpected_hybrid_small_expansion_count;
      }
      break;
    case ExpansionScheduleKind::fixed_larger_ring:
      if (options.schedule.fixed_ring_size < 2U) {
        return BatchedExpansionOptionsError::invalid_fixed_ring_size;
      }
      if (options.schedule.hybrid_small_expansion_count != 0U) {
        return BatchedExpansionOptionsError::unexpected_hybrid_small_expansion_count;
      }
      break;
    case ExpansionScheduleKind::hybrid_small_then_doubling:
      if (options.schedule.fixed_ring_size != 0U) {
        return BatchedExpansionOptionsError::unexpected_fixed_ring_size;
      }
      if (options.schedule.hybrid_small_expansion_count == 0U) {
        return BatchedExpansionOptionsError::invalid_hybrid_small_expansion_count;
      }
      break;
    default:
      return BatchedExpansionOptionsError::invalid_schedule;
  }
  switch (options.terminal_policy) {
    case ExpansionTerminalPolicy::full_region_fallback:
    case ExpansionTerminalPolicy::explicit_failure:
      break;
    default:
      return BatchedExpansionOptionsError::invalid_terminal_policy;
  }
  if (options.enable_compact_paths > 1U) {
    return BatchedExpansionOptionsError::invalid_compact_paths_flag;
  }
  return BatchedExpansionOptionsError::none;
}

BatchedExpansionRunResult run_batched_expansion(
    const WeightedGraph& graph,
    const TileDirectory& directory,
    const TileRunLayout64& tile_runs,
    const std::span<const RouteQuery> queries,
    const BatchedExpansionOptions& options,
    const ExpansionBatchRunner& runner) {
  if (validate_batched_expansion_options(options) !=
      BatchedExpansionOptionsError::none) {
    throw std::invalid_argument{"batched expansion options are invalid"};
  }
  if (!runner) {
    throw std::invalid_argument{"batched expansion requires a batch runner"};
  }
  if (!validate_weighted_graph(graph).ok() ||
      !validate_tile_directory(graph, directory).ok() ||
      !validate_tile_run_layout(graph, tile_runs).ok()) {
    throw std::invalid_argument{
        "batched expansion requires a deeply valid spatial graph"};
  }

  const auto total_begin = std::chrono::steady_clock::now();
  BatchedExpansionRunResult output;
  output.metrics.input_queries = size64(queries.size(), "query count overflow");
  if (queries.empty()) {
    if (options.enable_compact_paths != 0U) {
      output.metrics.compact_host_timing =
          CompactStageTimingEvidence::measured;
    }
    output.metrics.total_nanoseconds = elapsed_nanoseconds(
        total_begin, std::chrono::steady_clock::now());
    output.metrics.expansion_count_histogram.assign(1U, 0U);
    return output;
  }

  const SelectedRegionIndex selected_regions{graph, tile_runs};
  const std::vector<TileId> full_tiles = full_region_tiles(graph);
  std::vector<QueryState> states;
  states.reserve(queries.size());
  for (const RouteQuery& query : queries) {
    if (!validate_route_query(graph, query).ok()) {
      throw std::invalid_argument{
          "batched expansion requires deeply valid route queries"};
    }
    QueryState state;
    state.query = query;
    state.anchor = located_bounds(graph, query.selected_tiles);
    states.push_back(std::move(state));
  }
  std::sort(
      states.begin(), states.end(), [](const QueryState& left, const QueryState& right) {
        return left.query.query_id < right.query.query_id;
      });
  if (std::adjacent_find(
          states.begin(),
          states.end(),
          [](const QueryState& left, const QueryState& right) {
            return left.query.query_id == right.query.query_id;
          }) != states.end()) {
    throw std::invalid_argument{"batched expansion query IDs must be unique"};
  }
  output.metrics.schedule_comparison_fingerprint =
      comparison_fingerprint(graph, tile_runs, states, options);

  std::vector<std::size_t> active;
  active.reserve(states.size());
  for (std::size_t index = 0U; index < states.size(); ++index) {
    active.push_back(index);
  }

  std::uint64_t execution_ordinal = 0U;
  std::uint32_t planning_pass = 0U;
  while (!active.empty()) {
    std::vector<RouteQuery> active_queries;
    active_queries.reserve(active.size());
    for (const std::size_t index : active) {
      if (index >= states.size() || states[index].terminal) {
        throw std::logic_error{"batched expansion active-state ledger is invalid"};
      }
      active_queries.push_back(states[index].query);
    }

    const auto planning_begin = std::chrono::steady_clock::now();
    const std::vector<BatchQueryFeatures> features = make_batch_query_features(
        graph, selected_regions, active_queries);
    const BatchPlan plan = make_overlapping_batch_plan(
        selected_regions, features, options.planner_policy);
    if (!validate_batch_plan(selected_regions, features, plan).ok()) {
      throw std::logic_error{"retry batch plan failed deep validation"};
    }
    const std::uint64_t planning_elapsed = elapsed_nanoseconds(
        planning_begin, std::chrono::steady_clock::now());
    if (planning_pass == 0U) {
      output.metrics.initial_planning_nanoseconds = planning_elapsed;
    } else {
      checked_add(
          output.metrics.replanning_nanoseconds,
          planning_elapsed,
          "replanning time overflow");
    }
    checked_add(output.metrics.planning_passes, 1U, "planning pass count overflow");

    std::vector<std::size_t> failed;
    for (std::size_t batch_index = 0U; batch_index < plan.batches.size();
         ++batch_index) {
      const BatchPlanEntry& batch = plan.batches[batch_index];
      const ExpansionBatchContext context{
          execution_ordinal,
          planning_pass,
          size32(batch_index, "batch index overflow"),
          planning_pass != 0U};

      const auto execution_begin = std::chrono::steady_clock::now();
      ExpansionBatchExecution execution =
          runner(active_queries, features, batch, context);
      checked_add(
          output.metrics.execution_nanoseconds,
          elapsed_nanoseconds(
              execution_begin, std::chrono::steady_clock::now()),
          "execution time overflow");
      checked_add(execution_ordinal, 1U, "execution ordinal overflow");

      if (execution.result.engine_kind !=
              static_cast<std::uint32_t>(options.run_options.engine) ||
          execution.result.control_mode !=
              static_cast<std::uint32_t>(options.run_options.control_mode) ||
          validate_device_run_status(execution.result.status) !=
              DeviceRunStatusError::none ||
          execution.result.status.valid_lane_mask != batch.valid_lane_mask) {
        throw std::logic_error{
            "batch runner returned status for a different or invalid request"};
      }
      if (execution.work_evidence != ExpansionWorkEvidence::unavailable &&
          execution.work_evidence != ExpansionWorkEvidence::measured) {
        throw std::logic_error{"batch runner returned an invalid work evidence state"};
      }
      if (execution.work_evidence == ExpansionWorkEvidence::unavailable &&
          (execution.shared_edge_work != 0U ||
           execution.logical_lane_edge_work != 0U ||
           !device_work_is_zero(execution.result.work))) {
        throw std::logic_error{
            "unavailable batch work evidence must not carry numeric work"};
      }
      if (execution.work_evidence == ExpansionWorkEvidence::measured &&
          execution.logical_lane_edge_work < execution.shared_edge_work) {
        throw std::logic_error{
            "measured logical lane-edge work cannot be below shared edge work"};
      }
      const std::uint64_t expected_size =
          expected_distance_image_size(graph, batch);
      if (!execution.final_distances.empty() &&
          size64(execution.final_distances.size(), "distance image size overflow") !=
              expected_size) {
        throw std::logic_error{"batch runner returned a malformed distance image"};
      }
      if (!transfer_accounting_is_consistent(execution.compact_transfer) ||
          !compact_transport_is_consistent(execution) ||
          !compact_execution_is_valid(execution.compact_execution)) {
        throw std::logic_error{
            "batch runner returned malformed compact transfer/timing evidence"};
      }
      if (options.enable_compact_paths == 0U) {
        if (!execution.compact_paths.empty() ||
            !transfer_accounting_is_zero(execution.compact_transfer) ||
            execution.compact_status_bytes != 0U ||
            execution.compact_error_bytes != 0U ||
            execution.compact_total_device_to_host_bytes != 0U ||
            execution.compact_controller_poll_count != 0U ||
            execution.compact_controller_poll_bytes != 0U ||
            execution.compact_overall_device_to_host_bytes != 0U ||
            execution.compact_execution.device_timing !=
                CompactStageTimingEvidence::unavailable ||
            execution.compact_execution.host_timing !=
                CompactStageTimingEvidence::unavailable) {
          throw std::logic_error{
              "compact payloads require explicit Phase 18 production mode"};
        }
      } else {
        if (execution.compact_execution.host_timing !=
            CompactStageTimingEvidence::measured) {
          throw std::logic_error{
              "compact production requires measured host stage timing"};
        }
        const bool clean = clean_convergence(execution.result.status);
        const std::size_t expected_payload_count =
            clean ? static_cast<std::size_t>(std::popcount(batch.valid_lane_mask))
                  : 0U;
        std::uint64_t expected_summary_count = 0U;
        for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
          if ((batch.valid_lane_mask & (LaneMask{1U} << lane)) == 0U) {
            continue;
          }
          checked_add(
              expected_summary_count,
              size64(
                  find_query(active_queries, batch.query_ids_by_lane[lane])
                      .targets.size(),
                  "compact target count overflow"),
              "compact target count overflow");
        }
        if (execution.compact_paths.size() != expected_payload_count ||
            !std::is_sorted(
                execution.compact_paths.begin(),
                execution.compact_paths.end(),
                [](const CompactPathPayload& left,
                   const CompactPathPayload& right) {
                  return left.query_id < right.query_id;
                }) ||
            std::adjacent_find(
                execution.compact_paths.begin(),
                execution.compact_paths.end(),
                [](const CompactPathPayload& left,
                   const CompactPathPayload& right) {
                  return left.query_id == right.query_id;
                }) != execution.compact_paths.end()) {
          throw std::logic_error{
              "compact payloads do not match reached-lane cardinality/order"};
        }
        for (const CompactPathPayload& payload : execution.compact_paths) {
          bool matching_lane = false;
          for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
            const LaneMask bit = LaneMask{1U} << lane;
            if ((batch.valid_lane_mask & bit) != 0U &&
                batch.query_ids_by_lane[lane] == payload.query_id &&
                batch.expansion_generations_by_lane[lane] ==
                    payload.expansion_generation) {
              matching_lane = true;
              break;
            }
          }
          if (!matching_lane ||
              !validate_compact_path_payload(
                   graph,
                   find_query(active_queries, payload.query_id),
                   payload)
                   .ok()) {
            throw std::logic_error{
                "compact payload crossed lane identity or failed validation"};
          }
        }
        const CompactTransferAccounting expected_transfer =
            measure_compact_transfer(execution.compact_paths);
        if (clean && execution.compact_transfer != expected_transfer) {
          throw std::logic_error{
              "compact transfer bytes disagree with the returned path arenas"};
        }
        if (!clean && execution.compact_total_device_to_host_bytes == 0U &&
            !transfer_accounting_is_zero(execution.compact_transfer)) {
          throw std::logic_error{
              "unmeasured nonclean compact execution carried transfer bytes"};
        }
        if (!clean && execution.compact_total_device_to_host_bytes != 0U) {
          std::uint64_t expected_summary_bytes = expected_summary_count;
          if (expected_summary_bytes >
              std::numeric_limits<std::uint64_t>::max() /
                  sizeof(CompactTargetSummary)) {
            throw std::overflow_error{"compact target-summary bytes overflow"};
          }
          expected_summary_bytes *= sizeof(CompactTargetSummary);
          if (execution.compact_transfer.summary_bytes !=
                  expected_summary_bytes ||
              execution.compact_transfer.vertex_bytes != 0U ||
              execution.compact_transfer.distance_label_bytes != 0U ||
              execution.compact_transfer.edge_id_bytes != 0U) {
            throw std::logic_error{
                "nonclean compact transfer is not the fixed-summary shape"};
          }
        }
        if (execution.compact_total_device_to_host_bytes != 0U) {
          if (execution.compact_status_bytes != sizeof(DeviceRunStatus) ||
              execution.compact_error_bytes !=
                  sizeof(std::uint32_t) *
                      (execution.compact_transfer.vertex_bytes == 0U ? 1U
                                                                     : 2U)) {
            throw std::logic_error{
                "compact control transfer envelope is not the Phase 18 shape"};
          }
        }
      }

      checked_add(output.metrics.batches_executed, 1U, "batch count overflow");
      if (planning_pass == 0U) {
        checked_add(
            output.metrics.initial_batches_executed,
            1U,
            "initial batch count overflow");
      } else {
        checked_add(
            output.metrics.retry_batches_executed,
            1U,
            "retry batch count overflow");
        checked_add(
            output.metrics.retry_valid_lane_observations,
            static_cast<std::uint64_t>(
                std::popcount(batch.valid_lane_mask)),
            "retry valid-lane count overflow");
        checked_add(
            output.metrics.retry_lane_capacity,
            batch.lane_width,
            "retry lane capacity overflow");
      }
      checked_add(
          output.metrics.attempted_selected_vertex_count,
          batch.selected_lane_vertex_count,
          "attempted selected-vertex count overflow");
      checked_add(
          output.metrics.attempted_selected_edge_estimate,
          batch.selected_lane_edge_estimate,
          "attempted selected-edge estimate overflow");
      if (execution.work_evidence == ExpansionWorkEvidence::measured) {
        checked_add(
            output.metrics.work_measured_batches,
            1U,
            "measured-work batch count overflow");
        checked_add(
            output.metrics.shared_edge_work,
            execution.shared_edge_work,
            "shared edge-work count overflow");
        checked_add(
            output.metrics.logical_lane_edge_work,
            execution.logical_lane_edge_work,
            "logical lane-edge work count overflow");
        if (planning_pass != 0U) {
          checked_add(
              output.metrics.retry_work_measured_batches,
              1U,
              "retry measured-work batch count overflow");
          checked_add(
              output.metrics.retry_shared_edge_work,
              execution.shared_edge_work,
              "retry shared edge-work count overflow");
          checked_add(
              output.metrics.retry_logical_lane_edge_work,
              execution.logical_lane_edge_work,
              "retry logical lane-edge work count overflow");
        }
      } else {
        output.metrics.work_evidence = ExpansionWorkEvidence::unavailable;
      }
      add_device_work(output.metrics.device_work, execution.result.work);
      add_transfer_accounting(
          output.metrics.compact_transfer, execution.compact_transfer);
      checked_add(
          output.metrics.compact_status_bytes,
          execution.compact_status_bytes,
          "compact status-transfer bytes overflowed");
      checked_add(
          output.metrics.compact_error_bytes,
          execution.compact_error_bytes,
          "compact error-transfer bytes overflowed");
      checked_add(
          output.metrics.compact_total_device_to_host_bytes,
          execution.compact_total_device_to_host_bytes,
          "compact total device-to-host bytes overflowed");
      checked_add(
          output.metrics.compact_controller_poll_count,
          execution.compact_controller_poll_count,
          "compact controller-poll count overflowed");
      checked_add(
          output.metrics.compact_controller_poll_bytes,
          execution.compact_controller_poll_bytes,
          "compact controller-poll bytes overflowed");
      checked_add(
          output.metrics.compact_overall_device_to_host_bytes,
          execution.compact_overall_device_to_host_bytes,
          "compact overall device-to-host bytes overflowed");
      if (execution.compact_execution.device_timing ==
          CompactStageTimingEvidence::measured) {
        checked_add(
            output.metrics.compact_device_timing_measured_batches,
            1U,
            "compact device-timing batch count overflow");
        checked_add(
            output.metrics.sssp_device_nanoseconds,
            execution.compact_execution.sssp_device_nanoseconds,
            "SSSP device time overflow");
        checked_add(
            output.metrics.reconstruction_device_nanoseconds,
            execution.compact_execution.reconstruction_device_nanoseconds,
            "reconstruction device time overflow");
        checked_add(
            output.metrics.result_transfer_device_nanoseconds,
            execution.compact_execution.result_transfer_device_nanoseconds,
            "result-transfer device time overflow");
      }
      if (execution.compact_execution.host_timing ==
          CompactStageTimingEvidence::measured) {
        checked_add(
            output.metrics.compact_host_timing_measured_batches,
            1U,
            "compact host-timing batch count overflow");
        checked_add(
            output.metrics.sssp_host_nanoseconds,
            execution.compact_execution.sssp_host_nanoseconds,
            "SSSP host time overflow");
        checked_add(
            output.metrics.reconstruction_host_nanoseconds,
            execution.compact_execution.reconstruction_host_nanoseconds,
            "reconstruction host time overflow");
        checked_add(
            output.metrics.result_transfer_host_nanoseconds,
            execution.compact_execution.result_transfer_host_nanoseconds,
            "result-transfer host time overflow");
        checked_add(
            output.metrics.compact_end_to_end_host_nanoseconds,
            execution.compact_execution.end_to_end_host_nanoseconds,
            "compact end-to-end host time overflow");
      }

      const DeviceRunStatus& status = execution.result.status;
      const bool converged = clean_convergence(status);
      if (converged &&
          ((status.reached_target_mask | status.bounding_box_miss_mask) !=
               batch.valid_lane_mask ||
           (status.reached_target_mask & status.bounding_box_miss_mask) != 0U)) {
        throw std::logic_error{
            "clean batch status does not partition valid lanes into reached/miss"};
      }
      if (!converged &&
          (status.reached_target_mask != 0U ||
           status.bounding_box_miss_mask != 0U)) {
        throw std::logic_error{
            "non-clean batch status must not classify reached or missed lanes"};
      }

      ExpansionBatchTrace trace;
      trace.context = context;
      trace.lane_width = batch.lane_width;
      trace.valid_lane_mask = batch.valid_lane_mask;
      trace.reached_lane_mask = status.reached_target_mask;
      trace.miss_lane_mask = status.bounding_box_miss_mask;
      trace.query_ids_by_lane = batch.query_ids_by_lane;
      trace.expansion_generations_by_lane =
          batch.expansion_generations_by_lane;
      trace.union_vertex_count = batch.union_vertex_count;
      trace.union_edge_estimate = batch.union_edge_estimate;
      trace.selected_lane_vertex_count = batch.selected_lane_vertex_count;
      trace.selected_lane_edge_estimate = batch.selected_lane_edge_estimate;
      trace.shared_edge_work = execution.shared_edge_work;
      trace.logical_lane_edge_work = execution.logical_lane_edge_work;
      trace.work_evidence = execution.work_evidence;
      trace.stop_reason = status.stop_reason;
      trace.error_bits = status.error_bits;
      output.trace.push_back(std::move(trace));

      if (converged && status.bounding_box_miss_mask != 0U) {
        checked_add(
            output.metrics.failed_lane_observations,
            static_cast<std::uint64_t>(
                std::popcount(status.bounding_box_miss_mask)),
            "failed lane count overflow");
        checked_add(
            output.metrics.failed_origin_valid_lane_observations,
            static_cast<std::uint64_t>(
                std::popcount(batch.valid_lane_mask)),
            "failed origin capacity overflow");
        if (execution.work_evidence == ExpansionWorkEvidence::measured) {
          checked_add(
              output.metrics.failed_batch_work_measured_batches,
              1U,
              "failed-batch measured-work count overflow");
          checked_add(
              output.metrics.failed_batch_shared_edge_work,
              execution.shared_edge_work,
              "failed-batch shared edge-work count overflow");
          checked_add(
              output.metrics.failed_batch_logical_lane_edge_work,
              execution.logical_lane_edge_work,
              "failed-batch logical lane-edge work count overflow");
        }
      }

      for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
        const LaneMask lane_bit = LaneMask{1U} << lane;
        if ((batch.valid_lane_mask & lane_bit) == 0U) {
          continue;
        }
        QueryState& state = find_state(states, batch.query_ids_by_lane[lane]);
        if (state.terminal ||
            state.query.expansion_generation !=
                batch.expansion_generations_by_lane[lane]) {
          throw std::logic_error{"batch runner crossed query generation state"};
        }
        if (state.attempts == std::numeric_limits<std::uint32_t>::max()) {
          state.latest_distances.clear();
          state.latest_compact_paths.reset();
          set_terminal(
              state,
              ExpansionQueryDisposition::identity_or_count_overflow,
              status);
          continue;
        }
        ++state.attempts;
        if (!execution.final_distances.empty()) {
          copy_lane_distances(
              execution.final_distances,
              graph.vertex_count(),
              batch.lane_width,
              lane,
              state.latest_distances);
        } else {
          state.latest_distances.clear();
        }
        state.latest_compact_paths.reset();

        if (!converged) {
          state.latest_distances.clear();
          set_terminal(
              state, ExpansionQueryDisposition::engine_failure, status);
          continue;
        }
        if ((status.reached_target_mask & lane_bit) != 0U) {
          if (options.enable_compact_paths != 0U) {
            const auto payload = std::lower_bound(
                execution.compact_paths.begin(),
                execution.compact_paths.end(),
                state.query.query_id,
                [](const CompactPathPayload& value, const QueryId query_id) {
                  return value.query_id < query_id;
                });
            if (payload == execution.compact_paths.end() ||
                payload->query_id != state.query.query_id) {
              throw std::logic_error{
                  "reached lane lost its compact payload before retirement"};
            }
            state.latest_compact_paths = std::move(*payload);
            state.latest_distances.clear();
            if (!compact_path_payload_complete(*state.latest_compact_paths)) {
              const bool explicit_reconstruction_failure =
                  std::ranges::any_of(
                      state.latest_compact_paths->targets,
                      [](const CompactTargetPath& target) {
                        return target.summary.reconstruction ==
                                   CompactPathStatus::no_tight_path ||
                               target.summary.reconstruction ==
                                   CompactPathStatus::path_length_overflow ||
                               target.summary.reconstruction ==
                                   CompactPathStatus::query_terminal_failure;
                      });
              if (!explicit_reconstruction_failure) {
                state.latest_compact_paths.reset();
              }
              set_terminal(
                  state, ExpansionQueryDisposition::engine_failure, status);
              continue;
            }
          }
          set_terminal(state, ExpansionQueryDisposition::reached, status);
          if (planning_pass == 0U) {
            checked_add(
                output.metrics.initial_reached_queries,
                1U,
                "initial reached count overflow");
          }
          continue;
        }
        if ((status.bounding_box_miss_mask & lane_bit) == 0U) {
          throw std::logic_error{"a converged valid lane has no terminal class"};
        }
        if (options.enable_compact_paths != 0U) {
          const auto payload = std::lower_bound(
              execution.compact_paths.begin(),
              execution.compact_paths.end(),
              state.query.query_id,
              [](const CompactPathPayload& value, const QueryId query_id) {
                return value.query_id < query_id;
              });
          if (payload == execution.compact_paths.end() ||
              payload->query_id != state.query.query_id) {
            throw std::logic_error{
                "miss lane lost its compact payload before retirement"};
          }
          state.latest_compact_paths = std::move(*payload);
          const bool has_unreachable = std::ranges::any_of(
              state.latest_compact_paths->targets,
              [](const CompactTargetPath& target) {
                return target.summary.reconstruction ==
                       CompactPathStatus::unreachable;
              });
          const bool all_classified = std::ranges::all_of(
              state.latest_compact_paths->targets,
              [](const CompactTargetPath& target) {
                return target.summary.reconstruction ==
                           CompactPathStatus::complete ||
                       target.summary.reconstruction ==
                           CompactPathStatus::unreachable;
              });
          if (!has_unreachable || !all_classified) {
            // A clean lane-level miss is a per-target classification, not an
            // invitation to discard malformed reconstruction evidence.  A
            // miss with only complete targets has no explicit failure payload
            // that can survive terminal validation; other nonclassified
            // payloads retain their explicit reconstruction-failure status.
            if (all_classified) {
              state.latest_compact_paths.reset();
            }
            state.latest_distances.clear();
            set_terminal(
                state, ExpansionQueryDisposition::engine_failure, status);
            continue;
          }
        }
        failed.push_back(static_cast<std::size_t>(&state - states.data()));
      }
    }

    std::sort(
        failed.begin(), failed.end(), [&states](const std::size_t left,
                                               const std::size_t right) {
          return states[left].query.query_id < states[right].query.query_id;
        });
    if (std::adjacent_find(failed.begin(), failed.end()) != failed.end()) {
      throw std::logic_error{"a failed query appeared more than once in a pass"};
    }

    active.clear();
    const auto geometric_expansion_begin = std::chrono::steady_clock::now();
    for (const std::size_t index : failed) {
      QueryState& state = states[index];
      DeviceRunStatus miss_status{};
      miss_status.valid_lane_mask = 1U;
      miss_status.converged_lane_mask = 1U;
      miss_status.converged = 1U;
      miss_status.bounding_box_miss_mask = 1U;
      miss_status.stop_reason =
          static_cast<std::uint32_t>(DeviceStopReason::converged);
      const RetryDecision decision = advance_failed_query(
              graph,
              directory,
              selected_regions,
              full_tiles,
              options,
              miss_status,
              state,
              output.metrics);
      if (decision == RetryDecision::retry) {
        active.push_back(index);
      } else if (state.disposition !=
                 ExpansionQueryDisposition::unreachable_in_full_region) {
        state.latest_compact_paths.reset();
      } else if (options.enable_compact_paths != 0U) {
        if (!state.latest_compact_paths.has_value()) {
          throw std::logic_error{
              "terminal full-region miss lost its compact target payload"};
        }
        const bool has_unreachable = std::ranges::any_of(
            state.latest_compact_paths->targets,
            [](const CompactTargetPath& target) {
              return target.summary.reconstruction ==
                     CompactPathStatus::unreachable;
            });
        const bool all_classified = std::ranges::all_of(
            state.latest_compact_paths->targets,
            [](const CompactTargetPath& target) {
              return target.summary.reconstruction ==
                         CompactPathStatus::complete ||
                     target.summary.reconstruction ==
                         CompactPathStatus::unreachable;
            });
        if (!has_unreachable) {
          throw std::logic_error{
              "full-region miss payload contains no unreachable target"};
        }
        if (!all_classified) {
          state.disposition = ExpansionQueryDisposition::engine_failure;
        }
      }
    }
    if (!failed.empty()) {
      checked_add(
          output.metrics.geometric_expansion_host_nanoseconds,
          elapsed_nanoseconds(
              geometric_expansion_begin, std::chrono::steady_clock::now()),
          "geometric expansion host time overflow");
    }

    if (!active.empty()) {
      if (planning_pass == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error{"planning-pass aggregate overflow"};
      } else {
        ++planning_pass;
      }
    }
  }

  output.queries.reserve(states.size());
  std::uint32_t largest_expansion_count = 0U;
  for (QueryState& state : states) {
    if (!state.terminal) {
      throw std::logic_error{"batched expansion omitted a terminal query outcome"};
    }
    ExpansionQueryOutcome outcome;
    outcome.final_query = std::move(state.query);
    outcome.disposition = state.disposition;
    outcome.attempts = state.attempts;
    outcome.scheduled_expansions = state.scheduled_expansions;
    outcome.total_expansions = state.total_expansions;
    outcome.used_full_region_fallback = state.fallback_used;
    outcome.selected_vertex_count =
        selected_regions.selected_vertex_count(outcome.final_query.selected_tiles);
    outcome.selected_edge_count =
        selected_regions.selected_edge_count(outcome.final_query.selected_tiles);
    outcome.terminal_stop_reason = state.terminal_stop_reason;
    outcome.terminal_error_bits = state.terminal_error_bits;
    outcome.final_distances = std::move(state.latest_distances);
    outcome.compact_paths = std::move(state.latest_compact_paths);
    largest_expansion_count =
        std::max(largest_expansion_count, outcome.total_expansions);

    switch (outcome.disposition) {
      case ExpansionQueryDisposition::reached:
        checked_add(output.metrics.reached_queries, 1U, "reached count overflow");
        break;
      case ExpansionQueryDisposition::unreachable_in_full_region:
        checked_add(
            output.metrics.unreachable_full_region_queries,
            1U,
            "unreachable count overflow");
        break;
      case ExpansionQueryDisposition::expansion_limit:
        checked_add(
            output.metrics.expansion_limit_queries,
            1U,
            "expansion-limit count overflow");
        break;
      case ExpansionQueryDisposition::region_stalled:
        checked_add(
            output.metrics.stalled_region_queries,
            1U,
            "stalled-region count overflow");
        break;
      case ExpansionQueryDisposition::identity_or_count_overflow:
        checked_add(
            output.metrics.identity_or_count_overflow_queries,
            1U,
            "identity/count-overflow query count overflow");
        break;
      case ExpansionQueryDisposition::engine_failure:
        checked_add(
            output.metrics.engine_failure_queries,
            1U,
            "engine-failure count overflow");
        break;
    }
    checked_add(
        output.metrics.final_selected_tile_count,
        size64(outcome.final_query.selected_tiles.size(), "final tile count overflow"),
        "aggregate final tile count overflow");
    checked_add(
        output.metrics.final_selected_vertex_count,
        outcome.selected_vertex_count,
        "aggregate final vertex count overflow");
    checked_add(
        output.metrics.final_selected_edge_count,
        outcome.selected_edge_count,
        "aggregate final edge count overflow");
    output.queries.push_back(std::move(outcome));
  }

  output.metrics.expansion_count_histogram.assign(
      static_cast<std::size_t>(largest_expansion_count) + 1U, 0U);
  for (const ExpansionQueryOutcome& outcome : output.queries) {
    checked_add(
        output.metrics.expansion_count_histogram[outcome.total_expansions],
        1U,
        "expansion histogram overflow");
  }
  output.metrics.device_work.expansion_count =
      output.metrics.scheduled_expansions;
  checked_add(
      output.metrics.device_work.expansion_count,
      output.metrics.full_region_fallbacks,
      "aggregate expansion count overflow");
  if (output.metrics.batches_executed != 0U &&
      output.metrics.compact_device_timing_measured_batches ==
          output.metrics.batches_executed) {
    output.metrics.compact_device_timing =
        CompactStageTimingEvidence::measured;
  }
  if (output.metrics.batches_executed != 0U &&
      output.metrics.compact_host_timing_measured_batches ==
          output.metrics.batches_executed) {
    output.metrics.compact_host_timing =
        CompactStageTimingEvidence::measured;
  }
  output.metrics.total_nanoseconds = elapsed_nanoseconds(
      total_begin, std::chrono::steady_clock::now());
  output.metrics.all_query_throughput_milliqueries_per_second =
      throughput_milliqueries_per_second(
          output.metrics.input_queries, output.metrics.total_nanoseconds);
  return output;
}

BatchedExpansionRunResult run_host_batched_expansion(
    const WeightedGraph& graph,
    const TileDirectory& directory,
    const TileRunLayout64& tile_runs,
    const DeviceGraphLayout32& device_graph,
    const std::span<const RouteQuery> queries,
    const BatchedExpansionOptions& options,
    const HostBatchedExpansionOptions& host_options) {
  if (validate_batched_expansion_options(options) !=
      BatchedExpansionOptionsError::none) {
    throw std::invalid_argument{
        "portable expansion requires valid, runner-bound options"};
  }
  if (!validate_device_graph_layout32(graph, tile_runs, device_graph).ok()) {
    throw std::invalid_argument{
        "portable expansion requires the matching checked device graph image"};
  }

  BatchedExpansionOptions bound_options = options;
  bound_options.execution_configuration_fingerprint =
      host_execution_configuration_fingerprint(
          options.execution_configuration_fingerprint, host_options);

  BatchDeviceDescription description;
  const ExpansionBatchRunner runner =
      [&](const std::span<const RouteQuery> current_queries,
          const std::span<const BatchQueryFeatures> features,
          const BatchPlanEntry& batch,
          const ExpansionBatchContext&) {
        const auto callback_begin = std::chrono::steady_clock::now();
        prepare_batch_device_description(
            graph,
            tile_runs,
            current_queries,
            features,
            batch,
            host_options.run_representation,
            description);

        ExpansionBatchExecution execution;
        execution.work_evidence = ExpansionWorkEvidence::measured;
        const auto sssp_begin = std::chrono::steady_clock::now();
        switch (bound_options.run_options.engine) {
          case EngineKind::jacobi_pull: {
            HostBatchedJacobiRunResult result = run_host_batched_jacobi_pull(
                device_graph,
                current_queries,
                batch,
                description,
                bound_options.run_options);
            execution.result = result.result;
            execution.shared_edge_work = result.batch_work.csc_edge_loads;
            execution.logical_lane_edge_work =
                result.batch_work.lane_edge_relaxations;
            execution.final_distances = std::move(result.final_distances);
            break;
          }
          case EngineKind::dense_chaotic_push: {
            HostBatchedDenseRunResult result =
                run_host_batched_dense_chaotic_push(
                    device_graph,
                    current_queries,
                    batch,
                    description,
                    bound_options.run_options,
                    host_options.dense_schedule);
            execution.result = result.result;
            execution.shared_edge_work = result.batch_work.csr_edge_loads;
            execution.logical_lane_edge_work =
                result.batch_work.lane_edge_relaxations;
            execution.final_distances = std::move(result.distances);
            break;
          }
          case EngineKind::frontier_push: {
            HostBatchedFrontierRunResult result =
                run_host_batched_frontier_push(
                    device_graph,
                    current_queries,
                    batch,
                    description,
                    bound_options.run_options,
                    host_options.frontier_queue_capacity);
            execution.result = result.result;
            execution.shared_edge_work = result.batch_work.csr_edge_loads;
            execution.logical_lane_edge_work =
                result.batch_work.lane_edge_relaxations;
            execution.final_distances = std::move(result.distances);
            break;
          }
        }
        const auto sssp_end = std::chrono::steady_clock::now();
        if (bound_options.enable_compact_paths != 0U) {
          const auto reconstruction_begin = std::chrono::steady_clock::now();
          std::vector<float> lane_distances;
          const bool clean = clean_convergence(execution.result.status);
          for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
            if (!clean ||
                (batch.valid_lane_mask & (LaneMask{1U} << lane)) == 0U) {
              continue;
            }
            copy_lane_distances(
                execution.final_distances,
                graph.vertex_count(),
                batch.lane_width,
                lane,
                lane_distances);
            execution.compact_paths.push_back(reconstruct_compact_path_payload(
                graph,
                find_query(current_queries, batch.query_ids_by_lane[lane]),
                lane_distances));
          }
          std::sort(
              execution.compact_paths.begin(),
              execution.compact_paths.end(),
              [](const CompactPathPayload& left,
                 const CompactPathPayload& right) {
                return left.query_id < right.query_id;
              });
          const auto reconstruction_end = std::chrono::steady_clock::now();
          const auto transfer_begin = std::chrono::steady_clock::now();
          execution.compact_transfer =
              measure_compact_transfer(execution.compact_paths);
          const auto transfer_end = std::chrono::steady_clock::now();
          execution.final_distances.clear();
          execution.final_distances.shrink_to_fit();
          execution.compact_execution.host_timing =
              CompactStageTimingEvidence::measured;
          execution.compact_execution.sssp_host_nanoseconds =
              elapsed_nanoseconds(sssp_begin, sssp_end);
          execution.compact_execution.reconstruction_host_nanoseconds =
              elapsed_nanoseconds(reconstruction_begin, reconstruction_end);
          execution.compact_execution.result_transfer_host_nanoseconds =
              elapsed_nanoseconds(transfer_begin, transfer_end);
          execution.compact_execution.end_to_end_host_nanoseconds =
              elapsed_nanoseconds(
                  callback_begin, std::chrono::steady_clock::now());
        }
        return execution;
      };

  return run_batched_expansion(
      graph, directory, tile_runs, queries, bound_options, runner);
}

ExpansionScheduleEvidence make_expansion_schedule_evidence(
    const ExpansionSchedulePolicy& schedule,
    const BatchedExpansionRunResult& result) noexcept {
  const BatchedExpansionMetrics& metrics = result.metrics;
  const std::uint64_t terminal_failures =
      metrics.input_queries >= metrics.reached_queries
          ? metrics.input_queries - metrics.reached_queries
          : std::numeric_limits<std::uint64_t>::max();
  std::uint64_t total_expansions = 0U;
  const bool expansion_total_valid = try_add_u64(
      metrics.scheduled_expansions,
      metrics.full_region_fallbacks,
      total_expansions);
  if (!expansion_total_valid) {
    total_expansions = std::numeric_limits<std::uint64_t>::max();
  }
  return ExpansionScheduleEvidence{
      schedule,
      metrics.input_queries,
      metrics.reached_queries,
      terminal_failures,
      metrics.batches_executed,
      total_expansions,
      metrics.attempted_selected_edge_estimate,
      metrics.work_evidence,
      metrics.shared_edge_work,
      metrics.logical_lane_edge_work,
      metrics.schedule_comparison_fingerprint,
      metrics.engine_failure_queries == 0U &&
          metrics.identity_or_count_overflow_queries == 0U &&
          expansion_total_valid};
}

std::optional<ExpansionSchedulePolicy> select_expansion_schedule_from_evidence(
    const std::span<const ExpansionScheduleEvidence> evidence) noexcept {
  constexpr std::size_t schedule_count = 4U;
  if (evidence.size() != schedule_count) {
    return std::nullopt;
  }
  const std::uint64_t query_count = evidence.front().input_queries;
  const std::uint64_t comparison_identity =
      evidence.front().comparison_fingerprint;
  if (query_count == 0U || comparison_identity == 0U) {
    return std::nullopt;
  }
  std::uint32_t seen = 0U;
  bool all_work_measured = true;
  for (const ExpansionScheduleEvidence& record : evidence) {
    if (!valid_schedule_policy(record.schedule) ||
        record.input_queries != query_count ||
        record.comparison_fingerprint != comparison_identity ||
        record.reached_queries > record.input_queries ||
        record.batches_executed == 0U || !record.campaign_valid ||
        record.terminal_failures !=
            record.input_queries - record.reached_queries ||
        (record.work_evidence != ExpansionWorkEvidence::unavailable &&
         record.work_evidence != ExpansionWorkEvidence::measured) ||
        (record.work_evidence == ExpansionWorkEvidence::unavailable &&
         (record.shared_edge_work != 0U ||
          record.logical_lane_edge_work != 0U)) ||
        (record.work_evidence == ExpansionWorkEvidence::measured &&
         record.logical_lane_edge_work < record.shared_edge_work)) {
      return std::nullopt;
    }
    const auto raw_kind = static_cast<std::uint32_t>(record.schedule.kind);
    if (raw_kind == 0U || raw_kind > schedule_count) {
      return std::nullopt;
    }
    const std::uint32_t bit = std::uint32_t{1U} << (raw_kind - 1U);
    if ((seen & bit) != 0U) {
      return std::nullopt;
    }
    seen |= bit;
    all_work_measured =
        all_work_measured &&
        record.work_evidence == ExpansionWorkEvidence::measured;
  }
  if (seen != ((std::uint32_t{1U} << schedule_count) - 1U)) {
    return std::nullopt;
  }

  const ExpansionScheduleEvidence* best = &evidence.front();
  bool unique_best = true;
  for (const ExpansionScheduleEvidence& candidate : evidence.subspan(1U)) {
    if (evidence_precedes(candidate, *best, all_work_measured)) {
      best = &candidate;
      unique_best = true;
    } else if (!evidence_precedes(*best, candidate, all_work_measured)) {
      unique_best = false;
    }
  }
  if (!unique_best) {
    return std::nullopt;
  }
  return best->schedule;
}

}  // namespace bfnew
