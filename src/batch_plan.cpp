#include "bfnew/batch_plan.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

constexpr const char* batch_plan_schema = "bfnew.batch-plan.v1";

[[nodiscard]] std::uint64_t checked_add(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* const description) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error{description};
  }
  return left + right;
}

[[nodiscard]] std::uint64_t checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* const description) {
  if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error{description};
  }
  return left * right;
}

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

[[nodiscard]] bool supported_lane_width(const std::uint32_t width) noexcept {
  return width == 1U || width == 8U || width == 16U || width == 32U;
}

[[nodiscard]] bool canonical_tiles(
    const std::span<const TileId> tiles,
    const std::uint32_t tile_count) noexcept {
  TileId previous{};
  bool has_previous = false;
  for (const TileId tile : tiles) {
    if (tile.value() >= tile_count ||
        (has_previous && !(previous < tile))) {
      return false;
    }
    previous = tile;
    has_previous = true;
  }
  return true;
}

[[nodiscard]] bool canonical_tiles_without_bound(
    const std::span<const TileId> tiles) noexcept {
  TileId previous{};
  bool has_previous = false;
  for (const TileId tile : tiles) {
    if (has_previous && !(previous < tile)) {
      return false;
    }
    previous = tile;
    has_previous = true;
  }
  return true;
}

[[nodiscard]] bool tile_subset(
    const std::span<const TileId> subset,
    const std::span<const TileId> superset) noexcept {
  return std::includes(
      superset.begin(), superset.end(), subset.begin(), subset.end());
}

[[nodiscard]] std::uint64_t intersection_size(
    const std::span<const TileId> left,
    const std::span<const TileId> right) noexcept {
  std::size_t left_position = 0U;
  std::size_t right_position = 0U;
  std::uint64_t result = 0U;
  while (left_position < left.size() && right_position < right.size()) {
    if (left[left_position] < right[right_position]) {
      ++left_position;
    } else if (right[right_position] < left[left_position]) {
      ++right_position;
    } else {
      ++result;
      ++left_position;
      ++right_position;
    }
  }
  return result;
}

[[nodiscard]] std::vector<TileId> tile_union(
    const std::span<const TileId> left,
    const std::span<const TileId> right) {
  if (right.size() > std::numeric_limits<std::size_t>::max() - left.size()) {
    throw std::overflow_error{"tile union size overflow"};
  }
  std::vector<TileId> result;
  result.reserve(left.size() + right.size());
  std::set_union(
      left.begin(),
      left.end(),
      right.begin(),
      right.end(),
      std::back_inserter(result));
  return result;
}

// Exact fraction comparison without a widened integer extension. Continued
// fraction terms avoid overflowing cross-products even for uint64_t counts.
[[nodiscard]] int compare_fractions(
    std::uint64_t left_numerator,
    std::uint64_t left_denominator,
    std::uint64_t right_numerator,
    std::uint64_t right_denominator) noexcept {
  bool reversed = false;
  for (;;) {
    const std::uint64_t left_quotient = left_numerator / left_denominator;
    const std::uint64_t right_quotient = right_numerator / right_denominator;
    if (left_quotient != right_quotient) {
      const int direct = left_quotient < right_quotient ? -1 : 1;
      return reversed ? -direct : direct;
    }

    const std::uint64_t left_remainder = left_numerator % left_denominator;
    const std::uint64_t right_remainder = right_numerator % right_denominator;
    if (left_remainder == 0U || right_remainder == 0U) {
      if (left_remainder == right_remainder) {
        return 0;
      }
      const int direct = left_remainder == 0U ? -1 : 1;
      return reversed ? -direct : direct;
    }

    left_numerator = left_denominator;
    left_denominator = left_remainder;
    right_numerator = right_denominator;
    right_denominator = right_remainder;
    reversed = !reversed;
  }
}

[[nodiscard]] bool fraction_at_least(
    const BatchFraction value,
    const std::uint32_t numerator,
    const std::uint32_t denominator) noexcept {
  return compare_fractions(
             value.numerator,
             value.denominator,
             numerator,
             denominator) >= 0;
}

[[nodiscard]] bool fraction_at_most(
    const BatchFraction value,
    const std::uint32_t numerator,
    const std::uint32_t denominator) noexcept {
  return compare_fractions(
             value.numerator,
             value.denominator,
             numerator,
             denominator) <= 0;
}

[[nodiscard]] std::uint32_t absolute_difference(
    const std::uint32_t left,
    const std::uint32_t right) noexcept {
  return left < right ? right - left : left - right;
}

[[nodiscard]] LaneMask low_lane_mask(const std::uint32_t lane_count) noexcept {
  if (lane_count == 32U) {
    return std::numeric_limits<LaneMask>::max();
  }
  return (LaneMask{1U} << lane_count) - LaneMask{1U};
}

[[nodiscard]] std::vector<TileId> terminal_tiles(
    const WeightedGraph& graph,
    const std::span<const VertexId> terminals) {
  std::vector<TileId> result;
  result.reserve(terminals.size());
  for (const VertexId terminal : terminals) {
    result.push_back(graph.owner_tiles()[terminal.value()]);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

[[nodiscard]] bool valid_features(
    const SelectedRegionIndex& selected_regions,
    const BatchQueryFeatures& features) noexcept {
  try {
    const std::uint32_t tile_count = selected_regions.tile_count();
    if (features.source_count == 0U || features.target_count == 0U ||
        features.source_tiles.empty() || features.target_tiles.empty() ||
        features.selected_tiles.empty() ||
        features.source_tiles.size() >
            static_cast<std::size_t>(features.source_count) ||
        features.target_tiles.size() >
            static_cast<std::size_t>(features.target_count) ||
        !canonical_tiles(features.source_tiles, tile_count) ||
        !canonical_tiles(features.target_tiles, tile_count) ||
        !canonical_tiles(features.selected_tiles, tile_count) ||
        !tile_subset(features.source_tiles, features.selected_tiles) ||
        !tile_subset(features.target_tiles, features.selected_tiles)) {
      return false;
    }
    const std::uint64_t vertices =
        selected_regions.selected_vertex_count(features.selected_tiles);
    return vertices != 0U && vertices == features.selected_vertex_count &&
           selected_regions.selected_edge_count(features.selected_tiles) ==
               features.selected_edge_estimate;
  } catch (...) {
    return false;
  }
}

void require_valid_features(
    const SelectedRegionIndex& selected_regions,
    const std::span<const BatchQueryFeatures> queries) {
  if (queries.empty()) {
    throw std::invalid_argument{"batch planning requires at least one query"};
  }
  static_cast<void>(size32(queries.size(), "batch query count overflow"));

  std::vector<std::uint32_t> query_ids;
  query_ids.reserve(queries.size());
  for (const BatchQueryFeatures& query : queries) {
    if (!valid_features(selected_regions, query)) {
      throw std::invalid_argument{"batch planner query features are invalid"};
    }
    query_ids.push_back(query.query_id.value());
  }
  std::sort(query_ids.begin(), query_ids.end());
  if (std::adjacent_find(query_ids.begin(), query_ids.end()) != query_ids.end()) {
    throw std::invalid_argument{"batch planner query IDs must be unique"};
  }
}

[[nodiscard]] bool anchor_precedes(
    const BatchQueryFeatures& left,
    const BatchQueryFeatures& right) noexcept {
  if (left.selected_edge_estimate != right.selected_edge_estimate) {
    return left.selected_edge_estimate > right.selected_edge_estimate;
  }
  if (left.selected_vertex_count != right.selected_vertex_count) {
    return left.selected_vertex_count > right.selected_vertex_count;
  }
  if (left.selected_tiles.size() != right.selected_tiles.size()) {
    return left.selected_tiles.size() > right.selected_tiles.size();
  }
  if (left.expansion_generation != right.expansion_generation) {
    return left.expansion_generation < right.expansion_generation;
  }
  return left.query_id < right.query_id;
}

struct Candidate final {
  std::uint32_t query_index{};
  BatchFraction jaccard{};
  std::uint64_t terminal_overlap{};
  BatchFraction projected_inflation{};
  std::uint64_t projected_union_edges{};
  std::uint64_t projected_union_vertices{};
  std::uint64_t new_tiles{};
  std::uint32_t generation_difference{};
  std::uint32_t source_count_difference{};
  std::uint32_t target_count_difference{};
  QueryId query_id{};
  std::vector<TileId> projected_union_tiles;
};

[[nodiscard]] bool candidate_precedes(
    const Candidate& left,
    const Candidate& right) noexcept {
  const int jaccard_order = compare_fractions(
      left.jaccard.numerator,
      left.jaccard.denominator,
      right.jaccard.numerator,
      right.jaccard.denominator);
  if (jaccard_order != 0) {
    return jaccard_order > 0;
  }
  if (left.terminal_overlap != right.terminal_overlap) {
    return left.terminal_overlap > right.terminal_overlap;
  }
  const int inflation_order = compare_fractions(
      left.projected_inflation.numerator,
      left.projected_inflation.denominator,
      right.projected_inflation.numerator,
      right.projected_inflation.denominator);
  if (inflation_order != 0) {
    return inflation_order < 0;
  }
  return std::tie(
             left.projected_union_edges,
             left.projected_union_vertices,
             left.new_tiles,
             left.generation_difference,
             left.source_count_difference,
             left.target_count_difference,
             left.query_id) <
         std::tie(
             right.projected_union_edges,
             right.projected_union_vertices,
             right.new_tiles,
             right.generation_difference,
             right.source_count_difference,
             right.target_count_difference,
             right.query_id);
}

[[nodiscard]] BatchFraction inflation_fraction(
    const std::uint64_t lane_count,
    const std::span<const TileId> union_tiles,
    const std::uint64_t selected_lane_tile_count) {
  if (lane_count == 0U || selected_lane_tile_count == 0U) {
    throw std::invalid_argument{"batch inflation requires nonempty valid lanes"};
  }
  return BatchFraction{
      checked_multiply(
          lane_count,
          size64(union_tiles.size(), "batch union tile count overflow"),
          "batch union tile-position count overflow"),
      selected_lane_tile_count};
}

[[nodiscard]] Candidate make_candidate(
    const SelectedRegionIndex& selected_regions,
    const std::span<const BatchQueryFeatures> queries,
    const std::uint32_t query_index,
    const BatchQueryFeatures& anchor,
    const std::span<const TileId> current_union_tiles,
    const std::span<const TileId> current_source_tiles,
    const std::span<const TileId> current_target_tiles,
    const std::uint64_t current_lane_count,
    const std::uint64_t current_selected_lane_tile_count) {
  const BatchQueryFeatures& query = queries[query_index];
  Candidate result;
  result.query_index = query_index;
  result.jaccard = tile_set_jaccard(current_union_tiles, query.selected_tiles);
  result.terminal_overlap = checked_add(
      intersection_size(current_source_tiles, query.source_tiles),
      intersection_size(current_target_tiles, query.target_tiles),
      "batch terminal-overlap count overflow");
  result.projected_union_tiles =
      tile_union(current_union_tiles, query.selected_tiles);
  const std::uint64_t projected_selected_lane_tile_count = checked_add(
      current_selected_lane_tile_count,
      size64(query.selected_tiles.size(), "selected tile count overflow"),
      "selected lane-tile count overflow");
  result.projected_inflation = inflation_fraction(
      checked_add(current_lane_count, 1U, "batch lane count overflow"),
      result.projected_union_tiles,
      projected_selected_lane_tile_count);
  result.projected_union_edges =
      selected_regions.selected_edge_count(result.projected_union_tiles);
  result.projected_union_vertices =
      selected_regions.selected_vertex_count(result.projected_union_tiles);
  result.new_tiles = size64(
      result.projected_union_tiles.size() - current_union_tiles.size(),
      "new tile count overflow");
  result.generation_difference = absolute_difference(
      query.expansion_generation, anchor.expansion_generation);
  result.source_count_difference =
      absolute_difference(query.source_count, anchor.source_count);
  result.target_count_difference =
      absolute_difference(query.target_count, anchor.target_count);
  result.query_id = query.query_id;
  return result;
}

[[nodiscard]] BatchPlan build_plan(
    const SelectedRegionIndex& selected_regions,
    const std::span<const BatchQueryFeatures> queries,
    const BatchPlannerPolicy& policy) {
  BatchPlan plan;
  plan.policy = policy;
  plan.input_query_count = size32(queries.size(), "batch query count overflow");

  std::vector<std::uint32_t> anchor_order;
  anchor_order.reserve(queries.size());
  for (std::size_t index = 0U; index < queries.size(); ++index) {
    anchor_order.push_back(size32(index, "batch query index overflow"));
  }
  std::sort(
      anchor_order.begin(),
      anchor_order.end(),
      [queries](const std::uint32_t left, const std::uint32_t right) {
        return anchor_precedes(queries[left], queries[right]);
      });

  std::vector<bool> assigned(queries.size(), false);
  std::size_t assigned_count = 0U;
  while (assigned_count < queries.size()) {
    const auto anchor_position = std::find_if(
        anchor_order.begin(), anchor_order.end(), [&assigned](const std::uint32_t index) {
          return !assigned[index];
        });
    if (anchor_position == anchor_order.end()) {
      throw std::logic_error{"batch planner lost an unassigned query"};
    }

    const std::uint32_t anchor_index = *anchor_position;
    const BatchQueryFeatures& anchor = queries[anchor_index];
    std::vector<std::uint32_t> lane_indices{anchor_index};
    assigned[anchor_index] = true;
    ++assigned_count;

    std::vector<TileId> union_tiles = anchor.selected_tiles;
    std::vector<TileId> union_source_tiles = anchor.source_tiles;
    std::vector<TileId> union_target_tiles = anchor.target_tiles;
    std::uint64_t selected_lane_tile_count =
        size64(anchor.selected_tiles.size(), "selected tile count overflow");

    while (lane_indices.size() < static_cast<std::size_t>(policy.lane_width)) {
      Candidate best;
      bool has_best = false;
      for (std::size_t candidate_position = 0U;
           candidate_position < queries.size();
           ++candidate_position) {
        if (assigned[candidate_position]) {
          continue;
        }
        const std::uint32_t candidate_index =
            size32(candidate_position, "batch query index overflow");
        Candidate candidate = make_candidate(
            selected_regions,
            queries,
            candidate_index,
            anchor,
            union_tiles,
            union_source_tiles,
            union_target_tiles,
            size64(lane_indices.size(), "batch lane count overflow"),
            selected_lane_tile_count);
        if (!fraction_at_least(
                candidate.jaccard,
                policy.minimum_jaccard_numerator,
                policy.minimum_jaccard_denominator) ||
            !fraction_at_most(
                candidate.projected_inflation,
                policy.maximum_union_inflation_numerator,
                policy.maximum_union_inflation_denominator)) {
          continue;
        }
        if (!has_best || candidate_precedes(candidate, best)) {
          best = std::move(candidate);
          has_best = true;
        }
      }
      if (!has_best) {
        break;
      }

      const BatchQueryFeatures& selected = queries[best.query_index];
      lane_indices.push_back(best.query_index);
      assigned[best.query_index] = true;
      ++assigned_count;
      selected_lane_tile_count = checked_add(
          selected_lane_tile_count,
          size64(selected.selected_tiles.size(), "selected tile count overflow"),
          "selected lane-tile count overflow");
      union_tiles = std::move(best.projected_union_tiles);
      union_source_tiles = tile_union(union_source_tiles, selected.source_tiles);
      union_target_tiles = tile_union(union_target_tiles, selected.target_tiles);
    }

    BatchPlanEntry entry;
    entry.lane_width = policy.lane_width;
    entry.valid_lane_mask = low_lane_mask(size32(
        lane_indices.size(), "valid batch lane count overflow"));
    entry.query_indices_by_lane.assign(
        policy.lane_width, invalid_batch_query_index);
    entry.query_ids_by_lane.assign(policy.lane_width, invalid_batch_query_id);
    entry.expansion_generations_by_lane.assign(policy.lane_width, 0U);
    entry.union_tiles = std::move(union_tiles);
    entry.union_tile_lane_masks.assign(entry.union_tiles.size(), LaneMask{0U});

    for (std::size_t lane = 0U; lane < lane_indices.size(); ++lane) {
      const std::uint32_t query_index = lane_indices[lane];
      const BatchQueryFeatures& query = queries[query_index];
      entry.query_indices_by_lane[lane] = query_index;
      entry.query_ids_by_lane[lane] = query.query_id;
      entry.expansion_generations_by_lane[lane] = query.expansion_generation;
      entry.selected_lane_vertex_count = checked_add(
          entry.selected_lane_vertex_count,
          query.selected_vertex_count,
          "batch selected lane-vertex count overflow");
      entry.selected_lane_edge_estimate = checked_add(
          entry.selected_lane_edge_estimate,
          query.selected_edge_estimate,
          "batch selected lane-edge count overflow");

      const LaneMask lane_bit = LaneMask{1U} << size32(lane, "lane index overflow");
      for (std::size_t tile = 0U; tile < entry.union_tiles.size(); ++tile) {
        if (std::binary_search(
                query.selected_tiles.begin(),
                query.selected_tiles.end(),
                entry.union_tiles[tile])) {
          entry.union_tile_lane_masks[tile] |= lane_bit;
        }
      }
    }
    entry.union_vertex_count =
        selected_regions.selected_vertex_count(entry.union_tiles);
    entry.union_edge_estimate =
        selected_regions.selected_edge_count(entry.union_tiles);
    plan.batches.push_back(std::move(entry));
  }
  return plan;
}

[[nodiscard]] BatchPlanValidationResult validation_error(
    const BatchPlanValidationErrorCode code,
    const std::size_t batch = BatchPlanValidationResult::no_position,
    const std::size_t lane = BatchPlanValidationResult::no_position) noexcept {
  return BatchPlanValidationResult{code, batch, lane};
}

[[nodiscard]] bool valid_low_lane_mask(
    const LaneMask mask,
    const std::uint32_t width) noexcept {
  const std::uint32_t count = static_cast<std::uint32_t>(std::popcount(mask));
  return count != 0U && count <= width && mask == low_lane_mask(count);
}

}  // namespace

BatchPlannerPolicyError validate_batch_planner_policy(
    const BatchPlannerPolicy& policy) noexcept {
  if (!supported_lane_width(policy.lane_width)) {
    return BatchPlannerPolicyError::unsupported_lane_width;
  }
  if (policy.minimum_jaccard_denominator == 0U ||
      policy.minimum_jaccard_numerator > policy.minimum_jaccard_denominator) {
    return BatchPlannerPolicyError::invalid_jaccard_fraction;
  }
  if (policy.maximum_union_inflation_denominator == 0U ||
      policy.maximum_union_inflation_numerator <
          policy.maximum_union_inflation_denominator) {
    return BatchPlannerPolicyError::invalid_union_inflation_fraction;
  }
  return BatchPlannerPolicyError::none;
}

std::vector<BatchQueryFeatures> make_batch_query_features(
    const WeightedGraph& graph,
    const SelectedRegionIndex& selected_regions,
    const std::span<const RouteQuery> queries) {
  if (!graph.has_spatial_ordering() || !selected_regions.matches_graph(graph)) {
    throw std::invalid_argument{
        "batch query features require a matching spatial graph and index"};
  }

  std::vector<BatchQueryFeatures> result;
  result.reserve(queries.size());
  std::vector<std::uint32_t> query_ids;
  query_ids.reserve(queries.size());
  for (const RouteQuery& query : queries) {
    if (!validate_route_query(graph, query).ok()) {
      throw std::invalid_argument{
          "batch query features require deeply valid route queries"};
    }
    BatchQueryFeatures features;
    features.query_id = query.query_id;
    features.expansion_generation = query.expansion_generation;
    features.source_count = size32(query.sources.size(), "source count overflow");
    features.target_count = size32(query.targets.size(), "target count overflow");
    features.source_tiles = terminal_tiles(graph, query.sources);
    features.target_tiles = terminal_tiles(graph, query.targets);
    features.selected_tiles = query.selected_tiles;
    features.selected_vertex_count =
        selected_regions.selected_vertex_count(features.selected_tiles);
    features.selected_edge_estimate =
        selected_regions.selected_edge_count(features.selected_tiles);
    if (!valid_features(selected_regions, features)) {
      throw std::logic_error{"constructed batch query features are invalid"};
    }
    query_ids.push_back(features.query_id.value());
    result.push_back(std::move(features));
  }
  std::sort(query_ids.begin(), query_ids.end());
  if (std::adjacent_find(query_ids.begin(), query_ids.end()) != query_ids.end()) {
    throw std::invalid_argument{"batch query IDs must be unique"};
  }
  std::sort(
      result.begin(),
      result.end(),
      [](const BatchQueryFeatures& left, const BatchQueryFeatures& right) {
        return left.query_id < right.query_id;
      });
  return result;
}

BatchFraction tile_set_jaccard(
    const std::span<const TileId> left,
    const std::span<const TileId> right) {
  if (!canonical_tiles_without_bound(left) ||
      !canonical_tiles_without_bound(right)) {
    throw std::invalid_argument{
        "Jaccard tile sets must be strictly increasing and unique"};
  }
  const std::uint64_t intersection = intersection_size(left, right);
  const std::uint64_t total = checked_add(
      size64(left.size(), "Jaccard tile count overflow"),
      size64(right.size(), "Jaccard tile count overflow"),
      "Jaccard tile count overflow");
  const std::uint64_t set_union_size = total - intersection;
  if (set_union_size == 0U) {
    return BatchFraction{1U, 1U};
  }
  return BatchFraction{intersection, set_union_size};
}

BatchFraction batch_union_tile_inflation(
    const std::span<const BatchQueryFeatures> queries,
    const std::span<const std::uint32_t> query_indices,
    const std::span<const TileId> union_tiles) {
  if (query_indices.empty() || !canonical_tiles_without_bound(union_tiles)) {
    throw std::invalid_argument{"batch inflation requires canonical nonempty lanes"};
  }
  std::vector<std::uint32_t> unique_indices(
      query_indices.begin(), query_indices.end());
  std::sort(unique_indices.begin(), unique_indices.end());
  if (std::adjacent_find(unique_indices.begin(), unique_indices.end()) !=
      unique_indices.end()) {
    throw std::invalid_argument{"batch inflation lanes must be unique"};
  }

  std::uint64_t selected_lane_tile_count = 0U;
  for (const std::uint32_t index : query_indices) {
    if (index >= queries.size() || queries[index].selected_tiles.empty() ||
        !canonical_tiles_without_bound(queries[index].selected_tiles) ||
        !tile_subset(queries[index].selected_tiles, union_tiles)) {
      throw std::invalid_argument{"batch inflation query or union is invalid"};
    }
    selected_lane_tile_count = checked_add(
        selected_lane_tile_count,
        size64(queries[index].selected_tiles.size(), "selected tile count overflow"),
        "selected lane-tile count overflow");
  }
  return inflation_fraction(
      size64(query_indices.size(), "batch lane count overflow"),
      union_tiles,
      selected_lane_tile_count);
}

BatchPlan make_overlapping_batch_plan(
    const SelectedRegionIndex& selected_regions,
    const std::span<const BatchQueryFeatures> queries,
    const BatchPlannerPolicy& policy) {
  if (validate_batch_planner_policy(policy) != BatchPlannerPolicyError::none) {
    throw std::invalid_argument{"batch planner policy is invalid"};
  }
  require_valid_features(selected_regions, queries);
  return build_plan(selected_regions, queries, policy);
}

BatchPlanFamily make_standard_batch_plan_family(
    const SelectedRegionIndex& selected_regions,
    const std::span<const BatchQueryFeatures> queries,
    BatchPlannerPolicy policy) {
  if (validate_batch_planner_policy(policy) != BatchPlannerPolicyError::none) {
    throw std::invalid_argument{"batch planner policy is invalid"};
  }
  require_valid_features(selected_regions, queries);

  BatchPlanFamily result;
  constexpr std::uint32_t widths[3U]{32U, 16U, 8U};
  for (std::size_t index = 0U; index < result.plans.size(); ++index) {
    policy.lane_width = widths[index];
    result.plans[index] = build_plan(selected_regions, queries, policy);
  }
  return result;
}

BatchPlanValidationResult validate_batch_plan(
    const SelectedRegionIndex& selected_regions,
    const std::span<const BatchQueryFeatures> queries,
    const BatchPlan& plan) noexcept {
  try {
    if (validate_batch_planner_policy(plan.policy) !=
        BatchPlannerPolicyError::none) {
      return validation_error(BatchPlanValidationErrorCode::invalid_policy);
    }
    if (queries.size() > std::numeric_limits<std::uint32_t>::max()) {
      return validation_error(BatchPlanValidationErrorCode::query_count_overflow);
    }
    if (queries.empty()) {
      return validation_error(BatchPlanValidationErrorCode::empty_query_set);
    }

    std::vector<std::uint32_t> query_ids;
    query_ids.reserve(queries.size());
    for (const BatchQueryFeatures& query : queries) {
      if (!valid_features(selected_regions, query)) {
        return validation_error(BatchPlanValidationErrorCode::invalid_query_feature);
      }
      query_ids.push_back(query.query_id.value());
    }
    std::sort(query_ids.begin(), query_ids.end());
    if (std::adjacent_find(query_ids.begin(), query_ids.end()) !=
        query_ids.end()) {
      return validation_error(BatchPlanValidationErrorCode::duplicate_query_id);
    }
    if (plan.input_query_count != queries.size()) {
      return validation_error(BatchPlanValidationErrorCode::invalid_batch_shape);
    }
    if (plan.batches.empty()) {
      return validation_error(BatchPlanValidationErrorCode::empty_plan);
    }

    std::vector<bool> assigned(queries.size(), false);
    for (std::size_t batch_index = 0U;
         batch_index < plan.batches.size();
         ++batch_index) {
      const BatchPlanEntry& batch = plan.batches[batch_index];
      if (batch.lane_width != plan.policy.lane_width ||
          !supported_lane_width(batch.lane_width)) {
        return validation_error(
            BatchPlanValidationErrorCode::invalid_batch_width, batch_index);
      }
      if (batch.query_indices_by_lane.size() != batch.lane_width ||
          batch.query_ids_by_lane.size() != batch.lane_width ||
          batch.expansion_generations_by_lane.size() != batch.lane_width ||
          batch.union_tiles.empty() ||
          batch.union_tile_lane_masks.size() != batch.union_tiles.size()) {
        return validation_error(
            BatchPlanValidationErrorCode::invalid_batch_shape, batch_index);
      }
      if (!valid_low_lane_mask(batch.valid_lane_mask, batch.lane_width)) {
        return validation_error(
            BatchPlanValidationErrorCode::invalid_valid_lane_mask, batch_index);
      }
      const std::uint32_t valid_lane_count = static_cast<std::uint32_t>(
          std::popcount(batch.valid_lane_mask));

      std::vector<TileId> expected_union;
      std::uint64_t expected_selected_vertices = 0U;
      std::uint64_t expected_selected_edges = 0U;
      for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
        const bool valid = lane < valid_lane_count;
        if (!valid) {
          if (batch.query_indices_by_lane[lane] != invalid_batch_query_index ||
              batch.query_ids_by_lane[lane] != invalid_batch_query_id ||
              batch.expansion_generations_by_lane[lane] != 0U) {
            return validation_error(
                BatchPlanValidationErrorCode::invalid_padding,
                batch_index,
                lane);
          }
          continue;
        }

        const std::uint32_t query_index = batch.query_indices_by_lane[lane];
        if (query_index >= queries.size()) {
          return validation_error(
              BatchPlanValidationErrorCode::query_index_out_of_range,
              batch_index,
              lane);
        }
        const BatchQueryFeatures& query = queries[query_index];
        if (batch.query_ids_by_lane[lane] != query.query_id ||
            batch.expansion_generations_by_lane[lane] !=
                query.expansion_generation) {
          return validation_error(
              BatchPlanValidationErrorCode::query_identity_mismatch,
              batch_index,
              lane);
        }
        if (assigned[query_index]) {
          return validation_error(
              BatchPlanValidationErrorCode::duplicate_assignment,
              batch_index,
              lane);
        }
        assigned[query_index] = true;
        expected_union = tile_union(expected_union, query.selected_tiles);
        expected_selected_vertices = checked_add(
            expected_selected_vertices,
            query.selected_vertex_count,
            "batch selected vertex count overflow");
        expected_selected_edges = checked_add(
            expected_selected_edges,
            query.selected_edge_estimate,
            "batch selected edge count overflow");
      }

      if (batch.union_tiles != expected_union) {
        return validation_error(
            BatchPlanValidationErrorCode::union_tile_mismatch, batch_index);
      }
      for (std::size_t tile = 0U; tile < batch.union_tiles.size(); ++tile) {
        LaneMask expected_mask = 0U;
        for (std::size_t lane = 0U; lane < valid_lane_count; ++lane) {
          const BatchQueryFeatures& query =
              queries[batch.query_indices_by_lane[lane]];
          if (std::binary_search(
                  query.selected_tiles.begin(),
                  query.selected_tiles.end(),
                  batch.union_tiles[tile])) {
            expected_mask |= LaneMask{1U} << size32(lane, "lane index overflow");
          }
        }
        if (batch.union_tile_lane_masks[tile] != expected_mask) {
          return validation_error(
              BatchPlanValidationErrorCode::tile_lane_mask_mismatch,
              batch_index,
              tile);
        }
      }
      if (batch.union_vertex_count !=
              selected_regions.selected_vertex_count(batch.union_tiles) ||
          batch.union_edge_estimate !=
              selected_regions.selected_edge_count(batch.union_tiles) ||
          batch.selected_lane_vertex_count != expected_selected_vertices ||
          batch.selected_lane_edge_estimate != expected_selected_edges) {
        return validation_error(
            BatchPlanValidationErrorCode::estimate_mismatch, batch_index);
      }
    }
    if (std::find(assigned.begin(), assigned.end(), false) != assigned.end()) {
      return validation_error(BatchPlanValidationErrorCode::omitted_query);
    }

    // Structural validity alone is insufficient: a differently ordered but
    // arithmetically consistent plan could evade the frozen greedy contract.
    // Rebuild from canonical query identities and compare the complete image.
    if (build_plan(selected_regions, queries, plan.policy) != plan) {
      return validation_error(BatchPlanValidationErrorCode::invalid_batch_shape);
    }
    return {};
  } catch (...) {
    return validation_error(BatchPlanValidationErrorCode::invalid_query_feature);
  }
}

std::string serialize_batch_plan_tsv(const BatchPlan& plan) {
  if (validate_batch_planner_policy(plan.policy) !=
          BatchPlannerPolicyError::none ||
      plan.input_query_count == 0U || plan.batches.empty()) {
    throw std::invalid_argument{"cannot serialize an invalid batch plan"};
  }

  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << "schema\t" << batch_plan_schema << '\n';
  output << "record\tbatch\tposition\tlane_width\tvalid_lane_mask\tquery_index"
            "\tquery_id\texpansion_generation\ttile_id\ttile_lane_mask"
            "\tunion_vertex_count\tunion_edge_estimate"
            "\tselected_lane_vertex_count\tselected_lane_edge_estimate"
            "\tinput_query_count\tminimum_jaccard\tmaximum_union_inflation\n";
  output << "plan\t-\t-\t" << plan.policy.lane_width
         << "\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t"
         << plan.input_query_count << '\t'
         << plan.policy.minimum_jaccard_numerator << '/'
         << plan.policy.minimum_jaccard_denominator << '\t'
         << plan.policy.maximum_union_inflation_numerator << '/'
         << plan.policy.maximum_union_inflation_denominator << '\n';

  for (std::size_t batch_index = 0U;
       batch_index < plan.batches.size();
       ++batch_index) {
    const BatchPlanEntry& batch = plan.batches[batch_index];
    if (batch.lane_width != plan.policy.lane_width ||
        batch.query_indices_by_lane.size() != batch.lane_width ||
        batch.query_ids_by_lane.size() != batch.lane_width ||
        batch.expansion_generations_by_lane.size() != batch.lane_width ||
        !valid_low_lane_mask(batch.valid_lane_mask, batch.lane_width) ||
        batch.union_tiles.size() != batch.union_tile_lane_masks.size()) {
      throw std::invalid_argument{"cannot serialize a malformed batch entry"};
    }
    output << "batch\t" << batch_index << "\t-\t" << batch.lane_width
           << '\t' << batch.valid_lane_mask
           << "\t-\t-\t-\t-\t-\t" << batch.union_vertex_count << '\t'
           << batch.union_edge_estimate << '\t'
           << batch.selected_lane_vertex_count << '\t'
           << batch.selected_lane_edge_estimate
           << "\t-\t-\t-\n";
    for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
      output << "lane\t" << batch_index << '\t' << lane << '\t'
             << batch.lane_width << '\t' << batch.valid_lane_mask << '\t'
             << batch.query_indices_by_lane[lane] << '\t'
             << batch.query_ids_by_lane[lane].value() << '\t'
             << batch.expansion_generations_by_lane[lane]
             << "\t-\t-\t-\t-\t-\t-\t-\t-\t-\n";
    }
    for (std::size_t tile = 0U; tile < batch.union_tiles.size(); ++tile) {
      output << "tile\t" << batch_index << '\t' << tile << '\t'
             << batch.lane_width << '\t' << batch.valid_lane_mask
             << "\t-\t-\t-\t" << batch.union_tiles[tile].value() << '\t'
             << batch.union_tile_lane_masks[tile]
             << "\t-\t-\t-\t-\t-\t-\t-\n";
    }
  }
  return output.str();
}

std::uint64_t fingerprint_batch_plan(const BatchPlan& plan) {
  constexpr std::uint64_t offset_basis = 14'695'981'039'346'656'037ULL;
  constexpr std::uint64_t prime = 1'099'511'628'211ULL;
  std::uint64_t fingerprint = offset_basis;
  for (const char value : serialize_batch_plan_tsv(plan)) {
    fingerprint ^= static_cast<std::uint64_t>(
        static_cast<unsigned char>(value));
    fingerprint *= prime;
  }
  return fingerprint;
}

}  // namespace bfnew
