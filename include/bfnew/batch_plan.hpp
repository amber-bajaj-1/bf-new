#pragma once

#include "bfnew/gpu_api.hpp"
#include "bfnew/selected_region_index.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace bfnew {

inline constexpr std::uint32_t invalid_batch_query_index =
    static_cast<std::uint32_t>(-1);
inline constexpr QueryId invalid_batch_query_id{
    static_cast<std::uint32_t>(-1)};

struct BatchFraction {
  std::uint64_t numerator{};
  std::uint64_t denominator{1U};

  constexpr bool operator==(const BatchFraction&) const noexcept = default;
};

struct BatchPlannerPolicy {
  std::uint32_t lane_width{32U};
  std::uint32_t minimum_jaccard_numerator{1U};
  std::uint32_t minimum_jaccard_denominator{8U};
  std::uint32_t maximum_union_inflation_numerator{2U};
  std::uint32_t maximum_union_inflation_denominator{1U};

  constexpr bool operator==(const BatchPlannerPolicy&) const noexcept = default;
};

enum class BatchPlannerPolicyError : std::uint8_t {
  none,
  unsupported_lane_width,
  invalid_jaccard_fraction,
  invalid_union_inflation_fraction,
};

[[nodiscard]] BatchPlannerPolicyError validate_batch_planner_policy(
    const BatchPlannerPolicy& policy) noexcept;

struct BatchQueryFeatures {
  QueryId query_id{};
  std::uint32_t expansion_generation{};
  std::uint32_t source_count{};
  std::uint32_t target_count{};
  std::vector<TileId> source_tiles;
  std::vector<TileId> target_tiles;
  std::vector<TileId> selected_tiles;
  std::uint64_t selected_vertex_count{};
  std::uint64_t selected_edge_estimate{};

  bool operator==(const BatchQueryFeatures&) const = default;
};

[[nodiscard]] std::vector<BatchQueryFeatures> make_batch_query_features(
    const WeightedGraph& graph,
    const SelectedRegionIndex& selected_regions,
    std::span<const RouteQuery> queries);

[[nodiscard]] BatchFraction tile_set_jaccard(
    std::span<const TileId> left,
    std::span<const TileId> right);

// Union inflation is the number of union tile/lane positions divided by the
// useful selected tile/lane positions for the valid lanes.
[[nodiscard]] BatchFraction batch_union_tile_inflation(
    std::span<const BatchQueryFeatures> queries,
    std::span<const std::uint32_t> query_indices,
    std::span<const TileId> union_tiles);

struct BatchPlanEntry {
  std::uint32_t lane_width{};
  LaneMask valid_lane_mask{};
  std::vector<std::uint32_t> query_indices_by_lane;
  std::vector<QueryId> query_ids_by_lane;
  std::vector<std::uint32_t> expansion_generations_by_lane;
  std::vector<TileId> union_tiles;
  std::vector<LaneMask> union_tile_lane_masks;
  std::uint64_t union_vertex_count{};
  std::uint64_t union_edge_estimate{};
  std::uint64_t selected_lane_vertex_count{};
  std::uint64_t selected_lane_edge_estimate{};

  bool operator==(const BatchPlanEntry&) const = default;
};

struct BatchPlan {
  BatchPlannerPolicy policy{};
  std::uint32_t input_query_count{};
  std::vector<BatchPlanEntry> batches;

  bool operator==(const BatchPlan&) const = default;
};

// Canonical preference order: wave32 first, then the requested width-16 and
// width-8 comparison plans.
struct BatchPlanFamily {
  std::array<BatchPlan, 3U> plans;

  bool operator==(const BatchPlanFamily&) const = default;
};

[[nodiscard]] BatchPlan make_overlapping_batch_plan(
    const SelectedRegionIndex& selected_regions,
    std::span<const BatchQueryFeatures> queries,
    const BatchPlannerPolicy& policy);

[[nodiscard]] BatchPlanFamily make_standard_batch_plan_family(
    const SelectedRegionIndex& selected_regions,
    std::span<const BatchQueryFeatures> queries,
    BatchPlannerPolicy policy = {});

enum class BatchPlanValidationErrorCode : std::uint8_t {
  none,
  invalid_policy,
  query_count_overflow,
  empty_query_set,
  duplicate_query_id,
  invalid_query_feature,
  empty_plan,
  invalid_batch_width,
  invalid_batch_shape,
  invalid_valid_lane_mask,
  invalid_padding,
  query_index_out_of_range,
  query_identity_mismatch,
  duplicate_assignment,
  omitted_query,
  union_tile_mismatch,
  tile_lane_mask_mismatch,
  estimate_mismatch,
};

struct BatchPlanValidationResult {
  static constexpr std::size_t no_position = static_cast<std::size_t>(-1);

  BatchPlanValidationErrorCode code{BatchPlanValidationErrorCode::none};
  std::size_t batch{no_position};
  std::size_t lane{no_position};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == BatchPlanValidationErrorCode::none;
  }
};

[[nodiscard]] BatchPlanValidationResult validate_batch_plan(
    const SelectedRegionIndex& selected_regions,
    std::span<const BatchQueryFeatures> queries,
    const BatchPlan& plan) noexcept;

[[nodiscard]] std::string serialize_batch_plan_tsv(const BatchPlan& plan);

// Frozen 64-bit FNV-1a over the exact bfnew.batch-plan.v1 TSV bytes. This is
// a deterministic evidence identity, not a cryptographic digest.
[[nodiscard]] std::uint64_t fingerprint_batch_plan(const BatchPlan& plan);

}  // namespace bfnew
