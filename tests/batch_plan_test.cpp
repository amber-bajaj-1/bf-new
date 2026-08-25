#include "bfnew/batch_layout.hpp"
#include "bfnew/batch_plan.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/query.hpp"
#include "bfnew/spatial.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using bfnew::BatchDeviceDescription;
using bfnew::BatchPlan;
using bfnew::BatchPlanEntry;
using bfnew::BatchPlanFamily;
using bfnew::BatchPlannerPolicy;
using bfnew::BatchQueryFeatures;
using bfnew::BatchRunRepresentation;
using bfnew::InputGraph;
using bfnew::LaneMask;
using bfnew::PartitionedGraph;
using bfnew::QueryId;
using bfnew::RouteQuery;
using bfnew::SelectedRegionIndex;
using bfnew::TileId;
using bfnew::TileRunLayout64;
using bfnew::VertexId;

int failures = 0;

void expect(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAILED: " << description << '\n';
    ++failures;
  }
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, const std::string_view description) {
  try {
    function();
    expect(false, description);
  } catch (const Exception&) {
  } catch (...) {
    expect(false, description);
  }
}

struct Phase13Fixture {
  PartitionedGraph partitioned;
  TileRunLayout64 tile_runs;
  std::vector<RouteQuery> queries;
};

[[nodiscard]] bfnew::PhysicalProvenance provenance(
    const std::uint64_t record) noexcept {
  return bfnew::PhysicalProvenance{
      bfnew::provenance_domain::synthetic,
      bfnew::provenance_kind::synthetic_edge,
      record,
  };
}

[[nodiscard]] InputGraph make_phase13_input_graph() {
  const bfnew::ResourceClassId resource{1U};
  std::vector<bfnew::VertexMetadata> vertices{
      bfnew::VertexMetadata::located(0, 0, resource),
      bfnew::VertexMetadata::located(1, 0, resource),
      bfnew::VertexMetadata::located(10, 0, resource),
      bfnew::VertexMetadata::located(30, 0, resource),
      bfnew::VertexMetadata::unlocated(resource),
      bfnew::VertexMetadata::located(11, 0, resource),
      bfnew::VertexMetadata::located(2, 0, resource),
  };
  std::vector<bfnew::EdgeInputRecord> edges{
      {VertexId{0U}, VertexId{1U}, 1.0F, provenance(0U)},
      {VertexId{0U}, VertexId{2U}, 2.0F, provenance(1U)},
      {VertexId{0U}, VertexId{3U}, 3.0F, provenance(2U)},
      {VertexId{0U}, VertexId{3U}, 4.0F, provenance(3U)},
      {VertexId{0U}, VertexId{4U}, 5.0F, provenance(4U)},
      {VertexId{4U}, VertexId{0U}, 6.0F, provenance(5U)},
      {VertexId{2U}, VertexId{0U}, 7.0F, provenance(6U)},
      {VertexId{1U}, VertexId{2U}, 8.0F, provenance(7U)},
  };
  return InputGraph{std::move(vertices), std::move(edges)};
}

[[nodiscard]] RouteQuery with_selected_tiles(
    RouteQuery query,
    std::initializer_list<TileId> selected_tiles,
    const bfnew::WeightedGraph& graph) {
  query.selected_tiles.assign(selected_tiles.begin(), selected_tiles.end());
  std::ranges::sort(query.selected_tiles);
  if (!bfnew::validate_route_query(graph, query).ok()) {
    throw std::runtime_error{"invalid Phase 13 test query"};
  }
  return query;
}

[[nodiscard]] Phase13Fixture make_phase13_fixture() {
  const bfnew::UniformGridPartitioner partitioner{
      bfnew::SpatialOrderConfig{0, 0, 10U, 10U}};
  PartitionedGraph partitioned = partitioner.partition(make_phase13_input_graph());
  const bfnew::WeightedGraph& graph = partitioned.graph;
  const auto old_to_new = graph.old_to_new();
  const auto owner_tiles = graph.owner_tiles();
  const auto owner_of_old = [&](const std::size_t old_vertex) {
    return owner_tiles[old_to_new[old_vertex].value()];
  };
  const TileId tile_zero = owner_of_old(0U);
  const TileId tile_one = owner_of_old(2U);
  const TileId tile_two = owner_of_old(3U);
  const TileId spill_tile = owner_of_old(4U);

  const std::array q10_sources{old_to_new[0U], old_to_new[2U]};
  const std::array q10_targets{old_to_new[5U]};
  const std::array q20_sources{old_to_new[0U]};
  const std::array q20_targets{old_to_new[4U]};
  const std::array q30_sources{old_to_new[4U]};
  const std::array q30_targets{old_to_new[4U]};
  const std::array q40_sources{old_to_new[2U]};
  const std::array q40_targets{old_to_new[3U]};

  std::vector<RouteQuery> queries;
  queries.push_back(with_selected_tiles(
      bfnew::make_route_query(
          QueryId{10U}, graph, q10_sources, q10_targets, 0U, 0U),
      {tile_zero, tile_one},
      graph));
  queries.push_back(with_selected_tiles(
      bfnew::make_route_query(
          QueryId{20U}, graph, q20_sources, q20_targets, 0U, 1U),
      {tile_zero, spill_tile},
      graph));
  queries.push_back(with_selected_tiles(
      bfnew::make_route_query(
          QueryId{30U}, graph, q30_sources, q30_targets, 0U, 2U),
      {spill_tile},
      graph));
  queries.push_back(with_selected_tiles(
      bfnew::make_route_query(
          QueryId{40U}, graph, q40_sources, q40_targets, 0U, 3U),
      {tile_one, tile_two},
      graph));

  TileRunLayout64 tile_runs = bfnew::build_tile_run_layout(graph);
  return Phase13Fixture{
      std::move(partitioned), std::move(tile_runs), std::move(queries)};
}

[[nodiscard]] BatchPlannerPolicy golden_policy(
    const std::uint32_t lane_width) {
  BatchPlannerPolicy policy;
  policy.lane_width = lane_width;
  policy.maximum_union_inflation_numerator = 3U;
  policy.maximum_union_inflation_denominator = 1U;
  return policy;
}

[[nodiscard]] LaneMask low_lane_mask(const std::uint32_t lanes) {
  if (lanes == 32U) {
    return std::numeric_limits<LaneMask>::max();
  }
  return (LaneMask{1U} << lanes) - 1U;
}

[[nodiscard]] std::vector<QueryId> valid_query_ids(const BatchPlan& plan) {
  std::vector<QueryId> result;
  result.reserve(plan.input_query_count);
  for (const BatchPlanEntry& batch : plan.batches) {
    for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
      if ((batch.valid_lane_mask & (LaneMask{1U} << lane)) != 0U) {
        result.push_back(batch.query_ids_by_lane[lane]);
      }
    }
  }
  return result;
}

[[nodiscard]] bool plans_have_same_semantics(
    const BatchPlan& left,
    const BatchPlan& right) {
  if (left.policy != right.policy ||
      left.input_query_count != right.input_query_count ||
      left.batches.size() != right.batches.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.batches.size(); ++index) {
    BatchPlanEntry left_batch = left.batches[index];
    BatchPlanEntry right_batch = right.batches[index];
    left_batch.query_indices_by_lane.clear();
    right_batch.query_indices_by_lane.clear();
    if (left_batch != right_batch) {
      return false;
    }
  }
  return true;
}

void test_policy_and_fraction_contract() {
  BatchPlannerPolicy policy;
  expect(
      bfnew::validate_batch_planner_policy(policy) ==
          bfnew::BatchPlannerPolicyError::none,
      "the wave32 planner policy is valid");
  for (const std::uint32_t width : {1U, 8U, 16U, 32U}) {
    policy.lane_width = width;
    expect(
        bfnew::validate_batch_planner_policy(policy) ==
            bfnew::BatchPlannerPolicyError::none,
        "the Phase 14 singleton width and all Phase 13 widths are accepted");
  }
  for (const std::uint32_t width : {0U, 7U, 64U}) {
    policy.lane_width = width;
    expect(
        bfnew::validate_batch_planner_policy(policy) ==
            bfnew::BatchPlannerPolicyError::unsupported_lane_width,
        "widths outside 1/8/16/32 are rejected");
  }
  policy = BatchPlannerPolicy{};
  policy.minimum_jaccard_denominator = 0U;
  expect(
      bfnew::validate_batch_planner_policy(policy) ==
          bfnew::BatchPlannerPolicyError::invalid_jaccard_fraction,
      "a zero Jaccard denominator is rejected");
  policy = BatchPlannerPolicy{};
  policy.maximum_union_inflation_denominator = 0U;
  expect(
      bfnew::validate_batch_planner_policy(policy) ==
          bfnew::BatchPlannerPolicyError::invalid_union_inflation_fraction,
      "a zero union-inflation denominator is rejected");

  const std::array<TileId, 2U> left{TileId{0U}, TileId{1U}};
  const std::array<TileId, 2U> right{TileId{1U}, TileId{2U}};
  const bfnew::BatchFraction jaccard = bfnew::tile_set_jaccard(left, right);
  expect(
      jaccard.numerator * 3U == jaccard.denominator,
      "tile Jaccard uses the exact one-third ratio");
}

void test_golden_planner_and_features() {
  const Phase13Fixture fixture = make_phase13_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const SelectedRegionIndex selected_regions{graph, fixture.tile_runs};
  const std::vector<BatchQueryFeatures> features =
      bfnew::make_batch_query_features(graph, selected_regions, fixture.queries);

  expect(features.size() == 4U, "the golden corpus has four query profiles");
  expect(
      features[0U].source_count == 2U &&
          features[1U].source_count == 1U &&
          features[2U].source_count == 1U &&
          features[3U].source_count == 1U,
      "one- and two-source queries remain independent planner inputs");
  expect(
      features[0U].source_tiles ==
              std::vector<TileId>{TileId{0U}, TileId{1U}} &&
          features[0U].target_tiles == std::vector<TileId>{TileId{1U}} &&
          features[1U].source_tiles == std::vector<TileId>{TileId{0U}} &&
          features[1U].target_tiles == std::vector<TileId>{TileId{3U}},
      "source and target tile sets remain separate canonical planner features");
  expect(
      features[0U].selected_vertex_count == 5U &&
          features[1U].selected_vertex_count == 4U &&
          features[2U].selected_vertex_count == 1U &&
          features[3U].selected_vertex_count == 3U,
      "selected-region vertex estimates use actual selected tiles");
  expect(
      features[0U].selected_edge_estimate == 4U &&
          features[1U].selected_edge_estimate == 3U &&
          features[2U].selected_edge_estimate == 0U &&
          features[3U].selected_edge_estimate == 0U,
      "selected-region edge estimates count only induced admitted edges");
  for (std::size_t index = 0U; index < fixture.queries.size(); ++index) {
    expect(
        features[index].selected_vertex_count ==
                bfnew::estimate_selected_vertex_count(
                    graph, fixture.queries[index].selected_tiles) &&
            features[index].selected_edge_estimate ==
                bfnew::estimate_selected_edge_count(
                    graph, fixture.queries[index].selected_tiles),
        "selected-region index exactly matches the legacy bounded estimators");
  }

  const Phase13Fixture separate_fixture = make_phase13_fixture();
  const SelectedRegionIndex separate_index{
      separate_fixture.partitioned.graph, separate_fixture.tile_runs};
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::make_batch_query_features(
            graph, separate_index, fixture.queries));
      },
      "a same-shaped index bound to a separate graph object is rejected");

  RouteQuery expanded = fixture.queries.front();
  expanded.query_id = QueryId{11U};
  expanded.expansion_generation = 4U;
  expanded.selected_tiles.push_back(TileId{2U});
  expect(
      bfnew::validate_route_query(graph, expanded).ok(),
      "expanded selected-tile fixture remains a valid query");
  const std::array same_terminals_different_regions{
      fixture.queries.front(), expanded};
  const std::vector<BatchQueryFeatures> expansion_features =
      bfnew::make_batch_query_features(
          graph, selected_regions, same_terminals_different_regions);
  const BatchPlan expansion_plan = bfnew::make_overlapping_batch_plan(
      selected_regions, expansion_features, golden_policy(8U));
  expect(
      expansion_plan.batches.size() == 1U &&
          expansion_plan.batches.front().query_ids_by_lane[0U] == QueryId{11U} &&
          expansion_plan.batches.front().query_ids_by_lane[1U] == QueryId{10U} &&
          expansion_plan.batches.front().union_tiles ==
              std::vector<TileId>{TileId{0U}, TileId{1U}, TileId{2U}} &&
          expansion_plan.batches.front().union_tile_lane_masks ==
              std::vector<LaneMask>{0b11U, 0b11U, 0b01U},
      "planner uses actual expanded selected tiles rather than terminal rectangles");

  const BatchPlan plan = bfnew::make_overlapping_batch_plan(
      selected_regions, features, golden_policy(8U));
  expect(
      bfnew::validate_batch_plan(selected_regions, features, plan).ok(),
      "the golden width-eight plan passes deep validation");
  expect(plan.batches.size() == 1U, "the permissive golden policy makes one batch");
  const BatchPlanEntry& batch = plan.batches.front();
  expect(
      batch.lane_width == 8U && batch.valid_lane_mask == 0x0fU,
      "the four queries occupy a padded wave-eight prefix");
  expect(
      batch.query_ids_by_lane ==
          std::vector<QueryId>{
              QueryId{10U},
              QueryId{20U},
              QueryId{30U},
              QueryId{40U},
              bfnew::invalid_batch_query_id,
              bfnew::invalid_batch_query_id,
              bfnew::invalid_batch_query_id,
              bfnew::invalid_batch_query_id,
          },
      "exact Jaccard/edge/query tie breaks produce the golden lane order");
  expect(
      batch.expansion_generations_by_lane ==
          std::vector<std::uint32_t>{0U, 1U, 2U, 3U, 0U, 0U, 0U, 0U},
      "expansion generations stay attached to their query lanes");
  expect(
      batch.union_tiles ==
              std::vector<TileId>{TileId{0U}, TileId{1U}, TileId{2U}, TileId{3U}} &&
          batch.union_tile_lane_masks ==
              std::vector<LaneMask>{0b0011U, 0b1001U, 0b1000U, 0b0110U},
      "union tiles and exact lane masks derive from selected tile sets");
  expect(
      batch.union_vertex_count == 7U && batch.union_edge_estimate == 8U &&
          batch.selected_lane_vertex_count == 13U &&
          batch.selected_lane_edge_estimate == 7U,
      "union and useful-work estimates retain selected-versus-wasted accounting");
  const std::array<std::uint32_t, 4U> query_indices{0U, 1U, 2U, 3U};
  const bfnew::BatchFraction inflation = bfnew::batch_union_tile_inflation(
      features, query_indices, batch.union_tiles);
  expect(
      inflation.numerator * 7U == inflation.denominator * 16U,
      "union inflation is exact union tile/lane positions over useful positions");

  const std::string first = bfnew::serialize_batch_plan_tsv(plan);
  const std::string second = bfnew::serialize_batch_plan_tsv(plan);
  constexpr std::string_view literal_golden =
      "schema\tbfnew.batch-plan.v1\n"
      "record\tbatch\tposition\tlane_width\tvalid_lane_mask\tquery_index\tquery_id\t"
      "expansion_generation\ttile_id\ttile_lane_mask\tunion_vertex_count\t"
      "union_edge_estimate\tselected_lane_vertex_count\t"
      "selected_lane_edge_estimate\tinput_query_count\tminimum_jaccard\t"
      "maximum_union_inflation\n"
      "plan\t-\t-\t8\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t4\t1/8\t3/1\n"
      "batch\t0\t-\t8\t15\t-\t-\t-\t-\t-\t7\t8\t13\t7\t-\t-\t-\n"
      "lane\t0\t0\t8\t15\t0\t10\t0\t-\t-\t-\t-\t-\t-\t-\t-\t-\n"
      "lane\t0\t1\t8\t15\t1\t20\t1\t-\t-\t-\t-\t-\t-\t-\t-\t-\n"
      "lane\t0\t2\t8\t15\t2\t30\t2\t-\t-\t-\t-\t-\t-\t-\t-\t-\n"
      "lane\t0\t3\t8\t15\t3\t40\t3\t-\t-\t-\t-\t-\t-\t-\t-\t-\n"
      "lane\t0\t4\t8\t15\t4294967295\t4294967295\t0\t-\t-\t-\t-\t-\t-\t-\t-\t-\n"
      "lane\t0\t5\t8\t15\t4294967295\t4294967295\t0\t-\t-\t-\t-\t-\t-\t-\t-\t-\n"
      "lane\t0\t6\t8\t15\t4294967295\t4294967295\t0\t-\t-\t-\t-\t-\t-\t-\t-\t-\n"
      "lane\t0\t7\t8\t15\t4294967295\t4294967295\t0\t-\t-\t-\t-\t-\t-\t-\t-\t-\n"
      "tile\t0\t0\t8\t15\t-\t-\t-\t0\t3\t-\t-\t-\t-\t-\t-\t-\n"
      "tile\t0\t1\t8\t15\t-\t-\t-\t1\t9\t-\t-\t-\t-\t-\t-\t-\n"
      "tile\t0\t2\t8\t15\t-\t-\t-\t2\t8\t-\t-\t-\t-\t-\t-\t-\n"
      "tile\t0\t3\t8\t15\t-\t-\t-\t3\t6\t-\t-\t-\t-\t-\t-\t-\n";
  expect(
      first == second && first == literal_golden,
      "complete four-query TSV serialization matches the literal golden");
  constexpr std::uint64_t literal_fingerprint = 16'700'197'086'403'570'866ULL;
  expect(
      bfnew::fingerprint_batch_plan(plan) == literal_fingerprint &&
          bfnew::fingerprint_batch_plan(plan) ==
              bfnew::fingerprint_batch_plan(plan),
      "the frozen TSV fingerprint matches the literal deterministic golden");
  BatchPlan changed_fingerprint = plan;
  ++changed_fingerprint.batches.front().union_edge_estimate;
  expect(
      bfnew::fingerprint_batch_plan(changed_fingerprint) !=
          literal_fingerprint,
      "a changed serialized plan changes its evidence fingerprint");

  BatchPlan corrupted = plan;
  corrupted.batches.front().union_tile_lane_masks.front() ^= 1U;
  expect(
      !bfnew::validate_batch_plan(selected_regions, features, corrupted).ok(),
      "plan validation rejects a changed tile lane mask");
  corrupted = plan;
  ++corrupted.batches.front().selected_lane_edge_estimate;
  expect(
      !bfnew::validate_batch_plan(selected_regions, features, corrupted).ok(),
      "plan validation rejects a changed selected-edge estimate");
  corrupted = plan;
  corrupted.batches.front().query_ids_by_lane[4U] = QueryId{99U};
  expect(
      !bfnew::validate_batch_plan(selected_regions, features, corrupted).ok(),
      "plan validation rejects semantic data in a padded lane");
  corrupted = plan;
  corrupted.batches.front().valid_lane_mask = 0b1011U;
  expect(
      !bfnew::validate_batch_plan(selected_regions, features, corrupted).ok(),
      "plan validation rejects a non-prefix validity mask");
}

[[nodiscard]] std::vector<RouteQuery> make_tie_corpus(
    const Phase13Fixture& fixture,
    const std::size_t count,
    const bool reverse_order) {
  std::vector<RouteQuery> queries;
  queries.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    RouteQuery query = fixture.queries.front();
    query.query_id = QueryId{static_cast<std::uint32_t>(100U + index)};
    query.expansion_generation = 0U;
    queries.push_back(std::move(query));
  }
  if (reverse_order) {
    std::ranges::reverse(queries);
  }
  return queries;
}

void expect_standard_partition(
    const BatchPlan& plan,
    const std::uint32_t width,
    const std::size_t query_count) {
  const std::size_t expected_batches =
      (query_count + static_cast<std::size_t>(width) - 1U) /
      static_cast<std::size_t>(width);
  expect(plan.policy.lane_width == width, "standard family retains its width");
  expect(plan.batches.size() == expected_batches, "batch count is exact");
  std::size_t assigned = 0U;
  for (std::size_t batch_index = 0U; batch_index < plan.batches.size(); ++batch_index) {
    const BatchPlanEntry& batch = plan.batches[batch_index];
    const std::size_t remaining = query_count - assigned;
    const std::uint32_t valid = static_cast<std::uint32_t>(
        std::min<std::size_t>(remaining, width));
    expect(
        batch.valid_lane_mask == low_lane_mask(valid),
        "each batch uses a canonical prefix validity mask");
    for (std::uint32_t lane = 0U; lane < width; ++lane) {
      if (lane < valid) {
        expect(
            batch.query_ids_by_lane[lane] ==
                QueryId{static_cast<std::uint32_t>(100U + assigned + lane)},
            "all-tie lanes use QueryId as the deterministic final tie break");
      } else {
        expect(
            batch.query_indices_by_lane[lane] ==
                    bfnew::invalid_batch_query_index &&
                batch.query_ids_by_lane[lane] ==
                    bfnew::invalid_batch_query_id &&
                batch.expansion_generations_by_lane[lane] == 0U,
            "padded lanes contain no query identity");
      }
    }
    assigned += valid;
  }
  expect(assigned == query_count, "every query is assigned exactly once");
}

void test_family_once_only_singletons_and_ties() {
  const Phase13Fixture fixture = make_phase13_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const SelectedRegionIndex selected_regions{graph, fixture.tile_runs};

  constexpr std::array<std::size_t, 11U> corpus_sizes{
      1U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 32U, 33U, 10U};
  const std::array<std::uint32_t, 3U> widths{32U, 16U, 8U};
  for (const std::size_t corpus_size : corpus_sizes) {
    const std::vector<RouteQuery> ordered_queries =
        make_tie_corpus(fixture, corpus_size, false);
    const std::vector<RouteQuery> permuted_queries =
        make_tie_corpus(fixture, corpus_size, true);
    const std::vector<BatchQueryFeatures> ordered_features =
        bfnew::make_batch_query_features(
            graph, selected_regions, ordered_queries);
    const std::vector<BatchQueryFeatures> permuted_features =
        bfnew::make_batch_query_features(
            graph, selected_regions, permuted_queries);
    expect(
        ordered_features == permuted_features,
        "canonical feature sorting removes RouteQuery input permutation order");
    const BatchPlanFamily ordered = bfnew::make_standard_batch_plan_family(
        selected_regions, ordered_features);
    const BatchPlanFamily permuted = bfnew::make_standard_batch_plan_family(
        selected_regions, permuted_features);
    expect(
        ordered == permuted,
        "the literal full plan family is invariant after canonical feature sorting");
    for (std::size_t index = 0U; index < widths.size(); ++index) {
      expect(
          bfnew::validate_batch_plan(
              selected_regions, ordered_features, ordered.plans[index])
              .ok(),
          "every boundary-size standard plan passes exact assignment validation");
      expect_standard_partition(
          ordered.plans[index], widths[index], corpus_size);
      expect(
          bfnew::serialize_batch_plan_tsv(ordered.plans[index]) ==
              bfnew::serialize_batch_plan_tsv(permuted.plans[index]),
          "serialized plans are literally permutation invariant");
      expect(
          bfnew::fingerprint_batch_plan(ordered.plans[index]) ==
              bfnew::fingerprint_batch_plan(permuted.plans[index]),
          "plan fingerprints are invariant to RouteQuery input order");
    }
  }

  const std::vector<RouteQuery> forward_queries =
      make_tie_corpus(fixture, 33U, false);
  const std::vector<RouteQuery> reverse_queries =
      make_tie_corpus(fixture, 33U, true);
  const std::vector<BatchQueryFeatures> forward_features =
      bfnew::make_batch_query_features(graph, selected_regions, forward_queries);
  const std::vector<BatchQueryFeatures> reverse_features =
      bfnew::make_batch_query_features(graph, selected_regions, reverse_queries);

  const BatchPlanFamily forward = bfnew::make_standard_batch_plan_family(
      selected_regions, forward_features);
  const BatchPlanFamily reverse = bfnew::make_standard_batch_plan_family(
      selected_regions, reverse_features);
  expect(
      forward == reverse,
      "the complete 33-query plan family is literally input-order invariant");
  for (std::size_t index = 0U; index < widths.size(); ++index) {
    expect(
        bfnew::validate_batch_plan(
            selected_regions, forward_features, forward.plans[index])
            .ok(),
        "every standard-width plan passes assignment validation");
    expect_standard_partition(forward.plans[index], widths[index], 33U);
    expect(
        plans_have_same_semantics(forward.plans[index], reverse.plans[index]),
        "tie results are independent of input ordering after index resolution");
    expect(
        valid_query_ids(forward.plans[index]) ==
            valid_query_ids(reverse.plans[index]),
        "permutation determinism preserves the exact QueryId sequence");
  }
  expect(
      forward.plans[0U].batches.size() == 2U &&
          forward.plans[0U].batches.back().valid_lane_mask == 1U,
      "width 32 retains a singleton padded remainder");
  expect(
      forward.plans[1U].batches.size() == 3U &&
          forward.plans[1U].batches.back().valid_lane_mask == 1U,
      "width 16 retains a singleton padded remainder");
  expect(
      forward.plans[2U].batches.size() == 5U &&
          forward.plans[2U].batches.back().valid_lane_mask == 1U,
      "width 8 retains a singleton padded remainder");

  std::vector<BatchQueryFeatures> ten_features(
      forward_features.begin(), forward_features.begin() + 10);
  const BatchPlan ten = bfnew::make_overlapping_batch_plan(
      selected_regions, ten_features, BatchPlannerPolicy{8U});
  expect(
      ten.batches.size() == 2U && ten.batches.back().valid_lane_mask == 0b11U,
      "a non-singleton padded remainder retains both remaining queries");

  BatchPlan omitted = forward.plans[2U];
  omitted.batches.pop_back();
  expect(
      bfnew::validate_batch_plan(selected_regions, forward_features, omitted).code ==
          bfnew::BatchPlanValidationErrorCode::omitted_query,
      "deep validation detects omitted queries");
  BatchPlan duplicated = forward.plans[2U];
  BatchPlanEntry& final_batch = duplicated.batches.back();
  const BatchPlanEntry& first_batch = duplicated.batches.front();
  final_batch.query_indices_by_lane[0U] = first_batch.query_indices_by_lane[0U];
  final_batch.query_ids_by_lane[0U] = first_batch.query_ids_by_lane[0U];
  final_batch.expansion_generations_by_lane[0U] =
      first_batch.expansion_generations_by_lane[0U];
  expect(
      bfnew::validate_batch_plan(selected_regions, forward_features, duplicated).code ==
          bfnew::BatchPlanValidationErrorCode::duplicate_assignment,
      "deep validation detects duplicate assignments");

  std::vector<BatchQueryFeatures> duplicate_ids = forward_features;
  duplicate_ids[1U].query_id = duplicate_ids[0U].query_id;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::make_overlapping_batch_plan(
            selected_regions, duplicate_ids, BatchPlannerPolicy{8U}));
      },
      "planner input rejects duplicate stable query IDs");

  RouteQuery maximum_id_query = fixture.queries.front();
  maximum_id_query.query_id = bfnew::invalid_batch_query_id;
  const std::array maximum_id_queries{maximum_id_query};
  const std::vector<BatchQueryFeatures> maximum_id_features =
      bfnew::make_batch_query_features(
          graph, selected_regions, maximum_id_queries);
  const BatchPlan maximum_id_plan = bfnew::make_overlapping_batch_plan(
      selected_regions, maximum_id_features, BatchPlannerPolicy{8U});
  expect(
      bfnew::validate_batch_plan(
          selected_regions, maximum_id_features, maximum_id_plan)
          .ok(),
      "UINT32_MAX remains a legal query ID in a valid lane");
  const BatchPlanEntry& maximum_batch = maximum_id_plan.batches.front();
  expect(
      maximum_batch.valid_lane_mask == 1U &&
          maximum_batch.query_ids_by_lane[0U] ==
              bfnew::invalid_batch_query_id &&
          maximum_batch.query_indices_by_lane[0U] == 0U &&
          maximum_batch.query_ids_by_lane[1U] ==
              bfnew::invalid_batch_query_id &&
          maximum_batch.query_indices_by_lane[1U] ==
              bfnew::invalid_batch_query_index,
      "validity and query-index padding distinguish max QueryId from padding");
}

void test_device_layout_golden_and_run_proof() {
  const Phase13Fixture fixture = make_phase13_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const SelectedRegionIndex selected_regions{graph, fixture.tile_runs};
  const std::vector<BatchQueryFeatures> features =
      bfnew::make_batch_query_features(graph, selected_regions, fixture.queries);
  const BatchPlan plan = bfnew::make_overlapping_batch_plan(
      selected_regions, features, golden_policy(8U));
  const BatchPlanEntry& batch = plan.batches.front();

  BatchDeviceDescription retained;
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.queries,
      features,
      batch,
      BatchRunRepresentation::retained_per_run_masks,
      retained);
  expect(
      !retained.compact_vertex_mapping_valid,
      "run preparation leaves compact mapping work separate and measurable");
  bfnew::prepare_compact_vertex_mapping(graph, batch, retained);
  expect(
      bfnew::validate_batch_device_description(
          graph, fixture.tile_runs, fixture.queries, features, batch, retained)
          .ok(),
      "retained-mask device description passes deep validation");
  expect(
      bfnew::validate_compact_vertex_mapping(graph, batch, retained).ok() &&
          retained.touched_compact_tiles ==
              std::vector<std::uint32_t>{0U, 1U, 2U, 3U} &&
          retained.compact_vertex_biases_by_tile ==
              std::vector<std::uint32_t>{0U, 0U, 0U, 0U} &&
          retained.compact_mapping_report.entries_initialized == 4U &&
          retained.compact_mapping_report.entries_cleared == 0U &&
          retained.compact_mapping_report.entries_written == 4U,
      "dense compact mapping records every union tile and exact packed biases");
  expect(
      bfnew::prove_batch_endpoint_admission(graph, fixture.tile_runs, retained).ok(),
      "retained run masks agree with endpoint-by-endpoint admission");
  expect(
      retained.source_offsets ==
              std::vector<std::uint32_t>{0U, 2U, 3U, 4U, 5U, 5U, 5U, 5U, 5U} &&
          retained.target_offsets ==
              std::vector<std::uint32_t>{0U, 1U, 2U, 3U, 4U, 4U, 4U, 4U, 4U},
      "one/two-source slices and padded source/target offsets are exact");
  expect(
      retained.selected_vertex_counts_by_lane ==
              std::vector<std::uint64_t>{5U, 4U, 1U, 3U, 0U, 0U, 0U, 0U} &&
          retained.selected_edge_estimates_by_lane ==
              std::vector<std::uint64_t>{4U, 3U, 0U, 0U, 0U, 0U, 0U, 0U},
      "per-lane estimates retain useful selected work and zero padding");
  expect(
      retained.tile_lane_masks ==
          std::vector<LaneMask>{0b0011U, 0b1001U, 0b1000U, 0b0110U},
      "dense device tile masks match the planner golden");
  expect(
      retained.selected_vertex_ranges ==
          std::vector<bfnew::BatchVertexRange>{
              {0U, 3U, 0b0011U},
              {3U, 5U, 0b1001U},
              {5U, 6U, 0b1000U},
              {6U, 7U, 0b0110U},
          },
      "selected ranges preserve contiguous tile ranges and lane masks");
  expect(
      retained.csr_run_lane_masks ==
              std::vector<LaneMask>{3U, 1U, 0U, 2U, 1U, 1U, 2U} &&
          retained.csc_run_lane_masks ==
              std::vector<LaneMask>{1U, 2U, 3U, 1U, 0U, 2U},
      "retained CSR/CSC masks intersect owner and remote tile masks once per run");
  expect(
      retained.touched_csr_runs ==
              std::vector<std::uint32_t>{0U, 1U, 3U, 4U, 5U, 6U} &&
          retained.touched_csc_runs ==
              std::vector<std::uint32_t>{0U, 1U, 2U, 3U, 5U},
      "retained touched ledgers contain exactly the sorted nonzero runs");
  expect(
      retained.run_report.csr_runs_visited == 7U &&
          retained.run_report.csc_runs_visited == 6U &&
          retained.run_report.active_csr_runs == 6U &&
          retained.run_report.active_csc_runs == 5U &&
          retained.run_report.csr_lane_edge_pairs == 7U &&
          retained.run_report.csc_lane_edge_pairs == 7U &&
          retained.run_report.retained_entries_initialized == 13U &&
          retained.run_report.csr_retained_entries_initialized == 7U &&
          retained.run_report.csc_retained_entries_initialized == 6U &&
          retained.run_report.retained_entries_cleared == 0U &&
          retained.run_report.retained_entries_written == 11U &&
          retained.run_report.csr_retained_entries_written == 6U &&
          retained.run_report.csc_retained_entries_written == 5U,
      "run report counts visited, active, and admitted parallel-edge work exactly");

  const std::size_t csr_capacity = retained.csr_run_lane_masks.capacity();
  const std::size_t csc_capacity = retained.csc_run_lane_masks.capacity();
  const std::size_t mapping_capacity =
      retained.compact_vertex_biases_by_tile.capacity();
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.queries,
      features,
      batch,
      BatchRunRepresentation::retained_per_run_masks,
      retained);
  expect(
      !retained.compact_vertex_mapping_valid,
      "a newly prepared batch invalidates, but preserves, the old mapping ledger");
  bfnew::prepare_compact_vertex_mapping(graph, batch, retained);
  expect(
      retained.run_report.retained_entries_cleared == 11U &&
          retained.run_report.retained_entries_written == 11U &&
          retained.csr_run_lane_masks.capacity() == csr_capacity &&
          retained.csc_run_lane_masks.capacity() == csc_capacity &&
          retained.compact_vertex_biases_by_tile.capacity() == mapping_capacity &&
          retained.compact_mapping_report.entries_initialized == 0U &&
          retained.compact_mapping_report.entries_cleared == 4U &&
          retained.compact_mapping_report.entries_written == 4U,
      "repeated preparation clears only touched entries and reuses run/mapping capacity");

  const std::array sparse_query{fixture.queries[2U]};
  const std::vector<BatchQueryFeatures> sparse_features =
      bfnew::make_batch_query_features(graph, selected_regions, sparse_query);
  const BatchPlan sparse_plan = bfnew::make_overlapping_batch_plan(
      selected_regions, sparse_features, BatchPlannerPolicy{8U});
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      sparse_query,
      sparse_features,
      sparse_plan.batches.front(),
      BatchRunRepresentation::retained_per_run_masks,
      retained);
  bfnew::prepare_compact_vertex_mapping(
      graph, sparse_plan.batches.front(), retained);
  const std::uint32_t sparse_tile =
      sparse_plan.batches.front().union_tiles.front().value();
  const std::uint32_t sparse_global_begin =
      retained.selected_vertex_ranges.front().begin;
  expect(
      retained.run_report.retained_entries_cleared == 11U &&
          retained.run_report.retained_entries_written == 0U &&
          retained.touched_csr_runs.empty() && retained.touched_csc_runs.empty() &&
          std::ranges::all_of(
              retained.csr_run_lane_masks,
              [](const LaneMask mask) { return mask == 0U; }) &&
          std::ranges::all_of(
              retained.csc_run_lane_masks,
              [](const LaneMask mask) { return mask == 0U; }) &&
          bfnew::prove_batch_endpoint_admission(
              graph, fixture.tile_runs, retained)
              .ok() &&
          bfnew::validate_compact_vertex_mapping(
              graph, sparse_plan.batches.front(), retained)
              .ok() &&
          retained.touched_compact_tiles ==
              std::vector<std::uint32_t>{sparse_tile} &&
          retained.compact_vertex_biases_by_tile[sparse_tile] ==
              sparse_global_begin &&
          sparse_global_begin -
                  retained.compact_vertex_biases_by_tile[sparse_tile] ==
              0U &&
          retained.compact_mapping_report.entries_cleared == 4U &&
          retained.compact_mapping_report.entries_written == 1U,
      "reused retained storage clears stale dense-batch masks for an all-zero batch");

  BatchDeviceDescription switched_representation = retained;
  expect_throws<std::invalid_argument>(
      [&] {
        bfnew::prepare_batch_device_description(
            graph,
            fixture.tile_runs,
            sparse_query,
            sparse_features,
            sparse_plan.batches.front(),
            BatchRunRepresentation::compact_nonzero_descriptors,
            switched_representation);
      },
      "a warmed reusable description rejects a run-representation switch");

  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.queries,
      features,
      batch,
      BatchRunRepresentation::retained_per_run_masks,
      retained);
  bfnew::prepare_compact_vertex_mapping(graph, batch, retained);

  BatchDeviceDescription descriptors;
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.queries,
      features,
      batch,
      BatchRunRepresentation::compact_nonzero_descriptors,
      descriptors);
  bfnew::prepare_compact_vertex_mapping(graph, batch, descriptors);
  expect(
      bfnew::validate_batch_device_description(
          graph, fixture.tile_runs, fixture.queries, features, batch, descriptors)
          .ok() &&
          bfnew::prove_batch_endpoint_admission(
              graph, fixture.tile_runs, descriptors)
              .ok(),
      "compact descriptors validate and prove the same endpoint admission");
  expect(
      descriptors.csr_run_descriptors ==
              std::vector<bfnew::RunLaneMaskDescriptor>{
                  {0U, 3U},
                  {1U, 1U},
                  {3U, 2U},
                  {4U, 1U},
                  {5U, 1U},
                  {6U, 2U},
              } &&
          descriptors.csc_run_descriptors ==
              std::vector<bfnew::RunLaneMaskDescriptor>{
                  {0U, 1U},
                  {1U, 2U},
                  {2U, 3U},
                  {3U, 1U},
                  {5U, 2U},
              },
      "compact descriptors store only sorted nonzero run IDs and exact masks");
  expect(
      descriptors.csr_descriptor_offsets_by_union_vertex ==
              std::vector<std::uint32_t>{0U, 3U, 4U, 4U, 5U, 5U, 5U, 6U} &&
          descriptors.csc_descriptor_offsets_by_union_vertex ==
              std::vector<std::uint32_t>{0U, 2U, 3U, 3U, 4U, 4U, 4U, 5U} &&
          retained.csr_descriptor_offsets_by_union_vertex.empty() &&
          retained.csc_descriptor_offsets_by_union_vertex.empty(),
      "descriptor offsets give exact traversable CSR/CSC spans per union vertex");
  expect(
      descriptors.run_report.descriptor_entries_written == 11U &&
          descriptors.run_report.csr_descriptor_entries_written == 6U &&
          descriptors.run_report.csc_descriptor_entries_written == 5U &&
          descriptors.run_report.csr_lane_edge_pairs == 7U &&
          descriptors.run_report.csc_lane_edge_pairs == 7U,
      "descriptor reporting preserves the same semantic work totals");

  BatchDeviceDescription device_materialized;
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.queries,
      features,
      batch,
      BatchRunRepresentation::device_materialized_run_masks,
      device_materialized);
  expect(
      bfnew::validate_batch_device_description(
          graph,
          fixture.tile_runs,
          fixture.queries,
          features,
          batch,
          device_materialized)
              .ok() &&
          device_materialized.csr_run_lane_masks.empty() &&
          device_materialized.csc_run_lane_masks.empty() &&
          device_materialized.touched_csr_runs.empty() &&
          device_materialized.touched_csc_runs.empty() &&
          device_materialized.csr_run_descriptors.empty() &&
          device_materialized.csc_run_descriptors.empty() &&
          device_materialized.run_report ==
              bfnew::BatchRunPreparationReport{} &&
          bfnew::prove_batch_endpoint_admission(
              graph, fixture.tile_runs, device_materialized)
              .ok(),
      "device-materialized preparation retains no host run image while deep "
      "validation proves the same endpoint admission");
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.queries,
      features,
      batch,
      BatchRunRepresentation::device_materialized_run_masks,
      device_materialized);
  expect(
      device_materialized.csr_run_lane_masks.empty() &&
          device_materialized.csc_run_lane_masks.empty() &&
          device_materialized.run_report ==
              bfnew::BatchRunPreparationReport{},
      "reused device-materialized preparation performs no host run writes");

  BatchDeviceDescription corrupted = retained;
  corrupted.reached_lane_mask = 1U;
  expect(
      bfnew::validate_batch_device_description(
          graph, fixture.tile_runs, fixture.queries, features, batch, corrupted)
              .code ==
          bfnew::BatchLayoutValidationErrorCode::nonzero_initial_result_mask,
      "Phase 13 descriptions reject precomputed reached/miss results");
  corrupted = retained;
  corrupted.source_offsets[2U] = corrupted.source_offsets[1U];
  expect(
      !bfnew::validate_batch_device_description(
           graph, fixture.tile_runs, fixture.queries, features, batch, corrupted)
           .ok(),
      "layout validation rejects a dropped two-source lane entry");
  corrupted = retained;
  corrupted.csr_run_lane_masks[2U] = 1U;
  expect(
      !bfnew::validate_batch_device_description(
           graph, fixture.tile_runs, fixture.queries, features, batch, corrupted)
           .ok(),
      "layout validation rejects admission of the zero-mask parallel-edge run");
  corrupted = descriptors;
  std::swap(
      corrupted.csr_run_descriptors[0U], corrupted.csr_run_descriptors[1U]);
  expect(
      bfnew::validate_batch_device_description(
          graph, fixture.tile_runs, fixture.queries, features, batch, corrupted)
              .code ==
          bfnew::BatchLayoutValidationErrorCode::invalid_descriptor_order,
      "layout validation rejects unsorted compact descriptors");
  corrupted = descriptors;
  corrupted.csr_descriptor_offsets_by_union_vertex[1U] = 2U;
  expect(
      bfnew::validate_batch_device_description(
          graph, fixture.tile_runs, fixture.queries, features, batch, corrupted)
              .code ==
          bfnew::BatchLayoutValidationErrorCode::invalid_descriptor_offsets,
      "deep validation rejects a structurally valid offset that assigns a run to the wrong vertex");
  corrupted = descriptors;
  ++corrupted.compact_vertex_biases_by_tile[1U];
  expect(
      bfnew::validate_compact_vertex_mapping(graph, batch, corrupted).code ==
          bfnew::BatchLayoutValidationErrorCode::invalid_compact_vertex_mapping,
      "deep compact validation rejects a corrupted tile bias");
}

template <typename T>
concept HasDistances = requires(T value) { value.distances; };

template <typename T>
concept HasRunBatch = requires(T value) { value.run_batch(); };

static_assert(!HasDistances<BatchDeviceDescription>);
static_assert(!HasRunBatch<BatchDeviceDescription>);

[[nodiscard]] std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return {};
  }
  return std::string{
      std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

void test_phase14_scope_guard() {
  const std::filesystem::path source_root =
      std::filesystem::path{__FILE__}.parent_path().parent_path();
  const std::array<std::filesystem::path, 8U> phase13_files{
      source_root / "include/bfnew/selected_region_index.hpp",
      source_root / "include/bfnew/batch_plan.hpp",
      source_root / "include/bfnew/batch_layout.hpp",
      source_root / "include/bfnew/batch_workspace.hpp",
      source_root / "src/selected_region_index.cpp",
      source_root / "src/batch_plan.cpp",
      source_root / "src/batch_layout.cpp",
      source_root / "src/batch_workspace.cpp",
  };
  const std::array<std::string_view, 7U> forbidden{
      "#include <hip/",
      "hipLaunchKernelGGL",
      "__global__",
      "cooperative_groups::",
      "run_batch(",
      "run_batched_",
      "GpuSsspEngine",
  };
  bool inspected_any = false;
  for (const std::filesystem::path& path : phase13_files) {
    const std::string source = read_file(path);
    if (source.empty()) {
      continue;
    }
    inspected_any = true;
    for (const std::string_view token : forbidden) {
      expect(
          source.find(token) == std::string::npos,
          "Phase 13 host planning sources contain no HIP or batched SSSP execution");
    }
  }
  expect(inspected_any, "the Phase 13 structural guard inspected host sources");
}

}  // namespace

int main() {
  test_policy_and_fraction_contract();
  test_golden_planner_and_features();
  test_family_once_only_singletons_and_ties();
  test_device_layout_golden_and_run_proof();
  test_phase14_scope_guard();

  if (failures != 0) {
    std::cerr << failures << " Phase 13 planner/layout check(s) failed\n";
    return 1;
  }
  std::cout
      << "Phase 13 bounded planner/layout goldens passed; no HIP/GPU batch "
         "execution was performed.\n";
  return 0;
}
