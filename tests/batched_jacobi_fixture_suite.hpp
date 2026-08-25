#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/batch_plan.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/selected_region_index.hpp"
#include "bfnew/spatial.hpp"
#include "jacobi_fixture_suite.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bfnew::test {

struct BatchedJacobiFixture {
  PartitionedGraph partitioned;
  TileRunLayout64 tile_runs;
  DeviceGraphLayout32 device_graph;
  std::vector<RouteQuery> queries;
};

struct PreparedBatchedJacobiFixture {
  std::vector<BatchQueryFeatures> features;
  BatchPlan plan;
  BatchDeviceDescription description;
};

[[nodiscard]] inline RouteQuery with_exact_selected_tiles(
    RouteQuery query,
    std::vector<TileId> selected_tiles,
    const WeightedGraph& graph) {
  std::ranges::sort(selected_tiles);
  selected_tiles.erase(
      std::unique(selected_tiles.begin(), selected_tiles.end()),
      selected_tiles.end());
  query.selected_tiles = std::move(selected_tiles);
  if (!validate_route_query(graph, query).ok()) {
    throw std::logic_error{"invalid batched Jacobi fixture query"};
  }
  return query;
}

// Five lanes intentionally finish at different times:
//   1400: no first-round decrease and its target is already a source;
//   1401: one decrease round, then the terminal no-change round;
//   1402: a five-edge chain plus the terminal no-change round;
//   1403: no decrease and an unreachable admitted target;
//   1404: two independent sources and a multi-round completion.
// The tile-1 -> tile-2 run is in the union but has no admitting query lane,
// while tile-0 -> tile-2 is admitted only by lane 1404.
[[nodiscard]] inline BatchedJacobiFixture
make_mixed_duration_batched_jacobi_fixture() {
  const ResourceClassId resource{1U};
  const std::array<std::int32_t, 10U> x_coordinates{
      0, 1, 2, 10, 11, 12, 20, 21, 22, 23};
  std::vector<VertexMetadata> vertices;
  vertices.reserve(x_coordinates.size());
  for (const std::int32_t x : x_coordinates) {
    vertices.push_back(VertexMetadata::located(x, 0, resource));
  }
  InputGraph input{
      std::move(vertices),
      {
          jacobi_edge(0U, 1U, 1.0F, 14000U),
          jacobi_edge(1U, 2U, 1.0F, 14001U),
          jacobi_edge(2U, 3U, 1.0F, 14002U),
          jacobi_edge(3U, 4U, 1.0F, 14003U),
          jacobi_edge(4U, 5U, 1.0F, 14004U),
          jacobi_edge(5U, 6U, 1.0F, 14005U),
          jacobi_edge(6U, 7U, 1.0F, 14006U),
          jacobi_edge(7U, 8U, 1.0F, 14007U),
          jacobi_edge(8U, 9U, 1.0F, 14008U),
          jacobi_edge(1U, 7U, 2.0F, 14009U),
      },
  };
  const UniformGridPartitioner partitioner{
      SpatialOrderConfig{0, 0, 10U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const WeightedGraph& graph = partitioned.graph;
  const auto old_to_new = graph.old_to_new();
  const auto owners = graph.owner_tiles();
  const auto owner_of_old = [&](const std::size_t old_vertex) {
    return owners[old_to_new[old_vertex].value()];
  };
  const TileId tile_zero = owner_of_old(0U);
  const TileId tile_one = owner_of_old(3U);
  const TileId tile_two = owner_of_old(6U);

  std::vector<RouteQuery> queries;
  const std::array q1400_sources{old_to_new[2U]};
  const std::array q1400_targets{old_to_new[2U]};
  queries.push_back(with_exact_selected_tiles(
      make_route_query(
          QueryId{1400U}, graph, q1400_sources, q1400_targets),
      {tile_zero},
      graph));

  const std::array q1401_sources{old_to_new[1U]};
  const std::array q1401_targets{old_to_new[2U]};
  queries.push_back(with_exact_selected_tiles(
      make_route_query(
          QueryId{1401U}, graph, q1401_sources, q1401_targets),
      {tile_zero},
      graph));

  const std::array q1402_sources{old_to_new[0U]};
  const std::array q1402_targets{old_to_new[5U]};
  queries.push_back(with_exact_selected_tiles(
      make_route_query(
          QueryId{1402U}, graph, q1402_sources, q1402_targets),
      {tile_zero, tile_one},
      graph));

  const std::array q1403_sources{old_to_new[5U]};
  const std::array q1403_targets{old_to_new[3U]};
  queries.push_back(with_exact_selected_tiles(
      make_route_query(
          QueryId{1403U}, graph, q1403_sources, q1403_targets),
      {tile_one},
      graph));

  const std::array q1404_sources{old_to_new[0U], old_to_new[6U]};
  const std::array q1404_targets{old_to_new[8U]};
  queries.push_back(with_exact_selected_tiles(
      make_route_query(
          QueryId{1404U}, graph, q1404_sources, q1404_targets),
      {tile_zero, tile_two},
      graph));

  TileRunLayout64 tile_runs = build_tile_run_layout(graph);
  DeviceGraphLayout32 device_graph =
      build_device_graph_layout32(graph, tile_runs);
  return BatchedJacobiFixture{
      std::move(partitioned),
      std::move(tile_runs),
      std::move(device_graph),
      std::move(queries),
  };
}

[[nodiscard]] inline PreparedBatchedJacobiFixture prepare_batched_fixture(
    const BatchedJacobiFixture& fixture,
    const std::span<const RouteQuery> queries,
    const std::uint32_t lane_width,
    const BatchRunRepresentation representation) {
  const WeightedGraph& graph = fixture.partitioned.graph;
  const SelectedRegionIndex selected_regions{graph, fixture.tile_runs};
  PreparedBatchedJacobiFixture prepared;
  prepared.features =
      make_batch_query_features(graph, selected_regions, queries);
  BatchPlannerPolicy policy;
  policy.lane_width = lane_width;
  policy.minimum_jaccard_numerator = 0U;
  policy.minimum_jaccard_denominator = 1U;
  policy.maximum_union_inflation_numerator = 4U;
  policy.maximum_union_inflation_denominator = 1U;
  prepared.plan =
      make_overlapping_batch_plan(selected_regions, prepared.features, policy);
  if (prepared.plan.batches.size() != 1U) {
    throw std::logic_error{"batched Jacobi fixture did not form one batch"};
  }
  prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      queries,
      prepared.features,
      prepared.plan.batches.front(),
      representation,
      prepared.description);
  return prepared;
}

[[nodiscard]] inline std::size_t lane_for_query(
    const BatchPlanEntry& batch,
    const QueryId query_id) {
  for (std::size_t lane = 0U; lane < batch.lane_width; ++lane) {
    if ((batch.valid_lane_mask & (LaneMask{1U} << lane)) != 0U &&
        batch.query_ids_by_lane[lane] == query_id) {
      return lane;
    }
  }
  throw std::logic_error{"batched Jacobi fixture query lane is missing"};
}

}  // namespace bfnew::test
