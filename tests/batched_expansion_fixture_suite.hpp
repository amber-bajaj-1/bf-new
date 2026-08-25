#pragma once

#include "bfnew/device_layout.hpp"
#include "bfnew/query.hpp"
#include "bfnew/spatial.hpp"
#include "jacobi_fixture_suite.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bfnew::test {

// A deliberately tiny Phase 17 corpus.  Every route owns a disjoint x range
// so tests can select exactly the reachability shape they need without an
// unrelated vertex entering its initial terminal rectangle.
struct BatchedExpansionFixture {
  PartitionedGraph partitioned;
  TileDirectory directory;
  TileRunLayout64 tile_runs;
  DeviceGraphLayout32 device_graph;
  std::vector<RouteQuery> queries;
};

inline constexpr QueryId phase17_immediate_query_id{1700U};
inline constexpr QueryId phase17_two_ring_query_id{1701U};
inline constexpr QueryId phase17_long_edge_query_id{1702U};
inline constexpr QueryId phase17_spill_query_id{1703U};
inline constexpr QueryId phase17_unreachable_query_id{1704U};
inline constexpr QueryId phase17_second_miss_query_id{1705U};
inline constexpr QueryId phase17_multi_source_query_id{1706U};

[[nodiscard]] inline BatchedExpansionFixture
make_batched_expansion_fixture() {
  const ResourceClassId resource{1U};
  const std::vector<VertexMetadata> vertices{
      // Immediate, fully admitted three-vertex chain (old 0..2).
      VertexMetadata::located(40, 0, resource),
      VertexMetadata::located(50, 0, resource),
      VertexMetadata::located(60, 0, resource),

      // Two-ring northern detour (old 3..6).
      VertexMetadata::located(0, 0, resource),
      VertexMetadata::located(20, 0, resource),
      VertexMetadata::located(0, 10, resource),
      VertexMetadata::located(10, 20, resource),

      // A cross-tile jump to a fourth-ring intermediate (old 7..9).
      VertexMetadata::located(80, 0, resource),
      VertexMetadata::located(100, 0, resource),
      VertexMetadata::located(80, 40, resource),

      // Located source, spill target, and a second-ring intermediate
      // (old 10..12).
      VertexMetadata::located(120, 0, resource),
      VertexMetadata::located(120, 20, resource),
      VertexMetadata::unlocated(resource),

      // Globally unreachable terminals (old 13..14).
      VertexMetadata::located(160, 0, resource),
      VertexMetadata::located(180, 0, resource),

      // A separate three-ring southern detour, useful for a second failed
      // lane whose expanded shape differs from the northern route (old 15..18).
      VertexMetadata::located(220, 0, resource),
      VertexMetadata::located(240, 0, resource),
      VertexMetadata::located(220, -10, resource),
      VertexMetadata::located(230, -30, resource),

      // Otherwise-isolated tiles make every scheduled ring a strict region
      // growth step before the long/spill/southern path is finally admitted
      // (old 19..23).
      VertexMetadata::located(80, 10, resource),
      VertexMetadata::located(80, 20, resource),
      VertexMetadata::located(80, 30, resource),
      VertexMetadata::located(120, 10, resource),
      VertexMetadata::located(220, -20, resource),

      // Second source in the northern route's original source tile (old 24).
      VertexMetadata::located(1, 0, resource),
  };

  InputGraph input{
      vertices,
      {
          jacobi_edge(0U, 1U, 1.0F, 17000U),
          jacobi_edge(1U, 2U, 1.0F, 17001U),
          jacobi_edge(3U, 5U, 1.0F, 17002U),
          jacobi_edge(5U, 6U, 1.0F, 17003U),
          jacobi_edge(6U, 4U, 1.0F, 17004U),
          jacobi_edge(7U, 9U, 2.0F, 17005U),
          jacobi_edge(9U, 8U, 3.0F, 17006U),
          jacobi_edge(10U, 11U, 4.0F, 17007U),
          jacobi_edge(11U, 12U, 5.0F, 17008U),
          jacobi_edge(15U, 17U, 1.0F, 17009U),
          jacobi_edge(17U, 18U, 1.0F, 17010U),
          jacobi_edge(18U, 16U, 1.0F, 17011U),
          jacobi_edge(24U, 5U, 0.5F, 17012U),
      }};
  const UniformGridPartitioner partitioner{
      SpatialOrderConfig{0, 0, 10U, 10U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const WeightedGraph& graph = partitioned.graph;
  const std::span<const VertexId> old_to_new = graph.old_to_new();

  const auto make_query = [&](const QueryId id,
                              const std::uint32_t old_source,
                              const std::uint32_t old_target) {
    const std::array sources{old_to_new[old_source]};
    const std::array targets{old_to_new[old_target]};
    return make_route_query(id, graph, sources, targets);
  };

  std::vector<RouteQuery> queries;
  queries.push_back(make_query(phase17_immediate_query_id, 0U, 2U));
  queries.push_back(make_query(phase17_two_ring_query_id, 3U, 4U));
  queries.push_back(make_query(phase17_long_edge_query_id, 7U, 8U));
  queries.push_back(make_query(phase17_spill_query_id, 10U, 12U));
  queries.push_back(make_query(phase17_unreachable_query_id, 13U, 14U));
  queries.push_back(make_query(phase17_second_miss_query_id, 15U, 16U));
  {
    const std::array sources{old_to_new[3U], old_to_new[24U]};
    const std::array targets{old_to_new[4U]};
    queries.push_back(make_route_query(
        phase17_multi_source_query_id, graph, sources, targets));
  }

  for (const RouteQuery& query : queries) {
    if (!validate_route_query(graph, query).ok()) {
      throw std::logic_error{"invalid Phase 17 expansion fixture query"};
    }
  }

  TileDirectory directory = build_tile_directory(graph);
  TileRunLayout64 tile_runs = build_tile_run_layout(graph);
  DeviceGraphLayout32 device_graph =
      build_device_graph_layout32(graph, tile_runs);
  return BatchedExpansionFixture{
      std::move(partitioned),
      std::move(directory),
      std::move(tile_runs),
      std::move(device_graph),
      std::move(queries)};
}

[[nodiscard]] inline const RouteQuery& expansion_query(
    const BatchedExpansionFixture& fixture,
    const QueryId query_id) {
  for (const RouteQuery& query : fixture.queries) {
    if (query.query_id == query_id) {
      return query;
    }
  }
  throw std::logic_error{"Phase 17 fixture query is missing"};
}

}  // namespace bfnew::test
