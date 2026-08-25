#include "bfnew/graph.hpp"
#include "bfnew/query.hpp"
#include "bfnew/spatial.hpp"
#include "bfnew/types.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
  using namespace bfnew;

  const std::vector<VertexMetadata> vertices{
      VertexMetadata::located(0, 0, ResourceClassId{1U}),
      VertexMetadata::located(1, 0, ResourceClassId{1U}),
      VertexMetadata::located(2, 0, ResourceClassId{2U}),
      VertexMetadata::located(3, 0, ResourceClassId{2U}),
      VertexMetadata::unlocated(ResourceClassId{3U}),
  };
  const PhysicalProvenance provenance{
      provenance_domain::synthetic, provenance_kind::synthetic_edge, 7U};
  const std::vector<EdgeInputRecord> edges{
      {VertexId{0U}, VertexId{1U}, 1.0F, provenance},
      {VertexId{1U}, VertexId{2U}, 2.0F, provenance},
      {VertexId{2U}, VertexId{3U}, 3.0F, provenance},
      {VertexId{4U}, VertexId{0U}, 4.0F, provenance},
  };

  const InputGraph input(vertices, edges);
  const UniformGridPartitioner partitioner(SpatialOrderConfig{0, 0, 2U, 1U});
  const PartitionedGraph partitioned = partitioner.partition(input);
  assert(validate_weighted_graph(partitioned.graph).ok());
  assert(validate_tile_directory(partitioned.graph, partitioned.tiles).ok());

  const auto map = partitioned.graph.old_to_new();
  const std::array source_terminals{map[0U], map[0U]};
  const std::array target_terminals{map[3U], map[2U], map[3U]};
  const RouteQuery query = make_route_query(
      QueryId{9U}, partitioned.graph, source_terminals, target_terminals, 0U);
  assert(validate_route_query(partitioned.graph, query).ok());
  assert(query.source_terminals.size() == 2U);
  assert(query.sources.size() == 1U);
  assert(query.target_terminals.size() == 3U);
  assert(query.targets.size() == 2U);
  assert(query.selected_tiles.size() == 2U);
  assert(estimate_selected_vertex_count(partitioned.graph, query.selected_tiles) == 4U);
  assert(estimate_selected_edge_count(partitioned.graph, query.selected_tiles) == 3U);

  const InducedQueryGraph induced =
      build_induced_query_graph(partitioned.graph, query);
  assert(induced.graph.vertex_count() == 4U);
  assert(induced.graph.edge_count() == 3U);
  assert(induced.sources.size() == 1U);
  assert(induced.targets.size() == 2U);
  assert(validate_weighted_graph(induced.graph).ok());

  const std::array spill_source{map[4U]};
  const std::array located_target{map[0U]};
  const RouteQuery spill_query = make_route_query(
      QueryId{10U}, partitioned.graph, spill_source, located_target, 0U);
  assert(spill_query.selected_tiles.size() == 2U);
  assert(spill_query.selected_tiles.back() == partitioned.tiles.spill_tile());

  return 0;
}
