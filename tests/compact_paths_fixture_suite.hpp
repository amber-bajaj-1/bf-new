#pragma once

#include "bfnew/query.hpp"
#include "bfnew/spatial.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace bfnew::test {

inline constexpr QueryId phase18_cycle_query_id{1800U};
inline constexpr QueryId phase18_tie_query_id{1801U};
inline constexpr QueryId phase18_bounded_query_id{1802U};
inline constexpr QueryId phase18_multisource_query_id{1803U};
inline constexpr QueryId phase18_unreachable_query_id{1804U};

struct CompactPathFixture {
  PartitionedGraph partitioned;
  RouteQuery cycle_query;
  RouteQuery tie_query;
  RouteQuery bounded_query;
  RouteQuery multisource_query;
  RouteQuery unreachable_query;
};

[[nodiscard]] inline EdgeInputRecord phase18_edge(
    const std::uint32_t source,
    const std::uint32_t destination,
    const float weight,
    const std::uint64_t record) {
  return EdgeInputRecord{
      VertexId{source},
      VertexId{destination},
      weight,
      PhysicalProvenance{
          provenance_domain::synthetic,
          provenance_kind::synthetic_edge,
          record},
  };
}

[[nodiscard]] inline CompactPathFixture make_compact_path_fixture() {
  const ResourceClassId resource{1U};
  const std::vector<VertexMetadata> vertices{
      // Zero-cycle/backtracking component (old 0..3).  For target old 2,
      // tight edge 0->1 has a lower stable ID than source edge 3->1, but
      // following it reaches the on-path 1->0 cycle and must backtrack.
      VertexMetadata::located(10, 0, resource),
      VertexMetadata::located(20, 0, resource),
      VertexMetadata::located(30, 0, resource),
      VertexMetadata::located(0, 0, resource),

      // Equal-cost alternatives (old 4..7).  The old-5 branch owns the
      // smaller stable incoming edge ID at the target, and its target edge has
      // an equal-cost parallel copy with a larger stable ID.
      VertexMetadata::located(40, 0, resource),
      VertexMetadata::located(50, 0, resource),
      VertexMetadata::located(50, 0, resource),
      VertexMetadata::located(60, 0, resource),

      // Bounded path (old 8..11).  The y=20 detour is globally cheaper but
      // is outside the query's zero-padding terminal rectangle.
      VertexMetadata::located(80, 0, resource),
      VertexMetadata::located(100, 0, resource),
      VertexMetadata::located(90, 0, resource),
      VertexMetadata::located(90, 20, resource),

      // Duplicate terminals and two sources (old 12..14).
      VertexMetadata::located(120, 0, resource),
      VertexMetadata::located(121, 0, resource),
      VertexMetadata::located(140, 0, resource),

      // Disconnected target for an explicit terminal failure (old 15).
      VertexMetadata::located(160, 0, resource),
  };

  const InputGraph input{
      vertices,
      {
          phase18_edge(0U, 1U, 0.0F, 18'000U),
          phase18_edge(1U, 0U, 0.0F, 18'001U),
          phase18_edge(1U, 2U, 1.0F, 18'002U),
          phase18_edge(3U, 1U, 1.0F, 18'003U),

          phase18_edge(4U, 5U, 0.5F, 18'004U),
          phase18_edge(4U, 6U, 0.5F, 18'005U),
          phase18_edge(5U, 7U, 0.5F, 18'006U),
          phase18_edge(6U, 7U, 0.5F, 18'007U),
          phase18_edge(5U, 7U, 0.5F, 18'014U),

          phase18_edge(8U, 10U, 2.0F, 18'008U),
          phase18_edge(10U, 9U, 2.0F, 18'009U),
          phase18_edge(8U, 11U, 0.5F, 18'010U),
          phase18_edge(11U, 9U, 0.5F, 18'011U),

          phase18_edge(12U, 14U, 3.0F, 18'012U),
          phase18_edge(13U, 14U, 1.0F, 18'013U),
      }};

  const UniformGridPartitioner partitioner{
      SpatialOrderConfig{0, 0, 10U, 10U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const WeightedGraph& graph = partitioned.graph;
  const std::span<const VertexId> map = graph.old_to_new();

  const auto query = [&](const QueryId id,
                         const std::span<const VertexId> sources,
                         const std::span<const VertexId> targets) {
    return make_route_query(id, graph, sources, targets);
  };

  const std::array cycle_sources{map[3U]};
  const std::array cycle_targets{map[2U]};
  const std::array tie_sources{map[4U]};
  const std::array tie_targets{map[7U]};
  const std::array bounded_sources{map[8U]};
  const std::array bounded_targets{map[9U]};
  const std::array multi_sources{map[12U], map[13U], map[12U]};
  const std::array multi_targets{map[14U], map[14U], map[12U]};
  const std::array unreachable_sources{map[12U]};
  const std::array unreachable_targets{map[15U]};

  RouteQuery cycle = query(phase18_cycle_query_id, cycle_sources, cycle_targets);
  RouteQuery tie = query(phase18_tie_query_id, tie_sources, tie_targets);
  RouteQuery bounded =
      query(phase18_bounded_query_id, bounded_sources, bounded_targets);
  RouteQuery multi =
      query(phase18_multisource_query_id, multi_sources, multi_targets);
  RouteQuery unreachable = query(
      phase18_unreachable_query_id,
      unreachable_sources,
      unreachable_targets);

  return CompactPathFixture{
      std::move(partitioned),
      std::move(cycle),
      std::move(tie),
      std::move(bounded),
      std::move(multi),
      std::move(unreachable),
  };
}

}  // namespace bfnew::test
