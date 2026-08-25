#pragma once

#include "dense_fixture_suite.hpp"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace bfnew::test {

[[nodiscard]] inline JacobiFixtureCase make_expanding_grid_frontier_fixture() {
  constexpr std::uint32_t side = 5U;
  constexpr std::uint32_t vertex_count = side * side;
  std::vector<EdgeInputRecord> edges;
  for (std::uint32_t y = 0U; y < side; ++y) {
    for (std::uint32_t x = 0U; x < side; ++x) {
      const std::uint32_t source = y * side + x;
      if (x + 1U < side) {
        edges.push_back(jacobi_edge(
            source, source + 1U, 0.5F, 4000U + edges.size()));
      }
      if (y + 1U < side) {
        edges.push_back(jacobi_edge(
            source, source + side, 0.5F, 4000U + edges.size()));
      }
    }
  }
  InputGraph input{located_vertices(vertex_count, side), std::move(edges)};
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 8U, 8U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[0U]};
  const std::array targets{
      partitioned.graph.old_to_new()[vertex_count - 1U]};
  RouteQuery query = make_full_region_query(
      QueryId{1200U}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "frontier_rapidly_expanding_grid",
      std::move(partitioned),
      std::move(query),
  };
}

[[nodiscard]] inline JacobiFixtureCase
make_repeated_improvement_frontier_fixture() {
  std::vector<EdgeInputRecord> edges{
      jacobi_edge(0U, 1U, 0.0F, 4100U),
      jacobi_edge(0U, 2U, 0.0F, 4101U),
      jacobi_edge(0U, 3U, 0.0F, 4102U),
      jacobi_edge(1U, 4U, 3.0F, 4103U),
      jacobi_edge(2U, 4U, 2.0F, 4104U),
      jacobi_edge(3U, 4U, 1.0F, 4105U),
      jacobi_edge(4U, 5U, 0.5F, 4106U),
  };
  InputGraph input{located_vertices(6U), std::move(edges)};
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 16U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[0U]};
  const std::array targets{partitioned.graph.old_to_new()[5U]};
  RouteQuery query = make_full_region_query(
      QueryId{1201U}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "frontier_repeated_improvement_single_claim",
      std::move(partitioned),
      std::move(query),
  };
}

[[nodiscard]] inline std::vector<JacobiFixtureCase>
make_frontier_fixture_suite() {
  std::vector<JacobiFixtureCase> fixtures = make_dense_fixture_suite();
  fixtures.reserve(fixtures.size() + 2U);
  fixtures.push_back(make_expanding_grid_frontier_fixture());
  fixtures.push_back(make_repeated_improvement_frontier_fixture());
  return fixtures;
}

}  // namespace bfnew::test
