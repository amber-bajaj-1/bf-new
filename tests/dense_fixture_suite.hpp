#pragma once

#include "jacobi_fixture_suite.hpp"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace bfnew::test {

[[nodiscard]] inline JacobiFixtureCase make_high_fan_in_dense_fixture() {
  constexpr std::uint32_t intermediate_count = 12U;
  constexpr std::uint32_t hub = intermediate_count + 1U;
  std::vector<EdgeInputRecord> edges;
  edges.reserve(intermediate_count * 2U);
  for (std::uint32_t index = 0U; index < intermediate_count; ++index) {
    const std::uint32_t intermediate = index + 1U;
    edges.push_back(jacobi_edge(
        0U,
        intermediate,
        static_cast<float>((index % 4U) + 1U) * 0.25F,
        3000U + index));
    edges.push_back(jacobi_edge(
        intermediate,
        hub,
        static_cast<float>((intermediate_count - index) % 4U) * 0.25F,
        3100U + index));
  }
  InputGraph input{located_vertices(hub + 1U), std::move(edges)};
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 64U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[0U]};
  const std::array targets{partitioned.graph.old_to_new()[hub]};
  RouteQuery query = make_full_region_query(
      QueryId{1100U}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "dense_high_fan_in_contention",
      std::move(partitioned),
      std::move(query),
  };
}

[[nodiscard]] inline JacobiFixtureCase make_hub_fan_out_dense_fixture() {
  constexpr std::uint32_t vertex_count = 34U;
  constexpr std::uint32_t hub = 1U;
  std::vector<EdgeInputRecord> edges;
  edges.reserve(vertex_count - 1U);
  edges.push_back(jacobi_edge(0U, hub, 0.5F, 3200U));
  for (std::uint32_t destination = 2U; destination < vertex_count;
       ++destination) {
    edges.push_back(jacobi_edge(
        hub,
        destination,
        static_cast<float>((destination % 8U) + 1U) * 0.125F,
        3200U + destination));
  }
  InputGraph input{located_vertices(vertex_count), std::move(edges)};
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 64U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[0U]};
  const std::array targets{partitioned.graph.old_to_new()[vertex_count - 1U]};
  RouteQuery query = make_full_region_query(
      QueryId{1101U}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "dense_hub_high_fan_out",
      std::move(partitioned),
      std::move(query),
  };
}

[[nodiscard]] inline std::vector<JacobiFixtureCase> make_dense_fixture_suite() {
  std::vector<JacobiFixtureCase> fixtures = make_jacobi_fixture_suite();
  fixtures.reserve(fixtures.size() + 2U);
  fixtures.push_back(make_high_fan_in_dense_fixture());
  fixtures.push_back(make_hub_fan_out_dense_fixture());
  return fixtures;
}

}  // namespace bfnew::test
