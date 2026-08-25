#pragma once

#include "bfnew/graph.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/query.hpp"
#include "bfnew/spatial.hpp"
#include "graph_fixtures.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bfnew::test {

enum class JacobiComparisonPolicy : std::uint8_t {
  bitwise,
  four_ulps,
};

struct JacobiFixtureCase {
  std::string name;
  PartitionedGraph partitioned;
  RouteQuery query;
  JacobiComparisonPolicy comparison{JacobiComparisonPolicy::bitwise};
  std::uint64_t expected_rounds{};
  LaneMask expected_reached_mask{1U};
  LaneMask expected_miss_mask{};
  std::optional<float> expected_bounded_primary_target;
  std::optional<float> expected_unbounded_primary_target;

  JacobiFixtureCase(
      std::string case_name,
      PartitionedGraph case_partitioned,
      RouteQuery case_query,
      const JacobiComparisonPolicy case_comparison =
          JacobiComparisonPolicy::bitwise,
      const std::uint64_t case_expected_rounds = 0U,
      const LaneMask case_expected_reached_mask = 1U,
      const LaneMask case_expected_miss_mask = 0U,
      const std::optional<float> case_expected_bounded_primary_target =
          std::nullopt,
      const std::optional<float> case_expected_unbounded_primary_target =
          std::nullopt)
      : name{std::move(case_name)},
        partitioned{std::move(case_partitioned)},
        query{std::move(case_query)},
        comparison{case_comparison},
        expected_rounds{case_expected_rounds},
        expected_reached_mask{case_expected_reached_mask},
        expected_miss_mask{case_expected_miss_mask},
        expected_bounded_primary_target{
            case_expected_bounded_primary_target},
        expected_unbounded_primary_target{
            case_expected_unbounded_primary_target} {}
};

[[nodiscard]] inline RouteQuery make_full_region_query(
    const QueryId query_id,
    const WeightedGraph& graph,
    const std::span<const VertexId> sources,
    const std::span<const VertexId> targets) {
  RouteQuery query = make_route_query(query_id, graph, sources, targets, 0U);
  query.selected_tiles.clear();
  query.selected_tiles.reserve(graph.tile_coordinates().size());
  for (std::size_t tile = 0U; tile < graph.tile_coordinates().size(); ++tile) {
    query.selected_tiles.push_back(checked_id<TileId>(tile));
  }
  if (!validate_route_query(graph, query).ok()) {
    throw std::logic_error{"full-region Jacobi fixture query is invalid"};
  }
  return query;
}

[[nodiscard]] inline std::vector<VertexMetadata> located_vertices(
    const std::size_t count,
    const std::int32_t y = 0) {
  std::vector<VertexMetadata> vertices;
  vertices.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    vertices.push_back(VertexMetadata::located(
        static_cast<std::int32_t>(index),
        y,
        ResourceClassId{1U}));
  }
  return vertices;
}

[[nodiscard]] inline EdgeInputRecord jacobi_edge(
    const std::uint32_t source,
    const std::uint32_t destination,
    const float weight,
    const std::uint64_t record) {
  return EdgeInputRecord{
      VertexId{source},
      VertexId{destination},
      weight,
      synthetic_provenance(record),
  };
}

[[nodiscard]] inline JacobiFixtureCase make_phase5_core_jacobi_fixture() {
  const CoreWeightedFixture fixture = make_core_weighted_fixture();
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 64U, 64U}};
  PartitionedGraph partitioned = partitioner.partition(fixture.graph);
  std::vector<VertexId> sources;
  std::vector<VertexId> targets;
  for (const VertexId source : fixture.sources) {
    sources.push_back(partitioned.graph.old_to_new()[source.value()]);
  }
  for (const VertexId target : fixture.targets) {
    targets.push_back(partitioned.graph.old_to_new()[target.value()]);
  }
  RouteQuery query = make_full_region_query(
      QueryId{900U}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "phase5_core",
      std::move(partitioned),
      std::move(query),
  };
}

[[nodiscard]] inline JacobiFixtureCase make_phase5_spatial_jacobi_fixture() {
  const SpatialReorderFixture fixture = make_spatial_reorder_fixture();
  const UniformGridPartitioner partitioner{fixture.config};
  PartitionedGraph partitioned = partitioner.partition(fixture.graph);
  const std::array sources{partitioned.graph.old_to_new()[1U]};
  const std::array targets{
      partitioned.graph.old_to_new()[0U],
      partitioned.graph.old_to_new()[2U],
      partitioned.graph.old_to_new()[9U],
  };
  RouteQuery query = make_full_region_query(
      QueryId{901U}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "phase5_spatial",
      std::move(partitioned),
      std::move(query),
  };
}

[[nodiscard]] inline JacobiFixtureCase make_phase5_tie_jacobi_fixture() {
  InputGraph input{
      located_vertices(4U),
      {
          jacobi_edge(0U, 1U, 0.5F, 200U),
          jacobi_edge(0U, 2U, 0.5F, 201U),
          jacobi_edge(1U, 3U, 0.5F, 202U),
          jacobi_edge(2U, 3U, 0.5F, 203U),
      },
  };
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 8U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[0U]};
  const std::array targets{partitioned.graph.old_to_new()[3U]};
  RouteQuery query = make_full_region_query(
      QueryId{902U}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "phase5_equal_path_tie",
      std::move(partitioned),
      std::move(query),
  };
}

[[nodiscard]] inline JacobiFixtureCase make_phase5_zero_cycle_jacobi_fixture() {
  InputGraph input{
      located_vertices(4U),
      {
          jacobi_edge(0U, 1U, 0.0F, 300U),
          jacobi_edge(1U, 0U, 0.0F, 301U),
          jacobi_edge(1U, 2U, 1.0F, 302U),
          jacobi_edge(3U, 1U, 1.0F, 303U),
      },
  };
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 8U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[3U]};
  const std::array targets{partitioned.graph.old_to_new()[2U]};
  RouteQuery query = make_full_region_query(
      QueryId{903U}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "phase5_zero_weight_cycle",
      std::move(partitioned),
      std::move(query),
  };
}

[[nodiscard]] inline JacobiFixtureCase make_long_chain_jacobi_fixture() {
  constexpr std::uint32_t vertex_count = 33U;
  std::vector<EdgeInputRecord> edges;
  edges.reserve(vertex_count - 1U);
  for (std::uint32_t source = 0U; source + 1U < vertex_count; ++source) {
    edges.push_back(jacobi_edge(source, source + 1U, 0.25F, 400U + source));
  }
  InputGraph input{located_vertices(vertex_count), std::move(edges)};
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 64U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[0U]};
  // The target is reached in round one, but the selected tile contains the
  // complete chain and therefore needs 32 change rounds plus one no-change scan.
  const std::array targets{partitioned.graph.old_to_new()[1U]};
  RouteQuery query = make_full_region_query(
      QueryId{904U}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "long_chain_no_target_early_stop",
      std::move(partitioned),
      std::move(query),
      JacobiComparisonPolicy::bitwise,
      33U,
  };
}

[[nodiscard]] inline JacobiFixtureCase make_disconnected_jacobi_fixture() {
  InputGraph input{
      located_vertices(5U),
      {jacobi_edge(0U, 1U, 0.5F, 500U)},
  };
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 8U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[0U]};
  const std::array targets{partitioned.graph.old_to_new()[1U]};
  RouteQuery query = make_full_region_query(
      QueryId{905U}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "disconnected_non_targets",
      std::move(partitioned),
      std::move(query),
      JacobiComparisonPolicy::bitwise,
      2U,
  };
}

[[nodiscard]] inline JacobiFixtureCase make_parallel_jacobi_fixture() {
  InputGraph input{
      located_vertices(3U),
      {
          jacobi_edge(0U, 1U, 1.0F, 600U),
          jacobi_edge(0U, 1U, 0.5F, 601U),
          jacobi_edge(1U, 2U, 0.25F, 602U),
      },
  };
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 8U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[0U]};
  const std::array targets{partitioned.graph.old_to_new()[2U]};
  RouteQuery query = make_full_region_query(
      QueryId{906U}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "parallel_edges_short_chain",
      std::move(partitioned),
      std::move(query),
      JacobiComparisonPolicy::bitwise,
      3U,
  };
}

[[nodiscard]] inline JacobiFixtureCase make_multi_source_jacobi_fixture() {
  InputGraph input{
      located_vertices(4U),
      {
          jacobi_edge(0U, 2U, 2.0F, 700U),
          jacobi_edge(1U, 2U, 0.5F, 701U),
          jacobi_edge(2U, 3U, 0.25F, 702U),
      },
  };
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 8U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array source_terminals{
      partitioned.graph.old_to_new()[0U],
      partitioned.graph.old_to_new()[1U],
      partitioned.graph.old_to_new()[1U],
  };
  const std::array targets{partitioned.graph.old_to_new()[3U]};
  RouteQuery query = make_full_region_query(
      QueryId{907U}, partitioned.graph, source_terminals, targets);
  return JacobiFixtureCase{
      "multiple_sources",
      std::move(partitioned),
      std::move(query),
      JacobiComparisonPolicy::bitwise,
      3U,
  };
}

[[nodiscard]] inline std::vector<VertexMetadata> box_vertices() {
  return {
      VertexMetadata::located(0, 0, ResourceClassId{1U}),  // source
      VertexMetadata::located(1, 0, ResourceClassId{1U}),  // admitted detour
      VertexMetadata::located(2, 0, ResourceClassId{1U}),  // target
      VertexMetadata::located(1, 1, ResourceClassId{1U}),  // outside box
  };
}

[[nodiscard]] inline JacobiFixtureCase make_cheaper_outside_box_fixture() {
  InputGraph input{
      box_vertices(),
      {
          jacobi_edge(0U, 1U, 2.0F, 800U),
          jacobi_edge(1U, 2U, 2.0F, 801U),
          jacobi_edge(0U, 3U, 0.25F, 802U),
          jacobi_edge(3U, 2U, 0.25F, 803U),
      },
  };
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 1U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[0U]};
  const std::array targets{partitioned.graph.old_to_new()[2U]};
  RouteQuery query = make_route_query(
      QueryId{908U}, partitioned.graph, sources, targets, 0U);
  return JacobiFixtureCase{
      "bounded_route_ignores_cheaper_outside_path",
      std::move(partitioned),
      std::move(query),
      JacobiComparisonPolicy::bitwise,
      3U,
      1U,
      0U,
      4.0F,
      0.5F,
  };
}

[[nodiscard]] inline JacobiFixtureCase make_box_miss_fixture() {
  InputGraph input{
      box_vertices(),
      {
          jacobi_edge(0U, 3U, 1.0F, 900U),
          jacobi_edge(3U, 2U, 1.0F, 901U),
      },
  };
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 1U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{partitioned.graph.old_to_new()[0U]};
  const std::array targets{partitioned.graph.old_to_new()[2U]};
  RouteQuery query = make_route_query(
      QueryId{909U}, partitioned.graph, sources, targets, 0U);
  return JacobiFixtureCase{
      "bounded_box_miss_with_global_path",
      std::move(partitioned),
      std::move(query),
      JacobiComparisonPolicy::bitwise,
      1U,
      0U,
      1U,
      std::numeric_limits<float>::infinity(),
      2.0F,
  };
}

[[nodiscard]] inline std::uint32_t xorshift32(std::uint32_t& state) noexcept {
  state ^= state << 13U;
  state ^= state >> 17U;
  state ^= state << 5U;
  return state;
}

[[nodiscard]] inline float random_general_weight(std::uint32_t& state) noexcept {
  const std::uint32_t bits =
      0x3f000000U | (xorshift32(state) & 0x007fffffU);
  return std::bit_cast<float>(bits);
}

[[nodiscard]] inline JacobiFixtureCase make_random_jacobi_fixture(
    const std::uint32_t seed,
    const std::uint32_t ordinal) {
  constexpr std::uint32_t vertex_count = 16U;
  constexpr std::uint32_t edge_count = 48U;
  std::vector<VertexMetadata> vertices;
  vertices.reserve(vertex_count);
  for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex) {
    vertices.push_back(VertexMetadata::located(
        static_cast<std::int32_t>(vertex % 4U),
        static_cast<std::int32_t>(vertex / 4U),
        ResourceClassId{1U}));
  }

  std::uint32_t state = seed;
  std::vector<EdgeInputRecord> edges;
  edges.reserve(edge_count);
  for (std::uint32_t source = 0U; source + 1U < vertex_count; ++source) {
    edges.push_back(jacobi_edge(
        source,
        source + 1U,
        random_general_weight(state),
        1000U + static_cast<std::uint64_t>(ordinal) * 100U + source));
  }
  while (edges.size() < edge_count) {
    const std::uint32_t source = xorshift32(state) % vertex_count;
    const std::uint32_t destination = xorshift32(state) % vertex_count;
    edges.push_back(jacobi_edge(
        source,
        destination,
        random_general_weight(state),
        1000U + static_cast<std::uint64_t>(ordinal) * 100U + edges.size()));
  }

  InputGraph input{std::move(vertices), std::move(edges)};
  const UniformGridPartitioner partitioner{SpatialOrderConfig{0, 0, 1U, 1U}};
  PartitionedGraph partitioned = partitioner.partition(input);
  const std::array sources{
      partitioned.graph.old_to_new()[0U],
      partitioned.graph.old_to_new()[8U],
  };
  const std::array targets{partitioned.graph.old_to_new()[15U]};
  RouteQuery query = make_full_region_query(
      QueryId{1000U + ordinal}, partitioned.graph, sources, targets);
  return JacobiFixtureCase{
      "random_seed_" + std::to_string(seed),
      std::move(partitioned),
      std::move(query),
      JacobiComparisonPolicy::four_ulps,
  };
}

[[nodiscard]] inline std::vector<JacobiFixtureCase> make_jacobi_fixture_suite() {
  std::vector<JacobiFixtureCase> fixtures;
  fixtures.reserve(13U);
  fixtures.push_back(make_phase5_core_jacobi_fixture());
  fixtures.push_back(make_phase5_spatial_jacobi_fixture());
  fixtures.push_back(make_phase5_tie_jacobi_fixture());
  fixtures.push_back(make_phase5_zero_cycle_jacobi_fixture());
  fixtures.push_back(make_long_chain_jacobi_fixture());
  fixtures.push_back(make_disconnected_jacobi_fixture());
  fixtures.push_back(make_parallel_jacobi_fixture());
  fixtures.push_back(make_multi_source_jacobi_fixture());
  fixtures.push_back(make_cheaper_outside_box_fixture());
  fixtures.push_back(make_box_miss_fixture());
  fixtures.push_back(make_random_jacobi_fixture(0x9e3779b9U, 0U));
  fixtures.push_back(make_random_jacobi_fixture(0x243f6a88U, 1U));
  fixtures.push_back(make_random_jacobi_fixture(0xb7e15162U, 2U));
  return fixtures;
}

}  // namespace bfnew::test
