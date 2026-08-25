#include "bfnew/graph.hpp"
#include "bfnew/spatial.hpp"
#include "graph_fixtures.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using bfnew::EdgeId;
using bfnew::EdgeOffset;
using bfnew::PartitionedGraph;
using bfnew::TileDirectory;
using bfnew::TileId;
using bfnew::VertexId;
using bfnew::WeightedGraph;

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

template <typename T>
  requires std::is_trivially_copyable_v<T>
[[nodiscard]] bool byte_equal(const std::span<const T> left, const std::span<const T> right) {
  return left.size() == right.size() &&
         std::ranges::equal(std::as_bytes(left), std::as_bytes(right));
}

template <typename T>
[[nodiscard]] std::span<const T> tile_group(
    const std::span<const EdgeOffset> offsets,
    const std::span<const T> values,
    const TileId tile) {
  const std::size_t begin = static_cast<std::size_t>(offsets[tile.value()]);
  const std::size_t end = static_cast<std::size_t>(offsets[tile.value() + 1U]);
  return values.subspan(begin, end - begin);
}

template <typename T>
[[nodiscard]] bool group_contains(const std::span<const T> group, const T value) {
  return std::ranges::find(group, value) != group.end();
}

[[nodiscard]] EdgeId edge_id_for_source_record(
    const WeightedGraph& graph,
    const std::uint64_t source_record) {
  for (std::size_t edge_id = 0U; edge_id < graph.edge_provenance().size(); ++edge_id) {
    if (graph.edge_provenance()[edge_id].source_record == source_record) {
      return EdgeId{static_cast<std::uint64_t>(edge_id)};
    }
  }
  throw std::logic_error{"synthetic source record is absent from the graph"};
}

void test_uniform_partitioner_and_directory_counts() {
  const bfnew::test::SpatialReorderFixture fixture =
      bfnew::test::make_spatial_reorder_fixture();
  const std::unique_ptr<bfnew::SpatialPartitioner> partitioner =
      std::make_unique<bfnew::UniformGridPartitioner>(fixture.config);
  const PartitionedGraph partitioned = partitioner->partition(fixture.graph);
  const WeightedGraph& graph = partitioned.graph;
  const TileDirectory& directory = partitioned.tiles;

  expect(graph.has_spatial_ordering(), "uniform partitioner spatially orders the graph");
  expect(directory.tile_count() == 4U, "directory contains three located tiles and spill");
  expect(directory.spill_tile() == TileId{3U},
         "directory identifies the last dense tile as spill");
  expect(bfnew::validate_tile_directory(graph, directory).ok(),
         "tile directory passes deep validation");

  expect(directory.internal_edge_ids().size() == 6U,
         "fixture has six source-owned internal edges");
  expect(directory.outgoing_cross_edge_ids().size() == 6U,
         "fixture has six source-owned outgoing cross edges");
  expect(directory.incoming_cross_edge_ids().size() == 6U,
         "each cross edge has one incoming metadata entry");
  expect(directory.halo_vertices().size() == 9U,
         "halo lists deduplicate remote endpoints per tile");

  std::vector<bool> source_classification(graph.edge_count(), false);
  for (const EdgeId edge_id : directory.internal_edge_ids()) {
    const std::size_t id = static_cast<std::size_t>(edge_id.value());
    expect(!source_classification[id], "internal edge is not source-classified twice");
    source_classification[id] = true;
  }
  for (const EdgeId edge_id : directory.outgoing_cross_edge_ids()) {
    const std::size_t id = static_cast<std::size_t>(edge_id.value());
    expect(!source_classification[id], "cross edge is not source-classified twice");
    source_classification[id] = true;
  }
  expect(std::ranges::all_of(
             source_classification, [](const bool classified) { return classified; }),
         "every graph edge is classified exactly once for its source owner");

  for (std::size_t tile = 0U; tile < directory.tile_count(); ++tile) {
    const TileId tile_id{static_cast<std::uint32_t>(tile)};
    const auto halos = tile_group(
        directory.halo_vertex_offsets(), directory.halo_vertices(), tile_id);
    for (const VertexId halo : halos) {
      expect(halo.value() < graph.vertex_count(),
             "halo endpoint resolves to a global reordered vertex");
      expect(graph.owner_tiles()[halo.value()] != tile_id,
             "halo endpoint is remote from the tile that references it");
    }
  }
}

void test_adjacent_long_and_spill_relationships() {
  const bfnew::test::SpatialReorderFixture fixture =
      bfnew::test::make_spatial_reorder_fixture();
  const bfnew::UniformGridPartitioner partitioner{fixture.config};
  const PartitionedGraph partitioned = partitioner.partition(fixture.graph);
  const WeightedGraph& graph = partitioned.graph;
  const TileDirectory& directory = partitioned.tiles;

  const auto tile_zero_neighbors = tile_group(
      directory.neighbor_tile_offsets(), directory.neighbor_tiles(), TileId{0U});
  const auto tile_one_neighbors = tile_group(
      directory.neighbor_tile_offsets(), directory.neighbor_tiles(), TileId{1U});
  const auto tile_two_neighbors = tile_group(
      directory.neighbor_tile_offsets(), directory.neighbor_tiles(), TileId{2U});
  const auto spill_neighbors = tile_group(
      directory.neighbor_tile_offsets(), directory.neighbor_tiles(), TileId{3U});
  expect(std::ranges::equal(tile_zero_neighbors, std::vector{TileId{1U}, TileId{3U}}),
         "tile zero has its geometric neighbor and actual spill neighbor");
  expect(std::ranges::equal(tile_one_neighbors, std::vector{TileId{0U}, TileId{2U}}),
         "middle located tile has symmetric geometric neighbors");
  expect(std::ranges::equal(tile_two_neighbors, std::vector{TileId{1U}}),
         "long-edge endpoint tile is not made a geometric neighbor");
  expect(std::ranges::equal(spill_neighbors, std::vector{TileId{0U}}),
         "spill adjacency comes only from actual incident edges");

  const EdgeId adjacent_edge = edge_id_for_source_record(graph, 101U);
  const EdgeId long_edge = edge_id_for_source_record(graph, 100U);
  const EdgeId to_spill_edge = edge_id_for_source_record(graph, 107U);
  const auto tile_zero_outgoing = tile_group(
      directory.outgoing_cross_edge_offsets(),
      directory.outgoing_cross_edge_ids(),
      TileId{0U});
  expect(group_contains(tile_zero_outgoing, adjacent_edge),
         "adjacent-tile edge appears in outgoing cross metadata");
  expect(group_contains(tile_zero_outgoing, long_edge),
         "tile-skipping edge appears in outgoing cross metadata");
  expect(group_contains(tile_zero_outgoing, to_spill_edge),
         "located-to-spill edge appears in outgoing cross metadata");

  const auto tile_one_incoming = tile_group(
      directory.incoming_cross_edge_offsets(),
      directory.incoming_cross_edge_ids(),
      TileId{1U});
  const auto tile_two_incoming = tile_group(
      directory.incoming_cross_edge_offsets(),
      directory.incoming_cross_edge_ids(),
      TileId{2U});
  const auto spill_incoming = tile_group(
      directory.incoming_cross_edge_offsets(),
      directory.incoming_cross_edge_ids(),
      TileId{3U});
  expect(group_contains(tile_one_incoming, adjacent_edge),
         "adjacent edge appears in destination-owned incoming metadata");
  expect(group_contains(tile_two_incoming, long_edge),
         "long edge appears in destination-owned incoming metadata");
  expect(group_contains(spill_incoming, to_spill_edge),
         "spill edge appears in spill-owned incoming metadata");

  const VertexId reordered_source = graph.old_to_new()[1U];
  const VertexId reordered_long_destination = graph.old_to_new()[0U];
  const auto tile_zero_halos = tile_group(
      directory.halo_vertex_offsets(), directory.halo_vertices(), TileId{0U});
  const auto tile_two_halos = tile_group(
      directory.halo_vertex_offsets(), directory.halo_vertices(), TileId{2U});
  expect(group_contains(tile_zero_halos, reordered_long_destination),
         "source tile halo references a long edge's remote destination");
  expect(group_contains(tile_two_halos, reordered_source),
         "destination tile halo references a long edge's remote source");
}

void test_diagonal_geometric_neighbors_without_edges() {
  const bfnew::ResourceClassId resource_class{1U};
  const bfnew::InputGraph input{
      {
          bfnew::VertexMetadata::located(0, 0, resource_class),
          bfnew::VertexMetadata::located(10, 10, resource_class),
      },
      {},
  };
  const bfnew::UniformGridPartitioner partitioner{
      bfnew::SpatialOrderConfig{0, 0, 10U, 10U}};
  const PartitionedGraph partitioned = partitioner.partition(input);
  const TileDirectory& directory = partitioned.tiles;
  const auto first_neighbors = tile_group(
      directory.neighbor_tile_offsets(), directory.neighbor_tiles(), TileId{0U});
  const auto second_neighbors = tile_group(
      directory.neighbor_tile_offsets(), directory.neighbor_tiles(), TileId{1U});
  const auto spill_neighbors = tile_group(
      directory.neighbor_tile_offsets(), directory.neighbor_tiles(), TileId{2U});
  expect(std::ranges::equal(first_neighbors, std::vector{TileId{1U}}) &&
             std::ranges::equal(second_neighbors, std::vector{TileId{0U}}),
         "diagonal tiles are symmetric Chebyshev neighbors without requiring an edge");
  expect(spill_neighbors.empty(), "spill tile has no synthetic geometric neighbors");
  expect(bfnew::validate_tile_directory(partitioned.graph, directory).ok(),
         "diagonal-neighbor directory passes validation");
}

void test_directory_determinism_and_input_requirements() {
  const bfnew::test::SpatialReorderFixture fixture =
      bfnew::test::make_spatial_reorder_fixture();
  const bfnew::UniformGridPartitioner partitioner{fixture.config};
  const PartitionedGraph first = partitioner.partition(fixture.graph);
  const PartitionedGraph second = partitioner.partition(fixture.graph);
  const TileDirectory& left = first.tiles;
  const TileDirectory& right = second.tiles;

  expect(byte_equal<EdgeOffset>(
             left.neighbor_tile_offsets(), right.neighbor_tile_offsets()) &&
             byte_equal<TileId>(left.neighbor_tiles(), right.neighbor_tiles()) &&
             byte_equal<EdgeOffset>(
                 left.internal_edge_offsets(), right.internal_edge_offsets()) &&
             byte_equal<EdgeId>(left.internal_edge_ids(), right.internal_edge_ids()) &&
             byte_equal<EdgeOffset>(
                 left.outgoing_cross_edge_offsets(),
                 right.outgoing_cross_edge_offsets()) &&
             byte_equal<EdgeId>(
                 left.outgoing_cross_edge_ids(), right.outgoing_cross_edge_ids()) &&
             byte_equal<EdgeOffset>(
                 left.incoming_cross_edge_offsets(),
                 right.incoming_cross_edge_offsets()) &&
             byte_equal<EdgeId>(
                 left.incoming_cross_edge_ids(), right.incoming_cross_edge_ids()) &&
             byte_equal<EdgeOffset>(
                 left.halo_vertex_offsets(), right.halo_vertex_offsets()) &&
             byte_equal<VertexId>(left.halo_vertices(), right.halo_vertices()),
         "repeat partitioning produces byte-identical tile metadata");

  const WeightedGraph nonspatial = bfnew::build_weighted_graph(fixture.graph);
  expect_throws<std::invalid_argument>(
      [&nonspatial] { static_cast<void>(bfnew::build_tile_directory(nonspatial)); },
      "tile directory rejects a graph without spatial ordering");
}

}  // namespace

int main() {
  test_uniform_partitioner_and_directory_counts();
  test_adjacent_long_and_spill_relationships();
  test_diagonal_geometric_neighbors_without_edges();
  test_directory_determinism_and_input_requirements();

  if (failures != 0) {
    std::cerr << failures << " tile test assertion(s) failed\n";
    return 1;
  }
  return 0;
}
