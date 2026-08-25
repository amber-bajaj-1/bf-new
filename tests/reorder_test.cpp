#include "bfnew/graph.hpp"
#include "bfnew/spatial.hpp"
#include "graph_fixtures.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using bfnew::EdgeId;
using bfnew::InputGraph;
using bfnew::PhysicalProvenance;
using bfnew::TileId;
using bfnew::VertexId;
using bfnew::VertexMetadata;
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

struct OriginalEdgeSignature {
  VertexId source;
  VertexId destination;
  std::uint32_t weight_bits;
  PhysicalProvenance provenance;

  bool operator==(const OriginalEdgeSignature&) const noexcept = default;
};

[[nodiscard]] std::vector<OriginalEdgeSignature> collect_original_edges(
    const WeightedGraph& graph) {
  std::vector<OriginalEdgeSignature> signatures(graph.edge_count());
  const bfnew::OutgoingCsrView outgoing = graph.outgoing();
  for (std::size_t source = 0U; source < graph.vertex_count(); ++source) {
    const std::size_t row_begin = static_cast<std::size_t>(outgoing.row_offsets[source]);
    const std::size_t row_end =
        static_cast<std::size_t>(outgoing.row_offsets[source + 1U]);
    for (std::size_t position = row_begin; position < row_end; ++position) {
      const std::size_t edge_id =
          static_cast<std::size_t>(outgoing.edge_ids[position].value());
      signatures[edge_id] = OriginalEdgeSignature{
          graph.original_vertex_ids()[source],
          graph.original_vertex_ids()[outgoing.destinations[position].value()],
          std::bit_cast<std::uint32_t>(outgoing.weights[position]),
          graph.edge_provenance()[edge_id],
      };
    }
  }
  return signatures;
}

void test_morton_policy() {
  const bfnew::MortonLocalityPolicy morton;
  expect(morton(0U, 0U) == 0U, "Morton origin key is zero");
  expect(morton(1U, 0U) == 1U, "Morton x bit occupies the even position");
  expect(morton(0U, 1U) == 2U, "Morton y bit occupies the odd position");
  expect(morton(1U, 1U) == 3U, "Morton interleaves both low bits");
  expect(morton(2U, 0U) == 4U, "Morton spreads higher coordinate bits");
}

void test_permutation_tiles_and_spans() {
  const bfnew::test::SpatialReorderFixture fixture =
      bfnew::test::make_spatial_reorder_fixture();
  const WeightedGraph graph = bfnew::build_spatially_ordered_graph(
      fixture.graph, fixture.config);

  expect(graph.has_spatial_ordering(), "spatially built graph records its ordering mode");
  expect(bfnew::validate_weighted_graph(graph).ok(),
         "spatially reordered graph passes deep validation");

  const std::vector<VertexId> expected_new_to_old{
      VertexId{1U},
      VertexId{7U},
      VertexId{8U},
      VertexId{3U},
      VertexId{2U},
      VertexId{4U},
      VertexId{5U},
      VertexId{0U},
      VertexId{9U},
      VertexId{6U},
  };
  expect(std::ranges::equal(graph.new_to_old(), expected_new_to_old),
         "vertex order follows tile, resource, Morton, and original ID keys");

  for (std::size_t old_index = 0U; old_index < graph.vertex_count(); ++old_index) {
    const VertexId new_id = graph.old_to_new()[old_index];
    expect(graph.new_to_old()[new_id.value()] == VertexId{static_cast<std::uint32_t>(old_index)},
           "old-to-new and new-to-old permutations round trip");
  }
  expect(std::ranges::equal(graph.original_vertex_ids(), graph.new_to_old()),
         "original vertex IDs follow reordered vertex positions");

  const auto tiles = graph.tile_coordinates();
  expect(tiles.size() == 4U, "three located tiles plus one spill tile are stored");
  expect(tiles[0U] == bfnew::TileCoordinate::located(0, 0) &&
             tiles[1U] == bfnew::TileCoordinate::located(1, 0) &&
             tiles[2U] == bfnew::TileCoordinate::located(2, 0),
         "located tile IDs use row-major (tile_y, tile_x) order");
  expect(tiles[3U] == bfnew::TileCoordinate::spill(),
         "spill tile follows every located tile");

  const std::vector<bfnew::EdgeOffset> expected_offsets{0U, 4U, 6U, 8U, 10U};
  expect(std::ranges::equal(graph.tile_vertex_offsets(), expected_offsets),
         "every tile, including spill, has a contiguous vertex span");
  for (std::size_t tile = 0U; tile + 1U < expected_offsets.size(); ++tile) {
    for (std::size_t vertex = expected_offsets[tile];
         vertex < expected_offsets[tile + 1U];
         ++vertex) {
      expect(graph.owner_tiles()[vertex] == TileId{static_cast<std::uint32_t>(tile)},
             "owner tile agrees with its contiguous span");
    }
  }

  expect(graph.new_to_old()[1U] == VertexId{7U} &&
             graph.new_to_old()[2U] == VertexId{8U},
         "identical coordinates share a Morton key and fall back to original ID");
  expect(graph.new_to_old()[6U] == VertexId{5U} &&
             graph.new_to_old()[7U] == VertexId{0U},
         "resource class precedes Morton locality inside a tile");
  expect(graph.new_to_old()[8U] == VertexId{9U} &&
             graph.new_to_old()[9U] == VertexId{6U},
         "spill vertices use resource class and original ID only");
}

void test_edge_preservation_and_determinism() {
  const bfnew::test::SpatialReorderFixture fixture =
      bfnew::test::make_spatial_reorder_fixture();
  const WeightedGraph before = bfnew::build_weighted_graph(fixture.graph);
  const WeightedGraph first = bfnew::build_spatially_ordered_graph(
      fixture.graph, fixture.config);
  const WeightedGraph second = bfnew::build_spatially_ordered_graph(
      fixture.graph, fixture.config);

  expect(collect_original_edges(before) == collect_original_edges(first),
         "spatial permutation preserves every original logical edge and weight");
  expect(collect_original_edges(first) == collect_original_edges(second),
         "repeat construction preserves logical edge identity");

  expect(byte_equal<VertexId>(first.old_to_new(), second.old_to_new()) &&
             byte_equal<VertexId>(first.new_to_old(), second.new_to_old()) &&
             byte_equal<VertexId>(first.original_vertex_ids(), second.original_vertex_ids()),
         "repeat construction has byte-identical permutation arrays");
  expect(byte_equal<TileId>(first.owner_tiles(), second.owner_tiles()) &&
             byte_equal<bfnew::EdgeOffset>(
                 first.tile_vertex_offsets(), second.tile_vertex_offsets()),
         "repeat construction has byte-identical tile IDs and spans");
  expect(std::ranges::equal(first.tile_coordinates(), second.tile_coordinates()),
         "repeat construction has identical dense tile coordinates");

  const bfnew::OutgoingCsrView first_outgoing = first.outgoing();
  const bfnew::OutgoingCsrView second_outgoing = second.outgoing();
  expect(byte_equal<bfnew::EdgeOffset>(
             first_outgoing.row_offsets, second_outgoing.row_offsets) &&
             byte_equal<VertexId>(
                 first_outgoing.destinations, second_outgoing.destinations) &&
             byte_equal<float>(first_outgoing.weights, second_outgoing.weights) &&
             byte_equal<EdgeId>(first_outgoing.edge_ids, second_outgoing.edge_ids),
         "repeat construction has byte-identical CSR ordering");

  const bfnew::IncomingCscView first_incoming = first.incoming();
  const bfnew::IncomingCscView second_incoming = second.incoming();
  expect(byte_equal<bfnew::EdgeOffset>(
             first_incoming.column_offsets, second_incoming.column_offsets) &&
             byte_equal<VertexId>(first_incoming.sources, second_incoming.sources) &&
             byte_equal<float>(first_incoming.weights, second_incoming.weights) &&
             byte_equal<EdgeId>(first_incoming.edge_ids, second_incoming.edge_ids),
         "repeat construction has byte-identical CSC ordering");
}

void test_floor_division_and_policy_injection() {
  const bfnew::ResourceClassId resource_class{1U};
  const InputGraph negative_graph{
      {
          VertexMetadata::located(-1, 0, resource_class),
          VertexMetadata::located(-10, 0, resource_class),
          VertexMetadata::located(-11, 0, resource_class),
      },
      {},
  };
  const bfnew::SpatialOrderConfig config{0, 0, 10U, 10U};
  const WeightedGraph reordered =
      bfnew::build_spatially_ordered_graph(negative_graph, config);
  expect(reordered.tile_coordinates()[0U] == bfnew::TileCoordinate::located(-2, 0) &&
             reordered.tile_coordinates()[1U] == bfnew::TileCoordinate::located(-1, 0),
         "negative coordinates use mathematical floor division");
  const std::vector<VertexId> expected_order{VertexId{2U}, VertexId{1U}, VertexId{0U}};
  expect(std::ranges::equal(reordered.new_to_old(), expected_order),
         "negative-coordinate local positions are derived from the tile lower origin");

  const InputGraph row_major_graph{
      {
          VertexMetadata::located(0, 10, resource_class),
          VertexMetadata::located(20, 0, resource_class),
      },
      {},
  };
  const WeightedGraph row_major =
      bfnew::build_spatially_ordered_graph(row_major_graph, config);
  expect(row_major.tile_coordinates()[0U] == bfnew::TileCoordinate::located(2, 0) &&
             row_major.tile_coordinates()[1U] == bfnew::TileCoordinate::located(0, 1),
         "dense tile IDs sort by tile_y before tile_x");

  const bfnew::test::SpatialReorderFixture fixture =
      bfnew::test::make_spatial_reorder_fixture();
  std::size_t policy_calls = 0U;
  const bfnew::LocalityKeyPolicy reverse_morton =
      [&policy_calls](const std::uint32_t x, const std::uint32_t y) {
        ++policy_calls;
        return std::numeric_limits<std::uint64_t>::max() -
               bfnew::MortonLocalityPolicy{}(x, y);
      };
  const WeightedGraph custom = bfnew::build_spatially_ordered_graph(
      fixture.graph, fixture.config, reverse_morton);
  expect(policy_calls == 8U, "injected locality policy runs once per located vertex");
  expect(custom.new_to_old()[0U] != VertexId{1U},
         "injected locality policy can change within-resource ordering");
  expect(bfnew::validate_weighted_graph(custom).ok(),
         "graph built with an injected locality policy remains valid");

  expect_throws<std::invalid_argument>(
      [&fixture] {
        static_cast<void>(bfnew::build_spatially_ordered_graph(
            fixture.graph, bfnew::SpatialOrderConfig{0, 0, 0U, 10U}));
      },
      "zero tile width is rejected");
  expect_throws<std::invalid_argument>(
      [&fixture] {
        static_cast<void>(bfnew::build_spatially_ordered_graph(
            fixture.graph, bfnew::SpatialOrderConfig{0, 0, 10U, 0U}));
      },
      "zero tile height is rejected");
  expect_throws<std::invalid_argument>(
      [&fixture] {
        static_cast<void>(bfnew::build_spatially_ordered_graph(
            fixture.graph, fixture.config, bfnew::LocalityKeyPolicy{}));
      },
      "empty locality policy is rejected");
}

[[nodiscard]] std::size_t destination_tile_runs_before(
    const WeightedGraph& before,
    const WeightedGraph& after,
    const VertexId original_source) {
  const bfnew::OutgoingCsrView outgoing = before.outgoing();
  const std::size_t row_begin =
      static_cast<std::size_t>(outgoing.row_offsets[original_source.value()]);
  const std::size_t row_end =
      static_cast<std::size_t>(outgoing.row_offsets[original_source.value() + 1U]);
  std::size_t runs = 0U;
  TileId previous_tile{};
  bool has_previous = false;
  for (std::size_t position = row_begin; position < row_end; ++position) {
    const VertexId reordered_destination =
        after.old_to_new()[outgoing.destinations[position].value()];
    const TileId tile = after.owner_tiles()[reordered_destination.value()];
    if (!has_previous || tile != previous_tile) {
      ++runs;
      previous_tile = tile;
      has_previous = true;
    }
  }
  return runs;
}

[[nodiscard]] std::size_t destination_tile_runs_after(
    const WeightedGraph& graph,
    const VertexId original_source) {
  const VertexId source = graph.old_to_new()[original_source.value()];
  const bfnew::OutgoingCsrView outgoing = graph.outgoing();
  const std::size_t row_begin =
      static_cast<std::size_t>(outgoing.row_offsets[source.value()]);
  const std::size_t row_end =
      static_cast<std::size_t>(outgoing.row_offsets[source.value() + 1U]);
  std::size_t runs = 0U;
  TileId previous_tile{};
  bool has_previous = false;
  for (std::size_t position = row_begin; position < row_end; ++position) {
    const TileId tile = graph.owner_tiles()[outgoing.destinations[position].value()];
    if (!has_previous || tile != previous_tile) {
      ++runs;
      previous_tile = tile;
      has_previous = true;
    }
  }
  return runs;
}

void print_locality_report() {
  const bfnew::test::SpatialReorderFixture fixture =
      bfnew::test::make_spatial_reorder_fixture();
  const WeightedGraph before = bfnew::build_weighted_graph(fixture.graph);
  const WeightedGraph after = bfnew::build_spatially_ordered_graph(
      fixture.graph, fixture.config);
  const std::size_t runs_before =
      destination_tile_runs_before(before, after, fixture.locality_report_source);
  const std::size_t runs_after =
      destination_tile_runs_after(after, fixture.locality_report_source);
  expect(runs_after < runs_before,
         "destination-tile ordering reduces tile runs in the synthetic report row");
  std::cout << "Phase 3 locality report: source="
            << fixture.locality_report_source.value() << " edges=8"
            << " destination_tile_runs_before=" << runs_before
            << " destination_tile_runs_after=" << runs_after << '\n';
}

}  // namespace

int main() {
  test_morton_policy();
  test_permutation_tiles_and_spans();
  test_edge_preservation_and_determinism();
  test_floor_division_and_policy_injection();
  print_locality_report();

  if (failures != 0) {
    std::cerr << failures << " reorder test assertion(s) failed\n";
    return 1;
  }
  return 0;
}
