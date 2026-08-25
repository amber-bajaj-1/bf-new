#include "bfnew/graph.hpp"
#include "bfnew/spatial.hpp"
#include "bfnew/sssp.hpp"
#include "graph_fixtures.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using bfnew::EdgeInputRecord;
using bfnew::InputGraph;
using bfnew::ReconstructedPath;
using bfnew::SsspResult;
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

[[nodiscard]] EdgeInputRecord make_edge(
    const VertexId source,
    const VertexId destination,
    const float weight,
    const std::uint64_t source_record) {
  return EdgeInputRecord{
      source,
      destination,
      weight,
      bfnew::test::synthetic_provenance(source_record),
  };
}

[[nodiscard]] std::vector<VertexMetadata> plain_vertices(const std::size_t count) {
  std::vector<VertexMetadata> vertices;
  vertices.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    vertices.push_back(VertexMetadata::located(
        static_cast<std::int32_t>(index),
        0,
        bfnew::ResourceClassId{1U}));
  }
  return vertices;
}

void expect_exact_distance_agreement(
    const SsspResult& oracle,
    const SsspResult& reference,
    const std::string_view description) {
  expect(oracle.sources == reference.sources, description);
  expect(oracle.distances == reference.distances, description);
}

void validate_all_requested_paths(
    const WeightedGraph& graph,
    const SsspResult& result) {
  for (std::size_t target_index = 0U; target_index < graph.vertex_count(); ++target_index) {
    const VertexId target{static_cast<std::uint32_t>(target_index)};
    const auto path = bfnew::reconstruct_path_from_distances(graph, result, target);
    if (std::isfinite(result.distances[target_index])) {
      expect(path.has_value(), "every finite target has a reconstructed path");
      if (path) {
        expect(bfnew::validate_reconstructed_path(graph, result, target, *path),
               "reconstructed path is tight, continuous, and source-terminated");
      }
    } else {
      expect(!path.has_value(), "unreachable target has no reconstructed path");
    }
  }
}

void test_core_fixture_distances_and_paths() {
  const bfnew::test::CoreWeightedFixture fixture =
      bfnew::test::make_core_weighted_fixture();
  const WeightedGraph graph = bfnew::build_weighted_graph(fixture.graph);
  const SsspResult oracle = bfnew::dijkstra_oracle(graph, fixture.sources);
  const SsspResult reference =
      bfnew::synchronous_bellman_ford(graph, fixture.sources);
  expect_exact_distance_agreement(
      oracle, reference, "Dijkstra and synchronous min-plus labels agree exactly");

  const std::vector<float> expected{
      0.0F,
      0.5F,
      0.5F,
      2.0F,
      4.0F,
      4.25F,
      std::numeric_limits<float>::infinity(),
  };
  expect(oracle.distances == expected,
         "core fixture distances match deliberately representable expectations");
  validate_all_requested_paths(graph, oracle);
  validate_all_requested_paths(graph, reference);

  const auto target_path =
      bfnew::reconstruct_path_from_distances(graph, oracle, VertexId{5U});
  const std::vector<VertexId> expected_vertices{
      VertexId{0U},
      VertexId{1U},
      VertexId{2U},
      VertexId{3U},
      VertexId{4U},
      VertexId{5U},
  };
  expect(target_path && target_path->vertices == expected_vertices &&
             target_path->cost == 4.25F,
         "post-pass path follows the zero edge and reports exact float cost");

  const auto unreachable =
      bfnew::reconstruct_path_from_distances(graph, oracle, fixture.disconnected_vertex);
  expect(!unreachable, "disconnected fixture vertex remains unreachable");
}

void test_spatial_fixture_and_multiple_sources() {
  const bfnew::test::SpatialReorderFixture fixture =
      bfnew::test::make_spatial_reorder_fixture();
  const bfnew::UniformGridPartitioner partitioner{fixture.config};
  const bfnew::PartitionedGraph partitioned = partitioner.partition(fixture.graph);
  const VertexId source = partitioned.graph.old_to_new()[1U];
  const std::vector<VertexId> sources{source};
  const SsspResult oracle = bfnew::dijkstra_oracle(partitioned.graph, sources);
  const SsspResult reference =
      bfnew::synchronous_bellman_ford(partitioned.graph, sources);
  expect_exact_distance_agreement(
      oracle, reference, "both solvers agree after spatial permutation");
  validate_all_requested_paths(partitioned.graph, oracle);
  validate_all_requested_paths(partitioned.graph, reference);

  const WeightedGraph core =
      bfnew::build_weighted_graph(bfnew::test::make_core_weighted_fixture().graph);
  const std::vector<VertexId> duplicate_sources{
      VertexId{6U}, VertexId{0U}, VertexId{6U}};
  const SsspResult multi_oracle = bfnew::dijkstra_oracle(core, duplicate_sources);
  const SsspResult multi_reference =
      bfnew::synchronous_bellman_ford(core, duplicate_sources);
  expect(multi_oracle.sources == std::vector{VertexId{0U}, VertexId{6U}},
         "solver sources are sorted and deduplicated");
  expect_exact_distance_agreement(
      multi_oracle, multi_reference, "multiple-source labels agree exactly");
  const auto source_path =
      bfnew::reconstruct_path_from_distances(core, multi_oracle, VertexId{6U});
  expect(source_path && source_path->vertices == std::vector{VertexId{6U}} &&
             source_path->edge_ids.empty() && source_path->cost == 0.0F,
         "source target reconstructs as a zero-edge path");
}

void test_stable_edge_tie_breaking() {
  const InputGraph input{
      plain_vertices(4U),
      {
          make_edge(VertexId{0U}, VertexId{1U}, 0.5F, 200U),
          make_edge(VertexId{0U}, VertexId{2U}, 0.5F, 201U),
          make_edge(VertexId{1U}, VertexId{3U}, 0.5F, 202U),
          make_edge(VertexId{2U}, VertexId{3U}, 0.5F, 203U),
      },
  };
  const WeightedGraph graph = bfnew::build_weighted_graph(input);
  const std::vector<VertexId> sources{VertexId{0U}};
  const SsspResult result = bfnew::synchronous_bellman_ford(graph, sources);
  const auto path = bfnew::reconstruct_path_from_distances(graph, result, VertexId{3U});
  expect(path && path->vertices ==
                     std::vector{VertexId{0U}, VertexId{1U}, VertexId{3U}},
         "equal-cost incoming choices use the smaller stable edge ID");
  if (path) {
    expect(bfnew::validate_reconstructed_path(graph, result, VertexId{3U}, *path),
           "tie-broken path validates independently");
    ReconstructedPath corrupted = *path;
    corrupted.cost = 2.0F;
    expect(!bfnew::validate_reconstructed_path(
               graph, result, VertexId{3U}, corrupted),
           "path validator rejects an incorrect reported cost");
  }
}

void test_zero_weight_cycle_backtracking() {
  const InputGraph input{
      plain_vertices(4U),
      {
          make_edge(VertexId{0U}, VertexId{1U}, 0.0F, 300U),
          make_edge(VertexId{1U}, VertexId{0U}, 0.0F, 301U),
          make_edge(VertexId{1U}, VertexId{2U}, 1.0F, 302U),
          make_edge(VertexId{3U}, VertexId{1U}, 1.0F, 303U),
      },
  };
  const WeightedGraph graph = bfnew::build_weighted_graph(input);
  const std::vector<VertexId> sources{VertexId{3U}};
  const SsspResult oracle = bfnew::dijkstra_oracle(graph, sources);
  const SsspResult reference = bfnew::synchronous_bellman_ford(graph, sources);
  expect_exact_distance_agreement(
      oracle, reference, "zero-cycle fixture labels agree exactly");

  const auto path = bfnew::reconstruct_path_from_distances(graph, reference, VertexId{2U});
  const std::vector<VertexId> expected_vertices{VertexId{3U}, VertexId{1U}, VertexId{2U}};
  expect(path && path->vertices == expected_vertices && path->cost == 2.0F,
         "incoming-edge search backtracks out of a zero-weight cycle");
  if (path) {
    expect(bfnew::validate_reconstructed_path(graph, reference, VertexId{2U}, *path),
           "cycle-safe reconstructed path terminates at its source");
  }
}

void test_validation_and_ulp_policy() {
  const WeightedGraph graph =
      bfnew::build_weighted_graph(bfnew::test::make_core_weighted_fixture().graph);
  expect_throws<std::invalid_argument>(
      [&graph] { static_cast<void>(bfnew::dijkstra_oracle(graph, {})); },
      "Dijkstra oracle rejects an empty source set");
  expect_throws<std::invalid_argument>(
      [&graph] { static_cast<void>(bfnew::synchronous_bellman_ford(graph, {})); },
      "Bellman-Ford reference rejects an empty source set");
  const std::vector<VertexId> invalid_source{VertexId{7U}};
  expect_throws<std::out_of_range>(
      [&graph, &invalid_source] {
        static_cast<void>(bfnew::dijkstra_oracle(graph, invalid_source));
      },
      "Dijkstra oracle rejects an out-of-range source");
  expect_throws<std::out_of_range>(
      [&graph, &invalid_source] {
        static_cast<void>(bfnew::synchronous_bellman_ford(graph, invalid_source));
      },
      "Bellman-Ford reference rejects an out-of-range source");

  const float next = std::nextafter(1.0F, 2.0F);
  expect(bfnew::nonnegative_distance_within_ulps(1.0F, next, 1U),
         "general-data comparison accepts one ULP of float difference");
  expect(!bfnew::nonnegative_distance_within_ulps(1.0F, next, 0U),
         "zero-ULP policy remains exact");
  expect(bfnew::nonnegative_distance_within_ulps(
             std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::infinity()),
         "matching unreachable labels compare equal");
  expect(!bfnew::nonnegative_distance_within_ulps(
             std::numeric_limits<float>::quiet_NaN(), 1.0F),
         "NaN is never a valid distance comparison");
}

}  // namespace

int main() {
  test_core_fixture_distances_and_paths();
  test_spatial_fixture_and_multiple_sources();
  test_stable_edge_tie_breaking();
  test_zero_weight_cycle_backtracking();
  test_validation_and_ulp_policy();

  if (failures != 0) {
    std::cerr << failures << " SSSP test assertion(s) failed\n";
    return 1;
  }
  return 0;
}
