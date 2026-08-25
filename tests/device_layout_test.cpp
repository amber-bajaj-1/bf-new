#include "bfnew/device_layout.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/spatial.hpp"
#include "graph_fixtures.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using bfnew::DeviceGraphLayout32;
using bfnew::EdgeOffset;
using bfnew::InputGraph;
using bfnew::PartitionedGraph;
using bfnew::TileId;
using bfnew::TileRunLaneMasks;
using bfnew::TileRunLayout64;
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
concept HasCsrEdgeIds = requires(T value) { value.csr_edge_ids; };

template <typename T>
concept HasCscEdgeIds = requires(T value) { value.csc_edge_ids; };

template <typename T>
concept HasProvenance = requires(T value) { value.edge_provenance; };

static_assert(!HasCsrEdgeIds<DeviceGraphLayout32>);
static_assert(HasCscEdgeIds<DeviceGraphLayout32>);
static_assert(!HasProvenance<DeviceGraphLayout32>);
static_assert(std::is_same_v<
              typename decltype(DeviceGraphLayout32::csr_row_offsets)::value_type,
              std::uint32_t>);
static_assert(std::is_same_v<
              typename decltype(DeviceGraphLayout32::csc_column_offsets)::value_type,
              std::uint32_t>);
static_assert(std::is_same_v<
              typename decltype(DeviceGraphLayout32::csc_edge_ids)::value_type,
              std::uint32_t>);

[[nodiscard]] InputGraph make_tile_run_fixture() {
  const bfnew::ResourceClassId resource_class{1U};
  std::vector<bfnew::VertexMetadata> vertices{
      bfnew::VertexMetadata::located(0, 0, resource_class),
      bfnew::VertexMetadata::located(1, 0, resource_class),
      bfnew::VertexMetadata::located(10, 0, resource_class),
      bfnew::VertexMetadata::located(30, 0, resource_class),
      bfnew::VertexMetadata::unlocated(resource_class),
      bfnew::VertexMetadata::located(11, 0, resource_class),
      bfnew::VertexMetadata::located(2, 0, resource_class),
  };
  std::vector<bfnew::EdgeInputRecord> edges{
      {VertexId{0U}, VertexId{1U}, 1.0F, bfnew::test::synthetic_provenance(0U)},
      {VertexId{0U}, VertexId{2U}, 2.0F, bfnew::test::synthetic_provenance(1U)},
      {VertexId{0U}, VertexId{3U}, 3.0F, bfnew::test::synthetic_provenance(2U)},
      {VertexId{0U}, VertexId{3U}, 4.0F, bfnew::test::synthetic_provenance(3U)},
      {VertexId{0U}, VertexId{4U}, 5.0F, bfnew::test::synthetic_provenance(4U)},
      {VertexId{4U}, VertexId{0U}, 6.0F, bfnew::test::synthetic_provenance(5U)},
      {VertexId{2U}, VertexId{0U}, 7.0F, bfnew::test::synthetic_provenance(6U)},
      {VertexId{1U}, VertexId{2U}, 8.0F, bfnew::test::synthetic_provenance(7U)},
  };
  return InputGraph{std::move(vertices), std::move(edges)};
}

[[nodiscard]] PartitionedGraph make_partitioned_tile_run_fixture() {
  const bfnew::UniformGridPartitioner partitioner{
      bfnew::SpatialOrderConfig{0, 0, 10U, 10U}};
  return partitioner.partition(make_tile_run_fixture());
}

[[nodiscard]] std::span<const TileId> csr_run_tiles_for_row(
    const TileRunLayout64& runs,
    const VertexId row) {
  const std::size_t begin =
      static_cast<std::size_t>(runs.csr_row_run_offsets[row.value()]);
  const std::size_t end =
      static_cast<std::size_t>(runs.csr_row_run_offsets[row.value() + 1U]);
  return std::span<const TileId>{runs.csr_run_destination_tiles}.subspan(
      begin, end - begin);
}

[[nodiscard]] std::span<const std::uint32_t> csr_run_masks_for_row(
    const TileRunLayout64& runs,
    const TileRunLaneMasks& masks,
    const VertexId row) {
  const std::size_t begin =
      static_cast<std::size_t>(runs.csr_row_run_offsets[row.value()]);
  const std::size_t end =
      static_cast<std::size_t>(runs.csr_row_run_offsets[row.value() + 1U]);
  return std::span<const std::uint32_t>{masks.csr_run_masks}.subspan(
      begin, end - begin);
}

void test_maximal_tile_runs_and_empty_buckets() {
  const PartitionedGraph partitioned = make_partitioned_tile_run_fixture();
  const WeightedGraph& graph = partitioned.graph;
  const TileRunLayout64 runs = bfnew::build_tile_run_layout(graph);
  expect(bfnew::validate_tile_run_layout(graph, runs).ok(),
         "tile-run layout passes deep host validation");
  expect(runs.csr_run_destination_tiles.size() == 7U,
         "fixture produces seven maximal CSR runs");
  expect(runs.csc_run_source_tiles.size() == 6U,
         "fixture produces six maximal CSC runs");

  const VertexId source = graph.old_to_new()[0U];
  const std::span<const TileId> source_run_tiles = csr_run_tiles_for_row(runs, source);
  expect(
      source_run_tiles.size() == 4U && source_run_tiles[0U] == TileId{0U} &&
          source_run_tiles[1U] == TileId{1U} &&
          source_run_tiles[2U] == TileId{2U} &&
          source_run_tiles[3U] == TileId{3U},
      "one CSR row has internal, adjacent, long, and spill runs in layout order");

  const std::size_t source_run_begin =
      static_cast<std::size_t>(runs.csr_row_run_offsets[source.value()]);
  const std::vector<EdgeOffset> expected_lengths{1U, 1U, 2U, 1U};
  for (std::size_t local_run = 0U; local_run < expected_lengths.size(); ++local_run) {
    const std::size_t run = source_run_begin + local_run;
    expect(
        runs.csr_run_edge_offsets[run + 1U] - runs.csr_run_edge_offsets[run] ==
            expected_lengths[local_run],
        "CSR tile-run length preserves each sparse-layout edge");
  }

  const VertexId parallel_destination = graph.old_to_new()[3U];
  const std::size_t parallel_run_begin = static_cast<std::size_t>(
      runs.csc_column_run_offsets[parallel_destination.value()]);
  const std::size_t parallel_run_end = static_cast<std::size_t>(
      runs.csc_column_run_offsets[parallel_destination.value() + 1U]);
  expect(parallel_run_end - parallel_run_begin == 1U,
         "parallel incoming edges remain together in one maximal CSC source-tile run");
  expect(
      runs.csc_run_edge_offsets[parallel_run_begin + 1U] -
              runs.csc_run_edge_offsets[parallel_run_begin] ==
          2U,
      "the CSC run retains both parallel edges");

  const VertexId isolated = graph.old_to_new()[6U];
  expect(
      runs.csr_row_run_offsets[isolated.value()] ==
          runs.csr_row_run_offsets[isolated.value() + 1U] &&
          graph.outgoing().row_offsets[isolated.value()] ==
              graph.outgoing().row_offsets[isolated.value() + 1U],
      "an empty CSR row has zero runs");
  expect(
      runs.csc_column_run_offsets[isolated.value()] ==
          runs.csc_column_run_offsets[isolated.value() + 1U] &&
          graph.incoming().column_offsets[isolated.value()] ==
              graph.incoming().column_offsets[isolated.value() + 1U],
      "an empty CSC column has zero runs");

  const TileRunLayout64 repeated = bfnew::build_tile_run_layout(graph);
  expect(runs == repeated, "tile-run construction is deterministic");
}

void test_deep_tile_run_validation_rejects_corruption() {
  const PartitionedGraph partitioned = make_partitioned_tile_run_fixture();
  const WeightedGraph& graph = partitioned.graph;
  const TileRunLayout64 valid = bfnew::build_tile_run_layout(graph);

  TileRunLayout64 empty_run = valid;
  empty_run.csr_run_edge_offsets[1U] = empty_run.csr_run_edge_offsets[0U];
  expect(
      bfnew::validate_tile_run_layout(graph, empty_run).code ==
          bfnew::TileRunValidationErrorCode::csr_empty_run,
      "deep validation rejects an empty CSR run");

  TileRunLayout64 nonmaximal = valid;
  const VertexId source = graph.old_to_new()[0U];
  const std::size_t first_run =
      static_cast<std::size_t>(valid.csr_row_run_offsets[source.value()]);
  nonmaximal.csr_run_destination_tiles[first_run + 1U] =
      nonmaximal.csr_run_destination_tiles[first_run];
  expect(
      bfnew::validate_tile_run_layout(graph, nonmaximal).code ==
          bfnew::TileRunValidationErrorCode::csr_nonmaximal_runs,
      "deep validation rejects adjacent same-tile runs");

  TileRunLayout64 wrong_spill = valid;
  wrong_spill.csr_run_destination_tiles[first_run + 3U] = TileId{0U};
  expect(
      bfnew::validate_tile_run_layout(graph, wrong_spill).code ==
          bfnew::TileRunValidationErrorCode::csr_run_tile_mismatch,
      "deep validation checks spill-run endpoint ownership");

  const WeightedGraph nonspatial =
      bfnew::build_weighted_graph(bfnew::test::make_core_weighted_fixture().graph);
  expect_throws<std::invalid_argument>(
      [&nonspatial] { static_cast<void>(bfnew::build_tile_run_layout(nonspatial)); },
      "tile-run construction rejects a graph without spatial ordering");
}

void test_checked_device_layout_and_memory_report() {
  const PartitionedGraph partitioned = make_partitioned_tile_run_fixture();
  const WeightedGraph& graph = partitioned.graph;
  const TileRunLayout64 runs = bfnew::build_tile_run_layout(graph);
  const DeviceGraphLayout32 layout =
      bfnew::build_device_graph_layout32(graph, runs);
  expect(bfnew::validate_device_graph_layout32(graph, runs, layout).ok(),
         "32-bit hot layout passes deep equality validation against its host graph");
  expect(layout.edge_count == graph.edge_count(),
         "checked conversion preserves the sparse edge count");
  expect(layout.csr_destinations.size() == graph.edge_count() &&
             layout.csc_sources.size() == graph.edge_count() &&
             layout.csc_edge_ids.size() == graph.edge_count(),
         "device layout retains both relaxation directions and stable CSC IDs");

  const DeviceGraphLayout32 repeated =
      bfnew::build_device_graph_layout32(graph, runs);
  expect(bfnew::device_graph_layouts_deep_equal(layout, repeated),
         "repeated 32-bit layouts are bit-for-bit deeply equal");
  const bfnew::DeviceGraphFingerprint layout_fingerprint =
      bfnew::fingerprint_device_graph_layout32(layout);
  expect(
      layout_fingerprint == bfnew::fingerprint_device_graph_layout32(repeated) &&
          layout_fingerprint ==
              bfnew::fingerprint_device_graph_source32(graph, runs),
      "layout and allocation-free source fingerprints agree deterministically");

  DeviceGraphLayout32 corrupted = layout;
  corrupted.csr_destinations[0U] ^= 1U;
  expect(
      bfnew::validate_device_graph_layout32(graph, runs, corrupted).code ==
          bfnew::DeviceGraphLayoutValidationErrorCode::csr_destinations_mismatch,
      "device-layout validation detects a hot-array round-trip mismatch");
  expect(!bfnew::device_graph_layouts_deep_equal(layout, corrupted),
         "deep layout equality detects one changed component");
  expect(
      layout_fingerprint != bfnew::fingerprint_device_graph_layout32(corrupted),
      "content fingerprint distinguishes a same-shaped hot graph image");

  DeviceGraphLayout32 corrupted_edge_id = layout;
  corrupted_edge_id.csc_edge_ids[0U] ^= 1U;
  expect(
      bfnew::validate_device_graph_layout32(
          graph, runs, corrupted_edge_id).code ==
          bfnew::DeviceGraphLayoutValidationErrorCode::csc_edge_ids_mismatch,
      "device-layout validation detects a changed stable CSC edge ID");
  expect(
      layout_fingerprint !=
          bfnew::fingerprint_device_graph_layout32(corrupted_edge_id),
      "content fingerprint includes stable CSC edge IDs");

  const bfnew::DeviceGraphMemoryReport report =
      bfnew::report_device_graph_memory(layout);
  const std::uint64_t vertex_count = graph.vertex_count();
  const std::uint64_t edge_count = graph.edge_count();
  const std::uint64_t csr_run_count = runs.csr_run_destination_tiles.size();
  const std::uint64_t csc_run_count = runs.csc_run_source_tiles.size();
  const std::uint64_t expected_bytes =
      vertex_count * 4U + (vertex_count + 1U) * 4U + edge_count * 4U +
      edge_count * 4U + (vertex_count + 1U) * 4U +
      (csr_run_count + 1U) * 4U + csr_run_count * 4U +
      (vertex_count + 1U) * 4U + edge_count * 4U + edge_count * 4U +
      edge_count * 4U +
      (vertex_count + 1U) * 4U + (csc_run_count + 1U) * 4U +
      csc_run_count * 4U;
  expect(report.total_bytes == expected_bytes,
         "resident graph memory report totals every hot array by component");
  expect(report.csr_weights_bytes == edge_count * sizeof(float) &&
             report.csc_weights_bytes == edge_count * sizeof(float) &&
             report.csc_edge_ids_bytes == edge_count * sizeof(std::uint32_t),
         "memory report exposes weights and stable reconstruction IDs");

  expect(
      bfnew::checked_device_offset32(std::numeric_limits<std::uint32_t>::max()) ==
          std::numeric_limits<std::uint32_t>::max(),
      "checked narrowing accepts the largest 32-bit offset");
  expect_throws<std::overflow_error>(
      [] {
        static_cast<void>(bfnew::checked_device_offset32(
            static_cast<EdgeOffset>(std::numeric_limits<std::uint32_t>::max()) + 1U));
      },
      "checked narrowing rejects an unrepresentable host offset");
}

void test_reusable_run_lane_masks_and_endpoint_equivalence() {
  const PartitionedGraph partitioned = make_partitioned_tile_run_fixture();
  const WeightedGraph& graph = partitioned.graph;
  const TileRunLayout64 runs = bfnew::build_tile_run_layout(graph);
  const std::vector<std::uint32_t> tile_lane_masks{
      0b0011U,
      0b0110U,
      0b0101U,
      0b1011U,
  };

  TileRunLaneMasks masks;
  bfnew::compute_tile_run_lane_masks(graph, runs, tile_lane_masks, masks);
  expect(
      bfnew::prove_run_admission_equivalence(graph, runs, tile_lane_masks, masks).ok(),
      "CSR and CSC run masks admit exactly the endpoint-admitted lane/edge pairs");

  const VertexId source = graph.old_to_new()[0U];
  const std::span<const std::uint32_t> source_masks =
      csr_run_masks_for_row(runs, masks, source);
  expect(
      source_masks.size() == 4U && source_masks[0U] == 0b0011U &&
          source_masks[1U] == 0b0010U && source_masks[2U] == 0b0001U &&
          source_masks[3U] == 0b0011U,
      "CSR masks implement owner-tile intersection for internal/adjacent/long/spill runs");

  const std::size_t csr_capacity = masks.csr_run_masks.capacity();
  const std::size_t csc_capacity = masks.csc_run_masks.capacity();
  bfnew::compute_tile_run_lane_masks(graph, runs, tile_lane_masks, masks);
  expect(masks.csr_run_masks.capacity() == csr_capacity &&
             masks.csc_run_masks.capacity() == csc_capacity,
         "repeated lane-mask materialization reuses retained vector capacity");

  TileRunLaneMasks wrong_csr = masks;
  wrong_csr.csr_run_masks.front() ^= 0b1000U;
  expect(
      bfnew::prove_run_admission_equivalence(
          graph, runs, tile_lane_masks, wrong_csr)
              .code == bfnew::RunAdmissionProofErrorCode::csr_endpoint_mismatch,
      "equivalence proof detects a CSR run-mask admission error");

  TileRunLaneMasks wrong_csc = masks;
  wrong_csc.csc_run_masks.front() ^= 0b1000U;
  expect(
      bfnew::prove_run_admission_equivalence(
          graph, runs, tile_lane_masks, wrong_csc)
              .code == bfnew::RunAdmissionProofErrorCode::csc_endpoint_mismatch,
      "equivalence proof detects a CSC run-mask admission error");
}

void test_existing_spatial_fixture() {
  const bfnew::test::SpatialReorderFixture fixture =
      bfnew::test::make_spatial_reorder_fixture();
  const bfnew::UniformGridPartitioner partitioner{fixture.config};
  const PartitionedGraph partitioned = partitioner.partition(fixture.graph);
  const TileRunLayout64 runs = bfnew::build_tile_run_layout(partitioned.graph);
  const DeviceGraphLayout32 layout =
      bfnew::build_device_graph_layout32(partitioned.graph, runs);
  std::vector<std::uint32_t> tile_masks(
      partitioned.graph.tile_coordinates().size(), 0b1111U);
  TileRunLaneMasks run_masks;
  bfnew::compute_tile_run_lane_masks(
      partitioned.graph, runs, tile_masks, run_masks);
  expect(bfnew::validate_device_graph_layout32(partitioned.graph, runs, layout).ok() &&
             bfnew::prove_run_admission_equivalence(
                 partitioned.graph, runs, tile_masks, run_masks)
                 .ok(),
         "current Phase 5 spatial fixture passes layout and run-admission validation");
}

}  // namespace

int main() {
  test_maximal_tile_runs_and_empty_buckets();
  test_deep_tile_run_validation_rejects_corruption();
  test_checked_device_layout_and_memory_report();
  test_reusable_run_lane_masks_and_endpoint_equivalence();
  test_existing_spatial_fixture();

  if (failures != 0) {
    std::cerr << failures << " device-layout test assertion(s) failed\n";
    return 1;
  }
  return 0;
}
