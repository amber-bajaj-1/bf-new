#include "bfnew/batch_layout.hpp"
#include "bfnew/batch_plan.hpp"
#include "bfnew/batch_workspace.hpp"
#include "bfnew/device_layout.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/query.hpp"
#include "bfnew/spatial.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using bfnew::BatchDeviceDescription;
using bfnew::BatchPlan;
using bfnew::BatchPlanEntry;
using bfnew::BatchPlannerPolicy;
using bfnew::BatchQueryFeatures;
using bfnew::BatchRunRepresentation;
using bfnew::BatchVertexStorageStrategy;
using bfnew::BatchWorkspaceBudget;
using bfnew::BatchWorkspaceEstimate;
using bfnew::InputGraph;
using bfnew::PartitionedGraph;
using bfnew::QueryId;
using bfnew::RouteQuery;
using bfnew::SelectedRegionIndex;
using bfnew::TileId;
using bfnew::TileRunLayout64;
using bfnew::VertexId;

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

struct WorkspaceFixture {
  PartitionedGraph partitioned;
  TileRunLayout64 tile_runs;
  std::vector<RouteQuery> dense_queries;
  std::vector<RouteQuery> sparse_queries;
};

[[nodiscard]] bfnew::PhysicalProvenance provenance(
    const std::uint64_t record) noexcept {
  return bfnew::PhysicalProvenance{
      bfnew::provenance_domain::synthetic,
      bfnew::provenance_kind::synthetic_edge,
      record,
  };
}

[[nodiscard]] InputGraph make_input_graph() {
  const bfnew::ResourceClassId resource{1U};
  std::vector<bfnew::VertexMetadata> vertices{
      bfnew::VertexMetadata::located(0, 0, resource),
      bfnew::VertexMetadata::located(1, 0, resource),
      bfnew::VertexMetadata::located(10, 0, resource),
      bfnew::VertexMetadata::located(30, 0, resource),
      bfnew::VertexMetadata::unlocated(resource),
      bfnew::VertexMetadata::located(11, 0, resource),
      bfnew::VertexMetadata::located(2, 0, resource),
  };
  std::vector<bfnew::EdgeInputRecord> edges{
      {VertexId{0U}, VertexId{1U}, 1.0F, provenance(0U)},
      {VertexId{0U}, VertexId{2U}, 2.0F, provenance(1U)},
      {VertexId{0U}, VertexId{3U}, 3.0F, provenance(2U)},
      {VertexId{0U}, VertexId{3U}, 4.0F, provenance(3U)},
      {VertexId{0U}, VertexId{4U}, 5.0F, provenance(4U)},
      {VertexId{4U}, VertexId{0U}, 6.0F, provenance(5U)},
      {VertexId{2U}, VertexId{0U}, 7.0F, provenance(6U)},
      {VertexId{1U}, VertexId{2U}, 8.0F, provenance(7U)},
  };
  return InputGraph{std::move(vertices), std::move(edges)};
}

[[nodiscard]] RouteQuery selected_query(
    RouteQuery query,
    std::vector<TileId> tiles,
    const bfnew::WeightedGraph& graph) {
  std::ranges::sort(tiles);
  query.selected_tiles = std::move(tiles);
  if (!bfnew::validate_route_query(graph, query).ok()) {
    throw std::runtime_error{"invalid workspace fixture query"};
  }
  return query;
}

[[nodiscard]] WorkspaceFixture make_fixture() {
  const bfnew::UniformGridPartitioner partitioner{
      bfnew::SpatialOrderConfig{0, 0, 10U, 10U}};
  PartitionedGraph partitioned = partitioner.partition(make_input_graph());
  const bfnew::WeightedGraph& graph = partitioned.graph;
  const auto map = graph.old_to_new();
  const auto owners = graph.owner_tiles();
  const auto owner_of_old = [&](const std::size_t old_vertex) {
    return owners[map[old_vertex].value()];
  };
  const TileId tile_zero = owner_of_old(0U);
  const TileId tile_one = owner_of_old(2U);
  const TileId tile_two = owner_of_old(3U);
  const TileId spill = owner_of_old(4U);

  std::vector<RouteQuery> dense;
  const std::array q10_sources{map[0U], map[2U]};
  const std::array q10_targets{map[5U]};
  dense.push_back(selected_query(
      bfnew::make_route_query(
          QueryId{10U}, graph, q10_sources, q10_targets, 0U, 0U),
      {tile_zero, tile_one},
      graph));
  const std::array q20_sources{map[0U]};
  const std::array q20_targets{map[4U]};
  dense.push_back(selected_query(
      bfnew::make_route_query(
          QueryId{20U}, graph, q20_sources, q20_targets, 0U, 1U),
      {tile_zero, spill},
      graph));
  const std::array q30_sources{map[4U]};
  const std::array q30_targets{map[4U]};
  dense.push_back(selected_query(
      bfnew::make_route_query(
          QueryId{30U}, graph, q30_sources, q30_targets, 0U, 2U),
      {spill},
      graph));
  const std::array q40_sources{map[2U]};
  const std::array q40_targets{map[3U]};
  dense.push_back(selected_query(
      bfnew::make_route_query(
          QueryId{40U}, graph, q40_sources, q40_targets, 0U, 3U),
      {tile_one, tile_two},
      graph));

  std::vector<RouteQuery> sparse;
  const std::array q50_sources{map[2U]};
  const std::array q50_targets{map[5U]};
  sparse.push_back(selected_query(
      bfnew::make_route_query(
          QueryId{50U}, graph, q50_sources, q50_targets, 0U, 0U),
      {tile_one},
      graph));
  const std::array q60_sources{map[4U]};
  const std::array q60_targets{map[4U]};
  sparse.push_back(selected_query(
      bfnew::make_route_query(
          QueryId{60U}, graph, q60_sources, q60_targets, 0U, 0U),
      {spill},
      graph));

  TileRunLayout64 tile_runs = bfnew::build_tile_run_layout(graph);
  return WorkspaceFixture{
      std::move(partitioned),
      std::move(tile_runs),
      std::move(dense),
      std::move(sparse),
  };
}

struct PreparedBatch {
  std::vector<BatchQueryFeatures> features;
  BatchPlan plan;
  BatchDeviceDescription retained;
  BatchDeviceDescription descriptors;
};

[[nodiscard]] PreparedBatch prepare(
    const WorkspaceFixture& fixture,
    const SelectedRegionIndex& selected_regions,
    const std::span<const RouteQuery> queries,
    const std::uint32_t width,
    const bool allow_zero_overlap) {
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  PreparedBatch prepared;
  prepared.features =
      bfnew::make_batch_query_features(graph, selected_regions, queries);
  BatchPlannerPolicy policy;
  policy.lane_width = width;
  policy.maximum_union_inflation_numerator = 3U;
  if (allow_zero_overlap) {
    policy.minimum_jaccard_numerator = 0U;
    policy.minimum_jaccard_denominator = 1U;
  }
  prepared.plan = bfnew::make_overlapping_batch_plan(
      selected_regions, prepared.features, policy);
  if (prepared.plan.batches.size() != 1U) {
    throw std::runtime_error{"workspace fixture unexpectedly split into batches"};
  }
  const BatchPlanEntry& batch = prepared.plan.batches.front();
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      queries,
      prepared.features,
      batch,
      BatchRunRepresentation::retained_per_run_masks,
      prepared.retained);
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      queries,
      prepared.features,
      batch,
      BatchRunRepresentation::compact_nonzero_descriptors,
      prepared.descriptors);
  bfnew::prepare_compact_vertex_mapping(graph, batch, prepared.retained);
  bfnew::prepare_compact_vertex_mapping(graph, batch, prepared.descriptors);
  // Workspace strategy evidence models the warmed reusable path. Repeat the
  // identical preparation so allocation initialization is outside the
  // reported clear/write counters.
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      queries,
      prepared.features,
      batch,
      BatchRunRepresentation::retained_per_run_masks,
      prepared.retained);
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      queries,
      prepared.features,
      batch,
      BatchRunRepresentation::compact_nonzero_descriptors,
      prepared.descriptors);
  bfnew::prepare_compact_vertex_mapping(graph, batch, prepared.retained);
  bfnew::prepare_compact_vertex_mapping(graph, batch, prepared.descriptors);
  if (!bfnew::validate_compact_vertex_mapping(
           graph, batch, prepared.retained)
           .ok() ||
      !bfnew::validate_compact_vertex_mapping(
           graph, batch, prepared.descriptors)
           .ok()) {
    throw std::runtime_error{"workspace fixture compact mapping is invalid"};
  }
  return prepared;
}

[[nodiscard]] const BatchWorkspaceEstimate& find_estimate(
    const std::vector<BatchWorkspaceEstimate>& estimates,
    const BatchVertexStorageStrategy vertices,
    const BatchRunRepresentation runs) {
  const auto position = std::ranges::find_if(
      estimates,
      [vertices, runs](const BatchWorkspaceEstimate& estimate) {
        return estimate.vertex_storage == vertices &&
               estimate.run_representation == runs;
      });
  if (position == estimates.end()) {
    throw std::runtime_error{"workspace strategy estimate is missing"};
  }
  return *position;
}

void test_dense_workspace_matrix() {
  const WorkspaceFixture fixture = make_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const SelectedRegionIndex selected_regions{graph, fixture.tile_runs};
  const BatchWorkspaceBudget budget{1'000'000U, 1'000U, 500U};

  for (const std::uint32_t width : {8U, 16U, 32U}) {
    const PreparedBatch prepared =
        prepare(fixture, selected_regions, fixture.dense_queries, width, false);
    const BatchPlanEntry& batch = prepared.plan.batches.front();
    expect(
        batch.union_vertex_count == 7U &&
            batch.selected_lane_vertex_count == 13U,
        "dense workspace fixture has exact union and useful lane vertices");
    const std::uint64_t expected_metadata =
        32U * static_cast<std::uint64_t>(width) + 280U;

    for (const std::uint32_t slots : {1U, 2U}) {
      const std::vector<BatchWorkspaceEstimate> estimates =
          bfnew::compare_batch_workspace_strategies(
              graph,
              fixture.tile_runs,
              batch,
              prepared.retained,
              prepared.descriptors,
              slots,
              budget);
      expect(estimates.size() == 4U, "the full 2x2 workspace strategy matrix exists");
      const std::uint64_t distance_bytes =
          7U * static_cast<std::uint64_t>(width) * slots * sizeof(float);
      const std::uint64_t reset_bytes = 13U * slots * sizeof(float);
      for (const BatchVertexStorageStrategy vertex_strategy : {
               BatchVertexStorageStrategy::full_graph_vertex_major,
               BatchVertexStorageStrategy::compact_union_tiles}) {
        const BatchWorkspaceEstimate& retained = find_estimate(
            estimates,
            vertex_strategy,
            BatchRunRepresentation::retained_per_run_masks);
        const BatchWorkspaceEstimate& descriptors = find_estimate(
            estimates,
            vertex_strategy,
            BatchRunRepresentation::compact_nonzero_descriptors);
        const bool compact =
            vertex_strategy == BatchVertexStorageStrategy::compact_union_tiles;
        expect(
            retained.distance_bytes == distance_bytes &&
                descriptors.distance_bytes == distance_bytes &&
                retained.distance_reset_bytes == reset_bytes &&
                descriptors.distance_reset_bytes == reset_bytes,
            "distance allocation/reset formulas are exact at 8/16/32 lanes");
        expect(
            retained.allocated_lane_vertex_count ==
                    7U * static_cast<std::uint64_t>(width) &&
                retained.wasted_lane_vertex_count ==
                    7U * static_cast<std::uint64_t>(width) - 13U,
            "allocated and wasted lane vertices are exact");
        expect(
            retained.batch_metadata_bytes == expected_metadata &&
                descriptors.batch_metadata_bytes == expected_metadata,
            "device-shaped padded metadata bytes scale exactly with lane width");
        expect(
            retained.run_storage_bytes == 28U &&
                retained.descriptor_offset_bytes == 0U &&
                retained.tile_mapping_bytes == (compact ? 16U : 0U),
            "retained masks allocate one reusable maximum-orientation array");
        expect(
            descriptors.run_storage_bytes == 48U &&
                descriptors.descriptor_offset_bytes == 32U &&
                descriptors.tile_mapping_bytes == (compact ? 16U : 0U),
            "descriptors allocate one active orientation plus compact mapping only when selected");
        expect(
            retained.active_csr_runs == 6U && retained.active_csc_runs == 5U &&
                retained.zero_csr_runs == 1U && retained.zero_csc_runs == 1U &&
                descriptors.active_csr_runs == 6U &&
                descriptors.active_csc_runs == 5U,
            "active/zero run measurements agree for both representations");
        expect(
            retained.run_preparation_write_bytes == 48U &&
                descriptors.run_preparation_write_bytes == 80U,
            "one-orientation preparation traffic uses warmed clear/write counters");
        expect(
            retained.tile_mapping_write_bytes == (compact ? 32U : 0U) &&
                descriptors.tile_mapping_write_bytes == (compact ? 32U : 0U),
            "mapping traffic uses warmed clears plus writes only for compact storage");
        const std::uint64_t retained_expected_total =
            distance_bytes + (compact ? 16U : 0U) + 28U + expected_metadata;
        const std::uint64_t descriptor_expected_total =
            distance_bytes + (compact ? 16U : 0U) + 48U + 32U +
            expected_metadata;
        expect(
            retained.total_workspace_bytes == retained_expected_total &&
                descriptors.total_workspace_bytes == descriptor_expected_total,
            "workspace component totals are exact and include no hidden allocation");
        expect(
            retained.maximum_concurrent_workspaces ==
                    (budget.device_capacity_bytes - budget.resident_graph_bytes -
                     budget.explicit_reserve_bytes) /
                        retained.total_workspace_bytes &&
                descriptors.maximum_concurrent_workspaces ==
                    (budget.device_capacity_bytes - budget.resident_graph_bytes -
                     budget.explicit_reserve_bytes) /
                        descriptors.total_workspace_bytes,
            "maximum concurrent workspace counts use the explicit remaining budget");
      }
    }
  }
}

void test_sparse_compact_measurement() {
  const WorkspaceFixture fixture = make_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const SelectedRegionIndex selected_regions{graph, fixture.tile_runs};
  const PreparedBatch prepared =
      prepare(fixture, selected_regions, fixture.sparse_queries, 8U, true);
  const BatchPlanEntry& batch = prepared.plan.batches.front();
  expect(
      batch.union_tiles == std::vector<TileId>{TileId{1U}, TileId{3U}} &&
          batch.union_vertex_count == 3U &&
          batch.selected_lane_vertex_count == 3U,
      "sparse workspace fixture has a noncontiguous compact union");
  expect(
      prepared.retained.touched_compact_tiles ==
              std::vector<std::uint32_t>{1U, 3U} &&
          prepared.retained.compact_vertex_biases_by_tile ==
              std::vector<std::uint32_t>{0U, 3U, 0U, 4U} &&
          3U - prepared.retained.compact_vertex_biases_by_tile[1U] == 0U &&
          4U - prepared.retained.compact_vertex_biases_by_tile[1U] == 1U &&
          6U - prepared.retained.compact_vertex_biases_by_tile[3U] == 2U,
      "noncontiguous tile biases round-trip representative global vertices to packed indices");
  BatchDeviceDescription stale_mapping = prepared.retained;
  stale_mapping.compact_vertex_biases_by_tile[0U] = 9U;
  expect(
      bfnew::validate_compact_vertex_mapping(graph, batch, stale_mapping).code ==
          bfnew::BatchLayoutValidationErrorCode::invalid_compact_vertex_mapping,
      "deep mapping validation rejects a nonzero tile outside the touched union");
  expect(
      prepared.retained.run_report.active_csr_runs == 0U &&
          prepared.retained.run_report.active_csc_runs == 0U &&
          prepared.descriptors.csr_run_descriptors.empty() &&
          prepared.descriptors.csc_run_descriptors.empty(),
      "disjoint single-tile lanes produce no admitted runs or descriptors");
  const BatchWorkspaceBudget budget{100'000U, 0U, 0U};
  const std::vector<BatchWorkspaceEstimate> estimates =
      bfnew::compare_batch_workspace_strategies(
          graph,
          fixture.tile_runs,
          batch,
          prepared.retained,
          prepared.descriptors,
          2U,
          budget);
  const BatchWorkspaceEstimate& full_retained = find_estimate(
      estimates,
      BatchVertexStorageStrategy::full_graph_vertex_major,
      BatchRunRepresentation::retained_per_run_masks);
  const BatchWorkspaceEstimate& compact_retained = find_estimate(
      estimates,
      BatchVertexStorageStrategy::compact_union_tiles,
      BatchRunRepresentation::retained_per_run_masks);
  const BatchWorkspaceEstimate& full_descriptors = find_estimate(
      estimates,
      BatchVertexStorageStrategy::full_graph_vertex_major,
      BatchRunRepresentation::compact_nonzero_descriptors);
  const BatchWorkspaceEstimate& compact_descriptors = find_estimate(
      estimates,
      BatchVertexStorageStrategy::compact_union_tiles,
      BatchRunRepresentation::compact_nonzero_descriptors);
  BatchDeviceDescription cold_compact;
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.sparse_queries,
      prepared.features,
      batch,
      BatchRunRepresentation::retained_per_run_masks,
      cold_compact);
  bfnew::prepare_compact_vertex_mapping(graph, batch, cold_compact);
  const BatchWorkspaceEstimate cold_compact_estimate =
      bfnew::estimate_batch_workspace(
          graph,
          fixture.tile_runs,
          batch,
          cold_compact,
          BatchVertexStorageStrategy::compact_union_tiles,
          BatchRunRepresentation::retained_per_run_masks,
          2U,
          budget);
  expect(
      cold_compact_estimate.tile_mapping_write_bytes == 24U &&
          cold_compact_estimate.run_preparation_write_bytes == 28U &&
          cold_compact_estimate.total_preparation_write_bytes == 76U,
      "cold preparation traffic includes dense mapping/run initialization writes");
  expect(
      full_retained.distance_bytes == 448U &&
          compact_retained.distance_bytes == 192U &&
          full_retained.wasted_lane_vertex_count == 53U &&
          compact_retained.wasted_lane_vertex_count == 21U,
      "compact storage measures exact allocation and wasted-vertex savings");
  expect(
      full_retained.distance_reset_bytes == 24U &&
          compact_retained.distance_reset_bytes == 24U,
      "selected-tile reset traffic depends on useful lanes, not allocation size");
  expect(
          full_retained.total_workspace_bytes == 960U &&
          compact_retained.total_workspace_bytes == 720U &&
          full_descriptors.total_workspace_bytes == 948U &&
          compact_descriptors.total_workspace_bytes == 708U,
      "sparse full/compact and retained/descriptor totals match the literal golden");
  expect(
      compact_descriptors.total_workspace_bytes <
              compact_retained.total_workspace_bytes &&
          compact_descriptors.total_workspace_bytes <
              full_descriptors.total_workspace_bytes,
      "zero-active-run measurement exposes when compact descriptors save capacity");
}

void test_budget_boundaries_overflow_and_reuse() {
  const WorkspaceFixture fixture = make_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const SelectedRegionIndex selected_regions{graph, fixture.tile_runs};
  const PreparedBatch prepared =
      prepare(fixture, selected_regions, fixture.dense_queries, 8U, false);
  const BatchPlanEntry& batch = prepared.plan.batches.front();
  const BatchWorkspaceBudget generous{1'000'000U, 0U, 0U};
  const BatchWorkspaceEstimate base = bfnew::estimate_batch_workspace(
      graph,
      fixture.tile_runs,
      batch,
      prepared.retained,
      BatchVertexStorageStrategy::full_graph_vertex_major,
      BatchRunRepresentation::retained_per_run_masks,
      2U,
      generous);
  const std::uint64_t bytes = base.total_workspace_bytes;
  for (const std::pair<std::uint64_t, std::uint64_t> boundary : {
           std::pair{bytes - 1U, 0U},
           std::pair{bytes, 1U},
           std::pair{3U * bytes - 1U, 2U},
           std::pair{3U * bytes, 3U},
       }) {
    const BatchWorkspaceEstimate estimate = bfnew::estimate_batch_workspace(
        graph,
        fixture.tile_runs,
        batch,
        prepared.retained,
        BatchVertexStorageStrategy::full_graph_vertex_major,
        BatchRunRepresentation::retained_per_run_masks,
        2U,
        BatchWorkspaceBudget{boundary.first, 0U, 0U});
    expect(
        estimate.maximum_concurrent_workspaces == boundary.second,
        "concurrency changes exactly at whole-workspace budget boundaries");
  }

  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::estimate_batch_workspace(
            graph,
            fixture.tile_runs,
            batch,
            prepared.retained,
            BatchVertexStorageStrategy::full_graph_vertex_major,
            BatchRunRepresentation::retained_per_run_masks,
            0U,
            generous));
      },
      "zero distance slots are rejected");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::estimate_batch_workspace(
            graph,
            fixture.tile_runs,
            batch,
            prepared.retained,
            static_cast<BatchVertexStorageStrategy>(255U),
            BatchRunRepresentation::retained_per_run_masks,
            2U,
            generous));
      },
      "unknown vertex-storage strategies are rejected");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::estimate_batch_workspace(
            graph,
            fixture.tile_runs,
            batch,
            prepared.retained,
            BatchVertexStorageStrategy::full_graph_vertex_major,
            static_cast<BatchRunRepresentation>(255U),
            2U,
            generous));
      },
      "unknown run representations are rejected");
  BatchDeviceDescription unmapped = prepared.retained;
  unmapped.compact_vertex_mapping_valid = false;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::estimate_batch_workspace(
            graph,
            fixture.tile_runs,
            batch,
            unmapped,
            BatchVertexStorageStrategy::compact_union_tiles,
            BatchRunRepresentation::retained_per_run_masks,
            2U,
            generous));
      },
      "compact estimates require the separately prepared reusable mapping");
  expect_throws<std::overflow_error>(
      [&] {
        static_cast<void>(bfnew::estimate_batch_workspace(
            graph,
            fixture.tile_runs,
            batch,
            prepared.retained,
            BatchVertexStorageStrategy::full_graph_vertex_major,
            BatchRunRepresentation::retained_per_run_masks,
            2U,
            BatchWorkspaceBudget{
                std::numeric_limits<std::uint64_t>::max(),
                std::numeric_limits<std::uint64_t>::max(),
                1U}));
      },
      "budget commitment addition is overflow checked");
  BatchPlanEntry overflow_batch = batch;
  overflow_batch.union_vertex_count = std::numeric_limits<std::uint64_t>::max();
  overflow_batch.selected_lane_vertex_count = 0U;
  expect_throws<std::overflow_error>(
      [&] {
        static_cast<void>(bfnew::estimate_batch_workspace(
            graph,
            fixture.tile_runs,
            overflow_batch,
            prepared.retained,
            BatchVertexStorageStrategy::compact_union_tiles,
            BatchRunRepresentation::retained_per_run_masks,
            2U,
            generous));
      },
      "compact union lane multiplication is overflow checked before allocation");
  BatchPlanEntry impossible = batch;
  impossible.selected_lane_vertex_count =
      static_cast<std::uint64_t>(graph.vertex_count()) * batch.lane_width + 1U;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::estimate_batch_workspace(
            graph,
            fixture.tile_runs,
            impossible,
            prepared.retained,
            BatchVertexStorageStrategy::full_graph_vertex_major,
            BatchRunRepresentation::retained_per_run_masks,
            2U,
            generous));
      },
      "selected lane vertices cannot exceed the described allocation");

  bfnew::ReusableBatchWorkspaceReservation reservation;
  const bfnew::BatchWorkspaceReservationResult first = reservation.reserve(base);
  const bfnew::BatchWorkspaceReservationResult repeated = reservation.reserve(base);
  expect(
      first.capacity_grew && !repeated.capacity_grew &&
          first.generation == 1U && repeated.generation == 2U &&
          reservation.growth_events() == 1U,
      "identical batches reuse one retained allocation with a fresh generation");
  PreparedBatch width_16 =
      prepare(fixture, selected_regions, fixture.dense_queries, 16U, false);
  const BatchWorkspaceEstimate larger = bfnew::estimate_batch_workspace(
      graph,
      fixture.tile_runs,
      width_16.plan.batches.front(),
      width_16.retained,
      BatchVertexStorageStrategy::full_graph_vertex_major,
      BatchRunRepresentation::retained_per_run_masks,
      2U,
      generous);
  const bfnew::BatchWorkspaceReservationResult grown = reservation.reserve(larger);
  expect(
      grown.capacity_grew && grown.generation == 3U &&
          reservation.growth_events() == 2U &&
          reservation.capacity().total_bytes() >= larger.total_workspace_bytes,
      "a wider plan grows component capacities once and remains reusable");
  BatchWorkspaceEstimate inconsistent = base;
  ++inconsistent.total_workspace_bytes;
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(reservation.reserve(inconsistent)); },
      "reservation rejects an inconsistent component sum");
}

void test_device_materialized_jacobi_workspace_estimate() {
  const WorkspaceFixture fixture = make_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;
  const SelectedRegionIndex selected_regions{graph, fixture.tile_runs};
  const PreparedBatch prepared =
      prepare(fixture, selected_regions, fixture.dense_queries, 8U, false);
  const BatchPlanEntry& batch = prepared.plan.batches.front();
  BatchDeviceDescription device_materialized;
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.dense_queries,
      prepared.features,
      batch,
      BatchRunRepresentation::device_materialized_run_masks,
      device_materialized);

  const BatchWorkspaceEstimate estimate = bfnew::estimate_batch_workspace(
      graph,
      fixture.tile_runs,
      batch,
      device_materialized,
      BatchVertexStorageStrategy::full_graph_vertex_major,
      BatchRunRepresentation::device_materialized_run_masks,
      2U,
      BatchWorkspaceBudget{1'000'000U, 0U, 0U});
  expect(
      estimate.run_storage_bytes ==
              fixture.tile_runs.csc_run_source_tiles.size() *
                  sizeof(bfnew::LaneMask) &&
          estimate.descriptor_offset_bytes == 0U &&
          estimate.active_csr_runs == 0U && estimate.zero_csr_runs == 0U &&
          estimate.active_csc_runs ==
              prepared.retained.run_report.active_csc_runs &&
          estimate.zero_csc_runs + estimate.active_csc_runs ==
              fixture.tile_runs.csc_run_source_tiles.size(),
      "device-materialized Jacobi estimation models one dense CSC device image");
  expect(
      estimate.run_preparation_write_bytes ==
              prepared.retained.run_report.csc_runs_visited *
                  sizeof(bfnew::LaneMask) &&
          estimate.run_build_nanoseconds == 0U,
      "device-materialized preparation counts every selected-column device "
      "write without attributing a host run build");
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(bfnew::estimate_batch_workspace(
            graph,
            fixture.tile_runs,
            batch,
            device_materialized,
            BatchVertexStorageStrategy::full_graph_vertex_major,
            BatchRunRepresentation::device_materialized_run_masks,
            2U,
            BatchWorkspaceBudget{1'000'000U, 0U, 0U},
            bfnew::BatchWorkspaceStrategyTiming{0U, 1U}));
      },
      "device-materialized estimation rejects fictitious host run-build time");
  for (const std::pair<BatchVertexStorageStrategy, std::uint32_t> invalid : {
           std::pair{
               BatchVertexStorageStrategy::compact_union_tiles, 2U},
           std::pair{
               BatchVertexStorageStrategy::full_graph_vertex_major, 1U},
       }) {
    expect_throws<std::invalid_argument>(
        [&] {
          static_cast<void>(bfnew::estimate_batch_workspace(
              graph,
              fixture.tile_runs,
              batch,
              device_materialized,
              invalid.first,
              BatchRunRepresentation::device_materialized_run_masks,
              invalid.second,
              BatchWorkspaceBudget{1'000'000U, 0U, 0U}));
        },
        "device-materialized estimation rejects an unimplemented Jacobi "
        "storage shape");
  }
}

[[nodiscard]] std::uint64_t elapsed_nanoseconds(
    const std::chrono::steady_clock::time_point begin,
    const std::chrono::steady_clock::time_point end) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

void test_decision_and_bounded_measurement_output() {
  const WorkspaceFixture fixture = make_fixture();
  const bfnew::WeightedGraph& graph = fixture.partitioned.graph;

  const auto index_begin = std::chrono::steady_clock::now();
  const SelectedRegionIndex selected_regions{graph, fixture.tile_runs};
  const auto index_end = std::chrono::steady_clock::now();
  const std::vector<BatchQueryFeatures> features =
      bfnew::make_batch_query_features(graph, selected_regions, fixture.dense_queries);
  BatchPlannerPolicy policy;
  policy.lane_width = 8U;
  policy.maximum_union_inflation_numerator = 3U;
  const auto planner_begin = std::chrono::steady_clock::now();
  const BatchPlan plan =
      bfnew::make_overlapping_batch_plan(selected_regions, features, policy);
  const auto planner_end = std::chrono::steady_clock::now();
  expect(plan.batches.size() == 1U, "bounded measurement has one golden batch");

  BatchDeviceDescription retained;
  BatchDeviceDescription descriptors;
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.dense_queries,
      features,
      plan.batches.front(),
      BatchRunRepresentation::retained_per_run_masks,
      retained);
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.dense_queries,
      features,
      plan.batches.front(),
      BatchRunRepresentation::compact_nonzero_descriptors,
      descriptors);
  bfnew::prepare_compact_vertex_mapping(
      graph, plan.batches.front(), retained);
  bfnew::prepare_compact_vertex_mapping(
      graph, plan.batches.front(), descriptors);
  const std::size_t retained_csr_capacity =
      retained.csr_run_lane_masks.capacity();
  const std::size_t retained_csc_capacity =
      retained.csc_run_lane_masks.capacity();
  const std::size_t descriptor_csr_capacity =
      descriptors.csr_run_descriptors.capacity();
  const std::size_t descriptor_csc_capacity =
      descriptors.csc_run_descriptors.capacity();
  const std::size_t descriptor_csr_offset_capacity =
      descriptors.csr_descriptor_offsets_by_union_vertex.capacity();
  const std::size_t descriptor_csc_offset_capacity =
      descriptors.csc_descriptor_offsets_by_union_vertex.capacity();
  const std::size_t retained_mapping_capacity =
      retained.compact_vertex_biases_by_tile.capacity();
  const std::size_t descriptor_mapping_capacity =
      descriptors.compact_vertex_biases_by_tile.capacity();

  const auto retained_begin = std::chrono::steady_clock::now();
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.dense_queries,
      features,
      plan.batches.front(),
      BatchRunRepresentation::retained_per_run_masks,
      retained);
  const auto retained_end = std::chrono::steady_clock::now();
  const auto descriptor_begin = std::chrono::steady_clock::now();
  bfnew::prepare_batch_device_description(
      graph,
      fixture.tile_runs,
      fixture.dense_queries,
      features,
      plan.batches.front(),
      BatchRunRepresentation::compact_nonzero_descriptors,
      descriptors);
  const auto descriptor_end = std::chrono::steady_clock::now();
  // Average a bounded set of identical warmed preparations so the reported
  // mapping measurement remains observable on coarse host clocks. This is
  // structural evidence, not a performance threshold.
  constexpr std::uint64_t mapping_repetitions = 1'024U;
  const auto mapping_begin = std::chrono::steady_clock::now();
  for (std::uint64_t repetition = 0U; repetition < mapping_repetitions;
       ++repetition) {
    bfnew::prepare_compact_vertex_mapping(
        graph, plan.batches.front(), retained);
  }
  const auto mapping_end = std::chrono::steady_clock::now();
  bfnew::prepare_compact_vertex_mapping(
      graph, plan.batches.front(), descriptors);
  expect(
      retained.csr_run_lane_masks.capacity() == retained_csr_capacity &&
          retained.csc_run_lane_masks.capacity() == retained_csc_capacity &&
          descriptors.csr_run_descriptors.capacity() ==
              descriptor_csr_capacity &&
          descriptors.csc_run_descriptors.capacity() ==
              descriptor_csc_capacity &&
          descriptors.csr_descriptor_offsets_by_union_vertex.capacity() ==
              descriptor_csr_offset_capacity &&
          descriptors.csc_descriptor_offsets_by_union_vertex.capacity() ==
              descriptor_csc_offset_capacity &&
          retained.compact_vertex_biases_by_tile.capacity() ==
              retained_mapping_capacity &&
          descriptors.compact_vertex_biases_by_tile.capacity() ==
              descriptor_mapping_capacity &&
          retained.compact_mapping_report.entries_cleared == 4U &&
          retained.compact_mapping_report.entries_written == 4U &&
          retained.run_report.csr_retained_entries_cleared == 6U &&
          retained.run_report.csr_retained_entries_written == 6U,
      "bounded timings cover a warmed identical preparation with no capacity growth");
  expect(
      retained.run_report.csr_lane_edge_pairs == 7U &&
          retained.run_report.csc_lane_edge_pairs == 7U &&
          descriptors.run_report.csr_lane_edge_pairs == 7U &&
          descriptors.run_report.csc_lane_edge_pairs == 7U,
      "timed preparation has an asserted semantic checksum of 28 lane-edge pairs");

  const BatchWorkspaceBudget budget{65'536U, 4'096U, 4'096U};
  const std::uint64_t mapping_ns =
      elapsed_nanoseconds(mapping_begin, mapping_end) / mapping_repetitions;
  const std::uint64_t retained_ns =
      elapsed_nanoseconds(retained_begin, retained_end);
  const std::uint64_t descriptor_ns =
      elapsed_nanoseconds(descriptor_begin, descriptor_end);
  const bfnew::BatchWorkspaceBuildMeasurements measurements{
      mapping_ns,
      retained_ns,
      descriptor_ns,
  };
  const std::vector<BatchWorkspaceEstimate> estimates =
      bfnew::compare_batch_workspace_strategies(
          graph,
          fixture.tile_runs,
          plan.batches.front(),
          retained,
          descriptors,
          2U,
          budget,
          measurements);
  for (const BatchWorkspaceEstimate& estimate : estimates) {
    const bool compact = estimate.vertex_storage ==
                         BatchVertexStorageStrategy::compact_union_tiles;
    const bool retained_runs = estimate.run_representation ==
                               BatchRunRepresentation::retained_per_run_masks;
    expect(
        estimate.mapping_build_nanoseconds == (compact ? mapping_ns : 0U) &&
            estimate.run_build_nanoseconds ==
                (retained_runs ? retained_ns : descriptor_ns),
        "every 2x2 strategy row preserves its measured mapping/run evidence");
  }
  expect(
      find_estimate(
          estimates,
          BatchVertexStorageStrategy::full_graph_vertex_major,
          BatchRunRepresentation::retained_per_run_masks)
              .total_preparation_write_bytes == 152U &&
          find_estimate(
              estimates,
              BatchVertexStorageStrategy::full_graph_vertex_major,
              BatchRunRepresentation::compact_nonzero_descriptors)
                  .total_preparation_write_bytes == 184U &&
          find_estimate(
              estimates,
              BatchVertexStorageStrategy::compact_union_tiles,
              BatchRunRepresentation::retained_per_run_masks)
                  .total_preparation_write_bytes == 184U &&
          find_estimate(
              estimates,
              BatchVertexStorageStrategy::compact_union_tiles,
              BatchRunRepresentation::compact_nonzero_descriptors)
                  .total_preparation_write_bytes == 216U,
      "warmed strategy rows use observed mapping clears/writes and one active run orientation");
  bfnew::BatchWorkspaceDecision decision;
  decision.plan_fingerprint = bfnew::fingerprint_batch_plan(plan);
  decision.measurement_scope = bfnew::BatchMeasurementScope::bounded_synthetic;
  decision.budget = budget;
  decision.graph_vertex_count = graph.vertex_count();
  decision.graph_tile_count = graph.tile_coordinates().size();
  decision.lane_width = 8U;
  decision.distance_slot_count = 2U;
  decision.compared_strategies = estimates;
  decision.selected_vertex_storage =
      BatchVertexStorageStrategy::full_graph_vertex_major;
  decision.selected_run_representation =
      BatchRunRepresentation::retained_per_run_masks;
  decision.selected_mapping_build_nanoseconds = 0U;
  decision.selected_run_build_nanoseconds = retained_ns;
  decision.quantitative_reason =
      "bounded union covers all fixture vertices; full/retained uses the fewest "
      "bytes and requires no mapping table";
  expect(
      bfnew::validate_batch_workspace_decision(decision),
      "the explicit bounded full/retained decision is complete and quantitative");
  const auto mutable_estimate = [](
                                    bfnew::BatchWorkspaceDecision& candidate,
                                    const BatchVertexStorageStrategy vertices,
                                    const BatchRunRepresentation runs)
      -> BatchWorkspaceEstimate& {
    const auto position = std::ranges::find_if(
        candidate.compared_strategies,
        [vertices, runs](const BatchWorkspaceEstimate& estimate) {
          return estimate.vertex_storage == vertices &&
                 estimate.run_representation == runs;
        });
    if (position == candidate.compared_strategies.end()) {
      throw std::runtime_error{"mutable workspace strategy row is missing"};
    }
    return *position;
  };
  const auto refresh_workspace_total = [](
                                           bfnew::BatchWorkspaceDecision& candidate,
                                           BatchWorkspaceEstimate& estimate) {
    estimate.total_workspace_bytes =
        estimate.distance_bytes + estimate.tile_mapping_bytes +
        estimate.run_storage_bytes + estimate.descriptor_offset_bytes +
        estimate.batch_metadata_bytes;
    const std::uint64_t committed =
        candidate.budget.resident_graph_bytes +
        candidate.budget.explicit_reserve_bytes;
    estimate.maximum_concurrent_workspaces =
        committed <= candidate.budget.device_capacity_bytes
            ? (candidate.budget.device_capacity_bytes - committed) /
                  estimate.total_workspace_bytes
            : 0U;
  };
  const auto refresh_preparation_total = [](BatchWorkspaceEstimate& estimate) {
    estimate.total_preparation_write_bytes =
        estimate.distance_reset_bytes + estimate.tile_mapping_write_bytes +
        estimate.run_preparation_write_bytes;
  };
  const auto refresh_vertex_storage = [&refresh_workspace_total](
                                          bfnew::BatchWorkspaceDecision& candidate,
                                          BatchWorkspaceEstimate& estimate) {
    estimate.allocated_lane_vertex_count =
        estimate.storage_vertex_count * estimate.lane_width;
    estimate.wasted_lane_vertex_count =
        estimate.allocated_lane_vertex_count -
        estimate.selected_lane_vertex_count;
    estimate.distance_bytes =
        estimate.allocated_lane_vertex_count *
        estimate.distance_slot_count * sizeof(float);
    refresh_workspace_total(candidate, estimate);
  };
  const std::string first = bfnew::serialize_batch_workspace_decision(decision);
  const std::string second = bfnew::serialize_batch_workspace_decision(decision);
  expect(
      first == second &&
          first.find("selected_vertex_storage=full_graph_vertex_major") !=
              std::string::npos &&
          first.find("selected_run_representation=retained_per_run_masks") !=
              std::string::npos &&
          first.find("graph_vertex_count=7") != std::string::npos &&
          first.find("graph_tile_count=4") != std::string::npos &&
          first.find("total_preparation_write_bytes") != std::string::npos &&
          first.find(
              "mapping_build_ns\tdual_orientation_host_run_build_ns") !=
              std::string::npos &&
          first.find("run_build_scope=dual_orientation_host_proof_image") !=
              std::string::npos,
      "workspace serialization is deterministic and retains all row timings");
  bfnew::BatchWorkspaceDecision invalid = decision;
  invalid.compared_strategies.pop_back();
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation requires the complete 2x2 comparison");
  invalid = decision;
  invalid.quantitative_reason.clear();
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation requires a quantitative reason");
  invalid = decision;
  invalid.quantitative_reason += "\ninjected_row";
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation rejects control characters that could inject serialized rows");
  invalid = decision;
  invalid.plan_fingerprint = 0U;
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation rejects the reserved missing plan identity");
  invalid = decision;
  invalid.graph_vertex_count = 0U;
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation requires an immutable graph vertex count");
  invalid = decision;
  invalid.graph_tile_count = 0U;
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation requires an immutable graph tile count");
  invalid = decision;
  invalid.graph_vertex_count =
      static_cast<std::uint64_t>(
          std::numeric_limits<std::uint32_t>::max()) +
      1U;
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation rejects a graph vertex count outside VertexId range");
  invalid = decision;
  invalid.graph_tile_count =
      static_cast<std::uint64_t>(
          std::numeric_limits<std::uint32_t>::max()) +
      1U;
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation rejects a graph tile count outside TileId range");
  invalid = decision;
  invalid.measurement_scope = static_cast<bfnew::BatchMeasurementScope>(255U);
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation rejects an invalid measurement scope");
  invalid = decision;
  invalid.budget.device_capacity_bytes /= 2U;
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation binds concurrency to its recorded budget");
  invalid = decision;
  ++invalid.selected_run_build_nanoseconds;
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation binds selected timings to the selected strategy row");
  invalid = decision;
  ++invalid.compared_strategies.front().total_preparation_write_bytes;
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation binds total preparation traffic to its components");
  invalid = decision;
  ++invalid.compared_strategies.front().allocated_lane_vertex_count;
  ++invalid.compared_strategies.front().wasted_lane_vertex_count;
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation binds allocated lanes to storage vertices and width");
  invalid = decision;
  invalid.compared_strategies.front().mapping_build_nanoseconds = 1U;
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "full-graph decision rows reject compact mapping time");
  invalid = decision;
  BatchWorkspaceEstimate& full_mapping_allocation = mutable_estimate(
      invalid,
      BatchVertexStorageStrategy::full_graph_vertex_major,
      BatchRunRepresentation::retained_per_run_masks);
  full_mapping_allocation.tile_mapping_bytes += sizeof(std::uint32_t);
  refresh_workspace_total(invalid, full_mapping_allocation);
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "a coherent full-row mapping allocation/total/concurrency mutation is rejected");
  invalid = decision;
  BatchWorkspaceEstimate& full_mapping_write = mutable_estimate(
      invalid,
      BatchVertexStorageStrategy::full_graph_vertex_major,
      BatchRunRepresentation::retained_per_run_masks);
  full_mapping_write.tile_mapping_write_bytes = sizeof(std::uint32_t);
  refresh_preparation_total(full_mapping_write);
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "full-graph rows reject coherently totaled compact mapping writes");
  invalid = decision;
  BatchWorkspaceEstimate& divergent_full_vertices = mutable_estimate(
      invalid,
      BatchVertexStorageStrategy::full_graph_vertex_major,
      BatchRunRepresentation::compact_nonzero_descriptors);
  ++divergent_full_vertices.storage_vertex_count;
  refresh_vertex_storage(invalid, divergent_full_vertices);
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "full rows must represent the same graph vertex count after coherent recomputation");
  invalid = decision;
  for (BatchWorkspaceEstimate& estimate : invalid.compared_strategies) {
    if (estimate.vertex_storage ==
        BatchVertexStorageStrategy::full_graph_vertex_major) {
      ++estimate.storage_vertex_count;
      refresh_vertex_storage(invalid, estimate);
    }
  }
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "coherently changed full rows remain bound to the recorded immutable graph vertex count");
  invalid = decision;
  BatchWorkspaceEstimate& invalid_compact_vertices = mutable_estimate(
      invalid,
      BatchVertexStorageStrategy::compact_union_tiles,
      BatchRunRepresentation::retained_per_run_masks);
  ++invalid_compact_vertices.storage_vertex_count;
  refresh_vertex_storage(invalid, invalid_compact_vertices);
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "compact storage count remains bound to the shared union vertex count");
  invalid = decision;
  BatchWorkspaceEstimate& divergent_compact_mapping = mutable_estimate(
      invalid,
      BatchVertexStorageStrategy::compact_union_tiles,
      BatchRunRepresentation::compact_nonzero_descriptors);
  divergent_compact_mapping.tile_mapping_bytes += sizeof(std::uint32_t);
  refresh_workspace_total(invalid, divergent_compact_mapping);
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "compact rows must share one graph-wide mapping allocation");
  invalid = decision;
  for (BatchWorkspaceEstimate& estimate : invalid.compared_strategies) {
    if (estimate.vertex_storage ==
        BatchVertexStorageStrategy::compact_union_tiles) {
      estimate.tile_mapping_bytes += sizeof(std::uint32_t);
      refresh_workspace_total(invalid, estimate);
    }
  }
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "coherently changed compact rows remain bound to graph tile count times uint32 bias bytes");
  invalid = decision;
  for (BatchWorkspaceEstimate& estimate : invalid.compared_strategies) {
    if (estimate.vertex_storage ==
        BatchVertexStorageStrategy::compact_union_tiles) {
      estimate.tile_mapping_write_bytes =
          2U * estimate.tile_mapping_bytes + sizeof(std::uint32_t);
      refresh_preparation_total(estimate);
    }
  }
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "compact mapping writes cannot exceed initialization-or-clear plus current writes");
  invalid = decision;
  BatchWorkspaceEstimate& divergent_metadata = mutable_estimate(
      invalid,
      BatchVertexStorageStrategy::compact_union_tiles,
      BatchRunRepresentation::compact_nonzero_descriptors);
  divergent_metadata.batch_metadata_bytes += sizeof(std::uint32_t);
  refresh_workspace_total(invalid, divergent_metadata);
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "all rows must describe the same fixed batch metadata");
  invalid = decision;
  BatchWorkspaceEstimate& divergent_active_counts = mutable_estimate(
      invalid,
      BatchVertexStorageStrategy::compact_union_tiles,
      BatchRunRepresentation::retained_per_run_masks);
  --divergent_active_counts.active_csr_runs;
  ++divergent_active_counts.zero_csr_runs;
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "all rows must share active and zero run counts for one batch");
  invalid = decision;
  BatchWorkspaceEstimate& divergent_retained_preparation = mutable_estimate(
      invalid,
      BatchVertexStorageStrategy::compact_union_tiles,
      BatchRunRepresentation::retained_per_run_masks);
  divergent_retained_preparation.run_preparation_write_bytes -=
      sizeof(bfnew::LaneMask);
  refresh_preparation_total(divergent_retained_preparation);
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "retained run preparation components agree across vertex storage variants");
  invalid = decision;
  for (BatchWorkspaceEstimate& estimate : invalid.compared_strategies) {
    if (estimate.run_representation ==
        BatchRunRepresentation::retained_per_run_masks) {
      estimate.run_preparation_write_bytes = 0U;
      refresh_preparation_total(estimate);
    }
  }
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "retained preparation writes cover at least every currently active run");
  invalid = decision;
  for (BatchWorkspaceEstimate& estimate : invalid.compared_strategies) {
    if (estimate.run_representation ==
        BatchRunRepresentation::compact_nonzero_descriptors) {
      estimate.descriptor_offset_bytes += sizeof(std::uint32_t);
      estimate.run_preparation_write_bytes += sizeof(std::uint32_t);
      refresh_preparation_total(estimate);
      refresh_workspace_total(invalid, estimate);
    }
  }
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "coherently mutated descriptor rows still charge exactly (union vertices plus one) uint32 offsets");
  invalid = decision;
  invalid.compared_strategies.back() = invalid.compared_strategies[2U];
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation requires exactly one row for every 2x2 combination");
  invalid = decision;
  invalid.compared_strategies.front().vertex_storage =
      static_cast<BatchVertexStorageStrategy>(255U);
  expect(
      !bfnew::validate_batch_workspace_decision(invalid),
      "decision validation rejects invalid strategy enum values");

  const std::uint64_t index_ns = elapsed_nanoseconds(index_begin, index_end);
  const std::uint64_t planner_ns = elapsed_nanoseconds(planner_begin, planner_end);
  std::cout << "phase13_bounded_measurement index_build_ns=" << index_ns
            << " planner_ns=" << planner_ns
            << " compact_mapping_ns=" << mapping_ns
            << " retained_dual_orientation_host_prep_ns=" << retained_ns
            << " descriptor_dual_orientation_host_prep_ns=" << descriptor_ns
            << " semantic_lane_edge_checksum=28"
            << " preparation_scope=warmed_reusable"
            << " compact_mapping_repetitions=" << mapping_repetitions
            << " decision=full_graph_vertex_major+retained_per_run_masks"
            << " synthetic_budget_bytes=" << budget.device_capacity_bytes << '\n';

  constexpr std::uint64_t phase7_vertices = 28'226'432U;
  constexpr std::array<std::uint32_t, 3U> widths{8U, 16U, 32U};
  constexpr std::array<std::uint64_t, 3U> one_slot{
      903'245'824U, 1'806'491'648U, 3'612'983'296U};
  constexpr std::array<std::uint64_t, 3U> two_slots{
      1'806'491'648U, 3'612'983'296U, 7'225'966'592U};
  for (std::size_t index = 0U; index < widths.size(); ++index) {
    const std::uint64_t computed_one =
        phase7_vertices * widths[index] * sizeof(float);
    const std::uint64_t computed_two = computed_one * 2U;
    expect(
        computed_one == one_slot[index] && computed_two == two_slots[index],
        "Phase 7 V-only analytical distance bytes use 64-bit arithmetic");
  }
  std::cout
      << "phase13_phase7_v_only vertices=28226432"
      << " w8_one_two=903245824,1806491648"
      << " w16_one_two=1806491648,3612983296"
      << " w32_one_two=3612983296,7225966592\n";
}

}  // namespace

int main() {
  test_dense_workspace_matrix();
  test_sparse_compact_measurement();
  test_budget_boundaries_overflow_and_reuse();
  test_device_materialized_jacobi_workspace_estimate();
  test_decision_and_bounded_measurement_output();

  if (failures != 0) {
    std::cerr << failures << " Phase 13 workspace check(s) failed\n";
    return 1;
  }
  return 0;
}
