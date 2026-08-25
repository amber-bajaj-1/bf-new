#pragma once

// Phase 16 reuses the exact five-lane overlap fixture from Phases 14/15. It
// includes a shared source across two lanes, a shared destination wavefront,
// immediate and multi-round exhaustion, two sources in one lane, and a
// bounded-unreachable target.
#include "batched_jacobi_fixture_suite.hpp"
#include "frontier_fixture_suite.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace bfnew::test {

using BatchedFrontierFixture = BatchedJacobiFixture;
using PreparedBatchedFrontierFixture = PreparedBatchedJacobiFixture;

[[nodiscard]] inline BatchedFrontierFixture
make_mixed_duration_batched_frontier_fixture() {
  return make_mixed_duration_batched_jacobi_fixture();
}

[[nodiscard]] inline BatchedFrontierFixture
make_single_query_batched_frontier_fixture(JacobiFixtureCase fixture) {
  TileRunLayout64 tile_runs = build_tile_run_layout(fixture.partitioned.graph);
  DeviceGraphLayout32 device_graph =
      build_device_graph_layout32(fixture.partitioned.graph, tile_runs);
  std::vector<RouteQuery> queries;
  queries.push_back(std::move(fixture.query));
  return BatchedFrontierFixture{
      std::move(fixture.partitioned),
      std::move(tile_runs),
      std::move(device_graph),
      std::move(queries),
  };
}

[[nodiscard]] inline PreparedBatchedFrontierFixture
prepare_batched_frontier_fixture(
    const BatchedFrontierFixture& fixture,
    const std::span<const RouteQuery> queries,
    const std::uint32_t lane_width,
    const BatchRunRepresentation representation) {
  return prepare_batched_fixture(
      fixture, queries, lane_width, representation);
}

[[nodiscard]] inline std::size_t frontier_lane_for_query(
    const BatchPlanEntry& batch,
    const QueryId query_id) {
  return lane_for_query(batch, query_id);
}

}  // namespace bfnew::test
