#pragma once

// Phase 14 and Phase 15 deliberately share this tiny graph so pull and dense
// push can be compared without introducing a second synthetic workload. Dense
// push uses the reverse CSR schedule to expose one edge of progress per scan.
#include "batched_jacobi_fixture_suite.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace bfnew::test {

using BatchedDenseFixture = BatchedJacobiFixture;
using PreparedBatchedDenseFixture = PreparedBatchedJacobiFixture;

[[nodiscard]] inline BatchedDenseFixture
make_mixed_duration_batched_dense_fixture() {
  return make_mixed_duration_batched_jacobi_fixture();
}

[[nodiscard]] inline PreparedBatchedDenseFixture prepare_batched_dense_fixture(
    const BatchedDenseFixture& fixture,
    const std::span<const RouteQuery> queries,
    const std::uint32_t lane_width,
    const BatchRunRepresentation representation) {
  return prepare_batched_fixture(
      fixture, queries, lane_width, representation);
}

[[nodiscard]] inline std::size_t dense_lane_for_query(
    const BatchPlanEntry& batch,
    const QueryId query_id) {
  return lane_for_query(batch, query_id);
}

}  // namespace bfnew::test
