#include "bfnew/sssp.hpp"

#include "bfnew/graph.hpp"
#include "sssp_common.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace bfnew {

SsspResult synchronous_bellman_ford(
    const WeightedGraph& graph,
    const std::span<const VertexId> sources) {
  SsspResult result = cpu::detail::make_initial_result(graph, sources);
  const OutgoingCsrView outgoing = graph.outgoing();

  for (std::size_t round = 1U; round < graph.vertex_count(); ++round) {
    std::vector<float> next_distances = result.distances;
    bool changed = false;
    for (std::size_t source = 0U; source < graph.vertex_count(); ++source) {
      const float source_distance = result.distances[source];
      if (!std::isfinite(source_distance)) {
        continue;
      }
      const std::size_t row_begin =
          static_cast<std::size_t>(outgoing.row_offsets[source]);
      const std::size_t row_end =
          static_cast<std::size_t>(outgoing.row_offsets[source + 1U]);
      for (std::size_t position = row_begin; position < row_end; ++position) {
        const VertexId destination = outgoing.destinations[position];
        const float candidate = source_distance + outgoing.weights[position];
        if (candidate < next_distances[destination.value()]) {
          next_distances[destination.value()] = candidate;
          changed = true;
        }
      }
    }
    result.distances = std::move(next_distances);
    if (!changed) {
      break;
    }
  }
  return result;
}

}  // namespace bfnew
