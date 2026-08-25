#include "bfnew/sssp.hpp"

#include "bfnew/graph.hpp"
#include "sssp_common.hpp"

#include <cstddef>
#include <cstdint>
#include <queue>
#include <span>
#include <vector>

namespace bfnew {
namespace {

struct QueueEntry {
  float distance;
  VertexId vertex;
};

struct QueueEntryGreater {
  [[nodiscard]] bool operator()(const QueueEntry& left, const QueueEntry& right) const
      noexcept {
    if (left.distance != right.distance) {
      return left.distance > right.distance;
    }
    return left.vertex > right.vertex;
  }
};

}  // namespace

SsspResult dijkstra_oracle(
    const WeightedGraph& graph,
    const std::span<const VertexId> sources) {
  SsspResult result = cpu::detail::make_initial_result(graph, sources);
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater> queue;
  for (const VertexId source : result.sources) {
    queue.push(QueueEntry{0.0F, source});
  }

  const OutgoingCsrView outgoing = graph.outgoing();
  while (!queue.empty()) {
    const QueueEntry entry = queue.top();
    queue.pop();
    if (entry.distance != result.distances[entry.vertex.value()]) {
      continue;
    }

    const std::size_t row_begin =
        static_cast<std::size_t>(outgoing.row_offsets[entry.vertex.value()]);
    const std::size_t row_end =
        static_cast<std::size_t>(outgoing.row_offsets[entry.vertex.value() + 1U]);
    for (std::size_t position = row_begin; position < row_end; ++position) {
      const VertexId destination = outgoing.destinations[position];
      const float candidate = entry.distance + outgoing.weights[position];
      if (candidate < result.distances[destination.value()]) {
        result.distances[destination.value()] = candidate;
        queue.push(QueueEntry{candidate, destination});
      }
    }
  }
  return result;
}

}  // namespace bfnew
