#include "bfnew/sssp.hpp"

#include "bfnew/graph.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

struct TightIncomingEdge {
  EdgeId edge_id;
  VertexId predecessor;
  float weight;
};

struct BacktrackingFrame {
  VertexId vertex;
  std::vector<TightIncomingEdge> candidates;
  std::size_t next_candidate{};
};

struct EdgeDetails {
  VertexId source{};
  VertexId destination{};
  float weight{};
};

[[nodiscard]] bool result_shape_is_valid(
    const WeightedGraph& graph,
    const SsspResult& result) {
  if (result.distances.size() != graph.vertex_count() || result.sources.empty() ||
      !std::is_sorted(result.sources.begin(), result.sources.end()) ||
      std::adjacent_find(result.sources.begin(), result.sources.end()) !=
          result.sources.end()) {
    return false;
  }
  for (const VertexId source : result.sources) {
    if (!is_valid_vertex_id(source, graph.vertex_count()) ||
        result.distances[source.value()] != 0.0F) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::vector<TightIncomingEdge> tight_incoming_candidates(
    const WeightedGraph& graph,
    const SsspResult& result,
    const VertexId vertex) {
  std::vector<TightIncomingEdge> candidates;
  const IncomingCscView incoming = graph.incoming();
  const std::size_t column_begin =
      static_cast<std::size_t>(incoming.column_offsets[vertex.value()]);
  const std::size_t column_end =
      static_cast<std::size_t>(incoming.column_offsets[vertex.value() + 1U]);
  candidates.reserve(column_end - column_begin);
  for (std::size_t position = column_begin; position < column_end; ++position) {
    const VertexId predecessor = incoming.sources[position];
    const float predecessor_distance = result.distances[predecessor.value()];
    if (!std::isfinite(predecessor_distance)) {
      continue;
    }
    const float candidate = predecessor_distance + incoming.weights[position];
    if (candidate == result.distances[vertex.value()]) {
      candidates.push_back(TightIncomingEdge{
          incoming.edge_ids[position],
          predecessor,
          incoming.weights[position],
      });
    }
  }
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const TightIncomingEdge& left, const TightIncomingEdge& right) {
        return left.edge_id < right.edge_id;
      });
  return candidates;
}

[[nodiscard]] std::vector<EdgeDetails> edge_details_by_id(const WeightedGraph& graph) {
  std::vector<EdgeDetails> details(graph.edge_count());
  const OutgoingCsrView outgoing = graph.outgoing();
  for (std::size_t source = 0U; source < graph.vertex_count(); ++source) {
    const std::size_t row_begin = static_cast<std::size_t>(outgoing.row_offsets[source]);
    const std::size_t row_end =
        static_cast<std::size_t>(outgoing.row_offsets[source + 1U]);
    for (std::size_t position = row_begin; position < row_end; ++position) {
      const std::size_t edge_id =
          static_cast<std::size_t>(outgoing.edge_ids[position].value());
      details[edge_id] = EdgeDetails{
          VertexId{static_cast<std::uint32_t>(source)},
          outgoing.destinations[position],
          outgoing.weights[position],
      };
    }
  }
  return details;
}

}  // namespace

std::optional<ReconstructedPath> reconstruct_path_from_distances(
    const WeightedGraph& graph,
    const SsspResult& result,
    const VertexId target) {
  if (!result_shape_is_valid(graph, result)) {
    throw std::invalid_argument{"path reconstruction requires a valid distance-only result"};
  }
  if (!is_valid_vertex_id(target, graph.vertex_count())) {
    throw std::out_of_range{"path target is outside the graph vertex range"};
  }
  if (!std::isfinite(result.distances[target.value()])) {
    return std::nullopt;
  }

  std::vector<bool> source_mask(graph.vertex_count(), false);
  for (const VertexId source : result.sources) {
    source_mask[source.value()] = true;
  }

  std::vector<bool> on_path(graph.vertex_count(), false);
  std::vector<VertexId> backward_vertices{target};
  std::vector<EdgeId> backward_edges;
  std::vector<float> backward_weights;
  std::vector<BacktrackingFrame> stack;
  on_path[target.value()] = true;
  stack.push_back(BacktrackingFrame{
      target,
      tight_incoming_candidates(graph, result, target),
      0U,
  });

  while (!stack.empty()) {
    BacktrackingFrame& frame = stack.back();
    if (source_mask[frame.vertex.value()]) {
      ReconstructedPath path;
      path.vertices.assign(backward_vertices.rbegin(), backward_vertices.rend());
      path.edge_ids.assign(backward_edges.rbegin(), backward_edges.rend());
      path.cost = 0.0F;
      for (auto weight = backward_weights.rbegin(); weight != backward_weights.rend();
           ++weight) {
        path.cost = path.cost + *weight;
      }
      return path;
    }

    bool advanced = false;
    while (frame.next_candidate < frame.candidates.size()) {
      const TightIncomingEdge candidate = frame.candidates[frame.next_candidate];
      ++frame.next_candidate;
      if (on_path[candidate.predecessor.value()]) {
        continue;
      }
      backward_edges.push_back(candidate.edge_id);
      backward_weights.push_back(candidate.weight);
      backward_vertices.push_back(candidate.predecessor);
      on_path[candidate.predecessor.value()] = true;
      stack.push_back(BacktrackingFrame{
          candidate.predecessor,
          tight_incoming_candidates(graph, result, candidate.predecessor),
          0U,
      });
      advanced = true;
      break;
    }
    if (advanced) {
      continue;
    }

    on_path[frame.vertex.value()] = false;
    stack.pop_back();
    backward_vertices.pop_back();
    if (!backward_edges.empty()) {
      backward_edges.pop_back();
      backward_weights.pop_back();
    }
  }

  throw std::logic_error{"finite distance labels contain no tight path to a source"};
}

bool validate_reconstructed_path(
    const WeightedGraph& graph,
    const SsspResult& result,
    const VertexId target,
    const ReconstructedPath& path) {
  if (!result_shape_is_valid(graph, result) ||
      !is_valid_vertex_id(target, graph.vertex_count()) || path.vertices.empty() ||
      path.vertices.back() != target || path.edge_ids.size() + 1U != path.vertices.size() ||
      !std::binary_search(
          result.sources.begin(), result.sources.end(), path.vertices.front())) {
    return false;
  }

  std::vector<bool> vertex_seen(graph.vertex_count(), false);
  for (const VertexId vertex : path.vertices) {
    if (!is_valid_vertex_id(vertex, graph.vertex_count()) || vertex_seen[vertex.value()]) {
      return false;
    }
    vertex_seen[vertex.value()] = true;
  }

  const std::vector<EdgeDetails> edge_details = edge_details_by_id(graph);
  float cost = 0.0F;
  for (std::size_t edge_index = 0U; edge_index < path.edge_ids.size(); ++edge_index) {
    const EdgeId edge_id = path.edge_ids[edge_index];
    if (edge_id.value() >= graph.edge_count()) {
      return false;
    }
    const EdgeDetails& edge = edge_details[static_cast<std::size_t>(edge_id.value())];
    if (edge.source != path.vertices[edge_index] ||
        edge.destination != path.vertices[edge_index + 1U]) {
      return false;
    }
    if (result.distances[edge.source.value()] + edge.weight !=
        result.distances[edge.destination.value()]) {
      return false;
    }
    cost = cost + edge.weight;
  }
  return cost == path.cost && path.cost == result.distances[target.value()];
}

bool nonnegative_distance_within_ulps(
    const float left,
    const float right,
    const std::uint32_t maximum_ulps) noexcept {
  if (std::isnan(left) || std::isnan(right) || left < 0.0F || right < 0.0F) {
    return false;
  }
  if (left == right) {
    return true;
  }
  if (!std::isfinite(left) || !std::isfinite(right)) {
    return false;
  }
  const std::uint32_t left_bits = std::bit_cast<std::uint32_t>(left);
  const std::uint32_t right_bits = std::bit_cast<std::uint32_t>(right);
  const std::uint32_t difference =
      left_bits > right_bits ? left_bits - right_bits : right_bits - left_bits;
  return difference <= maximum_ulps;
}

}  // namespace bfnew
