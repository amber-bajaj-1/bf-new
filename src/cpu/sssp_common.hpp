#pragma once

#include "bfnew/graph.hpp"
#include "bfnew/sssp.hpp"

#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace bfnew::cpu::detail {

[[nodiscard]] inline std::vector<VertexId> canonicalize_sources(
    const WeightedGraph& graph,
    const std::span<const VertexId> sources) {
  if (sources.empty()) {
    throw std::invalid_argument{"SSSP requires at least one source vertex"};
  }

  std::vector<VertexId> canonical_sources(sources.begin(), sources.end());
  for (const VertexId source : canonical_sources) {
    if (!is_valid_vertex_id(source, graph.vertex_count())) {
      throw std::out_of_range{"SSSP source is outside the graph vertex range"};
    }
  }
  std::sort(canonical_sources.begin(), canonical_sources.end());
  canonical_sources.erase(
      std::unique(canonical_sources.begin(), canonical_sources.end()),
      canonical_sources.end());
  return canonical_sources;
}

[[nodiscard]] inline SsspResult make_initial_result(
    const WeightedGraph& graph,
    const std::span<const VertexId> sources) {
  SsspResult result;
  result.sources = canonicalize_sources(graph, sources);
  result.distances.assign(
      graph.vertex_count(), std::numeric_limits<float>::infinity());
  for (const VertexId source : result.sources) {
    result.distances[source.value()] = 0.0F;
  }
  return result;
}

}  // namespace bfnew::cpu::detail
