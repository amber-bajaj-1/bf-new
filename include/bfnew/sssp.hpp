#pragma once

#include "bfnew/types.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace bfnew {

class WeightedGraph;

struct SsspResult {
  std::vector<float> distances;
  std::vector<VertexId> sources;
};

struct ReconstructedPath {
  std::vector<VertexId> vertices;
  std::vector<EdgeId> edge_ids;
  float cost{};
};

[[nodiscard]] SsspResult dijkstra_oracle(
    const WeightedGraph& graph,
    std::span<const VertexId> sources);

[[nodiscard]] SsspResult synchronous_bellman_ford(
    const WeightedGraph& graph,
    std::span<const VertexId> sources);

[[nodiscard]] std::optional<ReconstructedPath> reconstruct_path_from_distances(
    const WeightedGraph& graph,
    const SsspResult& result,
    VertexId target);

[[nodiscard]] bool validate_reconstructed_path(
    const WeightedGraph& graph,
    const SsspResult& result,
    VertexId target,
    const ReconstructedPath& path);

[[nodiscard]] bool nonnegative_distance_within_ulps(
    float left,
    float right,
    std::uint32_t maximum_ulps = 4U) noexcept;

}  // namespace bfnew
