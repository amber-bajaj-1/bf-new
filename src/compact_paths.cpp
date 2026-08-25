#include "bfnew/compact_paths.hpp"

#include "bfnew/graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

struct IncomingChoice {
  EdgeId edge_id{};
  VertexId predecessor{};
  float weight{};
};

struct SearchFrame {
  VertexId vertex{};
  std::vector<IncomingChoice> choices;
  std::size_t cursor{};
};

struct EdgeDetails {
  VertexId source{};
  VertexId destination{};
  float weight{};
};

[[nodiscard]] bool valid_disposition(
    const ExpansionQueryDisposition disposition) noexcept {
  switch (disposition) {
    case ExpansionQueryDisposition::reached:
    case ExpansionQueryDisposition::unreachable_in_full_region:
    case ExpansionQueryDisposition::expansion_limit:
    case ExpansionQueryDisposition::region_stalled:
    case ExpansionQueryDisposition::identity_or_count_overflow:
    case ExpansionQueryDisposition::engine_failure:
      return true;
  }
  return false;
}

[[nodiscard]] bool valid_distance_value(const float value) noexcept {
  return !std::isnan(value) && value >= 0.0F;
}

[[nodiscard]] bool valid_distance_image(
    const WeightedGraph& graph,
    const RouteQuery& query,
    const std::span<const float> distances) noexcept {
  if (distances.size() != graph.vertex_count()) {
    return false;
  }
  for (const float distance : distances) {
    if (!valid_distance_value(distance)) {
      return false;
    }
  }
  return std::ranges::all_of(query.sources, [&](const VertexId source) {
    return distances[source.value()] == 0.0F;
  });
}

[[nodiscard]] std::vector<bool> selected_vertex_mask(
    const WeightedGraph& graph,
    const RouteQuery& query) {
  std::vector<bool> selected_tiles(graph.tile_coordinates().size(), false);
  for (const TileId tile : query.selected_tiles) {
    selected_tiles[tile.value()] = true;
  }
  std::vector<bool> selected_vertices(graph.vertex_count(), false);
  for (std::size_t vertex = 0U; vertex < graph.vertex_count(); ++vertex) {
    selected_vertices[vertex] = selected_tiles[graph.owner_tiles()[vertex].value()];
  }
  return selected_vertices;
}

[[nodiscard]] std::vector<bool> source_mask(
    const WeightedGraph& graph,
    const RouteQuery& query) {
  std::vector<bool> result(graph.vertex_count(), false);
  for (const VertexId source : query.sources) {
    result[source.value()] = true;
  }
  return result;
}

[[nodiscard]] std::vector<IncomingChoice> tight_choices(
    const WeightedGraph& graph,
    const std::span<const float> distances,
    const std::vector<bool>& selected,
    const VertexId vertex) {
  std::vector<IncomingChoice> result;
  const IncomingCscView incoming = graph.incoming();
  const std::size_t begin =
      static_cast<std::size_t>(incoming.column_offsets[vertex.value()]);
  const std::size_t end =
      static_cast<std::size_t>(incoming.column_offsets[vertex.value() + 1U]);
  result.reserve(end - begin);
  for (std::size_t position = begin; position < end; ++position) {
    const VertexId predecessor = incoming.sources[position];
    if (!selected[predecessor.value()] ||
        !std::isfinite(distances[predecessor.value()])) {
      continue;
    }
    if (distances[predecessor.value()] + incoming.weights[position] ==
        distances[vertex.value()]) {
      result.push_back(IncomingChoice{
          incoming.edge_ids[position],
          predecessor,
          incoming.weights[position]});
    }
  }
  std::sort(
      result.begin(),
      result.end(),
      [](const IncomingChoice& left, const IncomingChoice& right) {
        return left.edge_id < right.edge_id;
      });
  return result;
}

[[nodiscard]] CompactTargetPath reconstruct_target(
    const WeightedGraph& graph,
    const std::span<const float> distances,
    const std::vector<bool>& selected,
    const std::vector<bool>& sources,
    const VertexId target) {
  CompactTargetPath output;
  output.summary.target = target;
  output.summary.distance = distances[target.value()];
  if (!std::isfinite(output.summary.distance)) {
    output.summary.reached = CompactTargetReachStatus::not_reached;
    output.summary.reconstruction = CompactPathStatus::unreachable;
    return output;
  }

  output.summary.reached = CompactTargetReachStatus::reached;
  std::vector<bool> on_path(graph.vertex_count(), false);
  std::vector<VertexId> reverse_vertices{target};
  std::vector<EdgeId> reverse_edges;
  std::vector<SearchFrame> stack;
  on_path[target.value()] = true;
  stack.push_back(SearchFrame{
      target, tight_choices(graph, distances, selected, target), 0U});

  while (!stack.empty()) {
    SearchFrame& frame = stack.back();
    if (sources[frame.vertex.value()]) {
      output.vertices.assign(reverse_vertices.rbegin(), reverse_vertices.rend());
      output.edge_ids.assign(reverse_edges.rbegin(), reverse_edges.rend());
      output.distance_labels.reserve(output.vertices.size());
      for (const VertexId vertex : output.vertices) {
        output.distance_labels.push_back(distances[vertex.value()]);
      }
      if (output.edge_ids.size() >
          std::numeric_limits<std::uint32_t>::max()) {
        output.vertices.clear();
        output.distance_labels.clear();
        output.edge_ids.clear();
        output.summary.reconstruction = CompactPathStatus::path_length_overflow;
        return output;
      }
      output.summary.selected_source = output.vertices.front();
      output.summary.has_selected_source = 1U;
      output.summary.path_length =
          static_cast<std::uint32_t>(output.edge_ids.size());
      output.summary.reconstruction = CompactPathStatus::complete;
      return output;
    }

    bool advanced = false;
    while (frame.cursor < frame.choices.size()) {
      const IncomingChoice choice = frame.choices[frame.cursor++];
      if (on_path[choice.predecessor.value()]) {
        continue;
      }
      reverse_edges.push_back(choice.edge_id);
      reverse_vertices.push_back(choice.predecessor);
      on_path[choice.predecessor.value()] = true;
      stack.push_back(SearchFrame{
          choice.predecessor,
          tight_choices(graph, distances, selected, choice.predecessor),
          0U});
      advanced = true;
      break;
    }
    if (advanced) {
      continue;
    }

    const VertexId abandoned = frame.vertex;
    stack.pop_back();
    on_path[abandoned.value()] = false;
    reverse_vertices.pop_back();
    if (!stack.empty()) {
      reverse_edges.pop_back();
    }
  }

  output.summary.reconstruction = CompactPathStatus::no_tight_path;
  return output;
}

[[nodiscard]] std::vector<EdgeDetails> edge_details(const WeightedGraph& graph) {
  if (graph.edge_count() > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error{"stable edge table exceeds host size range"};
  }
  std::vector<EdgeDetails> result(static_cast<std::size_t>(graph.edge_count()));
  const OutgoingCsrView outgoing = graph.outgoing();
  for (std::size_t source = 0U; source < graph.vertex_count(); ++source) {
    const std::size_t begin =
        static_cast<std::size_t>(outgoing.row_offsets[source]);
    const std::size_t end =
        static_cast<std::size_t>(outgoing.row_offsets[source + 1U]);
    for (std::size_t position = begin; position < end; ++position) {
      result[static_cast<std::size_t>(outgoing.edge_ids[position].value())] =
          EdgeDetails{
              VertexId{static_cast<std::uint32_t>(source)},
              outgoing.destinations[position],
              outgoing.weights[position]};
    }
  }
  return result;
}

[[nodiscard]] CompactPathValidationResult validation_error(
    const CompactPathValidationErrorCode code,
    const std::size_t target = CompactPathValidationResult::no_position,
    const std::size_t path = CompactPathValidationResult::no_position) noexcept {
  return CompactPathValidationResult{code, target, path};
}

[[nodiscard]] bool valid_reach_status(
    const CompactTargetReachStatus status) noexcept {
  return status == CompactTargetReachStatus::not_reached ||
         status == CompactTargetReachStatus::reached;
}

[[nodiscard]] bool valid_path_status(const CompactPathStatus status) noexcept {
  switch (status) {
    case CompactPathStatus::complete:
    case CompactPathStatus::unreachable:
    case CompactPathStatus::query_terminal_failure:
    case CompactPathStatus::no_tight_path:
    case CompactPathStatus::path_length_overflow:
      return true;
  }
  return false;
}

[[nodiscard]] CompactPathPayload payload_from_result(
    const CompactQueryResult& result) {
  return CompactPathPayload{
      result.query_id,
      result.expansion_generation,
      result.target_terminal_to_target,
      result.targets};
}

[[nodiscard]] CompactPathValidationResult validate_payload_impl(
    const WeightedGraph& graph,
    const RouteQuery& query,
    const CompactPathPayload& payload) {
  if (!validate_weighted_graph(graph).ok()) {
    return validation_error(
        CompactPathValidationErrorCode::graph_validation_failed);
  }
  if (!validate_route_query(graph, query).ok()) {
    return validation_error(
        CompactPathValidationErrorCode::query_validation_failed);
  }
  if (payload.query_id != query.query_id) {
    return validation_error(
        CompactPathValidationErrorCode::query_identity_mismatch);
  }
  if (payload.expansion_generation != query.expansion_generation) {
    return validation_error(
        CompactPathValidationErrorCode::expansion_generation_mismatch);
  }
  if (payload.target_terminal_to_target != query.target_terminal_to_target) {
    return validation_error(
        CompactPathValidationErrorCode::terminal_map_mismatch);
  }
  if (payload.targets.size() != query.targets.size()) {
    return validation_error(
        CompactPathValidationErrorCode::target_count_mismatch);
  }

  const std::vector<bool> selected = selected_vertex_mask(graph, query);
  const std::vector<EdgeDetails> edges = edge_details(graph);
  for (std::size_t target_index = 0U; target_index < payload.targets.size();
       ++target_index) {
    const CompactTargetPath& path = payload.targets[target_index];
    const CompactTargetSummary& summary = path.summary;
    if (summary.target != query.targets[target_index]) {
      return validation_error(
          CompactPathValidationErrorCode::target_identity_mismatch,
          target_index);
    }
    if (!valid_reach_status(summary.reached)) {
      return validation_error(
          CompactPathValidationErrorCode::invalid_reach_status,
          target_index);
    }
    if (!valid_path_status(summary.reconstruction)) {
      return validation_error(
          CompactPathValidationErrorCode::invalid_reconstruction_status,
          target_index);
    }
    if (summary.has_selected_source > 1U) {
      return validation_error(
          CompactPathValidationErrorCode::invalid_selected_source_flag,
          target_index);
    }
    if (!valid_distance_value(summary.distance)) {
      return validation_error(
          CompactPathValidationErrorCode::invalid_distance, target_index);
    }

    const bool arenas_empty = path.vertices.empty() &&
                              path.distance_labels.empty() &&
                              path.edge_ids.empty();
    switch (summary.reconstruction) {
      case CompactPathStatus::complete: {
        if (summary.reached != CompactTargetReachStatus::reached ||
            !std::isfinite(summary.distance) ||
            summary.has_selected_source != 1U) {
          return validation_error(
              CompactPathValidationErrorCode::path_shape_mismatch,
              target_index);
        }
        if (!is_valid_vertex_id(summary.selected_source, graph.vertex_count())) {
          return validation_error(
              CompactPathValidationErrorCode::selected_source_out_of_range,
              target_index);
        }
        if (!std::binary_search(
                query.sources.begin(),
                query.sources.end(),
                summary.selected_source)) {
          return validation_error(
              CompactPathValidationErrorCode::selected_source_not_query_source,
              target_index);
        }
        if (path.vertices.size() != path.distance_labels.size() ||
            path.vertices.size() != path.edge_ids.size() + 1U) {
          return validation_error(
              CompactPathValidationErrorCode::path_shape_mismatch,
              target_index);
        }
        if (path.edge_ids.size() != summary.path_length) {
          return validation_error(
              CompactPathValidationErrorCode::path_length_mismatch,
              target_index);
        }
        if (path.vertices.empty() ||
            path.vertices.front() != summary.selected_source) {
          return validation_error(
              CompactPathValidationErrorCode::source_termination_mismatch,
              target_index);
        }
        if (path.vertices.back() != summary.target) {
          return validation_error(
              CompactPathValidationErrorCode::target_termination_mismatch,
              target_index);
        }

        std::vector<bool> seen(graph.vertex_count(), false);
        for (std::size_t vertex_index = 0U;
             vertex_index < path.vertices.size(); ++vertex_index) {
          const VertexId vertex = path.vertices[vertex_index];
          if (!is_valid_vertex_id(vertex, graph.vertex_count())) {
            return validation_error(
                CompactPathValidationErrorCode::vertex_out_of_range,
                target_index,
                vertex_index);
          }
          if (!selected[vertex.value()]) {
            return validation_error(
                CompactPathValidationErrorCode::vertex_outside_selected_region,
                target_index,
                vertex_index);
          }
          if (seen[vertex.value()]) {
            return validation_error(
                CompactPathValidationErrorCode::repeated_vertex,
                target_index,
                vertex_index);
          }
          seen[vertex.value()] = true;
          if (!std::isfinite(path.distance_labels[vertex_index]) ||
              path.distance_labels[vertex_index] < 0.0F) {
            return validation_error(
                CompactPathValidationErrorCode::invalid_distance,
                target_index,
                vertex_index);
          }
        }
        if (path.distance_labels.front() != 0.0F ||
            path.distance_labels.back() != summary.distance) {
          return validation_error(
              CompactPathValidationErrorCode::target_distance_mismatch,
              target_index);
        }

        float cost = 0.0F;
        for (std::size_t edge_index = 0U; edge_index < path.edge_ids.size();
             ++edge_index) {
          const EdgeId edge_id = path.edge_ids[edge_index];
          if (edge_id.value() >= graph.edge_count()) {
            return validation_error(
                CompactPathValidationErrorCode::edge_id_out_of_range,
                target_index,
                edge_index);
          }
          const EdgeDetails& edge =
              edges[static_cast<std::size_t>(edge_id.value())];
          if (edge.source != path.vertices[edge_index] ||
              edge.destination != path.vertices[edge_index + 1U]) {
            return validation_error(
                CompactPathValidationErrorCode::edge_continuity_mismatch,
                target_index,
                edge_index);
          }
          if (path.distance_labels[edge_index] + edge.weight !=
              path.distance_labels[edge_index + 1U]) {
            return validation_error(
                CompactPathValidationErrorCode::non_tight_edge,
                target_index,
                edge_index);
          }
          cost = cost + edge.weight;
        }
        if (cost != summary.distance) {
          return validation_error(
              CompactPathValidationErrorCode::reported_cost_mismatch,
              target_index);
        }
        break;
      }
      case CompactPathStatus::unreachable:
        if (summary.reached != CompactTargetReachStatus::not_reached ||
            std::isfinite(summary.distance) ||
            summary.has_selected_source != 0U || summary.path_length != 0U ||
            !arenas_empty) {
          return validation_error(
              CompactPathValidationErrorCode::inconsistent_unreachable_result,
              target_index);
        }
        break;
      case CompactPathStatus::query_terminal_failure:
        if (summary.reached != CompactTargetReachStatus::not_reached ||
            std::isfinite(summary.distance) ||
            summary.has_selected_source != 0U || summary.path_length != 0U ||
            !arenas_empty) {
          return validation_error(
              CompactPathValidationErrorCode::inconsistent_failure_result,
              target_index);
        }
        break;
      case CompactPathStatus::no_tight_path:
      case CompactPathStatus::path_length_overflow:
        if (summary.reached != CompactTargetReachStatus::reached ||
            !std::isfinite(summary.distance) ||
            summary.has_selected_source != 0U || summary.path_length != 0U ||
            !arenas_empty) {
          return validation_error(
              CompactPathValidationErrorCode::inconsistent_reconstruction_failure,
              target_index);
        }
        break;
    }
  }
  return {};
}

void checked_add(
    std::uint64_t& destination,
    const std::uint64_t value,
    const char* const message) {
  if (value > std::numeric_limits<std::uint64_t>::max() - destination) {
    throw std::overflow_error{message};
  }
  destination += value;
}

[[nodiscard]] std::uint64_t checked_bytes(
    const std::size_t count,
    const std::uint64_t width,
    const char* const message) {
  if (count > std::numeric_limits<std::uint64_t>::max() ||
      (width != 0U &&
       static_cast<std::uint64_t>(count) >
           std::numeric_limits<std::uint64_t>::max() / width)) {
    throw std::overflow_error{message};
  }
  return static_cast<std::uint64_t>(count) * width;
}

}  // namespace

CompactQueryResult make_compact_query_result(
    CompactPathPayload payload,
    const ExpansionQueryDisposition disposition) {
  if (!valid_disposition(disposition)) {
    throw std::invalid_argument{"compact result disposition is invalid"};
  }
  CompactQueryResult result;
  result.query_id = payload.query_id;
  result.expansion_generation = payload.expansion_generation;
  result.disposition = disposition;
  result.target_terminal_to_target =
      std::move(payload.target_terminal_to_target);
  result.targets = std::move(payload.targets);
  return result;
}

bool compact_path_payload_complete(
    const CompactPathPayload& payload) noexcept {
  return !payload.targets.empty() &&
         std::ranges::all_of(
             payload.targets,
             [](const CompactTargetPath& target) {
               return target.summary.reached ==
                          CompactTargetReachStatus::reached &&
                      target.summary.reconstruction ==
                          CompactPathStatus::complete;
             });
}

CompactPathPayload reconstruct_compact_path_payload(
    const WeightedGraph& graph,
    const RouteQuery& query,
    const std::span<const float> final_distances) {
  if (!validate_weighted_graph(graph).ok() ||
      !validate_route_query(graph, query).ok()) {
    throw std::invalid_argument{
        "compact reconstruction requires a valid spatial query graph"};
  }
  if (!valid_distance_image(graph, query, final_distances)) {
    throw std::invalid_argument{
        "compact reconstruction requires a valid graph-sized distance image"};
  }

  CompactPathPayload payload;
  payload.query_id = query.query_id;
  payload.expansion_generation = query.expansion_generation;
  payload.target_terminal_to_target = query.target_terminal_to_target;
  payload.targets.reserve(query.targets.size());
  const std::vector<bool> selected = selected_vertex_mask(graph, query);
  const std::vector<bool> sources = source_mask(graph, query);
  for (const VertexId target : query.targets) {
    payload.targets.push_back(reconstruct_target(
        graph, final_distances, selected, sources, target));
  }

  const CompactPathValidationResult validation =
      validate_compact_path_payload(graph, query, payload);
  if (!validation.ok()) {
    throw std::logic_error{"compact reconstruction produced an invalid payload"};
  }
  return payload;
}

CompactQueryResult reconstruct_compact_query_paths(
    const WeightedGraph& graph,
    const RouteQuery& query,
    const std::span<const float> final_distances,
    const ExpansionQueryDisposition disposition) {
  if (!valid_disposition(disposition)) {
    throw std::invalid_argument{"compact result disposition is invalid"};
  }
  CompactQueryResult output = make_compact_query_result(
      reconstruct_compact_path_payload(graph, query, final_distances),
      disposition);
  const CompactPathValidationResult validation =
      validate_compact_query_result(graph, query, output);
  if (!validation.ok()) {
    throw std::logic_error{
        "compact reconstruction produced a disposition-inconsistent result"};
  }
  return output;
}

CompactQueryResult make_failed_compact_query_result(
    const RouteQuery& query,
    const ExpansionQueryDisposition disposition) {
  if (!valid_disposition(disposition) ||
      disposition == ExpansionQueryDisposition::reached ||
      disposition ==
          ExpansionQueryDisposition::unreachable_in_full_region) {
    throw std::invalid_argument{
        "a reached, full-region-unreachable, or invalid query cannot publish "
        "a label-free failure"};
  }
  CompactPathPayload payload;
  payload.query_id = query.query_id;
  payload.expansion_generation = query.expansion_generation;
  payload.target_terminal_to_target = query.target_terminal_to_target;
  payload.targets.reserve(query.targets.size());
  for (const VertexId target : query.targets) {
    CompactTargetPath path;
    path.summary.target = target;
    path.summary.reconstruction = CompactPathStatus::query_terminal_failure;
    payload.targets.push_back(std::move(path));
  }
  return make_compact_query_result(std::move(payload), disposition);
}

std::vector<CompactQueryResult> extract_host_compact_paths(
    const WeightedGraph& graph,
    const std::span<ExpansionQueryOutcome> outcomes) {
  if (!validate_weighted_graph(graph).ok()) {
    throw std::invalid_argument{
        "compact extraction requires a deeply valid weighted graph"};
  }
  for (std::size_t index = 0U; index < outcomes.size(); ++index) {
    if (!validate_route_query(graph, outcomes[index].final_query).ok() ||
        (index != 0U &&
         !(outcomes[index - 1U].final_query.query_id <
           outcomes[index].final_query.query_id))) {
      throw std::invalid_argument{
          "compact extraction requires canonical terminal query outcomes"};
    }
  }

  std::vector<CompactQueryResult> output;
  output.reserve(outcomes.size());
  for (ExpansionQueryOutcome& outcome : outcomes) {
    CompactQueryResult result;
    if (outcome.compact_paths.has_value()) {
      result = make_compact_query_result(
          std::move(*outcome.compact_paths), outcome.disposition);
      outcome.compact_paths.reset();
      const CompactPathValidationResult validation =
          validate_compact_query_result(graph, outcome.final_query, result);
      if (!validation.ok()) {
        throw std::logic_error{"terminal compact payload failed validation"};
      }
      if (!outcome.final_distances.empty() &&
          !validate_compact_query_result_against_distances(
               graph, outcome.final_query, result, outcome.final_distances)
               .ok()) {
        throw std::logic_error{
            "terminal compact payload disagrees with diagnostic distances"};
      }
    } else if (!outcome.final_distances.empty()) {
      if (outcome.disposition == ExpansionQueryDisposition::reached ||
          outcome.disposition ==
              ExpansionQueryDisposition::unreachable_in_full_region) {
        result = reconstruct_compact_query_paths(
            graph,
            outcome.final_query,
            outcome.final_distances,
            outcome.disposition);
      } else {
        result = make_failed_compact_query_result(
            outcome.final_query, outcome.disposition);
      }
    } else {
      if (outcome.reached()) {
        throw std::logic_error{
            "reached compact extraction has neither paths nor distances"};
      }
      result = make_failed_compact_query_result(
          outcome.final_query, outcome.disposition);
    }
    std::vector<float>{}.swap(outcome.final_distances);
    output.push_back(std::move(result));
  }
  return output;
}

CompactTransferAccounting measure_compact_transfer(
    const std::span<const CompactPathPayload> payloads) {
  CompactTransferAccounting output;
  for (const CompactPathPayload& payload : payloads) {
    checked_add(
        output.summary_bytes,
        checked_bytes(
            payload.targets.size(),
            sizeof(CompactTargetSummary),
            "compact summary bytes overflowed"),
        "compact summary byte aggregate overflowed");
    for (const CompactTargetPath& target : payload.targets) {
      checked_add(
          output.vertex_bytes,
          checked_bytes(
              target.vertices.size(),
              sizeof(std::uint32_t),
              "compact vertex bytes overflowed"),
          "compact vertex byte aggregate overflowed");
      checked_add(
          output.distance_label_bytes,
          checked_bytes(
              target.distance_labels.size(),
              sizeof(float),
              "compact distance-label bytes overflowed"),
          "compact distance-label byte aggregate overflowed");
      // Device graphs are 32-bit representable.  Stable IDs are widened into
      // the host's EdgeId after transfer, so this is actual D2H traffic.
      checked_add(
          output.edge_id_bytes,
          checked_bytes(
              target.edge_ids.size(),
              sizeof(std::uint32_t),
              "compact edge-ID bytes overflowed"),
          "compact edge-ID byte aggregate overflowed");
    }
  }
  checked_add(output.total_bytes, output.summary_bytes, "compact bytes overflowed");
  checked_add(output.total_bytes, output.vertex_bytes, "compact bytes overflowed");
  checked_add(
      output.total_bytes,
      output.distance_label_bytes,
      "compact bytes overflowed");
  checked_add(output.total_bytes, output.edge_id_bytes, "compact bytes overflowed");
  return output;
}

CompactPathValidationResult validate_compact_path_payload(
    const WeightedGraph& graph,
    const RouteQuery& query,
    const CompactPathPayload& payload) {
  return validate_payload_impl(graph, query, payload);
}

CompactPathValidationResult validate_compact_query_result(
    const WeightedGraph& graph,
    const RouteQuery& query,
    const CompactQueryResult& result) {
  if (!valid_disposition(result.disposition)) {
    return validation_error(
        CompactPathValidationErrorCode::invalid_query_disposition);
  }
  const CompactPathValidationResult payload_validation =
      validate_payload_impl(graph, query, payload_from_result(result));
  if (!payload_validation.ok()) {
    return payload_validation;
  }
  const bool all_complete = std::ranges::all_of(
      result.targets,
      [](const CompactTargetPath& target) {
        return target.summary.reconstruction == CompactPathStatus::complete;
      });
  const bool any_unreachable = std::ranges::any_of(
      result.targets,
      [](const CompactTargetPath& target) {
        return target.summary.reconstruction == CompactPathStatus::unreachable;
      });
  const bool all_complete_or_unreachable = std::ranges::all_of(
      result.targets,
      [](const CompactTargetPath& target) {
        return target.summary.reconstruction == CompactPathStatus::complete ||
               target.summary.reconstruction == CompactPathStatus::unreachable;
      });
  const bool any_reconstruction_failure = std::ranges::any_of(
      result.targets,
      [](const CompactTargetPath& target) {
        return target.summary.reconstruction ==
                   CompactPathStatus::query_terminal_failure ||
               target.summary.reconstruction ==
                   CompactPathStatus::no_tight_path ||
               target.summary.reconstruction ==
                   CompactPathStatus::path_length_overflow;
      });
  const bool all_terminal_failure = std::ranges::all_of(
      result.targets,
      [](const CompactTargetPath& target) {
        return target.summary.reconstruction ==
               CompactPathStatus::query_terminal_failure;
      });
  switch (result.disposition) {
    case ExpansionQueryDisposition::reached:
      if (!all_complete) {
        return validation_error(
            CompactPathValidationErrorCode::reached_query_incomplete);
      }
      break;
    case ExpansionQueryDisposition::unreachable_in_full_region:
      if (!any_unreachable || !all_complete_or_unreachable) {
        return validation_error(
            CompactPathValidationErrorCode::inconsistent_unreachable_result);
      }
      break;
    case ExpansionQueryDisposition::engine_failure:
      if (!any_reconstruction_failure) {
        return validation_error(
            CompactPathValidationErrorCode::failed_query_reported_complete);
      }
      break;
    case ExpansionQueryDisposition::expansion_limit:
    case ExpansionQueryDisposition::region_stalled:
    case ExpansionQueryDisposition::identity_or_count_overflow:
      if (!all_terminal_failure) {
        return validation_error(
            CompactPathValidationErrorCode::inconsistent_failure_result);
      }
      break;
  }
  return {};
}

CompactPathValidationResult validate_compact_query_result_against_distances(
    const WeightedGraph& graph,
    const RouteQuery& query,
    const CompactQueryResult& result,
    const std::span<const float> final_distances) {
  const CompactPathValidationResult structural =
      validate_compact_query_result(graph, query, result);
  if (!structural.ok()) {
    return structural;
  }
  if (final_distances.size() != graph.vertex_count()) {
    return validation_error(
        CompactPathValidationErrorCode::distance_image_size_mismatch);
  }
  if (!valid_distance_image(graph, query, final_distances)) {
    return validation_error(
        CompactPathValidationErrorCode::invalid_distance_image);
  }
  for (std::size_t target_index = 0U; target_index < result.targets.size();
       ++target_index) {
    const CompactTargetPath& path = result.targets[target_index];
    switch (path.summary.reconstruction) {
      case CompactPathStatus::complete:
        for (std::size_t vertex_index = 0U;
             vertex_index < path.vertices.size(); ++vertex_index) {
          if (path.distance_labels[vertex_index] !=
              final_distances[path.vertices[vertex_index].value()]) {
            return validation_error(
                CompactPathValidationErrorCode::target_distance_mismatch,
                target_index,
                vertex_index);
          }
        }
        break;
      case CompactPathStatus::unreachable:
        if (std::isfinite(final_distances[path.summary.target.value()])) {
          return validation_error(
              CompactPathValidationErrorCode::target_distance_mismatch,
              target_index);
        }
        break;
      case CompactPathStatus::no_tight_path:
      case CompactPathStatus::path_length_overflow:
        if (path.summary.distance !=
            final_distances[path.summary.target.value()]) {
          return validation_error(
              CompactPathValidationErrorCode::target_distance_mismatch,
              target_index);
        }
        break;
      case CompactPathStatus::query_terminal_failure:
        break;
    }
  }
  return {};
}

}  // namespace bfnew
