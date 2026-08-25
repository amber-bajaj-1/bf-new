#pragma once

#include "bfnew/batched_expansion.hpp"
#include "bfnew/compact_path_types.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace bfnew {

struct CompactQueryResult {
  QueryId query_id{};
  std::uint32_t expansion_generation{};
  ExpansionQueryDisposition disposition{
      ExpansionQueryDisposition::engine_failure};
  // Canonical target order matches RouteQuery::targets.  Duplicate input
  // terminals are recovered through the unchanged terminal-to-target map.
  std::vector<std::uint32_t> target_terminal_to_target;
  std::vector<CompactTargetPath> targets;

  bool operator==(const CompactQueryResult&) const = default;
};

// Attaches a final expansion disposition without changing the payload's
// canonical identity or arenas.  Used by expansion orchestration after it has
// retained a reached lane before workspace reuse.
[[nodiscard]] CompactQueryResult make_compact_query_result(
    CompactPathPayload payload,
    ExpansionQueryDisposition disposition);

[[nodiscard]] bool compact_path_payload_complete(
    const CompactPathPayload& payload) noexcept;

// Independently reconstructs compact paths from final labels and incoming
// CSC.  Tight predecessors are tried in stable-EdgeId order with path-local
// cycle detection and real backtracking.  Only vertices owned by one of the
// query's selected tiles may enter a path.
[[nodiscard]] CompactPathPayload reconstruct_compact_path_payload(
    const WeightedGraph& graph,
    const RouteQuery& query,
    std::span<const float> final_distances);

[[nodiscard]] CompactQueryResult reconstruct_compact_query_paths(
    const WeightedGraph& graph,
    const RouteQuery& query,
    std::span<const float> final_distances,
    ExpansionQueryDisposition disposition);

// Produces an explicit result for a query whose terminal engine/identity
// failure published no distance image.  A reached query may not use this API.
[[nodiscard]] CompactQueryResult make_failed_compact_query_result(
    const RouteQuery& query,
    ExpansionQueryDisposition disposition);

// Portable Phase 18 adapter for Phase 17 outcomes. A reached outcome must own
// either its generation-bound compact payload or a graph-sized diagnostic
// image. It never copies a batch lane matrix into the returned compact result.
[[nodiscard]] std::vector<CompactQueryResult> extract_host_compact_paths(
    const WeightedGraph& graph,
    std::span<ExpansionQueryOutcome> outcomes);

// Checked serialized byte accounting for the mandatory summaries and exact
// compact path arenas.  Query metadata and terminal maps are already resident
// on the host and are not counted as device result transfer.
[[nodiscard]] CompactTransferAccounting measure_compact_transfer(
    std::span<const CompactPathPayload> payloads);

enum class CompactPathValidationErrorCode : std::uint8_t {
  none,
  graph_validation_failed,
  query_validation_failed,
  invalid_query_disposition,
  query_identity_mismatch,
  expansion_generation_mismatch,
  terminal_map_mismatch,
  target_count_mismatch,
  target_identity_mismatch,
  invalid_reach_status,
  invalid_reconstruction_status,
  invalid_distance,
  reached_query_incomplete,
  failed_query_reported_complete,
  inconsistent_unreachable_result,
  inconsistent_failure_result,
  inconsistent_reconstruction_failure,
  invalid_selected_source_flag,
  selected_source_out_of_range,
  selected_source_not_query_source,
  path_shape_mismatch,
  path_length_mismatch,
  vertex_out_of_range,
  vertex_outside_selected_region,
  repeated_vertex,
  source_termination_mismatch,
  target_termination_mismatch,
  edge_id_out_of_range,
  edge_continuity_mismatch,
  reported_cost_mismatch,
  distance_image_size_mismatch,
  invalid_distance_image,
  target_distance_mismatch,
  non_tight_edge,
};

struct CompactPathValidationResult {
  static constexpr std::size_t no_position =
      std::numeric_limits<std::size_t>::max();

  CompactPathValidationErrorCode code{CompactPathValidationErrorCode::none};
  std::size_t target_index{no_position};
  std::size_t path_index{no_position};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == CompactPathValidationErrorCode::none;
  }
};

// Validates target/source termination, simple paths, stable-edge continuity,
// exact tightness against the transferred per-path labels, reported cost, and
// selected-region membership without a diagnostic full label image.
[[nodiscard]] CompactPathValidationResult validate_compact_query_result(
    const WeightedGraph& graph,
    const RouteQuery& query,
    const CompactQueryResult& result);

[[nodiscard]] CompactPathValidationResult validate_compact_path_payload(
    const WeightedGraph& graph,
    const RouteQuery& query,
    const CompactPathPayload& payload);

// Adds exact per-edge tightness checks against a sampled full GPU distance
// image.  This diagnostic API is intentionally separate from the compact
// production path; callers must not obtain every lane matrix merely to use the
// ordinary validator.
[[nodiscard]] CompactPathValidationResult
validate_compact_query_result_against_distances(
    const WeightedGraph& graph,
    const RouteQuery& query,
    const CompactQueryResult& result,
    std::span<const float> final_distances);

}  // namespace bfnew
