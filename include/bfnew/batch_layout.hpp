#pragma once

#include "bfnew/batch_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace bfnew {

enum class BatchRunRepresentation : std::uint8_t {
  retained_per_run_masks,
  compact_nonzero_descriptors,
  // No host run image is built. A GPU engine derives the selected run masks
  // from the dense tile masks during its initialization kernel. This keeps a
  // dense device mask for the hot repeated rounds without a graph-wide H2D
  // copy or clear on every batch.
  device_materialized_run_masks,
};

struct BatchVertexRange {
  std::uint32_t begin{};
  std::uint32_t end{};
  LaneMask lane_mask{};

  constexpr bool operator==(const BatchVertexRange&) const noexcept = default;
};

struct RunLaneMaskDescriptor {
  std::uint32_t run_id{};
  LaneMask lane_mask{};

  constexpr bool operator==(const RunLaneMaskDescriptor&) const noexcept = default;
};

struct BatchRunPreparationReport {
  std::uint64_t csr_runs_visited{};
  std::uint64_t csc_runs_visited{};
  std::uint64_t active_csr_runs{};
  std::uint64_t active_csc_runs{};
  std::uint64_t csr_lane_edge_pairs{};
  std::uint64_t csc_lane_edge_pairs{};
  std::uint64_t retained_entries_initialized{};
  std::uint64_t retained_entries_cleared{};
  std::uint64_t retained_entries_written{};
  std::uint64_t descriptor_entries_written{};
  std::uint64_t csr_retained_entries_initialized{};
  std::uint64_t csc_retained_entries_initialized{};
  std::uint64_t csr_retained_entries_cleared{};
  std::uint64_t csc_retained_entries_cleared{};
  std::uint64_t csr_retained_entries_written{};
  std::uint64_t csc_retained_entries_written{};
  std::uint64_t csr_descriptor_entries_written{};
  std::uint64_t csc_descriptor_entries_written{};

  constexpr bool operator==(const BatchRunPreparationReport&) const noexcept =
      default;
};

struct BatchCompactMappingReport {
  std::uint64_t entries_initialized{};
  std::uint64_t entries_cleared{};
  std::uint64_t entries_written{};

  constexpr bool operator==(const BatchCompactMappingReport&) const noexcept =
      default;
};

// Owning, reusable host preparation image for one planned batch. Invalid
// padded lanes have empty source/target slices and no tile/run mask bits.
struct BatchDeviceDescription {
  std::uint32_t lane_width{};
  LaneMask valid_lane_mask{};
  LaneMask reached_lane_mask{};
  LaneMask miss_lane_mask{};

  std::vector<std::uint32_t> query_ids_by_lane;
  std::vector<std::uint32_t> expansion_generations_by_lane;
  std::vector<std::uint32_t> source_offsets;
  std::vector<std::uint32_t> sources;
  std::vector<std::uint32_t> target_offsets;
  std::vector<std::uint32_t> targets;
  std::vector<std::uint64_t> selected_vertex_counts_by_lane;
  std::vector<std::uint64_t> selected_edge_estimates_by_lane;

  std::vector<std::uint32_t> union_tiles;
  std::vector<LaneMask> tile_lane_masks;
  std::vector<BatchVertexRange> selected_vertex_ranges;

  // Optional compact-label mapping. For a selected tile and global vertex v,
  // compact index = v - compact_vertex_biases_by_tile[tile]. The stored bias
  // is (global tile begin - packed tile begin), so one uint32_t per tile is
  // sufficient even though the Phase 8 device graph has no tile offsets.
  // Values are meaningful only for `touched_compact_tiles`. The ledger permits
  // O(|old union| + |new union|) reuse without clearing all tiles. Batch
  // preparation invalidates, but preserves, this storage until the explicit
  // compact mapping builder runs.
  std::vector<std::uint32_t> compact_vertex_biases_by_tile;
  std::vector<std::uint32_t> touched_compact_tiles;
  bool compact_vertex_mapping_valid{};
  BatchCompactMappingReport compact_mapping_report{};

  BatchRunRepresentation run_representation{
      BatchRunRepresentation::retained_per_run_masks};
  bool run_representation_initialized{};
  std::vector<LaneMask> csr_run_lane_masks;
  std::vector<LaneMask> csc_run_lane_masks;
  std::vector<std::uint32_t> touched_csr_runs;
  std::vector<std::uint32_t> touched_csc_runs;
  std::vector<RunLaneMaskDescriptor> csr_run_descriptors;
  std::vector<RunLaneMaskDescriptor> csc_run_descriptors;
  // Descriptor positions for each packed union vertex. Both host arrays are
  // retained for validation; a device workspace needs only the active
  // orientation. Retained-mask descriptions leave both arrays empty.
  std::vector<std::uint32_t> csr_descriptor_offsets_by_union_vertex;
  std::vector<std::uint32_t> csc_descriptor_offsets_by_union_vertex;
  BatchRunPreparationReport run_report{};
};

// Reuses every vector capacity. Retained masks clear only the prior touched
// ledger, then write masks for the current union-owner ranges. Device-
// materialized descriptions intentionally leave every host run-storage vector
// empty; their engine initialization owns run-mask construction. The graph and
// tile-run layout must have been deeply validated once before entering a
// preparation loop (SelectedRegionIndex construction provides that gate).
void prepare_batch_device_description(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    std::span<const RouteQuery> queries,
    std::span<const BatchQueryFeatures> features,
    const BatchPlanEntry& batch,
    BatchRunRepresentation run_representation,
    BatchDeviceDescription& output);

// Builds the optional dense tile-to-packed-vertex-base mapping for compact
// union-tile label storage. This is separate from run preparation so its cost
// can be measured independently. The prepared description must refer to the
// same batch. Existing allocation and the prior touched ledger are reused.
void prepare_compact_vertex_mapping(
    const WeightedGraph& graph,
    const BatchPlanEntry& batch,
    BatchDeviceDescription& output);

enum class BatchLayoutValidationErrorCode : std::uint8_t {
  none,
  invalid_tile_runs,
  invalid_batch_shape,
  invalid_lane_identity,
  invalid_source_offsets,
  invalid_target_offsets,
  invalid_terminal_payload,
  invalid_estimates,
  invalid_union_tiles,
  invalid_tile_lane_masks,
  invalid_vertex_ranges,
  invalid_compact_vertex_mapping,
  invalid_run_storage_shape,
  invalid_run_mask,
  invalid_descriptor_order,
  invalid_descriptor_offsets,
  admitted_edge_coverage_mismatch,
  nonzero_initial_result_mask,
};

struct BatchLayoutValidationResult {
  static constexpr std::size_t no_position = static_cast<std::size_t>(-1);

  BatchLayoutValidationErrorCode code{BatchLayoutValidationErrorCode::none};
  std::size_t position{no_position};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == BatchLayoutValidationErrorCode::none;
  }
};

[[nodiscard]] BatchLayoutValidationResult validate_batch_device_description(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    std::span<const RouteQuery> queries,
    std::span<const BatchQueryFeatures> features,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description) noexcept;

// Deep compact-mapping check. Unlike hot-path preparation/estimation, this
// scans the dense tile table to prove that only the touched union is live and
// that every packed base agrees with immutable tile offsets.
[[nodiscard]] BatchLayoutValidationResult validate_compact_vertex_mapping(
    const WeightedGraph& graph,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description) noexcept;

// Intended for bounded acceptance fixtures. It expands compact descriptors as
// needed and invokes the Phase 8 endpoint-by-endpoint admission proof.
[[nodiscard]] RunAdmissionProofResult prove_batch_endpoint_admission(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const BatchDeviceDescription& description);

static_assert(std::is_standard_layout_v<BatchVertexRange>);
static_assert(std::is_trivially_copyable_v<BatchVertexRange>);
static_assert(sizeof(BatchVertexRange) == 12U);
static_assert(std::is_standard_layout_v<RunLaneMaskDescriptor>);
static_assert(std::is_trivially_copyable_v<RunLaneMaskDescriptor>);
static_assert(sizeof(RunLaneMaskDescriptor) == 8U);

}  // namespace bfnew
