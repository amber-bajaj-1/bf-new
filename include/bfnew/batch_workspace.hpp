#pragma once

#include "bfnew/batch_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace bfnew {

enum class BatchVertexStorageStrategy : std::uint8_t {
  full_graph_vertex_major,
  compact_union_tiles,
};

struct BatchWorkspaceBudget {
  std::uint64_t device_capacity_bytes{};
  std::uint64_t resident_graph_bytes{};
  std::uint64_t explicit_reserve_bytes{};

  constexpr bool operator==(const BatchWorkspaceBudget&) const noexcept = default;
};

struct BatchWorkspaceStrategyTiming {
  std::uint64_t mapping_build_nanoseconds{};
  std::uint64_t run_build_nanoseconds{};

  constexpr bool operator==(
      const BatchWorkspaceStrategyTiming&) const noexcept = default;
};

struct BatchWorkspaceBuildMeasurements {
  std::uint64_t compact_mapping_build_nanoseconds{};
  // Each run timing covers construction of the dual-orientation host proof
  // image. Device capacity/write bytes separately model the larger single
  // orientation because only one orientation is active at a time.
  std::uint64_t retained_run_build_nanoseconds{};
  std::uint64_t descriptor_run_build_nanoseconds{};

  constexpr bool operator==(
      const BatchWorkspaceBuildMeasurements&) const noexcept = default;
};

struct BatchWorkspaceEstimate {
  BatchVertexStorageStrategy vertex_storage{
      BatchVertexStorageStrategy::full_graph_vertex_major};
  BatchRunRepresentation run_representation{
      BatchRunRepresentation::retained_per_run_masks};
  std::uint32_t lane_width{};
  std::uint32_t distance_slot_count{};

  std::uint64_t storage_vertex_count{};
  std::uint64_t union_vertex_count{};
  std::uint64_t selected_lane_vertex_count{};
  std::uint64_t allocated_lane_vertex_count{};
  std::uint64_t wasted_lane_vertex_count{};

  std::uint64_t distance_bytes{};
  std::uint64_t tile_mapping_bytes{};
  std::uint64_t run_storage_bytes{};
  std::uint64_t descriptor_offset_bytes{};
  std::uint64_t batch_metadata_bytes{};
  std::uint64_t total_workspace_bytes{};

  std::uint64_t distance_reset_bytes{};
  // Preparation write traffic implied by the most recent image. Retained and
  // descriptor rows model host-built writes; device-materialized Jacobi rows
  // model selected-column CSC stores performed by the initialization kernel,
  // including zero masks, with no corresponding host run image or H2D copy.
  // Mapping bytes use actual prior-ledger clears plus current writes. Decision
  // measurements warm reusable allocations before recording these fields.
  std::uint64_t tile_mapping_write_bytes{};
  std::uint64_t run_preparation_write_bytes{};
  std::uint64_t total_preparation_write_bytes{};

  // Measured host build evidence for this exact strategy row. Full-graph rows
  // require zero mapping time; compact rows record the reusable mapping build.
  // Run time is the dual-orientation host proof-image build, not device-kernel
  // time; byte/write capacity is for the larger one active orientation.
  std::uint64_t mapping_build_nanoseconds{};
  std::uint64_t run_build_nanoseconds{};

  std::uint64_t active_csr_runs{};
  std::uint64_t active_csc_runs{};
  std::uint64_t zero_csr_runs{};
  std::uint64_t zero_csc_runs{};
  std::uint64_t maximum_concurrent_workspaces{};
  bool reusable_allocation{};

  constexpr bool operator==(const BatchWorkspaceEstimate&) const noexcept =
      default;
};

// Pure checked count model used before constructing or allocating a graph.
// It is also the analytical guard for recorded real-graph vertex counts.
[[nodiscard]] std::uint64_t estimate_vertex_major_distance_bytes(
    std::uint64_t storage_vertex_count,
    std::uint32_t lane_width,
    std::uint32_t distance_slot_count);

// The graph, tile-run layout, batch, and prepared description must share the
// same already-validated immutable inputs. Retained and descriptor rows use
// constant-time shape/count checks. A device-materialized Jacobi row is an
// offline diagnostic that scans only selected CSC destination runs to count
// exact nonzero masks and initialization writes; hot batch preparation does
// not perform that scan.
//
// Retained/descriptor run storage and preparation traffic use max(CSR, CSC).
// Device-materialized storage is Jacobi-specific: it models one full dense CSC
// device mask plus selected-column initialization stores and requires the
// implemented full-graph, two-distance-slot layout.
[[nodiscard]] BatchWorkspaceEstimate estimate_batch_workspace(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description,
    BatchVertexStorageStrategy vertex_storage,
    BatchRunRepresentation run_representation,
    std::uint32_t distance_slot_count,
    const BatchWorkspaceBudget& budget,
    const BatchWorkspaceStrategyTiming& timing = {});

[[nodiscard]] std::vector<BatchWorkspaceEstimate>
compare_batch_workspace_strategies(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& retained_description,
    const BatchDeviceDescription& descriptor_description,
    std::uint32_t distance_slot_count,
    const BatchWorkspaceBudget& budget,
    const BatchWorkspaceBuildMeasurements& measurements = {});

struct BatchWorkspaceCapacity {
  std::uint64_t distance_bytes{};
  std::uint64_t tile_mapping_bytes{};
  std::uint64_t run_storage_bytes{};
  std::uint64_t descriptor_offset_bytes{};
  std::uint64_t batch_metadata_bytes{};

  [[nodiscard]] std::uint64_t total_bytes() const;
};

struct BatchWorkspaceReservationResult {
  bool capacity_grew{};
  std::uint64_t generation{};
};

class ReusableBatchWorkspaceReservation final {
 public:
  [[nodiscard]] BatchWorkspaceReservationResult reserve(
      const BatchWorkspaceEstimate& estimate);
  [[nodiscard]] const BatchWorkspaceCapacity& capacity() const noexcept;
  [[nodiscard]] std::uint64_t growth_events() const noexcept;

 private:
  BatchWorkspaceCapacity capacity_{};
  std::uint64_t generation_{};
  std::uint64_t growth_events_{};
};

enum class BatchMeasurementScope : std::uint8_t {
  bounded_synthetic,
  real_query_corpus,
};

struct BatchWorkspaceDecision {
  std::uint64_t plan_fingerprint{};
  BatchMeasurementScope measurement_scope{
      BatchMeasurementScope::bounded_synthetic};
  BatchWorkspaceBudget budget{};
  // Recorded graph dimensions bind every serialized 2x2 row formula to the
  // graph counts used when the decision was constructed.
  std::uint64_t graph_vertex_count{};
  std::uint64_t graph_tile_count{};
  std::uint32_t lane_width{};
  std::uint32_t distance_slot_count{};
  std::vector<BatchWorkspaceEstimate> compared_strategies;
  BatchVertexStorageStrategy selected_vertex_storage{
      BatchVertexStorageStrategy::full_graph_vertex_major};
  BatchRunRepresentation selected_run_representation{
      BatchRunRepresentation::retained_per_run_masks};
  std::uint64_t selected_mapping_build_nanoseconds{};
  std::uint64_t selected_run_build_nanoseconds{};
  std::string quantitative_reason;
};

[[nodiscard]] bool validate_batch_workspace_decision(
    const BatchWorkspaceDecision& decision) noexcept;

[[nodiscard]] std::string serialize_batch_workspace_decision(
    const BatchWorkspaceDecision& decision);

}  // namespace bfnew
