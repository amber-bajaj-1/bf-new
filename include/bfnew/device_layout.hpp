#pragma once

#include "bfnew/graph.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace bfnew {

// Sparse-layout edge runs retain host-width offsets.  A run never crosses a
// CSR row or CSC column boundary, even when the neighboring bucket uses the
// same remote tile.
struct TileRunLayout64 {
  std::vector<EdgeOffset> csr_row_run_offsets;
  std::vector<EdgeOffset> csr_run_edge_offsets;
  std::vector<TileId> csr_run_destination_tiles;

  std::vector<EdgeOffset> csc_column_run_offsets;
  std::vector<EdgeOffset> csc_run_edge_offsets;
  std::vector<TileId> csc_run_source_tiles;

  bool operator==(const TileRunLayout64&) const = default;
};

enum class TileRunValidationErrorCode : std::uint8_t {
  none,
  graph_validation_failed,
  graph_is_not_spatially_ordered,
  invalid_csr_row_run_offsets,
  invalid_csr_run_edge_offsets,
  csr_run_tile_count_mismatch,
  csr_empty_run,
  csr_run_crosses_row,
  csr_edge_coverage_mismatch,
  csr_run_tile_out_of_range,
  csr_run_tile_mismatch,
  csr_nonmaximal_runs,
  invalid_csc_column_run_offsets,
  invalid_csc_run_edge_offsets,
  csc_run_tile_count_mismatch,
  csc_empty_run,
  csc_run_crosses_column,
  csc_edge_coverage_mismatch,
  csc_run_tile_out_of_range,
  csc_run_tile_mismatch,
  csc_nonmaximal_runs,
};

struct TileRunValidationResult {
  static constexpr EdgeOffset no_position = std::numeric_limits<EdgeOffset>::max();

  TileRunValidationErrorCode code{TileRunValidationErrorCode::none};
  EdgeOffset position{no_position};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == TileRunValidationErrorCode::none;
  }
};

[[nodiscard]] TileRunLayout64 build_tile_run_layout(const WeightedGraph& graph);

[[nodiscard]] TileRunValidationResult validate_tile_run_layout(
    const WeightedGraph& graph,
    const TileRunLayout64& layout);

// This is a host staging representation of the immutable device graph.
// Relaxation does not consume stable edge IDs, but Phase 18 reconstruction
// uses the CSC IDs to reproduce the accepted deterministic tight-edge order.
// Physical provenance and vertex metadata remain host-only.
struct DeviceGraphLayout32 {
  std::uint32_t vertex_count{};
  std::uint32_t edge_count{};
  std::uint32_t tile_count{};

  std::vector<std::uint32_t> owner_tiles;

  std::vector<std::uint32_t> csr_row_offsets;
  std::vector<std::uint32_t> csr_destinations;
  std::vector<float> csr_weights;
  std::vector<std::uint32_t> csr_row_run_offsets;
  std::vector<std::uint32_t> csr_run_edge_offsets;
  std::vector<std::uint32_t> csr_run_destination_tiles;

  std::vector<std::uint32_t> csc_column_offsets;
  std::vector<std::uint32_t> csc_sources;
  std::vector<float> csc_weights;
  std::vector<std::uint32_t> csc_edge_ids;
  std::vector<std::uint32_t> csc_column_run_offsets;
  std::vector<std::uint32_t> csc_run_edge_offsets;
  std::vector<std::uint32_t> csc_run_source_tiles;
};

struct DeviceGraphMemoryReport {
  std::uint64_t owner_tiles_bytes{};

  std::uint64_t csr_row_offsets_bytes{};
  std::uint64_t csr_destinations_bytes{};
  std::uint64_t csr_weights_bytes{};
  std::uint64_t csr_row_run_offsets_bytes{};
  std::uint64_t csr_run_edge_offsets_bytes{};
  std::uint64_t csr_run_destination_tiles_bytes{};

  std::uint64_t csc_column_offsets_bytes{};
  std::uint64_t csc_sources_bytes{};
  std::uint64_t csc_weights_bytes{};
  std::uint64_t csc_edge_ids_bytes{};
  std::uint64_t csc_column_run_offsets_bytes{};
  std::uint64_t csc_run_edge_offsets_bytes{};
  std::uint64_t csc_run_source_tiles_bytes{};

  std::uint64_t total_bytes{};

  constexpr bool operator==(const DeviceGraphMemoryReport&) const noexcept = default;
};

// A deterministic content identity for the immutable relaxation-hot image.
// It prevents an engine from accidentally binding a same-shaped resident
// allocation uploaded from a different graph. It is an identity guard, not a
// cryptographic integrity primitive.
struct DeviceGraphFingerprint {
  std::uint64_t first{};
  std::uint64_t second{};

  constexpr bool operator==(const DeviceGraphFingerprint&) const noexcept = default;
};

enum class DeviceGraphLayoutValidationErrorCode : std::uint8_t {
  none,
  tile_run_layout_invalid,
  graph_not_representable,
  scalar_mismatch,
  owner_tiles_mismatch,
  csr_row_offsets_mismatch,
  csr_destinations_mismatch,
  csr_weights_mismatch,
  csr_row_run_offsets_mismatch,
  csr_run_edge_offsets_mismatch,
  csr_run_destination_tiles_mismatch,
  csc_column_offsets_mismatch,
  csc_sources_mismatch,
  csc_weights_mismatch,
  csc_edge_ids_mismatch,
  csc_column_run_offsets_mismatch,
  csc_run_edge_offsets_mismatch,
  csc_run_source_tiles_mismatch,
};

struct DeviceGraphLayoutValidationResult {
  static constexpr EdgeOffset no_position = std::numeric_limits<EdgeOffset>::max();

  DeviceGraphLayoutValidationErrorCode code{
      DeviceGraphLayoutValidationErrorCode::none};
  EdgeOffset position{no_position};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == DeviceGraphLayoutValidationErrorCode::none;
  }
};

[[nodiscard]] std::uint32_t checked_device_offset32(EdgeOffset offset);

[[nodiscard]] DeviceGraphLayout32 build_device_graph_layout32(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs);

[[nodiscard]] DeviceGraphLayoutValidationResult validate_device_graph_layout32(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const DeviceGraphLayout32& layout);

[[nodiscard]] bool device_graph_layouts_deep_equal(
    const DeviceGraphLayout32& left,
    const DeviceGraphLayout32& right);

[[nodiscard]] DeviceGraphMemoryReport report_device_graph_memory(
    const DeviceGraphLayout32& layout) noexcept;

[[nodiscard]] DeviceGraphFingerprint fingerprint_device_graph_layout32(
    const DeviceGraphLayout32& layout) noexcept;

// Hashes the exact checked 32-bit image directly from the host graph/run
// sources without allocating a duplicate DeviceGraphLayout32.
[[nodiscard]] DeviceGraphFingerprint fingerprint_device_graph_source32(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs);

struct TileRunLaneMasks {
  std::vector<std::uint32_t> csr_run_masks;
  std::vector<std::uint32_t> csc_run_masks;

  bool operator==(const TileRunLaneMasks&) const = default;
};

// Reuses the capacity of output across batches.  tile_lane_masks is indexed by
// dense TileId and uses one bit per batch lane.
void compute_tile_run_lane_masks(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    std::span<const std::uint32_t> tile_lane_masks,
    TileRunLaneMasks& output);

enum class RunAdmissionProofErrorCode : std::uint8_t {
  none,
  tile_run_layout_invalid,
  tile_lane_mask_size_mismatch,
  run_lane_mask_size_mismatch,
  csr_endpoint_mismatch,
  csc_endpoint_mismatch,
};

struct RunAdmissionProofResult {
  static constexpr EdgeOffset no_position = std::numeric_limits<EdgeOffset>::max();

  RunAdmissionProofErrorCode code{RunAdmissionProofErrorCode::none};
  EdgeOffset position{no_position};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == RunAdmissionProofErrorCode::none;
  }
};

// Checks every CSR and CSC sparse position.  A successful result proves that
// the materialized run mask admits exactly the same lane/edge pairs as an
// endpoint-by-endpoint owner-tile intersection.
[[nodiscard]] RunAdmissionProofResult prove_run_admission_equivalence(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    std::span<const std::uint32_t> tile_lane_masks,
    const TileRunLaneMasks& run_lane_masks);

}  // namespace bfnew
