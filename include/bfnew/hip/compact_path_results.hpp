#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/compact_path_types.hpp"
#include "bfnew/hip/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace bfnew::hip {

enum class CompactDistanceEncoding : std::uint32_t {
  floating_point = 0U,
  nonnegative_float_bits = 1U,
};

// Non-owning final-lane matrix view. Exactly one data pointer is nonnull.
// Dense/frontier use one slot. Jacobi passes both slots; the reconstruction
// kernel resolves DeviceRunStatus::final_distance_slot on device before any
// label is read, avoiding a preliminary status transfer.
struct DeviceCompactDistanceMatrix {
  std::uint32_t vertex_count{};
  std::uint32_t lane_width{};
  std::uint32_t slot_count{1U};
  std::uint64_t slot_stride_elements{};
  CompactDistanceEncoding encoding{CompactDistanceEncoding::floating_point};
  const float* floating_point{};
  const std::uint32_t* nonnegative_float_bits{};
};

struct DeviceCompactBatchView {
  std::uint32_t lane_width{};
  LaneMask valid_lane_mask{};
  const std::uint32_t* source_offsets{};
  const std::uint32_t* target_offsets{};
};

static_assert(std::is_standard_layout_v<DeviceCompactDistanceMatrix>);
static_assert(std::is_trivially_copyable_v<DeviceCompactDistanceMatrix>);
static_assert(std::is_standard_layout_v<DeviceCompactBatchView>);
static_assert(std::is_trivially_copyable_v<DeviceCompactBatchView>);

struct CompactPathBatchMetrics {
  float sssp_device_milliseconds{};
  float reconstruction_device_milliseconds{};
  float result_transfer_device_milliseconds{};
  double end_to_end_wall_milliseconds{};

  constexpr bool operator==(const CompactPathBatchMetrics&) const noexcept =
      default;
};

// CompactTransferAccounting intentionally covers only the public target
// payload. total_device_to_host_bytes adds the fixed status/error transfers
// performed by reconstruction. Controller polls belong to SSSP, remain
// separate, and are included only in overall_device_to_host_bytes.
struct CompactPathTransportAccounting {
  std::uint64_t status_bytes{};
  std::uint64_t error_bytes{};
  std::uint64_t total_device_to_host_bytes{};
  std::uint64_t controller_poll_count{};
  std::uint64_t controller_poll_bytes{};
  std::uint64_t overall_device_to_host_bytes{};

  constexpr bool operator==(
      const CompactPathTransportAccounting&) const noexcept = default;
};

struct CompactPathBatchOutput {
  DeviceRunStatus status{};
  // Flattened in BatchDeviceDescription::targets order. Every finite target
  // in a clean lane is reconstructed, including the reachable subset of an
  // all-targets miss lane. Infinite targets have explicit unreachable
  // summaries. Nonclean lanes have query-terminal-failure summaries.
  std::vector<CompactTargetPath> targets;
  CompactTransferAccounting transfer{};
  CompactPathTransportAccounting transport{};
  CompactPathBatchMetrics metrics{};
};

// Called by an engine after reconstruction, once the exact number of ordinary
// or chunked convergence checks is known. Persistent execution passes zero.
// The helper validates the compact payload/control subtotal and fails closed
// on either multiplication or aggregate overflow.
void account_compact_path_controller_polls(
    CompactPathBatchOutput& output,
    std::uint64_t controller_poll_count);

struct CompactPathWorkspaceCapacity {
  std::size_t device_bytes{};
  std::uint32_t dfs_vertex_capacity{};
  std::uint32_t target_capacity{};
  std::uint64_t allocation_events{};

  constexpr bool operator==(
      const CompactPathWorkspaceCapacity&) const noexcept = default;
};

// One deterministic device worker processes targets lane-by-lane and reuses a
// single O(V) DFS/backtracking stack. Pass one transfers only status and fixed
// summaries. After checked host prefixing, pass two emits exact compact path
// vertices, their actual GPU distance labels, and checked-u32 stable EdgeIds.
// No graph-sized lane matrix is copied to the host.
class ReusableCompactPathWorkspace final {
 public:
  ReusableCompactPathWorkspace();
  ~ReusableCompactPathWorkspace() noexcept;

  ReusableCompactPathWorkspace(const ReusableCompactPathWorkspace&) = delete;
  ReusableCompactPathWorkspace& operator=(
      const ReusableCompactPathWorkspace&) = delete;
  ReusableCompactPathWorkspace(ReusableCompactPathWorkspace&&) = delete;
  ReusableCompactPathWorkspace& operator=(
      ReusableCompactPathWorkspace&&) = delete;

  [[nodiscard]] CompactPathWorkspaceCapacity capacity() const noexcept;

  // Engine integration point. The finalizer, distance matrix, and all batch
  // metadata must be live on stream until this returns. It synchronizes only
  // the two mandatory compact transfer boundaries.
  [[nodiscard]] CompactPathBatchOutput reconstruct(
      const DeviceGraphView32& graph,
      const DeviceWorkspaceView& workspace,
      const DeviceCompactBatchView& batch_view,
      const DeviceCompactDistanceMatrix& distances,
      const BatchDeviceDescription& host_batch,
      const HipStream& stream);

 private:
  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
