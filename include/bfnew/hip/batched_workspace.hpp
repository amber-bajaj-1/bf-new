#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/hip/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace bfnew::hip {

namespace detail {
struct DeviceBatchJacobiView;
struct DeviceBatchJacobiStatistics;
}  // namespace detail

struct BatchedJacobiWorkspaceCapacity {
  std::size_t device_bytes{};
  std::size_t engine_scratch_bytes{};
  std::uint64_t allocation_events{};
};

// Batch metadata cannot be represented by Phase 8's single-query staging
// record because it needs per-lane terminal offsets and selected ranges, and
// the Phase 8 preparation path unconditionally clears all scratch. This
// additive owner reuses the Phase 8 DeviceBuffer/stream runtime while
// retaining a batch-specific lease and never clearing distance scratch.
class ReusableBatchedJacobiWorkspace final {
 public:
  ReusableBatchedJacobiWorkspace();
  ~ReusableBatchedJacobiWorkspace() noexcept;

  ReusableBatchedJacobiWorkspace(
      const ReusableBatchedJacobiWorkspace&) = delete;
  ReusableBatchedJacobiWorkspace& operator=(
      const ReusableBatchedJacobiWorkspace&) = delete;
  ReusableBatchedJacobiWorkspace(ReusableBatchedJacobiWorkspace&&) = delete;
  ReusableBatchedJacobiWorkspace& operator=(
      ReusableBatchedJacobiWorkspace&&) = delete;

  [[nodiscard]] BatchedJacobiWorkspaceCapacity capacity() const noexcept;

 private:
  friend class BatchedJacobiPullEngine;

  void prepare_async(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      std::size_t scratch_bytes,
      std::uint32_t csc_run_count,
      const HipStream& stream);
  [[nodiscard]] DeviceWorkspaceView device_workspace_view() const;
  [[nodiscard]] detail::DeviceBatchJacobiView device_batch_view() const;
  [[nodiscard]] detail::DeviceBatchJacobiStatistics*
  device_batch_statistics() const noexcept;

  void download_controller_async(
      DeviceController& controller,
      const HipStream& stream) const;

  // Phase 17 all-query execution needs only the on-device all-targets masks
  // and terminal state. This deliberately excludes the controller, lane
  // traces, counters, and distance slots downloaded by download_async().
  void download_status_async(
      DeviceRunStatus& status,
      const HipStream& stream) const;

  void download_async(
      DeviceController& controller,
      DeviceRunStatus& status,
      DeviceWorkStatistics* work,
      detail::DeviceBatchJacobiStatistics* batch_work,
      std::span<std::uint64_t> lane_convergence_rounds,
      std::span<float> distance_slots,
      const HipStream& stream) const;
  void retire(const HipStream& stream);
  // Clears a lease without another fence after a same-stream operation has
  // already established completion (currently compact reconstruction only).
  void retire_after_stream_completion(const HipStream& stream);
  void recover_noexcept() noexcept;

  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
