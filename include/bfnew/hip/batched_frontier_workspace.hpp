#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/hip/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace bfnew::hip {

namespace detail {
struct DeviceBatchFrontierView;
struct DeviceBatchFrontierStatistics;
}  // namespace detail

struct BatchedFrontierWorkspaceCapacity {
  std::size_t device_bytes{};
  std::size_t engine_scratch_bytes{};
  std::uint64_t allocation_events{};
};

// A batch needs per-lane terminals, selected ranges, convergence records, and
// selected-only initialization. This owner therefore does not route through
// the Phase 8 single-query preparation path, whose unconditional scratch clear
// would erase the Phase 13 reset model. One stream owns each active lease.
class ReusableBatchedFrontierWorkspace final {
 public:
  ReusableBatchedFrontierWorkspace();
  ~ReusableBatchedFrontierWorkspace() noexcept;

  ReusableBatchedFrontierWorkspace(
      const ReusableBatchedFrontierWorkspace&) = delete;
  ReusableBatchedFrontierWorkspace& operator=(
      const ReusableBatchedFrontierWorkspace&) = delete;
  ReusableBatchedFrontierWorkspace(
      ReusableBatchedFrontierWorkspace&&) = delete;
  ReusableBatchedFrontierWorkspace& operator=(
      ReusableBatchedFrontierWorkspace&&) = delete;

  [[nodiscard]] BatchedFrontierWorkspaceCapacity capacity() const noexcept;

 private:
  friend class BatchedFrontierPushEngine;

  void prepare_async(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      std::uint32_t queue_capacity,
      std::size_t scratch_bytes,
      const HipStream& stream);
  [[nodiscard]] DeviceWorkspaceView device_workspace_view() const;
  [[nodiscard]] detail::DeviceBatchFrontierView device_batch_view() const;
  [[nodiscard]] detail::DeviceBatchFrontierStatistics*
  device_batch_statistics() const noexcept;

  void download_controller_async(
      DeviceController& controller,
      const HipStream& stream) const;
  // Compact Phase 17 terminal transfer: the GPU finalizer has already folded
  // every lane's targets into the masks in this single status record.
  void download_status_async(
      DeviceRunStatus& status,
      const HipStream& stream) const;
  void download_async(
      DeviceController& controller,
      DeviceRunStatus& status,
      DeviceWorkStatistics* work,
      detail::DeviceBatchFrontierStatistics* batch_work,
      std::span<std::uint64_t> lane_convergence_rounds,
      std::span<std::uint32_t> distance_bits,
      const HipStream& stream) const;
  void retire(const HipStream& stream);
  void retire_after_stream_completion(const HipStream& stream);
  void recover_noexcept() noexcept;

  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
