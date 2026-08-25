#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/hip/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace bfnew::hip {

namespace detail {
struct DeviceBatchDenseView;
struct DeviceBatchDenseStatistics;
}  // namespace detail

struct BatchedDenseWorkspaceCapacity {
  std::size_t device_bytes{};
  std::size_t engine_scratch_bytes{};
  std::uint64_t allocation_events{};
};

// Phase 8's single-query workspace cannot carry per-lane terminal offsets or
// selected ranges and clears the entire scratch allocation. Dense batching
// instead retains this dedicated one-slot CSR workspace. Preparation uploads
// metadata but never clears distance scratch; the initialization kernel writes
// exactly the selected vertex/lane words before any relaxation can read them.
class ReusableBatchedDenseWorkspace final {
 public:
  ReusableBatchedDenseWorkspace();
  ~ReusableBatchedDenseWorkspace() noexcept;

  ReusableBatchedDenseWorkspace(const ReusableBatchedDenseWorkspace&) =
      delete;
  ReusableBatchedDenseWorkspace& operator=(
      const ReusableBatchedDenseWorkspace&) = delete;
  ReusableBatchedDenseWorkspace(ReusableBatchedDenseWorkspace&&) = delete;
  ReusableBatchedDenseWorkspace& operator=(
      ReusableBatchedDenseWorkspace&&) = delete;

  [[nodiscard]] BatchedDenseWorkspaceCapacity capacity() const noexcept;

 private:
  friend class BatchedDenseChaoticPushEngine;

  void prepare_async(
      const BatchDeviceDescription& batch,
      const GpuRunOptions& options,
      std::size_t scratch_bytes,
      const HipStream& stream);
  [[nodiscard]] DeviceWorkspaceView device_workspace_view() const;
  [[nodiscard]] detail::DeviceBatchDenseView device_batch_view() const;
  [[nodiscard]] detail::DeviceBatchDenseStatistics*
  device_batch_statistics() const noexcept;

  void download_controller_async(
      DeviceController& controller,
      const HipStream& stream) const;
  // Compact Phase 17 terminal transfer: no controller, lane traces,
  // instrumentation records, or V*W distance words are copied.
  void download_status_async(
      DeviceRunStatus& status,
      const HipStream& stream) const;
  void download_async(
      DeviceController& controller,
      DeviceRunStatus& status,
      DeviceWorkStatistics* work,
      detail::DeviceBatchDenseStatistics* batch_work,
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
