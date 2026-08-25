#pragma once

#include "bfnew/device_layout.hpp"
#include "bfnew/gpu_api.hpp"
#include "bfnew/workspace.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace bfnew::hip {

// HIP headers intentionally stay out of this public header.  That keeps the
// host-only library usable when BFNEW_ENABLE_HIP is OFF and makes the device
// ABI below depend only on fixed-width C++ types.
class HipRuntimeError final : public std::runtime_error {
 public:
  HipRuntimeError(std::int32_t status, std::string message);

  [[nodiscard]] std::int32_t status() const noexcept { return status_; }

 private:
  std::int32_t status_{};
};

void throw_if_hip_error(
    std::int32_t status,
    std::string_view expression,
    std::source_location location = std::source_location::current());

class HipStream final {
 public:
  explicit HipStream(bool nonblocking = true);
  ~HipStream() noexcept;

  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;
  HipStream(HipStream&& other) noexcept;
  HipStream& operator=(HipStream&& other) noexcept;

  void synchronize() const;
  [[nodiscard]] void* native_handle() const noexcept { return handle_; }

 private:
  void* handle_{};
};

class HipEvent final {
 public:
  explicit HipEvent(bool enable_timing = false);
  ~HipEvent() noexcept;

  HipEvent(const HipEvent&) = delete;
  HipEvent& operator=(const HipEvent&) = delete;
  HipEvent(HipEvent&& other) noexcept;
  HipEvent& operator=(HipEvent&& other) noexcept;

  void record(const HipStream& stream);
  void wait(const HipStream& stream) const;
  void synchronize() const;
  [[nodiscard]] bool query() const;
  [[nodiscard]] static float elapsed_milliseconds(
      const HipEvent& start,
      const HipEvent& stop);

  [[nodiscard]] void* native_handle() const noexcept { return handle_; }
  [[nodiscard]] bool timing_enabled() const noexcept { return timing_enabled_; }

 private:
  void* handle_{};
  bool timing_enabled_{};
};

enum class BufferGrowth : std::uint8_t {
  exact,
  geometric,
};

class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() noexcept;

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  // Returns true only when a new persistent allocation was made.
  [[nodiscard]] bool reserve(
      std::size_t minimum_bytes,
      BufferGrowth growth = BufferGrowth::geometric);
  void release() noexcept;

  void clear_async(
      std::size_t bytes,
      const HipStream& stream,
      std::size_t destination_offset = 0U);
  // Copies are ordered in stream order. Guaranteed overlap with the host
  // requires the host endpoint to be HIP page-locked; otherwise HIP may stage
  // the transfer or block the calling thread.
  void copy_from_host_async(
      const void* source,
      std::size_t bytes,
      const HipStream& stream,
      std::size_t destination_offset = 0U);
  void copy_to_host_async(
      void* destination,
      std::size_t bytes,
      const HipStream& stream,
      std::size_t source_offset = 0U) const;

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t capacity_bytes() const noexcept {
    return capacity_bytes_;
  }

 private:
  void* data_{};
  std::size_t capacity_bytes_{};
};

class HipEventTimer final {
 public:
  HipEventTimer();

  void start(const HipStream& stream);
  void stop(const HipStream& stream);
  [[nodiscard]] float elapsed_milliseconds();
  // Use only after the containing stream (or the stop event) is already
  // known complete. Unlike elapsed_milliseconds(), this does not issue a
  // synchronization and therefore lets several adjacent device intervals be
  // observed after one shared result-transfer synchronization.
  [[nodiscard]] float elapsed_milliseconds_after_stream_synchronization()
      const;

 private:
  HipEvent start_;
  HipEvent stop_;
  bool started_{};
  bool stopped_{};
};

class SteadyClockTimer final {
 public:
  SteadyClockTimer() noexcept;

  void reset() noexcept;
  [[nodiscard]] double elapsed_milliseconds() const noexcept;

 private:
  std::chrono::steady_clock::time_point start_;
};

// These structures are copied into kernel argument blocks.  They deliberately
// contain no std::span, size_t, host strong-ID wrappers, or owning objects.
struct DeviceCsrView32 {
  std::uint32_t vertex_count{};
  std::uint32_t edge_count{};
  std::uint32_t run_count{};
  const std::uint32_t* row_offsets{};
  const std::uint32_t* destinations{};
  const float* weights{};
  const std::uint32_t* row_run_offsets{};
  const std::uint32_t* run_edge_offsets{};
  const std::uint32_t* run_destination_tiles{};
};

struct DeviceCscView32 {
  std::uint32_t vertex_count{};
  std::uint32_t edge_count{};
  std::uint32_t run_count{};
  const std::uint32_t* column_offsets{};
  const std::uint32_t* sources{};
  const float* weights{};
  const std::uint32_t* edge_ids{};
  const std::uint32_t* column_run_offsets{};
  const std::uint32_t* run_edge_offsets{};
  const std::uint32_t* run_source_tiles{};
};

struct DeviceGraphView32 {
  std::uint32_t vertex_count{};
  std::uint32_t edge_count{};
  std::uint32_t tile_count{};
  const std::uint32_t* owner_tiles{};
  DeviceCsrView32 csr{};
  DeviceCscView32 csc{};
};

struct DeviceWorkspaceView {
  std::uint64_t generation{};
  std::uint64_t engine_scratch_bytes{};
  std::uint32_t engine_kind{};
  std::uint32_t source_count{};
  std::uint32_t target_count{};
  std::uint32_t selected_tile_count{};
  std::uint32_t tile_lane_mask_count{};
  std::uint32_t run_lane_mask_count{};
  const std::uint32_t* sources{};
  const std::uint32_t* targets{};
  const std::uint32_t* selected_tiles{};
  LaneMask* tile_lane_masks{};
  LaneMask* run_lane_masks{};
  DeviceController* controller{};
  DeviceRunStatus* status{};
  DeviceWorkStatistics* instrumentation{};
  std::uint8_t* engine_scratch{};
};

static_assert(std::is_standard_layout_v<DeviceCsrView32>);
static_assert(std::is_trivially_copyable_v<DeviceCsrView32>);
static_assert(std::is_standard_layout_v<DeviceCscView32>);
static_assert(std::is_trivially_copyable_v<DeviceCscView32>);
static_assert(std::is_standard_layout_v<DeviceGraphView32>);
static_assert(std::is_trivially_copyable_v<DeviceGraphView32>);
static_assert(std::is_standard_layout_v<DeviceWorkspaceView>);
static_assert(std::is_trivially_copyable_v<DeviceWorkspaceView>);

// Constructing a plan performs all host-side shape checks and exposes the
// component memory report before ResidentDeviceGraph allocates anything.
struct ResidentGraphPlan {
  DeviceGraphLayout32 layout;
  DeviceGraphMemoryReport memory;
  DeviceGraphFingerprint fingerprint;
};

[[nodiscard]] ResidentGraphPlan make_resident_graph_plan(
    DeviceGraphLayout32 layout);

class ResidentDeviceGraph final {
 public:
  ResidentDeviceGraph();
  // If the graph was uploaded, destruction conservatively waits for all HIP
  // device work because Phase 8 does not yet register consuming kernels.
  ~ResidentDeviceGraph() noexcept;

  ResidentDeviceGraph(const ResidentDeviceGraph&) = delete;
  ResidentDeviceGraph& operator=(const ResidentDeviceGraph&) = delete;
  ResidentDeviceGraph(ResidentDeviceGraph&& other) noexcept;
  ResidentDeviceGraph& operator=(ResidentDeviceGraph&& other) noexcept;

  // This object is immutable after the first upload attempt.  The staging
  // layout is retained until the ready event completes, so the caller does not
  // need to keep the plan alive.
  void upload_once_async(ResidentGraphPlan plan, const HipStream& stream);
  void wait_until_ready(const HipStream& stream) const;
  void synchronize_upload();

  [[nodiscard]] bool has_upload() const noexcept;
  [[nodiscard]] bool upload_complete() const;
  [[nodiscard]] const DeviceGraphView32& view() const;
  [[nodiscard]] const DeviceGraphMemoryReport& memory_report() const;
  [[nodiscard]] const DeviceGraphFingerprint& fingerprint() const;

  // The destination owns the host memory and must remain alive until stream
  // completion.  This is intended for small validation round trips.
  void download_async(DeviceGraphLayout32& destination, const HipStream& stream) const;

 private:
  class Impl;
  Impl* impl_{};
};

class ReusableDeviceWorkspace final {
 public:
  ReusableDeviceWorkspace();
  // Retiring a lease is the normal path.  Destruction of an unretired lease
  // conservatively waits for all HIP device work before releasing buffers.
  ~ReusableDeviceWorkspace() noexcept;

  ReusableDeviceWorkspace(const ReusableDeviceWorkspace&) = delete;
  ReusableDeviceWorkspace& operator=(const ReusableDeviceWorkspace&) = delete;
  ReusableDeviceWorkspace(ReusableDeviceWorkspace&& other) noexcept;
  ReusableDeviceWorkspace& operator=(ReusableDeviceWorkspace&& other) noexcept;

  // Reservation is the only operation allowed to allocate device or pinned
  // staging storage.  All capacities grow geometrically and CSR/CSC run masks
  // share one buffer.  All engines likewise share one large scratch buffer.
  [[nodiscard]] bool reserve(const WorkspaceMemoryRequirements& requirements);

  // Requires a prior sufficient reservation.  This function performs no
  // allocation, never touches ResidentDeviceGraph, overwrites all uploaded
  // prefixes, and clears every active reusable state buffer.
  [[nodiscard]] WorkspaceLease prepare_query_async(
      const WorkspaceMemoryRequirements& requirements,
      const RouteQuery& query,
      std::span<const LaneMask> tile_lane_masks,
      std::span<const LaneMask> run_lane_masks,
      const DeviceController& controller,
      const HipStream& stream);

  [[nodiscard]] bool accepts(const WorkspaceLease& lease) const noexcept;
  [[nodiscard]] DeviceWorkspaceView view(const WorkspaceLease& lease) const;

  void wait_until_prepared(const HipStream& stream, const WorkspaceLease& lease) const;
  void synchronize_preparation(const WorkspaceLease& lease) const;

  void download_status_async(
      const WorkspaceLease& lease,
      DeviceRunStatus& destination,
      const HipStream& stream) const;
  void download_instrumentation_async(
      const WorkspaceLease& lease,
      DeviceWorkStatistics& destination,
      const HipStream& stream) const;

  // Phase 8 enforces one stream per lease: preparation, all consumers, result
  // copies, and retirement must use the same stream. Retirement synchronizes
  // it before invalidating the lease. Cross-stream execution is deferred until
  // a completion-registration API exists.
  void retire_query(const WorkspaceLease& lease, const HipStream& stream);

  [[nodiscard]] const WorkspaceCapacity& capacity() const noexcept;
  [[nodiscard]] std::uint64_t allocation_events() const noexcept;

 private:
  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
