#include "bfnew/hip/batched_workspace.hpp"

#include "batched_workspace_internal.hpp"

#include <hip/hip_runtime.h>

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bfnew::hip {
namespace {

[[nodiscard]] std::size_t checked_multiply(
    const std::size_t left,
    const std::size_t right,
    const char* const what) {
  if (right != 0U && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error{what};
  }
  return left * right;
}

template <typename T>
[[nodiscard]] std::size_t bytes(const std::size_t count) {
  return checked_multiply(count, sizeof(T), "batch workspace byte overflow");
}

template <typename T>
[[nodiscard]] T* device_pointer(void* const pointer) noexcept {
  return static_cast<T*>(pointer);
}

template <typename T>
[[nodiscard]] const T* device_pointer(const void* const pointer) noexcept {
  return static_cast<const T*>(pointer);
}

}  // namespace

class ReusableBatchedJacobiWorkspace::Impl final {
 public:
  DeviceBuffer sources;
  DeviceBuffer targets;
  DeviceBuffer tile_lane_masks;
  DeviceBuffer csc_run_lane_masks;
  DeviceBuffer source_offsets;
  DeviceBuffer target_offsets;
  DeviceBuffer selected_ranges;
  DeviceBuffer selected_range_vertex_offsets;
  DeviceBuffer controller;
  DeviceBuffer status;
  DeviceBuffer instrumentation;
  DeviceBuffer lane_convergence_rounds;
  DeviceBuffer batch_statistics;
  DeviceBuffer engine_scratch;

  DeviceWorkspaceView workspace_view{};
  detail::DeviceBatchJacobiView batch_view{};
  std::vector<std::uint32_t> selected_range_vertex_offsets_host;
  std::uint64_t generation{};
  std::uint64_t allocation_events{};
  bool lease_active{};
  bool device_work_may_be_in_flight{};
  bool instrumentation_enabled{};
  void* lease_stream{};

  [[nodiscard]] bool reserve(
      const BatchDeviceDescription& batch,
      const std::size_t scratch_bytes,
      const std::uint32_t csc_run_count) {
    if (lease_active) {
      throw std::logic_error{
          "retire the active batched Jacobi lease before reserving"};
    }
    bool grew = false;
    const auto grow = [&grew](DeviceBuffer& buffer, const std::size_t size) {
      grew = buffer.reserve(size, BufferGrowth::geometric) || grew;
    };
    grow(sources, bytes<std::uint32_t>(batch.sources.size()));
    grow(targets, bytes<std::uint32_t>(batch.targets.size()));
    grow(tile_lane_masks, bytes<LaneMask>(batch.tile_lane_masks.size()));
    grow(
        csc_run_lane_masks,
        bytes<LaneMask>(csc_run_count));
    grow(source_offsets, bytes<std::uint32_t>(batch.source_offsets.size()));
    grow(target_offsets, bytes<std::uint32_t>(batch.target_offsets.size()));
    grow(
        selected_ranges,
        bytes<BatchVertexRange>(batch.selected_vertex_ranges.size()));
    grow(
        selected_range_vertex_offsets,
        bytes<std::uint32_t>(batch.selected_vertex_ranges.size() + 1U));
    grow(controller, sizeof(DeviceController));
    grow(status, sizeof(DeviceRunStatus));
    // These two records are fixed-size and tiny. Reserve them with the first
    // batch so a later evidence replay cannot introduce allocator activity
    // into an otherwise warmed workspace. None instrumentation still exposes
    // null device pointers and performs no statistics clears or updates.
    grow(instrumentation, sizeof(DeviceWorkStatistics));
    grow(
        lane_convergence_rounds,
        bytes<std::uint64_t>(batch.lane_width));
    grow(batch_statistics, sizeof(detail::DeviceBatchJacobiStatistics));
    grow(engine_scratch, scratch_bytes);
    if (grew) {
      ++allocation_events;
    }
    return grew;
  }

  [[nodiscard]] std::size_t total_device_capacity() const noexcept {
    const DeviceBuffer* const buffers[] = {
        &sources,
        &targets,
        &tile_lane_masks,
        &csc_run_lane_masks,
        &source_offsets,
        &target_offsets,
        &selected_ranges,
        &selected_range_vertex_offsets,
        &controller,
        &status,
        &instrumentation,
        &lane_convergence_rounds,
        &batch_statistics,
        &engine_scratch,
    };
    std::size_t total = 0U;
    for (const DeviceBuffer* const buffer : buffers) {
      if (total >
          std::numeric_limits<std::size_t>::max() - buffer->capacity_bytes()) {
        return std::numeric_limits<std::size_t>::max();
      }
      total += buffer->capacity_bytes();
    }
    return total;
  }

  void clear_lease() noexcept {
    lease_active = false;
    device_work_may_be_in_flight = false;
    instrumentation_enabled = false;
    lease_stream = nullptr;
  }
};

ReusableBatchedJacobiWorkspace::ReusableBatchedJacobiWorkspace()
    : impl_{new Impl{}} {}

ReusableBatchedJacobiWorkspace::~ReusableBatchedJacobiWorkspace() noexcept {
  if (impl_ != nullptr) {
    if (impl_->lease_active || impl_->device_work_may_be_in_flight) {
      static_cast<void>(hipDeviceSynchronize());
    }
    delete impl_;
  }
}

BatchedJacobiWorkspaceCapacity ReusableBatchedJacobiWorkspace::capacity()
    const noexcept {
  if (impl_ == nullptr) {
    return {};
  }
  return BatchedJacobiWorkspaceCapacity{
      impl_->total_device_capacity(),
      impl_->engine_scratch.capacity_bytes(),
      impl_->allocation_events,
  };
}

void ReusableBatchedJacobiWorkspace::prepare_async(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    const std::size_t scratch_bytes,
    const std::uint32_t csc_run_count,
    const HipStream& stream) {
  if (impl_ == nullptr) {
    throw std::logic_error{
        "cannot prepare a moved-from batched Jacobi workspace"};
  }
  if (impl_->lease_active) {
    throw std::logic_error{
        "retire the active batched Jacobi lease before preparing another"};
  }
  if (impl_->device_work_may_be_in_flight) {
    static_cast<void>(hipDeviceSynchronize());
    impl_->clear_lease();
  }

  impl_->selected_range_vertex_offsets_host.clear();
  impl_->selected_range_vertex_offsets_host.reserve(
      batch.selected_vertex_ranges.size() + 1U);
  impl_->selected_range_vertex_offsets_host.push_back(0U);
  for (const BatchVertexRange range : batch.selected_vertex_ranges) {
    const std::uint64_t next =
        static_cast<std::uint64_t>(
            impl_->selected_range_vertex_offsets_host.back()) +
        (range.end - range.begin);
    if (next > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error{
          "batched Jacobi union vertex count exceeds the device ABI"};
    }
    impl_->selected_range_vertex_offsets_host.push_back(
        static_cast<std::uint32_t>(next));
  }
  if (impl_->selected_range_vertex_offsets_host.back() == 0U) {
    throw std::invalid_argument{"batched Jacobi union has no vertices"};
  }

  static_cast<void>(impl_->reserve(batch, scratch_bytes, csc_run_count));
  if (impl_->generation == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{"batched Jacobi workspace generation overflow"};
  }
  ++impl_->generation;
  impl_->lease_active = true;
  impl_->device_work_may_be_in_flight = true;
  impl_->instrumentation_enabled =
      options.instrumentation != InstrumentationLevel::none;
  impl_->lease_stream = stream.native_handle();

  try {
    const auto copy = [&](DeviceBuffer& destination,
                          const void* const source,
                          const std::size_t size) {
      if (size != 0U) {
        destination.copy_from_host_async(source, size, stream);
      }
    };
    copy(
        impl_->sources,
        batch.sources.data(),
        bytes<std::uint32_t>(batch.sources.size()));
    copy(
        impl_->targets,
        batch.targets.data(),
        bytes<std::uint32_t>(batch.targets.size()));
    copy(
        impl_->tile_lane_masks,
        batch.tile_lane_masks.data(),
        bytes<LaneMask>(batch.tile_lane_masks.size()));
    // The Jacobi initializer materializes masks only for selected destination
    // columns. It overwrites both zero and nonzero entries that the batch can
    // read, so neither a graph-wide clear nor a host mask upload is needed.
    copy(
        impl_->source_offsets,
        batch.source_offsets.data(),
        bytes<std::uint32_t>(batch.source_offsets.size()));
    copy(
        impl_->target_offsets,
        batch.target_offsets.data(),
        bytes<std::uint32_t>(batch.target_offsets.size()));
    copy(
        impl_->selected_ranges,
        batch.selected_vertex_ranges.data(),
        bytes<BatchVertexRange>(batch.selected_vertex_ranges.size()));
    copy(
        impl_->selected_range_vertex_offsets,
        impl_->selected_range_vertex_offsets_host.data(),
        bytes<std::uint32_t>(
            impl_->selected_range_vertex_offsets_host.size()));
    // Both ordinary and persistent initialization assign every controller
    // field before any read. Their finalizers likewise assign the complete
    // status record, so an initial controller H2D and status clear would only
    // add overwritten requests to every batch.
    if (impl_->instrumentation_enabled) {
      impl_->instrumentation.clear_async(sizeof(DeviceWorkStatistics), stream);
      impl_->batch_statistics.clear_async(
          sizeof(detail::DeviceBatchJacobiStatistics), stream);
    }
    // No engine-scratch memset occurs here. The initialization kernel writes
    // exactly both selected range/lane cells before any of them can be read.

    impl_->workspace_view = DeviceWorkspaceView{
        impl_->generation,
        static_cast<std::uint64_t>(scratch_bytes),
        static_cast<std::uint32_t>(EngineKind::jacobi_pull),
        static_cast<std::uint32_t>(batch.sources.size()),
        static_cast<std::uint32_t>(batch.targets.size()),
        0U,
        static_cast<std::uint32_t>(batch.tile_lane_masks.size()),
        csc_run_count,
        device_pointer<const std::uint32_t>(impl_->sources.data()),
        device_pointer<const std::uint32_t>(impl_->targets.data()),
        nullptr,
        device_pointer<LaneMask>(impl_->tile_lane_masks.data()),
        device_pointer<LaneMask>(impl_->csc_run_lane_masks.data()),
        device_pointer<DeviceController>(impl_->controller.data()),
        device_pointer<DeviceRunStatus>(impl_->status.data()),
        impl_->instrumentation_enabled
            ? device_pointer<DeviceWorkStatistics>(impl_->instrumentation.data())
            : nullptr,
        device_pointer<std::uint8_t>(impl_->engine_scratch.data()),
    };
    impl_->batch_view = detail::DeviceBatchJacobiView{
        batch.lane_width,
        batch.valid_lane_mask,
        static_cast<std::uint32_t>(batch.selected_vertex_ranges.size()),
        impl_->selected_range_vertex_offsets_host.back(),
        device_pointer<const std::uint32_t>(impl_->source_offsets.data()),
        device_pointer<const std::uint32_t>(impl_->target_offsets.data()),
        device_pointer<const BatchVertexRange>(impl_->selected_ranges.data()),
        device_pointer<const std::uint32_t>(
            impl_->selected_range_vertex_offsets.data()),
        device_pointer<std::uint64_t>(
            impl_->lane_convergence_rounds.data()),
    };
  } catch (...) {
    static_cast<void>(hipDeviceSynchronize());
    impl_->clear_lease();
    throw;
  }
}

DeviceWorkspaceView ReusableBatchedJacobiWorkspace::device_workspace_view()
    const {
  if (impl_ == nullptr || !impl_->lease_active) {
    throw std::logic_error{"batched Jacobi workspace has no active lease"};
  }
  return impl_->workspace_view;
}

detail::DeviceBatchJacobiView
ReusableBatchedJacobiWorkspace::device_batch_view() const {
  if (impl_ == nullptr || !impl_->lease_active) {
    throw std::logic_error{"batched Jacobi workspace has no active lease"};
  }
  return impl_->batch_view;
}

detail::DeviceBatchJacobiStatistics*
ReusableBatchedJacobiWorkspace::device_batch_statistics() const noexcept {
  return impl_ != nullptr && impl_->lease_active && impl_->instrumentation_enabled
             ? device_pointer<detail::DeviceBatchJacobiStatistics>(
                   impl_->batch_statistics.data())
             : nullptr;
}

void ReusableBatchedJacobiWorkspace::download_controller_async(
    DeviceController& controller_value,
    const HipStream& stream) const {
  if (impl_ == nullptr || !impl_->lease_active ||
      impl_->lease_stream != stream.native_handle()) {
    throw std::invalid_argument{
        "batched Jacobi poll rejected a stale/cross-stream lease"};
  }
  impl_->controller.copy_to_host_async(
      &controller_value, sizeof(controller_value), stream);
}

void ReusableBatchedJacobiWorkspace::download_status_async(
    DeviceRunStatus& status_value,
    const HipStream& stream) const {
  if (impl_ == nullptr || !impl_->lease_active ||
      impl_->lease_stream != stream.native_handle()) {
    throw std::invalid_argument{
        "batched Jacobi status download rejected a stale/cross-stream lease"};
  }
  impl_->status.copy_to_host_async(
      &status_value, sizeof(status_value), stream);
}

void ReusableBatchedJacobiWorkspace::download_async(
    DeviceController& controller_value,
    DeviceRunStatus& status_value,
    DeviceWorkStatistics* const work,
    detail::DeviceBatchJacobiStatistics* const batch_work,
    const std::span<std::uint64_t> lane_convergence_rounds,
    const std::span<float> distance_slots,
    const HipStream& stream) const {
  if (impl_ == nullptr || !impl_->lease_active ||
      impl_->lease_stream != stream.native_handle()) {
    throw std::invalid_argument{
        "batched Jacobi download rejected a stale/cross-stream lease"};
  }
  if (lane_convergence_rounds.size() != impl_->batch_view.lane_width) {
    throw std::invalid_argument{
        "batched Jacobi convergence download has the wrong width"};
  }
  if ((work == nullptr) != !impl_->instrumentation_enabled ||
      (batch_work == nullptr) != !impl_->instrumentation_enabled) {
    throw std::invalid_argument{
        "batched Jacobi statistics download disagrees with instrumentation"};
  }
  impl_->controller.copy_to_host_async(
      &controller_value, sizeof(controller_value), stream);
  impl_->status.copy_to_host_async(&status_value, sizeof(status_value), stream);
  impl_->lane_convergence_rounds.copy_to_host_async(
      lane_convergence_rounds.data(),
      bytes<std::uint64_t>(lane_convergence_rounds.size()),
      stream);
  if (impl_->instrumentation_enabled) {
    impl_->instrumentation.copy_to_host_async(
        work, sizeof(*work), stream);
    impl_->batch_statistics.copy_to_host_async(
        batch_work, sizeof(*batch_work), stream);
  }
  if (!distance_slots.empty()) {
    const std::size_t distance_bytes = bytes<float>(distance_slots.size());
    if (distance_bytes != impl_->workspace_view.engine_scratch_bytes) {
      throw std::invalid_argument{
          "batched Jacobi distance download has the wrong size"};
    }
    impl_->engine_scratch.copy_to_host_async(
        distance_slots.data(), distance_bytes, stream);
  }
}

void ReusableBatchedJacobiWorkspace::retire(const HipStream& stream) {
  if (impl_ == nullptr || !impl_->lease_active ||
      stream.native_handle() != impl_->lease_stream) {
    throw std::invalid_argument{
        "batched Jacobi workspace rejected a stale/cross-stream lease"};
  }
  stream.synchronize();
  impl_->clear_lease();
}

void ReusableBatchedJacobiWorkspace::retire_after_stream_completion(
    const HipStream& stream) {
  if (impl_ == nullptr || !impl_->lease_active ||
      stream.native_handle() != impl_->lease_stream) {
    throw std::invalid_argument{
        "batched Jacobi workspace rejected a stale/cross-stream lease"};
  }
  impl_->clear_lease();
}

void ReusableBatchedJacobiWorkspace::recover_noexcept() noexcept {
  if (impl_ != nullptr) {
    if (impl_->lease_active || impl_->device_work_may_be_in_flight) {
      static_cast<void>(hipDeviceSynchronize());
    }
    impl_->clear_lease();
  }
}

}  // namespace bfnew::hip
