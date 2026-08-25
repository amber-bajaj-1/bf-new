#include "bfnew/hip/batched_frontier_workspace.hpp"

#include "batched_frontier_workspace_internal.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <iterator>
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
  return checked_multiply(
      count, sizeof(T), "batched frontier workspace byte overflow");
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

class ReusableBatchedFrontierWorkspace::Impl final {
 public:
  DeviceBuffer sources;
  DeviceBuffer targets;
  DeviceBuffer selected_tiles;
  DeviceBuffer tile_lane_masks;
  DeviceBuffer csr_run_lane_masks;
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
  detail::DeviceBatchFrontierView batch_view{};
  DeviceController controller_staging{};
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
      const InstrumentationLevel instrumentation_level) {
    if (lease_active) {
      throw std::logic_error{
          "retire the active batched frontier lease before reserving"};
    }
    bool grew = false;
    const auto grow = [&grew](DeviceBuffer& buffer, const std::size_t size) {
      grew = buffer.reserve(size, BufferGrowth::geometric) || grew;
    };
    grow(sources, bytes<std::uint32_t>(batch.sources.size()));
    grow(targets, bytes<std::uint32_t>(batch.targets.size()));
    grow(selected_tiles, bytes<std::uint32_t>(batch.union_tiles.size()));
    grow(tile_lane_masks, bytes<LaneMask>(batch.tile_lane_masks.size()));
    grow(
        csr_run_lane_masks,
        bytes<LaneMask>(batch.csr_run_lane_masks.size()));
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
    grow(
        instrumentation,
        instrumentation_level == InstrumentationLevel::none
            ? 0U
            : sizeof(DeviceWorkStatistics));
    grow(
        lane_convergence_rounds,
        bytes<std::uint64_t>(batch.lane_width));
    grow(
        batch_statistics,
        instrumentation_level == InstrumentationLevel::none
            ? 0U
            : sizeof(detail::DeviceBatchFrontierStatistics));
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
        &selected_tiles,
        &tile_lane_masks,
        &csr_run_lane_masks,
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

ReusableBatchedFrontierWorkspace::ReusableBatchedFrontierWorkspace()
    : impl_{new Impl{}} {}

ReusableBatchedFrontierWorkspace::~ReusableBatchedFrontierWorkspace() noexcept {
  if (impl_ != nullptr) {
    if (impl_->lease_active || impl_->device_work_may_be_in_flight) {
      static_cast<void>(hipDeviceSynchronize());
    }
    delete impl_;
  }
}

BatchedFrontierWorkspaceCapacity ReusableBatchedFrontierWorkspace::capacity()
    const noexcept {
  if (impl_ == nullptr) {
    return {};
  }
  return BatchedFrontierWorkspaceCapacity{
      impl_->total_device_capacity(),
      impl_->engine_scratch.capacity_bytes(),
      impl_->allocation_events,
  };
}

void ReusableBatchedFrontierWorkspace::prepare_async(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    const std::uint32_t queue_capacity,
    const std::size_t scratch_bytes,
    const HipStream& stream) {
  if (impl_ == nullptr) {
    throw std::logic_error{
        "cannot prepare a moved-from batched frontier workspace"};
  }
  if (impl_->lease_active) {
    throw std::logic_error{
        "retire the active batched frontier lease before preparing another"};
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
          "batched frontier union vertex count exceeds the device ABI"};
    }
    impl_->selected_range_vertex_offsets_host.push_back(
        static_cast<std::uint32_t>(next));
  }
  if (impl_->selected_range_vertex_offsets_host.back() == 0U) {
    throw std::invalid_argument{"batched frontier union has no vertices"};
  }

  static_cast<void>(impl_->reserve(batch, scratch_bytes, options.instrumentation));
  if (impl_->generation == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{"batched frontier workspace generation overflow"};
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
        impl_->selected_tiles,
        batch.union_tiles.data(),
        bytes<std::uint32_t>(batch.union_tiles.size()));
    copy(
        impl_->tile_lane_masks,
        batch.tile_lane_masks.data(),
        bytes<LaneMask>(batch.tile_lane_masks.size()));
    copy(
        impl_->csr_run_lane_masks,
        batch.csr_run_lane_masks.data(),
        bytes<LaneMask>(batch.csr_run_lane_masks.size()));
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

    std::vector<std::uint32_t> distinct_sources = batch.sources;
    std::sort(distinct_sources.begin(), distinct_sources.end());
    const auto distinct_end =
        std::unique(distinct_sources.begin(), distinct_sources.end());
    const std::uint64_t distinct_source_count = static_cast<std::uint64_t>(
        std::distance(distinct_sources.begin(), distinct_end));
    if (distinct_source_count == 0U) {
      throw std::invalid_argument{
          "batched frontier requires at least one distinct source vertex"};
    }
    // This valid staging controller protects the async lease before the
    // ordered device initializer resets size zero and constructs the merged
    // source queue. No round can observe the intermediate staging value.
    impl_->controller_staging = initialize_device_controller(
        options, batch.valid_lane_mask, distinct_source_count);
    impl_->controller.copy_from_host_async(
        &impl_->controller_staging,
        sizeof(impl_->controller_staging),
        stream);
    impl_->status.clear_async(sizeof(DeviceRunStatus), stream);
    if (impl_->instrumentation_enabled) {
      impl_->instrumentation.clear_async(sizeof(DeviceWorkStatistics), stream);
      impl_->batch_statistics.clear_async(
          sizeof(detail::DeviceBatchFrontierStatistics), stream);
    }
    // Deliberately no full scratch clear: selected range/lane words are
    // initialized once by the first device kernel; all other words stay stale
    // and nonsemantic for this lease.

    impl_->workspace_view = DeviceWorkspaceView{
        impl_->generation,
        static_cast<std::uint64_t>(scratch_bytes),
        static_cast<std::uint32_t>(EngineKind::frontier_push),
        static_cast<std::uint32_t>(batch.sources.size()),
        static_cast<std::uint32_t>(batch.targets.size()),
        static_cast<std::uint32_t>(batch.union_tiles.size()),
        static_cast<std::uint32_t>(batch.tile_lane_masks.size()),
        static_cast<std::uint32_t>(batch.csr_run_lane_masks.size()),
        device_pointer<const std::uint32_t>(impl_->sources.data()),
        device_pointer<const std::uint32_t>(impl_->targets.data()),
        device_pointer<const std::uint32_t>(impl_->selected_tiles.data()),
        device_pointer<LaneMask>(impl_->tile_lane_masks.data()),
        device_pointer<LaneMask>(impl_->csr_run_lane_masks.data()),
        device_pointer<DeviceController>(impl_->controller.data()),
        device_pointer<DeviceRunStatus>(impl_->status.data()),
        impl_->instrumentation_enabled
            ? device_pointer<DeviceWorkStatistics>(impl_->instrumentation.data())
            : nullptr,
        device_pointer<std::uint8_t>(impl_->engine_scratch.data()),
    };
    impl_->batch_view = detail::DeviceBatchFrontierView{
        batch.lane_width,
        batch.valid_lane_mask,
        static_cast<std::uint32_t>(batch.selected_vertex_ranges.size()),
        impl_->selected_range_vertex_offsets_host.back(),
        queue_capacity,
        device_pointer<const std::uint32_t>(impl_->source_offsets.data()),
        device_pointer<const std::uint32_t>(impl_->target_offsets.data()),
        device_pointer<const BatchVertexRange>(impl_->selected_ranges.data()),
        device_pointer<const std::uint32_t>(
            impl_->selected_range_vertex_offsets.data()),
        device_pointer<std::uint64_t>(impl_->lane_convergence_rounds.data()),
    };
  } catch (...) {
    static_cast<void>(hipDeviceSynchronize());
    impl_->clear_lease();
    throw;
  }
}

DeviceWorkspaceView ReusableBatchedFrontierWorkspace::device_workspace_view()
    const {
  if (impl_ == nullptr || !impl_->lease_active) {
    throw std::logic_error{"batched frontier workspace has no active lease"};
  }
  return impl_->workspace_view;
}

detail::DeviceBatchFrontierView
ReusableBatchedFrontierWorkspace::device_batch_view() const {
  if (impl_ == nullptr || !impl_->lease_active) {
    throw std::logic_error{"batched frontier workspace has no active lease"};
  }
  return impl_->batch_view;
}

detail::DeviceBatchFrontierStatistics*
ReusableBatchedFrontierWorkspace::device_batch_statistics() const noexcept {
  return impl_ != nullptr && impl_->lease_active && impl_->instrumentation_enabled
             ? device_pointer<detail::DeviceBatchFrontierStatistics>(
                   impl_->batch_statistics.data())
             : nullptr;
}

void ReusableBatchedFrontierWorkspace::download_controller_async(
    DeviceController& controller_value,
    const HipStream& stream) const {
  if (impl_ == nullptr || !impl_->lease_active ||
      impl_->lease_stream != stream.native_handle()) {
    throw std::invalid_argument{
        "batched frontier poll rejected a stale/cross-stream lease"};
  }
  impl_->controller.copy_to_host_async(
      &controller_value, sizeof(controller_value), stream);
}

void ReusableBatchedFrontierWorkspace::download_status_async(
    DeviceRunStatus& status_value,
    const HipStream& stream) const {
  if (impl_ == nullptr || !impl_->lease_active ||
      impl_->lease_stream != stream.native_handle()) {
    throw std::invalid_argument{
        "batched frontier status download rejected a stale/cross-stream lease"};
  }
  impl_->status.copy_to_host_async(
      &status_value, sizeof(status_value), stream);
}

void ReusableBatchedFrontierWorkspace::download_async(
    DeviceController& controller_value,
    DeviceRunStatus& status_value,
    DeviceWorkStatistics* const work,
    detail::DeviceBatchFrontierStatistics* const batch_work,
    const std::span<std::uint64_t> lane_convergence_rounds,
    const std::span<std::uint32_t> distance_bits,
    const HipStream& stream) const {
  if (impl_ == nullptr || !impl_->lease_active ||
      impl_->lease_stream != stream.native_handle()) {
    throw std::invalid_argument{
        "batched frontier download rejected a stale/cross-stream lease"};
  }
  if (lane_convergence_rounds.size() != impl_->batch_view.lane_width) {
    throw std::invalid_argument{
        "batched frontier convergence download has the wrong width"};
  }
  if ((work == nullptr) != !impl_->instrumentation_enabled ||
      (batch_work == nullptr) != !impl_->instrumentation_enabled) {
    throw std::invalid_argument{
        "batched frontier statistics download disagrees with instrumentation"};
  }
  impl_->controller.copy_to_host_async(
      &controller_value, sizeof(controller_value), stream);
  impl_->status.copy_to_host_async(&status_value, sizeof(status_value), stream);
  impl_->lane_convergence_rounds.copy_to_host_async(
      lane_convergence_rounds.data(),
      bytes<std::uint64_t>(lane_convergence_rounds.size()),
      stream);
  if (impl_->instrumentation_enabled) {
    impl_->instrumentation.copy_to_host_async(work, sizeof(*work), stream);
    impl_->batch_statistics.copy_to_host_async(
        batch_work, sizeof(*batch_work), stream);
  }
  if (!distance_bits.empty()) {
    const std::size_t distance_bytes =
        bytes<std::uint32_t>(distance_bits.size());
    if (distance_bytes > impl_->workspace_view.engine_scratch_bytes) {
      throw std::invalid_argument{
          "batched frontier distance download has the wrong size"};
    }
    impl_->engine_scratch.copy_to_host_async(
        distance_bits.data(), distance_bytes, stream);
  }
}

void ReusableBatchedFrontierWorkspace::retire(const HipStream& stream) {
  if (impl_ == nullptr || !impl_->lease_active ||
      stream.native_handle() != impl_->lease_stream) {
    throw std::invalid_argument{
        "batched frontier workspace rejected a stale/cross-stream lease"};
  }
  stream.synchronize();
  impl_->clear_lease();
}

void ReusableBatchedFrontierWorkspace::retire_after_stream_completion(
    const HipStream& stream) {
  if (impl_ == nullptr || !impl_->lease_active ||
      stream.native_handle() != impl_->lease_stream) {
    throw std::invalid_argument{
        "batched frontier workspace rejected a stale/cross-stream lease"};
  }
  impl_->clear_lease();
}

void ReusableBatchedFrontierWorkspace::recover_noexcept() noexcept {
  if (impl_ != nullptr) {
    if (impl_->lease_active || impl_->device_work_may_be_in_flight) {
      static_cast<void>(hipDeviceSynchronize());
    }
    impl_->clear_lease();
  }
}

}  // namespace bfnew::hip
