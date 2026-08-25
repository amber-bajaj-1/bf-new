#include "bfnew/hip/runtime.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace bfnew::hip {
namespace {

void check(
    const hipError_t status,
    const std::string_view expression,
    const std::source_location location = std::source_location::current()) {
  throw_if_hip_error(static_cast<std::int32_t>(status), expression, location);
}

[[nodiscard]] std::size_t checked_add(
    const std::size_t left,
    const std::size_t right,
    const std::string_view what) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  return left + right;
}

[[nodiscard]] std::size_t checked_multiply(
    const std::size_t count,
    const std::size_t width,
    const std::string_view what) {
  if (width != 0U && count > std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  return count * width;
}

[[nodiscard]] std::size_t geometric_capacity(
    const std::size_t current,
    const std::size_t requested) {
  if (requested <= current) {
    return current;
  }
  std::size_t next = current == 0U ? requested : current;
  while (next < requested) {
    if (next > std::numeric_limits<std::size_t>::max() / 2U) {
      return requested;
    }
    next *= 2U;
  }
  return next;
}

template <typename T>
[[nodiscard]] T* device_pointer(void* pointer) noexcept {
  return static_cast<T*>(pointer);
}

template <typename T>
[[nodiscard]] const T* device_pointer(const void* pointer) noexcept {
  return static_cast<const T*>(pointer);
}

class PinnedHostBuffer final {
 public:
  PinnedHostBuffer() = default;
  ~PinnedHostBuffer() noexcept { release(); }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  [[nodiscard]] bool reserve(const std::size_t minimum_bytes) {
    if (minimum_bytes <= capacity_bytes_) {
      return false;
    }
    const std::size_t next = geometric_capacity(capacity_bytes_, minimum_bytes);
    void* replacement = nullptr;
    check(hipHostMalloc(&replacement, next, hipHostMallocDefault), "hipHostMalloc");
    if (data_ != nullptr) {
      const hipError_t status = hipHostFree(data_);
      if (status != hipSuccess) {
        static_cast<void>(hipHostFree(replacement));
        check(status, "hipHostFree");
      }
    }
    data_ = replacement;
    capacity_bytes_ = next;
    return true;
  }

  void release() noexcept {
    if (data_ != nullptr) {
      static_cast<void>(hipHostFree(data_));
    }
    data_ = nullptr;
    capacity_bytes_ = 0U;
  }

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] std::size_t capacity_bytes() const noexcept {
    return capacity_bytes_;
  }

 private:
  void* data_{};
  std::size_t capacity_bytes_{};
};

void validate_workspace_requirements(const WorkspaceMemoryRequirements& requirements) {
  if (requirements.lane_capacity == 0U ||
      requirements.lane_capacity > maximum_batch_lanes) {
    throw std::invalid_argument{
        "device workspace lane capacity exceeds maximum_batch_lanes"};
  }
  GpuRunOptions options;
  options.engine = requirements.engine;
  options.instrumentation = requirements.instrumentation;
  if (validate_gpu_run_options(options) != GpuRunOptionsError::none) {
    throw std::invalid_argument{
        "device workspace uses an invalid engine or instrumentation level"};
  }
  const std::size_t expected_instrumentation_bytes =
      requirements.instrumentation == InstrumentationLevel::none
          ? 0U
          : sizeof(DeviceWorkStatistics);
  if (requirements.source_bytes % sizeof(std::uint32_t) != 0U ||
      requirements.target_bytes % sizeof(std::uint32_t) != 0U ||
      requirements.selected_tile_bytes % sizeof(std::uint32_t) != 0U ||
      requirements.tile_lane_mask_bytes % sizeof(LaneMask) != 0U ||
      requirements.run_lane_mask_bytes % sizeof(LaneMask) != 0U ||
      requirements.controller_bytes != sizeof(DeviceController) ||
      requirements.status_bytes != sizeof(DeviceRunStatus) ||
      requirements.instrumentation_bytes != expected_instrumentation_bytes) {
    throw std::invalid_argument{"device workspace requirements violate the fixed ABI"};
  }

  const auto require_count32 = [](const std::size_t bytes, const std::size_t width) {
    return bytes / width <= std::numeric_limits<std::uint32_t>::max();
  };
  if (!require_count32(requirements.source_bytes, sizeof(std::uint32_t)) ||
      !require_count32(requirements.target_bytes, sizeof(std::uint32_t)) ||
      !require_count32(requirements.selected_tile_bytes, sizeof(std::uint32_t)) ||
      !require_count32(requirements.tile_lane_mask_bytes, sizeof(LaneMask)) ||
      !require_count32(requirements.run_lane_mask_bytes, sizeof(LaneMask))) {
    throw std::overflow_error{"device workspace element count exceeds the 32-bit ABI"};
  }

  std::size_t component_sum = 0U;
  const std::size_t components[] = {
      requirements.source_bytes,
      requirements.target_bytes,
      requirements.selected_tile_bytes,
      requirements.tile_lane_mask_bytes,
      requirements.run_lane_mask_bytes,
      requirements.controller_bytes,
      requirements.status_bytes,
      requirements.instrumentation_bytes,
      requirements.engine_scratch_bytes,
  };
  for (const std::size_t component : components) {
    component_sum = checked_add(component_sum, component, "workspace byte total");
  }
  if (component_sum != requirements.total_bytes) {
    throw std::invalid_argument{"device workspace component total is inconsistent"};
  }
  std::size_t pinned_sum = 0U;
  const std::size_t pinned_components[] = {
      requirements.source_bytes,
      requirements.target_bytes,
      requirements.selected_tile_bytes,
      requirements.tile_lane_mask_bytes,
      requirements.run_lane_mask_bytes,
      requirements.controller_bytes,
  };
  for (const std::size_t component : pinned_components) {
    pinned_sum = checked_add(pinned_sum, component, "pinned staging byte total");
  }
  if (pinned_sum != requirements.pinned_staging_bytes ||
      checked_add(component_sum, pinned_sum, "combined workspace byte total") !=
          requirements.combined_total_bytes) {
    throw std::invalid_argument{
        "device workspace pinned or combined byte total is inconsistent"};
  }
}

[[nodiscard]] std::uint32_t count32(
    const std::size_t bytes,
    const std::size_t width) noexcept {
  return static_cast<std::uint32_t>(bytes / width);
}

[[nodiscard]] LaneMask low_lane_mask(const std::uint32_t lanes) noexcept {
  return lanes == maximum_batch_lanes
             ? std::numeric_limits<LaneMask>::max()
             : (LaneMask{1U} << lanes) - LaneMask{1U};
}

}  // namespace

class ReusableDeviceWorkspace::Impl final {
 public:
  PinnedHostBuffer source_staging;
  PinnedHostBuffer target_staging;
  PinnedHostBuffer selected_tile_staging;
  PinnedHostBuffer tile_lane_mask_staging;
  PinnedHostBuffer run_lane_mask_staging;
  PinnedHostBuffer controller_staging;

  DeviceBuffer sources;
  DeviceBuffer targets;
  DeviceBuffer selected_tiles;
  DeviceBuffer tile_lane_masks;
  DeviceBuffer run_lane_masks;
  DeviceBuffer controller;
  DeviceBuffer status;
  DeviceBuffer instrumentation;
  DeviceBuffer engine_scratch;

  HipEvent prepared_event{false};
  WorkspaceCapacity capacity{};
  DeviceWorkspaceView workspace_view{};
  EngineKind active_engine{EngineKind::jacobi_pull};
  std::uint64_t generation{};
  std::uint64_t allocation_events{};
  bool lease_active{};
  bool preparation_recorded{};
  bool device_work_may_be_in_flight{};
  void* lease_stream{};

  void synchronize_noexcept() noexcept {
    if (lease_active || device_work_may_be_in_flight) {
      // Phase 8 has no kernel-completion registration surface yet.  A leaked
      // active lease therefore requires the conservative device-wide fence.
      static_cast<void>(hipDeviceSynchronize());
    } else if (preparation_recorded) {
      try {
        prepared_event.synchronize();
      } catch (...) {
      }
    }
  }
};

ReusableDeviceWorkspace::ReusableDeviceWorkspace() : impl_{new Impl{}} {}

ReusableDeviceWorkspace::~ReusableDeviceWorkspace() noexcept {
  if (impl_ != nullptr) {
    impl_->synchronize_noexcept();
    delete impl_;
  }
}

ReusableDeviceWorkspace::ReusableDeviceWorkspace(
    ReusableDeviceWorkspace&& other) noexcept
    : impl_{std::exchange(other.impl_, nullptr)} {}

ReusableDeviceWorkspace& ReusableDeviceWorkspace::operator=(
    ReusableDeviceWorkspace&& other) noexcept {
  if (this != &other) {
    if (impl_ != nullptr) {
      impl_->synchronize_noexcept();
      delete impl_;
    }
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

bool ReusableDeviceWorkspace::reserve(
    const WorkspaceMemoryRequirements& requirements) {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot reserve a moved-from device workspace"};
  }
  if (impl_->lease_active) {
    throw std::logic_error{"retire the active query before growing its workspace"};
  }
  if (impl_->device_work_may_be_in_flight) {
    check(
        hipDeviceSynchronize(),
        "hipDeviceSynchronize(recover failed query preparation)");
    impl_->device_work_may_be_in_flight = false;
    impl_->preparation_recorded = false;
  }
  validate_workspace_requirements(requirements);

  bool grew = false;
  const auto reserve_device = [&grew](DeviceBuffer& buffer, const std::size_t bytes) {
    grew = buffer.reserve(bytes, BufferGrowth::geometric) || grew;
  };
  const auto reserve_host = [&grew](
                                PinnedHostBuffer& buffer,
                                const std::size_t bytes) {
    grew = buffer.reserve(bytes) || grew;
  };

  try {
    reserve_device(impl_->sources, requirements.source_bytes);
    reserve_device(impl_->targets, requirements.target_bytes);
    reserve_device(impl_->selected_tiles, requirements.selected_tile_bytes);
    reserve_device(impl_->tile_lane_masks, requirements.tile_lane_mask_bytes);
    reserve_device(impl_->run_lane_masks, requirements.run_lane_mask_bytes);
    reserve_device(impl_->controller, requirements.controller_bytes);
    reserve_device(impl_->status, requirements.status_bytes);
    reserve_device(impl_->instrumentation, requirements.instrumentation_bytes);
    reserve_device(impl_->engine_scratch, requirements.engine_scratch_bytes);

    reserve_host(impl_->source_staging, requirements.source_bytes);
    reserve_host(impl_->target_staging, requirements.target_bytes);
    reserve_host(impl_->selected_tile_staging, requirements.selected_tile_bytes);
    reserve_host(
        impl_->tile_lane_mask_staging, requirements.tile_lane_mask_bytes);
    reserve_host(
        impl_->run_lane_mask_staging, requirements.run_lane_mask_bytes);
    reserve_host(impl_->controller_staging, requirements.controller_bytes);
  } catch (...) {
    if (grew) {
      ++impl_->allocation_events;
    }
    throw;
  }
  if (grew) {
    ++impl_->allocation_events;
  }

  impl_->capacity = WorkspaceCapacity{
      impl_->sources.capacity_bytes(),
      impl_->targets.capacity_bytes(),
      impl_->selected_tiles.capacity_bytes(),
      impl_->tile_lane_masks.capacity_bytes(),
      impl_->run_lane_masks.capacity_bytes(),
      impl_->controller.capacity_bytes(),
      impl_->status.capacity_bytes(),
      impl_->instrumentation.capacity_bytes(),
      impl_->engine_scratch.capacity_bytes(),
  };
  return grew;
}

WorkspaceLease ReusableDeviceWorkspace::prepare_query_async(
    const WorkspaceMemoryRequirements& requirements,
    const RouteQuery& query,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> run_lane_masks,
    const DeviceController& controller,
    const HipStream& stream) {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot prepare a moved-from device workspace"};
  }
  if (impl_->lease_active) {
    throw std::logic_error{"retire the active query before preparing another"};
  }
  if (impl_->device_work_may_be_in_flight) {
    check(
        hipDeviceSynchronize(),
        "hipDeviceSynchronize(recover failed query preparation)");
    impl_->device_work_may_be_in_flight = false;
    impl_->preparation_recorded = false;
  }
  validate_workspace_requirements(requirements);
  if (query.sources.empty() || query.targets.empty() ||
      query.selected_tiles.empty()) {
    throw std::invalid_argument{"query preparation requires populated query sets"};
  }
  if (validate_device_controller(controller) != DeviceControllerError::none ||
      controller.engine_kind != static_cast<std::uint32_t>(requirements.engine)) {
    throw std::invalid_argument{"query controller does not match its engine plan"};
  }
  if ((controller.valid_lane_mask &
       ~low_lane_mask(requirements.lane_capacity)) != 0U) {
    throw std::invalid_argument{"controller lane mask exceeds workspace lane capacity"};
  }
  const auto mask_exceeds_valid_lanes = [&controller](const LaneMask mask) {
    return (mask & ~controller.valid_lane_mask) != 0U;
  };
  if (std::ranges::any_of(tile_lane_masks, mask_exceeds_valid_lanes) ||
      std::ranges::any_of(run_lane_masks, mask_exceeds_valid_lanes)) {
    throw std::invalid_argument{"query lane mask exceeds the controller's valid lanes"};
  }

  const std::size_t source_bytes = checked_multiply(
      query.sources.size(), sizeof(std::uint32_t), "query source bytes");
  const std::size_t target_bytes = checked_multiply(
      query.targets.size(), sizeof(std::uint32_t), "query target bytes");
  const std::size_t selected_tile_bytes = checked_multiply(
      query.selected_tiles.size(), sizeof(std::uint32_t), "selected tile bytes");
  const std::size_t tile_mask_bytes = checked_multiply(
      tile_lane_masks.size(), sizeof(LaneMask), "tile lane-mask bytes");
  const std::size_t run_mask_bytes = checked_multiply(
      run_lane_masks.size(), sizeof(LaneMask), "run lane-mask bytes");
  if (source_bytes != requirements.source_bytes ||
      target_bytes != requirements.target_bytes ||
      selected_tile_bytes != requirements.selected_tile_bytes ||
      tile_mask_bytes != requirements.tile_lane_mask_bytes ||
      run_mask_bytes > requirements.run_lane_mask_bytes) {
    throw std::invalid_argument{"query payload does not match its workspace plan"};
  }

  const auto require_capacity = [](const std::size_t have, const std::size_t need) {
    if (have < need) {
      throw std::length_error{
          "query preparation cannot allocate; call device workspace reserve first"};
    }
  };
  require_capacity(impl_->sources.capacity_bytes(), requirements.source_bytes);
  require_capacity(impl_->targets.capacity_bytes(), requirements.target_bytes);
  require_capacity(
      impl_->selected_tiles.capacity_bytes(), requirements.selected_tile_bytes);
  require_capacity(
      impl_->tile_lane_masks.capacity_bytes(), requirements.tile_lane_mask_bytes);
  require_capacity(
      impl_->run_lane_masks.capacity_bytes(), requirements.run_lane_mask_bytes);
  require_capacity(impl_->controller.capacity_bytes(), requirements.controller_bytes);
  require_capacity(impl_->status.capacity_bytes(), requirements.status_bytes);
  require_capacity(
      impl_->instrumentation.capacity_bytes(), requirements.instrumentation_bytes);
  require_capacity(
      impl_->engine_scratch.capacity_bytes(), requirements.engine_scratch_bytes);
  require_capacity(impl_->source_staging.capacity_bytes(), requirements.source_bytes);
  require_capacity(impl_->target_staging.capacity_bytes(), requirements.target_bytes);
  require_capacity(
      impl_->selected_tile_staging.capacity_bytes(),
      requirements.selected_tile_bytes);
  require_capacity(
      impl_->tile_lane_mask_staging.capacity_bytes(),
      requirements.tile_lane_mask_bytes);
  require_capacity(
      impl_->run_lane_mask_staging.capacity_bytes(),
      requirements.run_lane_mask_bytes);
  require_capacity(
      impl_->controller_staging.capacity_bytes(), requirements.controller_bytes);

  if (impl_->generation == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{"device workspace generation overflow"};
  }
  ++impl_->generation;
  impl_->preparation_recorded = false;

  auto* staged_sources =
      static_cast<std::uint32_t*>(impl_->source_staging.data());
  for (std::size_t index = 0U; index < query.sources.size(); ++index) {
    staged_sources[index] = query.sources[index].value();
  }
  auto* staged_targets =
      static_cast<std::uint32_t*>(impl_->target_staging.data());
  for (std::size_t index = 0U; index < query.targets.size(); ++index) {
    staged_targets[index] = query.targets[index].value();
  }
  auto* staged_tiles =
      static_cast<std::uint32_t*>(impl_->selected_tile_staging.data());
  for (std::size_t index = 0U; index < query.selected_tiles.size(); ++index) {
    staged_tiles[index] = query.selected_tiles[index].value();
  }
  if (tile_mask_bytes != 0U) {
    std::memcpy(
        impl_->tile_lane_mask_staging.data(),
        tile_lane_masks.data(),
        tile_mask_bytes);
  }
  if (run_mask_bytes != 0U) {
    std::memcpy(
        impl_->run_lane_mask_staging.data(),
        run_lane_masks.data(),
        run_mask_bytes);
  }
  std::memcpy(impl_->controller_staging.data(), &controller, sizeof(controller));

  // From this point onward, an exception may leave asynchronous work queued.
  // The recovery and noexcept teardown paths conservatively fence the device.
  impl_->lease_stream = stream.native_handle();
  impl_->device_work_may_be_in_flight = true;
  impl_->sources.copy_from_host_async(
      impl_->source_staging.data(), requirements.source_bytes, stream);
  impl_->targets.copy_from_host_async(
      impl_->target_staging.data(), requirements.target_bytes, stream);
  impl_->selected_tiles.copy_from_host_async(
      impl_->selected_tile_staging.data(),
      requirements.selected_tile_bytes,
      stream);
  impl_->tile_lane_masks.copy_from_host_async(
      impl_->tile_lane_mask_staging.data(),
      requirements.tile_lane_mask_bytes,
      stream);
  impl_->controller.copy_from_host_async(
      impl_->controller_staging.data(), requirements.controller_bytes, stream);

  // Clear the whole retained active requirement before materializing the
  // selected orientation's exact run-mask prefix.  The unused CSR/CSC tail is
  // then inaccessible through the view.
  impl_->run_lane_masks.clear_async(requirements.run_lane_mask_bytes, stream);
  impl_->run_lane_masks.copy_from_host_async(
      impl_->run_lane_mask_staging.data(), run_mask_bytes, stream);
  impl_->status.clear_async(requirements.status_bytes, stream);
  impl_->instrumentation.clear_async(requirements.instrumentation_bytes, stream);
  impl_->engine_scratch.clear_async(requirements.engine_scratch_bytes, stream);

  impl_->active_engine = requirements.engine;
  impl_->workspace_view = DeviceWorkspaceView{
      impl_->generation,
      static_cast<std::uint64_t>(requirements.engine_scratch_bytes),
      static_cast<std::uint32_t>(requirements.engine),
      count32(requirements.source_bytes, sizeof(std::uint32_t)),
      count32(requirements.target_bytes, sizeof(std::uint32_t)),
      count32(requirements.selected_tile_bytes, sizeof(std::uint32_t)),
      count32(requirements.tile_lane_mask_bytes, sizeof(LaneMask)),
      count32(run_mask_bytes, sizeof(LaneMask)),
      device_pointer<const std::uint32_t>(impl_->sources.data()),
      device_pointer<const std::uint32_t>(impl_->targets.data()),
      device_pointer<const std::uint32_t>(impl_->selected_tiles.data()),
      device_pointer<LaneMask>(impl_->tile_lane_masks.data()),
      device_pointer<LaneMask>(impl_->run_lane_masks.data()),
      device_pointer<DeviceController>(impl_->controller.data()),
      device_pointer<DeviceRunStatus>(impl_->status.data()),
      requirements.instrumentation_bytes == 0U
          ? nullptr
          : device_pointer<DeviceWorkStatistics>(impl_->instrumentation.data()),
      device_pointer<std::uint8_t>(impl_->engine_scratch.data()),
  };
  impl_->prepared_event.record(stream);
  impl_->preparation_recorded = true;
  impl_->lease_active = true;
  return WorkspaceLease{requirements.engine, impl_->generation};
}

bool ReusableDeviceWorkspace::accepts(const WorkspaceLease& lease) const noexcept {
  return impl_ != nullptr && impl_->lease_active &&
         lease.engine == impl_->active_engine && lease.generation == impl_->generation;
}

DeviceWorkspaceView ReusableDeviceWorkspace::view(
    const WorkspaceLease& lease) const {
  if (!accepts(lease)) {
    throw std::invalid_argument{"device workspace rejected a stale lease"};
  }
  return impl_->workspace_view;
}

void ReusableDeviceWorkspace::wait_until_prepared(
    const HipStream& stream,
    const WorkspaceLease& lease) const {
  if (!accepts(lease) || !impl_->preparation_recorded) {
    throw std::invalid_argument{"cannot wait on an unrecognized workspace lease"};
  }
  if (stream.native_handle() != impl_->lease_stream) {
    throw std::invalid_argument{"workspace lease operations require one stream"};
  }
  impl_->prepared_event.wait(stream);
}

void ReusableDeviceWorkspace::synchronize_preparation(
    const WorkspaceLease& lease) const {
  if (!accepts(lease) || !impl_->preparation_recorded) {
    throw std::invalid_argument{"cannot synchronize an unrecognized workspace lease"};
  }
  impl_->prepared_event.synchronize();
}

void ReusableDeviceWorkspace::download_status_async(
    const WorkspaceLease& lease,
    DeviceRunStatus& destination,
    const HipStream& stream) const {
  if (!accepts(lease)) {
    throw std::invalid_argument{"cannot download status through a stale lease"};
  }
  if (stream.native_handle() != impl_->lease_stream) {
    throw std::invalid_argument{"workspace lease operations require one stream"};
  }
  impl_->status.copy_to_host_async(&destination, sizeof(destination), stream);
}

void ReusableDeviceWorkspace::download_instrumentation_async(
    const WorkspaceLease& lease,
    DeviceWorkStatistics& destination,
    const HipStream& stream) const {
  if (!accepts(lease)) {
    throw std::invalid_argument{"cannot download instrumentation through a stale lease"};
  }
  if (stream.native_handle() != impl_->lease_stream) {
    throw std::invalid_argument{"workspace lease operations require one stream"};
  }
  if (impl_->workspace_view.instrumentation == nullptr) {
    throw std::logic_error{"instrumentation was not reserved for this query"};
  }
  impl_->instrumentation.copy_to_host_async(
      &destination, sizeof(destination), stream);
}

void ReusableDeviceWorkspace::retire_query(
    const WorkspaceLease& lease,
    const HipStream& stream) {
  if (!accepts(lease)) {
    throw std::invalid_argument{"cannot retire an unrecognized workspace lease"};
  }
  if (stream.native_handle() != impl_->lease_stream) {
    throw std::invalid_argument{"workspace lease operations require one stream"};
  }
  stream.synchronize();
  impl_->lease_active = false;
  impl_->preparation_recorded = false;
  impl_->device_work_may_be_in_flight = false;
  impl_->lease_stream = nullptr;
  impl_->workspace_view = {};
}

const WorkspaceCapacity& ReusableDeviceWorkspace::capacity() const noexcept {
  static constexpr WorkspaceCapacity empty{};
  return impl_ == nullptr ? empty : impl_->capacity;
}

std::uint64_t ReusableDeviceWorkspace::allocation_events() const noexcept {
  return impl_ == nullptr ? 0U : impl_->allocation_events;
}


}  // namespace bfnew::hip
