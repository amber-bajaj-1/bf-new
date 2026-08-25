#include "bfnew/workspace.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace bfnew {
namespace {

[[nodiscard]] std::size_t checked_add(
    const std::size_t left,
    const std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error{"workspace byte total overflow"};
  }
  return left + right;
}

[[nodiscard]] std::size_t checked_multiply(
    const std::size_t count,
    const std::size_t element_size) {
  if (element_size != 0U &&
      count > std::numeric_limits<std::size_t>::max() / element_size) {
    throw std::overflow_error{"workspace component byte size overflow"};
  }
  return count * element_size;
}

void add_component(std::size_t& total, const std::size_t component) {
  total = checked_add(total, component);
}

void require_device_count32(
    const std::size_t count,
    const char* const component_name) {
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{
        std::string{component_name} + " count exceeds the 32-bit device ABI"};
  }
}

[[nodiscard]] bool grow_geometrically(
    std::size_t& capacity,
    const std::size_t requested) {
  if (requested <= capacity) {
    return false;
  }
  std::size_t next = capacity == 0U ? requested : capacity;
  while (next < requested) {
    if (next > std::numeric_limits<std::size_t>::max() / 2U) {
      next = requested;
      break;
    }
    next *= 2U;
  }
  capacity = next;
  return true;
}

}  // namespace

WorkspaceMemoryRequirements estimate_workspace_memory(
    const EngineKind engine,
    const RouteQuery& query,
    const std::size_t tile_count,
    const std::size_t csr_run_count,
    const std::size_t csc_run_count,
    const std::uint32_t lane_capacity,
    const InstrumentationLevel instrumentation,
    const std::size_t engine_scratch_bytes) {
  if (lane_capacity == 0U || lane_capacity > maximum_batch_lanes) {
    throw std::invalid_argument{"Phase 8 workspace lane capacity must be in [1,32]"};
  }
  if (query.sources.empty() || query.targets.empty() || query.selected_tiles.empty()) {
    throw std::invalid_argument{"workspace planning requires a populated RouteQuery"};
  }
  require_device_count32(query.sources.size(), "source");
  require_device_count32(query.targets.size(), "target");
  require_device_count32(query.selected_tiles.size(), "selected tile");
  require_device_count32(tile_count, "tile lane mask");
  require_device_count32(
      std::max(csr_run_count, csc_run_count), "run lane mask");
  GpuRunOptions options;
  options.engine = engine;
  options.instrumentation = instrumentation;
  if (validate_gpu_run_options(options) != GpuRunOptionsError::none) {
    throw std::invalid_argument{"workspace request uses an invalid engine or instrumentation"};
  }

  WorkspaceMemoryRequirements requirements;
  requirements.engine = engine;
  requirements.instrumentation = instrumentation;
  requirements.lane_capacity = lane_capacity;
  requirements.source_bytes = checked_multiply(query.sources.size(), sizeof(std::uint32_t));
  requirements.target_bytes = checked_multiply(query.targets.size(), sizeof(std::uint32_t));
  requirements.selected_tile_bytes =
      checked_multiply(query.selected_tiles.size(), sizeof(std::uint32_t));
  requirements.tile_lane_mask_bytes =
      checked_multiply(tile_count, sizeof(LaneMask));
  requirements.run_lane_mask_bytes = checked_multiply(
      std::max(csr_run_count, csc_run_count), sizeof(LaneMask));
  requirements.controller_bytes = sizeof(DeviceController);
  requirements.status_bytes = sizeof(DeviceRunStatus);
  requirements.instrumentation_bytes =
      instrumentation == InstrumentationLevel::none
          ? 0U
          : sizeof(DeviceWorkStatistics);
  requirements.engine_scratch_bytes = engine_scratch_bytes;

  add_component(requirements.total_bytes, requirements.source_bytes);
  add_component(requirements.total_bytes, requirements.target_bytes);
  add_component(requirements.total_bytes, requirements.selected_tile_bytes);
  add_component(requirements.total_bytes, requirements.tile_lane_mask_bytes);
  add_component(requirements.total_bytes, requirements.run_lane_mask_bytes);
  add_component(requirements.total_bytes, requirements.controller_bytes);
  add_component(requirements.total_bytes, requirements.status_bytes);
  add_component(requirements.total_bytes, requirements.instrumentation_bytes);
  add_component(requirements.total_bytes, requirements.engine_scratch_bytes);

  add_component(requirements.pinned_staging_bytes, requirements.source_bytes);
  add_component(requirements.pinned_staging_bytes, requirements.target_bytes);
  add_component(
      requirements.pinned_staging_bytes, requirements.selected_tile_bytes);
  add_component(
      requirements.pinned_staging_bytes, requirements.tile_lane_mask_bytes);
  add_component(
      requirements.pinned_staging_bytes, requirements.run_lane_mask_bytes);
  add_component(requirements.pinned_staging_bytes, requirements.controller_bytes);
  requirements.combined_total_bytes = checked_add(
      requirements.total_bytes, requirements.pinned_staging_bytes);
  return requirements;
}

std::size_t WorkspaceCapacity::total_bytes() const {
  std::size_t total = 0U;
  add_component(total, source_bytes);
  add_component(total, target_bytes);
  add_component(total, selected_tile_bytes);
  add_component(total, tile_lane_mask_bytes);
  add_component(total, run_lane_mask_bytes);
  add_component(total, controller_bytes);
  add_component(total, status_bytes);
  add_component(total, instrumentation_bytes);
  add_component(total, engine_scratch_bytes);
  return total;
}

WorkspaceReservationResult ReusableWorkspaceReservation::begin_query(
    const WorkspaceMemoryRequirements& requirements) {
  std::size_t component_sum = 0U;
  add_component(component_sum, requirements.source_bytes);
  add_component(component_sum, requirements.target_bytes);
  add_component(component_sum, requirements.selected_tile_bytes);
  add_component(component_sum, requirements.tile_lane_mask_bytes);
  add_component(component_sum, requirements.run_lane_mask_bytes);
  add_component(component_sum, requirements.controller_bytes);
  add_component(component_sum, requirements.status_bytes);
  add_component(component_sum, requirements.instrumentation_bytes);
  add_component(component_sum, requirements.engine_scratch_bytes);
  if (component_sum != requirements.total_bytes) {
    throw std::invalid_argument{"workspace requirement component sum is inconsistent"};
  }
  std::size_t pinned_sum = 0U;
  add_component(pinned_sum, requirements.source_bytes);
  add_component(pinned_sum, requirements.target_bytes);
  add_component(pinned_sum, requirements.selected_tile_bytes);
  add_component(pinned_sum, requirements.tile_lane_mask_bytes);
  add_component(pinned_sum, requirements.run_lane_mask_bytes);
  add_component(pinned_sum, requirements.controller_bytes);
  if (pinned_sum != requirements.pinned_staging_bytes ||
      checked_add(component_sum, pinned_sum) !=
          requirements.combined_total_bytes) {
    throw std::invalid_argument{
        "workspace pinned or combined byte total is inconsistent"};
  }

  GpuRunOptions options;
  options.engine = requirements.engine;
  options.instrumentation = requirements.instrumentation;
  const std::size_t expected_instrumentation_bytes =
      requirements.instrumentation == InstrumentationLevel::none
          ? 0U
          : sizeof(DeviceWorkStatistics);
  if (requirements.lane_capacity == 0U ||
      requirements.lane_capacity > maximum_batch_lanes ||
      validate_gpu_run_options(options) != GpuRunOptionsError::none ||
      requirements.controller_bytes != sizeof(DeviceController) ||
      requirements.status_bytes != sizeof(DeviceRunStatus) ||
      requirements.instrumentation_bytes != expected_instrumentation_bytes) {
    throw std::invalid_argument{"workspace requirement has an invalid engine or lane capacity"};
  }

  bool grew = false;
  grew = grow_geometrically(capacity_.source_bytes, requirements.source_bytes) || grew;
  grew = grow_geometrically(capacity_.target_bytes, requirements.target_bytes) || grew;
  grew = grow_geometrically(
             capacity_.selected_tile_bytes, requirements.selected_tile_bytes) ||
         grew;
  grew = grow_geometrically(
             capacity_.tile_lane_mask_bytes, requirements.tile_lane_mask_bytes) ||
         grew;
  grew = grow_geometrically(
             capacity_.run_lane_mask_bytes, requirements.run_lane_mask_bytes) ||
         grew;
  grew = grow_geometrically(capacity_.controller_bytes, requirements.controller_bytes) ||
         grew;
  grew = grow_geometrically(capacity_.status_bytes, requirements.status_bytes) || grew;
  grew = grow_geometrically(
             capacity_.instrumentation_bytes, requirements.instrumentation_bytes) ||
         grew;
  // A single shared scratch allocation retains the maximum requested capacity;
  // there is never one large allocation per engine.
  grew = grow_geometrically(
             capacity_.engine_scratch_bytes, requirements.engine_scratch_bytes) ||
         grew;
  if (grew) {
    ++growth_events_;
  }

  const bool engine_changed =
      has_active_engine_ && active_engine_ != requirements.engine;
  active_engine_ = requirements.engine;
  has_active_engine_ = true;
  if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{"workspace generation overflow"};
  }
  ++generation_;
  return WorkspaceReservationResult{
      WorkspaceLease{active_engine_, generation_},
      grew,
      engine_changed,
      true,
  };
}

bool ReusableWorkspaceReservation::accepts(
    const WorkspaceLease& lease) const noexcept {
  return has_active_engine_ && lease.engine == active_engine_ &&
         lease.generation == generation_;
}

}  // namespace bfnew
