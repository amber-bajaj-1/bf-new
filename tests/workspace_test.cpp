#include "bfnew/gpu_api.hpp"
#include "bfnew/graph.hpp"
#include "bfnew/hip/runtime.hpp"
#include "bfnew/query.hpp"
#include "bfnew/spatial.hpp"
#include "bfnew/workspace.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void expect(const bool condition) {
  if (!condition) {
    throw std::runtime_error{"workspace test expectation failed"};
  }
}

}  // namespace

int main() {
  using namespace bfnew;

  GpuRunOptions options;
  expect(validate_gpu_run_options(options) == GpuRunOptionsError::none);
  options.control_mode = ControlMode::chunked_host_poll;
  for (const std::uint32_t rounds : {2U, 4U, 8U, 16U, 32U}) {
    options.rounds_per_chunk = rounds;
    expect(validate_gpu_run_options(options) == GpuRunOptionsError::none);
  }
  options.rounds_per_chunk = 0U;
  expect(validate_gpu_run_options(options) ==
         GpuRunOptionsError::invalid_rounds_per_chunk);
  options.rounds_per_chunk = 8U;
  options.grid_policy = GridPolicy::fixed_blocks_per_wgp;
  expect(validate_gpu_run_options(options) ==
         GpuRunOptionsError::invalid_blocks_per_wgp);
  options.blocks_per_wgp = 2U;
  expect(validate_gpu_run_options(options) == GpuRunOptionsError::none);

  for (const EngineKind engine : {
           EngineKind::jacobi_pull,
           EngineKind::dense_chaotic_push,
           EngineKind::frontier_push}) {
    options.engine = engine;
    const DeviceController controller =
        initialize_device_controller(options, LaneMask{0b1011U}, 3U);
    expect(validate_device_controller(controller) == DeviceControllerError::none);
    expect(controller.valid_lane_mask == 0b1011U);
    expect(controller.active_lane_mask == controller.valid_lane_mask);
    expect(controller.execute_lane_mask == controller.valid_lane_mask);
    expect(controller.frontier_size[0] ==
           (engine == EngineKind::frontier_push ? 3U : 0U));
    if (engine == EngineKind::jacobi_pull) {
      expect(controller.distance_read_slot != controller.distance_write_slot);
    } else {
      expect(controller.distance_read_slot == 0U);
      expect(controller.distance_write_slot == 0U);
    }
  }

  DeviceController invalid = initialize_device_controller(options, 1U, 1U);
  invalid.execute_lane_mask = 0U;
  expect(validate_device_controller(invalid) ==
         DeviceControllerError::execute_mask_mismatch);
  invalid = initialize_device_controller(options, 1U, 1U);
  invalid.done = 1U;
  invalid.active_lane_mask = 0U;
  invalid.converged_lane_mask = invalid.valid_lane_mask;
  invalid.execute_lane_mask = 0U;
  invalid.stop_reason = static_cast<std::uint32_t>(DeviceStopReason::converged);
  expect(validate_device_controller(invalid) == DeviceControllerError::none);
  invalid.active_lane_mask = invalid.valid_lane_mask;
  invalid.converged_lane_mask = 0U;
  expect(validate_device_controller(invalid) ==
         DeviceControllerError::inconsistent_terminal_state);

  invalid = initialize_device_controller(options, 1U, 1U);
  invalid.done = 1U;
  invalid.active_lane_mask = 0U;
  invalid.execute_lane_mask = 0U;
  invalid.stop_reason =
      static_cast<std::uint32_t>(DeviceStopReason::queue_overflow);
  invalid.error_bits = device_error::device_failure;
  expect(validate_device_controller(invalid) ==
         DeviceControllerError::inconsistent_error_state);

  DeviceRunStatus status;
  status.final_distance_slot = 1U;
  status.converged = 1U;
  status.reached_target_mask = 0b0011U;
  status.valid_lane_mask = 0b1011U;
  status.converged_lane_mask = 0b1011U;
  status.stop_reason = static_cast<std::uint32_t>(DeviceStopReason::converged);
  expect(validate_device_run_status(status) == DeviceRunStatusError::none);
  status.converged = 0U;
  status.converged_lane_mask = 0U;
  status.reached_target_mask = 0U;
  status.bounding_box_miss_mask = 0b0001U;
  status.stop_reason =
      static_cast<std::uint32_t>(DeviceStopReason::device_failure);
  status.error_bits = device_error::device_failure;
  expect(validate_device_run_status(status) ==
         DeviceRunStatusError::error_reported_as_bounding_box_miss);

  const ResourceClassId resource{1U};
  const InputGraph input(
      {
          VertexMetadata::located(0, 0, resource),
          VertexMetadata::located(1, 0, resource),
          VertexMetadata::located(2, 0, resource),
      },
      {});
  const UniformGridPartitioner partitioner(SpatialOrderConfig{0, 0, 1U, 1U});
  const PartitionedGraph partitioned = partitioner.partition(input);
  const std::array source_terminals{
      partitioned.graph.old_to_new()[0U], partitioned.graph.old_to_new()[1U]};
  const std::array target_terminals{
      partitioned.graph.old_to_new()[2U], partitioned.graph.old_to_new()[2U]};
  const RouteQuery query = make_route_query(
      QueryId{7U}, partitioned.graph, source_terminals, target_terminals, 0U);

  const WorkspaceMemoryRequirements first = estimate_workspace_memory(
      EngineKind::jacobi_pull,
      query,
      partitioned.graph.tile_coordinates().size(),
      5U,
      6U,
      1U,
      InstrumentationLevel::light,
      4096U);
  expect(first.source_bytes == query.sources.size() * sizeof(std::uint32_t));
  expect(first.target_bytes == query.targets.size() * sizeof(std::uint32_t));
  expect(first.instrumentation_bytes == sizeof(DeviceWorkStatistics));
  expect(first.run_lane_mask_bytes == 6U * sizeof(LaneMask));
  expect(first.total_bytes > first.engine_scratch_bytes);
  expect(first.pinned_staging_bytes > first.controller_bytes);
  expect(first.combined_total_bytes ==
         first.total_bytes + first.pinned_staging_bytes);

  ReusableWorkspaceReservation reservation;
  const WorkspaceReservationResult first_reservation =
      reservation.begin_query(first);
  expect(first_reservation.capacity_grew);
  expect(!first_reservation.engine_changed);
  expect(first_reservation.clear_active_prefixes);
  expect(reservation.accepts(first_reservation.lease));
  expect(reservation.capacity().total_bytes() == first.total_bytes);

  const WorkspaceReservationResult reused = reservation.begin_query(first);
  expect(!reused.capacity_grew);
  expect(!reused.engine_changed);
  expect(!reservation.accepts(first_reservation.lease));
  expect(reservation.accepts(reused.lease));
  expect(reservation.growth_events() == 1U);

  const WorkspaceMemoryRequirements switched = estimate_workspace_memory(
      EngineKind::frontier_push,
      query,
      partitioned.graph.tile_coordinates().size(),
      5U,
      6U,
      1U,
      InstrumentationLevel::none,
      1024U);
  const WorkspaceReservationResult switched_reservation =
      reservation.begin_query(switched);
  expect(!switched_reservation.capacity_grew);
  expect(switched_reservation.engine_changed);
  expect(!reservation.accepts(reused.lease));
  expect(reservation.accepts(switched_reservation.lease));
  expect(reservation.capacity().engine_scratch_bytes == 4096U);

  WorkspaceMemoryRequirements modest_growth = switched;
  modest_growth.source_bytes += sizeof(std::uint32_t);
  modest_growth.total_bytes += sizeof(std::uint32_t);
  modest_growth.pinned_staging_bytes += sizeof(std::uint32_t);
  modest_growth.combined_total_bytes += 2U * sizeof(std::uint32_t);
  const std::size_t previous_source_capacity = reservation.capacity().source_bytes;
  const WorkspaceReservationResult grown = reservation.begin_query(modest_growth);
  expect(grown.capacity_grew);
  expect(reservation.capacity().source_bytes >= modest_growth.source_bytes);
  expect(reservation.capacity().source_bytes >= previous_source_capacity * 2U);

  WorkspaceMemoryRequirements within_spare_capacity = modest_growth;
  within_spare_capacity.source_bytes = reservation.capacity().source_bytes;
  const std::size_t source_delta =
      within_spare_capacity.source_bytes - switched.source_bytes;
  within_spare_capacity.total_bytes = switched.total_bytes + source_delta;
  within_spare_capacity.pinned_staging_bytes =
      switched.pinned_staging_bytes + source_delta;
  within_spare_capacity.combined_total_bytes =
      switched.combined_total_bytes + 2U * source_delta;
  const WorkspaceReservationResult spare =
      reservation.begin_query(within_spare_capacity);
  expect(!spare.capacity_grew);

  bool overflow_rejected = false;
  try {
    static_cast<void>(estimate_workspace_memory(
        EngineKind::jacobi_pull,
        query,
        std::numeric_limits<std::size_t>::max(),
        1U,
        1U,
        1U,
        InstrumentationLevel::none,
        0U));
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  expect(overflow_rejected);

  return 0;
}
