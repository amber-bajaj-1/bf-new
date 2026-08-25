#include "bfnew/hip/batched_jacobi.hpp"

#include "bfnew/jacobi_pull.hpp"
#include "bfnew/workspace.hpp"

#include "batched_workspace_internal.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bfnew::hip {
namespace {

namespace cg = ::cooperative_groups;

inline constexpr std::uint32_t phase14_wave_width = 32U;

[[nodiscard]] constexpr bool supported_lane_width(
    const std::uint32_t width) noexcept {
  return width == 1U || width == 8U || width == 16U || width == 32U;
}

[[nodiscard]] constexpr LaneMask width_mask(
    const std::uint32_t width) noexcept {
  return width == 32U ? std::numeric_limits<LaneMask>::max()
                      : (LaneMask{1U} << width) - LaneMask{1U};
}

[[nodiscard]] hipStream_t as_stream(const void* const handle) noexcept {
  return reinterpret_cast<hipStream_t>(const_cast<void*>(handle));
}

void check(
    const hipError_t status,
    const std::string_view expression,
    const std::source_location location = std::source_location::current()) {
  throw_if_hip_error(static_cast<std::int32_t>(status), expression, location);
}

[[nodiscard]] std::size_t checked_multiply(
    const std::size_t left,
    const std::size_t right,
    const std::string_view what) {
  if (right != 0U && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  return left * right;
}

[[nodiscard]] std::uint64_t checked_multiply_u64(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view what) {
  if (right != 0U && left > std::numeric_limits<std::uint64_t>::max() / right) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  return left * right;
}

void checked_add_u64(
    std::uint64_t& destination,
    const std::uint64_t increment,
    const std::string_view what) {
  if (destination >
      std::numeric_limits<std::uint64_t>::max() - increment) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  destination += increment;
}

[[nodiscard]] std::uint32_t checked_grid_blocks(
    const std::uint32_t blocks_per_wgp,
    const std::uint32_t wgp_count) {
  const std::uint64_t blocks =
      static_cast<std::uint64_t>(blocks_per_wgp) * wgp_count;
  if (blocks == 0U || blocks > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{
        "batched Jacobi grid block count exceeds the HIP launch ABI"};
  }
  return static_cast<std::uint32_t>(blocks);
}

[[nodiscard]] bool valid_offsets(
    const std::vector<std::uint32_t>& offsets,
    const std::size_t buckets,
    const std::size_t entries) noexcept {
  return offsets.size() == buckets + 1U && !offsets.empty() &&
         offsets.front() == 0U && offsets.back() == entries &&
         std::is_sorted(offsets.begin(), offsets.end());
}

void validate_batch_description_shape(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const BatchDeviceDescription& batch,
    const bool validate_device_materialized_edge_estimates) {
  if (!supported_lane_width(batch.lane_width) ||
      batch.valid_lane_mask == 0U ||
      (batch.valid_lane_mask & ~width_mask(batch.lane_width)) != 0U ||
      (batch.valid_lane_mask & (batch.valid_lane_mask + 1U)) != 0U) {
    throw std::invalid_argument{
        "batched Jacobi requires width 1/8/16/32 and a nonempty low valid mask"};
  }
  const std::size_t width = batch.lane_width;
  if (batch.query_ids_by_lane.size() != width ||
      batch.expansion_generations_by_lane.size() != width ||
      batch.selected_vertex_counts_by_lane.size() != width ||
      batch.selected_edge_estimates_by_lane.size() != width ||
      !valid_offsets(batch.source_offsets, width, batch.sources.size()) ||
      !valid_offsets(batch.target_offsets, width, batch.targets.size())) {
    throw std::invalid_argument{
        "batched Jacobi terminal or lane metadata has an invalid shape"};
  }
  if (batch.reached_lane_mask != 0U || batch.miss_lane_mask != 0U) {
    throw std::invalid_argument{
        "batched Jacobi requires zero initial reached/miss masks"};
  }
  const bool retained_masks =
      batch.run_representation ==
      BatchRunRepresentation::retained_per_run_masks;
  const bool device_materialized_masks =
      batch.run_representation ==
      BatchRunRepresentation::device_materialized_run_masks;
  if (batch.tile_lane_masks.size() != graph.tile_coordinates().size() ||
      (!retained_masks && !device_materialized_masks) ||
      !batch.run_representation_initialized ||
      (retained_masks &&
       batch.csc_run_lane_masks.size() !=
           tile_runs.csc_run_source_tiles.size()) ||
      (device_materialized_masks &&
       (!batch.csr_run_lane_masks.empty() ||
        !batch.csc_run_lane_masks.empty() ||
        !batch.touched_csr_runs.empty() ||
        !batch.touched_csc_runs.empty() ||
        !batch.csr_run_descriptors.empty() ||
        !batch.csc_run_descriptors.empty() ||
        !batch.csr_descriptor_offsets_by_union_vertex.empty() ||
        !batch.csc_descriptor_offsets_by_union_vertex.empty() ||
        batch.run_report != BatchRunPreparationReport{}))) {
    throw std::invalid_argument{
        "batched Jacobi requires retained masks or device-materialized masks"};
  }
  if (batch.union_tiles.empty() ||
      batch.selected_vertex_ranges.size() != batch.union_tiles.size()) {
    throw std::invalid_argument{
        "batched Jacobi requires one selected range per union tile"};
  }
  const auto tile_offsets = graph.tile_vertex_offsets();
  std::vector<bool> union_tile_flags(graph.tile_coordinates().size(), false);
  std::uint32_t preceding_tile = 0U;
  bool has_preceding_tile = false;
  for (std::size_t index = 0U; index < batch.union_tiles.size(); ++index) {
    const std::uint32_t tile = batch.union_tiles[index];
    if (tile >= graph.tile_coordinates().size() ||
        (has_preceding_tile && tile <= preceding_tile)) {
      throw std::invalid_argument{"batch union tiles are not canonical"};
    }
    const BatchVertexRange range = batch.selected_vertex_ranges[index];
    if (range.begin != tile_offsets[tile] ||
        range.end != tile_offsets[tile + 1U] || range.begin > range.end ||
        range.lane_mask != batch.tile_lane_masks[tile] ||
        range.lane_mask == 0U) {
      throw std::invalid_argument{
          "batch selected range disagrees with its union tile mask"};
    }
    union_tile_flags[tile] = true;
    preceding_tile = tile;
    has_preceding_tile = true;
  }
  for (std::size_t tile = 0U; tile < batch.tile_lane_masks.size(); ++tile) {
    const LaneMask mask = batch.tile_lane_masks[tile];
    if ((mask & ~batch.valid_lane_mask) != 0U) {
      throw std::invalid_argument{
          "batch tile mask contains an invalid or padded lane"};
    }
    if (union_tile_flags[tile] != (mask != 0U)) {
      throw std::invalid_argument{
          "batch nonzero tile masks must correspond exactly to union ranges"};
    }
  }
  std::vector<std::uint64_t> observed_vertex_counts(width, 0U);
  for (const BatchVertexRange range : batch.selected_vertex_ranges) {
    const std::uint64_t vertices = range.end - range.begin;
    for (std::size_t lane = 0U; lane < width; ++lane) {
      if ((range.lane_mask & (LaneMask{1U} << lane)) != 0U) {
        checked_add_u64(
            observed_vertex_counts[lane],
            vertices,
            "batch selected vertex count");
      }
    }
  }
  if (observed_vertex_counts != batch.selected_vertex_counts_by_lane) {
    throw std::invalid_argument{
        "batch selected vertex counts disagree with admitted ranges"};
  }
  if (device_materialized_masks &&
      validate_device_materialized_edge_estimates) {
    // Full/evidence modes already perform selected-run host accounting after
    // execution. Validate their planner estimates here as a correctness gate.
    // Compact timing/path modes skip this diagnostic scan and rely on the
    // previously validated batch plan, preserving run-free hot preparation.
    std::vector<std::uint64_t> observed_edge_counts(width, 0U);
    for (const BatchVertexRange range : batch.selected_vertex_ranges) {
      for (std::uint32_t vertex = range.begin; vertex < range.end; ++vertex) {
        const EdgeOffset run_begin =
            tile_runs.csc_column_run_offsets[vertex];
        const EdgeOffset run_end =
            tile_runs.csc_column_run_offsets[vertex + 1U];
        for (EdgeOffset run = run_begin; run < run_end; ++run) {
          const TileId source_tile = tile_runs.csc_run_source_tiles[run];
          const LaneMask mask =
              range.lane_mask & batch.tile_lane_masks[source_tile.value()];
          const std::uint64_t edges =
              tile_runs.csc_run_edge_offsets[run + 1U] -
              tile_runs.csc_run_edge_offsets[run];
          for (std::size_t lane = 0U; lane < width; ++lane) {
            if ((mask & (LaneMask{1U} << lane)) != 0U) {
              checked_add_u64(
                  observed_edge_counts[lane],
                  edges,
                  "batch selected edge count");
            }
          }
        }
      }
    }
    if (observed_edge_counts != batch.selected_edge_estimates_by_lane) {
      throw std::invalid_argument{
          "batch selected edge estimates disagree with admitted CSC runs"};
    }
  }
  if (retained_masks) {
    std::vector<bool> selected_destination_run(
        batch.csc_run_lane_masks.size(), false);
    for (const BatchVertexRange range : batch.selected_vertex_ranges) {
      for (std::uint32_t vertex = range.begin; vertex < range.end; ++vertex) {
        const std::size_t run_begin =
            static_cast<std::size_t>(tile_runs.csc_column_run_offsets[vertex]);
        const std::size_t run_end = static_cast<std::size_t>(
            tile_runs.csc_column_run_offsets[vertex + 1U]);
        for (std::size_t run = run_begin; run < run_end; ++run) {
          selected_destination_run[run] = true;
          const std::uint32_t source_tile =
              tile_runs.csc_run_source_tiles[run].value();
          const LaneMask expected =
              range.lane_mask & batch.tile_lane_masks[source_tile];
          if (batch.csc_run_lane_masks[run] != expected) {
            throw std::invalid_argument{
                "batch retained CSC mask differs from endpoint-tile admission"};
          }
        }
      }
    }
    for (std::size_t run = 0U; run < batch.csc_run_lane_masks.size(); ++run) {
      const LaneMask mask = batch.csc_run_lane_masks[run];
      if ((mask & ~batch.valid_lane_mask) != 0U ||
          (!selected_destination_run[run] && mask != 0U)) {
        throw std::invalid_argument{
            "batch CSC mask is nonzero outside admitted destination columns"};
      }
    }
  }
  for (std::size_t lane = 0U; lane < width; ++lane) {
    const LaneMask bit = LaneMask{1U} << lane;
    const std::size_t source_count =
        batch.source_offsets[lane + 1U] - batch.source_offsets[lane];
    const std::size_t target_count =
        batch.target_offsets[lane + 1U] - batch.target_offsets[lane];
    const bool valid = (batch.valid_lane_mask & bit) != 0U;
    if ((valid && (source_count == 0U || target_count == 0U)) ||
        (!valid && (source_count != 0U || target_count != 0U))) {
      throw std::invalid_argument{
          "valid lanes require terminals and padded lanes require empty slices"};
    }
  }
  const auto require_terminal = [&](const std::uint32_t vertex,
                                    const std::size_t lane) {
    if (vertex >= graph.vertex_count()) {
      throw std::invalid_argument{"batch terminal vertex is out of range"};
    }
    const std::uint32_t tile = graph.owner_tiles()[vertex].value();
    if ((batch.tile_lane_masks[tile] & (LaneMask{1U} << lane)) == 0U) {
      throw std::invalid_argument{
          "batch terminal owner tile does not admit its lane"};
    }
  };
  for (std::size_t lane = 0U; lane < width; ++lane) {
    for (std::size_t index = batch.source_offsets[lane];
         index < batch.source_offsets[lane + 1U];
         ++index) {
      require_terminal(batch.sources[index], lane);
    }
    for (std::size_t index = batch.target_offsets[lane];
         index < batch.target_offsets[lane + 1U];
         ++index) {
      require_terminal(batch.targets[index], lane);
    }
  }
}

struct DeviceProperties {
  std::uint32_t wgp_count{};
  std::uint32_t maximum_threads_per_block{};
  std::uint32_t wave_width{};
  bool cooperative_launch{};
};

[[nodiscard]] DeviceProperties current_device_properties() {
  int device = 0;
  check(hipGetDevice(&device), "hipGetDevice");
  hipDeviceProp_t properties{};
  check(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
  int cooperative = 0;
  check(
      hipDeviceGetAttribute(
          &cooperative, hipDeviceAttributeCooperativeLaunch, device),
      "hipDeviceGetAttribute(cooperative launch)");
  if (properties.multiProcessorCount <= 0 ||
      properties.maxThreadsPerBlock <= 0 || properties.warpSize <= 0) {
    throw std::runtime_error{"HIP returned invalid device scheduling limits"};
  }
  return DeviceProperties{
      static_cast<std::uint32_t>(properties.multiProcessorCount),
      static_cast<std::uint32_t>(properties.maxThreadsPerBlock),
      static_cast<std::uint32_t>(properties.warpSize),
      cooperative != 0,
  };
}

using detail::DeviceBatchJacobiStatistics;
using detail::DeviceBatchJacobiView;

__device__ void atomic_add_u64(
    std::uint64_t* const destination,
    const unsigned long long value) {
  atomicAdd(reinterpret_cast<unsigned long long*>(destination), value);
}

[[nodiscard]] __device__ float* batch_distance_slot(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchJacobiView& batch,
    const std::uint32_t slot) {
  auto* const base = reinterpret_cast<float*>(workspace.engine_scratch);
  const std::uint64_t slot_elements =
      static_cast<std::uint64_t>(graph.vertex_count) * batch.lane_width;
  return base + static_cast<std::uint64_t>(slot) * slot_elements;
}

__device__ void initialize_batched_jacobi_state(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence) {
  const std::uint64_t global_thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  float* const distance_zero = batch_distance_slot(graph, workspace, batch, 0U);
  float* const distance_one = batch_distance_slot(graph, workspace, batch, 1U);

  const std::uint32_t wave_lane = threadIdx.x & (phase14_wave_width - 1U);
  const std::uint32_t wave_in_block = threadIdx.x / phase14_wave_width;
  const std::uint32_t waves_per_block = blockDim.x / phase14_wave_width;
  const std::uint64_t global_wave =
      static_cast<std::uint64_t>(blockIdx.x) * waves_per_block + wave_in_block;
  const std::uint64_t grid_waves =
      static_cast<std::uint64_t>(gridDim.x) * waves_per_block;
  for (std::uint64_t packed_vertex = global_wave;
       packed_vertex < batch.union_vertex_count;
       packed_vertex += grid_waves) {
    std::uint32_t low = 0U;
    std::uint32_t high = batch.selected_range_count;
    while (low + 1U < high) {
      const std::uint32_t middle = low + (high - low) / 2U;
      if (batch.selected_range_vertex_offsets[middle] <= packed_vertex) {
        low = middle;
      } else {
        high = middle;
      }
    }
    const BatchVertexRange range = batch.selected_ranges[low];
    const std::uint32_t vertex =
        range.begin + static_cast<std::uint32_t>(
                          packed_vertex -
                          batch.selected_range_vertex_offsets[low]);
    // Every selected CSC run is overwritten before the batch can read it.
    // Runs belong to exactly one destination column, so waves never race.
    // Physical wave lanes cooperate even when they do not map to a query.
    const std::uint32_t run_begin =
        graph.csc.column_run_offsets[vertex];
    const std::uint32_t run_end =
        graph.csc.column_run_offsets[vertex + 1U];
    for (std::uint64_t run =
             static_cast<std::uint64_t>(run_begin) + wave_lane;
         run < run_end;
         run += phase14_wave_width) {
      const std::uint32_t source_tile =
          graph.csc.run_source_tiles[run];
      workspace.run_lane_masks[run] =
          range.lane_mask & workspace.tile_lane_masks[source_tile];
    }
    const bool mapped_lane = wave_lane < batch.lane_width;
    const LaneMask lane_bit =
        mapped_lane ? LaneMask{1U} << wave_lane : LaneMask{0U};
    if (!mapped_lane || (range.lane_mask & lane_bit) == 0U) {
      continue;
    }
    bool source = false;
    const std::uint32_t begin = batch.source_offsets[wave_lane];
    const std::uint32_t end = batch.source_offsets[wave_lane + 1U];
    for (std::uint32_t index = begin; index < end; ++index) {
      source = source || workspace.sources[index] == vertex;
    }
    const float initial =
        source ? 0.0F : std::numeric_limits<float>::infinity();
    const std::uint64_t element =
        static_cast<std::uint64_t>(vertex) * batch.lane_width + wave_lane;
    distance_zero[element] = initial;
    distance_one[element] = initial;
  }
  if (global_thread < batch.lane_width) {
    batch.lane_convergence_rounds[global_thread] = 0U;
  }

  if (global_thread == 0U) {
    DeviceController* const controller = workspace.controller;
    controller->valid_lane_mask = batch.valid_lane_mask;
    controller->active_lane_mask = batch.valid_lane_mask;
    controller->changed_lane_mask = 0U;
    controller->converged_lane_mask = 0U;
    controller->execute_lane_mask = batch.valid_lane_mask;
    controller->engine_kind =
        static_cast<std::uint32_t>(EngineKind::jacobi_pull);
    controller->enable_per_lane_convergence = enable_per_lane_convergence;
    controller->rounds_completed = 0U;
    controller->maximum_rounds = maximum_rounds;
    controller->distance_read_slot = 0U;
    controller->distance_write_slot = 1U;
    controller->frontier_read_slot = 0U;
    controller->frontier_write_slot = 1U;
    controller->frontier_size[0] = 0U;
    controller->frontier_size[1] = 0U;
    controller->next_frontier_lane_mask = 0U;
    controller->done = 0U;
    controller->stop_reason =
        static_cast<std::uint32_t>(DeviceStopReason::none);
    controller->error_bits = 0U;
  }
}

template <bool explicit_broadcast, bool collect_statistics>
__device__ void perform_batched_jacobi_round(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    DeviceBatchJacobiStatistics* const batch_statistics,
    const std::uint32_t instrumentation_level) {
  if constexpr (!collect_statistics) {
    static_cast<void>(batch_statistics);
    static_cast<void>(instrumentation_level);
  }
  const std::uint32_t done = workspace.controller->done;
  const LaneMask execute_lane_mask = workspace.controller->execute_lane_mask;
  const std::uint32_t read_slot = workspace.controller->distance_read_slot;
  const std::uint32_t write_slot = workspace.controller->distance_write_slot;
  if (done != 0U || execute_lane_mask == 0U) {
    return;
  }

  const std::uint64_t required_bytes =
      static_cast<std::uint64_t>(graph.vertex_count) * batch.lane_width *
      sizeof(float) * 2U;
  if (workspace.engine_scratch == nullptr ||
      workspace.engine_scratch_bytes < required_bytes || read_slot > 1U ||
      write_slot > 1U || read_slot == write_slot) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      atomicOr(
          &workspace.controller->error_bits,
          device_error::invalid_controller_state);
    }
    return;
  }

  const std::uint32_t wave_lane = threadIdx.x & (phase14_wave_width - 1U);
  const std::uint32_t wave_in_block = threadIdx.x / phase14_wave_width;
  const std::uint32_t waves_per_block = blockDim.x / phase14_wave_width;
  const std::uint64_t global_wave =
      static_cast<std::uint64_t>(blockIdx.x) * waves_per_block + wave_in_block;
  const std::uint64_t grid_waves =
      static_cast<std::uint64_t>(gridDim.x) * waves_per_block;
  const bool mapped_lane = wave_lane < batch.lane_width;
  const LaneMask lane_bit =
      mapped_lane ? LaneMask{1U} << wave_lane : LaneMask{0U};
  const std::uint64_t slot_elements =
      static_cast<std::uint64_t>(graph.vertex_count) * batch.lane_width;
  const float* const read =
      reinterpret_cast<const float*>(workspace.engine_scratch) +
      static_cast<std::uint64_t>(read_slot) * slot_elements;
  float* const write = reinterpret_cast<float*>(workspace.engine_scratch) +
                       static_cast<std::uint64_t>(write_slot) * slot_elements;

  bool lane_changed = false;
  unsigned long long local_runs_visited = 0U;
  unsigned long long local_runs_skipped = 0U;
  unsigned long long local_nonzero_runs = 0U;
  unsigned long long local_edge_records = 0U;
  unsigned long long local_lane_edge_pairs = 0U;
  unsigned long long local_active_lanes = 0U;
  unsigned long long local_active_vertex_lanes = 0U;
  unsigned long long local_successful = 0U;

  for (std::uint64_t packed_vertex = global_wave;
       packed_vertex < batch.union_vertex_count;
       packed_vertex += grid_waves) {
    std::uint32_t low = 0U;
    std::uint32_t high = batch.selected_range_count;
    while (low + 1U < high) {
      const std::uint32_t middle = low + (high - low) / 2U;
      if (batch.selected_range_vertex_offsets[middle] <= packed_vertex) {
        low = middle;
      } else {
        high = middle;
      }
    }
    const BatchVertexRange range = batch.selected_ranges[low];
    const std::uint32_t destination =
        range.begin + static_cast<std::uint32_t>(
                          packed_vertex -
                          batch.selected_range_vertex_offsets[low]);
    const std::uint32_t destination_tile = graph.owner_tiles[destination];
    const LaneMask destination_execute_mask =
        workspace.tile_lane_masks[destination_tile] & execute_lane_mask;
    if (destination_execute_mask == 0U) {
      continue;
    }
    if constexpr (collect_statistics) {
      if (wave_lane == 0U) {
        local_active_vertex_lanes += __popc(destination_execute_mask);
      }
    }

    const bool execute_lane =
        mapped_lane && (destination_execute_mask & lane_bit) != 0U;
    const std::uint64_t destination_element =
        static_cast<std::uint64_t>(destination) * batch.lane_width + wave_lane;
    const float preceding = execute_lane ? read[destination_element] : 0.0F;
    float best = preceding;
    const std::uint32_t run_begin =
        graph.csc.column_run_offsets[destination];
    const std::uint32_t run_end =
        graph.csc.column_run_offsets[destination + 1U];
    for (std::uint32_t run = run_begin; run < run_end; ++run) {
      LaneMask prepared_mask = 0U;
      if constexpr (explicit_broadcast) {
        if (wave_lane == 0U) {
          prepared_mask = workspace.run_lane_masks[run];
        }
        prepared_mask = __shfl(prepared_mask, 0, phase14_wave_width);
      } else {
        prepared_mask = workspace.run_lane_masks[run];
      }
      const LaneMask active_run_mask =
          prepared_mask & destination_execute_mask;
      const std::uint32_t edge_begin = graph.csc.run_edge_offsets[run];
      const std::uint32_t edge_end = graph.csc.run_edge_offsets[run + 1U];
      if constexpr (collect_statistics) {
        if (wave_lane == 0U) {
          ++local_runs_visited;
          if (active_run_mask == 0U) {
            ++local_runs_skipped;
          } else {
            ++local_nonzero_runs;
            const unsigned int active_lanes = __popc(active_run_mask);
            local_active_lanes += active_lanes;
            local_edge_records += edge_end - edge_begin;
            local_lane_edge_pairs +=
                static_cast<unsigned long long>(edge_end - edge_begin) *
                active_lanes;
          }
        }
      }
      if (active_run_mask == 0U) {
        continue;
      }
      const bool run_executes_lane =
          execute_lane && (active_run_mask & lane_bit) != 0U;
      for (std::uint32_t edge = edge_begin; edge < edge_end; ++edge) {
        std::uint32_t source = 0U;
        float weight = 0.0F;
        if constexpr (explicit_broadcast) {
          if (wave_lane == 0U) {
            source = graph.csc.sources[edge];
            weight = graph.csc.weights[edge];
          }
          source = __shfl(source, 0, phase14_wave_width);
          weight = __shfl(weight, 0, phase14_wave_width);
        } else {
          // Every mapped lane addresses the same immutable edge record. This
          // is the compiler-uniform baseline retained as the default until an
          // explicit-broadcast comparison is measured on the target GPU.
          source = graph.csc.sources[edge];
          weight = graph.csc.weights[edge];
        }
        if (run_executes_lane) {
          const float candidate =
              read[static_cast<std::uint64_t>(source) * batch.lane_width +
                   wave_lane] +
              weight;
          if (candidate < best) {
            best = candidate;
          }
        }
      }
    }
    const bool decreased = execute_lane && best < preceding;
    if (execute_lane) {
      // This writes the old value as well as improvements. Thus a lane's two
      // selected columns are equal before a no-change lane can be frozen.
      write[destination_element] = best;
      lane_changed = lane_changed || decreased;
    }
    if constexpr (collect_statistics) {
      const LaneMask decreases =
          static_cast<LaneMask>(__ballot(decreased));
      if (wave_lane == 0U) {
        local_successful += __popc(decreases);
      }
    }
  }

  const LaneMask changed = static_cast<LaneMask>(__ballot(lane_changed));
  if (wave_lane == 0U && changed != 0U) {
    atomicOr(&workspace.controller->changed_lane_mask, changed);
  }
  if constexpr (collect_statistics) {
    if (wave_lane == 0U) {
      if (batch_statistics != nullptr) {
        atomic_add_u64(
            &batch_statistics->csc_runs_considered, local_runs_visited);
        atomic_add_u64(
            &batch_statistics->csc_runs_skipped, local_runs_skipped);
        atomic_add_u64(
            &batch_statistics->csc_nonzero_runs_visited,
            local_nonzero_runs);
        atomic_add_u64(
            &batch_statistics->csc_edge_records_loaded,
            local_edge_records);
        atomic_add_u64(
            &batch_statistics->admitted_lane_edge_pairs,
            local_lane_edge_pairs);
        atomic_add_u64(
            &batch_statistics->active_lanes_across_nonzero_runs,
            local_active_lanes);
        atomic_add_u64(
            &batch_statistics->active_vertex_lane_evaluations,
            local_active_vertex_lanes);
        atomic_add_u64(
            &batch_statistics->successful_decreases, local_successful);
      }
      if (workspace.instrumentation != nullptr) {
        atomic_add_u64(
            &workspace.instrumentation->edges_examined,
            local_lane_edge_pairs);
        atomic_add_u64(
            &workspace.instrumentation->successful_decreases,
            local_successful);
        atomic_add_u64(
            &workspace.instrumentation->active_vertices,
            local_active_vertex_lanes);
        if (instrumentation_level ==
            static_cast<std::uint32_t>(InstrumentationLevel::debug)) {
          atomic_add_u64(
              &workspace.instrumentation->mask_operations,
              local_runs_visited);
        }
      }
    }
  }
}

__device__ void advance_batched_jacobi_controller(
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch) {
  DeviceController* const controller = workspace.controller;
  const LaneMask execute = controller->execute_lane_mask;
  const LaneMask changed = controller->changed_lane_mask & execute;
  const bool records_a_round = controller->done == 0U && execute != 0U &&
                               controller->error_bits == device_error::none;
  const std::uint64_t completed_round = controller->rounds_completed + 1U;
  if (records_a_round) {
    const LaneMask no_change = execute & ~changed;
    for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
      const LaneMask bit = LaneMask{1U} << lane;
      if ((no_change & bit) != 0U &&
          batch.lane_convergence_rounds[lane] == 0U) {
        batch.lane_convergence_rounds[lane] = completed_round;
      }
    }
  }
  const JacobiAdvanceResult transition = advance_jacobi_controller(*controller);
  if (workspace.instrumentation != nullptr && records_a_round &&
      transition != JacobiAdvanceResult::invalid_controller_state &&
      transition != JacobiAdvanceResult::no_op) {
    workspace.instrumentation->active_lane_rounds += __popc(execute);
  }
}

[[nodiscard]] __device__ DeviceRunStatus completed_batched_jacobi_status(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchJacobiView& batch) {
  const DeviceController controller = *workspace.controller;
  const bool converged =
      controller.stop_reason ==
          static_cast<std::uint32_t>(DeviceStopReason::converged) &&
      controller.error_bits == device_error::none &&
      controller.distance_read_slot <= 1U;
  LaneMask reached = 0U;
  LaneMask missed = 0U;
  if (converged) {
    const float* const distances =
        batch_distance_slot(
            graph, workspace, batch, controller.distance_read_slot);
    for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
      const LaneMask bit = LaneMask{1U} << lane;
      if ((batch.valid_lane_mask & bit) == 0U) {
        continue;
      }
      bool lane_reached = true;
      const std::uint32_t target_begin = batch.target_offsets[lane];
      const std::uint32_t target_end = batch.target_offsets[lane + 1U];
      for (std::uint32_t index = target_begin; index < target_end; ++index) {
        const std::uint32_t target = workspace.targets[index];
        lane_reached =
            lane_reached &&
            isfinite(
                distances[static_cast<std::uint64_t>(target) *
                              batch.lane_width +
                          lane]);
      }
      if (lane_reached) {
        reached |= bit;
      } else {
        missed |= bit;
      }
    }
  }
  return make_jacobi_run_status(controller, reached, missed);
}

__global__ void initialize_batched_jacobi_query(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence) {
  initialize_batched_jacobi_state(
      graph,
      workspace,
      batch,
      maximum_rounds,
      enable_per_lane_convergence);
}

__global__ void batched_jacobi_round_none_uniform_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    DeviceBatchJacobiStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  perform_batched_jacobi_round<false, false>(
      graph, workspace, batch, statistics, instrumentation_level);
}

__global__ void batched_jacobi_round_none_broadcast_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    DeviceBatchJacobiStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  perform_batched_jacobi_round<true, false>(
      graph, workspace, batch, statistics, instrumentation_level);
}

__global__ void batched_jacobi_round_instrumented_uniform_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    DeviceBatchJacobiStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  perform_batched_jacobi_round<false, true>(
      graph, workspace, batch, statistics, instrumentation_level);
}

__global__ void batched_jacobi_round_instrumented_broadcast_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    DeviceBatchJacobiStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  perform_batched_jacobi_round<true, true>(
      graph, workspace, batch, statistics, instrumentation_level);
}

__global__ void batched_jacobi_advance_kernel(
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    advance_batched_jacobi_controller(workspace, batch);
  }
}

__global__ void finalize_batched_jacobi_status(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *workspace.status =
        completed_batched_jacobi_status(graph, workspace, batch);
  }
}

template <bool explicit_broadcast, bool collect_statistics>
__device__ void execute_batched_jacobi_persistent(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    DeviceBatchJacobiStatistics* const statistics,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
  initialize_batched_jacobi_state(
      graph,
      workspace,
      batch,
      maximum_rounds,
      enable_per_lane_convergence);
  grid.sync();

  for (;;) {
    perform_batched_jacobi_round<explicit_broadcast, collect_statistics>(
        graph, workspace, batch, statistics, instrumentation_level);
    grid.sync();
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      advance_batched_jacobi_controller(workspace, batch);
    }
    // The controller owner cannot publish slots/masks while another block is
    // still using the preceding round. Every block also reaches this barrier
    // on the terminal iteration before observing done.
    grid.sync();
    if (workspace.controller->done != 0U) {
      break;
    }
  }

  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *workspace.status =
        completed_batched_jacobi_status(graph, workspace, batch);
  }
  grid.sync();
}

__global__ void batched_jacobi_persistent_none_uniform_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    DeviceBatchJacobiStatistics* const statistics,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  execute_batched_jacobi_persistent<false, false>(
      graph,
      workspace,
      batch,
      statistics,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level);
}

__global__ void batched_jacobi_persistent_none_broadcast_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    DeviceBatchJacobiStatistics* const statistics,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  execute_batched_jacobi_persistent<true, false>(
      graph,
      workspace,
      batch,
      statistics,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level);
}

__global__ void batched_jacobi_persistent_instrumented_uniform_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    DeviceBatchJacobiStatistics* const statistics,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  execute_batched_jacobi_persistent<false, true>(
      graph,
      workspace,
      batch,
      statistics,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level);
}

__global__ void batched_jacobi_persistent_instrumented_broadcast_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchJacobiView batch,
    DeviceBatchJacobiStatistics* const statistics,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  execute_batched_jacobi_persistent<true, true>(
      graph,
      workspace,
      batch,
      statistics,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level);
}

[[nodiscard]] const void* round_kernel_pointer(
    const InstrumentationLevel instrumentation,
    const BatchedJacobiLoadStrategy strategy) noexcept {
  if (instrumentation == InstrumentationLevel::none) {
    return strategy == BatchedJacobiLoadStrategy::compiler_uniform
               ? reinterpret_cast<const void*>(
                     batched_jacobi_round_none_uniform_kernel)
               : reinterpret_cast<const void*>(
                     batched_jacobi_round_none_broadcast_kernel);
  }
  return strategy == BatchedJacobiLoadStrategy::compiler_uniform
             ? reinterpret_cast<const void*>(
                   batched_jacobi_round_instrumented_uniform_kernel)
             : reinterpret_cast<const void*>(
                   batched_jacobi_round_instrumented_broadcast_kernel);
}

[[nodiscard]] const void* persistent_kernel_pointer(
    const InstrumentationLevel instrumentation,
    const BatchedJacobiLoadStrategy strategy) noexcept {
  if (instrumentation == InstrumentationLevel::none) {
    return strategy == BatchedJacobiLoadStrategy::compiler_uniform
               ? reinterpret_cast<const void*>(
                     batched_jacobi_persistent_none_uniform_kernel)
               : reinterpret_cast<const void*>(
                     batched_jacobi_persistent_none_broadcast_kernel);
  }
  return strategy == BatchedJacobiLoadStrategy::compiler_uniform
             ? reinterpret_cast<const void*>(
                   batched_jacobi_persistent_instrumented_uniform_kernel)
             : reinterpret_cast<const void*>(
                   batched_jacobi_persistent_instrumented_broadcast_kernel);
}

[[nodiscard]] std::uint32_t kernel_occupancy(
    const void* const kernel,
    const std::uint32_t block_size) {
  int active = 0;
  check(
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active, kernel, static_cast<int>(block_size), 0U),
      "hipOccupancyMaxActiveBlocksPerMultiprocessor(batched Jacobi kernel)");
  if (active <= 0) {
    throw std::runtime_error{
        "batched Jacobi kernel has zero reported occupancy"};
  }
  return static_cast<std::uint32_t>(active);
}

[[nodiscard]] std::uint32_t kernel_register_count(const void* const kernel) {
  hipFuncAttributes attributes{};
  check(
      hipFuncGetAttributes(&attributes, kernel),
      "hipFuncGetAttributes(batched Jacobi kernel)");
  if (attributes.numRegs < 0) {
    throw std::runtime_error{"HIP returned a negative kernel register count"};
  }
  return static_cast<std::uint32_t>(attributes.numRegs);
}

struct CooperativeGrid {
  std::uint32_t blocks{};
  std::uint32_t active_blocks_per_wgp{};
};

[[nodiscard]] CooperativeGrid cooperative_grid(
    const DeviceProperties& properties,
    const GpuRunOptions& options,
    const BatchedJacobiLoadStrategy strategy) {
  if (!properties.cooperative_launch) {
    throw std::runtime_error{
        "selected HIP device does not support cooperative launch"};
  }
  const std::uint32_t active = kernel_occupancy(
      persistent_kernel_pointer(options.instrumentation, strategy),
      options.block_size);
  std::uint32_t blocks_per_wgp = active;
  if (options.grid_policy == GridPolicy::fixed_blocks_per_wgp) {
    if (options.blocks_per_wgp > active) {
      throw std::invalid_argument{
          "requested cooperative grid exceeds batched Jacobi occupancy"};
    }
    blocks_per_wgp = options.blocks_per_wgp;
  }
  return CooperativeGrid{
      checked_grid_blocks(blocks_per_wgp, properties.wgp_count), active};
}

[[nodiscard]] std::uint32_t ordinary_grid_blocks(
    const DeviceProperties& properties,
    const GpuRunOptions& options,
    const BatchedJacobiLoadStrategy strategy,
    const std::uint32_t union_vertex_count,
    std::uint32_t& active_blocks_per_wgp) {
  active_blocks_per_wgp = kernel_occupancy(
      round_kernel_pointer(options.instrumentation, strategy),
      options.block_size);
  std::uint32_t blocks_per_wgp = active_blocks_per_wgp;
  if (options.grid_policy == GridPolicy::fixed_blocks_per_wgp) {
    if (options.blocks_per_wgp > active_blocks_per_wgp) {
      throw std::invalid_argument{
          "requested ordinary grid exceeds batched Jacobi occupancy"};
    }
    blocks_per_wgp = options.blocks_per_wgp;
  }
  const std::uint32_t resident =
      checked_grid_blocks(blocks_per_wgp, properties.wgp_count);
  const std::uint32_t waves_per_block =
      options.block_size / phase14_wave_width;
  const std::uint64_t work_blocks = std::max<std::uint64_t>(
      1U,
      (static_cast<std::uint64_t>(union_vertex_count) + waves_per_block - 1U) /
          waves_per_block);
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(resident, work_blocks));
}

void check_launch(const std::string_view name) {
  check(hipGetLastError(), name);
}

void launch_initialize(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchJacobiView& batch,
    const GpuRunOptions& options,
    const hipStream_t stream) {
  const std::uint32_t waves_per_block =
      options.block_size / phase14_wave_width;
  const std::uint32_t blocks = static_cast<std::uint32_t>(
      std::max<std::uint64_t>(
          1U,
          (static_cast<std::uint64_t>(batch.union_vertex_count) +
           waves_per_block - 1U) /
              waves_per_block));
  hipLaunchKernelGGL(
      initialize_batched_jacobi_query,
      dim3(blocks),
      dim3(options.block_size),
      0U,
      stream,
      graph,
      workspace,
      batch,
      options.maximum_rounds,
      options.enable_per_lane_convergence);
  check_launch("initialize_batched_jacobi_query launch");
}

void launch_round_pair(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchJacobiView& batch,
    DeviceBatchJacobiStatistics* const statistics,
    const GpuRunOptions& options,
    const BatchedJacobiLoadStrategy strategy,
    const std::uint32_t blocks,
    const hipStream_t stream) {
  const std::uint32_t instrumentation =
      static_cast<std::uint32_t>(options.instrumentation);
  if (options.instrumentation == InstrumentationLevel::none) {
    if (strategy == BatchedJacobiLoadStrategy::compiler_uniform) {
      hipLaunchKernelGGL(
          batched_jacobi_round_none_uniform_kernel,
          dim3(blocks),
          dim3(options.block_size),
          0U,
          stream,
          graph,
          workspace,
          batch,
          statistics,
          instrumentation);
    } else {
      hipLaunchKernelGGL(
          batched_jacobi_round_none_broadcast_kernel,
          dim3(blocks),
          dim3(options.block_size),
          0U,
          stream,
          graph,
          workspace,
          batch,
          statistics,
          instrumentation);
    }
  } else if (strategy == BatchedJacobiLoadStrategy::compiler_uniform) {
    hipLaunchKernelGGL(
        batched_jacobi_round_instrumented_uniform_kernel,
        dim3(blocks),
        dim3(options.block_size),
        0U,
        stream,
        graph,
        workspace,
        batch,
        statistics,
        instrumentation);
  } else {
    hipLaunchKernelGGL(
        batched_jacobi_round_instrumented_broadcast_kernel,
        dim3(blocks),
        dim3(options.block_size),
        0U,
        stream,
        graph,
        workspace,
        batch,
        statistics,
        instrumentation);
  }
  check_launch("batched Jacobi round launch");
  hipLaunchKernelGGL(
      batched_jacobi_advance_kernel,
      dim3(1U),
      dim3(1U),
      0U,
      stream,
      workspace,
      batch);
  check_launch("batched_jacobi_advance_kernel launch");
}

void launch_finalize(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchJacobiView& batch,
    const hipStream_t stream) {
  hipLaunchKernelGGL(
      finalize_batched_jacobi_status,
      dim3(1U),
      dim3(1U),
      0U,
      stream,
      graph,
      workspace,
      batch);
  check_launch("finalize_batched_jacobi_status launch");
}

void launch_persistent(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchJacobiView& batch,
    DeviceBatchJacobiStatistics* const statistics,
    const CooperativeGrid grid,
    const GpuRunOptions& options,
    const BatchedJacobiLoadStrategy strategy,
    const hipStream_t stream) {
  DeviceGraphView32 graph_argument = graph;
  DeviceWorkspaceView workspace_argument = workspace;
  DeviceBatchJacobiView batch_argument = batch;
  DeviceBatchJacobiStatistics* statistics_argument = statistics;
  std::uint64_t maximum_rounds_argument = options.maximum_rounds;
  std::uint32_t convergence_argument = options.enable_per_lane_convergence;
  std::uint32_t instrumentation_argument =
      static_cast<std::uint32_t>(options.instrumentation);
  void* arguments[] = {
      &graph_argument,
      &workspace_argument,
      &batch_argument,
      &statistics_argument,
      &maximum_rounds_argument,
      &convergence_argument,
      &instrumentation_argument,
  };
  check(
      hipLaunchCooperativeKernel(
          persistent_kernel_pointer(options.instrumentation, strategy),
          dim3(grid.blocks),
          dim3(options.block_size),
          arguments,
          0U,
          stream),
      "hipLaunchCooperativeKernel(batched Jacobi persistent)");
}

[[nodiscard]] std::uint64_t selected_lane_vertex_count(
    const BatchDeviceDescription& batch) {
  std::uint64_t count = 0U;
  for (const BatchVertexRange range : batch.selected_vertex_ranges) {
    const std::uint64_t vertices = range.end - range.begin;
    const std::uint64_t lanes =
        static_cast<std::uint64_t>(std::popcount(range.lane_mask));
    const std::uint64_t increment =
        checked_multiply_u64(vertices, lanes, "selected lane vertices");
    if (count > std::numeric_limits<std::uint64_t>::max() - increment) {
      throw std::overflow_error{"selected lane vertex count overflow"};
    }
    count += increment;
  }
  return count;
}

template <typename Visitor>
void for_each_nonzero_csc_run(
    const TileRunLayout64& runs,
    const BatchDeviceDescription& batch,
    Visitor&& visitor) {
  if (batch.run_representation ==
      BatchRunRepresentation::retained_per_run_masks) {
    for (const std::uint32_t run : batch.touched_csc_runs) {
      visitor(run, batch.csc_run_lane_masks[run]);
    }
    return;
  }
  if (batch.run_representation ==
      BatchRunRepresentation::compact_nonzero_descriptors) {
    for (const RunLaneMaskDescriptor descriptor :
         batch.csc_run_descriptors) {
      visitor(descriptor.run_id, descriptor.lane_mask);
    }
    return;
  }
  for (const BatchVertexRange range : batch.selected_vertex_ranges) {
    for (std::uint32_t vertex = range.begin; vertex < range.end; ++vertex) {
      const std::size_t run_begin = static_cast<std::size_t>(
          runs.csc_column_run_offsets[vertex]);
      const std::size_t run_end = static_cast<std::size_t>(
          runs.csc_column_run_offsets[vertex + 1U]);
      for (std::size_t run = run_begin; run < run_end; ++run) {
        const LaneMask mask =
            range.lane_mask &
            batch.tile_lane_masks[
                runs.csc_run_source_tiles[run].value()];
        if (mask != 0U) {
          visitor(run, mask);
        }
      }
    }
  }
}

[[nodiscard]] std::uint64_t lane_edge_pairs_per_complete_batch_round(
    const TileRunLayout64& runs,
    const BatchDeviceDescription& batch) {
  std::uint64_t pairs = 0U;
  for_each_nonzero_csc_run(runs, batch, [&](const std::size_t run,
                                            const LaneMask mask) {
    const std::uint64_t edges =
        runs.csc_run_edge_offsets[run + 1U] -
        runs.csc_run_edge_offsets[run];
    const std::uint64_t lanes = static_cast<std::uint64_t>(
        std::popcount(mask));
    const std::uint64_t increment =
        checked_multiply_u64(edges, lanes, "batch-round lane edges");
    if (pairs > std::numeric_limits<std::uint64_t>::max() - increment) {
      throw std::overflow_error{"batch-round lane edge count overflow"};
    }
    pairs += increment;
  });
  return pairs;
}

}  // namespace

std::size_t batched_jacobi_distance_scratch_bytes(
    const std::uint32_t vertex_count,
    const std::uint32_t lane_width) {
  if (vertex_count == 0U || !supported_lane_width(lane_width)) {
    throw std::invalid_argument{
        "batched Jacobi scratch requires vertices and width 1/8/16/32"};
  }
  const std::size_t lane_vertices = checked_multiply(
      static_cast<std::size_t>(vertex_count),
      lane_width,
      "batched Jacobi lane vertices");
  return checked_multiply(
      lane_vertices,
      2U * sizeof(float),
      "batched Jacobi two distance buffers");
}

class BatchedJacobiPullEngine::Impl final {
 public:
  Impl(
      const WeightedGraph& host_graph_value,
      const TileRunLayout64& tile_runs_value,
      const ResidentDeviceGraph& resident_graph_value,
      ReusableBatchedJacobiWorkspace& workspace_value,
      const HipStream& stream_value)
      : host_graph{host_graph_value},
        tile_runs{tile_runs_value},
        resident_graph{resident_graph_value},
        workspace{workspace_value},
        stream{stream_value} {
    if (!validate_weighted_graph(host_graph).ok() ||
        !host_graph.has_spatial_ordering()) {
      throw std::invalid_argument{
          "batched Jacobi requires a valid spatially ordered host graph"};
    }
    if (!validate_tile_run_layout(host_graph, tile_runs).ok()) {
      throw std::invalid_argument{
          "batched Jacobi requires deeply valid tile-run metadata"};
    }
    if (!resident_graph.has_upload()) {
      throw std::invalid_argument{
          "batched Jacobi requires an uploaded resident graph"};
    }
    const DeviceGraphView32& device = resident_graph.view();
    if (device.vertex_count != host_graph.vertex_count() ||
        device.edge_count != host_graph.edge_count() ||
        device.tile_count != host_graph.tile_coordinates().size() ||
        device.csc.run_count != tile_runs.csc_run_source_tiles.size()) {
      throw std::invalid_argument{
          "batched Jacobi host graph/run shapes disagree with resident graph"};
    }
    if (resident_graph.fingerprint() !=
        fingerprint_device_graph_source32(host_graph, tile_runs)) {
      throw std::invalid_argument{
          "batched Jacobi host graph/run content disagrees with resident graph"};
    }
  }

  const WeightedGraph& host_graph;
  const TileRunLayout64& tile_runs;
  const ResidentDeviceGraph& resident_graph;
  ReusableBatchedJacobiWorkspace& workspace;
  const HipStream& stream;

  std::vector<float> downloaded_distance_slots;
  HipEventTimer preparation_gpu_timer;
  HipEventTimer sssp_gpu_timer;
  HipEventTimer result_transfer_gpu_timer;
};

BatchedJacobiPullEngine::BatchedJacobiPullEngine(
    const WeightedGraph& host_graph,
    const TileRunLayout64& tile_runs,
    const ResidentDeviceGraph& resident_graph,
    ReusableBatchedJacobiWorkspace& workspace,
    const HipStream& stream)
    : impl_{new Impl{
          host_graph, tile_runs, resident_graph, workspace, stream}} {}

BatchedJacobiPullEngine::~BatchedJacobiPullEngine() { delete impl_; }

BatchedJacobiRunOutput BatchedJacobiPullEngine::run_status_only(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    const BatchedJacobiLoadStrategy load_strategy) {
  return run_impl(
      batch, options, load_strategy, false, false, nullptr, nullptr);
}

DeviceRunStatus BatchedJacobiPullEngine::run_compact_status(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    const BatchedJacobiLoadStrategy load_strategy) {
  return run_impl(
             batch, options, load_strategy, false, true, nullptr, nullptr)
      .result.status;
}

CompactPathBatchOutput BatchedJacobiPullEngine::run_compact_paths(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    ReusableCompactPathWorkspace& compact_workspace,
    const BatchedJacobiLoadStrategy load_strategy) {
  CompactPathBatchOutput compact_output;
  static_cast<void>(run_impl(
      batch,
      options,
      load_strategy,
      false,
      false,
      &compact_workspace,
      &compact_output));
  return compact_output;
}

BatchedJacobiRunOutput BatchedJacobiPullEngine::run_with_distances(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    const BatchedJacobiLoadStrategy load_strategy) {
  return run_impl(
      batch, options, load_strategy, true, false, nullptr, nullptr);
}

BatchedJacobiRunOutput BatchedJacobiPullEngine::run_impl(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    const BatchedJacobiLoadStrategy load_strategy,
    const bool download_distances,
    const bool compact_status_only,
    ReusableCompactPathWorkspace* const compact_workspace,
    CompactPathBatchOutput* const compact_output) {
  SteadyClockTimer end_to_end_timer;
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot run an invalid batched Jacobi engine"};
  }
  if (validate_gpu_run_options(options) != GpuRunOptionsError::none ||
      options.engine != EngineKind::jacobi_pull ||
      !valid_batched_jacobi_load_strategy(load_strategy)) {
    throw std::invalid_argument{
        "batched Jacobi options are invalid or select another engine"};
  }
  if ((compact_workspace == nullptr) != (compact_output == nullptr) ||
      (compact_output != nullptr &&
       (download_distances || compact_status_only ||
        options.instrumentation != InstrumentationLevel::none))) {
    throw std::invalid_argument{
        "batched Jacobi compact-path mode requires paired workspace/output, "
        "None instrumentation, and no other readback mode"};
  }
  validate_batch_description_shape(
      impl_->host_graph,
      impl_->tile_runs,
      batch,
      compact_workspace == nullptr && !compact_status_only);

  const DeviceProperties properties = current_device_properties();
  if (properties.wave_width != phase14_wave_width) {
    throw std::runtime_error{
        "Phase 14 wave-to-query mapping requires runtime wave32"};
  }
  if (options.block_size > properties.maximum_threads_per_block ||
      options.block_size % phase14_wave_width != 0U) {
    throw std::invalid_argument{
        "batched Jacobi block size must be a legal multiple of wave32"};
  }

  const DeviceGraphView32 graph = impl_->resident_graph.view();
  const std::size_t scratch_bytes = batched_jacobi_distance_scratch_bytes(
      graph.vertex_count, batch.lane_width);
  bool lease_active = false;

  BatchedJacobiRunOutput output;
  output.distances_downloaded = download_distances;
  output.result.engine_kind =
      static_cast<std::uint32_t>(EngineKind::jacobi_pull);
  output.result.control_mode = static_cast<std::uint32_t>(options.control_mode);
  output.metrics.lane_width = batch.lane_width;
  output.metrics.valid_lane_count = static_cast<std::uint32_t>(
      std::popcount(batch.valid_lane_mask));
  output.metrics.load_strategy = load_strategy;

  try {
    impl_->resident_graph.wait_until_ready(impl_->stream);
    impl_->preparation_gpu_timer.start(impl_->stream);
    impl_->workspace.prepare_async(
        batch, options, scratch_bytes, graph.csc.run_count, impl_->stream);
    lease_active = true;
    impl_->preparation_gpu_timer.stop(impl_->stream);

    const DeviceWorkspaceView workspace_view =
        impl_->workspace.device_workspace_view();
    const DeviceBatchJacobiView batch_view =
        impl_->workspace.device_batch_view();

    const hipStream_t native_stream = as_stream(impl_->stream.native_handle());
    DeviceBatchJacobiStatistics* const device_batch_statistics =
        impl_->workspace.device_batch_statistics();
    const void* const selected_kernel =
        options.control_mode == ControlMode::persistent_cooperative
            ? persistent_kernel_pointer(options.instrumentation, load_strategy)
            : round_kernel_pointer(options.instrumentation, load_strategy);
    output.metrics.kernel_registers_per_thread =
        kernel_register_count(selected_kernel);
    impl_->sssp_gpu_timer.start(impl_->stream);
    if (options.control_mode != ControlMode::persistent_cooperative) {
      launch_initialize(
          graph,
          workspace_view,
          batch_view,
          options,
          native_stream);
    }

    std::uint64_t convergence_kernel_dispatches = 0U;
    DeviceController observed_controller{};
    if (options.control_mode == ControlMode::persistent_cooperative) {
      const CooperativeGrid grid =
          cooperative_grid(properties, options, load_strategy);
      output.metrics.cooperative_grid_blocks = grid.blocks;
      output.metrics.cooperative_active_blocks_per_wgp =
          grid.active_blocks_per_wgp;
      launch_persistent(
          graph,
          workspace_view,
          batch_view,
          device_batch_statistics,
          grid,
          options,
          load_strategy,
          native_stream);
      convergence_kernel_dispatches = 1U;
    } else {
      const std::uint32_t blocks = ordinary_grid_blocks(
          properties,
          options,
          load_strategy,
          batch_view.union_vertex_count,
          output.metrics.ordinary_active_blocks_per_wgp);
      const std::uint32_t chunk_rounds =
          options.control_mode == ControlMode::per_round_host_poll
              ? 1U
              : options.rounds_per_chunk;
      do {
        for (std::uint32_t round = 0U; round < chunk_rounds; ++round) {
          launch_round_pair(
              graph,
              workspace_view,
              batch_view,
              device_batch_statistics,
              options,
              load_strategy,
              blocks,
              native_stream);
          checked_add_u64(
              output.metrics.engine_round_dispatches,
              1U,
              "engine round dispatches");
          checked_add_u64(
              output.metrics.controller_advance_dispatches,
              1U,
              "controller advance dispatches");
          checked_add_u64(
              convergence_kernel_dispatches,
              2U,
              "convergence kernel dispatches");
        }
        impl_->workspace.download_controller_async(
            observed_controller, impl_->stream);
        impl_->stream.synchronize();
        checked_add_u64(
            output.metrics.convergence_host_checks,
            1U,
            "convergence host checks");
      } while (observed_controller.done == 0U);
      launch_finalize(
          graph,
          workspace_view,
          batch_view,
          native_stream);
    }
    impl_->sssp_gpu_timer.stop(impl_->stream);

    if (compact_output != nullptr) {
      const std::uint64_t slot_stride =
          static_cast<std::uint64_t>(graph.vertex_count) * batch.lane_width;
      const DeviceCompactBatchView compact_batch{
          batch_view.lane_width,
          batch_view.valid_lane_mask,
          batch_view.source_offsets,
          batch_view.target_offsets};
      const DeviceCompactDistanceMatrix compact_distances{
          graph.vertex_count,
          batch.lane_width,
          2U,
          slot_stride,
          CompactDistanceEncoding::floating_point,
          reinterpret_cast<const float*>(workspace_view.engine_scratch),
          nullptr};
      *compact_output = compact_workspace->reconstruct(
          graph,
          workspace_view,
          compact_batch,
          compact_distances,
          batch,
          impl_->stream);
      // Reconstruction's mandatory count/emit transfer boundary has already
      // completed this stream. Releasing the lease must not fence it again.
      impl_->workspace.retire_after_stream_completion(impl_->stream);
      lease_active = false;
      account_compact_path_controller_polls(
          *compact_output, output.metrics.convergence_host_checks);
      const DeviceRunStatus& compact_status = compact_output->status;
      const LaneMask classified = compact_status.reached_target_mask |
                                  compact_status.bounding_box_miss_mask;
      const bool normal_convergence =
          compact_status.stop_reason ==
              static_cast<std::uint32_t>(DeviceStopReason::converged) &&
          compact_status.error_bits == device_error::none;
      if (validate_device_run_status(compact_status) !=
              DeviceRunStatusError::none ||
          compact_status.valid_lane_mask != batch.valid_lane_mask ||
          compact_status.final_distance_slot > 1U ||
          (normal_convergence
               ? classified != batch.valid_lane_mask
               : classified != 0U)) {
        throw std::runtime_error{
            "batched Jacobi produced an invalid compact-path status"};
      }
      output.result.status = compact_status;
      output.metrics.sssp_device_timeline_milliseconds =
          impl_->sssp_gpu_timer
              .elapsed_milliseconds_after_stream_synchronization();
      compact_output->metrics.sssp_device_milliseconds =
          output.metrics.sssp_device_timeline_milliseconds;
      output.metrics.end_to_end_wall_milliseconds =
          end_to_end_timer.elapsed_milliseconds();
      compact_output->metrics.end_to_end_wall_milliseconds =
          output.metrics.end_to_end_wall_milliseconds;
      return output;
    }

    if (compact_status_only) {
      DeviceRunStatus compact_status{};
      impl_->workspace.download_status_async(compact_status, impl_->stream);
      // Retirement is the single terminal D2H completion boundary.
      impl_->workspace.retire(impl_->stream);
      lease_active = false;
      const LaneMask classified = compact_status.reached_target_mask |
                                  compact_status.bounding_box_miss_mask;
      const bool normal_convergence =
          compact_status.stop_reason ==
              static_cast<std::uint32_t>(DeviceStopReason::converged) &&
          compact_status.error_bits == device_error::none;
      if (validate_device_run_status(compact_status) !=
              DeviceRunStatusError::none ||
          compact_status.valid_lane_mask != batch.valid_lane_mask ||
          compact_status.final_distance_slot > 1U ||
          (normal_convergence
               ? classified != batch.valid_lane_mask
               : classified != 0U)) {
        throw std::runtime_error{
            "batched Jacobi produced an invalid compact terminal status"};
      }
      output.result.status = compact_status;
      output.metrics.end_to_end_wall_milliseconds =
          end_to_end_timer.elapsed_milliseconds();
      return output;
    }

    DeviceController final_controller{};
    DeviceRunStatus final_status{};
    DeviceWorkStatistics final_statistics{};
    DeviceBatchJacobiStatistics final_batch_statistics{};
    output.lane_convergence_rounds.resize(batch.lane_width);
    if (download_distances) {
      impl_->downloaded_distance_slots.resize(scratch_bytes / sizeof(float));
    } else {
      impl_->downloaded_distance_slots.clear();
    }
    impl_->result_transfer_gpu_timer.start(impl_->stream);
    impl_->workspace.download_async(
        final_controller,
        final_status,
        options.instrumentation == InstrumentationLevel::none
            ? nullptr
            : &final_statistics,
        options.instrumentation == InstrumentationLevel::none
            ? nullptr
            : &final_batch_statistics,
        output.lane_convergence_rounds,
        impl_->downloaded_distance_slots,
        impl_->stream);
    impl_->result_transfer_gpu_timer.stop(impl_->stream);
    // Retirement is the single terminal result-transfer completion boundary.
    impl_->workspace.retire(impl_->stream);
    lease_active = false;

    output.metrics.preparation_gpu_milliseconds =
        impl_->preparation_gpu_timer
            .elapsed_milliseconds_after_stream_synchronization();
    output.metrics.sssp_device_timeline_milliseconds =
        impl_->sssp_gpu_timer
            .elapsed_milliseconds_after_stream_synchronization();
    output.metrics.result_transfer_gpu_milliseconds =
        impl_->result_transfer_gpu_timer
            .elapsed_milliseconds_after_stream_synchronization();
    if (validate_device_controller(final_controller) !=
            DeviceControllerError::none ||
        validate_device_run_status(final_status) !=
            DeviceRunStatusError::none ||
        final_status.final_distance_slot !=
            final_controller.distance_read_slot ||
        final_status.rounds_completed != final_controller.rounds_completed ||
        final_status.valid_lane_mask != batch.valid_lane_mask) {
      throw std::runtime_error{
          "batched Jacobi produced inconsistent terminal controller/status"};
    }

    const std::size_t slot_elements = checked_multiply(
        graph.vertex_count,
        batch.lane_width,
        "batched Jacobi result elements");
    if (download_distances) {
      output.distances.assign(
          slot_elements, std::numeric_limits<float>::infinity());
      const std::size_t final_offset =
          static_cast<std::size_t>(final_status.final_distance_slot) *
          slot_elements;
      for (const BatchVertexRange range : batch.selected_vertex_ranges) {
        for (std::uint32_t vertex = range.begin; vertex < range.end; ++vertex) {
          for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
            const LaneMask bit = LaneMask{1U} << lane;
            if ((range.lane_mask & bit) == 0U) {
              continue;
            }
            const std::size_t element =
                static_cast<std::size_t>(vertex) * batch.lane_width + lane;
            output.distances[element] =
                impl_->downloaded_distance_slots[final_offset + element];
          }
        }
      }
      if (final_status.converged != 0U) {
        LaneMask identical = 0U;
        for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
          const LaneMask bit = LaneMask{1U} << lane;
          if ((batch.valid_lane_mask & bit) == 0U) {
            continue;
          }
          bool lane_identical = true;
          for (const BatchVertexRange range : batch.selected_vertex_ranges) {
            if ((range.lane_mask & bit) == 0U) {
              continue;
            }
            for (std::uint32_t vertex = range.begin; vertex < range.end;
                 ++vertex) {
              const std::size_t element =
                  static_cast<std::size_t>(vertex) * batch.lane_width + lane;
              lane_identical =
                  lane_identical &&
                  std::bit_cast<std::uint32_t>(
                      impl_->downloaded_distance_slots[element]) ==
                      std::bit_cast<std::uint32_t>(
                          impl_->downloaded_distance_slots[
                              slot_elements + element]);
            }
          }
          if (lane_identical) {
            identical |= bit;
          }
        }
        output.converged_slots_bitwise_identical_mask = identical;
      }
    }

    output.lane_executed_rounds.assign(batch.lane_width, 0U);
    output.lane_tail_rounds.assign(batch.lane_width, 0U);
    for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
      const LaneMask bit = LaneMask{1U} << lane;
      if ((batch.valid_lane_mask & bit) == 0U) {
        output.lane_convergence_rounds[lane] = 0U;
        continue;
      }
      const std::uint64_t convergence =
          output.lane_convergence_rounds[lane];
      const std::uint64_t executed =
          options.enable_per_lane_convergence != 0U && convergence != 0U
              ? convergence
              : final_status.rounds_completed;
      output.lane_executed_rounds[lane] = executed;
      if (convergence != 0U && convergence <= final_status.rounds_completed) {
        output.lane_tail_rounds[lane] =
            final_status.rounds_completed - convergence;
      }
      checked_add_u64(
          output.metrics.executed_lane_rounds,
          executed,
          "executed lane rounds");
      if (options.enable_per_lane_convergence != 0U) {
        checked_add_u64(
            output.metrics.lane_rounds_avoided_by_early_convergence,
            final_status.rounds_completed - executed,
            "avoided lane rounds");
      }
    }

    std::uint64_t exact_lane_edge_pairs = 0U;
    std::uint64_t exact_physical_edge_records = 0U;
    std::uint64_t exact_nonzero_run_visits = 0U;
    std::uint64_t exact_active_lanes_across_runs = 0U;
    for_each_nonzero_csc_run(
        impl_->tile_runs,
        batch,
        [&](const std::size_t run, const LaneMask mask) {
      const std::uint64_t edge_count =
          impl_->tile_runs.csc_run_edge_offsets[run + 1U] -
          impl_->tile_runs.csc_run_edge_offsets[run];
      std::uint64_t maximum_lane_rounds = 0U;
      std::uint64_t summed_lane_rounds = 0U;
      for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
        if ((mask & (LaneMask{1U} << lane)) != 0U) {
          maximum_lane_rounds = std::max(
              maximum_lane_rounds, output.lane_executed_rounds[lane]);
          checked_add_u64(
              summed_lane_rounds,
              output.lane_executed_rounds[lane],
              "run active lane rounds");
        }
      }
      checked_add_u64(
          exact_lane_edge_pairs,
          checked_multiply_u64(
              edge_count, summed_lane_rounds, "run logical lane edges"),
          "logical lane-edge total");
      checked_add_u64(
          exact_physical_edge_records,
          checked_multiply_u64(
              edge_count,
              maximum_lane_rounds,
              "run physical edge records"),
          "physical edge-record total");
      checked_add_u64(
          exact_nonzero_run_visits,
          maximum_lane_rounds,
          "visited run total");
      checked_add_u64(
          exact_active_lanes_across_runs,
          summed_lane_rounds,
          "active lanes across runs");
        });
    output.metrics.admitted_lane_edge_pairs = exact_lane_edge_pairs;
    output.metrics.csc_edge_records_loaded = exact_physical_edge_records;
    output.metrics.csc_runs_visited = exact_nonzero_run_visits;
    output.metrics.active_lanes_across_nonzero_runs =
        exact_active_lanes_across_runs;
    const std::uint64_t baseline_lane_edges = checked_multiply_u64(
        lane_edge_pairs_per_complete_batch_round(impl_->tile_runs, batch),
        final_status.rounds_completed,
        "full-duration batch lane edges");
    if (exact_lane_edge_pairs > baseline_lane_edges) {
      throw std::runtime_error{
          "batched Jacobi logical lane-edge accounting exceeded baseline"};
    }
    output.metrics.lane_edge_relaxations_avoided_by_early_convergence =
        baseline_lane_edges - exact_lane_edge_pairs;

    if (options.instrumentation != InstrumentationLevel::none) {
      if (final_batch_statistics.admitted_lane_edge_pairs !=
              exact_lane_edge_pairs ||
          final_batch_statistics.csc_edge_records_loaded !=
              exact_physical_edge_records ||
          final_batch_statistics.csc_nonzero_runs_visited !=
              exact_nonzero_run_visits ||
          final_batch_statistics.active_lanes_across_nonzero_runs !=
              exact_active_lanes_across_runs) {
        throw std::runtime_error{
            "batched Jacobi device/host work accounting disagrees"};
      }
      output.metrics.csc_runs_considered =
          final_batch_statistics.csc_runs_considered;
      output.metrics.csc_runs_visited =
          final_batch_statistics.csc_nonzero_runs_visited;
      output.metrics.csc_runs_skipped =
          final_batch_statistics.csc_runs_skipped;
      output.metrics.active_vertex_lane_evaluations =
          final_batch_statistics.active_vertex_lane_evaluations;
    }

    final_statistics.host_checks = output.metrics.convergence_host_checks;
    final_statistics.host_synchronizations =
        output.metrics.convergence_host_checks;
    checked_add_u64(
        final_statistics.host_synchronizations,
        1U,
        "host synchronization count");
    final_statistics.controller_copies =
        output.metrics.convergence_host_checks;
    checked_add_u64(
        final_statistics.controller_copies,
        1U,
        "controller copy count");
    final_statistics.kernel_dispatches = convergence_kernel_dispatches;
    if (options.control_mode != ControlMode::persistent_cooperative) {
      checked_add_u64(
          final_statistics.kernel_dispatches,
          2U,
          "kernel dispatch count");
    }
    output.result.status = final_status;
    output.result.work = final_statistics;

    const std::uint64_t selected_lane_vertices =
        selected_lane_vertex_count(batch);
    output.metrics.distance_reset_bytes = checked_multiply_u64(
        selected_lane_vertices,
        2U * sizeof(float),
        "batched Jacobi selected reset bytes");
    output.metrics.source_seed_write_bytes = checked_multiply_u64(
        batch.sources.size(),
        2U * sizeof(float),
        "batched Jacobi source seed bytes");
    output.metrics.padded_distance_reset_bytes = 0U;
    const std::uint32_t valid_lane_count = static_cast<std::uint32_t>(
        std::popcount(batch.valid_lane_mask));
    output.metrics.padded_lane_rounds = checked_multiply_u64(
        batch.lane_width - valid_lane_count,
        final_status.rounds_completed,
        "padded lane rounds");
    output.metrics.unused_wave_lane_rounds = checked_multiply_u64(
        phase14_wave_width - batch.lane_width,
        final_status.rounds_completed,
        "unused physical wave lane rounds");
    output.metrics.union_tile_lane_positions = checked_multiply_u64(
        batch.union_tiles.size(),
        valid_lane_count,
        "union tile lane positions");
    for (const std::uint32_t tile : batch.union_tiles) {
      checked_add_u64(
          output.metrics.selected_tile_lane_positions,
          static_cast<std::uint64_t>(
              std::popcount(batch.tile_lane_masks[tile])),
          "selected tile lane positions");
    }
    output.metrics.edge_record_read_bytes_requested =
        checked_multiply_u64(
            exact_physical_edge_records,
            sizeof(std::uint32_t) + sizeof(float),
            "batched Jacobi edge record request bytes");
    output.metrics.source_distance_read_bytes_requested =
        checked_multiply_u64(
            exact_lane_edge_pairs,
            sizeof(float),
            "batched Jacobi source distance request bytes");
    if (exact_nonzero_run_visits != 0U) {
      output.metrics.average_active_lanes_per_nonzero_run =
          static_cast<double>(exact_active_lanes_across_runs) /
          static_cast<double>(exact_nonzero_run_visits);
    }
    const std::uint64_t possible_lane_rounds = checked_multiply_u64(
        final_status.rounds_completed,
        batch.lane_width,
        "possible lane rounds");
    if (possible_lane_rounds != 0U) {
      output.metrics.lane_round_utilization =
          static_cast<double>(output.metrics.executed_lane_rounds) /
          static_cast<double>(possible_lane_rounds);
    }
    if (output.metrics.sssp_device_timeline_milliseconds > 0.0F) {
      output.metrics.batch_queries_per_second =
          static_cast<double>(std::popcount(batch.valid_lane_mask)) * 1000.0 /
          output.metrics.sssp_device_timeline_milliseconds;
    }

    output.metrics.end_to_end_wall_milliseconds =
        end_to_end_timer.elapsed_milliseconds();
    return output;
  } catch (...) {
    if (lease_active) {
      try {
        impl_->workspace.retire(impl_->stream);
      } catch (...) {
        impl_->workspace.recover_noexcept();
      }
    }
    throw;
  }
}

}  // namespace bfnew::hip
