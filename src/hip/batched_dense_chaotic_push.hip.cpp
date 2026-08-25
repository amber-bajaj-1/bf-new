#include "bfnew/hip/batched_dense_chaotic_push.hpp"

#include "bfnew/dense_chaotic_push.hpp"
#include "bfnew/workspace.hpp"

#include "batched_dense_workspace_internal.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

inline constexpr std::uint32_t phase15_wave_width = 32U;
inline constexpr std::uint32_t positive_infinity_bits = 0x7f800000U;

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
        "batched dense grid block count exceeds the HIP launch ABI"};
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
    const BatchDeviceDescription& batch) {
  if (!supported_batched_dense_width(batch.lane_width) ||
      batch.valid_lane_mask == 0U ||
      (batch.valid_lane_mask & ~width_mask(batch.lane_width)) != 0U ||
      (batch.valid_lane_mask & (batch.valid_lane_mask + 1U)) != 0U) {
    throw std::invalid_argument{
        "batched dense requires width 1/8/16/32 and a nonempty low valid mask"};
  }
  const std::size_t width = batch.lane_width;
  if (batch.query_ids_by_lane.size() != width ||
      batch.expansion_generations_by_lane.size() != width ||
      batch.selected_vertex_counts_by_lane.size() != width ||
      batch.selected_edge_estimates_by_lane.size() != width ||
      !valid_offsets(batch.source_offsets, width, batch.sources.size()) ||
      !valid_offsets(batch.target_offsets, width, batch.targets.size())) {
    throw std::invalid_argument{
        "batched dense terminal or lane metadata has an invalid shape"};
  }
  if (batch.reached_lane_mask != 0U || batch.miss_lane_mask != 0U) {
    throw std::invalid_argument{
        "batched dense requires zero initial reached/miss masks"};
  }
  if (batch.tile_lane_masks.size() != graph.tile_coordinates().size() ||
      batch.run_representation !=
          BatchRunRepresentation::retained_per_run_masks ||
      !batch.run_representation_initialized ||
      batch.csr_run_lane_masks.size() !=
          tile_runs.csr_run_destination_tiles.size()) {
    throw std::invalid_argument{
        "batched dense requires the Phase 13 retained CSR mask image"};
  }
  if (batch.union_tiles.empty() ||
      batch.selected_vertex_ranges.size() != batch.union_tiles.size()) {
    throw std::invalid_argument{
        "batched dense requires one selected range per union tile"};
  }

  const auto tile_offsets = graph.tile_vertex_offsets();
  std::vector<bool> union_tile_flags(graph.tile_coordinates().size(), false);
  std::uint32_t preceding_tile = 0U;
  bool has_preceding_tile = false;
  for (std::size_t index = 0U; index < batch.union_tiles.size(); ++index) {
    const std::uint32_t tile = batch.union_tiles[index];
    if (tile >= graph.tile_coordinates().size() ||
        (has_preceding_tile && tile <= preceding_tile)) {
      throw std::invalid_argument{"batched dense union tiles are not canonical"};
    }
    const BatchVertexRange range = batch.selected_vertex_ranges[index];
    if (range.begin != tile_offsets[tile] ||
        range.end != tile_offsets[tile + 1U] || range.begin > range.end ||
        range.lane_mask != batch.tile_lane_masks[tile] ||
        range.lane_mask == 0U) {
      throw std::invalid_argument{
          "batched dense selected range disagrees with its union tile mask"};
    }
    union_tile_flags[tile] = true;
    preceding_tile = tile;
    has_preceding_tile = true;
  }
  for (std::size_t tile = 0U; tile < batch.tile_lane_masks.size(); ++tile) {
    const LaneMask mask = batch.tile_lane_masks[tile];
    if ((mask & ~batch.valid_lane_mask) != 0U) {
      throw std::invalid_argument{
          "batched dense tile mask contains an invalid or padded lane"};
    }
    if (union_tile_flags[tile] != (mask != 0U)) {
      throw std::invalid_argument{
          "batched dense nonzero tile masks must match union ranges exactly"};
    }
  }

  std::vector<bool> selected_source_run(
      batch.csr_run_lane_masks.size(), false);
  for (const BatchVertexRange range : batch.selected_vertex_ranges) {
    for (std::uint32_t vertex = range.begin; vertex < range.end; ++vertex) {
      const std::size_t run_begin =
          static_cast<std::size_t>(tile_runs.csr_row_run_offsets[vertex]);
      const std::size_t run_end = static_cast<std::size_t>(
          tile_runs.csr_row_run_offsets[vertex + 1U]);
      for (std::size_t run = run_begin; run < run_end; ++run) {
        selected_source_run[run] = true;
        const std::uint32_t destination_tile =
            tile_runs.csr_run_destination_tiles[run].value();
        const LaneMask expected =
            range.lane_mask & batch.tile_lane_masks[destination_tile];
        if (batch.csr_run_lane_masks[run] != expected) {
          throw std::invalid_argument{
              "batched dense retained CSR mask differs from endpoint admission"};
        }
      }
    }
  }
  for (std::size_t run = 0U; run < batch.csr_run_lane_masks.size(); ++run) {
    const LaneMask mask = batch.csr_run_lane_masks[run];
    if ((mask & ~batch.valid_lane_mask) != 0U ||
        (!selected_source_run[run] && mask != 0U)) {
      throw std::invalid_argument{
          "batched dense CSR mask is nonzero outside selected source rows"};
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
          "batched dense valid lanes require terminals and padding does not"};
    }
  }
  const auto require_terminal = [&](const std::uint32_t vertex,
                                    const std::size_t lane) {
    if (vertex >= graph.vertex_count()) {
      throw std::invalid_argument{"batched dense terminal is out of range"};
    }
    const std::uint32_t tile = graph.owner_tiles()[vertex].value();
    if ((batch.tile_lane_masks[tile] & (LaneMask{1U} << lane)) == 0U) {
      throw std::invalid_argument{
          "batched dense terminal owner tile does not admit its lane"};
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

using detail::DeviceBatchDenseStatistics;
using detail::DeviceBatchDenseView;

__device__ void atomic_add_u64(
    std::uint64_t* const destination,
    const unsigned long long value) {
  atomicAdd(reinterpret_cast<unsigned long long*>(destination), value);
}

// All relaxation-time accesses to a distance word use this atomic domain. A
// CAS with identical compare/replacement values supplies an atomic-compatible
// source load even when another source wave is decreasing the same word.
[[nodiscard]] __device__ std::uint32_t atomic_load_nonnegative_float_bits(
    const std::uint32_t* const address) {
  return atomicCAS(const_cast<std::uint32_t*>(address), 0U, 0U);
}

// Over nonnegative finite IEEE-754 values and +infinity, unsigned bit order is
// float order. Every destination lane owns an independent word and CAS loop.
[[nodiscard]] __device__ bool atomic_min_nonnegative_float_bits(
    std::uint32_t* const address,
    const std::uint32_t candidate) {
  std::uint32_t observed = atomic_load_nonnegative_float_bits(address);
  while (candidate < observed) {
    const std::uint32_t preceding = atomicCAS(address, observed, candidate);
    if (preceding == observed) {
      return true;
    }
    observed = preceding;
  }
  return false;
}

[[nodiscard]] __device__ std::uint32_t* batch_dense_distance_bits(
    const DeviceWorkspaceView& workspace) {
  return reinterpret_cast<std::uint32_t*>(workspace.engine_scratch);
}

__device__ void initialize_batched_dense_state(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence) {
  static_cast<void>(graph);
  const std::uint64_t global_thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::uint32_t* const distances = batch_dense_distance_bits(workspace);

  const std::uint32_t wave_lane = threadIdx.x & (phase15_wave_width - 1U);
  const std::uint32_t wave_in_block = threadIdx.x / phase15_wave_width;
  const std::uint32_t waves_per_block = blockDim.x / phase15_wave_width;
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
    const std::uint64_t element =
        static_cast<std::uint64_t>(vertex) * batch.lane_width + wave_lane;
    distances[element] = source ? 0U : positive_infinity_bits;
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
        static_cast<std::uint32_t>(EngineKind::dense_chaotic_push);
    controller->enable_per_lane_convergence = enable_per_lane_convergence;
    controller->rounds_completed = 0U;
    controller->maximum_rounds = maximum_rounds;
    controller->distance_read_slot = 0U;
    controller->distance_write_slot = 0U;
    controller->frontier_read_slot = 0U;
    controller->frontier_write_slot = 1U;
    controller->frontier_size[0] = 0U;
    controller->frontier_size[1] = 0U;
    controller->next_frontier_lane_mask = 0U;
    controller->done = 0U;
    controller->stop_reason =
        static_cast<std::uint32_t>(DeviceStopReason::none);
    controller->error_bits = device_error::none;
  }
}

template <bool explicit_broadcast, bool collect_statistics>
__device__ void perform_batched_dense_round(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const batch_statistics,
    const std::uint32_t instrumentation_level) {
  if constexpr (!collect_statistics) {
    static_cast<void>(batch_statistics);
    static_cast<void>(instrumentation_level);
  }
  const std::uint32_t done = workspace.controller->done;
  const LaneMask execute_lane_mask = workspace.controller->execute_lane_mask;
  if (done != 0U || execute_lane_mask == 0U) {
    return;
  }

  const std::uint64_t required_bytes =
      static_cast<std::uint64_t>(graph.vertex_count) * batch.lane_width *
      sizeof(std::uint32_t);
  if (workspace.engine_scratch == nullptr ||
      workspace.engine_scratch_bytes < required_bytes ||
      workspace.controller->distance_read_slot != 0U ||
      workspace.controller->distance_write_slot != 0U) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      atomicOr(
          &workspace.controller->error_bits,
          device_error::invalid_controller_state);
    }
    return;
  }

  const std::uint32_t wave_lane = threadIdx.x & (phase15_wave_width - 1U);
  const std::uint32_t wave_in_block = threadIdx.x / phase15_wave_width;
  const std::uint32_t waves_per_block = blockDim.x / phase15_wave_width;
  const std::uint64_t global_wave =
      static_cast<std::uint64_t>(blockIdx.x) * waves_per_block + wave_in_block;
  const std::uint64_t grid_waves =
      static_cast<std::uint64_t>(gridDim.x) * waves_per_block;
  const bool mapped_lane = wave_lane < batch.lane_width;
  const LaneMask lane_bit =
      mapped_lane ? LaneMask{1U} << wave_lane : LaneMask{0U};
  std::uint32_t* const distances = batch_dense_distance_bits(workspace);

  bool lane_changed = false;
  unsigned long long local_runs_considered = 0U;
  unsigned long long local_runs_skipped = 0U;
  unsigned long long local_nonzero_runs = 0U;
  unsigned long long local_edge_records = 0U;
  unsigned long long local_lane_edge_pairs = 0U;
  unsigned long long local_successful = 0U;
  unsigned long long local_active_lanes = 0U;
  unsigned long long local_active_source_lanes = 0U;

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
    const std::uint32_t source =
        range.begin + static_cast<std::uint32_t>(
                          packed_vertex -
                          batch.selected_range_vertex_offsets[low]);
    const LaneMask source_execute_mask =
        range.lane_mask & execute_lane_mask;
    if (source_execute_mask == 0U) {
      continue;
    }
    if constexpr (collect_statistics) {
      if (wave_lane == 0U) {
        local_active_source_lanes += __popc(source_execute_mask);
      }
    }

    const std::uint32_t run_begin = graph.csr.row_run_offsets[source];
    const std::uint32_t run_end = graph.csr.row_run_offsets[source + 1U];
    for (std::uint32_t run = run_begin; run < run_end; ++run) {
      LaneMask prepared_mask = 0U;
      if constexpr (explicit_broadcast) {
        if (wave_lane == 0U) {
          prepared_mask = workspace.run_lane_masks[run];
        }
        prepared_mask = __shfl(prepared_mask, 0, phase15_wave_width);
      } else {
        prepared_mask = workspace.run_lane_masks[run];
      }
      // The retained mask is endpoint-exact. Intersecting it once per run is
      // sufficient; the result is reused for every edge in this CSR run.
      const LaneMask active_run_mask = prepared_mask & execute_lane_mask;
      const std::uint32_t edge_begin = graph.csr.run_edge_offsets[run];
      const std::uint32_t edge_end = graph.csr.run_edge_offsets[run + 1U];
      if constexpr (collect_statistics) {
        if (wave_lane == 0U) {
          ++local_runs_considered;
          if (active_run_mask == 0U) {
            ++local_runs_skipped;
          } else {
            ++local_nonzero_runs;
            const unsigned int active_lanes = __popc(active_run_mask);
            const unsigned long long edge_count = edge_end - edge_begin;
            local_active_lanes += active_lanes;
            local_edge_records += edge_count;
            local_lane_edge_pairs += edge_count * active_lanes;
          }
        }
      }
      if (active_run_mask == 0U) {
        continue;
      }
      const bool run_executes_lane =
          mapped_lane && (active_run_mask & lane_bit) != 0U;
      for (std::uint32_t edge = edge_begin; edge < edge_end; ++edge) {
        std::uint32_t destination = 0U;
        float weight = 0.0F;
        if constexpr (explicit_broadcast) {
          if (wave_lane == 0U) {
            destination = graph.csr.destinations[edge];
            weight = graph.csr.weights[edge];
          }
          destination = __shfl(destination, 0, phase15_wave_width);
          weight = __shfl(weight, 0, phase15_wave_width);
        } else {
          // Uniform immutable edge addresses are the unmeasured default.
          destination = graph.csr.destinations[edge];
          weight = graph.csr.weights[edge];
        }

        bool decreased = false;
        if (run_executes_lane) {
          const std::uint64_t source_element =
              static_cast<std::uint64_t>(source) * batch.lane_width +
              wave_lane;
          // Reload once per admitted edge/lane. A concurrent wave may have
          // just decreased this source word during the same chaotic scan.
          const std::uint32_t source_bits =
              atomic_load_nonnegative_float_bits(
                  distances + source_element);
          const float candidate_value =
              __builtin_bit_cast(float, source_bits) + weight;
          const std::uint32_t candidate =
              __builtin_bit_cast(std::uint32_t, candidate_value);
          const std::uint64_t destination_element =
              static_cast<std::uint64_t>(destination) * batch.lane_width +
              wave_lane;
          decreased = atomic_min_nonnegative_float_bits(
              distances + destination_element, candidate);
          lane_changed = lane_changed || decreased;
        }
        if constexpr (collect_statistics) {
          const LaneMask decreased_lanes =
              static_cast<LaneMask>(__ballot(decreased));
          if (wave_lane == 0U) {
            local_successful += __popc(decreased_lanes);
          }
        }
      }
    }
  }

  const LaneMask changed_lanes =
      static_cast<LaneMask>(__ballot(lane_changed));
  if (wave_lane == 0U && changed_lanes != 0U) {
    atomicOr(&workspace.controller->changed_lane_mask, changed_lanes);
  }
  if constexpr (collect_statistics) {
    if (wave_lane == 0U) {
      if (batch_statistics != nullptr) {
        atomic_add_u64(
            &batch_statistics->csr_runs_considered,
            local_runs_considered);
        atomic_add_u64(
            &batch_statistics->csr_runs_skipped,
            local_runs_skipped);
        atomic_add_u64(
            &batch_statistics->csr_nonzero_runs_visited,
            local_nonzero_runs);
        atomic_add_u64(
            &batch_statistics->csr_edge_records_loaded,
            local_edge_records);
        atomic_add_u64(
            &batch_statistics->admitted_lane_edge_pairs,
            local_lane_edge_pairs);
        atomic_add_u64(
            &batch_statistics->active_lanes_across_nonzero_runs,
            local_active_lanes);
        atomic_add_u64(
            &batch_statistics->active_source_lane_evaluations,
            local_active_source_lanes);
        if (instrumentation_level ==
            static_cast<std::uint32_t>(InstrumentationLevel::debug)) {
          atomic_add_u64(
              &batch_statistics->atomic_source_loads,
              local_lane_edge_pairs);
          atomic_add_u64(
              &batch_statistics->atomic_min_attempts,
              local_lane_edge_pairs);
          atomic_add_u64(
              &batch_statistics->successful_atomic_updates,
              local_successful);
        }
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
            local_active_source_lanes);
        if (instrumentation_level ==
            static_cast<std::uint32_t>(InstrumentationLevel::debug)) {
          atomic_add_u64(
              &workspace.instrumentation->mask_operations,
              local_runs_considered);
          atomic_add_u64(
              &workspace.instrumentation->atomic_attempts,
              local_lane_edge_pairs);
          atomic_add_u64(
              &workspace.instrumentation->successful_atomic_updates,
              local_successful);
        }
      }
    }
  }
}

__device__ void advance_batched_dense_controller(
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const batch_statistics,
    const std::uint32_t instrumentation_level) {
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

  const DenseAdvanceResult transition = advance_dense_controller(*controller);
  if (records_a_round && transition != DenseAdvanceResult::no_op &&
      transition != DenseAdvanceResult::invalid_controller_state) {
    if (batch_statistics != nullptr) {
      ++batch_statistics->full_edge_rounds;
      if (instrumentation_level ==
              static_cast<std::uint32_t>(InstrumentationLevel::debug) &&
          changed != 0U) {
        ++batch_statistics->changed_round_publications;
      }
    }
    if (workspace.instrumentation != nullptr) {
      workspace.instrumentation->active_lane_rounds += __popc(execute);
      ++workspace.instrumentation->full_edge_rounds;
      if (instrumentation_level ==
              static_cast<std::uint32_t>(InstrumentationLevel::debug) &&
          changed != 0U) {
        ++workspace.instrumentation->changed_flag_updates;
      }
    }
  }
}

[[nodiscard]] __device__ DeviceRunStatus completed_batched_dense_status(
    const DeviceWorkspaceView& workspace,
    const DeviceBatchDenseView& batch) {
  const DeviceController controller = *workspace.controller;
  const bool converged =
      controller.stop_reason ==
          static_cast<std::uint32_t>(DeviceStopReason::converged) &&
      controller.error_bits == device_error::none &&
      controller.distance_read_slot == 0U &&
      controller.distance_write_slot == 0U;
  LaneMask reached = 0U;
  LaneMask missed = 0U;
  if (converged) {
    const std::uint32_t* const distances =
        batch_dense_distance_bits(workspace);
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
        const std::uint64_t element =
            static_cast<std::uint64_t>(target) * batch.lane_width + lane;
        const std::uint32_t target_bits =
            atomic_load_nonnegative_float_bits(distances + element);
        lane_reached =
            lane_reached && isfinite(__builtin_bit_cast(float, target_bits));
      }
      if (lane_reached) {
        reached |= bit;
      } else {
        missed |= bit;
      }
    }
  }
  return make_dense_run_status(controller, reached, missed);
}

__global__ void initialize_batched_dense_query(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence) {
  initialize_batched_dense_state(
      graph,
      workspace,
      batch,
      maximum_rounds,
      enable_per_lane_convergence);
}

__global__ void batched_dense_round_none_uniform_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  perform_batched_dense_round<false, false>(
      graph, workspace, batch, statistics, instrumentation_level);
}

__global__ void batched_dense_round_none_broadcast_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  perform_batched_dense_round<true, false>(
      graph, workspace, batch, statistics, instrumentation_level);
}

__global__ void batched_dense_round_instrumented_uniform_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  perform_batched_dense_round<false, true>(
      graph, workspace, batch, statistics, instrumentation_level);
}

__global__ void batched_dense_round_instrumented_broadcast_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  perform_batched_dense_round<true, true>(
      graph, workspace, batch, statistics, instrumentation_level);
}

__global__ void batched_dense_advance_kernel(
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    advance_batched_dense_controller(
        workspace, batch, statistics, instrumentation_level);
  }
}

__global__ void finalize_batched_dense_status(
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *workspace.status = completed_batched_dense_status(workspace, batch);
  }
}

template <bool explicit_broadcast, bool collect_statistics>
__device__ void execute_batched_dense_persistent(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const statistics,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
  initialize_batched_dense_state(
      graph,
      workspace,
      batch,
      maximum_rounds,
      enable_per_lane_convergence);
  grid.sync();

  for (;;) {
    perform_batched_dense_round<explicit_broadcast, collect_statistics>(
        graph, workspace, batch, statistics, instrumentation_level);
    // All CAS updates in this complete scan finish before the single owner
    // freezes no-change lanes or publishes the next execute mask.
    grid.sync();
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      advance_batched_dense_controller(
          workspace, batch, statistics, instrumentation_level);
    }
    grid.sync();
    if (workspace.controller->done != 0U) {
      break;
    }
  }

  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *workspace.status = completed_batched_dense_status(workspace, batch);
  }
  grid.sync();
}

__global__ void batched_dense_persistent_none_uniform_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const statistics,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  execute_batched_dense_persistent<false, false>(
      graph,
      workspace,
      batch,
      statistics,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level);
}

__global__ void batched_dense_persistent_none_broadcast_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const statistics,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  execute_batched_dense_persistent<true, false>(
      graph,
      workspace,
      batch,
      statistics,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level);
}

__global__ void batched_dense_persistent_instrumented_uniform_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const statistics,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  execute_batched_dense_persistent<false, true>(
      graph,
      workspace,
      batch,
      statistics,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level);
}

__global__ void batched_dense_persistent_instrumented_broadcast_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchDenseView batch,
    DeviceBatchDenseStatistics* const statistics,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  execute_batched_dense_persistent<true, true>(
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
    const BatchedDenseLoadStrategy strategy) noexcept {
  if (instrumentation == InstrumentationLevel::none) {
    return strategy == BatchedDenseLoadStrategy::compiler_uniform
               ? reinterpret_cast<const void*>(
                     batched_dense_round_none_uniform_kernel)
               : reinterpret_cast<const void*>(
                     batched_dense_round_none_broadcast_kernel);
  }
  return strategy == BatchedDenseLoadStrategy::compiler_uniform
             ? reinterpret_cast<const void*>(
                   batched_dense_round_instrumented_uniform_kernel)
             : reinterpret_cast<const void*>(
                   batched_dense_round_instrumented_broadcast_kernel);
}

[[nodiscard]] const void* persistent_kernel_pointer(
    const InstrumentationLevel instrumentation,
    const BatchedDenseLoadStrategy strategy) noexcept {
  if (instrumentation == InstrumentationLevel::none) {
    return strategy == BatchedDenseLoadStrategy::compiler_uniform
               ? reinterpret_cast<const void*>(
                     batched_dense_persistent_none_uniform_kernel)
               : reinterpret_cast<const void*>(
                     batched_dense_persistent_none_broadcast_kernel);
  }
  return strategy == BatchedDenseLoadStrategy::compiler_uniform
             ? reinterpret_cast<const void*>(
                   batched_dense_persistent_instrumented_uniform_kernel)
             : reinterpret_cast<const void*>(
                   batched_dense_persistent_instrumented_broadcast_kernel);
}

[[nodiscard]] std::uint32_t kernel_occupancy(
    const void* const kernel,
    const std::uint32_t block_size) {
  int active = 0;
  check(
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active, kernel, static_cast<int>(block_size), 0U),
      "hipOccupancyMaxActiveBlocksPerMultiprocessor(batched dense kernel)");
  if (active <= 0) {
    throw std::runtime_error{
        "batched dense kernel has zero reported occupancy"};
  }
  return static_cast<std::uint32_t>(active);
}

[[nodiscard]] std::uint32_t kernel_register_count(const void* const kernel) {
  hipFuncAttributes attributes{};
  check(
      hipFuncGetAttributes(&attributes, kernel),
      "hipFuncGetAttributes(batched dense kernel)");
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
    const BatchedDenseLoadStrategy strategy) {
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
          "requested cooperative grid exceeds batched dense occupancy"};
    }
    blocks_per_wgp = options.blocks_per_wgp;
  }
  return CooperativeGrid{
      checked_grid_blocks(blocks_per_wgp, properties.wgp_count), active};
}

[[nodiscard]] std::uint32_t ordinary_grid_blocks(
    const DeviceProperties& properties,
    const GpuRunOptions& options,
    const BatchedDenseLoadStrategy strategy,
    const std::uint32_t union_vertex_count,
    std::uint32_t& active_blocks_per_wgp) {
  active_blocks_per_wgp = kernel_occupancy(
      round_kernel_pointer(options.instrumentation, strategy),
      options.block_size);
  std::uint32_t blocks_per_wgp = active_blocks_per_wgp;
  if (options.grid_policy == GridPolicy::fixed_blocks_per_wgp) {
    if (options.blocks_per_wgp > active_blocks_per_wgp) {
      throw std::invalid_argument{
          "requested ordinary grid exceeds batched dense occupancy"};
    }
    blocks_per_wgp = options.blocks_per_wgp;
  }
  const std::uint32_t resident =
      checked_grid_blocks(blocks_per_wgp, properties.wgp_count);
  const std::uint32_t waves_per_block =
      options.block_size / phase15_wave_width;
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
    const DeviceBatchDenseView& batch,
    const GpuRunOptions& options,
    const hipStream_t stream) {
  const std::uint32_t waves_per_block =
      options.block_size / phase15_wave_width;
  const std::uint32_t blocks = static_cast<std::uint32_t>(
      std::max<std::uint64_t>(
          1U,
          (static_cast<std::uint64_t>(batch.union_vertex_count) +
           waves_per_block - 1U) /
              waves_per_block));
  hipLaunchKernelGGL(
      initialize_batched_dense_query,
      dim3(blocks),
      dim3(options.block_size),
      0U,
      stream,
      graph,
      workspace,
      batch,
      options.maximum_rounds,
      options.enable_per_lane_convergence);
  check_launch("initialize_batched_dense_query launch");
}

void launch_round_pair(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchDenseView& batch,
    DeviceBatchDenseStatistics* const statistics,
    const GpuRunOptions& options,
    const BatchedDenseLoadStrategy strategy,
    const std::uint32_t blocks,
    const hipStream_t stream) {
  const std::uint32_t instrumentation =
      static_cast<std::uint32_t>(options.instrumentation);
  if (options.instrumentation == InstrumentationLevel::none) {
    if (strategy == BatchedDenseLoadStrategy::compiler_uniform) {
      hipLaunchKernelGGL(
          batched_dense_round_none_uniform_kernel,
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
          batched_dense_round_none_broadcast_kernel,
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
  } else if (strategy == BatchedDenseLoadStrategy::compiler_uniform) {
    hipLaunchKernelGGL(
        batched_dense_round_instrumented_uniform_kernel,
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
        batched_dense_round_instrumented_broadcast_kernel,
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
  check_launch("batched dense round launch");
  hipLaunchKernelGGL(
      batched_dense_advance_kernel,
      dim3(1U),
      dim3(1U),
      0U,
      stream,
      workspace,
      batch,
      statistics,
      instrumentation);
  check_launch("batched_dense_advance_kernel launch");
}

void launch_finalize(
    const DeviceWorkspaceView& workspace,
    const DeviceBatchDenseView& batch,
    const hipStream_t stream) {
  hipLaunchKernelGGL(
      finalize_batched_dense_status,
      dim3(1U),
      dim3(1U),
      0U,
      stream,
      workspace,
      batch);
  check_launch("finalize_batched_dense_status launch");
}

void launch_persistent(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchDenseView& batch,
    DeviceBatchDenseStatistics* const statistics,
    const CooperativeGrid grid,
    const GpuRunOptions& options,
    const BatchedDenseLoadStrategy strategy,
    const hipStream_t stream) {
  DeviceGraphView32 graph_argument = graph;
  DeviceWorkspaceView workspace_argument = workspace;
  DeviceBatchDenseView batch_argument = batch;
  DeviceBatchDenseStatistics* statistics_argument = statistics;
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
      "hipLaunchCooperativeKernel(batched dense persistent)");
}

[[nodiscard]] std::uint64_t selected_lane_vertex_count(
    const BatchDeviceDescription& batch) {
  std::uint64_t count = 0U;
  for (const BatchVertexRange range : batch.selected_vertex_ranges) {
    checked_add_u64(
        count,
        checked_multiply_u64(
            range.end - range.begin,
            static_cast<std::uint64_t>(std::popcount(range.lane_mask)),
            "batched dense selected lane vertices"),
        "batched dense selected lane vertices");
  }
  return count;
}

[[nodiscard]] std::uint64_t maximum_rounds_for_mask(
    const LaneMask mask,
    const std::vector<std::uint64_t>& lane_rounds,
    const std::uint32_t lane_width) {
  std::uint64_t maximum = 0U;
  for (std::uint32_t lane = 0U; lane < lane_width; ++lane) {
    if ((mask & (LaneMask{1U} << lane)) != 0U) {
      maximum = std::max(maximum, lane_rounds[lane]);
    }
  }
  return maximum;
}

[[nodiscard]] std::uint64_t summed_rounds_for_mask(
    const LaneMask mask,
    const std::vector<std::uint64_t>& lane_rounds,
    const std::uint32_t lane_width,
    const std::string_view what) {
  std::uint64_t sum = 0U;
  for (std::uint32_t lane = 0U; lane < lane_width; ++lane) {
    if ((mask & (LaneMask{1U} << lane)) != 0U) {
      checked_add_u64(sum, lane_rounds[lane], what);
    }
  }
  return sum;
}

struct ExactDenseWork {
  std::uint64_t runs_considered{};
  std::uint64_t runs_visited{};
  std::uint64_t runs_skipped{};
  std::uint64_t active_lanes_across_runs{};
  std::uint64_t edge_records{};
  std::uint64_t lane_edge_pairs{};
  std::uint64_t active_source_lanes{};
};

[[nodiscard]] ExactDenseWork reconstruct_exact_dense_work(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const BatchDeviceDescription& batch,
    const std::vector<std::uint64_t>& lane_rounds) {
  ExactDenseWork exact;
  for (const BatchVertexRange range : batch.selected_vertex_ranges) {
    const std::uint64_t source_maximum = maximum_rounds_for_mask(
        range.lane_mask, lane_rounds, batch.lane_width);
    const std::uint64_t source_sum = summed_rounds_for_mask(
        range.lane_mask,
        lane_rounds,
        batch.lane_width,
        "batched dense active source lanes");
    for (std::uint32_t source = range.begin; source < range.end; ++source) {
      checked_add_u64(
          exact.active_source_lanes,
          source_sum,
          "batched dense active source lanes");
      const std::size_t run_begin =
          static_cast<std::size_t>(tile_runs.csr_row_run_offsets[source]);
      const std::size_t run_end = static_cast<std::size_t>(
          tile_runs.csr_row_run_offsets[source + 1U]);
      for (std::size_t run = run_begin; run < run_end; ++run) {
        const LaneMask mask = batch.csr_run_lane_masks[run];
        const std::uint64_t run_maximum = maximum_rounds_for_mask(
            mask, lane_rounds, batch.lane_width);
        const std::uint64_t run_sum = summed_rounds_for_mask(
            mask,
            lane_rounds,
            batch.lane_width,
            "batched dense active lanes over runs");
        if (run_maximum > source_maximum) {
          throw std::runtime_error{
              "batched dense reconstructed run outlives its source tile"};
        }
        checked_add_u64(
            exact.runs_considered,
            source_maximum,
            "batched dense considered runs");
        checked_add_u64(
            exact.runs_visited,
            run_maximum,
            "batched dense visited runs");
        checked_add_u64(
            exact.runs_skipped,
            source_maximum - run_maximum,
            "batched dense skipped runs");
        checked_add_u64(
            exact.active_lanes_across_runs,
            run_sum,
            "batched dense active lanes over runs");
        const std::uint64_t edge_count =
            tile_runs.csr_run_edge_offsets[run + 1U] -
            tile_runs.csr_run_edge_offsets[run];
        checked_add_u64(
            exact.edge_records,
            checked_multiply_u64(
                edge_count, run_maximum, "batched dense edge records"),
            "batched dense edge records");
        checked_add_u64(
            exact.lane_edge_pairs,
            checked_multiply_u64(
                edge_count, run_sum, "batched dense lane-edge pairs"),
            "batched dense lane-edge pairs");
      }
    }
  }
  static_cast<void>(graph);
  return exact;
}

[[nodiscard]] std::uint64_t lane_edge_pairs_per_complete_batch_round(
    const TileRunLayout64& tile_runs,
    const BatchDeviceDescription& batch) {
  std::uint64_t pairs = 0U;
  for (std::size_t run = 0U; run < batch.csr_run_lane_masks.size(); ++run) {
    const std::uint64_t edges =
        tile_runs.csr_run_edge_offsets[run + 1U] -
        tile_runs.csr_run_edge_offsets[run];
    const std::uint64_t lanes = static_cast<std::uint64_t>(
        std::popcount(batch.csr_run_lane_masks[run]));
    checked_add_u64(
        pairs,
        checked_multiply_u64(edges, lanes, "batched dense batch-round edges"),
        "batched dense batch-round edges");
  }
  return pairs;
}

}  // namespace

std::size_t batched_dense_distance_scratch_bytes(
    const std::uint32_t vertex_count,
    const std::uint32_t lane_width) {
  if (vertex_count == 0U || !supported_batched_dense_width(lane_width)) {
    throw std::invalid_argument{
        "batched dense scratch requires vertices and width 1/8/16/32"};
  }
  const std::size_t lane_vertices = checked_multiply(
      static_cast<std::size_t>(vertex_count),
      lane_width,
      "batched dense lane vertices");
  return checked_multiply(
      lane_vertices,
      sizeof(std::uint32_t),
      "batched dense distance words");
}

class BatchedDenseChaoticPushEngine::Impl final {
 public:
  Impl(
      const WeightedGraph& host_graph_value,
      const TileRunLayout64& tile_runs_value,
      const ResidentDeviceGraph& resident_graph_value,
      ReusableBatchedDenseWorkspace& workspace_value,
      const HipStream& stream_value)
      : host_graph{host_graph_value},
        tile_runs{tile_runs_value},
        resident_graph{resident_graph_value},
        workspace{workspace_value},
        stream{stream_value} {
    if (!validate_weighted_graph(host_graph).ok() ||
        !host_graph.has_spatial_ordering()) {
      throw std::invalid_argument{
          "batched dense requires a valid spatially ordered host graph"};
    }
    if (!validate_tile_run_layout(host_graph, tile_runs).ok()) {
      throw std::invalid_argument{
          "batched dense requires deeply valid tile-run metadata"};
    }
    if (!resident_graph.has_upload()) {
      throw std::invalid_argument{
          "batched dense requires an uploaded resident graph"};
    }
    const DeviceGraphView32& device = resident_graph.view();
    if (device.vertex_count != host_graph.vertex_count() ||
        device.edge_count != host_graph.edge_count() ||
        device.tile_count != host_graph.tile_coordinates().size() ||
        device.csr.run_count != tile_runs.csr_run_destination_tiles.size()) {
      throw std::invalid_argument{
          "batched dense host graph/run shapes disagree with resident graph"};
    }
    if (resident_graph.fingerprint() !=
        fingerprint_device_graph_source32(host_graph, tile_runs)) {
      throw std::invalid_argument{
          "batched dense host graph/run content disagrees with resident graph"};
    }
  }

  const WeightedGraph& host_graph;
  const TileRunLayout64& tile_runs;
  const ResidentDeviceGraph& resident_graph;
  ReusableBatchedDenseWorkspace& workspace;
  const HipStream& stream;

  std::vector<std::uint32_t> downloaded_distance_bits;
  HipEventTimer preparation_gpu_timer;
  HipEventTimer sssp_gpu_timer;
  HipEventTimer result_transfer_gpu_timer;
};

BatchedDenseChaoticPushEngine::BatchedDenseChaoticPushEngine(
    const WeightedGraph& host_graph,
    const TileRunLayout64& tile_runs,
    const ResidentDeviceGraph& resident_graph,
    ReusableBatchedDenseWorkspace& workspace,
    const HipStream& stream)
    : impl_{new Impl{
          host_graph, tile_runs, resident_graph, workspace, stream}} {}

BatchedDenseChaoticPushEngine::~BatchedDenseChaoticPushEngine() {
  delete impl_;
}

BatchedDenseRunOutput BatchedDenseChaoticPushEngine::run_status_only(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    const BatchedDenseLoadStrategy load_strategy) {
  return run_impl(
      batch, options, load_strategy, false, false, nullptr, nullptr);
}

DeviceRunStatus BatchedDenseChaoticPushEngine::run_compact_status(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    const BatchedDenseLoadStrategy load_strategy) {
  return run_impl(
             batch, options, load_strategy, false, true, nullptr, nullptr)
      .result.status;
}

CompactPathBatchOutput BatchedDenseChaoticPushEngine::run_compact_paths(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    ReusableCompactPathWorkspace& compact_workspace,
    const BatchedDenseLoadStrategy load_strategy) {
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

BatchedDenseRunOutput BatchedDenseChaoticPushEngine::run_with_distances(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    const BatchedDenseLoadStrategy load_strategy) {
  return run_impl(
      batch, options, load_strategy, true, false, nullptr, nullptr);
}

BatchedDenseRunOutput BatchedDenseChaoticPushEngine::run_impl(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    const BatchedDenseLoadStrategy load_strategy,
    const bool download_distances,
    const bool compact_status_only,
    ReusableCompactPathWorkspace* const compact_workspace,
    CompactPathBatchOutput* const compact_output) {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot run an invalid batched dense engine"};
  }
  if (validate_gpu_run_options(options) != GpuRunOptionsError::none ||
      options.engine != EngineKind::dense_chaotic_push ||
      !valid_batched_dense_load_strategy(load_strategy)) {
    throw std::invalid_argument{
        "batched dense options are invalid or select another engine"};
  }
  if ((compact_workspace == nullptr) != (compact_output == nullptr) ||
      (compact_output != nullptr &&
       (download_distances || compact_status_only ||
        options.instrumentation != InstrumentationLevel::none))) {
    throw std::invalid_argument{
        "batched dense compact-path mode requires paired workspace/output, "
        "None instrumentation, and no other readback mode"};
  }
  validate_batch_description_shape(impl_->host_graph, impl_->tile_runs, batch);

  const DeviceProperties properties = current_device_properties();
  if (properties.wave_width != phase15_wave_width) {
    throw std::runtime_error{
        "Phase 15 wave-to-query mapping requires runtime wave32"};
  }
  if (options.block_size > properties.maximum_threads_per_block ||
      options.block_size % phase15_wave_width != 0U) {
    throw std::invalid_argument{
        "batched dense block size must be a legal multiple of wave32"};
  }

  const DeviceGraphView32 graph = impl_->resident_graph.view();
  const std::size_t scratch_bytes = batched_dense_distance_scratch_bytes(
      graph.vertex_count, batch.lane_width);
  bool lease_active = false;

  SteadyClockTimer end_to_end_timer;
  BatchedDenseRunOutput output;
  output.distances_downloaded = download_distances;
  output.result.engine_kind =
      static_cast<std::uint32_t>(EngineKind::dense_chaotic_push);
  output.result.control_mode =
      static_cast<std::uint32_t>(options.control_mode);
  output.metrics.lane_width = batch.lane_width;
  output.metrics.valid_lane_count = static_cast<std::uint32_t>(
      std::popcount(batch.valid_lane_mask));
  output.metrics.load_strategy = load_strategy;

  try {
    impl_->resident_graph.wait_until_ready(impl_->stream);
    impl_->preparation_gpu_timer.start(impl_->stream);
    impl_->workspace.prepare_async(
        batch, options, scratch_bytes, impl_->stream);
    lease_active = true;
    impl_->preparation_gpu_timer.stop(impl_->stream);

    const DeviceWorkspaceView workspace_view =
        impl_->workspace.device_workspace_view();
    const DeviceBatchDenseView batch_view =
        impl_->workspace.device_batch_view();
    DeviceBatchDenseStatistics* const device_batch_statistics =
        impl_->workspace.device_batch_statistics();
    const hipStream_t native_stream = as_stream(impl_->stream.native_handle());
    const void* const selected_kernel =
        options.control_mode == ControlMode::persistent_cooperative
            ? persistent_kernel_pointer(options.instrumentation, load_strategy)
            : round_kernel_pointer(options.instrumentation, load_strategy);
    output.metrics.kernel_registers_per_thread =
        kernel_register_count(selected_kernel);

    impl_->sssp_gpu_timer.start(impl_->stream);
    if (options.control_mode != ControlMode::persistent_cooperative) {
      launch_initialize(
          graph, workspace_view, batch_view, options, native_stream);
    }

    std::uint64_t total_kernel_dispatches = 0U;
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
      total_kernel_dispatches = 1U;
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
              "batched dense engine round dispatches");
          checked_add_u64(
              output.metrics.controller_advance_dispatches,
              1U,
              "batched dense controller dispatches");
          checked_add_u64(
              total_kernel_dispatches,
              2U,
              "batched dense total kernel dispatches");
        }
        impl_->workspace.download_controller_async(
            observed_controller, impl_->stream);
        impl_->stream.synchronize();
        checked_add_u64(
            output.metrics.convergence_host_checks,
            1U,
            "batched dense convergence host checks");
      } while (observed_controller.done == 0U);
      launch_finalize(workspace_view, batch_view, native_stream);
      // Initialization and final status are separate ordinary kernels.
      checked_add_u64(
          total_kernel_dispatches,
          2U,
          "batched dense total kernel dispatches");
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
          1U,
          slot_stride,
          CompactDistanceEncoding::nonnegative_float_bits,
          nullptr,
          reinterpret_cast<const std::uint32_t*>(
              workspace_view.engine_scratch)};
      *compact_output = compact_workspace->reconstruct(
          graph,
          workspace_view,
          compact_batch,
          compact_distances,
          batch,
          impl_->stream);
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
          compact_status.final_distance_slot != 0U ||
          (normal_convergence
               ? classified != batch.valid_lane_mask
               : classified != 0U)) {
        throw std::runtime_error{
            "batched dense produced an invalid compact-path status"};
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
          compact_status.final_distance_slot != 0U ||
          (normal_convergence
               ? classified != batch.valid_lane_mask
               : classified != 0U)) {
        throw std::runtime_error{
            "batched dense produced an invalid compact terminal status"};
      }
      output.result.status = compact_status;
      output.metrics.end_to_end_wall_milliseconds =
          end_to_end_timer.elapsed_milliseconds();
      return output;
    }

    DeviceController final_controller{};
    DeviceRunStatus final_status{};
    DeviceWorkStatistics final_statistics{};
    DeviceBatchDenseStatistics final_batch_statistics{};
    output.lane_convergence_rounds.resize(batch.lane_width);
    if (download_distances) {
      impl_->downloaded_distance_bits.resize(
          scratch_bytes / sizeof(std::uint32_t));
    } else {
      impl_->downloaded_distance_bits.clear();
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
        impl_->downloaded_distance_bits,
        impl_->stream);
    impl_->result_transfer_gpu_timer.stop(impl_->stream);
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
        final_status.final_distance_slot != 0U ||
        final_status.rounds_completed != final_controller.rounds_completed ||
        final_status.valid_lane_mask != batch.valid_lane_mask) {
      throw std::runtime_error{
          "batched dense produced inconsistent terminal controller/status"};
    }

    const std::size_t elements = checked_multiply(
        graph.vertex_count,
        batch.lane_width,
        "batched dense result elements");
    if (download_distances) {
      output.distance_bits.assign(elements, positive_infinity_bits);
      for (const BatchVertexRange range : batch.selected_vertex_ranges) {
        for (std::uint32_t vertex = range.begin; vertex < range.end; ++vertex) {
          for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
            const LaneMask bit = LaneMask{1U} << lane;
            if ((range.lane_mask & bit) == 0U) {
              continue;
            }
            const std::size_t element =
                static_cast<std::size_t>(vertex) * batch.lane_width + lane;
            output.distance_bits[element] =
                impl_->downloaded_distance_bits[element];
          }
        }
      }
      output.distances.reserve(elements);
      for (const std::uint32_t bits : output.distance_bits) {
        output.distances.push_back(std::bit_cast<float>(bits));
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
      checked_add_u64(
          output.batch_work.active_lane_rounds,
          executed,
          "batched dense active lane rounds");
      if (convergence != 0U) {
        if (convergence > final_status.rounds_completed) {
          throw std::runtime_error{
              "batched dense convergence round exceeds batch duration"};
        }
        const std::uint64_t tail =
            final_status.rounds_completed - convergence;
        output.lane_tail_rounds[lane] = tail;
        checked_add_u64(
            output.batch_work.tail_lane_rounds,
            tail,
            "batched dense tail lane rounds");
        if (options.enable_per_lane_convergence != 0U) {
          checked_add_u64(
              output.batch_work.tail_lane_rounds_avoided,
              tail,
              "batched dense avoided tail rounds");
        } else {
          checked_add_u64(
              output.batch_work.tail_lane_rounds_executed,
              tail,
              "batched dense executed tail rounds");
        }
      }
    }

    const ExactDenseWork exact = reconstruct_exact_dense_work(
        impl_->host_graph,
        impl_->tile_runs,
        batch,
        output.lane_executed_rounds);
    output.batch_work.csr_runs_considered = exact.runs_considered;
    output.batch_work.csr_runs_visited = exact.runs_visited;
    output.batch_work.csr_runs_skipped = exact.runs_skipped;
    output.batch_work.active_lanes_over_visited_runs =
        exact.active_lanes_across_runs;
    output.batch_work.csr_edge_loads = exact.edge_records;
    output.batch_work.lane_edge_relaxations = exact.lane_edge_pairs;
    output.batch_work.atomic_source_loads = exact.lane_edge_pairs;
    output.batch_work.atomic_min_attempts = exact.lane_edge_pairs;
    output.batch_work.active_source_lane_evaluations =
        exact.active_source_lanes;
    output.batch_work.full_edge_rounds = final_status.rounds_completed;

    const std::uint64_t baseline_lane_edges = checked_multiply_u64(
        lane_edge_pairs_per_complete_batch_round(impl_->tile_runs, batch),
        final_status.rounds_completed,
        "batched dense full-duration lane edges");
    if (exact.lane_edge_pairs > baseline_lane_edges) {
      throw std::runtime_error{
          "batched dense logical edge accounting exceeded its baseline"};
    }
    output.batch_work.lane_edge_relaxations_avoided_by_early_convergence =
        baseline_lane_edges - exact.lane_edge_pairs;

    if (options.instrumentation != InstrumentationLevel::none) {
      if (final_batch_statistics.csr_runs_considered !=
              exact.runs_considered ||
          final_batch_statistics.csr_nonzero_runs_visited !=
              exact.runs_visited ||
          final_batch_statistics.csr_runs_skipped != exact.runs_skipped ||
          final_batch_statistics.active_lanes_across_nonzero_runs !=
              exact.active_lanes_across_runs ||
          final_batch_statistics.csr_edge_records_loaded !=
              exact.edge_records ||
          final_batch_statistics.admitted_lane_edge_pairs !=
              exact.lane_edge_pairs ||
          final_batch_statistics.active_source_lane_evaluations !=
              exact.active_source_lanes ||
          final_batch_statistics.full_edge_rounds !=
              final_status.rounds_completed) {
        throw std::runtime_error{
            "batched dense device/host work accounting disagrees"};
      }
      output.batch_work.successful_atomic_updates =
          final_statistics.successful_decreases;
      output.batch_work.changed_round_publications =
          final_batch_statistics.changed_round_publications;

      if (options.instrumentation == InstrumentationLevel::debug &&
          (final_batch_statistics.atomic_source_loads !=
               exact.lane_edge_pairs ||
           final_batch_statistics.atomic_min_attempts !=
               exact.lane_edge_pairs ||
           final_batch_statistics.successful_atomic_updates !=
               final_statistics.successful_decreases)) {
        throw std::runtime_error{
            "batched dense Debug atomic counters disagree"};
      }

      const bool common_light_matches =
          final_statistics.edges_examined == exact.lane_edge_pairs &&
          final_statistics.successful_decreases ==
              output.batch_work.successful_atomic_updates &&
          final_statistics.active_vertices == exact.active_source_lanes &&
          final_statistics.active_lane_rounds ==
              output.batch_work.active_lane_rounds &&
          final_statistics.full_edge_rounds == final_status.rounds_completed;
      const bool common_debug_matches =
          options.instrumentation != InstrumentationLevel::debug ||
          (final_statistics.atomic_attempts == exact.lane_edge_pairs &&
           final_statistics.successful_atomic_updates ==
               output.batch_work.successful_atomic_updates &&
           final_statistics.mask_operations == exact.runs_considered &&
           final_statistics.changed_flag_updates ==
               final_batch_statistics.changed_round_publications);
      if (!common_light_matches || !common_debug_matches) {
        throw std::runtime_error{
            "batched dense common/device counters disagree"};
      }
    }

    const std::uint64_t valid_lanes =
        static_cast<std::uint64_t>(std::popcount(batch.valid_lane_mask));
    output.batch_work.valid_lane_round_capacity = checked_multiply_u64(
        final_status.rounds_completed,
        valid_lanes,
        "batched dense valid lane-round capacity");
    output.batch_work.lane_width_round_capacity = checked_multiply_u64(
        final_status.rounds_completed,
        batch.lane_width,
        "batched dense width lane-round capacity");
    output.batch_work.wave32_lane_round_capacity = checked_multiply_u64(
        final_status.rounds_completed,
        phase15_wave_width,
        "batched dense wave32 lane-round capacity");
    if (output.batch_work.lane_width_round_capacity >
        output.batch_work.wave32_lane_round_capacity) {
      throw std::runtime_error{
          "batched dense lane width exceeds wave32 capacity"};
    }
    output.batch_work.unused_wave_lane_round_capacity =
        output.batch_work.wave32_lane_round_capacity -
        output.batch_work.lane_width_round_capacity;
    output.batch_work.padded_lane_round_capacity = checked_multiply_u64(
        final_status.rounds_completed,
        static_cast<std::uint64_t>(batch.lane_width) - valid_lanes,
        "batched dense padded lane-round capacity");
    if (output.batch_work.active_lane_rounds >
        output.batch_work.valid_lane_round_capacity) {
      throw std::runtime_error{
          "batched dense active lane rounds exceed valid capacity"};
    }
    output.batch_work.inactive_valid_lane_rounds =
        output.batch_work.valid_lane_round_capacity -
        output.batch_work.active_lane_rounds;
    output.batch_work.edge_wave_lane_capacity = checked_multiply_u64(
        exact.edge_records,
        phase15_wave_width,
        "batched dense edge-wave lane capacity");
    if (exact.lane_edge_pairs > output.batch_work.edge_wave_lane_capacity) {
      throw std::runtime_error{
          "batched dense lane-edge work exceeds wave32 capacity"};
    }
    output.batch_work.unused_edge_wave_lane_capacity =
        output.batch_work.edge_wave_lane_capacity - exact.lane_edge_pairs;
    output.batch_work.distance_reset_bytes = checked_multiply_u64(
        selected_lane_vertex_count(batch),
        sizeof(std::uint32_t),
        "batched dense selected reset bytes");
    output.batch_work.source_seed_write_bytes = checked_multiply_u64(
        batch.sources.size(),
        sizeof(std::uint32_t),
        "batched dense source seed bytes");
    output.batch_work.union_tile_lane_positions = checked_multiply_u64(
        batch.union_tiles.size(),
        valid_lanes,
        "batched dense union tile/lane positions");
    for (const std::uint32_t tile : batch.union_tiles) {
      checked_add_u64(
          output.batch_work.selected_tile_lane_positions,
          static_cast<std::uint64_t>(
              std::popcount(batch.tile_lane_masks[tile])),
          "batched dense selected tile/lane positions");
    }

    output.metrics.unused_wave_lane_rounds =
        output.batch_work.unused_wave_lane_round_capacity;
    output.metrics.edge_record_read_bytes_requested = checked_multiply_u64(
        exact.edge_records,
        sizeof(std::uint32_t) + sizeof(float),
        "batched dense edge record request bytes");
    output.metrics.atomic_source_read_bytes_requested = checked_multiply_u64(
        exact.lane_edge_pairs,
        sizeof(std::uint32_t),
        "batched dense atomic source request bytes");
    output.metrics.atomic_destination_access_bytes_requested =
        checked_multiply_u64(
            exact.lane_edge_pairs,
            sizeof(std::uint32_t),
            "batched dense atomic destination request bytes");
    if (exact.runs_visited != 0U) {
      output.metrics.average_active_lanes_per_nonzero_run =
          static_cast<double>(exact.active_lanes_across_runs) /
          static_cast<double>(exact.runs_visited);
    }
    if (output.batch_work.wave32_lane_round_capacity != 0U) {
      output.metrics.lane_round_utilization =
          static_cast<double>(output.batch_work.active_lane_rounds) /
          static_cast<double>(output.batch_work.wave32_lane_round_capacity);
    }
    if (output.metrics.sssp_device_timeline_milliseconds > 0.0F) {
      output.metrics.batch_queries_per_second =
          static_cast<double>(valid_lanes) * 1000.0 /
          output.metrics.sssp_device_timeline_milliseconds;
    }

    final_statistics.host_checks =
        output.metrics.convergence_host_checks;
    final_statistics.host_synchronizations =
        output.metrics.convergence_host_checks;
    checked_add_u64(
        final_statistics.host_synchronizations,
        1U,
        "batched dense host synchronization count");
    final_statistics.controller_copies =
        output.metrics.convergence_host_checks;
    checked_add_u64(
        final_statistics.controller_copies,
        2U,
        "batched dense controller copy count");
    final_statistics.kernel_dispatches = total_kernel_dispatches;
    output.result.status = final_status;
    output.result.work = final_statistics;

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
