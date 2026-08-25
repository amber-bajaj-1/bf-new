#include "bfnew/hip/batched_frontier_push.hpp"

#include "bfnew/frontier_push.hpp"
#include "bfnew/workspace.hpp"

#include "batched_frontier_workspace_internal.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
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
        "batched frontier grid block count exceeds the HIP launch ABI"};
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
  if (!supported_batched_frontier_width(batch.lane_width) ||
      batch.valid_lane_mask == 0U ||
      (batch.valid_lane_mask & ~width_mask(batch.lane_width)) != 0U ||
      (batch.valid_lane_mask & (batch.valid_lane_mask + 1U)) != 0U) {
    throw std::invalid_argument{
        "batched frontier requires width 1/8/16/32 and a nonempty low valid mask"};
  }
  const std::size_t width = batch.lane_width;
  if (batch.query_ids_by_lane.size() != width ||
      batch.expansion_generations_by_lane.size() != width ||
      batch.selected_vertex_counts_by_lane.size() != width ||
      batch.selected_edge_estimates_by_lane.size() != width ||
      !valid_offsets(batch.source_offsets, width, batch.sources.size()) ||
      !valid_offsets(batch.target_offsets, width, batch.targets.size())) {
    throw std::invalid_argument{
        "batched frontier terminal or lane metadata has an invalid shape"};
  }
  if (batch.reached_lane_mask != 0U || batch.miss_lane_mask != 0U) {
    throw std::invalid_argument{
        "batched frontier requires zero initial reached/miss masks"};
  }
  if (batch.tile_lane_masks.size() != graph.tile_coordinates().size() ||
      batch.run_representation !=
          BatchRunRepresentation::retained_per_run_masks ||
      !batch.run_representation_initialized ||
      batch.csr_run_lane_masks.size() !=
          tile_runs.csr_run_destination_tiles.size()) {
    throw std::invalid_argument{
        "batched frontier requires the Phase 13 retained CSR mask image"};
  }
  if (batch.union_tiles.empty() ||
      batch.selected_vertex_ranges.size() != batch.union_tiles.size()) {
    throw std::invalid_argument{
        "batched frontier requires one selected range per union tile"};
  }

  const auto tile_offsets = graph.tile_vertex_offsets();
  std::vector<bool> union_tile_flags(graph.tile_coordinates().size(), false);
  std::uint32_t preceding_tile = 0U;
  bool has_preceding_tile = false;
  for (std::size_t index = 0U; index < batch.union_tiles.size(); ++index) {
    const std::uint32_t tile = batch.union_tiles[index];
    if (tile >= graph.tile_coordinates().size() ||
        (has_preceding_tile && tile <= preceding_tile)) {
      throw std::invalid_argument{"batched frontier union tiles are not canonical"};
    }
    const BatchVertexRange range = batch.selected_vertex_ranges[index];
    if (range.begin != tile_offsets[tile] ||
        range.end != tile_offsets[tile + 1U] || range.begin > range.end ||
        range.lane_mask != batch.tile_lane_masks[tile] ||
        range.lane_mask == 0U) {
      throw std::invalid_argument{
          "batched frontier selected range disagrees with its union tile mask"};
    }
    union_tile_flags[tile] = true;
    preceding_tile = tile;
    has_preceding_tile = true;
  }
  for (std::size_t tile = 0U; tile < batch.tile_lane_masks.size(); ++tile) {
    const LaneMask mask = batch.tile_lane_masks[tile];
    if ((mask & ~batch.valid_lane_mask) != 0U) {
      throw std::invalid_argument{
          "batched frontier tile mask contains an invalid or padded lane"};
    }
    if (union_tile_flags[tile] != (mask != 0U)) {
      throw std::invalid_argument{
          "batched frontier nonzero tile masks must match union ranges exactly"};
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
              "batched frontier retained CSR mask differs from endpoint admission"};
        }
      }
    }
  }
  for (std::size_t run = 0U; run < batch.csr_run_lane_masks.size(); ++run) {
    const LaneMask mask = batch.csr_run_lane_masks[run];
    if ((mask & ~batch.valid_lane_mask) != 0U ||
        (!selected_source_run[run] && mask != 0U)) {
      throw std::invalid_argument{
          "batched frontier CSR mask is nonzero outside selected source rows"};
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
          "batched frontier valid lanes require terminals and padding does not"};
    }
    if (!valid) {
      if (batch.query_ids_by_lane[lane] != invalid_batch_query_id.value() ||
          batch.expansion_generations_by_lane[lane] != 0U ||
          batch.selected_vertex_counts_by_lane[lane] != 0U ||
          batch.selected_edge_estimates_by_lane[lane] != 0U) {
        throw std::invalid_argument{
            "batched frontier padded lane metadata is not canonical"};
      }
      continue;
    }
    for (std::size_t preceding_lane = 0U; preceding_lane < lane;
         ++preceding_lane) {
      const LaneMask preceding_bit = LaneMask{1U} << preceding_lane;
      if ((batch.valid_lane_mask & preceding_bit) != 0U &&
          batch.query_ids_by_lane[preceding_lane] ==
              batch.query_ids_by_lane[lane]) {
        throw std::invalid_argument{
            "batched frontier valid lane query IDs must be unique"};
      }
    }
  }
  const auto require_terminal = [&](const std::uint32_t vertex,
                                    const std::size_t lane) {
    if (vertex >= graph.vertex_count()) {
      throw std::invalid_argument{"batched frontier terminal is out of range"};
    }
    const std::uint32_t tile = graph.owner_tiles()[vertex].value();
    if ((batch.tile_lane_masks[tile] & (LaneMask{1U} << lane)) == 0U) {
      throw std::invalid_argument{
          "batched frontier terminal owner tile does not admit its lane"};
    }
  };
  for (std::size_t lane = 0U; lane < width; ++lane) {
    const std::size_t source_begin = batch.source_offsets[lane];
    const std::size_t source_end = batch.source_offsets[lane + 1U];
    for (std::size_t index = source_begin;
         index < source_end;
         ++index) {
      if (index != source_begin &&
          batch.sources[index - 1U] >= batch.sources[index]) {
        throw std::invalid_argument{
            "batched frontier source slices must be strictly increasing"};
      }
      require_terminal(batch.sources[index], lane);
    }
    const std::size_t target_begin = batch.target_offsets[lane];
    const std::size_t target_end = batch.target_offsets[lane + 1U];
    for (std::size_t index = target_begin;
         index < target_end;
         ++index) {
      if (index != target_begin &&
          batch.targets[index - 1U] >= batch.targets[index]) {
        throw std::invalid_argument{
            "batched frontier target slices must be strictly increasing"};
      }
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

using detail::DeviceBatchFrontierStatistics;
using detail::DeviceBatchFrontierView;

struct DeviceBatchedFrontierScratchView {
  std::uint32_t* distance_bits{};
  std::uint32_t* queues[2]{};
  LaneMask* vertex_lane_masks[2]{};
  std::uint32_t vertex_count{};
  std::uint32_t lane_width{};
  std::uint32_t queue_capacity{};
};

[[nodiscard]] __device__ DeviceBatchedFrontierScratchView
bind_batched_frontier_scratch(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchFrontierView& batch) {
  const std::uint64_t distance_words =
      static_cast<std::uint64_t>(graph.vertex_count) * batch.lane_width;
  const std::uint64_t queue_words = batch.queue_capacity;
  const std::uint64_t mask_words = graph.vertex_count;
  const std::uint64_t required_words =
      distance_words + 2U * queue_words + 2U * mask_words;
  if (workspace.engine_scratch == nullptr || graph.vertex_count == 0U ||
      batch.lane_width == 0U || batch.queue_capacity == 0U ||
      batch.queue_capacity > graph.vertex_count ||
      workspace.engine_scratch_bytes <
          required_words * sizeof(std::uint32_t)) {
    return {};
  }
  auto* const words =
      reinterpret_cast<std::uint32_t*>(workspace.engine_scratch);
  LaneMask* const mask_zero =
      reinterpret_cast<LaneMask*>(words + distance_words);
  LaneMask* const mask_one = mask_zero + mask_words;
  std::uint32_t* const queue_zero =
      reinterpret_cast<std::uint32_t*>(mask_one + mask_words);
  std::uint32_t* const queue_one = queue_zero + queue_words;
  return DeviceBatchedFrontierScratchView{
      words,
      {queue_zero, queue_one},
      {mask_zero, mask_one},
      graph.vertex_count,
      batch.lane_width,
      batch.queue_capacity,
  };
}

[[nodiscard]] __device__ bool valid_batched_frontier_scratch(
    const DeviceBatchedFrontierScratchView& scratch) {
  return scratch.distance_bits != nullptr && scratch.queues[0] != nullptr &&
         scratch.queues[1] != nullptr &&
         scratch.vertex_lane_masks[0] != nullptr &&
         scratch.vertex_lane_masks[1] != nullptr &&
         scratch.vertex_count != 0U && scratch.lane_width != 0U &&
         scratch.queue_capacity != 0U &&
         scratch.queue_capacity <= scratch.vertex_count;
}

__device__ void atomic_add_u64(
    std::uint64_t* const destination,
    const unsigned long long value) {
  atomicAdd(reinterpret_cast<unsigned long long*>(destination), value);
}

__device__ void atomic_max_u64(
    std::uint64_t* const destination,
    const unsigned long long value) {
  atomicMax(reinterpret_cast<unsigned long long*>(destination), value);
}

[[nodiscard]] __device__ std::uint32_t atomic_load_nonnegative_float_bits(
    const std::uint32_t* const address) {
  return atomicCAS(const_cast<std::uint32_t*>(address), 0U, 0U);
}

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

[[nodiscard]] __device__ std::uint64_t reserve_queue_position(
    std::uint64_t* const size) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(size), 1ULL);
}

[[nodiscard]] __device__ std::uint32_t source_lane_for_position(
    const DeviceBatchFrontierView& batch,
    const std::uint32_t position) {
  for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
    if (batch.source_offsets[lane] <= position &&
        position < batch.source_offsets[lane + 1U]) {
      return lane;
    }
  }
  return batch.lane_width;
}

__device__ void initialize_batched_frontier_storage(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchFrontierView batch,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence) {
  const DeviceBatchedFrontierScratchView scratch =
      bind_batched_frontier_scratch(graph, workspace, batch);
  const std::uint64_t global_thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t grid_threads =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  if (!valid_batched_frontier_scratch(scratch)) {
    if (global_thread == 0U) {
      atomicOr(
          &workspace.controller->error_bits,
          device_error::invalid_controller_state);
    }
    return;
  }

  for (std::uint64_t packed_vertex = global_thread;
       packed_vertex < batch.union_vertex_count;
       packed_vertex += grid_threads) {
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
    scratch.vertex_lane_masks[0][vertex] = 0U;
    scratch.vertex_lane_masks[1][vertex] = 0U;
    for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
      const LaneMask bit = LaneMask{1U} << lane;
      if ((range.lane_mask & bit) != 0U) {
        const std::uint64_t element =
            static_cast<std::uint64_t>(vertex) * batch.lane_width + lane;
        scratch.distance_bits[element] = positive_infinity_bits;
      }
    }
  }
  for (std::uint64_t lane = global_thread;
       lane < batch.lane_width;
       lane += grid_threads) {
    batch.lane_convergence_rounds[lane] = 0U;
  }

  if (global_thread == 0U) {
    DeviceController* const controller = workspace.controller;
    controller->valid_lane_mask = batch.valid_lane_mask;
    controller->active_lane_mask = batch.valid_lane_mask;
    controller->changed_lane_mask = 0U;
    controller->converged_lane_mask = 0U;
    controller->execute_lane_mask = batch.valid_lane_mask;
    controller->engine_kind =
        static_cast<std::uint32_t>(EngineKind::frontier_push);
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

__device__ void seed_batched_frontier_sources(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchFrontierView batch,
    DeviceBatchFrontierStatistics* const batch_statistics,
    const std::uint32_t instrumentation_level) {
  const DeviceBatchedFrontierScratchView scratch =
      bind_batched_frontier_scratch(graph, workspace, batch);
  const std::uint64_t global_thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t grid_threads =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  if (!valid_batched_frontier_scratch(scratch)) {
    return;
  }
  for (std::uint64_t source_position = global_thread;
       source_position < workspace.source_count;
       source_position += grid_threads) {
    const std::uint32_t position =
        static_cast<std::uint32_t>(source_position);
    const std::uint32_t lane = source_lane_for_position(batch, position);
    if (lane >= batch.lane_width) {
      atomicOr(
          &workspace.controller->error_bits,
          device_error::invalid_controller_state);
      continue;
    }
    const LaneMask bit = LaneMask{1U} << lane;
    const std::uint32_t source = workspace.sources[position];
    const std::uint64_t element =
        static_cast<std::uint64_t>(source) * batch.lane_width + lane;
    scratch.distance_bits[element] = 0U;
    const LaneMask preceding =
        atomicOr(scratch.vertex_lane_masks[0] + source, bit);
    if (preceding != 0U) {
      continue;
    }
    const std::uint64_t claim = reserve_queue_position(
        &workspace.controller->frontier_size[0]);
    const std::uint64_t queue_size = claim + 1U;
    if (workspace.instrumentation != nullptr) {
      atomic_max_u64(
          &workspace.instrumentation->maximum_queue_size, queue_size);
    }
    if (batch_statistics != nullptr) {
      atomic_max_u64(&batch_statistics->maximum_queue_size, queue_size);
    }
    if (claim < batch.queue_capacity) {
      scratch.queues[0][claim] = source;
    } else {
      const std::uint32_t old_error = atomicOr(
          &workspace.controller->error_bits, device_error::queue_overflow);
      if ((old_error & device_error::queue_overflow) == 0U &&
          instrumentation_level ==
              static_cast<std::uint32_t>(InstrumentationLevel::debug)) {
        if (workspace.instrumentation != nullptr) {
          atomic_add_u64(&workspace.instrumentation->overflow_events, 1U);
        }
        if (batch_statistics != nullptr) {
          atomic_add_u64(&batch_statistics->overflow_events, 1U);
        }
      }
    }
  }
}

__device__ void finish_batched_frontier_seed(
    const DeviceWorkspaceView workspace) {
  DeviceController* const controller = workspace.controller;
  if ((controller->error_bits & device_error::queue_overflow) != 0U) {
    ::bfnew::detail::canonicalize_frontier_error(
        *controller,
        DeviceStopReason::queue_overflow,
        device_error::queue_overflow);
  } else if (controller->frontier_size[0] == 0U ||
             controller->error_bits != device_error::none) {
    ::bfnew::detail::canonicalize_frontier_error(
        *controller,
        DeviceStopReason::invalid_controller_state,
        device_error::invalid_controller_state);
  }
}

struct FrontierThreadStatistics {
  unsigned long long worklist_vertices{};
  unsigned long long active_vertex_lanes{};
  unsigned long long multi_lane_worklist_vertices{};
  unsigned long long runs_considered{};
  unsigned long long runs_skipped{};
  unsigned long long runs_visited{};
  unsigned long long active_lanes_over_runs{};
  unsigned long long edge_records{};
  unsigned long long multi_lane_edge_records{};
  unsigned long long lane_edge_pairs{};
  unsigned long long successful_lane_updates{};
  unsigned long long current_mask_exchanges{};
  unsigned long long next_mask_ors{};
  unsigned long long controller_mask_ors{};
  unsigned long long unique_lane_activations{};
  unsigned long long queue_claims{};
  unsigned long long queue_entries_saved{};
  unsigned long long same_lane_duplicate_suppressions{};
  unsigned long long duplicate_suppressions{};
  unsigned long long maximum_queue_size{};
  unsigned long long overflow_events{};
};

__device__ void flush_batched_frontier_statistics(
    const DeviceWorkspaceView workspace,
    DeviceBatchFrontierStatistics* const batch_statistics,
    const FrontierThreadStatistics& local,
    const std::uint32_t instrumentation_level) {
  if (batch_statistics != nullptr) {
    atomic_add_u64(
        &batch_statistics->worklist_vertices, local.worklist_vertices);
    atomic_add_u64(
        &batch_statistics->active_vertex_lanes,
        local.active_vertex_lanes);
    atomic_add_u64(
        &batch_statistics->multi_lane_worklist_vertices,
        local.multi_lane_worklist_vertices);
    atomic_add_u64(
        &batch_statistics->csr_runs_considered, local.runs_considered);
    atomic_add_u64(
        &batch_statistics->csr_runs_skipped, local.runs_skipped);
    atomic_add_u64(
        &batch_statistics->csr_nonzero_runs_visited, local.runs_visited);
    atomic_add_u64(
        &batch_statistics->active_lanes_across_nonzero_runs,
        local.active_lanes_over_runs);
    atomic_add_u64(
        &batch_statistics->csr_edge_records_loaded, local.edge_records);
    atomic_add_u64(
        &batch_statistics->multi_lane_csr_edge_records,
        local.multi_lane_edge_records);
    atomic_add_u64(
        &batch_statistics->admitted_lane_edge_pairs,
        local.lane_edge_pairs);
    atomic_add_u64(
        &batch_statistics->distance_atomic_successes,
        local.successful_lane_updates);
    atomic_max_u64(
        &batch_statistics->maximum_queue_size, local.maximum_queue_size);
    if (instrumentation_level ==
        static_cast<std::uint32_t>(InstrumentationLevel::debug)) {
      atomic_add_u64(
          &batch_statistics->distance_atomic_source_loads,
          local.lane_edge_pairs);
      atomic_add_u64(
          &batch_statistics->distance_atomic_attempts,
          local.lane_edge_pairs);
      atomic_add_u64(
          &batch_statistics->current_mask_atomic_exchanges,
          local.current_mask_exchanges);
      atomic_add_u64(
          &batch_statistics->next_mask_atomic_ors, local.next_mask_ors);
      atomic_add_u64(
          &batch_statistics->controller_mask_atomic_ors,
          local.controller_mask_ors);
      atomic_add_u64(
          &batch_statistics->lane_enqueue_transitions,
          local.unique_lane_activations);
      atomic_add_u64(
          &batch_statistics->queue_claims, local.queue_claims);
      atomic_add_u64(
          &batch_statistics->queue_entries_saved_by_lane_merging,
          local.queue_entries_saved);
      atomic_add_u64(
          &batch_statistics->same_lane_duplicate_suppressions,
          local.same_lane_duplicate_suppressions);
      atomic_add_u64(
          &batch_statistics->duplicate_suppressions,
          local.duplicate_suppressions);
      atomic_add_u64(
          &batch_statistics->overflow_events, local.overflow_events);
    }
  }

  if (workspace.instrumentation != nullptr) {
    atomic_add_u64(
        &workspace.instrumentation->edges_examined,
        local.lane_edge_pairs);
    atomic_add_u64(
        &workspace.instrumentation->successful_decreases,
        local.successful_lane_updates);
    atomic_add_u64(
        &workspace.instrumentation->active_vertices,
        local.worklist_vertices);
    atomic_max_u64(
        &workspace.instrumentation->maximum_queue_size,
        local.maximum_queue_size);
    if (instrumentation_level ==
        static_cast<std::uint32_t>(InstrumentationLevel::debug)) {
      atomic_add_u64(
          &workspace.instrumentation->mask_operations,
          local.runs_considered + local.current_mask_exchanges +
              local.next_mask_ors + local.controller_mask_ors);
      atomic_add_u64(
          &workspace.instrumentation->atomic_attempts,
          local.lane_edge_pairs);
      atomic_add_u64(
          &workspace.instrumentation->successful_atomic_updates,
          local.successful_lane_updates);
      atomic_add_u64(
          &workspace.instrumentation->queue_claims, local.queue_claims);
      atomic_add_u64(
          &workspace.instrumentation->duplicate_suppressions,
          local.duplicate_suppressions);
      atomic_add_u64(
          &workspace.instrumentation->overflow_events,
          local.overflow_events);
    }
  }
}

__device__ void perform_batched_frontier_round(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchFrontierView batch,
    DeviceBatchFrontierStatistics* const batch_statistics,
    const std::uint32_t instrumentation_level) {
  const std::uint32_t done = workspace.controller->done;
  const LaneMask execute_lane_mask = workspace.controller->execute_lane_mask;
  const std::uint32_t read_slot = workspace.controller->frontier_read_slot;
  const std::uint32_t write_slot = workspace.controller->frontier_write_slot;
  const std::uint64_t current_size =
      read_slot <= 1U ? workspace.controller->frontier_size[read_slot] : 0U;
  if (done != 0U || execute_lane_mask == 0U) {
    return;
  }
  const DeviceBatchedFrontierScratchView scratch =
      bind_batched_frontier_scratch(graph, workspace, batch);
  if (!valid_batched_frontier_scratch(scratch) || read_slot > 1U ||
      write_slot > 1U || read_slot == write_slot || current_size == 0U ||
      current_size > batch.queue_capacity) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      atomicOr(
          &workspace.controller->error_bits,
          device_error::invalid_controller_state);
    }
    return;
  }

  const bool collect_light =
      instrumentation_level !=
      static_cast<std::uint32_t>(InstrumentationLevel::none);
  const bool collect_debug =
      instrumentation_level ==
      static_cast<std::uint32_t>(InstrumentationLevel::debug);
  FrontierThreadStatistics local{};
  const std::uint64_t global_thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t grid_threads =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t queue_index = global_thread;
       queue_index < current_size;
       queue_index += grid_threads) {
    const std::uint32_t source = scratch.queues[read_slot][queue_index];
    if (source >= graph.vertex_count) {
      atomicOr(
          &workspace.controller->error_bits,
          device_error::invalid_controller_state);
      continue;
    }

    const LaneMask queued_mask =
        atomicExch(scratch.vertex_lane_masks[read_slot] + source, 0U);
    const std::uint32_t source_tile = graph.owner_tiles[source];
    const LaneMask current_lane_mask =
        queued_mask & execute_lane_mask & workspace.tile_lane_masks[source_tile];
    if (queued_mask == 0U || current_lane_mask == 0U ||
        queued_mask != current_lane_mask ||
        (queued_mask & ~batch.valid_lane_mask) != 0U) {
      atomicOr(
          &workspace.controller->error_bits,
          device_error::invalid_controller_state);
      continue;
    }
    if (collect_light) {
      const unsigned int active_lanes = __popc(current_lane_mask);
      ++local.worklist_vertices;
      local.active_vertex_lanes += active_lanes;
      if (active_lanes > 1U) {
        ++local.multi_lane_worklist_vertices;
      }
    }
    if (collect_debug) {
      ++local.current_mask_exchanges;
    }

    const std::uint32_t run_begin = graph.csr.row_run_offsets[source];
    const std::uint32_t run_end = graph.csr.row_run_offsets[source + 1U];
    for (std::uint32_t run = run_begin; run < run_end; ++run) {
      const LaneMask prepared_mask = workspace.run_lane_masks[run];
      const LaneMask active_run_mask =
          prepared_mask & current_lane_mask & execute_lane_mask;
      const std::uint32_t edge_begin = graph.csr.run_edge_offsets[run];
      const std::uint32_t edge_end = graph.csr.run_edge_offsets[run + 1U];
      if (collect_light) {
        ++local.runs_considered;
        if (active_run_mask == 0U) {
          ++local.runs_skipped;
        } else {
          ++local.runs_visited;
          const unsigned int active_lanes = __popc(active_run_mask);
          const unsigned long long edge_count = edge_end - edge_begin;
          local.active_lanes_over_runs += active_lanes;
          local.edge_records += edge_count;
          if (active_lanes > 1U) {
            local.multi_lane_edge_records += edge_count;
          }
          local.lane_edge_pairs += edge_count * active_lanes;
        }
      }
      if (active_run_mask == 0U) {
        continue;
      }

      for (std::uint32_t edge = edge_begin; edge < edge_end; ++edge) {
        // Phase 16 retains Phase 11's simple one-owner-thread scheduling. The
        // owner loops query bits; edge balancing and wave aggregation remain
        // future work rather than silently changing the scheduling policy.
        const std::uint32_t destination = graph.csr.destinations[edge];
        const float weight = graph.csr.weights[edge];
        LaneMask successful_lanes = 0U;
        for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
          const LaneMask bit = LaneMask{1U} << lane;
          if ((active_run_mask & bit) == 0U) {
            continue;
          }
          const std::uint64_t source_element =
              static_cast<std::uint64_t>(source) * batch.lane_width + lane;
          const std::uint32_t source_bits =
              atomic_load_nonnegative_float_bits(
                  scratch.distance_bits + source_element);
          const float candidate_value =
              __builtin_bit_cast(float, source_bits) + weight;
          const std::uint32_t candidate =
              __builtin_bit_cast(std::uint32_t, candidate_value);
          const std::uint64_t destination_element =
              static_cast<std::uint64_t>(destination) * batch.lane_width +
              lane;
          if (atomic_min_nonnegative_float_bits(
                  scratch.distance_bits + destination_element, candidate)) {
            successful_lanes |= bit;
          }
        }
        if (successful_lanes == 0U) {
          continue;
        }
        if (collect_light) {
          local.successful_lane_updates += __popc(successful_lanes);
        }

        const LaneMask old_destination_mask = atomicOr(
            scratch.vertex_lane_masks[write_slot] + destination,
            successful_lanes);
        const LaneMask newly_active_lanes =
            successful_lanes & ~old_destination_mask;
        atomicOr(
            &workspace.controller->next_frontier_lane_mask,
            successful_lanes);
        if (collect_debug) {
          ++local.next_mask_ors;
          ++local.controller_mask_ors;
          local.unique_lane_activations += __popc(newly_active_lanes);
        }

        if (old_destination_mask == 0U) {
          const std::uint64_t claim = reserve_queue_position(
              &workspace.controller->frontier_size[write_slot]);
          const std::uint64_t queue_size = claim + 1U;
          if (collect_light && local.maximum_queue_size < queue_size) {
            local.maximum_queue_size = queue_size;
          }
          if (collect_debug) {
            ++local.queue_claims;
            const unsigned int activated = __popc(newly_active_lanes);
            if (activated != 0U) {
              local.queue_entries_saved += activated - 1U;
            }
            local.duplicate_suppressions +=
                __popc(successful_lanes) - 1U;
          }
          if (claim < batch.queue_capacity) {
            scratch.queues[write_slot][claim] = destination;
          } else {
            const std::uint32_t old_error = atomicOr(
                &workspace.controller->error_bits,
                device_error::queue_overflow);
            if ((old_error & device_error::queue_overflow) == 0U &&
                collect_debug) {
              ++local.overflow_events;
            }
          }
        } else if (collect_debug) {
          const unsigned int newly_active = __popc(newly_active_lanes);
          const unsigned int successful = __popc(successful_lanes);
          local.queue_entries_saved += newly_active;
          local.same_lane_duplicate_suppressions +=
              successful - newly_active;
          local.duplicate_suppressions += successful;
        }
      }
    }
  }

  if (collect_light) {
    flush_batched_frontier_statistics(
        workspace, batch_statistics, local, instrumentation_level);
  }
}

__device__ void advance_batched_frontier_and_count(
    const DeviceWorkspaceView workspace,
    const DeviceBatchFrontierView batch,
    DeviceBatchFrontierStatistics* const batch_statistics,
    const std::uint32_t instrumentation_level) {
  DeviceController* const controller = workspace.controller;
  const LaneMask execute = controller->execute_lane_mask;
  const LaneMask next = controller->next_frontier_lane_mask & execute;
  const std::uint32_t read_slot = controller->frontier_read_slot;
  const std::uint32_t write_slot = controller->frontier_write_slot;
  const std::uint64_t current_size =
      read_slot <= 1U ? controller->frontier_size[read_slot] : 0U;
  const std::uint64_t next_size =
      write_slot <= 1U ? controller->frontier_size[write_slot] : 0U;
  const std::uint32_t error_bits = controller->error_bits;
  const bool records_a_round =
      controller->done == 0U && execute != 0U && current_size != 0U;
  const LaneMask no_next = execute & ~next;

  const BatchedFrontierAdvanceResult transition =
      advance_batched_frontier_controller(*controller);
  if (!records_a_round ||
      transition == BatchedFrontierAdvanceResult::no_op ||
      transition == BatchedFrontierAdvanceResult::invalid_controller_state) {
    return;
  }
  const bool accepted_clean_round =
      error_bits == device_error::none &&
      (transition == BatchedFrontierAdvanceResult::continue_execution ||
       transition == BatchedFrontierAdvanceResult::converged ||
       transition == BatchedFrontierAdvanceResult::maximum_rounds);
  if (accepted_clean_round) {
    // Error, invalid, and no-op transitions never publish convergence proof.
    const std::uint64_t completed_round = controller->rounds_completed;
    for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
      const LaneMask bit = LaneMask{1U} << lane;
      if ((no_next & bit) != 0U &&
          batch.lane_convergence_rounds[lane] == 0U) {
        batch.lane_convergence_rounds[lane] = completed_round;
      }
    }
  }
  if (batch_statistics != nullptr) {
    ++batch_statistics->frontier_rounds;
    if (error_bits == device_error::none && next_size == 0U) {
      ++batch_statistics->empty_frontier_rounds;
    }
    if (current_size < 32U) {
      ++batch_statistics->small_frontier_rounds;
    }
  }
  if (workspace.instrumentation != nullptr) {
    workspace.instrumentation->active_lane_rounds += __popc(execute);
    if (error_bits == device_error::none && next_size == 0U) {
      ++workspace.instrumentation->empty_frontier_rounds;
    }
    if (current_size < 32U) {
      ++workspace.instrumentation->small_frontier_rounds;
    }
    static_cast<void>(instrumentation_level);
  }
}

[[nodiscard]] __device__ DeviceRunStatus completed_batched_frontier_status(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchFrontierView& batch) {
  const DeviceController controller = *workspace.controller;
  const bool converged =
      controller.stop_reason ==
          static_cast<std::uint32_t>(DeviceStopReason::converged) &&
      controller.error_bits == device_error::none;
  LaneMask reached = 0U;
  LaneMask missed = 0U;
  const DeviceBatchedFrontierScratchView scratch =
      bind_batched_frontier_scratch(graph, workspace, batch);
  if (converged && valid_batched_frontier_scratch(scratch)) {
    for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
      const LaneMask bit = LaneMask{1U} << lane;
      if ((batch.valid_lane_mask & bit) == 0U) {
        continue;
      }
      bool lane_reached = true;
      const std::uint32_t begin = batch.target_offsets[lane];
      const std::uint32_t end = batch.target_offsets[lane + 1U];
      for (std::uint32_t index = begin; index < end; ++index) {
        const std::uint32_t target = workspace.targets[index];
        const std::uint64_t element =
            static_cast<std::uint64_t>(target) * batch.lane_width + lane;
        lane_reached =
            lane_reached &&
            atomic_load_nonnegative_float_bits(
                scratch.distance_bits + element) != positive_infinity_bits;
      }
      if (lane_reached) {
        reached |= bit;
      } else {
        missed |= bit;
      }
    }
  }
  return make_frontier_run_status(controller, reached, missed);
}

__global__ void initialize_batched_frontier_storage_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchFrontierView batch,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence) {
  initialize_batched_frontier_storage(
      graph,
      workspace,
      batch,
      maximum_rounds,
      enable_per_lane_convergence);
}

__global__ void seed_batched_frontier_sources_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchFrontierView batch,
    DeviceBatchFrontierStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  seed_batched_frontier_sources(
      graph, workspace, batch, statistics, instrumentation_level);
}

__global__ void finish_batched_frontier_seed_kernel(
    const DeviceWorkspaceView workspace) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    finish_batched_frontier_seed(workspace);
  }
}

__global__ void batched_frontier_round_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchFrontierView batch,
    DeviceBatchFrontierStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  perform_batched_frontier_round(
      graph, workspace, batch, statistics, instrumentation_level);
}

__global__ void batched_frontier_advance_kernel(
    const DeviceWorkspaceView workspace,
    const DeviceBatchFrontierView batch,
    DeviceBatchFrontierStatistics* const statistics,
    const std::uint32_t instrumentation_level) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    advance_batched_frontier_and_count(
        workspace, batch, statistics, instrumentation_level);
  }
}

__global__ void finalize_batched_frontier_status(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchFrontierView batch) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *workspace.status =
        completed_batched_frontier_status(graph, workspace, batch);
  }
}

__global__ void batched_frontier_persistent_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceBatchFrontierView batch,
    DeviceBatchFrontierStatistics* const statistics,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
  initialize_batched_frontier_storage(
      graph,
      workspace,
      batch,
      maximum_rounds,
      enable_per_lane_convergence);
  grid.sync();
  seed_batched_frontier_sources(
      graph, workspace, batch, statistics, instrumentation_level);
  grid.sync();
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    finish_batched_frontier_seed(workspace);
  }
  grid.sync();

  while (workspace.controller->done == 0U) {
    perform_batched_frontier_round(
        graph, workspace, batch, statistics, instrumentation_level);
    grid.sync();
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      advance_batched_frontier_and_count(
          workspace, batch, statistics, instrumentation_level);
    }
    grid.sync();
  }
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *workspace.status =
        completed_batched_frontier_status(graph, workspace, batch);
  }
  grid.sync();
}

[[nodiscard]] std::uint32_t kernel_occupancy(
    const void* const kernel,
    const std::uint32_t block_size) {
  int active = 0;
  check(
      hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &active, kernel, static_cast<int>(block_size), 0U),
      "hipOccupancyMaxActiveBlocksPerMultiprocessor(batched frontier kernel)");
  if (active <= 0) {
    throw std::runtime_error{
        "batched frontier kernel has zero reported occupancy"};
  }
  return static_cast<std::uint32_t>(active);
}

[[nodiscard]] std::uint32_t kernel_register_count(const void* const kernel) {
  hipFuncAttributes attributes{};
  check(
      hipFuncGetAttributes(&attributes, kernel),
      "hipFuncGetAttributes(batched frontier kernel)");
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
    const GpuRunOptions& options) {
  if (!properties.cooperative_launch) {
    throw std::runtime_error{
        "selected HIP device does not support cooperative launch"};
  }
  const std::uint32_t active = kernel_occupancy(
      reinterpret_cast<const void*>(batched_frontier_persistent_kernel),
      options.block_size);
  std::uint32_t blocks_per_wgp = active;
  if (options.grid_policy == GridPolicy::fixed_blocks_per_wgp) {
    if (options.blocks_per_wgp > active) {
      throw std::invalid_argument{
          "requested cooperative grid exceeds batched frontier occupancy"};
    }
    blocks_per_wgp = options.blocks_per_wgp;
  }
  return CooperativeGrid{
      checked_grid_blocks(blocks_per_wgp, properties.wgp_count), active};
}

[[nodiscard]] std::uint32_t ordinary_grid_blocks(
    const DeviceProperties& properties,
    const GpuRunOptions& options,
    const std::uint32_t queue_capacity,
    std::uint32_t& active_blocks_per_wgp) {
  active_blocks_per_wgp = kernel_occupancy(
      reinterpret_cast<const void*>(batched_frontier_round_kernel),
      options.block_size);
  std::uint32_t blocks_per_wgp = active_blocks_per_wgp;
  if (options.grid_policy == GridPolicy::fixed_blocks_per_wgp) {
    if (options.blocks_per_wgp > active_blocks_per_wgp) {
      throw std::invalid_argument{
          "requested ordinary grid exceeds batched frontier occupancy"};
    }
    blocks_per_wgp = options.blocks_per_wgp;
  }
  const std::uint32_t resident =
      checked_grid_blocks(blocks_per_wgp, properties.wgp_count);
  const std::uint64_t work_blocks = std::max<std::uint64_t>(
      1U,
      (static_cast<std::uint64_t>(queue_capacity) + options.block_size - 1U) /
          options.block_size);
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(resident, work_blocks));
}

[[nodiscard]] std::uint32_t initialization_grid_blocks(
    const std::uint64_t work_items,
    const std::uint32_t block_size) {
  const std::uint64_t blocks = std::max<std::uint64_t>(
      1U, (work_items + block_size - 1U) / block_size);
  if (blocks > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{
        "batched frontier initialization grid exceeds 32 bits"};
  }
  return static_cast<std::uint32_t>(blocks);
}

void check_launch(const std::string_view name) {
  check(hipGetLastError(), name);
}

void launch_initialize(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchFrontierView& batch,
    DeviceBatchFrontierStatistics* const statistics,
    const GpuRunOptions& options,
    const hipStream_t stream) {
  hipLaunchKernelGGL(
      initialize_batched_frontier_storage_kernel,
      dim3(initialization_grid_blocks(
          batch.union_vertex_count, options.block_size)),
      dim3(options.block_size),
      0U,
      stream,
      graph,
      workspace,
      batch,
      options.maximum_rounds,
      options.enable_per_lane_convergence);
  check_launch("initialize_batched_frontier_storage_kernel launch");
  hipLaunchKernelGGL(
      seed_batched_frontier_sources_kernel,
      dim3(initialization_grid_blocks(
          workspace.source_count, options.block_size)),
      dim3(options.block_size),
      0U,
      stream,
      graph,
      workspace,
      batch,
      statistics,
      static_cast<std::uint32_t>(options.instrumentation));
  check_launch("seed_batched_frontier_sources_kernel launch");
  hipLaunchKernelGGL(
      finish_batched_frontier_seed_kernel,
      dim3(1U),
      dim3(1U),
      0U,
      stream,
      workspace);
  check_launch("finish_batched_frontier_seed_kernel launch");
}

void launch_round_pair(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchFrontierView& batch,
    DeviceBatchFrontierStatistics* const statistics,
    const std::uint32_t blocks,
    const GpuRunOptions& options,
    const hipStream_t stream) {
  const std::uint32_t instrumentation =
      static_cast<std::uint32_t>(options.instrumentation);
  hipLaunchKernelGGL(
      batched_frontier_round_kernel,
      dim3(blocks),
      dim3(options.block_size),
      0U,
      stream,
      graph,
      workspace,
      batch,
      statistics,
      instrumentation);
  check_launch("batched_frontier_round_kernel launch");
  hipLaunchKernelGGL(
      batched_frontier_advance_kernel,
      dim3(1U),
      dim3(1U),
      0U,
      stream,
      workspace,
      batch,
      statistics,
      instrumentation);
  check_launch("batched_frontier_advance_kernel launch");
}

void launch_finalize(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchFrontierView& batch,
    const hipStream_t stream) {
  hipLaunchKernelGGL(
      finalize_batched_frontier_status,
      dim3(1U),
      dim3(1U),
      0U,
      stream,
      graph,
      workspace,
      batch);
  check_launch("finalize_batched_frontier_status launch");
}

void launch_persistent(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceBatchFrontierView& batch,
    DeviceBatchFrontierStatistics* const statistics,
    const CooperativeGrid grid,
    const GpuRunOptions& options,
    const hipStream_t stream) {
  DeviceGraphView32 graph_argument = graph;
  DeviceWorkspaceView workspace_argument = workspace;
  DeviceBatchFrontierView batch_argument = batch;
  DeviceBatchFrontierStatistics* statistics_argument = statistics;
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
          reinterpret_cast<const void*>(batched_frontier_persistent_kernel),
          dim3(grid.blocks),
          dim3(options.block_size),
          arguments,
          0U,
          stream),
      "hipLaunchCooperativeKernel(batched frontier persistent)");
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
            "batched frontier selected lane vertices"),
        "batched frontier selected lane vertices");
  }
  return count;
}

[[nodiscard]] std::uint64_t union_vertex_count(
    const BatchDeviceDescription& batch) {
  std::uint64_t count = 0U;
  for (const BatchVertexRange range : batch.selected_vertex_ranges) {
    checked_add_u64(
        count,
        range.end - range.begin,
        "batched frontier union vertices");
  }
  return count;
}

[[nodiscard]] std::uint64_t distinct_source_vertex_count(
    const BatchDeviceDescription& batch) {
  std::vector<std::uint32_t> sources = batch.sources;
  std::sort(sources.begin(), sources.end());
  const auto end = std::unique(sources.begin(), sources.end());
  return static_cast<std::uint64_t>(std::distance(sources.begin(), end));
}

}  // namespace

std::size_t batched_frontier_scratch_bytes(
    const std::uint32_t vertex_count,
    const std::uint32_t lane_width,
    const std::uint32_t queue_capacity) {
  const BatchedFrontierScratchLayout layout =
      make_batched_frontier_scratch_layout(
          vertex_count, lane_width, queue_capacity);
  if (layout.total_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error{"batched frontier scratch bytes overflow"};
  }
  return static_cast<std::size_t>(layout.total_bytes);
}

class BatchedFrontierPushEngine::Impl final {
 public:
  Impl(
      const WeightedGraph& host_graph_value,
      const TileRunLayout64& tile_runs_value,
      const ResidentDeviceGraph& resident_graph_value,
      ReusableBatchedFrontierWorkspace& workspace_value,
      const HipStream& stream_value,
      const std::uint32_t requested_queue_capacity)
      : host_graph{host_graph_value},
        tile_runs{tile_runs_value},
        resident_graph{resident_graph_value},
        workspace{workspace_value},
        stream{stream_value} {
    if (!validate_weighted_graph(host_graph).ok() ||
        !host_graph.has_spatial_ordering()) {
      throw std::invalid_argument{
          "batched frontier requires a valid spatially ordered host graph"};
    }
    if (!validate_tile_run_layout(host_graph, tile_runs).ok()) {
      throw std::invalid_argument{
          "batched frontier requires deeply valid tile-run metadata"};
    }
    if (!resident_graph.has_upload()) {
      throw std::invalid_argument{
          "batched frontier requires an uploaded resident graph"};
    }
    const DeviceGraphView32& device = resident_graph.view();
    if (device.vertex_count != host_graph.vertex_count() ||
        device.edge_count != host_graph.edge_count() ||
        device.tile_count != host_graph.tile_coordinates().size() ||
        device.csr.run_count != tile_runs.csr_run_destination_tiles.size()) {
      throw std::invalid_argument{
          "batched frontier host graph/run shapes disagree with resident graph"};
    }
    if (resident_graph.fingerprint() !=
        fingerprint_device_graph_source32(host_graph, tile_runs)) {
      throw std::invalid_argument{
          "batched frontier host graph/run content disagrees with resident graph"};
    }
    if (requested_queue_capacity > host_graph.vertex_count()) {
      throw std::invalid_argument{
          "batched frontier queue capacity exceeds vertex count"};
    }
    queue_capacity =
        requested_queue_capacity == 0U
            ? static_cast<std::uint32_t>(host_graph.vertex_count())
            : requested_queue_capacity;
    if (queue_capacity == 0U) {
      throw std::invalid_argument{
          "batched frontier queue capacity must be nonzero"};
    }
  }

  const WeightedGraph& host_graph;
  const TileRunLayout64& tile_runs;
  const ResidentDeviceGraph& resident_graph;
  ReusableBatchedFrontierWorkspace& workspace;
  const HipStream& stream;
  std::uint32_t queue_capacity{};
  std::vector<std::uint32_t> downloaded_distance_bits;
  HipEventTimer preparation_gpu_timer;
  HipEventTimer sssp_gpu_timer;
  HipEventTimer result_transfer_gpu_timer;
};

BatchedFrontierPushEngine::BatchedFrontierPushEngine(
    const WeightedGraph& host_graph,
    const TileRunLayout64& tile_runs,
    const ResidentDeviceGraph& resident_graph,
    ReusableBatchedFrontierWorkspace& workspace,
    const HipStream& stream,
    const std::uint32_t queue_capacity)
    : impl_{new Impl{
          host_graph,
          tile_runs,
          resident_graph,
          workspace,
          stream,
          queue_capacity}} {}

BatchedFrontierPushEngine::~BatchedFrontierPushEngine() { delete impl_; }

BatchedFrontierRunOutput BatchedFrontierPushEngine::run_status_only(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options) {
  return run_impl(batch, options, false, false, nullptr, nullptr);
}

DeviceRunStatus BatchedFrontierPushEngine::run_compact_status(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options) {
  return run_impl(batch, options, false, true, nullptr, nullptr).result.status;
}

CompactPathBatchOutput BatchedFrontierPushEngine::run_compact_paths(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    ReusableCompactPathWorkspace& compact_workspace) {
  CompactPathBatchOutput compact_output;
  static_cast<void>(run_impl(
      batch,
      options,
      false,
      false,
      &compact_workspace,
      &compact_output));
  return compact_output;
}

BatchedFrontierRunOutput BatchedFrontierPushEngine::run_with_distances(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options) {
  return run_impl(batch, options, true, false, nullptr, nullptr);
}

BatchedFrontierRunOutput BatchedFrontierPushEngine::run_impl(
    const BatchDeviceDescription& batch,
    const GpuRunOptions& options,
    const bool download_distances,
    const bool compact_status_only,
    ReusableCompactPathWorkspace* const compact_workspace,
    CompactPathBatchOutput* const compact_output) {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot run an invalid batched frontier engine"};
  }
  if (validate_gpu_run_options(options) != GpuRunOptionsError::none ||
      options.engine != EngineKind::frontier_push) {
    throw std::invalid_argument{
        "batched frontier options are invalid or select another engine"};
  }
  if ((compact_workspace == nullptr) != (compact_output == nullptr) ||
      (compact_output != nullptr &&
       (download_distances || compact_status_only ||
        options.instrumentation != InstrumentationLevel::none))) {
    throw std::invalid_argument{
        "batched frontier compact-path mode requires paired workspace/output, "
        "None instrumentation, and no other readback mode"};
  }
  validate_batch_description_shape(impl_->host_graph, impl_->tile_runs, batch);

  const DeviceProperties properties = current_device_properties();
  if (options.block_size > properties.maximum_threads_per_block) {
    throw std::invalid_argument{
        "batched frontier block size exceeds the selected device limit"};
  }
  const DeviceGraphView32 graph = impl_->resident_graph.view();
  const std::size_t scratch_bytes = batched_frontier_scratch_bytes(
      graph.vertex_count, batch.lane_width, impl_->queue_capacity);
  const std::size_t distance_elements = checked_multiply(
      graph.vertex_count,
      batch.lane_width,
      "batched frontier distance elements");
  bool lease_active = false;

  SteadyClockTimer end_to_end_timer;
  BatchedFrontierRunOutput output;
  output.distances_downloaded = download_distances;
  output.result.engine_kind =
      static_cast<std::uint32_t>(EngineKind::frontier_push);
  output.result.control_mode =
      static_cast<std::uint32_t>(options.control_mode);
  output.metrics.lane_width = batch.lane_width;
  output.metrics.valid_lane_count = static_cast<std::uint32_t>(
      std::popcount(batch.valid_lane_mask));
  output.metrics.queue_capacity = impl_->queue_capacity;

  try {
    impl_->resident_graph.wait_until_ready(impl_->stream);
    impl_->preparation_gpu_timer.start(impl_->stream);
    impl_->workspace.prepare_async(
        batch,
        options,
        impl_->queue_capacity,
        scratch_bytes,
        impl_->stream);
    lease_active = true;
    impl_->preparation_gpu_timer.stop(impl_->stream);

    const DeviceWorkspaceView workspace_view =
        impl_->workspace.device_workspace_view();
    const DeviceBatchFrontierView batch_view =
        impl_->workspace.device_batch_view();
    DeviceBatchFrontierStatistics* const device_batch_statistics =
        impl_->workspace.device_batch_statistics();
    const hipStream_t native_stream = as_stream(impl_->stream.native_handle());
    const void* const selected_kernel =
        options.control_mode == ControlMode::persistent_cooperative
            ? reinterpret_cast<const void*>(batched_frontier_persistent_kernel)
            : reinterpret_cast<const void*>(batched_frontier_round_kernel);
    output.metrics.kernel_registers_per_thread =
        kernel_register_count(selected_kernel);

    impl_->sssp_gpu_timer.start(impl_->stream);
    std::uint64_t total_kernel_dispatches = 0U;
    if (options.control_mode != ControlMode::persistent_cooperative) {
      launch_initialize(
          graph,
          workspace_view,
          batch_view,
          device_batch_statistics,
          options,
          native_stream);
      total_kernel_dispatches = 3U;
    }

    DeviceController observed_controller{};
    if (options.control_mode == ControlMode::persistent_cooperative) {
      const CooperativeGrid grid = cooperative_grid(properties, options);
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
          native_stream);
      total_kernel_dispatches = 1U;
    } else {
      const std::uint32_t blocks = ordinary_grid_blocks(
          properties,
          options,
          impl_->queue_capacity,
          output.metrics.ordinary_active_blocks_per_wgp);
      output.metrics.ordinary_grid_blocks = blocks;
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
              blocks,
              options,
              native_stream);
          checked_add_u64(
              output.metrics.engine_round_dispatches,
              1U,
              "batched frontier round dispatches");
          checked_add_u64(
              output.metrics.controller_advance_dispatches,
              1U,
              "batched frontier advance dispatches");
          checked_add_u64(
              total_kernel_dispatches,
              2U,
              "batched frontier total dispatches");
        }
        impl_->workspace.download_controller_async(
            observed_controller, impl_->stream);
        impl_->stream.synchronize();
        checked_add_u64(
            output.metrics.convergence_host_checks,
            1U,
            "batched frontier host checks");
      } while (observed_controller.done == 0U);
      launch_finalize(graph, workspace_view, batch_view, native_stream);
      checked_add_u64(
          total_kernel_dispatches,
          1U,
          "batched frontier total dispatches");
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
            "batched frontier produced an invalid compact-path status"};
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
            "batched frontier produced an invalid compact terminal status"};
      }
      output.result.status = compact_status;
      output.metrics.end_to_end_wall_milliseconds =
          end_to_end_timer.elapsed_milliseconds();
      return output;
    }

    DeviceController final_controller{};
    DeviceRunStatus final_status{};
    DeviceWorkStatistics final_statistics{};
    DeviceBatchFrontierStatistics final_batch_statistics{};
    output.lane_convergence_rounds.resize(batch.lane_width);
    if (download_distances) {
      impl_->downloaded_distance_bits.resize(distance_elements);
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
          "batched frontier produced inconsistent terminal controller/status"};
    }
    output.final_controller = final_controller;

    if (download_distances) {
      output.distance_bits.assign(distance_elements, positive_infinity_bits);
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
      output.distances.reserve(distance_elements);
      for (const std::uint32_t bits : output.distance_bits) {
        output.distances.push_back(std::bit_cast<float>(bits));
      }
    }

    output.frontier_rounds_by_lane.assign(batch.lane_width, 0U);
    output.lane_tail_rounds.assign(batch.lane_width, 0U);
    for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
      const LaneMask bit = LaneMask{1U} << lane;
      if ((batch.valid_lane_mask & bit) == 0U) {
        output.lane_convergence_rounds[lane] = 0U;
        continue;
      }
      const std::uint64_t convergence =
          output.lane_convergence_rounds[lane];
      if (convergence > final_status.rounds_completed) {
        throw std::runtime_error{
            "batched frontier convergence round exceeds batch rounds"};
      }
      const std::uint64_t executed =
          convergence == 0U ? final_status.rounds_completed : convergence;
      output.frontier_rounds_by_lane[lane] = executed;
      checked_add_u64(
          output.batch_work.active_lane_rounds,
          executed,
          "batched frontier active lane rounds");
      if (convergence != 0U) {
        const std::uint64_t tail =
            final_status.rounds_completed - convergence;
        output.lane_tail_rounds[lane] = tail;
        checked_add_u64(
            output.batch_work.tail_lane_rounds,
            tail,
            "batched frontier tail lane rounds");
        checked_add_u64(
            output.batch_work.tail_lane_rounds_without_frontier_work,
            tail,
            "batched frontier tail rounds without work");
      }
    }

    output.batch_work.frontier_rounds = final_status.rounds_completed;
    if (final_status.stop_reason ==
        static_cast<std::uint32_t>(DeviceStopReason::converged)) {
      output.batch_work.empty_frontier_rounds = 1U;
    }
    if (options.instrumentation != InstrumentationLevel::none) {
      output.batch_work.frontier_vertex_entries =
          final_batch_statistics.worklist_vertices;
      output.batch_work.active_vertex_lane_pairs =
          final_batch_statistics.active_vertex_lanes;
      output.batch_work.multi_lane_frontier_vertex_entries =
          final_batch_statistics.multi_lane_worklist_vertices;
      output.batch_work.csr_runs_considered =
          final_batch_statistics.csr_runs_considered;
      output.batch_work.csr_runs_visited =
          final_batch_statistics.csr_nonzero_runs_visited;
      output.batch_work.csr_runs_skipped =
          final_batch_statistics.csr_runs_skipped;
      output.batch_work.active_lanes_over_visited_runs =
          final_batch_statistics.active_lanes_across_nonzero_runs;
      output.batch_work.csr_edge_loads =
          final_batch_statistics.csr_edge_records_loaded;
      output.batch_work.multi_lane_csr_edge_loads =
          final_batch_statistics.multi_lane_csr_edge_records;
      output.batch_work.lane_edge_relaxations =
          final_batch_statistics.admitted_lane_edge_pairs;
      output.batch_work.distance_atomic_source_loads =
          final_batch_statistics.admitted_lane_edge_pairs;
      output.batch_work.distance_atomic_attempts =
          final_batch_statistics.admitted_lane_edge_pairs;
      output.batch_work.successful_distance_atomic_updates =
          final_batch_statistics.distance_atomic_successes;
      output.batch_work.maximum_queue_size =
          final_batch_statistics.maximum_queue_size;
      output.batch_work.small_frontier_rounds =
          final_batch_statistics.small_frontier_rounds;

      if (output.batch_work.active_vertex_lane_pairs <
              output.batch_work.frontier_vertex_entries ||
          output.batch_work.lane_edge_relaxations <
              output.batch_work.csr_edge_loads) {
        throw std::runtime_error{
            "batched frontier sharing counters are inconsistent"};
      }
      output.batch_work.shared_vertex_entries_saved =
          output.batch_work.active_vertex_lane_pairs -
          output.batch_work.frontier_vertex_entries;
      output.batch_work.shared_edge_lane_work_saved =
          output.batch_work.lane_edge_relaxations -
          output.batch_work.csr_edge_loads;

      if (final_batch_statistics.frontier_rounds !=
              final_status.rounds_completed ||
          final_batch_statistics.empty_frontier_rounds !=
              output.batch_work.empty_frontier_rounds ||
          final_statistics.edges_examined !=
              output.batch_work.lane_edge_relaxations ||
          final_statistics.successful_decreases !=
              output.batch_work.successful_distance_atomic_updates ||
          final_statistics.active_vertices !=
              output.batch_work.frontier_vertex_entries ||
          final_statistics.active_lane_rounds !=
              output.batch_work.active_lane_rounds ||
          final_statistics.maximum_queue_size !=
              output.batch_work.maximum_queue_size ||
          final_statistics.empty_frontier_rounds !=
              output.batch_work.empty_frontier_rounds ||
          final_statistics.small_frontier_rounds !=
              output.batch_work.small_frontier_rounds) {
        throw std::runtime_error{
            "batched frontier Light/common counters disagree"};
      }

      if (options.instrumentation == InstrumentationLevel::debug) {
        output.batch_work.current_mask_atomic_exchanges =
            final_batch_statistics.current_mask_atomic_exchanges;
        output.batch_work.next_mask_atomic_ors =
            final_batch_statistics.next_mask_atomic_ors;
        output.batch_work.controller_mask_atomic_ors =
            final_batch_statistics.controller_mask_atomic_ors;
        output.batch_work.unique_next_vertex_lane_activations =
            final_batch_statistics.lane_enqueue_transitions;
        output.batch_work.queue_claims =
            final_batch_statistics.queue_claims;
        output.batch_work.queue_entries_saved_by_lane_merging =
            final_batch_statistics.queue_entries_saved_by_lane_merging;
        output.batch_work.same_lane_duplicate_suppressions =
            final_batch_statistics.same_lane_duplicate_suppressions;
        output.batch_work.duplicate_suppressions =
            final_batch_statistics.duplicate_suppressions;
        output.batch_work.overflow_events =
            final_batch_statistics.overflow_events;
        if (final_batch_statistics.distance_atomic_source_loads !=
                output.batch_work.lane_edge_relaxations ||
            final_batch_statistics.distance_atomic_attempts !=
                output.batch_work.lane_edge_relaxations ||
            final_batch_statistics.distance_atomic_successes !=
                output.batch_work.successful_distance_atomic_updates ||
            output.batch_work.unique_next_vertex_lane_activations <
                output.batch_work.queue_claims ||
            output.batch_work.successful_distance_atomic_updates <
                output.batch_work.unique_next_vertex_lane_activations ||
            output.batch_work.queue_entries_saved_by_lane_merging !=
                output.batch_work.unique_next_vertex_lane_activations -
                    output.batch_work.queue_claims ||
            output.batch_work.same_lane_duplicate_suppressions !=
                output.batch_work.successful_distance_atomic_updates -
                    output.batch_work.unique_next_vertex_lane_activations ||
            output.batch_work.duplicate_suppressions !=
                output.batch_work.successful_distance_atomic_updates -
                    output.batch_work.queue_claims ||
            final_statistics.atomic_attempts !=
                output.batch_work.distance_atomic_attempts ||
            final_statistics.successful_atomic_updates !=
                output.batch_work.successful_distance_atomic_updates ||
            final_statistics.queue_claims !=
                output.batch_work.queue_claims ||
            final_statistics.duplicate_suppressions !=
                output.batch_work.duplicate_suppressions ||
            final_statistics.overflow_events !=
                output.batch_work.overflow_events) {
          throw std::runtime_error{
              "batched frontier Debug counters disagree"};
        }
      }
    }

    const std::uint64_t initial_queue_entries =
        distinct_source_vertex_count(batch);
    output.batch_work.initial_source_lane_activations = batch.sources.size();
    output.batch_work.initial_queue_entries = initial_queue_entries;
    if (output.batch_work.initial_source_lane_activations <
        initial_queue_entries) {
      throw std::runtime_error{
          "batched frontier source queue accounting is inconsistent"};
    }
    output.batch_work.initial_queue_entries_saved_by_lane_merging =
        output.batch_work.initial_source_lane_activations -
        initial_queue_entries;

    const std::uint64_t valid_lanes =
        static_cast<std::uint64_t>(std::popcount(batch.valid_lane_mask));
    output.batch_work.valid_lane_round_capacity = checked_multiply_u64(
        final_status.rounds_completed,
        valid_lanes,
        "batched frontier valid lane-round capacity");
    output.batch_work.lane_width_round_capacity = checked_multiply_u64(
        final_status.rounds_completed,
        batch.lane_width,
        "batched frontier configured lane-round capacity");
    output.batch_work.wave32_lane_round_capacity = checked_multiply_u64(
        final_status.rounds_completed,
        32U,
        "batched frontier wave32 lane-round capacity");
    output.batch_work.unused_wave_lane_round_capacity =
        output.batch_work.wave32_lane_round_capacity -
        output.batch_work.lane_width_round_capacity;
    output.batch_work.padded_lane_round_capacity = checked_multiply_u64(
        final_status.rounds_completed,
        static_cast<std::uint64_t>(batch.lane_width) - valid_lanes,
        "batched frontier padded lane-round capacity");
    if (output.batch_work.active_lane_rounds >
        output.batch_work.valid_lane_round_capacity) {
      throw std::runtime_error{
          "batched frontier active lane rounds exceed valid capacity"};
    }
    output.batch_work.inactive_valid_lane_rounds =
        output.batch_work.valid_lane_round_capacity -
        output.batch_work.active_lane_rounds;
    output.batch_work.semantic_lane_edge_work_avoided_by_per_lane_convergence =
        0U;

    const std::uint64_t union_vertices = union_vertex_count(batch);
    output.batch_work.distance_reset_bytes = checked_multiply_u64(
        selected_lane_vertex_count(batch),
        sizeof(std::uint32_t),
        "batched frontier distance reset bytes");
    output.batch_work.activity_mask_reset_bytes = checked_multiply_u64(
        union_vertices,
        2U * sizeof(LaneMask),
        "batched frontier activity-mask reset bytes");
    output.batch_work.source_seed_write_bytes = checked_multiply_u64(
        batch.sources.size(),
        sizeof(std::uint32_t),
        "batched frontier source seed bytes");
    output.batch_work.frontier_queue_storage_bytes = checked_multiply_u64(
        impl_->queue_capacity,
        2U * sizeof(std::uint32_t),
        "batched frontier queue storage bytes");
    output.batch_work.union_tile_lane_positions = checked_multiply_u64(
        batch.union_tiles.size(),
        valid_lanes,
        "batched frontier union tile/lane positions");
    for (const std::uint32_t tile : batch.union_tiles) {
      checked_add_u64(
          output.batch_work.selected_tile_lane_positions,
          static_cast<std::uint64_t>(
              std::popcount(batch.tile_lane_masks[tile])),
          "batched frontier selected tile/lane positions");
    }

    output.metrics.edge_record_read_bytes_requested = checked_multiply_u64(
        output.batch_work.csr_edge_loads,
        sizeof(std::uint32_t) + sizeof(float),
        "batched frontier edge request bytes");
    output.metrics.distance_atomic_read_bytes_requested = checked_multiply_u64(
        output.batch_work.distance_atomic_source_loads,
        sizeof(std::uint32_t),
        "batched frontier distance read request bytes");
    output.metrics.distance_atomic_min_bytes_requested = checked_multiply_u64(
        output.batch_work.distance_atomic_attempts,
        sizeof(std::uint32_t),
        "batched frontier distance min request bytes");
    std::uint64_t mask_atomic_operations =
        output.batch_work.current_mask_atomic_exchanges;
    checked_add_u64(
        mask_atomic_operations,
        output.batch_work.next_mask_atomic_ors,
        "batched frontier activity-mask operations");
    checked_add_u64(
        mask_atomic_operations,
        output.batch_work.controller_mask_atomic_ors,
        "batched frontier activity-mask operations");
    output.metrics.activity_mask_atomic_bytes_requested = checked_multiply_u64(
        mask_atomic_operations,
        sizeof(LaneMask),
        "batched frontier activity-mask request bytes");
    if (output.batch_work.frontier_vertex_entries != 0U) {
      output.metrics.average_active_lanes_per_frontier_vertex =
          static_cast<double>(output.batch_work.active_vertex_lane_pairs) /
          static_cast<double>(output.batch_work.frontier_vertex_entries);
      output.metrics.frontier_lane_utilization =
          static_cast<double>(output.batch_work.active_vertex_lane_pairs) /
          static_cast<double>(checked_multiply_u64(
              output.batch_work.frontier_vertex_entries,
              32U,
              "batched frontier vertex wave32 capacity"));
    }
    if (output.batch_work.csr_runs_visited != 0U) {
      output.metrics.average_active_lanes_per_nonzero_run =
          static_cast<double>(
              output.batch_work.active_lanes_over_visited_runs) /
          static_cast<double>(output.batch_work.csr_runs_visited);
    }
    if (output.batch_work.wave32_lane_round_capacity != 0U) {
      output.metrics.lane_round_utilization =
          static_cast<double>(output.batch_work.active_lane_rounds) /
          static_cast<double>(
              output.batch_work.wave32_lane_round_capacity);
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
        "batched frontier host synchronization count");
    final_statistics.controller_copies =
        output.metrics.convergence_host_checks;
    checked_add_u64(
        final_statistics.controller_copies,
        2U,
        "batched frontier controller copy count");
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
