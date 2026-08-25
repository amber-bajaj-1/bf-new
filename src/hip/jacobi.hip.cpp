#include "bfnew/hip/jacobi.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bfnew::hip {
namespace {

namespace cg = ::cooperative_groups;

constexpr LaneMask standalone_lane = LaneMask{1U};

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
    const std::string_view name) {
  if (right != 0U && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error{std::string{name} + " overflow"};
  }
  return left * right;
}

[[nodiscard]] std::uint32_t checked_grid_blocks(
    const std::uint32_t blocks_per_wgp,
    const std::uint32_t wgp_count) {
  const std::uint64_t blocks =
      static_cast<std::uint64_t>(blocks_per_wgp) * wgp_count;
  if (blocks == 0U || blocks > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{"Jacobi grid block count exceeds the HIP launch ABI"};
  }
  return static_cast<std::uint32_t>(blocks);
}

struct DeviceProperties {
  std::uint32_t wgp_count{};
  std::uint32_t maximum_threads_per_block{};
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
  if (properties.multiProcessorCount <= 0 || properties.maxThreadsPerBlock <= 0) {
    throw std::runtime_error{"HIP returned invalid device scheduling limits"};
  }
  return DeviceProperties{
      static_cast<std::uint32_t>(properties.multiProcessorCount),
      static_cast<std::uint32_t>(properties.maxThreadsPerBlock),
      cooperative != 0,
  };
}

struct ThreadRoundStatistics {
  unsigned long long edges_examined{};
  unsigned long long successful_decreases{};
  unsigned long long active_vertices{};
  unsigned long long mask_operations{};
  unsigned int changed{};
};

__device__ void atomic_add_u64(
    std::uint64_t* const destination,
    const unsigned long long value) {
  atomicAdd(
      reinterpret_cast<unsigned long long*>(destination), value);
}

__device__ float* distance_slot(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const std::uint32_t slot) {
  auto* const base = reinterpret_cast<float*>(workspace.engine_scratch);
  return base + static_cast<std::uint64_t>(slot) * graph.vertex_count;
}

__device__ void advance_controller_and_count(
    DeviceController* const controller,
    DeviceWorkStatistics* const statistics) {
  const JacobiAdvanceResult transition =
      bfnew::advance_jacobi_controller(*controller);
  if (statistics != nullptr &&
      transition != JacobiAdvanceResult::no_op &&
      transition != JacobiAdvanceResult::invalid_controller_state) {
    ++statistics->active_lane_rounds;
  }
}

template <bool collect_statistics>
__device__ void perform_jacobi_round_thread_work(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const std::uint32_t instrumentation_level,
    ThreadRoundStatistics& output) {
  ThreadRoundStatistics local{};
  if constexpr (!collect_statistics) {
    static_cast<void>(instrumentation_level);
  }
  // Read only fields that are stable for the complete round. Other blocks
  // atomically OR changed_lane_mask below, so copying the whole controller
  // here would mix an ordinary read with those concurrent atomic updates.
  const std::uint32_t done = workspace.controller->done;
  const LaneMask execute_lane_mask = workspace.controller->execute_lane_mask;
  const std::uint32_t read_slot = workspace.controller->distance_read_slot;
  const std::uint32_t write_slot = workspace.controller->distance_write_slot;
  const bool execute = done == 0U &&
                       (execute_lane_mask & standalone_lane) != 0U;

  if (execute) {
    const JacobiDistanceView distances = bind_jacobi_distance_slots(
        workspace.engine_scratch,
        workspace.engine_scratch_bytes,
        graph.vertex_count,
        read_slot,
        write_slot);
    if (!valid_jacobi_distance_view(distances)) {
      if (blockIdx.x == 0U && threadIdx.x == 0U) {
        // Publish only an atomic error indication during the round. The
        // separate controller owner converts it into the terminal state after
        // every block has completed, avoiding a read/write race on done/slots.
        atomicOr(
            &workspace.controller->error_bits,
            device_error::invalid_controller_state);
      }
    } else {
      const std::uint64_t global_thread =
          static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      const std::uint64_t grid_threads =
          static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

      for (std::uint64_t destination = global_thread;
           destination < graph.vertex_count;
           destination += grid_threads) {
        const std::uint32_t vertex = static_cast<std::uint32_t>(destination);
        const std::uint32_t destination_tile = graph.owner_tiles[vertex];
        if ((workspace.tile_lane_masks[destination_tile] & standalone_lane) == 0U) {
          continue;
        }

        if constexpr (collect_statistics) {
          ++local.active_vertices;
        }
        const float preceding = distances.read_distances[vertex];
        float best = preceding;
        const std::uint32_t run_begin = graph.csc.column_run_offsets[vertex];
        const std::uint32_t run_end = graph.csc.column_run_offsets[vertex + 1U];
        for (std::uint32_t run = run_begin; run < run_end; ++run) {
          if constexpr (collect_statistics) {
            if (instrumentation_level ==
                static_cast<std::uint32_t>(InstrumentationLevel::debug)) {
              ++local.mask_operations;
            }
          }
          if ((workspace.run_lane_masks[run] & standalone_lane) == 0U) {
            continue;
          }
          const std::uint32_t edge_begin = graph.csc.run_edge_offsets[run];
          const std::uint32_t edge_end = graph.csc.run_edge_offsets[run + 1U];
          if constexpr (collect_statistics) {
            local.edges_examined +=
                static_cast<unsigned long long>(edge_end - edge_begin);
          }
          for (std::uint32_t edge = edge_begin; edge < edge_end; ++edge) {
            const std::uint32_t source = graph.csc.sources[edge];
            const float candidate =
                distances.read_distances[source] + graph.csc.weights[edge];
            if (candidate < best) {
              best = candidate;
            }
          }
        }
        distances.write_distances[vertex] = best;
        if (best < preceding) {
          local.changed = 1U;
          if constexpr (collect_statistics) {
            ++local.successful_decreases;
          }
        }
      }
    }
  }

  output = local;
}

__device__ void perform_jacobi_round_none(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    unsigned int* const block_changes) {
  ThreadRoundStatistics local{};
  perform_jacobi_round_thread_work<false>(
      graph,
      workspace,
      static_cast<std::uint32_t>(InstrumentationLevel::none),
      local);
  block_changes[threadIdx.x] = local.changed;
  __syncthreads();

  if (threadIdx.x == 0U) {
    unsigned int changed = 0U;
    for (std::uint32_t thread = 0U; thread < blockDim.x; ++thread) {
      changed |= block_changes[thread];
    }
    if (changed != 0U) {
      atomicOr(&workspace.controller->changed_lane_mask, standalone_lane);
    }
  }
}

__device__ void perform_jacobi_round_instrumented(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    ThreadRoundStatistics* const block_statistics,
    const std::uint32_t instrumentation_level) {
  ThreadRoundStatistics local{};
  perform_jacobi_round_thread_work<true>(
      graph, workspace, instrumentation_level, local);
  block_statistics[threadIdx.x] = local;
  __syncthreads();

  // Exactly one global changed update per block.  This avoids a device-wide
  // atomic for every useful destination while preserving strict float logic.
  if (threadIdx.x == 0U) {
    ThreadRoundStatistics reduced{};
    for (std::uint32_t thread = 0U; thread < blockDim.x; ++thread) {
      reduced.edges_examined += block_statistics[thread].edges_examined;
      reduced.successful_decreases +=
          block_statistics[thread].successful_decreases;
      reduced.active_vertices += block_statistics[thread].active_vertices;
      reduced.mask_operations += block_statistics[thread].mask_operations;
      reduced.changed |= block_statistics[thread].changed;
    }
    if (reduced.changed != 0U) {
      atomicOr(&workspace.controller->changed_lane_mask, standalone_lane);
    }
    if (workspace.instrumentation != nullptr) {
      atomic_add_u64(
          &workspace.instrumentation->edges_examined, reduced.edges_examined);
      atomic_add_u64(
          &workspace.instrumentation->successful_decreases,
          reduced.successful_decreases);
      atomic_add_u64(
          &workspace.instrumentation->active_vertices, reduced.active_vertices);
      atomic_add_u64(
          &workspace.instrumentation->mask_operations, reduced.mask_operations);
    }
  }
}

__device__ void initialize_jacobi_state(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence) {
  const std::uint64_t global_thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t grid_threads =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  float* const distance_zero = distance_slot(graph, workspace, 0U);
  float* const distance_one = distance_slot(graph, workspace, 1U);

  for (std::uint64_t vertex = global_thread; vertex < graph.vertex_count;
       vertex += grid_threads) {
    bool source = false;
    for (std::uint32_t index = 0U; index < workspace.source_count; ++index) {
      source = source || workspace.sources[index] == vertex;
    }
    const float initial =
        source ? 0.0F : std::numeric_limits<float>::infinity();
    distance_zero[vertex] = initial;
    distance_one[vertex] = initial;

    // Exactly one owner writes every immutable CSC run for this destination.
    // The run mask is the same endpoint-tile intersection proved in Phase 8.
    const std::uint32_t vertex32 = static_cast<std::uint32_t>(vertex);
    const std::uint32_t destination_tile = graph.owner_tiles[vertex32];
    const LaneMask destination_mask =
        workspace.tile_lane_masks[destination_tile];
    const std::uint32_t run_begin =
        graph.csc.column_run_offsets[vertex32];
    const std::uint32_t run_end =
        graph.csc.column_run_offsets[vertex32 + 1U];
    for (std::uint32_t run = run_begin; run < run_end; ++run) {
      const std::uint32_t source_tile = graph.csc.run_source_tiles[run];
      workspace.run_lane_masks[run] =
          destination_mask & workspace.tile_lane_masks[source_tile] &
          standalone_lane;
    }
  }

  if (global_thread == 0U) {
    DeviceController* const controller = workspace.controller;
    controller->valid_lane_mask = standalone_lane;
    controller->active_lane_mask = standalone_lane;
    controller->changed_lane_mask = 0U;
    controller->converged_lane_mask = 0U;
    controller->execute_lane_mask = standalone_lane;
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

__global__ void initialize_jacobi_query(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence) {
  initialize_jacobi_state(
      graph, workspace, maximum_rounds, enable_per_lane_convergence);
}

__global__ void jacobi_round_none_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint32_t instrumentation_level) {
  static_cast<void>(instrumentation_level);
  extern __shared__ unsigned int block_changes[];
  perform_jacobi_round_none(graph, workspace, block_changes);
}

__global__ void jacobi_round_instrumented_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint32_t instrumentation_level) {
  extern __shared__ ThreadRoundStatistics block_statistics[];
  perform_jacobi_round_instrumented(
      graph, workspace, block_statistics, instrumentation_level);
}

__global__ void jacobi_advance_kernel(const DeviceWorkspaceView workspace) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    advance_controller_and_count(
        workspace.controller, workspace.instrumentation);
  }
}

[[nodiscard]] __device__ DeviceRunStatus completed_jacobi_status(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace) {
  const DeviceController controller = *workspace.controller;
  const bool converged =
      controller.stop_reason ==
      static_cast<std::uint32_t>(DeviceStopReason::converged);
  bool reached_all_targets = converged &&
                             controller.distance_read_slot <= 1U;
  if (reached_all_targets) {
    const float* const distances =
        distance_slot(graph, workspace, controller.distance_read_slot);
    for (std::uint32_t index = 0U; index < workspace.target_count; ++index) {
      reached_all_targets = reached_all_targets &&
                            isfinite(distances[workspace.targets[index]]);
    }
  }
  const LaneMask reached =
      converged && reached_all_targets ? standalone_lane : 0U;
  const LaneMask missed =
      converged && !reached_all_targets ? standalone_lane : 0U;
  return make_jacobi_run_status(controller, reached, missed);
}

template <bool collect_statistics>
__device__ void execute_jacobi_persistent(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level,
    void* const block_shared) {
  cg::grid_group grid = cg::this_grid();

  // Persistent mode is one cooperative query kernel.  Every workgroup enters
  // both initialization barriers; distance seeding, immutable-run admission,
  // and controller reset therefore complete before the first round begins.
  grid.sync();
  initialize_jacobi_state(
      graph, workspace, maximum_rounds, enable_per_lane_convergence);
  grid.sync();

  for (;;) {
    if constexpr (collect_statistics) {
      perform_jacobi_round_instrumented(
          graph,
          workspace,
          static_cast<ThreadRoundStatistics*>(block_shared),
          instrumentation_level);
    } else {
      perform_jacobi_round_none(
          graph, workspace, static_cast<unsigned int*>(block_shared));
    }
    grid.sync();

    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      advance_controller_and_count(
          workspace.controller, workspace.instrumentation);
    }
    // Every workgroup reaches the post-advance barrier, including the round
    // that sets done.  Only after this uniform barrier may the grid return.
    grid.sync();
    if (workspace.controller->done != 0U) {
      break;
    }
  }

  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *workspace.status = completed_jacobi_status(graph, workspace);
  }
  grid.sync();
}

__global__ void jacobi_persistent_none_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  extern __shared__ unsigned int block_changes[];
  execute_jacobi_persistent<false>(
      graph,
      workspace,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level,
      block_changes);
}

__global__ void jacobi_persistent_instrumented_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  extern __shared__ ThreadRoundStatistics block_statistics[];
  execute_jacobi_persistent<true>(
      graph,
      workspace,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level,
      block_statistics);
}

__global__ void finalize_jacobi_status(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  *workspace.status = completed_jacobi_status(graph, workspace);
}

[[nodiscard]] std::uint32_t initialize_grid_blocks(
    const std::uint32_t vertex_count,
    const std::uint32_t block_size) {
  const std::uint64_t blocks =
      std::max<std::uint64_t>(
          1U,
          (static_cast<std::uint64_t>(vertex_count) + block_size - 1U) /
              block_size);
  if (blocks > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{"Jacobi initialization grid exceeds 32 bits"};
  }
  return static_cast<std::uint32_t>(blocks);
}

[[nodiscard]] int occupancy_for_round_kernel(
    const std::uint32_t block_size,
    const std::size_t shared_bytes,
    const InstrumentationLevel instrumentation) {
  int active = 0;
  if (instrumentation == InstrumentationLevel::none) {
    check(
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &active,
            jacobi_round_none_kernel,
            static_cast<int>(block_size),
            shared_bytes),
        "hipOccupancyMaxActiveBlocksPerMultiprocessor(Jacobi None round)");
  } else {
    check(
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &active,
            jacobi_round_instrumented_kernel,
            static_cast<int>(block_size),
            shared_bytes),
        "hipOccupancyMaxActiveBlocksPerMultiprocessor(Jacobi instrumented round)");
  }
  if (active <= 0) {
    throw std::runtime_error{"Jacobi ordinary kernel has zero reported occupancy"};
  }
  return active;
}

[[nodiscard]] int occupancy_for_persistent_kernel(
    const std::uint32_t block_size,
    const std::size_t shared_bytes,
    const InstrumentationLevel instrumentation) {
  int active = 0;
  if (instrumentation == InstrumentationLevel::none) {
    check(
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &active,
            jacobi_persistent_none_kernel,
            static_cast<int>(block_size),
            shared_bytes),
        "hipOccupancyMaxActiveBlocksPerMultiprocessor(Jacobi None persistent)");
  } else {
    check(
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &active,
            jacobi_persistent_instrumented_kernel,
            static_cast<int>(block_size),
            shared_bytes),
        "hipOccupancyMaxActiveBlocksPerMultiprocessor(Jacobi instrumented persistent)");
  }
  if (active <= 0) {
    throw std::runtime_error{"Jacobi persistent kernel has zero reported occupancy"};
  }
  return active;
}

[[nodiscard]] std::uint32_t ordinary_grid_blocks(
    const DeviceProperties& properties,
    const GpuRunOptions& options,
    const std::uint32_t vertex_count,
    const std::size_t shared_bytes) {
  const int occupancy = occupancy_for_round_kernel(
      options.block_size, shared_bytes, options.instrumentation);
  std::uint32_t blocks_per_wgp = static_cast<std::uint32_t>(occupancy);
  if (options.grid_policy == GridPolicy::fixed_blocks_per_wgp) {
    if (options.blocks_per_wgp > blocks_per_wgp) {
      throw std::invalid_argument{
          "requested Jacobi ordinary grid exceeds real-kernel occupancy"};
    }
    blocks_per_wgp = options.blocks_per_wgp;
  }
  const std::uint32_t resident_grid =
      checked_grid_blocks(blocks_per_wgp, properties.wgp_count);
  const std::uint64_t work_blocks =
      std::max<std::uint64_t>(
          1U,
          (static_cast<std::uint64_t>(vertex_count) + options.block_size - 1U) /
              options.block_size);
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(resident_grid, work_blocks));
}

struct CooperativeGrid {
  std::uint32_t blocks{};
  std::uint32_t active_blocks_per_wgp{};
};

[[nodiscard]] CooperativeGrid cooperative_grid(
    const DeviceProperties& properties,
    const GpuRunOptions& options,
    const std::size_t shared_bytes) {
  if (!properties.cooperative_launch) {
    throw std::runtime_error{"selected HIP device does not support cooperative launch"};
  }
  const int occupancy =
      occupancy_for_persistent_kernel(
          options.block_size, shared_bytes, options.instrumentation);
  const std::uint32_t active = static_cast<std::uint32_t>(occupancy);
  std::uint32_t blocks_per_wgp = active;
  if (options.grid_policy == GridPolicy::fixed_blocks_per_wgp) {
    if (options.blocks_per_wgp > active) {
      throw std::invalid_argument{
          "requested cooperative grid exceeds real Jacobi kernel occupancy"};
    }
    blocks_per_wgp = options.blocks_per_wgp;
  }
  return CooperativeGrid{
      checked_grid_blocks(blocks_per_wgp, properties.wgp_count), active};
}

void check_launch(const std::string_view kernel) {
  check(hipGetLastError(), kernel);
}

void launch_initialize(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const GpuRunOptions& options,
    const hipStream_t stream) {
  const std::uint32_t blocks =
      initialize_grid_blocks(graph.vertex_count, options.block_size);
  hipLaunchKernelGGL(
      initialize_jacobi_query,
      dim3(blocks),
      dim3(options.block_size),
      0U,
      stream,
      graph,
      workspace,
      options.maximum_rounds,
      options.enable_per_lane_convergence);
  check_launch("initialize_jacobi_query launch");
}

void launch_round_pair(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const std::uint32_t blocks,
    const std::uint32_t block_size,
    const std::size_t shared_bytes,
    const InstrumentationLevel instrumentation,
    const hipStream_t stream) {
  if (instrumentation == InstrumentationLevel::none) {
    hipLaunchKernelGGL(
        jacobi_round_none_kernel,
        dim3(blocks),
        dim3(block_size),
        shared_bytes,
        stream,
        graph,
        workspace,
        static_cast<std::uint32_t>(instrumentation));
    check_launch("jacobi_round_none_kernel launch");
  } else {
    hipLaunchKernelGGL(
        jacobi_round_instrumented_kernel,
        dim3(blocks),
        dim3(block_size),
        shared_bytes,
        stream,
        graph,
        workspace,
        static_cast<std::uint32_t>(instrumentation));
    check_launch("jacobi_round_instrumented_kernel launch");
  }
  hipLaunchKernelGGL(
      jacobi_advance_kernel,
      dim3(1U),
      dim3(1U),
      0U,
      stream,
      workspace);
  check_launch("jacobi_advance_kernel launch");
}

void launch_persistent(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const CooperativeGrid grid,
    const std::uint32_t block_size,
    const std::size_t shared_bytes,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const InstrumentationLevel instrumentation,
    const hipStream_t stream) {
  DeviceGraphView32 graph_argument = graph;
  DeviceWorkspaceView workspace_argument = workspace;
  std::uint64_t maximum_rounds_argument = maximum_rounds;
  std::uint32_t convergence_argument = enable_per_lane_convergence;
  std::uint32_t instrumentation_argument =
      static_cast<std::uint32_t>(instrumentation);
  void* arguments[] = {
      &graph_argument,
      &workspace_argument,
      &maximum_rounds_argument,
      &convergence_argument,
      &instrumentation_argument,
  };
  const void* const kernel =
      instrumentation == InstrumentationLevel::none
          ? reinterpret_cast<const void*>(jacobi_persistent_none_kernel)
          : reinterpret_cast<const void*>(
                jacobi_persistent_instrumented_kernel);
  check(
      hipLaunchCooperativeKernel(
          kernel,
          dim3(grid.blocks),
          dim3(block_size),
          arguments,
          shared_bytes,
          stream),
      instrumentation == InstrumentationLevel::none
          ? "hipLaunchCooperativeKernel(Jacobi None persistent)"
          : "hipLaunchCooperativeKernel(Jacobi instrumented persistent)");
}

void launch_finalize(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const hipStream_t stream) {
  hipLaunchKernelGGL(
      finalize_jacobi_status,
      dim3(1U),
      dim3(1U),
      0U,
      stream,
      graph,
      workspace);
  check_launch("finalize_jacobi_status launch");
}

}  // namespace

std::size_t jacobi_distance_scratch_bytes(const std::uint32_t vertex_count) {
  const JacobiScratchLayout layout =
      make_jacobi_scratch_layout(vertex_count);
  if (layout.total_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error{"Jacobi distance scratch bytes overflow"};
  }
  return static_cast<std::size_t>(layout.total_bytes);
}

namespace {

[[nodiscard]] std::size_t selected_vertex_count(
    const WeightedGraph& graph,
    const RouteQuery& query) {
  const auto offsets = graph.tile_vertex_offsets();
  std::size_t count = 0U;
  for (const TileId tile : query.selected_tiles) {
    count += static_cast<std::size_t>(
        offsets[tile.value() + 1U] - offsets[tile.value()]);
  }
  return count;
}

}  // namespace

class JacobiPullEngine::Impl final {
 public:
  Impl(
      const WeightedGraph& host_graph_value,
      const TileRunLayout64& tile_runs_value,
      const ResidentDeviceGraph& resident_graph_value,
      ReusableDeviceWorkspace& workspace_value,
      const HipStream& stream_value)
      : host_graph{host_graph_value},
        tile_runs{tile_runs_value},
        resident_graph{resident_graph_value},
        workspace{workspace_value},
        stream{stream_value} {
    if (!validate_weighted_graph(host_graph).ok() ||
        !host_graph.has_spatial_ordering()) {
      throw std::invalid_argument{
          "Jacobi requires a valid spatially ordered host graph"};
    }
    if (!validate_tile_run_layout(host_graph, tile_runs).ok()) {
      throw std::invalid_argument{"Jacobi requires deeply valid tile-run metadata"};
    }
    if (!resident_graph.has_upload()) {
      throw std::invalid_argument{"Jacobi requires an uploaded resident graph"};
    }
    const DeviceGraphView32& device = resident_graph.view();
    if (device.vertex_count != host_graph.vertex_count() ||
        device.edge_count != host_graph.edge_count() ||
        device.tile_count != host_graph.tile_coordinates().size() ||
        device.csc.run_count != tile_runs.csc_run_source_tiles.size()) {
      throw std::invalid_argument{
          "Jacobi host graph/run shapes disagree with the resident graph"};
    }
    if (resident_graph.fingerprint() !=
        fingerprint_device_graph_source32(host_graph, tile_runs)) {
      throw std::invalid_argument{
          "Jacobi host graph/run content disagrees with the resident graph"};
    }
  }

  const WeightedGraph& host_graph;
  const TileRunLayout64& tile_runs;
  const ResidentDeviceGraph& resident_graph;
  ReusableDeviceWorkspace& workspace;
  const HipStream& stream;

  std::vector<LaneMask> tile_lane_masks;
  std::vector<LaneMask> empty_csc_run_masks;
  std::vector<float> downloaded_distance_slots;
  HipEventTimer preparation_gpu_timer;
  HipEventTimer sssp_gpu_timer;
  HipEventTimer result_transfer_gpu_timer;
};

JacobiPullEngine::JacobiPullEngine(
    const WeightedGraph& host_graph,
    const TileRunLayout64& tile_runs,
    const ResidentDeviceGraph& resident_graph,
    ReusableDeviceWorkspace& workspace,
    const HipStream& stream)
    : impl_{new Impl{
          host_graph, tile_runs, resident_graph, workspace, stream}} {}

JacobiPullEngine::~JacobiPullEngine() { delete impl_; }

GpuSsspResult JacobiPullEngine::run(
    const RouteQuery& query,
    const GpuRunOptions& options) {
  return run_status_only(query, options).result;
}

JacobiRunOutput JacobiPullEngine::run_status_only(
    const RouteQuery& query,
    const GpuRunOptions& options) {
  return run_impl(query, options, DistanceReadback::none);
}

JacobiRunOutput JacobiPullEngine::run_with_distances(
    const RouteQuery& query,
    const GpuRunOptions& options) {
  return run_impl(query, options, DistanceReadback::full_graph);
}

JacobiRunOutput JacobiPullEngine::run_with_selected_distances(
    const RouteQuery& query,
    const GpuRunOptions& options) {
  return run_impl(query, options, DistanceReadback::selected_ranges);
}

JacobiRunOutput JacobiPullEngine::run_impl(
    const RouteQuery& query,
    const GpuRunOptions& options,
    const DistanceReadback readback) {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot run a moved-from Jacobi engine"};
  }
  if (validate_gpu_run_options(options) != GpuRunOptionsError::none ||
      options.engine != EngineKind::jacobi_pull) {
    throw std::invalid_argument{"Jacobi run options are invalid or select another engine"};
  }
  if (!validate_route_query(impl_->host_graph, query).ok()) {
    throw std::invalid_argument{"Jacobi requires a deeply valid route query"};
  }

  const DeviceProperties properties = current_device_properties();
  if (options.block_size > properties.maximum_threads_per_block) {
    throw std::invalid_argument{"Jacobi block size exceeds the selected HIP device limit"};
  }

  const DeviceGraphView32 graph = impl_->resident_graph.view();
  const std::size_t scratch_bytes =
      jacobi_distance_scratch_bytes(graph.vertex_count);
  const std::size_t thread_statistics_bytes =
      options.instrumentation == InstrumentationLevel::none
          ? checked_multiply(
                options.block_size,
                sizeof(unsigned int),
                "Jacobi None changed-reduction shared bytes")
          : checked_multiply(
                options.block_size,
                sizeof(ThreadRoundStatistics),
                "Jacobi instrumented block-reduction shared bytes");
  SteadyClockTimer end_to_end_timer;

  impl_->tile_lane_masks.assign(graph.tile_count, 0U);
  for (const TileId tile : query.selected_tiles) {
    impl_->tile_lane_masks[tile.value()] = standalone_lane;
  }
  // prepare_query_async requires an exact active-orientation prefix.  The
  // initialization kernel replaces every entry from immutable CSC run tiles.
  impl_->empty_csc_run_masks.assign(graph.csc.run_count, 0U);

  const WorkspaceMemoryRequirements requirements = estimate_workspace_memory(
      EngineKind::jacobi_pull,
      query,
      graph.tile_count,
      graph.csr.run_count,
      graph.csc.run_count,
      1U,
      options.instrumentation,
      scratch_bytes);
  static_cast<void>(impl_->workspace.reserve(requirements));
  const DeviceController initial_controller =
      initialize_device_controller(options, standalone_lane);

  impl_->resident_graph.wait_until_ready(impl_->stream);
  impl_->preparation_gpu_timer.start(impl_->stream);
  const WorkspaceLease lease = impl_->workspace.prepare_query_async(
      requirements,
      query,
      impl_->tile_lane_masks,
      impl_->empty_csc_run_masks,
      initial_controller,
      impl_->stream);
  impl_->preparation_gpu_timer.stop(impl_->stream);
  bool lease_active = true;

  JacobiRunOutput output;
  output.distances_downloaded = readback != DistanceReadback::none;
  output.result.engine_kind =
      static_cast<std::uint32_t>(EngineKind::jacobi_pull);
  output.result.control_mode = static_cast<std::uint32_t>(options.control_mode);

  try {
    const DeviceWorkspaceView workspace_view = impl_->workspace.view(lease);
    const hipStream_t native_stream = as_stream(impl_->stream.native_handle());
    // The event interval has the same scope in every mode: device
    // initialization through terminal status materialization.
    impl_->sssp_gpu_timer.start(impl_->stream);
    if (options.control_mode != ControlMode::persistent_cooperative) {
      launch_initialize(graph, workspace_view, options, native_stream);
    }

    std::uint64_t convergence_kernel_dispatches = 0U;
    DeviceController observed_controller{};
    if (options.control_mode == ControlMode::persistent_cooperative) {
      const CooperativeGrid grid = cooperative_grid(
          properties, options, thread_statistics_bytes);
      output.metrics.cooperative_grid_blocks = grid.blocks;
      output.metrics.cooperative_active_blocks_per_wgp =
          grid.active_blocks_per_wgp;
      launch_persistent(
          graph,
          workspace_view,
          grid,
          options.block_size,
          thread_statistics_bytes,
          options.maximum_rounds,
          options.enable_per_lane_convergence,
          options.instrumentation,
          native_stream);
      convergence_kernel_dispatches = 1U;
    } else {
      const std::uint32_t grid = ordinary_grid_blocks(
          properties,
          options,
          graph.vertex_count,
          thread_statistics_bytes);
      const std::uint32_t chunk_rounds =
          options.control_mode == ControlMode::per_round_host_poll
              ? 1U
              : options.rounds_per_chunk;
      do {
        for (std::uint32_t round = 0U; round < chunk_rounds; ++round) {
          launch_round_pair(
              graph,
              workspace_view,
              grid,
              options.block_size,
              thread_statistics_bytes,
              options.instrumentation,
              native_stream);
          ++output.metrics.engine_round_dispatches;
          ++output.metrics.controller_advance_dispatches;
          convergence_kernel_dispatches += 2U;
        }
        check(
            hipMemcpyAsync(
                &observed_controller,
                workspace_view.controller,
                sizeof(observed_controller),
                hipMemcpyDeviceToHost,
                native_stream),
            "hipMemcpyAsync(Jacobi controller poll)");
        impl_->stream.synchronize();
        ++output.metrics.convergence_host_checks;
      } while (observed_controller.done == 0U);
    }

    if (options.control_mode != ControlMode::persistent_cooperative) {
      launch_finalize(graph, workspace_view, native_stream);
    }
    impl_->sssp_gpu_timer.stop(impl_->stream);

    DeviceController final_controller{};
    DeviceRunStatus final_status{};
    DeviceWorkStatistics final_statistics{};
    impl_->result_transfer_gpu_timer.start(impl_->stream);
    check(
        hipMemcpyAsync(
            &final_controller,
            workspace_view.controller,
            sizeof(final_controller),
            hipMemcpyDeviceToHost,
            native_stream),
        "hipMemcpyAsync(Jacobi final controller)");
    impl_->workspace.download_status_async(
        lease, final_status, impl_->stream);
    if (options.instrumentation != InstrumentationLevel::none) {
      impl_->workspace.download_instrumentation_async(
          lease, final_statistics, impl_->stream);
    }
    const std::size_t selected_vertices =
        readback == DistanceReadback::selected_ranges
            ? selected_vertex_count(impl_->host_graph, query)
            : 0U;
    if (readback == DistanceReadback::full_graph) {
      impl_->downloaded_distance_slots.resize(
          scratch_bytes / sizeof(float));
      check(
          hipMemcpyAsync(
              impl_->downloaded_distance_slots.data(),
              workspace_view.engine_scratch,
              scratch_bytes,
              hipMemcpyDeviceToHost,
              native_stream),
          "hipMemcpyAsync(Jacobi distance slots)");
    } else if (readback == DistanceReadback::selected_ranges) {
      impl_->downloaded_distance_slots.resize(2U * selected_vertices);
      const auto tile_offsets = impl_->host_graph.tile_vertex_offsets();
      const auto* const device_distances =
          reinterpret_cast<const float*>(workspace_view.engine_scratch);
      for (std::size_t slot = 0U; slot < 2U; ++slot) {
        std::size_t packed_offset = 0U;
        for (const TileId tile : query.selected_tiles) {
          const std::size_t begin =
              static_cast<std::size_t>(tile_offsets[tile.value()]);
          const std::size_t end = static_cast<std::size_t>(
              tile_offsets[tile.value() + 1U]);
          const std::size_t count = end - begin;
          check(
              hipMemcpyAsync(
                  impl_->downloaded_distance_slots.data() +
                      slot * selected_vertices + packed_offset,
                  device_distances + slot * graph.vertex_count + begin,
                  count * sizeof(float),
                  hipMemcpyDeviceToHost,
                  native_stream),
              "hipMemcpyAsync(Jacobi selected distance range)");
          packed_offset += count;
        }
        if (packed_offset != selected_vertices) {
          throw std::logic_error{
              "Jacobi selected-distance packing omitted a vertex"};
        }
      }
    }
    impl_->result_transfer_gpu_timer.stop(impl_->stream);
    // Lease retirement is the single terminal result-transfer fence.
    impl_->workspace.retire_query(lease, impl_->stream);
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
    output.metrics.gpu_milliseconds =
        output.metrics.sssp_device_timeline_milliseconds;
    if (validate_device_controller(final_controller) !=
            DeviceControllerError::none ||
        validate_device_run_status(final_status) !=
            DeviceRunStatusError::none ||
        final_status.final_distance_slot !=
            final_controller.distance_read_slot ||
        final_status.rounds_completed != final_controller.rounds_completed) {
      throw std::runtime_error{
          "Jacobi produced an invalid or inconsistent terminal controller/status"};
    }

    if (readback != DistanceReadback::none) {
      const std::size_t vertices =
          readback == DistanceReadback::full_graph
              ? static_cast<std::size_t>(graph.vertex_count)
              : selected_vertices;
      const std::size_t final_offset =
          static_cast<std::size_t>(final_status.final_distance_slot) * vertices;
      output.distances.assign(
          impl_->downloaded_distance_slots.begin() +
              static_cast<std::ptrdiff_t>(final_offset),
          impl_->downloaded_distance_slots.begin() +
              static_cast<std::ptrdiff_t>(final_offset + vertices));
      output.converged_slots_bitwise_identical =
          final_status.converged != 0U &&
          std::memcmp(
              impl_->downloaded_distance_slots.data(),
              impl_->downloaded_distance_slots.data() + vertices,
              vertices * sizeof(float)) == 0;
    }

    final_statistics.host_checks =
        output.metrics.convergence_host_checks;
    // Synchronizations count convergence polls plus one shared final
    // result-transfer/lease-retirement boundary.
    final_statistics.host_synchronizations =
        output.metrics.convergence_host_checks + 1U;
    // Controller copies include the initial workspace H2D record, convergence
    // polls, and one terminal validation D2H record. The distinct status and
    // instrumentation transfers are not controller records.
    final_statistics.controller_copies =
        output.metrics.convergence_host_checks + 2U;
    final_statistics.kernel_dispatches =
        options.control_mode == ControlMode::persistent_cooperative
            ? convergence_kernel_dispatches
            : 1U + convergence_kernel_dispatches + 1U;
    output.result.status = final_status;
    output.result.work = final_statistics;

    output.metrics.end_to_end_wall_milliseconds =
        end_to_end_timer.elapsed_milliseconds();
    output.metrics.wall_milliseconds =
        output.metrics.end_to_end_wall_milliseconds;
    return output;
  } catch (...) {
    if (lease_active) {
      try {
        impl_->workspace.retire_query(lease, impl_->stream);
      } catch (...) {
      }
    }
    throw;
  }
}

}  // namespace bfnew::hip
