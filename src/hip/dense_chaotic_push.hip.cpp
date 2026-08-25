#include "bfnew/hip/dense_chaotic_push.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace bfnew::hip {
namespace {

namespace cg = ::cooperative_groups;

constexpr LaneMask standalone_lane = LaneMask{1U};
constexpr std::uint32_t positive_infinity_bits = 0x7f800000U;

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

[[nodiscard]] unsigned int checked_cooperative_shared_bytes(
    const std::size_t shared_bytes) {
  if (shared_bytes > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error{
        "dense dynamic shared-memory request exceeds the HIP cooperative-launch ABI"};
  }
  return static_cast<unsigned int>(shared_bytes);
}

[[nodiscard]] std::uint32_t checked_grid_blocks(
    const std::uint32_t blocks_per_wgp,
    const std::uint32_t wgp_count) {
  const std::uint64_t blocks =
      static_cast<std::uint64_t>(blocks_per_wgp) * wgp_count;
  if (blocks == 0U || blocks > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{"dense grid block count exceeds the HIP ABI"};
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

struct DenseThreadStatistics {
  unsigned long long edges_examined{};
  unsigned long long successful_decreases{};
  unsigned long long active_vertices{};
  unsigned long long mask_operations{};
  unsigned long long atomic_attempts{};
  unsigned long long successful_atomic_updates{};
  unsigned int changed{};
};

__device__ void atomic_add_u64(
    std::uint64_t* const destination,
    const unsigned long long value) {
  atomicAdd(reinterpret_cast<unsigned long long*>(destination), value);
}

// All distance accesses use the same atomic domain. A CAS with identical
// compare/replacement operands is an atomic-compatible load and therefore is
// never mixed with an ordinary load while another thread may update the word.
[[nodiscard]] __device__ std::uint32_t atomic_load_nonnegative_float_bits(
    const std::uint32_t* const address) {
  return atomicCAS(const_cast<std::uint32_t*>(address), 0U, 0U);
}

// Nonnegative finite IEEE-754 values and +infinity have sign bit zero, so
// unsigned integer order is exactly floating-point order over this domain.
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

[[nodiscard]] __device__ DenseDistanceView dense_distance_view(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace) {
  return bind_dense_distance_view(
      workspace.engine_scratch,
      workspace.engine_scratch_bytes,
      graph.vertex_count);
}

__device__ void advance_controller_and_count(
    DeviceController* const controller,
    DeviceWorkStatistics* const statistics) {
  const DenseAdvanceResult transition = advance_dense_controller(*controller);
  if (statistics != nullptr &&
      transition != DenseAdvanceResult::no_op &&
      transition != DenseAdvanceResult::invalid_controller_state) {
    ++statistics->active_lane_rounds;
    ++statistics->full_edge_rounds;
  }
}

template <bool collect_statistics>
__device__ void perform_dense_round_thread_work(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const std::uint32_t instrumentation_level,
    DenseThreadStatistics& output) {
  DenseThreadStatistics local{};
  if constexpr (!collect_statistics) {
    static_cast<void>(instrumentation_level);
  }
  const std::uint32_t done = workspace.controller->done;
  const LaneMask execute_lane_mask = workspace.controller->execute_lane_mask;
  const bool execute =
      done == 0U && (execute_lane_mask & standalone_lane) != 0U;

  if (execute) {
    const DenseDistanceView distances = dense_distance_view(graph, workspace);
    if (!valid_dense_distance_view(distances)) {
      if (blockIdx.x == 0U && threadIdx.x == 0U) {
        atomicOr(
            &workspace.controller->error_bits,
            device_error::invalid_controller_state);
      }
    } else {
      const std::uint64_t global_thread =
          static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      const std::uint64_t grid_threads =
          static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
      for (std::uint64_t source = global_thread;
           source < graph.vertex_count;
           source += grid_threads) {
        const std::uint32_t source32 = static_cast<std::uint32_t>(source);
        const std::uint32_t source_tile = graph.owner_tiles[source32];
        if ((workspace.tile_lane_masks[source_tile] & standalone_lane) == 0U) {
          continue;
        }
        if constexpr (collect_statistics) {
          ++local.active_vertices;
        }
        const std::uint32_t run_begin = graph.csr.row_run_offsets[source32];
        const std::uint32_t run_end = graph.csr.row_run_offsets[source32 + 1U];
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
          const std::uint32_t edge_begin = graph.csr.run_edge_offsets[run];
          const std::uint32_t edge_end = graph.csr.run_edge_offsets[run + 1U];
          if constexpr (collect_statistics) {
            local.edges_examined +=
                static_cast<unsigned long long>(edge_end - edge_begin);
          }
          for (std::uint32_t edge = edge_begin; edge < edge_end; ++edge) {
            // Reload d[source] for every edge. Other threads may decrease it
            // during this complete chaotic scan.
            const std::uint32_t source_bits =
                atomic_load_nonnegative_float_bits(
                    distances.distance_bits + source32);
            const float candidate_value =
                __builtin_bit_cast(float, source_bits) + graph.csr.weights[edge];
            const std::uint32_t candidate =
                __builtin_bit_cast(std::uint32_t, candidate_value);
            if constexpr (collect_statistics) {
              if (instrumentation_level ==
                  static_cast<std::uint32_t>(InstrumentationLevel::debug)) {
                ++local.atomic_attempts;
              }
            }
            const std::uint32_t destination = graph.csr.destinations[edge];
            if (atomic_min_nonnegative_float_bits(
                    distances.distance_bits + destination, candidate)) {
              local.changed = 1U;
              if constexpr (collect_statistics) {
                ++local.successful_decreases;
                if (instrumentation_level ==
                    static_cast<std::uint32_t>(InstrumentationLevel::debug)) {
                  ++local.successful_atomic_updates;
                }
              }
            }
          }
        }
      }
    }
  }

  output = local;
}

__device__ void perform_dense_round_none(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    unsigned int* const block_changes) {
  DenseThreadStatistics local{};
  perform_dense_round_thread_work<false>(
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

__device__ void perform_dense_round_instrumented(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    DenseThreadStatistics* const block_statistics,
    const std::uint32_t instrumentation_level) {
  DenseThreadStatistics local{};
  perform_dense_round_thread_work<true>(
      graph, workspace, instrumentation_level, local);
  block_statistics[threadIdx.x] = local;
  __syncthreads();
  if (threadIdx.x == 0U) {
    DenseThreadStatistics reduced{};
    for (std::uint32_t thread = 0U; thread < blockDim.x; ++thread) {
      reduced.edges_examined += block_statistics[thread].edges_examined;
      reduced.successful_decreases +=
          block_statistics[thread].successful_decreases;
      reduced.active_vertices += block_statistics[thread].active_vertices;
      reduced.mask_operations += block_statistics[thread].mask_operations;
      reduced.atomic_attempts += block_statistics[thread].atomic_attempts;
      reduced.successful_atomic_updates +=
          block_statistics[thread].successful_atomic_updates;
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
          &workspace.instrumentation->active_vertices,
          reduced.active_vertices);
      atomic_add_u64(
          &workspace.instrumentation->mask_operations,
          reduced.mask_operations);
      atomic_add_u64(
          &workspace.instrumentation->atomic_attempts,
          reduced.atomic_attempts);
      atomic_add_u64(
          &workspace.instrumentation->successful_atomic_updates,
          reduced.successful_atomic_updates);
      if (instrumentation_level ==
              static_cast<std::uint32_t>(InstrumentationLevel::debug) &&
          reduced.changed != 0U) {
        atomic_add_u64(&workspace.instrumentation->changed_flag_updates, 1U);
      }
    }
  }
}

__device__ void initialize_dense_state(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  const std::uint64_t global_thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t grid_threads =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const DenseDistanceView distances = dense_distance_view(graph, workspace);

  for (std::uint64_t vertex = global_thread; vertex < graph.vertex_count;
       vertex += grid_threads) {
    bool source = false;
    for (std::uint32_t index = 0U; index < workspace.source_count; ++index) {
      source = source || workspace.sources[index] == vertex;
    }
    *(distances.distance_bits + vertex) = source ? 0U : positive_infinity_bits;

    const std::uint32_t vertex32 = static_cast<std::uint32_t>(vertex);
    const std::uint32_t source_tile = graph.owner_tiles[vertex32];
    const LaneMask source_mask = workspace.tile_lane_masks[source_tile];
    const std::uint32_t run_begin = graph.csr.row_run_offsets[vertex32];
    const std::uint32_t run_end = graph.csr.row_run_offsets[vertex32 + 1U];
    for (std::uint32_t run = run_begin; run < run_end; ++run) {
      const std::uint32_t destination_tile =
          graph.csr.run_destination_tiles[run];
      workspace.run_lane_masks[run] =
          source_mask & workspace.tile_lane_masks[destination_tile] &
          standalone_lane;
    }

    if (workspace.instrumentation != nullptr &&
        instrumentation_level ==
            static_cast<std::uint32_t>(InstrumentationLevel::debug) &&
        (workspace.tile_lane_masks[graph.owner_tiles[vertex32]] &
         standalone_lane) != 0U) {
      std::uint64_t admitted_incoming = 0U;
      const std::uint32_t column_run_begin =
          graph.csc.column_run_offsets[vertex32];
      const std::uint32_t column_run_end =
          graph.csc.column_run_offsets[vertex32 + 1U];
      for (std::uint32_t run = column_run_begin; run < column_run_end; ++run) {
        const std::uint32_t incoming_tile = graph.csc.run_source_tiles[run];
        if ((workspace.tile_lane_masks[incoming_tile] & standalone_lane) != 0U) {
          admitted_incoming +=
              graph.csc.run_edge_offsets[run + 1U] -
              graph.csc.run_edge_offsets[run];
        }
      }
      if (admitted_incoming >= 2U) {
        atomic_add_u64(
            &workspace.instrumentation->high_contention_destinations, 1U);
      }
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
    controller->error_bits = 0U;
  }
}

__global__ void initialize_dense_query(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  initialize_dense_state(
      graph,
      workspace,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level);
}

__global__ void dense_round_none_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint32_t instrumentation_level) {
  static_cast<void>(instrumentation_level);
  extern __shared__ unsigned int block_changes[];
  perform_dense_round_none(graph, workspace, block_changes);
}

__global__ void dense_round_instrumented_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint32_t instrumentation_level) {
  extern __shared__ DenseThreadStatistics block_statistics[];
  perform_dense_round_instrumented(
      graph, workspace, block_statistics, instrumentation_level);
}

__global__ void dense_advance_kernel(const DeviceWorkspaceView workspace) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    advance_controller_and_count(
        workspace.controller, workspace.instrumentation);
  }
}

[[nodiscard]] __device__ DeviceRunStatus completed_dense_status(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace) {
  const DeviceController controller = *workspace.controller;
  const bool converged =
      controller.stop_reason ==
      static_cast<std::uint32_t>(DeviceStopReason::converged);
  bool reached_all_targets = converged;
  if (reached_all_targets) {
    const DenseDistanceView distances = dense_distance_view(graph, workspace);
    reached_all_targets = valid_dense_distance_view(distances);
    for (std::uint32_t index = 0U;
         reached_all_targets && index < workspace.target_count;
         ++index) {
      const std::uint32_t bits = atomic_load_nonnegative_float_bits(
          distances.distance_bits + workspace.targets[index]);
      reached_all_targets = bits != positive_infinity_bits;
    }
  }
  const LaneMask reached =
      converged && reached_all_targets ? standalone_lane : 0U;
  const LaneMask missed =
      converged && !reached_all_targets ? standalone_lane : 0U;
  return make_dense_run_status(controller, reached, missed);
}

template <bool collect_statistics>
__device__ void execute_dense_persistent(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level,
    void* const block_shared) {
  cg::grid_group grid = cg::this_grid();

  grid.sync();
  initialize_dense_state(
      graph,
      workspace,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level);
  grid.sync();
  for (;;) {
    if constexpr (collect_statistics) {
      perform_dense_round_instrumented(
          graph,
          workspace,
          static_cast<DenseThreadStatistics*>(block_shared),
          instrumentation_level);
    } else {
      perform_dense_round_none(
          graph, workspace, static_cast<unsigned int*>(block_shared));
    }
    grid.sync();
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      advance_controller_and_count(
          workspace.controller, workspace.instrumentation);
    }
    grid.sync();
    if (workspace.controller->done != 0U) {
      break;
    }
  }
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *workspace.status = completed_dense_status(graph, workspace);
  }
  grid.sync();
}

__global__ void dense_persistent_none_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  extern __shared__ unsigned int block_changes[];
  execute_dense_persistent<false>(
      graph,
      workspace,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level,
      block_changes);
}

__global__ void dense_persistent_instrumented_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint64_t maximum_rounds,
    const std::uint32_t enable_per_lane_convergence,
    const std::uint32_t instrumentation_level) {
  extern __shared__ DenseThreadStatistics block_statistics[];
  execute_dense_persistent<true>(
      graph,
      workspace,
      maximum_rounds,
      enable_per_lane_convergence,
      instrumentation_level,
      block_statistics);
}

__global__ void finalize_dense_status(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    *workspace.status = completed_dense_status(graph, workspace);
  }
}

[[nodiscard]] std::uint32_t initialization_grid_blocks(
    const std::uint32_t vertex_count,
    const std::uint32_t block_size) {
  const std::uint64_t blocks = std::max<std::uint64_t>(
      1U,
      (static_cast<std::uint64_t>(vertex_count) + block_size - 1U) /
          block_size);
  if (blocks > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{"dense initialization grid exceeds 32 bits"};
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
            dense_round_none_kernel,
            static_cast<int>(block_size),
            shared_bytes),
        "hipOccupancyMaxActiveBlocksPerMultiprocessor(dense None round)");
  } else {
    check(
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &active,
            dense_round_instrumented_kernel,
            static_cast<int>(block_size),
            shared_bytes),
        "hipOccupancyMaxActiveBlocksPerMultiprocessor(dense instrumented round)");
  }
  if (active <= 0) {
    throw std::runtime_error{"dense round kernel has zero occupancy"};
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
            dense_persistent_none_kernel,
            static_cast<int>(block_size),
            shared_bytes),
        "hipOccupancyMaxActiveBlocksPerMultiprocessor(dense None persistent)");
  } else {
    check(
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &active,
            dense_persistent_instrumented_kernel,
            static_cast<int>(block_size),
            shared_bytes),
        "hipOccupancyMaxActiveBlocksPerMultiprocessor(dense instrumented persistent)");
  }
  if (active <= 0) {
    throw std::runtime_error{"dense persistent kernel has zero occupancy"};
  }
  return active;
}

[[nodiscard]] std::uint32_t ordinary_grid_blocks(
    const DeviceProperties& properties,
    const GpuRunOptions& options,
    const std::uint32_t vertex_count,
    const std::size_t shared_bytes) {
  const int occupancy =
      occupancy_for_round_kernel(
          options.block_size, shared_bytes, options.instrumentation);
  std::uint32_t blocks_per_wgp = static_cast<std::uint32_t>(occupancy);
  if (options.grid_policy == GridPolicy::fixed_blocks_per_wgp) {
    if (options.blocks_per_wgp > blocks_per_wgp) {
      throw std::invalid_argument{
          "requested dense ordinary grid exceeds real-kernel occupancy"};
    }
    blocks_per_wgp = options.blocks_per_wgp;
  }
  const std::uint32_t resident_grid =
      checked_grid_blocks(blocks_per_wgp, properties.wgp_count);
  const std::uint64_t work_blocks = std::max<std::uint64_t>(
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
    throw std::runtime_error{
        "selected HIP device does not support cooperative launch"};
  }
  const int occupancy =
      occupancy_for_persistent_kernel(
          options.block_size, shared_bytes, options.instrumentation);
  const std::uint32_t active = static_cast<std::uint32_t>(occupancy);
  std::uint32_t blocks_per_wgp = active;
  if (options.grid_policy == GridPolicy::fixed_blocks_per_wgp) {
    if (options.blocks_per_wgp > active) {
      throw std::invalid_argument{
          "requested cooperative grid exceeds dense-kernel occupancy"};
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
  hipLaunchKernelGGL(
      initialize_dense_query,
      dim3(initialization_grid_blocks(graph.vertex_count, options.block_size)),
      dim3(options.block_size),
      0U,
      stream,
      graph,
      workspace,
      options.maximum_rounds,
      options.enable_per_lane_convergence,
      static_cast<std::uint32_t>(options.instrumentation));
  check_launch("initialize_dense_query launch");
}

void launch_round_pair(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const std::uint32_t blocks,
    const GpuRunOptions& options,
    const std::size_t shared_bytes,
    const hipStream_t stream) {
  if (options.instrumentation == InstrumentationLevel::none) {
    hipLaunchKernelGGL(
        dense_round_none_kernel,
        dim3(blocks),
        dim3(options.block_size),
        shared_bytes,
        stream,
        graph,
        workspace,
        static_cast<std::uint32_t>(options.instrumentation));
    check_launch("dense_round_none_kernel launch");
  } else {
    hipLaunchKernelGGL(
        dense_round_instrumented_kernel,
        dim3(blocks),
        dim3(options.block_size),
        shared_bytes,
        stream,
        graph,
        workspace,
        static_cast<std::uint32_t>(options.instrumentation));
    check_launch("dense_round_instrumented_kernel launch");
  }
  hipLaunchKernelGGL(
      dense_advance_kernel,
      dim3(1U),
      dim3(1U),
      0U,
      stream,
      workspace);
  check_launch("dense_advance_kernel launch");
}

void launch_persistent(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const CooperativeGrid grid,
    const GpuRunOptions& options,
    const std::size_t shared_bytes,
    const hipStream_t stream) {
  DeviceGraphView32 graph_argument = graph;
  DeviceWorkspaceView workspace_argument = workspace;
  std::uint64_t maximum_rounds_argument = options.maximum_rounds;
  std::uint32_t convergence_argument = options.enable_per_lane_convergence;
  std::uint32_t instrumentation_argument =
      static_cast<std::uint32_t>(options.instrumentation);
  void* arguments[] = {
      &graph_argument,
      &workspace_argument,
      &maximum_rounds_argument,
      &convergence_argument,
      &instrumentation_argument,
  };
  const void* const kernel =
      options.instrumentation == InstrumentationLevel::none
          ? reinterpret_cast<const void*>(dense_persistent_none_kernel)
          : reinterpret_cast<const void*>(dense_persistent_instrumented_kernel);
  const unsigned int launch_shared_bytes =
      checked_cooperative_shared_bytes(shared_bytes);
  check(
      hipLaunchCooperativeKernel(
          kernel,
          dim3(grid.blocks),
          dim3(options.block_size),
          arguments,
          launch_shared_bytes,
          stream),
      options.instrumentation == InstrumentationLevel::none
          ? "hipLaunchCooperativeKernel(dense None persistent)"
          : "hipLaunchCooperativeKernel(dense instrumented persistent)");
}

void launch_finalize(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const hipStream_t stream) {
  hipLaunchKernelGGL(
      finalize_dense_status,
      dim3(1U),
      dim3(1U),
      0U,
      stream,
      graph,
      workspace);
  check_launch("finalize_dense_status launch");
}

}  // namespace

std::size_t dense_distance_scratch_bytes(const std::uint32_t vertex_count) {
  const DenseScratchLayout layout = make_dense_scratch_layout(vertex_count);
  if (layout.total_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error{"dense distance scratch bytes overflow"};
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

class DenseChaoticPushEngine::Impl final {
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
          "dense push requires a valid spatially ordered host graph"};
    }
    if (!validate_tile_run_layout(host_graph, tile_runs).ok()) {
      throw std::invalid_argument{
          "dense push requires deeply valid tile-run metadata"};
    }
    if (!resident_graph.has_upload()) {
      throw std::invalid_argument{
          "dense push requires an uploaded resident graph"};
    }
    const DeviceGraphView32& device = resident_graph.view();
    if (device.vertex_count != host_graph.vertex_count() ||
        device.edge_count != host_graph.edge_count() ||
        device.tile_count != host_graph.tile_coordinates().size() ||
        device.csr.run_count != tile_runs.csr_run_destination_tiles.size()) {
      throw std::invalid_argument{
          "dense host graph/run shapes disagree with resident graph"};
    }
    if (resident_graph.fingerprint() !=
        fingerprint_device_graph_source32(host_graph, tile_runs)) {
      throw std::invalid_argument{
          "dense host graph/run content disagrees with resident graph"};
    }
  }

  const WeightedGraph& host_graph;
  const TileRunLayout64& tile_runs;
  const ResidentDeviceGraph& resident_graph;
  ReusableDeviceWorkspace& workspace;
  const HipStream& stream;
  std::vector<LaneMask> tile_lane_masks;
  std::vector<LaneMask> empty_csr_run_masks;
  std::vector<std::uint32_t> downloaded_distance_bits;
  HipEventTimer preparation_gpu_timer;
  HipEventTimer sssp_gpu_timer;
  HipEventTimer result_transfer_gpu_timer;
};

DenseChaoticPushEngine::DenseChaoticPushEngine(
    const WeightedGraph& host_graph,
    const TileRunLayout64& tile_runs,
    const ResidentDeviceGraph& resident_graph,
    ReusableDeviceWorkspace& workspace,
    const HipStream& stream)
    : impl_{new Impl{
          host_graph, tile_runs, resident_graph, workspace, stream}} {}

DenseChaoticPushEngine::~DenseChaoticPushEngine() { delete impl_; }

GpuSsspResult DenseChaoticPushEngine::run(
    const RouteQuery& query,
    const GpuRunOptions& options) {
  return run_status_only(query, options).result;
}

DenseRunOutput DenseChaoticPushEngine::run_status_only(
    const RouteQuery& query,
    const GpuRunOptions& options) {
  return run_impl(query, options, DistanceReadback::none);
}

DenseRunOutput DenseChaoticPushEngine::run_with_distances(
    const RouteQuery& query,
    const GpuRunOptions& options) {
  return run_impl(query, options, DistanceReadback::full_graph);
}

DenseRunOutput DenseChaoticPushEngine::run_with_selected_distances(
    const RouteQuery& query,
    const GpuRunOptions& options) {
  return run_impl(query, options, DistanceReadback::selected_ranges);
}

DenseRunOutput DenseChaoticPushEngine::run_impl(
    const RouteQuery& query,
    const GpuRunOptions& options,
    const DistanceReadback readback) {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot run a moved-from dense engine"};
  }
  if (validate_gpu_run_options(options) != GpuRunOptionsError::none ||
      options.engine != EngineKind::dense_chaotic_push) {
    throw std::invalid_argument{
        "dense run options are invalid or select another engine"};
  }
  if (!validate_route_query(impl_->host_graph, query).ok()) {
    throw std::invalid_argument{"dense push requires a valid route query"};
  }

  const DeviceProperties properties = current_device_properties();
  if (options.block_size > properties.maximum_threads_per_block) {
    throw std::invalid_argument{
        "dense block size exceeds the selected HIP device limit"};
  }
  const DeviceGraphView32 graph = impl_->resident_graph.view();
  const std::size_t scratch_bytes =
      dense_distance_scratch_bytes(graph.vertex_count);
  const std::size_t shared_bytes =
      options.instrumentation == InstrumentationLevel::none
          ? checked_multiply(
                options.block_size,
                sizeof(unsigned int),
                "dense None changed-reduction shared bytes")
          : checked_multiply(
                options.block_size,
                sizeof(DenseThreadStatistics),
                "dense instrumented block-reduction shared bytes");
  SteadyClockTimer end_to_end_timer;

  impl_->tile_lane_masks.assign(graph.tile_count, 0U);
  for (const TileId tile : query.selected_tiles) {
    impl_->tile_lane_masks[tile.value()] = standalone_lane;
  }
  // The device initialization writes every exact CSR source/destination-tile
  // intersection before the first scan.
  impl_->empty_csr_run_masks.assign(graph.csr.run_count, 0U);

  const WorkspaceMemoryRequirements requirements = estimate_workspace_memory(
      EngineKind::dense_chaotic_push,
      query,
      graph.tile_count,
      graph.csr.run_count,
      graph.csc.run_count,
      1U,
      options.instrumentation,
      scratch_bytes);
  static_cast<void>(impl_->workspace.reserve(requirements));
  DeviceController initial_controller =
      initialize_device_controller(options, standalone_lane);
  initial_controller.distance_write_slot = 0U;

  impl_->resident_graph.wait_until_ready(impl_->stream);
  impl_->preparation_gpu_timer.start(impl_->stream);
  const WorkspaceLease lease = impl_->workspace.prepare_query_async(
      requirements,
      query,
      impl_->tile_lane_masks,
      impl_->empty_csr_run_masks,
      initial_controller,
      impl_->stream);
  impl_->preparation_gpu_timer.stop(impl_->stream);
  bool lease_active = true;

  DenseRunOutput output;
  output.distances_downloaded = readback != DistanceReadback::none;
  output.result.engine_kind =
      static_cast<std::uint32_t>(EngineKind::dense_chaotic_push);
  output.result.control_mode = static_cast<std::uint32_t>(options.control_mode);

  try {
    const DeviceWorkspaceView workspace_view = impl_->workspace.view(lease);
    const hipStream_t native_stream = as_stream(impl_->stream.native_handle());
    impl_->sssp_gpu_timer.start(impl_->stream);
    if (options.control_mode != ControlMode::persistent_cooperative) {
      launch_initialize(graph, workspace_view, options, native_stream);
    }

    std::uint64_t convergence_kernel_dispatches = 0U;
    DeviceController observed_controller{};
    if (options.control_mode == ControlMode::persistent_cooperative) {
      const CooperativeGrid grid =
          cooperative_grid(properties, options, shared_bytes);
      output.metrics.cooperative_grid_blocks = grid.blocks;
      output.metrics.cooperative_active_blocks_per_wgp =
          grid.active_blocks_per_wgp;
      launch_persistent(
          graph, workspace_view, grid, options, shared_bytes, native_stream);
      convergence_kernel_dispatches = 1U;
    } else {
      const std::uint32_t grid = ordinary_grid_blocks(
          properties,
          options,
          graph.vertex_count,
          shared_bytes);
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
              options,
              shared_bytes,
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
            "hipMemcpyAsync(dense controller poll)");
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
        "hipMemcpyAsync(dense final controller)");
    impl_->workspace.download_status_async(lease, final_status, impl_->stream);
    if (options.instrumentation != InstrumentationLevel::none) {
      impl_->workspace.download_instrumentation_async(
          lease, final_statistics, impl_->stream);
    }
    if (readback == DistanceReadback::full_graph) {
      impl_->downloaded_distance_bits.resize(graph.vertex_count);
      check(
          hipMemcpyAsync(
              impl_->downloaded_distance_bits.data(),
              workspace_view.engine_scratch,
              scratch_bytes,
              hipMemcpyDeviceToHost,
              native_stream),
          "hipMemcpyAsync(dense distance bits)");
    } else if (readback == DistanceReadback::selected_ranges) {
      const std::size_t selected_vertices =
          selected_vertex_count(impl_->host_graph, query);
      impl_->downloaded_distance_bits.resize(selected_vertices);
      const auto tile_offsets = impl_->host_graph.tile_vertex_offsets();
      const auto* const device_distances =
          reinterpret_cast<const std::uint32_t*>(
              workspace_view.engine_scratch);
      std::size_t packed_offset = 0U;
      for (const TileId tile : query.selected_tiles) {
        const std::size_t begin =
            static_cast<std::size_t>(tile_offsets[tile.value()]);
        const std::size_t end = static_cast<std::size_t>(
            tile_offsets[tile.value() + 1U]);
        const std::size_t count = end - begin;
        check(
            hipMemcpyAsync(
                impl_->downloaded_distance_bits.data() + packed_offset,
                device_distances + begin,
                count * sizeof(std::uint32_t),
                hipMemcpyDeviceToHost,
                native_stream),
            "hipMemcpyAsync(dense selected distance range)");
        packed_offset += count;
      }
      if (packed_offset != selected_vertices) {
        throw std::logic_error{
            "dense selected-distance packing omitted a vertex"};
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
        final_controller.distance_read_slot != 0U ||
        final_controller.distance_write_slot != 0U ||
        final_status.final_distance_slot != 0U ||
        final_status.rounds_completed != final_controller.rounds_completed) {
      throw std::runtime_error{
          "dense push produced an invalid terminal controller/status"};
    }

    if (readback != DistanceReadback::none) {
      output.distances.reserve(impl_->downloaded_distance_bits.size());
      for (const std::uint32_t bits : impl_->downloaded_distance_bits) {
        output.distances.push_back(dense_atomic_bits_float(bits));
      }
    }
    final_statistics.host_checks = output.metrics.convergence_host_checks;
    final_statistics.host_synchronizations =
        output.metrics.convergence_host_checks + 1U;
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
