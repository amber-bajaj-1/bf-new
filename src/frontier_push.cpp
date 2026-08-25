#include "bfnew/frontier_push.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bfnew {
namespace {

constexpr LaneMask standalone_lane = LaneMask{1U};
constexpr std::uint32_t positive_infinity_bits = 0x7f800000U;

[[nodiscard]] std::size_t checked_add(
    const std::size_t left,
    const std::size_t right,
    const char* const description) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error{description};
  }
  return left + right;
}

[[nodiscard]] std::size_t checked_multiply(
    const std::size_t left,
    const std::size_t right,
    const char* const description) {
  if (right != 0U && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error{description};
  }
  return left * right;
}

[[nodiscard]] std::size_t align_up(
    const std::size_t value,
    const std::size_t alignment) {
  const std::size_t remainder = value % alignment;
  return remainder == 0U
             ? value
             : checked_add(
                   value,
                   alignment - remainder,
                   "frontier scratch alignment overflow");
}

void require(const bool condition, const char* const description) {
  if (!condition) {
    throw std::invalid_argument{description};
  }
}

[[nodiscard]] bool valid_offsets(
    const std::span<const std::uint32_t> offsets,
    const std::size_t bucket_count,
    const std::size_t entry_count) noexcept {
  return offsets.size() == bucket_count + 1U && !offsets.empty() &&
         offsets.front() == 0U &&
         static_cast<std::size_t>(offsets.back()) == entry_count &&
         std::is_sorted(offsets.begin(), offsets.end());
}

void validate_host_problem(
    const DeviceGraphLayout32& graph,
    const RouteQuery& query,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csr_run_lane_masks,
    const GpuRunOptions& options,
    const std::size_t queue_capacity) {
  require(
      validate_gpu_run_options(options) == GpuRunOptionsError::none,
      "host frontier execution requires valid run options");
  require(
      options.engine == EngineKind::frontier_push,
      "host frontier execution rejects another engine kind");
  require(graph.vertex_count != 0U, "host frontier graph must contain a vertex");
  require(
      queue_capacity != 0U && queue_capacity <= graph.vertex_count,
      "frontier queue capacity must be in [1, vertex_count]");

  const std::size_t vertex_count = graph.vertex_count;
  const std::size_t edge_count = graph.edge_count;
  const std::size_t tile_count = graph.tile_count;
  const std::size_t csr_run_count = graph.csr_run_destination_tiles.size();
  require(
      graph.owner_tiles.size() == vertex_count,
      "host frontier owner-tile shape mismatch");
  require(
      graph.csr_destinations.size() == edge_count &&
          graph.csr_weights.size() == edge_count &&
          valid_offsets(graph.csr_row_offsets, vertex_count, edge_count),
      "host frontier CSR edge fields or rows are invalid");
  require(
      valid_offsets(
          graph.csr_row_run_offsets, vertex_count, csr_run_count) &&
          valid_offsets(
              graph.csr_run_edge_offsets, csr_run_count, edge_count) &&
          graph.csr_run_destination_tiles.size() == csr_run_count,
      "host frontier CSR run fields are invalid");
  require(
      tile_lane_masks.size() == tile_count,
      "host frontier tile-mask shape mismatch");
  require(
      csr_run_lane_masks.size() == csr_run_count,
      "host frontier run-mask shape mismatch");
  require(
      !query.sources.empty() && !query.targets.empty() &&
          !query.selected_tiles.empty(),
      "host frontier query sets must be populated");
  const auto strictly_increasing_vertices = [](const auto& vertices) {
    for (std::size_t index = 1U; index < vertices.size(); ++index) {
      if (vertices[index - 1U].value() >= vertices[index].value()) {
        return false;
      }
    }
    return true;
  };
  require(
      strictly_increasing_vertices(query.sources) &&
          strictly_increasing_vertices(query.targets),
      "host frontier terminal sets must be sorted and deduplicated");

  std::vector<bool> selected(tile_count, false);
  std::uint32_t preceding_tile = 0U;
  bool has_preceding_tile = false;
  for (const TileId tile : query.selected_tiles) {
    require(tile.value() < graph.tile_count, "query selected tile is out of range");
    require(
        !has_preceding_tile || preceding_tile < tile.value(),
        "query selected tiles must be canonical");
    selected[tile.value()] = true;
    preceding_tile = tile.value();
    has_preceding_tile = true;
  }
  for (std::size_t tile = 0U; tile < tile_count; ++tile) {
    require(
        tile_lane_masks[tile] ==
            (selected[tile] ? standalone_lane : LaneMask{0U}),
        "frontier tile masks do not match selected tiles");
  }
  const auto require_terminal = [&](const VertexId vertex) {
    require(vertex.value() < graph.vertex_count, "query terminal is out of range");
    const std::uint32_t owner = graph.owner_tiles[vertex.value()];
    require(owner < graph.tile_count, "query terminal owner is out of range");
    require(selected[owner], "query terminal owner tile is not selected");
  };
  for (const VertexId source : query.sources) {
    require_terminal(source);
  }
  for (const VertexId target : query.targets) {
    require_terminal(target);
  }

  for (std::size_t source = 0U; source < vertex_count; ++source) {
    const std::uint32_t source_tile = graph.owner_tiles[source];
    require(source_tile < tile_count, "vertex owner tile is out of range");
    const std::size_t edge_begin = graph.csr_row_offsets[source];
    const std::size_t edge_end = graph.csr_row_offsets[source + 1U];
    const std::size_t run_begin = graph.csr_row_run_offsets[source];
    const std::size_t run_end = graph.csr_row_run_offsets[source + 1U];
    std::size_t cursor = edge_begin;
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const std::size_t begin = graph.csr_run_edge_offsets[run];
      const std::size_t end = graph.csr_run_edge_offsets[run + 1U];
      require(
          begin == cursor && begin < end && end <= edge_end,
          "CSR run does not exactly cover its source row");
      const std::uint32_t destination_tile =
          graph.csr_run_destination_tiles[run];
      require(destination_tile < tile_count, "CSR run tile is out of range");
      require(
          csr_run_lane_masks[run] ==
              (tile_lane_masks[source_tile] &
               tile_lane_masks[destination_tile]),
          "CSR run mask differs from endpoint-tile admission");
      for (std::size_t edge = begin; edge < end; ++edge) {
        const std::uint32_t destination = graph.csr_destinations[edge];
        require(destination < graph.vertex_count, "CSR destination is out of range");
        require(
            graph.owner_tiles[destination] == destination_tile,
            "CSR run destination tile is inconsistent");
        require(
            std::isfinite(graph.csr_weights[edge]) &&
                graph.csr_weights[edge] >= 0.0F &&
                !(graph.csr_weights[edge] == 0.0F &&
                  std::signbit(graph.csr_weights[edge])),
            "CSR contains an invalid frontier weight");
      }
      cursor = end;
    }
    require(cursor == edge_end, "CSR runs do not cover their source row");
  }
}

struct HostFrontierState {
  std::vector<std::uint32_t> distances;
  std::vector<std::uint32_t> frontiers[2];
  std::vector<std::uint64_t> enqueue_generation;
  std::size_t queue_capacity{};
};

void initialize_host_state(
    const DeviceGraphLayout32& graph,
    const RouteQuery& query,
    const GpuRunOptions& options,
    HostFrontierState& state,
    DeviceController& controller,
    DeviceWorkStatistics& work) {
  state.distances.assign(graph.vertex_count, positive_infinity_bits);
  state.frontiers[0].assign(state.queue_capacity, 0U);
  state.frontiers[1].assign(state.queue_capacity, 0U);
  state.enqueue_generation.assign(graph.vertex_count, 0U);
  controller = initialize_device_controller(
      options, standalone_lane, query.sources.size());
  for (std::size_t source = 0U; source < query.sources.size(); ++source) {
    const std::uint32_t vertex = query.sources[source].value();
    state.distances[vertex] = 0U;
    if (source < state.queue_capacity) {
      state.frontiers[0][source] = vertex;
    }
  }
  if (options.instrumentation != InstrumentationLevel::none) {
    work.maximum_queue_size = query.sources.size();
  }
  if (query.sources.size() > state.queue_capacity) {
    if (options.instrumentation == InstrumentationLevel::debug) {
      work.overflow_events = 1U;
    }
    detail::canonicalize_frontier_error(
        controller,
        DeviceStopReason::queue_overflow,
        device_error::queue_overflow);
  }
}

void execute_host_round(
    const DeviceGraphLayout32& graph,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csr_run_lane_masks,
    HostFrontierState& state,
    DeviceController& controller,
    DeviceWorkStatistics& work,
    std::vector<std::uint64_t>& frontier_trace,
    const InstrumentationLevel instrumentation) {
  if (controller.done != 0U || controller.execute_lane_mask == 0U) {
    return;
  }
  const bool collect_light = instrumentation != InstrumentationLevel::none;
  const bool collect_debug = instrumentation == InstrumentationLevel::debug;
  const std::uint32_t read_slot = controller.frontier_read_slot;
  const std::uint32_t write_slot = controller.frontier_write_slot;
  const std::uint64_t current_size = controller.frontier_size[read_slot];
  if (current_size == 0U || current_size > state.queue_capacity ||
      controller.frontier_size[write_slot] != 0U) {
    detail::canonicalize_frontier_error(
        controller,
        DeviceStopReason::invalid_controller_state,
        device_error::invalid_controller_state);
    return;
  }
  frontier_trace.push_back(current_size);
  if (collect_light) {
    ++work.active_lane_rounds;
    work.active_vertices += current_size;
    if (current_size < 32U) {
      ++work.small_frontier_rounds;
    }
  }

  const std::uint64_t generation = controller.rounds_completed + 1U;
  for (std::uint64_t queue_index = 0U; queue_index < current_size;
       ++queue_index) {
    const std::uint32_t source =
        state.frontiers[read_slot][static_cast<std::size_t>(queue_index)];
    const std::uint32_t source_tile = graph.owner_tiles[source];
    if ((tile_lane_masks[source_tile] & standalone_lane) == 0U) {
      detail::canonicalize_frontier_error(
          controller,
          DeviceStopReason::invalid_controller_state,
          device_error::invalid_controller_state);
      return;
    }
    const std::size_t run_begin = graph.csr_row_run_offsets[source];
    const std::size_t run_end = graph.csr_row_run_offsets[source + 1U];
    for (std::size_t run = run_begin; run < run_end; ++run) {
      if (collect_debug) {
        ++work.mask_operations;
      }
      if ((csr_run_lane_masks[run] & standalone_lane) == 0U) {
        continue;
      }
      const std::size_t edge_begin = graph.csr_run_edge_offsets[run];
      const std::size_t edge_end = graph.csr_run_edge_offsets[run + 1U];
      for (std::size_t edge = edge_begin; edge < edge_end; ++edge) {
        if (collect_light) {
          ++work.edges_examined;
        }
        if (collect_debug) {
          ++work.atomic_attempts;
        }
        const float source_distance =
            dense_atomic_bits_float(state.distances[source]);
        const float candidate = source_distance + graph.csr_weights[edge];
        const std::uint32_t candidate_bits = dense_atomic_float_bits(candidate);
        const std::uint32_t destination = graph.csr_destinations[edge];
        const DenseAtomicMinResult update =
            dense_atomic_min_bits(state.distances[destination], candidate_bits);
        if (!update.decreased) {
          continue;
        }
        state.distances[destination] = update.final_bits;
        if (collect_light) {
          ++work.successful_decreases;
        }
        if (collect_debug) {
          ++work.successful_atomic_updates;
        }

        if (state.enqueue_generation[destination] == generation) {
          if (collect_debug) {
            ++work.duplicate_suppressions;
          }
          continue;
        }
        state.enqueue_generation[destination] = generation;
        const std::uint64_t claim = controller.frontier_size[write_slot]++;
        controller.next_frontier_lane_mask |= standalone_lane;
        if (collect_debug) {
          ++work.queue_claims;
        }
        if (collect_light) {
          work.maximum_queue_size =
              std::max(work.maximum_queue_size, claim + 1U);
        }
        if (claim < state.queue_capacity) {
          state.frontiers[write_slot][static_cast<std::size_t>(claim)] =
              destination;
        } else {
          if (controller.error_bits == device_error::none && collect_debug) {
            ++work.overflow_events;
          }
          controller.error_bits = device_error::queue_overflow;
        }
      }
    }
  }
  if (collect_light && controller.frontier_size[write_slot] == 0U) {
    ++work.empty_frontier_rounds;
  }
}

void execute_pair(
    const DeviceGraphLayout32& graph,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csr_run_lane_masks,
    HostFrontierState& state,
    DeviceController& controller,
    DeviceWorkStatistics& work,
    std::vector<std::uint64_t>& frontier_trace,
    const InstrumentationLevel instrumentation) {
  execute_host_round(
      graph,
      tile_lane_masks,
      csr_run_lane_masks,
      state,
      controller,
      work,
      frontier_trace,
      instrumentation);
  static_cast<void>(advance_frontier_controller(controller));
}

[[nodiscard]] bool all_targets_reached(
    const RouteQuery& query,
    const std::vector<std::uint32_t>& distances) {
  for (const VertexId target : query.targets) {
    if (distances[target.value()] == positive_infinity_bits) {
      return false;
    }
  }
  return true;
}

}  // namespace

FrontierScratchLayout make_frontier_scratch_layout(
    const std::size_t vertex_count,
    const std::size_t requested_queue_capacity) {
  if (vertex_count == 0U) {
    throw std::invalid_argument{"frontier scratch requires at least one vertex"};
  }
  if (vertex_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{"frontier vertex count exceeds the device ABI"};
  }
  const std::size_t queue_capacity =
      requested_queue_capacity == 0U ? vertex_count : requested_queue_capacity;
  if (queue_capacity == 0U || queue_capacity > vertex_count ||
      queue_capacity > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument{
        "frontier queue capacity must be in [1, vertex_count]"};
  }

  const std::size_t distance_bytes = checked_multiply(
      vertex_count,
      sizeof(std::uint32_t),
      "frontier distance byte size overflow");
  const std::size_t queue_bytes = checked_multiply(
      queue_capacity,
      sizeof(std::uint32_t),
      "frontier queue byte size overflow");
  const std::size_t frontier_zero_offset = distance_bytes;
  const std::size_t frontier_one_offset = checked_add(
      frontier_zero_offset,
      queue_bytes,
      "frontier-one offset overflow");
  const std::size_t generation_offset = align_up(
      checked_add(
          frontier_one_offset,
          queue_bytes,
          "frontier generation offset overflow"),
      alignof(std::uint64_t));
  const std::size_t generation_bytes = checked_multiply(
      vertex_count,
      sizeof(std::uint64_t),
      "frontier generation byte size overflow");
  const std::size_t total_bytes = checked_add(
      generation_offset,
      generation_bytes,
      "frontier scratch total overflow");
  return FrontierScratchLayout{
      vertex_count,
      queue_capacity,
      0U,
      distance_bytes,
      {frontier_zero_offset, frontier_one_offset},
      queue_bytes,
      generation_offset,
      generation_bytes,
      total_bytes,
  };
}

HostFrontierRunResult run_host_frontier_push(
    const DeviceGraphLayout32& graph,
    const RouteQuery& query,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csr_run_lane_masks,
    const GpuRunOptions& options,
    const std::size_t requested_queue_capacity) {
  const FrontierScratchLayout layout = make_frontier_scratch_layout(
      graph.vertex_count, requested_queue_capacity);
  const std::size_t queue_capacity =
      static_cast<std::size_t>(layout.queue_capacity);
  validate_host_problem(
      graph,
      query,
      tile_lane_masks,
      csr_run_lane_masks,
      options,
      queue_capacity);

  HostFrontierRunResult output;
  HostFrontierState state;
  state.queue_capacity = queue_capacity;
  DeviceWorkStatistics work;
  initialize_host_state(
      graph, query, options, state, output.controller, work);

  if (options.control_mode == ControlMode::persistent_cooperative) {
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      execute_pair(
          graph,
          tile_lane_masks,
          csr_run_lane_masks,
          state,
          output.controller,
          work,
          output.current_frontier_sizes,
          options.instrumentation);
    }
    work.controller_copies = 1U;
    work.host_synchronizations = 1U;
    work.host_checks = 1U;
  } else {
    work.kernel_dispatches = 1U;
    const std::uint32_t chunk_size =
        options.control_mode == ControlMode::per_round_host_poll
            ? 1U
            : options.rounds_per_chunk;
    do {
      for (std::uint32_t pair = 0U; pair < chunk_size; ++pair) {
        work.kernel_dispatches += 2U;
        ++output.queued_round_pairs;
        execute_pair(
            graph,
            tile_lane_masks,
            csr_run_lane_masks,
            state,
            output.controller,
            work,
            output.current_frontier_sizes,
            options.instrumentation);
      }
      ++work.controller_copies;
      ++work.host_synchronizations;
      ++work.host_checks;
      ++output.completed_host_chunks;
    } while (output.controller.done == 0U);
  }

  output.distance_bits = std::move(state.distances);
  output.distances.reserve(output.distance_bits.size());
  for (const std::uint32_t bits : output.distance_bits) {
    output.distances.push_back(dense_atomic_bits_float(bits));
  }
  LaneMask reached = 0U;
  LaneMask miss = 0U;
  if (output.controller.stop_reason ==
      static_cast<std::uint32_t>(DeviceStopReason::converged)) {
    if (all_targets_reached(query, output.distance_bits)) {
      reached = standalone_lane;
    } else {
      miss = standalone_lane;
    }
  }
  output.result.engine_kind =
      static_cast<std::uint32_t>(EngineKind::frontier_push);
  output.result.control_mode = static_cast<std::uint32_t>(options.control_mode);
  output.result.status =
      make_frontier_run_status(output.controller, reached, miss);
  output.result.work = work;
  if (validate_device_controller(output.controller) !=
          DeviceControllerError::none ||
      validate_device_run_status(output.result.status) !=
          DeviceRunStatusError::none) {
    throw std::logic_error{
        "portable frontier execution produced invalid terminal state"};
  }
  return output;
}

}  // namespace bfnew
