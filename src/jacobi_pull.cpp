#include "bfnew/jacobi_pull.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bfnew {
namespace {

constexpr LaneMask standalone_lane = LaneMask{1U};

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

void require(const bool condition, const char* const description) {
  if (!condition) {
    throw std::invalid_argument{description};
  }
}

[[nodiscard]] bool valid_offsets(
    const std::span<const std::uint32_t> offsets,
    const std::size_t bucket_count,
    const std::size_t entry_count) noexcept {
  if (offsets.size() != bucket_count + 1U || offsets.empty() ||
      offsets.front() != 0U ||
      static_cast<std::size_t>(offsets.back()) != entry_count) {
    return false;
  }
  return std::is_sorted(offsets.begin(), offsets.end()) &&
         static_cast<std::size_t>(offsets.back()) == entry_count;
}

void validate_host_problem(
    const DeviceGraphLayout32& graph,
    const RouteQuery& query,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csc_run_lane_masks,
    const GpuRunOptions& options) {
  require(
      validate_gpu_run_options(options) == GpuRunOptionsError::none,
      "host Jacobi execution requires valid run options");
  require(
      options.engine == EngineKind::jacobi_pull,
      "host Jacobi execution rejects another engine kind");
  require(graph.vertex_count != 0U, "host Jacobi graph must contain a vertex");
  const std::size_t vertex_count = graph.vertex_count;
  const std::size_t edge_count = graph.edge_count;
  const std::size_t tile_count = graph.tile_count;
  const std::size_t run_count = graph.csc_run_source_tiles.size();
  require(
      graph.owner_tiles.size() == vertex_count,
      "host Jacobi owner-tile shape mismatch");
  require(
      graph.csc_sources.size() == edge_count &&
          graph.csc_weights.size() == edge_count,
      "host Jacobi CSC edge-field shape mismatch");
  require(
      valid_offsets(graph.csc_column_offsets, vertex_count, edge_count),
      "host Jacobi CSC column offsets are invalid");
  require(
      valid_offsets(graph.csc_column_run_offsets, vertex_count, run_count),
      "host Jacobi CSC column-run offsets are invalid");
  require(
      graph.csc_run_edge_offsets.size() == run_count + 1U &&
          !graph.csc_run_edge_offsets.empty() &&
          graph.csc_run_edge_offsets.front() == 0U &&
          static_cast<std::size_t>(graph.csc_run_edge_offsets.back()) == edge_count &&
          std::is_sorted(
              graph.csc_run_edge_offsets.begin(),
              graph.csc_run_edge_offsets.end()),
      "host Jacobi CSC run-edge offsets are invalid");
  require(
      tile_lane_masks.size() == tile_count,
      "host Jacobi tile-mask shape mismatch");
  require(
      csc_run_lane_masks.size() == run_count,
      "host Jacobi run-mask shape mismatch");
  require(
      !query.sources.empty() && !query.targets.empty() &&
          !query.selected_tiles.empty(),
      "host Jacobi query sets must be populated");

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
    const LaneMask expected = selected[tile] ? standalone_lane : LaneMask{0U};
    require(
        tile_lane_masks[tile] == expected,
        "standalone tile mask does not match selected tiles");
  }

  const auto require_terminal = [&](const VertexId vertex) {
    require(vertex.value() < graph.vertex_count, "query terminal is out of range");
    const std::uint32_t owner = graph.owner_tiles[vertex.value()];
    require(owner < graph.tile_count, "query terminal owner tile is out of range");
    require(selected[owner], "query terminal owner tile is not selected");
  };
  for (const VertexId source : query.sources) {
    require_terminal(source);
  }
  for (const VertexId target : query.targets) {
    require_terminal(target);
  }

  for (std::size_t vertex = 0U; vertex < vertex_count; ++vertex) {
    const std::uint32_t owner = graph.owner_tiles[vertex];
    require(owner < graph.tile_count, "vertex owner tile is out of range");
    const std::size_t edge_begin = graph.csc_column_offsets[vertex];
    const std::size_t edge_end = graph.csc_column_offsets[vertex + 1U];
    const std::size_t run_begin = graph.csc_column_run_offsets[vertex];
    const std::size_t run_end = graph.csc_column_run_offsets[vertex + 1U];
    std::size_t cursor = edge_begin;
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const std::size_t begin = graph.csc_run_edge_offsets[run];
      const std::size_t end = graph.csc_run_edge_offsets[run + 1U];
      require(
          begin == cursor && begin < end && end <= edge_end,
          "CSC run does not exactly cover its destination column");
      const std::uint32_t remote_tile = graph.csc_run_source_tiles[run];
      require(remote_tile < graph.tile_count, "CSC run source tile is out of range");
      const LaneMask expected_mask = tile_lane_masks[owner] &
                                     tile_lane_masks[remote_tile];
      require(
          csc_run_lane_masks[run] == expected_mask,
          "CSC run mask differs from endpoint-tile admission");
      for (std::size_t edge = begin; edge < end; ++edge) {
        const std::uint32_t source = graph.csc_sources[edge];
        require(source < graph.vertex_count, "CSC source vertex is out of range");
        require(
            graph.owner_tiles[source] == remote_tile,
            "CSC run source-tile metadata is inconsistent");
        require(
            std::isfinite(graph.csc_weights[edge]) && graph.csc_weights[edge] >= 0.0F,
            "CSC contains an invalid weight");
      }
      cursor = end;
    }
    require(cursor == edge_end, "CSC runs do not cover their destination column");
  }
}

[[nodiscard]] bool execute_host_round(
    const DeviceGraphLayout32& graph,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csc_run_lane_masks,
    std::array<std::vector<float>, 2>& slots,
    DeviceController& controller,
    DeviceWorkStatistics& work,
    const InstrumentationLevel instrumentation) {
  if (controller.done != 0U || controller.execute_lane_mask == 0U) {
    return false;
  }

  const std::size_t read_slot = controller.distance_read_slot;
  const std::size_t write_slot = controller.distance_write_slot;
  if (read_slot > 1U || write_slot > 1U || read_slot == write_slot) {
    detail::fail_jacobi_controller(controller);
    return false;
  }
  const std::vector<float>& old_distances = slots[read_slot];
  std::vector<float>& next_distances = slots[write_slot];
  LaneMask changed = 0U;
  const bool collect_light = instrumentation != InstrumentationLevel::none;
  const bool collect_debug = instrumentation == InstrumentationLevel::debug;
  if (collect_light) {
    ++work.active_lane_rounds;
  }

  for (std::size_t destination = 0U; destination < graph.vertex_count;
       ++destination) {
    const std::uint32_t owner = graph.owner_tiles[destination];
    if ((tile_lane_masks[owner] & controller.execute_lane_mask) == 0U) {
      continue;
    }
    if (collect_light) {
      ++work.active_vertices;
    }
    float best = old_distances[destination];
    const std::size_t run_begin = graph.csc_column_run_offsets[destination];
    const std::size_t run_end = graph.csc_column_run_offsets[destination + 1U];
    for (std::size_t run = run_begin; run < run_end; ++run) {
      if (collect_debug) {
        ++work.mask_operations;
      }
      if ((csc_run_lane_masks[run] & controller.execute_lane_mask) == 0U) {
        continue;
      }
      const std::size_t edge_begin = graph.csc_run_edge_offsets[run];
      const std::size_t edge_end = graph.csc_run_edge_offsets[run + 1U];
      for (std::size_t edge = edge_begin; edge < edge_end; ++edge) {
        if (collect_light) {
          ++work.edges_examined;
        }
        const std::uint32_t source = graph.csc_sources[edge];
        const float candidate = old_distances[source] + graph.csc_weights[edge];
        if (candidate < best) {
          best = candidate;
        }
      }
    }
    next_distances[destination] = best;
    if (best < old_distances[destination]) {
      changed |= standalone_lane;
      if (collect_light) {
        ++work.successful_decreases;
      }
    }
  }
  controller.changed_lane_mask |= changed;
  return true;
}

void execute_pair(
    const DeviceGraphLayout32& graph,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csc_run_lane_masks,
    std::array<std::vector<float>, 2>& slots,
    DeviceController& controller,
    DeviceWorkStatistics& work,
    const InstrumentationLevel instrumentation) {
  static_cast<void>(execute_host_round(
      graph,
      tile_lane_masks,
      csc_run_lane_masks,
      slots,
      controller,
      work,
      instrumentation));
  static_cast<void>(advance_jacobi_controller(controller));
}

[[nodiscard]] bool all_targets_reached(
    const RouteQuery& query,
    const std::vector<float>& distances) noexcept {
  for (const VertexId target : query.targets) {
    if (!std::isfinite(distances[target.value()])) {
      return false;
    }
  }
  return true;
}

}  // namespace

JacobiScratchLayout make_jacobi_scratch_layout(const std::size_t vertex_count) {
  if (vertex_count == 0U) {
    throw std::invalid_argument{"Jacobi scratch requires at least one vertex"};
  }
  if (vertex_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{"Jacobi vertex count exceeds the device ABI"};
  }
  const std::size_t slot_bytes = checked_multiply(
      vertex_count, sizeof(float), "Jacobi distance-slot byte size overflow");
  const std::size_t total_bytes = checked_add(
      slot_bytes, slot_bytes, "Jacobi two-slot byte size overflow");
  return JacobiScratchLayout{
      static_cast<std::uint64_t>(vertex_count),
      static_cast<std::uint64_t>(slot_bytes),
      {0U, static_cast<std::uint64_t>(slot_bytes)},
      static_cast<std::uint64_t>(total_bytes),
  };
}

HostJacobiRunResult run_host_jacobi_pull(
    const DeviceGraphLayout32& graph,
    const RouteQuery& query,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csc_run_lane_masks,
    const GpuRunOptions& options) {
  validate_host_problem(
      graph, query, tile_lane_masks, csc_run_lane_masks, options);
  static_cast<void>(make_jacobi_scratch_layout(graph.vertex_count));

  HostJacobiRunResult output;
  const float infinity = std::numeric_limits<float>::infinity();
  output.distance_slots[0].assign(graph.vertex_count, infinity);
  output.distance_slots[1].assign(graph.vertex_count, infinity);
  for (const VertexId source : query.sources) {
    output.distance_slots[0][source.value()] = 0.0F;
    output.distance_slots[1][source.value()] = 0.0F;
  }
  output.controller = initialize_device_controller(options, standalone_lane);

  DeviceWorkStatistics work;
  if (options.control_mode == ControlMode::persistent_cooperative) {
    // The real persistent path must fold initialization and all rounds into
    // this one cooperative dispatch.
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      execute_pair(
          graph,
          tile_lane_masks,
          csc_run_lane_masks,
          output.distance_slots,
          output.controller,
          work,
          options.instrumentation);
    }
    work.controller_copies = 1U;
    work.host_synchronizations = 1U;
    work.host_checks = 1U;
  } else if (options.control_mode == ControlMode::per_round_host_poll) {
    // Ordinary execution requires one explicit distance initialization launch
    // before its exact (round, advance) pairs.
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      work.kernel_dispatches += 2U;
      ++output.queued_round_pairs;
      execute_pair(
          graph,
          tile_lane_masks,
          csc_run_lane_masks,
          output.distance_slots,
          output.controller,
          work,
          options.instrumentation);
      ++work.controller_copies;
      ++work.host_synchronizations;
      ++work.host_checks;
      ++output.completed_host_chunks;
    }
  } else {
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      // Match the HIP stream protocol exactly: enqueue the complete K-pair
      // chunk, even when convergence or the round limit occurs inside it.
      // Device-done state makes the remaining already-queued pairs no-ops.
      const std::uint64_t pairs = options.rounds_per_chunk;
      for (std::uint64_t pair = 0U; pair < pairs; ++pair) {
        work.kernel_dispatches += 2U;
        ++output.queued_round_pairs;
        execute_pair(
            graph,
            tile_lane_masks,
            csc_run_lane_masks,
            output.distance_slots,
            output.controller,
            work,
            options.instrumentation);
      }
      ++work.controller_copies;
      ++work.host_synchronizations;
      ++work.host_checks;
      ++output.completed_host_chunks;
    }
  }

  const std::size_t final_slot = output.controller.distance_read_slot;
  output.distances = output.distance_slots[final_slot];
  LaneMask reached = 0U;
  LaneMask miss = 0U;
  if (output.controller.stop_reason ==
      static_cast<std::uint32_t>(DeviceStopReason::converged)) {
    if (all_targets_reached(query, output.distances)) {
      reached = standalone_lane;
    } else {
      miss = standalone_lane;
    }
  }
  output.result.engine_kind = static_cast<std::uint32_t>(EngineKind::jacobi_pull);
  output.result.control_mode = static_cast<std::uint32_t>(options.control_mode);
  output.result.status = make_jacobi_run_status(output.controller, reached, miss);
  output.result.work = work;
  if (validate_device_controller(output.controller) != DeviceControllerError::none ||
      validate_device_run_status(output.result.status) != DeviceRunStatusError::none) {
    throw std::logic_error{"portable Jacobi execution produced invalid terminal state"};
  }
  return output;
}

}  // namespace bfnew
