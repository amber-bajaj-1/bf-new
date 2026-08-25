#include "bfnew/dense_chaotic_push.hpp"

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
  return offsets.size() == bucket_count + 1U && !offsets.empty() &&
         offsets.front() == 0U && offsets.back() == entry_count &&
         std::is_sorted(offsets.begin(), offsets.end());
}

void validate_host_problem(
    const DeviceGraphLayout32& graph,
    const RouteQuery& query,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csr_run_lane_masks,
    const GpuRunOptions& options) {
  require(
      validate_gpu_run_options(options) == GpuRunOptionsError::none,
      "host dense execution requires valid run options");
  require(
      options.engine == EngineKind::dense_chaotic_push,
      "host dense execution rejects another engine kind");
  require(graph.vertex_count != 0U, "host dense graph must contain a vertex");

  const std::size_t vertex_count = graph.vertex_count;
  const std::size_t edge_count = graph.edge_count;
  const std::size_t tile_count = graph.tile_count;
  const std::size_t csr_run_count = graph.csr_run_destination_tiles.size();
  const std::size_t csc_run_count = graph.csc_run_source_tiles.size();
  require(
      graph.owner_tiles.size() == vertex_count,
      "host dense owner-tile shape mismatch");
  require(
      graph.csr_destinations.size() == edge_count &&
          graph.csr_weights.size() == edge_count,
      "host dense CSR edge-field shape mismatch");
  require(
      valid_offsets(graph.csr_row_offsets, vertex_count, edge_count),
      "host dense CSR row offsets are invalid");
  require(
      valid_offsets(graph.csr_row_run_offsets, vertex_count, csr_run_count),
      "host dense CSR row-run offsets are invalid");
  require(
      valid_offsets(graph.csr_run_edge_offsets, csr_run_count, edge_count),
      "host dense CSR run-edge offsets are invalid");
  require(
      tile_lane_masks.size() == tile_count,
      "host dense tile-mask shape mismatch");
  require(
      csr_run_lane_masks.size() == csr_run_count,
      "host dense run-mask shape mismatch");
  require(
      graph.csc_sources.size() == edge_count &&
          valid_offsets(graph.csc_column_offsets, vertex_count, edge_count) &&
          valid_offsets(
              graph.csc_column_run_offsets, vertex_count, csc_run_count) &&
          valid_offsets(
              graph.csc_run_edge_offsets, csc_run_count, edge_count),
      "host dense CSC shape is invalid for contention instrumentation");
  require(
      !query.sources.empty() && !query.targets.empty() &&
          !query.selected_tiles.empty(),
      "host dense query sets must be populated");

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

  for (std::size_t source = 0U; source < vertex_count; ++source) {
    const std::uint32_t owner = graph.owner_tiles[source];
    require(owner < graph.tile_count, "vertex owner tile is out of range");
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
      const std::uint32_t remote_tile = graph.csr_run_destination_tiles[run];
      require(remote_tile < graph.tile_count, "CSR run tile is out of range");
      require(
          csr_run_lane_masks[run] ==
              (tile_lane_masks[owner] & tile_lane_masks[remote_tile]),
          "CSR run mask differs from endpoint-tile admission");
      for (std::size_t edge = begin; edge < end; ++edge) {
        const std::uint32_t destination = graph.csr_destinations[edge];
        require(destination < graph.vertex_count, "CSR destination is out of range");
        require(
            graph.owner_tiles[destination] == remote_tile,
            "CSR run destination-tile metadata is inconsistent");
        require(
            std::isfinite(graph.csr_weights[edge]) &&
                graph.csr_weights[edge] >= 0.0F &&
                !(graph.csr_weights[edge] == 0.0F &&
                  std::signbit(graph.csr_weights[edge])),
            "CSR contains an invalid weight");
      }
      cursor = end;
    }
    require(cursor == edge_end, "CSR runs do not cover their source row");
  }
}

[[nodiscard]] std::uint64_t count_high_contention_destinations(
    const DeviceGraphLayout32& graph,
    const std::span<const LaneMask> tile_lane_masks) {
  std::uint64_t count = 0U;
  for (std::size_t destination = 0U; destination < graph.vertex_count;
       ++destination) {
    const std::uint32_t owner = graph.owner_tiles[destination];
    if ((tile_lane_masks[owner] & standalone_lane) == 0U) {
      continue;
    }
    std::uint64_t admitted_incoming = 0U;
    const std::size_t run_begin = graph.csc_column_run_offsets[destination];
    const std::size_t run_end = graph.csc_column_run_offsets[destination + 1U];
    for (std::size_t run = run_begin; run < run_end; ++run) {
      const std::uint32_t source_tile = graph.csc_run_source_tiles[run];
      if ((tile_lane_masks[source_tile] & standalone_lane) == 0U) {
        continue;
      }
      admitted_incoming += graph.csc_run_edge_offsets[run + 1U] -
                           graph.csc_run_edge_offsets[run];
    }
    if (admitted_incoming >= 2U) {
      ++count;
    }
  }
  return count;
}

[[nodiscard]] bool execute_host_round(
    const DeviceGraphLayout32& graph,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csr_run_lane_masks,
    std::vector<std::uint32_t>& distances,
    DeviceController& controller,
    DeviceWorkStatistics& work,
    const InstrumentationLevel instrumentation,
    const DenseHostSchedule schedule) {
  if (controller.done != 0U || controller.execute_lane_mask == 0U) {
    return false;
  }
  if (controller.distance_read_slot != 0U ||
      controller.distance_write_slot != 0U) {
    detail::fail_dense_controller(controller);
    return false;
  }

  const bool collect_light = instrumentation != InstrumentationLevel::none;
  const bool collect_debug = instrumentation == InstrumentationLevel::debug;
  if (collect_light) {
    ++work.active_lane_rounds;
    ++work.full_edge_rounds;
  }

  const bool reverse = schedule == DenseHostSchedule::csr_reverse ||
                       (schedule == DenseHostSchedule::alternating &&
                        (controller.rounds_completed & 1U) != 0U);
  LaneMask changed = 0U;
  const std::size_t vertex_count = graph.vertex_count;
  for (std::size_t visit = 0U; visit < vertex_count; ++visit) {
    const std::size_t source = reverse ? vertex_count - 1U - visit : visit;
    const std::uint32_t source_tile = graph.owner_tiles[source];
    if ((tile_lane_masks[source_tile] & controller.execute_lane_mask) == 0U) {
      continue;
    }
    if (collect_light) {
      ++work.active_vertices;
    }

    const std::size_t run_begin = graph.csr_row_run_offsets[source];
    const std::size_t run_end = graph.csr_row_run_offsets[source + 1U];
    const std::size_t run_count = run_end - run_begin;
    for (std::size_t run_visit = 0U; run_visit < run_count; ++run_visit) {
      const std::size_t run =
          reverse ? run_end - 1U - run_visit : run_begin + run_visit;
      if (collect_debug) {
        ++work.mask_operations;
      }
      if ((csr_run_lane_masks[run] & controller.execute_lane_mask) == 0U) {
        continue;
      }
      const std::size_t edge_begin = graph.csr_run_edge_offsets[run];
      const std::size_t edge_end = graph.csr_run_edge_offsets[run + 1U];
      const std::size_t edge_count = edge_end - edge_begin;
      for (std::size_t edge_visit = 0U; edge_visit < edge_count; ++edge_visit) {
        const std::size_t edge =
            reverse ? edge_end - 1U - edge_visit : edge_begin + edge_visit;
        if (collect_light) {
          ++work.edges_examined;
        }
        if (collect_debug) {
          ++work.atomic_attempts;
        }
        // This direct read models the value returned by the HIP atomic CAS
        // load. No portable thread races exist in this semantic model.
        const float source_distance = dense_atomic_bits_float(distances[source]);
        const float candidate = source_distance + graph.csr_weights[edge];
        const std::uint32_t candidate_bits = dense_atomic_float_bits(candidate);
        const std::uint32_t destination = graph.csr_destinations[edge];
        const DenseAtomicMinResult update =
            dense_atomic_min_bits(distances[destination], candidate_bits);
        if (update.decreased) {
          distances[destination] = update.final_bits;
          changed |= standalone_lane;
          if (collect_light) {
            ++work.successful_decreases;
          }
          if (collect_debug) {
            ++work.successful_atomic_updates;
          }
        }
      }
    }
  }
  controller.changed_lane_mask |= changed;
  if (collect_debug && changed != 0U) {
    // The single-threaded semantic model publishes one changed flag per
    // changing round. Device execution records one publication per changed
    // block, so this counter is not expected to match across implementations.
    ++work.changed_flag_updates;
  }
  return true;
}

void execute_pair(
    const DeviceGraphLayout32& graph,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csr_run_lane_masks,
    std::vector<std::uint32_t>& distances,
    DeviceController& controller,
    DeviceWorkStatistics& work,
    const InstrumentationLevel instrumentation,
    const DenseHostSchedule schedule) {
  static_cast<void>(execute_host_round(
      graph,
      tile_lane_masks,
      csr_run_lane_masks,
      distances,
      controller,
      work,
      instrumentation,
      schedule));
  static_cast<void>(advance_dense_controller(controller));
}

[[nodiscard]] bool all_targets_reached(
    const RouteQuery& query,
    const std::vector<std::uint32_t>& distances) {
  for (const VertexId target : query.targets) {
    if (!std::isfinite(dense_atomic_bits_float(distances[target.value()]))) {
      return false;
    }
  }
  return true;
}

}  // namespace

DenseScratchLayout make_dense_scratch_layout(const std::size_t vertex_count) {
  if (vertex_count == 0U) {
    throw std::invalid_argument{"dense scratch requires at least one vertex"};
  }
  if (vertex_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{"dense vertex count exceeds the device ABI"};
  }
  const std::size_t bytes = checked_multiply(
      vertex_count,
      sizeof(std::uint32_t),
      "dense distance byte size overflow");
  return DenseScratchLayout{
      static_cast<std::uint64_t>(vertex_count),
      static_cast<std::uint64_t>(bytes),
      static_cast<std::uint64_t>(bytes),
  };
}

bool is_dense_atomic_float(const float value) noexcept {
  if (value == 0.0F && std::signbit(value)) {
    return false;
  }
  return (std::isfinite(value) && value >= 0.0F) ||
         (std::isinf(value) && !std::signbit(value));
}

std::uint32_t dense_atomic_float_bits(const float value) {
  if (!is_dense_atomic_float(value)) {
    throw std::invalid_argument{
        "dense atomic float must be finite nonnegative or positive infinity"};
  }
  return std::bit_cast<std::uint32_t>(value);
}

float dense_atomic_bits_float(const std::uint32_t bits) {
  if (bits > positive_infinity_bits) {
    throw std::invalid_argument{"dense atomic bits are outside the supported domain"};
  }
  return std::bit_cast<float>(bits);
}

DenseAtomicMinResult dense_atomic_min_bits(
    const std::uint32_t current_bits,
    const std::uint32_t candidate_bits) {
  static_cast<void>(dense_atomic_bits_float(current_bits));
  static_cast<void>(dense_atomic_bits_float(candidate_bits));
  if (candidate_bits < current_bits) {
    return DenseAtomicMinResult{candidate_bits, true};
  }
  return DenseAtomicMinResult{current_bits, false};
}

HostDenseRunResult run_host_dense_chaotic_push(
    const DeviceGraphLayout32& graph,
    const RouteQuery& query,
    const std::span<const LaneMask> tile_lane_masks,
    const std::span<const LaneMask> csr_run_lane_masks,
    const GpuRunOptions& options,
    const DenseHostSchedule schedule) {
  validate_host_problem(
      graph, query, tile_lane_masks, csr_run_lane_masks, options);
  static_cast<void>(make_dense_scratch_layout(graph.vertex_count));

  HostDenseRunResult output;
  output.distance_bits.assign(graph.vertex_count, positive_infinity_bits);
  for (const VertexId source : query.sources) {
    output.distance_bits[source.value()] = 0U;
  }
  output.controller = initialize_device_controller(options, standalone_lane);

  DeviceWorkStatistics work;
  if (options.instrumentation == InstrumentationLevel::debug) {
    work.high_contention_destinations =
        count_high_contention_destinations(graph, tile_lane_masks);
  }

  if (options.control_mode == ControlMode::persistent_cooperative) {
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      execute_pair(
          graph,
          tile_lane_masks,
          csr_run_lane_masks,
          output.distance_bits,
          output.controller,
          work,
          options.instrumentation,
          schedule);
    }
    work.controller_copies = 1U;
    work.host_synchronizations = 1U;
    work.host_checks = 1U;
  } else if (options.control_mode == ControlMode::per_round_host_poll) {
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      work.kernel_dispatches += 2U;
      ++output.queued_round_pairs;
      execute_pair(
          graph,
          tile_lane_masks,
          csr_run_lane_masks,
          output.distance_bits,
          output.controller,
          work,
          options.instrumentation,
          schedule);
      ++work.controller_copies;
      ++work.host_synchronizations;
      ++work.host_checks;
      ++output.completed_host_chunks;
    }
  } else {
    work.kernel_dispatches = 1U;
    while (output.controller.done == 0U) {
      for (std::uint32_t pair = 0U; pair < options.rounds_per_chunk; ++pair) {
        work.kernel_dispatches += 2U;
        ++output.queued_round_pairs;
        execute_pair(
            graph,
            tile_lane_masks,
            csr_run_lane_masks,
            output.distance_bits,
            output.controller,
            work,
            options.instrumentation,
            schedule);
      }
      ++work.controller_copies;
      ++work.host_synchronizations;
      ++work.host_checks;
      ++output.completed_host_chunks;
    }
  }

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
      static_cast<std::uint32_t>(EngineKind::dense_chaotic_push);
  output.result.control_mode = static_cast<std::uint32_t>(options.control_mode);
  output.result.status = make_dense_run_status(output.controller, reached, miss);
  output.result.work = work;
  if (validate_device_controller(output.controller) != DeviceControllerError::none ||
      validate_device_run_status(output.result.status) !=
          DeviceRunStatusError::none) {
    throw std::logic_error{
        "portable dense execution produced invalid terminal state"};
  }
  return output;
}

}  // namespace bfnew
