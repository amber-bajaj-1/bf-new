#include "bfnew/hip/compact_path_results.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bfnew::hip {
namespace {

inline constexpr std::uint32_t positive_infinity_bits = 0x7f800000U;

namespace reconstruction_error {

inline constexpr std::uint32_t invalid_shape = 1U << 0U;
inline constexpr std::uint32_t replay_mismatch = 1U << 1U;
inline constexpr std::uint32_t arena_overflow = 1U << 2U;

}  // namespace reconstruction_error

struct DeviceCompactTargetSummary {
  std::uint32_t target{};
  std::uint32_t selected_source{};
  float distance{};
  std::uint32_t path_length{};
  std::uint32_t reached{};
  std::uint32_t reconstruction{};
  std::uint32_t has_selected_source{};
};

struct DeviceCompactOutputRange {
  std::uint64_t vertex_offset{};
  std::uint64_t edge_offset{};
};

static_assert(sizeof(DeviceCompactTargetSummary) ==
              sizeof(CompactTargetSummary));
static_assert(alignof(DeviceCompactTargetSummary) ==
              alignof(CompactTargetSummary));
static_assert(offsetof(DeviceCompactTargetSummary, target) ==
              offsetof(CompactTargetSummary, target));
static_assert(offsetof(DeviceCompactTargetSummary, selected_source) ==
              offsetof(CompactTargetSummary, selected_source));
static_assert(offsetof(DeviceCompactTargetSummary, distance) ==
              offsetof(CompactTargetSummary, distance));
static_assert(offsetof(DeviceCompactTargetSummary, path_length) ==
              offsetof(CompactTargetSummary, path_length));
static_assert(offsetof(DeviceCompactTargetSummary, reached) ==
              offsetof(CompactTargetSummary, reached));
static_assert(offsetof(DeviceCompactTargetSummary, reconstruction) ==
              offsetof(CompactTargetSummary, reconstruction));
static_assert(offsetof(DeviceCompactTargetSummary, has_selected_source) ==
              offsetof(CompactTargetSummary, has_selected_source));

[[nodiscard]] std::size_t checked_multiply(
    const std::size_t count,
    const std::size_t width,
    const std::string_view what) {
  if (width != 0U &&
      count > std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  return count * width;
}

[[nodiscard]] std::uint64_t checked_add_u64(
    const std::uint64_t left,
    const std::uint64_t right,
    const std::string_view what) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  return left + right;
}

template <typename T>
[[nodiscard]] T* device_pointer(void* const pointer) noexcept {
  return static_cast<T*>(pointer);
}

template <typename T>
[[nodiscard]] const T* device_pointer(const void* const pointer) noexcept {
  return static_cast<const T*>(pointer);
}

[[nodiscard]] __device__ float bits_to_float(
    const std::uint32_t bits) noexcept {
  return __builtin_bit_cast(float, bits);
}

[[nodiscard]] __device__ std::uint32_t float_to_bits(
    const float value) noexcept {
  return __builtin_bit_cast(std::uint32_t, value);
}

[[nodiscard]] __device__ bool finite_nonnegative(
    const float value) noexcept {
  const std::uint32_t bits = float_to_bits(value);
  return (bits & 0x80000000U) == 0U &&
         (bits & 0x7f800000U) != 0x7f800000U;
}

[[nodiscard]] __device__ float load_distance(
    const DeviceCompactDistanceMatrix distances,
    const std::uint32_t vertex,
    const std::uint32_t lane) noexcept {
  const std::uint64_t element =
      static_cast<std::uint64_t>(vertex) * distances.lane_width + lane;
  if (distances.encoding == CompactDistanceEncoding::floating_point) {
    return distances.floating_point[element];
  }
  return bits_to_float(distances.nonnegative_float_bits[element]);
}

[[nodiscard]] __device__ bool lane_admits_vertex(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const std::uint32_t lane,
    const std::uint32_t vertex) noexcept {
  if (vertex >= graph.vertex_count) {
    return false;
  }
  const std::uint32_t tile = graph.owner_tiles[vertex];
  return tile < graph.tile_count && tile < workspace.tile_lane_mask_count &&
         (workspace.tile_lane_masks[tile] & (LaneMask{1U} << lane)) != 0U;
}

[[nodiscard]] __device__ bool lane_source(
    const DeviceWorkspaceView workspace,
    const DeviceCompactBatchView batch,
    const std::uint32_t lane,
    const std::uint32_t vertex) noexcept {
  const std::uint32_t begin = batch.source_offsets[lane];
  const std::uint32_t end = batch.source_offsets[lane + 1U];
  for (std::uint32_t position = begin; position < end; ++position) {
    if (workspace.sources[position] == vertex) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] __device__ bool vertex_is_on_stack(
    const std::uint32_t* const stack_vertices,
    const std::uint32_t depth,
    const std::uint32_t vertex) noexcept {
  for (std::uint32_t position = 0U; position <= depth; ++position) {
    if (stack_vertices[position] == vertex) {
      return true;
    }
  }
  return false;
}

struct DeviceSearchResult {
  std::uint32_t path_length{};
  std::uint32_t selected_source{};
  std::uint32_t complete{};
  std::uint32_t overflow{};
};

[[nodiscard]] __device__ DeviceSearchResult search_tight_path(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceCompactBatchView batch,
    const DeviceCompactDistanceMatrix distances,
    const std::uint32_t lane,
    const std::uint32_t target,
    std::uint32_t* const stack_vertices,
    std::uint32_t* const stack_edge_ids,
    std::uint32_t* const next_edge_thresholds) noexcept {
  DeviceSearchResult result{};
  if (graph.vertex_count == 0U ||
      !lane_admits_vertex(graph, workspace, lane, target) ||
      !finite_nonnegative(load_distance(distances, target, lane))) {
    return result;
  }

  std::uint32_t depth = 0U;
  stack_vertices[0U] = target;
  next_edge_thresholds[0U] = 0U;
  while (true) {
    const std::uint32_t current = stack_vertices[depth];
    if (lane_source(workspace, batch, lane, current)) {
      result.path_length = depth;
      result.selected_source = current;
      result.complete = 1U;
      return result;
    }

    const float current_distance = load_distance(distances, current, lane);
    const std::uint32_t column_begin = graph.csc.column_offsets[current];
    const std::uint32_t column_end = graph.csc.column_offsets[current + 1U];
    const std::uint32_t threshold = next_edge_thresholds[depth];
    bool found = false;
    std::uint32_t best_edge_id = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t best_predecessor = 0U;
    for (std::uint32_t position = column_begin; position < column_end;
         ++position) {
      const std::uint32_t edge_id = graph.csc.edge_ids[position];
      if (edge_id < threshold || (found && edge_id >= best_edge_id)) {
        continue;
      }
      const std::uint32_t predecessor = graph.csc.sources[position];
      if (!lane_admits_vertex(graph, workspace, lane, predecessor) ||
          vertex_is_on_stack(stack_vertices, depth, predecessor)) {
        continue;
      }
      const float predecessor_distance =
          load_distance(distances, predecessor, lane);
      if (!finite_nonnegative(predecessor_distance) ||
          predecessor_distance + graph.csc.weights[position] !=
              current_distance) {
        continue;
      }
      found = true;
      best_edge_id = edge_id;
      best_predecessor = predecessor;
    }

    if (found) {
      // A representable graph has edge IDs in [0,E), with E<=UINT32_MAX,
      // hence the largest ID is at most UINT32_MAX-1 and this increment is
      // the exact next stable-ID search threshold.
      next_edge_thresholds[depth] = best_edge_id + 1U;
      stack_edge_ids[depth] = best_edge_id;
      if (depth + 1U >= graph.vertex_count) {
        result.overflow = 1U;
        return result;
      }
      ++depth;
      stack_vertices[depth] = best_predecessor;
      next_edge_thresholds[depth] = 0U;
      continue;
    }

    if (depth == 0U) {
      return result;
    }
    --depth;
  }
}

__global__ void compact_path_reconstruction_kernel(
    const DeviceGraphView32 graph,
    const DeviceWorkspaceView workspace,
    const DeviceCompactBatchView batch,
    const DeviceCompactDistanceMatrix distances,
    DeviceRunStatus* const status_snapshot,
    DeviceCompactTargetSummary* const summaries,
    std::uint32_t* const stack_vertices,
    std::uint32_t* const stack_edge_ids,
    std::uint32_t* const next_edge_thresholds,
    const DeviceCompactOutputRange* const output_ranges,
    const std::uint64_t vertex_capacity,
    const std::uint64_t edge_capacity,
    std::uint32_t* const output_vertices,
    float* const output_distance_labels,
    std::uint32_t* const output_edge_ids,
    const std::uint32_t emit_paths,
    std::uint32_t* const error_bits) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  if (workspace.status == nullptr || status_snapshot == nullptr ||
      summaries == nullptr || stack_vertices == nullptr ||
      stack_edge_ids == nullptr || next_edge_thresholds == nullptr ||
      error_bits == nullptr ||
      (graph.edge_count != 0U && graph.csc.edge_ids == nullptr)) {
    *error_bits = reconstruction_error::invalid_shape;
    return;
  }
  *status_snapshot = *workspace.status;
  const DeviceRunStatus status = *workspace.status;
  if (distances.slot_count == 0U ||
      status.final_distance_slot >= distances.slot_count) {
    *error_bits = reconstruction_error::invalid_shape;
    return;
  }
  DeviceCompactDistanceMatrix resolved_distances = distances;
  const std::uint64_t slot_offset =
      static_cast<std::uint64_t>(status.final_distance_slot) *
      distances.slot_stride_elements;
  if (resolved_distances.encoding ==
      CompactDistanceEncoding::floating_point) {
    resolved_distances.floating_point += slot_offset;
  } else {
    resolved_distances.nonnegative_float_bits += slot_offset;
  }
  resolved_distances.slot_count = 1U;
  resolved_distances.slot_stride_elements = 0U;
  const bool clean =
      status.stop_reason ==
          static_cast<std::uint32_t>(DeviceStopReason::converged) &&
      status.error_bits == device_error::none;

  std::uint32_t target_cursor = 0U;
  for (std::uint32_t lane = 0U; lane < batch.lane_width; ++lane) {
    const std::uint32_t target_begin = batch.target_offsets[lane];
    const std::uint32_t target_end = batch.target_offsets[lane + 1U];
    if (target_begin != target_cursor || target_end < target_begin ||
        target_end > workspace.target_count) {
      *error_bits |= reconstruction_error::invalid_shape;
      return;
    }
    for (std::uint32_t target_index = target_begin;
         target_index < target_end;
         ++target_index) {
      const std::uint32_t target = workspace.targets[target_index];
      if (emit_paths == 0U) {
        DeviceCompactTargetSummary summary{};
        summary.target = target;
        summary.distance = bits_to_float(positive_infinity_bits);
        summary.reached = static_cast<std::uint32_t>(
            CompactTargetReachStatus::not_reached);
        summary.reconstruction = static_cast<std::uint32_t>(
            CompactPathStatus::query_terminal_failure);
        if (clean) {
          summary.distance = load_distance(resolved_distances, target, lane);
          if (finite_nonnegative(summary.distance)) {
            summary.reached = static_cast<std::uint32_t>(
                CompactTargetReachStatus::reached);
            const DeviceSearchResult search = search_tight_path(
                graph,
                workspace,
                batch,
                resolved_distances,
                lane,
                target,
                stack_vertices,
                stack_edge_ids,
                next_edge_thresholds);
            if (search.complete != 0U) {
              summary.selected_source = search.selected_source;
              summary.path_length = search.path_length;
              summary.reconstruction =
                  static_cast<std::uint32_t>(CompactPathStatus::complete);
              summary.has_selected_source = 1U;
            } else if (search.overflow != 0U) {
              summary.reconstruction = static_cast<std::uint32_t>(
                  CompactPathStatus::path_length_overflow);
            } else {
              summary.reconstruction =
                  static_cast<std::uint32_t>(CompactPathStatus::no_tight_path);
            }
          } else {
            summary.reconstruction =
                static_cast<std::uint32_t>(CompactPathStatus::unreachable);
          }
        }
        summaries[target_index] = summary;
      } else {
        const DeviceCompactTargetSummary summary = summaries[target_index];
        if (summary.reconstruction !=
            static_cast<std::uint32_t>(CompactPathStatus::complete)) {
          continue;
        }
        const DeviceSearchResult search = search_tight_path(
            graph,
            workspace,
            batch,
            resolved_distances,
            lane,
            target,
            stack_vertices,
            stack_edge_ids,
            next_edge_thresholds);
        if (search.complete == 0U ||
            search.path_length != summary.path_length ||
            search.selected_source != summary.selected_source ||
            output_ranges == nullptr) {
          *error_bits |= reconstruction_error::replay_mismatch;
          continue;
        }
        const DeviceCompactOutputRange range = output_ranges[target_index];
        const std::uint64_t vertex_count =
            static_cast<std::uint64_t>(search.path_length) + 1U;
        if (range.vertex_offset > vertex_capacity ||
            vertex_count > vertex_capacity - range.vertex_offset ||
            range.edge_offset > edge_capacity ||
            search.path_length > edge_capacity - range.edge_offset ||
            output_vertices == nullptr || output_distance_labels == nullptr ||
            (search.path_length != 0U && output_edge_ids == nullptr)) {
          *error_bits |= reconstruction_error::arena_overflow;
          continue;
        }
        for (std::uint32_t path_vertex = 0U;
             path_vertex <= search.path_length;
             ++path_vertex) {
          const std::uint32_t stack_position =
              search.path_length - path_vertex;
          const std::uint32_t vertex = stack_vertices[stack_position];
          const std::uint64_t output = range.vertex_offset + path_vertex;
          output_vertices[output] = vertex;
          output_distance_labels[output] =
              load_distance(resolved_distances, vertex, lane);
        }
        for (std::uint32_t path_edge = 0U;
             path_edge < search.path_length;
             ++path_edge) {
          const std::uint32_t stack_position =
              search.path_length - 1U - path_edge;
          output_edge_ids[range.edge_offset + path_edge] =
              stack_edge_ids[stack_position];
        }
      }
    }
    target_cursor = target_end;
  }
  if (target_cursor != workspace.target_count) {
    *error_bits |= reconstruction_error::invalid_shape;
  }
}

[[nodiscard]] bool valid_encoding(
    const CompactDistanceEncoding encoding) noexcept {
  return encoding == CompactDistanceEncoding::floating_point ||
         encoding == CompactDistanceEncoding::nonnegative_float_bits;
}

[[nodiscard]] bool valid_summary_enums(
    const CompactTargetSummary& summary) noexcept {
  const bool valid_reach =
      summary.reached == CompactTargetReachStatus::not_reached ||
      summary.reached == CompactTargetReachStatus::reached;
  const bool valid_path =
      summary.reconstruction == CompactPathStatus::complete ||
      summary.reconstruction == CompactPathStatus::unreachable ||
      summary.reconstruction == CompactPathStatus::query_terminal_failure ||
      summary.reconstruction == CompactPathStatus::no_tight_path ||
      summary.reconstruction == CompactPathStatus::path_length_overflow;
  return valid_reach && valid_path;
}

void validate_host_shape(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceCompactBatchView& batch_view,
    const DeviceCompactDistanceMatrix& distances,
    const BatchDeviceDescription& host_batch) {
  const bool floating =
      distances.encoding == CompactDistanceEncoding::floating_point;
  if (!valid_encoding(distances.encoding) || graph.vertex_count == 0U ||
      graph.vertex_count != distances.vertex_count ||
      graph.csc.vertex_count != graph.vertex_count ||
      graph.csc.edge_count != graph.edge_count ||
      graph.csc.column_offsets == nullptr ||
      (graph.edge_count != 0U &&
       (graph.csc.sources == nullptr || graph.csc.weights == nullptr ||
        graph.csc.edge_ids == nullptr)) ||
      graph.owner_tiles == nullptr || workspace.sources == nullptr ||
      workspace.targets == nullptr || workspace.tile_lane_masks == nullptr ||
      workspace.status == nullptr || batch_view.source_offsets == nullptr ||
      batch_view.target_offsets == nullptr ||
      batch_view.lane_width != host_batch.lane_width ||
      batch_view.valid_lane_mask != host_batch.valid_lane_mask ||
      distances.lane_width != host_batch.lane_width ||
      distances.slot_count == 0U || distances.slot_count > 2U ||
      distances.slot_stride_elements !=
          static_cast<std::uint64_t>(graph.vertex_count) *
              distances.lane_width ||
      (floating ? distances.floating_point == nullptr
                : distances.nonnegative_float_bits == nullptr) ||
      (floating ? distances.nonnegative_float_bits != nullptr
                : distances.floating_point != nullptr) ||
      workspace.source_count != host_batch.sources.size() ||
      workspace.target_count != host_batch.targets.size() ||
      workspace.tile_lane_mask_count != host_batch.tile_lane_masks.size() ||
      host_batch.source_offsets.size() != host_batch.lane_width + 1U ||
      host_batch.target_offsets.size() != host_batch.lane_width + 1U ||
      host_batch.targets.empty() ||
      host_batch.targets.size() >
          std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument{
        "compact path reconstruction received inconsistent device/host shape"};
  }
}

}  // namespace

class ReusableCompactPathWorkspace::Impl final {
 public:
  DeviceBuffer status_snapshot;
  DeviceBuffer summaries;
  DeviceBuffer stack_vertices;
  DeviceBuffer stack_edge_ids;
  DeviceBuffer next_edge_thresholds;
  DeviceBuffer output_ranges;
  DeviceBuffer output_vertices;
  DeviceBuffer output_distance_labels;
  DeviceBuffer output_edge_ids;
  DeviceBuffer error_bits;

  std::vector<CompactTargetSummary> host_summaries;
  std::vector<DeviceCompactOutputRange> host_ranges;
  std::vector<std::uint32_t> host_vertices;
  std::vector<float> host_distance_labels;
  std::vector<std::uint32_t> host_edge_ids;
  std::uint64_t allocation_events{};
  std::uint32_t dfs_vertex_capacity{};
  std::uint32_t target_capacity{};

  HipEventTimer count_reconstruction_timer;
  HipEventTimer summary_transfer_timer;
  HipEventTimer emit_reconstruction_timer;
  HipEventTimer arena_transfer_timer;

  [[nodiscard]] bool reserve_initial(
      const std::uint32_t vertex_count,
      const std::uint32_t target_count) {
    bool grew = false;
    const auto grow = [&grew](DeviceBuffer& buffer, const std::size_t bytes) {
      grew = buffer.reserve(bytes, BufferGrowth::geometric) || grew;
    };
    grow(status_snapshot, sizeof(DeviceRunStatus));
    grow(
        summaries,
        checked_multiply(
            target_count,
            sizeof(DeviceCompactTargetSummary),
            "compact summary bytes"));
    grow(
        stack_vertices,
        checked_multiply(
            vertex_count, sizeof(std::uint32_t), "compact DFS vertices"));
    grow(
        stack_edge_ids,
        checked_multiply(
            vertex_count, sizeof(std::uint32_t), "compact DFS edge IDs"));
    grow(
        next_edge_thresholds,
        checked_multiply(
            vertex_count, sizeof(std::uint32_t), "compact DFS cursors"));
    grow(
        output_ranges,
        checked_multiply(
            target_count,
            sizeof(DeviceCompactOutputRange),
            "compact output ranges"));
    grow(error_bits, sizeof(std::uint32_t));
    if (grew) {
      ++allocation_events;
    }
    if (vertex_count > dfs_vertex_capacity) {
      dfs_vertex_capacity = vertex_count;
    }
    if (target_count > target_capacity) {
      target_capacity = target_count;
    }
    return grew;
  }

  void reserve_output(
      const std::size_t vertex_count,
      const std::size_t edge_count) {
    bool grew = false;
    const auto grow = [&grew](DeviceBuffer& buffer, const std::size_t bytes) {
      grew = buffer.reserve(bytes, BufferGrowth::geometric) || grew;
    };
    grow(
        output_vertices,
        checked_multiply(
            vertex_count, sizeof(std::uint32_t), "compact output vertices"));
    grow(
        output_distance_labels,
        checked_multiply(
            vertex_count, sizeof(float), "compact output distance labels"));
    grow(
        output_edge_ids,
        checked_multiply(
            edge_count, sizeof(std::uint32_t), "compact output edge IDs"));
    if (grew) {
      ++allocation_events;
    }
  }

  [[nodiscard]] std::size_t device_capacity() const noexcept {
    const DeviceBuffer* const buffers[]{
        &status_snapshot,
        &summaries,
        &stack_vertices,
        &stack_edge_ids,
        &next_edge_thresholds,
        &output_ranges,
        &output_vertices,
        &output_distance_labels,
        &output_edge_ids,
        &error_bits};
    std::size_t total = 0U;
    for (const DeviceBuffer* const buffer : buffers) {
      if (total > std::numeric_limits<std::size_t>::max() -
                      buffer->capacity_bytes()) {
        return std::numeric_limits<std::size_t>::max();
      }
      total += buffer->capacity_bytes();
    }
    return total;
  }
};

ReusableCompactPathWorkspace::ReusableCompactPathWorkspace()
    : impl_{new Impl{}} {}

ReusableCompactPathWorkspace::~ReusableCompactPathWorkspace() noexcept {
  delete impl_;
}

CompactPathWorkspaceCapacity ReusableCompactPathWorkspace::capacity()
    const noexcept {
  if (impl_ == nullptr) {
    return {};
  }
  return CompactPathWorkspaceCapacity{
      impl_->device_capacity(),
      impl_->dfs_vertex_capacity,
      impl_->target_capacity,
      impl_->allocation_events};
}

void account_compact_path_controller_polls(
    CompactPathBatchOutput& output,
    const std::uint64_t controller_poll_count) {
  const std::uint64_t payload_components = checked_add_u64(
      checked_add_u64(
          output.transfer.summary_bytes,
          output.transfer.vertex_bytes,
          "compact payload component accounting"),
      checked_add_u64(
          output.transfer.distance_label_bytes,
          output.transfer.edge_id_bytes,
          "compact payload component accounting"),
      "compact payload component accounting");
  if (payload_components != output.transfer.total_bytes) {
    throw std::logic_error{
        "compact payload components disagree before controller accounting"};
  }
  const std::uint64_t compact_total = checked_add_u64(
      output.transfer.total_bytes,
      checked_add_u64(
          output.transport.status_bytes,
          output.transport.error_bytes,
          "compact control transfer accounting"),
      "compact transfer accounting");
  if (compact_total != output.transport.total_device_to_host_bytes) {
    throw std::logic_error{
        "compact transfer subtotal disagrees before controller accounting"};
  }
  constexpr std::uint64_t controller_width = sizeof(DeviceController);
  if (controller_poll_count >
      std::numeric_limits<std::uint64_t>::max() / controller_width) {
    throw std::overflow_error{"compact controller-poll bytes overflow"};
  }
  output.transport.controller_poll_count = controller_poll_count;
  output.transport.controller_poll_bytes =
      controller_poll_count * controller_width;
  output.transport.overall_device_to_host_bytes = checked_add_u64(
      compact_total,
      output.transport.controller_poll_bytes,
      "compact overall device-to-host accounting");
}

CompactPathBatchOutput ReusableCompactPathWorkspace::reconstruct(
    const DeviceGraphView32& graph,
    const DeviceWorkspaceView& workspace,
    const DeviceCompactBatchView& batch_view,
    const DeviceCompactDistanceMatrix& distances,
    const BatchDeviceDescription& host_batch,
    const HipStream& stream) {
  if (impl_ == nullptr) {
    throw std::logic_error{
        "cannot reconstruct through an invalid compact path workspace"};
  }
  validate_host_shape(
      graph, workspace, batch_view, distances, host_batch);
  try {
  SteadyClockTimer wall_timer;
  const std::uint32_t target_count =
      static_cast<std::uint32_t>(host_batch.targets.size());
  static_cast<void>(impl_->reserve_initial(graph.vertex_count, target_count));
  impl_->host_summaries.resize(target_count);

  impl_->error_bits.clear_async(sizeof(std::uint32_t), stream);
  impl_->count_reconstruction_timer.start(stream);
  hipLaunchKernelGGL(
      compact_path_reconstruction_kernel,
      dim3{1U},
      dim3{1U},
      0U,
      static_cast<hipStream_t>(stream.native_handle()),
      graph,
      workspace,
      batch_view,
      distances,
      device_pointer<DeviceRunStatus>(impl_->status_snapshot.data()),
      device_pointer<DeviceCompactTargetSummary>(impl_->summaries.data()),
      device_pointer<std::uint32_t>(impl_->stack_vertices.data()),
      device_pointer<std::uint32_t>(impl_->stack_edge_ids.data()),
      device_pointer<std::uint32_t>(impl_->next_edge_thresholds.data()),
      nullptr,
      0U,
      0U,
      nullptr,
      nullptr,
      nullptr,
      0U,
      device_pointer<std::uint32_t>(impl_->error_bits.data()));
  throw_if_hip_error(
      static_cast<std::int32_t>(hipGetLastError()),
      "compact path count kernel launch");
  impl_->count_reconstruction_timer.stop(stream);

  DeviceRunStatus status{};
  std::uint32_t count_error_bits = 0U;
  impl_->summary_transfer_timer.start(stream);
  impl_->status_snapshot.copy_to_host_async(
      &status, sizeof(status), stream);
  impl_->summaries.copy_to_host_async(
      impl_->host_summaries.data(),
      checked_multiply(
          target_count,
          sizeof(CompactTargetSummary),
          "compact summary transfer bytes"),
      stream);
  impl_->error_bits.copy_to_host_async(
      &count_error_bits, sizeof(count_error_bits), stream);
  impl_->summary_transfer_timer.stop(stream);
  stream.synchronize();
  if (count_error_bits != 0U) {
    throw std::runtime_error{
        "compact path count pass rejected device metadata"};
  }

  impl_->host_ranges.assign(target_count, DeviceCompactOutputRange{});
  std::uint64_t total_vertices = 0U;
  std::uint64_t total_edges = 0U;
  for (std::size_t index = 0U; index < impl_->host_summaries.size(); ++index) {
    const CompactTargetSummary& summary = impl_->host_summaries[index];
    if (!valid_summary_enums(summary) ||
        summary.target.value() != host_batch.targets[index]) {
      throw std::runtime_error{
          "compact path count pass returned an invalid target summary"};
    }
    if (summary.reconstruction != CompactPathStatus::complete) {
      if (summary.path_length != 0U || summary.has_selected_source != 0U) {
        throw std::runtime_error{
            "failed compact target summary carried a path/source"};
      }
      continue;
    }
    if (summary.reached != CompactTargetReachStatus::reached ||
        summary.has_selected_source != 1U ||
        !std::isfinite(summary.distance) || summary.distance < 0.0F ||
        summary.path_length >= graph.vertex_count) {
      throw std::runtime_error{
          "complete compact target summary has invalid path metadata"};
    }
    impl_->host_ranges[index] =
        DeviceCompactOutputRange{total_vertices, total_edges};
    total_vertices = checked_add_u64(
        total_vertices,
        static_cast<std::uint64_t>(summary.path_length) + 1U,
        "compact vertex count");
    total_edges = checked_add_u64(
        total_edges, summary.path_length, "compact edge count");
  }
  if (total_vertices > std::numeric_limits<std::size_t>::max() ||
      total_edges > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error{
        "compact path arenas exceed the host address space"};
  }
  const std::size_t vertex_count = static_cast<std::size_t>(total_vertices);
  const std::size_t edge_count = static_cast<std::size_t>(total_edges);
  impl_->reserve_output(vertex_count, edge_count);
  impl_->host_vertices.resize(vertex_count);
  impl_->host_distance_labels.resize(vertex_count);
  impl_->host_edge_ids.resize(edge_count);

  float emit_milliseconds = 0.0F;
  float arena_transfer_milliseconds = 0.0F;
  if (vertex_count != 0U) {
    impl_->emit_reconstruction_timer.start(stream);
    impl_->output_ranges.copy_from_host_async(
        impl_->host_ranges.data(),
        checked_multiply(
            target_count,
            sizeof(DeviceCompactOutputRange),
            "compact range upload bytes"),
        stream);
    impl_->error_bits.clear_async(sizeof(std::uint32_t), stream);
    hipLaunchKernelGGL(
        compact_path_reconstruction_kernel,
        dim3{1U},
        dim3{1U},
        0U,
        static_cast<hipStream_t>(stream.native_handle()),
        graph,
        workspace,
        batch_view,
        distances,
        device_pointer<DeviceRunStatus>(impl_->status_snapshot.data()),
        device_pointer<DeviceCompactTargetSummary>(impl_->summaries.data()),
        device_pointer<std::uint32_t>(impl_->stack_vertices.data()),
        device_pointer<std::uint32_t>(impl_->stack_edge_ids.data()),
        device_pointer<std::uint32_t>(impl_->next_edge_thresholds.data()),
        device_pointer<const DeviceCompactOutputRange>(
            impl_->output_ranges.data()),
        total_vertices,
        total_edges,
        device_pointer<std::uint32_t>(impl_->output_vertices.data()),
        device_pointer<float>(impl_->output_distance_labels.data()),
        edge_count == 0U
            ? nullptr
            : device_pointer<std::uint32_t>(impl_->output_edge_ids.data()),
        1U,
        device_pointer<std::uint32_t>(impl_->error_bits.data()));
    throw_if_hip_error(
        static_cast<std::int32_t>(hipGetLastError()),
        "compact path emit kernel launch");
    impl_->emit_reconstruction_timer.stop(stream);

    std::uint32_t emit_error_bits = 0U;
    impl_->arena_transfer_timer.start(stream);
    impl_->output_vertices.copy_to_host_async(
        impl_->host_vertices.data(),
        checked_multiply(
            vertex_count,
            sizeof(std::uint32_t),
            "compact vertex transfer bytes"),
        stream);
    impl_->output_distance_labels.copy_to_host_async(
        impl_->host_distance_labels.data(),
        checked_multiply(
            vertex_count, sizeof(float), "compact label transfer bytes"),
        stream);
    if (edge_count != 0U) {
      impl_->output_edge_ids.copy_to_host_async(
          impl_->host_edge_ids.data(),
          checked_multiply(
              edge_count,
              sizeof(std::uint32_t),
              "compact edge-ID transfer bytes"),
          stream);
    }
    impl_->error_bits.copy_to_host_async(
        &emit_error_bits, sizeof(emit_error_bits), stream);
    impl_->arena_transfer_timer.stop(stream);
    stream.synchronize();
    if (emit_error_bits != 0U) {
      throw std::runtime_error{
          "compact path emit pass disagreed with the count pass"};
    }
    emit_milliseconds = impl_->emit_reconstruction_timer
                            .elapsed_milliseconds_after_stream_synchronization();
    arena_transfer_milliseconds = impl_->arena_transfer_timer
                                      .elapsed_milliseconds_after_stream_synchronization();
  }

  CompactPathBatchOutput output;
  output.status = status;
  output.targets.reserve(target_count);
  for (std::size_t index = 0U; index < impl_->host_summaries.size(); ++index) {
    CompactTargetPath target;
    target.summary = impl_->host_summaries[index];
    if (target.summary.reconstruction == CompactPathStatus::complete) {
      const DeviceCompactOutputRange range = impl_->host_ranges[index];
      const std::size_t vertex_begin =
          static_cast<std::size_t>(range.vertex_offset);
      const std::size_t edge_begin =
          static_cast<std::size_t>(range.edge_offset);
      const std::size_t path_vertices =
          static_cast<std::size_t>(target.summary.path_length) + 1U;
      const std::size_t path_edges = target.summary.path_length;
      target.vertices.reserve(path_vertices);
      target.distance_labels.reserve(path_vertices);
      target.edge_ids.reserve(path_edges);
      for (std::size_t path_index = 0U; path_index < path_vertices;
           ++path_index) {
        target.vertices.push_back(
            VertexId{impl_->host_vertices[vertex_begin + path_index]});
        target.distance_labels.push_back(
            impl_->host_distance_labels[vertex_begin + path_index]);
      }
      for (std::size_t path_index = 0U; path_index < path_edges;
           ++path_index) {
        target.edge_ids.push_back(
            EdgeId{impl_->host_edge_ids[edge_begin + path_index]});
      }
    }
    output.targets.push_back(std::move(target));
  }

  output.transfer.summary_bytes = checked_multiply(
      target_count,
      sizeof(CompactTargetSummary),
      "compact summary accounting");
  output.transfer.vertex_bytes = checked_multiply(
      vertex_count, sizeof(std::uint32_t), "compact vertex accounting");
  output.transfer.distance_label_bytes = checked_multiply(
      vertex_count, sizeof(float), "compact label accounting");
  output.transfer.edge_id_bytes = checked_multiply(
      edge_count, sizeof(std::uint32_t), "compact edge-ID accounting");
  output.transfer.total_bytes = checked_add_u64(
      checked_add_u64(
          output.transfer.summary_bytes,
          output.transfer.vertex_bytes,
          "compact transfer accounting"),
      checked_add_u64(
          output.transfer.distance_label_bytes,
          output.transfer.edge_id_bytes,
          "compact transfer accounting"),
      "compact transfer accounting");
  output.transport.status_bytes = sizeof(DeviceRunStatus);
  output.transport.error_bytes = sizeof(std::uint32_t);
  if (vertex_count != 0U) {
    output.transport.error_bytes = checked_add_u64(
        output.transport.error_bytes,
        sizeof(std::uint32_t),
        "compact error transfer accounting");
  }
  output.transport.total_device_to_host_bytes = checked_add_u64(
      output.transfer.total_bytes,
      checked_add_u64(
          output.transport.status_bytes,
          output.transport.error_bytes,
          "compact control transfer accounting"),
      "compact total transfer accounting");
  output.transport.overall_device_to_host_bytes =
      output.transport.total_device_to_host_bytes;
  output.metrics.reconstruction_device_milliseconds =
      impl_->count_reconstruction_timer
          .elapsed_milliseconds_after_stream_synchronization() +
      emit_milliseconds;
  output.metrics.result_transfer_device_milliseconds =
      impl_->summary_transfer_timer
          .elapsed_milliseconds_after_stream_synchronization() +
      arena_transfer_milliseconds;
  output.metrics.end_to_end_wall_milliseconds =
      wall_timer.elapsed_milliseconds();
  return output;
  } catch (...) {
    // This workspace owns buffers referenced by both reconstruction passes.
    // Make an exceptional return safe even when used outside an engine (whose
    // lease recovery also synchronizes outstanding device work).
    try {
      stream.synchronize();
    } catch (...) {
      static_cast<void>(hipDeviceSynchronize());
    }
    throw;
  }
}

}  // namespace bfnew::hip
