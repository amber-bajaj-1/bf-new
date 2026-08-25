#include "bfnew/batch_workspace.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace bfnew {
namespace {

[[nodiscard]] std::uint64_t checked_add(
    const std::uint64_t left,
    const std::uint64_t right) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw std::overflow_error{"batch workspace byte total overflow"};
  }
  return left + right;
}

[[nodiscard]] std::uint64_t checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right) {
  if (right != 0U &&
      left > std::numeric_limits<std::uint64_t>::max() / right) {
    throw std::overflow_error{"batch workspace component overflow"};
  }
  return left * right;
}

void add_component(std::uint64_t& total, const std::uint64_t component) {
  total = checked_add(total, component);
}

[[nodiscard]] std::uint64_t vector_bytes(
    const std::size_t count,
    const std::size_t element_size) {
  return checked_multiply(
      static_cast<std::uint64_t>(count),
      static_cast<std::uint64_t>(element_size));
}

[[nodiscard]] std::uint64_t metadata_bytes(
    const BatchDeviceDescription& description) {
  std::uint64_t bytes = 0U;
  add_component(
      bytes,
      vector_bytes(
          description.query_ids_by_lane.size(), sizeof(std::uint32_t)));
  add_component(
      bytes,
      vector_bytes(
          description.expansion_generations_by_lane.size(),
          sizeof(std::uint32_t)));
  add_component(
      bytes,
      vector_bytes(description.source_offsets.size(), sizeof(std::uint32_t)));
  add_component(
      bytes, vector_bytes(description.sources.size(), sizeof(std::uint32_t)));
  add_component(
      bytes,
      vector_bytes(description.target_offsets.size(), sizeof(std::uint32_t)));
  add_component(
      bytes, vector_bytes(description.targets.size(), sizeof(std::uint32_t)));
  add_component(
      bytes,
      vector_bytes(
          description.selected_vertex_counts_by_lane.size(),
          sizeof(std::uint64_t)));
  add_component(
      bytes,
      vector_bytes(
          description.selected_edge_estimates_by_lane.size(),
          sizeof(std::uint64_t)));
  add_component(
      bytes, vector_bytes(description.union_tiles.size(), sizeof(std::uint32_t)));
  add_component(
      bytes,
      vector_bytes(description.tile_lane_masks.size(), sizeof(LaneMask)));
  add_component(
      bytes,
      vector_bytes(
          description.selected_vertex_ranges.size(), sizeof(BatchVertexRange)));
  add_component(bytes, 3U * sizeof(LaneMask));
  add_component(bytes, sizeof(DeviceController));
  add_component(bytes, sizeof(DeviceRunStatus));
  return bytes;
}

[[nodiscard]] bool grow_geometrically(
    std::uint64_t& capacity,
    const std::uint64_t requested) {
  if (requested <= capacity) {
    return false;
  }
  std::uint64_t next = capacity == 0U ? requested : capacity;
  while (next < requested) {
    if (next > std::numeric_limits<std::uint64_t>::max() / 2U) {
      next = requested;
      break;
    }
    next *= 2U;
  }
  capacity = next;
  return true;
}

[[nodiscard]] const char* vertex_storage_name(
    const BatchVertexStorageStrategy strategy) noexcept {
  switch (strategy) {
    case BatchVertexStorageStrategy::full_graph_vertex_major:
      return "full_graph_vertex_major";
    case BatchVertexStorageStrategy::compact_union_tiles:
      return "compact_union_tiles";
  }
  return "invalid";
}

[[nodiscard]] const char* run_representation_name(
    const BatchRunRepresentation representation) noexcept {
  switch (representation) {
    case BatchRunRepresentation::retained_per_run_masks:
      return "retained_per_run_masks";
    case BatchRunRepresentation::compact_nonzero_descriptors:
      return "compact_nonzero_descriptors";
    case BatchRunRepresentation::device_materialized_run_masks:
      return "device_materialized_run_masks";
  }
  return "invalid";
}

[[nodiscard]] const BatchWorkspaceEstimate* find_selected_estimate(
    const BatchWorkspaceDecision& decision) noexcept {
  const auto position = std::find_if(
      decision.compared_strategies.begin(),
      decision.compared_strategies.end(),
      [&decision](const BatchWorkspaceEstimate& estimate) {
        return estimate.vertex_storage == decision.selected_vertex_storage &&
               estimate.run_representation ==
                   decision.selected_run_representation;
      });
  return position == decision.compared_strategies.end() ? nullptr : &*position;
}

}  // namespace

std::uint64_t estimate_vertex_major_distance_bytes(
    const std::uint64_t storage_vertex_count,
    const std::uint32_t lane_width,
    const std::uint32_t distance_slot_count) {
  if ((lane_width != 1U && lane_width != 8U && lane_width != 16U &&
       lane_width != 32U) ||
      (distance_slot_count != 1U && distance_slot_count != 2U)) {
    throw std::invalid_argument{
        "batch distance estimate requires width 1/8/16/32 and one/two slots"};
  }
  return checked_multiply(
      checked_multiply(storage_vertex_count, lane_width),
      checked_multiply(distance_slot_count, sizeof(float)));
}

BatchWorkspaceEstimate estimate_batch_workspace(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& description,
    const BatchVertexStorageStrategy vertex_storage,
    const BatchRunRepresentation run_representation,
    const std::uint32_t distance_slot_count,
    const BatchWorkspaceBudget& budget,
    const BatchWorkspaceStrategyTiming& timing) {
  if (vertex_storage !=
          BatchVertexStorageStrategy::full_graph_vertex_major &&
      vertex_storage != BatchVertexStorageStrategy::compact_union_tiles) {
    throw std::invalid_argument{"invalid batch vertex-storage strategy"};
  }
  if (run_representation !=
          BatchRunRepresentation::retained_per_run_masks &&
      run_representation !=
          BatchRunRepresentation::compact_nonzero_descriptors &&
      run_representation !=
          BatchRunRepresentation::device_materialized_run_masks) {
    throw std::invalid_argument{"invalid batch run representation"};
  }
  if (distance_slot_count != 1U && distance_slot_count != 2U) {
    throw std::invalid_argument{
        "batch workspace distance slot count must be one or two"};
  }
  if (run_representation ==
          BatchRunRepresentation::device_materialized_run_masks &&
      (vertex_storage !=
           BatchVertexStorageStrategy::full_graph_vertex_major ||
       distance_slot_count != 2U)) {
    throw std::invalid_argument{
        "device-materialized Jacobi requires full-graph two-slot storage"};
  }
  if (batch.lane_width != description.lane_width ||
      description.run_representation != run_representation ||
      (batch.lane_width != 1U && batch.lane_width != 8U &&
       batch.lane_width != 16U && batch.lane_width != 32U)) {
    throw std::invalid_argument{
        "batch workspace request does not match its prepared description"};
  }
  if (description.run_report.active_csr_runs >
          tile_runs.csr_run_destination_tiles.size() ||
      description.run_report.active_csc_runs >
          tile_runs.csc_run_source_tiles.size()) {
    throw std::invalid_argument{"batch run counts exceed immutable metadata"};
  }
  if (vertex_storage == BatchVertexStorageStrategy::full_graph_vertex_major &&
      timing.mapping_build_nanoseconds != 0U) {
    throw std::invalid_argument{
        "full-graph workspace rows cannot record compact mapping work"};
  }
  if (vertex_storage == BatchVertexStorageStrategy::compact_union_tiles &&
      (!description.compact_vertex_mapping_valid ||
       description.compact_vertex_biases_by_tile.size() !=
           graph.tile_coordinates().size() ||
       description.touched_compact_tiles.size() != batch.union_tiles.size() ||
       description.compact_mapping_report.entries_written !=
           batch.union_tiles.size() ||
       description.compact_mapping_report.entries_initialized >
           graph.tile_coordinates().size() ||
       description.compact_mapping_report.entries_cleared >
           graph.tile_coordinates().size())) {
    throw std::invalid_argument{
        "compact workspace requires a prepared compact vertex mapping"};
  }
  if (run_representation ==
          BatchRunRepresentation::compact_nonzero_descriptors &&
      (batch.union_vertex_count == std::numeric_limits<std::uint64_t>::max() ||
       !std::in_range<std::size_t>(batch.union_vertex_count + 1U) ||
       description.csr_descriptor_offsets_by_union_vertex.size() !=
           static_cast<std::size_t>(batch.union_vertex_count + 1U) ||
       description.csc_descriptor_offsets_by_union_vertex.size() !=
           static_cast<std::size_t>(batch.union_vertex_count + 1U) ||
       description.csr_descriptor_offsets_by_union_vertex.back() !=
           description.csr_run_descriptors.size() ||
       description.csc_descriptor_offsets_by_union_vertex.back() !=
           description.csc_run_descriptors.size())) {
    throw std::invalid_argument{
        "descriptor workspace requires traversable per-vertex offsets"};
  }
  if (run_representation ==
          BatchRunRepresentation::device_materialized_run_masks &&
      (description.tile_lane_masks.size() !=
           graph.tile_coordinates().size() ||
       description.selected_vertex_ranges.empty() ||
       !description.csr_run_lane_masks.empty() ||
       !description.csc_run_lane_masks.empty() ||
       !description.touched_csr_runs.empty() ||
       !description.touched_csc_runs.empty() ||
       !description.csr_run_descriptors.empty() ||
       !description.csc_run_descriptors.empty() ||
       !description.csr_descriptor_offsets_by_union_vertex.empty() ||
       !description.csc_descriptor_offsets_by_union_vertex.empty() ||
       description.run_report != BatchRunPreparationReport{} ||
       timing.run_build_nanoseconds != 0U)) {
    throw std::invalid_argument{
        "device-materialized Jacobi workspace requires an empty host run "
        "image and zero host run-build time"};
  }
  const BatchRunPreparationReport& run_report = description.run_report;
  if (!description.run_representation_initialized ||
      run_report.csr_retained_entries_initialized >
          tile_runs.csr_run_destination_tiles.size() ||
      run_report.csc_retained_entries_initialized >
          tile_runs.csc_run_source_tiles.size() ||
      run_report.csr_retained_entries_cleared >
          tile_runs.csr_run_destination_tiles.size() ||
      run_report.csc_retained_entries_cleared >
          tile_runs.csc_run_source_tiles.size() ||
      run_report.csr_retained_entries_written >
          tile_runs.csr_run_destination_tiles.size() ||
      run_report.csc_retained_entries_written >
          tile_runs.csc_run_source_tiles.size() ||
      run_report.csr_descriptor_entries_written >
          tile_runs.csr_run_destination_tiles.size() ||
      run_report.csc_descriptor_entries_written >
          tile_runs.csc_run_source_tiles.size() ||
      (run_representation ==
           BatchRunRepresentation::retained_per_run_masks &&
       (run_report.csr_descriptor_entries_written != 0U ||
        run_report.csc_descriptor_entries_written != 0U)) ||
      (run_representation ==
           BatchRunRepresentation::compact_nonzero_descriptors &&
       (run_report.retained_entries_initialized != 0U ||
        run_report.csr_retained_entries_written != 0U ||
        run_report.csc_retained_entries_written != 0U)) ||
      (run_representation ==
           BatchRunRepresentation::device_materialized_run_masks &&
       run_report != BatchRunPreparationReport{})) {
    throw std::invalid_argument{
        "batch run preparation counters are inconsistent"};
  }

  BatchWorkspaceEstimate estimate;
  estimate.vertex_storage = vertex_storage;
  estimate.run_representation = run_representation;
  estimate.lane_width = batch.lane_width;
  estimate.distance_slot_count = distance_slot_count;
  estimate.mapping_build_nanoseconds = timing.mapping_build_nanoseconds;
  estimate.run_build_nanoseconds = timing.run_build_nanoseconds;
  estimate.union_vertex_count = batch.union_vertex_count;
  estimate.selected_lane_vertex_count = batch.selected_lane_vertex_count;
  estimate.storage_vertex_count =
      vertex_storage == BatchVertexStorageStrategy::full_graph_vertex_major
          ? graph.vertex_count()
          : batch.union_vertex_count;
  estimate.allocated_lane_vertex_count = checked_multiply(
      estimate.storage_vertex_count, estimate.lane_width);
  if (estimate.selected_lane_vertex_count >
      estimate.allocated_lane_vertex_count) {
    throw std::invalid_argument{
        "selected lane vertices exceed batch workspace storage"};
  }
  estimate.wasted_lane_vertex_count =
      estimate.allocated_lane_vertex_count -
      estimate.selected_lane_vertex_count;

  estimate.distance_bytes = estimate_vertex_major_distance_bytes(
      estimate.storage_vertex_count,
      estimate.lane_width,
      distance_slot_count);

  // Compact labels use one retained uint32 bias per tile. A selected global
  // vertex maps as compact = global - bias, guarded by the dense tile mask.
  // Full-graph labels need no such mapping, including descriptor traversal.
  const bool needs_tile_mapping =
      vertex_storage == BatchVertexStorageStrategy::compact_union_tiles;
  estimate.tile_mapping_bytes =
      needs_tile_mapping
          ? checked_multiply(
                graph.tile_coordinates().size(), sizeof(std::uint32_t))
          : 0U;

  std::uint64_t device_materialized_csc_writes = 0U;
  estimate.active_csr_runs = description.run_report.active_csr_runs;
  estimate.active_csc_runs = description.run_report.active_csc_runs;
  if (run_representation ==
      BatchRunRepresentation::device_materialized_run_masks) {
    // This representation is deliberately Jacobi/CSC-specific. Estimation is
    // an offline diagnostic: it scans only selected destination columns to
    // report exact nonzero masks and writes, while hot preparation performs no
    // host run scan. Every selected run is written, including a zero mask.
    estimate.active_csr_runs = 0U;
    estimate.active_csc_runs = 0U;
    for (const BatchVertexRange range : description.selected_vertex_ranges) {
      if (range.begin > range.end || range.end > graph.vertex_count()) {
        throw std::invalid_argument{
            "device-materialized selected vertex range is invalid"};
      }
      for (std::uint32_t vertex = range.begin; vertex < range.end; ++vertex) {
        const EdgeOffset run_begin =
            tile_runs.csc_column_run_offsets[vertex];
        const EdgeOffset run_end =
            tile_runs.csc_column_run_offsets[vertex + 1U];
        device_materialized_csc_writes = checked_add(
            device_materialized_csc_writes, run_end - run_begin);
        for (EdgeOffset run = run_begin; run < run_end; ++run) {
          const TileId source_tile = tile_runs.csc_run_source_tiles[run];
          if ((range.lane_mask &
               description.tile_lane_masks[source_tile.value()]) != 0U) {
            estimate.active_csc_runs =
                checked_add(estimate.active_csc_runs, 1U);
          }
        }
      }
    }
  }
  estimate.zero_csr_runs =
      run_representation ==
              BatchRunRepresentation::device_materialized_run_masks
          ? 0U
          : static_cast<std::uint64_t>(
                tile_runs.csr_run_destination_tiles.size()) -
                estimate.active_csr_runs;
  estimate.zero_csc_runs =
      static_cast<std::uint64_t>(tile_runs.csc_run_source_tiles.size()) -
      estimate.active_csc_runs;
  const std::uint64_t maximum_run_count = std::max<std::uint64_t>(
      tile_runs.csr_run_destination_tiles.size(),
      tile_runs.csc_run_source_tiles.size());
  const std::uint64_t maximum_active_run_count = std::max(
      estimate.active_csr_runs, estimate.active_csc_runs);
  if (run_representation == BatchRunRepresentation::retained_per_run_masks) {
    estimate.run_storage_bytes = checked_multiply(
        maximum_run_count, sizeof(LaneMask));
  } else if (run_representation ==
             BatchRunRepresentation::compact_nonzero_descriptors) {
    estimate.run_storage_bytes = checked_multiply(
        maximum_active_run_count, sizeof(RunLaneMaskDescriptor));
    estimate.descriptor_offset_bytes = checked_multiply(
        checked_add(estimate.union_vertex_count, 1U), sizeof(std::uint32_t));
  } else {
    estimate.run_storage_bytes = checked_multiply(
        tile_runs.csc_run_source_tiles.size(), sizeof(LaneMask));
  }

  estimate.batch_metadata_bytes = metadata_bytes(description);
  add_component(estimate.total_workspace_bytes, estimate.distance_bytes);
  add_component(estimate.total_workspace_bytes, estimate.tile_mapping_bytes);
  add_component(estimate.total_workspace_bytes, estimate.run_storage_bytes);
  add_component(
      estimate.total_workspace_bytes, estimate.descriptor_offset_bytes);
  add_component(estimate.total_workspace_bytes, estimate.batch_metadata_bytes);

  estimate.distance_reset_bytes = checked_multiply(
      checked_multiply(
          estimate.selected_lane_vertex_count, distance_slot_count),
      sizeof(float));
  estimate.tile_mapping_write_bytes =
      needs_tile_mapping
          ? checked_multiply(
                checked_add(
                    description.compact_mapping_report.entries_initialized,
                    checked_add(
                        description.compact_mapping_report.entries_cleared,
                        description.compact_mapping_report.entries_written)),
                sizeof(std::uint32_t))
          : 0U;
  if (run_representation == BatchRunRepresentation::retained_per_run_masks) {
    const std::uint64_t csr_writes = checked_add(
        description.run_report.csr_retained_entries_initialized,
        checked_add(
            description.run_report.csr_retained_entries_cleared,
            description.run_report.csr_retained_entries_written));
    const std::uint64_t csc_writes = checked_add(
        description.run_report.csc_retained_entries_initialized,
        checked_add(
            description.run_report.csc_retained_entries_cleared,
            description.run_report.csc_retained_entries_written));
    estimate.run_preparation_write_bytes = checked_multiply(
        std::max(csr_writes, csc_writes), sizeof(LaneMask));
  } else if (run_representation ==
             BatchRunRepresentation::compact_nonzero_descriptors) {
    const std::uint64_t csr_writes = checked_add(
        checked_multiply(
            description.run_report.csr_descriptor_entries_written,
            sizeof(RunLaneMaskDescriptor)),
        vector_bytes(
            description.csr_descriptor_offsets_by_union_vertex.size(),
            sizeof(std::uint32_t)));
    const std::uint64_t csc_writes = checked_add(
        checked_multiply(
            description.run_report.csc_descriptor_entries_written,
            sizeof(RunLaneMaskDescriptor)),
        vector_bytes(
            description.csc_descriptor_offsets_by_union_vertex.size(),
            sizeof(std::uint32_t)));
    // The host proof image builds both orientations. Capacity and preparation
    // traffic model the larger single orientation because the device engine
    // activates only one orientation at a time.
    estimate.run_preparation_write_bytes = std::max(csr_writes, csc_writes);
  } else {
    estimate.run_preparation_write_bytes = checked_multiply(
        device_materialized_csc_writes, sizeof(LaneMask));
  }
  estimate.total_preparation_write_bytes = checked_add(
      estimate.distance_reset_bytes,
      checked_add(
          estimate.tile_mapping_write_bytes,
          estimate.run_preparation_write_bytes));
  estimate.reusable_allocation = true;

  const std::uint64_t committed = checked_add(
      budget.resident_graph_bytes, budget.explicit_reserve_bytes);
  if (committed <= budget.device_capacity_bytes &&
      estimate.total_workspace_bytes != 0U) {
    estimate.maximum_concurrent_workspaces =
        (budget.device_capacity_bytes - committed) /
        estimate.total_workspace_bytes;
  }
  return estimate;
}

std::vector<BatchWorkspaceEstimate> compare_batch_workspace_strategies(
    const WeightedGraph& graph,
    const TileRunLayout64& tile_runs,
    const BatchPlanEntry& batch,
    const BatchDeviceDescription& retained_description,
    const BatchDeviceDescription& descriptor_description,
    const std::uint32_t distance_slot_count,
    const BatchWorkspaceBudget& budget,
    const BatchWorkspaceBuildMeasurements& measurements) {
  if (retained_description.run_representation !=
          BatchRunRepresentation::retained_per_run_masks ||
      descriptor_description.run_representation !=
          BatchRunRepresentation::compact_nonzero_descriptors) {
    throw std::invalid_argument{
        "batch workspace comparison requires both run representations"};
  }
  std::vector<BatchWorkspaceEstimate> estimates;
  estimates.reserve(4U);
  for (const BatchVertexStorageStrategy vertex_storage : {
           BatchVertexStorageStrategy::full_graph_vertex_major,
           BatchVertexStorageStrategy::compact_union_tiles}) {
    estimates.push_back(estimate_batch_workspace(
        graph,
        tile_runs,
        batch,
        retained_description,
        vertex_storage,
        BatchRunRepresentation::retained_per_run_masks,
        distance_slot_count,
        budget,
        BatchWorkspaceStrategyTiming{
            vertex_storage == BatchVertexStorageStrategy::compact_union_tiles
                ? measurements.compact_mapping_build_nanoseconds
                : 0U,
            measurements.retained_run_build_nanoseconds}));
    estimates.push_back(estimate_batch_workspace(
        graph,
        tile_runs,
        batch,
        descriptor_description,
        vertex_storage,
        BatchRunRepresentation::compact_nonzero_descriptors,
        distance_slot_count,
        budget,
        BatchWorkspaceStrategyTiming{
            vertex_storage == BatchVertexStorageStrategy::compact_union_tiles
                ? measurements.compact_mapping_build_nanoseconds
                : 0U,
            measurements.descriptor_run_build_nanoseconds}));
  }
  return estimates;
}

std::uint64_t BatchWorkspaceCapacity::total_bytes() const {
  std::uint64_t total = 0U;
  add_component(total, distance_bytes);
  add_component(total, tile_mapping_bytes);
  add_component(total, run_storage_bytes);
  add_component(total, descriptor_offset_bytes);
  add_component(total, batch_metadata_bytes);
  return total;
}

BatchWorkspaceReservationResult ReusableBatchWorkspaceReservation::reserve(
    const BatchWorkspaceEstimate& estimate) {
  if (!estimate.reusable_allocation || estimate.total_workspace_bytes == 0U) {
    throw std::invalid_argument{
        "batch workspace reservation requires a validated nonempty estimate"};
  }
  const std::uint64_t component_sum = checked_add(
      checked_add(estimate.distance_bytes, estimate.tile_mapping_bytes),
      checked_add(
          checked_add(
              estimate.run_storage_bytes, estimate.descriptor_offset_bytes),
          estimate.batch_metadata_bytes));
  if (component_sum != estimate.total_workspace_bytes) {
    throw std::invalid_argument{
        "batch workspace estimate component sum is inconsistent"};
  }

  bool grew = false;
  grew = grow_geometrically(capacity_.distance_bytes, estimate.distance_bytes) ||
         grew;
  grew = grow_geometrically(
             capacity_.tile_mapping_bytes, estimate.tile_mapping_bytes) ||
         grew;
  grew = grow_geometrically(
             capacity_.run_storage_bytes, estimate.run_storage_bytes) ||
         grew;
  grew = grow_geometrically(
             capacity_.descriptor_offset_bytes,
             estimate.descriptor_offset_bytes) ||
         grew;
  grew = grow_geometrically(
             capacity_.batch_metadata_bytes, estimate.batch_metadata_bytes) ||
         grew;
  if (grew) {
    ++growth_events_;
  }
  if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{"batch workspace generation overflow"};
  }
  ++generation_;
  return BatchWorkspaceReservationResult{grew, generation_};
}

const BatchWorkspaceCapacity& ReusableBatchWorkspaceReservation::capacity()
    const noexcept {
  return capacity_;
}

std::uint64_t ReusableBatchWorkspaceReservation::growth_events() const noexcept {
  return growth_events_;
}

bool validate_batch_workspace_decision(
    const BatchWorkspaceDecision& decision) noexcept {
  if ((decision.measurement_scope !=
           BatchMeasurementScope::bounded_synthetic &&
       decision.measurement_scope !=
           BatchMeasurementScope::real_query_corpus) ||
      (decision.selected_vertex_storage !=
           BatchVertexStorageStrategy::full_graph_vertex_major &&
       decision.selected_vertex_storage !=
           BatchVertexStorageStrategy::compact_union_tiles) ||
      (decision.selected_run_representation !=
           BatchRunRepresentation::retained_per_run_masks &&
       decision.selected_run_representation !=
           BatchRunRepresentation::compact_nonzero_descriptors) ||
      (decision.lane_width != 1U && decision.lane_width != 8U &&
       decision.lane_width != 16U && decision.lane_width != 32U) ||
      (decision.distance_slot_count != 1U &&
       decision.distance_slot_count != 2U) ||
      decision.plan_fingerprint == 0U ||
      decision.graph_vertex_count == 0U ||
      decision.graph_tile_count == 0U ||
      decision.graph_vertex_count >
          std::numeric_limits<std::uint32_t>::max() ||
      decision.graph_tile_count >
          std::numeric_limits<std::uint32_t>::max() ||
      decision.compared_strategies.size() != 4U ||
      decision.quantitative_reason.empty() ||
      decision.quantitative_reason.find_first_of("\r\n\t") !=
          std::string::npos) {
    return false;
  }
  if (decision.budget.resident_graph_bytes >
      std::numeric_limits<std::uint64_t>::max() -
          decision.budget.explicit_reserve_bytes) {
    return false;
  }
  const std::uint64_t committed = decision.budget.resident_graph_bytes +
                                  decision.budget.explicit_reserve_bytes;
  const auto multiply_without_overflow = [](
                                             const std::uint64_t left,
                                             const std::uint64_t right,
                                             std::uint64_t& product) noexcept {
    if (left != 0U &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
      return false;
    }
    product = left * right;
    return true;
  };
  const auto add_without_overflow = [](
                                        const std::uint64_t left,
                                        const std::uint64_t right,
                                        std::uint64_t& sum) noexcept {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
      return false;
    }
    sum = left + right;
    return true;
  };
  std::uint64_t expected_tile_mapping_bytes = 0U;
  if (!multiply_without_overflow(
          decision.graph_tile_count,
          sizeof(std::uint32_t),
          expected_tile_mapping_bytes)) {
    return false;
  }
  std::array<bool, 4U> combinations{};
  for (const BatchWorkspaceEstimate& estimate :
       decision.compared_strategies) {
    if ((estimate.vertex_storage !=
             BatchVertexStorageStrategy::full_graph_vertex_major &&
         estimate.vertex_storage !=
             BatchVertexStorageStrategy::compact_union_tiles) ||
        (estimate.run_representation !=
             BatchRunRepresentation::retained_per_run_masks &&
         estimate.run_representation !=
             BatchRunRepresentation::compact_nonzero_descriptors) ||
        estimate.lane_width != decision.lane_width ||
        estimate.distance_slot_count != decision.distance_slot_count ||
        estimate.union_vertex_count == 0U ||
        estimate.selected_lane_vertex_count == 0U ||
        estimate.total_workspace_bytes == 0U ||
        !estimate.reusable_allocation ||
        (estimate.vertex_storage ==
             BatchVertexStorageStrategy::full_graph_vertex_major &&
         estimate.mapping_build_nanoseconds != 0U)) {
      return false;
    }
    std::uint64_t component_sum = 0U;
    for (const std::uint64_t component : {
             estimate.distance_bytes,
             estimate.tile_mapping_bytes,
             estimate.run_storage_bytes,
             estimate.descriptor_offset_bytes,
             estimate.batch_metadata_bytes}) {
      if (component_sum >
          std::numeric_limits<std::uint64_t>::max() - component) {
        return false;
      }
      component_sum += component;
    }
    std::uint64_t expected_allocated_lanes = 0U;
    std::uint64_t expected_distance_bytes = 0U;
    std::uint64_t expected_reset_bytes = 0U;
    std::uint64_t preparation_sum = estimate.distance_reset_bytes;
    if (!multiply_without_overflow(
            estimate.storage_vertex_count,
            estimate.lane_width,
            expected_allocated_lanes) ||
        !multiply_without_overflow(
            expected_allocated_lanes,
            estimate.distance_slot_count,
            expected_distance_bytes) ||
        !multiply_without_overflow(
            expected_distance_bytes,
            sizeof(float),
            expected_distance_bytes) ||
        !multiply_without_overflow(
            estimate.selected_lane_vertex_count,
            estimate.distance_slot_count,
            expected_reset_bytes) ||
        !multiply_without_overflow(
            expected_reset_bytes, sizeof(float), expected_reset_bytes) ||
        preparation_sum > std::numeric_limits<std::uint64_t>::max() -
                              estimate.tile_mapping_write_bytes) {
      return false;
    }
    preparation_sum += estimate.tile_mapping_write_bytes;
    if (preparation_sum > std::numeric_limits<std::uint64_t>::max() -
                              estimate.run_preparation_write_bytes) {
      return false;
    }
    preparation_sum += estimate.run_preparation_write_bytes;
    if (component_sum != estimate.total_workspace_bytes ||
        expected_allocated_lanes != estimate.allocated_lane_vertex_count ||
        expected_distance_bytes != estimate.distance_bytes ||
        expected_reset_bytes != estimate.distance_reset_bytes ||
        preparation_sum != estimate.total_preparation_write_bytes ||
        estimate.selected_lane_vertex_count >
            estimate.allocated_lane_vertex_count ||
        estimate.wasted_lane_vertex_count !=
            estimate.allocated_lane_vertex_count -
                estimate.selected_lane_vertex_count) {
      return false;
    }

    std::uint64_t total_csr_runs = 0U;
    std::uint64_t total_csc_runs = 0U;
    if (!add_without_overflow(
            estimate.active_csr_runs,
            estimate.zero_csr_runs,
            total_csr_runs) ||
        !add_without_overflow(
            estimate.active_csc_runs,
            estimate.zero_csc_runs,
            total_csc_runs)) {
      return false;
    }
    if (estimate.vertex_storage ==
        BatchVertexStorageStrategy::full_graph_vertex_major) {
      if (estimate.storage_vertex_count != decision.graph_vertex_count ||
          estimate.tile_mapping_bytes != 0U ||
          estimate.tile_mapping_write_bytes != 0U ||
          estimate.mapping_build_nanoseconds != 0U) {
        return false;
      }
    } else {
      // The compact mapping is a dense uint32 bias table. A valid nonempty
      // batch writes at least one entry. Initialization and stale clearing are
      // mutually exclusive, so either path plus current writes touches at
      // most two table images.
      const std::uint64_t mapping_entries =
          estimate.tile_mapping_bytes / sizeof(std::uint32_t);
      const std::uint64_t mapping_write_entries =
          estimate.tile_mapping_write_bytes / sizeof(std::uint32_t);
      const std::uint64_t minimum_entries_for_writes =
          mapping_write_entries / 2U +
          (mapping_write_entries % 2U != 0U ? 1U : 0U);
      if (estimate.storage_vertex_count != estimate.union_vertex_count ||
          estimate.tile_mapping_bytes != expected_tile_mapping_bytes ||
          estimate.tile_mapping_write_bytes == 0U ||
          estimate.tile_mapping_bytes % sizeof(std::uint32_t) != 0U ||
          estimate.tile_mapping_write_bytes % sizeof(std::uint32_t) != 0U ||
          mapping_entries < minimum_entries_for_writes) {
        return false;
      }
    }

    std::uint64_t expected_run_storage_bytes = 0U;
    if (estimate.run_representation ==
        BatchRunRepresentation::retained_per_run_masks) {
      std::uint64_t minimum_run_preparation_bytes = 0U;
      if (!multiply_without_overflow(
              std::max(total_csr_runs, total_csc_runs),
              sizeof(LaneMask),
              expected_run_storage_bytes) ||
          !multiply_without_overflow(
              std::max(
                  estimate.active_csr_runs, estimate.active_csc_runs),
              sizeof(LaneMask),
              minimum_run_preparation_bytes) ||
          estimate.run_storage_bytes != expected_run_storage_bytes ||
          estimate.descriptor_offset_bytes != 0U ||
          estimate.run_preparation_write_bytes <
              minimum_run_preparation_bytes ||
          estimate.run_preparation_write_bytes % sizeof(LaneMask) != 0U) {
        return false;
      }
      // Initialization/clear and current writes each touch no more than one
      // retained image for the active orientation.
      if (estimate.run_preparation_write_bytes / 2U >
              estimate.run_storage_bytes ||
          (estimate.run_preparation_write_bytes % 2U != 0U &&
           estimate.run_preparation_write_bytes / 2U >=
               estimate.run_storage_bytes)) {
        return false;
      }
    } else {
      std::uint64_t union_vertices_with_sentinel = 0U;
      std::uint64_t expected_descriptor_offset_bytes = 0U;
      std::uint64_t expected_run_preparation_bytes = 0U;
      if (!multiply_without_overflow(
              std::max(
                  estimate.active_csr_runs, estimate.active_csc_runs),
              sizeof(RunLaneMaskDescriptor),
              expected_run_storage_bytes) ||
          !add_without_overflow(
              estimate.union_vertex_count,
              1U,
              union_vertices_with_sentinel) ||
          !multiply_without_overflow(
              union_vertices_with_sentinel,
              sizeof(std::uint32_t),
              expected_descriptor_offset_bytes) ||
          !add_without_overflow(
              expected_run_storage_bytes,
              expected_descriptor_offset_bytes,
              expected_run_preparation_bytes) ||
          estimate.run_storage_bytes != expected_run_storage_bytes ||
          estimate.descriptor_offset_bytes !=
              expected_descriptor_offset_bytes ||
          estimate.run_preparation_write_bytes !=
              expected_run_preparation_bytes) {
        return false;
      }
    }
    const std::uint64_t expected_concurrent =
        committed <= decision.budget.device_capacity_bytes
            ? (decision.budget.device_capacity_bytes - committed) /
                  estimate.total_workspace_bytes
            : 0U;
    if (estimate.maximum_concurrent_workspaces != expected_concurrent) {
      return false;
    }
    const std::size_t vertex =
        estimate.vertex_storage ==
                BatchVertexStorageStrategy::compact_union_tiles
            ? 1U
            : 0U;
    const std::size_t runs =
        estimate.run_representation ==
                BatchRunRepresentation::compact_nonzero_descriptors
            ? 1U
            : 0U;
    const std::size_t index = vertex * 2U + runs;
    if (combinations[index]) {
      return false;
    }
    combinations[index] = true;
  }
  if (!std::all_of(
          combinations.begin(), combinations.end(), [](const bool value) {
            return value;
          })) {
    return false;
  }
  const BatchWorkspaceEstimate* const selected =
      find_selected_estimate(decision);
  if (selected == nullptr || selected->maximum_concurrent_workspaces == 0U ||
      decision.selected_mapping_build_nanoseconds !=
          selected->mapping_build_nanoseconds ||
      decision.selected_run_build_nanoseconds !=
          selected->run_build_nanoseconds) {
    return false;
  }
  const auto find_row = [&decision](
                            const BatchVertexStorageStrategy vertices,
                            const BatchRunRepresentation runs) noexcept {
    return std::find_if(
        decision.compared_strategies.begin(),
        decision.compared_strategies.end(),
        [vertices, runs](const BatchWorkspaceEstimate& estimate) {
          return estimate.vertex_storage == vertices &&
                 estimate.run_representation == runs;
        });
  };
  const auto full_retained = find_row(
      BatchVertexStorageStrategy::full_graph_vertex_major,
      BatchRunRepresentation::retained_per_run_masks);
  const auto full_descriptors = find_row(
      BatchVertexStorageStrategy::full_graph_vertex_major,
      BatchRunRepresentation::compact_nonzero_descriptors);
  const auto compact_retained = find_row(
      BatchVertexStorageStrategy::compact_union_tiles,
      BatchRunRepresentation::retained_per_run_masks);
  const auto compact_descriptors = find_row(
      BatchVertexStorageStrategy::compact_union_tiles,
      BatchRunRepresentation::compact_nonzero_descriptors);
  if (full_retained == decision.compared_strategies.end() ||
      full_descriptors == decision.compared_strategies.end() ||
      compact_retained == decision.compared_strategies.end() ||
      compact_descriptors == decision.compared_strategies.end()) {
    return false;
  }

  const auto same_batch_evidence = [](
                                       const BatchWorkspaceEstimate& left,
                                       const BatchWorkspaceEstimate& right) {
    return left.union_vertex_count == right.union_vertex_count &&
           left.selected_lane_vertex_count ==
               right.selected_lane_vertex_count &&
           left.batch_metadata_bytes == right.batch_metadata_bytes &&
           left.active_csr_runs == right.active_csr_runs &&
           left.active_csc_runs == right.active_csc_runs &&
           left.zero_csr_runs == right.zero_csr_runs &&
           left.zero_csc_runs == right.zero_csc_runs;
  };
  const auto same_vertex_components = [](
                                          const BatchWorkspaceEstimate& left,
                                          const BatchWorkspaceEstimate& right) {
    return left.storage_vertex_count == right.storage_vertex_count &&
           left.allocated_lane_vertex_count ==
               right.allocated_lane_vertex_count &&
           left.wasted_lane_vertex_count == right.wasted_lane_vertex_count &&
           left.distance_bytes == right.distance_bytes &&
           left.distance_reset_bytes == right.distance_reset_bytes &&
           left.tile_mapping_bytes == right.tile_mapping_bytes &&
           left.tile_mapping_write_bytes ==
               right.tile_mapping_write_bytes &&
           left.mapping_build_nanoseconds ==
               right.mapping_build_nanoseconds;
  };
  const auto same_run_components = [](
                                       const BatchWorkspaceEstimate& left,
                                       const BatchWorkspaceEstimate& right) {
    return left.run_storage_bytes == right.run_storage_bytes &&
           left.descriptor_offset_bytes ==
               right.descriptor_offset_bytes &&
           left.run_preparation_write_bytes ==
               right.run_preparation_write_bytes &&
           left.run_build_nanoseconds == right.run_build_nanoseconds;
  };

  return same_batch_evidence(*full_retained, *full_descriptors) &&
         same_batch_evidence(*full_retained, *compact_retained) &&
         same_batch_evidence(*full_retained, *compact_descriptors) &&
         same_vertex_components(*full_retained, *full_descriptors) &&
         same_vertex_components(*compact_retained, *compact_descriptors) &&
         same_run_components(*full_retained, *compact_retained) &&
         same_run_components(*full_descriptors, *compact_descriptors) &&
         full_retained->storage_vertex_count >=
             full_retained->union_vertex_count;
}

std::string serialize_batch_workspace_decision(
    const BatchWorkspaceDecision& decision) {
  if (!validate_batch_workspace_decision(decision)) {
    throw std::invalid_argument{
        "cannot serialize an invalid batch workspace decision"};
  }
  std::string output;
  output += "bfnew_phase13_workspace_decision_v1\n";
  output += "plan_fingerprint=" + std::to_string(decision.plan_fingerprint) +
            "\n";
  output += "measurement_scope=" +
            std::to_string(
                static_cast<std::uint32_t>(decision.measurement_scope)) +
            "\n";
  output += "device_capacity_bytes=" +
            std::to_string(decision.budget.device_capacity_bytes) + "\n";
  output += "resident_graph_bytes=" +
            std::to_string(decision.budget.resident_graph_bytes) + "\n";
  output += "explicit_reserve_bytes=" +
            std::to_string(decision.budget.explicit_reserve_bytes) + "\n";
  output += "graph_vertex_count=" +
            std::to_string(decision.graph_vertex_count) + "\n";
  output += "graph_tile_count=" +
            std::to_string(decision.graph_tile_count) + "\n";
  output += "lane_width=" + std::to_string(decision.lane_width) + "\n";
  output += "distance_slot_count=" +
            std::to_string(decision.distance_slot_count) + "\n";
  output += "selected_vertex_storage=" +
            std::string{vertex_storage_name(
                decision.selected_vertex_storage)} +
            "\n";
  output += "selected_run_representation=" +
            std::string{run_representation_name(
                decision.selected_run_representation)} +
            "\n";
  output += "selected_mapping_build_nanoseconds=" +
            std::to_string(decision.selected_mapping_build_nanoseconds) +
            "\n";
  output += "selected_run_build_nanoseconds=" +
            std::to_string(decision.selected_run_build_nanoseconds) + "\n";
  output += "run_build_scope=dual_orientation_host_proof_image\n";
  output += "reason=" + decision.quantitative_reason + "\n";
  output +=
      "vertex_storage\trun_representation\ttotal_bytes\t"
      "total_preparation_write_bytes\t"
      "wasted_lane_vertices\tactive_csr_runs\tactive_csc_runs\t"
      "mapping_build_ns\tdual_orientation_host_run_build_ns\t"
      "max_concurrent\n";
  for (const BatchWorkspaceEstimate& estimate :
       decision.compared_strategies) {
    output += std::string{vertex_storage_name(estimate.vertex_storage)} + "\t" +
              run_representation_name(estimate.run_representation) + "\t" +
              std::to_string(estimate.total_workspace_bytes) + "\t" +
              std::to_string(estimate.total_preparation_write_bytes) + "\t" +
              std::to_string(estimate.wasted_lane_vertex_count) + "\t" +
              std::to_string(estimate.active_csr_runs) + "\t" +
              std::to_string(estimate.active_csc_runs) + "\t" +
              std::to_string(estimate.mapping_build_nanoseconds) + "\t" +
              std::to_string(estimate.run_build_nanoseconds) + "\t" +
              std::to_string(estimate.maximum_concurrent_workspaces) + "\n";
  }
  return output;
}

}  // namespace bfnew
