#include "bfnew/hip/batched_expansion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace bfnew::hip {
namespace {

void validate_binding(
    const GpuRunOptions& options,
    const EngineKind required_engine,
    const BatchedExpansionTransferMode transfer_mode) {
  if (validate_gpu_run_options(options) != GpuRunOptionsError::none ||
      options.engine != required_engine) {
    throw std::invalid_argument{
        "batched expansion executor received invalid or mismatched run options"};
  }
  if (!valid_batched_expansion_transfer_mode(transfer_mode)) {
    throw std::invalid_argument{
        "batched expansion executor received an unknown transfer mode"};
  }
  if ((transfer_mode == BatchedExpansionTransferMode::compact_status ||
       transfer_mode == BatchedExpansionTransferMode::compact_paths) &&
      options.instrumentation != InstrumentationLevel::none) {
    throw std::invalid_argument{
        "compact batched expansion requires None instrumentation; use the "
        "evidence transfer mode for instrumented execution"};
  }
}

[[nodiscard]] std::uint64_t milliseconds_to_nanoseconds(
    const double milliseconds,
    const char* const stage) {
  constexpr double nanoseconds_per_millisecond = 1'000'000.0;
  if (!std::isfinite(milliseconds) || milliseconds < 0.0 ||
      milliseconds >
          static_cast<double>(std::numeric_limits<std::uint64_t>::max()) /
              nanoseconds_per_millisecond) {
    throw std::runtime_error{std::string{"invalid compact-path "} + stage +
                             " duration"};
  }
  return static_cast<std::uint64_t>(
      milliseconds * nanoseconds_per_millisecond);
}

[[nodiscard]] const RouteQuery& find_executor_query(
    const std::span<const RouteQuery> queries,
    const QueryId query_id,
    const std::uint32_t generation) {
  const RouteQuery* found = nullptr;
  for (const RouteQuery& query : queries) {
    if (query.query_id != query_id) {
      continue;
    }
    if (found != nullptr || query.expansion_generation != generation) {
      throw std::runtime_error{
          "compact-path executor received ambiguous query identity"};
    }
    found = &query;
  }
  if (found == nullptr) {
    throw std::runtime_error{
        "compact-path executor could not resolve a retry-local query"};
  }
  return *found;
}

[[nodiscard]] ExpansionBatchExecution compact_path_execution(
    const EngineKind engine,
    const ControlMode control,
    CompactPathBatchOutput compact,
    const std::span<const RouteQuery> queries,
    const BatchDeviceDescription& description) {
  ExpansionBatchExecution execution;
  execution.result.engine_kind = static_cast<std::uint32_t>(engine);
  execution.result.control_mode = static_cast<std::uint32_t>(control);
  execution.result.status = compact.status;
  execution.work_evidence = ExpansionWorkEvidence::unavailable;
  execution.compact_transfer = compact.transfer;
  execution.compact_status_bytes = compact.transport.status_bytes;
  execution.compact_error_bytes = compact.transport.error_bytes;
  execution.compact_total_device_to_host_bytes =
      compact.transport.total_device_to_host_bytes;
  execution.compact_controller_poll_count =
      compact.transport.controller_poll_count;
  execution.compact_controller_poll_bytes =
      compact.transport.controller_poll_bytes;
  execution.compact_overall_device_to_host_bytes =
      compact.transport.overall_device_to_host_bytes;
  execution.compact_execution.device_timing =
      CompactStageTimingEvidence::measured;
  execution.compact_execution.host_timing =
      CompactStageTimingEvidence::measured;
  execution.compact_execution.sssp_device_nanoseconds =
      milliseconds_to_nanoseconds(
          compact.metrics.sssp_device_milliseconds, "SSSP device");
  execution.compact_execution.reconstruction_device_nanoseconds =
      milliseconds_to_nanoseconds(
          compact.metrics.reconstruction_device_milliseconds,
          "reconstruction device");
  execution.compact_execution.result_transfer_device_nanoseconds =
      milliseconds_to_nanoseconds(
          compact.metrics.result_transfer_device_milliseconds,
          "result-transfer device");
  // Named host-stage fields deliberately remain zero: the two mandatory D2H
  // boundaries are stream-ordered after reconstruction, so partitioning them
  // exactly would add production-only synchronizations. Device events are the
  // authoritative stage separation; this is the exact callback wall span.
  execution.compact_execution.end_to_end_host_nanoseconds =
      milliseconds_to_nanoseconds(
          compact.metrics.end_to_end_wall_milliseconds, "end-to-end host");

  if (compact.targets.size() != description.targets.size() ||
      description.target_offsets.size() != description.lane_width + 1U ||
      description.query_ids_by_lane.size() != description.lane_width ||
      description.expansion_generations_by_lane.size() !=
          description.lane_width) {
    throw std::runtime_error{
        "compact-path engine returned inconsistent target metadata"};
  }
  const bool clean =
      compact.status.stop_reason ==
          static_cast<std::uint32_t>(DeviceStopReason::converged) &&
      compact.status.error_bits == device_error::none;
  for (std::uint32_t lane = 0U; lane < description.lane_width; ++lane) {
    const LaneMask lane_bit = LaneMask{1U} << lane;
    if (!clean || (description.valid_lane_mask & lane_bit) == 0U) {
      continue;
    }
    const QueryId query_id{description.query_ids_by_lane[lane]};
    const std::uint32_t generation =
        description.expansion_generations_by_lane[lane];
    const RouteQuery& query =
        find_executor_query(queries, query_id, generation);
    const std::size_t begin = description.target_offsets[lane];
    const std::size_t end = description.target_offsets[lane + 1U];
    if (end < begin || end > compact.targets.size() ||
        end - begin != query.targets.size()) {
      throw std::runtime_error{
          "compact-path target slice disagrees with its query"};
    }
    CompactPathPayload payload;
    payload.query_id = query_id;
    payload.expansion_generation = generation;
    payload.target_terminal_to_target = query.target_terminal_to_target;
    payload.targets.reserve(end - begin);
    for (std::size_t target = begin; target < end; ++target) {
      payload.targets.push_back(std::move(compact.targets[target]));
    }
    execution.compact_paths.push_back(std::move(payload));
  }
  std::sort(
      execution.compact_paths.begin(),
      execution.compact_paths.end(),
      [](const CompactPathPayload& left, const CompactPathPayload& right) {
        return left.query_id.value() < right.query_id.value();
      });
  return execution;
}

[[nodiscard]] ExpansionBatchExecution compact_execution(
    const EngineKind engine,
    const ControlMode control,
    const DeviceRunStatus& status) {
  ExpansionBatchExecution execution;
  execution.result.engine_kind = static_cast<std::uint32_t>(engine);
  execution.result.control_mode = static_cast<std::uint32_t>(control);
  execution.result.status = status;
  execution.work_evidence = ExpansionWorkEvidence::unavailable;
  return execution;
}

}  // namespace

class BatchedExpansionExecutor::Impl final {
 public:
  Impl(
      const WeightedGraph& host_graph_value,
      const TileRunLayout64& tile_runs_value,
      const ResidentDeviceGraph& resident_graph,
      ReusableBatchedJacobiWorkspace& workspace,
      const HipStream& stream,
      const GpuRunOptions run_options_value,
      const BatchedExpansionTransferMode transfer_mode_value,
      const BatchedJacobiLoadStrategy load_strategy)
      : host_graph{host_graph_value},
        tile_runs{tile_runs_value},
        run_options{run_options_value},
        transfer{transfer_mode_value},
        jacobi_load_strategy{load_strategy} {
    validate_binding(run_options, EngineKind::jacobi_pull, transfer);
    if (!valid_batched_jacobi_load_strategy(jacobi_load_strategy)) {
      throw std::invalid_argument{
          "batched expansion executor received an invalid Jacobi load strategy"};
    }
    jacobi = std::make_unique<BatchedJacobiPullEngine>(
        host_graph, tile_runs, resident_graph, workspace, stream);
    if (transfer == BatchedExpansionTransferMode::compact_paths) {
      compact_workspace = std::make_unique<ReusableCompactPathWorkspace>();
    }
  }

  Impl(
      const WeightedGraph& host_graph_value,
      const TileRunLayout64& tile_runs_value,
      const ResidentDeviceGraph& resident_graph,
      ReusableBatchedDenseWorkspace& workspace,
      const HipStream& stream,
      const GpuRunOptions run_options_value,
      const BatchedExpansionTransferMode transfer_mode_value,
      const BatchedDenseLoadStrategy load_strategy)
      : host_graph{host_graph_value},
        tile_runs{tile_runs_value},
        run_options{run_options_value},
        transfer{transfer_mode_value},
        dense_load_strategy{load_strategy} {
    validate_binding(run_options, EngineKind::dense_chaotic_push, transfer);
    if (!valid_batched_dense_load_strategy(dense_load_strategy)) {
      throw std::invalid_argument{
          "batched expansion executor received an invalid dense load strategy"};
    }
    dense = std::make_unique<BatchedDenseChaoticPushEngine>(
        host_graph, tile_runs, resident_graph, workspace, stream);
    if (transfer == BatchedExpansionTransferMode::compact_paths) {
      compact_workspace = std::make_unique<ReusableCompactPathWorkspace>();
    }
  }

  Impl(
      const WeightedGraph& host_graph_value,
      const TileRunLayout64& tile_runs_value,
      const ResidentDeviceGraph& resident_graph,
      ReusableBatchedFrontierWorkspace& workspace,
      const HipStream& stream,
      const GpuRunOptions run_options_value,
      const BatchedExpansionTransferMode transfer_mode_value,
      const std::uint32_t queue_capacity)
      : host_graph{host_graph_value},
        tile_runs{tile_runs_value},
        run_options{run_options_value},
        transfer{transfer_mode_value} {
    validate_binding(run_options, EngineKind::frontier_push, transfer);
    frontier = std::make_unique<BatchedFrontierPushEngine>(
        host_graph,
        tile_runs,
        resident_graph,
        workspace,
        stream,
        queue_capacity);
    if (transfer == BatchedExpansionTransferMode::compact_paths) {
      compact_workspace = std::make_unique<ReusableCompactPathWorkspace>();
    }
  }

  [[nodiscard]] ExpansionBatchExecution execute(
      const std::span<const RouteQuery> queries,
      const std::span<const BatchQueryFeatures> features,
      const BatchPlanEntry& batch,
      const ExpansionBatchContext& context) {
    // The ordinal is evidence identity owned by the generic orchestrator. It
    // must not perturb preparation, engine choice, or relaxation semantics.
    static_cast<void>(context);
    prepare_batch_device_description(
        host_graph,
        tile_runs,
        queries,
        features,
        batch,
        jacobi != nullptr
            ? BatchRunRepresentation::device_materialized_run_masks
            : BatchRunRepresentation::retained_per_run_masks,
        description);

    if (jacobi != nullptr) {
      if (transfer == BatchedExpansionTransferMode::compact_paths) {
        return compact_path_execution(
            EngineKind::jacobi_pull,
            run_options.control_mode,
            jacobi->run_compact_paths(
                description,
                run_options,
                *compact_workspace,
                jacobi_load_strategy),
            queries,
            description);
      }
      if (transfer == BatchedExpansionTransferMode::compact_status) {
        return compact_execution(
            EngineKind::jacobi_pull,
            run_options.control_mode,
            jacobi->run_compact_status(
                description, run_options, jacobi_load_strategy));
      }
      BatchedJacobiRunOutput output = jacobi->run_status_only(
          description, run_options, jacobi_load_strategy);
      ExpansionBatchExecution execution;
      execution.result = std::move(output.result);
      execution.work_evidence = ExpansionWorkEvidence::measured;
      execution.shared_edge_work = output.metrics.csc_edge_records_loaded;
      execution.logical_lane_edge_work =
          output.metrics.admitted_lane_edge_pairs;
      return execution;
    }

    if (dense != nullptr) {
      if (transfer == BatchedExpansionTransferMode::compact_paths) {
        return compact_path_execution(
            EngineKind::dense_chaotic_push,
            run_options.control_mode,
            dense->run_compact_paths(
                description,
                run_options,
                *compact_workspace,
                dense_load_strategy),
            queries,
            description);
      }
      if (transfer == BatchedExpansionTransferMode::compact_status) {
        return compact_execution(
            EngineKind::dense_chaotic_push,
            run_options.control_mode,
            dense->run_compact_status(
                description, run_options, dense_load_strategy));
      }
      BatchedDenseRunOutput output = dense->run_status_only(
          description, run_options, dense_load_strategy);
      ExpansionBatchExecution execution;
      execution.result = std::move(output.result);
      execution.work_evidence = ExpansionWorkEvidence::measured;
      execution.shared_edge_work = output.batch_work.csr_edge_loads;
      execution.logical_lane_edge_work =
          output.batch_work.lane_edge_relaxations;
      return execution;
    }

    if (frontier != nullptr) {
      if (transfer == BatchedExpansionTransferMode::compact_paths) {
        return compact_path_execution(
            EngineKind::frontier_push,
            run_options.control_mode,
            frontier->run_compact_paths(
                description, run_options, *compact_workspace),
            queries,
            description);
      }
      if (transfer == BatchedExpansionTransferMode::compact_status) {
        return compact_execution(
            EngineKind::frontier_push,
            run_options.control_mode,
            frontier->run_compact_status(description, run_options));
      }
      BatchedFrontierRunOutput output =
          frontier->run_status_only(description, run_options);
      ExpansionBatchExecution execution;
      execution.result = std::move(output.result);
      if (run_options.instrumentation != InstrumentationLevel::none) {
        execution.work_evidence = ExpansionWorkEvidence::measured;
        execution.shared_edge_work = output.batch_work.csr_edge_loads;
        execution.logical_lane_edge_work =
            output.batch_work.lane_edge_relaxations;
      }
      return execution;
    }

    throw std::logic_error{
        "batched expansion executor has no bound engine"};
  }

  const WeightedGraph& host_graph;
  const TileRunLayout64& tile_runs;
  GpuRunOptions run_options{};
  BatchedExpansionTransferMode transfer{
      BatchedExpansionTransferMode::compact_status};
  BatchedJacobiLoadStrategy jacobi_load_strategy{
      BatchedJacobiLoadStrategy::compiler_uniform};
  BatchedDenseLoadStrategy dense_load_strategy{
      BatchedDenseLoadStrategy::compiler_uniform};
  BatchDeviceDescription description;
  std::unique_ptr<BatchedJacobiPullEngine> jacobi;
  std::unique_ptr<BatchedDenseChaoticPushEngine> dense;
  std::unique_ptr<BatchedFrontierPushEngine> frontier;
  std::unique_ptr<ReusableCompactPathWorkspace> compact_workspace;
};

BatchedExpansionExecutor::BatchedExpansionExecutor(
    const WeightedGraph& host_graph,
    const TileRunLayout64& tile_runs,
    const ResidentDeviceGraph& resident_graph,
    ReusableBatchedJacobiWorkspace& workspace,
    const HipStream& stream,
    const GpuRunOptions run_options,
    const BatchedExpansionTransferMode transfer_mode,
    const BatchedJacobiLoadStrategy load_strategy)
    : impl_{new Impl{
          host_graph,
          tile_runs,
          resident_graph,
          workspace,
          stream,
          run_options,
          transfer_mode,
          load_strategy}} {}

BatchedExpansionExecutor::BatchedExpansionExecutor(
    const WeightedGraph& host_graph,
    const TileRunLayout64& tile_runs,
    const ResidentDeviceGraph& resident_graph,
    ReusableBatchedDenseWorkspace& workspace,
    const HipStream& stream,
    const GpuRunOptions run_options,
    const BatchedExpansionTransferMode transfer_mode,
    const BatchedDenseLoadStrategy load_strategy)
    : impl_{new Impl{
          host_graph,
          tile_runs,
          resident_graph,
          workspace,
          stream,
          run_options,
          transfer_mode,
          load_strategy}} {}

BatchedExpansionExecutor::BatchedExpansionExecutor(
    const WeightedGraph& host_graph,
    const TileRunLayout64& tile_runs,
    const ResidentDeviceGraph& resident_graph,
    ReusableBatchedFrontierWorkspace& workspace,
    const HipStream& stream,
    const GpuRunOptions run_options,
    const BatchedExpansionTransferMode transfer_mode,
    const std::uint32_t queue_capacity)
    : impl_{new Impl{
          host_graph,
          tile_runs,
          resident_graph,
          workspace,
          stream,
          run_options,
          transfer_mode,
          queue_capacity}} {}

BatchedExpansionExecutor::~BatchedExpansionExecutor() { delete impl_; }

EngineKind BatchedExpansionExecutor::kind() const noexcept {
  return impl_ == nullptr ? EngineKind::jacobi_pull : impl_->run_options.engine;
}

BatchedExpansionTransferMode BatchedExpansionExecutor::transfer_mode()
    const noexcept {
  return impl_ == nullptr ? BatchedExpansionTransferMode::compact_status
                          : impl_->transfer;
}

ExpansionBatchExecution BatchedExpansionExecutor::operator()(
    const std::span<const RouteQuery> queries,
    const std::span<const BatchQueryFeatures> features,
    const BatchPlanEntry& batch,
    const ExpansionBatchContext& context) {
  if (impl_ == nullptr) {
    throw std::logic_error{
        "cannot execute a batched expansion with an invalid executor"};
  }
  return impl_->execute(queries, features, batch, context);
}

}  // namespace bfnew::hip
