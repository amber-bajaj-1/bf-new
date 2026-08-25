#pragma once

#include "bfnew/batch_layout.hpp"
#include "bfnew/compact_path_types.hpp"
#include "bfnew/dense_chaotic_push.hpp"
#include "bfnew/device_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace bfnew {

// Phase 17 deliberately has no production schedule default.  A caller must
// name a schedule explicitly or obtain one from measured comparison evidence.
enum class ExpansionScheduleKind : std::uint8_t {
  unspecified,
  one_geometric_ring,
  fixed_larger_ring,
  doubling_xy_margins,
  hybrid_small_then_doubling,
};

enum class ExpansionTerminalPolicy : std::uint8_t {
  full_region_fallback,
  explicit_failure,
};

struct ExpansionSchedulePolicy {
  ExpansionScheduleKind kind{ExpansionScheduleKind::unspecified};
  // Used only by fixed_larger_ring.  It is the additive number of geometric
  // tile rings applied to each x/y side per expansion.
  std::uint32_t fixed_ring_size{};
  // Used only by hybrid_small_then_doubling.  This many initial expansions
  // add one ring; subsequent expansions double the intended x/y margins.
  std::uint32_t hybrid_small_expansion_count{};

  constexpr bool operator==(
      const ExpansionSchedulePolicy&) const noexcept = default;
};

[[nodiscard]] constexpr ExpansionSchedulePolicy one_ring_expansion() noexcept {
  return ExpansionSchedulePolicy{
      ExpansionScheduleKind::one_geometric_ring, 0U, 0U};
}

[[nodiscard]] constexpr ExpansionSchedulePolicy fixed_ring_expansion(
    const std::uint32_t ring_size) noexcept {
  return ExpansionSchedulePolicy{
      ExpansionScheduleKind::fixed_larger_ring, ring_size, 0U};
}

[[nodiscard]] constexpr ExpansionSchedulePolicy doubling_margin_expansion()
    noexcept {
  return ExpansionSchedulePolicy{
      ExpansionScheduleKind::doubling_xy_margins, 0U, 0U};
}

[[nodiscard]] constexpr ExpansionSchedulePolicy hybrid_margin_expansion(
    const std::uint32_t small_expansion_count) noexcept {
  return ExpansionSchedulePolicy{
      ExpansionScheduleKind::hybrid_small_then_doubling,
      0U,
      small_expansion_count};
}

struct BatchedExpansionOptions {
  GpuRunOptions run_options{};
  BatchPlannerPolicy planner_policy{};
  // Nonzero caller-owned identity for execution settings that are outside the
  // shared engine/planner options, such as a HIP transfer/load strategy or a
  // portable run representation. It is folded into schedule-comparison
  // evidence so unlike runner configurations cannot be compared silently.
  std::uint64_t execution_configuration_fingerprint{};
  ExpansionSchedulePolicy schedule{};
  // This limits scheduled geometric expansions.  A configured full-region
  // fallback is a separate, final restart and occurs at most once.
  std::uint32_t maximum_expansions{};
  ExpansionTerminalPolicy terminal_policy{
      ExpansionTerminalPolicy::full_region_fallback};
  // Phase 18 production mode. Every valid lane in a clean batch returns its
  // generation-bound CompactPathPayload before workspace reuse. Retry misses
  // are discarded; terminal full-region misses retain their classification.
  std::uint32_t enable_compact_paths{};

  [[nodiscard]] constexpr bool operator==(
      const BatchedExpansionOptions& other) const noexcept {
    return run_options.engine == other.run_options.engine &&
           run_options.control_mode == other.run_options.control_mode &&
           run_options.rounds_per_chunk ==
               other.run_options.rounds_per_chunk &&
           run_options.block_size == other.run_options.block_size &&
           run_options.grid_policy == other.run_options.grid_policy &&
           run_options.blocks_per_wgp == other.run_options.blocks_per_wgp &&
           run_options.instrumentation == other.run_options.instrumentation &&
           run_options.maximum_rounds == other.run_options.maximum_rounds &&
           run_options.enable_per_lane_convergence ==
               other.run_options.enable_per_lane_convergence &&
           planner_policy == other.planner_policy &&
           execution_configuration_fingerprint ==
               other.execution_configuration_fingerprint &&
           schedule == other.schedule &&
           maximum_expansions == other.maximum_expansions &&
           terminal_policy == other.terminal_policy &&
           enable_compact_paths == other.enable_compact_paths;
  }
};

enum class BatchedExpansionOptionsError : std::uint8_t {
  none,
  invalid_run_options,
  invalid_planner_policy,
  missing_execution_configuration_fingerprint,
  unspecified_schedule,
  invalid_schedule,
  invalid_fixed_ring_size,
  unexpected_fixed_ring_size,
  invalid_hybrid_small_expansion_count,
  unexpected_hybrid_small_expansion_count,
  invalid_terminal_policy,
  invalid_compact_paths_flag,
};

[[nodiscard]] BatchedExpansionOptionsError validate_batched_expansion_options(
    const BatchedExpansionOptions& options) noexcept;

enum class ExpansionWorkEvidence : std::uint8_t {
  unavailable,
  measured,
};

struct ExpansionBatchContext {
  std::uint64_t execution_ordinal{};
  std::uint32_t planning_pass{};
  std::uint32_t batch_index{};
  bool retry_pass{};

  constexpr bool operator==(const ExpansionBatchContext&) const noexcept =
      default;
};

struct ExpansionBatchExecution {
  GpuSsspResult result{};
  ExpansionWorkEvidence work_evidence{ExpansionWorkEvidence::unavailable};
  // Shared work counts physical graph-edge record consumption; logical work
  // counts per-lane edge relaxations.  Zero is meaningful only when evidence
  // is measured.  A timing run may leave both unavailable.
  std::uint64_t shared_edge_work{};
  std::uint64_t logical_lane_edge_work{};
  // Optional portable correctness image, vertex-major and lane-contiguous.
  // HIP status-only execution leaves this empty.  A nonempty image must have
  // exactly graph.vertex_count() * batch.lane_width entries.
  std::vector<float> final_distances;
  // When compact production is enabled, a clean batch contains exactly one
  // payload per valid lane, sorted by QueryId. Retry-miss payloads are
  // discarded; a terminal full-region miss retains its mixed
  // complete/unreachable target summaries.
  std::vector<CompactPathPayload> compact_paths;
  CompactTransferAccounting compact_transfer{};
  // Fixed control/error transfers that accompany the compact payload. The
  // portable adapter leaves these zero; HIP reports the exact status and
  // reconstruction-error words copied at its mandatory transfer boundaries.
  std::uint64_t compact_status_bytes{};
  std::uint64_t compact_error_bytes{};
  std::uint64_t compact_total_device_to_host_bytes{};
  // Exact convergence-controller polling is separate from the compact
  // payload/status/error subtotal above. Persistent execution reports zero.
  std::uint64_t compact_controller_poll_count{};
  std::uint64_t compact_controller_poll_bytes{};
  std::uint64_t compact_overall_device_to_host_bytes{};
  CompactPathExecutionMetrics compact_execution{};
};

using ExpansionBatchRunner = std::function<ExpansionBatchExecution(
    std::span<const RouteQuery> queries,
    std::span<const BatchQueryFeatures> features,
    const BatchPlanEntry& batch,
    const ExpansionBatchContext& context)>;

enum class ExpansionQueryDisposition : std::uint8_t {
  reached,
  unreachable_in_full_region,
  expansion_limit,
  region_stalled,
  identity_or_count_overflow,
  engine_failure,
};

struct ExpansionQueryOutcome {
  RouteQuery final_query;
  ExpansionQueryDisposition disposition{
      ExpansionQueryDisposition::engine_failure};
  std::uint32_t attempts{};
  std::uint32_t scheduled_expansions{};
  std::uint32_t total_expansions{};
  bool used_full_region_fallback{};
  std::uint64_t selected_vertex_count{};
  std::uint64_t selected_edge_count{};
  std::uint32_t terminal_stop_reason{};
  std::uint32_t terminal_error_bits{};
  // Present only when the runner supplied a full correctness image for the
  // terminal attempt.  It contains one graph-sized lane projection.
  std::vector<float> final_distances;
  // Present only for a reached terminal generation or a terminal full-region
  // miss captured before workspace reuse. A retry discards both stale labels
  // and stale compact paths; nonclean and label-free failures carry none.
  std::optional<CompactPathPayload> compact_paths;

  [[nodiscard]] constexpr bool reached() const noexcept {
    return disposition == ExpansionQueryDisposition::reached;
  }
};

struct ExpansionBatchTrace {
  ExpansionBatchContext context{};
  std::uint32_t lane_width{};
  LaneMask valid_lane_mask{};
  LaneMask reached_lane_mask{};
  LaneMask miss_lane_mask{};
  std::vector<QueryId> query_ids_by_lane;
  std::vector<std::uint32_t> expansion_generations_by_lane;
  std::uint64_t union_vertex_count{};
  std::uint64_t union_edge_estimate{};
  std::uint64_t selected_lane_vertex_count{};
  std::uint64_t selected_lane_edge_estimate{};
  std::uint64_t shared_edge_work{};
  std::uint64_t logical_lane_edge_work{};
  ExpansionWorkEvidence work_evidence{ExpansionWorkEvidence::unavailable};
  std::uint32_t stop_reason{};
  std::uint32_t error_bits{};

  bool operator==(const ExpansionBatchTrace&) const = default;
};

struct BatchedExpansionMetrics {
  std::uint64_t input_queries{};
  std::uint64_t initial_reached_queries{};
  std::uint64_t reached_queries{};
  std::uint64_t unreachable_full_region_queries{};
  std::uint64_t expansion_limit_queries{};
  std::uint64_t stalled_region_queries{};
  std::uint64_t identity_or_count_overflow_queries{};
  std::uint64_t engine_failure_queries{};

  std::uint64_t planning_passes{};
  std::uint64_t batches_executed{};
  std::uint64_t initial_batches_executed{};
  std::uint64_t retry_batches_executed{};
  std::uint64_t failed_lane_observations{};
  std::uint64_t failed_origin_valid_lane_observations{};
  std::uint64_t retry_valid_lane_observations{};
  std::uint64_t retry_lane_capacity{};

  std::uint64_t scheduled_expansions{};
  std::uint64_t full_region_fallbacks{};
  std::uint64_t repeated_selected_edge_estimate{};
  std::uint64_t attempted_selected_vertex_count{};
  std::uint64_t attempted_selected_edge_estimate{};
  std::uint64_t final_selected_tile_count{};
  std::uint64_t final_selected_vertex_count{};
  std::uint64_t final_selected_edge_count{};

  ExpansionWorkEvidence work_evidence{ExpansionWorkEvidence::measured};
  std::uint64_t work_measured_batches{};
  std::uint64_t shared_edge_work{};
  std::uint64_t logical_lane_edge_work{};
  // Retry work is actual measured work from planning passes after the first.
  // Failed-batch work counts whole batches that produced at least one miss;
  // shared physical work cannot be apportioned exactly between reached and
  // missed lanes in a mixed batch.
  std::uint64_t retry_work_measured_batches{};
  std::uint64_t retry_shared_edge_work{};
  std::uint64_t retry_logical_lane_edge_work{};
  std::uint64_t failed_batch_work_measured_batches{};
  std::uint64_t failed_batch_shared_edge_work{};
  std::uint64_t failed_batch_logical_lane_edge_work{};
  DeviceWorkStatistics device_work{};

  CompactStageTimingEvidence compact_device_timing{
      CompactStageTimingEvidence::unavailable};
  CompactStageTimingEvidence compact_host_timing{
      CompactStageTimingEvidence::unavailable};
  std::uint64_t compact_device_timing_measured_batches{};
  std::uint64_t compact_host_timing_measured_batches{};
  std::uint64_t sssp_device_nanoseconds{};
  std::uint64_t reconstruction_device_nanoseconds{};
  std::uint64_t result_transfer_device_nanoseconds{};
  std::uint64_t sssp_host_nanoseconds{};
  std::uint64_t reconstruction_host_nanoseconds{};
  std::uint64_t result_transfer_host_nanoseconds{};
  std::uint64_t compact_end_to_end_host_nanoseconds{};
  std::uint64_t geometric_expansion_host_nanoseconds{};
  CompactTransferAccounting compact_transfer{};
  std::uint64_t compact_status_bytes{};
  std::uint64_t compact_error_bytes{};
  std::uint64_t compact_total_device_to_host_bytes{};
  std::uint64_t compact_controller_poll_count{};
  std::uint64_t compact_controller_poll_bytes{};
  std::uint64_t compact_overall_device_to_host_bytes{};

  // CPU host-wall observations.  They are not GPU timing and are excluded
  // from deterministic trace equality.
  std::uint64_t initial_planning_nanoseconds{};
  std::uint64_t replanning_nanoseconds{};
  std::uint64_t execution_nanoseconds{};
  std::uint64_t total_nanoseconds{};
  // Floor of queries/second scaled by 1,000.  Zero means unavailable when
  // total_nanoseconds is zero.
  std::uint64_t all_query_throughput_milliqueries_per_second{};

  // Deterministic identity of graph, initial queries, engine/control/tuning,
  // planner, limit, and terminal policy.  The schedule itself is excluded so
  // records from the four schedule candidates remain comparable.
  std::uint64_t schedule_comparison_fingerprint{};

  // Index i counts terminal queries that performed i total expansions,
  // including the optional final full-region fallback.
  std::vector<std::uint64_t> expansion_count_histogram;
};

struct BatchedExpansionRunResult {
  // Canonical QueryId order, independent of input order and retry-local plan
  // indices.
  std::vector<ExpansionQueryOutcome> queries;
  std::vector<ExpansionBatchTrace> trace;
  BatchedExpansionMetrics metrics{};
};

// Orchestrates compact-status execution, reachability-only retry collection,
// geometric expansion, and deterministic replanning.  The runner owns the
// engine/workspace mechanics and must restart every request from original
// query sources.  Controller errors and maximum-round exits never expand.
[[nodiscard]] BatchedExpansionRunResult run_batched_expansion(
    const WeightedGraph& graph,
    const TileDirectory& directory,
    const TileRunLayout64& tile_runs,
    std::span<const RouteQuery> queries,
    const BatchedExpansionOptions& options,
    const ExpansionBatchRunner& runner);

struct HostBatchedExpansionOptions {
  BatchRunRepresentation run_representation{
      BatchRunRepresentation::retained_per_run_masks};
  DenseHostSchedule dense_schedule{DenseHostSchedule::csr_forward};
  std::size_t frontier_queue_capacity{};
};

// Portable Phase 17 adapter.  Each callback invocation rebuilds the batch
// description and invokes exactly one existing Phase 14/15/16 host engine;
// no distance image is carried from a failed attempt into a retry.
[[nodiscard]] BatchedExpansionRunResult run_host_batched_expansion(
    const WeightedGraph& graph,
    const TileDirectory& directory,
    const TileRunLayout64& tile_runs,
    const DeviceGraphLayout32& device_graph,
    std::span<const RouteQuery> queries,
    const BatchedExpansionOptions& options,
    const HostBatchedExpansionOptions& host_options = {});

struct ExpansionScheduleEvidence {
  ExpansionSchedulePolicy schedule{};
  std::uint64_t input_queries{};
  std::uint64_t reached_queries{};
  std::uint64_t terminal_failures{};
  std::uint64_t batches_executed{};
  std::uint64_t total_expansions{};
  std::uint64_t attempted_selected_edge_estimate{};
  ExpansionWorkEvidence work_evidence{ExpansionWorkEvidence::unavailable};
  std::uint64_t shared_edge_work{};
  std::uint64_t logical_lane_edge_work{};
  std::uint64_t comparison_fingerprint{};
  // False if the run contained an engine/identity failure or otherwise cannot
  // participate in a production schedule recommendation.
  bool campaign_valid{};

  constexpr bool operator==(const ExpansionScheduleEvidence&) const noexcept =
      default;
};

[[nodiscard]] ExpansionScheduleEvidence make_expansion_schedule_evidence(
    const ExpansionSchedulePolicy& schedule,
    const BatchedExpansionRunResult& result) noexcept;

// Returns no recommendation unless evidence contains one comparable record
// for every schedule kind.  Correct completion is ranked before measured work,
// then deterministic selected-edge/retry evidence breaks unavailable-work
// cases. This is the only Phase 17 evidence-backed recommendation mechanism;
// bounded synthetic evidence is not a production default.
[[nodiscard]] std::optional<ExpansionSchedulePolicy>
select_expansion_schedule_from_evidence(
    std::span<const ExpansionScheduleEvidence> evidence) noexcept;

}  // namespace bfnew
