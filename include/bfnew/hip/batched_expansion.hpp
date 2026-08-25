#pragma once

#include "bfnew/batched_expansion.hpp"
#include "bfnew/hip/batched_dense_chaotic_push.hpp"
#include "bfnew/hip/batched_frontier_push.hpp"
#include "bfnew/hip/batched_jacobi.hpp"

#include <cstdint>
#include <span>

namespace bfnew::hip {

// Compact execution is the production all-query boundary: after the existing
// GPU finalizer classifies every valid lane, exactly one DeviceRunStatus is
// copied. Evidence execution retains the Phase 14/15/16 status-only download
// paths so logical/shared work can be reported when those paths expose it.
enum class BatchedExpansionTransferMode : std::uint8_t {
  compact_status = 0U,
  status_and_work_evidence = 1U,
  compact_paths = 2U,
};

[[nodiscard]] constexpr bool valid_batched_expansion_transfer_mode(
    const BatchedExpansionTransferMode mode) noexcept {
  return mode == BatchedExpansionTransferMode::compact_status ||
         mode == BatchedExpansionTransferMode::status_and_work_evidence ||
         mode == BatchedExpansionTransferMode::compact_paths;
}

// Sequential adapter for run_batched_expansion(). One instance is bound to
// exactly one engine, immutable run options, one reusable host description,
// and that engine's reusable device workspace. The referenced graph, tile-run
// layout, resident graph, workspace, and stream must outlive this object.
//
// Every callback rebuilds retry-local QueryId/region metadata. Jacobi
// materializes selected CSC masks on-device; push engines retain prepared CSR
// masks. Every engine starts from its ordinary selected-only reset/source-seed
// path. Failed-attempt labels are never downloaded or reused.
class BatchedExpansionExecutor final {
 public:
  BatchedExpansionExecutor(
      const WeightedGraph& host_graph,
      const TileRunLayout64& tile_runs,
      const ResidentDeviceGraph& resident_graph,
      ReusableBatchedJacobiWorkspace& workspace,
      const HipStream& stream,
      GpuRunOptions run_options,
      BatchedExpansionTransferMode transfer_mode =
          BatchedExpansionTransferMode::compact_status,
      BatchedJacobiLoadStrategy load_strategy =
          BatchedJacobiLoadStrategy::compiler_uniform);

  BatchedExpansionExecutor(
      const WeightedGraph& host_graph,
      const TileRunLayout64& tile_runs,
      const ResidentDeviceGraph& resident_graph,
      ReusableBatchedDenseWorkspace& workspace,
      const HipStream& stream,
      GpuRunOptions run_options,
      BatchedExpansionTransferMode transfer_mode =
          BatchedExpansionTransferMode::compact_status,
      BatchedDenseLoadStrategy load_strategy =
          BatchedDenseLoadStrategy::compiler_uniform);

  BatchedExpansionExecutor(
      const WeightedGraph& host_graph,
      const TileRunLayout64& tile_runs,
      const ResidentDeviceGraph& resident_graph,
      ReusableBatchedFrontierWorkspace& workspace,
      const HipStream& stream,
      GpuRunOptions run_options,
      BatchedExpansionTransferMode transfer_mode =
          BatchedExpansionTransferMode::compact_status,
      std::uint32_t queue_capacity = 0U);

  ~BatchedExpansionExecutor();

  BatchedExpansionExecutor(const BatchedExpansionExecutor&) = delete;
  BatchedExpansionExecutor& operator=(
      const BatchedExpansionExecutor&) = delete;
  BatchedExpansionExecutor(BatchedExpansionExecutor&&) = delete;
  BatchedExpansionExecutor& operator=(BatchedExpansionExecutor&&) = delete;

  [[nodiscard]] EngineKind kind() const noexcept;
  [[nodiscard]] BatchedExpansionTransferMode transfer_mode() const noexcept;

  // Signature-compatible with ExpansionBatchRunner. Use std::ref(executor)
  // when constructing that std::function so this non-owning binding is not
  // copied. Calls must remain sequential because the description/workspace is
  // deliberately reused between retries.
  [[nodiscard]] ExpansionBatchExecution operator()(
      std::span<const RouteQuery> queries,
      std::span<const BatchQueryFeatures> features,
      const BatchPlanEntry& batch,
      const ExpansionBatchContext& context);

 private:
  class Impl;
  Impl* impl_{};
};

}  // namespace bfnew::hip
