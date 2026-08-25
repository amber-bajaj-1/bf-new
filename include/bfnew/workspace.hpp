#pragma once

#include "bfnew/gpu_api.hpp"
#include "bfnew/query.hpp"

#include <cstddef>
#include <cstdint>

namespace bfnew {

struct WorkspaceMemoryRequirements {
  EngineKind engine{EngineKind::jacobi_pull};
  InstrumentationLevel instrumentation{InstrumentationLevel::none};
  std::uint32_t lane_capacity{1U};
  std::size_t source_bytes{};
  std::size_t target_bytes{};
  std::size_t selected_tile_bytes{};
  std::size_t tile_lane_mask_bytes{};
  // The active engine consumes exactly one sparse orientation, so CSR and
  // CSC run masks share this retained allocation.
  std::size_t run_lane_mask_bytes{};
  std::size_t controller_bytes{};
  std::size_t status_bytes{};
  std::size_t instrumentation_bytes{};
  std::size_t engine_scratch_bytes{};
  // Device allocations only. Pinned staging is reported separately because
  // it consumes host memory rather than device memory.
  std::size_t total_bytes{};
  std::size_t pinned_staging_bytes{};
  std::size_t combined_total_bytes{};

  constexpr bool operator==(const WorkspaceMemoryRequirements&) const noexcept =
      default;
};

[[nodiscard]] WorkspaceMemoryRequirements estimate_workspace_memory(
    EngineKind engine,
    const RouteQuery& query,
    std::size_t tile_count,
    std::size_t csr_run_count,
    std::size_t csc_run_count,
    std::uint32_t lane_capacity,
    InstrumentationLevel instrumentation,
    std::size_t engine_scratch_bytes);

struct WorkspaceCapacity {
  std::size_t source_bytes{};
  std::size_t target_bytes{};
  std::size_t selected_tile_bytes{};
  std::size_t tile_lane_mask_bytes{};
  std::size_t run_lane_mask_bytes{};
  std::size_t controller_bytes{};
  std::size_t status_bytes{};
  std::size_t instrumentation_bytes{};
  std::size_t engine_scratch_bytes{};

  // Retained device capacity; HIP pinned-host staging is not represented by
  // this host-only allocation-pool model.
  [[nodiscard]] std::size_t total_bytes() const;
};

struct WorkspaceLease {
  EngineKind engine{EngineKind::jacobi_pull};
  std::uint64_t generation{};

  constexpr bool operator==(const WorkspaceLease&) const noexcept = default;
};

struct WorkspaceReservationResult {
  WorkspaceLease lease{};
  bool capacity_grew{};
  bool engine_changed{};
  bool clear_active_prefixes{};
};

class ReusableWorkspaceReservation {
 public:
  [[nodiscard]] WorkspaceReservationResult begin_query(
      const WorkspaceMemoryRequirements& requirements);

  [[nodiscard]] const WorkspaceCapacity& capacity() const noexcept {
    return capacity_;
  }

  [[nodiscard]] std::uint64_t growth_events() const noexcept {
    return growth_events_;
  }

  [[nodiscard]] bool accepts(const WorkspaceLease& lease) const noexcept;

 private:
  WorkspaceCapacity capacity_{};
  EngineKind active_engine_{EngineKind::jacobi_pull};
  bool has_active_engine_{};
  std::uint64_t generation_{};
  std::uint64_t growth_events_{};
};

}  // namespace bfnew
