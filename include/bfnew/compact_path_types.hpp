#pragma once

#include "bfnew/types.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace bfnew {

// Fixed-width values copied directly from a reconstruction device pass.
enum class CompactTargetReachStatus : std::uint32_t {
  not_reached = 0U,
  reached = 1U,
};

enum class CompactPathStatus : std::uint32_t {
  complete = 0U,
  unreachable = 1U,
  query_terminal_failure = 2U,
  no_tight_path = 3U,
  path_length_overflow = 4U,
};

// The first and only mandatory target-result transfer.  Path vertices, their
// actual final distance labels, and stable edge IDs are transferred separately
// after the host has inspected these lengths/statuses.  path_length is an edge
// count and is meaningful only when reconstruction is complete.  The explicit
// flag avoids assigning a sentinel value to a strong vertex ID.
struct CompactTargetSummary {
  VertexId target{};
  VertexId selected_source{};
  float distance{std::numeric_limits<float>::infinity()};
  std::uint32_t path_length{};
  CompactTargetReachStatus reached{CompactTargetReachStatus::not_reached};
  CompactPathStatus reconstruction{CompactPathStatus::query_terminal_failure};
  std::uint32_t has_selected_source{};

  constexpr bool operator==(const CompactTargetSummary&) const noexcept =
      default;
};

static_assert(std::is_standard_layout_v<CompactTargetSummary>);
static_assert(std::is_trivially_copyable_v<CompactTargetSummary>);
static_assert(sizeof(CompactTargetSummary) == 28U);
static_assert(alignof(CompactTargetSummary) == 4U);

struct CompactTargetPath {
  CompactTargetSummary summary{};
  // Global reordered vertex IDs.  A complete path has path_length + 1
  // vertices and distance labels, plus path_length stable logical edge IDs.
  // The aligned labels are compact tightness evidence from the actual final
  // lane; sampled validation therefore never requires a V-by-W label copy.
  std::vector<VertexId> vertices;
  std::vector<float> distance_labels;
  std::vector<EdgeId> edge_ids;

  bool operator==(const CompactTargetPath&) const = default;
};

// Low-level, generation-bound payload returned before an engine workspace may
// be reused.  Canonical targets appear exactly once in RouteQuery::targets
// order; duplicate original terminals retain identity through the map.
struct CompactPathPayload {
  QueryId query_id{};
  std::uint32_t expansion_generation{};
  std::vector<std::uint32_t> target_terminal_to_target;
  std::vector<CompactTargetPath> targets;

  bool operator==(const CompactPathPayload&) const = default;
};

struct CompactTransferAccounting {
  std::uint64_t summary_bytes{};
  std::uint64_t vertex_bytes{};
  std::uint64_t distance_label_bytes{};
  std::uint64_t edge_id_bytes{};
  std::uint64_t total_bytes{};

  constexpr bool operator==(const CompactTransferAccounting&) const noexcept =
      default;
};

enum class CompactStageTimingEvidence : std::uint32_t {
  unavailable = 0U,
  measured = 1U,
};

// Device-event durations and host-wall durations are kept separate.  Device
// fields must come from events on the execution stream; nonzero named host
// fields enclose only that stage.  An asynchronous implementation may leave
// the three named host fields zero rather than add synchronizations, while
// still measuring end_to_end_host_nanoseconds around the complete callback,
// including launch/control overhead between stages.
struct CompactPathExecutionMetrics {
  CompactStageTimingEvidence device_timing{
      CompactStageTimingEvidence::unavailable};
  CompactStageTimingEvidence host_timing{
      CompactStageTimingEvidence::unavailable};
  std::uint64_t sssp_device_nanoseconds{};
  std::uint64_t reconstruction_device_nanoseconds{};
  std::uint64_t result_transfer_device_nanoseconds{};
  std::uint64_t sssp_host_nanoseconds{};
  std::uint64_t reconstruction_host_nanoseconds{};
  std::uint64_t result_transfer_host_nanoseconds{};
  std::uint64_t end_to_end_host_nanoseconds{};

  constexpr bool operator==(const CompactPathExecutionMetrics&) const noexcept =
      default;
};

}  // namespace bfnew
