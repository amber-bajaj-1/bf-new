#include "bfnew/device_layout.hpp"
#include "bfnew/hip/runtime.hpp"
#include "bfnew/spatial.hpp"
#include "graph_fixtures.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const char* message) {
  std::cerr << "device transfer test failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void expect(const bool condition, const char* message) {
  if (!condition) {
    fail(message);
  }
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, const char* message) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  fail(message);
}

[[nodiscard]] bfnew::DeviceGraphLayout32 tiny_layout() {
  bfnew::DeviceGraphLayout32 layout;
  layout.vertex_count = 3U;
  layout.edge_count = 4U;
  layout.tile_count = 2U;
  layout.owner_tiles = {0U, 0U, 1U};

  layout.csr_row_offsets = {0U, 2U, 3U, 4U};
  layout.csr_destinations = {1U, 2U, 2U, 0U};
  layout.csr_weights = {1.25F, 2.5F, 3.75F, 4.125F};
  layout.csr_row_run_offsets = {0U, 2U, 3U, 4U};
  layout.csr_run_edge_offsets = {0U, 1U, 2U, 3U, 4U};
  layout.csr_run_destination_tiles = {0U, 1U, 1U, 0U};

  layout.csc_column_offsets = {0U, 1U, 2U, 4U};
  layout.csc_sources = {2U, 0U, 0U, 1U};
  layout.csc_weights = {4.125F, 1.25F, 2.5F, 3.75F};
  layout.csc_edge_ids = {3U, 0U, 1U, 2U};
  layout.csc_column_run_offsets = {0U, 1U, 2U, 3U};
  layout.csc_run_edge_offsets = {0U, 1U, 2U, 4U};
  layout.csc_run_source_tiles = {1U, 0U, 0U};
  return layout;
}

[[nodiscard]] bfnew::RouteQuery tiny_query() {
  bfnew::RouteQuery query;
  query.query_id = bfnew::QueryId{9U};
  query.sources = {bfnew::VertexId{0U}, bfnew::VertexId{1U}};
  query.targets = {bfnew::VertexId{1U}, bfnew::VertexId{2U}};
  query.selected_tiles = {bfnew::TileId{0U}, bfnew::TileId{1U}};
  return query;
}

namespace validation_error {

inline constexpr std::uint32_t shape = 1U << 0U;
inline constexpr std::uint32_t bucket_bounds = 1U << 1U;
inline constexpr std::uint32_t run_bounds = 1U << 2U;
inline constexpr std::uint32_t tile_ownership = 1U << 3U;
inline constexpr std::uint32_t nonmaximal = 1U << 4U;
inline constexpr std::uint32_t admission = 1U << 5U;

}  // namespace validation_error

// Test-only structural validation: this is deliberately not an SSSP kernel.
// A single thread walks the tiny fixture so that later HIP execution can prove
// the raw device views agree with endpoint-by-endpoint admission.
__global__ void validate_run_metadata(
    const bfnew::hip::DeviceGraphView32 graph,
    const bfnew::hip::DeviceWorkspaceView workspace,
    const std::uint32_t use_csc,
    std::uint32_t* const error_bits) {
  std::uint32_t errors = 0U;
  const std::uint32_t expected_runs =
      use_csc != 0U ? graph.csc.run_count : graph.csr.run_count;
  if (workspace.tile_lane_mask_count < graph.tile_count ||
      workspace.run_lane_mask_count != expected_runs ||
      workspace.tile_lane_masks == nullptr || workspace.run_lane_masks == nullptr) {
    *error_bits = validation_error::shape;
    return;
  }

  for (std::uint32_t vertex = 0U; vertex < graph.vertex_count; ++vertex) {
    const std::uint32_t local_tile = graph.owner_tiles[vertex];
    if (local_tile >= graph.tile_count) {
      errors |= validation_error::tile_ownership;
      continue;
    }

    const std::uint32_t bucket_edge_begin =
        use_csc != 0U ? graph.csc.column_offsets[vertex]
                      : graph.csr.row_offsets[vertex];
    const std::uint32_t bucket_edge_end =
        use_csc != 0U ? graph.csc.column_offsets[vertex + 1U]
                      : graph.csr.row_offsets[vertex + 1U];
    const std::uint32_t bucket_run_begin =
        use_csc != 0U ? graph.csc.column_run_offsets[vertex]
                      : graph.csr.row_run_offsets[vertex];
    const std::uint32_t bucket_run_end =
        use_csc != 0U ? graph.csc.column_run_offsets[vertex + 1U]
                      : graph.csr.row_run_offsets[vertex + 1U];
    if (bucket_edge_begin > bucket_edge_end || bucket_edge_end > graph.edge_count ||
        bucket_run_begin > bucket_run_end || bucket_run_end > expected_runs) {
      errors |= validation_error::bucket_bounds;
      continue;
    }

    std::uint32_t cursor = bucket_edge_begin;
    std::uint32_t preceding_tile = 0U;
    bool has_preceding_tile = false;
    for (std::uint32_t run = bucket_run_begin; run < bucket_run_end; ++run) {
      const std::uint32_t edge_begin =
          use_csc != 0U ? graph.csc.run_edge_offsets[run]
                        : graph.csr.run_edge_offsets[run];
      const std::uint32_t edge_end =
          use_csc != 0U ? graph.csc.run_edge_offsets[run + 1U]
                        : graph.csr.run_edge_offsets[run + 1U];
      const std::uint32_t remote_tile =
          use_csc != 0U ? graph.csc.run_source_tiles[run]
                        : graph.csr.run_destination_tiles[run];
      if (edge_begin != cursor || edge_end <= edge_begin ||
          edge_end > bucket_edge_end) {
        errors |= validation_error::run_bounds;
        continue;
      }
      if (remote_tile >= graph.tile_count) {
        errors |= validation_error::tile_ownership;
        continue;
      }
      if (has_preceding_tile && remote_tile == preceding_tile) {
        errors |= validation_error::nonmaximal;
      }

      const bfnew::LaneMask expected_mask =
          workspace.tile_lane_masks[local_tile] &
          workspace.tile_lane_masks[remote_tile];
      if (workspace.run_lane_masks[run] != expected_mask) {
        errors |= validation_error::admission;
      }
      for (std::uint32_t edge = edge_begin; edge < edge_end; ++edge) {
        const std::uint32_t endpoint =
            use_csc != 0U ? graph.csc.sources[edge]
                          : graph.csr.destinations[edge];
        if (endpoint >= graph.vertex_count ||
            graph.owner_tiles[endpoint] != remote_tile) {
          errors |= validation_error::tile_ownership;
        }
      }
      cursor = edge_end;
      preceding_tile = remote_tile;
      has_preceding_tile = true;
    }
    if (cursor != bucket_edge_end) {
      errors |= validation_error::run_bounds;
    }
  }
  *error_bits = errors;
}

void check_hip(const hipError_t status, const char* expression) {
  bfnew::hip::throw_if_hip_error(
      static_cast<std::int32_t>(status), expression);
}

void validate_runs_on_device(
    const bfnew::hip::DeviceGraphView32 graph,
    const bfnew::hip::DeviceWorkspaceView workspace,
    const bool use_csc,
    bfnew::hip::DeviceBuffer& error_buffer,
    const bfnew::hip::HipStream& stream) {
  error_buffer.clear_async(sizeof(std::uint32_t), stream);
  hipLaunchKernelGGL(
      validate_run_metadata,
      dim3(1U),
      dim3(1U),
      0U,
      reinterpret_cast<hipStream_t>(stream.native_handle()),
      graph,
      workspace,
      use_csc ? 1U : 0U,
      static_cast<std::uint32_t*>(error_buffer.data()));
  check_hip(hipGetLastError(), "validate_run_metadata launch");
  std::uint32_t errors = ~std::uint32_t{0U};
  error_buffer.copy_to_host_async(&errors, sizeof(errors), stream);
  stream.synchronize();
  expect(errors == 0U, "device run-layout/admission validation failed");
}

void expect_query_payload_round_trip(
    const bfnew::RouteQuery& query,
    const std::vector<bfnew::LaneMask>& tile_masks,
    const std::vector<bfnew::LaneMask>& run_masks,
    const bfnew::DeviceController& expected_controller,
    const bfnew::hip::DeviceWorkspaceView view,
    const bfnew::hip::HipStream& stream) {
  std::vector<std::uint32_t> sources(view.source_count);
  std::vector<std::uint32_t> targets(view.target_count);
  std::vector<std::uint32_t> selected_tiles(view.selected_tile_count);
  std::vector<bfnew::LaneMask> downloaded_tile_masks(view.tile_lane_mask_count);
  std::vector<bfnew::LaneMask> downloaded_run_masks(view.run_lane_mask_count);
  bfnew::DeviceController downloaded_controller{};
  const hipStream_t native =
      reinterpret_cast<hipStream_t>(stream.native_handle());
  check_hip(
      hipMemcpyAsync(
          sources.data(),
          view.sources,
          sources.size() * sizeof(std::uint32_t),
          hipMemcpyDeviceToHost,
          native),
      "query sources download");
  check_hip(
      hipMemcpyAsync(
          targets.data(),
          view.targets,
          targets.size() * sizeof(std::uint32_t),
          hipMemcpyDeviceToHost,
          native),
      "query targets download");
  check_hip(
      hipMemcpyAsync(
          selected_tiles.data(),
          view.selected_tiles,
          selected_tiles.size() * sizeof(std::uint32_t),
          hipMemcpyDeviceToHost,
          native),
      "selected tiles download");
  check_hip(
      hipMemcpyAsync(
          downloaded_tile_masks.data(),
          view.tile_lane_masks,
          downloaded_tile_masks.size() * sizeof(bfnew::LaneMask),
          hipMemcpyDeviceToHost,
          native),
      "tile masks download");
  check_hip(
      hipMemcpyAsync(
          downloaded_run_masks.data(),
          view.run_lane_masks,
          downloaded_run_masks.size() * sizeof(bfnew::LaneMask),
          hipMemcpyDeviceToHost,
          native),
      "run masks download");
  check_hip(
      hipMemcpyAsync(
          &downloaded_controller,
          view.controller,
          sizeof(downloaded_controller),
          hipMemcpyDeviceToHost,
          native),
      "controller download");
  stream.synchronize();

  expect(sources.size() == query.sources.size(), "source count changed on device");
  expect(targets.size() == query.targets.size(), "target count changed on device");
  expect(
      selected_tiles.size() == query.selected_tiles.size(),
      "selected tile count changed on device");
  for (std::size_t index = 0U; index < sources.size(); ++index) {
    expect(sources[index] == query.sources[index].value(), "source ID changed");
  }
  for (std::size_t index = 0U; index < targets.size(); ++index) {
    expect(targets[index] == query.targets[index].value(), "target ID changed");
  }
  for (std::size_t index = 0U; index < selected_tiles.size(); ++index) {
    expect(
        selected_tiles[index] == query.selected_tiles[index].value(),
        "selected tile ID changed");
  }
  expect(downloaded_tile_masks == tile_masks, "tile masks changed on device");
  expect(downloaded_run_masks == run_masks, "run masks changed on device");
  expect(
      std::memcmp(
          &downloaded_controller,
          &expected_controller,
          sizeof(expected_controller)) == 0,
      "controller bytes changed on device");
}

void test_resident_graph_round_trip(const bfnew::hip::HipStream& stream) {
  const bfnew::DeviceGraphLayout32 expected = tiny_layout();
  bfnew::DeviceGraphLayout32 empty_run = expected;
  empty_run.csr_run_edge_offsets[1U] =
      empty_run.csr_run_edge_offsets[0U];
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            bfnew::hip::make_resident_graph_plan(std::move(empty_run)));
      },
      "resident plan accepted an empty CSR run");

  bfnew::DeviceGraphLayout32 wrong_owner = expected;
  wrong_owner.csc_run_source_tiles[0U] = 0U;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            bfnew::hip::make_resident_graph_plan(std::move(wrong_owner)));
      },
      "resident plan accepted a CSC run with wrong endpoint ownership");

  bfnew::DeviceGraphLayout32 duplicate_edge_id = expected;
  duplicate_edge_id.csc_edge_ids[0U] = duplicate_edge_id.csc_edge_ids[1U];
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            bfnew::hip::make_resident_graph_plan(std::move(duplicate_edge_id)));
      },
      "resident plan accepted duplicate stable CSC edge IDs");

  bfnew::hip::ResidentGraphPlan plan =
      bfnew::hip::make_resident_graph_plan(expected);
  expect(plan.memory.total_bytes > 0U, "graph plan did not report memory");

  bfnew::hip::ResidentDeviceGraph graph;
  graph.upload_once_async(std::move(plan), stream);
  const bfnew::hip::DeviceGraphView32 before = graph.view();
  expect(before.csr.weights != nullptr, "resident CSR pointer is null");
  expect(before.csc.weights != nullptr, "resident CSC pointer is null");
  expect(before.csc.edge_ids != nullptr, "resident CSC edge-ID pointer is null");

  expect_throws<std::logic_error>(
      [&] {
        graph.upload_once_async(
            bfnew::hip::make_resident_graph_plan(expected), stream);
      },
      "resident graph accepted a second upload");

  // Exercise reconstruction from the retained memory report after pageable
  // upload staging has been released.
  graph.synchronize_upload();
  expect(graph.upload_complete(), "resident graph did not report ready");
  bfnew::DeviceGraphLayout32 downloaded;
  graph.download_async(downloaded, stream);
  stream.synchronize();
  expect(
      bfnew::device_graph_layouts_deep_equal(expected, downloaded),
      "resident graph byte/bit round trip changed the layout");
  for (std::size_t edge = 0U; edge < expected.csr_weights.size(); ++edge) {
    expect(
        std::bit_cast<std::uint32_t>(expected.csr_weights[edge]) ==
            std::bit_cast<std::uint32_t>(downloaded.csr_weights[edge]),
        "CSR weight bits changed during transfer");
  }

  const bfnew::hip::DeviceGraphView32 after = graph.view();
  expect(before.owner_tiles == after.owner_tiles, "owner pointer was not stable");
  expect(before.csr.weights == after.csr.weights, "CSR pointer was not stable");
  expect(before.csc.weights == after.csc.weights, "CSC pointer was not stable");
  expect(
      before.csc.edge_ids == after.csc.edge_ids,
      "CSC stable edge-ID pointer was not stable");
}

void round_trip_additional_layout(
    const bfnew::DeviceGraphLayout32& expected,
    const bfnew::hip::HipStream& stream,
    const char* failure_message) {
  bfnew::hip::ResidentDeviceGraph graph;
  graph.upload_once_async(
      bfnew::hip::make_resident_graph_plan(expected), stream);
  bfnew::DeviceGraphLayout32 downloaded;
  graph.download_async(downloaded, stream);
  stream.synchronize();
  expect(
      bfnew::device_graph_layouts_deep_equal(expected, downloaded),
      failure_message);
}

void test_phase5_spatial_spill_and_empty_round_trips(
    const bfnew::hip::HipStream& stream) {
  const bfnew::test::SpatialReorderFixture fixture =
      bfnew::test::make_spatial_reorder_fixture();
  const bfnew::UniformGridPartitioner partitioner{fixture.config};
  const bfnew::PartitionedGraph partitioned = partitioner.partition(fixture.graph);
  const bfnew::TileRunLayout64 runs =
      bfnew::build_tile_run_layout(partitioned.graph);
  const bfnew::DeviceGraphLayout32 spatial =
      bfnew::build_device_graph_layout32(partitioned.graph, runs);
  const std::uint32_t spill_tile = partitioned.tiles.spill_tile().value();
  expect(
      std::find(spatial.owner_tiles.begin(), spatial.owner_tiles.end(), spill_tile) !=
          spatial.owner_tiles.end(),
      "Phase 5 spatial transfer fixture lost its spill-owned vertices");
  round_trip_additional_layout(
      spatial, stream, "Phase 5 spatial/spill device round trip changed bits");

  const bfnew::InputGraph empty_input{
      std::vector<bfnew::VertexMetadata>{},
      std::vector<bfnew::EdgeInputRecord>{}};
  const bfnew::PartitionedGraph empty = partitioner.partition(empty_input);
  const bfnew::TileRunLayout64 empty_runs =
      bfnew::build_tile_run_layout(empty.graph);
  const bfnew::DeviceGraphLayout32 empty_layout =
      bfnew::build_device_graph_layout32(empty.graph, empty_runs);
  expect(
      empty_layout.vertex_count == 0U && empty_layout.edge_count == 0U,
      "empty spatial layout has nonzero graph counts");
  round_trip_additional_layout(
      empty_layout, stream, "empty spatial device round trip changed its layout");
}

void test_reusable_workspace(const bfnew::hip::HipStream& stream) {
  const bfnew::RouteQuery query = tiny_query();
  bfnew::hip::ResidentDeviceGraph resident_graph;
  resident_graph.upload_once_async(
      bfnew::hip::make_resident_graph_plan(tiny_layout()), stream);
  bfnew::hip::DeviceBuffer validation_errors;
  static_cast<void>(validation_errors.reserve(sizeof(std::uint32_t)));
  const auto requirements = bfnew::estimate_workspace_memory(
      bfnew::EngineKind::jacobi_pull,
      query,
      2U,
      4U,
      3U,
      2U,
      bfnew::InstrumentationLevel::debug,
      64U);
  expect(
      requirements.combined_total_bytes ==
          requirements.total_bytes + requirements.pinned_staging_bytes,
      "workspace combined device/pinned report is inconsistent");
  const std::vector<bfnew::LaneMask> tile_masks{0x3U, 0x2U};
  const std::vector<bfnew::LaneMask> csc_run_masks{0x2U, 0x3U, 0x2U};

  bfnew::GpuRunOptions options;
  options.engine = bfnew::EngineKind::jacobi_pull;
  options.maximum_rounds = 17U;
  const bfnew::DeviceController controller =
      bfnew::initialize_device_controller(options, 0x3U);

  bfnew::hip::ReusableDeviceWorkspace workspace;
  expect(workspace.reserve(requirements), "initial workspace reserve did not grow");
  const std::uint64_t initial_allocations = workspace.allocation_events();
  expect(initial_allocations == 1U, "initial reserve event count is wrong");

  const bfnew::WorkspaceLease first = workspace.prepare_query_async(
      requirements,
      query,
      tile_masks,
      csc_run_masks,
      controller,
      stream);
  const bfnew::hip::DeviceWorkspaceView first_view = workspace.view(first);
  expect(first_view.run_lane_mask_count == 3U, "active CSC run count is wrong");
  bfnew::hip::HipStream wrong_stream;
  bfnew::DeviceRunStatus wrong_stream_status{};
  expect_throws<std::invalid_argument>(
      [&] {
        workspace.download_status_async(
            first, wrong_stream_status, wrong_stream);
      },
      "workspace accepted a result copy on the wrong stream");
  expect_query_payload_round_trip(
      query, tile_masks, csc_run_masks, controller, first_view, stream);
  validate_runs_on_device(
      resident_graph.view(), first_view, true, validation_errors, stream);

  bfnew::DeviceRunStatus cleared_status;
  cleared_status.error_bits = 0xFFFFFFFFU;
  bfnew::DeviceWorkStatistics cleared_work;
  cleared_work.edges_examined = 0xFFFFFFFFFFFFFFFFULL;
  workspace.download_status_async(first, cleared_status, stream);
  workspace.download_instrumentation_async(first, cleared_work, stream);
  stream.synchronize();
  expect(cleared_status.error_bits == 0U, "reused status was not cleared");
  expect(cleared_work.edges_examined == 0U, "instrumentation was not cleared");
  workspace.retire_query(first, stream);

  expect(!workspace.reserve(requirements), "equal workspace reserve grew again");
  expect(
      workspace.allocation_events() == initial_allocations,
      "equal reserve changed the allocation count");
  const bfnew::WorkspaceLease second = workspace.prepare_query_async(
      requirements,
      query,
      tile_masks,
      csc_run_masks,
      controller,
      stream);
  const bfnew::hip::DeviceWorkspaceView second_view = workspace.view(second);
  expect(
      first_view.sources == second_view.sources,
      "source pointer changed without capacity growth");
  expect(
      first_view.run_lane_masks == second_view.run_lane_masks,
      "shared run-mask pointer changed without capacity growth");
  expect(
      first_view.engine_scratch == second_view.engine_scratch,
      "engine scratch pointer changed without capacity growth");
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(workspace.view(first)); },
      "workspace accepted a stale generation");
  workspace.retire_query(second, stream);

  const auto frontier_requirements = bfnew::estimate_workspace_memory(
      bfnew::EngineKind::frontier_push,
      query,
      2U,
      4U,
      3U,
      2U,
      bfnew::InstrumentationLevel::debug,
      64U);
  expect(
      !workspace.reserve(frontier_requirements),
      "engine switch allocated a second large scratch buffer");
  options.engine = bfnew::EngineKind::frontier_push;
  const bfnew::DeviceController frontier_controller =
      bfnew::initialize_device_controller(options, 0x3U, query.sources.size());
  const std::vector<bfnew::LaneMask> csr_run_masks{0x3U, 0x2U, 0x2U, 0x2U};
  const bfnew::WorkspaceLease frontier = workspace.prepare_query_async(
      frontier_requirements,
      query,
      tile_masks,
      csr_run_masks,
      frontier_controller,
      stream);
  const bfnew::hip::DeviceWorkspaceView frontier_view = workspace.view(frontier);
  expect(frontier_view.run_lane_mask_count == 4U, "active CSR run count is wrong");
  expect_query_payload_round_trip(
      query,
      tile_masks,
      csr_run_masks,
      frontier_controller,
      frontier_view,
      stream);
  validate_runs_on_device(
      resident_graph.view(), frontier_view, false, validation_errors, stream);
  expect(
      frontier_view.engine_scratch == first_view.engine_scratch,
      "engine switch did not reuse the single scratch allocation");
  workspace.retire_query(frontier, stream);
}

void trace_query_reuse_only(const bfnew::hip::HipStream& stream) {
  bfnew::hip::ResidentDeviceGraph graph;
  graph.upload_once_async(
      bfnew::hip::make_resident_graph_plan(tiny_layout()), stream);
  graph.synchronize_upload();
  const auto graph_view = graph.view();

  const bfnew::RouteQuery query = tiny_query();
  const auto requirements = bfnew::estimate_workspace_memory(
      bfnew::EngineKind::jacobi_pull,
      query,
      2U,
      4U,
      3U,
      2U,
      bfnew::InstrumentationLevel::none,
      64U);
  const std::vector<bfnew::LaneMask> tile_masks{0x3U, 0x2U};
  const std::vector<bfnew::LaneMask> csc_run_masks{0x2U, 0x3U, 0x2U};
  bfnew::GpuRunOptions options;
  options.engine = bfnew::EngineKind::jacobi_pull;
  options.maximum_rounds = 17U;
  const bfnew::DeviceController controller =
      bfnew::initialize_device_controller(options, 0x3U);

  bfnew::hip::ReusableDeviceWorkspace workspace;
  expect(workspace.reserve(requirements), "trace workspace did not reserve once");
  const std::uint64_t allocations = workspace.allocation_events();
  for (std::uint32_t repetition = 0U; repetition < 8U; ++repetition) {
    const bfnew::WorkspaceLease lease = workspace.prepare_query_async(
        requirements,
        query,
        tile_masks,
        csc_run_masks,
        controller,
        stream);
    static_cast<void>(workspace.view(lease));
    workspace.retire_query(lease, stream);
    expect(
        workspace.allocation_events() == allocations,
        "trace query unexpectedly allocated retained storage");
    expect(
        graph.view().owner_tiles == graph_view.owner_tiles,
        "trace query changed the resident graph pointer");
  }
}

}  // namespace

int main(const int argc, char** const argv) {
  try {
    bfnew::hip::HipStream stream;
    if (argc == 2 && std::string_view{argv[1]} == "--trace-query-reuse") {
      trace_query_reuse_only(stream);
      return EXIT_SUCCESS;
    }
    if (argc != 1) {
      fail("unknown command-line argument");
    }
    test_resident_graph_round_trip(stream);
    test_phase5_spatial_spill_and_empty_round_trips(stream);
    test_reusable_workspace(stream);
  } catch (const std::exception& error) {
    std::cerr << "device transfer test raised: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
