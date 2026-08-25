#include "bfnew/device_layout.hpp"
#include "bfnew/hip/jacobi.hpp"
#include "bfnew/query.hpp"
#include "bfnew/spatial.hpp"
#include "bfnew/sssp.hpp"
#include "graph_fixtures.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(
    const std::string_view fixture,
    const std::string_view message) {
  std::cerr << "Jacobi test failed [" << fixture << "]: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void expect(
    const bool condition,
    const std::string_view fixture,
    const std::string_view message) {
  if (!condition) {
    fail(fixture, message);
  }
}

template <typename Exception, typename Function>
void expect_throws(
    Function&& function,
    const std::string_view fixture,
    const std::string_view message) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  fail(fixture, message);
}

[[nodiscard]] bfnew::PhysicalProvenance provenance(
    const std::uint64_t record) noexcept {
  return bfnew::test::synthetic_provenance(10'000U + record);
}

[[nodiscard]] bfnew::VertexMetadata located_vertex(
    const std::uint32_t index,
    const std::uint32_t row_width = 8U) {
  return bfnew::VertexMetadata::located(
      static_cast<std::int32_t>(index % row_width),
      static_cast<std::int32_t>(index / row_width),
      bfnew::ResourceClassId{1U});
}

[[nodiscard]] bfnew::EdgeInputRecord edge(
    const std::uint32_t source,
    const std::uint32_t destination,
    const float weight,
    const std::uint64_t record) {
  return bfnew::EdgeInputRecord{
      bfnew::VertexId{source},
      bfnew::VertexId{destination},
      weight,
      provenance(record),
  };
}

[[nodiscard]] bfnew::InputGraph input_graph(
    const std::uint32_t vertex_count,
    std::vector<bfnew::EdgeInputRecord> edges,
    const std::uint32_t row_width = 8U) {
  std::vector<bfnew::VertexMetadata> vertices;
  vertices.reserve(vertex_count);
  for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex) {
    vertices.push_back(located_vertex(vertex, row_width));
  }
  return bfnew::InputGraph{std::move(vertices), std::move(edges)};
}

[[nodiscard]] std::vector<bfnew::VertexId> remap_vertices(
    const bfnew::WeightedGraph& graph,
    const std::span<const bfnew::VertexId> original) {
  std::vector<bfnew::VertexId> remapped;
  remapped.reserve(original.size());
  for (const bfnew::VertexId vertex : original) {
    remapped.push_back(graph.old_to_new()[vertex.value()]);
  }
  return remapped;
}

void select_full_region(
    const bfnew::WeightedGraph& graph,
    bfnew::RouteQuery& query) {
  query.selected_tiles.clear();
  query.selected_tiles.reserve(graph.tile_coordinates().size());
  for (std::size_t tile = 0U; tile < graph.tile_coordinates().size(); ++tile) {
    query.selected_tiles.push_back(
        bfnew::TileId{static_cast<std::uint32_t>(tile)});
  }
}

[[nodiscard]] std::vector<float> bounded_oracle(
    const bfnew::WeightedGraph& graph,
    const bfnew::RouteQuery& query) {
  const bfnew::InducedQueryGraph induced =
      bfnew::build_induced_query_graph(graph, query);
  const bfnew::SsspResult local =
      bfnew::dijkstra_oracle(induced.graph, induced.sources);
  std::vector<float> global(
      graph.vertex_count(), std::numeric_limits<float>::infinity());
  for (std::size_t local_vertex = 0U;
       local_vertex < induced.local_to_global.size();
       ++local_vertex) {
    global[induced.local_to_global[local_vertex].value()] =
        local.distances[local_vertex];
  }
  return global;
}

[[nodiscard]] bool bitwise_equal(
    const std::span<const float> left,
    const std::span<const float> right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (std::bit_cast<std::uint32_t>(left[index]) !=
        std::bit_cast<std::uint32_t>(right[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool within_ulps(
    const std::span<const float> left,
    const std::span<const float> right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (!bfnew::nonnegative_distance_within_ulps(
            left[index], right[index], 4U)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool all_targets_reached(
    const bfnew::RouteQuery& query,
    const std::span<const float> distances) {
  return std::ranges::all_of(query.targets, [&](const bfnew::VertexId target) {
    return std::isfinite(distances[target.value()]);
  });
}

struct CaseOptions {
  std::string_view name;
  bfnew::InputGraph input;
  bfnew::SpatialOrderConfig spatial;
  std::vector<bfnew::VertexId> original_sources;
  std::vector<bfnew::VertexId> original_targets;
  bool full_region{true};
  bool exact_representable{true};
  std::uint64_t minimum_completed_rounds{};
  bool prove_cheaper_unbounded_route{};
};

[[nodiscard]] bfnew::GpuRunOptions base_options(
    const std::uint64_t maximum_rounds) {
  bfnew::GpuRunOptions options;
  options.engine = bfnew::EngineKind::jacobi_pull;
  options.block_size = 128U;
  options.maximum_rounds = maximum_rounds;
  options.instrumentation = bfnew::InstrumentationLevel::debug;
  return options;
}

void validate_run(
    const std::string_view name,
    const bfnew::RouteQuery& query,
    const std::span<const float> oracle,
    const bool exact_representable,
    const std::uint64_t minimum_completed_rounds,
    const bfnew::GpuRunOptions& options,
    const bfnew::hip::JacobiRunOutput& output) {
  expect(
      output.result.engine_kind ==
          static_cast<std::uint32_t>(bfnew::EngineKind::jacobi_pull),
      name,
      "result lost the standalone Jacobi engine identity");
  expect(
      output.result.control_mode ==
          static_cast<std::uint32_t>(options.control_mode),
      name,
      "result lost the independently selected control mode");
  expect(
      bfnew::validate_device_run_status(output.result.status) ==
          bfnew::DeviceRunStatusError::none,
      name,
      "terminal device status is invalid");
  expect(
      output.result.status.converged == 1U &&
          output.result.status.stop_reason ==
              static_cast<std::uint32_t>(bfnew::DeviceStopReason::converged),
      name,
      "tiny fixture did not converge normally");
  expect(
      output.result.status.rounds_completed >= minimum_completed_rounds,
      name,
      "target presence stopped Jacobi before global label convergence");
  expect(
      output.result.status.final_distance_slot ==
          output.result.status.rounds_completed % 2U,
      name,
      "final distance-buffer identity disagrees with controller parity");
  expect(
      output.converged_slots_bitwise_identical,
      name,
      "converged distance columns are not bitwise identical");

  const bool reached = all_targets_reached(query, oracle);
  expect(
      output.result.status.reached_target_mask ==
          (reached ? bfnew::LaneMask{1U} : bfnew::LaneMask{0U}),
      name,
      "reached-target mask disagrees with bounded Dijkstra");
  expect(
      output.result.status.bounding_box_miss_mask ==
          (reached ? bfnew::LaneMask{0U} : bfnew::LaneMask{1U}),
      name,
      "bounding-box miss is not limited to converged unreachable targets");

  const bool oracle_match = exact_representable
                                ? bitwise_equal(output.distances, oracle)
                                : within_ulps(output.distances, oracle);
  expect(oracle_match, name, "Jacobi labels disagree with bounded CPU Dijkstra");
  expect(
      output.result.work.atomic_attempts == 0U &&
          output.result.work.successful_atomic_updates == 0U,
      name,
      "Jacobi reported a forbidden distance atomic");
  expect(
      output.metrics.gpu_milliseconds >= 0.0F &&
          output.metrics.wall_milliseconds >= 0.0,
      name,
      "timing result is invalid");
  expect(
      output.result.work.host_checks ==
              output.metrics.convergence_host_checks &&
          output.result.work.controller_copies ==
              output.metrics.convergence_host_checks + 2U &&
          output.result.work.host_synchronizations ==
              output.metrics.convergence_host_checks + 1U,
      name,
      "physical controller-copy or host-synchronization accounting is wrong");

  if (options.control_mode == bfnew::ControlMode::persistent_cooperative) {
    expect(
        output.metrics.cooperative_grid_blocks != 0U &&
            output.metrics.cooperative_active_blocks_per_wgp != 0U,
        name,
        "persistent result omitted real-kernel occupancy/grid evidence");
    expect(
        output.metrics.engine_round_dispatches == 0U &&
            output.metrics.controller_advance_dispatches == 0U &&
            output.metrics.convergence_host_checks == 0U &&
            output.result.work.kernel_dispatches == 1U,
        name,
        "persistent convergence used host polling or extra query kernels");
  } else {
    expect(
        output.metrics.engine_round_dispatches ==
                output.metrics.controller_advance_dispatches &&
            output.metrics.engine_round_dispatches >=
                output.result.status.rounds_completed,
        name,
        "ordinary execution violated the round/advance pair protocol");
    if (options.control_mode == bfnew::ControlMode::per_round_host_poll) {
      expect(
          output.metrics.engine_round_dispatches ==
                  output.result.status.rounds_completed &&
              output.metrics.convergence_host_checks ==
                  output.result.status.rounds_completed,
          name,
          "per-round mode did not perform exactly one pair per check");
    } else {
      expect(
          output.metrics.engine_round_dispatches % options.rounds_per_chunk ==
                  0U &&
              output.metrics.convergence_host_checks *
                      options.rounds_per_chunk ==
                  output.metrics.engine_round_dispatches,
          name,
          "chunked mode synchronized inside a K-round chunk");
    }
  }
}

void run_case(CaseOptions definition, const std::uint32_t query_number) {
  const bfnew::UniformGridPartitioner partitioner{definition.spatial};
  bfnew::PartitionedGraph partitioned = partitioner.partition(definition.input);
  const std::vector<bfnew::VertexId> sources =
      remap_vertices(partitioned.graph, definition.original_sources);
  const std::vector<bfnew::VertexId> targets =
      remap_vertices(partitioned.graph, definition.original_targets);
  bfnew::RouteQuery query = bfnew::make_route_query(
      bfnew::QueryId{query_number}, partitioned.graph, sources, targets, 0U);
  if (definition.full_region) {
    select_full_region(partitioned.graph, query);
  }
  expect(
      bfnew::validate_route_query(partitioned.graph, query).ok(),
      definition.name,
      "test constructed an invalid route query");

  const std::vector<float> oracle = bounded_oracle(partitioned.graph, query);
  if (definition.prove_cheaper_unbounded_route) {
    const bfnew::SsspResult unbounded =
        bfnew::dijkstra_oracle(partitioned.graph, query.sources);
    const bfnew::VertexId target = query.targets.front();
    expect(
        unbounded.distances[target.value()] < oracle[target.value()],
        definition.name,
        "fixture does not actually exclude its cheaper global route");
  }

  const bfnew::TileRunLayout64 runs =
      bfnew::build_tile_run_layout(partitioned.graph);
  const bfnew::DeviceGraphLayout32 layout =
      bfnew::build_device_graph_layout32(partitioned.graph, runs);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(layout), stream);
  bfnew::hip::ReusableDeviceWorkspace workspace;
  bfnew::hip::JacobiPullEngine engine{
      partitioned.graph, runs, resident, workspace, stream};

  const std::uint64_t maximum_rounds =
      static_cast<std::uint64_t>(partitioned.graph.vertex_count()) + 4U;
  std::vector<float> control_baseline;
  const auto exercise = [&](bfnew::GpuRunOptions options) {
    const bfnew::hip::JacobiRunOutput output =
        engine.run_with_distances(query, options);
    validate_run(
        definition.name,
        query,
        oracle,
        definition.exact_representable,
        definition.minimum_completed_rounds,
        options,
        output);
    if (control_baseline.empty()) {
      control_baseline = output.distances;
    } else {
      expect(
          bitwise_equal(control_baseline, output.distances),
          definition.name,
          "independently selectable control modes changed Jacobi bits");
    }
    return output;
  };

  bfnew::GpuRunOptions per_round = base_options(maximum_rounds);
  per_round.control_mode = bfnew::ControlMode::per_round_host_poll;
  const bfnew::hip::JacobiRunOutput debug_per_round = exercise(per_round);

  if (definition.name == "Phase 5 core") {
    bfnew::GpuRunOptions none = per_round;
    none.instrumentation = bfnew::InstrumentationLevel::none;
    const bfnew::hip::JacobiRunOutput none_output = exercise(none);
    bfnew::GpuRunOptions light = per_round;
    light.instrumentation = bfnew::InstrumentationLevel::light;
    const bfnew::hip::JacobiRunOutput light_output = exercise(light);
    expect(
        none_output.result.work.edges_examined == 0U &&
            none_output.result.work.successful_decreases == 0U &&
            none_output.result.work.active_vertices == 0U &&
            none_output.result.work.active_lane_rounds == 0U &&
            none_output.result.work.mask_operations == 0U,
        definition.name,
        "None instrumentation recorded algorithm counters");
    expect(
        light_output.result.work.edges_examined != 0U &&
            light_output.result.work.successful_decreases != 0U &&
            light_output.result.work.active_vertices != 0U &&
            light_output.result.work.active_lane_rounds != 0U &&
            light_output.result.work.mask_operations == 0U,
        definition.name,
        "Light instrumentation did not isolate its basic counters");
    expect(
        debug_per_round.result.work.edges_examined ==
                light_output.result.work.edges_examined &&
            debug_per_round.result.work.successful_decreases ==
                light_output.result.work.successful_decreases &&
            debug_per_round.result.work.active_vertices ==
                light_output.result.work.active_vertices &&
            debug_per_round.result.work.active_lane_rounds ==
                light_output.result.work.active_lane_rounds &&
            debug_per_round.result.work.mask_operations != 0U,
        definition.name,
        "Debug instrumentation does not extend Light with mask counters");
  }

  for (const std::uint32_t rounds_per_chunk :
       std::array<std::uint32_t, 5U>{2U, 4U, 8U, 16U, 32U}) {
    bfnew::GpuRunOptions chunked = base_options(maximum_rounds);
    chunked.control_mode = bfnew::ControlMode::chunked_host_poll;
    chunked.rounds_per_chunk = rounds_per_chunk;
    static_cast<void>(exercise(chunked));
  }

  bfnew::GpuRunOptions persistent = base_options(maximum_rounds);
  persistent.control_mode = bfnew::ControlMode::persistent_cooperative;
  const bfnew::hip::JacobiRunOutput occupancy_run = exercise(persistent);

  // The core fixture additionally exercises multiple legal cooperative grid
  // sizes.  Legality is derived from this real kernel's measured occupancy.
  if (definition.name == "Phase 5 core") {
    expect(
        occupancy_run.metrics.cooperative_active_blocks_per_wgp >= 2U,
        definition.name,
        "real Jacobi occupancy did not expose two legal grid sizes");
    std::uint32_t one_block_per_wgp_grid = 0U;
    for (const std::uint32_t blocks_per_wgp :
         std::array<std::uint32_t, 2U>{1U, 2U}) {
      bfnew::GpuRunOptions fixed = persistent;
      fixed.grid_policy = bfnew::GridPolicy::fixed_blocks_per_wgp;
      fixed.blocks_per_wgp = blocks_per_wgp;
      const bfnew::hip::JacobiRunOutput fixed_output = exercise(fixed);
      if (blocks_per_wgp == 1U) {
        one_block_per_wgp_grid =
            fixed_output.metrics.cooperative_grid_blocks;
        expect(
            one_block_per_wgp_grid != 0U,
            definition.name,
            "one-block residency produced an empty cooperative grid");
      } else {
        expect(
            fixed_output.metrics.cooperative_grid_blocks ==
                one_block_per_wgp_grid * blocks_per_wgp,
            definition.name,
            "cooperative grid does not scale with runtime WGP count");
      }
    }
  }

  bfnew::GpuRunOptions wrong_engine = base_options(maximum_rounds);
  wrong_engine.engine = bfnew::EngineKind::dense_chaotic_push;
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(engine.run_with_distances(query, wrong_engine)); },
      definition.name,
      "Jacobi accepted another engine kind");
}

[[nodiscard]] CaseOptions core_case() {
  bfnew::test::CoreWeightedFixture fixture =
      bfnew::test::make_core_weighted_fixture();
  return CaseOptions{
      "Phase 5 core",
      std::move(fixture.graph),
      bfnew::SpatialOrderConfig{0, 0, 2U, 2U},
      std::move(fixture.sources),
      std::move(fixture.targets),
      true,
      true,
      0U,
      false,
  };
}

[[nodiscard]] CaseOptions spatial_case() {
  bfnew::test::SpatialReorderFixture fixture =
      bfnew::test::make_spatial_reorder_fixture();
  return CaseOptions{
      "Phase 5 spatial/spill",
      std::move(fixture.graph),
      fixture.config,
      {bfnew::VertexId{1U}},
      {bfnew::VertexId{0U}, bfnew::VertexId{5U}, bfnew::VertexId{9U}},
      true,
      true,
      0U,
      false,
  };
}

[[nodiscard]] CaseOptions long_chain_case(const bool target_is_source) {
  constexpr std::uint32_t vertices = 40U;
  std::vector<bfnew::EdgeInputRecord> edges;
  for (std::uint32_t vertex = 0U; vertex + 1U < vertices; ++vertex) {
    edges.push_back(edge(vertex, vertex + 1U, 0.5F, vertex));
  }
  return CaseOptions{
      target_is_source ? "long chain without target stop" : "long chain",
      input_graph(vertices, std::move(edges), 8U),
      bfnew::SpatialOrderConfig{0, 0, 4U, 1U},
      {bfnew::VertexId{0U}},
      {bfnew::VertexId{target_is_source ? 0U : vertices - 1U}},
      true,
      true,
      target_is_source ? static_cast<std::uint64_t>(vertices) : 0U,
      false,
  };
}

[[nodiscard]] CaseOptions disconnected_case() {
  std::vector<bfnew::EdgeInputRecord> edges{
      edge(0U, 1U, 1.0F, 0U),
      edge(1U, 2U, 1.0F, 1U),
      edge(3U, 4U, 1.0F, 2U),
  };
  return CaseOptions{
      "disconnected graph",
      input_graph(6U, std::move(edges)),
      bfnew::SpatialOrderConfig{0, 0, 2U, 1U},
      {bfnew::VertexId{0U}},
      {bfnew::VertexId{5U}},
  };
}

[[nodiscard]] CaseOptions zero_cycle_case() {
  std::vector<bfnew::EdgeInputRecord> edges{
      edge(0U, 1U, 0.0F, 0U),
      edge(1U, 2U, 0.0F, 1U),
      edge(2U, 0U, 0.0F, 2U),
      edge(2U, 3U, 1.0F, 3U),
      edge(3U, 4U, 1.0F, 4U),
  };
  return CaseOptions{
      "zero-weight cycle",
      input_graph(5U, std::move(edges)),
      bfnew::SpatialOrderConfig{0, 0, 2U, 1U},
      {bfnew::VertexId{0U}},
      {bfnew::VertexId{4U}},
  };
}

[[nodiscard]] CaseOptions parallel_case() {
  std::vector<bfnew::EdgeInputRecord> edges{
      edge(0U, 1U, 4.0F, 0U),
      edge(0U, 1U, 0.5F, 1U),
      edge(0U, 1U, 2.0F, 2U),
      edge(1U, 2U, 0.25F, 3U),
  };
  return CaseOptions{
      "parallel edges",
      input_graph(3U, std::move(edges)),
      bfnew::SpatialOrderConfig{0, 0, 1U, 1U},
      {bfnew::VertexId{0U}},
      {bfnew::VertexId{2U}},
  };
}

[[nodiscard]] CaseOptions multiple_source_case() {
  std::vector<bfnew::EdgeInputRecord> edges{
      edge(0U, 2U, 4.0F, 0U),
      edge(1U, 2U, 0.5F, 1U),
      edge(2U, 3U, 0.5F, 2U),
      edge(3U, 4U, 1.0F, 3U),
  };
  return CaseOptions{
      "multiple sources",
      input_graph(5U, std::move(edges)),
      bfnew::SpatialOrderConfig{0, 0, 2U, 1U},
      {bfnew::VertexId{0U}, bfnew::VertexId{1U}},
      {bfnew::VertexId{4U}},
  };
}

[[nodiscard]] CaseOptions bounded_route_case(const bool include_direct) {
  std::vector<bfnew::VertexMetadata> vertices{
      bfnew::VertexMetadata::located(0, 0, bfnew::ResourceClassId{1U}),
      bfnew::VertexMetadata::located(1, 0, bfnew::ResourceClassId{1U}),
      bfnew::VertexMetadata::located(0, 2, bfnew::ResourceClassId{1U}),
      bfnew::VertexMetadata::located(1, 2, bfnew::ResourceClassId{1U}),
  };
  std::vector<bfnew::EdgeInputRecord> edges{
      edge(0U, 2U, 0.5F, 0U),
      edge(2U, 3U, 0.5F, 1U),
      edge(3U, 1U, 0.5F, 2U),
  };
  if (include_direct) {
    edges.push_back(edge(0U, 1U, 10.0F, 3U));
  }
  return CaseOptions{
      include_direct ? "box excludes cheaper route" : "box miss",
      bfnew::InputGraph{std::move(vertices), std::move(edges)},
      bfnew::SpatialOrderConfig{0, 0, 1U, 1U},
      {bfnew::VertexId{0U}},
      {bfnew::VertexId{1U}},
      false,
      true,
      0U,
      include_direct,
  };
}

[[nodiscard]] CaseOptions random_case(const std::uint32_t seed) {
  constexpr std::uint32_t vertices = 20U;
  std::mt19937 generator{seed};
  std::uniform_int_distribution<std::uint32_t> endpoint{0U, vertices - 1U};
  std::uniform_int_distribution<std::uint32_t> hundredths{1U, 300U};
  std::vector<bfnew::EdgeInputRecord> edges;
  for (std::uint32_t vertex = 0U; vertex + 1U < vertices; ++vertex) {
    edges.push_back(edge(
        vertex,
        vertex + 1U,
        static_cast<float>(hundredths(generator)) / 100.0F,
        edges.size()));
  }
  while (edges.size() < 80U) {
    const std::uint32_t source = endpoint(generator);
    const std::uint32_t destination = endpoint(generator);
    edges.push_back(edge(
        source,
        destination,
        static_cast<float>(hundredths(generator)) / 100.0F,
        edges.size()));
  }
  return CaseOptions{
      seed == 17U ? "random seed 17"
                  : (seed == 29U ? "random seed 29" : "random seed 43"),
      input_graph(vertices, std::move(edges), 5U),
      bfnew::SpatialOrderConfig{0, 0, 2U, 1U},
      {bfnew::VertexId{0U}},
      {bfnew::VertexId{vertices - 1U}},
      true,
      false,
      0U,
      false,
  };
}

void test_resident_graph_identity_guard() {
  CaseOptions definition = core_case();
  const bfnew::UniformGridPartitioner partitioner{definition.spatial};
  bfnew::PartitionedGraph partitioned = partitioner.partition(definition.input);
  const bfnew::TileRunLayout64 runs =
      bfnew::build_tile_run_layout(partitioned.graph);
  bfnew::DeviceGraphLayout32 wrong_layout =
      bfnew::build_device_graph_layout32(partitioned.graph, runs);
  expect(
      !wrong_layout.csc_weights.empty(),
      "resident identity guard",
      "core fixture unexpectedly has no weight to perturb");
  wrong_layout.csc_weights.front() += 0.25F;

  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(std::move(wrong_layout)), stream);
  bfnew::hip::ReusableDeviceWorkspace workspace;
  expect_throws<std::invalid_argument>(
      [&] {
        const bfnew::hip::JacobiPullEngine engine{
            partitioned.graph, runs, resident, workspace, stream};
        static_cast<void>(engine);
      },
      "resident identity guard",
      "Jacobi accepted a same-shaped resident graph with different content");
}

void test_maximum_round_exhaustion() {
  CaseOptions definition = long_chain_case(false);
  const bfnew::UniformGridPartitioner partitioner{definition.spatial};
  bfnew::PartitionedGraph partitioned = partitioner.partition(definition.input);
  const std::vector<bfnew::VertexId> sources =
      remap_vertices(partitioned.graph, definition.original_sources);
  const std::vector<bfnew::VertexId> targets =
      remap_vertices(partitioned.graph, definition.original_targets);
  bfnew::RouteQuery query = bfnew::make_route_query(
      bfnew::QueryId{900U}, partitioned.graph, sources, targets, 0U);
  select_full_region(partitioned.graph, query);
  const bfnew::TileRunLayout64 runs =
      bfnew::build_tile_run_layout(partitioned.graph);
  const bfnew::DeviceGraphLayout32 layout =
      bfnew::build_device_graph_layout32(partitioned.graph, runs);
  bfnew::hip::HipStream stream;
  bfnew::hip::ResidentDeviceGraph resident;
  resident.upload_once_async(
      bfnew::hip::make_resident_graph_plan(layout), stream);
  bfnew::hip::ReusableDeviceWorkspace workspace;
  bfnew::hip::JacobiPullEngine engine{
      partitioned.graph, runs, resident, workspace, stream};
  for (const bfnew::ControlMode mode : {
           bfnew::ControlMode::persistent_cooperative,
           bfnew::ControlMode::per_round_host_poll,
           bfnew::ControlMode::chunked_host_poll,
       }) {
    bfnew::GpuRunOptions options = base_options(1U);
    options.control_mode = mode;
    options.rounds_per_chunk = 32U;
    const bfnew::hip::JacobiRunOutput output =
        engine.run_with_distances(query, options);
    expect(
        output.result.status.stop_reason ==
                static_cast<std::uint32_t>(
                    bfnew::DeviceStopReason::maximum_rounds) &&
            output.result.status.rounds_completed == 1U &&
            output.result.status.final_distance_slot == 1U,
        "maximum-round exhaustion",
        "a control mode lost maximum-round stop or executed-round parity");
    expect(
        output.result.status.bounding_box_miss_mask == 0U,
        "maximum-round exhaustion",
        "algorithm exhaustion was misreported as a bounding-box miss");
    const std::uint64_t checks =
        mode == bfnew::ControlMode::persistent_cooperative ? 0U : 1U;
    expect(
        output.result.work.controller_copies == checks + 2U &&
            output.result.work.host_synchronizations == checks + 1U,
        "maximum-round exhaustion",
        "terminal control accounting disagrees with the physical protocol");
    if (mode == bfnew::ControlMode::chunked_host_poll) {
      expect(
          output.metrics.engine_round_dispatches == 32U &&
              output.metrics.controller_advance_dispatches == 32U &&
              output.metrics.convergence_host_checks == 1U &&
              output.result.work.kernel_dispatches == 66U,
          "maximum-round exhaustion",
          "chunked maximum stop failed to retain its already-queued no-op pairs");
    }
  }
}

}  // namespace

int main() {
  std::vector<CaseOptions> cases;
  cases.push_back(core_case());
  cases.push_back(spatial_case());
  cases.push_back(long_chain_case(false));
  cases.push_back(long_chain_case(true));
  cases.push_back(disconnected_case());
  cases.push_back(zero_cycle_case());
  cases.push_back(parallel_case());
  cases.push_back(multiple_source_case());
  cases.push_back(bounded_route_case(true));
  cases.push_back(bounded_route_case(false));
  cases.push_back(random_case(17U));
  cases.push_back(random_case(29U));
  cases.push_back(random_case(43U));

  std::uint32_t query = 100U;
  for (CaseOptions& definition : cases) {
    run_case(std::move(definition), query++);
  }
  test_resident_graph_identity_guard();
  test_maximum_round_exhaustion();
  return 0;
}
