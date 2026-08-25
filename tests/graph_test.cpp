#include "bfnew/graph.hpp"
#include "bfnew/types.hpp"
#include "graph_fixtures.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using bfnew::EdgeInputRecord;
using bfnew::GraphValidationErrorCode;
using bfnew::InputGraph;
using bfnew::PhysicalProvenance;
using bfnew::VertexId;
using bfnew::VertexMetadata;
using bfnew::WeightedGraph;

int failures = 0;

void expect(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAILED: " << description << '\n';
    ++failures;
  }
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, const std::string_view description) {
  try {
    function();
    expect(false, description);
  } catch (const Exception&) {
  } catch (...) {
    expect(false, description);
  }
}

[[nodiscard]] EdgeInputRecord make_edge(
    const VertexId source,
    const VertexId destination,
    const float weight,
    const std::uint64_t source_record = 0U) {
  return EdgeInputRecord{
      source,
      destination,
      weight,
      bfnew::test::synthetic_provenance(source_record),
  };
}

[[nodiscard]] InputGraph make_single_edge_graph(const float weight) {
  return InputGraph{
      {VertexMetadata::located(0, 0, bfnew::ResourceClassId{1U})},
      {make_edge(VertexId{0U}, VertexId{0U}, weight)},
  };
}

void test_strong_ids_and_provenance() {
  static_assert(!std::is_convertible_v<std::uint32_t, VertexId>);
  static_assert(!std::is_same_v<bfnew::VertexId, bfnew::TileId>);
  static_assert(!std::is_same_v<bfnew::EdgeId, bfnew::QueryId>);
  static_assert(std::is_trivially_copyable_v<PhysicalProvenance>);
  static_assert(sizeof(PhysicalProvenance) == 16U);

  expect(VertexId{7U}.value() == 7U, "strong ID exposes its fixed-width value");
  expect(bfnew::checked_id<VertexId>(7U) == VertexId{7U},
         "checked ID conversion accepts an in-range value");
  expect_throws<std::out_of_range>(
      [] { static_cast<void>(bfnew::checked_id<VertexId>(-1)); },
      "checked ID conversion rejects a negative value");
  expect_throws<std::out_of_range>(
      [] {
        static_cast<void>(bfnew::checked_id<VertexId>(
            static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) +
            1U));
      },
      "checked ID conversion rejects a value wider than VertexId");

  const PhysicalProvenance first = bfnew::test::synthetic_provenance(10U);
  const PhysicalProvenance second = bfnew::test::synthetic_provenance(11U);
  expect(first < second, "physical provenance has lexicographic ordering");
}

void test_fixture_coverage_and_determinism() {
  const bfnew::test::CoreWeightedFixture first =
      bfnew::test::make_core_weighted_fixture();
  const bfnew::test::CoreWeightedFixture second =
      bfnew::test::make_core_weighted_fixture();

  expect(first.graph.vertex_count() == 7U, "fixture has seven vertices");
  expect(first.graph.edge_count() == 9U, "fixture has nine directed edges");
  expect(first.sources == std::vector{VertexId{0U}}, "fixture has a deterministic source");
  expect(first.targets == std::vector{VertexId{3U}, VertexId{5U}},
         "fixture exposes multiple targets");
  expect(first.disconnected_vertex == VertexId{6U},
         "fixture identifies its disconnected vertex");

  expect(std::ranges::equal(first.graph.vertices(), second.graph.vertices()),
         "fixture vertex metadata is deterministic");
  expect(std::ranges::equal(first.graph.edges(), second.graph.edges()),
         "fixture edge records are deterministic");

  const auto vertices = first.graph.vertices();
  expect(!vertices[3U].has_location, "fixture has an explicitly unlocated vertex");
  expect(vertices[0U].resource_class != vertices[1U].resource_class,
         "fixture has at least two resource classes");

  std::size_t zero_weights = 0U;
  std::size_t parallel_edges = 0U;
  std::size_t self_loops = 0U;
  bool disconnected_has_incident_edge = false;
  for (const EdgeInputRecord& edge : first.graph.edges()) {
    zero_weights += edge.weight == 0.0F ? 1U : 0U;
    parallel_edges +=
        edge.source == VertexId{1U} && edge.destination == VertexId{2U} ? 1U : 0U;
    self_loops += edge.source == edge.destination ? 1U : 0U;
    disconnected_has_incident_edge =
        disconnected_has_incident_edge || edge.source == first.disconnected_vertex ||
        edge.destination == first.disconnected_vertex;
  }

  expect(zero_weights == 1U, "fixture has one zero-weight edge");
  expect(parallel_edges == 2U, "fixture preserves parallel directed edges");
  expect(self_loops == 1U, "fixture has one self-loop");
  expect(!disconnected_has_incident_edge, "fixture vertex is truly disconnected");
  expect(first.graph.edges()[0U].weight != first.graph.edges()[1U].weight,
         "fixture has unequal fractional weights");
}

void test_weight_normalization_and_validation() {
  const InputGraph signed_zero = make_single_edge_graph(-0.0F);
  const float stored_zero = signed_zero.edges().front().weight;
  expect(stored_zero == 0.0F, "signed zero remains a zero-cost edge");
  expect(!std::signbit(stored_zero), "signed zero is normalized to positive zero");
  expect(bfnew::canonical_weight_bits(-0.0F) == std::bit_cast<std::uint32_t>(0.0F),
         "canonical weight bits normalize signed zero");

  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(make_single_edge_graph(-0.25F)); },
      "negative weight is rejected");
  expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(make_single_edge_graph(
            std::numeric_limits<float>::quiet_NaN()));
      },
      "NaN weight is rejected");
  expect_throws<std::invalid_argument>(
      [] {
        static_cast<void>(make_single_edge_graph(std::numeric_limits<float>::infinity()));
      },
      "infinite weight is rejected");
}

void test_vertex_range_validation() {
  const std::vector<VertexMetadata> vertices{
      VertexMetadata::located(0, 0, bfnew::ResourceClassId{1U}),
      VertexMetadata::located(1, 0, bfnew::ResourceClassId{1U}),
  };

  expect_throws<std::out_of_range>(
      [&vertices] {
        static_cast<void>(InputGraph{
            vertices,
            {make_edge(VertexId{2U}, VertexId{0U}, 0.5F)},
        });
      },
      "out-of-range source ID is rejected");
  expect_throws<std::out_of_range>(
      [&vertices] {
        static_cast<void>(InputGraph{
            vertices,
            {make_edge(VertexId{0U}, VertexId{2U}, 0.5F)},
        });
      },
      "out-of-range destination ID is rejected");

  const EdgeInputRecord invalid_destination =
      make_edge(VertexId{0U}, VertexId{2U}, 0.5F);
  const auto validation =
      bfnew::validate_graph_input(vertices.size(), std::span{&invalid_destination, 1U});
  expect(validation.code == GraphValidationErrorCode::destination_out_of_range,
         "validator reports the precise endpoint failure");
  expect(validation.edge_index == 0U, "validator reports the failing edge index");

  if constexpr (std::numeric_limits<std::size_t>::max() >
                std::numeric_limits<std::uint32_t>::max()) {
    const auto oversized_vertex_count =
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1U;
    const auto count_validation =
        bfnew::validate_graph_input(oversized_vertex_count, {});
    expect(count_validation.code == GraphValidationErrorCode::vertex_count_overflow,
           "validator rejects a vertex count beyond the 32-bit representation");
  }
}

void expect_same_dual_layout(
    const WeightedGraph& left,
    const WeightedGraph& right,
    const std::string_view description) {
  const bfnew::OutgoingCsrView left_outgoing = left.outgoing();
  const bfnew::OutgoingCsrView right_outgoing = right.outgoing();
  const bfnew::IncomingCscView left_incoming = left.incoming();
  const bfnew::IncomingCscView right_incoming = right.incoming();

  const bool equal =
      left.vertex_count() == right.vertex_count() &&
      left.edge_count() == right.edge_count() &&
      std::ranges::equal(left.vertices(), right.vertices()) &&
      std::ranges::equal(left.edge_provenance(), right.edge_provenance()) &&
      std::ranges::equal(left_outgoing.row_offsets, right_outgoing.row_offsets) &&
      std::ranges::equal(left_outgoing.destinations, right_outgoing.destinations) &&
      std::ranges::equal(left_outgoing.weights, right_outgoing.weights) &&
      std::ranges::equal(left_outgoing.edge_ids, right_outgoing.edge_ids) &&
      std::ranges::equal(left_incoming.column_offsets, right_incoming.column_offsets) &&
      std::ranges::equal(left_incoming.sources, right_incoming.sources) &&
      std::ranges::equal(left_incoming.weights, right_incoming.weights) &&
      std::ranges::equal(left_incoming.edge_ids, right_incoming.edge_ids);
  expect(equal, description);
}

void test_dual_layout_and_transpose_invariants() {
  const bfnew::test::CoreWeightedFixture fixture =
      bfnew::test::make_core_weighted_fixture();
  const WeightedGraph graph = bfnew::build_weighted_graph(fixture.graph);
  const auto validation = bfnew::validate_weighted_graph(graph);
  expect(validation.ok(), "deep dual-graph validation succeeds");

  const bfnew::OutgoingCsrView outgoing = graph.outgoing();
  const bfnew::IncomingCscView incoming = graph.incoming();
  expect(outgoing.row_offsets.size() == 8U, "CSR has one offset per vertex plus one");
  expect(incoming.column_offsets.size() == 8U,
         "CSC has one offset per vertex plus one");
  expect(outgoing.row_offsets.front() == 0U &&
             outgoing.row_offsets.back() == graph.edge_count(),
         "CSR offsets cover every edge");
  expect(incoming.column_offsets.front() == 0U &&
             incoming.column_offsets.back() == graph.edge_count(),
         "CSC offsets cover every edge");

  const std::size_t disconnected = fixture.disconnected_vertex.value();
  expect(outgoing.row_offsets[disconnected] == outgoing.row_offsets[disconnected + 1U],
         "CSR preserves an empty outgoing row");
  expect(incoming.column_offsets[disconnected] ==
             incoming.column_offsets[disconnected + 1U],
         "CSC preserves an empty incoming column");

  std::vector<std::uint32_t> csr_weight_bits(graph.edge_count());
  std::vector<std::uint32_t> csc_weight_bits(graph.edge_count());
  std::vector<bool> csr_seen(graph.edge_count(), false);
  std::vector<bool> csc_seen(graph.edge_count(), false);
  bool csc_id_differs_from_position = false;
  for (std::size_t position = 0U; position < outgoing.edge_ids.size(); ++position) {
    const std::size_t logical_id =
        static_cast<std::size_t>(outgoing.edge_ids[position].value());
    csr_seen[logical_id] = true;
    csr_weight_bits[logical_id] = std::bit_cast<std::uint32_t>(outgoing.weights[position]);
  }
  for (std::size_t position = 0U; position < incoming.edge_ids.size(); ++position) {
    const std::size_t logical_id =
        static_cast<std::size_t>(incoming.edge_ids[position].value());
    csc_seen[logical_id] = true;
    csc_weight_bits[logical_id] = std::bit_cast<std::uint32_t>(incoming.weights[position]);
    csc_id_differs_from_position =
        csc_id_differs_from_position || logical_id != position;
  }
  expect(std::ranges::all_of(csr_seen, [](const bool value) { return value; }),
         "CSR contains every logical edge ID exactly once");
  expect(std::ranges::all_of(csc_seen, [](const bool value) { return value; }),
         "CSC contains every logical edge ID exactly once");
  expect(csr_weight_bits == csc_weight_bits,
         "CSR and CSC duplicate identical float weight bits by logical edge ID");
  expect(csc_id_differs_from_position,
         "logical edge IDs are independent of CSC array positions");

  const std::size_t row_one_begin = static_cast<std::size_t>(outgoing.row_offsets[1U]);
  const std::size_t row_one_end = static_cast<std::size_t>(outgoing.row_offsets[2U]);
  std::vector<bfnew::EdgeId> parallel_ids;
  for (std::size_t position = row_one_begin; position < row_one_end; ++position) {
    if (outgoing.destinations[position] == VertexId{2U}) {
      parallel_ids.push_back(outgoing.edge_ids[position]);
    }
  }
  expect(parallel_ids.size() == 2U && parallel_ids[0U] != parallel_ids[1U],
         "parallel edges retain distinct stable logical IDs");

  bool csr_has_self_loop = false;
  bool csc_has_same_self_loop = false;
  bfnew::EdgeId self_loop_id{};
  const std::size_t row_two_begin = static_cast<std::size_t>(outgoing.row_offsets[2U]);
  const std::size_t row_two_end = static_cast<std::size_t>(outgoing.row_offsets[3U]);
  for (std::size_t position = row_two_begin; position < row_two_end; ++position) {
    if (outgoing.destinations[position] == VertexId{2U}) {
      csr_has_self_loop = true;
      self_loop_id = outgoing.edge_ids[position];
    }
  }
  const std::size_t column_two_begin =
      static_cast<std::size_t>(incoming.column_offsets[2U]);
  const std::size_t column_two_end =
      static_cast<std::size_t>(incoming.column_offsets[3U]);
  for (std::size_t position = column_two_begin; position < column_two_end; ++position) {
    if (incoming.sources[position] == VertexId{2U} &&
        incoming.edge_ids[position] == self_loop_id) {
      csc_has_same_self_loop = true;
    }
  }
  expect(csr_has_self_loop && csc_has_same_self_loop,
         "self-loop has the same logical ID in CSR and CSC");
}

void test_canonical_ids_and_deterministic_rebuild() {
  const bfnew::test::CoreWeightedFixture fixture =
      bfnew::test::make_core_weighted_fixture();
  const WeightedGraph original = bfnew::build_weighted_graph(fixture.graph);

  std::vector<VertexMetadata> vertices(
      fixture.graph.vertices().begin(), fixture.graph.vertices().end());
  std::vector<EdgeInputRecord> reversed_edges(
      fixture.graph.edges().begin(), fixture.graph.edges().end());
  std::reverse(reversed_edges.begin(), reversed_edges.end());
  const InputGraph reversed_input{std::move(vertices), std::move(reversed_edges)};
  const WeightedGraph rebuilt = bfnew::build_weighted_graph(reversed_input);
  expect_same_dual_layout(
      original,
      rebuilt,
      "canonical IDs and both layouts are independent of input-record order");

  const std::vector<VertexMetadata> duplicate_vertices{
      VertexMetadata::located(0, 0, bfnew::ResourceClassId{1U}),
      VertexMetadata::located(1, 0, bfnew::ResourceClassId{1U}),
  };
  const EdgeInputRecord duplicate = make_edge(VertexId{0U}, VertexId{1U}, 0.5F, 100U);
  const WeightedGraph exact_parallel = bfnew::build_weighted_graph(
      InputGraph{duplicate_vertices, {duplicate, duplicate}});
  const bfnew::OutgoingCsrView duplicate_outgoing = exact_parallel.outgoing();
  expect(exact_parallel.edge_count() == 2U,
         "otherwise-identical parallel records are both preserved");
  expect(duplicate_outgoing.edge_ids[0U] == bfnew::EdgeId{0U} &&
             duplicate_outgoing.edge_ids[1U] == bfnew::EdgeId{1U},
         "implicit parallel ranks produce distinct consecutive logical IDs");
  expect(bfnew::validate_weighted_graph(exact_parallel).ok(),
         "deep validation accepts otherwise-identical parallel records");
}

void test_empty_graph_and_wide_offsets() {
  const WeightedGraph empty = bfnew::build_weighted_graph(InputGraph{{}, {}});
  expect(empty.vertex_count() == 0U && empty.edge_count() == 0U,
         "empty input graph builds an empty dual graph");
  expect(empty.outgoing().row_offsets.size() == 1U &&
             empty.outgoing().row_offsets.front() == 0U,
         "empty CSR offset array is valid");
  expect(empty.incoming().column_offsets.size() == 1U &&
             empty.incoming().column_offsets.front() == 0U,
         "empty CSC offset array is valid");
  expect(bfnew::validate_weighted_graph(empty).ok(),
         "deep validation accepts an empty dual graph");

  constexpr bfnew::EdgeCount beyond_32_bit =
      static_cast<bfnew::EdgeCount>(std::numeric_limits<std::uint32_t>::max()) + 1U;
  constexpr std::array<bfnew::EdgeOffset, 2U> simulated_offsets{0U, beyond_32_bit};
  static_assert(beyond_32_bit > std::numeric_limits<std::uint32_t>::max());
  expect(bfnew::are_valid_offsets(simulated_offsets, 1U, beyond_32_bit),
         "64-bit offset validation accepts a simulated edge count above 32 bits");

  constexpr std::array<bfnew::EdgeOffset, 3U> nonmonotonic_offsets{0U, 5U, 4U};
  expect(!bfnew::are_valid_offsets(nonmonotonic_offsets, 2U, 4U),
         "offset validation rejects a nonmonotonic sequence");
}

}  // namespace

int main() {
  test_strong_ids_and_provenance();
  test_fixture_coverage_and_determinism();
  test_weight_normalization_and_validation();
  test_vertex_range_validation();
  test_dual_layout_and_transpose_invariants();
  test_canonical_ids_and_deterministic_rebuild();
  test_empty_graph_and_wide_offsets();

  if (failures != 0) {
    std::cerr << failures << " graph/type test assertion(s) failed\n";
    return 1;
  }
  return 0;
}
