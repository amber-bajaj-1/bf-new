#pragma once

#include "bfnew/graph.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace bfnew::test {

struct CoreWeightedFixture {
  InputGraph graph;
  std::vector<VertexId> sources;
  std::vector<VertexId> targets;
  VertexId disconnected_vertex;
};

struct SpatialReorderFixture {
  InputGraph graph;
  SpatialOrderConfig config;
  VertexId locality_report_source;
};

[[nodiscard]] inline PhysicalProvenance synthetic_provenance(
    const std::uint64_t source_record) noexcept {
  return PhysicalProvenance{
      provenance_domain::synthetic,
      provenance_kind::synthetic_edge,
      source_record,
  };
}

[[nodiscard]] inline CoreWeightedFixture make_core_weighted_fixture() {
  const ResourceClassId logic_class{1U};
  const ResourceClassId routing_class{2U};

  std::vector<VertexMetadata> vertices{
      VertexMetadata::located(0, 0, logic_class),
      VertexMetadata::located(1, 0, routing_class),
      VertexMetadata::located(2, 0, logic_class),
      VertexMetadata::unlocated(routing_class),
      VertexMetadata::located(3, 1, routing_class),
      VertexMetadata::located(4, 1, logic_class),
      VertexMetadata::located(9, 9, routing_class),
  };

  std::vector<EdgeInputRecord> edges{
      {VertexId{0U}, VertexId{1U}, 0.5F, synthetic_provenance(0U)},
      {VertexId{0U}, VertexId{2U}, 2.25F, synthetic_provenance(1U)},
      {VertexId{1U}, VertexId{2U}, 0.0F, synthetic_provenance(2U)},
      {VertexId{1U}, VertexId{2U}, 0.75F, synthetic_provenance(3U)},
      {VertexId{2U}, VertexId{2U}, 0.125F, synthetic_provenance(4U)},
      {VertexId{2U}, VertexId{3U}, 1.5F, synthetic_provenance(5U)},
      {VertexId{3U}, VertexId{4U}, 2.0F, synthetic_provenance(6U)},
      {VertexId{0U}, VertexId{4U}, 7.75F, synthetic_provenance(7U)},
      {VertexId{4U}, VertexId{5U}, 0.25F, synthetic_provenance(8U)},
  };

  return CoreWeightedFixture{
      InputGraph{std::move(vertices), std::move(edges)},
      {VertexId{0U}},
      {VertexId{3U}, VertexId{5U}},
      VertexId{6U},
  };
}

[[nodiscard]] inline SpatialReorderFixture make_spatial_reorder_fixture() {
  const ResourceClassId class_one{1U};
  const ResourceClassId class_two{2U};

  std::vector<VertexMetadata> vertices{
      VertexMetadata::located(20, 1, class_two),
      VertexMetadata::located(1, 1, class_one),
      VertexMetadata::located(11, 1, class_one),
      VertexMetadata::located(2, 1, class_two),
      VertexMetadata::located(12, 1, class_one),
      VertexMetadata::located(21, 1, class_one),
      VertexMetadata::unlocated(class_two),
      VertexMetadata::located(3, 1, class_one),
      VertexMetadata::located(3, 1, class_one),
      VertexMetadata::unlocated(class_one),
  };

  std::vector<EdgeInputRecord> edges{
      {VertexId{1U}, VertexId{0U}, 0.5F, synthetic_provenance(100U)},
      {VertexId{1U}, VertexId{2U}, 0.75F, synthetic_provenance(101U)},
      {VertexId{1U}, VertexId{3U}, 1.25F, synthetic_provenance(102U)},
      {VertexId{1U}, VertexId{4U}, 1.5F, synthetic_provenance(103U)},
      {VertexId{1U}, VertexId{5U}, 2.25F, synthetic_provenance(104U)},
      {VertexId{1U}, VertexId{7U}, 0.0F, synthetic_provenance(105U)},
      {VertexId{1U}, VertexId{8U}, 0.25F, synthetic_provenance(106U)},
      {VertexId{1U}, VertexId{9U}, 3.5F, synthetic_provenance(107U)},
      {VertexId{3U}, VertexId{1U}, 0.625F, synthetic_provenance(108U)},
      {VertexId{6U}, VertexId{1U}, 4.0F, synthetic_provenance(109U)},
      {VertexId{2U}, VertexId{2U}, 0.125F, synthetic_provenance(110U)},
      {VertexId{5U}, VertexId{0U}, 1.75F, synthetic_provenance(111U)},
  };

  return SpatialReorderFixture{
      InputGraph{std::move(vertices), std::move(edges)},
      SpatialOrderConfig{0, 0, 10U, 10U},
      VertexId{1U},
  };
}

}  // namespace bfnew::test
