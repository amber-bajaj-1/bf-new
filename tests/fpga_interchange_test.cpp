#include "bfnew/fpga_interchange.hpp"

#include "DeviceResources.capnp.h"
#include "PhysicalNetlist.capnp.h"

#include "bfnew/graph.hpp"
#include "bfnew/query.hpp"
#include "bfnew/spatial.hpp"
#include "bfnew/sssp.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/compat/gzip.h>
#include <kj/io.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void write_gzip_message(
    const std::filesystem::path& path,
    capnp::MessageBuilder& message) {
  const int descriptor = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
  assert(descriptor >= 0);
  kj::FdOutputStream file(kj::AutoCloseFd{descriptor});
  kj::GzipOutputStream gzip(file);
  capnp::writeMessage(gzip, message);
}

void write_device_fixture(const std::filesystem::path& path) {
  capnp::MallocMessageBuilder message;
  auto device = message.initRoot<DeviceResources::Device>();
  device.setName("fixture");
  auto strings = device.initStrList(8U);
  strings.set(0U, "TILE0");
  strings.set(1U, "WIRE0");
  strings.set(2U, "WIRE1");
  strings.set(3U, "SITE0");
  strings.set(4U, "SLICEL");
  strings.set(5U, "OUT");
  strings.set(6U, "IN");
  strings.set(7U, "TTYPE");

  auto site_types = device.initSiteTypeList(1U);
  auto site_type = site_types[0U];
  site_type.setName(4U);
  auto site_pins = site_type.initPins(2U);
  site_pins[0U].setName(5U);
  site_pins[1U].setName(6U);
  site_type.initAltSiteTypes(0U);

  auto tile_types = device.initTileTypeList(1U);
  auto tile_type = tile_types[0U];
  tile_type.setName(7U);
  auto tile_site_types = tile_type.initSiteTypes(1U);
  tile_site_types[0U].setPrimaryType(0U);
  auto pin_wires = tile_site_types[0U].initPrimaryPinsToTileWires(2U);
  pin_wires.set(0U, 1U);
  pin_wires.set(1U, 2U);
  tile_site_types[0U].initAltPinsToPrimaryPins(0U);
  auto local_wires = tile_type.initWires(2U);
  local_wires.set(0U, 1U);
  local_wires.set(1U, 2U);
  auto pips = tile_type.initPips(1U);
  pips[0U].setWire0(0U);
  pips[0U].setWire1(1U);
  pips[0U].setDirectional(false);
  pips[0U].setConventional();
  pips[0U].setTiming(0U);

  auto tiles = device.initTileList(1U);
  tiles[0U].setName(0U);
  tiles[0U].setType(0U);
  tiles[0U].setRow(5U);
  tiles[0U].setCol(7U);
  auto sites = tiles[0U].initSites(1U);
  sites[0U].setName(3U);
  sites[0U].setType(0U);

  auto wires = device.initWires(2U);
  wires[0U].setTile(0U);
  wires[0U].setWire(1U);
  wires[0U].setType(0U);
  wires[1U].setTile(0U);
  wires[1U].setWire(2U);
  wires[1U].setType(0U);
  auto nodes = device.initNodes(2U);
  nodes[0U].initWires(1U).set(0U, 0U);
  nodes[1U].initWires(1U).set(0U, 1U);
  device.initWireTypes(1U)[0U].setName(7U);
  auto timing = device.initPipTimings(1U)[0U];
  timing.initInternalDelay().initSlow().initSlow().initTyp().setTyp(0.25F);

  write_gzip_message(path, message);
}

void set_site_pin_branch(
    PhysicalNetlist::PhysNetlist::RouteBranch::Builder branch,
    const std::uint32_t pin) {
  auto site_pin = branch.initRouteSegment().initSitePin();
  site_pin.setSite(1U);
  site_pin.setPin(pin);
  branch.initBranches(0U);
}

void write_physical_fixture(const std::filesystem::path& path) {
  capnp::MallocMessageBuilder message;
  auto physical = message.initRoot<PhysicalNetlist::PhysNetlist>();
  physical.setPart("fixture-part");
  auto strings = physical.initStrList(8U);
  strings.set(0U, "net0");
  strings.set(1U, "SITE0");
  strings.set(2U, "SLICEL");
  strings.set(3U, "OUT");
  strings.set(4U, "IN");
  strings.set(5U, "GLOBAL_USEDNET");
  strings.set(6U, "static");
  strings.set(7U, "driverless");
  auto site_instances = physical.initSiteInsts(1U);
  site_instances[0U].setSite(1U);
  site_instances[0U].setType(2U);

  auto nets = physical.initPhysNets(4U);
  nets[0U].setName(0U);
  nets[0U].setType(PhysicalNetlist::PhysNetlist::NetType::SIGNAL);
  set_site_pin_branch(nets[0U].initSources(1U)[0U], 3U);
  set_site_pin_branch(nets[0U].initStubs(1U)[0U], 4U);
  nets[0U].initStubNodes(0U);

  nets[1U].setName(5U);
  nets[1U].setType(PhysicalNetlist::PhysNetlist::NetType::SIGNAL);
  set_site_pin_branch(nets[1U].initSources(1U)[0U], 3U);
  set_site_pin_branch(nets[1U].initStubs(1U)[0U], 4U);

  nets[2U].setName(6U);
  nets[2U].setType(PhysicalNetlist::PhysNetlist::NetType::GND);
  set_site_pin_branch(nets[2U].initSources(1U)[0U], 3U);
  set_site_pin_branch(nets[2U].initStubs(1U)[0U], 4U);

  nets[3U].setName(7U);
  nets[3U].setType(PhysicalNetlist::PhysNetlist::NetType::SIGNAL);
  nets[3U].initSources(0U);
  set_site_pin_branch(nets[3U].initStubs(1U)[0U], 4U);

  write_gzip_message(path, message);
}

}  // namespace

int main() {
  using namespace bfnew;

  const std::filesystem::path directory =
      std::filesystem::current_path() /
      ("bfnew_fpgaif_test_" + std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::create_directories(directory);
  const auto device_path = directory / "fixture.device";
  const auto physical_path = directory / "fixture.phys";
  write_device_fixture(device_path);
  write_physical_fixture(physical_path);

  DeviceImportResult imported = load_fpga_interchange_device(device_path);
  assert(imported.graph->vertex_count() == 2U);
  assert(imported.graph->edge_count() == 2U);
  assert(imported.statistics.pip_count == 1U);
  assert(imported.statistics.bidirectional_pip_count == 1U);
  assert(imported.statistics.fallback_weight_edges == 0U);
  assert(imported.site_pin_mappings.size() == 2U);
  assert(imported.graph->edges()[0U].weight == 0.25F);

  const PhysicalCorpus corpus = scan_fpga_interchange_physical_netlist(
      physical_path, imported.site_pin_mappings);
  assert(corpus.statistics.total_nets == 4U);
  assert(corpus.statistics.accepted_queries == 1U);
  assert(corpus.statistics.excluded_global_usednet == 1U);
  assert(corpus.statistics.excluded_static == 1U);
  assert(corpus.statistics.excluded_driverless == 1U);
  assert(corpus.queries.size() == 1U);
  assert(corpus.queries[0U].sources.size() == 1U);
  assert(corpus.queries[0U].targets.size() == 1U);

  const UniformGridPartitioner partitioner(SpatialOrderConfig{0, 0, 1U, 1U});
  const PartitionedGraph partitioned = partitioner.partition(*imported.graph);
  assert(validate_weighted_graph(partitioned.graph).ok());
  assert(validate_tile_directory(partitioned.graph, partitioned.tiles).ok());
  const std::vector<MappedRouteQuery> queries =
      map_route_queries(corpus, partitioned.graph, 0U);
  assert(queries.size() == 1U);
  assert(validate_route_query(partitioned.graph, queries[0U].query).ok());
  const InducedQueryGraph bounded =
      build_induced_query_graph(partitioned.graph, queries[0U].query);
  const SsspResult distances = dijkstra_oracle(bounded.graph, bounded.sources);
  assert(distances.distances[bounded.targets[0U].value()] == 0.25F);

  const auto graph_a = directory / "graph-a.bfgraph";
  const auto graph_b = directory / "graph-b.bfgraph";
  write_graph_artifact(graph_a, *imported.graph);
  write_graph_artifact(graph_b, *imported.graph);
  assert(files_are_byte_identical(graph_a, graph_b));
  const InputGraph graph_round_trip = read_graph_artifact(graph_a);
  assert(graph_round_trip.vertices().size() == imported.graph->vertices().size());
  assert(std::equal(
      graph_round_trip.vertices().begin(),
      graph_round_trip.vertices().end(),
      imported.graph->vertices().begin()));
  assert(graph_round_trip.edges().size() == imported.graph->edges().size());
  assert(std::equal(
      graph_round_trip.edges().begin(),
      graph_round_trip.edges().end(),
      imported.graph->edges().begin()));

  const auto query_a = directory / "query-a.bfqueries";
  const auto query_b = directory / "query-b.bfqueries";
  write_query_artifact(query_a, queries);
  write_query_artifact(query_b, queries);
  assert(files_are_byte_identical(query_a, query_b));
  const auto query_round_trip = read_query_artifact(query_a, partitioned.graph);
  assert(query_round_trip == queries);

  std::filesystem::remove_all(directory);
  return 0;
}
