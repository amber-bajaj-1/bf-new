#include "bfnew/fpga_interchange.hpp"

#include "DeviceResources.capnp.h"
#include "PhysicalNetlist.capnp.h"

#include <capnp/serialize.h>
#include <kj/compat/gzip.h>
#include <kj/io.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace bfnew {
namespace {

using DeviceReader = DeviceResources::Device::Reader;
using PhysReader = PhysicalNetlist::PhysNetlist::Reader;

constexpr std::uint32_t unmapped = std::numeric_limits<std::uint32_t>::max();

[[nodiscard]] std::runtime_error schema_error(const std::string& message) {
  return std::runtime_error{"FPGA Interchange schema error: " + message};
}

[[nodiscard]] std::uint64_t tile_wire_key(
    const std::uint32_t tile_name,
    const std::uint32_t wire_name) noexcept {
  return (static_cast<std::uint64_t>(tile_name) << 32U) |
         static_cast<std::uint64_t>(wire_name);
}

[[nodiscard]] std::string copy_text(const capnp::Text::Reader text) {
  return std::string{text.begin(), text.end()};
}

[[nodiscard]] std::string indexed_string(
    const capnp::List<capnp::Text>::Reader strings,
    const std::uint32_t index,
    const char* field_name) {
  if (index >= strings.size()) {
    throw schema_error(std::string{field_name} + " string index is out of range");
  }
  return copy_text(strings[index]);
}

template <typename ValuesReader>
[[nodiscard]] std::optional<float> corner_values(const ValuesReader values) {
  if (values.getTyp().isTyp()) {
    return values.getTyp().getTyp();
  }
  if (values.getMax().isMax()) {
    return values.getMax().getMax();
  }
  if (values.getMin().isMin()) {
    return values.getMin().getMin();
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<float> pip_delay(
    const DeviceResources::Device::PIPTiming::Reader timing) {
  const auto delay = timing.getInternalDelay();
  if (delay.getSlow().isSlow()) {
    const auto value = corner_values(delay.getSlow().getSlow());
    if (value && is_valid_weight(*value)) {
      return canonicalize_weight(*value);
    }
  }
  if (delay.getFast().isFast()) {
    const auto value = corner_values(delay.getFast().getFast());
    if (value && is_valid_weight(*value)) {
      return canonicalize_weight(*value);
    }
  }
  return std::nullopt;
}

[[nodiscard]] float fallback_pip_weight(
    const VertexMetadata& source,
    const VertexMetadata& destination,
    const bool pseudo_pip) noexcept {
  std::uint32_t feature_units = 0U;
  if (source.has_location && destination.has_location) {
    const std::int64_t delta_x =
        std::abs(static_cast<std::int64_t>(source.x) - destination.x);
    const std::int64_t delta_y =
        std::abs(static_cast<std::int64_t>(source.y) - destination.y);
    const std::int64_t span = std::min<std::int64_t>(60, delta_x + delta_y);
    feature_units += static_cast<std::uint32_t>(span);
  }
  if (source.resource_class != destination.resource_class) {
    ++feature_units;
  }
  if (pseudo_pip) {
    feature_units += 2U;
  }
  return 1.0F + static_cast<float>(feature_units) / 64.0F;
}

class GzipMessage {
 public:
  explicit GzipMessage(const std::filesystem::path& path)
      : temporary_path_(make_temporary_path()) {
    try {
      decompress(path);
      map_message();
      std::filesystem::remove(temporary_path_);
      temporary_path_.clear();
    } catch (...) {
      if (!temporary_path_.empty()) {
        std::filesystem::remove(temporary_path_);
      }
      throw;
    }
  }

  ~GzipMessage() noexcept {
    reader_.reset();
    if (mapping_ != MAP_FAILED) {
      (void)::munmap(mapping_, byte_size_);
    }
    if (!temporary_path_.empty()) {
      std::error_code ignored;
      std::filesystem::remove(temporary_path_, ignored);
    }
  }

  GzipMessage(const GzipMessage&) = delete;
  GzipMessage& operator=(const GzipMessage&) = delete;

  template <typename Root>
  [[nodiscard]] typename Root::Reader root() {
    return reader_->getRoot<Root>();
  }

 private:
  [[nodiscard]] static kj::AutoCloseFd open_file(
      const std::filesystem::path& path,
      const int flags) {
    const int descriptor = ::open(path.c_str(), flags, 0600);
    if (descriptor < 0) {
      throw std::runtime_error{"cannot open FPGA Interchange file: " + path.string()};
    }
    return kj::AutoCloseFd{descriptor};
  }

  [[nodiscard]] static std::filesystem::path make_temporary_path() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "bfnew-fpgaif-XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int descriptor = ::mkstemp(writable.data());
    if (descriptor < 0) {
      throw std::runtime_error{"cannot create FPGA Interchange decompression cache"};
    }
    (void)::close(descriptor);
    return std::filesystem::path{writable.data()};
  }

  void decompress(const std::filesystem::path& source_path) {
    kj::FdInputStream source(open_file(source_path, O_RDONLY));
    kj::GzipInputStream gzip(source);
    kj::FdOutputStream destination(
        open_file(temporary_path_, O_WRONLY | O_TRUNC));
    std::array<kj::byte, 1U << 20U> buffer{};
    while (true) {
      const std::size_t count = gzip.tryRead(buffer.data(), 1U, buffer.size());
      if (count == 0U) {
        break;
      }
      destination.write(buffer.data(), count);
    }
  }

  void map_message() {
    byte_size_ = static_cast<std::size_t>(std::filesystem::file_size(temporary_path_));
    if (byte_size_ == 0U || byte_size_ % sizeof(capnp::word) != 0U) {
      throw schema_error("decompressed message is empty or not word-aligned");
    }
    const int descriptor = ::open(temporary_path_.c_str(), O_RDONLY);
    if (descriptor < 0) {
      throw std::runtime_error{"cannot reopen FPGA Interchange decompression cache"};
    }
    mapping_ = ::mmap(nullptr, byte_size_, PROT_READ, MAP_PRIVATE, descriptor, 0);
    (void)::close(descriptor);
    if (mapping_ == MAP_FAILED) {
      throw std::runtime_error{"cannot memory-map FPGA Interchange message"};
    }
    const auto words = kj::ArrayPtr<const capnp::word>{
        static_cast<const capnp::word*>(mapping_), byte_size_ / sizeof(capnp::word)};
    reader_ =
        std::make_unique<capnp::FlatArrayMessageReader>(words, reader_options());
  }

  [[nodiscard]] static capnp::ReaderOptions reader_options() {
    capnp::ReaderOptions options;
    options.traversalLimitInWords = std::numeric_limits<std::uint64_t>::max();
    options.nestingLimit = 256U;
    return options;
  }

  std::filesystem::path temporary_path_;
  void* mapping_{MAP_FAILED};
  std::size_t byte_size_{};
  std::unique_ptr<capnp::FlatArrayMessageReader> reader_;
};

struct TileSummary {
  std::uint32_t type{};
  std::uint16_t row{};
  std::uint16_t col{};
};

[[nodiscard]] bool site_pin_less(
    const SitePinMapping& left,
    const SitePinMapping& right) noexcept {
  return std::tie(left.site, left.site_type, left.pin) <
         std::tie(right.site, right.site_type, right.pin);
}

[[nodiscard]] bool same_site_pin_key(
    const SitePinMapping& left,
    const SitePinMapping& right) noexcept {
  return std::tie(left.site, left.site_type, left.pin) ==
         std::tie(right.site, right.site_type, right.pin);
}

[[nodiscard]] bool add_site_pin_mapping(
    std::vector<SitePinMapping>& mappings,
    const capnp::List<capnp::Text>::Reader strings,
    const std::string& site_name,
    const std::uint32_t site_type_index,
    const std::uint32_t pin_index,
    const std::uint32_t primary_pin_index,
    const capnp::List<DeviceResources::Device::SiteType>::Reader site_types,
    const DeviceResources::Device::SiteTypeInTileType::Reader tile_site_type,
    const std::unordered_map<std::uint64_t, std::uint32_t>& wire_index_by_key,
    const std::vector<std::uint32_t>& wire_to_node,
    const std::uint32_t tile_name_index) {
  if (site_type_index >= site_types.size()) {
    throw schema_error("site type index is out of range");
  }
  const auto pins = site_types[site_type_index].getPins();
  const auto primary_wires = tile_site_type.getPrimaryPinsToTileWires();
  if (pin_index >= pins.size() || primary_pin_index >= primary_wires.size()) {
    throw schema_error("site pin mapping index is out of range");
  }
  const std::uint32_t wire_name_index = primary_wires[primary_pin_index];
  const auto wire_position =
      wire_index_by_key.find(tile_wire_key(tile_name_index, wire_name_index));
  if (wire_position == wire_index_by_key.end()) {
    return false;
  }
  const std::uint32_t node = wire_to_node[wire_position->second];
  if (node == unmapped) {
    return false;
  }
  mappings.push_back(SitePinMapping{
      site_name,
      indexed_string(strings, site_types[site_type_index].getName(), "site type"),
      indexed_string(strings, pins[pin_index].getName(), "site pin"),
      VertexId{node},
  });
  return true;
}

[[nodiscard]] std::vector<SitePinMapping> build_site_pin_mappings(
    const DeviceReader device,
    const std::unordered_map<std::uint64_t, std::uint32_t>& wire_index_by_key,
    const std::vector<std::uint32_t>& wire_to_node,
    std::uint64_t& skipped_unmapped) {
  const auto strings = device.getStrList();
  const auto tiles = device.getTileList();
  const auto tile_types = device.getTileTypeList();
  const auto site_types = device.getSiteTypeList();
  std::vector<SitePinMapping> mappings;

  for (const auto tile : tiles) {
    if (tile.getType() >= tile_types.size()) {
      throw schema_error("tile type index is out of range");
    }
    const auto tile_type = tile_types[tile.getType()];
    const auto tile_site_types = tile_type.getSiteTypes();
    const std::string tile_name = indexed_string(strings, tile.getName(), "tile");
    (void)tile_name;

    for (const auto site : tile.getSites()) {
      if (site.getType() >= tile_site_types.size()) {
        throw schema_error("tile site-type index is out of range");
      }
      const auto tile_site_type = tile_site_types[site.getType()];
      const std::uint32_t primary_type_index = tile_site_type.getPrimaryType();
      if (primary_type_index >= site_types.size()) {
        throw schema_error("primary site type index is out of range");
      }
      const std::string site_name = indexed_string(strings, site.getName(), "site");
      const auto primary_type = site_types[primary_type_index];
      const auto primary_pins = primary_type.getPins();
      for (std::uint32_t pin = 0U; pin < primary_pins.size(); ++pin) {
        if (!add_site_pin_mapping(
            mappings,
            strings,
            site_name,
            primary_type_index,
            pin,
            pin,
            site_types,
            tile_site_type,
            wire_index_by_key,
            wire_to_node,
            tile.getName())) {
          ++skipped_unmapped;
        }
      }

      const auto alternatives = primary_type.getAltSiteTypes();
      const auto alternative_maps = tile_site_type.getAltPinsToPrimaryPins();
      if (alternatives.size() != alternative_maps.size()) {
        throw schema_error("alternative site type and parent-pin map sizes differ");
      }
      for (std::uint32_t alternative = 0U; alternative < alternatives.size();
           ++alternative) {
        const std::uint32_t alternative_type_index = alternatives[alternative];
        if (alternative_type_index >= site_types.size()) {
          throw schema_error("alternative site type index is out of range");
        }
        const auto parent_pins = alternative_maps[alternative].getPins();
        const auto alternative_pins = site_types[alternative_type_index].getPins();
        if (parent_pins.size() != alternative_pins.size()) {
          throw schema_error("alternative pin and parent-pin map sizes differ");
        }
        for (std::uint32_t pin = 0U; pin < alternative_pins.size(); ++pin) {
          if (!add_site_pin_mapping(
              mappings,
              strings,
              site_name,
              alternative_type_index,
              pin,
              parent_pins[pin],
              site_types,
              tile_site_type,
              wire_index_by_key,
              wire_to_node,
              tile.getName())) {
            ++skipped_unmapped;
          }
        }
      }
    }
  }

  std::sort(mappings.begin(), mappings.end(), site_pin_less);
  for (std::size_t index = 1U; index < mappings.size(); ++index) {
    if (same_site_pin_key(mappings[index - 1U], mappings[index])) {
      if (mappings[index - 1U].original_vertex != mappings[index].original_vertex) {
        throw schema_error("duplicate site pin maps to inconsistent routing nodes");
      }
      throw schema_error("duplicate site/type/pin mapping");
    }
  }
  return mappings;
}

struct BranchShape {
  bool contains_inter_site_pip{};
  std::vector<std::pair<std::string, std::string>> site_pins;
};

void inspect_branch(
    const PhysicalNetlist::PhysNetlist::RouteBranch::Reader branch,
    const capnp::List<capnp::Text>::Reader strings,
    BranchShape& shape) {
  const auto segment = branch.getRouteSegment();
  if (segment.isPip()) {
    shape.contains_inter_site_pip = true;
  } else if (segment.isSitePin()) {
    const auto site_pin = segment.getSitePin();
    shape.site_pins.emplace_back(
        indexed_string(strings, site_pin.getSite(), "physical site pin site"),
        indexed_string(strings, site_pin.getPin(), "physical site pin name"));
  }
  for (const auto child : branch.getBranches()) {
    inspect_branch(child, strings, shape);
  }
}

[[nodiscard]] std::optional<std::pair<std::string, std::string>> branch_endpoint(
    const PhysicalNetlist::PhysNetlist::RouteBranch::Reader branch,
    const capnp::List<capnp::Text>::Reader strings,
    bool& partial_shape) {
  BranchShape shape;
  inspect_branch(branch, strings, shape);
  partial_shape = partial_shape || shape.contains_inter_site_pip;
  std::sort(shape.site_pins.begin(), shape.site_pins.end());
  shape.site_pins.erase(
      std::unique(shape.site_pins.begin(), shape.site_pins.end()),
      shape.site_pins.end());
  if (shape.site_pins.size() != 1U) {
    partial_shape = true;
    return std::nullopt;
  }
  return shape.site_pins.front();
}

[[nodiscard]] const SitePinMapping* find_site_pin_mapping(
    const std::vector<SitePinMapping>& mappings,
    const std::string& site,
    const std::string& type,
    const std::string& pin) {
  const SitePinMapping key{site, type, pin, VertexId{}};
  const auto position = std::lower_bound(mappings.begin(), mappings.end(), key, site_pin_less);
  if (position == mappings.end() || !same_site_pin_key(*position, key)) {
    return nullptr;
  }
  return &*position;
}

[[nodiscard]] std::optional<std::vector<PhysicalEndpoint>> map_branch_endpoints(
    const capnp::List<PhysicalNetlist::PhysNetlist::RouteBranch>::Reader branches,
    const capnp::List<capnp::Text>::Reader strings,
    const std::unordered_map<std::string, std::string>& active_site_types,
    const std::vector<SitePinMapping>& mappings,
    bool& partial_shape) {
  std::vector<PhysicalEndpoint> endpoints;
  endpoints.reserve(branches.size());
  for (const auto branch : branches) {
    const auto endpoint = branch_endpoint(branch, strings, partial_shape);
    if (!endpoint) {
      continue;
    }
    const auto active_type = active_site_types.find(endpoint->first);
    if (active_type == active_site_types.end()) {
      return std::nullopt;
    }
    const SitePinMapping* mapping = find_site_pin_mapping(
        mappings, endpoint->first, active_type->second, endpoint->second);
    if (mapping == nullptr) {
      return std::nullopt;
    }
    endpoints.push_back(PhysicalEndpoint{
        endpoint->first, endpoint->second, mapping->original_vertex});
  }
  return endpoints;
}

[[nodiscard]] MappedTerminal map_terminal(
    const PhysicalEndpoint& endpoint,
    const WeightedGraph& graph) {
  if (endpoint.original_vertex.value() >= graph.old_to_new().size()) {
    throw std::out_of_range{"physical endpoint node is outside the device graph"};
  }
  const VertexId vertex = graph.old_to_new()[endpoint.original_vertex.value()];
  const VertexMetadata metadata = graph.vertices()[vertex.value()];
  return MappedTerminal{
      endpoint.site,
      endpoint.pin,
      endpoint.original_vertex,
      vertex,
      metadata.x,
      metadata.y,
      metadata.has_location,
      graph.owner_tiles()[vertex.value()],
  };
}

}  // namespace

DeviceImportResult load_fpga_interchange_device(
    const std::filesystem::path& gzip_device_path) {
  GzipMessage message(gzip_device_path);
  const DeviceReader device = message.root<DeviceResources::Device>();
  const auto strings = device.getStrList();
  const auto tiles = device.getTileList();
  const auto tile_types = device.getTileTypeList();
  const auto wires = device.getWires();
  const auto nodes = device.getNodes();
  const auto timings = device.getPipTimings();

  if (!std::in_range<std::uint32_t>(nodes.size()) ||
      !std::in_range<std::uint32_t>(wires.size())) {
    throw schema_error("device node or wire count exceeds 32-bit identifiers");
  }

  std::unordered_map<std::uint32_t, TileSummary> tile_by_name;
  tile_by_name.reserve(tiles.size());
  for (const auto tile : tiles) {
    if (tile.getName() >= strings.size() || tile.getType() >= tile_types.size()) {
      throw schema_error("tile string or type index is out of range");
    }
    if (!tile_by_name.emplace(
             tile.getName(), TileSummary{tile.getType(), tile.getRow(), tile.getCol()})
             .second) {
      throw schema_error("duplicate tile name index");
    }
  }

  std::unordered_map<std::uint64_t, std::uint32_t> wire_index_by_key;
  wire_index_by_key.reserve(wires.size());
  for (std::uint32_t wire_index = 0U; wire_index < wires.size(); ++wire_index) {
    const auto wire = wires[wire_index];
    if (wire.getTile() >= strings.size() || wire.getWire() >= strings.size()) {
      throw schema_error("wire tile or name string index is out of range");
    }
    if (!wire_index_by_key
             .emplace(tile_wire_key(wire.getTile(), wire.getWire()), wire_index)
             .second) {
      throw schema_error("duplicate tile/wire record");
    }
  }

  std::vector<std::uint32_t> wire_to_node(wires.size(), unmapped);
  std::vector<VertexMetadata> vertices;
  vertices.reserve(nodes.size());
  std::uint64_t missing_coordinate_nodes = 0U;
  for (std::uint32_t node_index = 0U; node_index < nodes.size(); ++node_index) {
    const auto node_wires = nodes[node_index].getWires();
    bool has_location = false;
    std::uint16_t minimum_resource = std::numeric_limits<std::uint16_t>::max();
    std::tuple<std::uint16_t, std::uint16_t> representative{
        std::numeric_limits<std::uint16_t>::max(),
        std::numeric_limits<std::uint16_t>::max()};
    for (const std::uint32_t wire_index : node_wires) {
      if (wire_index >= wires.size()) {
        throw schema_error("node wire index is out of range");
      }
      if (wire_to_node[wire_index] != unmapped) {
        throw schema_error("wire belongs to more than one node");
      }
      wire_to_node[wire_index] = node_index;
      const auto wire = wires[wire_index];
      if (!std::in_range<std::uint16_t>(wire.getType())) {
        throw schema_error("wire type exceeds ResourceClassId representation");
      }
      minimum_resource = std::min(
          minimum_resource, static_cast<std::uint16_t>(wire.getType()));
      const auto tile = tile_by_name.find(wire.getTile());
      if (tile != tile_by_name.end()) {
        has_location = true;
        representative = std::min(
            representative, std::tuple{tile->second.row, tile->second.col});
      }
    }
    if (minimum_resource == std::numeric_limits<std::uint16_t>::max()) {
      minimum_resource = 0U;
    }
    if (has_location) {
      vertices.push_back(VertexMetadata::located(
          static_cast<std::int32_t>(std::get<1>(representative)),
          static_cast<std::int32_t>(std::get<0>(representative)),
          ResourceClassId{minimum_resource}));
    } else {
      vertices.push_back(VertexMetadata::unlocated(ResourceClassId{minimum_resource}));
      ++missing_coordinate_nodes;
    }
  }

  std::uint64_t edge_reserve = 0U;
  for (const auto tile : tiles) {
    const auto pips = tile_types[tile.getType()].getPips();
    for (const auto pip : pips) {
      edge_reserve += pip.getDirectional() ? 1U : 2U;
    }
  }
  if (!std::in_range<std::size_t>(edge_reserve)) {
    throw schema_error("directed PIP count exceeds host size representation");
  }
  std::vector<EdgeInputRecord> edges;
  edges.reserve(static_cast<std::size_t>(edge_reserve));

  DeviceImportStatistics statistics;
  statistics.tile_count = tiles.size();
  statistics.wire_count = wires.size();
  statistics.node_count = nodes.size();
  statistics.missing_coordinate_nodes = missing_coordinate_nodes;
  for (std::uint32_t tile_index = 0U; tile_index < tiles.size(); ++tile_index) {
    const auto tile = tiles[tile_index];
    const auto tile_type = tile_types[tile.getType()];
    const auto local_wires = tile_type.getWires();
    const auto pips = tile_type.getPips();
    for (std::uint32_t pip_index = 0U; pip_index < pips.size(); ++pip_index) {
      const auto pip = pips[pip_index];
      ++statistics.pip_count;
      if (!pip.getDirectional()) {
        ++statistics.bidirectional_pip_count;
      }
      if (pip.getWire0() >= local_wires.size() || pip.getWire1() >= local_wires.size()) {
        throw schema_error("PIP wire index is out of tile-type range");
      }
      const auto wire0 = wire_index_by_key.find(
          tile_wire_key(tile.getName(), local_wires[pip.getWire0()]));
      const auto wire1 = wire_index_by_key.find(
          tile_wire_key(tile.getName(), local_wires[pip.getWire1()]));
      if (wire0 == wire_index_by_key.end() || wire1 == wire_index_by_key.end()) {
        ++statistics.skipped_unmapped_pips;
        continue;
      }
      const std::uint32_t node0 = wire_to_node[wire0->second];
      const std::uint32_t node1 = wire_to_node[wire1->second];
      if (node0 == unmapped || node1 == unmapped) {
        ++statistics.skipped_unmapped_pips;
        continue;
      }

      float weight = fallback_pip_weight(
          vertices[node0], vertices[node1], pip.isPseudoCells());
      if (pip.getTiming() < timings.size()) {
        if (const auto delay = pip_delay(timings[pip.getTiming()])) {
          weight = *delay;
        } else {
          statistics.fallback_weight_edges += pip.getDirectional() ? 1U : 2U;
        }
      } else {
        statistics.fallback_weight_edges += pip.getDirectional() ? 1U : 2U;
      }
      const std::uint64_t source_record =
          (static_cast<std::uint64_t>(tile_index) << 32U) | pip_index;
      edges.push_back(EdgeInputRecord{
          VertexId{node0},
          VertexId{node1},
          weight,
          PhysicalProvenance{
              provenance_domain::fpga_interchange,
              provenance_kind::fpga_interchange_pip_forward,
              source_record},
      });
      ++statistics.directed_pip_count;
      if (!pip.getDirectional()) {
        edges.push_back(EdgeInputRecord{
            VertexId{node1},
            VertexId{node0},
            weight,
            PhysicalProvenance{
                provenance_domain::fpga_interchange,
                provenance_kind::fpga_interchange_pip_reverse,
                source_record},
        });
        ++statistics.directed_pip_count;
      }
    }
  }

  std::vector<SitePinMapping> mappings =
      build_site_pin_mappings(
          device,
          wire_index_by_key,
          wire_to_node,
          statistics.skipped_unmapped_site_pins);
  statistics.site_pin_mappings = mappings.size();
  auto graph = std::make_unique<InputGraph>(std::move(vertices), std::move(edges));
  return DeviceImportResult{std::move(graph), std::move(mappings), statistics};
}

PhysicalCorpus scan_fpga_interchange_physical_netlist(
    const std::filesystem::path& gzip_phys_path,
    const std::vector<SitePinMapping>& site_pin_mappings,
    const bool retain_queries) {
  if (!std::is_sorted(site_pin_mappings.begin(), site_pin_mappings.end(), site_pin_less)) {
    throw std::invalid_argument{"site pin mapping table must be canonically sorted"};
  }

  GzipMessage message(gzip_phys_path);
  const PhysReader physical = message.root<PhysicalNetlist::PhysNetlist>();
  const auto strings = physical.getStrList();
  std::unordered_map<std::string, std::string> active_site_types;
  active_site_types.reserve(physical.getSiteInsts().size());
  for (const auto site_instance : physical.getSiteInsts()) {
    const std::string site =
        indexed_string(strings, site_instance.getSite(), "site instance site");
    const std::string type =
        indexed_string(strings, site_instance.getType(), "site instance type");
    const auto [position, inserted] = active_site_types.emplace(site, type);
    if (!inserted && position->second != type) {
      throw schema_error("site instance has conflicting active site types");
    }
  }

  PhysicalCorpus corpus;
  corpus.part = copy_text(physical.getPart());
  const auto nets = physical.getPhysNets();
  corpus.statistics.total_nets = nets.size();
  if (retain_queries) {
    corpus.queries.reserve(nets.size());
  }

  for (const auto net : nets) {
    const std::string name = indexed_string(strings, net.getName(), "physical net");
    if (name == "GLOBAL_USEDNET") {
      ++corpus.statistics.excluded_global_usednet;
      continue;
    }
    if (net.getType() != PhysicalNetlist::PhysNetlist::NetType::SIGNAL) {
      ++corpus.statistics.excluded_static;
      continue;
    }
    if (net.getSources().size() == 0U) {
      ++corpus.statistics.excluded_driverless;
      continue;
    }
    if (net.getStubs().size() == 0U) {
      ++corpus.statistics.excluded_no_targets;
      continue;
    }
    if (net.getSources().size() > 2U || net.getStubNodes().size() != 0U) {
      if (net.getSources().size() > 2U) {
        ++corpus.statistics.excluded_source_count;
      } else {
        ++corpus.statistics.excluded_partial_shape;
      }
      continue;
    }

    bool partial_shape = false;
    const auto sources = map_branch_endpoints(
        net.getSources(),
        strings,
        active_site_types,
        site_pin_mappings,
        partial_shape);
    const auto targets = map_branch_endpoints(
        net.getStubs(),
        strings,
        active_site_types,
        site_pin_mappings,
        partial_shape);
    if (partial_shape) {
      ++corpus.statistics.excluded_partial_shape;
      continue;
    }
    if (!sources || !targets) {
      ++corpus.statistics.excluded_missing_site_mapping;
      continue;
    }
    if (sources->empty()) {
      ++corpus.statistics.excluded_driverless;
      continue;
    }
    if (targets->empty()) {
      ++corpus.statistics.excluded_no_targets;
      continue;
    }
    if (sources->size() > 2U) {
      ++corpus.statistics.excluded_source_count;
      continue;
    }

    ++corpus.statistics.accepted_queries;
    if (sources->size() == 1U) {
      ++corpus.statistics.one_source_queries;
    } else {
      ++corpus.statistics.two_source_queries;
    }
    corpus.statistics.retained_sink_terminals += targets->size();
    if (retain_queries) {
      corpus.queries.push_back(
          PhysicalQueryRecord{name, std::move(*sources), std::move(*targets)});
    }
  }

  if (retain_queries) {
    std::sort(
        corpus.queries.begin(),
        corpus.queries.end(),
        [](const PhysicalQueryRecord& left, const PhysicalQueryRecord& right) {
          return left.net_name < right.net_name;
        });
    const auto duplicate = std::adjacent_find(
        corpus.queries.begin(),
        corpus.queries.end(),
        [](const PhysicalQueryRecord& left, const PhysicalQueryRecord& right) {
          return left.net_name == right.net_name;
        });
    if (duplicate != corpus.queries.end()) {
      throw schema_error("accepted physical net names are not unique");
    }
  }
  return corpus;
}

std::vector<MappedRouteQuery> map_route_queries(
    const PhysicalCorpus& corpus,
    const WeightedGraph& spatial_graph,
    const std::uint32_t tile_padding) {
  if (!spatial_graph.has_spatial_ordering()) {
    throw std::invalid_argument{"physical queries require a spatially ordered graph"};
  }
  if (!std::in_range<std::uint32_t>(corpus.queries.size())) {
    throw std::length_error{"query corpus exceeds QueryId representation"};
  }

  std::vector<MappedRouteQuery> mapped;
  mapped.reserve(corpus.queries.size());
  for (std::size_t query_index = 0U; query_index < corpus.queries.size(); ++query_index) {
    const PhysicalQueryRecord& physical = corpus.queries[query_index];
    std::vector<MappedTerminal> sources;
    std::vector<MappedTerminal> targets;
    std::vector<VertexId> source_vertices;
    std::vector<VertexId> target_vertices;
    sources.reserve(physical.sources.size());
    targets.reserve(physical.targets.size());
    source_vertices.reserve(physical.sources.size());
    target_vertices.reserve(physical.targets.size());
    for (const PhysicalEndpoint& endpoint : physical.sources) {
      sources.push_back(map_terminal(endpoint, spatial_graph));
      source_vertices.push_back(sources.back().vertex);
    }
    for (const PhysicalEndpoint& endpoint : physical.targets) {
      targets.push_back(map_terminal(endpoint, spatial_graph));
      target_vertices.push_back(targets.back().vertex);
    }

    RouteQuery query = make_route_query(
        checked_id<QueryId>(query_index),
        spatial_graph,
        source_vertices,
        target_vertices,
        tile_padding);
    mapped.push_back(MappedRouteQuery{
        physical.net_name, std::move(query), std::move(sources), std::move(targets)});
  }
  return mapped;
}

}  // namespace bfnew
