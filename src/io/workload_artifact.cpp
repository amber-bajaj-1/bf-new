#include "bfnew/fpga_interchange.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace bfnew {
namespace {

constexpr std::array<char, 8U> graph_magic{'B', 'F', 'G', 'R', 'P', 'H', '0', '7'};
constexpr std::array<char, 8U> query_magic{'B', 'F', 'Q', 'R', 'Y', '0', '0', '7'};

class BinaryWriter {
 public:
  explicit BinaryWriter(const std::filesystem::path& path) : stream_(path, std::ios::binary) {
    if (!stream_) {
      throw std::runtime_error{"cannot create workload artifact: " + path.string()};
    }
  }

  void bytes(const char* data, const std::size_t size) {
    stream_.write(data, static_cast<std::streamsize>(size));
    if (!stream_) {
      throw std::runtime_error{"failed while writing workload artifact"};
    }
  }

  template <std::unsigned_integral Integer>
  void unsigned_integer(Integer value) {
    std::array<char, sizeof(Integer)> encoded{};
    std::uint64_t remaining = static_cast<std::uint64_t>(value);
    for (std::size_t byte = 0U; byte < encoded.size(); ++byte) {
      encoded[byte] = static_cast<char>(remaining & 0xffU);
      remaining >>= 8U;
    }
    bytes(encoded.data(), encoded.size());
  }

  void signed_integer(const std::int32_t value) {
    unsigned_integer(std::bit_cast<std::uint32_t>(value));
  }

  void boolean(const bool value) { unsigned_integer<std::uint8_t>(value ? 1U : 0U); }

  void string(const std::string& value) {
    if (!std::in_range<std::uint32_t>(value.size())) {
      throw std::length_error{"artifact string exceeds 32-bit length"};
    }
    unsigned_integer(static_cast<std::uint32_t>(value.size()));
    bytes(value.data(), value.size());
  }

 private:
  std::ofstream stream_;
};

class BinaryReader {
 public:
  explicit BinaryReader(const std::filesystem::path& path) : stream_(path, std::ios::binary) {
    if (!stream_) {
      throw std::runtime_error{"cannot open workload artifact: " + path.string()};
    }
  }

  void bytes(char* data, const std::size_t size) {
    stream_.read(data, static_cast<std::streamsize>(size));
    if (stream_.gcount() != static_cast<std::streamsize>(size)) {
      throw std::runtime_error{"truncated workload artifact"};
    }
  }

  template <std::unsigned_integral Integer>
  [[nodiscard]] Integer unsigned_integer() {
    std::array<unsigned char, sizeof(Integer)> encoded{};
    bytes(reinterpret_cast<char*>(encoded.data()), encoded.size());
    Integer value = 0U;
    for (std::size_t byte = 0U; byte < encoded.size(); ++byte) {
      value |= static_cast<Integer>(encoded[byte]) << (byte * 8U);
    }
    return value;
  }

  [[nodiscard]] std::int32_t signed_integer() {
    return std::bit_cast<std::int32_t>(unsigned_integer<std::uint32_t>());
  }

  [[nodiscard]] bool boolean() {
    const std::uint8_t encoded = unsigned_integer<std::uint8_t>();
    if (encoded > 1U) {
      throw std::runtime_error{"invalid artifact boolean"};
    }
    return encoded != 0U;
  }

  [[nodiscard]] std::string string() {
    const std::uint32_t size = unsigned_integer<std::uint32_t>();
    std::string value(size, '\0');
    bytes(value.data(), value.size());
    return value;
  }

  void require_eof() {
    char trailing{};
    stream_.read(&trailing, 1);
    if (stream_.gcount() != 0) {
      throw std::runtime_error{"workload artifact has trailing data"};
    }
  }

 private:
  std::ifstream stream_;
};

template <std::size_t Size>
void write_magic(BinaryWriter& writer, const std::array<char, Size>& magic) {
  writer.bytes(magic.data(), magic.size());
}

template <std::size_t Size>
void require_magic(BinaryReader& reader, const std::array<char, Size>& expected) {
  std::array<char, Size> actual{};
  reader.bytes(actual.data(), actual.size());
  if (actual != expected) {
    throw std::runtime_error{"unrecognized workload artifact magic"};
  }
}

void prepare_parent(const std::filesystem::path& path) {
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

template <typename Id>
void write_id_vector(BinaryWriter& writer, const std::vector<Id>& values) {
  if (!std::in_range<std::uint32_t>(values.size())) {
    throw std::length_error{"artifact ID vector exceeds 32-bit length"};
  }
  writer.unsigned_integer(static_cast<std::uint32_t>(values.size()));
  for (const Id value : values) {
    writer.unsigned_integer(value.value());
  }
}

template <typename Id>
[[nodiscard]] std::vector<Id> read_id_vector(BinaryReader& reader) {
  const std::uint32_t count = reader.unsigned_integer<std::uint32_t>();
  std::vector<Id> values;
  values.reserve(count);
  for (std::uint32_t index = 0U; index < count; ++index) {
    values.push_back(Id{reader.unsigned_integer<typename Id::representation_type>()});
  }
  return values;
}

void write_u32_vector(BinaryWriter& writer, const std::vector<std::uint32_t>& values) {
  if (!std::in_range<std::uint32_t>(values.size())) {
    throw std::length_error{"artifact index vector exceeds 32-bit length"};
  }
  writer.unsigned_integer(static_cast<std::uint32_t>(values.size()));
  for (const std::uint32_t value : values) {
    writer.unsigned_integer(value);
  }
}

[[nodiscard]] std::vector<std::uint32_t> read_u32_vector(BinaryReader& reader) {
  const std::uint32_t count = reader.unsigned_integer<std::uint32_t>();
  std::vector<std::uint32_t> values;
  values.reserve(count);
  for (std::uint32_t index = 0U; index < count; ++index) {
    values.push_back(reader.unsigned_integer<std::uint32_t>());
  }
  return values;
}

void write_terminal(BinaryWriter& writer, const MappedTerminal& terminal) {
  writer.string(terminal.site);
  writer.string(terminal.pin);
  writer.unsigned_integer(terminal.original_vertex.value());
  writer.unsigned_integer(terminal.vertex.value());
  writer.signed_integer(terminal.x);
  writer.signed_integer(terminal.y);
  writer.boolean(terminal.has_location);
  writer.unsigned_integer(terminal.owner_tile.value());
}

[[nodiscard]] MappedTerminal read_terminal(BinaryReader& reader) {
  MappedTerminal terminal;
  terminal.site = reader.string();
  terminal.pin = reader.string();
  terminal.original_vertex = VertexId{reader.unsigned_integer<std::uint32_t>()};
  terminal.vertex = VertexId{reader.unsigned_integer<std::uint32_t>()};
  terminal.x = reader.signed_integer();
  terminal.y = reader.signed_integer();
  terminal.has_location = reader.boolean();
  terminal.owner_tile = TileId{reader.unsigned_integer<std::uint32_t>()};
  return terminal;
}

void validate_terminal(
    const MappedTerminal& terminal,
    const WeightedGraph& graph) {
  if (!is_valid_vertex_id(terminal.vertex, graph.vertex_count()) ||
      terminal.original_vertex.value() >= graph.old_to_new().size() ||
      graph.old_to_new()[terminal.original_vertex.value()] != terminal.vertex) {
    throw std::runtime_error{"query artifact terminal vertex mapping is invalid"};
  }
  const VertexMetadata metadata = graph.vertices()[terminal.vertex.value()];
  if (metadata.x != terminal.x || metadata.y != terminal.y ||
      metadata.has_location != terminal.has_location ||
      graph.owner_tiles()[terminal.vertex.value()] != terminal.owner_tile) {
    throw std::runtime_error{"query artifact terminal spatial metadata is invalid"};
  }
}

}  // namespace

void write_graph_artifact(
    const std::filesystem::path& path,
    const InputGraph& graph) {
  prepare_parent(path);
  BinaryWriter writer(path);
  write_magic(writer, graph_magic);
  writer.unsigned_integer(workload_artifact_version);
  writer.unsigned_integer(graph.vertex_count());
  writer.unsigned_integer(graph.edge_count());
  for (const VertexMetadata& vertex : graph.vertices()) {
    writer.signed_integer(vertex.x);
    writer.signed_integer(vertex.y);
    writer.boolean(vertex.has_location);
    writer.unsigned_integer(vertex.resource_class.value());
  }
  for (const EdgeInputRecord& edge : graph.edges()) {
    writer.unsigned_integer(edge.source.value());
    writer.unsigned_integer(edge.destination.value());
    writer.unsigned_integer(canonical_weight_bits(edge.weight));
    writer.unsigned_integer(edge.provenance.domain);
    writer.unsigned_integer(edge.provenance.kind_and_flags);
    writer.unsigned_integer(edge.provenance.source_record);
  }
}

InputGraph read_graph_artifact(const std::filesystem::path& path) {
  BinaryReader reader(path);
  require_magic(reader, graph_magic);
  if (reader.unsigned_integer<std::uint32_t>() != workload_artifact_version) {
    throw std::runtime_error{"unsupported graph artifact version"};
  }
  const std::uint32_t vertex_count = reader.unsigned_integer<std::uint32_t>();
  const std::uint64_t edge_count = reader.unsigned_integer<std::uint64_t>();
  if (!std::in_range<std::size_t>(edge_count)) {
    throw std::length_error{"graph artifact edge count exceeds host size"};
  }
  std::vector<VertexMetadata> vertices;
  vertices.reserve(vertex_count);
  for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex) {
    const std::int32_t x = reader.signed_integer();
    const std::int32_t y = reader.signed_integer();
    const bool has_location = reader.boolean();
    const ResourceClassId resource{
        reader.unsigned_integer<ResourceClassId::representation_type>()};
    vertices.push_back(VertexMetadata{x, y, has_location, resource});
  }
  std::vector<EdgeInputRecord> edges;
  edges.reserve(static_cast<std::size_t>(edge_count));
  for (std::uint64_t edge = 0U; edge < edge_count; ++edge) {
    const VertexId source{reader.unsigned_integer<std::uint32_t>()};
    const VertexId destination{reader.unsigned_integer<std::uint32_t>()};
    const float weight =
        std::bit_cast<float>(reader.unsigned_integer<std::uint32_t>());
    const std::uint32_t domain = reader.unsigned_integer<std::uint32_t>();
    const std::uint32_t kind = reader.unsigned_integer<std::uint32_t>();
    const std::uint64_t source_record = reader.unsigned_integer<std::uint64_t>();
    edges.push_back(EdgeInputRecord{
        source, destination, weight, PhysicalProvenance{domain, kind, source_record}});
  }
  reader.require_eof();
  return InputGraph(std::move(vertices), std::move(edges));
}

void write_query_artifact(
    const std::filesystem::path& path,
    const std::vector<MappedRouteQuery>& queries) {
  prepare_parent(path);
  BinaryWriter writer(path);
  write_magic(writer, query_magic);
  writer.unsigned_integer(workload_artifact_version);
  writer.unsigned_integer(static_cast<std::uint64_t>(queries.size()));
  for (const MappedRouteQuery& mapped : queries) {
    writer.string(mapped.net_name);
    writer.unsigned_integer(mapped.query.query_id.value());
    write_id_vector(writer, mapped.query.source_terminals);
    write_id_vector(writer, mapped.query.target_terminals);
    write_id_vector(writer, mapped.query.sources);
    write_id_vector(writer, mapped.query.targets);
    write_u32_vector(writer, mapped.query.source_terminal_to_source);
    write_u32_vector(writer, mapped.query.target_terminal_to_target);
    write_id_vector(writer, mapped.query.selected_tiles);
    writer.unsigned_integer(mapped.query.expansion_generation);

    if (!std::in_range<std::uint32_t>(mapped.source_terminals.size()) ||
        !std::in_range<std::uint32_t>(mapped.target_terminals.size())) {
      throw std::length_error{"artifact terminal vector exceeds 32-bit length"};
    }
    writer.unsigned_integer(static_cast<std::uint32_t>(mapped.source_terminals.size()));
    for (const MappedTerminal& terminal : mapped.source_terminals) {
      write_terminal(writer, terminal);
    }
    writer.unsigned_integer(static_cast<std::uint32_t>(mapped.target_terminals.size()));
    for (const MappedTerminal& terminal : mapped.target_terminals) {
      write_terminal(writer, terminal);
    }
  }
}

std::vector<MappedRouteQuery> read_query_artifact(
    const std::filesystem::path& path,
    const WeightedGraph& spatial_graph) {
  BinaryReader reader(path);
  require_magic(reader, query_magic);
  if (reader.unsigned_integer<std::uint32_t>() != workload_artifact_version) {
    throw std::runtime_error{"unsupported query artifact version"};
  }
  const std::uint64_t query_count = reader.unsigned_integer<std::uint64_t>();
  if (!std::in_range<std::size_t>(query_count)) {
    throw std::length_error{"query artifact count exceeds host size"};
  }
  std::vector<MappedRouteQuery> queries;
  queries.reserve(static_cast<std::size_t>(query_count));
  for (std::uint64_t query_index = 0U; query_index < query_count; ++query_index) {
    MappedRouteQuery mapped;
    mapped.net_name = reader.string();
    mapped.query.query_id = QueryId{reader.unsigned_integer<std::uint32_t>()};
    mapped.query.source_terminals = read_id_vector<VertexId>(reader);
    mapped.query.target_terminals = read_id_vector<VertexId>(reader);
    mapped.query.sources = read_id_vector<VertexId>(reader);
    mapped.query.targets = read_id_vector<VertexId>(reader);
    mapped.query.source_terminal_to_source = read_u32_vector(reader);
    mapped.query.target_terminal_to_target = read_u32_vector(reader);
    mapped.query.selected_tiles = read_id_vector<TileId>(reader);
    mapped.query.expansion_generation = reader.unsigned_integer<std::uint32_t>();

    const std::uint32_t source_count = reader.unsigned_integer<std::uint32_t>();
    mapped.source_terminals.reserve(source_count);
    for (std::uint32_t terminal = 0U; terminal < source_count; ++terminal) {
      mapped.source_terminals.push_back(read_terminal(reader));
    }
    const std::uint32_t target_count = reader.unsigned_integer<std::uint32_t>();
    mapped.target_terminals.reserve(target_count);
    for (std::uint32_t terminal = 0U; terminal < target_count; ++terminal) {
      mapped.target_terminals.push_back(read_terminal(reader));
    }

    if (mapped.source_terminals.size() != mapped.query.source_terminals.size() ||
        mapped.target_terminals.size() != mapped.query.target_terminals.size()) {
      throw std::runtime_error{"query artifact terminal descriptor count mismatch"};
    }
    if (!validate_route_query(spatial_graph, mapped.query).ok()) {
      throw std::runtime_error{"query artifact contains an invalid RouteQuery"};
    }
    for (std::size_t terminal = 0U; terminal < mapped.source_terminals.size();
         ++terminal) {
      validate_terminal(mapped.source_terminals[terminal], spatial_graph);
      if (mapped.source_terminals[terminal].vertex !=
          mapped.query.source_terminals[terminal]) {
        throw std::runtime_error{"source descriptor and query terminal disagree"};
      }
    }
    for (std::size_t terminal = 0U; terminal < mapped.target_terminals.size();
         ++terminal) {
      validate_terminal(mapped.target_terminals[terminal], spatial_graph);
      if (mapped.target_terminals[terminal].vertex !=
          mapped.query.target_terminals[terminal]) {
        throw std::runtime_error{"target descriptor and query terminal disagree"};
      }
    }
    queries.push_back(std::move(mapped));
  }
  reader.require_eof();
  return queries;
}

bool files_are_byte_identical(
    const std::filesystem::path& left,
    const std::filesystem::path& right) {
  std::ifstream left_stream(left, std::ios::binary);
  std::ifstream right_stream(right, std::ios::binary);
  if (!left_stream || !right_stream) {
    throw std::runtime_error{"cannot open artifact for deterministic comparison"};
  }
  constexpr std::size_t buffer_size = 64U * 1024U;
  std::array<char, buffer_size> left_buffer{};
  std::array<char, buffer_size> right_buffer{};
  while (left_stream || right_stream) {
    left_stream.read(left_buffer.data(), static_cast<std::streamsize>(left_buffer.size()));
    right_stream.read(
        right_buffer.data(), static_cast<std::streamsize>(right_buffer.size()));
    const std::streamsize left_count = left_stream.gcount();
    const std::streamsize right_count = right_stream.gcount();
    if (left_count != right_count ||
        !std::equal(
            left_buffer.begin(),
            left_buffer.begin() + left_count,
            right_buffer.begin())) {
      return false;
    }
  }
  return true;
}

}  // namespace bfnew
