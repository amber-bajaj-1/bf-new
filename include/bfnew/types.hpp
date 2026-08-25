#pragma once

#include <compare>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace bfnew {

template <typename Tag, std::unsigned_integral Representation>
class StrongId {
 public:
  using representation_type = Representation;

  constexpr StrongId() noexcept = default;
  explicit constexpr StrongId(Representation value) noexcept : value_{value} {}

  [[nodiscard]] constexpr Representation value() const noexcept { return value_; }

  constexpr auto operator<=>(const StrongId&) const noexcept = default;

 private:
  Representation value_{};
};

struct VertexIdTag;
struct EdgeIdTag;
struct TileIdTag;
struct QueryIdTag;
struct ResourceClassIdTag;

using VertexId = StrongId<VertexIdTag, std::uint32_t>;
using EdgeId = StrongId<EdgeIdTag, std::uint64_t>;
using TileId = StrongId<TileIdTag, std::uint32_t>;
using QueryId = StrongId<QueryIdTag, std::uint32_t>;
using ResourceClassId = StrongId<ResourceClassIdTag, std::uint16_t>;
using EdgeCount = std::uint64_t;
using EdgeOffset = std::uint64_t;

template <typename Id>
concept StrongIdType = requires {
  typename Id::representation_type;
  requires std::unsigned_integral<typename Id::representation_type>;
};

template <StrongIdType Id, std::integral Source>
[[nodiscard]] constexpr Id checked_id(Source value) {
  using Representation = typename Id::representation_type;
  if (!std::in_range<Representation>(value)) {
    throw std::out_of_range{"integer value is outside the strong ID range"};
  }
  return Id{static_cast<Representation>(value)};
}

namespace provenance_domain {

inline constexpr std::uint32_t synthetic = 1U;
inline constexpr std::uint32_t fpga_interchange = 2U;

}  // namespace provenance_domain

namespace provenance_kind {

inline constexpr std::uint32_t synthetic_edge = 1U;
inline constexpr std::uint32_t fpga_interchange_pip_forward = 2U;
inline constexpr std::uint32_t fpga_interchange_pip_reverse = 3U;

}  // namespace provenance_kind

struct PhysicalProvenance {
  std::uint32_t domain{};
  std::uint32_t kind_and_flags{};
  std::uint64_t source_record{};

  constexpr auto operator<=>(const PhysicalProvenance&) const noexcept = default;
};

static_assert(sizeof(VertexId) == sizeof(std::uint32_t));
static_assert(sizeof(EdgeId) == sizeof(std::uint64_t));
static_assert(sizeof(TileId) == sizeof(std::uint32_t));
static_assert(sizeof(QueryId) == sizeof(std::uint32_t));
static_assert(sizeof(ResourceClassId) == sizeof(std::uint16_t));
static_assert(std::is_trivially_copyable_v<VertexId>);
static_assert(std::is_trivially_copyable_v<EdgeId>);
static_assert(std::is_trivially_copyable_v<TileId>);
static_assert(std::is_trivially_copyable_v<QueryId>);
static_assert(std::is_standard_layout_v<PhysicalProvenance>);
static_assert(std::is_trivially_copyable_v<PhysicalProvenance>);
static_assert(sizeof(PhysicalProvenance) == 16U);

}  // namespace bfnew
