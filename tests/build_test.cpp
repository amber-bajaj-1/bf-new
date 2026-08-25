#include <array>
#include <concepts>
#include <span>

static_assert(__cplusplus >= 202002L);

template <std::integral T>
constexpr T identity(T value) {
  return value;
}

int main() {
  constexpr std::array values{identity(20), identity(22)};
  const std::span<const int> view{values};
  return view[0] + view[1] == 42 ? 0 : 1;
}
