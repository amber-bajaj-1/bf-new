#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace bfnew::hip_tool {

inline void check(
    const hipError_t error,
    const char* expression,
    const char* file,
    const int line) {
  if (error == hipSuccess) {
    return;
  }
  std::ostringstream message;
  message << file << ':' << line << ": " << expression << " failed: "
          << hipGetErrorName(error) << " (" << hipGetErrorString(error) << ')';
  throw std::runtime_error{message.str()};
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(const std::size_t element_count) : size_{element_count} {
    if (size_ != 0U) {
      check(
          hipMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)),
          "hipMalloc",
          __FILE__,
          __LINE__);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_{std::exchange(other.data_, nullptr)},
        size_{std::exchange(other.size_, 0U)} {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0U);
    }
    return *this;
  }

  ~DeviceBuffer() { reset(); }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  void reset() noexcept {
    if (data_ != nullptr) {
      static_cast<void>(hipFree(data_));
      data_ = nullptr;
      size_ = 0U;
    }
  }

  T* data_{};
  std::size_t size_{};
};

}  // namespace bfnew::hip_tool

#define BFNEW_HIP_CHECK(expression) \
  ::bfnew::hip_tool::check((expression), #expression, __FILE__, __LINE__)
