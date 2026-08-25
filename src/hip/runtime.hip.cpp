#include "bfnew/hip/runtime.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>
#include <sstream>
#include <utility>

namespace bfnew::hip {
namespace {

[[nodiscard]] hipStream_t as_stream(const void* handle) noexcept {
  return reinterpret_cast<hipStream_t>(const_cast<void*>(handle));
}

[[nodiscard]] hipEvent_t as_event(const void* handle) noexcept {
  return reinterpret_cast<hipEvent_t>(const_cast<void*>(handle));
}

[[nodiscard]] void* erase_stream(const hipStream_t handle) noexcept {
  return reinterpret_cast<void*>(handle);
}

[[nodiscard]] void* erase_event(const hipEvent_t handle) noexcept {
  return reinterpret_cast<void*>(handle);
}

void check(
    const hipError_t status,
    const std::string_view expression,
    const std::source_location location = std::source_location::current()) {
  throw_if_hip_error(static_cast<std::int32_t>(status), expression, location);
}

void require_region(
    const std::size_t capacity,
    const std::size_t offset,
    const std::size_t bytes,
    const std::string_view operation) {
  if (offset > capacity || bytes > capacity - offset) {
    throw std::out_of_range{
        std::string{operation} + " exceeds the persistent buffer capacity"};
  }
}

[[nodiscard]] std::size_t geometric_capacity(
    const std::size_t current,
    const std::size_t requested) {
  if (requested <= current) {
    return current;
  }
  std::size_t next = current == 0U ? requested : current;
  while (next < requested) {
    if (next > std::numeric_limits<std::size_t>::max() / 2U) {
      return requested;
    }
    next *= 2U;
  }
  return next;
}

}  // namespace

HipRuntimeError::HipRuntimeError(
    const std::int32_t status,
    std::string message)
    : std::runtime_error{std::move(message)}, status_{status} {}

void throw_if_hip_error(
    const std::int32_t status,
    const std::string_view expression,
    const std::source_location location) {
  if (status == static_cast<std::int32_t>(hipSuccess)) {
    return;
  }
  const char* description = hipGetErrorString(static_cast<hipError_t>(status));
  std::ostringstream message;
  message << "HIP call " << expression << " failed with status " << status;
  if (description != nullptr) {
    message << " (" << description << ')';
  }
  message << " at " << location.file_name() << ':' << location.line();
  throw HipRuntimeError{status, message.str()};
}

HipStream::HipStream(const bool nonblocking) {
  hipStream_t stream = nullptr;
  check(
      hipStreamCreateWithFlags(
          &stream, nonblocking ? hipStreamNonBlocking : hipStreamDefault),
      "hipStreamCreateWithFlags");
  handle_ = erase_stream(stream);
}

HipStream::~HipStream() noexcept {
  if (handle_ != nullptr) {
    static_cast<void>(hipStreamDestroy(as_stream(handle_)));
  }
}

HipStream::HipStream(HipStream&& other) noexcept
    : handle_{std::exchange(other.handle_, nullptr)} {}

HipStream& HipStream::operator=(HipStream&& other) noexcept {
  if (this != &other) {
    if (handle_ != nullptr) {
      static_cast<void>(hipStreamDestroy(as_stream(handle_)));
    }
    handle_ = std::exchange(other.handle_, nullptr);
  }
  return *this;
}

void HipStream::synchronize() const {
  if (handle_ == nullptr) {
    throw std::logic_error{"cannot synchronize a moved-from HIP stream"};
  }
  check(hipStreamSynchronize(as_stream(handle_)), "hipStreamSynchronize");
}

HipEvent::HipEvent(const bool enable_timing) : timing_enabled_{enable_timing} {
  hipEvent_t event = nullptr;
  check(
      hipEventCreateWithFlags(
          &event, enable_timing ? hipEventDefault : hipEventDisableTiming),
      "hipEventCreateWithFlags");
  handle_ = erase_event(event);
}

HipEvent::~HipEvent() noexcept {
  if (handle_ != nullptr) {
    static_cast<void>(hipEventDestroy(as_event(handle_)));
  }
}

HipEvent::HipEvent(HipEvent&& other) noexcept
    : handle_{std::exchange(other.handle_, nullptr)},
      timing_enabled_{std::exchange(other.timing_enabled_, false)} {}

HipEvent& HipEvent::operator=(HipEvent&& other) noexcept {
  if (this != &other) {
    if (handle_ != nullptr) {
      static_cast<void>(hipEventDestroy(as_event(handle_)));
    }
    handle_ = std::exchange(other.handle_, nullptr);
    timing_enabled_ = std::exchange(other.timing_enabled_, false);
  }
  return *this;
}

void HipEvent::record(const HipStream& stream) {
  if (handle_ == nullptr || stream.native_handle() == nullptr) {
    throw std::logic_error{"cannot record a moved-from HIP event or stream"};
  }
  check(
      hipEventRecord(as_event(handle_), as_stream(stream.native_handle())),
      "hipEventRecord");
}

void HipEvent::wait(const HipStream& stream) const {
  if (handle_ == nullptr || stream.native_handle() == nullptr) {
    throw std::logic_error{"cannot wait on a moved-from HIP event or stream"};
  }
  check(
      hipStreamWaitEvent(
          as_stream(stream.native_handle()), as_event(handle_), 0U),
      "hipStreamWaitEvent");
}

void HipEvent::synchronize() const {
  if (handle_ == nullptr) {
    throw std::logic_error{"cannot synchronize a moved-from HIP event"};
  }
  check(hipEventSynchronize(as_event(handle_)), "hipEventSynchronize");
}

bool HipEvent::query() const {
  if (handle_ == nullptr) {
    throw std::logic_error{"cannot query a moved-from HIP event"};
  }
  const hipError_t status = hipEventQuery(as_event(handle_));
  if (status == hipErrorNotReady) {
    return false;
  }
  check(status, "hipEventQuery");
  return true;
}

float HipEvent::elapsed_milliseconds(
    const HipEvent& start,
    const HipEvent& stop) {
  if (!start.timing_enabled_ || !stop.timing_enabled_ || start.handle_ == nullptr ||
      stop.handle_ == nullptr) {
    throw std::invalid_argument{"HIP event timing requires two live timing events"};
  }
  float milliseconds = 0.0F;
  check(
      hipEventElapsedTime(
          &milliseconds, as_event(start.handle_), as_event(stop.handle_)),
      "hipEventElapsedTime");
  return milliseconds;
}

DeviceBuffer::~DeviceBuffer() noexcept { release(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_{std::exchange(other.data_, nullptr)},
      capacity_bytes_{std::exchange(other.capacity_bytes_, 0U)} {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    release();
    data_ = std::exchange(other.data_, nullptr);
    capacity_bytes_ = std::exchange(other.capacity_bytes_, 0U);
  }
  return *this;
}

bool DeviceBuffer::reserve(
    const std::size_t minimum_bytes,
    const BufferGrowth growth) {
  if (minimum_bytes <= capacity_bytes_) {
    return false;
  }
  const std::size_t next = growth == BufferGrowth::exact
                               ? minimum_bytes
                               : geometric_capacity(capacity_bytes_, minimum_bytes);
  void* replacement = nullptr;
  check(hipMalloc(&replacement, next), "hipMalloc");
  if (data_ != nullptr) {
    const hipError_t status = hipFree(data_);
    if (status != hipSuccess) {
      static_cast<void>(hipFree(replacement));
      check(status, "hipFree");
    }
  }
  data_ = replacement;
  capacity_bytes_ = next;
  return true;
}

void DeviceBuffer::release() noexcept {
  if (data_ != nullptr) {
    static_cast<void>(hipFree(data_));
  }
  data_ = nullptr;
  capacity_bytes_ = 0U;
}

void DeviceBuffer::clear_async(
    const std::size_t bytes,
    const HipStream& stream,
    const std::size_t destination_offset) {
  require_region(capacity_bytes_, destination_offset, bytes, "device clear");
  if (bytes == 0U) {
    return;
  }
  if (stream.native_handle() == nullptr) {
    throw std::invalid_argument{"device clear requires a live HIP stream"};
  }
  auto* destination = static_cast<std::byte*>(data_) + destination_offset;
  check(
      hipMemsetAsync(destination, 0, bytes, as_stream(stream.native_handle())),
      "hipMemsetAsync");
}

void DeviceBuffer::copy_from_host_async(
    const void* source,
    const std::size_t bytes,
    const HipStream& stream,
    const std::size_t destination_offset) {
  require_region(capacity_bytes_, destination_offset, bytes, "host-to-device copy");
  if (bytes == 0U) {
    return;
  }
  if (source == nullptr || stream.native_handle() == nullptr) {
    throw std::invalid_argument{"host-to-device copy requires live endpoints"};
  }
  auto* destination = static_cast<std::byte*>(data_) + destination_offset;
  check(
      hipMemcpyAsync(
          destination,
          source,
          bytes,
          hipMemcpyHostToDevice,
          as_stream(stream.native_handle())),
      "hipMemcpyAsync(host-to-device)");
}

void DeviceBuffer::copy_to_host_async(
    void* destination,
    const std::size_t bytes,
    const HipStream& stream,
    const std::size_t source_offset) const {
  require_region(capacity_bytes_, source_offset, bytes, "device-to-host copy");
  if (bytes == 0U) {
    return;
  }
  if (destination == nullptr || stream.native_handle() == nullptr) {
    throw std::invalid_argument{"device-to-host copy requires live endpoints"};
  }
  const auto* source = static_cast<const std::byte*>(data_) + source_offset;
  check(
      hipMemcpyAsync(
          destination,
          source,
          bytes,
          hipMemcpyDeviceToHost,
          as_stream(stream.native_handle())),
      "hipMemcpyAsync(device-to-host)");
}

HipEventTimer::HipEventTimer() : start_{true}, stop_{true} {}

void HipEventTimer::start(const HipStream& stream) {
  start_.record(stream);
  started_ = true;
  stopped_ = false;
}

void HipEventTimer::stop(const HipStream& stream) {
  if (!started_) {
    throw std::logic_error{"HIP event timer was stopped before it was started"};
  }
  stop_.record(stream);
  stopped_ = true;
}

float HipEventTimer::elapsed_milliseconds() {
  if (!started_ || !stopped_) {
    throw std::logic_error{"HIP event timer requires a completed interval"};
  }
  stop_.synchronize();
  return elapsed_milliseconds_after_stream_synchronization();
}

float HipEventTimer::elapsed_milliseconds_after_stream_synchronization()
    const {
  if (!started_ || !stopped_) {
    throw std::logic_error{"HIP event timer requires a completed interval"};
  }
  return HipEvent::elapsed_milliseconds(start_, stop_);
}

SteadyClockTimer::SteadyClockTimer() noexcept
    : start_{std::chrono::steady_clock::now()} {}

void SteadyClockTimer::reset() noexcept {
  start_ = std::chrono::steady_clock::now();
}

double SteadyClockTimer::elapsed_milliseconds() const noexcept {
  const auto elapsed = std::chrono::steady_clock::now() - start_;
  return std::chrono::duration<double, std::milli>{elapsed}.count();
}

}  // namespace bfnew::hip
