#include "bfnew/hip/runtime.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <utility>

namespace bfnew::hip {
namespace {

[[nodiscard]] std::size_t checked_add(
    const std::size_t left,
    const std::size_t right,
    const std::string_view what) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  return left + right;
}

[[nodiscard]] std::size_t checked_multiply(
    const std::size_t count,
    const std::size_t width,
    const std::string_view what) {
  if (width != 0U && count > std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error{std::string{what} + " overflow"};
  }
  return count * width;
}

template <typename T>
[[nodiscard]] T* device_pointer(void* pointer) noexcept {
  return static_cast<T*>(pointer);
}

template <typename T>
[[nodiscard]] const T* device_pointer(const void* pointer) noexcept {
  return static_cast<const T*>(pointer);
}

template <typename T>
void validate_offsets(
    const std::vector<T>& offsets,
    const std::size_t bucket_count,
    const T entry_count,
    const std::string_view name) {
  if (offsets.size() != checked_add(bucket_count, 1U, name) || offsets.empty() ||
      offsets.front() != T{} || offsets.back() != entry_count) {
    throw std::invalid_argument{std::string{name} + " has an invalid shape"};
  }
  for (std::size_t index = 1U; index < offsets.size(); ++index) {
    if (offsets[index] < offsets[index - 1U] || offsets[index] > entry_count) {
      throw std::invalid_argument{std::string{name} + " is not a valid offset table"};
    }
  }
}

void validate_run_layout32(
    const std::vector<std::uint32_t>& bucket_edge_offsets,
    const std::vector<std::uint32_t>& bucket_run_offsets,
    const std::vector<std::uint32_t>& run_edge_offsets,
    const std::vector<std::uint32_t>& run_tiles,
    const std::vector<std::uint32_t>& remote_endpoints,
    const std::vector<std::uint32_t>& owner_tiles,
    const std::string_view name) {
  for (std::size_t bucket = 0U; bucket < owner_tiles.size(); ++bucket) {
    const std::uint32_t edge_begin = bucket_edge_offsets[bucket];
    const std::uint32_t edge_end = bucket_edge_offsets[bucket + 1U];
    const std::uint32_t run_begin = bucket_run_offsets[bucket];
    const std::uint32_t run_end = bucket_run_offsets[bucket + 1U];
    std::uint32_t cursor = edge_begin;
    std::uint32_t preceding_tile = 0U;
    bool has_preceding_tile = false;
    for (std::uint32_t run = run_begin; run < run_end; ++run) {
      const std::uint32_t current_edge_begin = run_edge_offsets[run];
      const std::uint32_t current_edge_end = run_edge_offsets[run + 1U];
      const std::uint32_t remote_tile = run_tiles[run];
      if (current_edge_begin != cursor || current_edge_end <= current_edge_begin ||
          current_edge_end > edge_end) {
        throw std::invalid_argument{
            std::string{name} + " has an empty, crossing, or uncovered run"};
      }
      if (has_preceding_tile && remote_tile <= preceding_tile) {
        throw std::invalid_argument{
            std::string{name} + " runs are nonmaximal or out of tile order"};
      }
      for (std::uint32_t edge = current_edge_begin; edge < current_edge_end;
           ++edge) {
        const std::uint32_t endpoint = remote_endpoints[edge];
        if (owner_tiles[endpoint] != remote_tile) {
          throw std::invalid_argument{
              std::string{name} + " run tile disagrees with endpoint ownership"};
        }
      }
      cursor = current_edge_end;
      preceding_tile = remote_tile;
      has_preceding_tile = true;
    }
    if (cursor != edge_end) {
      throw std::invalid_argument{
          std::string{name} + " runs do not cover their sparse bucket"};
    }
  }
}

void validate_resident_layout(const DeviceGraphLayout32& layout) {
  const std::size_t vertices = layout.vertex_count;
  const std::size_t edges = layout.edge_count;
  const std::size_t tiles = layout.tile_count;
  const std::size_t csr_runs = layout.csr_run_destination_tiles.size();
  const std::size_t csc_runs = layout.csc_run_source_tiles.size();
  if (csr_runs > std::numeric_limits<std::uint32_t>::max() ||
      csc_runs > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error{"device tile-run count exceeds the 32-bit ABI"};
  }
  if (layout.owner_tiles.size() != vertices ||
      layout.csr_destinations.size() != edges || layout.csr_weights.size() != edges ||
      layout.csc_sources.size() != edges || layout.csc_weights.size() != edges ||
      layout.csc_edge_ids.size() != edges ||
      layout.csr_run_edge_offsets.size() != csr_runs + 1U ||
      layout.csc_run_edge_offsets.size() != csc_runs + 1U) {
    throw std::invalid_argument{"resident device layout component sizes disagree"};
  }

  validate_offsets(
      layout.csr_row_offsets, vertices, layout.edge_count, "CSR row offsets");
  validate_offsets(
      layout.csc_column_offsets, vertices, layout.edge_count, "CSC column offsets");
  validate_offsets(
      layout.csr_row_run_offsets,
      vertices,
      static_cast<std::uint32_t>(csr_runs),
      "CSR row-run offsets");
  validate_offsets(
      layout.csc_column_run_offsets,
      vertices,
      static_cast<std::uint32_t>(csc_runs),
      "CSC column-run offsets");
  validate_offsets(
      layout.csr_run_edge_offsets, csr_runs, layout.edge_count, "CSR run edges");
  validate_offsets(
      layout.csc_run_edge_offsets, csc_runs, layout.edge_count, "CSC run edges");

  const auto invalid_owner = [tiles](const std::uint32_t tile) {
    return static_cast<std::size_t>(tile) >= tiles;
  };
  const auto invalid_vertex = [vertices](const std::uint32_t vertex) {
    return static_cast<std::size_t>(vertex) >= vertices;
  };
  if (std::ranges::any_of(layout.owner_tiles, invalid_owner) ||
      std::ranges::any_of(layout.csr_run_destination_tiles, invalid_owner) ||
      std::ranges::any_of(layout.csc_run_source_tiles, invalid_owner) ||
      std::ranges::any_of(layout.csr_destinations, invalid_vertex) ||
      std::ranges::any_of(layout.csc_sources, invalid_vertex)) {
    throw std::invalid_argument{"resident device layout contains an out-of-range ID"};
  }
  std::vector<bool> edge_id_seen(edges, false);
  for (const std::uint32_t edge_id : layout.csc_edge_ids) {
    if (edge_id >= layout.edge_count || edge_id_seen[edge_id]) {
      throw std::invalid_argument{
          "resident CSC stable edge IDs are not a permutation"};
    }
    edge_id_seen[edge_id] = true;
  }
  const auto invalid_weight = [](const float weight) {
    return !std::isfinite(weight) || weight < 0.0F ||
           (weight == 0.0F && std::signbit(weight));
  };
  if (std::ranges::any_of(layout.csr_weights, invalid_weight) ||
      std::ranges::any_of(layout.csc_weights, invalid_weight)) {
    throw std::invalid_argument{"resident device layout contains an invalid weight"};
  }

  validate_run_layout32(
      layout.csr_row_offsets,
      layout.csr_row_run_offsets,
      layout.csr_run_edge_offsets,
      layout.csr_run_destination_tiles,
      layout.csr_destinations,
      layout.owner_tiles,
      "CSR");
  validate_run_layout32(
      layout.csc_column_offsets,
      layout.csc_column_run_offsets,
      layout.csc_run_edge_offsets,
      layout.csc_run_source_tiles,
      layout.csc_sources,
      layout.owner_tiles,
      "CSC");
}

template <typename T>
void upload_vector(
    DeviceBuffer& destination,
    const std::vector<T>& source,
    const HipStream& stream) {
  destination.copy_from_host_async(
      source.data(),
      checked_multiply(source.size(), sizeof(T), "device graph component bytes"),
      stream);
}

template <typename T>
void download_vector(
    const DeviceBuffer& source,
    std::vector<T>& destination,
    const HipStream& stream) {
  source.copy_to_host_async(
      destination.data(),
      checked_multiply(destination.size(), sizeof(T), "device graph component bytes"),
      stream);
}

template <typename T>
void resize_from_reported_bytes(
    std::vector<T>& destination,
    const std::uint64_t bytes,
    const std::string_view name) {
  if (bytes % sizeof(T) != 0U ||
      bytes / sizeof(T) > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error{
        std::string{name} + " byte report cannot be represented on the host"};
  }
  destination.resize(static_cast<std::size_t>(bytes / sizeof(T)));
}

}  // namespace

ResidentGraphPlan make_resident_graph_plan(DeviceGraphLayout32 layout) {
  validate_resident_layout(layout);
  const DeviceGraphMemoryReport memory = report_device_graph_memory(layout);
  const DeviceGraphFingerprint fingerprint =
      fingerprint_device_graph_layout32(layout);
  return ResidentGraphPlan{std::move(layout), memory, fingerprint};
}

class ResidentDeviceGraph::Impl final {
 public:
  enum class State : std::uint8_t {
    empty,
    uploading,
    ready,
    failed,
  };

  // Staging precedes device buffers so it is destroyed after them if teardown
  // happens during stack unwinding.  The destructor normally synchronizes it.
  DeviceGraphLayout32 staging;
  DeviceBuffer owner_tiles;
  DeviceBuffer csr_row_offsets;
  DeviceBuffer csr_destinations;
  DeviceBuffer csr_weights;
  DeviceBuffer csr_row_run_offsets;
  DeviceBuffer csr_run_edge_offsets;
  DeviceBuffer csr_run_destination_tiles;
  DeviceBuffer csc_column_offsets;
  DeviceBuffer csc_sources;
  DeviceBuffer csc_weights;
  DeviceBuffer csc_edge_ids;
  DeviceBuffer csc_column_run_offsets;
  DeviceBuffer csc_run_edge_offsets;
  DeviceBuffer csc_run_source_tiles;
  HipEvent ready_event{false};
  DeviceGraphMemoryReport memory{};
  DeviceGraphFingerprint fingerprint{};
  DeviceGraphView32 graph_view{};
  mutable State state{State::empty};

  void release_upload_staging() noexcept {
    staging = DeviceGraphLayout32{};
  }

  void synchronize_noexcept() noexcept {
    if (state == State::uploading || state == State::ready ||
        state == State::failed) {
      // Until execution registration arrives, any uploaded view may have been
      // consumed by later kernels.  Device-wide synchronization is the only
      // conservative nonthrowing teardown fence.
      const hipError_t status = hipDeviceSynchronize();
      state = status == hipSuccess ? State::ready : State::failed;
      if (status == hipSuccess) {
        release_upload_staging();
      }
    }
  }
};

ResidentDeviceGraph::ResidentDeviceGraph() : impl_{new Impl{}} {}

ResidentDeviceGraph::~ResidentDeviceGraph() noexcept {
  if (impl_ != nullptr) {
    impl_->synchronize_noexcept();
    delete impl_;
  }
}

ResidentDeviceGraph::ResidentDeviceGraph(ResidentDeviceGraph&& other) noexcept
    : impl_{std::exchange(other.impl_, nullptr)} {}

ResidentDeviceGraph& ResidentDeviceGraph::operator=(
    ResidentDeviceGraph&& other) noexcept {
  if (this != &other) {
    if (impl_ != nullptr) {
      impl_->synchronize_noexcept();
      delete impl_;
    }
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

void ResidentDeviceGraph::upload_once_async(
    ResidentGraphPlan plan,
    const HipStream& stream) {
  if (impl_ == nullptr) {
    throw std::logic_error{"cannot upload through a moved-from resident graph"};
  }
  if (impl_->state != Impl::State::empty) {
    throw std::logic_error{"resident graph accepts exactly one upload attempt"};
  }
  validate_resident_layout(plan.layout);
  const DeviceGraphMemoryReport actual_report =
      report_device_graph_memory(plan.layout);
  if (actual_report != plan.memory) {
    throw std::invalid_argument{"resident graph plan memory report is stale"};
  }
  const DeviceGraphFingerprint actual_fingerprint =
      fingerprint_device_graph_layout32(plan.layout);
  if (actual_fingerprint != plan.fingerprint) {
    throw std::invalid_argument{"resident graph plan fingerprint is stale"};
  }

  impl_->state = Impl::State::failed;
  impl_->staging = std::move(plan.layout);
  impl_->memory = plan.memory;
  impl_->fingerprint = plan.fingerprint;
  const auto allocate = [](DeviceBuffer& buffer, const std::uint64_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error{"resident graph component exceeds host size_t"};
    }
    static_cast<void>(buffer.reserve(
        static_cast<std::size_t>(bytes), BufferGrowth::exact));
  };
  allocate(impl_->owner_tiles, impl_->memory.owner_tiles_bytes);
  allocate(impl_->csr_row_offsets, impl_->memory.csr_row_offsets_bytes);
  allocate(impl_->csr_destinations, impl_->memory.csr_destinations_bytes);
  allocate(impl_->csr_weights, impl_->memory.csr_weights_bytes);
  allocate(impl_->csr_row_run_offsets, impl_->memory.csr_row_run_offsets_bytes);
  allocate(impl_->csr_run_edge_offsets, impl_->memory.csr_run_edge_offsets_bytes);
  allocate(
      impl_->csr_run_destination_tiles,
      impl_->memory.csr_run_destination_tiles_bytes);
  allocate(impl_->csc_column_offsets, impl_->memory.csc_column_offsets_bytes);
  allocate(impl_->csc_sources, impl_->memory.csc_sources_bytes);
  allocate(impl_->csc_weights, impl_->memory.csc_weights_bytes);
  allocate(impl_->csc_edge_ids, impl_->memory.csc_edge_ids_bytes);
  allocate(impl_->csc_column_run_offsets, impl_->memory.csc_column_run_offsets_bytes);
  allocate(impl_->csc_run_edge_offsets, impl_->memory.csc_run_edge_offsets_bytes);
  allocate(impl_->csc_run_source_tiles, impl_->memory.csc_run_source_tiles_bytes);

  const DeviceGraphLayout32& source = impl_->staging;
  upload_vector(impl_->owner_tiles, source.owner_tiles, stream);
  upload_vector(impl_->csr_row_offsets, source.csr_row_offsets, stream);
  upload_vector(impl_->csr_destinations, source.csr_destinations, stream);
  upload_vector(impl_->csr_weights, source.csr_weights, stream);
  upload_vector(impl_->csr_row_run_offsets, source.csr_row_run_offsets, stream);
  upload_vector(impl_->csr_run_edge_offsets, source.csr_run_edge_offsets, stream);
  upload_vector(
      impl_->csr_run_destination_tiles, source.csr_run_destination_tiles, stream);
  upload_vector(impl_->csc_column_offsets, source.csc_column_offsets, stream);
  upload_vector(impl_->csc_sources, source.csc_sources, stream);
  upload_vector(impl_->csc_weights, source.csc_weights, stream);
  upload_vector(impl_->csc_edge_ids, source.csc_edge_ids, stream);
  upload_vector(impl_->csc_column_run_offsets, source.csc_column_run_offsets, stream);
  upload_vector(impl_->csc_run_edge_offsets, source.csc_run_edge_offsets, stream);
  upload_vector(impl_->csc_run_source_tiles, source.csc_run_source_tiles, stream);

  const auto csr_run_count = static_cast<std::uint32_t>(
      source.csr_run_destination_tiles.size());
  const auto csc_run_count =
      static_cast<std::uint32_t>(source.csc_run_source_tiles.size());
  impl_->graph_view = DeviceGraphView32{
      source.vertex_count,
      source.edge_count,
      source.tile_count,
      device_pointer<const std::uint32_t>(impl_->owner_tiles.data()),
      DeviceCsrView32{
          source.vertex_count,
          source.edge_count,
          csr_run_count,
          device_pointer<const std::uint32_t>(impl_->csr_row_offsets.data()),
          device_pointer<const std::uint32_t>(impl_->csr_destinations.data()),
          device_pointer<const float>(impl_->csr_weights.data()),
          device_pointer<const std::uint32_t>(impl_->csr_row_run_offsets.data()),
          device_pointer<const std::uint32_t>(impl_->csr_run_edge_offsets.data()),
          device_pointer<const std::uint32_t>(
              impl_->csr_run_destination_tiles.data()),
      },
      DeviceCscView32{
          source.vertex_count,
          source.edge_count,
          csc_run_count,
          device_pointer<const std::uint32_t>(impl_->csc_column_offsets.data()),
          device_pointer<const std::uint32_t>(impl_->csc_sources.data()),
          device_pointer<const float>(impl_->csc_weights.data()),
          device_pointer<const std::uint32_t>(impl_->csc_edge_ids.data()),
          device_pointer<const std::uint32_t>(impl_->csc_column_run_offsets.data()),
          device_pointer<const std::uint32_t>(impl_->csc_run_edge_offsets.data()),
          device_pointer<const std::uint32_t>(impl_->csc_run_source_tiles.data()),
      },
  };
  impl_->ready_event.record(stream);
  impl_->state = Impl::State::uploading;
}

void ResidentDeviceGraph::wait_until_ready(const HipStream& stream) const {
  if (!has_upload()) {
    throw std::logic_error{"resident graph has no successful upload enqueue"};
  }
  impl_->ready_event.wait(stream);
}

void ResidentDeviceGraph::synchronize_upload() {
  if (!has_upload()) {
    throw std::logic_error{"resident graph has no successful upload enqueue"};
  }
  impl_->ready_event.synchronize();
  impl_->state = Impl::State::ready;
  impl_->release_upload_staging();
}

bool ResidentDeviceGraph::has_upload() const noexcept {
  return impl_ != nullptr &&
         (impl_->state == Impl::State::uploading ||
          impl_->state == Impl::State::ready);
}

bool ResidentDeviceGraph::upload_complete() const {
  if (!has_upload()) {
    return false;
  }
  if (impl_->state == Impl::State::ready) {
    return true;
  }
  if (impl_->ready_event.query()) {
    impl_->state = Impl::State::ready;
    impl_->release_upload_staging();
    return true;
  }
  return false;
}

const DeviceGraphView32& ResidentDeviceGraph::view() const {
  if (!has_upload()) {
    throw std::logic_error{"resident graph view requested before upload"};
  }
  return impl_->graph_view;
}

const DeviceGraphMemoryReport& ResidentDeviceGraph::memory_report() const {
  if (!has_upload()) {
    throw std::logic_error{"resident graph memory requested before upload"};
  }
  return impl_->memory;
}

const DeviceGraphFingerprint& ResidentDeviceGraph::fingerprint() const {
  if (!has_upload()) {
    throw std::logic_error{"resident graph fingerprint requested before upload"};
  }
  return impl_->fingerprint;
}

void ResidentDeviceGraph::download_async(
    DeviceGraphLayout32& destination,
    const HipStream& stream) const {
  if (!has_upload()) {
    throw std::logic_error{"resident graph download requested before upload"};
  }
  impl_->ready_event.wait(stream);
  destination.vertex_count = impl_->graph_view.vertex_count;
  destination.edge_count = impl_->graph_view.edge_count;
  destination.tile_count = impl_->graph_view.tile_count;
  resize_from_reported_bytes(
      destination.owner_tiles, impl_->memory.owner_tiles_bytes, "owner tiles");
  resize_from_reported_bytes(
      destination.csr_row_offsets,
      impl_->memory.csr_row_offsets_bytes,
      "CSR row offsets");
  resize_from_reported_bytes(
      destination.csr_destinations,
      impl_->memory.csr_destinations_bytes,
      "CSR destinations");
  resize_from_reported_bytes(
      destination.csr_weights, impl_->memory.csr_weights_bytes, "CSR weights");
  resize_from_reported_bytes(
      destination.csr_row_run_offsets,
      impl_->memory.csr_row_run_offsets_bytes,
      "CSR row-run offsets");
  resize_from_reported_bytes(
      destination.csr_run_edge_offsets,
      impl_->memory.csr_run_edge_offsets_bytes,
      "CSR run-edge offsets");
  resize_from_reported_bytes(
      destination.csr_run_destination_tiles,
      impl_->memory.csr_run_destination_tiles_bytes,
      "CSR run destination tiles");
  resize_from_reported_bytes(
      destination.csc_column_offsets,
      impl_->memory.csc_column_offsets_bytes,
      "CSC column offsets");
  resize_from_reported_bytes(
      destination.csc_sources, impl_->memory.csc_sources_bytes, "CSC sources");
  resize_from_reported_bytes(
      destination.csc_weights, impl_->memory.csc_weights_bytes, "CSC weights");
  resize_from_reported_bytes(
      destination.csc_edge_ids,
      impl_->memory.csc_edge_ids_bytes,
      "CSC stable edge IDs");
  resize_from_reported_bytes(
      destination.csc_column_run_offsets,
      impl_->memory.csc_column_run_offsets_bytes,
      "CSC column-run offsets");
  resize_from_reported_bytes(
      destination.csc_run_edge_offsets,
      impl_->memory.csc_run_edge_offsets_bytes,
      "CSC run-edge offsets");
  resize_from_reported_bytes(
      destination.csc_run_source_tiles,
      impl_->memory.csc_run_source_tiles_bytes,
      "CSC run source tiles");

  download_vector(impl_->owner_tiles, destination.owner_tiles, stream);
  download_vector(impl_->csr_row_offsets, destination.csr_row_offsets, stream);
  download_vector(impl_->csr_destinations, destination.csr_destinations, stream);
  download_vector(impl_->csr_weights, destination.csr_weights, stream);
  download_vector(impl_->csr_row_run_offsets, destination.csr_row_run_offsets, stream);
  download_vector(impl_->csr_run_edge_offsets, destination.csr_run_edge_offsets, stream);
  download_vector(
      impl_->csr_run_destination_tiles,
      destination.csr_run_destination_tiles,
      stream);
  download_vector(impl_->csc_column_offsets, destination.csc_column_offsets, stream);
  download_vector(impl_->csc_sources, destination.csc_sources, stream);
  download_vector(impl_->csc_weights, destination.csc_weights, stream);
  download_vector(impl_->csc_edge_ids, destination.csc_edge_ids, stream);
  download_vector(
      impl_->csc_column_run_offsets, destination.csc_column_run_offsets, stream);
  download_vector(
      impl_->csc_run_edge_offsets, destination.csc_run_edge_offsets, stream);
  download_vector(
      impl_->csc_run_source_tiles, destination.csc_run_source_tiles, stream);
}

}  // namespace bfnew::hip
