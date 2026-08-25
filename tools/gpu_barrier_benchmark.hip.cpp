#include "hip_tool_support.hpp"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cooperative_groups = ::cooperative_groups;

namespace {

struct Options {
  std::vector<int> block_sizes{128, 256, 512};
  std::vector<int> blocks_per_scheduling_unit{1, 2, 4, -1};
  std::vector<std::uint32_t> barrier_counts{1U, 10U, 100U, 1000U};
  std::uint32_t repetitions{10U};
};

__global__ void cooperative_barrier_kernel(
    std::uint64_t* block_sinks,
    const std::uint32_t barrier_count) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  std::uint64_t accumulator =
      static_cast<std::uint64_t>(blockIdx.x + 1U) * 0x9E3779B97F4A7C15ULL +
      static_cast<std::uint64_t>(threadIdx.x + 1U);
  for (std::uint32_t barrier = 0U; barrier < barrier_count; ++barrier) {
    accumulator ^= static_cast<std::uint64_t>(barrier + 1U) +
                   (accumulator << 6U) + (accumulator >> 2U);
    grid.sync();
  }
  if (threadIdx.x == 0U) {
    block_sinks[blockIdx.x] = accumulator;
  }
}

[[nodiscard]] std::uint64_t parse_unsigned(
    const std::string_view text,
    const std::string_view option) {
  std::size_t consumed = 0U;
  unsigned long long value = 0U;
  try {
    value = std::stoull(std::string{text}, &consumed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument{std::string{option} + " requires integers"};
  }
  if (consumed != text.size()) {
    throw std::invalid_argument{std::string{option} + " requires integers"};
  }
  return static_cast<std::uint64_t>(value);
}

template <typename Value>
[[nodiscard]] std::vector<Value> parse_csv(
    const std::string_view input,
    const std::string_view option) {
  std::vector<Value> values;
  std::size_t begin = 0U;
  while (begin <= input.size()) {
    const std::size_t end = input.find(',', begin);
    const std::string_view item = input.substr(
        begin, end == std::string_view::npos ? input.size() - begin : end - begin);
    if (item.empty()) {
      throw std::invalid_argument{std::string{option} + " contains an empty value"};
    }
    const std::uint64_t parsed = parse_unsigned(item, option);
    if (parsed == 0U || parsed > static_cast<std::uint64_t>(
                                      std::numeric_limits<Value>::max())) {
      throw std::invalid_argument{std::string{option} + " value is out of range"};
    }
    values.push_back(static_cast<Value>(parsed));
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  return values;
}

[[nodiscard]] std::vector<int> parse_residency_csv(
    const std::string_view input) {
  std::vector<int> values;
  std::size_t begin = 0U;
  while (begin <= input.size()) {
    const std::size_t end = input.find(',', begin);
    const std::string_view item = input.substr(
        begin, end == std::string_view::npos ? input.size() - begin : end - begin);
    if (item == "max") {
      values.push_back(-1);
    } else {
      const std::uint64_t parsed = parse_unsigned(item, "--blocks-per-wgp");
      if (parsed == 0U || parsed > static_cast<std::uint64_t>(
                                      std::numeric_limits<int>::max())) {
        throw std::invalid_argument{"--blocks-per-wgp value is out of range"};
      }
      values.push_back(static_cast<int>(parsed));
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  return values;
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--help") {
      std::cout
          << "usage: bfnew_gpu_barrier_benchmark "
             "[--block-sizes 128,256,512] [--blocks-per-wgp 1,2,4,max] "
             "[--barriers 1,10,100,1000] [--repetitions N]\n";
      std::exit(0);
    }
    if (argument != "--block-sizes" && argument != "--blocks-per-wgp" &&
        argument != "--barriers" && argument != "--repetitions") {
      throw std::invalid_argument{"unknown option: " + std::string{argument}};
    }
    if (++index >= argc) {
      throw std::invalid_argument{std::string{argument} + " requires a value"};
    }
    const std::string_view value{argv[index]};
    if (argument == "--block-sizes") {
      options.block_sizes = parse_csv<int>(value, argument);
    } else if (argument == "--blocks-per-wgp") {
      options.blocks_per_scheduling_unit = parse_residency_csv(value);
    } else if (argument == "--barriers") {
      options.barrier_counts = parse_csv<std::uint32_t>(value, argument);
    } else {
      const std::uint64_t parsed = parse_unsigned(value, argument);
      if (parsed == 0U || parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument{"--repetitions is outside the valid range"};
      }
      options.repetitions = static_cast<std::uint32_t>(parsed);
    }
  }
  return options;
}

void launch_barrier_kernel(
    std::uint64_t* sinks,
    const int grid_blocks,
    const int block_size,
    const std::uint32_t barrier_count) {
  std::uint64_t* sink_pointer = sinks;
  std::uint32_t count = barrier_count;
  void* arguments[] = {&sink_pointer, &count};
  BFNEW_HIP_CHECK(hipLaunchCooperativeKernel(
      reinterpret_cast<void*>(cooperative_barrier_kernel),
      dim3(static_cast<unsigned int>(grid_blocks)),
      dim3(static_cast<unsigned int>(block_size)),
      arguments,
      0U,
      nullptr));
}

[[nodiscard]] float measure_configuration(
    std::uint64_t* sinks,
    const int grid_blocks,
    const int block_size,
    const std::uint32_t barrier_count,
    const std::uint32_t repetitions) {
  launch_barrier_kernel(sinks, grid_blocks, block_size, barrier_count);
  BFNEW_HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t begin{};
  hipEvent_t end{};
  BFNEW_HIP_CHECK(hipEventCreate(&begin));
  BFNEW_HIP_CHECK(hipEventCreate(&end));
  try {
    BFNEW_HIP_CHECK(hipEventRecord(begin, nullptr));
    for (std::uint32_t repetition = 0U; repetition < repetitions; ++repetition) {
      launch_barrier_kernel(sinks, grid_blocks, block_size, barrier_count);
    }
    BFNEW_HIP_CHECK(hipEventRecord(end, nullptr));
    BFNEW_HIP_CHECK(hipEventSynchronize(end));
    float elapsed_ms = 0.0F;
    BFNEW_HIP_CHECK(hipEventElapsedTime(&elapsed_ms, begin, end));
    BFNEW_HIP_CHECK(hipEventDestroy(begin));
    BFNEW_HIP_CHECK(hipEventDestroy(end));
    return elapsed_ms;
  } catch (...) {
    static_cast<void>(hipEventDestroy(begin));
    static_cast<void>(hipEventDestroy(end));
    throw;
  }
}

void run_benchmark(const Options& options) {
  int device = 0;
  BFNEW_HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t properties{};
  BFNEW_HIP_CHECK(hipGetDeviceProperties(&properties, device));
  if (properties.cooperativeLaunch == 0) {
    throw std::runtime_error{"selected device does not support cooperative launch"};
  }

  std::cout << "# device=" << properties.name
            << " architecture=" << properties.gcnArchName
            << " wave_size=" << properties.warpSize
            << " wgp_or_multiprocessor_count=" << properties.multiProcessorCount
            << " repetitions=" << options.repetitions << '\n';
  std::cout
      << "block_size,occupancy_ceiling_blocks_per_wgp_or_multiprocessor,"
         "requested_blocks_per_wgp_or_multiprocessor,grid_blocks,barriers,"
         "repetitions,total_gpu_event_ms,mean_kernel_us,"
         "amortized_kernel_us_per_grid_barrier\n";

  for (const int block_size : options.block_sizes) {
    if (block_size <= 0 || block_size > properties.maxThreadsPerBlock) {
      std::cerr << "skipping illegal block size " << block_size << '\n';
      continue;
    }
    int occupancy_ceiling = 0;
    BFNEW_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy_ceiling,
        cooperative_barrier_kernel,
        block_size,
        0U));
    if (occupancy_ceiling <= 0) {
      std::cerr << "skipping block size " << block_size
                << " because its occupancy ceiling is zero\n";
      continue;
    }

    std::set<int> resident_counts;
    for (const int requested : options.blocks_per_scheduling_unit) {
      const int resolved = requested < 0 ? occupancy_ceiling : requested;
      if (resolved <= occupancy_ceiling) {
        resident_counts.insert(resolved);
      }
    }
    for (const int resident_count : resident_counts) {
      const int grid_blocks = resident_count * properties.multiProcessorCount;
      bfnew::hip_tool::DeviceBuffer<std::uint64_t> sinks{
          static_cast<std::size_t>(grid_blocks)};
      for (const std::uint32_t barrier_count : options.barrier_counts) {
        const float elapsed_ms = measure_configuration(
            sinks.get(),
            grid_blocks,
            block_size,
            barrier_count,
            options.repetitions);
        const double mean_kernel_us =
            static_cast<double>(elapsed_ms) * 1000.0 /
            static_cast<double>(options.repetitions);
        const double amortized_barrier_us =
            mean_kernel_us / static_cast<double>(barrier_count);
        std::cout << block_size << ',' << occupancy_ceiling << ','
                  << resident_count << ',' << grid_blocks << ','
                  << barrier_count << ',' << options.repetitions << ','
                  << elapsed_ms << ',' << mean_kernel_us << ','
                  << amortized_barrier_us << '\n';
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    run_benchmark(parse_options(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "bfnew_gpu_barrier_benchmark: " << error.what() << '\n';
    return 1;
  }
}
