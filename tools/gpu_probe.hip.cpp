#include "hip_tool_support.hpp"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  bool workload_only{};
  bool include_workload{};
  std::size_t workload_elements{1U << 20U};
  std::uint32_t workload_launches{64U};
  std::string accepted_groups_path;
};

struct Capability {
  std::string device_name;
  std::string architecture;
  int wave_size{};
  int scheduling_units{};
  int architectural_compute_units{-1};
  int cooperative_launch_flag{};
  int probe_block_size{256};
  int probe_blocks_per_scheduling_unit{};
  int probe_grid_blocks{};
  bool cooperative_launch_executed{};
  int hip_runtime_version{};
  std::string profiler_version;
};

__global__ void cooperative_probe_kernel(
    std::uint32_t* state,
    const std::uint32_t counted_threads) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const std::uint64_t rank = grid.thread_rank();
  if (rank < counted_threads) {
    atomicAdd(&state[0], 1U);
  }
  grid.sync();
  if (rank == 0U) {
    state[1] = state[0];
  }
  grid.sync();
  if (rank + 1U == grid.size()) {
    state[2] = state[1];
  }
}

__global__ void profile_workload_kernel(
    float* values,
    const std::size_t element_count,
    const std::uint32_t launch_index) {
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = first; index < element_count; index += stride) {
    float value = values[index];
    #pragma unroll
    for (int step = 0; step < 16; ++step) {
      const float increment =
          static_cast<float>((launch_index + static_cast<std::uint32_t>(step)) & 7U) *
          0.0001220703125F;
      value = fmaf(value, 1.00000011920928955078125F, increment);
      value = value - floorf(value * 0.000244140625F) * 4096.0F;
    }
    values[index] = value;
  }
}

[[nodiscard]] std::uint64_t parse_unsigned(
    const std::string_view text,
    const std::string_view option_name) {
  std::size_t consumed = 0U;
  unsigned long long value = 0U;
  try {
    value = std::stoull(std::string{text}, &consumed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument{std::string{option_name} + " requires an integer"};
  }
  if (consumed != text.size()) {
    throw std::invalid_argument{std::string{option_name} + " requires an integer"};
  }
  return static_cast<std::uint64_t>(value);
}

[[nodiscard]] Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--workload-only") {
      options.workload_only = true;
      options.include_workload = true;
    } else if (argument == "--profile-workload") {
      options.include_workload = true;
    } else if (argument == "--workload-elements" ||
               argument == "--workload-launches" ||
               argument == "--accepted-pmc-groups") {
      if (++index >= argc) {
        throw std::invalid_argument{std::string{argument} + " requires a value"};
      }
      const std::string_view value{argv[index]};
      if (argument == "--workload-elements") {
        const std::uint64_t parsed = parse_unsigned(value, argument);
        if (parsed == 0U || parsed > std::numeric_limits<std::size_t>::max()) {
          throw std::invalid_argument{"--workload-elements is outside the valid range"};
        }
        options.workload_elements = static_cast<std::size_t>(parsed);
      } else if (argument == "--workload-launches") {
        const std::uint64_t parsed = parse_unsigned(value, argument);
        if (parsed == 0U || parsed > std::numeric_limits<std::uint32_t>::max()) {
          throw std::invalid_argument{"--workload-launches is outside the valid range"};
        }
        options.workload_launches = static_cast<std::uint32_t>(parsed);
      } else {
        options.accepted_groups_path = value;
      }
    } else if (argument == "--help") {
      std::cout
          << "usage: bfnew_gpu_probe [--profile-workload|--workload-only] "
             "[--workload-elements N] [--workload-launches N] "
             "[--accepted-pmc-groups FILE]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument{"unknown option: " + std::string{argument}};
    }
  }
  return options;
}

[[nodiscard]] std::string trim_line(std::string value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

[[nodiscard]] std::string profiler_version() {
  std::array<char, 512> buffer{};
  std::string version;
  FILE* pipe = popen("rocprofv3 --version 2>&1", "r");
  if (pipe == nullptr) {
    return {};
  }
  if (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    version = trim_line(buffer.data());
  }
  static_cast<void>(pclose(pipe));
  return version;
}

[[nodiscard]] std::string json_escape(const std::string_view input) {
  std::string output;
  output.reserve(input.size());
  for (const char character : input) {
    switch (character) {
      case '\\': output += "\\\\"; break;
      case '"': output += "\\\""; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default: output += character; break;
    }
  }
  return output;
}

[[nodiscard]] std::vector<std::string> read_accepted_groups(
    const std::string& path) {
  if (path.empty()) {
    return {};
  }
  std::ifstream input{path};
  if (!input) {
    throw std::runtime_error{"cannot read accepted PMC groups from " + path};
  }
  std::vector<std::string> groups;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      groups.push_back(line);
    }
  }
  return groups;
}

[[nodiscard]] Capability collect_capability() {
  int device = 0;
  BFNEW_HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t properties{};
  BFNEW_HIP_CHECK(hipGetDeviceProperties(&properties, device));

  Capability result;
  result.device_name = properties.name;
  result.architecture = properties.gcnArchName;
  result.wave_size = properties.warpSize;
  result.scheduling_units = properties.multiProcessorCount;
  result.cooperative_launch_flag = properties.cooperativeLaunch;
  BFNEW_HIP_CHECK(hipRuntimeGetVersion(&result.hip_runtime_version));
  result.profiler_version = profiler_version();

#if defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7
  int physical_compute_units = 0;
  const hipError_t physical_status = hipDeviceGetAttribute(
      &physical_compute_units,
      hipDeviceAttributePhysicalMultiProcessorCount,
      device);
  if (physical_status == hipSuccess) {
    result.architectural_compute_units = physical_compute_units;
  } else {
    static_cast<void>(hipGetLastError());
  }
#endif

  if (result.cooperative_launch_flag == 0) {
    return result;
  }

  BFNEW_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &result.probe_blocks_per_scheduling_unit,
      cooperative_probe_kernel,
      result.probe_block_size,
      0U));
  result.probe_grid_blocks =
      result.probe_blocks_per_scheduling_unit * result.scheduling_units;
  if (result.probe_grid_blocks <= 0) {
    throw std::runtime_error{"probe kernel reported no legal cooperative grid"};
  }

  bfnew::hip_tool::DeviceBuffer<std::uint32_t> state{3U};
  BFNEW_HIP_CHECK(hipMemset(state.get(), 0, state.size() * sizeof(std::uint32_t)));
  const std::uint64_t total_threads =
      static_cast<std::uint64_t>(result.probe_grid_blocks) *
      static_cast<std::uint64_t>(result.probe_block_size);
  std::uint32_t counted_threads = static_cast<std::uint32_t>(std::min(
      total_threads,
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())));
  std::uint32_t* state_pointer = state.get();
  void* arguments[] = {&state_pointer, &counted_threads};
  BFNEW_HIP_CHECK(hipLaunchCooperativeKernel(
      reinterpret_cast<void*>(cooperative_probe_kernel),
      dim3(static_cast<unsigned int>(result.probe_grid_blocks)),
      dim3(static_cast<unsigned int>(result.probe_block_size)),
      arguments,
      0U,
      nullptr));
  BFNEW_HIP_CHECK(hipDeviceSynchronize());

  std::array<std::uint32_t, 3> host_state{};
  BFNEW_HIP_CHECK(hipMemcpy(
      host_state.data(),
      state.get(),
      host_state.size() * sizeof(std::uint32_t),
      hipMemcpyDeviceToHost));
  result.cooperative_launch_executed =
      host_state[0] == counted_threads && host_state[1] == counted_threads &&
      host_state[2] == counted_threads;
  if (!result.cooperative_launch_executed) {
    throw std::runtime_error{"cooperative probe completed with invalid barrier state"};
  }
  return result;
}

[[nodiscard]] double run_profile_workload(
    const std::size_t element_count,
    const std::uint32_t launch_count) {
  std::vector<float> initial(element_count);
  for (std::size_t index = 0U; index < initial.size(); ++index) {
    initial[index] = static_cast<float>(index & 1023U) * 0.125F;
  }
  bfnew::hip_tool::DeviceBuffer<float> values{element_count};
  BFNEW_HIP_CHECK(hipMemcpy(
      values.get(),
      initial.data(),
      initial.size() * sizeof(float),
      hipMemcpyHostToDevice));

  constexpr unsigned int block_size = 256U;
  const std::size_t desired_blocks =
      (element_count + block_size - 1U) / block_size;
  const unsigned int grid_size = static_cast<unsigned int>(
      std::min<std::size_t>(desired_blocks, 1024U));
  for (std::uint32_t launch = 0U; launch < launch_count; ++launch) {
    hipLaunchKernelGGL(
        profile_workload_kernel,
        dim3(grid_size),
        dim3(block_size),
        0U,
        nullptr,
        values.get(),
        element_count,
        launch);
    BFNEW_HIP_CHECK(hipGetLastError());
  }
  BFNEW_HIP_CHECK(hipDeviceSynchronize());

  const std::size_t sample_count = std::min<std::size_t>(element_count, 32U);
  std::vector<float> sample(sample_count);
  BFNEW_HIP_CHECK(hipMemcpy(
      sample.data(),
      values.get(),
      sample.size() * sizeof(float),
      hipMemcpyDeviceToHost));
  double checksum = 0.0;
  for (const float value : sample) {
    if (!std::isfinite(value)) {
      throw std::runtime_error{"profile workload produced a nonfinite value"};
    }
    checksum += value;
  }
  return checksum;
}

void print_capability_json(
    const Capability& capability,
    const std::vector<std::string>& accepted_groups,
    const Options& options,
    const double workload_checksum) {
  std::cout << "{\n"
            << "  \"device_name\": \"" << json_escape(capability.device_name)
            << "\",\n"
            << "  \"target_architecture\": \""
            << json_escape(capability.architecture) << "\",\n"
            << "  \"wave_size\": " << capability.wave_size << ",\n"
            << "  \"wgp_or_multiprocessor_count\": "
            << capability.scheduling_units << ",\n"
            << "  \"architectural_cu_count\": ";
  if (capability.architectural_compute_units >= 0) {
    std::cout << capability.architectural_compute_units;
  } else {
    std::cout << "null";
  }
  std::cout << ",\n"
            << "  \"cooperative_launch_flag\": "
            << capability.cooperative_launch_flag << ",\n"
            << "  \"cooperative_launch_executed\": "
            << (capability.cooperative_launch_executed ? "true" : "false") << ",\n"
            << "  \"probe_block_size\": " << capability.probe_block_size << ",\n"
            << "  \"probe_blocks_per_wgp_or_multiprocessor\": "
            << capability.probe_blocks_per_scheduling_unit << ",\n"
            << "  \"probe_grid_blocks\": " << capability.probe_grid_blocks << ",\n"
            << "  \"hip_runtime_version_integer\": "
            << capability.hip_runtime_version << ",\n"
            << "  \"rocprofiler_version\": ";
  if (capability.profiler_version.empty()) {
    std::cout << "null";
  } else {
    std::cout << '"' << json_escape(capability.profiler_version) << '"';
  }
  std::cout << ",\n  \"accepted_pmc_groups\": [";
  for (std::size_t index = 0U; index < accepted_groups.size(); ++index) {
    std::cout << (index == 0U ? "\n    \"" : ",\n    \"")
              << json_escape(accepted_groups[index]) << '"';
  }
  if (!accepted_groups.empty()) {
    std::cout << '\n';
  }
  std::cout << "  ],\n"
            << "  \"profile_workload_included\": "
            << (options.include_workload ? "true" : "false");
  if (options.include_workload) {
    std::cout << ",\n  \"profile_workload_elements\": "
              << options.workload_elements
              << ",\n  \"profile_workload_launches\": "
              << options.workload_launches
              << ",\n  \"profile_workload_checksum\": " << workload_checksum;
  }
  std::cout << "\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.workload_only) {
      const double checksum = run_profile_workload(
          options.workload_elements, options.workload_launches);
      std::cout << "profile_workload_elements=" << options.workload_elements
                << " profile_workload_launches=" << options.workload_launches
                << " checksum=" << checksum << '\n';
      return 0;
    }

    const Capability capability = collect_capability();
    double checksum = 0.0;
    if (options.include_workload) {
      checksum = run_profile_workload(
          options.workload_elements, options.workload_launches);
    }
    print_capability_json(
        capability,
        read_accepted_groups(options.accepted_groups_path),
        options,
        checksum);
    return capability.cooperative_launch_executed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "bfnew_gpu_probe: " << error.what() << '\n';
    return 1;
  }
}
