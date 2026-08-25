cmake_minimum_required(VERSION 3.20)

project(
  bfnew
  VERSION 0.1.0
  DESCRIPTION "Independently implemented weighted shortest-path prototype"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(
  BFNEW_ENABLE_HIP
  "Build the optional HIP capability tools, resident infrastructure, and SSSP engines"
  OFF
)
option(
  BFNEW_ENABLE_FPGAIF
  "Build the Phase 7 FPGA Interchange import and workload tools"
  OFF
)
set(
  BFNEW_HIP_ARCHITECTURES
  "gfx1151"
  CACHE STRING
  "HIP GPU architectures used when BFNEW_ENABLE_HIP is ON"
)

if(BFNEW_ENABLE_HIP)
  if(CMAKE_VERSION VERSION_LESS 3.21)
    message(
      FATAL_ERROR
      "BFNEW_ENABLE_HIP requires CMake 3.21 or newer for HIP language support"
    )
  endif()

  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(
      CMAKE_HIP_ARCHITECTURES
      "${BFNEW_HIP_ARCHITECTURES}"
      CACHE STRING
      "HIP target architectures"
    )
  endif()

  include(CheckLanguage)
  check_language(HIP)
  if(NOT CMAKE_HIP_COMPILER)
    message(
      FATAL_ERROR
      "BFNEW_ENABLE_HIP=ON was requested, but no HIP compiler was found. "
      "Install ROCm/HIP or configure this project with BFNEW_ENABLE_HIP=OFF."
    )
  endif()

  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS OFF)
endif()

set(
  BFNEW_FPGA24_DATA_ROOT
  "/Users/amber_bajaj/Desktop/RIPS/fpga24_routing_contest"
  CACHE PATH
  "Read-only root containing FPGA24 benchmark data"
)
set(
  BFNEW_FPGA24_SCHEMA_ROOT
  "/Users/amber_bajaj/Desktop/RIPS/fpga24_routing_contest/fpga-interchange-schema/interchange"
  CACHE PATH
  "Read-only root containing FPGA Interchange schemas"
)

include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
  "
    #include <concepts>
    #include <span>

    template <std::integral T>
    constexpr T identity(T value) { return value; }

    int main() {
      int values[] = {identity(1)};
      std::span<int> view(values);
      return view.front() == 1 ? 0 : 1;
    }
  "
  BFNEW_COMPILER_SUPPORTS_CXX20
)

if(NOT BFNEW_COMPILER_SUPPORTS_CXX20)
  message(FATAL_ERROR "bfnew requires a compiler and standard library with C++20 support")
endif()

add_library(bfnew_project_warnings INTERFACE)
if(MSVC)
  target_compile_options(bfnew_project_warnings INTERFACE /W4 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(
    bfnew_project_warnings
    INTERFACE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wsign-conversion
  )
endif()

add_library(
  bfnew
  STATIC
    src/batched_expansion.cpp
    src/compact_paths.cpp
    src/no_congestion_pipeline.cpp
    src/batched_dense_chaotic_push.cpp
    src/batched_frontier_push.cpp
    src/batched_jacobi_pull.cpp
    src/batch_layout.cpp
    src/batch_plan.cpp
    src/batch_workspace.cpp
    src/device_layout.cpp
    src/dense_chaotic_push.cpp
    src/engine_shootout.cpp
    src/final_audit.cpp
    src/frontier_push.cpp
    src/gpu_api.cpp
    src/graph_builder.cpp
    src/jacobi_pull.cpp
    src/cpu/reference_bellman_ford.cpp
    src/cpu/reference_dijkstra.cpp
    src/path_reconstruct.cpp
    src/query.cpp
    src/reorder.cpp
    src/selected_region_index.cpp
    src/tiled_graph.cpp
    src/workspace.cpp
)
add_library(bfnew::bfnew ALIAS bfnew)
target_compile_features(bfnew PUBLIC cxx_std_20)
target_include_directories(
  bfnew
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(bfnew PRIVATE bfnew_project_warnings)

add_executable(bfnew_shootout_report tools/shootout_report.cpp)
target_compile_features(bfnew_shootout_report PRIVATE cxx_std_20)
target_link_libraries(
  bfnew_shootout_report
  PRIVATE
    bfnew::bfnew
    bfnew_project_warnings
)

add_executable(
  bfnew_shootout_profile_import
  tools/shootout_profile_import.cpp
)
target_compile_features(bfnew_shootout_profile_import PRIVATE cxx_std_20)
target_link_libraries(
  bfnew_shootout_profile_import
  PRIVATE
    bfnew::bfnew
    bfnew_project_warnings
)

if(BFNEW_ENABLE_FPGAIF)
  find_package(CapnProto CONFIG REQUIRED)

  set(_bfnew_fpgaif_schemas
    "${BFNEW_FPGA24_SCHEMA_ROOT}/References.capnp"
    "${BFNEW_FPGA24_SCHEMA_ROOT}/LogicalNetlist.capnp"
    "${BFNEW_FPGA24_SCHEMA_ROOT}/DeviceResources.capnp"
    "${BFNEW_FPGA24_SCHEMA_ROOT}/PhysicalNetlist.capnp"
  )
  set(_bfnew_fpgaif_required_schemas
    ${_bfnew_fpgaif_schemas}
    "${BFNEW_FPGA24_SCHEMA_ROOT}/capnp/java.capnp"
  )
  foreach(_bfnew_schema IN LISTS _bfnew_fpgaif_required_schemas)
    if(NOT EXISTS "${_bfnew_schema}")
      message(FATAL_ERROR "Required FPGA Interchange schema is missing: ${_bfnew_schema}")
    endif()
  endforeach()

  set(CAPNPC_SRC_PREFIX "${BFNEW_FPGA24_SCHEMA_ROOT}")
  set(CAPNPC_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(CAPNPC_IMPORT_DIRS "${BFNEW_FPGA24_SCHEMA_ROOT}")
  file(MAKE_DIRECTORY "${CAPNPC_OUTPUT_DIR}")
  capnp_generate_cpp(
    BFNEW_FPGAIF_SCHEMA_SOURCES
    BFNEW_FPGAIF_SCHEMA_HEADERS
    ${_bfnew_fpgaif_schemas}
  )

  add_library(
    bfnew_fpgaif
    STATIC
      ${BFNEW_FPGAIF_SCHEMA_SOURCES}
      src/io/fpga_interchange.cpp
      src/io/workload_artifact.cpp
  )
  add_library(bfnew::fpgaif ALIAS bfnew_fpgaif)
  target_compile_features(bfnew_fpgaif PUBLIC cxx_std_20)
  target_include_directories(
    bfnew_fpgaif
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>"
  )
  target_link_libraries(
    bfnew_fpgaif
    PUBLIC
      bfnew::bfnew
      CapnProto::capnp
    PRIVATE
      CapnProto::kj-gzip
      bfnew_project_warnings
  )

  add_executable(bfnew_build_fpga_workload tools/build_fpga_workload.cpp)
  target_compile_features(bfnew_build_fpga_workload PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_build_fpga_workload
    PRIVATE
      bfnew::fpgaif
      bfnew_project_warnings
  )
endif()

if(BFNEW_ENABLE_HIP)
  add_library(
    bfnew_hip
    STATIC
      src/hip/resident_graph.hip.cpp
      src/hip/runtime.hip.cpp
      src/hip/workspace.hip.cpp
      src/hip/batched_expansion.hip.cpp
      src/hip/compact_path_results.hip.cpp
      src/hip/batched_dense_workspace.hip.cpp
      src/hip/batched_dense_chaotic_push.hip.cpp
      src/hip/batched_frontier_workspace.hip.cpp
      src/hip/batched_frontier_push.hip.cpp
      src/hip/batched_workspace.hip.cpp
      src/hip/batched_jacobi.hip.cpp
      src/hip/jacobi.hip.cpp
      src/hip/dense_chaotic_push.hip.cpp
      src/hip/engine_shootout.hip.cpp
      src/hip/frontier_push.hip.cpp
  )
  add_library(bfnew::hip ALIAS bfnew_hip)
  set_source_files_properties(
    src/hip/resident_graph.hip.cpp
    src/hip/runtime.hip.cpp
    src/hip/workspace.hip.cpp
    src/hip/batched_expansion.hip.cpp
    src/hip/compact_path_results.hip.cpp
    src/hip/batched_dense_workspace.hip.cpp
    src/hip/batched_dense_chaotic_push.hip.cpp
    src/hip/batched_frontier_workspace.hip.cpp
    src/hip/batched_frontier_push.hip.cpp
    src/hip/batched_workspace.hip.cpp
    src/hip/batched_jacobi.hip.cpp
    src/hip/jacobi.hip.cpp
    src/hip/dense_chaotic_push.hip.cpp
    src/hip/engine_shootout.hip.cpp
    src/hip/frontier_push.hip.cpp
    PROPERTIES LANGUAGE HIP
  )
  target_compile_features(bfnew_hip PUBLIC cxx_std_20)
  target_link_libraries(
    bfnew_hip
    PUBLIC bfnew::bfnew
    PRIVATE bfnew_project_warnings
  )

  add_executable(bfnew_gpu_probe tools/gpu_probe.hip.cpp)
  set_source_files_properties(tools/gpu_probe.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_features(bfnew_gpu_probe PRIVATE cxx_std_20)
  target_link_libraries(bfnew_gpu_probe PRIVATE bfnew_project_warnings)

  add_executable(bfnew_gpu_barrier_benchmark tools/gpu_barrier_benchmark.hip.cpp)
  set_source_files_properties(
    tools/gpu_barrier_benchmark.hip.cpp
    PROPERTIES LANGUAGE HIP
  )
  target_compile_features(bfnew_gpu_barrier_benchmark PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_gpu_barrier_benchmark
    PRIVATE bfnew_project_warnings
  )

  if(BFNEW_ENABLE_FPGAIF)
    add_executable(bfnew_gpu_shootout tools/gpu_shootout.hip.cpp)
    set_source_files_properties(
      tools/gpu_shootout.hip.cpp
      PROPERTIES LANGUAGE HIP
    )
    target_compile_features(bfnew_gpu_shootout PRIVATE cxx_std_20)
    target_link_libraries(
      bfnew_gpu_shootout
      PRIVATE
        bfnew::hip
        bfnew::fpgaif
        bfnew_project_warnings
    )

    add_executable(
      bfnew_gpu_batched_expansion
      tools/gpu_batched_expansion.hip.cpp
    )
    set_source_files_properties(
      tools/gpu_batched_expansion.hip.cpp
      PROPERTIES LANGUAGE HIP
    )
    target_compile_features(
      bfnew_gpu_batched_expansion
      PRIVATE cxx_std_20
    )
    target_link_libraries(
      bfnew_gpu_batched_expansion
      PRIVATE
        bfnew::hip
        bfnew::fpgaif
        bfnew_project_warnings
    )
  endif()
endif()

include(CTest)

if(BUILD_TESTING)
  add_executable(bfnew_build_test tests/build_test.cpp)
  target_compile_features(bfnew_build_test PRIVATE cxx_std_20)
  target_link_libraries(bfnew_build_test PRIVATE bfnew_project_warnings)

  add_test(NAME bfnew.build COMMAND bfnew_build_test)

  add_executable(bfnew_graph_test tests/graph_test.cpp)
  target_compile_features(bfnew_graph_test PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_graph_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.graph COMMAND bfnew_graph_test)

  add_executable(bfnew_reorder_test tests/reorder_test.cpp)
  target_compile_features(bfnew_reorder_test PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_reorder_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.reorder COMMAND bfnew_reorder_test)

  add_executable(bfnew_tile_test tests/tile_test.cpp)
  target_compile_features(bfnew_tile_test PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_tile_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.tile COMMAND bfnew_tile_test)

  add_executable(bfnew_sssp_test tests/sssp_test.cpp)
  target_compile_features(bfnew_sssp_test PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_sssp_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.sssp COMMAND bfnew_sssp_test)

  add_executable(bfnew_query_test tests/query_test.cpp)
  target_compile_features(bfnew_query_test PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_query_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.query COMMAND bfnew_query_test)

  add_executable(bfnew_workspace_test tests/workspace_test.cpp)
  target_compile_features(bfnew_workspace_test PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_workspace_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.workspace COMMAND bfnew_workspace_test)

  add_executable(bfnew_device_layout_test tests/device_layout_test.cpp)
  target_compile_features(bfnew_device_layout_test PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_device_layout_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.device_layout COMMAND bfnew_device_layout_test)

  add_executable(bfnew_jacobi_pull_test tests/jacobi_pull_test.cpp)
  target_compile_features(bfnew_jacobi_pull_test PRIVATE cxx_std_20)
  target_compile_definitions(
    bfnew_jacobi_pull_test
    PRIVATE
      BFNEW_PHASE9_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/jacobi.hip.cpp"
  )
  target_link_libraries(
    bfnew_jacobi_pull_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.jacobi_pull COMMAND bfnew_jacobi_pull_test)

  add_executable(
    bfnew_dense_chaotic_push_test
    tests/dense_chaotic_push_test.cpp
  )
  target_compile_features(bfnew_dense_chaotic_push_test PRIVATE cxx_std_20)
  target_compile_definitions(
    bfnew_dense_chaotic_push_test
    PRIVATE
      BFNEW_PHASE10_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/dense_chaotic_push.hip.cpp"
  )
  target_link_libraries(
    bfnew_dense_chaotic_push_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(
    NAME bfnew.dense_chaotic_push
    COMMAND bfnew_dense_chaotic_push_test
  )

  add_executable(bfnew_frontier_push_test tests/frontier_push_test.cpp)
  target_compile_features(bfnew_frontier_push_test PRIVATE cxx_std_20)
  target_compile_definitions(
    bfnew_frontier_push_test
    PRIVATE
      BFNEW_PHASE11_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/frontier_push.hip.cpp"
  )
  target_link_libraries(
    bfnew_frontier_push_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.frontier_push COMMAND bfnew_frontier_push_test)

  add_executable(bfnew_shootout_test tests/engine_shootout_test.cpp)
  target_compile_features(bfnew_shootout_test PRIVATE cxx_std_20)
  target_compile_definitions(
    bfnew_shootout_test
    PRIVATE
      BFNEW_PHASE12_JACOBI_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/jacobi.hip.cpp"
      BFNEW_PHASE12_DENSE_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/dense_chaotic_push.hip.cpp"
      BFNEW_PHASE12_FRONTIER_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/frontier_push.hip.cpp"
      BFNEW_PHASE12_EXECUTOR_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/engine_shootout.hip.cpp"
      BFNEW_PHASE12_RUNNER_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tools/gpu_shootout.hip.cpp"
  )
  target_link_libraries(
    bfnew_shootout_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.shootout COMMAND bfnew_shootout_test)

  add_executable(bfnew_batch_plan_test tests/batch_plan_test.cpp)
  target_compile_features(bfnew_batch_plan_test PRIVATE cxx_std_20)
  target_compile_definitions(
    bfnew_batch_plan_test
    PRIVATE
      BFNEW_PHASE13_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(
    bfnew_batch_plan_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.batch_plan COMMAND bfnew_batch_plan_test)

  add_executable(
    bfnew_batch_workspace_test
    tests/batch_workspace_test.cpp
  )
  target_compile_features(bfnew_batch_workspace_test PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_batch_workspace_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.batch_workspace COMMAND bfnew_batch_workspace_test)

  add_executable(
    bfnew_batched_jacobi_pull_test
    tests/batched_jacobi_pull_test.cpp
  )
  target_compile_features(bfnew_batched_jacobi_pull_test PRIVATE cxx_std_20)
  target_compile_definitions(
    bfnew_batched_jacobi_pull_test
    PRIVATE
      BFNEW_PHASE14_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/batched_jacobi.hip.cpp"
      BFNEW_PHASE14_WORKSPACE_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/batched_workspace.hip.cpp"
  )
  target_link_libraries(
    bfnew_batched_jacobi_pull_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(
    NAME bfnew.batched_jacobi_pull
    COMMAND bfnew_batched_jacobi_pull_test
  )

  add_executable(
    bfnew_batched_dense_chaotic_push_test
    tests/batched_dense_chaotic_push_test.cpp
  )
  target_compile_features(
    bfnew_batched_dense_chaotic_push_test
    PRIVATE cxx_std_20
  )
  target_compile_definitions(
    bfnew_batched_dense_chaotic_push_test
    PRIVATE
      BFNEW_PHASE15_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/batched_dense_chaotic_push.hip.cpp"
      BFNEW_PHASE15_WORKSPACE_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/batched_dense_workspace.hip.cpp"
  )
  target_link_libraries(
    bfnew_batched_dense_chaotic_push_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(
    NAME bfnew.batched_dense_chaotic_push
    COMMAND bfnew_batched_dense_chaotic_push_test
  )

  add_executable(
    bfnew_batched_frontier_push_test
    tests/batched_frontier_push_test.cpp
  )
  target_compile_features(
    bfnew_batched_frontier_push_test
    PRIVATE cxx_std_20
  )
  target_compile_definitions(
    bfnew_batched_frontier_push_test
    PRIVATE
      BFNEW_PHASE16_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/batched_frontier_push.hip.cpp"
      BFNEW_PHASE16_WORKSPACE_HIP_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/hip/batched_frontier_workspace.hip.cpp"
  )
  target_link_libraries(
    bfnew_batched_frontier_push_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(
    NAME bfnew.batched_frontier_push
    COMMAND bfnew_batched_frontier_push_test
  )

  add_executable(
    bfnew_batched_expansion_test
    tests/batched_expansion_test.cpp
  )
  target_compile_features(
    bfnew_batched_expansion_test
    PRIVATE cxx_std_20
  )
  target_link_libraries(
    bfnew_batched_expansion_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(
    NAME bfnew.batched_expansion
    COMMAND bfnew_batched_expansion_test
  )

  add_executable(
    bfnew_compact_paths_test
    tests/compact_paths_test.cpp
  )
  target_compile_features(bfnew_compact_paths_test PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_compact_paths_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.compact_paths COMMAND bfnew_compact_paths_test)

  add_executable(
    bfnew_final_audit_test
    tests/final_audit_test.cpp
  )
  target_compile_features(bfnew_final_audit_test PRIVATE cxx_std_20)
  target_link_libraries(
    bfnew_final_audit_test
    PRIVATE
      bfnew::bfnew
      bfnew_project_warnings
  )

  add_test(NAME bfnew.final_audit COMMAND bfnew_final_audit_test)

  if(BFNEW_ENABLE_FPGAIF)
    add_executable(bfnew_fpgaif_test tests/fpga_interchange_test.cpp)
    target_compile_features(bfnew_fpgaif_test PRIVATE cxx_std_20)
    target_link_libraries(
      bfnew_fpgaif_test
      PRIVATE
        bfnew::fpgaif
        bfnew_project_warnings
    )
    add_test(NAME bfnew.fpgaif COMMAND bfnew_fpgaif_test)
  endif()

  if(BFNEW_ENABLE_HIP)
    add_executable(
      bfnew_device_transfer_test
      tests/device_transfer_test.hip.cpp
    )
    set_source_files_properties(
      tests/device_transfer_test.hip.cpp
      PROPERTIES LANGUAGE HIP
    )
    target_compile_features(bfnew_device_transfer_test PRIVATE cxx_std_20)
    target_link_libraries(
      bfnew_device_transfer_test
      PRIVATE
        bfnew::hip
        bfnew_project_warnings
    )
    add_test(NAME bfnew.device_transfer COMMAND bfnew_device_transfer_test)

    add_executable(bfnew_jacobi_test tests/jacobi_test.hip.cpp)
    set_source_files_properties(
      tests/jacobi_test.hip.cpp
      PROPERTIES LANGUAGE HIP
    )
    target_compile_features(bfnew_jacobi_test PRIVATE cxx_std_20)
    target_link_libraries(
      bfnew_jacobi_test
      PRIVATE
        bfnew::hip
        bfnew_project_warnings
    )
    add_test(NAME bfnew.jacobi COMMAND bfnew_jacobi_test)

    add_executable(
      bfnew_dense_chaotic_push_gpu_test
      tests/dense_chaotic_push_test.hip.cpp
    )
    set_source_files_properties(
      tests/dense_chaotic_push_test.hip.cpp
      PROPERTIES LANGUAGE HIP
    )
    target_compile_features(
      bfnew_dense_chaotic_push_gpu_test
      PRIVATE cxx_std_20
    )
    target_link_libraries(
      bfnew_dense_chaotic_push_gpu_test
      PRIVATE
        bfnew::hip
        bfnew_project_warnings
    )
    add_test(
      NAME bfnew.dense_chaotic_push_gpu
      COMMAND bfnew_dense_chaotic_push_gpu_test
    )

    add_executable(
      bfnew_frontier_push_gpu_test
      tests/frontier_push_test.hip.cpp
    )
    set_source_files_properties(
      tests/frontier_push_test.hip.cpp
      PROPERTIES LANGUAGE HIP
    )
    target_compile_features(
      bfnew_frontier_push_gpu_test
      PRIVATE cxx_std_20
    )
    target_link_libraries(
      bfnew_frontier_push_gpu_test
      PRIVATE
        bfnew::hip
        bfnew_project_warnings
    )
    add_test(
      NAME bfnew.frontier_push_gpu
      COMMAND bfnew_frontier_push_gpu_test
    )
    set_tests_properties(bfnew.frontier_push_gpu PROPERTIES TIMEOUT 180)

    add_executable(
      bfnew_batched_jacobi_hip_test
      tests/batched_jacobi_test.hip.cpp
    )
    set_source_files_properties(
      tests/batched_jacobi_test.hip.cpp
      PROPERTIES LANGUAGE HIP
    )
    target_compile_features(
      bfnew_batched_jacobi_hip_test
      PRIVATE cxx_std_20
    )
    target_link_libraries(
      bfnew_batched_jacobi_hip_test
      PRIVATE
        bfnew::hip
        bfnew_project_warnings
    )
    add_test(
      NAME bfnew.batched_jacobi
      COMMAND bfnew_batched_jacobi_hip_test
    )
    set_tests_properties(bfnew.batched_jacobi PROPERTIES TIMEOUT 240)

    add_executable(
      bfnew_batched_dense_chaotic_push_hip_test
      tests/batched_dense_chaotic_push_test.hip.cpp
    )
    set_source_files_properties(
      tests/batched_dense_chaotic_push_test.hip.cpp
      PROPERTIES LANGUAGE HIP
    )
    target_compile_features(
      bfnew_batched_dense_chaotic_push_hip_test
      PRIVATE cxx_std_20
    )
    target_link_libraries(
      bfnew_batched_dense_chaotic_push_hip_test
      PRIVATE
        bfnew::hip
        bfnew_project_warnings
    )
    add_test(
      NAME bfnew.batched_dense_chaotic_push_gpu
      COMMAND bfnew_batched_dense_chaotic_push_hip_test
    )
    set_tests_properties(
      bfnew.batched_dense_chaotic_push_gpu
      PROPERTIES TIMEOUT 240
    )

    add_executable(
      bfnew_batched_frontier_push_hip_test
      tests/batched_frontier_push_test.hip.cpp
    )
    set_source_files_properties(
      tests/batched_frontier_push_test.hip.cpp
      PROPERTIES LANGUAGE HIP
    )
    target_compile_features(
      bfnew_batched_frontier_push_hip_test
      PRIVATE cxx_std_20
    )
    target_link_libraries(
      bfnew_batched_frontier_push_hip_test
      PRIVATE
        bfnew::hip
        bfnew_project_warnings
    )
    add_test(
      NAME bfnew.batched_frontier_push_gpu
      COMMAND bfnew_batched_frontier_push_hip_test
    )
    set_tests_properties(
      bfnew.batched_frontier_push_gpu
      PROPERTIES TIMEOUT 240
    )

    add_executable(
      bfnew_batched_expansion_hip_test
      tests/batched_expansion_test.hip.cpp
    )
    set_source_files_properties(
      tests/batched_expansion_test.hip.cpp
      PROPERTIES LANGUAGE HIP
    )
    target_compile_features(
      bfnew_batched_expansion_hip_test
      PRIVATE cxx_std_20
    )
    target_link_libraries(
      bfnew_batched_expansion_hip_test
      PRIVATE
        bfnew::hip
        bfnew_project_warnings
    )
    add_test(
      NAME bfnew.batched_expansion_gpu
      COMMAND bfnew_batched_expansion_hip_test
    )
    set_tests_properties(
      bfnew.batched_expansion_gpu
      PROPERTIES TIMEOUT 300
    )

    add_executable(
      bfnew_compact_path_results_test
      tests/compact_path_results_test.hip.cpp
    )
    set_source_files_properties(
      tests/compact_path_results_test.hip.cpp
      PROPERTIES LANGUAGE HIP
    )
    target_compile_features(
      bfnew_compact_path_results_test
      PRIVATE cxx_std_20
    )
    target_link_libraries(
      bfnew_compact_path_results_test
      PRIVATE
        bfnew::hip
        bfnew_project_warnings
    )
    add_test(
      NAME bfnew.compact_path_results
      COMMAND bfnew_compact_path_results_test
    )
    set_tests_properties(
      bfnew.compact_path_results
      PROPERTIES TIMEOUT 300
    )

    add_test(NAME bfnew.gpu_capability COMMAND bfnew_gpu_probe)
    add_test(
      NAME bfnew.gpu_barrier_smoke
      COMMAND
        bfnew_gpu_barrier_benchmark
        --block-sizes 128
        --blocks-per-wgp 1
        --barriers 1
        --repetitions 1
    )
  endif()
endif()
