#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "usage: $0 [HIP_BUILD_DIRECTORY] [OUTPUT_DIRECTORY]" >&2
  exit 2
fi

build_directory="${1:-build-hip}"
output_directory="${2:-${build_directory}/phase6-profile}"
probe="${build_directory}/bfnew_gpu_probe"
barrier_benchmark="${build_directory}/bfnew_gpu_barrier_benchmark"

for command_name in rocprofv3 rocprofv3-avail; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "required command not found: ${command_name}" >&2
    exit 1
  fi
done
for executable in "${probe}" "${barrier_benchmark}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "required Phase 6 executable not found: ${executable}" >&2
    exit 1
  fi
done

mkdir -p "${output_directory}" "${output_directory}/pmc-check"
accepted_groups="${output_directory}/accepted-pmc-groups.txt"
rejected_counters="${output_directory}/rejected-pmc-counters.txt"
: >"${accepted_groups}"
: >"${rejected_counters}"

rocprofv3 --version >"${output_directory}/rocprofv3-version.txt" 2>&1
rocprofv3-avail -d 0 list --pmc >"${output_directory}/pmc-list.txt" 2>&1
rocprofv3-avail -d 0 info --pmc >"${output_directory}/pmc-info.txt" 2>&1

check_group() {
  local label="$1"
  shift
  local log="${output_directory}/pmc-check/${label}.txt"
  local result
  if result="$(rocprofv3-avail -d 0 pmc-check "$@" 2>&1)" &&
      grep -q "can be collected together" <<<"${result}"; then
    printf '%s\n' "${result}" >"${log}"
    printf '%s\t%s\n' "${label}" "$*" >>"${accepted_groups}"
    return
  fi

  printf '%s\n' "${result}" >"${log}"
  if [[ $# -eq 1 ]]; then
    printf '%s\t%s\n' "${label}" "$1" >>"${rejected_counters}"
    return
  fi

  local -a counters=("$@")
  local split=$(( ${#counters[@]} / 2 ))
  check_group "${label}-a" "${counters[@]:0:${split}}"
  check_group "${label}-b" "${counters[@]:${split}}"
}

check_group utilization \
  GRBM_COUNT GRBM_GUI_ACTIVE GPUBusy OccupancyPercent \
  MeanOccupancyPerCU MeanOccupancyPerActiveCU MemUnitBusy
check_group instruction-wave \
  SQ_WAVES_sum SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_SALU \
  SQ_INSTS_SMEM SQ_INSTS_FLAT SQ_INSTS_TEX_LOAD SQ_INSTS_TEX_STORE
check_group l2-memory \
  FETCH_SIZE WRITE_SIZE L2CacheHit GL2C_HIT_sum GL2C_MISS_sum \
  GL2C_MC_RDREQ_sum GL2C_MC_WRREQ_sum GL2C_EA_WRREQ_64B_sum \
  GL2C_WRREQ_STALL_max WriteUnitStalled

if [[ ! -s "${accepted_groups}" ]]; then
  echo "no candidate PMC group was accepted" >&2
  exit 1
fi

"${probe}" --accepted-pmc-groups "${accepted_groups}" \
  >"${output_directory}/gpu-capability.json"

read -r -a first_group <"${accepted_groups}"
if [[ ${#first_group[@]} -lt 2 ]]; then
  echo "accepted PMC group record is malformed" >&2
  exit 1
fi
profile_counters=("${first_group[@]:1}")
rocprofv3 \
  --pmc "${profile_counters[@]}" \
  --output-directory "${output_directory}/counter-collection" \
  --output-format csv \
  -- "${probe}" --workload-only --workload-elements 1048576 --workload-launches 64 \
  >"${output_directory}/counter-collection.log" 2>&1

if ! find "${output_directory}/counter-collection" \
    -type f -name '*counter_collection.csv' -size +0c -print -quit |
    grep -q .; then
  echo "rocprofv3 produced no nonempty counter-collection CSV" >&2
  exit 1
fi

rocprofv3 \
  --hip-trace \
  --kernel-trace \
  --memory-copy-trace \
  --output-directory "${output_directory}/trace" \
  --output-format csv \
  -- "${probe}" --workload-only --workload-elements 1048576 --workload-launches 64 \
  >"${output_directory}/trace.log" 2>&1

"${barrier_benchmark}" >"${output_directory}/cooperative-barriers.csv"

echo "Phase 6 profiling evidence written to ${output_directory}"
