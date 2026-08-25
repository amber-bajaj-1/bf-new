#!/usr/bin/env bash

# Bootstrap the inputs needed to build and benchmark bf-new on an AUP host.
#
# By default this creates:
#   $HOME/rips/bf-new
#   $HOME/rips/fpga24_routing_contest
#   $HOME/rips/download-cache
#   $HOME/rips/bfnew-env.sh
#
# Override the defaults when needed, for example:
#   AUP_WORKSPACE_ROOT=/scratch/$USER/rips \
#   BFNEW_REPO_URL=git@github.com:amber-bajaj-1/bf-new.git \
#     bash setup-aup.sh

set -Eeuo pipefail

readonly AUP_WORKSPACE_ROOT="${AUP_WORKSPACE_ROOT:-${HOME:?HOME must identify the current user home}/rips}"
readonly BFNEW_REPO_URL="${BFNEW_REPO_URL:-https://github.com/amber-bajaj-1/bf-new.git}"
readonly BFNEW_REPO_REF="${BFNEW_REPO_REF:-main}"

# Pin source/schema revisions so AUP preprocessing uses the same definitions as
# the local prototype. Benchmark and device assets use the contest's canonical
# release URLs because the upstream project publishes those assets separately
# from its Git source archive.
readonly FPGA24_SOURCE_REVISION="f86cb48769466601b9f8d70ae000fa7bae39b3d8"
readonly FPGA24_SCHEMA_REVISION="c985b4648e66414b250261c1ba4cbe45a2971b1c"
readonly FPGA24_SOURCE_URL="https://github.com/Xilinx/fpga24_routing_contest/archive/${FPGA24_SOURCE_REVISION}.tar.gz"
readonly FPGA24_SCHEMA_URL="https://github.com/chipsalliance/fpga-interchange-schema/archive/${FPGA24_SCHEMA_REVISION}.tar.gz"
readonly FPGA24_BENCHMARKS_URL="https://github.com/Xilinx/fpga24_routing_contest/releases/latest/download/benchmarks.tar.gz"
readonly FPGA24_DEVICE_URL="https://github.com/Xilinx/fpga24_routing_contest/releases/latest/download/xcvu3p.device"

readonly BFNEW_DIR="${AUP_WORKSPACE_ROOT}/bf-new"
readonly FPGA24_DIR="${AUP_WORKSPACE_ROOT}/fpga24_routing_contest"
readonly DOWNLOAD_CACHE="${AUP_WORKSPACE_ROOT}/download-cache"
readonly BFNEW_TOOLS="${BFNEW_TOOLS:-${AUP_WORKSPACE_ROOT}/bfnew-tools}"
readonly BFNEW_ENVIRONMENT="${AUP_WORKSPACE_ROOT}/bfnew-env.sh"
readonly USER_BASHRC="${HOME}/.bashrc"

readonly -a BENCHMARKS=(
  logicnets_jscl
  boom_med_pb
  vtr_mcml
  rosetta_fd
  corundum_25g
  finn_radioml
  vtr_lu64peeng
  corescore_500
  corescore_500_pb
  mlcad_d181_lefttwo3rds
  koios_dla_like_large
  boom_soc
  ispd16_example2
)

log() {
  printf '[bf-new setup] %s\n' "$*"
}

die() {
  printf '[bf-new setup] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    die "Required command is unavailable: $1"
}

download() {
  local url="$1"
  local destination="$2"

  if [[ -s "$destination" ]]; then
    log "Using cached $(basename "$destination")"
    return
  fi

  mkdir -p "$(dirname "$destination")"
  log "Downloading $(basename "$destination")"
  wget \
    --continue \
    --tries=5 \
    --timeout=30 \
    --output-document="${destination}.part" \
    "$url"
  [[ -s "${destination}.part" ]] ||
    die "Download produced an empty file: $url"
  mv -- "${destination}.part" "$destination"
}

validate_tar_paths() {
  local archive="$1"
  local entry

  tar -tzf "$archive" >/dev/null 2>&1 ||
    die "Downloaded file is not a valid gzip-compressed tar archive: $archive"

  while IFS= read -r entry; do
    [[ "$entry" != /* &&
       "$entry" != ".." &&
       "$entry" != ../* &&
       "$entry" != */../* &&
       "$entry" != */.. ]] ||
      die "Unsafe path in archive $archive: $entry"
  done < <(tar -tzf "$archive")
}

single_extracted_directory() {
  local parent="$1"
  local -a directories=()

  while IFS= read -r -d '' directory; do
    directories+=("$directory")
  done < <(find "$parent" -mindepth 1 -maxdepth 1 -type d -print0)

  [[ "${#directories[@]}" -eq 1 ]] ||
    die "Expected exactly one top-level directory below $parent"
  printf '%s\n' "${directories[0]}"
}

clone_bfnew() {
  if [[ -d "$BFNEW_DIR/.git" ]]; then
    log "Using existing bf-new clone at $BFNEW_DIR"
    return
  fi
  [[ ! -e "$BFNEW_DIR" ]] ||
    die "$BFNEW_DIR exists but is not a Git repository; move it aside and rerun setup."

  log "Cloning bf-new at revision $BFNEW_REPO_REF"
  git clone \
    --branch "$BFNEW_REPO_REF" \
    --single-branch \
    "$BFNEW_REPO_URL" \
    "$BFNEW_DIR"
}

contest_data_is_ready() {
  local benchmark
  local schema

  [[ -s "$FPGA24_DIR/xcvu3p.device" ]] || return 1
  for benchmark in "${BENCHMARKS[@]}"; do
    [[ -s "$FPGA24_DIR/${benchmark}_unrouted.phys" ]] || return 1
    [[ -s "$FPGA24_DIR/${benchmark}.netlist" ]] || return 1
  done
  for schema in \
    References.capnp \
    LogicalNetlist.capnp \
    DeviceResources.capnp \
    PhysicalNetlist.capnp; do
    [[ -s "$FPGA24_DIR/fpga-interchange-schema/interchange/$schema" ]] ||
      return 1
  done
}

report_missing_contest_artifacts() {
  local benchmark
  local schema

  [[ -s "$FPGA24_DIR/xcvu3p.device" ]] ||
    printf '  missing: %s\n' "$FPGA24_DIR/xcvu3p.device" >&2
  for benchmark in "${BENCHMARKS[@]}"; do
    [[ -s "$FPGA24_DIR/${benchmark}_unrouted.phys" ]] ||
      printf '  missing: %s\n' \
        "$FPGA24_DIR/${benchmark}_unrouted.phys" >&2
    [[ -s "$FPGA24_DIR/${benchmark}.netlist" ]] ||
      printf '  missing: %s\n' "$FPGA24_DIR/${benchmark}.netlist" >&2
  done
  for schema in \
    References.capnp \
    LogicalNetlist.capnp \
    DeviceResources.capnp \
    PhysicalNetlist.capnp; do
    [[ -s "$FPGA24_DIR/fpga-interchange-schema/interchange/$schema" ]] ||
      printf '  missing: %s\n' \
        "$FPGA24_DIR/fpga-interchange-schema/interchange/$schema" >&2
  done
}

place_archive_artifact_at_root() {
  local root="$1"
  local filename="$2"
  local expected="$root/$filename"
  local -a matches=()

  [[ -s "$expected" ]] && return
  while IFS= read -r -d '' match; do
    matches+=("$match")
  done < <(find "$root" -mindepth 2 -type f -name "$filename" -print0)

  [[ "${#matches[@]}" -eq 1 ]] ||
    die "Expected exactly one $filename in the downloaded benchmark archive."
  mv -- "${matches[0]}" "$expected"
}

prepare_contest_data() {
  if contest_data_is_ready; then
    log "Using existing FPGA24 data at $FPGA24_DIR"
    return
  fi
  if [[ -d "$FPGA24_DIR" ]]; then
    log "Repairing incomplete FPGA24 data at $FPGA24_DIR"
  elif [[ -e "$FPGA24_DIR" ]]; then
    die "$FPGA24_DIR exists but is not a directory."
  fi

  local source_archive="$DOWNLOAD_CACHE/fpga24-source-${FPGA24_SOURCE_REVISION}.tar.gz"
  local schema_archive="$DOWNLOAD_CACHE/fpga-interchange-schema-${FPGA24_SCHEMA_REVISION}.tar.gz"
  local benchmark_archive="$DOWNLOAD_CACHE/fpga24-benchmarks.tar.gz"
  local device_download="$DOWNLOAD_CACHE/xcvu3p.device"

  download "$FPGA24_SOURCE_URL" "$source_archive"
  download "$FPGA24_SCHEMA_URL" "$schema_archive"
  download "$FPGA24_BENCHMARKS_URL" "$benchmark_archive"
  download "$FPGA24_DEVICE_URL" "$device_download"
  validate_tar_paths "$source_archive"
  validate_tar_paths "$schema_archive"
  validate_tar_paths "$benchmark_archive"

  local staging_root
  staging_root="$(mktemp -d "${AUP_WORKSPACE_ROOT}/.fpga24-staging.XXXXXX")"
  trap 'rm -rf -- "${staging_root:-}"' EXIT

  local source_unpack="$staging_root/source"
  local schema_unpack="$staging_root/schema"
  mkdir -p "$source_unpack" "$schema_unpack"
  tar -xzf "$source_archive" -C "$source_unpack" \
    --no-same-owner --no-same-permissions
  tar -xzf "$schema_archive" -C "$schema_unpack" \
    --no-same-owner --no-same-permissions

  local staged_contest
  local staged_schema
  staged_contest="$(single_extracted_directory "$source_unpack")"
  staged_schema="$(single_extracted_directory "$schema_unpack")"

  # GitHub source archives may preserve an empty submodule placeholder. Copying
  # the schema contents into that path avoids accidentally nesting the schema
  # repository one directory too deep.
  mkdir -p "$staged_contest/fpga-interchange-schema"
  cp -a -- "$staged_schema/." "$staged_contest/fpga-interchange-schema/"
  tar -xzf "$benchmark_archive" -C "$staged_contest" \
    --no-same-owner --no-same-permissions
  cp -- "$device_download" "$staged_contest/xcvu3p.device"

  local benchmark
  for benchmark in "${BENCHMARKS[@]}"; do
    place_archive_artifact_at_root \
      "$staged_contest" "${benchmark}_unrouted.phys"
    place_archive_artifact_at_root \
      "$staged_contest" "${benchmark}.netlist"
  done

  if [[ -d "$FPGA24_DIR" ]]; then
    # Preserve any existing contest source and completed large downloads. Fill
    # only the required benchmark/device inputs, and refresh the pinned schemas.
    mkdir -p "$FPGA24_DIR/fpga-interchange-schema"
    cp -a -- \
      "$staged_contest/fpga-interchange-schema/." \
      "$FPGA24_DIR/fpga-interchange-schema/"
    [[ -s "$FPGA24_DIR/xcvu3p.device" ]] ||
      cp -- "$staged_contest/xcvu3p.device" "$FPGA24_DIR/xcvu3p.device"
    for benchmark in "${BENCHMARKS[@]}"; do
      [[ -s "$FPGA24_DIR/${benchmark}_unrouted.phys" ]] ||
        cp -- \
          "$staged_contest/${benchmark}_unrouted.phys" \
          "$FPGA24_DIR/${benchmark}_unrouted.phys"
      [[ -s "$FPGA24_DIR/${benchmark}.netlist" ]] ||
        cp -- \
          "$staged_contest/${benchmark}.netlist" \
          "$FPGA24_DIR/${benchmark}.netlist"
    done
  else
    mv -- "$staged_contest" "$FPGA24_DIR"
  fi
  trap - EXIT
  rm -rf -- "$staging_root"

  if ! contest_data_is_ready; then
    report_missing_contest_artifacts
    die "FPGA24 setup completed, but the artifacts listed above are missing."
  fi

  # These are benchmark inputs, not output locations.
  chmod a-w "$FPGA24_DIR/xcvu3p.device"
  for benchmark in "${BENCHMARKS[@]}"; do
    chmod a-w \
      "$FPGA24_DIR/${benchmark}_unrouted.phys" \
      "$FPGA24_DIR/${benchmark}.netlist"
  done
}

configure_persistent_environment() {
  local rocm_root="${ROCM_PATH:-/opt/rocm}"

  log "Writing persistent AUP environment to $BFNEW_ENVIRONMENT"
  {
    printf '%s\n' '# Generated by bf-new/setup-aup.sh.'
    printf 'export BFNEW_TOOLS=%q\n' "$BFNEW_TOOLS"
    printf 'export BFNEW_ROOT=%q\n' "$BFNEW_DIR"
    printf 'export FPGA24_ROOT=%q\n' "$FPGA24_DIR"
    printf 'export BFNEW_FPGA24_DATA_ROOT=%q\n' "$FPGA24_DIR"
    printf 'export BFNEW_FPGA24_SCHEMA_ROOT=%q\n' \
      "$FPGA24_DIR/fpga-interchange-schema/interchange"
    printf 'export CPU_BUILD=%q\n' "$BFNEW_DIR/build/aup-cpu"
    printf 'export HIP_BUILD=%q\n' "$BFNEW_DIR/build/aup-hip"
    printf 'export CAMPAIGN_BUILD=%q\n' "$BFNEW_DIR/build/aup-campaign"
    printf 'export WORKLOAD_ROOT=%q\n' "$BFNEW_DIR/out/aup-workload"
    printf 'export EVIDENCE=%q\n' "$BFNEW_DIR/build/aup-evidence"
    printf 'export ROCM_PATH=%q\n' "$rocm_root"
    printf '%s\n' \
      'export PATH="$BFNEW_TOOLS/bin:$ROCM_PATH/bin:$PATH"' \
      'export LD_LIBRARY_PATH="$BFNEW_TOOLS/lib:$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"' \
      'export PKG_CONFIG_PATH="$BFNEW_TOOLS/lib/pkgconfig:${PKG_CONFIG_PATH:-}"' \
      'export CMAKE_PREFIX_PATH="$BFNEW_TOOLS:$ROCM_PATH"'
  } >"$BFNEW_ENVIRONMENT"
  chmod 0644 "$BFNEW_ENVIRONMENT"

  local source_line
  if [[ "$AUP_WORKSPACE_ROOT" == "$HOME/rips" ]]; then
    source_line='source "$HOME/rips/bfnew-env.sh"'
  else
    local quoted_environment
    printf -v quoted_environment '%q' "$BFNEW_ENVIRONMENT"
    source_line="source $quoted_environment"
  fi
  touch "$USER_BASHRC"
  if ! grep -Fqx "$source_line" "$USER_BASHRC"; then
    printf '\n%s\n' "$source_line" >>"$USER_BASHRC"
    log "Registered the bf-new environment in $USER_BASHRC"
  else
    log "The bf-new environment is already registered in $USER_BASHRC"
  fi
}

main() {
  require_command find
  require_command git
  require_command mktemp
  require_command tar
  require_command wget

  mkdir -p "$AUP_WORKSPACE_ROOT" "$DOWNLOAD_CACHE" "$BFNEW_TOOLS"
  clone_bfnew
  prepare_contest_data
  configure_persistent_environment

  log "Setup complete"
  printf '%s\n' \
    "bf-new repository: $BFNEW_DIR" \
    "FPGA24 data root:  $FPGA24_DIR" \
    "Schema root:       $FPGA24_DIR/fpga-interchange-schema/interchange" \
    "Download cache:    $DOWNLOAD_CACHE" \
    "Build tools:       $BFNEW_TOOLS" \
    "Environment:       $BFNEW_ENVIRONMENT" \
    "" \
    "New Bash terminals load the environment automatically." \
    "For this already-open terminal, run:" \
    "  source $BFNEW_ENVIRONMENT" \
    "" \
    "Use these CMake settings on AUP:" \
    "  -DBFNEW_FPGA24_DATA_ROOT=$FPGA24_DIR" \
    "  -DBFNEW_FPGA24_SCHEMA_ROOT=$FPGA24_DIR/fpga-interchange-schema/interchange"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
