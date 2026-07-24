#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <left_gray_image> <right_gray_image> [output_dir]" >&2
    exit 1
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${BUILD_DIR:-${project_dir}/build}"
output_dir="${3:-${project_dir}/output}"
cmake_bin="${CMAKE_BIN:-/usr/bin/cmake}"

cmake_args=(
    -S "${project_dir}"
    -B "${build_dir}"
    -DCMAKE_BUILD_TYPE=Release
)

if [[ -n "${OpenCV_DIR:-}" ]]; then
    cmake_args+=("-DOpenCV_DIR=${OpenCV_DIR}")
elif [[ -f "${project_dir}/../opencv-install-min/lib/cmake/opencv4/OpenCVConfig.cmake" ]]; then
    cmake_args+=(
        "-DOpenCV_DIR=${project_dir}/../opencv-install-min/lib/cmake/opencv4"
    )
fi

"${cmake_bin}" "${cmake_args[@]}"
"${cmake_bin}" --build "${build_dir}" --parallel

mkdir -p "${output_dir}"

"${build_dir}/vitis_bm_cpu" \
    "$1" "$2" "${output_dir}/bm_disparity_visual.png"

"${build_dir}/vitis_sgbm_cpu" \
    "$1" "$2" "${output_dir}/sgbm_disparity_raw.png"

echo "CPU stereo tests completed. Results: ${output_dir}"
