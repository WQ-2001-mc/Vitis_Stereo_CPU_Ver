#!/usr/bin/env python3
"""Run the repository's Vitis StereoBM CPU reference over a D/W parameter grid.

With no command-line arguments this script is interactive.  It auto-detects a
left image, a right image, and a YAML calibration file in the selected dataset
directory, then asks for the output directory and sweep parameters.

The C++ executable writes the Vitis-testbench-style 8-bit disparity visual.  A
matching Python OpenCV StereoBM call keeps the signed Q4 disparity needed for
metric depth conversion and verifies that the two visual outputs agree.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np


DEFAULT_DATASET = Path(
    "/home/hcc/Desktop/Public/datasets/gemini335/1280*800分辨率/data"
)
DEFAULT_DISPARITIES = (64, 128, 256)
DEFAULT_WINDOWS = (5, 9, 11, 21)
DEFAULT_DEPTH_MIN_MM = 300.0
DEFAULT_DEPTH_MAX_MM = 3000.0
DEFAULT_RUNS = 5

# These are intentionally identical to src/vitis_bm_cpu.cpp.
PRE_FILTER_CAP = 31
UNIQUENESS_RATIO = 15
TEXTURE_THRESHOLD = 20
MIN_DISPARITY = 0

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Calibration:
    source: str
    image_width: int
    image_height: int
    input_rectified: bool
    focal_px: float
    baseline_mm: float
    disparity_offset_px: float
    fx_baseline_mm_px: float
    left_map_x: Optional[np.ndarray] = None
    left_map_y: Optional[np.ndarray] = None
    right_map_x: Optional[np.ndarray] = None
    right_map_y: Optional[np.ndarray] = None

    def metadata(self) -> dict:
        result = asdict(self)
        for key in ("left_map_x", "left_map_y", "right_map_x", "right_map_y"):
            result.pop(key, None)
        return result


@dataclass
class SweepConfig:
    disparities: int
    window: int

    @property
    def key(self) -> str:
        return f"d{self.disparities:03d}_w{self.window:03d}"


@dataclass
class Result:
    disparities: int
    window: int
    nearest_depth_mm: float
    common_roi_depth_coverage_pct: float
    disparity_saturation_pct: float
    median_disparity_px: float
    depth_p10_mm: float
    depth_median_mm: float
    depth_p90_mm: float
    photometric_median_abs_error_gray: float
    photometric_inlier_le15_pct: float
    local_outlier_gt2px_pct: float
    cpp_compute_median_ms: float
    cpp_compute_min_ms: float
    cpp_compute_max_ms: float
    cpp_python_visual_match_pct: float
    valid_depth_pixels: int
    common_roi_pixels: int
    config_directory: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the repository StereoBM CPU reference for a D/W grid and "
            "generate depth maps plus a Markdown comparison report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, help="Dataset directory")
    parser.add_argument("--left", type=Path, help="Left image (overrides auto-detect)")
    parser.add_argument("--right", type=Path, help="Right image (overrides auto-detect)")
    parser.add_argument(
        "--calibration", type=Path, help="OpenCV/Orbbec YAML calibration file"
    )
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument(
        "--disparities",
        help="Comma/space-separated D values; each must be divisible by 16",
    )
    parser.add_argument(
        "--windows", help="Comma/space-separated odd StereoBM block sizes W"
    )
    parser.add_argument(
        "--depth-min-mm", type=float, help="Near end of the shared depth color scale"
    )
    parser.add_argument(
        "--depth-max-mm", type=float, help="Far end of the shared depth color scale"
    )
    parser.add_argument(
        "--runs", type=int, help="Number of C++ compute-time repetitions per setting"
    )
    parser.add_argument(
        "--focal-px",
        type=float,
        help="Rectified focal length if no calibration YAML is available",
    )
    parser.add_argument(
        "--baseline-mm",
        type=float,
        help="Stereo baseline in mm if no calibration YAML is available",
    )
    parser.add_argument(
        "--disparity-offset-px",
        type=float,
        default=0.0,
        help="cx_right-cx_left used in Z=fB/(d+offset) without calibration YAML",
    )
    parser.add_argument(
        "--assume-rectified",
        action="store_true",
        help="Treat inputs as already rectified even if YAML does not say so",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Vitis_Stereo_CPU_Ver repository root",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt even when other command-line arguments are present",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use command-line values/defaults without prompts",
    )
    return parser.parse_args()


def prompt_text(label: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    response = input(f"{label}{suffix}: ").strip()
    return response if response else (default or "")


def parse_int_list(text: str, label: str) -> tuple[int, ...]:
    values = []
    for token in re.split(r"[\s,，;；]+", text.strip()):
        if token:
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"{label} contains a non-integer value: {token}") from exc
    if not values:
        raise ValueError(f"{label} cannot be empty")
    return tuple(dict.fromkeys(values))


def list_text(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def auto_detect_files(dataset: Path) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    if not dataset.is_dir():
        return None, None, None
    files = sorted(path for path in dataset.iterdir() if path.is_file())
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    left_candidates = [path for path in images if "left" in path.name.casefold()]
    right_candidates = [path for path in images if "right" in path.name.casefold()]
    yaml_candidates = [
        path for path in files if path.suffix.lower() in {".yaml", ".yml"}
    ]
    left = left_candidates[0] if len(left_candidates) == 1 else None
    right = right_candidates[0] if len(right_candidates) == 1 else None
    calibration = yaml_candidates[0] if len(yaml_candidates) == 1 else None
    return left, right, calibration


def resolve_inputs(args: argparse.Namespace) -> argparse.Namespace:
    interactive = args.interactive or (len(sys.argv) == 1 and not args.non_interactive)
    if args.interactive and args.non_interactive:
        raise ValueError("--interactive and --non-interactive cannot be used together")

    dataset = args.dataset or DEFAULT_DATASET
    if interactive:
        print("\nStereoBM D/W 深度图扫描（直接回车采用方括号内默认值）")
        dataset = Path(prompt_text("数据目录", str(dataset))).expanduser()

    auto_left, auto_right, auto_calibration = auto_detect_files(dataset)
    left = args.left or auto_left
    right = args.right or auto_right
    calibration = args.calibration or auto_calibration

    if interactive:
        left = Path(prompt_text("左图", str(left) if left else "")).expanduser()
        right = Path(prompt_text("右图", str(right) if right else "")).expanduser()
        calibration_text = prompt_text(
            "标定 YAML（已有 fx/baseline 参数时可留空）",
            str(calibration) if calibration else "",
        )
        calibration = Path(calibration_text).expanduser() if calibration_text else None

    default_output = dataset / "stereobm_depth_sweep"
    output = args.output or default_output
    d_text = args.disparities or list_text(DEFAULT_DISPARITIES)
    w_text = args.windows or list_text(DEFAULT_WINDOWS)
    depth_min = (
        args.depth_min_mm
        if args.depth_min_mm is not None
        else DEFAULT_DEPTH_MIN_MM
    )
    depth_max = (
        args.depth_max_mm
        if args.depth_max_mm is not None
        else DEFAULT_DEPTH_MAX_MM
    )
    runs = args.runs if args.runs is not None else DEFAULT_RUNS

    if interactive:
        output = Path(prompt_text("输出目录", str(output))).expanduser()
        d_text = prompt_text("视差范围 D（逗号分隔）", d_text)
        w_text = prompt_text("窗口大小 W（奇数，逗号分隔）", w_text)
        depth_min = float(prompt_text("统一色标最近深度/mm", f"{depth_min:g}"))
        depth_max = float(prompt_text("统一色标最远深度/mm", f"{depth_max:g}"))
        runs = int(prompt_text("每组 C++ 计时重复次数", str(runs)))

    if left is None or right is None:
        raise ValueError(
            "Could not auto-detect exactly one left and one right image; "
            "provide --left and --right."
        )

    args.dataset = dataset.resolve()
    args.left = left.resolve()
    args.right = right.resolve()
    args.calibration = calibration.resolve() if calibration else None
    args.output = output.resolve()
    args.disparity_values = parse_int_list(d_text, "D")
    args.window_values = parse_int_list(w_text, "W")
    args.depth_min_mm = depth_min
    args.depth_max_mm = depth_max
    args.runs = runs
    args.project_root = args.project_root.resolve()
    return args


def validate_paths_and_parameters(args: argparse.Namespace) -> None:
    for label, path in (("left image", args.left), ("right image", args.right)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.calibration and not args.calibration.is_file():
        raise FileNotFoundError(f"calibration file not found: {args.calibration}")
    if not args.project_root.is_dir():
        raise FileNotFoundError(f"project root not found: {args.project_root}")
    if args.depth_min_mm <= 0 or args.depth_max_mm <= args.depth_min_mm:
        raise ValueError("depth range must satisfy 0 < min < max")
    if args.runs < 1 or args.runs > 100:
        raise ValueError("--runs must be in [1, 100]")
    for value in args.disparity_values:
        if value <= 0 or value % 16 != 0:
            raise ValueError(f"D={value} must be positive and divisible by 16")
    for value in args.window_values:
        if value < 5 or value > 255 or value % 2 == 0:
            raise ValueError(f"W={value} must be odd and in [5, 255]")
    if args.calibration is None and (
        args.focal_px is None or args.baseline_mm is None
    ):
        raise ValueError(
            "No calibration YAML was found. Provide both --focal-px and --baseline-mm."
        )


def fs_real(fs: cv2.FileStorage, key: str) -> Optional[float]:
    node = fs.getNode(key)
    if node.empty():
        return None
    value = node.real()
    return float(value)


def fs_matrix(fs: cv2.FileStorage, *keys: str) -> Optional[np.ndarray]:
    for key in keys:
        node = fs.getNode(key)
        if not node.empty():
            matrix = node.mat()
            if matrix is not None and matrix.size:
                return np.asarray(matrix, dtype=np.float64)
    return None


def load_calibration(
    path: Optional[Path],
    image_size: tuple[int, int],
    focal_px_override: Optional[float],
    baseline_mm_override: Optional[float],
    disparity_offset_override: float,
    assume_rectified: bool,
) -> Calibration:
    width, height = image_size
    if path is None:
        focal = float(focal_px_override)
        baseline = float(baseline_mm_override)
        if focal <= 0 or baseline <= 0:
            raise ValueError("focal length and baseline must be positive")
        return Calibration(
            source="command-line focal/baseline",
            image_width=width,
            image_height=height,
            input_rectified=True,
            focal_px=focal,
            baseline_mm=baseline,
            disparity_offset_px=float(disparity_offset_override),
            fx_baseline_mm_px=focal * baseline,
        )

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"OpenCV could not open calibration YAML: {path}")
    try:
        yaml_width = int(fs_real(fs, "image_width") or width)
        yaml_height = int(fs_real(fs, "image_height") or height)
        if (yaml_width, yaml_height) != (width, height):
            raise ValueError(
                "calibration/image size mismatch: "
                f"YAML={yaml_width}x{yaml_height}, images={width}x{height}"
            )

        rectified_flag = fs_real(fs, "parameters_are_rectified")
        input_rectified = assume_rectified or bool(rectified_flag)
        p1 = fs_matrix(fs, "P1")
        p2 = fs_matrix(fs, "P2")
        m1 = fs_matrix(fs, "M1", "K1")
        m2 = fs_matrix(fs, "M2", "K2")
        d1 = fs_matrix(fs, "D1")
        d2 = fs_matrix(fs, "D2")
        rotation = fs_matrix(fs, "R")
        translation = fs_matrix(fs, "T")

        focal_node = fs_real(fs, "focal_length_px")
        baseline_node = fs_real(fs, "baseline_mm")
        if baseline_node is None:
            baseline_node = fs_real(fs, "baseline")
        bf_node = fs_real(fs, "fx_times_baseline_mm_px")

        if input_rectified:
            focal = float(p1[0, 0]) if p1 is not None else float(focal_node or 0)
            if p1 is not None and p2 is not None:
                disparity_offset = float(p2[0, 2] - p1[0, 2])
                bf_from_projection = abs(float(p2[0, 3]))
            else:
                disparity_offset = float(disparity_offset_override)
                bf_from_projection = 0.0
            bf = float(bf_node or bf_from_projection)
            baseline = float(baseline_node or (bf / focal if focal else 0))
            if not bf and focal and baseline:
                bf = focal * baseline
            if focal <= 0 or baseline <= 0 or bf <= 0:
                raise ValueError(
                    "rectified YAML must provide P1/P2 or focal/baseline information"
                )
            return Calibration(
                source=str(path),
                image_width=width,
                image_height=height,
                input_rectified=True,
                focal_px=focal,
                baseline_mm=baseline,
                disparity_offset_px=disparity_offset,
                fx_baseline_mm_px=bf,
            )

        required = {
            "M1/K1": m1,
            "M2/K2": m2,
            "D1": d1,
            "D2": d2,
            "R": rotation,
            "T": translation,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "unrectified calibration is missing: " + ", ".join(missing)
            )
        rect_l, rect_r, proj_l, proj_r, _, _, _ = cv2.stereoRectify(
            m1,
            d1,
            m2,
            d2,
            (width, height),
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        map_lx, map_ly = cv2.initUndistortRectifyMap(
            m1, d1, rect_l, proj_l, (width, height), cv2.CV_32FC1
        )
        map_rx, map_ry = cv2.initUndistortRectifyMap(
            m2, d2, rect_r, proj_r, (width, height), cv2.CV_32FC1
        )
        focal = float(proj_l[0, 0])
        disparity_offset = float(proj_r[0, 2] - proj_l[0, 2])
        baseline = float(baseline_node or np.linalg.norm(translation))
        bf = focal * baseline
        return Calibration(
            source=str(path),
            image_width=width,
            image_height=height,
            input_rectified=False,
            focal_px=focal,
            baseline_mm=baseline,
            disparity_offset_px=disparity_offset,
            fx_baseline_mm_px=bf,
            left_map_x=map_lx,
            left_map_y=map_ly,
            right_map_x=map_rx,
            right_map_y=map_ry,
        )
    finally:
        fs.release()


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV failed to read image: {path}")
    return image


def rectify_pair(
    left: np.ndarray, right: np.ndarray, calibration: Calibration
) -> tuple[np.ndarray, np.ndarray]:
    if calibration.input_rectified:
        return left, right
    rectified_left = cv2.remap(
        left,
        calibration.left_map_x,
        calibration.left_map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    rectified_right = cv2.remap(
        right,
        calibration.right_map_x,
        calibration.right_map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return rectified_left, rectified_right


def ensure_cpp_binary(project_root: Path) -> tuple[Path, str]:
    binary = project_root / "build" / "vitis_bm_cpu"
    source = project_root / "src" / "vitis_bm_cpu.cpp"
    cmake_file = project_root / "CMakeLists.txt"
    needs_build = not binary.is_file()
    if binary.is_file():
        newest_input = max(source.stat().st_mtime, cmake_file.stat().st_mtime)
        needs_build = binary.stat().st_mtime < newest_input
    build_log = ""
    if needs_build:
        commands = [
            ["cmake", "-S", str(project_root), "-B", str(project_root / "build")],
            ["cmake", "--build", str(project_root / "build"), "-j"],
        ]
        logs = []
        for command in commands:
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False
            )
            logs.append(
                "$ "
                + " ".join(command)
                + "\n"
                + completed.stdout
                + completed.stderr
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Failed to build vitis_bm_cpu:\n" + "\n".join(logs)
                )
        build_log = "\n".join(logs)
    if not binary.is_file():
        raise FileNotFoundError(f"C++ StereoBM binary was not produced: {binary}")
    return binary, build_log


def create_matcher(disparities: int, window: int) -> cv2.StereoBM:
    matcher = cv2.StereoBM_create(
        numDisparities=disparities, blockSize=window
    )
    matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
    matcher.setPreFilterCap(PRE_FILTER_CAP)
    matcher.setMinDisparity(MIN_DISPARITY)
    matcher.setTextureThreshold(TEXTURE_THRESHOLD)
    matcher.setUniquenessRatio(UNIQUENESS_RATIO)
    return matcher


def cpp_visual_from_q4(disparity_q4: np.ndarray, disparities: int) -> np.ndarray:
    scale = (256.0 / disparities) / 16.0
    scaled = np.rint(disparity_q4.astype(np.float64) * scale)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def depth_color(
    depth_mm: np.ndarray,
    valid_mask: np.ndarray,
    depth_min_mm: float,
    depth_max_mm: float,
) -> np.ndarray:
    normalized = (
        (depth_max_mm - np.clip(depth_mm, depth_min_mm, depth_max_mm))
        / (depth_max_mm - depth_min_mm)
        * 255.0
    )
    indices = np.clip(np.rint(normalized), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(indices, cv2.COLORMAP_TURBO)
    colored[~valid_mask] = 0
    return colored


def safe_percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def calculate_metrics(
    disparity_q4: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    config: SweepConfig,
    calibration: Calibration,
    common_roi: np.ndarray,
    cpp_times_ms: Sequence[float],
    cpp_match_pct: float,
    config_directory: str,
) -> tuple[Result, np.ndarray, np.ndarray]:
    disparity = disparity_q4.astype(np.float32) / 16.0
    denominator = disparity + calibration.disparity_offset_px
    algorithm_valid = disparity_q4 >= MIN_DISPARITY * 16
    valid_depth = algorithm_valid & (denominator > 0)
    depth_mm = np.zeros(disparity.shape, dtype=np.float32)
    depth_mm[valid_depth] = calibration.fx_baseline_mm_px / denominator[valid_depth]

    evaluation_mask = common_roi & valid_depth
    roi_pixels = int(np.count_nonzero(common_roi))
    valid_pixels = int(np.count_nonzero(evaluation_mask))
    disparities_in_roi = disparity[evaluation_mask]
    depths_in_roi = depth_mm[evaluation_mask]
    saturation = (
        disparities_in_roi >= float(config.disparities - 1)
        if disparities_in_roi.size
        else np.array([], dtype=bool)
    )

    rows, cols = disparity.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32)
    )
    right_x = grid_x - disparity
    warped_right = cv2.remap(
        right,
        right_x,
        grid_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    photo_mask = (
        evaluation_mask
        & (right_x >= 0.0)
        & (right_x <= cols - 1.0)
    )
    abs_error = cv2.absdiff(left, warped_right)
    photo_values = abs_error[photo_mask]

    # Require a fully valid 5x5 neighborhood so invalid sentinels cannot create
    # artificial local-disparity outliers.
    dense_valid = cv2.erode(
        evaluation_mask.astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    disparity_for_median = disparity.copy()
    disparity_for_median[~valid_depth] = 0.0
    local_median = cv2.medianBlur(disparity_for_median, 5)
    local_difference = np.abs(disparity - local_median)
    outlier_values = local_difference[dense_valid]

    nearest = calibration.fx_baseline_mm_px / (
        max(config.disparities - 1, 1) + calibration.disparity_offset_px
    )
    result = Result(
        disparities=config.disparities,
        window=config.window,
        nearest_depth_mm=float(nearest),
        common_roi_depth_coverage_pct=safe_percent(valid_pixels, roi_pixels),
        disparity_saturation_pct=safe_percent(
            int(np.count_nonzero(saturation)), int(saturation.size)
        ),
        median_disparity_px=finite_percentile(disparities_in_roi, 50),
        depth_p10_mm=finite_percentile(depths_in_roi, 10),
        depth_median_mm=finite_percentile(depths_in_roi, 50),
        depth_p90_mm=finite_percentile(depths_in_roi, 90),
        photometric_median_abs_error_gray=finite_percentile(photo_values, 50),
        photometric_inlier_le15_pct=safe_percent(
            int(np.count_nonzero(photo_values <= 15)), int(photo_values.size)
        ),
        local_outlier_gt2px_pct=safe_percent(
            int(np.count_nonzero(outlier_values > 2.0)), int(outlier_values.size)
        ),
        cpp_compute_median_ms=float(statistics.median(cpp_times_ms)),
        cpp_compute_min_ms=float(min(cpp_times_ms)),
        cpp_compute_max_ms=float(max(cpp_times_ms)),
        cpp_python_visual_match_pct=cpp_match_pct,
        valid_depth_pixels=valid_pixels,
        common_roi_pixels=roi_pixels,
        config_directory=config_directory,
    )
    return result, depth_mm, valid_depth


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def run_cpp_reference(
    binary: Path,
    left_path: Path,
    right_path: Path,
    output_visual: Path,
    config: SweepConfig,
    runs: int,
) -> tuple[list[float], str]:
    pattern = re.compile(r"CPU compute:\s*([0-9]+(?:\.[0-9]+)?)\s*ms")
    times: list[float] = []
    log_sections = []
    command = [
        str(binary),
        str(left_path),
        str(right_path),
        str(output_visual),
        str(config.disparities),
        str(config.window),
    ]
    for index in range(runs):
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        section = (
            f"=== run {index + 1}/{runs} ===\n"
            f"$ {' '.join(command)}\n"
            f"{completed.stdout}{completed.stderr}"
        )
        log_sections.append(section)
        if completed.returncode != 0:
            raise RuntimeError(
                f"C++ StereoBM failed for D={config.disparities}, "
                f"W={config.window}:\n{section}"
            )
        match = pattern.search(completed.stdout)
        if not match:
            raise RuntimeError(f"could not parse C++ compute time:\n{section}")
        times.append(float(match.group(1)))
    return times, "\n".join(log_sections)


def fit_panel_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_size = (
        max(1, int(round(image.shape[1] * scale))),
        max(1, int(round(image.shape[0] * scale))),
    )
    resized = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    x0 = (width - resized.shape[1]) // 2
    y0 = (height - resized.shape[0]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def put_centered(
    image: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    size, _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = max(4, (image.shape[1] - size[0]) // 2)
    cv2.putText(
        image, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA
    )


def add_horizontal_colorbar(
    canvas: np.ndarray, y: int, depth_min_mm: float, depth_max_mm: float
) -> None:
    margin = 100
    bar_width = canvas.shape[1] - 2 * margin
    if bar_width < 100:
        return
    index = np.linspace(255, 0, bar_width, dtype=np.uint8)[None, :]
    bar = cv2.applyColorMap(index, cv2.COLORMAP_TURBO)
    bar = cv2.resize(bar, (bar_width, 22), interpolation=cv2.INTER_NEAREST)
    canvas[y : y + 22, margin : margin + bar_width] = bar
    cv2.rectangle(
        canvas, (margin, y), (margin + bar_width - 1, y + 21), (220, 220, 220), 1
    )
    cv2.putText(
        canvas,
        f"{depth_min_mm:g} mm (near)",
        (margin, y + 48),
        FONT,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    far_text = f"{depth_max_mm:g} mm (far)"
    size, _ = cv2.getTextSize(far_text, FONT, 0.55, 1)
    cv2.putText(
        canvas,
        far_text,
        (margin + bar_width - size[0], y + 48),
        FONT,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def make_depth_grid(
    configurations: Sequence[SweepConfig],
    results_by_key: dict[str, Result],
    colors_by_key: dict[str, np.ndarray],
    rows: int,
    cols: int,
    title: str,
    depth_min_mm: float,
    depth_max_mm: float,
    output_path: Path,
) -> None:
    cell_width = 420
    image_height = 263
    header_height = 62
    footer_height = 58
    title_height = 62
    colorbar_height = 76
    cell_height = header_height + image_height + footer_height
    canvas = np.full(
        (
            title_height + rows * cell_height + colorbar_height,
            cols * cell_width,
            3,
        ),
        18,
        dtype=np.uint8,
    )
    put_centered(canvas, title, 39, 0.85, (245, 245, 245), 2)
    for index, config in enumerate(configurations):
        row = index // cols
        col = index % cols
        x0 = col * cell_width
        y0 = title_height + row * cell_height
        cell = canvas[y0 : y0 + cell_height, x0 : x0 + cell_width]
        result = results_by_key[config.key]
        put_centered(
            cell,
            f"D={config.disparities}   W={config.window}",
            35,
            0.75,
            (255, 255, 255),
            2,
        )
        fitted = fit_panel_image(
            colors_by_key[config.key], cell_width, image_height
        )
        cell[
            header_height : header_height + image_height, 0:cell_width
        ] = fitted
        stats = (
            f"coverage {result.common_roi_depth_coverage_pct:.1f}%  "
            f"photo<=15 {result.photometric_inlier_le15_pct:.1f}%  "
            f"CPU {result.cpp_compute_median_ms:.1f} ms"
        )
        put_centered(
            cell,
            stats,
            header_height + image_height + 34,
            0.49,
            (225, 225, 225),
            1,
        )
        cv2.rectangle(
            cell,
            (0, 0),
            (cell_width - 1, cell_height - 1),
            (75, 75, 75),
            1,
        )
    bar_y = title_height + rows * cell_height + 10
    add_horizontal_colorbar(canvas, bar_y, depth_min_mm, depth_max_mm)
    write_image(output_path, canvas)


def make_source_pair(
    left: np.ndarray, right: np.ndarray, output_path: Path
) -> None:
    annotated = []
    line_positions = np.linspace(80, left.shape[0] - 80, 7, dtype=int)
    for image, label in ((left, "RECTIFIED LEFT"), (right, "RECTIFIED RIGHT")):
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for y in line_positions:
            cv2.line(color, (0, int(y)), (color.shape[1] - 1, int(y)), (0, 220, 0), 1)
        panel = fit_panel_image(color, 640, 400)
        wrapped = np.full((455, 640, 3), 20, np.uint8)
        wrapped[55:, :] = panel
        put_centered(wrapped, label, 35, 0.72, (245, 245, 245), 2)
        annotated.append(wrapped)
    gap = np.full((455, 12, 3), 20, np.uint8)
    write_image(output_path, np.hstack((annotated[0], gap, annotated[1])))


def metric_limits(values: Iterable[float], percentage: bool = False) -> tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    low = 0.0
    high = max(finite)
    high = min(100.0, max(5.0, high * 1.12)) if percentage else max(1.0, high * 1.15)
    return low, high


def make_metrics_chart(
    results: Sequence[Result],
    disparity_values: Sequence[int],
    window_values: Sequence[int],
    output_path: Path,
) -> None:
    canvas = np.full((650, 1500, 3), 250, dtype=np.uint8)
    cv2.putText(
        canvas,
        "StereoBM diagnostic metrics (common ROI; CPU time is not FPGA latency)",
        (35, 42),
        FONT,
        0.85,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    panels = [
        (
            "Finite-depth coverage (%)",
            "common_roi_depth_coverage_pct",
            True,
        ),
        (
            "Photometric inliers <= 15 (%)",
            "photometric_inlier_le15_pct",
            True,
        ),
        ("C++ compute median (ms)", "cpp_compute_median_ms", False),
    ]
    colors = [(28, 91, 214), (54, 155, 61), (204, 99, 28), (145, 70, 180)]
    result_map = {(result.disparities, result.window): result for result in results}
    for panel_index, (title, field, percentage) in enumerate(panels):
        panel_x = 30 + panel_index * 490
        plot_x0, plot_x1 = panel_x + 62, panel_x + 455
        plot_y0, plot_y1 = 105, 535
        cv2.rectangle(
            canvas, (panel_x, 75), (panel_x + 470, 600), (210, 210, 210), 1
        )
        cv2.putText(
            canvas,
            title,
            (panel_x + 18, 101),
            FONT,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        panel_values = [
            getattr(result, field)
            for result in results
            if np.isfinite(getattr(result, field))
        ]
        y_min, y_max = metric_limits(panel_values, percentage)
        for tick_index in range(6):
            fraction = tick_index / 5.0
            y = int(round(plot_y1 - fraction * (plot_y1 - plot_y0)))
            value = y_min + fraction * (y_max - y_min)
            cv2.line(canvas, (plot_x0, y), (plot_x1, y), (225, 225, 225), 1)
            label = f"{value:.0f}" if y_max >= 10 else f"{value:.1f}"
            cv2.putText(
                canvas,
                label,
                (panel_x + 7, y + 5),
                FONT,
                0.42,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
        if len(window_values) == 1:
            x_positions = [(plot_x0 + plot_x1) // 2]
        else:
            x_positions = [
                int(round(plot_x0 + i * (plot_x1 - plot_x0) / (len(window_values) - 1)))
                for i in range(len(window_values))
            ]
        for x, window in zip(x_positions, window_values):
            cv2.line(canvas, (x, plot_y1), (x, plot_y1 + 5), (40, 40, 40), 1)
            label = str(window)
            size, _ = cv2.getTextSize(label, FONT, 0.45, 1)
            cv2.putText(
                canvas,
                label,
                (x - size[0] // 2, plot_y1 + 25),
                FONT,
                0.45,
                (50, 50, 50),
                1,
                cv2.LINE_AA,
            )
        for d_index, disparities in enumerate(disparity_values):
            color = colors[d_index % len(colors)]
            points = []
            for x, window in zip(x_positions, window_values):
                result = result_map[(disparities, window)]
                value = getattr(result, field)
                fraction = (
                    (value - y_min) / (y_max - y_min)
                    if np.isfinite(value) and y_max > y_min
                    else 0.0
                )
                y = int(round(plot_y1 - fraction * (plot_y1 - plot_y0)))
                points.append((x, y))
            for first, second in zip(points, points[1:]):
                cv2.line(canvas, first, second, color, 3, cv2.LINE_AA)
            for point in points:
                cv2.circle(canvas, point, 6, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, point, 6, (255, 255, 255), 1, cv2.LINE_AA)
            legend_x = panel_x + 18 + d_index * 120
            cv2.line(
                canvas,
                (legend_x, 579),
                (legend_x + 20, 579),
                color,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"D={disparities}",
                (legend_x + 25, 584),
                FONT,
                0.42,
                (45, 45, 45),
                1,
                cv2.LINE_AA,
            )
        cv2.line(canvas, (plot_x0, plot_y0), (plot_x0, plot_y1), (40, 40, 40), 1)
        cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), (40, 40, 40), 1)
    write_image(output_path, canvas)


def format_number(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"


def write_metrics_csv(results: Sequence[Result], output_path: Path) -> None:
    fieldnames = list(asdict(results[0]).keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    calibration: Calibration,
    results: Sequence[Result],
    common_roi_bounds: tuple[int, int, int, int],
    rectification_mode: str,
) -> None:
    best_coverage = max(results, key=lambda item: item.common_roi_depth_coverage_pct)
    best_photo = max(results, key=lambda item: item.photometric_inlier_le15_pct)
    fastest = min(results, key=lambda item: item.cpp_compute_median_ms)
    exact_match = min(item.cpp_python_visual_match_pct for item in results)
    result_map = {
        (item.disparities, item.window): item for item in results
    }
    default_candidate = result_map.get((128, 11), best_coverage)
    w9_candidate = result_map.get((default_candidate.disparities, 9))
    same_d_small = result_map.get(
        (default_candidate.disparities, min(args.window_values))
    )
    same_d_smooth = result_map.get(
        (default_candidate.disparities, max(args.window_values))
    )
    larger_d_same_w = result_map.get(
        (
            next(
                (
                    value
                    for value in args.disparity_values
                    if value > default_candidate.disparities
                ),
                default_candidate.disparities,
            ),
            default_candidate.window,
        )
    )
    smallest_d_results = [
        item for item in results if item.disparities == min(args.disparity_values)
    ]
    smallest_d_saturation = statistics.mean(
        item.disparity_saturation_pct for item in smallest_d_results
    )
    x0, y0, x1, y1 = common_roi_bounds

    lines = [
        "# Gemini335 StereoBM：D/W 参数深度图横向对比",
        "",
        "## 技术摘要",
        "",
        (
            f"本报告对仓库 `src/vitis_bm_cpu.cpp` 的 StereoBM 参数做了 "
            f"{len(args.disparity_values)}×{len(args.window_values)} 组扫描："
            f"D={list_text(args.disparity_values)}，W={list_text(args.window_values)}。"
            f"所有深度图使用同一色标 {args.depth_min_mm:g}–"
            f"{args.depth_max_mm:g} mm（红色近、蓝色远、黑色无效），因此可以直接比较。"
        ),
        "",
        (
            f"有限深度覆盖率最高的是 D={best_coverage.disparities}, "
            f"W={best_coverage.window}（{best_coverage.common_roi_depth_coverage_pct:.1f}%）；"
            f"光度残差≤15 的占比最高的是 D={best_photo.disparities}, "
            f"W={best_photo.window}（{best_photo.photometric_inlier_le15_pct:.1f}%）；"
            f"本机 C++ 中位计算时间最短的是 D={fastest.disparities}, "
            f"W={fastest.window}（{fastest.cpp_compute_median_ms:.1f} ms）。"
            "这些指标没有真值监督，只能用于诊断和参数筛选，不能等同于深度精度。"
        ),
        "",
        "### 本次数据的直接结论",
        "",
        (
            f"- **优先候选：D={default_candidate.disparities}, "
            f"W={default_candidate.window}。**它在共同 ROI 上得到 "
            f"{default_candidate.common_roi_depth_coverage_pct:.1f}% 的有限深度覆盖，"
            f"光度≤15 占比 {default_candidate.photometric_inlier_le15_pct:.1f}%，"
            f"搜索顶端饱和仅 {default_candidate.disparity_saturation_pct:.2f}%，"
            f"本机 C++ 中位时间 {default_candidate.cpp_compute_median_ms:.1f} ms。"
            "这表示它适合作为后续 HLS 候选，但不表示已经证明深度最准确。"
        ),
        (
            f"- **D={min(args.disparity_values)} 对本场景偏小。**其理论最近范围是 "
            f"{calibration.fx_baseline_mm_px / (min(args.disparity_values) - 1 + calibration.disparity_offset_px):.1f} mm，"
            f"本轮不同 W 的平均顶端饱和率为 {smallest_d_saturation:.2f}%；"
            "近处物体会更容易被截到最大视差。"
        ),
        (
            f"- **W 从 {min(args.window_values)} 增至 {max(args.window_values)} 后，"
            "图面明显更连续，但细杆、电缆及遮挡边界更容易被聚合。**"
            + (
                f"在 D={default_candidate.disparities} 下，光度≤15 占比从 "
                f"{same_d_small.photometric_inlier_le15_pct:.1f}% "
                f"变化到 {same_d_smooth.photometric_inlier_le15_pct:.1f}%，"
                f"覆盖率从 {same_d_small.common_roi_depth_coverage_pct:.1f}% "
                f"变化到 {same_d_smooth.common_roi_depth_coverage_pct:.1f}%。"
                if same_d_small
                and same_d_smooth
                and same_d_smooth.window != same_d_small.window
                else ""
            )
        ),
        *(
            [
                (
                    f"- **新增 W=9 与 W={default_candidate.window} 很接近，"
                    "适合作为硬件窗口折中候选。**"
                    f"在 D={default_candidate.disparities} 下，W=9 的覆盖率为 "
                    f"{w9_candidate.common_roi_depth_coverage_pct:.1f}%、"
                    f"光度≤15 占比为 {w9_candidate.photometric_inlier_le15_pct:.1f}%、"
                    f"本机 C++ 中位时间为 {w9_candidate.cpp_compute_median_ms:.1f} ms；"
                    f"W={default_candidate.window} 对应为 "
                    f"{default_candidate.common_roi_depth_coverage_pct:.1f}%、"
                    f"{default_candidate.photometric_inlier_le15_pct:.1f}% 和 "
                    f"{default_candidate.cpp_compute_median_ms:.1f} ms。"
                )
            ]
            if w9_candidate and w9_candidate.window != default_candidate.window
            else []
        ),
        (
            f"- **D={max(args.disparity_values)} 只在需要更近距离时值得优先考虑。**"
            + (
                f"同为 W={default_candidate.window}，其本机 CPU 中位时间为 "
                f"{larger_d_same_w.cpp_compute_median_ms:.1f} ms，"
                f"而 D={default_candidate.disparities} 为 "
                f"{default_candidate.cpp_compute_median_ms:.1f} ms；"
                f"共同 ROI 覆盖率分别为 "
                f"{larger_d_same_w.common_roi_depth_coverage_pct:.1f}% 和 "
                f"{default_candidate.common_roi_depth_coverage_pct:.1f}%。"
                if larger_d_same_w
                and larger_d_same_w.disparities != default_candidate.disparities
                else ""
            )
        ),
        "",
        "## 关键发现与可视证据",
        "",
        "![全部 D/W 深度图矩阵](comparison_all.png)",
        "",
        (
            "上图是最重要的对比：每个单元格都使用完全相同的毫米色标。"
            "D 决定最大搜索视差及最近可测距离；W 决定局部匹配聚合范围。"
            "通常小 W 更可能保留物体边缘但出现噪点，大 W 会提高局部连续性，"
            "同时更容易在深度边界产生前景/背景混合。黑色区域是无有效正视差，"
            "不是零毫米深度。"
        ),
        "",
    ]

    for disparities in args.disparity_values:
        lines.extend(
            [
                f"### D={disparities}：不同 W 横向对比",
                "",
                f"![D={disparities} 的 W 横向对比](comparison_D{disparities:03d}.png)",
                "",
                (
                    f"D={disparities} 的理论最大整数视差约为 {disparities - 1} px；"
                    f"按本标定参数估算，最近可测距离约 "
                    f"{calibration.fx_baseline_mm_px / (disparities - 1 + calibration.disparity_offset_px):.1f} mm。"
                    f"横向查看 W={list_text(args.window_values).replace(',', '、')}，"
                    "可重点观察支架边缘、右侧大斜面、"
                    "桌面电缆和远处暗区域的孔洞、噪声与边界扩散。"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "### 无真值诊断指标",
            "",
            "![覆盖率、光度一致性与 CPU 计算时间](metrics_comparison.png)",
            "",
            (
                "覆盖率和光度一致性统一在共同 ROI 上计算，以免 D 越大导致左侧天然搜索盲区越宽而产生不公平比较。"
                "光度指标把右图按估计视差回投到左图后计算灰度绝对差；遮挡、曝光变化和投影散斑都会影响它。"
                "图中的 CPU 时间只反映当前主机 OpenCV 参考实现，不是 FPGA 延时。"
            ),
            "",
            "## 完整指标表",
            "",
            "| D | W | 最近范围/mm | 有限深度覆盖/% | 顶端饱和/% | 中位视差/px | 深度 P10/P50/P90 mm | 光度中位误差 | 光度≤15/% | 局部离群>2px/% | C++ 中位时间/ms | C++/Python 图一致/% |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        lines.append(
            "| "
            f"{item.disparities} | {item.window} | {format_number(item.nearest_depth_mm)} | "
            f"{format_number(item.common_roi_depth_coverage_pct)} | "
            f"{format_number(item.disparity_saturation_pct, 2)} | "
            f"{format_number(item.median_disparity_px, 2)} | "
            f"{format_number(item.depth_p10_mm)}/"
            f"{format_number(item.depth_median_mm)}/"
            f"{format_number(item.depth_p90_mm)} | "
            f"{format_number(item.photometric_median_abs_error_gray)} | "
            f"{format_number(item.photometric_inlier_le15_pct)} | "
            f"{format_number(item.local_outlier_gt2px_pct)} | "
            f"{format_number(item.cpp_compute_median_ms, 2)} | "
            f"{format_number(item.cpp_python_visual_match_pct, 4)} |"
        )

    lines.extend(
        [
            "",
            "逐组的原始 Q4 视差、毫米深度、彩色深度图、C++ 输出和运行日志见 [`details/`](details/)，"
            "机器可读数据见 [`metrics.csv`](metrics.csv) 与 [`metadata.json`](metadata.json)。",
            "",
            "## 数据、标定与度量范围",
            "",
            f"- 左图：`{args.left}`",
            f"- 右图：`{args.right}`",
            f"- 标定来源：`{calibration.source}`",
            f"- 输入尺寸：{calibration.image_width}×{calibration.image_height}",
            f"- 校正方式：{rectification_mode}",
            f"- 焦距：{calibration.focal_px:.7f} px",
            f"- 基线：{calibration.baseline_mm:.7f} mm",
            f"- fx×baseline：{calibration.fx_baseline_mm_px:.7f} mm·px",
            f"- 主点视差偏移 `cx_right-cx_left`：{calibration.disparity_offset_px:.7f} px",
            (
                f"- 深度公式：`Z_mm = {calibration.fx_baseline_mm_px:.7f} / "
                f"(disparity_px + {calibration.disparity_offset_px:.7f})`"
            ),
            f"- 共同评价 ROI：`x=[{x0},{x1})`, `y=[{y0},{y1})`",
            f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
            "",
            "![校正后左右图与水平极线](source_rectified_pair.png)",
            "",
            (
                "绿色水平线用于快速检查校正方向；当前 Orbbec YAML 标明输入已经由 SDK 校正，"
                "脚本因此直接使用图像，没有再次 remap。"
                if calibration.input_rectified
                else "绿色水平线用于快速检查脚本根据标定参数完成的双目校正。"
            ),
            "",
            "## 方法",
            "",
            (
                "每一组参数都实际调用仓库构建出的 `build/vitis_bm_cpu`。"
                "固定参数与仓库源码一致：`PREFILTER_XSOBEL`、preFilterCap=31、"
                "uniquenessRatio=15、textureThreshold=20、minDisparity=0。"
                "C++ 程序只输出按 Vitis testbench 规则缩放的 8 位视差图，"
                "所以脚本再以相同 OpenCV 参数计算一次 signed Q4 视差用于毫米深度换算，"
                f"并逐像素核对两者；本轮所有配置的最低一致率为 {exact_match:.4f}%。"
            ),
            "",
            (
                f"每组 C++ 核心 compute 计时重复 {args.runs} 次并报告中位数；"
                "不包括 Python 绘图和报告生成。共同 ROI 根据本轮最大 D 和最大 W 构造，"
                "仅对有效正视差计算深度统计。局部离群率只在完整有效的 5×5 邻域中统计。"
            ),
            "",
            "## 限制与稳健性说明",
            "",
            "- 数据只有一对画面且没有真值深度，无法报告 MAE、RMSE 或绝对精度；视觉效果与诊断指标应结合使用。",
            "- D/W 在 FPGA HLS 实现中通常是编译期配置；CPU 参数扫描不能推出 LUT、FF、BRAM、DSP、II 或 Fmax。",
            "- C++ CPU 时间受主机、OpenCV 构建、线程与系统负载影响，不能作为 FPGA 延时估计。",
            "- D 增大会扩展近距离搜索范围，也会扩大图像左侧的天然不可匹配区域，并通常增加硬件代价。",
            "- W 增大会增强局部聚合与平滑，但可能模糊细杆、电缆和物体遮挡边界。",
            "",
            "## 下一步建议",
            "",
            (
                f"先从本报告中选 2–3 个互补候选组合；建议保留 "
                f"D={default_candidate.disparities}/W={default_candidate.window} 作为折中基线，"
                + (
                    f"同时比较 W=9 的较小窗口折中，"
                    if 9 in args.window_values
                    else ""
                )
                + f"再加入 W={min(args.window_values)} 的保边缘方向和 "
                f"W={max(args.window_values)} 的平滑方向。"
                f"本轮光度指标最高的是 D={best_photo.disparities}/W={best_photo.window}。"
                "然后用更多近景、弱纹理、强遮挡样本复测。"
            ),
            "",
            (
                "若目标是评估 FPGA 性能，下一阶段应直接对对应 Vitis Vision HLS 顶层做 C 仿真/综合："
                "C 仿真验证像素输出；C 综合报告 LUT/FF/BRAM/DSP、目标时钟与估计 latency/II；"
                "暂时不上板也能完成这两步。只有板上运行才能最终验证 DDR/AXI、DMA 和端到端帧率。"
            ),
            "",
            "## 可复现运行",
            "",
            "交互式运行：",
            "",
            "```bash",
            f"cd \"{args.project_root}\"",
            "./scripts/run_stereobm_depth_sweep.py",
            "```",
            "",
            "本轮等价的非交互命令：",
            "",
            "```bash",
            "python3 scripts/run_stereobm_depth_sweep.py \\",
            f"  --non-interactive --dataset \"{args.dataset}\" \\",
            f"  --left \"{args.left}\" --right \"{args.right}\" \\",
            f"  --calibration \"{args.calibration}\" \\"
            if args.calibration
            else (
                f"  --focal-px {calibration.focal_px:.7f} "
                f"--baseline-mm {calibration.baseline_mm:.7f} \\\n"
            ),
            f"  --output \"{args.output}\" \\",
            f"  --disparities {list_text(args.disparity_values)} \\",
            f"  --windows {list_text(args.window_values)} \\",
            f"  --depth-min-mm {args.depth_min_mm:g} "
            f"--depth-max-mm {args.depth_max_mm:g} --runs {args.runs}",
            "```",
            "",
            "## 进一步问题",
            "",
            "- 最终 FPGA 型号、目标时钟和 Vitis/Vitis Vision 版本是什么？这些决定资源与时序结论。",
            "- 实际工作距离范围和最小目标尺寸是多少？它们决定 D 与 W 的优先级。",
            "- 后续是否有测距标靶或结构光/激光真值？有真值后才能把“视觉更平滑”与“深度更准确”区分开。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    try:
        args = resolve_inputs(parse_args())
        validate_paths_and_parameters(args)
        args.output.mkdir(parents=True, exist_ok=True)
        details_dir = args.output / "details"
        details_dir.mkdir(parents=True, exist_ok=True)

        left = read_gray(args.left)
        right = read_gray(args.right)
        if left.shape != right.shape:
            raise ValueError(
                f"left/right sizes differ: {left.shape[::-1]} versus {right.shape[::-1]}"
            )
        height, width = left.shape
        if max(args.disparity_values) >= width:
            raise ValueError("maximum D must be smaller than image width")
        if max(args.window_values) > min(width, height):
            raise ValueError("maximum W is larger than the image")

        calibration = load_calibration(
            args.calibration,
            (width, height),
            args.focal_px,
            args.baseline_mm,
            args.disparity_offset_px,
            args.assume_rectified,
        )
        rectified_left, rectified_right = rectify_pair(left, right, calibration)
        left_rectified_path = details_dir / "rectified_left.png"
        right_rectified_path = details_dir / "rectified_right.png"
        write_image(left_rectified_path, rectified_left)
        write_image(right_rectified_path, rectified_right)
        make_source_pair(
            rectified_left,
            rectified_right,
            args.output / "source_rectified_pair.png",
        )

        max_d = max(args.disparity_values)
        half_w = max(args.window_values) // 2
        x0 = max_d + half_w
        y0 = half_w
        x1 = width - half_w
        y1 = height - half_w
        if x0 >= x1 or y0 >= y1:
            raise ValueError("D/W values leave no common evaluation ROI")
        common_roi = np.zeros((height, width), dtype=bool)
        common_roi[y0:y1, x0:x1] = True

        binary, build_log = ensure_cpp_binary(args.project_root)
        if build_log:
            (args.output / "build.log").write_text(build_log, encoding="utf-8")

        cv2.setUseOptimized(False)
        configurations = [
            SweepConfig(disparities, window)
            for disparities in args.disparity_values
            for window in args.window_values
        ]
        results: list[Result] = []
        results_by_key: dict[str, Result] = {}
        colors_by_key: dict[str, np.ndarray] = {}

        print(
            f"\nRunning {len(configurations)} configurations on "
            f"{width}x{height} images..."
        )
        for index, config in enumerate(configurations, start=1):
            print(
                f"[{index:02d}/{len(configurations):02d}] "
                f"D={config.disparities}, W={config.window}",
                flush=True,
            )
            config_dir = details_dir / config.key
            config_dir.mkdir(parents=True, exist_ok=True)
            cpp_visual_path = config_dir / "cpp_disparity_visual.png"
            times, cpp_log = run_cpp_reference(
                binary,
                left_rectified_path,
                right_rectified_path,
                cpp_visual_path,
                config,
                args.runs,
            )
            (config_dir / "cpp_run.log").write_text(cpp_log, encoding="utf-8")

            matcher = create_matcher(config.disparities, config.window)
            disparity_q4 = matcher.compute(rectified_left, rectified_right)
            if disparity_q4.dtype != np.int16:
                raise RuntimeError(
                    f"unexpected StereoBM disparity dtype: {disparity_q4.dtype}"
                )
            expected_visual = cpp_visual_from_q4(
                disparity_q4, config.disparities
            )
            cpp_visual = read_gray(cpp_visual_path)
            if cpp_visual.shape != expected_visual.shape:
                raise RuntimeError("C++/Python visual image sizes differ")
            match_pct = safe_percent(
                int(np.count_nonzero(cpp_visual == expected_visual)),
                int(cpp_visual.size),
            )

            result, depth_mm, valid_depth = calculate_metrics(
                disparity_q4,
                rectified_left,
                rectified_right,
                config,
                calibration,
                common_roi,
                times,
                match_pct,
                str(config_dir.relative_to(args.output)),
            )
            colored = depth_color(
                depth_mm,
                valid_depth,
                args.depth_min_mm,
                args.depth_max_mm,
            )
            depth_u16 = np.zeros(depth_mm.shape, dtype=np.uint16)
            depth_u16[valid_depth] = np.clip(
                np.rint(depth_mm[valid_depth]), 1, np.iinfo(np.uint16).max
            ).astype(np.uint16)

            np.save(config_dir / "disparity_q4_int16.npy", disparity_q4)
            np.save(config_dir / "depth_mm_float32.npy", depth_mm)
            write_image(config_dir / "disparity_visual.png", expected_visual)
            write_image(config_dir / "depth_mm_u16.png", depth_u16)
            write_image(config_dir / "depth_color.png", colored)
            (config_dir / "metrics.json").write_text(
                json.dumps(asdict(result), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            results.append(result)
            results_by_key[config.key] = result
            colors_by_key[config.key] = colored

        make_depth_grid(
            configurations,
            results_by_key,
            colors_by_key,
            len(args.disparity_values),
            len(args.window_values),
            "StereoBM depth comparison: rows=D, columns=W",
            args.depth_min_mm,
            args.depth_max_mm,
            args.output / "comparison_all.png",
        )
        for disparities in args.disparity_values:
            row_configs = [
                config for config in configurations if config.disparities == disparities
            ]
            make_depth_grid(
                row_configs,
                results_by_key,
                colors_by_key,
                1,
                len(row_configs),
                f"StereoBM D={disparities}: horizontal W comparison",
                args.depth_min_mm,
                args.depth_max_mm,
                args.output / f"comparison_D{disparities:03d}.png",
            )
        make_metrics_chart(
            results,
            args.disparity_values,
            args.window_values,
            args.output / "metrics_comparison.png",
        )
        write_metrics_csv(results, args.output / "metrics.csv")

        metadata = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "project_root": str(args.project_root),
            "cpp_binary": str(binary),
            "dataset": str(args.dataset),
            "left": str(args.left),
            "right": str(args.right),
            "calibration_path": str(args.calibration) if args.calibration else None,
            "calibration": calibration.metadata(),
            "parameters": {
                "disparities": list(args.disparity_values),
                "windows": list(args.window_values),
                "depth_min_mm": args.depth_min_mm,
                "depth_max_mm": args.depth_max_mm,
                "runs": args.runs,
                "pre_filter_type": "PREFILTER_XSOBEL",
                "pre_filter_cap": PRE_FILTER_CAP,
                "uniqueness_ratio": UNIQUENESS_RATIO,
                "texture_threshold": TEXTURE_THRESHOLD,
                "min_disparity": MIN_DISPARITY,
            },
            "common_roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "software": {
                "python": sys.version.split()[0],
                "python_opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "results": [asdict(result) for result in results],
        }
        (args.output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rectification_mode = (
            "YAML 标明输入已校正，未二次 remap"
            if calibration.input_rectified
            else "脚本依据 YAML 完成 stereoRectify/remap"
        )
        report_path = args.output / "STEREOBM_DEPTH_COMPARISON.md"
        write_report(
            report_path,
            args,
            calibration,
            results,
            (x0, y0, x1, y1),
            rectification_mode,
        )

        print("\nCompleted.")
        print(f"Report:  {report_path}")
        print(f"Metrics: {args.output / 'metrics.csv'}")
        print(f"Details: {details_dir}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
