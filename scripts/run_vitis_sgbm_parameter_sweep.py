#!/usr/bin/env python3
"""Sweep the repository's Vitis scalar SGM reference and build a depth report.

The program under test is ``src/vitis_sgbm_cpu.cpp``: 5x5 Census cost, four
semi-global aggregation paths, integer-disparity winner-takes-all, and tunable
D/P1/P2.  It is not OpenCV StereoSGBM.

Two controlled comparisons are generated:

1. D=64/128/256 at fixed P1/P2=20/40.
2. D=128 at P1/P2=10/20, 20/40, 40/80, 10/40, and 20/80.

The reference kernel has no confidence or invalid-disparity output.  Therefore
the script preserves its raw result and separately runs a horizontally flipped
right-to-left audit.  "LR-consistent" metrics require agreement within one
integer pixel and are external diagnostics, not part of the Vitis kernel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from run_stereobm_depth_sweep import (
    DEFAULT_DATASET,
    Calibration,
    add_horizontal_colorbar,
    auto_detect_files,
    depth_color,
    finite_percentile,
    fit_panel_image,
    load_calibration,
    make_source_pair,
    prompt_text,
    put_centered,
    read_gray,
    rectify_pair,
    safe_percent,
    write_image,
)


DEFAULT_DISPARITIES = (64, 128, 256)
DEFAULT_RANGE_P1 = 20
DEFAULT_RANGE_P2 = 40
DEFAULT_PENALTY_DISPARITY = 128
DEFAULT_PENALTY_PAIRS = ((10, 20), (20, 40), (40, 80), (10, 40), (20, 80))
DEFAULT_DEPTH_MIN_MM = 300.0
DEFAULT_DEPTH_MAX_MM = 3000.0
DEFAULT_RUNS = 3
DEFAULT_LR_TOLERANCE_PX = 1

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class Configuration:
    disparities: int
    p1: int
    p2: int

    @property
    def key(self) -> str:
        return f"d{self.disparities:03d}_p1{self.p1:03d}_p2{self.p2:03d}"

    @property
    def title(self) -> str:
        return f"D={self.disparities}  P1/P2={self.p1}/{self.p2}"


@dataclass
class Result:
    disparities: int
    p1: int
    p2: int
    nearest_depth_mm: float
    raw_positive_coverage_pct: float
    raw_usable_coverage_pct: float
    disparity_saturation_pct: float
    lr_consistent_coverage_pct: float
    lr_acceptance_of_raw_pct: float
    raw_photometric_median_abs_error_gray: float
    raw_photometric_inlier_le15_pct: float
    lr_photometric_median_abs_error_gray: float
    lr_photometric_inlier_le15_pct: float
    raw_local_outlier_gt2px_pct: float
    lr_local_outlier_gt2px_pct: float
    lr_median_disparity_px: float
    lr_depth_p10_mm: float
    lr_depth_median_mm: float
    lr_depth_p90_mm: float
    cpp_compute_median_ms: float
    cpp_compute_min_ms: float
    cpp_compute_max_ms: float
    right_audit_compute_ms: float
    estimated_cpu_working_set_mib: float
    repeated_raw_output_match_pct: float
    cpp_raw_visual_match_pct: float
    common_roi_pixels: int
    lr_consistent_pixels: int
    config_directory: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled D and P1/P2 sweeps with the repository Vitis "
            "scalar SGM reference, then generate depth comparisons and a "
            "Markdown technical report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, help="Dataset directory")
    parser.add_argument("--left", type=Path, help="Left image")
    parser.add_argument("--right", type=Path, help="Right image")
    parser.add_argument("--calibration", type=Path, help="Calibration YAML")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument(
        "--disparities",
        help="Comma/space-separated D values for the range sweep",
    )
    parser.add_argument(
        "--range-p1",
        type=int,
        help="P1 held fixed in the D sweep",
    )
    parser.add_argument(
        "--range-p2",
        type=int,
        help="P2 held fixed in the D sweep",
    )
    parser.add_argument(
        "--penalty-disparity",
        type=int,
        help="D held fixed in the P1/P2 sweep",
    )
    parser.add_argument(
        "--penalty-pairs",
        help="Comma-separated P1/P2 pairs, for example 10/20,20/40,40/80",
    )
    parser.add_argument("--depth-min-mm", type=float)
    parser.add_argument("--depth-max-mm", type=float)
    parser.add_argument(
        "--runs",
        type=int,
        help="Repeated left-disparity C++ runs per unique configuration",
    )
    parser.add_argument(
        "--lr-tolerance-px",
        type=int,
        help="Maximum integer disparity disagreement for LR consistency",
    )
    parser.add_argument("--focal-px", type=float)
    parser.add_argument("--baseline-mm", type=float)
    parser.add_argument("--disparity-offset-px", type=float, default=0.0)
    parser.add_argument("--assume-rectified", action="store_true")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse complete per-configuration outputs in the selected output "
            "directory; only use with identical inputs and evaluation options"
        ),
    )
    return parser.parse_args()


def parse_int_list(text: str, label: str) -> tuple[int, ...]:
    values = []
    for token in re.split(r"[\s,，;；]+", text.strip()):
        if token:
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"{label} contains a non-integer: {token}") from exc
    if not values:
        raise ValueError(f"{label} cannot be empty")
    return tuple(dict.fromkeys(values))


def parse_penalty_pairs(text: str) -> tuple[tuple[int, int], ...]:
    pairs = []
    for token in re.split(r"[\s,，;；]+", text.strip()):
        if not token:
            continue
        pieces = re.split(r"[/：:]", token)
        if len(pieces) != 2:
            raise ValueError(f"invalid P1/P2 pair: {token}")
        try:
            pair = (int(pieces[0]), int(pieces[1]))
        except ValueError as exc:
            raise ValueError(f"invalid P1/P2 pair: {token}") from exc
        pairs.append(pair)
    if not pairs:
        raise ValueError("P1/P2 list cannot be empty")
    return tuple(dict.fromkeys(pairs))


def list_text(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def pairs_text(pairs: Sequence[tuple[int, int]]) -> str:
    return ",".join(f"{p1}/{p2}" for p1, p2 in pairs)


def resolve_inputs(args: argparse.Namespace) -> argparse.Namespace:
    interactive = args.interactive or (len(sys.argv) == 1 and not args.non_interactive)
    if args.interactive and args.non_interactive:
        raise ValueError("--interactive and --non-interactive cannot be combined")

    dataset = args.dataset or DEFAULT_DATASET
    if interactive:
        print("\nVitis-SGBM/SGM 参数扫描（回车采用方括号默认值）")
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

    output = args.output or dataset / "vitis_sgbm_parameter_sweep"
    d_text = args.disparities or list_text(DEFAULT_DISPARITIES)
    range_p1 = args.range_p1 if args.range_p1 is not None else DEFAULT_RANGE_P1
    range_p2 = args.range_p2 if args.range_p2 is not None else DEFAULT_RANGE_P2
    penalty_d = (
        args.penalty_disparity
        if args.penalty_disparity is not None
        else DEFAULT_PENALTY_DISPARITY
    )
    penalty_text = args.penalty_pairs or pairs_text(DEFAULT_PENALTY_PAIRS)
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
    lr_tolerance = (
        args.lr_tolerance_px
        if args.lr_tolerance_px is not None
        else DEFAULT_LR_TOLERANCE_PX
    )

    if interactive:
        output = Path(prompt_text("输出目录", str(output))).expanduser()
        d_text = prompt_text("D 范围扫描（逗号分隔）", d_text)
        range_p1 = int(prompt_text("D 扫描固定 P1", str(range_p1)))
        range_p2 = int(prompt_text("D 扫描固定 P2", str(range_p2)))
        penalty_d = int(prompt_text("惩罚参数扫描固定 D", str(penalty_d)))
        penalty_text = prompt_text("P1/P2 列表", penalty_text)
        depth_min = float(prompt_text("统一色标最近深度/mm", f"{depth_min:g}"))
        depth_max = float(prompt_text("统一色标最远深度/mm", f"{depth_max:g}"))
        runs = int(prompt_text("每组左视差 C++ 计时次数", str(runs)))
        lr_tolerance = int(
            prompt_text("左右一致性容差/px", str(lr_tolerance))
        )

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
    args.range_p1 = range_p1
    args.range_p2 = range_p2
    args.penalty_disparity = penalty_d
    args.penalty_pair_values = parse_penalty_pairs(penalty_text)
    args.depth_min_mm = depth_min
    args.depth_max_mm = depth_max
    args.runs = runs
    args.lr_tolerance_px = lr_tolerance
    args.project_root = args.project_root.resolve()
    return args


def validate_inputs(args: argparse.Namespace) -> None:
    for label, path in (("left image", args.left), ("right image", args.right)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.calibration and not args.calibration.is_file():
        raise FileNotFoundError(f"calibration YAML not found: {args.calibration}")
    if args.calibration is None and (
        args.focal_px is None or args.baseline_mm is None
    ):
        raise ValueError(
            "No calibration YAML found; provide --focal-px and --baseline-mm."
        )
    if args.depth_min_mm <= 0 or args.depth_max_mm <= args.depth_min_mm:
        raise ValueError("depth range must satisfy 0 < min < max")
    if not 1 <= args.runs <= 20:
        raise ValueError("--runs must be in [1, 20]")
    if not 0 <= args.lr_tolerance_px <= 10:
        raise ValueError("--lr-tolerance-px must be in [0, 10]")
    for disparity in (*args.disparity_values, args.penalty_disparity):
        if disparity <= 0 or disparity > 256:
            raise ValueError(f"D={disparity} must be in [1, 256]")
    for p1, p2 in (
        (args.range_p1, args.range_p2),
        *args.penalty_pair_values,
    ):
        if p1 < 0 or p2 <= p1:
            raise ValueError(
                f"penalties must satisfy 0 <= P1 < P2, got {p1}/{p2}"
            )


def build_configurations(
    args: argparse.Namespace,
) -> tuple[list[Configuration], list[Configuration], list[Configuration]]:
    range_configs = [
        Configuration(disparity, args.range_p1, args.range_p2)
        for disparity in args.disparity_values
    ]
    penalty_configs = [
        Configuration(args.penalty_disparity, p1, p2)
        for p1, p2 in args.penalty_pair_values
    ]
    unique: dict[str, Configuration] = {}
    for config in (*range_configs, *penalty_configs):
        unique.setdefault(config.key, config)
    return range_configs, penalty_configs, list(unique.values())


def ensure_cpp_binary(project_root: Path) -> tuple[Path, str]:
    binary = project_root / "build" / "vitis_sgbm_cpu"
    source = project_root / "src" / "vitis_sgbm_cpu.cpp"
    cmake_file = project_root / "CMakeLists.txt"
    needs_build = not binary.is_file()
    if binary.is_file():
        needs_build = binary.stat().st_mtime < max(
            source.stat().st_mtime, cmake_file.stat().st_mtime
        )
    logs = []
    if needs_build:
        for command in (
            ["cmake", "-S", str(project_root), "-B", str(project_root / "build")],
            ["cmake", "--build", str(project_root / "build"), "-j"],
        ):
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False
            )
            logs.append(
                "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr
            )
            if completed.returncode != 0:
                raise RuntimeError("failed to build vitis_sgbm_cpu:\n" + "\n".join(logs))
    if not binary.is_file():
        raise FileNotFoundError(f"missing SGM executable: {binary}")
    return binary, "\n".join(logs)


def visual_path_for(raw_path: Path) -> Path:
    suffix = raw_path.suffix if raw_path.suffix else ".png"
    return raw_path.with_name(raw_path.stem + "_visual" + suffix)


def read_raw_disparity(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"SGM executable did not produce: {path}")
    if image.shape != expected_shape or image.dtype != np.uint8:
        raise RuntimeError(
            f"unexpected raw disparity format at {path}: "
            f"shape={image.shape}, dtype={image.dtype}"
        )
    return image


def parse_logged_float(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"could not parse {label} from C++ output:\n{text}")
    return float(match.group(1))


def run_cpp_repeated(
    binary: Path,
    left_path: Path,
    right_path: Path,
    raw_path: Path,
    config: Configuration,
    runs: int,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, list[float], float, float, str]:
    command = [
        str(binary),
        str(left_path),
        str(right_path),
        str(raw_path),
        str(config.disparities),
        str(config.p1),
        str(config.p2),
    ]
    times = []
    reference: Optional[np.ndarray] = None
    minimum_match = 100.0
    memory_mib = float("nan")
    logs = []
    for index in range(runs):
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        section = (
            f"=== run {index + 1}/{runs} ===\n"
            f"$ {' '.join(command)}\n"
            f"{completed.stdout}{completed.stderr}"
        )
        logs.append(section)
        if completed.returncode != 0:
            raise RuntimeError(f"C++ SGM failed for {config.title}:\n{section}")
        times.append(
            parse_logged_float(
                r"CPU compute:\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
                completed.stdout,
                "CPU compute time",
            )
        )
        memory_mib = parse_logged_float(
            r"Est\. memory:\s*([0-9]+(?:\.[0-9]+)?)\s*MiB",
            completed.stdout,
            "estimated memory",
        )
        current = read_raw_disparity(raw_path, expected_shape)
        if reference is None:
            reference = current.copy()
        else:
            minimum_match = min(
                minimum_match,
                safe_percent(
                    int(np.count_nonzero(current == reference)), current.size
                ),
            )
    return reference, times, memory_mib, minimum_match, "\n".join(logs)


def run_cpp_once(
    binary: Path,
    left_path: Path,
    right_path: Path,
    raw_path: Path,
    config: Configuration,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, float, str]:
    command = [
        str(binary),
        str(left_path),
        str(right_path),
        str(raw_path),
        str(config.disparities),
        str(config.p1),
        str(config.p2),
    ]
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False
    )
    log = "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"right-to-left audit failed for {config.title}:\n{log}")
    runtime = parse_logged_float(
        r"CPU compute:\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
        completed.stdout,
        "right-audit CPU compute time",
    )
    return read_raw_disparity(raw_path, expected_shape), runtime, log


def expected_cpp_visual(raw: np.ndarray, disparities: int) -> np.ndarray:
    scale = 255.0 / float(disparities - 1) if disparities > 1 else 0.0
    return np.clip(np.rint(raw.astype(np.float64) * scale), 0, 255).astype(
        np.uint8
    )


def photometric_metrics(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    rows, cols = disparity.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(cols, dtype=np.float32),
        np.arange(rows, dtype=np.float32),
    )
    right_x = grid_x - disparity.astype(np.float32)
    photo_mask = mask & (right_x >= 0.0) & (right_x <= cols - 1.0)
    if not np.any(photo_mask):
        return float("nan"), float("nan")
    warped = cv2.remap(
        right,
        right_x,
        grid_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    error = cv2.absdiff(left, warped)[photo_mask]
    return float(np.median(error)), safe_percent(
        int(np.count_nonzero(error <= 15)), int(error.size)
    )


def local_outlier_percent(disparity: np.ndarray, mask: np.ndarray) -> float:
    dense = cv2.erode(
        mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)
    ).astype(bool)
    if not np.any(dense):
        return float("nan")
    local_median = cv2.medianBlur(disparity.astype(np.float32), 5)
    difference = np.abs(disparity.astype(np.float32) - local_median)
    values = difference[dense]
    return safe_percent(int(np.count_nonzero(values > 2.0)), int(values.size))


def calculate_result(
    left: np.ndarray,
    right: np.ndarray,
    left_disparity: np.ndarray,
    right_disparity: np.ndarray,
    common_roi: np.ndarray,
    calibration: Calibration,
    config: Configuration,
    cpp_times: Sequence[float],
    right_runtime: float,
    memory_mib: float,
    repeat_match_pct: float,
    visual_match_pct: float,
    lr_tolerance_px: int,
    config_directory: str,
) -> tuple[Result, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = left_disparity.shape
    yy, xx = np.indices((rows, cols))
    raw_positive = common_roi & (left_disparity > 0)
    raw_usable = (
        common_roi
        & (left_disparity > 0)
        & (left_disparity < config.disparities - 1)
    )
    saturated = common_roi & (left_disparity >= config.disparities - 1)

    right_x = xx - left_disparity.astype(np.int32)
    in_bounds = raw_usable & (right_x >= 0) & (right_x < cols)
    sampled_right = np.zeros(left_disparity.shape, dtype=np.uint8)
    sampled_right[in_bounds] = right_disparity[
        yy[in_bounds], right_x[in_bounds]
    ]
    lr_consistent = (
        in_bounds
        & (sampled_right > 0)
        & (sampled_right < config.disparities - 1)
        & (
            np.abs(
                left_disparity.astype(np.int16)
                - sampled_right.astype(np.int16)
            )
            <= lr_tolerance_px
        )
    )

    disparity_float = left_disparity.astype(np.float32)
    denominator = disparity_float + calibration.disparity_offset_px
    depth_mm = np.zeros(disparity_float.shape, dtype=np.float32)
    depth_defined = denominator > 0
    depth_mm[depth_defined] = (
        calibration.fx_baseline_mm_px / denominator[depth_defined]
    )
    lr_depth_values = depth_mm[lr_consistent]
    lr_disparities = disparity_float[lr_consistent]
    raw_photo_median, raw_photo_inlier = photometric_metrics(
        left, right, left_disparity, raw_usable
    )
    lr_photo_median, lr_photo_inlier = photometric_metrics(
        left, right, left_disparity, lr_consistent
    )
    roi_pixels = int(np.count_nonzero(common_roi))
    lr_pixels = int(np.count_nonzero(lr_consistent))
    raw_pixels = int(np.count_nonzero(raw_usable))
    nearest_depth = calibration.fx_baseline_mm_px / (
        config.disparities - 1 + calibration.disparity_offset_px
    )
    result = Result(
        disparities=config.disparities,
        p1=config.p1,
        p2=config.p2,
        nearest_depth_mm=float(nearest_depth),
        raw_positive_coverage_pct=safe_percent(
            int(np.count_nonzero(raw_positive)), roi_pixels
        ),
        raw_usable_coverage_pct=safe_percent(raw_pixels, roi_pixels),
        disparity_saturation_pct=safe_percent(
            int(np.count_nonzero(saturated)), roi_pixels
        ),
        lr_consistent_coverage_pct=safe_percent(lr_pixels, roi_pixels),
        lr_acceptance_of_raw_pct=safe_percent(lr_pixels, raw_pixels),
        raw_photometric_median_abs_error_gray=raw_photo_median,
        raw_photometric_inlier_le15_pct=raw_photo_inlier,
        lr_photometric_median_abs_error_gray=lr_photo_median,
        lr_photometric_inlier_le15_pct=lr_photo_inlier,
        raw_local_outlier_gt2px_pct=local_outlier_percent(
            left_disparity, raw_usable
        ),
        lr_local_outlier_gt2px_pct=local_outlier_percent(
            left_disparity, lr_consistent
        ),
        lr_median_disparity_px=finite_percentile(lr_disparities, 50),
        lr_depth_p10_mm=finite_percentile(lr_depth_values, 10),
        lr_depth_median_mm=finite_percentile(lr_depth_values, 50),
        lr_depth_p90_mm=finite_percentile(lr_depth_values, 90),
        cpp_compute_median_ms=float(statistics.median(cpp_times)),
        cpp_compute_min_ms=float(min(cpp_times)),
        cpp_compute_max_ms=float(max(cpp_times)),
        right_audit_compute_ms=float(right_runtime),
        estimated_cpu_working_set_mib=float(memory_mib),
        repeated_raw_output_match_pct=float(repeat_match_pct),
        cpp_raw_visual_match_pct=float(visual_match_pct),
        common_roi_pixels=roi_pixels,
        lr_consistent_pixels=lr_pixels,
        config_directory=config_directory,
    )
    return result, depth_mm, raw_positive, lr_consistent


def depth_u16(depth_mm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros(depth_mm.shape, dtype=np.uint16)
    output[mask] = np.clip(
        np.rint(depth_mm[mask]), 1, np.iinfo(np.uint16).max
    ).astype(np.uint16)
    return output


def make_depth_grid(
    configurations: Sequence[Configuration],
    results_by_key: dict[str, Result],
    colors_by_key: dict[str, np.ndarray],
    title: str,
    mode: str,
    depth_min_mm: float,
    depth_max_mm: float,
    output_path: Path,
) -> None:
    cell_width = 420
    image_height = 263
    header_height = 64
    footer_height = 62
    title_height = 64
    colorbar_height = 76
    cell_height = header_height + image_height + footer_height
    canvas = np.full(
        (
            title_height + cell_height + colorbar_height,
            cell_width * len(configurations),
            3,
        ),
        18,
        dtype=np.uint8,
    )
    put_centered(canvas, title, 41, 0.82, (245, 245, 245), 2)
    for index, config in enumerate(configurations):
        x0 = index * cell_width
        cell = canvas[
            title_height : title_height + cell_height,
            x0 : x0 + cell_width,
        ]
        result = results_by_key[config.key]
        put_centered(cell, config.title, 38, 0.66, (255, 255, 255), 2)
        cell[header_height : header_height + image_height, :] = fit_panel_image(
            colors_by_key[config.key], cell_width, image_height
        )
        if mode == "raw":
            stats = (
                f"raw {result.raw_usable_coverage_pct:.1f}%  "
                f"sat {result.disparity_saturation_pct:.2f}%  "
                f"CPU {result.cpp_compute_median_ms / 1000.0:.2f} s"
            )
        else:
            stats = (
                f"LR {result.lr_consistent_coverage_pct:.1f}%  "
                f"accept {result.lr_acceptance_of_raw_pct:.1f}%  "
                f"photo<=15 {result.lr_photometric_inlier_le15_pct:.1f}%"
            )
        put_centered(
            cell,
            stats,
            header_height + image_height + 38,
            0.48,
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
    add_horizontal_colorbar(
        canvas,
        title_height + cell_height + 11,
        depth_min_mm,
        depth_max_mm,
    )
    write_image(output_path, canvas)


def metric_limits(values: Iterable[float], percentage: bool) -> tuple[float, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    maximum = max(finite)
    if percentage:
        return 0.0, min(100.0, max(5.0, maximum * 1.12))
    return 0.0, max(1.0, maximum * 1.16)


def make_metrics_dashboard(
    configurations: Sequence[Configuration],
    results_by_key: dict[str, Result],
    panels: Sequence[tuple[str, str, bool, float]],
    title: str,
    category_labels: Sequence[str],
    output_path: Path,
) -> None:
    canvas = np.full((900, 1600, 3), 248, dtype=np.uint8)
    cv2.putText(
        canvas, title, (35, 45), FONT, 0.85, (25, 25, 25), 2, cv2.LINE_AA
    )
    palette = [
        (28, 91, 214),
        (54, 155, 61),
        (204, 99, 28),
        (145, 70, 180),
        (30, 160, 180),
        (170, 80, 80),
    ]
    for panel_index, (panel_title, field, percentage, divisor) in enumerate(panels):
        panel_row, panel_col = divmod(panel_index, 2)
        px = 35 + panel_col * 780
        py = 78 + panel_row * 400
        plot_x0, plot_x1 = px + 75, px + 735
        plot_y0, plot_y1 = py + 50, py + 325
        cv2.rectangle(canvas, (px, py), (px + 750, py + 370), (205, 205, 205), 1)
        cv2.putText(
            canvas,
            panel_title,
            (px + 18, py + 32),
            FONT,
            0.6,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        values = [
            float(getattr(results_by_key[config.key], field)) / divisor
            for config in configurations
        ]
        ymin, ymax = metric_limits(values, percentage)
        for tick in range(6):
            fraction = tick / 5.0
            y = int(round(plot_y1 - fraction * (plot_y1 - plot_y0)))
            value = ymin + fraction * (ymax - ymin)
            cv2.line(canvas, (plot_x0, y), (plot_x1, y), (224, 224, 224), 1)
            label = f"{value:.0f}" if ymax >= 10 else f"{value:.1f}"
            cv2.putText(
                canvas,
                label,
                (px + 12, y + 5),
                FONT,
                0.42,
                (65, 65, 65),
                1,
                cv2.LINE_AA,
            )
        slot = (plot_x1 - plot_x0) / max(len(configurations), 1)
        bar_width = max(18, int(slot * 0.58))
        for index, (config, label, value) in enumerate(
            zip(configurations, category_labels, values)
        ):
            center = int(round(plot_x0 + (index + 0.5) * slot))
            fraction = (value - ymin) / (ymax - ymin) if ymax > ymin else 0
            top = int(round(plot_y1 - fraction * (plot_y1 - plot_y0)))
            color = palette[index % len(palette)]
            cv2.rectangle(
                canvas,
                (center - bar_width // 2, top),
                (center + bar_width // 2, plot_y1),
                color,
                -1,
            )
            value_label = f"{value:.1f}" if value < 100 else f"{value:.0f}"
            size, _ = cv2.getTextSize(value_label, FONT, 0.4, 1)
            cv2.putText(
                canvas,
                value_label,
                (center - size[0] // 2, max(plot_y0 + 12, top - 8)),
                FONT,
                0.4,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )
            size, _ = cv2.getTextSize(label, FONT, 0.42, 1)
            cv2.putText(
                canvas,
                label,
                (center - size[0] // 2, plot_y1 + 25),
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


def write_metrics_csv(results: Sequence[Result], path: Path) -> None:
    fieldnames = list(asdict(results[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    calibration: Calibration,
    range_configs: Sequence[Configuration],
    penalty_configs: Sequence[Configuration],
    unique_configs: Sequence[Configuration],
    results_by_key: dict[str, Result],
    common_roi_bounds: tuple[int, int, int, int],
    rectification_mode: str,
) -> None:
    range_results = [results_by_key[config.key] for config in range_configs]
    penalty_results = [results_by_key[config.key] for config in penalty_configs]
    all_results = [results_by_key[config.key] for config in unique_configs]
    best_range = max(range_results, key=lambda item: item.lr_consistent_coverage_pct)
    best_penalty = max(
        penalty_results, key=lambda item: item.lr_consistent_coverage_pct
    )
    default = results_by_key.get(
        Configuration(
            args.penalty_disparity, args.range_p1, args.range_p2
        ).key,
        best_range,
    )
    d64 = results_by_key.get(Configuration(64, args.range_p1, args.range_p2).key)
    d128 = results_by_key.get(Configuration(128, args.range_p1, args.range_p2).key)
    d256 = results_by_key.get(Configuration(256, args.range_p1, args.range_p2).key)
    weak = results_by_key.get(Configuration(args.penalty_disparity, 10, 20).key)
    strong = results_by_key.get(Configuration(args.penalty_disparity, 40, 80).key)
    lower_p1 = results_by_key.get(Configuration(args.penalty_disparity, 10, 40).key)
    higher_p2 = results_by_key.get(Configuration(args.penalty_disparity, 20, 80).key)
    min_repeat_match = min(
        result.repeated_raw_output_match_pct for result in all_results
    )
    min_visual_match = min(result.cpp_raw_visual_match_pct for result in all_results)
    far_disparity_floor = max(
        1, int(math.floor(calibration.fx_baseline_mm_px / args.depth_max_mm))
    )
    far_integer_depth_step_mm = (
        calibration.fx_baseline_mm_px / far_disparity_floor
        - calibration.fx_baseline_mm_px / (far_disparity_floor + 1)
    )
    x0, y0, x1, y1 = common_roi_bounds

    lines = [
        "# Gemini335 Vitis-SGBM/SGM：D 与 P1/P2 参数效果对比",
        "",
        "## 技术摘要",
        "",
        (
            "本报告测试的是仓库 `src/vitis_sgbm_cpu.cpp`：Vitis `sgbm` "
            "测试台的标量 SGM 参考路径，采用 5×5 Census、4 方向路径聚合和"
            "整数视差 winner-takes-all；它不是 OpenCV `StereoSGBM`。"
        ),
        "",
        (
            f"实验包含 {len(unique_configs)} 个唯一配置：第一组固定 "
            f"P1/P2={args.range_p1}/{args.range_p2} 比较 "
            f"D={list_text(args.disparity_values)}；第二组固定 "
            f"D={args.penalty_disparity} 比较 "
            f"P1/P2={pairs_text(args.penalty_pair_values)}。"
            f"所有深度图沿用 BM 报告的 {args.depth_min_mm:g}–"
            f"{args.depth_max_mm:g} mm 固定色标和共同 ROI。"
        ),
        "",
        (
            f"在固定 P1/P2 的 D 扫描中，左右一致覆盖最高的是 "
            f"D={best_range.disparities}（{best_range.lr_consistent_coverage_pct:.1f}%）；"
            f"在 D={args.penalty_disparity} 的惩罚扫描中最高的是 "
            f"P1/P2={best_penalty.p1}/{best_penalty.p2}"
            f"（{best_penalty.lr_consistent_coverage_pct:.1f}%）。"
            "由于没有像素级真值，这些是一致性与对应关系诊断，不能解释为绝对深度精度。"
        ),
        "",
        "### 本次数据的直接结论",
        "",
    ]
    if d64 and d128:
        lines.append(
            f"- **D=64 对当前近景仍偏小。**其最近搜索边界约 "
            f"{d64.nearest_depth_mm:.1f} mm，顶端饱和率 "
            f"{d64.disparity_saturation_pct:.2f}%；D=128 将饱和降到 "
            f"{d128.disparity_saturation_pct:.2f}%，左右一致覆盖由 "
            f"{d64.lr_consistent_coverage_pct:.1f}% 提升到 "
            f"{d128.lr_consistent_coverage_pct:.1f}%。"
        )
    if d128 and d256:
        lines.append(
            f"- **D=256 在这帧上没有带来一致覆盖收益。**D=128 与 D=256 "
            f"的左右一致覆盖分别是 {d128.lr_consistent_coverage_pct:.1f}% 和 "
            f"{d256.lr_consistent_coverage_pct:.1f}%，但左视差中位计算时间从 "
            f"{d128.cpp_compute_median_ms / 1000.0:.2f} s 增至 "
            f"{d256.cpp_compute_median_ms / 1000.0:.2f} s，估算 CPU 工作集从 "
            f"{d128.estimated_cpu_working_set_mib:.1f} MiB 增至 "
            f"{d256.estimated_cpu_working_set_mib:.1f} MiB。"
        )
    lines.append(
        f"- **默认 D={default.disparities}, P1/P2={default.p1}/{default.p2} "
        f"仍是可复现基线。**其左右一致覆盖为 "
        f"{default.lr_consistent_coverage_pct:.1f}%，LR 光度≤15 占比为 "
        f"{default.lr_photometric_inlier_le15_pct:.1f}%，中位深度为 "
        f"{default.lr_depth_median_mm:.1f} mm。"
    )
    if lower_p1 and higher_p2:
        lines.append(
            f"- **一致性优先时可继续复测 10/40 与 20/80。**本帧中两者的"
            f"左右一致覆盖分别为 {lower_p1.lr_consistent_coverage_pct:.1f}% 和 "
            f"{higher_p2.lr_consistent_coverage_pct:.1f}%，高于默认 "
            f"{default.lr_consistent_coverage_pct:.1f}%；但 10/40 更允许相邻"
            "视差变化，20/80 更强地抑制大跳变，二者可能在斜面和遮挡边界产生不同偏差，"
            "无真值时不能仅凭覆盖率定优劣。"
        )
    if weak and strong:
        lines.append(
            f"- **成比例减弱到 10/20 会明显放大原始局部离群。**原始局部"
            f"离群率由默认的 {default.raw_local_outlier_gt2px_pct:.1f}% 变为 "
            f"{weak.raw_local_outlier_gt2px_pct:.1f}%；增强到 40/80 后为 "
            f"{strong.raw_local_outlier_gt2px_pct:.1f}%，但更强平滑也需要复查"
            "细杆、电缆和深度边界。"
        )
    lines.extend(
        [
            "",
            "## D=128 在当前场景形成最佳范围折中",
            "",
            "### 原始 SGM 输出",
            "",
            "![D 范围原始深度对比](comparison_D_range_raw.png)",
            "",
            (
                "原始图完整保留 C++ 核的整数视差输出，只把视差 0 画为黑色。"
                "该参考实现会对几乎每个像素强制选出一个最小代价视差，因此彩色覆盖完整"
                "不等于对应可靠；D=64 的红色近景饱和尤其需要结合下图判断。"
            ),
            "",
            "### 左右一致性审计后的深度",
            "",
            "![D 范围 LR 一致深度对比](comparison_D_range_lr.png)",
            "",
            (
                f"可信视图只保留左右视差差值≤{args.lr_tolerance_px} px、"
                "且不在 0/D−1 边界的像素。它是脚本额外运行的审计层，不是原 Vitis "
                "核的输出。D=128 相比 D=64 消除了主要近距截断；D=256 增加搜索歧义、"
                "时间和软件工作集，在这帧上没有提高 LR 一致覆盖。"
            ),
            "",
            "![D 范围诊断指标](metrics_D_range.png)",
            "",
            (
                "四个面板依次比较 LR 一致覆盖、视差顶端饱和、C++ 左视差计算时间和"
                "程序估算的 CPU 工作集。后两项只描述当前标量参考程序，不能当作 FPGA "
                "latency 或 BRAM/LUT 用量；但 D 近似线性扩大代价体，是后续 HLS 取舍的重要方向。"
            ),
            "",
            "## P1/P2 改变平滑与边界保留方式",
            "",
            "### 原始 SGM 输出",
            "",
            "![惩罚参数原始深度对比](comparison_penalties_raw.png)",
            "",
            (
                "该横向图固定 D=128。10/20、20/40、40/80 用于观察整体惩罚强度；"
                "10/40 在固定 P2 时降低相邻视差惩罚，20/80 在固定 P1 时提高大跳变惩罚。"
                "这五组不是一个单调标尺，必须按参数作用分别解释。"
            ),
            "",
            "### 左右一致性审计后的深度",
            "",
            "![惩罚参数 LR 一致深度对比](comparison_penalties_lr.png)",
            "",
            (
                "左右一致视图会剔除互相不支持的匹配。较弱的 10/20 在投影散斑和遮挡区"
                "出现更多不一致；10/40 与 20/80 在本帧覆盖较高，但一个偏向允许渐变，"
                "另一个偏向抑制突变。没有真值时，应在斜面连续性和物体边界上继续做多场景复测。"
            ),
            "",
            "![惩罚参数诊断指标](metrics_penalties.png)",
            "",
            (
                "惩罚参数图展示 LR 一致覆盖、LR 光度一致性、原始局部离群率和 C++ 时间。"
                "P1/P2 主要改变输出结构而不改变代价体尺寸，因此这些配置的 CPU 时间和估算"
                "工作集接近；未来 FPGA 资源仍需以对应编译期配置的 HLS 综合报告为准。"
            ),
            "",
            "## 完整指标表",
            "",
            "| D | P1/P2 | 最近范围/mm | 原始可用/% | 顶端饱和/% | LR一致覆盖/% | LR接收原始/% | LR光度≤15/% | 原始离群>2px/% | LR深度P10/P50/P90 mm | C++中位时间/s | CPU工作集/MiB | 重复输出一致/% |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for config in unique_configs:
        item = results_by_key[config.key]
        lines.append(
            f"| {item.disparities} | {item.p1}/{item.p2} | "
            f"{format_number(item.nearest_depth_mm)} | "
            f"{format_number(item.raw_usable_coverage_pct)} | "
            f"{format_number(item.disparity_saturation_pct, 2)} | "
            f"{format_number(item.lr_consistent_coverage_pct)} | "
            f"{format_number(item.lr_acceptance_of_raw_pct)} | "
            f"{format_number(item.lr_photometric_inlier_le15_pct)} | "
            f"{format_number(item.raw_local_outlier_gt2px_pct)} | "
            f"{format_number(item.lr_depth_p10_mm)}/"
            f"{format_number(item.lr_depth_median_mm)}/"
            f"{format_number(item.lr_depth_p90_mm)} | "
            f"{format_number(item.cpp_compute_median_ms / 1000.0, 3)} | "
            f"{format_number(item.estimated_cpu_working_set_mib)} | "
            f"{format_number(item.repeated_raw_output_match_pct, 4)} |"
        )
    lines.extend(
        [
            "",
            "逐配置原始视差、反向审计视差、LR 掩膜、毫米深度、彩色图和运行日志见 "
            "[`details/`](details/)。机器可读结果见 [`metrics.csv`](metrics.csv) "
            "与 [`metadata.json`](metadata.json)。",
            "",
            "## 数据、标定与指标定义",
            "",
            f"- 左图：`{args.left}`",
            f"- 右图：`{args.right}`",
            f"- 标定来源：`{calibration.source}`",
            f"- 输入尺寸：{calibration.image_width}×{calibration.image_height}",
            f"- 校正方式：{rectification_mode}",
            f"- 焦距：{calibration.focal_px:.7f} px",
            f"- 基线：{calibration.baseline_mm:.7f} mm",
            f"- fx×baseline：{calibration.fx_baseline_mm_px:.7f} mm·px",
            f"- 主点视差偏移：{calibration.disparity_offset_px:.7f} px",
            (
                f"- 深度公式：`Z_mm={calibration.fx_baseline_mm_px:.7f}/"
                f"(d+{calibration.disparity_offset_px:.7f})`"
            ),
            f"- 共同 ROI：`x=[{x0},{x1})`, `y=[{y0},{y1})`，与 BM 报告一致",
            f"- LR 一致容差：≤{args.lr_tolerance_px} 个整数视差像素",
            "- 原始可用：ROI 内 `0 < d < D−1`；D−1 单独记为顶端饱和",
            "- LR 一致覆盖：原始可用且反向视差有效，并满足左右差值容差",
            (
                f"- 彩色色标：{args.depth_min_mm:g}–{args.depth_max_mm:g} mm；"
                "范围外深度仅在图中截到端点颜色，CSV/表格保留实际计算值"
            ),
            "- 光度≤15：右图按左视差回投后，灰度绝对误差不超过 15 的比例",
            "- 局部离群>2 px：完整有效 5×5 邻域内，相对局部中位数偏差超过 2 px",
            f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
            "",
            "![校正后左右图](source_rectified_pair.png)",
            "",
            (
                "绿色水平线用于检查极线方向。Orbbec YAML 标记该输入已经由 SDK 校正，"
                "所以本轮不执行二次 remap。"
                if calibration.input_rectified
                else "绿色水平线用于检查脚本根据 YAML 完成的双目校正。"
            ),
            "",
            "## 方法与可复现性",
            "",
            (
                "每个唯一配置均直接调用 `build/vitis_sgbm_cpu`。左视差运行 "
                f"{args.runs} 次，报告核心 compute 中位数并逐像素比较重复输出；"
                "反向审计再运行一次水平翻转后的 right→left 输入。"
                f"所有配置的最低重复一致率为 {min_repeat_match:.4f}%，"
                f"原始图与 C++ 自动生成可视化的最低一致率为 {min_visual_match:.4f}%。"
            ),
            "",
            (
                "右向审计构造方式是：水平翻转右图作为第一输入、水平翻转左图作为第二输入，"
                "运行同一 C++ 核后再水平翻回。这样程序仍然只执行 `x-d` 搜索，"
                "但得到右图坐标上的正视差，可用于 `d_left(x)` 与 `d_right(x-d)` 比较。"
            ),
            "",
            (
                "本报告与 [StereoBM D/W 对比](../gemini335_1280x800_bm_sweep/"
                "STEREOBM_DEPTH_COMPARISON.md) 使用同一图像、标定、深度色标和共同 ROI。"
                "不过 BM 是 OpenCV Q4 亚像素参考，当前 SGM 输出整数视差，二者量化精度与"
                "无效像素语义不同，不能仅按彩色覆盖直接排名。"
            ),
            "",
            "本轮非交互复现命令：",
            "",
            "```bash",
            f"cd \"{args.project_root}\"",
            "python3 scripts/run_vitis_sgbm_parameter_sweep.py \\",
            f"  --non-interactive --dataset \"{args.dataset}\" \\",
            f"  --calibration \"{args.calibration}\" \\"
            if args.calibration
            else (
                f"  --focal-px {calibration.focal_px:.7f} "
                f"--baseline-mm {calibration.baseline_mm:.7f} \\\n"
            ),
            f"  --output \"{args.output}\" \\",
            f"  --disparities {list_text(args.disparity_values)} \\",
            f"  --range-p1 {args.range_p1} --range-p2 {args.range_p2} \\",
            f"  --penalty-disparity {args.penalty_disparity} \\",
            f"  --penalty-pairs {pairs_text(args.penalty_pair_values)} \\",
            f"  --depth-min-mm {args.depth_min_mm:g} "
            f"--depth-max-mm {args.depth_max_mm:g} \\",
            f"  --runs {args.runs} --lr-tolerance-px {args.lr_tolerance_px}",
            "```",
            "",
            "## 限制、稳健性与不能推出的结论",
            "",
            "- 本数据只有一对画面且没有真值深度，无法报告 MAE、RMSE、P95 或毫米级绝对精度。",
            "- LR 一致性会排除许多遮挡和错误匹配，但左右双方共同犯错仍可能通过；它不是精度真值。",
            (
                f"- 当前输出是整数视差。{args.depth_max_mm / 1000.0:g} m "
                f"附近相邻整数视差对应约 {far_integer_depth_step_mm:.1f} mm "
                "的深度台阶；这只是量化间隔，不包含匹配和标定误差。"
            ),
            "- 原始核没有 uniqueness、置信度、speckle、LR check 或亚像素拟合；报告中的 LR 掩膜是外部审计。",
            "- 4 路径标量 CPU 时间和三份完整代价体的软件工作集不能作为 FPGA 延时或 BRAM/LUT/FF/DSP 用量。",
            "- P1/P2 在本实现中是全图常数，没有按图像梯度自适应；强平滑可能跨越真实深度边界。",
            "",
            "## 建议的下一步",
            "",
            (
                f"先保留 D={default.disparities}, P1/P2={default.p1}/{default.p2} "
                "作为 Vitis 默认基线，再把本帧 LR 一致覆盖较高的 "
                f"{best_penalty.p1}/{best_penalty.p2} 纳入候选。"
                "下一轮至少加入近距离、弱纹理、细杆/电缆、强遮挡和斜面场景，并固定真实距离标靶。"
            ),
            "",
            (
                "暂时不上板时，效果验证继续使用 C 仿真；资源与时序必须对与候选完全一致的 "
                "D、P1/P2、NUM_DIR、PARALLEL_UNITS 和图像上限做 HLS C 综合，记录 "
                "LUT/FF/BRAM/DSP、II、目标时钟和 estimated latency。"
            ),
            "",
            "## 进一步问题",
            "",
            "- 实际最近/最远工作距离和最小目标尺寸是多少？它们决定 D 与整数视差是否足够。",
            "- FPGA 型号、目标频率、Vitis Vision 版本、NUM_DIR 与 PARALLEL_UNITS 计划值是什么？",
            "- 是否能补充已知距离平面或深度真值？有真值后才能判断 10/40 与 20/80 哪个更准确。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    try:
        args = resolve_inputs(parse_args())
        validate_inputs(args)
        range_configs, penalty_configs, unique_configs = build_configurations(args)
        args.output.mkdir(parents=True, exist_ok=True)
        details_dir = args.output / "details"
        details_dir.mkdir(parents=True, exist_ok=True)

        left = read_gray(args.left)
        right = read_gray(args.right)
        if left.shape != right.shape:
            raise ValueError(
                f"left/right sizes differ: {left.shape[::-1]} vs {right.shape[::-1]}"
            )
        height, width = left.shape
        if max(config.disparities for config in unique_configs) >= width:
            raise ValueError("maximum D must be smaller than image width")
        calibration = load_calibration(
            args.calibration,
            (width, height),
            args.focal_px,
            args.baseline_mm,
            args.disparity_offset_px,
            args.assume_rectified,
        )
        rectified_left, rectified_right = rectify_pair(left, right, calibration)
        left_path = details_dir / "rectified_left.png"
        right_path = details_dir / "rectified_right.png"
        flipped_right_path = details_dir / "audit_flipped_right.png"
        flipped_left_path = details_dir / "audit_flipped_left.png"
        write_image(left_path, rectified_left)
        write_image(right_path, rectified_right)
        write_image(flipped_right_path, cv2.flip(rectified_right, 1))
        write_image(flipped_left_path, cv2.flip(rectified_left, 1))
        make_source_pair(
            rectified_left,
            rectified_right,
            args.output / "source_rectified_pair.png",
        )

        # Match the BM report exactly: max D=256 plus a 10-pixel comparison
        # border.  This is more conservative than the SGM Census radius and
        # permits direct cross-report diagnostics on the same pixel population.
        max_d = max(config.disparities for config in unique_configs)
        border = 10
        x0, y0 = max_d + border, border
        x1, y1 = width - border, height - border
        if x0 >= x1 or y0 >= y1:
            raise ValueError("configuration leaves no common evaluation ROI")
        common_roi = np.zeros((height, width), dtype=bool)
        common_roi[y0:y1, x0:x1] = True

        binary, build_log = ensure_cpp_binary(args.project_root)
        if build_log:
            (args.output / "build.log").write_text(build_log, encoding="utf-8")

        results: list[Result] = []
        results_by_key: dict[str, Result] = {}
        raw_colors: dict[str, np.ndarray] = {}
        lr_colors: dict[str, np.ndarray] = {}

        print(
            f"\nRunning {len(unique_configs)} unique Vitis-SGM configurations "
            f"on {width}x{height}; {args.runs} left runs + 1 right audit each..."
        )
        for index, config in enumerate(unique_configs, start=1):
            config_dir = details_dir / config.key
            config_dir.mkdir(parents=True, exist_ok=True)
            cached_paths = (
                config_dir / "metrics.json",
                config_dir / "depth_raw_color.png",
                config_dir / "depth_lr_color.png",
                config_dir / "left_disparity_raw.png",
                config_dir / "right_disparity_raw.png",
                config_dir / "lr_consistency_mask.png",
            )
            if args.resume and all(path.is_file() for path in cached_paths):
                cached = json.loads(cached_paths[0].read_text(encoding="utf-8"))
                if (
                    int(cached["disparities"]) != config.disparities
                    or int(cached["p1"]) != config.p1
                    or int(cached["p2"]) != config.p2
                ):
                    raise RuntimeError(
                        f"cached metrics/config mismatch in {config_dir}"
                    )
                raw_color = cv2.imread(
                    str(cached_paths[1]), cv2.IMREAD_COLOR
                )
                lr_color = cv2.imread(
                    str(cached_paths[2]), cv2.IMREAD_COLOR
                )
                if (
                    raw_color is None
                    or lr_color is None
                    or raw_color.shape[:2] != (height, width)
                    or lr_color.shape[:2] != (height, width)
                ):
                    raise RuntimeError(
                        f"cached color images are invalid in {config_dir}"
                    )
                result = Result(**cached)
                results.append(result)
                results_by_key[config.key] = result
                raw_colors[config.key] = raw_color
                lr_colors[config.key] = lr_color
                print(
                    f"[{index:02d}/{len(unique_configs):02d}] "
                    f"{config.title} [reused complete output]",
                    flush=True,
                )
                continue

            print(
                f"[{index:02d}/{len(unique_configs):02d}] {config.title}",
                flush=True,
            )
            left_raw_path = config_dir / "left_disparity_raw.png"
            (
                left_disparity,
                cpp_times,
                memory_mib,
                repeat_match_pct,
                cpp_log,
            ) = run_cpp_repeated(
                binary,
                left_path,
                right_path,
                left_raw_path,
                config,
                args.runs,
                (height, width),
            )
            (config_dir / "left_cpp_runs.log").write_text(
                cpp_log, encoding="utf-8"
            )

            right_flipped_raw_path = (
                config_dir / "right_audit_flipped_disparity_raw.png"
            )
            right_flipped, right_runtime, right_log = run_cpp_once(
                binary,
                flipped_right_path,
                flipped_left_path,
                right_flipped_raw_path,
                config,
                (height, width),
            )
            (config_dir / "right_audit_cpp_run.log").write_text(
                right_log, encoding="utf-8"
            )
            right_disparity = cv2.flip(right_flipped, 1)
            write_image(
                config_dir / "right_disparity_raw.png", right_disparity
            )

            cpp_visual = read_raw_disparity(
                visual_path_for(left_raw_path), (height, width)
            )
            expected_visual = expected_cpp_visual(
                left_disparity, config.disparities
            )
            visual_match_pct = safe_percent(
                int(np.count_nonzero(cpp_visual == expected_visual)),
                int(cpp_visual.size),
            )
            result, depth_mm, raw_positive, lr_consistent = calculate_result(
                rectified_left,
                rectified_right,
                left_disparity,
                right_disparity,
                common_roi,
                calibration,
                config,
                cpp_times,
                right_runtime,
                memory_mib,
                repeat_match_pct,
                visual_match_pct,
                args.lr_tolerance_px,
                str(config_dir.relative_to(args.output)),
            )

            raw_color = depth_color(
                depth_mm,
                raw_positive,
                args.depth_min_mm,
                args.depth_max_mm,
            )
            lr_color = depth_color(
                depth_mm,
                lr_consistent,
                args.depth_min_mm,
                args.depth_max_mm,
            )
            np.save(config_dir / "depth_mm_float32.npy", depth_mm)
            write_image(
                config_dir / "depth_raw_mm_u16.png",
                depth_u16(depth_mm, raw_positive),
            )
            write_image(
                config_dir / "depth_lr_mm_u16.png",
                depth_u16(depth_mm, lr_consistent),
            )
            write_image(config_dir / "depth_raw_color.png", raw_color)
            write_image(config_dir / "depth_lr_color.png", lr_color)
            write_image(
                config_dir / "lr_consistency_mask.png",
                lr_consistent.astype(np.uint8) * 255,
            )
            (config_dir / "metrics.json").write_text(
                json.dumps(asdict(result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            results.append(result)
            results_by_key[config.key] = result
            raw_colors[config.key] = raw_color
            lr_colors[config.key] = lr_color

        for configurations, stem, title in (
            (
                range_configs,
                "D_range",
                f"Vitis-SGM D sweep: P1/P2={args.range_p1}/{args.range_p2}",
            ),
            (
                penalty_configs,
                "penalties",
                f"Vitis-SGM penalty sweep: D={args.penalty_disparity}",
            ),
        ):
            make_depth_grid(
                configurations,
                results_by_key,
                raw_colors,
                title + " (raw kernel output)",
                "raw",
                args.depth_min_mm,
                args.depth_max_mm,
                args.output / f"comparison_{stem}_raw.png",
            )
            make_depth_grid(
                configurations,
                results_by_key,
                lr_colors,
                title + f" (LR consistent <= {args.lr_tolerance_px}px)",
                "lr",
                args.depth_min_mm,
                args.depth_max_mm,
                args.output / f"comparison_{stem}_lr.png",
            )

        make_metrics_dashboard(
            range_configs,
            results_by_key,
            (
                ("LR-consistent coverage (%)", "lr_consistent_coverage_pct", True, 1.0),
                ("Max-disparity saturation (%)", "disparity_saturation_pct", True, 1.0),
                ("Left C++ compute median (s)", "cpp_compute_median_ms", False, 1000.0),
                ("Estimated CPU working set (MiB)", "estimated_cpu_working_set_mib", False, 1.0),
            ),
            "Vitis-SGM disparity-range diagnostics",
            [f"D{config.disparities}" for config in range_configs],
            args.output / "metrics_D_range.png",
        )
        make_metrics_dashboard(
            penalty_configs,
            results_by_key,
            (
                ("LR-consistent coverage (%)", "lr_consistent_coverage_pct", True, 1.0),
                ("LR photometric inliers <=15 (%)", "lr_photometric_inlier_le15_pct", True, 1.0),
                ("Raw local outliers >2px (%)", "raw_local_outlier_gt2px_pct", True, 1.0),
                ("Left C++ compute median (s)", "cpp_compute_median_ms", False, 1000.0),
            ),
            f"Vitis-SGM P1/P2 diagnostics at D={args.penalty_disparity}",
            [f"{config.p1}/{config.p2}" for config in penalty_configs],
            args.output / "metrics_penalties.png",
        )

        write_metrics_csv(results, args.output / "metrics.csv")
        chart_map = [
            {
                "section": "D range raw output",
                "question": "How does D change the raw metric-depth result?",
                "type": "horizontal small multiples",
                "fields": ["D", "raw depth"],
                "palette": "fixed 300-3000 mm TURBO reversed; black=zero disparity",
                "artifact": "comparison_D_range_raw.png",
            },
            {
                "section": "D range LR audit",
                "question": "How much depth survives bidirectional consistency?",
                "type": "horizontal small multiples",
                "fields": ["D", "LR-consistent depth"],
                "palette": "same fixed depth scale; black=inconsistent/invalid",
                "artifact": "comparison_D_range_lr.png",
            },
            {
                "section": "Penalty raw output",
                "question": "How do P1/P2 choices change raw depth structure?",
                "type": "horizontal small multiples",
                "fields": ["P1", "P2", "raw depth"],
                "palette": "same fixed depth scale",
                "artifact": "comparison_penalties_raw.png",
            },
            {
                "section": "Penalty LR audit",
                "question": "Which P1/P2 choices retain bidirectional support?",
                "type": "horizontal small multiples",
                "fields": ["P1", "P2", "LR-consistent depth"],
                "palette": "same fixed depth scale",
                "artifact": "comparison_penalties_lr.png",
            },
        ]
        metadata = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "project_root": str(args.project_root),
            "cpp_binary": str(binary),
            "algorithm": {
                "name": "Vitis scalar SGM CPU reference",
                "source": "src/vitis_sgbm_cpu.cpp",
                "census_window": "5x5",
                "paths": 4,
                "disparity_output": "uint8 integer pixels",
                "not_opencv_stereosgbm": True,
            },
            "dataset": str(args.dataset),
            "left": str(args.left),
            "right": str(args.right),
            "calibration_path": str(args.calibration) if args.calibration else None,
            "calibration": calibration.metadata(),
            "parameters": {
                "range_disparities": list(args.disparity_values),
                "range_p1": args.range_p1,
                "range_p2": args.range_p2,
                "penalty_disparity": args.penalty_disparity,
                "penalty_pairs": [list(pair) for pair in args.penalty_pair_values],
                "depth_min_mm": args.depth_min_mm,
                "depth_max_mm": args.depth_max_mm,
                "left_runs": args.runs,
                "right_audit_runs": 1,
                "lr_tolerance_px": args.lr_tolerance_px,
                "resume_used": args.resume,
            },
            "common_roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "software": {
                "python": sys.version.split()[0],
                "python_opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "chart_map": chart_map,
            "results": [asdict(result) for result in results],
        }
        (args.output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rectification_mode = (
            "YAML 标明输入已校正，未二次 remap"
            if calibration.input_rectified
            else "脚本依据 YAML 完成 stereoRectify/remap"
        )
        report_path = args.output / "VITIS_SGBM_PARAMETER_COMPARISON.md"
        write_report(
            report_path,
            args,
            calibration,
            range_configs,
            penalty_configs,
            unique_configs,
            results_by_key,
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
