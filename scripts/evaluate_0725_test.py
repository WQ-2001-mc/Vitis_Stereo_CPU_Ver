#!/usr/bin/env python3
"""Generate a BM/Vitis-SGM metric-depth comparison for one stereo pair.

The pair has no per-pixel ground-truth depth.  The reported values are therefore
correspondence and coverage diagnostics, not absolute depth-accuracy metrics.
Both algorithms use D=128 and the same calibration-derived 0.3-3.0 m color
scale so that their output can be compared directly.  Inputs can be discovered
from a dataset directory or supplied as explicit left/right/calibration paths.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from evaluate_0724_depth import (
    BM_PRE_FILTER_CAP,
    BM_TEXTURE_THRESHOLD,
    BM_UNIQUENESS_RATIO,
    DEPTH_DISPLAY_MAX_MM,
    DEPTH_DISPLAY_MIN_MM,
    DEPTH_VALID_MAX_MM,
    DEPTH_VALID_MIN_MM,
    EVALUATION_ROI_XYXY,
    PHOTO_INLIER_THRESHOLD,
    Configuration,
    colorize_depth,
    compute_metrics,
    create_bm,
    depth_from_disparity,
    determine_order,
    make_comparison,
    make_rectification_check,
    parse_logged_float,
    run_sgm,
    sanitize_json,
    save_metrics,
    write_depth_mm,
)
from evaluate_camera_dataset import (
    Calibration,
    read_calibration,
    read_gray,
    rectify_pair,
)


DEFAULT_DATASET = Path(
    "/home/hcc/Desktop/Public/datasets/自研相机/0725-test"
)
DEFAULT_PROJECT = Path("/home/hcc/Desktop/HXB/Vitis_Stereo_CPU_Ver")

BM_CONFIG = Configuration(
    "bm_d128_b11", "BM", disparities=128, block_size=11
)
SGM_CONFIG = Configuration(
    "sgm_d128_p20_p40",
    "Vitis-SGM",
    disparities=128,
    p1=20,
    p2=40,
)
CONFIGURATIONS = (BM_CONFIG, SGM_CONFIG)
IMAGE_TIMESTAMP_PATTERN = re.compile(
    r"Image_(\d{8})(\d{6})(\d{3})(?:\D|$)"
)


@dataclass
class StereoInputs:
    mode: str
    dataset: Path | None
    calibration_path: Path
    calibration: Calibration
    left_source: Path
    right_source: Path
    rectified_left: np.ndarray
    rectified_right: np.ndarray
    order_candidates: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Directory mode: discover exactly two *.bmp images and one *.yaml "
            f"calibration, then determine left/right order (default: {DEFAULT_DATASET})"
        ),
    )
    parser.add_argument(
        "--left-image",
        "--left",
        dest="left_image",
        type=Path,
        default=None,
        help="Explicit mode: path to the already identified left image",
    )
    parser.add_argument(
        "--right-image",
        "--right",
        dest="right_image",
        type=Path,
        default=None,
        help="Explicit mode: path to the already identified right image",
    )
    parser.add_argument(
        "--calibration",
        "--calibration-file",
        dest="calibration",
        type=Path,
        default=None,
        help="Explicit mode: path to the OpenCV stereo-calibration YAML",
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: PROJECT/results/0725_test)",
    )
    parser.add_argument(
        "--bm-exe",
        type=Path,
        default=None,
        help="Extracted Vitis/OpenCV StereoBM CPU executable",
    )
    parser.add_argument(
        "--sgm-exe",
        type=Path,
        default=None,
        help="Extracted Vitis scalar SGM CPU executable",
    )
    args = parser.parse_args()
    explicit_paths = (args.left_image, args.right_image, args.calibration)
    if any(path is not None for path in explicit_paths) and not all(
        path is not None for path in explicit_paths
    ):
        parser.error(
            "--left-image, --right-image and --calibration must be provided together"
        )
    if args.left_image is not None and (
        args.left_image.resolve() == args.right_image.resolve()
    ):
        parser.error("--left-image and --right-image must be different files")
    return args


def filename_timestamp(path: Path) -> datetime | None:
    match = IMAGE_TIMESTAMP_PATTERN.search(path.stem)
    if match is None:
        return None
    date, clock, milliseconds = match.groups()
    return datetime.strptime(
        date + clock + milliseconds, "%Y%m%d%H%M%S%f"
    )


def filename_timestamp_gap_seconds(left: Path, right: Path) -> float | None:
    left_time = filename_timestamp(left)
    right_time = filename_timestamp(right)
    if left_time is None or right_time is None:
        return None
    return abs((left_time - right_time).total_seconds())


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Missing {label}: {resolved}")
    return resolved


def load_inputs(args: argparse.Namespace) -> StereoInputs:
    if args.left_image is not None:
        left_source = require_file(args.left_image, "left image")
        right_source = require_file(args.right_image, "right image")
        calibration_path = require_file(args.calibration, "calibration file")
        calibration = read_calibration(calibration_path)
        expected_size = (calibration.width, calibration.height)
        left_image = read_gray(left_source, expected_size)
        right_image = read_gray(right_source, expected_size)
        rectified_left, rectified_right = rectify_pair(
            left_image, right_image, calibration
        )
        return StereoInputs(
            mode="explicit_paths",
            dataset=None,
            calibration_path=calibration_path,
            calibration=calibration,
            left_source=left_source,
            right_source=right_source,
            rectified_left=rectified_left,
            rectified_right=rectified_right,
            order_candidates=[],
        )

    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        raise RuntimeError(f"Missing dataset directory: {dataset}")
    calibration_paths = sorted(dataset.glob("*.yaml"))
    image_paths = sorted(dataset.glob("*.bmp"))
    if len(calibration_paths) != 1:
        raise RuntimeError(
            f"Expected one YAML calibration, found {len(calibration_paths)}"
        )
    calibration_path = calibration_paths[0]
    calibration = read_calibration(calibration_path)
    (
        left_source,
        right_source,
        rectified_left,
        rectified_right,
        order_candidates,
    ) = determine_order(image_paths, calibration)
    return StereoInputs(
        mode="dataset_auto_order",
        dataset=dataset,
        calibration_path=calibration_path,
        calibration=calibration,
        left_source=left_source,
        right_source=right_source,
        rectified_left=rectified_left,
        rectified_right=rectified_right,
        order_candidates=order_candidates,
    )


def run_bm(
    executable: Path,
    left_path: Path,
    right_path: Path,
    details_dir: Path,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, float, float, str]:
    """Run the C++ BM program and retain its otherwise-discarded Q4 disparity."""
    visual_path = details_dir / "bm_d128_b11_disparity_visual.png"
    completed = subprocess.run(
        [
            str(executable),
            str(left_path),
            str(right_path),
            str(visual_path),
            str(BM_CONFIG.disparities),
            str(BM_CONFIG.block_size),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    runtime_ms = parse_logged_float(
        r"CPU compute:\s+([0-9.]+) ms", completed.stdout
    )

    # The C++ reference writes only an 8-bit display image.  Recompute with the
    # same OpenCV implementation and settings to retain Q4 disparity for metric
    # depth conversion, then verify that its 8-bit rendering matches C++.
    cv2.setUseOptimized(False)
    matcher = create_bm(BM_CONFIG.disparities, int(BM_CONFIG.block_size))
    disparity_q4 = matcher.compute(left, right)
    disparity = disparity_q4.astype(np.float32) / 16.0

    q4_output = np.zeros(disparity_q4.shape, dtype=np.uint16)
    positive = disparity_q4 > 0
    q4_output[positive] = disparity_q4[positive].astype(np.uint16)
    q4_path = details_dir / "bm_d128_b11_disparity_q4_u16.png"
    if not cv2.imwrite(str(q4_path), q4_output):
        raise RuntimeError(f"Cannot write BM Q4 disparity: {q4_path}")

    cpp_visual = cv2.imread(str(visual_path), cv2.IMREAD_GRAYSCALE)
    if cpp_visual is None:
        raise RuntimeError(f"BM executable did not produce {visual_path}")
    visual_scale = 256.0 / BM_CONFIG.disparities / 16.0
    reference_visual = np.clip(
        np.rint(disparity_q4.astype(np.float32) * visual_scale),
        0.0,
        255.0,
    ).astype(np.uint8)
    visual_match_percent = 100.0 * float(
        np.mean(cpp_visual == reference_visual)
    )
    return disparity, runtime_ms, visual_match_percent, completed.stdout


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else project / "results" / "0725_test"
    )
    details_dir = output_dir / "details"
    bm_exe = (
        args.bm_exe.resolve()
        if args.bm_exe is not None
        else project / "build" / "vitis_bm_cpu"
    )
    sgm_exe = (
        args.sgm_exe.resolve()
        if args.sgm_exe is not None
        else project / "build" / "vitis_sgbm_cpu"
    )

    for executable in (bm_exe, sgm_exe):
        if not executable.is_file():
            raise RuntimeError(
                f"Missing executable: {executable}. Build the project first."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    stereo_inputs = load_inputs(args)
    calibration_path = stereo_inputs.calibration_path
    calibration = stereo_inputs.calibration
    left_source = stereo_inputs.left_source
    right_source = stereo_inputs.right_source
    rectified_left = stereo_inputs.rectified_left
    rectified_right = stereo_inputs.rectified_right
    order_candidates = stereo_inputs.order_candidates

    left_path = details_dir / "rectified_left.png"
    right_path = details_dir / "rectified_right.png"
    if not cv2.imwrite(str(left_path), rectified_left):
        raise RuntimeError(f"Cannot write {left_path}")
    if not cv2.imwrite(str(right_path), rectified_right):
        raise RuntimeError(f"Cannot write {right_path}")
    make_rectification_check(
        rectified_left,
        rectified_right,
        details_dir / "rectification_check.png",
    )

    print(
        f"Input mode={stereo_inputs.mode}; "
        f"left={left_source}, right={right_source}; "
        f"rectified fx={calibration.focal_px:.3f}px, "
        f"baseline={calibration.baseline_mm:.3f}mm"
    )
    if order_candidates:
        for candidate in order_candidates:
            print(
                "  order candidate: "
                f"{candidate['left_file']} -> {candidate['right_file']}, "
                f"score={candidate['score']:.2f}, "
                f"valid={candidate['valid_percent']:.1f}%, "
                f"photo median={candidate['photo_median_abs_gray']:.1f}"
            )
    else:
        print("  left/right roles were supplied explicitly; auto-order skipped")

    bm_disparity, bm_runtime, bm_visual_match, bm_stdout = run_bm(
        bm_exe,
        left_path,
        right_path,
        details_dir,
        rectified_left,
        rectified_right,
    )
    (details_dir / "bm_d128_b11_run.log").write_text(
        bm_stdout, encoding="utf-8"
    )
    sgm_raw_path = details_dir / "sgm_d128_p20_p40_disparity_raw.png"
    (
        sgm_disparity,
        sgm_runtime,
        sgm_memory_mib,
        sgm_stdout,
    ) = run_sgm(
        sgm_exe,
        left_path,
        right_path,
        sgm_raw_path,
        SGM_CONFIG,
    )
    (details_dir / "sgm_d128_p20_p40_run.log").write_text(
        sgm_stdout, encoding="utf-8"
    )

    algorithm_inputs = (
        (BM_CONFIG, bm_disparity, bm_runtime, float("nan")),
        (SGM_CONFIG, sgm_disparity, sgm_runtime, sgm_memory_mib),
    )
    all_metrics = []
    result_images: dict[str, np.ndarray] = {}
    metrics_by_key = {}
    for config, disparity, runtime_ms, memory_mib in algorithm_inputs:
        depth, valid = depth_from_disparity(
            disparity, calibration, config.disparities
        )
        metrics = compute_metrics(
            config,
            runtime_ms,
            memory_mib,
            rectified_left,
            rectified_right,
            disparity,
            depth,
            valid,
        )
        color = colorize_depth(depth, valid)
        write_depth_mm(
            details_dir / f"{config.key}_depth_mm_u16.png", depth, valid
        )
        color_path = details_dir / f"{config.key}_depth_color.png"
        if not cv2.imwrite(str(color_path), color):
            raise RuntimeError(f"Cannot write depth visualization: {color_path}")
        all_metrics.append(metrics)
        result_images[config.key] = color
        metrics_by_key[config.key] = metrics
        print(
            f"  {config.title}: runtime={runtime_ms:.1f}ms, "
            f"valid={metrics.valid_percent:.1f}%, "
            f"saturated={metrics.saturated_percent:.2f}%, "
            f"photo median={metrics.photo_median_abs_gray:.1f}, "
            f"local outlier={metrics.local_outlier_percent:.1f}%"
        )

    make_comparison(
        rectified_left,
        result_images,
        metrics_by_key,
        tuple(config.key for config in CONFIGURATIONS),
        "BM and Vitis-SGM metric-depth comparison (D=128)",
        output_dir / "comparison_bm_sgbm.png",
    )
    save_metrics(all_metrics, output_dir)

    metadata = {
        "input_mode": stereo_inputs.mode,
        "dataset": (
            str(stereo_inputs.dataset)
            if stereo_inputs.dataset is not None
            else None
        ),
        "calibration_file": str(calibration_path),
        "ground_truth_available": False,
        "selected_left_file": left_source.name,
        "selected_right_file": right_source.name,
        "filename_timestamp_gap_seconds": filename_timestamp_gap_seconds(
            left_source, right_source
        ),
        "filename_timestamp_capture_semantics_known": False,
        "order_candidates": order_candidates,
        "image_size": [calibration.width, calibration.height],
        "rectified_focal_px": calibration.focal_px,
        "baseline_yaml_mm": calibration.baseline_yaml_mm,
        "rectified_baseline_mm": calibration.baseline_mm,
        "fx_times_baseline_px_mm": (
            calibration.focal_px * calibration.baseline_mm
        ),
        "doffs_px": calibration.doffs_px,
        "rectification_roi1": calibration.roi1,
        "rectification_roi2": calibration.roi2,
        "depth_equation": (
            "Z_mm = rectified_fx_px * baseline_mm "
            "/ (disparity_px + doffs_px)"
        ),
        "depth_display_mm": [
            DEPTH_DISPLAY_MIN_MM,
            DEPTH_DISPLAY_MAX_MM,
        ],
        "depth_valid_mm": [DEPTH_VALID_MIN_MM, DEPTH_VALID_MAX_MM],
        "evaluation_roi_xyxy": EVALUATION_ROI_XYXY,
        "photo_inlier_abs_gray_threshold": PHOTO_INLIER_THRESHOLD,
        "bm_cpp_visual_match_percent": bm_visual_match,
        "parameters": {
            "disparities": BM_CONFIG.disparities,
            "bm": {
                "block_size": BM_CONFIG.block_size,
                "pre_filter_cap": BM_PRE_FILTER_CAP,
                "uniqueness_ratio": BM_UNIQUENESS_RATIO,
                "texture_threshold": BM_TEXTURE_THRESHOLD,
                "min_disparity": 0,
            },
            "vitis_sgm": {
                "census_window": 5,
                "p1": SGM_CONFIG.p1,
                "p2": SGM_CONFIG.p2,
                "paths": 4,
            },
        },
        "configurations": [asdict(config) for config in CONFIGURATIONS],
    }
    with (output_dir / "metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            sanitize_json(metadata),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    print(
        f"BM C++/Python 8-bit visualization match: "
        f"{bm_visual_match:.3f}%"
    )
    print(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
