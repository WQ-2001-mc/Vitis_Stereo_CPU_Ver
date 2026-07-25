#!/usr/bin/env python3
"""Evaluate both CPU stereo algorithms on Gemini335 0724 data.

The script rectifies the explicitly labelled Gemini335 pair with its own
calibration, runs the C++ ``vitis_bm_cpu`` and ``vitis_sgbm_cpu`` programs,
converts disparity to metric depth, and compares the results with the previous
in-house camera capture of the same scene.

There is no pixel-aligned ground-truth depth for either camera.  The reported
metrics are therefore correspondence/coverage diagnostics, not absolute depth
accuracy measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from evaluate_camera_dataset import Calibration, read_calibration, read_gray, rectify_pair


DEFAULT_PROJECT = Path("/home/hcc/Desktop/HXB/Vitis_Stereo_CPU_Ver")
DEFAULT_DATASET = Path("/home/hcc/Desktop/Public/datasets/gemini335/0724")
DEFAULT_PREVIOUS_DATASET = Path(
    "/home/hcc/Desktop/Public/datasets/自研相机/0724-imgs"
)

DISPARITIES = 128
BM_BLOCK_SIZE = 11
BM_PRE_FILTER_CAP = 31
BM_UNIQUENESS_RATIO = 15
BM_TEXTURE_THRESHOLD = 20
SGM_P1 = 20
SGM_P2 = 40

DEPTH_DISPLAY_MIN_MM = 300.0
DEPTH_DISPLAY_MAX_MM = 3000.0
DEPTH_VALID_MIN_MM = 150.0
DEPTH_VALID_MAX_MM = 10000.0
PHOTO_INLIER_THRESHOLD = 15.0

# The normalized ROI excludes the left disparity-search blind area and a thin
# remap border.  Using the same normalized coordinates makes coverage metrics
# more comparable despite the cameras' different resolutions and aspect ratios.
NORMALIZED_ROI_XYXY = (0.23, 0.02, 0.984, 0.98)


@dataclass
class CameraInput:
    key: str
    label: str
    dataset: Path
    calibration_path: Path
    left_path: Path
    right_path: Path
    calibration: Calibration
    rectified_left: np.ndarray
    rectified_right: np.ndarray
    order_candidates: list[dict[str, object]]


@dataclass
class ResultMetrics:
    camera: str
    algorithm: str
    disparities: int
    block_size: int | None
    p1: int | None
    p2: int | None
    runtime_ms: float
    estimated_memory_mib: float
    valid_percent: float
    saturated_percent: float
    photo_median_abs_gray: float
    photo_inlier_percent: float
    local_outlier_percent: float
    median_disparity_px: float
    p10_depth_mm: float
    median_depth_mm: float
    p90_depth_mm: float
    bm_cpp_visual_match_percent: float | None


@dataclass
class AlgorithmResult:
    disparity: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    depth_color: np.ndarray
    metrics: ResultMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--previous-dataset", type=Path, default=DEFAULT_PREVIOUS_DATASET
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: PROJECT/results/gemini335_0724)",
    )
    parser.add_argument("--bm-exe", type=Path, default=None)
    parser.add_argument("--sgm-exe", type=Path, default=None)
    return parser.parse_args()


def find_single(dataset: Path, pattern: str, description: str) -> Path:
    paths = sorted(dataset.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one {description} matching {pattern!r} in {dataset}, "
            f"found {len(paths)}"
        )
    return paths[0]


def find_labelled_image(dataset: Path, label: str) -> Path:
    candidates = [
        path
        for path in dataset.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
        and label.casefold() in path.name.casefold()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one image containing {label!r} in {dataset}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def normalized_roi_mask(shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = shape
    nx0, ny0, nx1, ny1 = NORMALIZED_ROI_XYXY
    x0 = max(DISPARITIES + 2, int(round(nx0 * width)))
    y0 = int(round(ny0 * height))
    x1 = min(width, int(round(nx1 * width)))
    y1 = min(height, int(round(ny1 * height)))
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise RuntimeError(f"Invalid evaluation ROI for {width}x{height}")
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask, (x0, y0, x1, y1)


def create_bm() -> cv2.StereoBM:
    matcher = cv2.StereoBM_create(
        numDisparities=DISPARITIES, blockSize=BM_BLOCK_SIZE
    )
    matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
    matcher.setPreFilterCap(BM_PRE_FILTER_CAP)
    matcher.setMinDisparity(0)
    matcher.setTextureThreshold(BM_TEXTURE_THRESHOLD)
    matcher.setUniquenessRatio(BM_UNIQUENESS_RATIO)
    return matcher


def photometric_metrics(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float]:
    yy, xx = np.indices(disparity.shape, dtype=np.float32)
    map_x = xx - disparity.astype(np.float32)
    map_valid = valid & (map_x >= 0.0) & (map_x <= right.shape[1] - 1.0)
    if not np.any(map_valid):
        return float("nan"), float("nan")
    warped_right = cv2.remap(
        right,
        map_x,
        yy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    residual = np.abs(left.astype(np.float32) - warped_right.astype(np.float32))
    values = residual[map_valid]
    return (
        float(np.median(values)),
        100.0 * float(np.mean(values <= PHOTO_INLIER_THRESHOLD)),
    )


def score_order(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float]:
    cv2.setUseOptimized(False)
    disparity = create_bm().compute(left, right).astype(np.float32) / 16.0
    roi, _ = normalized_roi_mask(disparity.shape)
    valid = roi & (disparity > 0.0) & (disparity < DISPARITIES - 1.0)
    valid_percent = 100.0 * float(np.sum(valid) / np.sum(roi))
    photo_median, photo_inlier = photometric_metrics(left, right, disparity, valid)
    score = valid_percent + (
        photo_inlier if math.isfinite(photo_inlier) else 0.0
    )
    return {
        "score": score,
        "valid_percent": valid_percent,
        "photo_median_abs_gray": photo_median,
        "photo_inlier_percent": photo_inlier,
        "median_disparity_px": (
            float(np.median(disparity[valid])) if np.any(valid) else float("nan")
        ),
    }


def choose_unlabelled_order(
    image_paths: list[Path],
    calibration: Calibration,
) -> tuple[Path, Path, np.ndarray, np.ndarray, list[dict[str, object]]]:
    if len(image_paths) != 2:
        raise RuntimeError(
            f"Expected exactly two previous-camera images, found {len(image_paths)}"
        )
    images = [
        read_gray(path, (calibration.width, calibration.height))
        for path in image_paths
    ]
    candidates: list[dict[str, object]] = []
    for left_index, right_index in ((0, 1), (1, 0)):
        left, right = rectify_pair(
            images[left_index], images[right_index], calibration
        )
        candidates.append(
            {
                "left_file": image_paths[left_index].name,
                "right_file": image_paths[right_index].name,
                **score_order(left, right),
                "_left": left,
                "_right": right,
            }
        )
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    best = candidates[0]
    public = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in candidates
    ]
    left_path = next(path for path in image_paths if path.name == best["left_file"])
    right_path = next(path for path in image_paths if path.name == best["right_file"])
    return (
        left_path,
        right_path,
        np.asarray(best["_left"]),
        np.asarray(best["_right"]),
        public,
    )


def load_gemini335(dataset: Path) -> CameraInput:
    calibration_path = find_single(dataset, "*.yaml", "calibration YAML")
    calibration = read_calibration(calibration_path)
    left_path = find_labelled_image(dataset, "left")
    right_path = find_labelled_image(dataset, "right")
    left = read_gray(left_path, (calibration.width, calibration.height))
    right = read_gray(right_path, (calibration.width, calibration.height))
    rectified_left, rectified_right = rectify_pair(left, right, calibration)
    order_details = score_order(rectified_left, rectified_right)
    return CameraInput(
        key="gemini335",
        label="Gemini335 (0724)",
        dataset=dataset,
        calibration_path=calibration_path,
        left_path=left_path,
        right_path=right_path,
        calibration=calibration,
        rectified_left=rectified_left,
        rectified_right=rectified_right,
        order_candidates=[
            {
                "left_file": left_path.name,
                "right_file": right_path.name,
                "selection": "explicit filename labels",
                **order_details,
            }
        ],
    )


def load_previous_camera(dataset: Path) -> CameraInput:
    calibration_path = find_single(dataset, "*.yaml", "calibration YAML")
    calibration = read_calibration(calibration_path)
    image_paths = sorted(dataset.glob("*.bmp"))
    (
        left_path,
        right_path,
        rectified_left,
        rectified_right,
        order_candidates,
    ) = choose_unlabelled_order(image_paths, calibration)
    return CameraInput(
        key="previous_camera",
        label="Previous in-house camera",
        dataset=dataset,
        calibration_path=calibration_path,
        left_path=left_path,
        right_path=right_path,
        calibration=calibration,
        rectified_left=rectified_left,
        rectified_right=rectified_right,
        order_candidates=order_candidates,
    )


def parse_logged_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else float("nan")


def depth_from_disparity(
    disparity: np.ndarray,
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = disparity.astype(np.float32) + calibration.doffs_px
    valid = (
        np.isfinite(disparity)
        & (disparity > 0.0)
        & (disparity < DISPARITIES - 1.0)
        & (denominator > 1.0e-6)
    )
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = (
        calibration.focal_px * calibration.baseline_mm / denominator[valid]
    )
    valid &= (depth >= DEPTH_VALID_MIN_MM) & (depth <= DEPTH_VALID_MAX_MM)
    depth[~valid] = np.nan
    return depth, valid


def colorize_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth, DEPTH_DISPLAY_MIN_MM, DEPTH_DISPLAY_MAX_MM)
    scaled = 1.0 - (
        (clipped - DEPTH_DISPLAY_MIN_MM)
        / (DEPTH_DISPLAY_MAX_MM - DEPTH_DISPLAY_MIN_MM)
    )
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.rint(255.0 * scaled[valid]).astype(np.uint8)
    output = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    output[~valid] = 0
    return output


def write_depth_mm(path: Path, depth: np.ndarray, valid: np.ndarray) -> None:
    output = np.zeros(depth.shape, dtype=np.uint16)
    output[valid] = np.rint(np.clip(depth[valid], 0.0, 65535.0)).astype(
        np.uint16
    )
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"Cannot write depth image: {path}")


def compute_metrics(
    camera: CameraInput,
    algorithm: str,
    disparity: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    runtime_ms: float,
    memory_mib: float,
    bm_visual_match: float | None,
) -> ResultMetrics:
    roi, _ = normalized_roi_mask(disparity.shape)
    scored_valid = roi & valid
    roi_count = int(np.sum(roi))
    valid_percent = 100.0 * float(np.sum(scored_valid) / roi_count)
    saturated = roi & (disparity >= DISPARITIES - 1.0)
    saturated_percent = 100.0 * float(np.sum(saturated) / roi_count)
    photo_median, photo_inlier = photometric_metrics(
        camera.rectified_left,
        camera.rectified_right,
        disparity,
        scored_valid,
    )
    median_disparity = cv2.medianBlur(disparity.astype(np.float32), 5)
    local_outlier = scored_valid & (
        np.abs(disparity - median_disparity) > 2.0
    )
    local_outlier_percent = (
        100.0 * float(np.sum(local_outlier) / np.sum(scored_valid))
        if np.any(scored_valid)
        else float("nan")
    )
    if np.any(scored_valid):
        p10_depth, median_depth, p90_depth = (
            float(value)
            for value in np.percentile(
                depth[scored_valid], [10.0, 50.0, 90.0]
            )
        )
        median_disparity = float(np.median(disparity[scored_valid]))
    else:
        p10_depth = median_depth = p90_depth = float("nan")
        median_disparity = float("nan")
    return ResultMetrics(
        camera=camera.key,
        algorithm=algorithm,
        disparities=DISPARITIES,
        block_size=BM_BLOCK_SIZE if algorithm == "BM" else None,
        p1=SGM_P1 if algorithm == "Vitis-SGM" else None,
        p2=SGM_P2 if algorithm == "Vitis-SGM" else None,
        runtime_ms=runtime_ms,
        estimated_memory_mib=memory_mib,
        valid_percent=valid_percent,
        saturated_percent=saturated_percent,
        photo_median_abs_gray=photo_median,
        photo_inlier_percent=photo_inlier,
        local_outlier_percent=local_outlier_percent,
        median_disparity_px=median_disparity,
        p10_depth_mm=p10_depth,
        median_depth_mm=median_depth,
        p90_depth_mm=p90_depth,
        bm_cpp_visual_match_percent=bm_visual_match,
    )


def save_rectified_inputs(camera: CameraInput, details_dir: Path) -> tuple[Path, Path]:
    camera_dir = details_dir / camera.key
    camera_dir.mkdir(parents=True, exist_ok=True)
    left_path = camera_dir / "rectified_left.png"
    right_path = camera_dir / "rectified_right.png"
    if not cv2.imwrite(str(left_path), camera.rectified_left):
        raise RuntimeError(f"Cannot write {left_path}")
    if not cv2.imwrite(str(right_path), camera.rectified_right):
        raise RuntimeError(f"Cannot write {right_path}")
    return left_path, right_path


def run_bm(
    camera: CameraInput,
    executable: Path,
    left_path: Path,
    right_path: Path,
    details_dir: Path,
) -> AlgorithmResult:
    camera_dir = details_dir / camera.key
    cpp_visual_path = camera_dir / "bm_d128_b11_disparity_visual.png"
    completed = subprocess.run(
        [
            str(executable),
            str(left_path),
            str(right_path),
            str(cpp_visual_path),
            str(DISPARITIES),
            str(BM_BLOCK_SIZE),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    (camera_dir / "bm_d128_b11_run.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    runtime_ms = parse_logged_float(
        r"CPU compute:\s+([0-9.]+) ms", completed.stdout
    )

    # The original C++ BM reference writes only an 8-bit display image.
    # Recompute through the same OpenCV implementation/settings to preserve
    # the Q4 raw disparity needed for metric depth conversion.
    cv2.setUseOptimized(False)
    disparity_q4 = create_bm().compute(
        camera.rectified_left, camera.rectified_right
    )
    disparity = disparity_q4.astype(np.float32) / 16.0
    q4_output = np.zeros(disparity_q4.shape, dtype=np.uint16)
    positive = disparity_q4 > 0
    q4_output[positive] = disparity_q4[positive].astype(np.uint16)
    if not cv2.imwrite(
        str(camera_dir / "bm_d128_b11_disparity_q4_u16.png"), q4_output
    ):
        raise RuntimeError("Cannot write BM Q4 disparity")

    cpp_visual = cv2.imread(str(cpp_visual_path), cv2.IMREAD_GRAYSCALE)
    if cpp_visual is None:
        raise RuntimeError(f"BM executable did not produce {cpp_visual_path}")
    reference_visual = np.clip(
        np.rint(disparity_q4.astype(np.float32) / 8.0), 0.0, 255.0
    ).astype(np.uint8)
    visual_match = 100.0 * float(np.mean(cpp_visual == reference_visual))

    depth, valid = depth_from_disparity(disparity, camera.calibration)
    color = colorize_depth(depth, valid)
    write_depth_mm(camera_dir / "bm_d128_b11_depth_mm_u16.png", depth, valid)
    if not cv2.imwrite(str(camera_dir / "bm_d128_b11_depth_color.png"), color):
        raise RuntimeError("Cannot write BM color depth")
    metrics = compute_metrics(
        camera,
        "BM",
        disparity,
        depth,
        valid,
        runtime_ms,
        float("nan"),
        visual_match,
    )
    return AlgorithmResult(disparity, depth, valid, color, metrics)


def run_sgm(
    camera: CameraInput,
    executable: Path,
    left_path: Path,
    right_path: Path,
    details_dir: Path,
) -> AlgorithmResult:
    camera_dir = details_dir / camera.key
    raw_path = camera_dir / "sgm_d128_p20_p40_disparity_raw.png"
    completed = subprocess.run(
        [
            str(executable),
            str(left_path),
            str(right_path),
            str(raw_path),
            str(DISPARITIES),
            str(SGM_P1),
            str(SGM_P2),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    (camera_dir / "sgm_d128_p20_p40_run.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    disparity_raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
    if disparity_raw is None:
        raise RuntimeError(f"SGM executable did not produce {raw_path}")
    disparity = disparity_raw.astype(np.float32)
    runtime_ms = parse_logged_float(
        r"CPU compute:\s+([0-9.]+) ms", completed.stdout
    )
    memory_mib = parse_logged_float(
        r"Est\. memory:\s+([0-9.]+) MiB", completed.stdout
    )
    depth, valid = depth_from_disparity(disparity, camera.calibration)
    color = colorize_depth(depth, valid)
    write_depth_mm(camera_dir / "sgm_d128_p20_p40_depth_mm_u16.png", depth, valid)
    if not cv2.imwrite(
        str(camera_dir / "sgm_d128_p20_p40_depth_color.png"), color
    ):
        raise RuntimeError("Cannot write SGM color depth")
    metrics = compute_metrics(
        camera,
        "Vitis-SGM",
        disparity,
        depth,
        valid,
        runtime_ms,
        memory_mib,
        None,
    )
    return AlgorithmResult(disparity, depth, valid, color, metrics)


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


def panel(
    image: np.ndarray,
    title: str,
    lines: tuple[str, ...],
    width: int,
    body_height: int,
) -> np.ndarray:
    header_height = 34 + 22 * len(lines)
    header = np.full((header_height, width, 3), 24, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (9, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            header,
            line,
            (9, 48 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([header, fit_image(image, width, body_height)])


def format_value(value: float, decimals: int = 1) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def metrics_lines(metrics: ResultMetrics) -> tuple[str, str]:
    return (
        f"valid {metrics.valid_percent:.1f}% | saturated {metrics.saturated_percent:.2f}%",
        "photo median "
        f"{format_value(metrics.photo_median_abs_gray)} | local outlier "
        f"{format_value(metrics.local_outlier_percent)}%",
    )


def colorbar(width: int, heading: str, height: int = 72) -> np.ndarray:
    output = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(
        output,
        heading,
        (9, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    x0, x1 = 145, width - 145
    gradient = np.linspace(255, 0, x1 - x0, dtype=np.uint8)[None, :]
    gradient = np.repeat(gradient, 16, axis=0)
    output[29:45, x0:x1] = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    cv2.putText(
        output,
        f"near {DEPTH_DISPLAY_MIN_MM / 1000.0:.1f} m",
        (9, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"far {DEPTH_DISPLAY_MAX_MM / 1000.0:.1f} m",
        (width - 118, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Fixed metric scale; black = invalid/range saturation",
        (9, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )
    return output


def annotate_source(camera: CameraInput) -> np.ndarray:
    source = cv2.cvtColor(camera.rectified_left, cv2.COLOR_GRAY2BGR)
    _, (x0, y0, x1, y1) = normalized_roi_mask(camera.rectified_left.shape)
    cv2.rectangle(source, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
    return source


def make_gemini_comparison(
    camera: CameraInput,
    bm: AlgorithmResult,
    sgm: AlgorithmResult,
    output_path: Path,
) -> None:
    width = 430
    body_height = 300
    items = [
        panel(
            annotate_source(camera),
            "Gemini335 rectified left",
            ("green = normalized metrics ROI", "848x480 input"),
            width,
            body_height,
        ),
        panel(
            bm.depth_color,
            "StereoBM D=128, block=11",
            metrics_lines(bm.metrics),
            width,
            body_height,
        ),
        panel(
            sgm.depth_color,
            "Vitis-SGM D=128, P1/P2=20/40",
            metrics_lines(sgm.metrics),
            width,
            body_height,
        ),
    ]
    comparison = np.hstack(items)
    comparison = np.vstack(
        [comparison, colorbar(comparison.shape[1], "Gemini335 CPU stereo comparison")]
    )
    if not cv2.imwrite(str(output_path), comparison):
        raise RuntimeError(f"Cannot write {output_path}")


def make_camera_comparison(
    previous: CameraInput,
    gemini: CameraInput,
    results: dict[str, dict[str, AlgorithmResult]],
    output_path: Path,
) -> None:
    panel_width = 560
    body_height = 390
    rows = []
    rows.append(
        np.hstack(
            [
                panel(
                    annotate_source(previous),
                    "Previous in-house camera - rectified left",
                    (
                        f"{previous.calibration.width}x{previous.calibration.height}",
                        "green = normalized metrics ROI",
                    ),
                    panel_width,
                    body_height,
                ),
                panel(
                    annotate_source(gemini),
                    "Gemini335 - rectified left",
                    (
                        f"{gemini.calibration.width}x{gemini.calibration.height}",
                        "green = normalized metrics ROI",
                    ),
                    panel_width,
                    body_height,
                ),
            ]
        )
    )
    for key, title in (("bm", "StereoBM D=128, block=11"), ("sgm", "Vitis-SGM D=128, P1/P2=20/40")):
        rows.append(
            np.hstack(
                [
                    panel(
                        results[previous.key][key].depth_color,
                        f"Previous camera - {title}",
                        metrics_lines(results[previous.key][key].metrics),
                        panel_width,
                        body_height,
                    ),
                    panel(
                        results[gemini.key][key].depth_color,
                        f"Gemini335 - {title}",
                        metrics_lines(results[gemini.key][key].metrics),
                        panel_width,
                        body_height,
                    ),
                ]
            )
        )
    comparison = np.vstack(rows)
    comparison = np.vstack(
        [
            comparison,
            colorbar(
                comparison.shape[1],
                "Same scene, parameters and 0.3-3.0 m metric-depth scale",
            ),
        ]
    )
    if not cv2.imwrite(str(output_path), comparison):
        raise RuntimeError(f"Cannot write {output_path}")


def make_rectification_check(camera: CameraInput, output_path: Path) -> None:
    width = 600
    body_height = 340
    displays = []
    for image, label in (
        (camera.rectified_left, "Gemini335 rectified left"),
        (camera.rectified_right, "Gemini335 rectified right"),
    ):
        display = fit_image(image, width, body_height)
        for y in range(45, body_height, 45):
            cv2.line(display, (0, y), (width - 1, y), (0, 255, 0), 1)
        displays.append(panel(display, label, ("horizontal epipolar guides",), width, body_height))
    comparison = np.hstack(displays)
    if not cv2.imwrite(str(output_path), comparison):
        raise RuntimeError(f"Cannot write {output_path}")


def sanitize_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_metrics(metrics: Iterable[ResultMetrics], output_dir: Path) -> None:
    rows = [asdict(item) for item in metrics]
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            sanitize_json(rows),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    with (output_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def camera_metadata(camera: CameraInput) -> dict[str, object]:
    _, roi = normalized_roi_mask(camera.rectified_left.shape)
    calibration = camera.calibration
    return {
        "label": camera.label,
        "dataset": str(camera.dataset),
        "calibration_file": str(camera.calibration_path),
        "selected_left_file": camera.left_path.name,
        "selected_right_file": camera.right_path.name,
        "order_candidates": camera.order_candidates,
        "image_size": [calibration.width, calibration.height],
        "rectified_focal_px": calibration.focal_px,
        "baseline_yaml_mm": calibration.baseline_yaml_mm,
        "rectified_baseline_mm": calibration.baseline_mm,
        "fx_times_baseline_px_mm": calibration.focal_px * calibration.baseline_mm,
        "doffs_px": calibration.doffs_px,
        "rectification_roi1": calibration.roi1,
        "rectification_roi2": calibration.roi2,
        "evaluation_roi_xyxy": roi,
        "d128_theoretical_nearest_depth_mm": (
            calibration.focal_px * calibration.baseline_mm
            / (DISPARITIES - 1)
        ),
    }


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    dataset = args.dataset.resolve()
    previous_dataset = args.previous_dataset.resolve()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else project / "results" / "gemini335_0724"
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
    for directory in (dataset, previous_dataset):
        if not directory.is_dir():
            raise RuntimeError(f"Missing dataset directory: {directory}")
    for executable in (bm_exe, sgm_exe):
        if not executable.is_file():
            raise RuntimeError(
                f"Missing executable: {executable}. Build the CMake project first."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    gemini = load_gemini335(dataset)
    previous = load_previous_camera(previous_dataset)
    cameras = (previous, gemini)
    results: dict[str, dict[str, AlgorithmResult]] = {}
    all_metrics: list[ResultMetrics] = []

    for camera in cameras:
        left_path, right_path = save_rectified_inputs(camera, details_dir)
        print(
            f"{camera.label}: left={camera.left_path.name}, "
            f"right={camera.right_path.name}, "
            f"rectified fx={camera.calibration.focal_px:.3f}px, "
            f"baseline={camera.calibration.baseline_mm:.3f}mm"
        )
        bm = run_bm(camera, bm_exe, left_path, right_path, details_dir)
        sgm = run_sgm(camera, sgm_exe, left_path, right_path, details_dir)
        results[camera.key] = {"bm": bm, "sgm": sgm}
        all_metrics.extend((bm.metrics, sgm.metrics))
        for result in (bm, sgm):
            print(
                f"  {result.metrics.algorithm}: "
                f"runtime={result.metrics.runtime_ms:.3f}ms, "
                f"valid={result.metrics.valid_percent:.1f}%, "
                f"saturated={result.metrics.saturated_percent:.2f}%, "
                f"photo median={result.metrics.photo_median_abs_gray:.1f}, "
                f"local outlier={result.metrics.local_outlier_percent:.1f}%"
            )

    make_rectification_check(gemini, output_dir / "rectification_check.png")
    make_gemini_comparison(
        gemini,
        results[gemini.key]["bm"],
        results[gemini.key]["sgm"],
        output_dir / "comparison_gemini335_algorithms.png",
    )
    make_camera_comparison(
        previous,
        gemini,
        results,
        output_dir / "comparison_previous_camera.png",
    )
    save_metrics(all_metrics, output_dir)
    metadata = {
        "ground_truth_available": False,
        "comparison_note": (
            "Same scene, algorithm parameters and depth color scale; cameras "
            "have different resolution, optics, baseline and field of view."
        ),
        "depth_equation": (
            "Z_mm = rectified_fx_px * baseline_mm "
            "/ (disparity_px + doffs_px)"
        ),
        "depth_display_mm": [DEPTH_DISPLAY_MIN_MM, DEPTH_DISPLAY_MAX_MM],
        "depth_valid_mm": [DEPTH_VALID_MIN_MM, DEPTH_VALID_MAX_MM],
        "normalized_evaluation_roi_xyxy": NORMALIZED_ROI_XYXY,
        "photo_inlier_abs_gray_threshold": PHOTO_INLIER_THRESHOLD,
        "parameters": {
            "disparities": DISPARITIES,
            "bm": {
                "block_size": BM_BLOCK_SIZE,
                "pre_filter_cap": BM_PRE_FILTER_CAP,
                "uniqueness_ratio": BM_UNIQUENESS_RATIO,
                "texture_threshold": BM_TEXTURE_THRESHOLD,
                "min_disparity": 0,
            },
            "vitis_sgm": {
                "census_window": 5,
                "p1": SGM_P1,
                "p2": SGM_P2,
                "paths": 4,
            },
        },
        "cameras": {
            previous.key: camera_metadata(previous),
            gemini.key: camera_metadata(gemini),
        },
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
    print(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
