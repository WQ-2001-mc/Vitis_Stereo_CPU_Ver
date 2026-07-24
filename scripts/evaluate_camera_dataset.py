#!/usr/bin/env python3
"""Evaluate the extracted Vitis BM/SGM CPU references on the in-house camera data.

The script:
  * reads the OpenCV stereo calibration;
  * determines the unlabeled BMP left/right order;
  * rectifies each pair;
  * evaluates BM and the extracted Vitis scalar SGM at D=64/128/256;
  * converts disparity to metric depth with Z = fx * baseline / disparity;
  * writes 16-bit millimetre depth PNGs, color visualizations, metrics, and
    horizontal comparison figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_DATASET = Path("/home/hcc/Desktop/Public/datasets/自研相机")
DEFAULT_PROJECT = Path("/home/hcc/Desktop/HXB/Vitis_Stereo_CPU_Ver")
DISTANCE_NAMES = ("50cm", "75cm")
DISPARITY_COUNTS = (64, 128, 256)
NOMINAL_DEPTH_MM = {"50cm": 500.0, "75cm": 750.0}

BM_BLOCK_SIZE = 11
BM_PRE_FILTER_CAP = 31
BM_UNIQUENESS_RATIO = 15
BM_TEXTURE_THRESHOLD = 20

DEPTH_DISPLAY_MIN_MM = 300.0
DEPTH_DISPLAY_MAX_MM = 1200.0

# Interior rectangles of the central planar target. They intentionally exclude
# the stand, target edges, and most background pixels. Coordinates refer to the
# 1224x1024 rectified left images and were selected once from the source image,
# before inspecting any algorithm output.
TARGET_ROI = {
    "50cm": (640, 430, 710, 545),
    "75cm": (635, 450, 682, 535),
}


@dataclass
class Calibration:
    width: int
    height: int
    baseline_yaml_mm: float
    k1: np.ndarray
    d1: np.ndarray
    k2: np.ndarray
    d2: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    q: np.ndarray
    roi1: tuple[int, int, int, int]
    roi2: tuple[int, int, int, int]
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray

    @property
    def focal_px(self) -> float:
        return float(self.p1[0, 0])

    @property
    def doffs_px(self) -> float:
        return float(self.p2[0, 2] - self.p1[0, 2])

    @property
    def baseline_mm(self) -> float:
        projected = abs(float(self.p2[0, 3]) / self.focal_px)
        if projected > 1.0e-12:
            return projected
        return float(np.linalg.norm(self.translation))


@dataclass
class ResultMetrics:
    distance: str
    algorithm: str
    disparities: int
    runtime_ms: float
    estimated_memory_mib: float
    valid_percent: float
    nominal_inlier_percent: float
    median_depth_mm: float
    p10_depth_mm: float
    p90_depth_mm: float
    median_abs_error_mm: float
    expected_disparity_px: float
    median_disparity_px: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: PROJECT/results/camera_dataset)",
    )
    parser.add_argument(
        "--sgbm-exe",
        type=Path,
        default=None,
        help="Extracted Vitis scalar SGM executable",
    )
    return parser.parse_args()


def read_required_matrix(fs: cv2.FileStorage, key: str) -> np.ndarray:
    value = fs.getNode(key).mat()
    if value is None or value.size == 0:
        raise RuntimeError(f"Missing calibration matrix: {key}")
    return np.asarray(value, dtype=np.float64)


def read_calibration(path: Path) -> Calibration:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration: {path}")
    width = int(fs.getNode("image_width").real())
    height = int(fs.getNode("image_height").real())
    baseline = float(fs.getNode("baseline").real())
    k1 = read_required_matrix(fs, "K1")
    d1 = read_required_matrix(fs, "D1")
    k2 = read_required_matrix(fs, "K2")
    d2 = read_required_matrix(fs, "D2")
    rotation = read_required_matrix(fs, "R")
    translation = read_required_matrix(fs, "T")
    fs.release()

    image_size = (width, height)
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        k1,
        d1,
        k2,
        d2,
        image_size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
        newImageSize=image_size,
    )
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        k1, d1, r1, p1, image_size, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        k2, d2, r2, p2, image_size, cv2.CV_16SC2
    )
    return Calibration(
        width=width,
        height=height,
        baseline_yaml_mm=baseline,
        k1=k1,
        d1=d1,
        k2=k2,
        d2=d2,
        rotation=rotation,
        translation=translation,
        r1=r1,
        r2=r2,
        p1=p1,
        p2=p2,
        q=q,
        roi1=tuple(int(v) for v in roi1),
        roi2=tuple(int(v) for v in roi2),
        left_map1=left_map1,
        left_map2=left_map2,
        right_map1=right_map1,
        right_map2=right_map2,
    )


def read_gray(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if (image.shape[1], image.shape[0]) != expected_size:
        raise RuntimeError(
            f"Unexpected image size for {path}: "
            f"{image.shape[1]}x{image.shape[0]}, expected "
            f"{expected_size[0]}x{expected_size[1]}"
        )
    return image


def rectify_pair(
    camera1_image: np.ndarray,
    camera2_image: np.ndarray,
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray]:
    left = cv2.remap(
        camera1_image,
        calibration.left_map1,
        calibration.left_map2,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right = cv2.remap(
        camera2_image,
        calibration.right_map1,
        calibration.right_map2,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return left, right


def create_bm(disparities: int) -> cv2.StereoBM:
    matcher = cv2.StereoBM_create(
        numDisparities=disparities, blockSize=BM_BLOCK_SIZE
    )
    matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
    matcher.setPreFilterCap(BM_PRE_FILTER_CAP)
    matcher.setMinDisparity(0)
    matcher.setTextureThreshold(BM_TEXTURE_THRESHOLD)
    matcher.setUniquenessRatio(BM_UNIQUENESS_RATIO)
    return matcher


def compute_bm_disparity(
    left: np.ndarray, right: np.ndarray, disparities: int
) -> tuple[np.ndarray, float]:
    cv2.setUseOptimized(False)
    matcher = create_bm(disparities)
    start = time.perf_counter()
    disparity_q4 = matcher.compute(left, right)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return disparity_q4.astype(np.float32) / 16.0, elapsed_ms


def order_score(
    left: np.ndarray,
    right: np.ndarray,
    calibration: Calibration,
    nominal_depth_mm: float,
) -> tuple[float, dict[str, float]]:
    disparity, _ = compute_bm_disparity(left, right, 128)
    expected = calibration.focal_px * calibration.baseline_mm / nominal_depth_mm
    height, width = disparity.shape
    roi = disparity[
        int(0.15 * height) : int(0.85 * height),
        int(0.25 * width) : int(0.75 * width),
    ]
    valid = (roi > 1.0) & (roi < 126.0)
    plausible = valid & (roi > expected * 0.45) & (roi < expected * 1.8)
    valid_fraction = float(np.mean(valid))
    plausible_fraction = float(np.mean(plausible))
    if np.any(valid):
        median = float(np.median(roi[valid]))
        closeness = math.exp(-abs(math.log(max(median, 1.0e-6) / expected)))
    else:
        median = float("nan")
        closeness = 0.0
    score = valid_fraction + 2.0 * plausible_fraction + 0.25 * closeness
    return score, {
        "score": score,
        "valid_percent": 100.0 * valid_fraction,
        "plausible_percent": 100.0 * plausible_fraction,
        "median_disparity_px": median,
        "expected_disparity_px": expected,
    }


def determine_order(
    image_paths: list[Path],
    calibration: Calibration,
    nominal_depth_mm: float,
) -> tuple[Path, Path, np.ndarray, np.ndarray, list[dict[str, object]]]:
    if len(image_paths) != 2:
        raise RuntimeError(
            f"Expected exactly two BMP images, found {len(image_paths)}: "
            + ", ".join(str(path) for path in image_paths)
        )
    images = [
        read_gray(path, (calibration.width, calibration.height))
        for path in image_paths
    ]
    candidates: list[dict[str, object]] = []
    for first, second in ((0, 1), (1, 0)):
        rect_left, rect_right = rectify_pair(
            images[first], images[second], calibration
        )
        score, details = order_score(
            rect_left, rect_right, calibration, nominal_depth_mm
        )
        candidates.append(
            {
                "camera1_file": image_paths[first].name,
                "camera2_file": image_paths[second].name,
                **details,
                "_left": rect_left,
                "_right": rect_right,
            }
        )
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    best = candidates[0]
    public_candidates = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in candidates
    ]
    return (
        next(path for path in image_paths if path.name == best["camera1_file"]),
        next(path for path in image_paths if path.name == best["camera2_file"]),
        np.asarray(best["_left"]),
        np.asarray(best["_right"]),
        public_candidates,
    )


def parse_sgm_memory_mib(stdout: str) -> float:
    match = re.search(r"Est\. memory:\s+([0-9.]+) MiB", stdout)
    return float(match.group(1)) if match else float("nan")


def run_vitis_sgm(
    executable: Path,
    left_path: Path,
    right_path: Path,
    raw_output: Path,
    disparities: int,
) -> tuple[np.ndarray, float, float, str]:
    start = time.perf_counter()
    completed = subprocess.run(
        [
            str(executable),
            str(left_path),
            str(right_path),
            str(raw_output),
            str(disparities),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    disparity = cv2.imread(str(raw_output), cv2.IMREAD_UNCHANGED)
    if disparity is None:
        raise RuntimeError(f"Vitis SGM did not produce {raw_output}")
    return (
        disparity.astype(np.float32),
        wall_ms,
        parse_sgm_memory_mib(completed.stdout),
        completed.stdout,
    )


def depth_from_disparity(
    disparity: np.ndarray,
    calibration: Calibration,
    disparities: int,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = disparity.astype(np.float32) + calibration.doffs_px
    valid = (
        np.isfinite(disparity)
        & (disparity > 0.0)
        & (disparity < float(disparities - 1))
        & (denominator > 1.0e-6)
    )
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = (
        calibration.focal_px * calibration.baseline_mm / denominator[valid]
    )
    valid &= (depth >= 150.0) & (depth <= 10000.0)
    depth[~valid] = np.nan
    return depth, valid


def evaluation_roi(
    shape: tuple[int, int], distance_name: str
) -> tuple[slice, slice, tuple[int, int, int, int]]:
    height, width = shape
    x0, y0, x1, y1 = TARGET_ROI[distance_name]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise RuntimeError(
            f"Target ROI {TARGET_ROI[distance_name]} is outside "
            f"{width}x{height}"
        )
    return slice(y0, y1), slice(x0, x1), (x0, y0, x1, y1)


def compute_metrics(
    distance_name: str,
    algorithm: str,
    disparities: int,
    runtime_ms: float,
    estimated_memory_mib: float,
    disparity: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    calibration: Calibration,
) -> ResultMetrics:
    nominal = NOMINAL_DEPTH_MM[distance_name]
    ys, xs, _ = evaluation_roi(depth.shape, distance_name)
    roi_depth = depth[ys, xs]
    roi_disparity = disparity[ys, xs]
    roi_valid = valid[ys, xs] & np.isfinite(roi_depth)
    valid_values = roi_depth[roi_valid]
    disparity_values = roi_disparity[roi_valid]
    valid_percent = 100.0 * float(np.mean(roi_valid))

    tolerance_mm = 0.10 * nominal
    inliers = roi_valid & (np.abs(roi_depth - nominal) <= tolerance_mm)
    inlier_percent = 100.0 * float(np.mean(inliers))

    if valid_values.size:
        median = float(np.median(valid_values))
        p10, p90 = (float(value) for value in np.percentile(valid_values, [10, 90]))
        median_error = float(np.median(np.abs(valid_values - nominal)))
        median_disparity = float(np.median(disparity_values))
    else:
        median = p10 = p90 = median_error = median_disparity = float("nan")

    expected_disparity = (
        calibration.focal_px * calibration.baseline_mm / nominal
    )
    return ResultMetrics(
        distance=distance_name,
        algorithm=algorithm,
        disparities=disparities,
        runtime_ms=runtime_ms,
        estimated_memory_mib=estimated_memory_mib,
        valid_percent=valid_percent,
        nominal_inlier_percent=inlier_percent,
        median_depth_mm=median,
        p10_depth_mm=p10,
        p90_depth_mm=p90,
        median_abs_error_mm=median_error,
        expected_disparity_px=expected_disparity,
        median_disparity_px=median_disparity,
    )


def write_depth_mm(path: Path, depth: np.ndarray, valid: np.ndarray) -> None:
    output = np.zeros(depth.shape, dtype=np.uint16)
    rounded = np.rint(np.clip(depth[valid], 0.0, 65535.0)).astype(np.uint16)
    output[valid] = rounded
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"Cannot write depth image: {path}")


def colorize_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    clipped = np.clip(depth, DEPTH_DISPLAY_MIN_MM, DEPTH_DISPLAY_MAX_MM)
    # Invert so that closer points are warm and farther points are cool.
    scaled = 1.0 - (
        (clipped - DEPTH_DISPLAY_MIN_MM)
        / (DEPTH_DISPLAY_MAX_MM - DEPTH_DISPLAY_MIN_MM)
    )
    normalized[valid] = np.rint(255.0 * scaled[valid]).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def annotate_source(image: np.ndarray, distance_name: str) -> np.ndarray:
    source = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    _, _, box = evaluation_roi(image.shape, distance_name)
    x0, y0, x1, y1 = box
    cv2.rectangle(source, (x0, y0), (x1, y1), (0, 255, 0), 3)
    return source


def panel(
    image: np.ndarray,
    title: str,
    subtitle: str,
    width: int = 300,
) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    header = np.full((58, width, 3), 24, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        subtitle,
        (8, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, resized])


def colorbar(width: int, height: int = 42) -> np.ndarray:
    bar = np.full((height, width, 3), 24, dtype=np.uint8)
    x0, x1 = 120, width - 120
    gradient = np.linspace(255, 0, x1 - x0, dtype=np.uint8)[None, :]
    gradient = np.repeat(gradient, 15, axis=0)
    gradient_color = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    bar[3:18, x0:x1] = gradient_color
    cv2.putText(
        bar,
        f"near {DEPTH_DISPLAY_MIN_MM / 1000.0:.1f} m",
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        f"far {DEPTH_DISPLAY_MAX_MM / 1000.0:.1f} m",
        (width - 105, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        "Black = invalid; green rectangle in source = statistics ROI",
        (8, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return bar


def format_metric(value: float, suffix: str = "") -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.1f}{suffix}"


def make_comparison(
    distance_name: str,
    rectified_left: np.ndarray,
    visualizations: list[tuple[ResultMetrics, np.ndarray]],
    output_path: Path,
) -> None:
    items = [
        panel(
            annotate_source(rectified_left, distance_name),
            f"{distance_name} rectified left",
            "green box = metrics ROI",
        )
    ]
    for metrics, visualization in visualizations:
        items.append(
            panel(
                visualization,
                f"{metrics.algorithm}  D={metrics.disparities}",
                "valid "
                f"{metrics.valid_percent:.1f}% | median "
                f"{format_metric(metrics.median_depth_mm, ' mm')}",
            )
        )
    comparison = np.hstack(items)
    comparison = np.vstack([comparison, colorbar(comparison.shape[1])])
    if not cv2.imwrite(str(output_path), comparison):
        raise RuntimeError(f"Cannot write comparison: {output_path}")


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
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


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


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    dataset = args.dataset.resolve()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else project / "results" / "camera_dataset"
    )
    sgbm_exe = (
        args.sgbm_exe.resolve()
        if args.sgbm_exe is not None
        else project / "build" / "vitis_sgbm_cpu"
    )
    if not sgbm_exe.is_file():
        raise RuntimeError(
            f"Missing executable: {sgbm_exe}. Build the CMake project first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[ResultMetrics] = []
    metadata: dict[str, object] = {
        "dataset": str(dataset),
        "depth_equation": "Z_mm = rectified_fx_px * baseline_mm / disparity_px",
        "depth_display_mm": [
            DEPTH_DISPLAY_MIN_MM,
            DEPTH_DISPLAY_MAX_MM,
        ],
        "bm_parameters": {
            "block_size": BM_BLOCK_SIZE,
            "pre_filter_cap": BM_PRE_FILTER_CAP,
            "uniqueness_ratio": BM_UNIQUENESS_RATIO,
            "texture_threshold": BM_TEXTURE_THRESHOLD,
        },
        "vitis_sgm_parameters": {
            "census_window": 5,
            "p1": 20,
            "p2": 40,
            "paths": 4,
        },
        "distances": {},
    }

    for distance_name in DISTANCE_NAMES:
        source_dir = dataset / distance_name
        distance_output = output_dir / distance_name
        distance_output.mkdir(parents=True, exist_ok=True)
        calibration_path = next(source_dir.glob("*.yaml"))
        calibration = read_calibration(calibration_path)
        image_paths = sorted(source_dir.glob("*.bmp"))
        nominal_depth = NOMINAL_DEPTH_MM[distance_name]
        (
            camera1_path,
            camera2_path,
            rectified_left,
            rectified_right,
            order_candidates,
        ) = determine_order(
            image_paths,
            calibration,
            nominal_depth,
        )

        left_path = distance_output / "rectified_left.png"
        right_path = distance_output / "rectified_right.png"
        cv2.imwrite(str(left_path), rectified_left)
        cv2.imwrite(str(right_path), rectified_right)
        cv2.imwrite(
            str(distance_output / "rectified_left_with_roi.png"),
            annotate_source(rectified_left, distance_name),
        )

        distance_metadata = {
            "nominal_depth_mm": nominal_depth,
            "calibration_file": str(calibration_path),
            "selected_camera1_left_file": camera1_path.name,
            "selected_camera2_right_file": camera2_path.name,
            "order_candidates": order_candidates,
            "rectified_focal_px": calibration.focal_px,
            "baseline_yaml_mm": calibration.baseline_yaml_mm,
            "rectified_baseline_mm": calibration.baseline_mm,
            "doffs_px": calibration.doffs_px,
            "roi1": calibration.roi1,
            "roi2": calibration.roi2,
            "expected_disparity_px": (
                calibration.focal_px * calibration.baseline_mm / nominal_depth
            ),
            "target_roi_xyxy": TARGET_ROI[distance_name],
        }
        metadata["distances"][distance_name] = distance_metadata

        print(
            f"{distance_name}: selected left={camera1_path.name}, "
            f"right={camera2_path.name}, fx={calibration.focal_px:.3f}px, "
            f"baseline={calibration.baseline_mm:.3f}mm, "
            f"expected disparity={distance_metadata['expected_disparity_px']:.2f}px"
        )
        for candidate in order_candidates:
            print(
                "  order candidate: "
                f"{candidate['camera1_file']} -> {candidate['camera2_file']}, "
                f"score={candidate['score']:.3f}, "
                f"valid={candidate['valid_percent']:.1f}%, "
                f"plausible={candidate['plausible_percent']:.1f}%"
            )

        visualizations: list[tuple[ResultMetrics, np.ndarray]] = []
        for algorithm in ("BM", "Vitis-SGM"):
            for disparities in DISPARITY_COUNTS:
                stem = f"{algorithm.lower().replace('-', '_')}_d{disparities}"
                if algorithm == "BM":
                    disparity, runtime_ms = compute_bm_disparity(
                        rectified_left, rectified_right, disparities
                    )
                    estimated_memory_mib = float("nan")
                    disparity_q4 = np.zeros(disparity.shape, dtype=np.uint16)
                    positive = disparity > 0.0
                    disparity_q4[positive] = np.rint(
                        disparity[positive] * 16.0
                    ).astype(np.uint16)
                    cv2.imwrite(
                        str(distance_output / f"{stem}_disparity_q4_u16.png"),
                        disparity_q4,
                    )
                else:
                    raw_path = distance_output / f"{stem}_disparity_raw.png"
                    (
                        disparity,
                        runtime_ms,
                        estimated_memory_mib,
                        stdout,
                    ) = run_vitis_sgm(
                        sgbm_exe,
                        left_path,
                        right_path,
                        raw_path,
                        disparities,
                    )
                    (distance_output / f"{stem}_run.log").write_text(
                        stdout, encoding="utf-8"
                    )

                depth, valid = depth_from_disparity(
                    disparity, calibration, disparities
                )
                metrics = compute_metrics(
                    distance_name,
                    algorithm,
                    disparities,
                    runtime_ms,
                    estimated_memory_mib,
                    disparity,
                    depth,
                    valid,
                    calibration,
                )
                all_metrics.append(metrics)

                depth_path = distance_output / f"{stem}_depth_mm_u16.png"
                color_path = distance_output / f"{stem}_depth_color.png"
                write_depth_mm(depth_path, depth, valid)
                visualization = colorize_depth(depth, valid)
                cv2.imwrite(str(color_path), visualization)
                visualizations.append((metrics, visualization))

                print(
                    f"  {algorithm:9s} D={disparities:3d}: "
                    f"runtime={runtime_ms:8.1f} ms, "
                    f"valid={metrics.valid_percent:5.1f}%, "
                    f"inlier={metrics.nominal_inlier_percent:5.1f}%, "
                    f"median={metrics.median_depth_mm:7.1f} mm"
                )

        make_comparison(
            distance_name,
            rectified_left,
            visualizations,
            output_dir / f"comparison_{distance_name}.png",
        )

    save_metrics(all_metrics, output_dir)
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            sanitize_json(metadata),
            handle,
            ensure_ascii=False,
            indent=2,
            default=json_safe,
            allow_nan=False,
        )
    print(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
