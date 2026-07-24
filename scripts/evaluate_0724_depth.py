#!/usr/bin/env python3
"""Run a reproducible parameter comparison on the 0724 in-house stereo pair.

This data has no per-pixel ground-truth depth.  The script therefore reports
diagnostic, not accuracy, metrics:

* valid depth coverage in one common image region;
* pixels pinned to the upper end of the disparity search range;
* left-to-right photometric reprojection residual;
* local disparity outliers relative to a 5x5 median.

It also writes rectification and fixed-scale metric-depth comparison figures
for the README.
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

from evaluate_camera_dataset import Calibration, read_calibration, read_gray, rectify_pair


DEFAULT_DATASET = Path(
    "/home/hcc/Desktop/Public/datasets/自研相机/0724-imgs"
)
DEFAULT_PROJECT = Path("/home/hcc/Desktop/HXB/Vitis_Stereo_CPU_Ver")

# A fixed range is essential for an honest visual comparison.  Values outside
# this display range are clipped to the end colors; only invalid depth is black.
DEPTH_DISPLAY_MIN_MM = 300.0
DEPTH_DISPLAY_MAX_MM = 3000.0
DEPTH_VALID_MIN_MM = 150.0
DEPTH_VALID_MAX_MM = 10000.0

# All parameter sets are scored in exactly the same rectified-image region.
# x >= 280 excludes the unavoidable left search border even for D=256.
EVALUATION_ROI_XYXY = (280, 20, 1204, 1004)
ORDER_ROI_XYXY = (160, 20, 1200, 1000)
PHOTO_INLIER_THRESHOLD = 15.0

BM_PRE_FILTER_CAP = 31
BM_UNIQUENESS_RATIO = 15
BM_TEXTURE_THRESHOLD = 20


@dataclass(frozen=True)
class Configuration:
    key: str
    algorithm: str
    disparities: int
    block_size: int | None = None
    p1: int | None = None
    p2: int | None = None

    @property
    def title(self) -> str:
        if self.algorithm == "BM":
            return f"BM D={self.disparities}, block={self.block_size}"
        return f"Vitis-SGM D={self.disparities}, P1/P2={self.p1}/{self.p2}"


CONFIGURATIONS = (
    Configuration("bm_d128_b5", "BM", 128, block_size=5),
    Configuration("bm_d128_b11", "BM", 128, block_size=11),
    Configuration("bm_d128_b21", "BM", 128, block_size=21),
    Configuration("sgm_d64_p20_p40", "Vitis-SGM", 64, p1=20, p2=40),
    Configuration("sgm_d128_p20_p40", "Vitis-SGM", 128, p1=20, p2=40),
    Configuration("sgm_d256_p20_p40", "Vitis-SGM", 256, p1=20, p2=40),
    Configuration("sgm_d128_p10_p20", "Vitis-SGM", 128, p1=10, p2=20),
    Configuration("sgm_d128_p40_p80", "Vitis-SGM", 128, p1=40, p2=80),
)

COMPARISON_GROUPS = (
    (
        "comparison_sgm_range.png",
        "SGM disparity-range comparison",
        (
            "sgm_d64_p20_p40",
            "sgm_d128_p20_p40",
            "sgm_d256_p20_p40",
        ),
    ),
    (
        "comparison_sgm_penalties.png",
        "SGM smoothness-penalty comparison (D=128)",
        (
            "sgm_d128_p10_p20",
            "sgm_d128_p20_p40",
            "sgm_d128_p40_p80",
        ),
    ),
    (
        "comparison_bm_block_size.png",
        "BM block-size comparison (D=128)",
        (
            "bm_d128_b5",
            "bm_d128_b11",
            "bm_d128_b21",
        ),
    ),
)


@dataclass
class ResultMetrics:
    key: str
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: PROJECT/results/0724_imgs)",
    )
    parser.add_argument(
        "--sgm-exe",
        type=Path,
        default=None,
        help="Vitis scalar SGM executable with optional P1/P2 arguments",
    )
    return parser.parse_args()


def make_rect_mask(
    shape: tuple[int, int], roi_xyxy: tuple[int, int, int, int]
) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = roi_xyxy
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise RuntimeError(f"ROI {roi_xyxy} is outside {width}x{height}")
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def create_bm(disparities: int, block_size: int) -> cv2.StereoBM:
    matcher = cv2.StereoBM_create(
        numDisparities=disparities, blockSize=block_size
    )
    matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
    matcher.setPreFilterCap(BM_PRE_FILTER_CAP)
    matcher.setMinDisparity(0)
    matcher.setTextureThreshold(BM_TEXTURE_THRESHOLD)
    matcher.setUniquenessRatio(BM_UNIQUENESS_RATIO)
    return matcher


def compute_bm(
    left: np.ndarray,
    right: np.ndarray,
    disparities: int,
    block_size: int,
    repetitions: int = 5,
) -> tuple[np.ndarray, float]:
    cv2.setUseOptimized(False)
    matcher = create_bm(disparities, block_size)
    # Warm up allocations and dispatch before measuring.
    disparity_q4 = matcher.compute(left, right)
    runtimes: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        disparity_q4 = matcher.compute(left, right)
        runtimes.append((time.perf_counter() - start) * 1000.0)
    return disparity_q4.astype(np.float32) / 16.0, float(np.median(runtimes))


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
    residual = np.abs(
        left.astype(np.float32) - warped_right.astype(np.float32)
    )
    values = residual[map_valid]
    return (
        float(np.median(values)),
        100.0 * float(np.mean(values <= PHOTO_INLIER_THRESHOLD)),
    )


def score_order(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float]:
    disparity, _ = compute_bm(left, right, disparities=128, block_size=11)
    roi = make_rect_mask(disparity.shape, ORDER_ROI_XYXY)
    valid = roi & np.isfinite(disparity) & (disparity > 0.0) & (disparity < 127.0)
    valid_percent = 100.0 * float(np.sum(valid) / np.sum(roi))
    photo_median, photo_inlier = photometric_metrics(
        left, right, disparity, valid
    )
    # Both terms are independently useful.  Their sum makes the correct order
    # decisively preferable on this pair without assuming a target distance.
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


def determine_order(
    image_paths: list[Path],
    calibration: Calibration,
) -> tuple[Path, Path, np.ndarray, np.ndarray, list[dict[str, object]]]:
    if len(image_paths) != 2:
        raise RuntimeError(
            f"Expected exactly two BMP images, found {len(image_paths)}"
        )
    images = [
        read_gray(path, (calibration.width, calibration.height))
        for path in image_paths
    ]
    candidates: list[dict[str, object]] = []
    for first, second in ((0, 1), (1, 0)):
        left, right = rectify_pair(images[first], images[second], calibration)
        candidates.append(
            {
                "left_file": image_paths[first].name,
                "right_file": image_paths[second].name,
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
    left_path = next(
        path for path in image_paths if path.name == best["left_file"]
    )
    right_path = next(
        path for path in image_paths if path.name == best["right_file"]
    )
    return (
        left_path,
        right_path,
        np.asarray(best["_left"]),
        np.asarray(best["_right"]),
        public,
    )


def parse_logged_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else float("nan")


def run_sgm(
    executable: Path,
    left_path: Path,
    right_path: Path,
    raw_output: Path,
    config: Configuration,
) -> tuple[np.ndarray, float, float, str]:
    completed = subprocess.run(
        [
            str(executable),
            str(left_path),
            str(right_path),
            str(raw_output),
            str(config.disparities),
            str(config.p1),
            str(config.p2),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    disparity = cv2.imread(str(raw_output), cv2.IMREAD_UNCHANGED)
    if disparity is None:
        raise RuntimeError(f"SGM did not produce {raw_output}")
    runtime_ms = parse_logged_float(
        r"CPU compute:\s+([0-9.]+) ms", completed.stdout
    )
    memory_mib = parse_logged_float(
        r"Est\. memory:\s+([0-9.]+) MiB", completed.stdout
    )
    return (
        disparity.astype(np.float32),
        runtime_ms,
        memory_mib,
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
        # The last search bin is treated as range saturation, not a reliable
        # measurement.
        & (disparity < float(disparities - 1))
        & (denominator > 1.0e-6)
    )
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = (
        calibration.focal_px * calibration.baseline_mm / denominator[valid]
    )
    valid &= (
        (depth >= DEPTH_VALID_MIN_MM) & (depth <= DEPTH_VALID_MAX_MM)
    )
    depth[~valid] = np.nan
    return depth, valid


def compute_metrics(
    config: Configuration,
    runtime_ms: float,
    memory_mib: float,
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
) -> ResultMetrics:
    roi = make_rect_mask(disparity.shape, EVALUATION_ROI_XYXY)
    roi_count = int(np.sum(roi))
    scored_valid = roi & valid
    valid_percent = 100.0 * float(np.sum(scored_valid) / roi_count)
    saturated = roi & (disparity >= float(config.disparities - 1))
    saturated_percent = 100.0 * float(np.sum(saturated) / roi_count)

    photo_median, photo_inlier = photometric_metrics(
        left, right, disparity, scored_valid
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

    valid_depth = depth[scored_valid]
    valid_disparity = disparity[scored_valid]
    if valid_depth.size:
        p10_depth, median_depth, p90_depth = (
            float(value)
            for value in np.percentile(valid_depth, [10.0, 50.0, 90.0])
        )
        median_disparity_px = float(np.median(valid_disparity))
    else:
        p10_depth = median_depth = p90_depth = float("nan")
        median_disparity_px = float("nan")

    return ResultMetrics(
        key=config.key,
        algorithm=config.algorithm,
        disparities=config.disparities,
        block_size=config.block_size,
        p1=config.p1,
        p2=config.p2,
        runtime_ms=runtime_ms,
        estimated_memory_mib=memory_mib,
        valid_percent=valid_percent,
        saturated_percent=saturated_percent,
        photo_median_abs_gray=photo_median,
        photo_inlier_percent=photo_inlier,
        local_outlier_percent=local_outlier_percent,
        median_disparity_px=median_disparity_px,
        p10_depth_mm=p10_depth,
        median_depth_mm=median_depth,
        p90_depth_mm=p90_depth,
    )


def write_depth_mm(path: Path, depth: np.ndarray, valid: np.ndarray) -> None:
    output = np.zeros(depth.shape, dtype=np.uint16)
    output[valid] = np.rint(
        np.clip(depth[valid], 0.0, 65535.0)
    ).astype(np.uint16)
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"Cannot write depth image: {path}")


def colorize_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        depth, DEPTH_DISPLAY_MIN_MM, DEPTH_DISPLAY_MAX_MM
    )
    scaled = 1.0 - (
        (clipped - DEPTH_DISPLAY_MIN_MM)
        / (DEPTH_DISPLAY_MAX_MM - DEPTH_DISPLAY_MIN_MM)
    )
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.rint(255.0 * scaled[valid]).astype(np.uint8)
    output = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    output[~valid] = 0
    return output


def annotate_source(image: np.ndarray) -> np.ndarray:
    source = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = EVALUATION_ROI_XYXY
    cv2.rectangle(source, (x0, y0), (x1, y1), (0, 255, 0), 3)
    return source


def panel(
    image: np.ndarray,
    title: str,
    lines: tuple[str, ...],
    width: int = 330,
) -> np.ndarray:
    image_height = int(round(image.shape[0] * width / image.shape[1]))
    resized = cv2.resize(
        image, (width, image_height), interpolation=cv2.INTER_AREA
    )
    header_height = 31 + 22 * len(lines)
    header = np.full((header_height, width, 3), 24, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            header,
            line,
            (8, 45 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([header, resized])


def format_value(value: float, decimals: int = 1) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def colorbar(width: int, heading: str, height: int = 67) -> np.ndarray:
    output = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(
        output,
        heading,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    x0, x1 = 125, width - 125
    gradient = np.linspace(255, 0, x1 - x0, dtype=np.uint8)[None, :]
    gradient = np.repeat(gradient, 16, axis=0)
    output[27:43, x0:x1] = cv2.applyColorMap(
        gradient, cv2.COLORMAP_TURBO
    )
    cv2.putText(
        output,
        f"near {DEPTH_DISPLAY_MIN_MM / 1000.0:.1f} m",
        (8, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"far {DEPTH_DISPLAY_MAX_MM / 1000.0:.1f} m",
        (width - 103, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Fixed scale; black = invalid/range saturation; green = metrics ROI",
        (8, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )
    return output


def make_comparison(
    rectified_left: np.ndarray,
    result_images: dict[str, np.ndarray],
    metrics_by_key: dict[str, ResultMetrics],
    keys: tuple[str, ...],
    heading: str,
    output_path: Path,
) -> None:
    items = [
        panel(
            annotate_source(rectified_left),
            "Rectified left",
            ("green = common metrics ROI", "fixed depth colors in all panels"),
        )
    ]
    configs = {config.key: config for config in CONFIGURATIONS}
    for key in keys:
        config = configs[key]
        metrics = metrics_by_key[key]
        items.append(
            panel(
                result_images[key],
                config.title,
                (
                    "valid "
                    f"{format_value(metrics.valid_percent)}% | saturated "
                    f"{format_value(metrics.saturated_percent)}%",
                    "photo median "
                    f"{format_value(metrics.photo_median_abs_gray)} | local outlier "
                    f"{format_value(metrics.local_outlier_percent)}%",
                ),
            )
        )
    comparison = np.hstack(items)
    comparison = np.vstack(
        [comparison, colorbar(comparison.shape[1], heading)]
    )
    if not cv2.imwrite(str(output_path), comparison):
        raise RuntimeError(f"Cannot write comparison: {output_path}")


def make_rectification_check(
    left: np.ndarray,
    right: np.ndarray,
    output_path: Path,
) -> None:
    display_width = 612
    display_height = int(round(left.shape[0] * display_width / left.shape[1]))
    displays = []
    for image, label in ((left, "rectified left"), (right, "rectified right")):
        resized = cv2.resize(
            image, (display_width, display_height), interpolation=cv2.INTER_AREA
        )
        display = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            display,
            label,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for y in range(64, display_height, 64):
            cv2.line(display, (0, y), (display_width - 1, y), (0, 255, 0), 1)
        displays.append(display)
    comparison = np.hstack(displays)
    footer = np.full((38, comparison.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(
        footer,
        "Horizontal guides verify that corresponding features lie on the same scanline",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    comparison = np.vstack([comparison, footer])
    if not cv2.imwrite(str(output_path), comparison):
        raise RuntimeError(f"Cannot write rectification check: {output_path}")


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


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    dataset = args.dataset.resolve()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else project / "results" / "0724_imgs"
    )
    details_dir = output_dir / "details"
    sgm_exe = (
        args.sgm_exe.resolve()
        if args.sgm_exe is not None
        else project / "build" / "vitis_sgbm_cpu"
    )
    if not dataset.is_dir():
        raise RuntimeError(f"Missing dataset directory: {dataset}")
    if not sgm_exe.is_file():
        raise RuntimeError(
            f"Missing executable: {sgm_exe}. Build the CMake project first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

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

    rectified_left_path = details_dir / "rectified_left.png"
    rectified_right_path = details_dir / "rectified_right.png"
    if not cv2.imwrite(str(rectified_left_path), rectified_left):
        raise RuntimeError(f"Cannot write {rectified_left_path}")
    if not cv2.imwrite(str(rectified_right_path), rectified_right):
        raise RuntimeError(f"Cannot write {rectified_right_path}")
    make_rectification_check(
        rectified_left,
        rectified_right,
        output_dir / "rectification_check.png",
    )

    print(
        f"Selected left={left_source.name}, right={right_source.name}; "
        f"rectified fx={calibration.focal_px:.3f}px, "
        f"baseline={calibration.baseline_mm:.3f}mm"
    )
    for candidate in order_candidates:
        print(
            "  order candidate: "
            f"{candidate['left_file']} -> {candidate['right_file']}, "
            f"score={candidate['score']:.2f}, "
            f"valid={candidate['valid_percent']:.1f}%, "
            f"photo median={candidate['photo_median_abs_gray']:.1f}"
        )

    all_metrics: list[ResultMetrics] = []
    result_images: dict[str, np.ndarray] = {}
    metrics_by_key: dict[str, ResultMetrics] = {}
    for config in CONFIGURATIONS:
        if config.algorithm == "BM":
            assert config.block_size is not None
            disparity, runtime_ms = compute_bm(
                rectified_left,
                rectified_right,
                config.disparities,
                config.block_size,
            )
            memory_mib = float("nan")
            disparity_output = np.zeros(disparity.shape, dtype=np.uint16)
            positive = disparity > 0.0
            disparity_output[positive] = np.rint(
                disparity[positive] * 16.0
            ).astype(np.uint16)
            cv2.imwrite(
                str(details_dir / f"{config.key}_disparity_q4_u16.png"),
                disparity_output,
            )
        else:
            raw_path = details_dir / f"{config.key}_disparity_raw.png"
            disparity, runtime_ms, memory_mib, stdout = run_sgm(
                sgm_exe,
                rectified_left_path,
                rectified_right_path,
                raw_path,
                config,
            )
            (details_dir / f"{config.key}_run.log").write_text(
                stdout, encoding="utf-8"
            )

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
        cv2.imwrite(
            str(details_dir / f"{config.key}_depth_color.png"), color
        )
        all_metrics.append(metrics)
        result_images[config.key] = color
        metrics_by_key[config.key] = metrics
        print(
            f"  {config.title}: runtime={runtime_ms:.1f}ms, "
            f"valid={metrics.valid_percent:.1f}%, "
            f"saturated={metrics.saturated_percent:.1f}%, "
            f"photo median={metrics.photo_median_abs_gray:.1f}, "
            f"local outlier={metrics.local_outlier_percent:.1f}%"
        )

    for filename, heading, keys in COMPARISON_GROUPS:
        make_comparison(
            rectified_left,
            result_images,
            metrics_by_key,
            keys,
            heading,
            output_dir / filename,
        )

    save_metrics(all_metrics, output_dir)
    metadata = {
        "dataset": str(dataset),
        "calibration_file": str(calibration_path),
        "ground_truth_available": False,
        "selected_left_file": left_source.name,
        "selected_right_file": right_source.name,
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
        "bm_fixed_parameters": {
            "pre_filter_cap": BM_PRE_FILTER_CAP,
            "uniqueness_ratio": BM_UNIQUENESS_RATIO,
            "texture_threshold": BM_TEXTURE_THRESHOLD,
            "min_disparity": 0,
        },
        "sgm_fixed_parameters": {
            "census_window": 5,
            "paths": 4,
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
    print(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
