#!/usr/bin/env python3
"""Validate optimized scalar SGM outputs against exact Vitis HLS C simulation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from run_stereobm_depth_sweep import (
    add_horizontal_colorbar,
    depth_color,
    fit_panel_image,
    put_centered,
    write_image,
)


DEFAULT_RESULTS = Path(
    "results/gemini335_1280x800_sgbm_fpga_equiv"
)
DEFAULT_FPGA_ROOT = Path(
    "/home/hcc/Desktop/HXB/FPGA_Camera_V2/"
    "0805-AXU5EVB_Board_SGBM_1280x800/Vitis_Libraries/vision/L1/"
    "examples/gemini335_sgbm_pl"
)
DEFAULT_PENALTIES = "10/20,20/40,40/80,10/40,20/80"
BF_MM_PX = 31205.250210849994
DISPARITIES = 128
DEPTH_MIN_MM = 300.0
DEPTH_MAX_MM = 3000.0
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fpga-root", type=Path, default=DEFAULT_FPGA_ROOT)
    parser.add_argument("--penalties", default=DEFAULT_PENALTIES)
    return parser.parse_args()


def penalty_pairs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for entry in text.split(","):
        p1_text, separator, p2_text = entry.strip().partition("/")
        if not separator:
            raise ValueError(f"invalid P1/P2 entry: {entry}")
        p1, p2 = int(p1_text), int(p2_text)
        if p1 < 0 or p2 <= p1 or p2 > 100:
            raise ValueError(f"invalid Vitis penalties: {p1}/{p2}")
        pairs.append((p1, p2))
    if not pairs:
        raise ValueError("at least one P1/P2 pair is required")
    return pairs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_depth_lut() -> np.ndarray:
    lut = np.zeros(DISPARITIES, dtype=np.uint16)
    for disparity in range(2, DISPARITIES):
        depth = int(np.floor(BF_MM_PX / disparity + 0.5))
        if depth <= 20000:
            lut[disparity] = depth
    return lut


def read_metrics(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(row["p1"]), int(row["p2"])): row for row in rows}


def make_comparison(
    images: list[np.ndarray],
    validations: list[dict[str, object]],
    output: Path,
) -> None:
    panel_width = 360
    panel_image_height = 300
    header_height = 76
    footer_height = 76
    colorbar_height = 82
    canvas = np.full(
        (
            header_height + panel_image_height + footer_height + colorbar_height,
            panel_width * len(images),
            3,
        ),
        20,
        dtype=np.uint8,
    )
    for index, (image, item) in enumerate(zip(images, validations)):
        x0 = index * panel_width
        cell = canvas[:, x0 : x0 + panel_width]
        put_centered(
            cell,
            f"P1/P2 = {item['p1']}/{item['p2']}",
            32,
            0.68,
            (245, 245, 245),
            2,
        )
        put_centered(
            cell,
            "EXACT VITIS HLS C-SIM",
            59,
            0.43,
            (190, 190, 190),
            1,
        )
        cell[
            header_height : header_height + panel_image_height,
            :,
        ] = fit_panel_image(image, panel_width, panel_image_height)
        footer_y = header_height + panel_image_height
        put_centered(
            cell,
            f"0.3-3 m ROI {item['usable_roi_pct']:.1f}%",
            footer_y + 27,
            0.46,
            (230, 230, 230),
            1,
        )
        put_centered(
            cell,
            f"scalar match {item['exact_match_pct']:.4f}%",
            footer_y + 55,
            0.46,
            (230, 230, 230),
            1,
        )
        cv2.rectangle(
            cell,
            (0, 0),
            (panel_width - 1, header_height + panel_image_height + footer_height - 1),
            (78, 78, 78),
            1,
        )
    add_horizontal_colorbar(
        canvas,
        header_height + panel_image_height + footer_height + 8,
        DEPTH_MIN_MM,
        DEPTH_MAX_MM,
    )
    write_image(output, canvas)


def main() -> int:
    args = parse_args()
    results = args.results.resolve()
    fpga_root = args.fpga_root.resolve()
    penalties = penalty_pairs(args.penalties)
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    metrics = read_metrics(results / "metrics.csv")
    x0 = int(metadata["common_roi"]["x0"])
    y0 = int(metadata["common_roi"]["y0"])
    x1 = int(metadata["common_roi"]["x1"])
    y1 = int(metadata["common_roi"]["y1"])
    roi = np.zeros((800, 1280), dtype=bool)
    roi[y0:y1, x0:x1] = True
    lut = exact_depth_lut()

    validations: list[dict[str, object]] = []
    color_images: list[np.ndarray] = []
    csim_dir = results / "hls_csim"
    for p1, p2 in penalties:
        hls_pgm = csim_dir / f"hls_depth_p1_{p1:03d}_p2_{p2:03d}.pgm"
        hls_depth = cv2.imread(str(hls_pgm), cv2.IMREAD_UNCHANGED)
        if hls_depth is None or hls_depth.shape != (800, 1280):
            raise RuntimeError(f"invalid HLS C-sim output: {hls_pgm}")
        config_dir = results / metrics[(p1, p2)]["config_directory"]
        scalar_disparity = cv2.imread(
            str(config_dir / "left_disparity_raw.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if scalar_disparity is None or scalar_disparity.shape != hls_depth.shape:
            raise RuntimeError(f"invalid scalar disparity: {config_dir}")
        scalar_depth = lut[scalar_disparity]
        different = hls_depth != scalar_depth
        exact_match = 100.0 * float(np.count_nonzero(~different)) / hls_depth.size
        valid = hls_depth > 0
        usable = valid & (hls_depth >= DEPTH_MIN_MM) & (hls_depth <= DEPTH_MAX_MM)
        usable_roi = usable & roi
        valid_samples = hls_depth[valid]
        selected_samples = hls_depth[usable_roi]
        item: dict[str, object] = {
            "p1": p1,
            "p2": p2,
            "pixels": int(hls_depth.size),
            "different_pixels": int(np.count_nonzero(different)),
            "exact_match_pct": exact_match,
            "maximum_abs_difference_mm": int(
                np.max(
                    np.abs(
                        hls_depth.astype(np.int32) - scalar_depth.astype(np.int32)
                    )
                )
            ),
            "hls_valid_pixels": int(np.count_nonzero(valid)),
            "hls_valid_full_frame_pct": 100.0 * float(np.mean(valid)),
            "usable_roi_pct": 100.0
            * float(np.count_nonzero(usable_roi))
            / float(np.count_nonzero(roi)),
            "hls_valid_depth_p05_mm": float(np.percentile(valid_samples, 5)),
            "hls_valid_depth_p50_mm": float(np.percentile(valid_samples, 50)),
            "hls_valid_depth_p95_mm": float(np.percentile(valid_samples, 95)),
            "usable_roi_depth_p50_mm": float(np.percentile(selected_samples, 50)),
            "scalar_lr_consistent_coverage_pct": float(
                metrics[(p1, p2)]["lr_consistent_coverage_pct"]
            ),
            "scalar_raw_local_outlier_gt2px_pct": float(
                metrics[(p1, p2)]["raw_local_outlier_gt2px_pct"]
            ),
        }
        validations.append(item)
        cv2.imwrite(
            str(csim_dir / f"hls_depth_p1_{p1:03d}_p2_{p2:03d}_u16.png"),
            hls_depth,
        )
        color = depth_color(
            hls_depth.astype(np.float32),
            usable,
            DEPTH_MIN_MM,
            DEPTH_MAX_MM,
        )
        cv2.imwrite(
            str(csim_dir / f"hls_depth_p1_{p1:03d}_p2_{p2:03d}_color.png"),
            color,
        )
        color_images.append(color)

    sources = {
        "accelerator": fpga_root / "hls" / "gemini335_sgbm_accel.cpp",
        "configuration": fpga_root / "hls" / "gemini335_sgbm_config.hpp",
        "vitis_sgbm_header": fpga_root.parent.parent.parent.parent
        / "vision"
        / "L1"
        / "include"
        / "imgproc"
        / "xf_sgbm.hpp",
        "left_input": Path(metadata["left"]),
        "right_input": Path(metadata["right"]),
        "calibration": Path(metadata["calibration_path"]),
    }
    for label, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "verdict": (
            "bit-exact"
            if all(item["different_pixels"] == 0 for item in validations)
            else "different"
        ),
        "comparison": (
            "optimized scalar R=3 output mapped through the exact PL depth LUT "
            "versus Vitis HLS 2021.1 C simulation of the production accelerator"
        ),
        "fixed_hls_configuration": {
            "width": 1280,
            "height": 800,
            "total_disparity": 128,
            "parallel_units": 64,
            "num_directions": 3,
            "window_size": 5,
            "input_type": "XF_8UC1",
            "disparity_type": "XF_8UC1 integer pixels",
            "depth_type": "XF_16UC1 millimetres",
        },
        "runtime_penalties": [list(pair) for pair in penalties],
        "fx_times_baseline_mm_px": BF_MM_PX,
        "depth_lut_rule": (
            "d=0/1 -> 0; d=2..127 -> round(31205.250210849994/d) mm"
        ),
        "common_roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "source_sha256": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in sources.items()
        },
        "results": validations,
    }
    (results / "hls_csim_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (results / "hls_csim_validation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validations[0].keys()))
        writer.writeheader()
        writer.writerows(validations)

    make_comparison(
        color_images,
        validations,
        results / "comparison_hls_csim_penalties.png",
    )
    csim_log = (
        Path(__file__).resolve().parents[1]
        / "build"
        / "gemini335_dataset_hls_csim"
        / "sol1"
        / "csim"
        / "report"
        / "gemini335_sgbm_accel_csim.log"
    )
    if csim_log.is_file():
        shutil.copyfile(csim_log, csim_dir / "gemini335_sgbm_accel_csim.log")

    print(
        f"HLS_CSIM_VALIDATION verdict={summary['verdict']} "
        f"configurations={len(validations)}"
    )
    for item in validations:
        print(
            f"P1/P2={item['p1']}/{item['p2']} "
            f"exact={item['exact_match_pct']:.4f}% "
            f"different={item['different_pixels']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
