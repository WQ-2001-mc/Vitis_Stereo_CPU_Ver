#!/usr/bin/env python3
"""Run the production Gemini335 Vitis SGBM kernel in full-frame HLS C simulation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    "/home/hcc/Desktop/Public/datasets/gemini335/1280*800分辨率/data"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/gemini335_1280x800_sgbm_fpga_equiv"
DEFAULT_FPGA_ROOT = Path(
    "/home/hcc/Desktop/HXB/FPGA_Camera_V2/"
    "0805-AXU5EVB_Board_SGBM_1280x800/Vitis_Libraries/vision/L1/"
    "examples/gemini335_sgbm_pl"
)
DEFAULT_VITIS_HLS = Path("/tools/Xilinx/Vitis_HLS/2021.1/bin/vitis_hls")
DEFAULT_PENALTIES = "10/20,20/40,40/80,10/40,20/80"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fpga-root", type=Path, default=DEFAULT_FPGA_ROOT)
    parser.add_argument("--vitis-hls", type=Path, default=DEFAULT_VITIS_HLS)
    parser.add_argument("--penalties", default=DEFAULT_PENALTIES)
    parser.add_argument(
        "--reuse-csim",
        action="store_true",
        help="Skip Vitis HLS when all requested PGM outputs already exist",
    )
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


def read_gray(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype.name != "uint8" or image.shape != (800, 1280):
        raise ValueError(
            f"expected uint8 1280x800 input at {path}, got "
            f"shape={image.shape}, dtype={image.dtype}"
        )
    return image


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    fpga_root = args.fpga_root.resolve()
    left = (args.left or dataset / "left_IR_IR Left.png").resolve()
    right = (args.right or dataset / "right_IR_IR Right.png").resolve()
    pairs = penalty_pairs(args.penalties)
    required_sources = (
        fpga_root / "hls/gemini335_sgbm_accel.cpp",
        fpga_root / "hls/gemini335_sgbm_config.hpp",
        args.vitis_hls,
    )
    for path in required_sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    csim_dir = output / "hls_csim"
    csim_dir.mkdir(parents=True, exist_ok=True)
    left_pgm = csim_dir / "left_input_u8.pgm"
    right_pgm = csim_dir / "right_input_u8.pgm"
    if not cv2.imwrite(str(left_pgm), read_gray(left)):
        raise RuntimeError(f"failed to write {left_pgm}")
    if not cv2.imwrite(str(right_pgm), read_gray(right)):
        raise RuntimeError(f"failed to write {right_pgm}")

    expected = [
        csim_dir / f"hls_depth_p1_{p1:03d}_p2_{p2:03d}.pgm"
        for p1, p2 in pairs
    ]
    if not (args.reuse_csim and all(path.is_file() for path in expected)):
        environment = os.environ.copy()
        environment.update(
            {
                "GEMINI_FPGA_HLS_ROOT": str(fpga_root),
                "GEMINI_CSIM_LEFT_PGM": str(left_pgm),
                "GEMINI_CSIM_RIGHT_PGM": str(right_pgm),
                "GEMINI_CSIM_OUTPUT_DIR": str(csim_dir),
                "GEMINI_CSIM_PENALTIES": ",".join(
                    f"{p1}/{p2}" for p1, p2 in pairs
                ),
            }
        )
        completed = subprocess.run(
            [
                str(args.vitis_hls),
                "-f",
                str(PROJECT_ROOT / "hls_csim/run_dataset_csim.tcl"),
            ],
            cwd=csim_dir,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        console = completed.stdout + completed.stderr
        (csim_dir / "vitis_hls_console.log").write_text(
            console, encoding="utf-8"
        )
        print(console, end="")
        if completed.returncode != 0:
            return completed.returncode

    for path in expected:
        if not path.is_file():
            raise RuntimeError(f"missing HLS C-sim output: {path}")
    analyzer = PROJECT_ROOT / "scripts/analyze_vitis_hls_csim_results.py"
    return subprocess.run(
        [
            sys.executable,
            str(analyzer),
            "--results",
            str(output),
            "--fpga-root",
            str(fpga_root),
            "--penalties",
            ",".join(f"{p1}/{p2}" for p1, p2 in pairs),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
