# Vitis Stereo CPU Version

## 目录

1. [项目简介](#1-项目简介)
2. [包含的算法](#2-包含的算法)
3. [快速开始](#3-快速开始)
   - [3.1 依赖](#31-依赖)
   - [3.2 一键编译并运行](#32-一键编译并运行)
   - [3.3 手动编译](#33-手动编译)
   - [3.4 分别运行](#34-分别运行)
   - [3.5 交互式扫描 D/W 并生成深度报告](#35-交互式扫描-dw-并生成深度报告)
4. [输出](#4-输出)
5. [参数建议](#5-参数建议)
6. [最新参数扫描结果](#6-最新参数扫描结果)
7. [仓库结构](#7-仓库结构)
8. [详细文档](#8-详细文档)
   - [8.1 原理与使用说明](#81-原理与使用说明)
   - [8.2 实验报告](#82-实验报告)
9. [License](#9-license)

## 1. 项目简介

从 `Vitis_Libraries/vision` 立体匹配示例中提取的纯 CPU 测试工程，用于在
不依赖 FPGA 工具链的环境中验证 BM 与 SGM 两条立体匹配路线。

工程只依赖 C++17 和 OpenCV，不依赖 Vitis、HLS、`ap_int`、OpenCL/XRT
或 FPGA bitstream。

## 2. 包含的算法

| 程序 | 算法 | 实现来源 | 默认配置 |
|---|---|---|---|
| `vitis_bm_cpu` | StereoBM，局部 SAD 块匹配 | OpenCV `cv::StereoBM` | `D=32`，`block=11` |
| `vitis_sgbm_cpu` | Census 代价 + 4 路径 SGM | Vitis SGBM 测试台的标量 CPU 参考实现 | `D=64`，Census 5×5，`P1/P2=20/40` |

这里的 `vitis_sgbm_cpu` 不是 OpenCV `cv::StereoSGBM`。Vitis 示例目录名和
硬件接口使用 SGBM/SemiGlobalBM，而提取出的 CPU 参考函数实际执行
Census + 4 路径 SGM。

## 3. 快速开始

### 3.1 依赖

- 支持 C++17 的编译器
- CMake 3.10 或更高版本
- OpenCV：`core`、`imgproc`、`imgcodecs`、`calib3d`

输入必须是尺寸相同、已经完成双目校正的左右图。程序读取后会统一转换为
单通道灰度图。

### 3.2 一键编译并运行

```bash
cd /home/hcc/Desktop/HXB/Vitis_Stereo_CPU_Ver
./run_cpu_tests.sh LEFT.png RIGHT.png
```

脚本会自动配置和编译工程，随后依次运行 BM 与 Vitis-SGM。结果默认写入
`output/`；也可以传入第三个参数指定输出目录：

```bash
./run_cpu_tests.sh LEFT.png RIGHT.png OUTPUT_DIR
```

如需显式指定 OpenCV：

```bash
OpenCV_DIR=/path/to/opencv/lib/cmake/opencv4 \
  ./run_cpu_tests.sh LEFT.png RIGHT.png
```

### 3.3 手动编译

```bash
cd /home/hcc/Desktop/HXB/Vitis_Stereo_CPU_Ver
/usr/bin/cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/home/hcc/Desktop/HXB/opencv-install-min/lib/cmake/opencv4
/usr/bin/cmake --build build --parallel
```

如果 CMake 能自动找到系统安装的 OpenCV，可以省略 `OpenCV_DIR`。

### 3.4 分别运行

使用默认参数：

```bash
./build/vitis_bm_cpu LEFT.png RIGHT.png bm_visual.png
./build/vitis_sgbm_cpu LEFT.png RIGHT.png sgm_raw.png
```

覆盖视差数和 BM 窗口：

```bash
./build/vitis_bm_cpu LEFT.png RIGHT.png bm_visual.png 128 11
```

覆盖 Vitis-SGM 的视差数和 `P1/P2`：

```bash
./build/vitis_sgbm_cpu LEFT.png RIGHT.png sgm_raw.png 128 20 40
```

BM 的视差数必须是 16 的倍数。

### 3.5 交互式扫描 D/W 并生成深度报告

需要对一组校正后的双目图批量比较 `D=64/128/256` 和不同窗口 `W` 时，
直接运行：

```bash
cd /home/hcc/Desktop/HXB/Vitis_Stereo_CPU_Ver
./scripts/run_stereobm_depth_sweep.py
```

脚本会提示输入数据目录、左右图、标定 YAML、输出目录、D/W 列表、统一深度
色标和计时次数；如果数据目录中恰好各有一个名称包含 `left`、`right` 的图像
及一个 YAML，会自动识别。除 C++ 工程依赖外，脚本需要 Python 3、`cv2`
和 NumPy。

也可以完整指定参数后非交互运行：

```bash
python3 scripts/run_stereobm_depth_sweep.py \
  --non-interactive --dataset "/path/to/data" \
  --output "/path/to/results" \
  --disparities 64,128,256 --windows 5,9,11,21 \
  --depth-min-mm 300 --depth-max-mm 3000 --runs 5
```

脚本会实际调用本仓库的 `vitis_bm_cpu`，保存 signed Q4 视差和毫米深度，
并生成固定色标的横向对比图、CSV/JSON 指标及 Markdown 报告。当前
Gemini335 1280×800 样本的结果见
[StereoBM D/W 深度图横向对比](results/gemini335_1280x800_bm_sweep/STEREOBM_DEPTH_COMPARISON.md)。

## 4. 输出

- BM 输出按 Vitis 测试台的方式缩放为 8 位视差可视化图。
- Vitis-SGM 的指定输出文件保存原始整数视差，并自动生成同目录下带
  `_visual` 后缀的 8 位拉伸图。
- 一键脚本的默认结果是：
  `output/bm_disparity_visual.png`、
  `output/sgbm_disparity_raw.png` 和
  `output/sgbm_disparity_raw_visual.png`。

原始视差格式、内存优化方式等细节见
[输出格式说明](docs/output-format.md)。

## 5. 参数建议

针对当前自研相机的 0.5–3 m 目标范围，现有 0.5/0.75 m 真值实验和多距离
场景验证建议把 `D=128` 作为通用视差范围：

- 质量和有效覆盖优先：Vitis-SGM，`D=128`，`P1/P2=20/40`。
- 大平面连续性优先：Vitis-SGM，`D=128`，`P1/P2=40/80`。
- CPU 速度优先：StereoBM，`D=128`，`block=11`。

这些参数来自当前相机、标定和数据集，不应直接视为其他相机的最优值。完整的
量化结果、适用边界和资源取舍见
[0.5–3 m 参数与定量汇总](docs/parameter-recommendations.md)。

## 6. 最新参数扫描结果

Gemini335 1280×800 双目 IR 图已经完成 StereoBM 参数扫描，当前比较范围为
`D=64/128/256`、`W=5/9/11/21`，共 12 组配置。所有深度图使用统一的
300–3000 mm 色标和共同评价 ROI。

- [完整 Markdown 报告：Gemini335 StereoBM D/W 参数深度图横向对比](results/gemini335_1280x800_bm_sweep/STEREOBM_DEPTH_COMPARISON.md)
- [12 组参数总对比图](results/gemini335_1280x800_bm_sweep/comparison_all.png)
- [完整量化指标 CSV](results/gemini335_1280x800_bm_sweep/metrics.csv)
- [实验参数与可复现元数据](results/gemini335_1280x800_bm_sweep/metadata.json)

当前数据上可优先把 `D=128/W=11` 作为效果折中基线，同时保留
`D=128/W=9` 作为更小窗口的硬件候选。CPU 参考时间不能直接视为 FPGA
延时，资源和时序仍需通过对应 HLS 顶层的 C 综合报告评估。

## 7. 仓库结构

```text
Vitis_Stereo_CPU_Ver/
├── CMakeLists.txt                 # CMake 构建配置
├── run_cpu_tests.sh               # 编译并依次运行两种算法
├── src/
│   ├── vitis_bm_cpu.cpp           # OpenCV StereoBM CPU 程序
│   └── vitis_sgbm_cpu.cpp         # Vitis Census + 4 路径 SGM CPU 程序
├── scripts/                       # 数据集评估与结果生成脚本
├── docs/                          # 文档索引、算法、参数和实验报告
├── results/                       # 纳入版本管理的图表与指标
├── output/                        # 本地运行输出，Git 忽略
└── README.md
```

`results/*/details/` 保存可再生成的详细中间结果，默认由 Git 忽略；汇总图、
CSV、JSON 和元数据保留在各结果目录的顶层。

## 8. 详细文档

### 8.1 原理与使用说明

- [BM/SGBM/SGM 的来源与区别](docs/algorithm-background.md)
- [输出格式与内存实现说明](docs/output-format.md)
- [0.5–3 m 参数与定量汇总](docs/parameter-recommendations.md)

### 8.2 实验报告

- [自研相机 50 cm / 75 cm 深度实验](docs/experiment-camera-50cm-75cm.md)
- [0724 多距离室内场景参数验证](docs/experiment-0724-parameter-validation.md)
- [Gemini335 与之前相机同场景对比](docs/experiment-gemini335-comparison.md)
- [Gemini335 1280×800 StereoBM D/W 深度图横向对比](results/gemini335_1280x800_bm_sweep/STEREOBM_DEPTH_COMPARISON.md)
- [0725-test BM 与 Vitis-SGM 深度效果](docs/experiment-0725-bm-sgm.md)

## 9. License

见 [LICENSE](LICENSE)。
