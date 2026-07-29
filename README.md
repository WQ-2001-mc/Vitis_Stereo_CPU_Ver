# Vitis Stereo CPU Version

从 `Vitis_Libraries/vision` 立体匹配示例中提取的纯 CPU 测试工程，用于在
不依赖 FPGA 工具链的环境中验证 BM 与 SGM 两条立体匹配路线。

工程只依赖 C++17 和 OpenCV，不依赖 Vitis、HLS、`ap_int`、OpenCL/XRT
或 FPGA bitstream。

## 包含的算法

| 程序 | 算法 | 实现来源 | 默认配置 |
|---|---|---|---|
| `vitis_bm_cpu` | StereoBM，局部 SAD 块匹配 | OpenCV `cv::StereoBM` | `D=32`，`block=11` |
| `vitis_sgbm_cpu` | Census 代价 + 4 路径 SGM | Vitis SGBM 测试台的标量 CPU 参考实现 | `D=64`，Census 5×5，`P1/P2=20/40` |

这里的 `vitis_sgbm_cpu` 不是 OpenCV `cv::StereoSGBM`。Vitis 示例目录名和
硬件接口使用 SGBM/SemiGlobalBM，而提取出的 CPU 参考函数实际执行
Census + 4 路径 SGM。

## 快速开始

### 依赖

- 支持 C++17 的编译器
- CMake 3.10 或更高版本
- OpenCV：`core`、`imgproc`、`imgcodecs`、`calib3d`

输入必须是尺寸相同、已经完成双目校正的左右图。程序读取后会统一转换为
单通道灰度图。

### 一键编译并运行

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

### 手动编译

```bash
cd /home/hcc/Desktop/HXB/Vitis_Stereo_CPU_Ver
/usr/bin/cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/home/hcc/Desktop/HXB/opencv-install-min/lib/cmake/opencv4
/usr/bin/cmake --build build --parallel
```

如果 CMake 能自动找到系统安装的 OpenCV，可以省略 `OpenCV_DIR`。

### 分别运行

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

## 输出

- BM 输出按 Vitis 测试台的方式缩放为 8 位视差可视化图。
- Vitis-SGM 的指定输出文件保存原始整数视差，并自动生成同目录下带
  `_visual` 后缀的 8 位拉伸图。
- 一键脚本的默认结果是：
  `output/bm_disparity_visual.png`、
  `output/sgbm_disparity_raw.png` 和
  `output/sgbm_disparity_raw_visual.png`。

原始视差格式、内存优化方式等细节见
[输出格式说明](docs/output-format.md)。

## 参数建议

针对当前自研相机的 0.5–3 m 目标范围，现有 0.5/0.75 m 真值实验和多距离
场景验证建议把 `D=128` 作为通用视差范围：

- 质量和有效覆盖优先：Vitis-SGM，`D=128`，`P1/P2=20/40`。
- 大平面连续性优先：Vitis-SGM，`D=128`，`P1/P2=40/80`。
- CPU 速度优先：StereoBM，`D=128`，`block=11`。

这些参数来自当前相机、标定和数据集，不应直接视为其他相机的最优值。完整的
量化结果、适用边界和资源取舍见
[0.5–3 m 参数与定量汇总](docs/parameter-recommendations.md)。

## 仓库结构

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

## 详细文档

### 原理与使用说明

- [BM/SGBM/SGM 的来源与区别](docs/algorithm-background.md)
- [输出格式与内存实现说明](docs/output-format.md)
- [0.5–3 m 参数与定量汇总](docs/parameter-recommendations.md)

### 实验报告

- [自研相机 50 cm / 75 cm 深度实验](docs/experiment-camera-50cm-75cm.md)
- [0724 多距离室内场景参数验证](docs/experiment-0724-parameter-validation.md)
- [Gemini335 与之前相机同场景对比](docs/experiment-gemini335-comparison.md)
- [0725-test BM 与 Vitis-SGM 深度效果](docs/experiment-0725-bm-sgm.md)

## License

见 [LICENSE](LICENSE)。
