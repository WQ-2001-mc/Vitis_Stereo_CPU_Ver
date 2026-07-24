# Vitis Stereo CPU Version

这是从 `Vitis_Libraries/vision` 的立体匹配示例中独立出来的纯 CPU
测试工程。工程只依赖普通 C++17 和 OpenCV，不依赖 Vitis、HLS、
`ap_int`、OpenCL/XRT 或 FPGA bitstream。

## 目录

- [1. 原目录里是不是有 BM 和 SGBM 两套算法？](#chapter-1)
- [2. 本工程内容](#chapter-2)
- [3. 编译](#chapter-3)
- [4. 运行](#chapter-4)
- [5. 输出说明](#chapter-5)
- [6. 自研相机 50 cm / 75 cm 深度实验](#chapter-6)
  - [6.1 横向效果对比](#chapter-6-1)
    - [6.1.1 50 cm](#chapter-6-1-1)
    - [6.1.2 75 cm](#chapter-6-1-2)
  - [6.2 定量结果](#chapter-6-2)
    - [6.2.1 50 cm 目标板](#chapter-6-2-1)
    - [6.2.2 75 cm 目标板](#chapter-6-2-2)
  - [6.3 建议参数](#chapter-6-3)
    - [6.3.1 同时覆盖 50 cm 和 75 cm](#chapter-6-3-1)
    - [6.3.2 只处理约 75 cm 且场景中没有更近物体](#chapter-6-3-2)
  - [6.4 公制深度偏差说明](#chapter-6-4)
  - [6.5 复现实验](#chapter-6-5)

<a id="chapter-1"></a>

## 1. 原目录里是不是有 BM 和 SGBM 两套算法？

是。两套核心算法及原始入口如下：

| 路线 | Vitis 目录/接口 | 算法含义 | 原 CPU 参考代码 |
|---|---|---|---|
| BM | `L1/examples/stereolbm` / `xf::cv::StereoBM` | Local Block Matching，局部 SAD 块匹配 | `xf_stereolbm_tb.cpp` 中的 OpenCV `cv::StereoBM` |
| SGBM | `L1/examples/sgbm` / `xf::cv::SemiGlobalBM` | Census 代价 + 半全局路径聚合 | `xf_sgbm_tb.cpp` 中的标量 `compute_SGM` |

需要注意：Vitis 目录名是 `sgbm`，硬件函数名是 `SemiGlobalBM`，但 CPU
参考函数叫 `compute_SGM`。这里的 SGBM CPU 程序是从 Vitis 测试台提取的
自带参考实现，并不是 OpenCV 的 `cv::StereoSGBM`。

`L1` 和 `L2` 中也各有一份示例。算法并没有因此变成四套：L1 是 HLS/C
仿真入口，L2 增加了 OpenCL/XRT 主机调用；两层中的 CPU 参考计算基本相同。

<a id="chapter-2"></a>

## 2. 本工程内容

- `vitis_bm_cpu`：OpenCV `StereoBM` CPU 参考程序。
- `vitis_sgbm_cpu`：Vitis 的 Census + 4 路径 SGM 标量 CPU 参考程序。
- `run_cpu_tests.sh`：配置、编译并依次运行两套算法。

默认参数保持原 Vitis 示例配置：

| 参数 | BM | SGBM/SGM |
|---|---:|---:|
| 视差数 | 32 | 64 |
| 窗口 | 11×11 SAD | 5×5 Census |
| 小/大惩罚 P1/P2 | — | 20 / 40 |
| 聚合路径数 | — | 4 |
| BM 唯一性比例 | 15 | — |
| BM 纹理阈值 | 20 | — |
| BM 预滤波上限 | 31 | — |

<a id="chapter-3"></a>

## 3. 编译

当前机器已有的 OpenCV 可以这样使用：

```bash
cd /home/hcc/Desktop/HXB/Vitis_Stereo_CPU_Ver
/usr/bin/cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/home/hcc/Desktop/HXB/opencv-install-min/lib/cmake/opencv4
/usr/bin/cmake --build build --parallel
```

如果系统安装的 OpenCV 自带 `OpenCVConfig.cmake` 且 CMake 能自动找到，
可以不传 `OpenCV_DIR`。

<a id="chapter-4"></a>

## 4. 运行

一次运行 BM 和 SGBM：

```bash
./run_cpu_tests.sh LEFT.png RIGHT.png
```

结果默认写入 `output/`。当前 Vitis 仓库自带图像的完整命令是：

```bash
./run_cpu_tests.sh \
  /home/hcc/Desktop/HXB/Vitis_Libraries/vision/data/left.png \
  /home/hcc/Desktop/HXB/Vitis_Libraries/vision/data/right.png
```

也可以分别运行：

```bash
./build/vitis_bm_cpu LEFT.png RIGHT.png bm_visual.png
./build/vitis_sgbm_cpu LEFT.png RIGHT.png sgbm_raw.png
```

可选覆盖视差数和 BM 窗口：

```bash
./build/vitis_bm_cpu LEFT.png RIGHT.png bm_visual.png 64 15
./build/vitis_sgbm_cpu LEFT.png RIGHT.png sgbm_raw.png 32
```

BM 的视差数必须是 16 的倍数。两套程序都要求输入为相同尺寸的左右校正图；
读取时会强制转换为单通道灰度图。

<a id="chapter-5"></a>

## 5. 输出说明

- BM 输出是按 Vitis 测试台方式缩放到 8 位的可视化图。
- SGBM 的指定输出文件保存原始整数视差（默认范围 0–63），同时自动生成
  同目录、文件名带 `_visual` 后缀的 0–255 拉伸图。

原 SGBM 测试台会同时保留每条路径的完整三维代价体。独立版本保持完全相同
的 Census 代价、四个方向、递推公式和最小代价选取规则，但复用单个路径代价
缓存，并把每像素重复扫描视差最小值的实现改为等价的一次扫描。以
1280×720、64 视差为例，预计工作内存从约 1.4 GiB 降为约 0.7 GiB。

<a id="chapter-6"></a>

## 6. 自研相机 50 cm / 75 cm 深度实验

数据集：

```text
/home/hcc/Desktop/Public/datasets/自研相机/50cm
/home/hcc/Desktop/Public/datasets/自研相机/75cm
```

每个目录中的两张 BMP 没有 left/right 文件名。分别测试两个排列后，BM
正视差有效率更高的排列均为：

| 距离 | 左图（Camera 1） | 右图（Camera 2） |
|---|---|---|
| 50 cm | `Image_20260723221034165.bmp` | `Image_20260723220953229.bmp` |
| 75 cm | `Image_20260724094303759.bmp` | `Image_20260724094244564.bmp` |

实验先使用目录中的 `stereo_opencv_params-0723-1708.yaml` 完成双目校正，
然后分别运行 BM 和 Vitis-SGM，测试 `D=64/128/256`。校正后用于深度换算
的参数为：

| 参数 | 数值 |
|---|---:|
| 校正后焦距 `fx` | 658.169 px |
| 基线 | 61.791 mm |
| 校正后主点偏移 `doffs` | 0 px |
| `fx × baseline` | 40669.158 px·mm |

公制深度按下式计算：

```text
Z_mm = fx_px × baseline_mm / disparity_px
```

据此，标称 50 cm 和 75 cm 的理论视差分别约为 81.34 px 和 54.23 px。
`D=64` 的最大有效视差只有约 63 px，所以从参数范围上就无法正确表示
50 cm。

<a id="chapter-6-1"></a>

### 6.1 横向效果对比

颜色范围统一为 0.3–1.2 m：暖色表示近，冷色表示远，黑色表示无效深度。
源图上的绿色小框是中央平面目标板的统计区域；统计框只包含目标板内部，
不包含支架、边缘和大面积背景。

<a id="chapter-6-1-1"></a>

#### 6.1.1 50 cm

![50 cm BM/SGBM depth comparison](results/camera_dataset/comparison_50cm.png)

<a id="chapter-6-1-2"></a>

#### 6.1.2 75 cm

![75 cm BM/SGBM depth comparison](results/camera_dataset/comparison_75cm.png)

所有单独的视差图、16 位毫米深度图和伪彩深度图位于：

```text
results/camera_dataset/50cm
results/camera_dataset/75cm
```

其中：

- `*_depth_mm_u16.png`：16 位无符号毫米深度，`0` 表示无效。
- `*_depth_color.png`：固定 0.3–1.2 m 色标的可视化深度图。
- BM 的 `*_disparity_q4_u16.png`：Q4 定点视差，即保存值除以 16
  得到像素视差。
- Vitis-SGM 的 `*_disparity_raw.png`：Vitis 标量参考算法输出的整数像素
  视差。

<a id="chapter-6-2"></a>

### 6.2 定量结果

下表的统计范围是横向对比图绿色框中的平面目标板：

- “有效率”是能够转换为有效深度的 ROI 像素比例。
- “±10% 命中率”是深度落在标称距离 ±10% 内的 ROI 像素比例，无效像素
  也计为未命中。
- “中位绝对误差”是有效深度相对标称距离的中位绝对误差。
- 时间是本机单次运行结果；BM 仅统计匹配计算，Vitis-SGM 统计独立程序
  从启动到输出视差的时间。

<a id="chapter-6-2-1"></a>

#### 6.2.1 50 cm 目标板

| 算法 | D | 有效率 | ±10% 命中率 | 中位深度 | 中位绝对误差 | 时间 | SGM 预计内存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM | 64 | 60.4% | 0.0% | 655.3 mm | 155.3 mm | 7.5 ms | — |
| BM | 128 | 41.9% | 35.5% | 539.6 mm | 40.0 mm | 11.0 ms | — |
| BM | 256 | 41.9% | 35.4% | 539.6 mm | 40.0 mm | 17.7 ms | — |
| Vitis-SGM | 64 | 60.3% | 0.0% | 656.0 mm | 156.0 mm | 1.54 s | 927.6 MiB |
| Vitis-SGM | 128 | 100.0% | 99.2% | 535.1 mm | 35.1 mm | 2.76 s | 1845.6 MiB |
| Vitis-SGM | 256 | 100.0% | 99.4% | 535.1 mm | 35.1 mm | 5.23 s | 3681.6 MiB |

50 cm 下，BM 和 Vitis-SGM 的 `D=64` 都集中输出约 62 px，已经贴近搜索
范围上限，因此深度被错误推远到约 656 mm。`D=128` 后目标板视差恢复到
约 75–76 px。`D=256` 与 `D=128` 的目标板中位结果没有可见改善。

<a id="chapter-6-2-2"></a>

#### 6.2.2 75 cm 目标板

| 算法 | D | 有效率 | ±10% 命中率 | 中位深度 | 中位绝对误差 | 时间 | SGM 预计内存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM | 64 | 79.4% | 78.4% | 798.4 mm | 48.4 mm | 7.5 ms | — |
| BM | 128 | 71.8% | 70.4% | 799.4 mm | 49.4 mm | 11.2 ms | — |
| BM | 256 | 71.8% | 70.4% | 799.4 mm | 49.4 mm | 18.6 ms | — |
| Vitis-SGM | 64 | 98.9% | 94.6% | 797.4 mm | 47.4 mm | 1.50 s | 927.6 MiB |
| Vitis-SGM | 128 | 100.0% | 92.5% | 797.4 mm | 47.4 mm | 2.73 s | 1845.6 MiB |
| Vitis-SGM | 256 | 100.0% | 91.8% | 797.4 mm | 47.4 mm | 5.18 s | 3681.6 MiB |

75 cm 的理论视差约 54 px，`D=64` 可以覆盖，因此三种 D 的中位深度基本
不变。对固定 75 cm 场景，D 从 64 增加到 128/256 没有带来精度收益，
反而增加时间、内存和搜索歧义。

完整机器可读统计：

- [`results/camera_dataset/metrics.csv`](results/camera_dataset/metrics.csv)
- [`results/camera_dataset/metrics.json`](results/camera_dataset/metrics.json)
- [`results/camera_dataset/metadata.json`](results/camera_dataset/metadata.json)

<a id="chapter-6-3"></a>

### 6.3 建议参数

<a id="chapter-6-3-1"></a>

#### 6.3.1 同时覆盖 50 cm 和 75 cm

首选 Vitis-SGM：

| 参数 | 建议值 |
|---|---:|
| `TOTAL_DISPARITY` / D | **128** |
| Census 窗口 | 5×5 |
| `SMALL_PENALTY` / P1 | 20 |
| `LARGE_PENALTY` / P2 | 40 |
| 路径数 | 4 |

理由：

- D=64 无法覆盖 50 cm 所需的约 81 px 视差。
- D=128 在两个距离的目标板上均达到 100% 有效率；50 cm 的 ±10%
  命中率为 99.2%，75 cm 为 92.5%。
- D=256 与 D=128 的中位深度完全相同，但 Vitis-SGM 的预计内存从约
  1.85 GiB 增加到 3.68 GiB，运行时间从约 2.7 s 增加到 5.2 s。

如果必须优先考虑 CPU 速度，可使用 BM：

| 参数 | 建议值 |
|---|---:|
| D | **128** |
| SAD 窗口 | 11×11 |
| `preFilterCap` | 31 |
| `uniquenessRatio` | 15 |
| `textureThreshold` | 20 |
| `minDisparity` | 0 |

BM D=128 在当前机器约 11 ms，但目标板有效率明显低于 Vitis-SGM：
50 cm 为 41.9%，75 cm 为 71.8%。它适合速度优先或后续还会做空洞填充、
左右一致性检查和时域滤波的场景。

<a id="chapter-6-3-2"></a>

#### 6.3.2 只处理约 75 cm 且场景中没有更近物体

可使用 D=64。当前标定下各 D 对应的理论最近深度约为：

| D | 最大有效视差 | 理论最近深度 |
|---:|---:|---:|
| 64 | 63 px | 645.5 mm |
| 128 | 127 px | 320.2 mm |
| 256 | 255 px | 159.5 mm |

所以 D=64 只适用于深度基本不小于约 0.65 m 的场景。只要需要覆盖
50 cm，或者画面中可能出现 0.32–0.65 m 的近物体，就应使用 D=128。
D=256 只在确实需要约 0.16–0.32 m 的近距离范围时才值得使用。

<a id="chapter-6-4"></a>

### 6.4 公制深度偏差说明

两种算法在 D 足够时给出的目标板中位结果非常接近：

- 标称 500 mm，测得约 535 mm，偏大约 7.0%。
- 标称 750 mm，测得约 797 mm，偏大约 6.3%。

BM 和 Vitis-SGM 都出现方向和比例相近的偏差，说明主要问题不是匹配算法
或 D，而更可能是：

- “50 cm/75 cm”测量起点并非左相机光心；
- 目标板实际位置与目录标称距离存在偏差；
- 焦距×基线的公制尺度仍需要重新核验。

如果公制精度是重点，建议重新测量“左相机光心到目标板平面”的真实距离，
并用多个已知距离验证标定尺度。当前两点对应的尺度修正系数约为
0.934 和 0.941，但不建议仅凭两张图直接把约 0.937 的经验系数写死到代码；
应先确认测距基准和同步采集是否正确。

<a id="chapter-6-5"></a>

### 6.5 复现实验

先按前面的方式编译工程，然后运行：

```bash
cd /home/hcc/Desktop/HXB/Vitis_Stereo_CPU_Ver
./scripts/evaluate_camera_dataset.py
```

评估脚本还需要 Python 3、`numpy` 和 Python OpenCV (`cv2`)；当前机器已经
具备这些依赖。

脚本会重新完成左右顺序判定、校正、12 组视差/深度计算、毫米深度图生成、
指标统计和横向对比图生成。
