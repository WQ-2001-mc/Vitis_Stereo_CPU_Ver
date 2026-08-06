# 文档索引

> [返回项目 README](../README.md)

## 算法分类

本项目实际包含两条立体匹配路线。项目代码和 OpenCV 使用 **BM / StereoBM**
这个名称，它就是这里要归类的 **SBM（Stereo Block Matching）**；另一条是
**Vitis-SGBM**，其 CPU 核心是 Census 代价加多路径 SGM 聚合，不是 OpenCV
的 `cv::StereoSGBM`。

| 分类 | 项目中的名称 | 专属文档入口 |
|---|---|---|
| SBM | BM、StereoBM、`vitis_bm_cpu` | [SBM（StereoBM）文档](sbm/README.md) |
| SGBM | Vitis-SGBM、Vitis-SGM、`vitis_sgbm_cpu` | [SGBM（Vitis-SGM）文档](sgbm/README.md) |

当前 Gemini335 1280×800 FPGA 等效实验：

- [Vitis-SGBM FPGA 等效 P1/P2 参数对比](sgbm/gemini335-1280x800-vitis-sgbm-fpga-equivalent.md)

## 公共说明

- [BM/SGBM/SGM 的来源与区别](common/algorithm-background.md)
- [两种算法的输出格式与内存实现说明](common/output-format.md)

## 对比、参数与实验

以下文件都同时包含 SBM 和 SGBM，归入对比类，不强行归到任何一种算法：

- [0.5–3 m 参数与定量汇总](comparisons/parameter-recommendations.md)
- [自研相机 50 cm / 75 cm 深度实验](comparisons/experiment-camera-50cm-75cm.md)
- [0724 多距离室内场景参数验证](comparisons/experiment-0724-parameter-validation.md)
- [Gemini335 与之前相机同场景对比](comparisons/experiment-gemini335-comparison.md)
- [0725-test BM 与 Vitis-SGM 深度效果](comparisons/experiment-0725-bm-sgm.md)
