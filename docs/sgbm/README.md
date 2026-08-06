# SGBM（Vitis-SGM）文档

> [返回文档索引](../README.md) · [返回项目 README](../../README.md)

## 名称说明

本项目的第二条路线在目录和程序名中写作 **SGBM**，在算法描述和图表中常写作
**Vitis-SGM**：

- 程序：`src/vitis_sgbm_cpu.cpp`
- Vitis 接口：`xf::cv::SemiGlobalBM`
- CPU 参考函数：`compute_SGM`
- 匹配方式：Census 代价加半全局多路径代价聚合
- 主要参数：视差范围 `D`、Census 窗口、路径数、平滑惩罚 `P1/P2`

这些名称在本项目中指向同一条算法路线。它来自 Vitis 测试台的标量参考实现，
**不是** OpenCV 的 `cv::StereoSGBM`。

## 相关文档

- [BM/SGBM/SGM 的来源与区别](../common/algorithm-background.md)
- [两种算法的输出格式与 SGBM 内存实现](../common/output-format.md)
- [0.5–3 m 参数与定量汇总](../comparisons/parameter-recommendations.md)
- [自研相机 50 cm / 75 cm 对比实验](../comparisons/experiment-camera-50cm-75cm.md)
- [0724 视差范围和 P1/P2 验证](../comparisons/experiment-0724-parameter-validation.md)
- [Gemini335 与之前相机对比](../comparisons/experiment-gemini335-comparison.md)
- [0725-test BM 与 Vitis-SGM 对比](../comparisons/experiment-0725-bm-sgm.md)

单独针对 Vitis-SGBM 的最新参数扫描报告位于：

- [Gemini335 1280×800 Vitis-SGBM FPGA 等效参数对比](gemini335-1280x800-vitis-sgbm-fpga-equivalent.md)
- [旧版四路径 D 与 P1/P2 趋势扫描](../../results/gemini335_1280x800_sgbm_sweep/VITIS_SGBM_PARAMETER_COMPARISON.md)

当前 1280×800 FPGA 工程使用 `D=128、PU=64、R=3、W=5`。旧版报告使用
四路径标量参考，适合保留作 D/P1/P2 趋势资料，但不能替代上面的三路径
FPGA 等效报告。
