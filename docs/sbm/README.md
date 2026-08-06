# SBM（StereoBM）文档

> [返回文档索引](../README.md) · [返回项目 README](../../README.md)

## 名称说明

这里的 **SBM** 指 Stereo Block Matching。项目源码、OpenCV 接口和已有报告
统一写作 **BM / StereoBM**：

- 程序：`src/vitis_bm_cpu.cpp`
- 算法接口：OpenCV `cv::StereoBM`
- Vitis 来源：`L1/examples/stereolbm` / `xf::cv::StereoBM`
- 匹配方式：局部 SAD 块匹配
- 主要参数：视差范围 `D`、SAD 窗口 `blockSize`、`preFilterCap`、
  `uniquenessRatio`、`textureThreshold`

因此，文档中出现 `BM`、`StereoBM` 或 `vitis_bm_cpu` 时，都属于 SBM 路线。

## 相关文档

- [BM/SGBM/SGM 的来源与区别](../common/algorithm-background.md)
- [两种算法的输出格式](../common/output-format.md)
- [0.5–3 m 参数与定量汇总](../comparisons/parameter-recommendations.md)
- [自研相机 50 cm / 75 cm 对比实验](../comparisons/experiment-camera-50cm-75cm.md)
- [0724 视差范围和块大小验证](../comparisons/experiment-0724-parameter-validation.md)
- [Gemini335 与之前相机对比](../comparisons/experiment-gemini335-comparison.md)
- [0725-test BM 与 Vitis-SGM 对比](../comparisons/experiment-0725-bm-sgm.md)

单独针对 StereoBM 的最新 D/W 参数扫描报告位于：

- [Gemini335 StereoBM D/W 参数深度图横向对比](../../results/gemini335_1280x800_bm_sweep/STEREOBM_DEPTH_COMPARISON.md)
