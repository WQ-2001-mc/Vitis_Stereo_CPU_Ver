# 原目录里是不是有 BM 和 SGBM 两套算法？

> [返回项目 README](../../README.md)

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
