/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified for a standalone, CPU-only build. The OpenCV StereoBM reference
 * path and its default parameters come from:
 * Vitis_Libraries/vision/L1/examples/stereolbm/xf_stereolbm_tb.cpp
 */

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {

constexpr int kDefaultNumDisparities = 32;
constexpr int kDefaultBlockSize = 11;
constexpr int kPreFilterCap = 31;
constexpr int kUniquenessRatio = 15;
constexpr int kTextureThreshold = 20;
constexpr int kMinDisparity = 0;

void printUsage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " <left_gray_image> <right_gray_image> [output_visual.png]"
           " [num_disparities] [block_size]\n\n"
        << "Defaults (matching the Vitis stereolbm example):\n"
        << "  output_visual  bm_disparity_visual.png\n"
        << "  num_disparities  " << kDefaultNumDisparities << "\n"
        << "  block_size       " << kDefaultBlockSize << "\n";
}

bool parseInteger(const char* text, int& value) {
    try {
        std::size_t parsed = 0;
        const int candidate = std::stoi(text, &parsed);
        if (parsed != std::string(text).size()) {
            return false;
        }
        value = candidate;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool validateInputs(const cv::Mat& left,
                    const cv::Mat& right,
                    int num_disparities,
                    int block_size) {
    if (left.empty() || right.empty()) {
        std::cerr << "ERROR: failed to read one or both input images.\n";
        return false;
    }
    if (left.size() != right.size()) {
        std::cerr << "ERROR: left/right image sizes differ: "
                  << left.cols << "x" << left.rows << " versus "
                  << right.cols << "x" << right.rows << ".\n";
        return false;
    }
    if (num_disparities <= 0 || num_disparities % 16 != 0) {
        std::cerr << "ERROR: StereoBM num_disparities must be positive and divisible by 16.\n";
        return false;
    }
    if (num_disparities >= left.cols) {
        std::cerr << "ERROR: num_disparities must be smaller than the image width.\n";
        return false;
    }
    if (block_size < 5 || block_size > 255 || block_size % 2 == 0) {
        std::cerr << "ERROR: StereoBM block_size must be odd and in [5, 255].\n";
        return false;
    }
    if (block_size > left.rows || block_size > left.cols) {
        std::cerr << "ERROR: block_size is larger than the input image.\n";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string left_path = argv[1];
    const std::string right_path = argv[2];
    const std::string output_path = argc >= 4 ? argv[3] : "bm_disparity_visual.png";

    int num_disparities = kDefaultNumDisparities;
    int block_size = kDefaultBlockSize;
    if (argc >= 5 && !parseInteger(argv[4], num_disparities)) {
        std::cerr << "ERROR: invalid num_disparities: " << argv[4] << "\n";
        return EXIT_FAILURE;
    }
    if (argc >= 6 && !parseInteger(argv[5], block_size)) {
        std::cerr << "ERROR: invalid block_size: " << argv[5] << "\n";
        return EXIT_FAILURE;
    }

    const cv::Mat left = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
    if (!validateInputs(left, right, num_disparities, block_size)) {
        return EXIT_FAILURE;
    }

    // Keep the Vitis CPU reference behavior: do not use OpenCV's optimized
    // dispatch paths while producing the reference result.
    cv::setUseOptimized(false);

    cv::Ptr<cv::StereoBM> matcher = cv::StereoBM::create(num_disparities, block_size);
    matcher->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
    matcher->setPreFilterCap(kPreFilterCap);
    matcher->setMinDisparity(kMinDisparity);
    matcher->setTextureThreshold(kTextureThreshold);
    matcher->setUniquenessRatio(kUniquenessRatio);

    cv::Mat disparity_fixed_point;
    const auto start = std::chrono::steady_clock::now();
    matcher->compute(left, right, disparity_fixed_point);
    const auto stop = std::chrono::steady_clock::now();

    // StereoBM produces CV_16S disparities with four fractional bits.
    // This is the same visualization scaling used by the Vitis testbench.
    cv::Mat disparity_visual;
    disparity_fixed_point.convertTo(
        disparity_visual, CV_8U, (256.0 / num_disparities) / 16.0);

    if (!cv::imwrite(output_path, disparity_visual)) {
        std::cerr << "ERROR: failed to write output image: " << output_path << "\n";
        return EXIT_FAILURE;
    }

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    const int valid_pixels = cv::countNonZero(disparity_fixed_point >= 0);

    std::cout << "Algorithm:       OpenCV StereoBM (Vitis BM CPU reference)\n"
              << "Input size:      " << left.cols << "x" << left.rows << "\n"
              << "Disparities:     " << num_disparities << "\n"
              << "Block size:      " << block_size << "\n"
              << "Valid pixels:    " << valid_pixels << " / " << left.total() << "\n"
              << "CPU compute:     " << elapsed_ms << " ms\n"
              << "Output:          " << output_path << "\n";

    return EXIT_SUCCESS;
}
