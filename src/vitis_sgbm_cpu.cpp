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
 * Modified for a standalone, CPU-only build. The Census cost and SGM
 * recurrence come from:
 * Vitis_Libraries/vision/L1/examples/sgbm/xf_sgbm_tb.cpp
 *
 * The recurrence is unchanged. Repeated full-volume path buffers and the
 * O(D^2) previous-minimum scan have been replaced by one reusable path
 * buffer and an equivalent O(D) minimum scan.
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kDefaultNumDisparities = 64;
constexpr int kWindowSize = 5;
constexpr int kSmallPenalty = 20;
constexpr int kLargePenalty = 40;

// The Vitis example's NUM_DIR is 4. These are the exact four predecessor
// directions used by its scalar CPU reference.
constexpr std::pair<int, int> kDirections[] = {
    {0, -1},
    {-1, -1},
    {-1, 0},
    {-1, 1},
};

void printUsage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " <left_gray_image> <right_gray_image> [output_raw.png]"
           " [num_disparities]\n\n"
        << "Defaults (matching the Vitis sgbm example):\n"
        << "  output_raw       sgbm_disparity_raw.png\n"
        << "  num_disparities  " << kDefaultNumDisparities << "\n"
        << "  Census window    " << kWindowSize << "x" << kWindowSize << "\n"
        << "  P1/P2            " << kSmallPenalty << "/" << kLargePenalty << "\n"
        << "  paths            4\n";
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
                    int num_disparities) {
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
    if (num_disparities <= 0 || num_disparities > 256) {
        std::cerr << "ERROR: num_disparities must be in [1, 256].\n";
        return false;
    }
    if (num_disparities >= left.cols) {
        std::cerr << "ERROR: num_disparities must be smaller than the image width.\n";
        return false;
    }
    return true;
}

std::size_t checkedVolumeSize(int rows, int cols, int disparities) {
    const std::size_t pixels =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (pixels > std::numeric_limits<std::size_t>::max() /
                     static_cast<std::size_t>(disparities)) {
        throw std::overflow_error("cost-volume size overflow");
    }
    return pixels * static_cast<std::size_t>(disparities);
}

std::uint32_t hammingDistance(std::uint32_t a, std::uint32_t b) {
    return static_cast<std::uint32_t>(__builtin_popcount(a ^ b));
}

void computeCensusTransform(const cv::Mat& image,
                            std::vector<std::uint32_t>& census) {
    const int radius = kWindowSize / 2;
    census.resize(image.total());

    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            const std::uint8_t center = image.at<std::uint8_t>(row, col);
            std::uint32_t descriptor = 0;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dy == 0 && dx == 0) {
                        continue;
                    }

                    descriptor <<= 1;
                    const int sample_row = row + dy;
                    const int sample_col = col + dx;
                    std::uint8_t sample = 0;
                    if (sample_row >= 0 && sample_row < image.rows &&
                        sample_col >= 0 && sample_col < image.cols) {
                        sample = image.at<std::uint8_t>(sample_row, sample_col);
                    }
                    if (sample < center) {
                        descriptor |= 1U;
                    }
                }
            }
            census[static_cast<std::size_t>(row) * image.cols + col] = descriptor;
        }
    }
}

void computeInitialCost(const std::vector<std::uint32_t>& left_census,
                        const std::vector<std::uint32_t>& right_census,
                        int rows,
                        int cols,
                        int disparities,
                        std::vector<int>& cost) {
    cost.resize(checkedVolumeSize(rows, cols, disparities));

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const std::size_t pixel =
                static_cast<std::size_t>(row) * cols + col;
            const std::size_t base = pixel * disparities;
            for (int disparity = 0; disparity < disparities; ++disparity) {
                const std::uint32_t right_descriptor =
                    col - disparity >= 0
                        ? right_census[static_cast<std::size_t>(row) * cols +
                                       col - disparity]
                        : 0U;
                cost[base + disparity] = static_cast<int>(
                    hammingDistance(left_census[pixel], right_descriptor));
            }
        }
    }
}

void aggregateDirection(const std::vector<int>& initial_cost,
                        int rows,
                        int cols,
                        int disparities,
                        int delta_row,
                        int delta_col,
                        std::vector<int>& path_cost,
                        std::vector<int>& aggregated_cost) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * cols + col) * disparities;
            const int previous_row = row + delta_row;
            const int previous_col = col + delta_col;

            if (previous_row < 0 || previous_row >= rows ||
                previous_col < 0 || previous_col >= cols) {
                for (int disparity = 0; disparity < disparities; ++disparity) {
                    const int value = initial_cost[base + disparity];
                    path_cost[base + disparity] = value;
                    aggregated_cost[base + disparity] += value;
                }
                continue;
            }

            const std::size_t previous_base =
                (static_cast<std::size_t>(previous_row) * cols + previous_col) *
                disparities;
            const auto previous_begin = path_cost.begin() + previous_base;
            const int previous_min =
                *std::min_element(previous_begin, previous_begin + disparities);

            for (int disparity = 0; disparity < disparities; ++disparity) {
                const int same = path_cost[previous_base + disparity];
                const int lower =
                    disparity > 0
                        ? path_cost[previous_base + disparity - 1] + kSmallPenalty
                        : std::numeric_limits<int>::max();
                const int higher =
                    disparity + 1 < disparities
                        ? path_cost[previous_base + disparity + 1] + kSmallPenalty
                        : std::numeric_limits<int>::max();
                const int any = previous_min + kLargePenalty;
                const int transition = std::min({same, lower, higher, any});
                const int value =
                    initial_cost[base + disparity] + transition - previous_min;

                path_cost[base + disparity] = value;
                aggregated_cost[base + disparity] += value;
            }
        }
    }
}

cv::Mat computeVitisSgm(const cv::Mat& left,
                        const cv::Mat& right,
                        int disparities) {
    std::vector<std::uint32_t> left_census;
    std::vector<std::uint32_t> right_census;
    computeCensusTransform(left, left_census);
    computeCensusTransform(right, right_census);

    std::vector<int> initial_cost;
    computeInitialCost(
        left_census, right_census, left.rows, left.cols, disparities, initial_cost);

    const std::size_t volume_size =
        checkedVolumeSize(left.rows, left.cols, disparities);
    std::vector<int> path_cost(volume_size);
    std::vector<int> aggregated_cost(volume_size, 0);

    for (const auto& [delta_row, delta_col] : kDirections) {
        aggregateDirection(initial_cost,
                           left.rows,
                           left.cols,
                           disparities,
                           delta_row,
                           delta_col,
                           path_cost,
                           aggregated_cost);
    }

    cv::Mat disparity(left.rows, left.cols, CV_8UC1);
    for (int row = 0; row < left.rows; ++row) {
        for (int col = 0; col < left.cols; ++col) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * left.cols + col) * disparities;
            const auto begin = aggregated_cost.begin() + base;
            const auto best = std::min_element(begin, begin + disparities);
            disparity.at<std::uint8_t>(row, col) =
                static_cast<std::uint8_t>(std::distance(begin, best));
        }
    }
    return disparity;
}

std::filesystem::path visualPathFor(const std::filesystem::path& raw_path) {
    const std::string extension =
        raw_path.has_extension() ? raw_path.extension().string() : ".png";
    return raw_path.parent_path() /
           (raw_path.stem().string() + "_visual" + extension);
}

double estimatedWorkingSetMiB(int rows, int cols, int disparities) {
    const double volume =
        static_cast<double>(rows) * cols * disparities * sizeof(int) * 3;
    const double census =
        static_cast<double>(rows) * cols * sizeof(std::uint32_t) * 2;
    return (volume + census) / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string left_path = argv[1];
    const std::string right_path = argv[2];
    const std::filesystem::path raw_output =
        argc >= 4 ? argv[3] : "sgbm_disparity_raw.png";

    int disparities = kDefaultNumDisparities;
    if (argc >= 5 && !parseInteger(argv[4], disparities)) {
        std::cerr << "ERROR: invalid num_disparities: " << argv[4] << "\n";
        return EXIT_FAILURE;
    }

    const cv::Mat left = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
    if (!validateInputs(left, right, disparities)) {
        return EXIT_FAILURE;
    }

    std::cout << "Algorithm:       Vitis scalar SGM CPU reference\n"
              << "Input size:      " << left.cols << "x" << left.rows << "\n"
              << "Disparities:     " << disparities << "\n"
              << "Census window:   " << kWindowSize << "x" << kWindowSize << "\n"
              << "P1/P2:           " << kSmallPenalty << "/" << kLargePenalty << "\n"
              << "Paths:           4\n"
              << "Est. memory:     " << std::fixed << std::setprecision(1)
              << estimatedWorkingSetMiB(left.rows, left.cols, disparities)
              << " MiB\n";

    cv::Mat disparity_raw;
    const auto start = std::chrono::steady_clock::now();
    try {
        disparity_raw = computeVitisSgm(left, right, disparities);
    } catch (const std::bad_alloc&) {
        std::cerr << "ERROR: not enough memory for the SGM cost volumes.\n";
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: SGM computation failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    const auto stop = std::chrono::steady_clock::now();

    const std::filesystem::path visual_output = visualPathFor(raw_output);
    cv::Mat disparity_visual;
    const double visual_scale =
        disparities > 1 ? 255.0 / static_cast<double>(disparities - 1) : 0.0;
    disparity_raw.convertTo(disparity_visual, CV_8U, visual_scale);

    if (!cv::imwrite(raw_output.string(), disparity_raw)) {
        std::cerr << "ERROR: failed to write raw disparity: " << raw_output << "\n";
        return EXIT_FAILURE;
    }
    if (!cv::imwrite(visual_output.string(), disparity_visual)) {
        std::cerr << "ERROR: failed to write visualization: " << visual_output << "\n";
        return EXIT_FAILURE;
    }

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "CPU compute:     " << std::setprecision(3) << elapsed_ms << " ms\n"
              << "Raw output:      " << raw_output << "\n"
              << "Visualization:   " << visual_output << "\n";

    return EXIT_SUCCESS;
}
