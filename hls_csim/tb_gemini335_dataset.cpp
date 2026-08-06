#include "gemini335_sgbm_config.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
};

std::string requiredEnvironment(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        throw std::runtime_error(std::string("missing environment variable: ") +
                                 name);
    }
    return value;
}

std::string nextPgmToken(std::istream& input) {
    std::string token;
    while (input >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string ignored;
            std::getline(input, ignored);
            continue;
        }
        return token;
    }
    throw std::runtime_error("unexpected end of PGM header");
}

GrayImage readPgm8(const std::string& path) {
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input PGM: " + path);
    if (nextPgmToken(input) != "P5") {
        throw std::runtime_error("only binary P5 PGM input is supported: " +
                                 path);
    }
    GrayImage image;
    image.width = std::stoi(nextPgmToken(input));
    image.height = std::stoi(nextPgmToken(input));
    const int maxValue = std::stoi(nextPgmToken(input));
    if (image.width != GEMINI_WIDTH || image.height != GEMINI_HEIGHT ||
        maxValue != 255) {
        std::ostringstream message;
        message << "expected P5 " << GEMINI_WIDTH << "x" << GEMINI_HEIGHT
                << " max=255, got " << image.width << "x" << image.height
                << " max=" << maxValue;
        throw std::runtime_error(message.str());
    }
    input.get();
    image.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
    input.read(reinterpret_cast<char*>(image.pixels.data()),
               static_cast<std::streamsize>(image.pixels.size()));
    if (input.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
        throw std::runtime_error("short PGM pixel payload: " + path);
    }
    return image;
}

std::vector<std::pair<int, int> > parsePenalties(const std::string& text) {
    std::vector<std::pair<int, int> > penalties;
    std::stringstream entries(text);
    std::string entry;
    while (std::getline(entries, entry, ',')) {
        if (entry.empty()) continue;
        const std::size_t slash = entry.find('/');
        if (slash == std::string::npos) {
            throw std::runtime_error("invalid P1/P2 entry: " + entry);
        }
        const int p1 = std::stoi(entry.substr(0, slash));
        const int p2 = std::stoi(entry.substr(slash + 1));
        if (p1 < 0 || p2 <= p1 || p2 > 100) {
            throw std::runtime_error("P1/P2 must satisfy 0 <= P1 < P2 <= 100");
        }
        penalties.push_back(std::make_pair(p1, p2));
    }
    if (penalties.empty()) throw std::runtime_error("empty P1/P2 list");
    return penalties;
}

void packInput(
    const std::vector<std::uint8_t>& pixels,
    std::vector<ap_uint<GEMINI_INPUT_PTR_WIDTH> >& words) {
    const int lanes = GEMINI_INPUT_PTR_WIDTH / 8;
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        const std::size_t word = index / lanes;
        const int lane = static_cast<int>(index % lanes);
        words[word].range(lane * 8 + 7, lane * 8) = pixels[index];
    }
}

std::vector<std::uint16_t> unpackDepth(
    const std::vector<ap_uint<GEMINI_OUTPUT_PTR_WIDTH> >& words,
    std::size_t pixelCount) {
    const int lanes = GEMINI_OUTPUT_PTR_WIDTH / 16;
    std::vector<std::uint16_t> depth(pixelCount);
    for (std::size_t index = 0; index < pixelCount; ++index) {
        const std::size_t word = index / lanes;
        const int lane = static_cast<int>(index % lanes);
        depth[index] = static_cast<std::uint16_t>(
            words[word].range(lane * 16 + 15, lane * 16));
    }
    return depth;
}

void writePgm16(const std::string& path,
                const std::vector<std::uint16_t>& pixels,
                int width,
                int height) {
    std::ofstream output(path.c_str(), std::ios::binary);
    if (!output) throw std::runtime_error("cannot create output PGM: " + path);
    output << "P5\n" << width << " " << height << "\n65535\n";
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        const char bytes[2] = {
            static_cast<char>((pixels[index] >> 8) & 0xff),
            static_cast<char>(pixels[index] & 0xff),
        };
        output.write(bytes, 2);
    }
    if (!output) throw std::runtime_error("failed writing output PGM: " + path);
}

std::string outputName(const std::string& directory, int p1, int p2) {
    std::ostringstream name;
    name << directory << "/hls_depth_p1_" << std::setw(3)
         << std::setfill('0') << p1 << "_p2_" << std::setw(3) << p2
         << ".pgm";
    return name.str();
}

}  // namespace

int main() {
    try {
        const std::string leftPath = requiredEnvironment("GEMINI_CSIM_LEFT_PGM");
        const std::string rightPath = requiredEnvironment("GEMINI_CSIM_RIGHT_PGM");
        const std::string outputDirectory =
            requiredEnvironment("GEMINI_CSIM_OUTPUT_DIR");
        const char* penaltyEnvironment = std::getenv("GEMINI_CSIM_PENALTIES");
        const std::vector<std::pair<int, int> > penalties = parsePenalties(
            penaltyEnvironment && *penaltyEnvironment
                ? penaltyEnvironment
                : "20/40");

        const GrayImage left = readPgm8(leftPath);
        const GrayImage right = readPgm8(rightPath);
        if (left.width != right.width || left.height != right.height) {
            throw std::runtime_error("left/right PGM dimensions differ");
        }

        const std::size_t pixelCount = left.pixels.size();
        const int inputLanes = GEMINI_INPUT_PTR_WIDTH / 8;
        const int outputLanes = GEMINI_OUTPUT_PTR_WIDTH / 16;
        std::vector<ap_uint<GEMINI_INPUT_PTR_WIDTH> > leftWords(
            (pixelCount + inputLanes - 1) / inputLanes);
        std::vector<ap_uint<GEMINI_INPUT_PTR_WIDTH> > rightWords(
            (pixelCount + inputLanes - 1) / inputLanes);
        packInput(left.pixels, leftWords);
        packInput(right.pixels, rightWords);

        for (std::size_t index = 0; index < penalties.size(); ++index) {
            const int p1 = penalties[index].first;
            const int p2 = penalties[index].second;
            std::vector<ap_uint<GEMINI_OUTPUT_PTR_WIDTH> > outputWords(
                (pixelCount + outputLanes - 1) / outputLanes);
            unsigned int signature = 0;
            const std::chrono::steady_clock::time_point start =
                std::chrono::steady_clock::now();
            gemini335_sgbm_accel(leftWords.data(),
                                 rightWords.data(),
                                 outputWords.data(),
                                 static_cast<unsigned char>(p1),
                                 static_cast<unsigned char>(p2),
                                 left.height,
                                 left.width,
                                 signature);
            const std::chrono::steady_clock::time_point finish =
                std::chrono::steady_clock::now();
            if (signature != GEMINI_PL_BUILD_SIGNATURE) {
                throw std::runtime_error("PL build signature mismatch in C simulation");
            }
            const std::vector<std::uint16_t> depth =
                unpackDepth(outputWords, pixelCount);
            const std::string outputPath = outputName(outputDirectory, p1, p2);
            writePgm16(outputPath, depth, left.width, left.height);

            std::size_t valid = 0;
            std::uint64_t checksum = 0;
            std::uint16_t maximum = 0;
            for (std::size_t pixel = 0; pixel < depth.size(); ++pixel) {
                if (depth[pixel] != 0) ++valid;
                checksum += depth[pixel];
                maximum = std::max(maximum, depth[pixel]);
            }
            const double elapsed =
                std::chrono::duration<double>(finish - start).count();
            std::cout << "GEMINI335_DATASET_CSIM"
                      << " P1=" << p1 << " P2=" << p2
                      << " rows=" << left.height << " cols=" << left.width
                      << " D=" << GEMINI_SGBM_NUM_DISPARITIES
                      << " PU=" << GEMINI_SGBM_PARALLEL_UNITS
                      << " R=" << GEMINI_SGBM_NUM_DIRECTIONS
                      << " W=" << GEMINI_SGBM_WINDOW_SIZE
                      << " valid=" << valid
                      << " max_depth_mm=" << maximum
                      << " checksum=" << checksum
                      << " host_csim_seconds=" << std::fixed
                      << std::setprecision(3) << elapsed
                      << " output=" << outputPath << "\n";
        }
        std::cout << "GEMINI335_DATASET_CSIM_PASS configurations="
                  << penalties.size() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GEMINI335_DATASET_CSIM_FAIL " << error.what() << "\n";
        return 1;
    }
}
