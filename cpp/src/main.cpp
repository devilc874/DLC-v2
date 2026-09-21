/**
 * @file main.cpp
 * @brief DLC CLI entry point.
 *
 * Usage:
 *   ./dlc compress   -i input.bin -o output.dlc
 *   ./dlc decompress -i input.dlc -o output.bin
 */

#include "dlc/codec.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


static void print_usage() {
  std::cerr << "Usage:\n"
            << "  dlc compress   -i <input.bin> -o <output.dlc> [--precision "
               "N] [--chunk-size N] [--workers N] [--ablation-precision]\n"
            << "  dlc decompress -i <input.dlc> -o <output.bin>\n"
            << "\n"
            << "  --precision N          Requested precision cap (8–20, default 16).\n"
            << "                         Windows still adapt above the floor up to N.\n"
            << "  --ablation-precision   Disable the 16-bit error-bound floor so\n"
            << "                         precision can adapt over the full 8–20 range.\n";
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    print_usage();
    return 1;
  }

  std::string command = argv[1];
  std::string input_path, output_path;
  int precision = 16; // paper default
  uint32_t chunk_size = 100000;
  int workers = 0;
  bool ablation_precision = false;

  // Parse arguments
  for (int i = 2; i < argc; ++i) {
    if ((std::strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
      input_path = argv[++i];
    } else if ((std::strcmp(argv[i], "-o") == 0) && i + 1 < argc) {
      output_path = argv[++i];
    } else if ((std::strcmp(argv[i], "--precision") == 0) && i + 1 < argc) {
      precision = std::stoi(argv[++i]);
    } else if ((std::strcmp(argv[i], "--chunk-size") == 0) && i + 1 < argc) {
      chunk_size = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else if ((std::strcmp(argv[i], "--workers") == 0) && i + 1 < argc) {
      workers = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--ablation-precision") == 0) {
      ablation_precision = true;
    }
  }

  if (input_path.empty() || output_path.empty()) {
    print_usage();
    return 1;
  }

  if (precision < 8)
    precision = 8;
  if (precision > 20)
    precision = 20;

  try {
    if (command == "compress") {
      // Read raw float64 binary (little-endian)
      std::ifstream in(input_path, std::ios::binary);
      if (!in) {
        std::cerr << "Cannot open input file: " << input_path << "\n";
        return 1;
      }

      in.seekg(0, std::ios::end);
      size_t file_size = static_cast<size_t>(in.tellg());
      in.seekg(0, std::ios::beg);

      size_t n = file_size / sizeof(double);
      std::vector<double> data(n);
      in.read(reinterpret_cast<char *>(data.data()),
              static_cast<std::streamsize>(file_size));
      in.close();

      std::cout << "Read " << n << " samples from " << input_path << "\n";

      dlc::CompressOptions opts;
      opts.precision_bits = precision;
      opts.chunk_size = chunk_size;
      opts.num_workers = workers;
      opts.enforce_error_bound = !ablation_precision;

      dlc::compress(data.data(), n, output_path, opts);
      std::cout << "Compressed to " << output_path << "\n";

    } else if (command == "decompress") {
      auto result = dlc::decompress(input_path);

      std::ofstream out(output_path, std::ios::binary);
      if (!out) {
        std::cerr << "Cannot open output file: " << output_path << "\n";
        return 1;
      }
      out.write(reinterpret_cast<const char *>(result.data()),
                static_cast<std::streamsize>(result.size() * sizeof(double)));
      out.close();

      std::cout << "Decompressed " << result.size() << " samples to "
                << output_path << "\n";

    } else {
      std::cerr << "Unknown command: " << command << "\n";
      print_usage();
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
