/**
 * @file test_format.cpp
 * @brief GTest tests for DLC binary format.
 */

#include "dlc/format.hpp"
#include <cstring>
#include <gtest/gtest.h>
#include <sstream>


namespace dlc {
namespace test {

// ── Header Tests ────────────────────────────────────────────────────────────

TEST(FormatTest, HeaderSizeIs64) { EXPECT_EQ(sizeof(DLCHeader), 64u); }

TEST(FormatTest, FooterSizeIs8) { EXPECT_EQ(sizeof(DLCFooter), 8u); }

TEST(FormatTest, HeaderRoundTrip) {
  DLCHeader h{};
  std::memcpy(h.magic, MAGIC_BYTES, 4);
  h.major_version = VERSION_MAJOR;
  h.minor_version = VERSION_MINOR;
  h.precision_bits = 12;
  h.chunk_size = 100000;
  h.total_samples = 999999;
  std::memset(h.reserved, 0, 42);

  std::stringstream ss;
  pack_header(h, ss);
  std::string data = ss.str();
  EXPECT_EQ(data.size(), 64u);

  std::stringstream ss2(data);
  DLCHeader h2 = unpack_header(ss2);
  EXPECT_EQ(std::memcmp(h2.magic, MAGIC_BYTES, 4), 0);
  EXPECT_EQ(h2.precision_bits, 12);
  EXPECT_EQ(h2.chunk_size, 100000u);
  EXPECT_EQ(h2.total_samples, 999999u);
}

TEST(FormatTest, HeaderInvalidMagicThrows) {
  DLCHeader h{};
  h.magic[0] = 'X';
  std::stringstream ss;
  ss.write(reinterpret_cast<const char *>(&h), sizeof(h));
  ss.seekg(0);
  EXPECT_THROW(unpack_header(ss), std::runtime_error);
}

// ── Footer Tests ────────────────────────────────────────────────────────────

TEST(FormatTest, FooterRoundTrip) {
  DLCFooter f{};
  std::memcpy(f.magic, FOOTER_MAGIC, 4);
  f.crc32 = 0xDEADBEEF;

  std::stringstream ss;
  pack_footer(f, ss);
  std::string data = ss.str();
  EXPECT_EQ(data.size(), 8u);

  std::stringstream ss2(data);
  DLCFooter f2 = unpack_footer(ss2);
  EXPECT_EQ(f2.crc32, 0xDEADBEEF);
}

// ── Window Block Tests ──────────────────────────────────────────────────────

TEST(FormatTest, WindowBlockLinearRoundTrip) {
  WindowBlockMeta meta;
  meta.window_length = 256;
  meta.model_id = ModelId::Linear;
  meta.encoding_id = EncodingId::Bitpack;
  std::vector<double> params = {1.5, -0.3};
  std::vector<uint8_t> residuals = {1, 2, 3, 4};

  std::vector<uint8_t> buf;
  pack_window_block(meta, params, residuals, buf);

  WindowBlockMeta meta2;
  std::vector<double> params2;
  std::vector<uint8_t> residuals2;
  size_t consumed = unpack_window_block(buf.data(), buf.size(), 0, meta2,
                                        params2, residuals2);

  EXPECT_EQ(consumed, buf.size());
  EXPECT_EQ(meta2.window_length, 256u);
  EXPECT_EQ(static_cast<uint8_t>(meta2.model_id), 0u);
  EXPECT_EQ(static_cast<uint8_t>(meta2.encoding_id), 1u);
  ASSERT_EQ(params2.size(), 2u);
  EXPECT_DOUBLE_EQ(params2[0], 1.5);
  EXPECT_DOUBLE_EQ(params2[1], -0.3);
  EXPECT_EQ(residuals2, residuals);
}

TEST(FormatTest, WindowBlockQuadraticRoundTrip) {
  WindowBlockMeta meta;
  meta.window_length = 512;
  meta.model_id = ModelId::Quadratic;
  meta.encoding_id = EncodingId::ZigzagVarint;
  std::vector<double> params = {0.001, -2.0, 100.0};
  std::vector<uint8_t> residuals(20);
  for (int i = 0; i < 20; ++i)
    residuals[i] = static_cast<uint8_t>(i);

  std::vector<uint8_t> buf;
  pack_window_block(meta, params, residuals, buf);

  WindowBlockMeta meta2;
  std::vector<double> params2;
  std::vector<uint8_t> residuals2;
  unpack_window_block(buf.data(), buf.size(), 0, meta2, params2, residuals2);

  EXPECT_EQ(static_cast<uint8_t>(meta2.model_id), 1u);
  ASSERT_EQ(params2.size(), 3u);
  EXPECT_EQ(residuals2, residuals);
}

TEST(FormatTest, WindowBlockXorDeltaRoundTrip) {
  WindowBlockMeta meta;
  meta.window_length = 16;
  meta.model_id = ModelId::XorDelta;
  meta.encoding_id = EncodingId::OutlierSep;
  std::vector<double> params = {3.14159};
  std::vector<uint8_t> residuals(10, 0xFF);

  std::vector<uint8_t> buf;
  pack_window_block(meta, params, residuals, buf);

  WindowBlockMeta meta2;
  std::vector<double> params2;
  std::vector<uint8_t> residuals2;
  unpack_window_block(buf.data(), buf.size(), 0, meta2, params2, residuals2);

  EXPECT_EQ(static_cast<uint8_t>(meta2.model_id), 2u);
  ASSERT_EQ(params2.size(), 1u);
  EXPECT_NEAR(params2[0], 3.14159, 1e-10);
}

TEST(FormatTest, MultipleBlocksSequential) {
  std::vector<uint8_t> full_buf;

  for (int i = 0; i < 5; ++i) {
    WindowBlockMeta meta;
    meta.window_length = 100 + i;
    meta.model_id = ModelId::Linear;
    meta.encoding_id = EncodingId::ZigzagVarint;
    std::vector<double> params = {static_cast<double>(i),
                                  static_cast<double>(i * 2)};
    std::vector<uint8_t> residuals(10 + i, static_cast<uint8_t>(i));
    pack_window_block(meta, params, residuals, full_buf);
  }

  size_t offset = 0;
  for (int i = 0; i < 5; ++i) {
    WindowBlockMeta meta;
    std::vector<double> params;
    std::vector<uint8_t> residuals;
    size_t consumed = unpack_window_block(full_buf.data(), full_buf.size(),
                                          offset, meta, params, residuals);
    EXPECT_EQ(meta.window_length, static_cast<uint32_t>(100 + i));
    EXPECT_EQ(residuals.size(), static_cast<size_t>(10 + i));
    offset += consumed;
  }
  EXPECT_EQ(offset, full_buf.size());
}

} // namespace test
} // namespace dlc
