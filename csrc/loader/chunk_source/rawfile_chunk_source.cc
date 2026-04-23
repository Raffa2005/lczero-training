#include "loader/chunk_source/rawfile_chunk_source.h"

#include <absl/log/log.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "trainingdata/trainingdata_v6.h"
#include "utils/files.h"
#include "utils/gz.h"

namespace lczero {
namespace training {

namespace {

size_t GetFrameSize(ChunkSourceLoaderConfig::FrameFormat frame_format) {
  return frame_format == ChunkSourceLoaderConfig::V7TrainingData
             ? sizeof(V7TrainingData)
             : sizeof(V6TrainingData);
}

// Reads the uncompressed size from a single-member gzip file's ISIZE footer
// (RFC 1952). Only valid for files <4GB and single-member archives, both of
// which hold for self-play game files. Returns nullopt on any I/O issue.
std::optional<size_t> GetGzipUncompressedSize(
    const std::filesystem::path& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    LOG(WARNING) << "Could not open gzip file " << filename
                 << " to read its uncompressed size.";
    return std::nullopt;
  }
  file.seekg(0, std::ios::end);
  const auto end = file.tellg();
  if (end < static_cast<std::streamoff>(4)) {
    LOG(WARNING) << "Gzip file " << filename
                 << " is too small to contain an ISIZE footer.";
    return std::nullopt;
  }
  file.seekg(-4, std::ios::end);
  uint8_t footer[4];
  if (!file.read(reinterpret_cast<char*>(footer), sizeof(footer))) {
    LOG(WARNING) << "Failed reading gzip ISIZE footer from " << filename;
    return std::nullopt;
  }
  return static_cast<size_t>(footer[0]) |
         (static_cast<size_t>(footer[1]) << 8) |
         (static_cast<size_t>(footer[2]) << 16) |
         (static_cast<size_t>(footer[3]) << 24);
}

// Returns the number of training frames the file contains, computed from the
// file size without decompressing. Falls back to 1 (the historical bucket
// count) on any error.
size_t ComputeWindowUnits(const std::filesystem::path& filename,
                          ChunkSourceLoaderConfig::FrameFormat frame_format) {
  const size_t frame_size = GetFrameSize(frame_format);
  std::optional<size_t> raw_size;
  if (filename.extension() == ".gz") {
    raw_size = GetGzipUncompressedSize(filename);
  } else {
    std::error_code error;
    const auto size = std::filesystem::file_size(filename, error);
    if (!error) raw_size = size;
  }
  if (!raw_size.has_value()) {
    LOG(WARNING) << "Falling back to one window unit for " << filename;
    return 1;
  }
  if (*raw_size < frame_size) return 1;
  if (*raw_size % frame_size != 0) {
    LOG(WARNING) << "File " << filename << " size " << *raw_size
                 << " is not a multiple of input frame size " << frame_size
                 << "; rounding down for window accounting.";
  }
  return std::max<size_t>(1, *raw_size / frame_size);
}

}  // namespace

RawFileChunkSource::RawFileChunkSource(
    const std::filesystem::path& filename,
    ChunkSourceLoaderConfig::FrameFormat frame_format)
    : filename_(filename),
      frame_format_(frame_format),
      window_units_(ComputeWindowUnits(filename, frame_format)) {}

RawFileChunkSource::~RawFileChunkSource() = default;

std::string RawFileChunkSource::GetChunkSortKey() const {
  return std::filesystem::path(filename_).filename().string();
}

size_t RawFileChunkSource::GetChunkCount() const { return 1; }

size_t RawFileChunkSource::GetWindowUnits() const { return window_units_; }

std::optional<std::vector<FrameType>> RawFileChunkSource::GetChunkData(
    size_t index) {
  if (index != 0) return std::nullopt;
  std::string data = ReadFileToString(filename_);
  if (data.empty()) return std::nullopt;

  const size_t input_size = GetFrameSize(frame_format_);
  if (data.size() % input_size != 0) {
    LOG(WARNING) << "File " << filename_ << " size " << data.size()
                 << " is not a multiple of input frame size " << input_size;
    return std::nullopt;
  }

  const size_t num_frames = data.size() / input_size;
  std::vector<V7TrainingData> result(num_frames);

  if (frame_format_ == ChunkSourceLoaderConfig::V7TrainingData) {
    std::memcpy(result.data(), data.data(), data.size());
  } else {
    const auto* v6_data = reinterpret_cast<const V6TrainingData*>(data.data());
    for (size_t i = 0; i < num_frames; ++i) {
      std::memcpy(&result[i], &v6_data[i], sizeof(V6TrainingData));
    }
  }
  return result;
}

}  // namespace training
}  // namespace lczero
