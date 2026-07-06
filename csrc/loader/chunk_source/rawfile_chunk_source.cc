#include "loader/chunk_source/rawfile_chunk_source.h"

#include <absl/log/log.h>

#include <algorithm>
#include <cstring>

#include "trainingdata/trainingdata_v6.h"
#include "utils/files.h"
#include "utils/gz.h"

namespace lczero {
namespace training {

namespace {

constexpr size_t kEstimatedGzipBytesPerWindowUnit = 240;

size_t GetFrameSize(ChunkSourceLoaderConfig::FrameFormat frame_format) {
  return frame_format == ChunkSourceLoaderConfig::V7TrainingData
             ? sizeof(V7TrainingData)
             : sizeof(V6TrainingData);
}

size_t EstimateGzipWindowUnits(const std::filesystem::path& filename) {
  std::error_code error;
  const auto size = std::filesystem::file_size(filename, error);
  if (error) {
    LOG(WARNING) << "Could not stat gzip file " << filename
                 << " to estimate its window units: " << error.message();
    return 1;
  }
  const size_t estimated_units =
      (static_cast<size_t>(size) + kEstimatedGzipBytesPerWindowUnit - 1) /
      kEstimatedGzipBytesPerWindowUnit;
  return std::max<size_t>(1, estimated_units);
}

// Returns the number of training frames the file contains, computed from the
// file size without decompressing. For gzip files, use compressed size as a
// calibrated proxy; reading the gzip footer for every candidate is too costly
// on cold Lustre replay directories. Falls back to 1 (the historical bucket
// count) on any error.
size_t ComputeWindowUnits(const std::filesystem::path& filename,
                          ChunkSourceLoaderConfig::FrameFormat frame_format) {
  const size_t frame_size = GetFrameSize(frame_format);
  if (filename.extension() == ".gz") {
    return EstimateGzipWindowUnits(filename);
  }

  std::error_code error;
  const auto raw_size = std::filesystem::file_size(filename, error);
  if (error) {
    LOG(WARNING) << "Falling back to one window unit for " << filename;
    return 1;
  }
  if (raw_size < frame_size) return 1;
  if (raw_size % frame_size != 0) {
    LOG(WARNING) << "File " << filename << " size " << raw_size
                 << " is not a multiple of input frame size " << frame_size
                 << "; rounding down for window accounting.";
  }
  return std::max<size_t>(1, raw_size / frame_size);
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
