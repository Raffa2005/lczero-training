#pragma once

#include <algorithm>
#include <cstddef>

#include "proto/data_loader_config.pb.h"
#include "trainingdata/trainingdata_v6.h"

namespace lczero {
namespace training {

// The V6 estimate is calibrated on the retained v41.1 replay window. V7 has
// not been recalibrated, so preserve its historical estimate.
constexpr size_t kEstimatedGzipBytesPerV6WindowUnit = 208;
constexpr size_t kEstimatedGzipBytesPerV7WindowUnit = 240;

inline size_t GetReplayFrameSize(
    ChunkSourceLoaderConfig::FrameFormat frame_format) {
  return frame_format == ChunkSourceLoaderConfig::V7TrainingData
             ? sizeof(V7TrainingData)
             : sizeof(V6TrainingData);
}

inline size_t EstimateGzipWindowUnits(
    size_t compressed_size,
    ChunkSourceLoaderConfig::FrameFormat frame_format) {
  const size_t bytes_per_unit =
      frame_format == ChunkSourceLoaderConfig::V7TrainingData
          ? kEstimatedGzipBytesPerV7WindowUnit
          : kEstimatedGzipBytesPerV6WindowUnit;
  const size_t estimated_units =
      compressed_size / bytes_per_unit +
      static_cast<size_t>(compressed_size % bytes_per_unit != 0);
  return std::max<size_t>(1, estimated_units);
}

inline size_t EstimateRawWindowUnits(
    size_t raw_size, ChunkSourceLoaderConfig::FrameFormat frame_format) {
  return std::max<size_t>(1, raw_size / GetReplayFrameSize(frame_format));
}

}  // namespace training
}  // namespace lczero
