#include "loader/stages/chunk_source_loader.h"

#include <absl/cleanup/cleanup.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "loader/chunk_source/rawfile_chunk_source.h"
#include "loader/stages/file_path_provider.h"
#include "trainingdata/trainingdata_v6.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

namespace {

template <typename T>
class PassthroughStage : public Stage {
 public:
  explicit PassthroughStage(Queue<T>* queue) : queue_(queue) {}

  void Start() override {}
  void Stop() override {}
  StageMetricProto FlushMetrics() override { return StageMetricProto(); }
  QueueBase* GetOutput(std::string_view name = "") override {
    (void)name;
    return queue_;
  }
  void SetInputs(absl::Span<QueueBase* const> inputs) override {
    if (!inputs.empty()) {
      throw std::runtime_error("PassthroughStage expects no inputs");
    }
  }

 private:
  Queue<T>* queue_;
};

}  // namespace

TEST(ChunkSourceLoaderTest, ProcessesFiles) {
  Queue<FilePathProvider::File> input_queue(10);
  ChunkSourceLoaderConfig config;
  config.set_threads(1);
  config.mutable_output()->set_queue_capacity(10);
  ChunkSourceLoader feed(config);
  feed.SetInputs({&input_queue});
  feed.Start();

  {
    auto producer = input_queue.CreateProducer();
    // Add a file with unsupported extension (should not create ChunkSource)
    producer.Put(FilePathProvider::File{
        .filepath =
            std::filesystem::path("/test.txt"),  // unsupported extension
        .message_type = FilePathProvider::MessageType::kFile});
  }  // Producer destroyed here, closing input queue

  // Try to get output - there should be no valid ChunkSources for unsupported
  // files
  try {
    while (true) {
      auto output = feed.output_queue()->Get();
      // If we get output, it means a ChunkSource was created, which shouldn't
      // happen for unsupported files
      FAIL() << "Expected no output for unsupported file extension";
    }
  } catch (const QueueClosedException&) {
    // Expected: queue should be closed when input is done and no output
    // produced
    SUCCEED();
  }
}

TEST(ChunkSourceLoaderTest, HandlesPhases) {
  Queue<FilePathProvider::File> input_queue(10);
  ChunkSourceLoaderConfig config;
  config.set_threads(1);
  config.mutable_output()->set_queue_capacity(10);
  ChunkSourceLoader feed(config);
  feed.SetInputs({&input_queue});
  feed.Start();

  {
    auto producer = input_queue.CreateProducer();
    // Test different phases - all should be passed through even if no
    // ChunkSource is created
    producer.Put(FilePathProvider::File{
        .filepath = std::filesystem::path("/test1.gz"),
        .message_type = FilePathProvider::MessageType::kFile});

    producer.Put(FilePathProvider::File{
        .filepath = std::filesystem::path("/test2.gz"),
        .message_type = FilePathProvider::MessageType::kFile});
  }  // Producer destroyed here, closing input queue

  // Queue should eventually close when input is done
  try {
    while (true) {
      feed.output_queue()->Get();
    }
  } catch (const QueueClosedException&) {
    SUCCEED();
  }
}

TEST(ChunkSourceLoaderTest, PassesThroughInitialScanComplete) {
  Queue<FilePathProvider::File> input_queue(10);
  ChunkSourceLoaderConfig config;
  config.set_threads(1);
  config.mutable_output()->set_queue_capacity(10);
  ChunkSourceLoader feed(config);
  feed.SetInputs({&input_queue});
  feed.Start();

  {
    auto producer = input_queue.CreateProducer();
    producer.Put(FilePathProvider::File{
        .filepath = std::filesystem::path(""),
        .message_type = FilePathProvider::MessageType::kInitialScanComplete});
  }  // Producer destroyed here, closing input queue

  // Should get kInitialScanComplete in output with null ChunkSource
  auto output = feed.output_queue()->Get();
  EXPECT_EQ(output.message_type,
            FilePathProvider::MessageType::kInitialScanComplete);
  EXPECT_EQ(output.source, nullptr);

  // Queue should be closed after the single message
  try {
    feed.output_queue()->Get();
    FAIL() << "Expected queue to be closed";
  } catch (const QueueClosedException&) {
    SUCCEED();
  }
}

TEST(ChunkSourceLoaderTest, SentinelBarrierWithMultipleThreads) {
  Queue<FilePathProvider::File> input_queue(100);
  ChunkSourceLoaderConfig config;
  config.set_threads(4);
  config.mutable_output()->set_queue_capacity(100);
  ChunkSourceLoader feed(config);
  feed.SetInputs({&input_queue});
  feed.Start();

  {
    auto producer = input_queue.CreateProducer();
    // Add files that will be processed before sentinel.
    for (int i = 0; i < 20; ++i) {
      producer.Put(FilePathProvider::File{
          .filepath =
              std::filesystem::path("/test" + std::to_string(i) + ".txt"),
          .message_type = FilePathProvider::MessageType::kFile});
    }
    // Add sentinel.
    producer.Put(FilePathProvider::File{
        .filepath = std::filesystem::path(""),
        .message_type = FilePathProvider::MessageType::kInitialScanComplete});
    // Add files that arrive after sentinel.
    for (int i = 20; i < 30; ++i) {
      producer.Put(FilePathProvider::File{
          .filepath =
              std::filesystem::path("/test" + std::to_string(i) + ".txt"),
          .message_type = FilePathProvider::MessageType::kFile});
    }
  }  // Producer destroyed here, closing input queue

  // Read all outputs and verify sentinel comes after all pre-sentinel files.
  int files_before_sentinel = 0;
  int files_after_sentinel = 0;
  bool sentinel_seen = false;

  try {
    while (true) {
      auto output = feed.output_queue()->Get();
      if (output.message_type ==
          FilePathProvider::MessageType::kInitialScanComplete) {
        EXPECT_FALSE(sentinel_seen) << "Sentinel should appear exactly once";
        sentinel_seen = true;
      } else {
        if (sentinel_seen) {
          files_after_sentinel++;
        } else {
          files_before_sentinel++;
        }
      }
    }
  } catch (const QueueClosedException&) {
  }

  // Verify sentinel was seen.
  EXPECT_TRUE(sentinel_seen);
  // All 20 pre-sentinel files should be before sentinel (unsupported, so 0).
  EXPECT_EQ(files_before_sentinel, 0);
  // All 10 post-sentinel files should be after sentinel (unsupported, so 0).
  EXPECT_EQ(files_after_sentinel, 0);
}

TEST(ChunkSourceLoaderTest, RawFileChunkSourceUsesFrameCountForWindowUnits) {
  const auto path = std::filesystem::temp_directory_path() /
                    "rawfile_chunk_source_window_units.bin";
  std::error_code error;
  std::filesystem::remove(path, error);
  absl::Cleanup cleanup = [&] { std::filesystem::remove(path, error); };

  std::vector<V6TrainingData> frames(3);
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file.write(reinterpret_cast<const char*>(frames.data()),
             static_cast<std::streamsize>(frames.size() * sizeof(frames[0])));
  file.close();

  RawFileChunkSource raw_source(path, ChunkSourceLoaderConfig::V6TrainingData);
  ChunkSource& source = raw_source;
  EXPECT_EQ(source.GetChunkCount(), 1u);
  EXPECT_EQ(source.GetWindowUnits(), 3u);
}

TEST(ChunkSourceLoaderTest, RawGzipWindowAccountingUsesCompressedSizeEstimate) {
  const auto path = std::filesystem::temp_directory_path() /
                    "rawfile_chunk_source_window_units.gz";
  std::error_code error;
  std::filesystem::remove(path, error);
  absl::Cleanup cleanup = [&] { std::filesystem::remove(path, error); };

  // Window accounting intentionally does not read the gzip footer; it uses
  // compressed file size as a cheap estimate instead.
  constexpr size_t kFrames = 3;
  const uint32_t raw_size = kFrames * sizeof(V6TrainingData);
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  const std::string padding(8, 'x');
  file.write(padding.data(), static_cast<std::streamsize>(padding.size()));
  const char footer[4] = {
      static_cast<char>(raw_size & 0xff),
      static_cast<char>((raw_size >> 8) & 0xff),
      static_cast<char>((raw_size >> 16) & 0xff),
      static_cast<char>((raw_size >> 24) & 0xff),
  };
  file.write(footer, sizeof(footer));
  file.close();

  RawFileChunkSource raw_source(path, ChunkSourceLoaderConfig::V6TrainingData);
  ChunkSource& source = raw_source;
  EXPECT_EQ(source.GetChunkCount(), 1u);
  EXPECT_EQ(source.GetWindowUnits(), 1u);
}

}  // namespace training
}  // namespace lczero
