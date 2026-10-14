#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "trainingdata/trainingdata_v6.h"

ABSL_FLAG(std::string, input_dir, "",
          "Directory containing raw self-play .gz game files.");
ABSL_FLAG(std::string, input_file_list, "",
          "Optional text file containing one gzip path per line. Mutually "
          "exclusive with input_dir filtering.");
ABSL_FLAG(std::string, output_dir, "",
          "Directory where CSV and Markdown outputs will be written.");
ABSL_FLAG(int, num_threads, 0,
          "Worker threads. Defaults to hardware concurrency.");
ABSL_FLAG(double, clip_epsilon, 1e-6,
          "Clip root_q to [-1+epsilon, 1-epsilon] before atanh.");
ABSL_FLAG(int, max_games, 0,
          "Optional smoke-test cap after chronological sorting. 0 means all.");
ABSL_FLAG(std::string, filename_regex, "",
          "Optional ECMAScript regex applied to each gzip basename before "
          "max_games. Empty means all gzip files.");
ABSL_FLAG(bool, split_stream_outputs, false,
          "Also write p1 and p2 subsets, inferred from _p1_game_/_p2_game_.");
ABSL_FLAG(int, bootstrap_replicates, 0,
          "Whole-game bootstrap replicates for queen, rook, and minor buckets. "
          "0 disables bootstrap output.");
ABSL_FLAG(uint64_t, bootstrap_seed, 20260713,
          "Deterministic seed for whole-game bootstrap resampling.");
ABSL_FLAG(double, bootstrap_bandwidth, 0.25,
          "Kernel bandwidth used for bootstrap intervals.");
ABSL_FLAG(bool, write_bootstrap_moments, false,
          "Write each bootstrap replicate's compact kernel moments for later "
          "pooled-window confidence intervals.");
ABSL_FLAG(bool, write_joint_resource_rows, false,
          "Write side-to-move-oriented piece and ability differences for a "
          "joint pawn-normalized resource-value fit.");
ABSL_FLAG(bool, allow_non_slurm, false,
          "Allow running outside a SLURM allocation. Keep false for full runs.");

namespace {

namespace fs = std::filesystem;
using ::lczero::V6TrainingData;

constexpr int kBucketCount = 5;
constexpr int kThresholdCount = 6;
constexpr std::array<double, kThresholdCount> kThresholds = {
    0.10, 0.25, 0.50, 0.75, 0.95, 1.00};
constexpr int kBandwidthCount = 6;
constexpr std::array<double, kBandwidthCount> kBandwidths = {
    0.10, 0.15, 0.20, 0.25, 0.35, 0.50};
constexpr std::array<const char*, kBucketCount> kBucketNames = {
    "all", "queen_present", "rook_only", "minor_only", "pawns_only"};

struct Row {
  uint32_t game_index = 0;
  uint8_t bucket = 0;
  uint8_t stream = 0;
  double q_dm = 0.0;
  double z_dm = 0.0;
  double deficit = 0.0;
};

struct JointResourceRow {
  uint32_t game_index = 0;
  uint8_t stream = 0;
  uint8_t ability_bucket = 0;
  double q = 0.0;
  double result_q = 0.0;
  std::array<int8_t, 5> piece_differences{};
  int8_t ability_difference = 0;
  uint8_t our_ability = 0;
  uint8_t their_ability = 0;
};

struct GameStats {
  uint8_t stream = 0;
  uint64_t frames = 0;
  uint64_t one_sided = 0;
  uint64_t kept = 0;
  uint64_t corrected_mid_rows = 0;
  bool ok = false;
  std::string error;
};

struct ThreadRows {
  std::vector<Row> rows;
  std::vector<JointResourceRow> joint_resource_rows;
};

int PopCount(uint64_t x) { return static_cast<int>(std::popcount(x)); }

double Clip(double q, double epsilon) {
  return std::min(1.0 - epsilon, std::max(-1.0 + epsilon, q));
}

int PieceValueUsMinusThem(const V6TrainingData& frame) {
  const int pawns = PopCount(frame.planes[0]) - PopCount(frame.planes[6]);
  const int knights = PopCount(frame.planes[1]) - PopCount(frame.planes[7]);
  const int bishops = PopCount(frame.planes[2]) - PopCount(frame.planes[8]);
  const int rooks = PopCount(frame.planes[3]) - PopCount(frame.planes[9]);
  const int queens = PopCount(frame.planes[4]) - PopCount(frame.planes[10]);
  return pawns + 3 * knights + 3 * bishops + 5 * rooks + 9 * queens;
}

std::array<int8_t, 5> PieceDifferencesUsMinusThem(
    const V6TrainingData& frame) {
  std::array<int8_t, 5> differences{};
  for (int piece = 0; piece < 5; ++piece) {
    differences[piece] = static_cast<int8_t>(
        PopCount(frame.planes[piece]) - PopCount(frame.planes[piece + 6]));
  }
  return differences;
}

uint8_t HighestNoDmPieceBucket(const V6TrainingData& frame, bool dm_is_us) {
  const int offset = dm_is_us ? 6 : 0;
  if (PopCount(frame.planes[offset + 4]) > 0) return 1;
  if (PopCount(frame.planes[offset + 3]) > 0) return 2;
  if (PopCount(frame.planes[offset + 1]) + PopCount(frame.planes[offset + 2]) > 0) {
    return 3;
  }
  return 4;
}

bool ReadOneFrame(gzFile file, V6TrainingData* frame, std::string* error) {
  const int bytes = gzread(file, frame, sizeof(*frame));
  if (bytes == 0) return false;
  if (bytes < 0) {
    int errnum = 0;
    const char* message = gzerror(file, &errnum);
    *error = message ? message : "unknown gzip read error";
    return false;
  }
  if (bytes != static_cast<int>(sizeof(*frame))) {
    *error = "partial frame at end of file";
    return false;
  }
  return true;
}

uint8_t StreamFromFilename(const fs::path& path) {
  const std::string name = path.filename().string();
  if (name.find("_p1_game_") != std::string::npos) return 1;
  if (name.find("_p2_game_") != std::string::npos) return 2;
  return 0;
}

void ProcessGame(const fs::path& path, uint32_t game_index, uint8_t stream,
                 double clip_epsilon,
                 ThreadRows* out, GameStats* stats) {
  stats->stream = stream;
  gzFile file = gzopen(path.string().c_str(), "rb");
  if (file == nullptr) {
    stats->error = "failed to open gzip file";
    return;
  }

  while (true) {
    V6TrainingData frame;
    std::string read_error;
    const bool got_frame = ReadOneFrame(file, &frame, &read_error);
    if (!got_frame) {
      if (!read_error.empty()) stats->error = read_error;
      break;
    }
    ++stats->frames;

    if (frame.is_mid_doublemove) {
      ++stats->corrected_mid_rows;
      continue;
    }

    const bool our = frame.our_doublemove_available != 0;
    const bool their = frame.their_doublemove_available != 0;
    double q = frame.root_q;
    if (std::isfinite(q)) {
      q = Clip(q, clip_epsilon);
      const int8_t ability_difference =
          static_cast<int8_t>(static_cast<int>(our) - static_cast<int>(their));
      uint8_t ability_bucket = 0;
      if (ability_difference != 0) {
        ability_bucket = HighestNoDmPieceBucket(frame, ability_difference > 0);
      }
      out->joint_resource_rows.push_back(JointResourceRow{
          game_index, stream, ability_bucket, q,
          static_cast<double>(frame.result_q),
          PieceDifferencesUsMinusThem(frame), ability_difference,
          static_cast<uint8_t>(our), static_cast<uint8_t>(their)});
    }

    if (our == their) continue;
    ++stats->one_sided;

    if (!std::isfinite(q)) continue;
    // V6 root_q is stored in lc0's historical node convention; mid-double
    // roots were skipped above, so ordinary roots are already side-to-move.
    q = Clip(q, clip_epsilon);

    const bool dm_is_us = our && !their;
    const double q_dm = dm_is_us ? q : -q;
    const double z_dm = std::atanh(q_dm);
    if (!std::isfinite(z_dm)) continue;

    const int material_us_minus_them = PieceValueUsMinusThem(frame);
    const int deficit = dm_is_us ? -material_us_minus_them
                                 : material_us_minus_them;
    const uint8_t bucket = HighestNoDmPieceBucket(frame, dm_is_us);

    out->rows.push_back(Row{game_index, 0, stream, q_dm, z_dm,
                            static_cast<double>(deficit)});
    out->rows.push_back(Row{game_index, bucket, stream, q_dm, z_dm,
                            static_cast<double>(deficit)});
    ++stats->kept;
  }

  const int close_result = gzclose(file);
  if (close_result != Z_OK && stats->error.empty()) {
    stats->error = "gzip close failed";
  }
  stats->ok = stats->error.empty();
}

void WriteJointResourceRows(const fs::path& output_dir,
                            const std::vector<JointResourceRow>& rows) {
  fs::create_directories(output_dir);
  std::ofstream out(output_dir / "joint_resource_rows.csv");
  out << "game_index,stream,q,pawn_diff,knight_diff,bishop_diff,rook_diff,"
         "queen_diff,ability_diff,ability_bucket,result_q,our_ability,"
         "their_ability\n";
  out << std::setprecision(17);
  for (const JointResourceRow& row : rows) {
    out << row.game_index << ',' << static_cast<int>(row.stream) << ','
        << row.q;
    for (int8_t difference : row.piece_differences) {
      out << ',' << static_cast<int>(difference);
    }
    out << ',' << static_cast<int>(row.ability_difference) << ','
        << static_cast<int>(row.ability_bucket) << ',' << row.result_q << ','
        << static_cast<int>(row.our_ability) << ','
        << static_cast<int>(row.their_ability) << '\n';
  }
}

std::vector<fs::path> ReadGzipFileList(const fs::path& list_path,
                                       int max_games) {
  std::ifstream input(list_path);
  if (!input) LOG(FATAL) << "Could not open --input_file_list: " << list_path;
  std::vector<fs::path> files;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    files.emplace_back(line);
  }
  if (max_games > 0 && static_cast<size_t>(max_games) < files.size()) {
    files.resize(static_cast<size_t>(max_games));
  }
  return files;
}

std::vector<fs::path> CollectGzipFiles(const fs::path& input_dir,
                                       int max_games,
                                       const std::string& filename_regex) {
  std::optional<std::regex> filter;
  if (!filename_regex.empty()) {
    try {
      filter.emplace(filename_regex);
    } catch (const std::regex_error& error) {
      LOG(FATAL) << "Invalid --filename_regex: " << error.what();
    }
  }

  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".gz") continue;
    const std::string basename = entry.path().filename().string();
    if (filter.has_value() && !std::regex_search(basename, *filter)) continue;
    files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end(),
            [](const fs::path& a, const fs::path& b) {
              return a.filename().string() < b.filename().string();
            });
  if (max_games > 0 && static_cast<size_t>(max_games) < files.size()) {
    files.resize(static_cast<size_t>(max_games));
  }
  return files;
}

size_t WorkerCount(size_t item_count, int requested) {
  if (item_count == 0) return 0;
  size_t count = requested > 0 ? static_cast<size_t>(requested)
                              : std::thread::hardware_concurrency();
  if (count == 0) count = 1;
  return std::max<size_t>(1, std::min(count, item_count));
}

struct ThresholdStats {
  uint64_t positions = 0;
  uint64_t games = 0;
  double mean = std::numeric_limits<double>::quiet_NaN();
  double median = std::numeric_limits<double>::quiet_NaN();
  double p25 = std::numeric_limits<double>::quiet_NaN();
  double p75 = std::numeric_limits<double>::quiet_NaN();
};

struct KernelStats {
  uint64_t positions = 0;
  uint64_t games = 0;
  double weight_sum = 0.0;
  double weight_square_sum = 0.0;
  double weighted_q_sum = 0.0;
  double weighted_deficit_sum = 0.0;
  double weighted_q2_sum = 0.0;
  double weighted_q_deficit_sum = 0.0;
  double weighted_deficit2_sum = 0.0;
  double effective_n = std::numeric_limits<double>::quiet_NaN();
  double weighted_mean_deficit = std::numeric_limits<double>::quiet_NaN();
  double local_intercept = std::numeric_limits<double>::quiet_NaN();
  double local_slope = std::numeric_limits<double>::quiet_NaN();
  double local_rmse = std::numeric_limits<double>::quiet_NaN();
};

double Quantile(std::vector<double>* values, double p) {
  if (values->empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values->begin(), values->end());
  const double raw = p * static_cast<double>(values->size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(raw));
  const size_t hi = static_cast<size_t>(std::ceil(raw));
  if (lo == hi) return (*values)[lo];
  const double frac = raw - static_cast<double>(lo);
  return (*values)[lo] * (1.0 - frac) + (*values)[hi] * frac;
}

void CountGames(const std::vector<uint32_t>& games, uint64_t* out) {
  if (games.empty()) {
    *out = 0;
    return;
  }
  std::vector<uint32_t> copy = games;
  std::sort(copy.begin(), copy.end());
  *out = static_cast<uint64_t>(std::unique(copy.begin(), copy.end()) - copy.begin());
}

ThresholdStats ComputeThresholdStats(const std::vector<Row>& rows, int bucket,
                                     double abs_q_threshold) {
  std::vector<double> deficits;
  std::vector<uint32_t> games;
  deficits.reserve(rows.size() / 4);
  for (const Row& row : rows) {
    if (row.bucket != bucket || std::fabs(row.q_dm) >= abs_q_threshold) continue;
    deficits.push_back(row.deficit);
    games.push_back(row.game_index);
  }

  ThresholdStats stats;
  stats.positions = deficits.size();
  CountGames(games, &stats.games);
  if (deficits.empty()) return stats;

  const double sum = std::accumulate(deficits.begin(), deficits.end(), 0.0);
  stats.mean = sum / static_cast<double>(deficits.size());
  std::vector<double> qvals = deficits;
  stats.p25 = Quantile(&qvals, 0.25);
  qvals = deficits;
  stats.median = Quantile(&qvals, 0.50);
  qvals = deficits;
  stats.p75 = Quantile(&qvals, 0.75);
  return stats;
}

double CompactKernelWeight(double q, double bandwidth) {
  const double u = std::fabs(q) / bandwidth;
  if (u >= 1.0) return 0.0;
  const double t = 1.0 - u * u;
  return t * t;
}

KernelStats ComputeKernelStats(const std::vector<Row>& rows, int bucket,
                               double bandwidth) {
  double sw = 0.0;
  double sw2 = 0.0;
  double swx = 0.0;
  double swy = 0.0;
  double swxx = 0.0;
  double swxy = 0.0;
  double swyy = 0.0;
  std::vector<uint32_t> games;

  for (const Row& row : rows) {
    if (row.bucket != bucket) continue;
    const double w = CompactKernelWeight(row.q_dm, bandwidth);
    if (w <= 0.0) continue;
    const double x = row.q_dm;
    const double y = row.deficit;
    games.push_back(row.game_index);
    sw += w;
    sw2 += w * w;
    swx += w * x;
    swy += w * y;
    swxx += w * x * x;
    swxy += w * x * y;
    swyy += w * y * y;
  }

  KernelStats stats;
  stats.positions = games.size();
  CountGames(games, &stats.games);
  stats.weight_sum = sw;
  stats.weight_square_sum = sw2;
  stats.weighted_q_sum = swx;
  stats.weighted_deficit_sum = swy;
  stats.weighted_q2_sum = swxx;
  stats.weighted_q_deficit_sum = swxy;
  stats.weighted_deficit2_sum = swyy;
  if (sw2 > 0.0) stats.effective_n = sw * sw / sw2;
  if (sw <= 0.0) return stats;
  stats.weighted_mean_deficit = swy / sw;

  const double denom = sw * swxx - swx * swx;
  if (std::fabs(denom) < 1e-12) return stats;
  stats.local_slope = (sw * swxy - swx * swy) / denom;
  stats.local_intercept = (swy - stats.local_slope * swx) / sw;

  const double sse = std::max(
      0.0, swyy + sw * stats.local_intercept * stats.local_intercept +
               stats.local_slope * stats.local_slope * swxx +
               2.0 * stats.local_intercept * stats.local_slope * swx -
               2.0 * stats.local_intercept * swy -
               2.0 * stats.local_slope * swxy);
  stats.local_rmse = std::sqrt(sse / sw);
  return stats;
}

struct KernelMoments {
  double sw = 0.0;
  double swx = 0.0;
  double swy = 0.0;
  double swxx = 0.0;
  double swxy = 0.0;

  void Add(const Row& row, double weight) {
    sw += weight;
    swx += weight * row.q_dm;
    swy += weight * row.deficit;
    swxx += weight * row.q_dm * row.q_dm;
    swxy += weight * row.q_dm * row.deficit;
  }

  KernelMoments& operator+=(const KernelMoments& other) {
    sw += other.sw;
    swx += other.swx;
    swy += other.swy;
    swxx += other.swxx;
    swxy += other.swxy;
    return *this;
  }
};

struct BootstrapStats {
  int requested_replicates = 0;
  int valid_replicates = 0;
  double point_weighted_mean = std::numeric_limits<double>::quiet_NaN();
  double point_intercept = std::numeric_limits<double>::quiet_NaN();
  double weighted_mean_bootstrap_mean = std::numeric_limits<double>::quiet_NaN();
  double weighted_mean_bootstrap_se = std::numeric_limits<double>::quiet_NaN();
  double weighted_mean_ci_low = std::numeric_limits<double>::quiet_NaN();
  double weighted_mean_ci_high = std::numeric_limits<double>::quiet_NaN();
  double intercept_bootstrap_mean = std::numeric_limits<double>::quiet_NaN();
  double intercept_bootstrap_se = std::numeric_limits<double>::quiet_NaN();
  double intercept_ci_low = std::numeric_limits<double>::quiet_NaN();
  double intercept_ci_high = std::numeric_limits<double>::quiet_NaN();
  std::vector<KernelMoments> replicate_moments;
};

std::pair<double, double> FinalizeKernelMoments(const KernelMoments& moments) {
  if (moments.sw <= 0.0) {
    return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
  }
  const double weighted_mean = moments.swy / moments.sw;
  const double denom = moments.sw * moments.swxx - moments.swx * moments.swx;
  if (std::fabs(denom) < 1e-12) {
    return {weighted_mean, std::numeric_limits<double>::quiet_NaN()};
  }
  const double slope =
      (moments.sw * moments.swxy - moments.swx * moments.swy) / denom;
  const double intercept = (moments.swy - slope * moments.swx) / moments.sw;
  return {weighted_mean, intercept};
}

double SampleStandardDeviation(const std::vector<double>& values) {
  if (values.size() < 2) return std::numeric_limits<double>::quiet_NaN();
  const double mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double squared_sum = 0.0;
  for (double value : values) {
    const double delta = value - mean;
    squared_sum += delta * delta;
  }
  return std::sqrt(squared_sum / static_cast<double>(values.size() - 1));
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

BootstrapStats BootstrapKernelStats(const std::vector<Row>& rows,
                                    const std::vector<uint32_t>& game_indices,
                                    int bucket, double bandwidth, int replicates,
                                    uint64_t seed) {
  BootstrapStats stats;
  stats.requested_replicates = replicates;
  if (replicates <= 0 || game_indices.empty()) return stats;

  uint32_t max_game_index = 0;
  for (uint32_t game_index : game_indices) {
    max_game_index = std::max(max_game_index, game_index);
  }
  std::vector<int32_t> local_index(max_game_index + 1, -1);
  for (size_t i = 0; i < game_indices.size(); ++i) {
    local_index[game_indices[i]] = static_cast<int32_t>(i);
  }

  std::vector<KernelMoments> per_game(game_indices.size());
  for (const Row& row : rows) {
    if (row.bucket != bucket || row.game_index >= local_index.size()) continue;
    const int32_t local = local_index[row.game_index];
    if (local < 0) continue;
    const double weight = CompactKernelWeight(row.q_dm, bandwidth);
    if (weight > 0.0) per_game[local].Add(row, weight);
  }

  KernelMoments point_moments;
  for (const KernelMoments& moments : per_game) point_moments += moments;
  std::tie(stats.point_weighted_mean, stats.point_intercept) =
      FinalizeKernelMoments(point_moments);

  std::mt19937_64 random(seed);
  std::uniform_int_distribution<size_t> draw(0, per_game.size() - 1);
  std::vector<double> weighted_means;
  std::vector<double> intercepts;
  weighted_means.reserve(replicates);
  intercepts.reserve(replicates);
  stats.replicate_moments.reserve(replicates);
  for (int replicate = 0; replicate < replicates; ++replicate) {
    KernelMoments sampled;
    for (size_t game = 0; game < per_game.size(); ++game) {
      sampled += per_game[draw(random)];
    }
    stats.replicate_moments.push_back(sampled);
    const auto [weighted_mean, intercept] = FinalizeKernelMoments(sampled);
    if (std::isfinite(weighted_mean) && std::isfinite(intercept)) {
      weighted_means.push_back(weighted_mean);
      intercepts.push_back(intercept);
    }
  }

  stats.valid_replicates = static_cast<int>(intercepts.size());
  if (intercepts.empty()) return stats;
  stats.weighted_mean_bootstrap_mean = Mean(weighted_means);
  stats.weighted_mean_bootstrap_se = SampleStandardDeviation(weighted_means);
  stats.intercept_bootstrap_mean = Mean(intercepts);
  stats.intercept_bootstrap_se = SampleStandardDeviation(intercepts);
  std::vector<double> quantiles = weighted_means;
  stats.weighted_mean_ci_low = Quantile(&quantiles, 0.025);
  quantiles = weighted_means;
  stats.weighted_mean_ci_high = Quantile(&quantiles, 0.975);
  quantiles = intercepts;
  stats.intercept_ci_low = Quantile(&quantiles, 0.025);
  quantiles = intercepts;
  stats.intercept_ci_high = Quantile(&quantiles, 0.975);
  return stats;
}

std::string Format(double value, int precision = 3) {
  if (!std::isfinite(value)) return "nan";
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

void WriteOutputs(const fs::path& output_dir, const fs::path& input_dir,
                  const fs::path& input_file_list,
                  const std::string& filename_regex,
                  const std::vector<fs::path>& files,
                  const std::vector<GameStats>& game_stats,
                  const std::vector<uint32_t>& game_indices,
                  const std::vector<Row>& rows, int bootstrap_replicates,
                  uint64_t bootstrap_seed, double bootstrap_bandwidth,
                  bool write_bootstrap_moments) {
  fs::create_directories(output_dir);

  uint64_t total_frames = 0;
  uint64_t total_one_sided = 0;
  uint64_t total_kept = 0;
  uint64_t corrected_mid_rows = 0;
  uint64_t failed_games = 0;
  for (const GameStats& stats : game_stats) {
    total_frames += stats.frames;
    total_one_sided += stats.one_sided;
    total_kept += stats.kept;
    corrected_mid_rows += stats.corrected_mid_rows;
    if (!stats.ok) ++failed_games;
  }

  {
    std::ofstream out(output_dir / "threshold_summary.csv");
    out << "bucket,abs_q_lt,positions,games,mean_deficit,median_deficit,p25_deficit,p75_deficit\n";
    out << std::setprecision(17);
    for (int bucket = 0; bucket < kBucketCount; ++bucket) {
      for (double threshold : kThresholds) {
        const ThresholdStats stats = ComputeThresholdStats(rows, bucket, threshold);
        out << kBucketNames[bucket] << ',' << threshold << ',' << stats.positions
            << ',' << stats.games << ',' << stats.mean << ',' << stats.median
            << ',' << stats.p25 << ',' << stats.p75 << '\n';
      }
    }
  }

  {
    std::ofstream out(output_dir / "kernel_summary.csv");
    out << "bucket,bandwidth,positions,games,weight_sum,effective_n,"
           "weighted_mean_deficit,local_intercept,local_slope,local_rmse,"
           "weight_square_sum,weighted_q_sum,weighted_deficit_sum,"
           "weighted_q2_sum,weighted_q_deficit_sum,weighted_deficit2_sum\n";
    out << std::setprecision(17);
    for (int bucket = 0; bucket < kBucketCount; ++bucket) {
      for (double bandwidth : kBandwidths) {
        const KernelStats stats = ComputeKernelStats(rows, bucket, bandwidth);
        out << kBucketNames[bucket] << ',' << bandwidth << ','
            << stats.positions << ',' << stats.games << ','
            << stats.weight_sum << ',' << stats.effective_n << ','
            << stats.weighted_mean_deficit << ',' << stats.local_intercept
            << ',' << stats.local_slope << ',' << stats.local_rmse << ','
            << stats.weight_square_sum << ',' << stats.weighted_q_sum << ','
            << stats.weighted_deficit_sum << ',' << stats.weighted_q2_sum
            << ',' << stats.weighted_q_deficit_sum << ','
            << stats.weighted_deficit2_sum << '\n';
      }
    }
  }

  if (bootstrap_replicates > 0) {
    std::ofstream out(output_dir / "kernel_bootstrap.csv");
    std::ofstream moments_out;
    if (write_bootstrap_moments) {
      moments_out.open(output_dir / "kernel_bootstrap_moments.csv");
      moments_out
          << "bucket,bandwidth,replicate,weight_sum,weighted_q_sum,"
             "weighted_deficit_sum,weighted_q2_sum,"
             "weighted_q_deficit_sum\n";
      moments_out << std::setprecision(17);
    }
    out << "bucket,bandwidth,games,requested_replicates,valid_replicates,"
           "point_weighted_mean,weighted_mean_bootstrap_mean,"
           "weighted_mean_bootstrap_se,weighted_mean_ci_low,"
           "weighted_mean_ci_high,point_intercept,intercept_bootstrap_mean,"
           "intercept_bootstrap_se,intercept_ci_low,intercept_ci_high\n";
    out << std::setprecision(17);
    for (int bucket : {1, 2, 3}) {
      const BootstrapStats stats = BootstrapKernelStats(
          rows, game_indices, bucket, bootstrap_bandwidth,
          bootstrap_replicates, bootstrap_seed + static_cast<uint64_t>(bucket));
      out << kBucketNames[bucket] << ',' << bootstrap_bandwidth << ','
          << game_indices.size() << ',' << stats.requested_replicates << ','
          << stats.valid_replicates << ',' << stats.point_weighted_mean << ','
          << stats.weighted_mean_bootstrap_mean << ','
          << stats.weighted_mean_bootstrap_se << ','
          << stats.weighted_mean_ci_low << ',' << stats.weighted_mean_ci_high
          << ',' << stats.point_intercept << ','
          << stats.intercept_bootstrap_mean << ','
          << stats.intercept_bootstrap_se << ',' << stats.intercept_ci_low
          << ',' << stats.intercept_ci_high << '\n';
      if (write_bootstrap_moments) {
        for (size_t replicate = 0;
             replicate < stats.replicate_moments.size(); ++replicate) {
          const KernelMoments& moments = stats.replicate_moments[replicate];
          moments_out << kBucketNames[bucket] << ',' << bootstrap_bandwidth
                      << ',' << replicate << ',' << moments.sw << ','
                      << moments.swx << ',' << moments.swy << ','
                      << moments.swxx << ',' << moments.swxy << '\n';
        }
      }
    }
  }

  {
    std::ofstream out(output_dir / "summary.md");
    out << "# Double-Move Material Compensation\n\n";
    if (!input_file_list.empty()) {
      out << "- input_file_list: `" << input_file_list.string() << "`\n";
    } else {
      out << "- input_dir: `" << input_dir.string() << "`\n";
    }
    out << "- filename_regex: `"
        << (filename_regex.empty() ? "<all>" : filename_regex) << "`\n";
    out << "- output_dir: `" << output_dir.string() << "`\n";
    out << "- games: " << files.size() << "\n";
    out << "- total frames read: " << total_frames << "\n";
    out << "- one-sided non-mid positions: " << total_one_sided << "\n";
    out << "- kept one-sided positions: " << total_kept << "\n";
    out << "- analysis rows including all+buckets: " << rows.size() << "\n";
    out << "- skipped mid-double rows: " << corrected_mid_rows << "\n";
    out << "- failed game files: " << failed_games << "\n";
    out << "- bootstrap_replicates: " << bootstrap_replicates << "\n";
    if (bootstrap_replicates > 0) {
      out << "- bootstrap_bandwidth: " << bootstrap_bandwidth << "\n";
      out << "- bootstrap_seed: " << bootstrap_seed << "\n";
    }
    if (!files.empty()) {
      out << "- first file: `" << files.front().filename().string() << "`\n";
      out << "- last file: `" << files.back().filename().string() << "`\n";
    }
    out << "\nMaterial uses conventional values `P=1, N/B=3, R=5, Q=9`.\n";
    out << "`deficit = material(no-DM side) - material(DM side)`, so positive "
           "means the side with the remaining double move is down material.\n";
    out << "`q` is converted to the DM side's perspective before filtering/fitting.\n\n";

    out << "## Hard-Threshold Balanced-Position Deficit\n\n";
    out << "| bucket | |q| < 0.10 n | median | mean | IQR | |q| < 0.25 n | median | mean | IQR | |q| < 0.50 n | median | mean | IQR |\n";
    out << "|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|\n";
    for (int bucket = 0; bucket < kBucketCount; ++bucket) {
      const ThresholdStats s10 = ComputeThresholdStats(rows, bucket, 0.10);
      const ThresholdStats s25 = ComputeThresholdStats(rows, bucket, 0.25);
      const ThresholdStats s50 = ComputeThresholdStats(rows, bucket, 0.50);
      out << "| " << kBucketNames[bucket] << " | " << s10.positions << " | "
          << Format(s10.median) << " | " << Format(s10.mean) << " | ["
          << Format(s10.p25) << ", " << Format(s10.p75) << "] | "
          << s25.positions << " | "
          << Format(s25.median) << " | " << Format(s25.mean) << " | ["
          << Format(s25.p25) << ", " << Format(s25.p75) << "] | "
          << s50.positions << " | " << Format(s50.median) << " | "
          << Format(s50.mean) << " | [" << Format(s50.p25) << ", "
          << Format(s50.p75) << "] |\n";
    }

    out << "\n## Compact-Kernel Local Estimates\n\n";
    out << "Kernel: `w = (1 - (|q| / h)^2)^2` for `|q| < h`, else zero.\n";
    out << "Local fit: `deficit = a + b * q_dm`; the reported local-linear value is `a`.\n\n";

    out << "| bucket | h=0.10 weighted mean | h=0.10 local a | h=0.15 weighted mean | h=0.15 local a | h=0.20 weighted mean | h=0.20 local a | h=0.25 weighted mean | h=0.25 local a | h=0.35 weighted mean | h=0.35 local a | h=0.50 weighted mean | h=0.50 local a |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (int bucket = 0; bucket < kBucketCount; ++bucket) {
      out << "| " << kBucketNames[bucket];
      for (double bandwidth : kBandwidths) {
        const KernelStats stats = ComputeKernelStats(rows, bucket, bandwidth);
        out << " | " << Format(stats.weighted_mean_deficit)
            << " | " << Format(stats.local_intercept);
      }
      out << " |\n";
    }

    out << "\nFull numeric outputs are in `threshold_summary.csv`, "
           "and `kernel_summary.csv`.\n";
  }
}

struct OutputSubset {
  std::vector<fs::path> files;
  std::vector<GameStats> game_stats;
  std::vector<uint32_t> game_indices;
  std::vector<Row> rows;
};

OutputSubset SelectStream(const std::vector<fs::path>& files,
                          const std::vector<GameStats>& game_stats,
                          const std::vector<Row>& rows, uint8_t stream) {
  OutputSubset subset;
  for (size_t index = 0; index < files.size(); ++index) {
    if (game_stats[index].stream != stream) continue;
    subset.files.push_back(files[index]);
    subset.game_stats.push_back(game_stats[index]);
    subset.game_indices.push_back(static_cast<uint32_t>(index));
  }
  subset.rows.reserve(rows.size() / 2);
  for (const Row& row : rows) {
    if (row.stream == stream) subset.rows.push_back(row);
  }
  return subset;
}

bool IsRunningUnderSlurm() { return std::getenv("SLURM_JOB_ID") != nullptr; }

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  if (!IsRunningUnderSlurm() && !absl::GetFlag(FLAGS_allow_non_slurm)) {
    LOG(FATAL) << "Refusing to run outside a SLURM allocation. Use "
                  "--allow_non_slurm=true only for tiny smoke tests.";
  }

  const fs::path input_dir(absl::GetFlag(FLAGS_input_dir));
  const fs::path input_file_list(absl::GetFlag(FLAGS_input_file_list));
  const fs::path output_dir(absl::GetFlag(FLAGS_output_dir));
  const int num_threads = absl::GetFlag(FLAGS_num_threads);
  const int max_games = absl::GetFlag(FLAGS_max_games);
  const std::string filename_regex = absl::GetFlag(FLAGS_filename_regex);
  const bool split_stream_outputs = absl::GetFlag(FLAGS_split_stream_outputs);
  const bool write_bootstrap_moments =
      absl::GetFlag(FLAGS_write_bootstrap_moments);
  const bool write_joint_resource_rows =
      absl::GetFlag(FLAGS_write_joint_resource_rows);
  const int bootstrap_replicates = absl::GetFlag(FLAGS_bootstrap_replicates);
  const uint64_t bootstrap_seed = absl::GetFlag(FLAGS_bootstrap_seed);
  const double bootstrap_bandwidth = absl::GetFlag(FLAGS_bootstrap_bandwidth);
  const double clip_epsilon = absl::GetFlag(FLAGS_clip_epsilon);

  if (input_dir.empty() == input_file_list.empty()) {
    LOG(FATAL) << "Exactly one of --input_dir and --input_file_list is required.";
  }
  if (output_dir.empty()) LOG(FATAL) << "--output_dir is required.";
  if (!input_dir.empty() &&
      (!fs::exists(input_dir) || !fs::is_directory(input_dir))) {
    LOG(FATAL) << "Input directory does not exist: " << input_dir.string();
  }
  if (!input_file_list.empty() && !filename_regex.empty()) {
    LOG(FATAL) << "--filename_regex cannot be combined with --input_file_list.";
  }
  if (clip_epsilon <= 0.0 || clip_epsilon >= 1.0) {
    LOG(FATAL) << "--clip_epsilon must be in (0, 1).";
  }
  if (bootstrap_replicates < 0) {
    LOG(FATAL) << "--bootstrap_replicates must be non-negative.";
  }
  if (bootstrap_bandwidth <= 0.0 || bootstrap_bandwidth >= 1.0) {
    LOG(FATAL) << "--bootstrap_bandwidth must be in (0, 1).";
  }

  std::vector<fs::path> files = input_file_list.empty()
                                    ? CollectGzipFiles(input_dir, max_games,
                                                       filename_regex)
                                    : ReadGzipFileList(input_file_list, max_games);
  if (files.empty()) {
    LOG(FATAL) << "No matching .gz files found in " << input_dir;
  }
  LOG(INFO) << "Collected " << files.size() << " chronologically sorted files.";

  const size_t worker_count = WorkerCount(files.size(), num_threads);
  std::vector<ThreadRows> thread_rows(worker_count);
  std::vector<GameStats> game_stats(files.size());
  std::atomic<size_t> next_index(0);
  std::atomic<uint64_t> processed(0);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  for (size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&, worker]() {
      while (true) {
        const size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
        if (index >= files.size()) break;
        ProcessGame(files[index], static_cast<uint32_t>(index),
                    StreamFromFilename(files[index]), clip_epsilon,
                    &thread_rows[worker], &game_stats[index]);
        const uint64_t done =
            processed.fetch_add(1, std::memory_order_relaxed) + 1;
        if (done % 1000 == 0) {
          LOG(INFO) << "Processed " << done << " game files.";
        }
      }
    });
  }
  for (auto& worker : workers) worker.join();

  size_t total_rows = 0;
  size_t total_joint_resource_rows = 0;
  for (const ThreadRows& rows : thread_rows) total_rows += rows.rows.size();
  for (const ThreadRows& rows : thread_rows) {
    total_joint_resource_rows += rows.joint_resource_rows.size();
  }
  std::vector<Row> rows;
  rows.reserve(total_rows);
  std::vector<JointResourceRow> joint_resource_rows;
  joint_resource_rows.reserve(total_joint_resource_rows);
  for (ThreadRows& part : thread_rows) {
    rows.insert(rows.end(), part.rows.begin(), part.rows.end());
    joint_resource_rows.insert(joint_resource_rows.end(),
                               part.joint_resource_rows.begin(),
                               part.joint_resource_rows.end());
  }
  LOG(INFO) << "Collected " << rows.size() << " analysis rows.";
  if (write_joint_resource_rows) {
    const fs::path joint_output_dir =
        split_stream_outputs ? output_dir / "all" : output_dir;
    WriteJointResourceRows(joint_output_dir, joint_resource_rows);
    LOG(INFO) << "Wrote " << joint_resource_rows.size()
              << " joint resource rows.";
  }

  std::vector<uint32_t> all_game_indices(files.size());
  std::iota(all_game_indices.begin(), all_game_indices.end(), 0);
  if (split_stream_outputs) {
    WriteOutputs(output_dir / "all", input_dir, input_file_list, filename_regex,
                 files, game_stats, all_game_indices, rows,
                 bootstrap_replicates, bootstrap_seed, bootstrap_bandwidth,
                 write_bootstrap_moments);
    for (uint8_t stream : {1, 2}) {
      OutputSubset subset = SelectStream(files, game_stats, rows, stream);
      if (subset.files.empty()) {
        LOG(FATAL) << "No p" << static_cast<int>(stream)
                   << " files found while --split_stream_outputs=true.";
      }
      WriteOutputs(output_dir / (stream == 1 ? "p1" : "p2"), input_dir,
                   input_file_list, filename_regex, subset.files,
                   subset.game_stats, subset.game_indices, subset.rows, 0,
                   bootstrap_seed, bootstrap_bandwidth, false);
    }
  } else {
    WriteOutputs(output_dir, input_dir, input_file_list, filename_regex, files,
                 game_stats, all_game_indices, rows, bootstrap_replicates,
                 bootstrap_seed, bootstrap_bandwidth,
                 write_bootstrap_moments);
  }
  LOG(INFO) << "Wrote outputs under " << output_dir.string();
  return 0;
}
