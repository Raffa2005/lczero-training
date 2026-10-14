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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "trainingdata/trainingdata_v6.h"

ABSL_FLAG(std::string, input_dir, "",
          "Directory containing raw self-play .gz game files.");
ABSL_FLAG(std::string, output_dir, "",
          "Directory where manifest and fit CSV files will be written.");
ABSL_FLAG(int, window_games, 10000,
          "Number of chronological games per non-overlapping window.");
ABSL_FLAG(int, min_window_games, 1,
          "Drop a final partial window if it has fewer games than this.");
ABSL_FLAG(int, bootstrap_replicates, 1000,
          "Game-level bootstrap replicates per window/bin. Use 0 to skip.");
ABSL_FLAG(int, num_threads, 0,
          "Worker threads. Defaults to hardware concurrency.");
ABSL_FLAG(double, ridge_lambda, 1e-6,
          "Ridge penalty added to non-intercept diagonal entries. Use 0 for "
          "plain OLS; bins with constant columns may then be singular.");
ABSL_FLAG(double, clip_epsilon, 1e-6,
          "Clip root_q to [-1+epsilon, 1-epsilon] before atanh.");
ABSL_FLAG(uint64_t, seed, 20260623,
          "Base RNG seed for deterministic bootstrap samples.");
ABSL_FLAG(int, max_games, 0,
          "Optional smoke-test cap after chronological sorting. 0 means all.");
ABSL_FLAG(bool, allow_non_slurm, false,
          "Allow running outside a SLURM allocation. Keep false for full runs.");

namespace {

namespace fs = std::filesystem;

using ::lczero::V6TrainingData;

constexpr int kFeatures = 7;
constexpr int kRatioCount = 5;

enum Feature {
  kIntercept = 0,
  kPawn = 1,
  kKnight = 2,
  kBishop = 3,
  kRook = 4,
  kQueen = 5,
  kAbility = 6,
};

enum Bin {
  kGlobal = 0,
  kAnyAvailable = 1,
  kBothAvailable = 2,
  kOneSided = 3,
  kNobodyAvailable = 4,
  kBinCount = 5,
};

const std::array<const char*, kBinCount> kBinNames = {
    "global", "any_available", "both_available", "one_sided",
    "nobody_available"};

const std::array<const char*, kFeatures> kFeatureNames = {
    "intercept", "P", "N", "B", "R", "Q", "A"};

const std::array<const char*, kRatioCount> kRatioNames = {
    "N_over_P", "B_over_P", "R_over_P", "Q_over_P", "A_over_P"};

struct Summary {
  uint64_t n = 0;
  double xtx[kFeatures][kFeatures] = {};
  double xty[kFeatures] = {};
  double yy = 0.0;

  void AddRow(const std::array<double, kFeatures>& x, double y) {
    ++n;
    for (int i = 0; i < kFeatures; ++i) {
      xty[i] += x[i] * y;
      for (int j = 0; j < kFeatures; ++j) {
        xtx[i][j] += x[i] * x[j];
      }
    }
    yy += y * y;
  }

  void AddSummary(const Summary& other) {
    n += other.n;
    for (int i = 0; i < kFeatures; ++i) {
      xty[i] += other.xty[i];
      for (int j = 0; j < kFeatures; ++j) {
        xtx[i][j] += other.xtx[i][j];
      }
    }
    yy += other.yy;
  }
};

struct GameRecord {
  fs::path path;
  std::string filename;
  uintmax_t size_bytes = 0;
  bool ok = false;
  std::string error;
  uint64_t frames = 0;
  uint64_t skipped_rows = 0;
  uint64_t corrected_mid_rows = 0;
  std::array<Summary, kBinCount> bins;
};

struct Window {
  size_t index = 0;
  size_t start = 0;
  size_t end = 0;
};

struct FitResult {
  bool ok = false;
  std::string status = "not_fit";
  uint64_t positions = 0;
  std::array<double, kFeatures> w = {};
  std::array<double, kRatioCount> ratios = {};
  double sse = std::numeric_limits<double>::quiet_NaN();
  double rmse = std::numeric_limits<double>::quiet_NaN();
};

struct CiResult {
  size_t window_index = 0;
  Bin bin = kGlobal;
  std::string metric;
  double estimate = std::numeric_limits<double>::quiet_NaN();
  double p025 = std::numeric_limits<double>::quiet_NaN();
  double p500 = std::numeric_limits<double>::quiet_NaN();
  double p975 = std::numeric_limits<double>::quiet_NaN();
  int valid_replicates = 0;
};

std::string CsvEscape(const std::string& value) {
  bool needs_quotes = false;
  for (char c : value) {
    if (c == ',' || c == '"' || c == '\n' || c == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) return value;
  std::string out = "\"";
  for (char c : value) {
    if (c == '"') out += '"';
    out += c;
  }
  out += '"';
  return out;
}

int PopCount(uint64_t x) { return static_cast<int>(std::popcount(x)); }

double ClipRootQ(double q, double epsilon) {
  const double lo = -1.0 + epsilon;
  const double hi = 1.0 - epsilon;
  return std::min(hi, std::max(lo, q));
}

std::array<double, kFeatures> ExtractFeatures(const V6TrainingData& frame) {
  std::array<double, kFeatures> x = {};
  x[kIntercept] = 1.0;
  x[kPawn] = PopCount(frame.planes[0]) - PopCount(frame.planes[6]);
  x[kKnight] = PopCount(frame.planes[1]) - PopCount(frame.planes[7]);
  x[kBishop] = PopCount(frame.planes[2]) - PopCount(frame.planes[8]);
  x[kRook] = PopCount(frame.planes[3]) - PopCount(frame.planes[9]);
  x[kQueen] = PopCount(frame.planes[4]) - PopCount(frame.planes[10]);
  x[kAbility] = (frame.our_doublemove_available ? 1.0 : 0.0) -
                (frame.their_doublemove_available ? 1.0 : 0.0);
  return x;
}

void AddFrameToBins(const V6TrainingData& frame, double y, GameRecord* record) {
  const bool our = frame.our_doublemove_available != 0;
  const bool their = frame.their_doublemove_available != 0;
  const std::array<double, kFeatures> x = ExtractFeatures(frame);

  record->bins[kGlobal].AddRow(x, y);
  if (our || their) record->bins[kAnyAvailable].AddRow(x, y);
  if (our && their) record->bins[kBothAvailable].AddRow(x, y);
  if (our != their) record->bins[kOneSided].AddRow(x, y);
  if (!our && !their) record->bins[kNobodyAvailable].AddRow(x, y);
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

void ProcessGame(double clip_epsilon, GameRecord* record) {
  gzFile file = gzopen(record->path.string().c_str(), "rb");
  if (file == nullptr) {
    record->error = "failed to open gzip file";
    return;
  }

  while (true) {
    V6TrainingData frame;
    std::string read_error;
    const bool got_frame = ReadOneFrame(file, &frame, &read_error);
    if (!got_frame) {
      if (!read_error.empty()) record->error = read_error;
      break;
    }

    ++record->frames;
    double q = frame.root_q;
    if (!std::isfinite(q)) {
      ++record->skipped_rows;
      continue;
    }
    if (frame.is_mid_doublemove) {
      q = -q;
      ++record->corrected_mid_rows;
    }
    const double y = std::atanh(ClipRootQ(q, clip_epsilon));
    if (!std::isfinite(y)) {
      ++record->skipped_rows;
      continue;
    }
    AddFrameToBins(frame, y, record);
  }

  const int close_result = gzclose(file);
  if (close_result != Z_OK && record->error.empty()) {
    record->error = "gzip close failed";
  }
  record->ok = record->error.empty();
}

std::vector<fs::path> CollectGzipFiles(const fs::path& input_dir,
                                       int max_games) {
  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (!entry.is_regular_file()) continue;
    const fs::path& path = entry.path();
    if (path.extension() == ".gz") files.push_back(path);
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

size_t WorkerCount(size_t item_count, int flag_value) {
  if (item_count == 0) return 0;
  size_t count = 0;
  if (flag_value > 0) {
    count = static_cast<size_t>(flag_value);
  } else {
    const unsigned int hw = std::thread::hardware_concurrency();
    count = hw > 0 ? static_cast<size_t>(hw) : 1;
  }
  return std::max<size_t>(1, std::min(count, item_count));
}

void ExtractAllGames(std::vector<GameRecord>* records, int num_threads,
                     double clip_epsilon) {
  std::atomic<size_t> next_index(0);
  std::atomic<uint64_t> processed(0);
  const size_t worker_count = WorkerCount(records->size(), num_threads);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  for (size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([records, &next_index, &processed, clip_epsilon]() {
      while (true) {
        const size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
        if (index >= records->size()) break;
        ProcessGame(clip_epsilon, &(*records)[index]);
        const uint64_t done =
            processed.fetch_add(1, std::memory_order_relaxed) + 1;
        if (done % 1000 == 0) {
          LOG(INFO) << "Extracted " << done << " game files.";
        }
      }
    });
  }
  for (auto& worker : workers) worker.join();
}

std::vector<Window> BuildWindows(size_t game_count, int window_games,
                                 int min_window_games) {
  std::vector<Window> windows;
  if (window_games <= 0) return windows;
  size_t index = 0;
  for (size_t start = 0; start < game_count;
       start += static_cast<size_t>(window_games)) {
    const size_t end =
        std::min(game_count, start + static_cast<size_t>(window_games));
    if (end - start < static_cast<size_t>(std::max(1, min_window_games))) {
      continue;
    }
    windows.push_back(Window{index++, start, end});
  }
  return windows;
}

Summary SumWindowBin(const std::vector<GameRecord>& records, const Window& w,
                     Bin bin) {
  Summary sum;
  for (size_t i = w.start; i < w.end; ++i) {
    sum.AddSummary(records[i].bins[bin]);
  }
  return sum;
}

bool SolveLinearSystem(double matrix[kFeatures][kFeatures],
                       double rhs[kFeatures],
                       std::array<double, kFeatures>* solution) {
  constexpr double kPivotEps = 1e-12;
  for (int col = 0; col < kFeatures; ++col) {
    int pivot = col;
    double best = std::fabs(matrix[col][col]);
    for (int row = col + 1; row < kFeatures; ++row) {
      const double candidate = std::fabs(matrix[row][col]);
      if (candidate > best) {
        best = candidate;
        pivot = row;
      }
    }
    if (best < kPivotEps) return false;
    if (pivot != col) {
      for (int j = col; j < kFeatures; ++j) {
        std::swap(matrix[col][j], matrix[pivot][j]);
      }
      std::swap(rhs[col], rhs[pivot]);
    }

    const double pivot_value = matrix[col][col];
    for (int j = col; j < kFeatures; ++j) matrix[col][j] /= pivot_value;
    rhs[col] /= pivot_value;

    for (int row = 0; row < kFeatures; ++row) {
      if (row == col) continue;
      const double factor = matrix[row][col];
      if (factor == 0.0) continue;
      for (int j = col; j < kFeatures; ++j) {
        matrix[row][j] -= factor * matrix[col][j];
      }
      rhs[row] -= factor * rhs[col];
    }
  }

  for (int i = 0; i < kFeatures; ++i) (*solution)[i] = rhs[i];
  return true;
}

FitResult FitSummary(const Summary& summary, double ridge_lambda) {
  FitResult result;
  result.positions = summary.n;
  if (summary.n == 0) {
    result.status = "empty";
    return result;
  }

  double matrix[kFeatures][kFeatures] = {};
  double rhs[kFeatures] = {};
  for (int i = 0; i < kFeatures; ++i) {
    rhs[i] = summary.xty[i];
    for (int j = 0; j < kFeatures; ++j) matrix[i][j] = summary.xtx[i][j];
  }
  for (int i = 1; i < kFeatures; ++i) matrix[i][i] += ridge_lambda;

  if (!SolveLinearSystem(matrix, rhs, &result.w)) {
    result.status = "singular";
    return result;
  }

  constexpr double kDenomEps = 1e-12;
  const double pawn = result.w[kPawn];
  result.ratios.fill(std::numeric_limits<double>::quiet_NaN());
  if (std::fabs(pawn) >= kDenomEps) {
    result.ratios[0] = result.w[kKnight] / pawn;
    result.ratios[1] = result.w[kBishop] / pawn;
    result.ratios[2] = result.w[kRook] / pawn;
    result.ratios[3] = result.w[kQueen] / pawn;
    result.ratios[4] = result.w[kAbility] / pawn;
  }

  double wt_x_y = 0.0;
  double wt_x_x_w = 0.0;
  for (int i = 0; i < kFeatures; ++i) {
    wt_x_y += result.w[i] * summary.xty[i];
    for (int j = 0; j < kFeatures; ++j) {
      wt_x_x_w += result.w[i] * summary.xtx[i][j] * result.w[j];
    }
  }
  result.sse = std::max(0.0, summary.yy - 2.0 * wt_x_y + wt_x_x_w);
  result.rmse = std::sqrt(result.sse / static_cast<double>(summary.n));
  result.ok = true;
  result.status = "ok";
  return result;
}

std::vector<std::pair<std::string, double>> FitMetrics(
    const FitResult& fit) {
  std::vector<std::pair<std::string, double>> metrics;
  metrics.reserve(kFeatures + kRatioCount);
  for (int i = 0; i < kFeatures; ++i) {
    metrics.push_back(
        {std::string("coef_") + kFeatureNames[i], fit.w[i]});
  }
  for (int i = 0; i < kRatioCount; ++i) {
    metrics.push_back({kRatioNames[i], fit.ratios[i]});
  }
  return metrics;
}

double Percentile(std::vector<double>* values, double p) {
  if (values->empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values->begin(), values->end());
  const size_t index =
      static_cast<size_t>(std::floor(p * static_cast<double>(values->size() - 1)));
  return (*values)[index];
}

std::vector<CiResult> BootstrapWindowBin(const std::vector<GameRecord>& records,
                                         const Window& window, Bin bin,
                                         const FitResult& estimate,
                                         int replicates,
                                         uint64_t seed,
                                         double ridge_lambda) {
  std::vector<CiResult> output;
  const std::vector<std::pair<std::string, double>> estimate_metrics =
      FitMetrics(estimate);
  output.reserve(estimate_metrics.size());
  if (!estimate.ok || replicates <= 0) {
    for (const auto& [name, value] : estimate_metrics) {
      output.push_back(CiResult{window.index, bin, name, value});
    }
    return output;
  }

  std::vector<std::vector<double>> samples(estimate_metrics.size());
  const size_t games_in_window = window.end - window.start;
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> dist(0, games_in_window - 1);

  for (int rep = 0; rep < replicates; ++rep) {
    Summary sampled;
    for (size_t draw = 0; draw < games_in_window; ++draw) {
      const size_t game = window.start + dist(rng);
      sampled.AddSummary(records[game].bins[bin]);
    }
    const FitResult fit = FitSummary(sampled, ridge_lambda);
    if (!fit.ok) continue;
    const auto metrics = FitMetrics(fit);
    for (size_t i = 0; i < metrics.size(); ++i) {
      if (std::isfinite(metrics[i].second)) {
        samples[i].push_back(metrics[i].second);
      }
    }
  }

  for (size_t i = 0; i < estimate_metrics.size(); ++i) {
    std::vector<double> values = std::move(samples[i]);
    CiResult ci;
    ci.window_index = window.index;
    ci.bin = bin;
    ci.metric = estimate_metrics[i].first;
    ci.estimate = estimate_metrics[i].second;
    ci.valid_replicates = static_cast<int>(values.size());
    ci.p025 = Percentile(&values, 0.025);
    ci.p500 = Percentile(&values, 0.500);
    ci.p975 = Percentile(&values, 0.975);
    output.push_back(ci);
  }
  return output;
}

struct TaskResult {
  size_t window_index = 0;
  Bin bin = kGlobal;
  FitResult fit;
  std::vector<CiResult> cis;
};

uint64_t MixSeed(uint64_t seed, uint64_t task_index) {
  uint64_t x = seed + 0x9e3779b97f4a7c15ULL * (task_index + 1);
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

std::vector<TaskResult> FitAllTasks(const std::vector<GameRecord>& records,
                                    const std::vector<Window>& windows,
                                    int bootstrap_replicates, int num_threads,
                                    uint64_t seed, double ridge_lambda) {
  const size_t task_count = windows.size() * kBinCount;
  std::vector<TaskResult> results(task_count);
  std::atomic<size_t> next_task(0);
  std::atomic<uint64_t> done(0);
  const size_t worker_count = WorkerCount(task_count, num_threads);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  for (size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&]() {
      while (true) {
        const size_t task = next_task.fetch_add(1, std::memory_order_relaxed);
        if (task >= task_count) break;
        const size_t window_index = task / kBinCount;
        const Bin bin = static_cast<Bin>(task % kBinCount);
        const Summary summary = SumWindowBin(records, windows[window_index], bin);
        FitResult fit = FitSummary(summary, ridge_lambda);
        std::vector<CiResult> cis = BootstrapWindowBin(
            records, windows[window_index], bin, fit, bootstrap_replicates,
            MixSeed(seed, task), ridge_lambda);
        results[task] =
            TaskResult{windows[window_index].index, bin, std::move(fit),
                       std::move(cis)};
        const uint64_t finished =
            done.fetch_add(1, std::memory_order_relaxed) + 1;
        if (finished % 10 == 0) {
          LOG(INFO) << "Finished " << finished << " fit/bootstrap tasks.";
        }
      }
    });
  }
  for (auto& worker : workers) worker.join();
  return results;
}

void WriteMetadata(const fs::path& output_dir, const fs::path& input_dir,
                   const std::vector<GameRecord>& records,
                   const std::vector<Window>& windows, int window_games,
                   int bootstrap_replicates, int num_threads,
                   double ridge_lambda, double clip_epsilon, uint64_t seed) {
  std::ofstream out(output_dir / "metadata.txt", std::ios::out | std::ios::trunc);
  out << "tool=material_value_windows\n";
  out << "input_dir=" << input_dir.string() << "\n";
  out << "num_games=" << records.size() << "\n";
  out << "num_windows=" << windows.size() << "\n";
  out << "window_games=" << window_games << "\n";
  out << "bootstrap_replicates=" << bootstrap_replicates << "\n";
  out << "num_threads=" << num_threads << "\n";
  out << "ridge_lambda=" << std::setprecision(17) << ridge_lambda << "\n";
  out << "clip_epsilon=" << std::setprecision(17) << clip_epsilon << "\n";
  out << "seed=" << seed << "\n";
  if (!records.empty()) {
    out << "first_filename=" << records.front().filename << "\n";
    out << "last_filename=" << records.back().filename << "\n";
  }
  out << "root_q_correction=if is_mid_doublemove then q=-root_q else q=root_q\n";
  out << "features=1,dP,dN,dB,dR,dQ,dA\n";
  out << "dA=our_doublemove_available - their_doublemove_available\n";
}

void WriteManifest(const fs::path& output_dir,
                   const std::vector<GameRecord>& records) {
  std::ofstream out(output_dir / "chunk_manifest.csv",
                    std::ios::out | std::ios::trunc);
  out << "game_index,filename,path,size_bytes,ok,frames,skipped_rows,"
         "corrected_mid_rows";
  for (const char* name : kBinNames) out << ',' << name << "_positions";
  out << ",error\n";

  for (size_t i = 0; i < records.size(); ++i) {
    const GameRecord& r = records[i];
    out << i << ',' << CsvEscape(r.filename) << ','
        << CsvEscape(r.path.string()) << ',' << r.size_bytes << ','
        << (r.ok ? 1 : 0) << ',' << r.frames << ',' << r.skipped_rows << ','
        << r.corrected_mid_rows;
    for (const Summary& bin : r.bins) out << ',' << bin.n;
    out << ',' << CsvEscape(r.error) << '\n';
  }
}

void WriteWindowFits(const fs::path& output_dir,
                     const std::vector<Window>& windows,
                     const std::vector<TaskResult>& task_results,
                     double ridge_lambda) {
  std::ofstream out(output_dir / "window_fits.csv",
                    std::ios::out | std::ios::trunc);
  out << "window_index,start_game,end_game,n_games,bin,positions,status,"
         "ridge_lambda,sse,rmse";
  for (const char* name : kFeatureNames) out << ",coef_" << name;
  for (const char* name : kRatioNames) out << ',' << name;
  out << '\n';

  out << std::setprecision(17);
  for (const TaskResult& result : task_results) {
    const Window& w = windows[result.window_index];
    out << result.window_index << ',' << w.start << ',' << w.end << ','
        << (w.end - w.start) << ',' << kBinNames[result.bin] << ','
        << result.fit.positions << ',' << result.fit.status << ','
        << ridge_lambda << ',' << result.fit.sse << ',' << result.fit.rmse;
    for (double value : result.fit.w) out << ',' << value;
    for (double value : result.fit.ratios) out << ',' << value;
    out << '\n';
  }
}

void WriteBootstrapCis(const fs::path& output_dir,
                       const std::vector<Window>& windows,
                       const std::vector<TaskResult>& task_results,
                       int bootstrap_replicates) {
  std::ofstream out(output_dir / "bootstrap_cis.csv",
                    std::ios::out | std::ios::trunc);
  out << "window_index,start_game,end_game,n_games,bin,metric,estimate,p025,"
         "p50,p975,valid_replicates,requested_replicates\n";
  out << std::setprecision(17);
  for (const TaskResult& result : task_results) {
    const Window& w = windows[result.window_index];
    for (const CiResult& ci : result.cis) {
      out << ci.window_index << ',' << w.start << ',' << w.end << ','
          << (w.end - w.start) << ',' << kBinNames[ci.bin] << ','
          << ci.metric << ',' << ci.estimate << ',' << ci.p025 << ','
          << ci.p500 << ',' << ci.p975 << ',' << ci.valid_replicates << ','
          << bootstrap_replicates << '\n';
    }
  }
}

std::string FormatNumber(double value, int precision) {
  if (!std::isfinite(value)) return "nan";
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

const CiResult* FindMetricCi(const TaskResult& result,
                             const std::string& metric) {
  for (const CiResult& ci : result.cis) {
    if (ci.metric == metric) return &ci;
  }
  return nullptr;
}

void WriteHumanSummary(const fs::path& output_dir, const fs::path& input_dir,
                       const std::vector<GameRecord>& records,
                       const std::vector<Window>& windows,
                       const std::vector<TaskResult>& task_results,
                       int window_games, int bootstrap_replicates,
                       int num_threads, double ridge_lambda) {
  uint64_t total_frames = 0;
  uint64_t skipped_rows = 0;
  uint64_t corrected_mid_rows = 0;
  size_t failed_games = 0;
  for (const GameRecord& record : records) {
    total_frames += record.frames;
    skipped_rows += record.skipped_rows;
    corrected_mid_rows += record.corrected_mid_rows;
    if (!record.ok) ++failed_games;
  }

  std::ofstream out(output_dir / "summary.md", std::ios::out | std::ios::trunc);
  out << "# Material Value Windows Summary\n\n";
  out << "- input_dir: `" << input_dir.string() << "`\n";
  out << "- output_dir: `" << output_dir.string() << "`\n";
  out << "- games: " << records.size() << "\n";
  out << "- windows: " << windows.size() << "\n";
  out << "- window size: " << window_games
      << " chronological games, not positions\n";
  out << "- bootstrap replicates: " << bootstrap_replicates << "\n";
  out << "- worker threads: " << num_threads << "\n";
  out << "- ridge lambda: " << std::setprecision(17) << ridge_lambda << "\n";
  out << "- total frames read: " << total_frames << "\n";
  out << "- skipped rows: " << skipped_rows << "\n";
  out << "- corrected mid-double root rows: " << corrected_mid_rows << "\n";
  out << "- failed game files: " << failed_games << "\n";
  if (!records.empty()) {
    out << "- first file: `" << records.front().filename << "`\n";
    out << "- last file: `" << records.back().filename << "`\n";
  }
  out << "\n";
  out << "Root Q correction: `q = -root_q` for `is_mid_doublemove`, otherwise "
         "`q = root_q`.\n\n";
  out << "Feature vector: `1,dP,dN,dB,dR,dQ,dA`, where "
         "`dA = our_doublemove_available - their_doublemove_available`.\n\n";

  out << "## Fits\n\n";
  out << "| window | games | bin | positions | status | N/P | B/P | R/P | Q/P | "
         "A/P | A/P 95% CI |\n";
  out << "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|\n";
  for (const TaskResult& result : task_results) {
    const Window& window = windows[result.window_index];
    const FitResult& fit = result.fit;
    std::string ability_ci = "n/a";
    const CiResult* ci = FindMetricCi(result, "A_over_P");
    if (ci != nullptr && ci->valid_replicates > 0) {
      ability_ci = "[" + FormatNumber(ci->p025, 3) + ", " +
                   FormatNumber(ci->p975, 3) + "]";
    }
    out << "| " << result.window_index << " | " << window.start << "-"
        << window.end << " | " << kBinNames[result.bin] << " | "
        << fit.positions << " | " << fit.status << " | "
        << FormatNumber(fit.ratios[0], 3) << " | "
        << FormatNumber(fit.ratios[1], 3) << " | "
        << FormatNumber(fit.ratios[2], 3) << " | "
        << FormatNumber(fit.ratios[3], 3) << " | "
        << FormatNumber(fit.ratios[4], 3) << " | " << ability_ci << " |\n";
  }

  out << "\nPrecise coefficients, ratios, and all bootstrap intervals are in "
         "`window_fits.csv` and `bootstrap_cis.csv`.\n";
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
  const fs::path output_dir(absl::GetFlag(FLAGS_output_dir));
  if (input_dir.empty()) LOG(FATAL) << "--input_dir is required.";
  if (output_dir.empty()) LOG(FATAL) << "--output_dir is required.";
  if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
    LOG(FATAL) << "Input directory does not exist: " << input_dir.string();
  }
  fs::create_directories(output_dir);

  const int window_games = absl::GetFlag(FLAGS_window_games);
  const int min_window_games = absl::GetFlag(FLAGS_min_window_games);
  const int bootstrap_replicates = absl::GetFlag(FLAGS_bootstrap_replicates);
  const int num_threads = absl::GetFlag(FLAGS_num_threads);
  const double ridge_lambda = absl::GetFlag(FLAGS_ridge_lambda);
  const double clip_epsilon = absl::GetFlag(FLAGS_clip_epsilon);
  const uint64_t seed = absl::GetFlag(FLAGS_seed);
  const int max_games = absl::GetFlag(FLAGS_max_games);

  if (window_games <= 0) LOG(FATAL) << "--window_games must be positive.";
  if (bootstrap_replicates < 0) {
    LOG(FATAL) << "--bootstrap_replicates must be non-negative.";
  }
  if (clip_epsilon <= 0.0 || clip_epsilon >= 1.0) {
    LOG(FATAL) << "--clip_epsilon must be in (0, 1).";
  }
  if (ridge_lambda < 0.0) LOG(FATAL) << "--ridge_lambda must be non-negative.";

  std::vector<fs::path> files = CollectGzipFiles(input_dir, max_games);
  if (files.empty()) LOG(FATAL) << "No .gz files found in " << input_dir;
  LOG(INFO) << "Collected " << files.size()
            << " chronologically sorted gzip game files.";

  std::vector<GameRecord> records(files.size());
  for (size_t i = 0; i < files.size(); ++i) {
    records[i].path = files[i];
    records[i].filename = files[i].filename().string();
    std::error_code ec;
    records[i].size_bytes = fs::file_size(files[i], ec);
    if (ec) records[i].size_bytes = 0;
  }

  ExtractAllGames(&records, num_threads, clip_epsilon);
  const auto ok_count = std::count_if(records.begin(), records.end(),
                                      [](const GameRecord& r) { return r.ok; });
  LOG(INFO) << "Extraction complete. ok=" << ok_count
            << " failed=" << (records.size() - ok_count);

  const std::vector<Window> windows =
      BuildWindows(records.size(), window_games, min_window_games);
  if (windows.empty()) LOG(FATAL) << "No windows to fit.";
  LOG(INFO) << "Built " << windows.size() << " chronological window(s).";

  const std::vector<TaskResult> task_results =
      FitAllTasks(records, windows, bootstrap_replicates, num_threads, seed,
                  ridge_lambda);

  WriteMetadata(output_dir, input_dir, records, windows, window_games,
                bootstrap_replicates, num_threads, ridge_lambda, clip_epsilon,
                seed);
  WriteManifest(output_dir, records);
  WriteWindowFits(output_dir, windows, task_results, ridge_lambda);
  WriteBootstrapCis(output_dir, windows, task_results, bootstrap_replicates);
  WriteHumanSummary(output_dir, input_dir, records, windows, task_results,
                    window_games, bootstrap_replicates, num_threads,
                    ridge_lambda);

  LOG(INFO) << "Wrote results under " << output_dir.string();
  return 0;
}
