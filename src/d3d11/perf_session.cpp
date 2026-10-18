#include "perf_session.hpp"

#include "log/log.hpp"
#include "util_env.hpp"
#include "util_perf_gpu_window.hpp"
#include "util_string.hpp"
#include "util_win32_compat.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>

namespace dxmt {
namespace perf_session {

namespace {

using steady_clock = std::chrono::steady_clock;

constexpr const char *kSchema = "dxmt-perf-session-v1.1";
constexpr unsigned kSummaryPresentInterval = 60;
constexpr double kSummaryWallSeconds = 1.0;

std::optional<bool> g_enabled;
std::mutex g_mutex;
bool g_header_written = false;
std::ofstream g_out;
steady_clock::time_point g_session_start{};
steady_clock::time_point g_last_present{};
steady_clock::time_point g_window_start{};
bool g_have_last_present = false;

unsigned g_window_presents = 0;
double g_frame_ms_sum = 0.0;
double g_frame_ms_min = std::numeric_limits<double>::infinity();
double g_frame_ms_max = 0.0;
double g_fps_inst_min = std::numeric_limits<double>::infinity();
double g_fps_inst_max = 0.0;
unsigned g_window_frame_samples = 0;

static double
durationMs(steady_clock::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

static double
chronoDurationToMs(clock::duration d) {
  return d.count() / 1000000.0;
}

static const char *
scalerTypeName(ScalerType type) {
  switch (type) {
  case ScalerType::Spatial:
    return "Spatial";
  case ScalerType::Temporal:
    return "Temporal";
  default:
    return "None";
  }
}

static std::string
resolveOutputPath() {
  const std::string explicit_path = env::getEnvVar("DXMT_PERF_SESSION_PATH");
  if (!explicit_path.empty())
    return explicit_path;

  std::string path = env::getEnvVar("DXMT_LOG_PATH");
  if (path == "none" || path.empty())
    return "/tmp/dxmt-perf-session.jsonl";

  if (*path.rbegin() != '/')
    path += '/';
  path += "dxmt-perf-session.jsonl";
  return path;
}

static void
ensureFileOpenLocked() {
  if (g_out.is_open())
    return;

  const std::string path = resolveOutputPath();
  g_out.open(str::topath(path.c_str()).c_str(), std::ios::out | std::ios::app);
  if (!g_out.is_open()) {
    WARN("perf-session: failed to open ", path);
    return;
  }

  g_session_start = steady_clock::now();
  if (!g_header_written) {
    const auto t0 = std::chrono::duration_cast<std::chrono::milliseconds>(
                        g_session_start.time_since_epoch())
                        .count();
    g_out << std::format(
        R"({{"type":"header","schema":"{}","path":"{}","exe":"{}","t_ms":{}}})",
        kSchema, path, env::getExeBaseName(), t0
    ) << '\n';
    g_header_written = true;
  }
}

static void
resetWindowLocked(steady_clock::time_point now) {
  g_window_start = now;
  g_window_presents = 0;
  g_frame_ms_sum = 0.0;
  g_frame_ms_min = std::numeric_limits<double>::infinity();
  g_frame_ms_max = 0.0;
  g_fps_inst_min = std::numeric_limits<double>::infinity();
  g_fps_inst_max = 0.0;
  g_window_frame_samples = 0;
}

static void
emitSummaryLocked(
    const PresentCounters &counters, const FrameStatisticsContainer &statistics,
    uint64_t queue_frame_count_after_boundary, steady_clock::time_point now
) {
  if (!g_out.is_open())
    return;

  const double window_s = durationMs(now - g_window_start) / 1000.0;
  const double fps_avg =
      (window_s > 0.0 && g_window_presents > 0)
          ? (static_cast<double>(g_window_presents) / window_s)
          : 0.0;
  const double fps_min =
      g_window_frame_samples > 0 ? g_fps_inst_min : 0.0;
  const double fps_max =
      g_window_frame_samples > 0 ? g_fps_inst_max : 0.0;
  const double frame_ms_avg =
      g_window_frame_samples > 0 ? (g_frame_ms_sum / g_window_frame_samples) : 0.0;
  const double frame_ms_min =
      g_window_frame_samples > 0 ? g_frame_ms_min : 0.0;
  const double frame_ms_max =
      g_window_frame_samples > 0 ? g_frame_ms_max : 0.0;

  const auto &average = statistics.average();
  const uint64_t prev_frame =
      queue_frame_count_after_boundary > 0 ? queue_frame_count_after_boundary - 1 : 0;
  const auto &frame = statistics.at(prev_frame);
  const auto &info = frame.last_scaler_info;

  const auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch())
                        .count();

  const double encode_prepare_ms = chronoDurationToMs(average.encode_prepare_interval);
  const double encode_flush_ms = chronoDurationToMs(average.encode_flush_interval);
  const double commit_ms = chronoDurationToMs(average.commit_interval);
  const double present_latency_ms =
      chronoDurationToMs(average.present_latency_interval);

  const auto gpu_snap = perf_gpu_window::takeWindowSnapshotAndReset();
  const unsigned gpu_cb_finished = gpu_snap ? gpu_snap->command_buffers : 0;
  const bool have_gpu_ms = gpu_snap && gpu_snap->command_buffers > 0;
  const double gpu_ms_per_wall_s =
      (window_s > 0.0 && have_gpu_ms)
          ? (gpu_snap->gpu_ms_avg * static_cast<double>(gpu_cb_finished) / window_s)
          : 0.0;

  const auto emit_t0 = steady_clock::now();

  if (have_gpu_ms) {
    g_out << std::format(
        R"({{"type":"summary","t_ms":{},"presents":{},"rendered":{},"presented":{},"unified_encode":{},"unified_fi":{},"fps_avg":{:.3f},"fps_min":{:.3f},"fps_max":{:.3f},"frame_ms_avg":{:.3f},"frame_ms_min":{:.3f},"frame_ms_max":{:.3f},"encode_prepare_ms_avg":{:.3f},"encode_flush_ms_avg":{:.3f},"commit_ms_avg":{:.3f},"present_latency_ms_avg":{:.3f},"scaler":{{"type":"{}","in_w":{},"in_h":{},"out_w":{},"out_h":{}}},"gpu_cb_finished":{},"gpu_ms_avg":{:.3f},"gpu_ms_min":{:.3f},"gpu_ms_max":{:.3f},"gpu_ms_per_wall_s":{:.3f},"gpu_ms_note":"Metal_GPUStartTime_to_GPUEndTime_per_finished_command_buffer_in_window","cpu_pct":null,"gpu_pct":null,"gpu_pct_note":"not_implemented_no_fabricated_gpu_pct"}})",
        t_ms, counters.presents, counters.rendered, counters.presented,
        counters.unified_encode, counters.unified_fi, fps_avg, fps_min, fps_max,
        frame_ms_avg, frame_ms_min, frame_ms_max, encode_prepare_ms, encode_flush_ms,
        commit_ms, present_latency_ms, scalerTypeName(info.type), info.input_width,
        info.input_height, info.output_width, info.output_height, gpu_cb_finished,
        gpu_snap->gpu_ms_avg, gpu_snap->gpu_ms_min, gpu_snap->gpu_ms_max, gpu_ms_per_wall_s
    ) << '\n';
  } else {
    g_out << std::format(
        R"({{"type":"summary","t_ms":{},"presents":{},"rendered":{},"presented":{},"unified_encode":{},"unified_fi":{},"fps_avg":{:.3f},"fps_min":{:.3f},"fps_max":{:.3f},"frame_ms_avg":{:.3f},"frame_ms_min":{:.3f},"frame_ms_max":{:.3f},"encode_prepare_ms_avg":{:.3f},"encode_flush_ms_avg":{:.3f},"commit_ms_avg":{:.3f},"present_latency_ms_avg":{:.3f},"scaler":{{"type":"{}","in_w":{},"in_h":{},"out_w":{},"out_h":{}}},"gpu_cb_finished":{},"gpu_ms_avg":null,"gpu_ms_min":null,"gpu_ms_max":null,"gpu_ms_per_wall_s":null,"gpu_ms_note":"Metal_GPUStartTime_to_GPUEndTime_per_finished_command_buffer_in_window","cpu_pct":null,"gpu_pct":null,"gpu_pct_note":"not_implemented_no_fabricated_gpu_pct"}})",
        t_ms, counters.presents, counters.rendered, counters.presented,
        counters.unified_encode, counters.unified_fi, fps_avg, fps_min, fps_max,
        frame_ms_avg, frame_ms_min, frame_ms_max, encode_prepare_ms, encode_flush_ms,
        commit_ms, present_latency_ms, scalerTypeName(info.type), info.input_width,
        info.input_height, info.output_width, info.output_height, gpu_cb_finished
    ) << '\n';
  }
  g_out.flush();

  const auto perf_log_us = std::chrono::duration_cast<std::chrono::microseconds>(
                               steady_clock::now() - emit_t0)
                               .count();

  if (have_gpu_ms) {
    WARN(
        "perf-session: fps_avg=", std::format("{:.1f}", fps_avg),
        " fps_min=", std::format("{:.1f}", fps_min),
        " fps_max=", std::format("{:.1f}", fps_max),
        " frame_ms=", std::format("{:.1f}", frame_ms_avg),
        " encode=", std::format("{:.1f}+{:.1f}", encode_prepare_ms, encode_flush_ms),
        " commit=", std::format("{:.1f}", commit_ms),
        " present_lat=", std::format("{:.1f}", present_latency_ms),
        " gpu_ms=", std::format("{:.1f}", gpu_snap->gpu_ms_avg),
        " gpu_cb=", gpu_cb_finished,
        " perf_log_us=", perf_log_us
    );
  } else {
    WARN(
        "perf-session: fps_avg=", std::format("{:.1f}", fps_avg),
        " fps_min=", std::format("{:.1f}", fps_min),
        " fps_max=", std::format("{:.1f}", fps_max),
        " frame_ms=", std::format("{:.1f}", frame_ms_avg),
        " encode=", std::format("{:.1f}+{:.1f}", encode_prepare_ms, encode_flush_ms),
        " commit=", std::format("{:.1f}", commit_ms),
        " present_lat=", std::format("{:.1f}", present_latency_ms),
        " perf_log_us=", perf_log_us
    );
  }

  resetWindowLocked(now);
}

} // namespace

bool
enabled() {
  if (!g_enabled.has_value())
    g_enabled = (env::getEnvVar("DXMT_PERF_SESSION") == "1");
  return *g_enabled;
}

void
onPresent(
    const PresentCounters &counters, const FrameStatisticsContainer &statistics,
    uint64_t queue_frame_count_after_boundary
) {
  if (!enabled())
    return;

  std::lock_guard lock(g_mutex);
  ensureFileOpenLocked();
  if (!g_out.is_open())
    return;

  const auto now = steady_clock::now();
  if (!g_have_last_present) {
    g_last_present = now;
    g_have_last_present = true;
    resetWindowLocked(now);
  } else {
    const double frame_ms = durationMs(now - g_last_present);
    g_last_present = now;
    if (frame_ms > 0.0) {
      const double inst_fps = 1000.0 / frame_ms;
      g_frame_ms_sum += frame_ms;
      g_frame_ms_min = std::min(g_frame_ms_min, frame_ms);
      g_frame_ms_max = std::max(g_frame_ms_max, frame_ms);
      g_fps_inst_min = std::min(g_fps_inst_min, inst_fps);
      g_fps_inst_max = std::max(g_fps_inst_max, inst_fps);
      g_window_frame_samples++;
    }
  }

  g_window_presents++;

  const double window_s = durationMs(now - g_window_start) / 1000.0;
  const bool time_window = window_s >= kSummaryWallSeconds;
  const bool count_window =
      g_window_presents >= kSummaryPresentInterval && g_window_frame_samples > 0;
  if (time_window || count_window)
    emitSummaryLocked(
        counters, statistics, queue_frame_count_after_boundary, now
    );
}

} // namespace perf_session
} // namespace dxmt
