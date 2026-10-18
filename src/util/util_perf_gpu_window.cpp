#include "util_perf_gpu_window.hpp"

#include "util_env.hpp"

#include <algorithm>
#include <limits>
#include <mutex>

namespace dxmt {
namespace perf_gpu_window {

namespace {

std::optional<bool> g_enabled;
std::mutex g_mutex;
unsigned g_count = 0;
double g_sum_ms = 0.0;
double g_min_ms = std::numeric_limits<double>::infinity();
double g_max_ms = 0.0;

static bool
enabled() {
  if (!g_enabled.has_value())
    g_enabled = (env::getEnvVar("DXMT_PERF_SESSION") == "1");
  return *g_enabled;
}

} // namespace

void
recordFinishedCommandBuffer(uint64_t gpu_start_ns, uint64_t gpu_end_ns) {
  if (!enabled())
    return;
  if (gpu_start_ns == 0 || gpu_end_ns == 0 || gpu_end_ns <= gpu_start_ns)
    return;

  const double gpu_ms = static_cast<double>(gpu_end_ns - gpu_start_ns) / 1'000'000.0;
  std::lock_guard lock(g_mutex);
  g_count++;
  g_sum_ms += gpu_ms;
  g_min_ms = std::min(g_min_ms, gpu_ms);
  g_max_ms = std::max(g_max_ms, gpu_ms);
}

std::optional<Snapshot>
takeWindowSnapshotAndReset() {
  if (!enabled())
    return std::nullopt;

  std::lock_guard lock(g_mutex);
  if (g_count == 0) {
    return Snapshot{};
  }

  Snapshot snap;
  snap.command_buffers = g_count;
  snap.gpu_ms_avg = g_sum_ms / static_cast<double>(g_count);
  snap.gpu_ms_min = g_min_ms;
  snap.gpu_ms_max = g_max_ms;

  g_count = 0;
  g_sum_ms = 0.0;
  g_min_ms = std::numeric_limits<double>::infinity();
  g_max_ms = 0.0;

  return snap;
}

} // namespace perf_gpu_window
} // namespace dxmt
