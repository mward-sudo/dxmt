#pragma once

#include <cstdint>
#include <optional>

namespace dxmt {
namespace perf_gpu_window {

struct Snapshot {
  unsigned command_buffers = 0;
  double gpu_ms_avg = 0.0;
  double gpu_ms_min = 0.0;
  double gpu_ms_max = 0.0;
};

/** Record GPU duration for a finished command buffer (nanosecond timestamps). */
void recordFinishedCommandBuffer(uint64_t gpu_start_ns, uint64_t gpu_end_ns);

/** Take accumulated stats for the current perf-session window and reset accumulators. */
std::optional<Snapshot> takeWindowSnapshotAndReset();

} // namespace perf_gpu_window
} // namespace dxmt
