#pragma once

#include "dxmt_statistics.hpp"
#include <cstdint>

namespace dxmt {
namespace perf_session {

/** True when DXMT_PERF_SESSION=1 (cached; off by default). */
bool enabled();

struct PresentCounters {
  uint64_t presents = 0;
  uint64_t rendered = 0;
  uint64_t presented = 0;
  uint64_t unified_encode = 0;
  uint64_t unified_fi = 0;
};

/**
 * Present-to-present wall clock sample + DXMT frame statistics.
 * Call once per successful Present1 after PresentBoundary (statistics computed).
 */
void onPresent(
    const PresentCounters &counters, const FrameStatisticsContainer &statistics,
    uint64_t queue_frame_count_after_boundary
);

} // namespace perf_session
} // namespace dxmt
