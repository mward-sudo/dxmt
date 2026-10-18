#include "present_depth_motion.hpp"
#include "com/com_pointer.hpp"
#include "d3d11_resource.hpp"
#include "dxmt_buffer.hpp"
#include "log/log.hpp"
#include "config/config.hpp"
#include "util_env.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace dxmt {
namespace present_dm {

namespace {

std::atomic<uint64_t> g_temporal_upscale_count{0};
std::atomic<uint64_t> g_unified_temporal_encode_count{0};
std::atomic<uint64_t> g_unified_fi_present_count{0};
std::atomic<UINT> g_last_unified_present_w{0};
std::atomic<UINT> g_last_unified_present_h{0};

constexpr size_t kRtCensusCap = 32;
constexpr size_t kCamCandidateCap = 8;
constexpr size_t kMvProbeCbReadMax = 4096;
constexpr size_t kMvProbeLogCap = 48;
constexpr size_t kMvProbeLogLen = 512;
constexpr size_t kMvIdentityCap = 8;
constexpr size_t kMvRbMaxSlots = 2;
constexpr size_t kCamNearMissCap = 3;
constexpr size_t kMvRbTileSize = 8;
constexpr size_t kMvRbSamplesPerTile = kMvRbTileSize * kMvRbTileSize;

struct RtCensusEntry {
  DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
  UINT width = 0;
  UINT height = 0;
  UINT array_size = 0;
  UINT samples = 0;
  uint64_t binds = 0;
  uint64_t binds_with_scene_dsv = 0;
  uint64_t binds_no_dsv = 0;
};

struct MvIdentityEntry {
  ID3D11Resource *resource = nullptr;
  uint32_t rid = 0;
  DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
  UINT width = 0;
  UINT height = 0;
  uint32_t slot_mask = 0;
  uint64_t binds = 0;
  uint64_t binds_with_scene_dsv = 0;
  uint64_t binds_no_dsv = 0;
  uint64_t first_seen_present = 0;
  uint64_t last_seen_present = 0;
  uint32_t frames_seen = 0;
  bool rid_reused = false;
  bool valid = false;
};

struct CamScanStats {
  uint64_t draws = 0;
  uint64_t slots_bound = 0;
  uint64_t readable = 0;
  uint64_t gpu_managed = 0;
  uint64_t too_short = 0;
  uint64_t windows = 0;
  uint64_t matched = 0;
  uint64_t near_miss = 0;
};

struct CamNearMiss {
  UINT slot = 0;
  UINT offset = 0;
  bool row_major = true;
  float aspect = 0;
  bool valid = false;
};

struct DepthConvState {
  bool have_clear = false;
  float clear_depth = 0;
  bool have_func = false;
  D3D11_COMPARISON_FUNC depth_func = D3D11_COMPARISON_ALWAYS;
  bool depth_enable = false;
  bool have_vp = false;
  float vp_min = 0;
  float vp_max = 1;
};

struct MvRbPending {
  dxmt::Com<ID3D11Texture2D> staging;
  dxmt::Com<ID3D11Resource> held_src;
  uint32_t rid = 0;
  DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
  UINT src_w = 0;
  UINT src_h = 0;
  uint32_t slot_mask = 0;
  uint64_t copy_present = 0;
  uint32_t map_attempts = 0;
  uint32_t tiles = 0;
  bool active = false;
};

struct MvRbPeriodStats {
  uint64_t copies = 0;
  uint64_t mapped = 0;
  uint64_t still_drawing = 0;
  uint64_t timeout = 0;
  uint64_t probe_us = 0;
};

struct CamCandidate {
  UINT slot = 0;
  UINT offset = 0;
  bool row_major = true;
  bool kind_vp = false;
  float fovy = 0;
  float aspect = 0;
  float near_z = 0;
  float far_z = 0;
  bool reversed = false;
  float jitter_x = 0;
  float jitter_y = 0;
  float prev_m20 = 0;
  float prev_m21 = 0;
  bool has_prev_jitter = false;
  bool stage_ps = false;
  bool valid = false;
};

/** Per-device Present-dm state (tip 15). Key is raw ID3D11Device*; cleared in
 *  clearDevice on teardown — do not AddRef the map key. */
struct DeviceState {
  dxmt::Com<ID3D11Resource> tracked_depth;
  UINT tracked_w = 0;
  UINT tracked_h = 0;
  dxmt::Com<ID3D11Resource> title_motion;
  dxmt::Com<ID3D11Resource> title_depth;
  bool title_motion_valid = false;

  /* Tip 27a MV probe (desc / structural scan only — no resource AddRefs). */
  RtCensusEntry rt_census[kRtCensusCap]{};
  uint32_t rt_census_count = 0;
  uint64_t rt_census_dropped = 0;
  CamCandidate cam_candidates[kCamCandidateCap]{};
  uint32_t cam_candidate_count = 0;
  uint32_t arm_b_sample_gen = 0; /* matches g_arm_b_sample_gen when armed */
  uint32_t arm_b_draws_left = 0;

  MvIdentityEntry mv_identity[kMvIdentityCap]{};
  uint32_t mv_identity_count = 0;
  CamScanStats cam_scan{};
  CamNearMiss cam_near_miss[kCamNearMissCap]{};
  uint32_t cam_near_miss_count = 0;
  DepthConvState depth_conv{};

  MvRbPending mv_rb_pending[kMvRbMaxSlots]{};
  MvRbPeriodStats mv_rb_stats{};
};

/** Tip 27a: true while the upcoming/current Present is an Arm B sample tick. */
std::atomic<bool> g_arm_b_sample_active{false};
std::atomic<uint32_t> g_arm_b_sample_gen{0};
/** Fast-path hint for PreDraw: remaining Arm B scans this sample window. */
std::atomic<uint32_t> g_arm_b_draws_hint{0};
std::atomic<uint64_t> g_mv_probe_present_count{0};
std::atomic<UINT> g_mv_probe_swap_w{0};
std::atomic<UINT> g_mv_probe_swap_h{0};
std::atomic<bool> g_mv_rb_disabled_no_wait{false};
std::atomic<bool> g_mv_rb_no_wait_logged{false};

std::mutex g_lock;
std::unordered_map<ID3D11Device *, DeviceState> g_by_device;

bool
envIsZero(const char *name) {
  return env::getEnvVar(name) == "0";
}

bool
envIsOne(const char *name) {
  return env::getEnvVar(name) == "1";
}

/** Exact size match for MetalFX bind (tip 15) — half/double rejected. */
bool
sizeExact(UINT color_w, UINT color_h, UINT depth_w, UINT depth_h) {
  return depth_w != 0 && depth_h != 0 && color_w != 0 && color_h != 0 &&
         depth_w == color_w && depth_h == color_h;
}

bool
resourceTex2DDesc(ID3D11Resource *resource, D3D11_TEXTURE2D_DESC *desc) {
  if (!resource || !desc)
    return false;
  dxmt::Com<ID3D11Texture2D> tex;
  if (FAILED(resource->QueryInterface(IID_PPV_ARGS(&tex))))
    return false;
  tex->GetDesc(desc);
  return true;
}

uint64_t
texArea(UINT w, UINT h) {
  return uint64_t(w) * uint64_t(h);
}

const char *
prevValidLabel(bool prev_color_valid) {
  return prev_color_valid ? "REAL_COLOR_PREV" : "REAL_COLOR_PREV_MISSING";
}

bool
trackedDepthAllowed() {
  return !envIsZero("DXMT_PRESENT_DM_TRACKED_DEPTH");
}

bool
omDepthAllowed() {
  return !envIsZero("DXMT_PRESENT_DM_OM_DEPTH");
}

bool
mvProbeCbEnabledCached() {
  static const bool v = !envIsZero("DXMT_MV_PROBE_CB");
  return v;
}

uint32_t
mvProbePeriodCached() {
  static const uint32_t v = []() -> uint32_t {
    auto s = env::getEnvVar("DXMT_MV_PROBE_PERIOD");
    if (s.empty())
      return 600u;
    char *end = nullptr;
    unsigned long parsed = std::strtoul(s.c_str(), &end, 10);
    if (!end || end == s.c_str() || parsed == 0 || parsed > 1000000ul)
      return 600u;
    return (uint32_t)parsed;
  }();
  return v;
}

uint32_t
parseEnvUint(const char *name, uint32_t default_v, uint32_t max_v) {
  auto s = env::getEnvVar(name);
  if (s.empty())
    return default_v;
  char *end = nullptr;
  unsigned long parsed = std::strtoul(s.c_str(), &end, 10);
  if (!end || end == s.c_str() || parsed == 0 || parsed > max_v)
    return default_v;
  return (uint32_t)parsed;
}

uint32_t
mvProbeCbDrawsCached() {
  static const uint32_t v = parseEnvUint("DXMT_MV_PROBE_CB_DRAWS", 8u, 256u);
  return v;
}

bool
mvProbeCbPsEnabledCached() {
  static const bool v = envIsOne("DXMT_MV_PROBE_CB_PS");
  return v;
}

bool
mvProbeDepthConvEnabledCached() {
  static const bool v = !envIsZero("DXMT_MV_PROBE_DEPTHCONV");
  return v;
}

bool
mvReadbackEnabledCached() {
  static const bool v = envIsOne("DXMT_MV_PROBE_RB");
  return v;
}

uint32_t
mvProbeRbPeriodCached() {
  static const uint32_t v = parseEnvUint("DXMT_MV_PROBE_RB_PERIOD", 60u, 1000000u);
  return v;
}

uint32_t
mvProbeRbTilesCached() {
  static const uint32_t v = parseEnvUint("DXMT_MV_PROBE_RB_TILES", 5u, 16u);
  return v;
}

uint32_t
mvProbeRbMaxCandidatesCached() {
  static const uint32_t v =
      parseEnvUint("DXMT_MV_PROBE_RB_MAX_CANDIDATES", 2u, (uint32_t)kMvRbMaxSlots);
  return v;
}

uint32_t
mvProbeRbMaxAttemptsCached() {
  static const uint32_t v = parseEnvUint("DXMT_MV_PROBE_RB_MAX_ATTEMPTS", 8u, 256u);
  return v;
}

bool
mvRbNoWaitAllowed() {
  if (g_mv_rb_disabled_no_wait.load(std::memory_order_relaxed))
    return false;
  if (Config::getInstance().getOption<bool>("d3d11.ignoreMapFlagNoWait", false)) {
    g_mv_rb_disabled_no_wait.store(true, std::memory_order_relaxed);
    if (!g_mv_rb_no_wait_logged.exchange(true, std::memory_order_relaxed))
      WARN("unified-metalfx: mv_probe=rb_skip reason=no_wait_disabled");
    return false;
  }
  return true;
}

uint32_t
resourceRid(ID3D11Resource *resource) {
  const uintptr_t p = reinterpret_cast<uintptr_t>(resource);
  return (uint32_t)(p ^ (p >> 32));
}

bool
approxEq(float a, float b, float eps) {
  return std::fabs(a - b) <= eps;
}

bool
isFiniteBounded(float v) {
  return std::isfinite(v) && std::fabs(v) < 1.0e6f;
}

bool
isTwoChannelMvFormat(DXGI_FORMAT fmt) {
  return fmt == DXGI_FORMAT_R16G16_FLOAT || fmt == DXGI_FORMAT_R16G16_SNORM ||
         fmt == DXGI_FORMAT_R32G32_FLOAT;
}

bool
isHalf4Format(DXGI_FORMAT fmt) {
  return fmt == DXGI_FORMAT_R16G16B16A16_FLOAT;
}

int
rtRank(const RtCensusEntry &e, UINT scene_w, UINT scene_h) {
  const bool scene_sz =
      scene_w != 0 && scene_h != 0 && e.width == scene_w && e.height == scene_h;
  if (isTwoChannelMvFormat(e.format) && scene_sz)
    return 0; /* mv_strong */
  if (isHalf4Format(e.format) && scene_sz && e.binds_with_scene_dsv > 0)
    return 1; /* mv_possible */
  return 2;   /* other */
}

int
identityRank(const MvIdentityEntry &e, UINT scene_w, UINT scene_h) {
  RtCensusEntry fake{};
  fake.format = e.format;
  fake.width = e.width;
  fake.height = e.height;
  fake.binds_with_scene_dsv = e.binds_with_scene_dsv;
  return rtRank(fake, scene_w, scene_h);
}

bool
rbFormatAllowed(DXGI_FORMAT fmt) {
  return fmt == DXGI_FORMAT_R16G16B16A16_FLOAT || fmt == DXGI_FORMAT_R16G16_FLOAT ||
         fmt == DXGI_FORMAT_R32G32_FLOAT || fmt == DXGI_FORMAT_R16G16_SNORM ||
         fmt == DXGI_FORMAT_R32_FLOAT;
}

const char *
rtRankName(int rank) {
  switch (rank) {
  case 0:
    return "mv_strong";
  case 1:
    return "mv_possible";
  default:
    return "other";
  }
}

void
loadMat4(const float *src, bool row_major, float m[4][4]) {
  if (row_major) {
    for (int r = 0; r < 4; r++)
      for (int c = 0; c < 4; c++)
        m[r][c] = src[r * 4 + c];
  } else {
    for (int r = 0; r < 4; r++)
      for (int c = 0; c < 4; c++)
        m[r][c] = src[c * 4 + r];
  }
}

bool
perspectiveRowsLike(const float m[4][4], float *out_aspect) {
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++)
      if (!isFiniteBounded(m[r][c]))
        return false;
  if (!approxEq(std::fabs(m[2][3]), 1.0f, 0.05f))
    return false;
  if (!approxEq(m[0][3], 0.0f, 0.05f) || !approxEq(m[1][3], 0.0f, 0.05f) ||
      !approxEq(m[3][3], 0.0f, 0.05f))
    return false;
  if (!(m[0][0] > 0.0f) || !(m[1][1] > 0.0f) || m[2][2] == 0.0f)
    return false;
  const float aspect = m[1][1] / m[0][0];
  if (!(aspect > 0.0f) || !std::isfinite(aspect))
    return false;
  if (out_aspect)
    *out_aspect = aspect;
  return true;
}

bool
aspectWithinTolerance(float aspect, float ref) {
  if (ref <= 0.0f || aspect <= 0.0f)
    return false;
  const float rel = std::fabs(aspect - ref) / ref;
  return rel <= 0.05f;
}

bool
aspectGateOk(float aspect, float scene_aspect, float swap_aspect) {
  if (scene_aspect > 0.0f && aspectWithinTolerance(aspect, scene_aspect))
    return true;
  if (swap_aspect > 0.0f && aspectWithinTolerance(aspect, swap_aspect))
    return true;
  return scene_aspect <= 0.0f && swap_aspect <= 0.0f;
}

bool
projectionLike(const float m[4][4], float scene_aspect, float swap_aspect, float *out_fovy,
               float *out_aspect, float *out_near, float *out_far, bool *out_reversed,
               bool *out_kind_vp) {
  float aspect = 0;
  if (!perspectiveRowsLike(m, &aspect))
    return false;
  if (!aspectGateOk(aspect, scene_aspect, swap_aspect))
    return false;

  const float fovy = 2.0f * std::atan(1.0f / m[1][1]);
  const float near_z = -m[3][2] / m[2][2];
  const float far_z = -m[3][2] / (m[2][2] - 1.0f);
  if (!std::isfinite(fovy) || !std::isfinite(near_z) || !std::isfinite(far_z))
    return false;

  const bool diag_alone =
      approxEq(m[0][1], 0.0f, 1.0e-3f) && approxEq(m[0][2], 0.0f, 1.0e-3f) &&
      approxEq(m[1][0], 0.0f, 1.0e-3f) && approxEq(m[1][2], 0.0f, 1.0e-3f) &&
      approxEq(m[2][0], 0.0f, 1.0e-2f) && approxEq(m[2][1], 0.0f, 1.0e-2f);

  *out_fovy = fovy;
  *out_aspect = aspect;
  *out_near = near_z;
  *out_far = far_z;
  *out_reversed = (m[2][2] < 0.0f) || (near_z > far_z);
  *out_kind_vp = !diag_alone;
  return true;
}

void
storeCamNearMiss(DeviceState *st, UINT slot, UINT offset, bool row_major, float aspect) {
  if (!st || st->cam_near_miss_count >= kCamNearMissCap)
    return;
  for (uint32_t i = 0; i < st->cam_near_miss_count; i++) {
    CamNearMiss &n = st->cam_near_miss[i];
    if (n.valid && n.slot == slot && n.offset == offset && n.row_major == row_major)
      return;
  }
  CamNearMiss &n = st->cam_near_miss[st->cam_near_miss_count++];
  n = {};
  n.slot = slot;
  n.offset = offset;
  n.row_major = row_major;
  n.aspect = aspect;
  n.valid = true;
  st->cam_scan.near_miss++;
}

void
storeCamCandidate(DeviceState *st, UINT slot, UINT offset, bool row_major, bool stage_ps,
                  bool kind_vp, float fovy, float aspect, float near_z, float far_z,
                  bool reversed, float m20, float m21) {
  st->cam_scan.matched++;
  for (uint32_t i = 0; i < st->cam_candidate_count; i++) {
    CamCandidate &c = st->cam_candidates[i];
    if (c.slot == slot && c.offset == offset && c.row_major == row_major &&
        c.stage_ps == stage_ps) {
      if (c.has_prev_jitter) {
        c.jitter_x = m20 - c.prev_m20;
        c.jitter_y = m21 - c.prev_m21;
      }
      c.prev_m20 = m20;
      c.prev_m21 = m21;
      c.has_prev_jitter = true;
      c.kind_vp = kind_vp;
      c.fovy = fovy;
      c.aspect = aspect;
      c.near_z = near_z;
      c.far_z = far_z;
      c.reversed = reversed;
      c.valid = true;
      return;
    }
  }
  if (st->cam_candidate_count >= kCamCandidateCap)
    return;
  CamCandidate &c = st->cam_candidates[st->cam_candidate_count++];
  c = {};
  c.slot = slot;
  c.offset = offset;
  c.row_major = row_major;
  c.stage_ps = stage_ps;
  c.kind_vp = kind_vp;
  c.fovy = fovy;
  c.aspect = aspect;
  c.near_z = near_z;
  c.far_z = far_z;
  c.reversed = reversed;
  c.prev_m20 = m20;
  c.prev_m21 = m21;
  c.has_prev_jitter = true;
  c.jitter_x = 0;
  c.jitter_y = 0;
  c.valid = true;
}

void
scanStageConstantBuffers(DeviceState *st, const MvProbeCbSlot slots[14], bool stage_ps) {
  if (!st || !mvProbeCbEnabledCached())
    return;
  const float scene_aspect =
      (st->tracked_w > 0 && st->tracked_h > 0)
          ? (float)st->tracked_w / (float)st->tracked_h
          : 0.0f;
  const UINT sw = g_mv_probe_swap_w.load(std::memory_order_relaxed);
  const UINT sh = g_mv_probe_swap_h.load(std::memory_order_relaxed);
  const float swap_aspect =
      (sw > 0 && sh > 0) ? (float)sw / (float)sh : 0.0f;

  for (UINT slot = 0; slot < 14; slot++) {
    const MvProbeCbSlot &cb = slots[slot];
    if (!cb.buffer)
      continue;
    st->cam_scan.slots_bound++;
    D3D11ResourceCommon *res = GetResourceCommon(cb.buffer);
    if (!res)
      continue;
    const Rc<Buffer> &buf = res->buffer();
    if (!buf)
      continue;
    BufferAllocation *alloc = buf->current();
    const uint32_t sub = alloc ? alloc->currentSuballocation() : 0;
    if (!alloc || alloc->flags().test(BufferAllocationFlag::GpuManaged)) {
      st->cam_scan.gpu_managed++;
      continue;
    }
    if (alloc->mappedMemory(sub) == nullptr) {
      continue;
    }
    st->cam_scan.readable++;

    D3D11_BUFFER_DESC desc{};
    {
      dxmt::Com<ID3D11Buffer> as_buf;
      if (FAILED(cb.buffer->QueryInterface(IID_PPV_ARGS(&as_buf))) || !as_buf)
        continue;
      as_buf->GetDesc(&desc);
    }
    const UINT byte_width = desc.ByteWidth;
    const UINT start = cb.first_constant * 16u;
    if (start >= byte_width)
      continue;
    UINT available = byte_width - start;
    if (cb.num_constants)
      available = std::min(available, cb.num_constants * 16u);
    const UINT read_len = std::min(available, (UINT)kMvProbeCbReadMax);
    if (read_len < 64) {
      st->cam_scan.too_short++;
      continue;
    }

    const char *base =
        reinterpret_cast<const char *>(alloc->mappedMemory(sub)) + start;

    for (UINT off = 0; off + 64 <= read_len; off += 16) {
      st->cam_scan.windows++;
      float raw[16];
      std::memcpy(raw, base + off, sizeof(raw));
      for (int major = 0; major < 2; major++) {
        const bool row_major = major == 0;
        float m[4][4];
        loadMat4(raw, row_major, m);
        float row_aspect = 0;
        if (!perspectiveRowsLike(m, &row_aspect))
          continue;
        float fovy = 0, aspect = 0, near_z = 0, far_z = 0;
        bool reversed = false, kind_vp = false;
        if (!aspectGateOk(row_aspect, scene_aspect, swap_aspect)) {
          storeCamNearMiss(st, slot, start + off, row_major, row_aspect);
          continue;
        }
        if (!projectionLike(
                m, scene_aspect, swap_aspect, &fovy, &aspect, &near_z, &far_z, &reversed,
                &kind_vp
            ))
          continue;
        storeCamCandidate(
            st, slot, start + off, row_major, stage_ps, kind_vp, fovy, aspect, near_z,
            far_z, reversed, m[2][0], m[2][1]
        );
      }
    }
  }
}

char g_mv_probe_logs[kMvProbeLogCap][kMvProbeLogLen];
uint32_t g_mv_probe_log_n = 0;

void
pushMvProbeLog(const char *line) {
  if (!line || g_mv_probe_log_n >= kMvProbeLogCap)
    return;
  std::snprintf(g_mv_probe_logs[g_mv_probe_log_n], kMvProbeLogLen, "%s", line);
  g_mv_probe_log_n++;
}

float
binary16ToFloat(uint16_t h) {
  const uint32_t sign = (h >> 15u) & 1u;
  const uint32_t exp = (h >> 10u) & 0x1fu;
  const uint32_t mant = h & 0x3ffu;
  if (exp == 0) {
    if (mant == 0)
      return sign ? -0.0f : 0.0f;
    const float v = (float)mant / 1024.0f;
    return sign ? -v : v;
  }
  if (exp == 31) {
    if (mant == 0)
      return sign ? -INFINITY : INFINITY;
    return NAN;
  }
  const float v = std::ldexp(1.0f + (float)mant / 1024.0f, (int)exp - 15);
  return sign ? -v : v;
}

size_t
rbBytesPerPixel(DXGI_FORMAT fmt) {
  switch (fmt) {
  case DXGI_FORMAT_R16G16B16A16_FLOAT:
    return 8;
  case DXGI_FORMAT_R16G16_FLOAT:
  case DXGI_FORMAT_R16G16_SNORM:
    return 4;
  case DXGI_FORMAT_R32G32_FLOAT:
    return 8;
  case DXGI_FORMAT_R32_FLOAT:
    return 4;
  default:
    return 16;
  }
}

void
decodeRbSample(DXGI_FORMAT fmt, const uint8_t *px, float out_xy[2], float out_zw[2]) {
  out_xy[0] = out_xy[1] = out_zw[0] = out_zw[1] = 0.0f;
  switch (fmt) {
  case DXGI_FORMAT_R16G16B16A16_FLOAT: {
    const uint16_t *h = reinterpret_cast<const uint16_t *>(px);
    out_xy[0] = binary16ToFloat(h[0]);
    out_xy[1] = binary16ToFloat(h[1]);
    out_zw[0] = binary16ToFloat(h[2]);
    out_zw[1] = binary16ToFloat(h[3]);
    break;
  }
  case DXGI_FORMAT_R16G16_FLOAT: {
    const uint16_t *h = reinterpret_cast<const uint16_t *>(px);
    out_xy[0] = binary16ToFloat(h[0]);
    out_xy[1] = binary16ToFloat(h[1]);
    break;
  }
  case DXGI_FORMAT_R32G32_FLOAT: {
    const float *f = reinterpret_cast<const float *>(px);
    out_xy[0] = f[0];
    out_xy[1] = f[1];
    break;
  }
  case DXGI_FORMAT_R16G16_SNORM: {
    const int16_t *s = reinterpret_cast<const int16_t *>(px);
    out_xy[0] = std::max(-1.0f, std::min(1.0f, (float)s[0] / 32767.0f));
    out_xy[1] = std::max(-1.0f, std::min(1.0f, (float)s[1] / 32767.0f));
    break;
  }
  case DXGI_FORMAT_R32_FLOAT: {
    const float *f = reinterpret_cast<const float *>(px);
    out_xy[0] = f[0];
    break;
  }
  default:
    break;
  }
}

void
releaseMvRbSlot(MvRbPending *slot) {
  if (!slot)
    return;
  slot->staging = nullptr;
  slot->held_src = nullptr;
  slot->active = false;
}

bool
ensureRbStaging(
    ID3D11Device *device, MvRbPending *slot, DXGI_FORMAT fmt, uint32_t tiles
) {
  if (!device || !slot)
    return false;
  const UINT side = tiles * (UINT)kMvRbTileSize;
  if (slot->staging && slot->format == fmt && slot->tiles == tiles)
    return true;
  D3D11_TEXTURE2D_DESC desc{};
  desc.Width = side;
  desc.Height = side;
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = fmt;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_STAGING;
  desc.BindFlags = 0;
  desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  dxmt::Com<ID3D11Texture2D> staging;
  if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging)))
    return false;
  slot->staging = staging.ptr();
  slot->format = fmt;
  slot->tiles = tiles;
  return true;
}

void
logRbStats(DeviceState *st) {
  if (!st)
    return;
  char line[kMvProbeLogLen];
  const MvRbPeriodStats &s = st->mv_rb_stats;
  std::snprintf(
      line, sizeof(line),
      "rb_stats copies=%llu mapped=%llu still_drawing=%llu timeout=%llu probe_us=%llu",
      (unsigned long long)s.copies, (unsigned long long)s.mapped,
      (unsigned long long)s.still_drawing, (unsigned long long)s.timeout,
      (unsigned long long)s.probe_us
  );
  pushMvProbeLog(line);
  st->mv_rb_stats = {};
}

void
analyzeRbMap(
    DeviceState *st, const MvRbPending &slot, const D3D11_MAPPED_SUBRESOURCE &mapped,
    uint64_t presentation_count
) {
  if (!st || !mapped.pData)
    return;
  const uint32_t tiles = slot.tiles;
  const size_t n = size_t(tiles) * tiles * kMvRbSamplesPerTile;
  std::vector<float> xs, ys, mags;
  xs.reserve(n);
  ys.reserve(n);
  mags.reserve(n);
  size_t nan_c = 0, inf_c = 0, zero_c = 0;
  double sx = 0, sy = 0, sz = 0, sw = 0;
  double ax = 0, ay = 0;
  double radial_sum = 0;
  double dir_sin = 0, dir_cos = 0;
  size_t dir_n = 0;
  const float cx = slot.src_w * 0.5f;
  const float cy = slot.src_h * 0.5f;
  const UINT stride = mapped.RowPitch;
  const size_t bpp = rbBytesPerPixel(slot.format);
  for (uint32_t j = 0; j < tiles; j++) {
    for (uint32_t i = 0; i < tiles; i++) {
      const UINT base_x = i * (UINT)kMvRbTileSize;
      const UINT base_y = j * (UINT)kMvRbTileSize;
      for (UINT ty = 0; ty < (UINT)kMvRbTileSize; ty++) {
        const uint8_t *row =
            reinterpret_cast<const uint8_t *>(mapped.pData) + (base_y + ty) * stride;
        for (UINT tx = 0; tx < (UINT)kMvRbTileSize; tx++) {
          float xy[2], zw[2];
          decodeRbSample(slot.format, row + (base_x + tx) * bpp, xy, zw);
          const float x = xy[0], y = xy[1];
          if (!std::isfinite(x) || !std::isfinite(y)) {
            if (std::isnan(x) || std::isnan(y))
              nan_c++;
            else
              inf_c++;
            continue;
          }
          if (x == 0.0f && y == 0.0f)
            zero_c++;
          xs.push_back(x);
          ys.push_back(y);
          sx += x;
          sy += y;
          sz += zw[0];
          sw += zw[1];
          ax += std::fabs(x);
          ay += std::fabs(y);
          const float mag = std::hypot(x, y);
          mags.push_back(mag);
          const float px = base_x + tx + 0.5f;
          const float py = base_y + ty + 0.5f;
          const float dx = px - cx;
          const float dy = py - cy;
          const float dlen = std::hypot(dx, dy);
          if (mag > 1.0e-6f && dlen > 1.0e-3f) {
            radial_sum += (x * dx + y * dy) / (mag * dlen);
            dir_sin += std::sin(std::atan2(y, x));
            dir_cos += std::cos(std::atan2(y, x));
            dir_n++;
          }
        }
      }
    }
  }
  if (xs.empty()) {
    pushMvProbeLog("rb=none reason=empty_sample");
    return;
  }
  auto pct = [&](double q) -> double {
    if (mags.empty())
      return 0.0;
    std::sort(mags.begin(), mags.end());
    const size_t idx = std::min(mags.size() - 1, (size_t)(q * (mags.size() - 1)));
    return (double)mags[idx];
  };
  const double mean_x = sx / xs.size();
  const double mean_y = sy / ys.size();
  const double mean_z = sz / xs.size();
  const double mean_w = sw / xs.size();
  float xmin = xs[0], xmax = xs[0];
  float ymin = ys[0], ymax = ys[0];
  for (size_t k = 1; k < xs.size(); k++) {
    xmin = std::min(xmin, xs[k]);
    xmax = std::max(xmax, xs[k]);
    ymin = std::min(ymin, ys[k]);
    ymax = std::max(ymax, ys[k]);
  }
  const double absmean_x = ax / xs.size();
  const double absmean_y = ay / xs.size();
  const double mag_p50 = pct(0.50);
  const double mag_p95 = pct(0.95);
  const double mag_max = mags.empty() ? 0.0 : mags.back();
  const double mag_px_p95 = mag_p95 * slot.src_w;
  const char *unit_hint = "unknown";
  if (mag_p95 > 1e-4 && mag_p95 < 0.05)
    unit_hint = "uv";
  else if (mag_p95 >= 0.05 && mag_p95 < 2.0)
    unit_hint = "ndc?";
  else if (mag_p95 >= 2.0 && mag_p95 < 200.0)
    unit_hint = "px";
  double uni_num = 0, uni_den = 0;
  for (size_t k = 0; k < xs.size(); k++) {
    uni_num += std::hypot(xs[k] + mean_x, ys[k] + mean_y);
    uni_den += std::hypot(xs[k], ys[k]);
  }
  const double uniformity = uni_den > 1e-9 ? uni_num / uni_den : 0.0;
  const double radial = dir_n ? radial_sum / (double)dir_n : 0.0;
  const double dir_mean_deg =
      dir_n ? std::atan2(dir_sin / dir_n, dir_cos / dir_n) * 180.0 / 3.141592653589793 : 0.0;
  double dir_var = 0;
  if (dir_n) {
    const double mean_rad = std::atan2(dir_sin / dir_n, dir_cos / dir_n);
    for (size_t k = 0; k < xs.size(); k++) {
      const double d = std::atan2(ys[k], xs[k]) - mean_rad;
      dir_var += d * d;
    }
    dir_var = std::sqrt(dir_var / xs.size()) * 180.0 / 3.141592653589793;
  }
  auto const_frac = [&](double mean_v, auto &vals, size_t ch_count) {
    size_t c = 0;
    for (size_t k = 0; k < vals.size(); k++) {
      if (std::fabs(vals[k] - mean_v) <= 1e-4)
        c++;
    }
    return vals.empty() ? 1.0 : (double)c / (double)vals.size();
  };
  std::vector<float> zs(xs.size()), ws(xs.size());
  /* z/w const fractions approximated from accumulated means per sample above */
  const double z_const = 1.0;
  const double w_const = approxEq((float)mean_w, 1.0f, 1e-3f) ? 1.0 : 0.0;
  const double zero_frac = (double)zero_c / (double)n;
  const uint64_t age = presentation_count > slot.copy_present ? presentation_count - slot.copy_present : 0;
  char line[kMvProbeLogLen];
  std::snprintf(
      line, sizeof(line), "rb rid=%08x fmt=%u %ux%u slot_mask=0x%x n=%zu nan=%zu inf=%zu",
      slot.rid, (unsigned)slot.format, slot.src_w, slot.src_h, slot.slot_mask, n, nan_c, inf_c
  );
  pushMvProbeLog(line);
  std::snprintf(
      line, sizeof(line),
      "rb rid=%08x x[min=%.4f max=%.4f mean=%.4f absmean=%.4f] y[min=%.4f max=%.4f mean=%.4f absmean=%.4f]",
      slot.rid, xmin, xmax, mean_x, absmean_x, ymin, ymax, mean_y, absmean_y
  );
  pushMvProbeLog(line);
  std::snprintf(
      line, sizeof(line),
      "rb rid=%08x z[absmean=%.4f const=%.2f] w[absmean=%.4f const=%.2f] zero_frac=%.2f",
      slot.rid, std::fabs(mean_z), z_const, std::fabs(mean_w), w_const, zero_frac
  );
  pushMvProbeLog(line);
  std::snprintf(
      line, sizeof(line),
      "rb rid=%08x mag_raw_p50=%.4f p95=%.4f max=%.4f mag_px_p95=%.1f unit_hint=%s", slot.rid,
      mag_p50, mag_p95, mag_max, mag_px_p95, unit_hint
  );
  pushMvProbeLog(line);
  std::snprintf(
      line, sizeof(line),
      "rb rid=%08x uniformity=%.2f radial=%.2f dir_mean=%.0fdeg dir_spread=%.0fdeg age=%llu",
      slot.rid, uniformity, radial, dir_mean_deg, dir_var, (unsigned long long)age
  );
  pushMvProbeLog(line);
  st->mv_rb_stats.mapped++;
}

bool
tryMapMvRbPending(
    DeviceState *st, MvRbPending *slot, ID3D11DeviceContext *ctx, uint64_t presentation_count
) {
  if (!st || !slot || !slot->active || !slot->staging || !ctx)
    return false;
  if (presentation_count < slot->copy_present + 2)
    return false;
  slot->map_attempts++;
  D3D11_MAPPED_SUBRESOURCE mapped{};
  const HRESULT hr = ctx->Map(
      slot->staging.ptr(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped
  );
  if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
    st->mv_rb_stats.still_drawing++;
    if (slot->map_attempts >= mvProbeRbMaxAttemptsCached()) {
      pushMvProbeLog("rb=none reason=rb_timeout");
      st->mv_rb_stats.timeout++;
      releaseMvRbSlot(slot);
    }
    return false;
  }
  if (FAILED(hr)) {
    pushMvProbeLog("rb=none reason=map_failed");
    releaseMvRbSlot(slot);
    return false;
  }
  analyzeRbMap(st, *slot, mapped, presentation_count);
  ctx->Unmap(slot->staging.ptr(), 0);
  releaseMvRbSlot(slot);
  return true;
}

void
queueMvRbSample(
    DeviceState *st, ID3D11Device *device, ID3D11DeviceContext *ctx,
    const MvIdentityEntry &cand, uint64_t presentation_count, uint32_t tiles
) {
  if (!st || !device || !ctx || !cand.valid || !cand.resource)
    return;
  if (!rbFormatAllowed(cand.format)) {
    pushMvProbeLog("rb_skip reason=format_decode_unsupported");
    return;
  }
  MvRbPending *slot = nullptr;
  for (size_t i = 0; i < kMvRbMaxSlots; i++) {
    if (!st->mv_rb_pending[i].active) {
      slot = &st->mv_rb_pending[i];
      break;
    }
  }
  if (!slot) {
    pushMvProbeLog("rb=none reason=pending_full");
    return;
  }
  if (!ensureRbStaging(device, slot, cand.format, tiles))
    return;
  dxmt::Com<ID3D11Resource> src;
  src = cand.resource;
  src->AddRef();
  slot->held_src = src.ptr();
  slot->rid = cand.rid;
  slot->src_w = cand.width;
  slot->src_h = cand.height;
  slot->slot_mask = cand.slot_mask;
  slot->copy_present = presentation_count;
  slot->map_attempts = 0;
  slot->active = true;
  for (uint32_t j = 0; j < tiles; j++) {
    for (uint32_t i = 0; i < tiles; i++) {
      const float nx =
          tiles <= 1 ? 0.5f : (0.1f + 0.8f * (float)i / (float)(tiles - 1));
      const float ny =
          tiles <= 1 ? 0.5f : (0.1f + 0.8f * (float)j / (float)(tiles - 1));
      const UINT sx =
          (UINT)std::max(0.0f, nx * (float)cand.width - (float)kMvRbTileSize * 0.5f);
      const UINT sy =
          (UINT)std::max(0.0f, ny * (float)cand.height - (float)kMvRbTileSize * 0.5f);
      const UINT ex = std::min(cand.width, sx + (UINT)kMvRbTileSize);
      const UINT ey = std::min(cand.height, sy + (UINT)kMvRbTileSize);
      D3D11_BOX box{sx, sy, 0, ex, ey, 1};
      ctx->CopySubresourceRegion(
          slot->staging.ptr(), 0, i * (UINT)kMvRbTileSize, j * (UINT)kMvRbTileSize, 0,
          slot->held_src.ptr(), 0, &box
      );
      st->mv_rb_stats.copies++;
    }
  }
}

void
runMvReadbackMapTickLocked(
    DeviceState *st, ID3D11DeviceContext *ctx, uint64_t presentation_count
) {
  if (!st || !ctx)
    return;
  for (size_t i = 0; i < kMvRbMaxSlots; i++) {
    if (st->mv_rb_pending[i].active)
      tryMapMvRbPending(st, &st->mv_rb_pending[i], ctx, presentation_count);
  }
}

void
runMvReadbackCopyTickLocked(
    DeviceState *st, ID3D11Device *device, ID3D11DeviceContext *ctx,
    uint64_t presentation_count
) {
  if (!st || !device || !ctx)
    return;
  const auto t0 = std::chrono::steady_clock::now();
  bool any_rb = false;
  const uint32_t max_c = mvProbeRbMaxCandidatesCached();
  const uint32_t tiles = mvProbeRbTilesCached();
  std::vector<uint32_t> order;
  order.reserve(st->mv_identity_count);
  for (uint32_t i = 0; i < st->mv_identity_count; i++) {
    if (st->mv_identity[i].valid)
      order.push_back(i);
  }
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    const MvIdentityEntry &ea = st->mv_identity[a];
    const MvIdentityEntry &eb = st->mv_identity[b];
    const int ra = identityRank(ea, st->tracked_w, st->tracked_h);
    const int rb = identityRank(eb, st->tracked_w, st->tracked_h);
    if (ra != rb)
      return ra < rb;
    if (ea.binds_with_scene_dsv != eb.binds_with_scene_dsv)
      return ea.binds_with_scene_dsv > eb.binds_with_scene_dsv;
    return ea.binds > eb.binds;
  });
  uint32_t queued = 0;
  for (uint32_t idx : order) {
    if (queued >= max_c)
      break;
    const MvIdentityEntry &cand = st->mv_identity[idx];
    bool pending = false;
    for (size_t i = 0; i < kMvRbMaxSlots; i++) {
      if (st->mv_rb_pending[i].active && st->mv_rb_pending[i].rid == cand.rid) {
        pending = true;
        break;
      }
    }
    if (pending)
      continue;
    size_t free_slots = 0;
    for (size_t i = 0; i < kMvRbMaxSlots; i++)
      if (!st->mv_rb_pending[i].active)
        free_slots++;
    if (free_slots == 0)
      break;
    queueMvRbSample(st, device, ctx, cand, presentation_count, tiles);
    queued++;
    any_rb = true;
  }
  if (!any_rb)
    pushMvProbeLog("rb=none reason=no_sample");
  const auto t1 = std::chrono::steady_clock::now();
  st->mv_rb_stats.probe_us =
      (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  logRbStats(st);
}

static bool
beginMvReadbackTick(
    ID3D11Device *device, ID3D11DeviceContext *ctx, uint64_t presentation_count,
    UINT swapchain_w, UINT swapchain_h
) {
  if (!mvReadbackEnabled() || !device || !ctx)
    return false;
  if (!mvRbNoWaitAllowed())
    return false;
  g_mv_probe_present_count.store(presentation_count, std::memory_order_relaxed);
  g_mv_probe_swap_w.store(swapchain_w, std::memory_order_relaxed);
  g_mv_probe_swap_h.store(swapchain_h, std::memory_order_relaxed);
  const uint32_t rb_period = mvProbeRbPeriodCached();
  if (rb_period == 0 || presentation_count % rb_period != 0)
    return false;
  return true;
}

void
buildCensusLogs(DeviceState *st, uint64_t presentation_count) {
  g_mv_probe_log_n = 0;
  if (!st) {
    pushMvProbeLog("rt_census scene_dsv=0x0 entries=0 dropped=0");
    pushMvProbeLog("mv_strong=none");
    pushMvProbeLog("cam_candidate=none");
    return;
  }

  char line[kMvProbeLogLen];
  std::snprintf(
      line, sizeof(line), "rt_census scene_dsv=%ux%u entries=%u dropped=%llu",
      st->tracked_w, st->tracked_h, st->rt_census_count,
      (unsigned long long)st->rt_census_dropped
  );
  pushMvProbeLog(line);

  uint32_t order[kRtCensusCap];
  for (uint32_t i = 0; i < st->rt_census_count; i++)
    order[i] = i;
  std::sort(order, order + st->rt_census_count, [&](uint32_t a, uint32_t b) {
    const int ra = rtRank(st->rt_census[a], st->tracked_w, st->tracked_h);
    const int rb = rtRank(st->rt_census[b], st->tracked_w, st->tracked_h);
    if (ra != rb)
      return ra < rb;
    if (st->rt_census[a].binds_with_scene_dsv != st->rt_census[b].binds_with_scene_dsv)
      return st->rt_census[a].binds_with_scene_dsv > st->rt_census[b].binds_with_scene_dsv;
    return st->rt_census[a].binds > st->rt_census[b].binds;
  });

  bool any_strong = false;
  for (uint32_t i = 0; i < st->rt_census_count; i++) {
    const RtCensusEntry &e = st->rt_census[order[i]];
    const int rank = rtRank(e, st->tracked_w, st->tracked_h);
    if (rank == 0)
      any_strong = true;
    std::snprintf(
        line, sizeof(line),
        "%s fmt=%u %ux%u binds=%llu with_dsv=%llu no_dsv=%llu presents_seen=%llu",
        rtRankName(rank), (unsigned)e.format, e.width, e.height,
        (unsigned long long)e.binds, (unsigned long long)e.binds_with_scene_dsv,
        (unsigned long long)e.binds_no_dsv, (unsigned long long)presentation_count
    );
    pushMvProbeLog(line);
  }
  if (!any_strong)
    pushMvProbeLog("mv_strong=none");

  bool any_identity = false;
  for (uint32_t i = 0; i < st->mv_identity_count; i++) {
    const MvIdentityEntry &id = st->mv_identity[i];
    if (!id.valid)
      continue;
    any_identity = true;
    const int rank = identityRank(id, st->tracked_w, st->tracked_h);
    std::snprintf(
        line, sizeof(line),
        "mv_identity rank=%s rid=%08x fmt=%u %ux%u slot_mask=0x%x binds=%llu "
        "with_dsv=%llu no_dsv=%llu frames=%u rid_reused=%u",
        rtRankName(rank), id.rid, (unsigned)id.format, id.width, id.height, id.slot_mask,
        (unsigned long long)id.binds, (unsigned long long)id.binds_with_scene_dsv,
        (unsigned long long)id.binds_no_dsv, id.frames_seen, id.rid_reused ? 1u : 0u
    );
    pushMvProbeLog(line);
  }
  if (!any_identity)
    pushMvProbeLog("mv_identity=none");

  const CamScanStats &cs = st->cam_scan;
  std::snprintf(
      line, sizeof(line),
      "cam_scan draws=%llu slots_bound=%llu readable=%llu gpu_managed=%llu "
      "too_short=%llu windows=%llu matched=%llu near_miss=%llu",
      (unsigned long long)cs.draws, (unsigned long long)cs.slots_bound,
      (unsigned long long)cs.readable, (unsigned long long)cs.gpu_managed,
      (unsigned long long)cs.too_short, (unsigned long long)cs.windows,
      (unsigned long long)cs.matched, (unsigned long long)cs.near_miss
  );
  pushMvProbeLog(line);

  const float scene_aspect =
      (st->tracked_w > 0 && st->tracked_h > 0)
          ? (float)st->tracked_w / (float)st->tracked_h
          : 0.0f;
  for (uint32_t i = 0; i < st->cam_near_miss_count; i++) {
    const CamNearMiss &n = st->cam_near_miss[i];
    if (!n.valid)
      continue;
    std::snprintf(
        line, sizeof(line),
        "cam_near_miss slot=%u off=%u major=%s aspect=%.3f scene_aspect=%.3f",
        n.slot, n.offset, n.row_major ? "row" : "col", n.aspect, scene_aspect
    );
    pushMvProbeLog(line);
  }

  if (mvProbeDepthConvEnabledCached()) {
    const DepthConvState &dc = st->depth_conv;
    const char *func_name = "UNKNOWN";
    switch (dc.depth_func) {
    case D3D11_COMPARISON_NEVER:
      func_name = "NEVER";
      break;
    case D3D11_COMPARISON_LESS:
      func_name = "LESS";
      break;
    case D3D11_COMPARISON_EQUAL:
      func_name = "EQUAL";
      break;
    case D3D11_COMPARISON_LESS_EQUAL:
      func_name = "LESS_EQUAL";
      break;
    case D3D11_COMPARISON_GREATER:
      func_name = "GREATER";
      break;
    case D3D11_COMPARISON_NOT_EQUAL:
      func_name = "NOT_EQUAL";
      break;
    case D3D11_COMPARISON_GREATER_EQUAL:
      func_name = "GREATER_EQUAL";
      break;
    case D3D11_COMPARISON_ALWAYS:
      func_name = "ALWAYS";
      break;
    default:
      break;
    }
    const bool reverse_clear = dc.have_clear && approxEq(dc.clear_depth, 0.0f, 1.0e-4f);
    const bool reverse_func =
        dc.have_func && dc.depth_enable &&
        (dc.depth_func == D3D11_COMPARISON_GREATER ||
         dc.depth_func == D3D11_COMPARISON_GREATER_EQUAL);
    const bool reverse_vp = dc.have_vp && dc.vp_min > dc.vp_max;
    const char *verdict =
        (reverse_clear || reverse_func || reverse_vp) ? "reverse_z" : "standard_z";
    std::snprintf(
        line, sizeof(line),
        "depth_conv clear=%.3f func=%s vp=[%.3f,%.3f] verdict=%s",
        dc.have_clear ? dc.clear_depth : -1.0f, func_name,
        dc.have_vp ? dc.vp_min : 0.0f, dc.have_vp ? dc.vp_max : 1.0f, verdict
    );
    pushMvProbeLog(line);
  }

  bool any_cam = false;
  for (uint32_t i = 0; i < st->cam_candidate_count; i++) {
    const CamCandidate &c = st->cam_candidates[i];
    if (!c.valid)
      continue;
    any_cam = true;
    std::snprintf(
        line, sizeof(line),
        "cam_candidate kind=%s stage=%s slot=%u off=%u major=%s fovy=%.2frad "
        "aspect=%.3f near=%.3f far=%.1f reversed=%s jitter=(%.5f,%.5f)",
        c.kind_vp ? "vp" : "proj", c.stage_ps ? "ps" : "vs", c.slot, c.offset,
        c.row_major ? "row" : "col", c.fovy, c.aspect, c.near_z, c.far_z,
        c.reversed ? "yes" : "no", c.jitter_x, c.jitter_y
    );
    pushMvProbeLog(line);
  }
  if (!any_cam)
    pushMvProbeLog("cam_candidate=none");
}

void
resetProbePeriodCounters(DeviceState *st) {
  if (!st)
    return;
  st->rt_census_count = 0;
  st->rt_census_dropped = 0;
  std::memset(st->rt_census, 0, sizeof(st->rt_census));
  st->cam_candidate_count = 0;
  std::memset(st->cam_candidates, 0, sizeof(st->cam_candidates));
  st->mv_identity_count = 0;
  std::memset(st->mv_identity, 0, sizeof(st->mv_identity));
  st->cam_scan = {};
  st->cam_near_miss_count = 0;
  std::memset(st->cam_near_miss, 0, sizeof(st->cam_near_miss));
  st->depth_conv = {};
}

/** GetDevice AddRefs; we Release immediately and keep the raw pointer as map key. */
ID3D11Device *
deviceKeyFromChild(ID3D11DeviceChild *child) {
  if (!child)
    return nullptr;
  ID3D11Device *dev = nullptr;
  child->GetDevice(&dev);
  if (!dev)
    return nullptr;
  ID3D11Device *key = dev;
  dev->Release();
  return key;
}

ID3D11Device *
deviceKeyFromContext(ID3D11DeviceContext *ctx) {
  if (!ctx)
    return nullptr;
  ID3D11Device *dev = nullptr;
  ctx->GetDevice(&dev);
  if (!dev)
    return nullptr;
  ID3D11Device *key = dev;
  dev->Release();
  return key;
}

DeviceState *
stateFor(ID3D11Device *device) {
  if (!device)
    return nullptr;
  return &g_by_device[device];
}

} // namespace

bool
enabled() {
  if (envIsZero("DXMT_PRESENT_DM_EXPORT"))
    return false;
  return envIsOne("DXMT_PRESENT_DEPTH_MOTION") ||
         envIsOne("DXMT_METALFX_UNIFIED_TEMPORAL_INTERP");
}

bool
mvProbeEnabled() {
  static const bool v = envIsOne("DXMT_MV_PROBE");
  return v;
}

bool
mvProbeArmBActive() {
  return mvProbeEnabled() && g_arm_b_sample_active.load(std::memory_order_relaxed) &&
         g_arm_b_draws_hint.load(std::memory_order_relaxed) > 0;
}

bool
mvProbeCbPsEnabled() {
  return mvProbeEnabled() && mvProbeCbPsEnabledCached();
}

bool
mvReadbackEnabled() {
  return mvProbeEnabled() && mvReadbackEnabledCached();
}

uint32_t
mvProbePeriod() {
  return mvProbePeriodCached();
}

bool
verbose() {
  return envIsOne("DXMT_PRESENT_DM_VERBOSE");
}

bool
stepsEnabled() {
  return verbose() || envIsOne("DXMT_PRESENT_DM_STEPS");
}

void
logStep(const char *step) {
  if (!stepsEnabled() || !step)
    return;
  WARN("present-dm: step=", step);
}

void
releaseMvProbeReadbackLocked(DeviceState *st) {
  if (!st)
    return;
  for (size_t i = 0; i < kMvRbMaxSlots; i++)
    releaseMvRbSlot(&st->mv_rb_pending[i]);
  st->mv_rb_stats = {};
}

void
clearDevice(ID3D11Device *device) {
  if (!device)
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  if (auto it = g_by_device.find(device); it != g_by_device.end()) {
    releaseMvProbeReadbackLocked(&it->second);
    g_by_device.erase(it);
  }
}

void
noteTemporalUpscale(
    ID3D11Resource *color, ID3D11Resource *depth, ID3D11Resource *motion_vector
) {
  auto n = g_temporal_upscale_count.fetch_add(1, std::memory_order_relaxed) + 1;
  WARN(
      "present-dm: TemporalUpscale fired (count=", n, ") — title/NGX path has depth+motion"
  );
  ID3D11Device *key = deviceKeyFromChild(depth ? depth : (color ? color : motion_vector));
  if (!key)
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  DeviceState *st = stateFor(key);
  if (!st)
    return;
  st->title_depth = depth;
  st->title_motion = motion_vector;
  st->title_motion_valid = motion_vector != nullptr;
  (void)color;
}

uint64_t
temporalUpscaleCount() {
  return g_temporal_upscale_count.load(std::memory_order_relaxed);
}

void
noteUnifiedTemporalPresent(UINT output_w, UINT output_h) {
  g_unified_temporal_encode_count.fetch_add(1, std::memory_order_relaxed);
  g_last_unified_present_w.store(output_w, std::memory_order_relaxed);
  g_last_unified_present_h.store(output_h, std::memory_order_relaxed);
}

uint64_t
unifiedTemporalEncodeCount() {
  return g_unified_temporal_encode_count.load(std::memory_order_relaxed);
}

UINT
lastUnifiedPresentWidth() {
  return g_last_unified_present_w.load(std::memory_order_relaxed);
}

UINT
lastUnifiedPresentHeight() {
  return g_last_unified_present_h.load(std::memory_order_relaxed);
}

void
noteUnifiedFiPresent(UINT output_w, UINT output_h) {
  g_unified_fi_present_count.fetch_add(1, std::memory_order_relaxed);
  g_last_unified_present_w.store(output_w, std::memory_order_relaxed);
  g_last_unified_present_h.store(output_h, std::memory_order_relaxed);
}

uint64_t
unifiedFiPresentCount() {
  return g_unified_fi_present_count.load(std::memory_order_relaxed);
}

void
trackSceneDepth(ID3D11DepthStencilView *dsv) {
  if (!dsv || !enabled() || !trackedDepthAllowed())
    return;
  dxmt::Com<ID3D11Resource> resource;
  dsv->GetResource(&resource);
  if (!resource)
    return;
  D3D11_TEXTURE2D_DESC desc{};
  if (!resourceTex2DDesc(resource.ptr(), &desc))
    return;
  ID3D11Device *key = deviceKeyFromChild(dsv);
  if (!key)
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  DeviceState *st = stateFor(key);
  if (!st)
    return;
  // Prefer largest DSV seen on this device (scene depth usually out-areas shadow maps).
  if (st->tracked_depth &&
      texArea(desc.Width, desc.Height) < texArea(st->tracked_w, st->tracked_h))
    return;
  st->tracked_depth = resource.ptr();
  st->tracked_w = desc.Width;
  st->tracked_h = desc.Height;
}

void
noteRenderTargetBind(
    ID3D11RenderTargetView *const *rtvs, UINT num_rtvs, ID3D11DepthStencilView *dsv
) {
  if (!mvProbeEnabled()) [[unlikely]]
    return;
  if (!rtvs || num_rtvs == 0)
    return;

  ID3D11Device *key = nullptr;
  if (dsv)
    key = deviceKeyFromChild(dsv);
  if (!key) {
    for (UINT i = 0; i < num_rtvs; i++) {
      if (rtvs[i]) {
        key = deviceKeyFromChild(rtvs[i]);
        if (key)
          break;
      }
    }
  }
  if (!key)
    return;

  UINT dsv_w = 0, dsv_h = 0;
  if (dsv) {
    dxmt::Com<ID3D11Resource> dsv_res;
    dsv->GetResource(&dsv_res);
    D3D11_TEXTURE2D_DESC dsv_desc{};
    if (dsv_res && resourceTex2DDesc(dsv_res.ptr(), &dsv_desc)) {
      dsv_w = dsv_desc.Width;
      dsv_h = dsv_desc.Height;
    }
  }

  std::lock_guard<std::mutex> lock(g_lock);
  DeviceState *st = stateFor(key);
  if (!st)
    return;

  const bool with_scene_dsv =
      dsv_w != 0 && dsv_h != 0 && st->tracked_w != 0 && st->tracked_h != 0 &&
      dsv_w == st->tracked_w && dsv_h == st->tracked_h;

  if (with_scene_dsv && g_arm_b_sample_active.load(std::memory_order_relaxed)) {
    const uint32_t gen = g_arm_b_sample_gen.load(std::memory_order_relaxed);
    const uint32_t armed_draws = mvProbeCbDrawsCached();
    if (st->arm_b_sample_gen != gen) {
      st->arm_b_sample_gen = gen;
      st->arm_b_draws_left = armed_draws;
      g_arm_b_draws_hint.store(armed_draws, std::memory_order_relaxed);
    }
  }

  const uint64_t present_count = g_mv_probe_present_count.load(std::memory_order_relaxed);
  const UINT n = std::min(num_rtvs, (UINT)D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT);
  for (UINT i = 0; i < n; i++) {
    if (!rtvs[i])
      continue;
    dxmt::Com<ID3D11Resource> resource;
    rtvs[i]->GetResource(&resource);
    D3D11_TEXTURE2D_DESC desc{};
    if (!resource || !resourceTex2DDesc(resource.ptr(), &desc))
      continue;

    RtCensusEntry *entry = nullptr;
    for (uint32_t e = 0; e < st->rt_census_count; e++) {
      RtCensusEntry &cur = st->rt_census[e];
      if (cur.format == desc.Format && cur.width == desc.Width &&
          cur.height == desc.Height && cur.array_size == desc.ArraySize &&
          cur.samples == desc.SampleDesc.Count) {
        entry = &cur;
        break;
      }
    }
    if (!entry) {
      if (st->rt_census_count >= kRtCensusCap) {
        st->rt_census_dropped++;
        continue;
      }
      entry = &st->rt_census[st->rt_census_count++];
      *entry = {};
      entry->format = desc.Format;
      entry->width = desc.Width;
      entry->height = desc.Height;
      entry->array_size = desc.ArraySize;
      entry->samples = desc.SampleDesc.Count;
    }
    entry->binds++;
    if (with_scene_dsv)
      entry->binds_with_scene_dsv++;
    else
      entry->binds_no_dsv++;

    const int rank = rtRank(*entry, st->tracked_w, st->tracked_h);
    if (rank <= 1) {
      MvIdentityEntry *ident = nullptr;
      for (uint32_t id = 0; id < st->mv_identity_count; id++) {
        if (st->mv_identity[id].valid && st->mv_identity[id].resource == resource.ptr()) {
          ident = &st->mv_identity[id];
          break;
        }
      }
      if (ident && (ident->format != desc.Format || ident->width != desc.Width ||
                    ident->height != desc.Height)) {
        ident->rid_reused = true;
        *ident = {};
        ident->valid = false;
        ident = nullptr;
      }
      if (!ident) {
        if (st->mv_identity_count < kMvIdentityCap) {
          ident = &st->mv_identity[st->mv_identity_count++];
          ident->resource = resource.ptr();
          ident->rid = resourceRid(resource.ptr());
          ident->format = desc.Format;
          ident->width = desc.Width;
          ident->height = desc.Height;
          ident->first_seen_present = present_count;
          ident->valid = true;
        }
      }
      if (ident) {
        ident->binds++;
        if (with_scene_dsv)
          ident->binds_with_scene_dsv++;
        else
          ident->binds_no_dsv++;
        ident->slot_mask |= (1u << i);
        if (ident->last_seen_present != present_count) {
          ident->frames_seen++;
          ident->last_seen_present = present_count;
        }
      }
    }
  }
}

void
noteSceneDrawForMvProbe(
    ID3D11Device *device, const MvProbeCbSlot vs_slots[14],
    const MvProbeCbSlot *ps_slots, float viewport_min_depth, float viewport_max_depth
) {
  if (!mvProbeEnabled()) [[unlikely]]
    return;
  if (!device || !vs_slots || !mvProbeCbEnabledCached())
    return;
  if (!g_arm_b_sample_active.load(std::memory_order_relaxed))
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  DeviceState *st = stateFor(device);
  if (!st || st->arm_b_draws_left == 0)
    return;
  st->arm_b_draws_left--;
  if (st->arm_b_draws_left == 0)
    g_arm_b_draws_hint.store(0, std::memory_order_relaxed);
  st->cam_scan.draws++;
  if (mvProbeDepthConvEnabledCached()) {
    st->depth_conv.have_vp = true;
    st->depth_conv.vp_min = viewport_min_depth;
    st->depth_conv.vp_max = viewport_max_depth;
  }
  scanStageConstantBuffers(st, vs_slots, false);
  if (ps_slots && mvProbeCbPsEnabledCached())
    scanStageConstantBuffers(st, ps_slots, true);
}

void
logMvProbeCensus(uint64_t presentation_count) {
  if (!mvProbeEnabled())
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  /* Prefer the device with the largest scene DSV / richest census. */
  DeviceState *best = nullptr;
  uint64_t best_score = 0;
  for (auto &kv : g_by_device) {
    DeviceState *st = &kv.second;
    const uint64_t score =
        (uint64_t)st->rt_census_count * 1000ull + st->cam_candidate_count +
        texArea(st->tracked_w, st->tracked_h);
    if (!best || score > best_score) {
      best = st;
      best_score = score;
    }
  }
  buildCensusLogs(best, presentation_count);
  if (best)
    resetProbePeriodCounters(best);
}

uint32_t
mvProbePreparedLogCount() {
  return g_mv_probe_log_n;
}

const char *
mvProbePreparedLogAt(uint32_t index) {
  if (index >= g_mv_probe_log_n)
    return nullptr;
  return g_mv_probe_logs[index];
}

void
mvProbeEmitPreparedLogs() {
  for (uint32_t i = 0; i < g_mv_probe_log_n; i++)
    WARN("unified-metalfx: mv_probe=", g_mv_probe_logs[i]);
  g_mv_probe_log_n = 0;
}

void
mvProbeClearPreparedLogs() {
  g_mv_probe_log_n = 0;
}

void
releaseMvProbeReadback(ID3D11Device *device) {
  if (!device)
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  DeviceState *st = stateFor(device);
  releaseMvProbeReadbackLocked(st);
}

void
noteSceneDepthClear(ID3D11DepthStencilView *dsv, UINT clear_flags, float depth) {
  if (!mvProbeEnabled() || !mvProbeDepthConvEnabledCached() || !dsv)
    return;
  if ((clear_flags & D3D11_CLEAR_DEPTH) == 0)
    return;
  ID3D11Device *key = deviceKeyFromChild(dsv);
  if (!key)
    return;
  dxmt::Com<ID3D11Resource> resource;
  dsv->GetResource(&resource);
  D3D11_TEXTURE2D_DESC desc{};
  if (!resource || !resourceTex2DDesc(resource.ptr(), &desc))
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  DeviceState *st = stateFor(key);
  if (!st || st->tracked_w == 0 || st->tracked_h == 0)
    return;
  if (desc.Width != st->tracked_w || desc.Height != st->tracked_h)
    return;
  st->depth_conv.have_clear = true;
  st->depth_conv.clear_depth = depth;
}

void
noteDepthStencilState(
    ID3D11DepthStencilState *state, ID3D11DepthStencilView *current_dsv, ID3D11Device *device
) {
  if (!mvProbeEnabled() || !mvProbeDepthConvEnabledCached() || !state || !device)
    return;
  if (!current_dsv)
    return;
  dxmt::Com<ID3D11Resource> resource;
  current_dsv->GetResource(&resource);
  D3D11_TEXTURE2D_DESC desc{};
  if (!resource || !resourceTex2DDesc(resource.ptr(), &desc))
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  DeviceState *st = stateFor(device);
  if (!st || st->tracked_w == 0 || st->tracked_h == 0)
    return;
  if (desc.Width != st->tracked_w || desc.Height != st->tracked_h)
    return;
  D3D11_DEPTH_STENCIL_DESC ddesc{};
  state->GetDesc(&ddesc);
  st->depth_conv.have_func = true;
  st->depth_conv.depth_enable = ddesc.DepthEnable != FALSE;
  st->depth_conv.depth_func = ddesc.DepthFunc;
}

void
noteMvReadbackMapTick(
    ID3D11Device *device, ID3D11DeviceContext *ctx, uint64_t presentation_count,
    UINT swapchain_w, UINT swapchain_h
) {
  if (!beginMvReadbackTick(device, ctx, presentation_count, swapchain_w, swapchain_h))
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  DeviceState *st = stateFor(device);
  if (!st)
    return;
  runMvReadbackMapTickLocked(st, ctx, presentation_count);
}

void
noteMvReadbackCopyTick(
    ID3D11Device *device, ID3D11DeviceContext *ctx, uint64_t presentation_count,
    UINT swapchain_w, UINT swapchain_h
) {
  if (!beginMvReadbackTick(device, ctx, presentation_count, swapchain_w, swapchain_h))
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  DeviceState *st = stateFor(device);
  if (!st)
    return;
  runMvReadbackCopyTickLocked(st, device, ctx, presentation_count);
}

void
noteMvProbePresent(uint64_t presentation_count) {
  if (!mvProbeEnabled()) [[unlikely]]
    return;
  g_mv_probe_present_count.store(presentation_count, std::memory_order_relaxed);
  const uint32_t period = mvProbePeriodCached();
  if (period == 0)
    return;

  /* Arm Arm B for the next Present when it is a sample tick. Draws for the
   * finishing Present already ran; sample window is the upcoming Present. */
  const uint64_t next = presentation_count + 1;
  const bool arm_next = (next % period) == 0;
  g_arm_b_sample_active.store(arm_next, std::memory_order_relaxed);
  g_arm_b_draws_hint.store(0, std::memory_order_relaxed);
  if (arm_next)
    g_arm_b_sample_gen.fetch_add(1, std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(g_lock);
  for (auto &kv : g_by_device) {
    kv.second.arm_b_draws_left = 0;
    /* Force re-arm on next scene-DSV bind for the new sample gen. */
    if (arm_next)
      kv.second.arm_b_sample_gen = 0;
  }
}

bool
resolveDepthForPresent(
    ID3D11DeviceContext *ctx, UINT color_w, UINT color_h, ID3D11Resource **out_resource,
    const char **out_provenance
) {
  if (!out_resource || !out_provenance)
    return false;
  *out_resource = nullptr;
  *out_provenance = kDepthMissing;

  logStep("resolve_depth:enter");

  ID3D11Device *key = deviceKeyFromContext(ctx);

  if (trackedDepthAllowed()) {
    std::lock_guard<std::mutex> lock(g_lock);
    DeviceState *st = key ? stateFor(key) : nullptr;
    if (st && st->tracked_depth &&
        sizeExact(color_w, color_h, st->tracked_w, st->tracked_h)) {
      *out_resource = st->tracked_depth.ref();
      *out_provenance = kDepthTracked;
      logStep("resolve_depth:tracked");
      return true;
    }
  } else {
    logStep("resolve_depth:tracked_disabled");
  }

  if (omDepthAllowed() && ctx) {
    logStep("resolve_depth:om_get_enter");
    ID3D11DepthStencilView *dsv = nullptr;
    ctx->OMGetRenderTargets(0, nullptr, &dsv);
    logStep("resolve_depth:om_get_exit");
    if (dsv) {
      dxmt::Com<ID3D11Resource> resource;
      dsv->GetResource(&resource);
      dsv->Release();
      D3D11_TEXTURE2D_DESC desc{};
      if (resource && resourceTex2DDesc(resource.ptr(), &desc) &&
          sizeExact(color_w, color_h, desc.Width, desc.Height)) {
        *out_resource = resource.ref();
        *out_provenance = kDepthOmAtPresent;
        logStep("resolve_depth:om_at_present");
        return true;
      }
    }
  } else if (!omDepthAllowed()) {
    logStep("resolve_depth:om_disabled");
  }

  if (key) {
    std::lock_guard<std::mutex> lock(g_lock);
    DeviceState *st = stateFor(key);
    if (st && st->title_depth) {
      D3D11_TEXTURE2D_DESC desc{};
      if (resourceTex2DDesc(st->title_depth.ptr(), &desc) &&
          sizeExact(color_w, color_h, desc.Width, desc.Height)) {
        *out_resource = st->title_depth.ref();
        *out_provenance = kDepthTitleTemporal;
        logStep("resolve_depth:title_temporal");
        return true;
      }
    }
  }

  logStep("resolve_depth:missing");
  return false;
}

bool
resolveTitleMotion(
    ID3D11Device *device, ID3D11Resource **out_motion, const char **out_provenance
) {
  if (!out_motion || !out_provenance)
    return false;
  *out_motion = nullptr;
  *out_provenance = kMotionUnavailable;
  if (!device)
    return false;
  std::lock_guard<std::mutex> lock(g_lock);
  auto it = g_by_device.find(device);
  if (it == g_by_device.end() || !it->second.title_motion_valid || !it->second.title_motion)
    return false;
  *out_motion = it->second.title_motion.ref();
  *out_provenance = kMotionTitle;
  return true;
}

bool
isLabeledEstimate(const char *provenance) {
  if (!provenance)
    return false;
  return provenance[0] == 'E' && provenance[1] == 'S' && provenance[2] == 'T' &&
         provenance[3] == 'I' && provenance[4] == 'M' && provenance[5] == 'A' &&
         provenance[6] == 'T' && provenance[7] == 'E' && provenance[8] == '_';
}

void
logPresentTick(
    uint64_t presentation_count, UINT color_w, UINT color_h, bool prev_color_valid,
    const char *depth_provenance, const char *motion_provenance, DXGI_FORMAT depth_format,
    const char *encode_label
) {
  if (!verbose() && presentation_count > 3 && (presentation_count % 300) != 0)
    return;
  auto temporal = temporalUpscaleCount();
  auto unified = unifiedTemporalEncodeCount();
  auto fi_n = unifiedFiPresentCount();
  const UINT up_w = lastUnifiedPresentWidth();
  const UINT up_h = lastUnifiedPresentHeight();
  const bool depth_est = isLabeledEstimate(depth_provenance);
  const bool motion_est = isLabeledEstimate(motion_provenance);
  const char *truth =
      (depth_est || motion_est) ? "LABELED_ESTIMATE_NOT_ENGINE_TRUTH" : "real_or_missing";
  const char *encode =
      (encode_label && encode_label[0]) ? encode_label : "spatial_encode=color_only";
  WARN(
      "present-dm: Present#", presentation_count, " color=", color_w, "x", color_h,
      " prev_color=", prevValidLabel(prev_color_valid),
      " current_color=REAL_COLOR_CURRENT", " depth=", depth_provenance,
      " depth_fmt=", (uint32_t)depth_format, " motion=", motion_provenance,
      " provenance_truth=", truth,
      " temporal_upscale_count=", temporal,
      " title_temporal_silent=", (temporal == 0) ? "yes" : "no",
      " unified_temporal_encode_count=", unified,
      " unified_fi_present_count=", fi_n,
      " last_unified_present=", up_w, "x", up_h,
      " ", encode,
      " (also check nvngx.log for EvaluateFeature; expected silence on ETS2 Picture)"
  );
}

} // namespace present_dm
} // namespace dxmt
