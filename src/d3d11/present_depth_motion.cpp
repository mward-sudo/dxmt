#include "present_depth_motion.hpp"
#include "com/com_pointer.hpp"
#include "d3d11_resource.hpp"
#include "dxmt_buffer.hpp"
#include "log/log.hpp"
#include "util_env.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

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
constexpr size_t kMvProbeLogCap = 24;
constexpr size_t kMvProbeLogLen = 256;

struct RtCensusEntry {
  DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
  UINT width = 0;
  UINT height = 0;
  UINT array_size = 0;
  UINT samples = 0;
  uint64_t binds = 0;
  uint64_t binds_with_scene_dsv = 0;
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
  bool cb_not_readable_logged = false;
};

/** Tip 27a: true while the upcoming/current Present is an Arm B sample tick. */
std::atomic<bool> g_arm_b_sample_active{false};
std::atomic<uint32_t> g_arm_b_sample_gen{0};
/** Fast-path hint for PreDraw: remaining Arm B scans this sample window. */
std::atomic<uint32_t> g_arm_b_draws_hint{0};

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
  if (isTwoChannelMvFormat(e.format) && scene_sz && e.binds_with_scene_dsv > 0)
    return 0; /* mv_strong */
  if (isHalf4Format(e.format) && scene_sz && e.binds_with_scene_dsv > 0)
    return 1; /* mv_possible */
  return 2;   /* other */
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
projectionLike(const float m[4][4], float scene_aspect, float *out_fovy, float *out_aspect,
               float *out_near, float *out_far, bool *out_reversed, bool *out_kind_vp) {
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++)
      if (!isFiniteBounded(m[r][c]))
        return false;

  /* |m23|≈1; m03,m13,m33≈0 (D3D row-vector projection). */
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
  if (scene_aspect > 0.0f) {
    const float rel = std::fabs(aspect - scene_aspect) / scene_aspect;
    if (rel > 0.01f)
      return false;
  }

  const float fovy = 2.0f * std::atan(1.0f / m[1][1]);
  const float near_z = -m[3][2] / m[2][2];
  const float far_z = -m[3][2] / (m[2][2] - 1.0f);
  if (!std::isfinite(fovy) || !std::isfinite(near_z) || !std::isfinite(far_z))
    return false;

  /* Bare projection: m00/m11 alone on the diagonal of the upper-left 3×3. */
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
storeCamCandidate(DeviceState *st, UINT slot, UINT offset, bool row_major, bool kind_vp,
                  float fovy, float aspect, float near_z, float far_z, bool reversed,
                  float m20, float m21) {
  for (uint32_t i = 0; i < st->cam_candidate_count; i++) {
    CamCandidate &c = st->cam_candidates[i];
    if (c.slot == slot && c.offset == offset && c.row_major == row_major) {
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
scanVsConstantBuffers(DeviceState *st, const MvProbeCbSlot slots[14]) {
  if (!st || !mvProbeCbEnabledCached())
    return;
  const float scene_aspect =
      (st->tracked_w > 0 && st->tracked_h > 0)
          ? (float)st->tracked_w / (float)st->tracked_h
          : 0.0f;

  for (UINT slot = 0; slot < 14; slot++) {
    const MvProbeCbSlot &cb = slots[slot];
    if (!cb.buffer)
      continue;
    D3D11ResourceCommon *res = GetResourceCommon(cb.buffer);
    if (!res)
      continue;
    const Rc<Buffer> &buf = res->buffer();
    if (!buf)
      continue;
    BufferAllocation *alloc = buf->current();
    if (!alloc || alloc->flags().test(BufferAllocationFlag::GpuManaged) ||
        alloc->mappedMemory(0) == nullptr) {
      if (!st->cb_not_readable_logged) {
        st->cb_not_readable_logged = true;
        WARN("unified-metalfx: mv_probe=skip_reason=cb_not_cpu_readable");
      }
      continue;
    }

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
    if (read_len < 64)
      continue;

    const uint32_t sub = alloc->currentSuballocation();
    const char *base =
        reinterpret_cast<const char *>(alloc->mappedMemory(sub)) + start;

    for (UINT off = 0; off + 64 <= read_len; off += 16) {
      float raw[16];
      std::memcpy(raw, base + off, sizeof(raw));
      for (int major = 0; major < 2; major++) {
        const bool row_major = major == 0;
        float m[4][4];
        loadMat4(raw, row_major, m);
        float fovy = 0, aspect = 0, near_z = 0, far_z = 0;
        bool reversed = false, kind_vp = false;
        if (!projectionLike(m, scene_aspect, &fovy, &aspect, &near_z, &far_z, &reversed,
                            &kind_vp))
          continue;
        storeCamCandidate(
            st, slot, start + off, row_major, kind_vp, fovy, aspect, near_z, far_z,
            reversed, m[2][0], m[2][1]
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
        "%s fmt=%u %ux%u binds=%llu with_dsv=%llu presents_seen=%llu", rtRankName(rank),
        (unsigned)e.format, e.width, e.height, (unsigned long long)e.binds,
        (unsigned long long)e.binds_with_scene_dsv,
        (unsigned long long)presentation_count
    );
    pushMvProbeLog(line);
  }
  if (!any_strong)
    pushMvProbeLog("mv_strong=none");

  bool any_cam = false;
  for (uint32_t i = 0; i < st->cam_candidate_count; i++) {
    const CamCandidate &c = st->cam_candidates[i];
    if (!c.valid)
      continue;
    any_cam = true;
    std::snprintf(
        line, sizeof(line),
        "cam_candidate kind=%s stage=vs slot=%u off=%u major=%s fovy=%.2frad "
        "aspect=%.3f near=%.3f far=%.1f reversed=%s jitter=(%.5f,%.5f)",
        c.kind_vp ? "vp" : "proj", c.slot, c.offset, c.row_major ? "row" : "col",
        c.fovy, c.aspect, c.near_z, c.far_z, c.reversed ? "yes" : "no", c.jitter_x,
        c.jitter_y
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
clearDevice(ID3D11Device *device) {
  if (!device)
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  g_by_device.erase(device);
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
    if (st->arm_b_sample_gen != gen) {
      st->arm_b_sample_gen = gen;
      st->arm_b_draws_left = 2;
      g_arm_b_draws_hint.store(2, std::memory_order_relaxed);
    }
  }

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
  }
}

void
noteSceneDrawForMvProbe(ID3D11Device *device, const MvProbeCbSlot slots[14]) {
  if (!mvProbeEnabled()) [[unlikely]]
    return;
  if (!device || !slots || !mvProbeCbEnabledCached())
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
  scanVsConstantBuffers(st, slots);
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
noteMvProbePresent(uint64_t presentation_count) {
  if (!mvProbeEnabled()) [[unlikely]]
    return;
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
