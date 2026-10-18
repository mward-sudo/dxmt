#include "present_depth_motion.hpp"
#include "com/com_pointer.hpp"
#include "log/log.hpp"
#include "util_env.hpp"
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace dxmt {
namespace present_dm {

namespace {

std::atomic<uint64_t> g_temporal_upscale_count{0};
std::atomic<uint64_t> g_unified_temporal_encode_count{0};
std::atomic<UINT> g_last_unified_present_w{0};
std::atomic<UINT> g_last_unified_present_h{0};

/** Per-device Present-dm state (tip 15). Key is raw ID3D11Device*; cleared in
 *  clearDevice on teardown — do not AddRef the map key. */
struct DeviceState {
  dxmt::Com<ID3D11Resource> tracked_depth;
  UINT tracked_w = 0;
  UINT tracked_h = 0;
  dxmt::Com<ID3D11Resource> title_motion;
  dxmt::Com<ID3D11Resource> title_depth;
  bool title_motion_valid = false;
};

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
      " last_unified_present=", up_w, "x", up_h,
      " ", encode,
      " (also check nvngx.log for EvaluateFeature; expected silence on ETS2 Picture)"
  );
}

} // namespace present_dm
} // namespace dxmt
