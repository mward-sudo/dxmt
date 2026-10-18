#include "present_depth_motion.hpp"
#include "com/com_pointer.hpp"
#include "log/log.hpp"
#include "util_env.hpp"
#include <atomic>
#include <mutex>

namespace dxmt {
namespace present_dm {

namespace {

std::atomic<uint64_t> g_temporal_upscale_count{0};

std::mutex g_lock;
dxmt::Com<ID3D11Resource> g_tracked_depth;
UINT g_tracked_w = 0;
UINT g_tracked_h = 0;

dxmt::Com<ID3D11Resource> g_title_motion;
dxmt::Com<ID3D11Resource> g_title_depth;
bool g_title_motion_valid = false;

bool
envIsZero(const char *name) {
  return env::getEnvVar(name) == "0";
}

bool
envIsOne(const char *name) {
  return env::getEnvVar(name) == "1";
}

bool
sizeCompatible(UINT color_w, UINT color_h, UINT depth_w, UINT depth_h) {
  if (depth_w == 0 || depth_h == 0 || color_w == 0 || color_h == 0)
    return false;
  if (depth_w == color_w && depth_h == color_h)
    return true;
  if (depth_w * 2 == color_w && depth_h * 2 == color_h)
    return true;
  if (color_w * 2 == depth_w && color_h * 2 == depth_h)
    return true;
  return false;
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
  // Default on; DXMT_PRESENT_DM_TRACKED_DEPTH=0 disables for bisection.
  return !envIsZero("DXMT_PRESENT_DM_TRACKED_DEPTH");
}

bool
omDepthAllowed() {
  // Default on; DXMT_PRESENT_DM_OM_DEPTH=0 disables for bisection.
  return !envIsZero("DXMT_PRESENT_DM_OM_DEPTH");
}

} // namespace

bool
enabled() {
  // Bisection: DXMT_PRESENT_DM_EXPORT=0 kills all Present export (no OMGet,
  // no tracked globals, no estimate fill) even when unified temporal is armed.
  if (envIsZero("DXMT_PRESENT_DM_EXPORT"))
    return false;
  // Opt-in so daily spatial Present is unchanged unless proof/fork trial sets the env.
  // Unified temporal Present implies the same export (map: DXMT_METALFX_UNIFIED_TEMPORAL_INTERP).
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
noteTemporalUpscale(
    ID3D11Resource *color, ID3D11Resource *depth, ID3D11Resource *motion_vector
) {
  auto n = g_temporal_upscale_count.fetch_add(1, std::memory_order_relaxed) + 1;
  WARN(
      "present-dm: TemporalUpscale fired (count=", n, ") — title/NGX path has depth+motion"
  );
  std::lock_guard<std::mutex> lock(g_lock);
  g_title_depth = depth;
  g_title_motion = motion_vector;
  g_title_motion_valid = motion_vector != nullptr;
  (void)color;
}

uint64_t
temporalUpscaleCount() {
  return g_temporal_upscale_count.load(std::memory_order_relaxed);
}

void
trackSceneDepth(ID3D11DepthStencilView *dsv) {
  // EXPORT=0 / disabled: do not touch process-global tracked depth.
  if (!dsv || !enabled() || !trackedDepthAllowed())
    return;
  dxmt::Com<ID3D11Resource> resource;
  dsv->GetResource(&resource);
  if (!resource)
    return;
  D3D11_TEXTURE2D_DESC desc{};
  if (!resourceTex2DDesc(resource.ptr(), &desc))
    return;
  std::lock_guard<std::mutex> lock(g_lock);
  // Prefer largest DSV seen (scene depth usually out-areas shadow maps).
  if (g_tracked_depth && texArea(desc.Width, desc.Height) < texArea(g_tracked_w, g_tracked_h))
    return;
  g_tracked_depth = resource.ptr();
  g_tracked_w = desc.Width;
  g_tracked_h = desc.Height;
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

  // Prefer fork-tracked scene depth (OM often unbound / wrong after post).
  if (trackedDepthAllowed()) {
    std::lock_guard<std::mutex> lock(g_lock);
    if (g_tracked_depth && sizeCompatible(color_w, color_h, g_tracked_w, g_tracked_h)) {
      *out_resource = g_tracked_depth.ref();
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
          sizeCompatible(color_w, color_h, desc.Width, desc.Height)) {
        *out_resource = resource.ref();
        *out_provenance = kDepthOmAtPresent;
        logStep("resolve_depth:om_at_present");
        return true;
      }
    }
  } else if (!omDepthAllowed()) {
    logStep("resolve_depth:om_disabled");
  }

  {
    std::lock_guard<std::mutex> lock(g_lock);
    if (g_title_depth) {
      D3D11_TEXTURE2D_DESC desc{};
      if (resourceTex2DDesc(g_title_depth.ptr(), &desc) &&
          sizeCompatible(color_w, color_h, desc.Width, desc.Height)) {
        *out_resource = g_title_depth.ref();
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
resolveTitleMotion(ID3D11Resource **out_motion, const char **out_provenance) {
  if (!out_motion || !out_provenance)
    return false;
  *out_motion = nullptr;
  *out_provenance = kMotionUnavailable;
  std::lock_guard<std::mutex> lock(g_lock);
  if (!g_title_motion_valid || !g_title_motion)
    return false;
  *out_motion = g_title_motion.ref();
  *out_provenance = kMotionTitle;
  return true;
}

bool
isLabeledEstimate(const char *provenance) {
  if (!provenance)
    return false;
  // ESTIMATE_* labels are fork-synthesized; never engine / title truth.
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
  // Default: first 3 + every 300th. VERBOSE=1 closes the blind window before crashes.
  if (!verbose() && presentation_count > 3 && (presentation_count % 300) != 0)
    return;
  auto temporal = temporalUpscaleCount();
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
      " ", encode,
      " (also check nvngx.log for EvaluateFeature; expected silence on ETS2 Picture)"
  );
}

} // namespace present_dm
} // namespace dxmt
