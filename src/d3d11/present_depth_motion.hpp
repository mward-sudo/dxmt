/*
 * Fork helper: Present-path color / depth / motion export for MetalFX Frame
 * interpolation follow-on.
 *
 * Prefer real inputs. After real motion (and depth, if also missing) fail,
 * allow an explicitly labeled estimate — never unlabeled / invented MVs.
 *
 * Gates (either enables export; default off so stock spatial Present is unchanged):
 *   DXMT_PRESENT_DEPTH_MOTION=1 — export + provenance only (spatial encode stays
 *     color-only unless the unified path is also selected at CreateSwapChain).
 *   DXMT_METALFX_UNIFIED_TEMPORAL_INTERP=1 — implies the same export so an
 *     attachable temporal scaler can bind color/depth/motion on Present.
 *
 * Diagnostic kill-switches / verbosity (ticket/12):
 *   DXMT_PRESENT_DM_EXPORT=0 — force export off even when unified is armed.
 *   DXMT_PRESENT_DM_VERBOSE=1 — log every Present (not just #1–3 / %300).
 *   DXMT_PRESENT_DM_STEPS=1 — step markers inside resolve/export (also on with VERBOSE).
 *   DXMT_PRESENT_DM_TRACKED_DEPTH=0 — never use per-device tracked depth.
 *   DXMT_PRESENT_DM_OM_DEPTH=0 — never call OMGetRenderTargets at Present.
 *
 * Tip 15: tracked/title depth+motion are scoped per ID3D11Device (not process
 * globals) and cleared on device teardown. Size match for bind is exact.
 */
#pragma once

#include "d3d11.h"
#include <cstdint>

namespace dxmt {
namespace present_dm {

/**
 * True when DXMT_PRESENT_DEPTH_MOTION=1 or DXMT_METALFX_UNIFIED_TEMPORAL_INTERP=1,
 * unless DXMT_PRESENT_DM_EXPORT=0 forces export fully off (bisection kill-switch).
 */
bool enabled();

/** DXMT_PRESENT_DM_VERBOSE=1 — every Present tick is logged. */
bool verbose();

/** Step markers when VERBOSE=1 or DXMT_PRESENT_DM_STEPS=1. */
bool stepsEnabled();

/** Emit a Present-path step marker (no-op unless stepsEnabled()). */
void logStep(const char *step);

/** Drop per-device tracked/title state (call from device teardown). */
void clearDevice(ID3D11Device *device);

void noteTemporalUpscale(
    ID3D11Resource *color, ID3D11Resource *depth, ID3D11Resource *motion_vector
);

/** TemporalUpscale / title path count visible inside d3d11.dll (NGX/title proof). */
uint64_t temporalUpscaleCount();

/**
 * Tip 16: count unified Present-path temporal encodes that presented the
 * upscaled backbuffer (not NGX title EvaluateFeature; not pre-upscale fallback).
 */
void noteUnifiedTemporalPresent(UINT output_w, UINT output_h);
uint64_t unifiedTemporalEncodeCount();
UINT lastUnifiedPresentWidth();
UINT lastUnifiedPresentHeight();

/** Tip 18: count unified Present ticks that paced interpolated + upscaled. */
void noteUnifiedFiPresent(UINT output_w, UINT output_h);
uint64_t unifiedFiPresentCount();

/** Track last immediate-context scene DSV for this device (may be unbound at Present). */
void trackSceneDepth(ID3D11DepthStencilView *dsv);

/**
 * Tip 27a: default-off MV source census (`DXMT_MV_PROBE=1`). Log-only — never
 * binds motion, never changes provenance, never opens the tip-19 FI gate.
 */
bool mvProbeEnabled();
/** True during a sampled Present while Arm B may still scan draws. */
bool mvProbeArmBActive();
/** Optional PS-stage CB scan (`DXMT_MV_PROBE_CB_PS`, default off). */
bool mvProbeCbPsEnabled();
/** Presents between census dumps; `DXMT_MV_PROBE_PERIOD`, default 600. */
uint32_t mvProbePeriod();
/** Arm A: aggregate RTV binds at OMSetRenderTargets (desc metadata only). */
void noteRenderTargetBind(
    ID3D11RenderTargetView *const *rtvs, UINT num_rtvs, ID3D11DepthStencilView *dsv
);
/**
 * Arm B/B′: structural VS (and optional PS) CB projection scan. Call from
 * PreDraw with bound CB slots; scans at most DXMT_MV_PROBE_CB_DRAWS armed draws.
 */
struct MvProbeCbSlot {
  ID3D11Resource *buffer = nullptr;
  UINT first_constant = 0;
  UINT num_constants = 0;
};
void noteSceneDrawForMvProbe(
    ID3D11Device *device, const MvProbeCbSlot vs_slots[14],
    const MvProbeCbSlot *ps_slots, float viewport_min_depth, float viewport_max_depth
);
/** Advance Present tick: arm Arm B for the next sampled Present. */
void noteMvProbePresent(uint64_t presentation_count);
/** Arm D: rate-limited staging readback (`DXMT_MV_PROBE_RB=1`). */
bool mvReadbackEnabled();
/** Map/consume prior RB staging (no GPU copies). Run after temporal emitcc. */
void noteMvReadbackMapTick(
    ID3D11Device *device, ID3D11DeviceContext *ctx, uint64_t presentation_count,
    UINT swapchain_w, UINT swapchain_h
);
/** Queue CopySubresourceRegion RB samples — must run after temporal emitcc. */
void noteMvReadbackCopyTick(
    ID3D11Device *device, ID3D11DeviceContext *ctx, uint64_t presentation_count,
    UINT swapchain_w, UINT swapchain_h
);
/** Release readback staging / held refs (device teardown or swapchain resize). */
void releaseMvProbeReadback(ID3D11Device *device);
/** Arm E: depth convention probes (metadata only). */
void noteSceneDepthClear(
    ID3D11DepthStencilView *dsv, UINT clear_flags, float depth
);
void noteDepthStencilState(
    ID3D11DepthStencilState *state, ID3D11DepthStencilView *current_dsv,
    ID3D11Device *device
);
/** Clear prepared probe log buffer after swapchain emission. */
void mvProbeClearPreparedLogs();
/**
 * Build period-aggregated `mv_probe=` lines and reset census counters.
 * Caller emits via swapchain LogMvProbeStrategy (or mvProbeEmitPreparedLogs).
 */
void logMvProbeCensus(uint64_t presentation_count);
uint32_t mvProbePreparedLogCount();
const char *mvProbePreparedLogAt(uint32_t index);
/** Fallback emitter when the swapchain helper is unavailable. */
void mvProbeEmitPreparedLogs();

/**
 * Resolve a depth resource for Present export.
 * Prefer last tracked scene DSV (exact size match with color); else OM DSV at
 * Present; else title TemporalUpscale depth. Returns AddRef'd resource when true.
 * Kill-switches: DXMT_PRESENT_DM_TRACKED_DEPTH=0 / DXMT_PRESENT_DM_OM_DEPTH=0.
 */
bool resolveDepthForPresent(
    ID3D11DeviceContext *ctx, UINT color_w, UINT color_h, ID3D11Resource **out_resource,
    const char **out_provenance
);

/**
 * If TemporalUpscale / NGX supplied motion for this device, return last MV
 * resource (AddRef'd). Otherwise false → caller may try labeled estimate.
 */
bool resolveTitleMotion(
    ID3D11Device *device, ID3D11Resource **out_motion, const char **out_provenance
);

/** True when provenance is an ESTIMATE_* label (not engine / title truth). */
bool isLabeledEstimate(const char *provenance);

/* Real provenance */
constexpr const char *kMotionUnavailable = "REAL_MOTION_UNAVAILABLE";
constexpr const char *kDepthOmAtPresent = "REAL_DEPTH_OM_AT_PRESENT";
constexpr const char *kDepthTracked = "REAL_DEPTH_TRACKED";
constexpr const char *kDepthTitleTemporal = "REAL_DEPTH_TITLE_TEMPORAL";
constexpr const char *kDepthResolvedD32S8 = "REAL_DEPTH_RESOLVED_D32S8";
constexpr const char *kDepthMissing = "REAL_DEPTH_MISSING";
constexpr const char *kMotionTitle = "REAL_MOTION_TITLE_TEMPORAL";

/* Labeled estimates — only after real path failed; never treat as engine truth */
constexpr const char *kMotionEstimateZero = "ESTIMATE_MOTION_ZERO";
constexpr const char *kMotionEstimateFailed = "ESTIMATE_MOTION_FAILED";
constexpr const char *kDepthEstimateConstantFar = "ESTIMATE_DEPTH_CONSTANT_FAR";
constexpr const char *kDepthEstimateFailed = "ESTIMATE_DEPTH_FAILED";

/**
 * @param encode_label Honest Present encode path, e.g.
 *   "spatial_encode=color_only" or "temporal_encode=color+depth+motion_bound".
 */
void logPresentTick(
    uint64_t presentation_count, UINT color_w, UINT color_h, bool prev_color_valid,
    const char *depth_provenance, const char *motion_provenance, DXGI_FORMAT depth_format,
    const char *encode_label
);

} // namespace present_dm
} // namespace dxmt
