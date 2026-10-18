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

/** TemporalUpscale / title path count visible inside d3d11.dll (Present proof). */
uint64_t temporalUpscaleCount();

/** Track last immediate-context scene DSV for this device (may be unbound at Present). */
void trackSceneDepth(ID3D11DepthStencilView *dsv);

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
