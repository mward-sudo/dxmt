/*
 * Fork helper: Present-path color / depth / motion export for MetalFX Frame
 * interpolation follow-on.
 *
 * Prefer real inputs. After real motion (and depth, if also missing) fail,
 * allow an explicitly labeled estimate — never unlabeled / invented MVs.
 *
 * Gate: DXMT_PRESENT_DEPTH_MOTION=1 (opt-in; default off so stock spatial
 * Present is unchanged). Spatial encode stays color-only until a follow-on
 * replaces it; this path only retains synchronized buffers + provenance logs.
 */
#pragma once

#include "d3d11.h"
#include <cstdint>

namespace dxmt {
namespace present_dm {

/** Env DXMT_PRESENT_DEPTH_MOTION=1 enables; unset/0 leaves Present untouched. */
bool enabled();

void noteTemporalUpscale(
    ID3D11Resource *color, ID3D11Resource *depth, ID3D11Resource *motion_vector
);

/** TemporalUpscale / title path count visible inside d3d11.dll (Present proof). */
uint64_t temporalUpscaleCount();

/** Track last immediate-context scene DSV (may be unbound at Present). */
void trackSceneDepth(ID3D11DepthStencilView *dsv);

/**
 * Resolve a depth resource for Present export.
 * Prefer last tracked scene DSV (size-compatible with color); else OM DSV at
 * Present; else title TemporalUpscale depth. Returns AddRef'd resource when true.
 */
bool resolveDepthForPresent(
    ID3D11DeviceContext *ctx, UINT color_w, UINT color_h, ID3D11Resource **out_resource,
    const char **out_provenance
);

/**
 * If TemporalUpscale / NGX supplied motion this process, return last MV resource
 * (AddRef'd). Otherwise false → caller may try labeled estimate, else report
 * REAL_MOTION_UNAVAILABLE.
 */
bool resolveTitleMotion(ID3D11Resource **out_motion, const char **out_provenance);

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

void logPresentTick(
    uint64_t presentation_count, UINT color_w, UINT color_h, bool prev_color_valid,
    const char *depth_provenance, const char *motion_provenance, DXGI_FORMAT depth_format
);

} // namespace present_dm
} // namespace dxmt
