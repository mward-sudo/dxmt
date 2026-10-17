#include "d3d11_swapchain.hpp"
#include "com/com_guid.hpp"
#include "config/config.hpp"
#include "d3d11_private.h"
#include "dxgi_interfaces.h"
#include "dxgi_object.hpp"
#include "dxgi_output.hpp"
#include "d3d11_context.hpp"
#include "dxmt_context.hpp"
#include "dxmt_hud_state.hpp"
#include "dxmt_statistics.hpp"
#include "dxmt_presenter.hpp"
#include "log/log.hpp"
#include "d3d11_resource.hpp"
#include "d3d11_device.hpp"
#include "util_cpu_fence.hpp"
#include "util_env.hpp"
#include "util_string.hpp"
#include "util_win32_compat.h"
#include "util_hotpatch.h"
#include "wsi_monitor.hpp"
#include "wsi_platform_win32.hpp"
#include "wsi_window.hpp"
#include "dxmt_info.hpp"
#include "dxmt_presenter.hpp"
#include "present_depth_motion.hpp"
#include <atomic>
#include <cfloat>
#include <format>

/**
Ref: https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_3/nf-dxgi1_3-idxgiswapchain2-setmaximumframelatency
This value is 1 by default.
*/
constexpr size_t kSwapchainLatency = 1;

namespace dxmt {

WMTPixelFormat ConvertSwapChainFormat(DXGI_FORMAT format) {
  switch (format) {
  case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
  case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    return WMTPixelFormatBGRA8Unorm_sRGB;
  case DXGI_FORMAT_B8G8R8A8_UNORM:
  case DXGI_FORMAT_R8G8B8A8_UNORM:
    return WMTPixelFormatBGRA8Unorm;
  case DXGI_FORMAT_R10G10B10A2_UNORM:
    return WMTPixelFormatRGB10A2Unorm;
  case DXGI_FORMAT_R16G16B16A16_FLOAT:
    return WMTPixelFormatRGBA16Float;
  default:
    return WMTPixelFormatInvalid;
  }
}

WMTColorSpace ConvertColorSpace(DXGI_COLOR_SPACE_TYPE color_space, bool hdr) {
  switch (color_space) {
  case DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709:
    return WMTColorSpaceSRGB;
  case DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709:
    return hdr ? WMTColorSpaceHDR_scRGB : WMTColorSpaceSRGBLinear;
  case DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020:
    return WMTColorSpaceHDR_PQ;
  case DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P2020:
    return WMTColorSpaceBT2020;
  default:
    return WMTColorSpaceInvalid;
  }
}

/**
 FIXME: duplicated implementation in dxgi_output.cpp
*/
uint32_t
GetMonitorFormatBpp(DXGI_FORMAT Format) {
  switch (Format) {
  case DXGI_FORMAT_R8G8B8A8_UNORM:
  case DXGI_FORMAT_B8G8R8A8_UNORM:
  case DXGI_FORMAT_B8G8R8X8_UNORM:
  case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
  case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
  case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
  case DXGI_FORMAT_R10G10B10A2_UNORM:
  case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
    return 32;

  case DXGI_FORMAT_R16G16B16A16_FLOAT:
    return 64;

  default:
    Logger::warn(str::format("GetMonitorFormatBpp: Unknown format: ", Format));
    return 32;
  }
}

class ModeSetGuard {
  std::atomic_flag in_progress_;
public:
  class ModeSetInProgress {
    ModeSetGuard &guard_;
    bool protected_;

  public:
    ModeSetInProgress(ModeSetGuard &guard) : guard_(guard) {
      protected_ = guard_.in_progress_.test_and_set();
    }
    ~ModeSetInProgress() {
      if (!protected_) {
        guard_.in_progress_.clear();
      }
    }
    operator bool() {
      return protected_;
    }
  };
};

/**
 * Present MetalFX mode:
 *   Off — stock Present (no MetalFX upscale)
 *   Spatial — DXMT_METALFX_SPATIAL_SWAPCHAIN (interpolation-off Picture)
 *   Temporal — DXMT_METALFX_UNIFIED_TEMPORAL_INTERP (attachable temporal scaler;
 *     FrameInterpolator is a later ticket — not wired here)
 */
enum class SwapChainMetalFXMode {
  Off = 0,
  Spatial = 1,
  Temporal = 2,
};

template <SwapChainMetalFXMode MetalFX>
class MTLD3D11SwapChain final : public MTLDXGISubObject<IDXGISwapChain4, MTLD3D11Device> {
  static constexpr bool kUpscale = MetalFX != SwapChainMetalFXMode::Off;
  static constexpr bool kSpatial = MetalFX == SwapChainMetalFXMode::Spatial;
  static constexpr bool kTemporal = MetalFX == SwapChainMetalFXMode::Temporal;

public:
  MTLD3D11SwapChain(
      IDXGIFactory1 *pFactory, MTLD3D11Device *pDevice, IMTLDXGIDevice *pDXGIDevice,
       HWND hWnd, const DXGI_SWAP_CHAIN_DESC1 *pDesc,
      const DXGI_SWAP_CHAIN_FULLSCREEN_DESC *pFullscreenDesc
  ) :
      MTLDXGISubObject(pDevice),
      factory_(pFactory),
      dxgi_device_(pDXGIDevice),
      presentation_count_(0),
      desc_(*pDesc),
      device_context_(pDevice->GetImmediateContextPrivate()),
      hWnd(hWnd),
      monitor_(wsi::getWindowMonitor(hWnd)),
      hud(WMT::DeveloperHUDProperties::instance()) {

    native_view_ = WMT::CreateMetalViewFromHWND((intptr_t)hWnd, pDevice->GetMTLDevice(), layer_weak_);

    if (!native_view_) {
      ERR("Failed to create metal view, it seems like your Wine has no exported symbols needed by DXMT.");
      abort();
    }

    if constexpr (kUpscale) {
      scale_factor = std::max(Config::getInstance().getOption<float>("d3d11.metalSpatialUpscaleFactor", 2), 1.0f);
    }

    presenter = Rc(new Presenter(pDevice->GetMTLDevice(), layer_weak_,
                                 pDevice->GetDXMTDevice().queue().cmd_library,
                                 scale_factor, desc_.SampleDesc.Count));

    frame_latency = kSwapchainLatency;
    present_semaphore_ = CreateSemaphore(nullptr, frame_latency,
                                         DXGI_MAX_SWAP_CHAIN_BUFFERS, nullptr);

    // without this flag, there is still a DXGIDevice level of frame latency control
    if (desc_.Flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT) {
      frame_latency_fence_ = std::make_unique<CpuFence>();
    }

    if (desc_.Width == 0 || desc_.Height == 0) {
      wsi::getWindowSize(hWnd, &desc_.Width, &desc_.Height);
    }

    if (pFullscreenDesc) {
      fullscreen_desc_ = *pFullscreenDesc;
    } else {
      fullscreen_desc_.Windowed = true;
    }

    preferred_max_frame_rate =
        Config::getInstance().getOption<int>("d3d11.preferredMaxFrameRate", 0);
    wsi::WsiMode current_mode;
    if (wsi::getCurrentDisplayMode(monitor_, &current_mode) &&
        current_mode.refreshRate.denominator != 0 &&
        current_mode.refreshRate.numerator != 0) {
      init_refresh_rate_ = (double)current_mode.refreshRate.numerator /
                           (double)current_mode.refreshRate.denominator;
    }

    handle_alt_tab_ = Config::getInstance().getOption<bool>("dxgi.handleAltTab", false);

    hud.initialize(GetVersionDescriptionText(device_->GetDirectXVersion(), device_->GetFeatureLevel()));

    backbuffer_desc_ = D3D11_TEXTURE2D_DESC1 {
      .Width = desc_.Width,
      .Height = desc_.Height,
      .MipLevels = 1,
      .ArraySize = 1,
      .Format = desc_.Format,
      .SampleDesc = desc_.SampleDesc,
      .Usage = D3D11_USAGE_DEFAULT,
      .BindFlags = D3D11_BIND_RENDER_TARGET,
      .CPUAccessFlags = {},
      .MiscFlags = {},
      .TextureLayout = {},
    };
    
    if (desc_.BufferUsage & DXGI_USAGE_SHADER_INPUT)
      backbuffer_desc_.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
    if (desc_.BufferUsage & DXGI_USAGE_UNORDERED_ACCESS)
      backbuffer_desc_.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;

    // FIXME: check HRESULT!
    ResizeBuffers(0, desc_.Width, desc_.Height, DXGI_FORMAT_UNKNOWN, desc_.Flags);
    if (!fullscreen_desc_.Windowed)
      EnterFullscreenMode(nullptr);
  };

  ~MTLD3D11SwapChain() {
    device_context_->WaitUntilGPUIdle();
    WMT::ReleaseMetalView(native_view_);
    native_view_ = {};
    CloseHandle(present_semaphore_);
  };

  HRESULT
  STDMETHODCALLTYPE
  QueryInterface(REFIID riid, void **ppvObject) final {
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) ||
        riid == __uuidof(IDXGIDeviceSubObject) ||
        riid == __uuidof(IDXGISwapChain) || riid == __uuidof(IDXGISwapChain1) ||
        riid == __uuidof(IDXGISwapChain2) ||
        riid == __uuidof(IDXGISwapChain3) ||
        riid == __uuidof(IDXGISwapChain4)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (logQueryInterfaceError(__uuidof(IDXGISwapChain1), riid)) {
      WARN("DXGISwapChain: Unknown interface query ", str::format(riid));
    }

    return E_NOINTERFACE;
  };

  HRESULT
  STDMETHODCALLTYPE
  GetParent(REFIID riid, void **parent) final {
    return factory_->QueryInterface(riid, parent);
  };

  DXMT_HOTPATCHABLE
  HRESULT
  STDMETHODCALLTYPE
  Present(UINT sync_interval, UINT flags) final {
    return Present1(sync_interval, flags, nullptr);
  };

  HRESULT
  STDMETHODCALLTYPE
  GetBuffer(UINT buffer_idx, REFIID riid, void **surface) final {
    if (buffer_idx == 0) {
      return backbuffer_->QueryInterface(riid, surface);
    } else {
      ERR("Non zero-index buffer is not supported");
      return DXGI_ERROR_UNSUPPORTED;
    }
  };

  DXMT_HOTPATCHABLE
  HRESULT
  STDMETHODCALLTYPE
  SetFullscreenState(BOOL Fullscreen, IDXGIOutput *pTarget) final {
    Com<IDXGIOutput1> target;

    if (pTarget) {
      DXGI_OUTPUT_DESC desc;

      pTarget->QueryInterface(IID_PPV_ARGS(&target));
      target->GetDesc(&desc);

      if (!fullscreen_desc_.Windowed && Fullscreen && monitor_ != desc.Monitor) {
        HRESULT hr = LeaveFullscreenMode();
        if (FAILED(hr))
          return hr;
      }
    }

    if (fullscreen_desc_.Windowed && Fullscreen)
      return EnterFullscreenMode(target.ptr());
    else if (!fullscreen_desc_.Windowed && !Fullscreen)
      return LeaveFullscreenMode();

    return S_OK;
  };

  HRESULT EnterFullscreenMode(IDXGIOutput1* pTarget) {
    ModeSetGuard::ModeSetInProgress modeset_inprogress(modeset_guard_);
    if (modeset_inprogress)
      return DXGI_STATUS_MODE_CHANGE_IN_PROGRESS;

    Com<IDXGIOutput1> output = pTarget;

    if (!wsi::isWindow(hWnd))
      return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;
    
    if (output == nullptr) {
      if (FAILED(GetOutputFromMonitor(wsi::getWindowMonitor(hWnd), &output))) {
       ERR("DXGI: EnterFullscreenMode: Cannot query containing output");
        return E_FAIL;
      }
    }

    std::unique_lock<dxmt::mutex> lock(mutex_);

    DXGI_MODE_DESC1 preferred_display_mode = {
        desc_.Width,
        desc_.Height,
        fullscreen_desc_.RefreshRate,
        desc_.Format,
        DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED,
        DXGI_MODE_SCALING_UNSPECIFIED
    };
    if (FAILED(ChangeDisplayMode(output.ptr(), &preferred_display_mode))) {
      ERR("DXGI: EnterFullscreenMode: Failed to change display mode");
      return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;
    }

    // Update swap chain description
    fullscreen_desc_.Windowed = FALSE;
    
    // Move the window so that it covers the entire output
    bool modeSwitch = (desc_.Flags & DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH) != 0u;

    DXGI_OUTPUT_DESC desc;
    output->GetDesc(&desc);

    monitor_ = desc.Monitor;
    target_  = std::move(output);

    lock = {};

    if (!wsi::enterFullscreenMode(desc.Monitor, hWnd, &window_state_, modeSwitch)) {
      ERR("DXGI: EnterFullscreenMode: Failed to enter fullscreen mode");
      return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;
    }

    return S_OK;
  }
  
  
  HRESULT LeaveFullscreenMode() {
    ModeSetGuard::ModeSetInProgress modeset_inprogress(modeset_guard_);
    if (modeset_inprogress)
      return DXGI_STATUS_MODE_CHANGE_IN_PROGRESS;

    std::lock_guard<dxmt::mutex> lock(mutex_);

    if (FAILED(RestoreDisplayMode(monitor_)))
      WARN("DXGI: LeaveFullscreenMode: Failed to restore display mode");

    // Restore internal state
    fullscreen_desc_.Windowed = TRUE;
    target_  = nullptr;
    monitor_ = wsi::getWindowMonitor(hWnd);
    
    if (!wsi::isWindow(hWnd))
      return S_OK;
    
    if (!wsi::leaveFullscreenMode(hWnd, &window_state_, true)) {
      ERR("DXGI: LeaveFullscreenMode: Failed to exit fullscreen mode");
      return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;
    }
    
    presenter->changeGammaRamp(nullptr);

    return S_OK;
  }

  HRESULT
  ChangeDisplayMode(IDXGIOutput1 *pOutput, DXGI_MODE_DESC1 *pDisplayMode) {
    if (!pOutput)
      return DXGI_ERROR_INVALID_CALL;

    // Find a mode that the output supports

    DXGI_MODE_DESC1 preferred_mode = *pDisplayMode;
    DXGI_MODE_DESC1 selected_mode = {};

    if (!(desc_.Flags & DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH)) {
      preferred_mode.Width = 0;
      preferred_mode.Height = 0;
    }

    if (preferred_mode.Format == DXGI_FORMAT_UNKNOWN)
      preferred_mode.Format = desc_.Format;

    HRESULT hr = pOutput->FindClosestMatchingMode1(&preferred_mode, &selected_mode, nullptr);

    if (FAILED(hr)) {
      ERR("DXGI: Failed to query closest mode:"
          "\n"
          "  Format: ",
          preferred_mode.Format,
          "\n"
          "  Mode:   ",
          preferred_mode.Width, "x", preferred_mode.Height, "@",
          preferred_mode.RefreshRate.Numerator / std::max(preferred_mode.RefreshRate.Denominator, 1u));
      return hr;
    }

    if (!selected_mode.RefreshRate.Denominator)
      selected_mode.RefreshRate.Denominator = 1;

    DXGI_OUTPUT_DESC output_desc;
    pOutput->GetDesc(&output_desc);
    wsi::WsiMode wsi_mode{
        selected_mode.Width,
        selected_mode.Height,
        {selected_mode.RefreshRate.Numerator, selected_mode.RefreshRate.Denominator},
        GetMonitorFormatBpp(selected_mode.Format),
        selected_mode.ScanlineOrdering == DXGI_MODE_SCANLINE_ORDER_UPPER_FIELD_FIRST ||
            selected_mode.ScanlineOrdering == DXGI_MODE_SCANLINE_ORDER_LOWER_FIELD_FIRST
    };
    if (!wsi::setWindowMode(output_desc.Monitor, hWnd, wsi_mode))
      return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;

    *pDisplayMode = selected_mode;
    init_refresh_rate_ = double(selected_mode.RefreshRate.Numerator) / double(selected_mode.RefreshRate.Denominator);
    return S_OK;
  }

  HRESULT
  RestoreDisplayMode(HMONITOR hMonitor) {
    if (!hMonitor)
      return DXGI_ERROR_INVALID_CALL;

    if (!wsi::restoreDisplayMode(hMonitor))
      return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;

    init_refresh_rate_ = DBL_MAX;
    return S_OK;
  }

  HRESULT
  STDMETHODCALLTYPE
  GetFullscreenState(BOOL *pFullscreen, IDXGIOutput **ppTarget) final {
    HRESULT hr = S_OK;

    if (handle_alt_tab_ && !fullscreen_desc_.Windowed && !wsi::isForeground(hWnd))
      SetFullscreenState(FALSE, nullptr);

    if (pFullscreen != nullptr)
      *pFullscreen = !fullscreen_desc_.Windowed;
    
    if (ppTarget != nullptr)
      *ppTarget = target_.ref();

    return hr;
  };

  HRESULT
  STDMETHODCALLTYPE
  GetDesc(DXGI_SWAP_CHAIN_DESC *pDesc) final {
    if (!pDesc)
      return E_INVALIDARG;

    pDesc->BufferDesc.Width = desc_.Width;
    pDesc->BufferDesc.Height = desc_.Height;
    pDesc->BufferDesc.RefreshRate = fullscreen_desc_.RefreshRate;
    pDesc->BufferDesc.Format = desc_.Format;
    pDesc->BufferDesc.ScanlineOrdering = fullscreen_desc_.ScanlineOrdering;
    pDesc->BufferDesc.Scaling = fullscreen_desc_.Scaling;
    pDesc->SampleDesc = desc_.SampleDesc;
    pDesc->BufferUsage = desc_.BufferUsage;
    pDesc->BufferCount = desc_.BufferCount;
    pDesc->OutputWindow = hWnd;
    pDesc->Windowed = fullscreen_desc_.Windowed;
    pDesc->SwapEffect = desc_.SwapEffect;
    pDesc->Flags = desc_.Flags;
    return S_OK;
  };

  DXMT_HOTPATCHABLE
  HRESULT
  STDMETHODCALLTYPE
  ResizeBuffers(UINT BufferCount, UINT Width, UINT Height, DXGI_FORMAT Format,
                UINT flags) final {
    /* BufferCount ignored */
    if (Width == 0 || Height == 0) {
      wsi::getWindowSize(hWnd, &desc_.Width, &desc_.Height);
    } else {
      desc_.Width = Width;
      desc_.Height = Height;
    }
    if (Format != DXGI_FORMAT_UNKNOWN) {
      if (ConvertSwapChainFormat(Format) != WMTPixelFormatInvalid)
        desc_.Format = Format;
    }

    backbuffer_ = nullptr;
    if (desc_.Width == 0 || desc_.Height == 0) {
      backbuffer_desc_.Width = 1;
      backbuffer_desc_.Height = 1;
    } else {
      backbuffer_desc_.Width = desc_.Width;
      backbuffer_desc_.Height = desc_.Height;
    }

    ApplyLayerProps();

    backbuffer_desc_.Format = desc_.Format;

    backbuffer_ = nullptr;
    if (FAILED(dxmt::CreateDeviceTexture2D(
            device_, &backbuffer_desc_, nullptr, reinterpret_cast<ID3D11Texture2D1 **>(&backbuffer_)
        )))
      return E_FAIL;
    // CreateDeviceTexture2D returns public reference, change to private one here
    backbuffer_->AddRefPrivate();
    backbuffer_->Release();

    // present-dm: reset history on resize
    present_prev_color_ = nullptr;
    present_depth_ = nullptr;
    present_motion_ = nullptr;
    present_prev_color_valid_ = false;
    if (present_dm::enabled())
      EnsurePresentPrevColorTexture();

    if constexpr (kUpscale) {
      D3D11_TEXTURE2D_DESC1 upscaled_desc_ = backbuffer_desc_;
      upscaled_desc_.Height *= scale_factor;
      upscaled_desc_.Width *= scale_factor;
      upscaled_backbuffer_ = nullptr;
      if (FAILED(dxmt::CreateDeviceTexture2D(
              device_, &upscaled_desc_, nullptr, reinterpret_cast<ID3D11Texture2D1 **>(&upscaled_backbuffer_)
          )))
        return E_FAIL;
    }

    if constexpr (kSpatial) {
      WMTFXSpatialScalerInfo info;
      info.input_height = desc_.Height;
      info.input_width = desc_.Width;
      info.output_height = desc_.Height * scale_factor;
      info.output_width = desc_.Width * scale_factor;
      info.color_format = backbuffer_->texture()->pixelFormat();
      info.output_format = upscaled_backbuffer_->texture()->pixelFormat();
      metalfx_spatial_scaler_ = new SpatialScaler(device_->GetMTLDevice(), info);
      D3D11_ASSERT(metalfx_spatial_scaler_ && "otherwise metalfx spatial failed to initialize");
    }

    if constexpr (kTemporal) {
      // Temporal scaler is created lazily once Present-export depth/motion formats
      // are known (real or labeled estimate). Invalidate on resize.
      metalfx_temporal_scaler_ = nullptr;
      temporal_depth_format_ = WMTPixelFormatInvalid;
      temporal_motion_format_ = WMTPixelFormatInvalid;
      temporal_needs_reset_ = true;
      WARN(
          "unified-metalfx: temporal Present path armed (input=", desc_.Width, "x", desc_.Height,
          " output=", (uint32_t)(desc_.Width * scale_factor), "x", (uint32_t)(desc_.Height * scale_factor),
          ") — FrameInterpolator not wired yet"
      );
    }

    return S_OK;
  };

  DXMT_HOTPATCHABLE
  HRESULT
  STDMETHODCALLTYPE
  ResizeTarget(const DXGI_MODE_DESC *pDesc) final {
    if (!pDesc)
      return DXGI_ERROR_INVALID_CALL;

    if (!wsi::isWindow(hWnd))
      return DXGI_ERROR_INVALID_CALL;

    std::unique_lock<dxmt::mutex> lock(mutex_);

    // Promote display mode
    DXGI_MODE_DESC1 newDisplayMode = {};
    newDisplayMode.Width = pDesc->Width;
    newDisplayMode.Height = pDesc->Height;
    newDisplayMode.RefreshRate = pDesc->RefreshRate;
    newDisplayMode.Format = pDesc->Format;
    newDisplayMode.ScanlineOrdering = pDesc->ScanlineOrdering;
    newDisplayMode.Scaling = pDesc->Scaling;

    // Update the swap chain description
    if (newDisplayMode.RefreshRate.Numerator != 0)
      fullscreen_desc_.RefreshRate = newDisplayMode.RefreshRate;

    fullscreen_desc_.ScanlineOrdering = newDisplayMode.ScanlineOrdering;
    fullscreen_desc_.Scaling = newDisplayMode.Scaling;

    if (fullscreen_desc_.Windowed) {
      lock = {};
      wsi::resizeWindow(hWnd, &window_state_, newDisplayMode.Width,
                        newDisplayMode.Height);
    } else {
      ChangeDisplayMode(target_.ptr(), &newDisplayMode);
      lock = {};
      wsi::updateFullscreenWindow(monitor_, hWnd, false);
    }

    return S_OK;
  };

  void ApplyLayerProps() {
    auto target_color_space =
        ConvertColorSpace(desc_.Format == DXGI_FORMAT_R16G16B16A16_FLOAT
                              ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709
                              : colorspace_,
                          LayerSupportEDR());
    if (presenter->changeLayerProperties(
            ConvertSwapChainFormat(desc_.Format), target_color_space, desc_.Width * scale_factor,
            desc_.Height * scale_factor, desc_.SampleDesc.Count
        ))
      device_context_->WaitUntilGPUIdle();
  };

  HRESULT GetOutputFromMonitor(
          HMONITOR                  Monitor,
          IDXGIOutput1**            ppOutput) {
    if (!ppOutput)
      return DXGI_ERROR_INVALID_CALL;

    Com<IDXGIAdapter> adapter;
    Com<IDXGIOutput> output;

    if (FAILED(dxgi_device_->GetAdapter(&adapter)))
      return E_FAIL;

    for (uint32_t i = 0; SUCCEEDED(adapter->EnumOutputs(i, &output)); i++) {
      DXGI_OUTPUT_DESC outputDesc;
      output->GetDesc(&outputDesc);
      
      if (outputDesc.Monitor == Monitor)
        return output->QueryInterface(IID_PPV_ARGS(ppOutput));
      
      output = nullptr;
    }
    
    return DXGI_ERROR_NOT_FOUND;
  }

  HRESULT
  STDMETHODCALLTYPE
  GetContainingOutput(IDXGIOutput **ppOutput) final {
    InitReturnPtr(ppOutput);
    
    if (!wsi::isWindow(hWnd))
      return DXGI_ERROR_INVALID_CALL;
    
    Com<IDXGIOutput1> output;

    if (target_ == nullptr) {
      HRESULT hr = GetOutputFromMonitor(wsi::getWindowMonitor(hWnd), &output);

      if (FAILED(hr))
        return hr;
    } else {
      output = target_;
    }

    *ppOutput = output.ref();
    return S_OK;
  };

  HRESULT
  STDMETHODCALLTYPE
  GetFrameStatistics(DXGI_FRAME_STATISTICS *stats) final {
    DEBUG("DXGISwapChain::GetFrameStatistics: stub");
    std::unique_lock<d3d11_device_mutex> lock(device_->mutex);
    stats->PresentCount = presentation_count_;
    stats->SyncRefreshCount = presentation_count_;
    stats->PresentRefreshCount = presentation_count_;
    stats->SyncGPUTime = {};
    stats->SyncQPCTime = {};
    return S_OK;
  };

  HRESULT
  STDMETHODCALLTYPE
  GetLastPresentCount(UINT *last_present_count) final {
    if (last_present_count == NULL) {
      return E_POINTER;
    }
    std::unique_lock<d3d11_device_mutex> lock(device_->mutex);
    *last_present_count = presentation_count_;
    return S_OK;
  };

  HRESULT
  STDMETHODCALLTYPE
  GetDesc1(DXGI_SWAP_CHAIN_DESC1 *pDesc) final {
    if (pDesc == NULL) {
      return E_POINTER;
    }
    *pDesc = desc_;
    return S_OK;
  };

  HRESULT
  STDMETHODCALLTYPE
  GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC *pDesc) final {
    if (pDesc == NULL) {
      return E_POINTER;
    }
    *pDesc = fullscreen_desc_;
    return S_OK;
  };

  HRESULT
  STDMETHODCALLTYPE
  GetHwnd(HWND *pHwnd) final {
    if (pHwnd == NULL) {
      return E_POINTER;
    }
    *pHwnd = hWnd;
    return S_OK;
  };

  HRESULT
  STDMETHODCALLTYPE
  GetCoreWindow(REFIID refiid, void **ppUnk) final {
    ERR("Not implemented");
    return E_NOTIMPL;
  };

  class SyncFrameState {
    uint64_t frame_id_ = 0;
    CpuFence *frame_latency_fence_ = nullptr;

  public:
    SyncFrameState() {}
    SyncFrameState(uint64_t frame_id, CpuFence *frame_latency_fence)
        : frame_id_(frame_id), frame_latency_fence_(frame_latency_fence) {}

    SyncFrameState(const SyncFrameState &) = delete;
    SyncFrameState(SyncFrameState &&move) {
      frame_id_ = move.frame_id_;
      frame_latency_fence_ = move.frame_latency_fence_;
      move.frame_latency_fence_ = nullptr;
    };
    ~SyncFrameState() {
      if (frame_latency_fence_) {
        frame_latency_fence_->signal(frame_id_);
        frame_latency_fence_ = nullptr;
      }
    }
  };

  SyncFrameState SyncFrame(uint64_t current_frame_id) {
    if (frame_latency_fence_) {
      if (current_frame_id > frame_latency)
        frame_latency_fence_->wait(current_frame_id - frame_latency);
      return SyncFrameState(current_frame_id, frame_latency_fence_.get());
    }
    return SyncFrameState();
  };

  DXMT_HOTPATCHABLE
  HRESULT
  STDMETHODCALLTYPE
  Present1(UINT SyncInterval, UINT PresentFlags,
           const DXGI_PRESENT_PARAMETERS *pPresentParameters) final {
    if (SyncInterval > 4)
      return DXGI_ERROR_INVALID_CALL;

    HRESULT hr = S_OK;
    bool window_minimized = wsi::isMinimized(hWnd);
    if ((window_minimized || desc_.Width == 0 || desc_.Height == 0)
        // MSDN: You will not receive DXGI_STATUS_OCCLUDED if you're using a flip model swap chain.
        && desc_.SwapEffect <= DXGI_SWAP_EFFECT_SEQUENTIAL)
      hr = DXGI_STATUS_OCCLUDED;
    bool should_exit_fs = handle_alt_tab_ // At the moment this is still broken for certain games
                          && !fullscreen_desc_.Windowed && !window_minimized && !wsi::isForeground(hWnd);
    if (hr == S_OK && should_exit_fs)
      hr = DXGI_STATUS_OCCLUDED;
    if (PresentFlags & DXGI_PRESENT_TEST)
      return hr;

    if (should_exit_fs)
      SetFullscreenState(FALSE, nullptr);

    std::unique_lock<d3d11_device_mutex> lock(device_->mutex);

    device_context_->PrepareFlush();
    if (hr == DXGI_STATUS_OCCLUDED) {
      // flush commands without presenting
      device_context_->Commit();
      return hr;
    }

    // present-dm: export synchronized color (+prev), depth, motion provenance
    // before MetalFX encode. Temporal path binds those textures; spatial stays
    // color-only. Does not wire FrameInterpolator (ticket 03).
    if (present_dm::enabled()) {
      if constexpr (kTemporal)
        ExportPresentDepthMotion("temporal_encode=color+depth+motion_bound");
      else
        ExportPresentDepthMotion("spatial_encode=color_only");
    }

    double vsync_duration =
        std::max(SyncInterval * 1.0 /
                     (preferred_max_frame_rate ? preferred_max_frame_rate
                                               : init_refresh_rate_),
                 preferred_max_frame_rate ? 1.0 / preferred_max_frame_rate : 0);

    auto &cmd_queue = device_->GetDXMTDevice().queue();
    auto chunk = cmd_queue.CurrentChunk();
    chunk->signal_frame_latency_fence_ = cmd_queue.CurrentFrameSeq();
    if (target_) {
      auto output = static_cast<MTLDXGIOutput *>(target_.ptr());
      presenter->changeGammaRamp(output->GetGammaRamp());
    }
    if constexpr (kSpatial) {
      chunk->emitcc([
        this, vsync_duration, backbuffer = backbuffer_->texture(),
        sync_state = SyncFrame(++presentation_count_),
        upscaled = upscaled_backbuffer_->texture(),
        scaler = this->metalfx_spatial_scaler_, state = presenter->synchronizeLayerProperties()
      ](ArgumentEncodingContext &ctx) mutable {
        auto &scaler_info = ctx.currentFrameStatistics().last_scaler_info;
        scaler_info.type = ScalerType::Spatial;
        scaler_info.input_width = backbuffer->width();
        scaler_info.input_height = backbuffer->height();
        scaler_info.output_width = upscaled->width();
        scaler_info.output_height = upscaled->height();
        ctx.upscale(backbuffer, upscaled, scaler);
        ctx.present(upscaled, presenter, vsync_duration, state.metadata);
        ReleaseSemaphore(present_semaphore_, 1, nullptr);
        this->UpdateStatistics(ctx.queue().statistics, ctx.currentFrameId());
      });
    } else if constexpr (kTemporal) {
      Rc<Texture> depth_tex = present_depth_ ? present_depth_->texture() : nullptr;
      Rc<Texture> motion_tex = present_motion_ ? present_motion_->texture() : nullptr;
      const bool encoded = depth_tex && motion_tex && EnsureTemporalScaler(depth_tex, motion_tex);
      if (!encoded) {
        WARN(
            "unified-metalfx: temporal encode skipped (depth=", (bool)depth_tex,
            " motion=", (bool)motion_tex, " scaler=", (bool)metalfx_temporal_scaler_,
            ") — presenting pre-upscale backbuffer"
        );
        chunk->emitcc([
          this, vsync_duration, state = presenter->synchronizeLayerProperties(),
          sync_state = SyncFrame(++presentation_count_),
          backbuffer = backbuffer_->texture()
        ](ArgumentEncodingContext &ctx) mutable {
          ctx.present(backbuffer, presenter, vsync_duration, state.metadata);
          ReleaseSemaphore(present_semaphore_, 1, nullptr);
          this->UpdateStatistics(ctx.queue().statistics, ctx.currentFrameId());
        });
      } else {
        const bool reset = temporal_needs_reset_;
        temporal_needs_reset_ = false;
        const uint32_t content_w = backbuffer_desc_.Width;
        const uint32_t content_h = backbuffer_desc_.Height;
        chunk->emitcc([
          this, vsync_duration, backbuffer = backbuffer_->texture(),
          sync_state = SyncFrame(++presentation_count_),
          upscaled = upscaled_backbuffer_->texture(), depth = std::move(depth_tex),
          motion = std::move(motion_tex), scaler = this->metalfx_temporal_scaler_,
          motion_format = temporal_motion_format_, reset, content_w, content_h,
          state = presenter->synchronizeLayerProperties()
        ](ArgumentEncodingContext &ctx) mutable {
          auto &scaler_info = ctx.currentFrameStatistics().last_scaler_info;
          scaler_info.type = ScalerType::Temporal;
          scaler_info.auto_exposure = true;
          scaler_info.motion_vector_highres = false;
          scaler_info.input_width = content_w;
          scaler_info.input_height = content_h;
          scaler_info.output_width = upscaled->width();
          scaler_info.output_height = upscaled->height();

          auto mv_view = motion->createView({
              .format = motion_format,
              .type = WMTTextureType2D,
              .firstMiplevel = 0,
              .miplevelCount = 1,
              .firstArraySlice = 0,
              .arraySize = 1,
          });
          Rc<Texture> exposure;
          WMTFXTemporalScalerProps props{};
          props.input_content_width = content_w;
          props.input_content_height = content_h;
          props.reset = reset;
          // Labeled depth estimate is conventional far=1; real tracked depth may be
          // reversed-Z — not forged here; ticket 03/polish can refine.
          props.depth_reversed = false;
          props.motion_vector_scale_x = float(content_w);
          props.motion_vector_scale_y = float(content_h);
          props.jitter_offset_x = 0.f;
          props.jitter_offset_y = 0.f;
          props.pre_exposure = 1.f;
          ctx.upscaleTemporal(
              backbuffer, upscaled, depth, motion, mv_view, exposure, scaler, props
          );
          ctx.present(upscaled, presenter, vsync_duration, state.metadata);
          ReleaseSemaphore(present_semaphore_, 1, nullptr);
          this->UpdateStatistics(ctx.queue().statistics, ctx.currentFrameId());
        });
      }
    } else {
      chunk->emitcc([
        this, vsync_duration, state = presenter->synchronizeLayerProperties(),
        sync_state = SyncFrame(++presentation_count_),
        backbuffer = backbuffer_->texture()
      ](ArgumentEncodingContext &ctx) mutable {
        ctx.present(backbuffer, presenter, vsync_duration, state.metadata);
        ReleaseSemaphore(present_semaphore_, 1, nullptr);
        this->UpdateStatistics(ctx.queue().statistics, ctx.currentFrameId());
      });
    }
    device_context_->Commit();

    lock.unlock(); // since PresentBoundary() will and should only stall current thread

    cmd_queue.PresentBoundary();

    return hr;
  };

  void UpdateStatistics(const FrameStatisticsContainer& statistics, uint64_t frame_id) {
    hud.begin();
    auto &frame = statistics.at(frame_id - 1); // show the previous one frame statistics
    auto &average = statistics.average();
    Flags<FeatureCompatibility> flags = frame.compatibility_flags;
    char text[] = "---------------------------";
    if (flags.test(FeatureCompatibility::UnsupportedGeometryDraw)) {
      text[3] = 'G';
    }
    if (flags.test(FeatureCompatibility::UnsupportedTessellationOutputPrimitive)) {
      text[5] = 'T';
      text[6] = 'O';
    }
    if (flags.test(FeatureCompatibility::UnsupportedIndirectTessellationDraw)) {
      text[8] = 'I';
      text[9] = 'T';
    }
    if (flags.test(FeatureCompatibility::UnsupportedGeometryTessellationDraw)) {
      text[11] = 'G';
      text[12] = 'T';
    }
    if (flags.test(FeatureCompatibility::UnsupportedDrawAuto)) {
      text[14] = 'A';
    }
    if (flags.test(FeatureCompatibility::UnsupportedPredication)) {
      text[16] = 'P';
    }
    if (flags.test(FeatureCompatibility::UnsupportedStreamOutputAppending)) {
      text[18] = 'S';
      text[19] = 'A';
    }
    if (flags.test(FeatureCompatibility::UnsupportedMultipleStreamOutput)) {
      text[21] = 'M';
      text[22] = 'S';
    }
    hud.printLine(text);
    hud.printLine(std::format(
        "Commit: {:2} -{:4.1f} -{:4.1f}", std::min(frame.command_buffer_count, 99u),
        std::min(average.commit_interval.count() / 1000000.0, 99.9),
        std::min(statistics.max().commit_interval.count() / 1000000.0, 99.9)
    ));
    hud.printLine(std::format(
        "Sync:   {:2} {:4.1f}  {:2} {:4.1f} {:2}", std::min(frame.sync_count, 99u),
        std::min(average.sync_interval.count() / 1000000.0, 99.9), std::min(statistics.max().event_stall, 99u),
        std::min(average.present_latency_interval.count() / 1000000.0, 99.9), frame.latency
    ));
    hud.printLine(std::format(
        "Encode: {:4.1f}+{:4.1f}+{:4.1f}={:4.1f}", std::min(average.encode_prepare_interval.count() / 1000000.0, 99.9),
        std::min((average.encode_flush_interval - average.drawable_blocking_interval).count() / 1000000.0, 99.9),
        std::min(average.drawable_blocking_interval.count() / 1000000.0, 99.9),
        std::min((average.encode_prepare_interval + average.encode_flush_interval).count() / 1000000.0, 99.9)
    ));
    hud.printLine(std::format(
        "Render:{:3}+{:<3} Clear:{:3}+{:<2}", std::min(frame.render_pass_count - frame.render_pass_optimized, 999u),
        std::min(frame.render_pass_optimized, 999u),
        std::min(frame.clear_pass_count - frame.clear_pass_optimized, 999u), std::min(frame.clear_pass_optimized, 99u)
    ));
    {
      /* scaler info */
      auto &info = frame.last_scaler_info;
      if (info.type == ScalerType::Temporal) {
        auto &info = frame.last_scaler_info;
        hud.printLine(std::format(
            "MetalFX: Temporal {} {}", info.auto_exposure ? "AEXP" : "", info.motion_vector_highres ? "HMV" : ""
        ));
        hud.printLine(std::format(
            "Scale: {:4}x{:4}->{:4}x{:4}", info.input_width, info.input_height, info.output_width, info.output_height
        ));
      }
      if (info.type == ScalerType::Spatial) {
        hud.printLine(std::format(
            "MetalFX: Spatial"
        ));
        hud.printLine(std::format(
            "Scale: {:4}x{:4}->{:4}x{:4}", info.input_width, info.input_height, info.output_width, info.output_height
        ));
      }
    }
    hud.end();
  }

  BOOL STDMETHODCALLTYPE IsTemporaryMonoSupported() final { return FALSE; };

  HRESULT
  STDMETHODCALLTYPE
  GetRestrictToOutput(IDXGIOutput **ppRestrictToOutput) final {
    InitReturnPtr(ppRestrictToOutput);
    ERR("DXGISwapChain1::GetRestrictToOutput: not implemented");
    return E_NOTIMPL;
  };

  HRESULT
  STDMETHODCALLTYPE
  SetBackgroundColor(const DXGI_RGBA *pColor) final {
    ERR("DXGISwapChain1::SetBackgroundColor: not implemented");
    return E_NOTIMPL;
  };

  HRESULT
  STDMETHODCALLTYPE
  GetBackgroundColor(DXGI_RGBA *pColor) final {
    if (!pColor)
      return E_INVALIDARG;
    // TODO(swapchain): check if native returns transparent or opaque black
    *pColor = {0, 0, 0, 0};
    return S_OK;
  };

  HRESULT
  STDMETHODCALLTYPE
  SetRotation(DXGI_MODE_ROTATION Rotation) final {
    ERR("DXGISwapChain1::SetRotation: not implemented");
    return E_NOTIMPL;
  };

  HRESULT
  STDMETHODCALLTYPE
  GetRotation(DXGI_MODE_ROTATION *pRotation) final {
    if (!pRotation)
      return E_INVALIDARG;
    *pRotation = DXGI_MODE_ROTATION_IDENTITY;
    return S_OK;
  };

  HRESULT STDMETHODCALLTYPE
  SetSourceSize(UINT Width, UINT Height) override {
    ERR("DXGISwapChain2::SetSourceSize: not implemented");
    return E_NOTIMPL;
  };

  HRESULT STDMETHODCALLTYPE
  GetSourceSize(UINT *pWidth, UINT *pHeight) override {
    if (pWidth)
      *pWidth = desc_.Width;
    if (pHeight)
      *pHeight = desc_.Height;
    return S_OK;
  };

  HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT max_latency) override {
    if (max_latency == 0 || max_latency > DXGI_MAX_SWAP_CHAIN_BUFFERS) {
      return E_INVALIDARG;
    }
    if (max_latency > frame_latency) {
      ReleaseSemaphore(present_semaphore_, max_latency - frame_latency,
                       nullptr);
    }
    frame_latency = max_latency;

    return S_OK;
  };

  HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT *max_latency) override {
    if (max_latency) {
      *max_latency = frame_latency;
    }
    return S_OK;
  };

  HANDLE STDMETHODCALLTYPE GetFrameLatencyWaitableObject() override {
    if (!(desc_.Flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT)) {
      return nullptr;
    }

    HANDLE result = nullptr;
    HANDLE processHandle = GetCurrentProcess();

    if (!DuplicateHandle(processHandle, present_semaphore_, processHandle,
                         &result, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
      return nullptr;
    }

    return result;
  };

  HRESULT STDMETHODCALLTYPE
  SetMatrixTransform(const DXGI_MATRIX_3X2_F *matrix) override {
    ERR("DXGISwapChain2::SetMatrixTransform: not implemented");
    return E_NOTIMPL;
  };

  HRESULT STDMETHODCALLTYPE
  GetMatrixTransform(DXGI_MATRIX_3X2_F *matrix) override {
    ERR("DXGISwapChain2::GetMatrixTransform: not implemented");
    return E_NOTIMPL;
  };

  HRESULT STDMETHODCALLTYPE CheckColorSpaceSupport(
      DXGI_COLOR_SPACE_TYPE ColorSpace, UINT *pColorSpaceSupport) override {
    if (!pColorSpaceSupport)
      return E_INVALIDARG;
    *pColorSpaceSupport = CGColorSpace_checkColorSpaceSupported(ConvertColorSpace(ColorSpace, false))
            ? DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT
            : 0;
    return S_OK;
  };

  UINT STDMETHODCALLTYPE GetCurrentBackBufferIndex() override {
    // TODO(swapchain): can be non-zero once sequential swapchain is implemented
    return 0;
  }

  DXMT_HOTPATCHABLE
  HRESULT STDMETHODCALLTYPE ResizeBuffers1(UINT BufferCount, UINT Width, UINT Height,
                         DXGI_FORMAT Format, UINT SwapChainFlags,
                         const UINT *pCreationNodeMask,
                         IUnknown *const *ppPresentQueue) override {
    WARN("DXGISwapChain3::ResizeBuffers1: ignoring d3d12 related parameters");
    return ResizeBuffers(BufferCount, Width, Height, Format, SwapChainFlags);
  }

  HRESULT STDMETHODCALLTYPE
  SetColorSpace1(DXGI_COLOR_SPACE_TYPE ColorSpace) override {
    auto target_color_space = ConvertColorSpace(ColorSpace, LayerSupportEDR());
    if (presenter->changeLayerColorSpace(target_color_space))
      device_context_->WaitUntilGPUIdle();
    colorspace_ = ColorSpace;
    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE SetHDRMetaData(DXGI_HDR_METADATA_TYPE Type,
                                           UINT Size,
                                           void *pMetaData) override {
                                            return S_OK;
    if (Type == DXGI_HDR_METADATA_TYPE_NONE) {
      presenter->changeHDRMetadata(nullptr);
      return S_OK;
    }
    if (Type == DXGI_HDR_METADATA_TYPE_HDR10) {
      if (Size != sizeof(WMTHDRMetadata))
        return E_INVALIDARG;
      presenter->changeHDRMetadata(reinterpret_cast<const WMTHDRMetadata *>(pMetaData));
      return S_OK;
    }
    return DXGI_ERROR_UNSUPPORTED;
  }

private:
  bool LayerSupportEDR() {
    WMTEDRValue edr_value;
    MetalLayer_getEDRValue(layer_weak_, &edr_value);
    return edr_value.maximum_potential_edr_color_component_value > 1.0f;
  };

  void EnsurePresentPrevColorTexture() {
    if (present_prev_color_)
      return;
    D3D11_TEXTURE2D_DESC1 prev_desc = backbuffer_desc_;
    prev_desc.Usage = D3D11_USAGE_DEFAULT;
    prev_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    prev_desc.CPUAccessFlags = 0;
    prev_desc.MiscFlags = 0;
    if (FAILED(dxmt::CreateDeviceTexture2D(
            device_, &prev_desc, nullptr, reinterpret_cast<ID3D11Texture2D1 **>(&present_prev_color_)
        ))) {
      WARN("present-dm: failed to allocate previous-color texture");
      present_prev_color_ = nullptr;
      return;
    }
    present_prev_color_->AddRefPrivate();
    present_prev_color_->Release();
  }

  bool EnsurePresentCopyTexture(
      Com<D3D11ResourceCommon, false> &slot, ID3D11Resource *source, UINT bind_flags
  ) {
    if (!source)
      return false;
    Com<ID3D11Texture2D> src_tex;
    if (FAILED(source->QueryInterface(IID_PPV_ARGS(&src_tex))))
      return false;
    D3D11_TEXTURE2D_DESC src_desc{};
    src_tex->GetDesc(&src_desc);
    if (slot && slot->texture() && slot->texture()->width() == src_desc.Width &&
        slot->texture()->height() == src_desc.Height) {
      D3D11_TEXTURE2D_DESC1 existing{};
      slot->GetDesc1(&existing);
      if (existing.Format == src_desc.Format)
        return true;
    }
    slot = nullptr;
    D3D11_TEXTURE2D_DESC1 dst_desc{};
    dst_desc.Width = src_desc.Width;
    dst_desc.Height = src_desc.Height;
    dst_desc.MipLevels = 1;
    dst_desc.ArraySize = 1;
    dst_desc.Format = src_desc.Format;
    dst_desc.SampleDesc = src_desc.SampleDesc;
    dst_desc.Usage = D3D11_USAGE_DEFAULT;
    dst_desc.BindFlags = bind_flags ? bind_flags : src_desc.BindFlags;
    dst_desc.CPUAccessFlags = 0;
    dst_desc.MiscFlags = 0;
    if (FAILED(dxmt::CreateDeviceTexture2D(
            device_, &dst_desc, nullptr, reinterpret_cast<ID3D11Texture2D1 **>(&slot)
        ))) {
      WARN("present-dm: failed to allocate Present export texture fmt=", (uint32_t)src_desc.Format);
      slot = nullptr;
      return false;
    }
    slot->AddRefPrivate();
    slot->Release();
    return true;
  }

  /** Allocate / reuse color-sized texture for labeled estimates; bind as RT+SRV. */
  bool EnsurePresentEstimateTexture(
      Com<D3D11ResourceCommon, false> &slot, DXGI_FORMAT format, UINT width, UINT height
  ) {
    if (width == 0 || height == 0)
      return false;
    if (slot && slot->texture() && slot->texture()->width() == width &&
        slot->texture()->height() == height) {
      D3D11_TEXTURE2D_DESC1 existing{};
      slot->GetDesc1(&existing);
      if (existing.Format == format)
        return true;
    }
    slot = nullptr;
    D3D11_TEXTURE2D_DESC1 desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;
    if (FAILED(dxmt::CreateDeviceTexture2D(
            device_, &desc, nullptr, reinterpret_cast<ID3D11Texture2D1 **>(&slot)
        ))) {
      WARN(
          "present-dm: failed to allocate labeled-estimate texture fmt=", (uint32_t)format,
          " ", width, "x", height
      );
      slot = nullptr;
      return false;
    }
    slot->AddRefPrivate();
    slot->Release();
    return true;
  }

  bool ClearPresentEstimateTexture(ID3D11Resource *resource, const FLOAT rgba[4]) {
    if (!resource || !device_ || !device_context_)
      return false;
    Com<ID3D11RenderTargetView> rtv;
    if (FAILED(device_->CreateRenderTargetView(resource, nullptr, &rtv))) {
      WARN("present-dm: failed to create RTV for labeled-estimate clear");
      return false;
    }
    device_context_->ClearRenderTargetView(rtv.ptr(), rgba);
    return true;
  }

  /**
   * Labeled motion estimate after REAL_MOTION_UNAVAILABLE: zero RG16Float MVs
   * (screen-space displacement assumed static). Explicitly not engine truth.
   */
  bool FillLabeledMotionEstimateZero(const char **out_provenance) {
    if (!out_provenance)
      return false;
    *out_provenance = present_dm::kMotionEstimateFailed;
    if (!EnsurePresentEstimateTexture(
            present_motion_, DXGI_FORMAT_R16G16_FLOAT, backbuffer_desc_.Width,
            backbuffer_desc_.Height
        ))
      return false;
    const FLOAT zeros[4] = {0.f, 0.f, 0.f, 0.f};
    if (!ClearPresentEstimateTexture(
            static_cast<ID3D11Resource *>(present_motion_.ptr()), zeros
        )) {
      present_motion_ = nullptr;
      return false;
    }
    *out_provenance = present_dm::kMotionEstimateZero;
    return true;
  }

  /**
   * Labeled depth estimate only when real depth is missing: constant far (1.0)
   * R32Float. Explicitly not engine truth / reversed-Z aware.
   */
  bool FillLabeledDepthEstimateConstantFar(const char **out_provenance, DXGI_FORMAT *out_fmt) {
    if (!out_provenance)
      return false;
    *out_provenance = present_dm::kDepthEstimateFailed;
    if (out_fmt)
      *out_fmt = DXGI_FORMAT_UNKNOWN;
    if (!EnsurePresentEstimateTexture(
            present_depth_, DXGI_FORMAT_R32_FLOAT, backbuffer_desc_.Width, backbuffer_desc_.Height
        ))
      return false;
    // Normalized far plane for conventional depth; estimate only — not reversed-Z truth.
    const FLOAT far_plane[4] = {1.f, 1.f, 1.f, 1.f};
    if (!ClearPresentEstimateTexture(
            static_cast<ID3D11Resource *>(present_depth_.ptr()), far_plane
        )) {
      present_depth_ = nullptr;
      return false;
    }
    *out_provenance = present_dm::kDepthEstimateConstantFar;
    if (out_fmt)
      *out_fmt = DXGI_FORMAT_R32_FLOAT;
    return true;
  }

  static WMTPixelFormat CorrectMotionVectorFormat(WMTPixelFormat format) {
    switch (format) {
    case WMTPixelFormatRG16Uint:
    case WMTPixelFormatRG16Float:
    case WMTPixelFormatRG16Sint:
    case WMTPixelFormatRG16Snorm:
    case WMTPixelFormatRG16Unorm:
      return WMTPixelFormatRG16Float;
    case WMTPixelFormatRG32Uint:
    case WMTPixelFormatRG32Float:
    case WMTPixelFormatRG32Sint:
      return WMTPixelFormatRG32Float;
    case WMTPixelFormatRGBA16Sint:
    case WMTPixelFormatRGBA16Snorm:
    case WMTPixelFormatRGBA16Uint:
    case WMTPixelFormatRGBA16Unorm:
    case WMTPixelFormatRGBA16Float:
      return WMTPixelFormatRGBA16Float;
    default:
      break;
    }
    return WMTPixelFormatInvalid;
  }

  /** Create / refresh attachable MTLFXTemporalScaler for Present (ticket 02). */
  bool EnsureTemporalScaler(Rc<Texture> &depth, Rc<Texture> &motion) {
    if constexpr (!kTemporal)
      return false;
    if (!depth || !motion || !backbuffer_ || !upscaled_backbuffer_)
      return false;
    WMTPixelFormat motion_fmt = CorrectMotionVectorFormat(motion->pixelFormat());
    if (motion_fmt == WMTPixelFormatInvalid) {
      ERR("unified-metalfx: invalid Present motion format ", motion->pixelFormat());
      return false;
    }
    WMTPixelFormat depth_fmt = depth->pixelFormat();
    WMTPixelFormat color_fmt = backbuffer_->texture()->pixelFormat();
    WMTPixelFormat output_fmt = upscaled_backbuffer_->texture()->pixelFormat();
    const uint32_t in_w = backbuffer_desc_.Width;
    const uint32_t in_h = backbuffer_desc_.Height;
    const uint32_t out_w = (uint32_t)(desc_.Width * scale_factor);
    const uint32_t out_h = (uint32_t)(desc_.Height * scale_factor);
    if (metalfx_temporal_scaler_ && temporal_depth_format_ == depth_fmt &&
        temporal_motion_format_ == motion_fmt && temporal_color_format_ == color_fmt &&
        temporal_output_format_ == output_fmt && temporal_input_width_ == in_w &&
        temporal_input_height_ == in_h && temporal_output_width_ == out_w &&
        temporal_output_height_ == out_h) {
      return true;
    }

    WMTFXTemporalScalerInfo info{};
    info.color_format = color_fmt;
    info.output_format = output_fmt;
    info.depth_format = depth_fmt;
    info.motion_format = motion_fmt;
    info.input_width = in_w;
    info.input_height = in_h;
    info.output_width = out_w;
    info.output_height = out_h;
    info.auto_exposure = true;
    info.input_content_properties_enabled = true;
    info.input_content_min_scale = 1.0f;
    info.input_content_max_scale = 3.0f;
    info.requires_synchronous_initialization = true;
    metalfx_temporal_scaler_ = new TemporalScaler(device_->GetMTLDevice(), info);
    if (!metalfx_temporal_scaler_) {
      ERR("unified-metalfx: TemporalScaler create failed");
      return false;
    }
    temporal_depth_format_ = depth_fmt;
    temporal_motion_format_ = motion_fmt;
    temporal_color_format_ = color_fmt;
    temporal_output_format_ = output_fmt;
    temporal_input_width_ = in_w;
    temporal_input_height_ = in_h;
    temporal_output_width_ = out_w;
    temporal_output_height_ = out_h;
    temporal_needs_reset_ = true;
    WARN(
        "unified-metalfx: TemporalScaler created (attachable; interpolator not wired) "
        "in=", in_w, "x", in_h, " out=", out_w, "x", out_h,
        " depth_fmt=", (uint32_t)depth_fmt, " motion_fmt=", (uint32_t)motion_fmt
    );
    return true;
  }

  void ExportPresentDepthMotion(const char *encode_label) {
    EnsurePresentPrevColorTexture();
    const bool prev_valid = present_prev_color_valid_ && present_prev_color_;
    const char *depth_prov = present_dm::kDepthMissing;
    const char *motion_prov = present_dm::kMotionUnavailable;
    DXGI_FORMAT depth_fmt = DXGI_FORMAT_UNKNOWN;

    Com<ID3D11Resource> depth_src;
    if (present_dm::resolveDepthForPresent(
            device_context_, backbuffer_desc_.Width, backbuffer_desc_.Height, &depth_src, &depth_prov
        )) {
      D3D11_TEXTURE2D_DESC ddesc{};
      if (Com<ID3D11Texture2D> t; SUCCEEDED(depth_src->QueryInterface(IID_PPV_ARGS(&t)))) {
        t->GetDesc(&ddesc);
        depth_fmt = ddesc.Format;
      }
      // Inherit source bind flags (depth formats often reject mismatched binds).
      if (EnsurePresentCopyTexture(present_depth_, depth_src.ptr(), 0)) {
        device_context_->CopyResource(
            static_cast<ID3D11Resource *>(present_depth_.ptr()), depth_src.ptr()
        );
      } else {
        depth_prov = present_dm::kDepthMissing;
        present_depth_ = nullptr;
      }
    } else {
      present_depth_ = nullptr;
    }

    // Real depth missing → labeled constant-far estimate (only after real failure).
    if (!present_depth_ || depth_prov == present_dm::kDepthMissing) {
      if (!FillLabeledDepthEstimateConstantFar(&depth_prov, &depth_fmt)) {
        present_depth_ = nullptr;
        depth_prov = present_dm::kDepthEstimateFailed;
        depth_fmt = DXGI_FORMAT_UNKNOWN;
      }
    }

    Com<ID3D11Resource> motion_src;
    if (present_dm::resolveTitleMotion(&motion_src, &motion_prov)) {
      if (EnsurePresentCopyTexture(present_motion_, motion_src.ptr(), 0)) {
        device_context_->CopyResource(
            static_cast<ID3D11Resource *>(present_motion_.ptr()), motion_src.ptr()
        );
      } else {
        motion_prov = present_dm::kMotionUnavailable;
        present_motion_ = nullptr;
      }
    } else {
      present_motion_ = nullptr;
      motion_prov = present_dm::kMotionUnavailable;
    }

    // Real motion unavailable → labeled zero-MV estimate (only after real failure).
    if (!present_motion_ || motion_prov == present_dm::kMotionUnavailable) {
      if (!FillLabeledMotionEstimateZero(&motion_prov)) {
        present_motion_ = nullptr;
        motion_prov = present_dm::kMotionEstimateFailed;
      }
    }

    present_dm::logPresentTick(
        presentation_count_ + 1, backbuffer_desc_.Width, backbuffer_desc_.Height, prev_valid, depth_prov,
        motion_prov, depth_fmt, encode_label
    );

    // Retain current color as previous for the next Present tick (pre-upscale).
    if (present_prev_color_ && backbuffer_) {
      device_context_->CopyResource(
          static_cast<ID3D11Resource *>(present_prev_color_.ptr()),
          static_cast<ID3D11Resource *>(backbuffer_.ptr())
      );
      present_prev_color_valid_ = true;
    }
  }

  Com<IDXGIFactory1> factory_;
  Com<IMTLDXGIDevice> dxgi_device_;
  WMT::Object native_view_;
  WMT::MetalLayer layer_weak_;
  ULONG presentation_count_;
  DXGI_SWAP_CHAIN_DESC1 desc_;
  DXGI_SWAP_CHAIN_FULLSCREEN_DESC fullscreen_desc_;
  D3D11_TEXTURE2D_DESC1 backbuffer_desc_;
  IMTLD3D11DeviceContext* device_context_;
  Com<D3D11ResourceCommon, false> backbuffer_;
  /** present-dm export targets (pre-upscale color history + optional depth/MV). */
  Com<D3D11ResourceCommon, false> present_prev_color_;
  Com<D3D11ResourceCommon, false> present_depth_;
  Com<D3D11ResourceCommon, false> present_motion_;
  bool present_prev_color_valid_ = false;
  HANDLE present_semaphore_;
  std::unique_ptr<CpuFence> frame_latency_fence_;
  HWND hWnd;
  HMONITOR monitor_;
  Com<IDXGIOutput1> target_;
  wsi::DXMTWindowState window_state_;
  uint32_t frame_latency;
  DXGI_COLOR_SPACE_TYPE colorspace_ = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
  double init_refresh_rate_ = DBL_MAX;
  int preferred_max_frame_rate = 0;
  HUDState hud;
  Rc<Presenter> presenter;
  ModeSetGuard modeset_guard_;
  dxmt::mutex mutex_;

  bool handle_alt_tab_;

  std::conditional_t<kSpatial, Rc<SpatialScaler>, std::monostate> metalfx_spatial_scaler_;
  std::conditional_t<kTemporal, Rc<TemporalScaler>, std::monostate> metalfx_temporal_scaler_;
  std::conditional_t<kUpscale, Com<D3D11ResourceCommon>, std::monostate> upscaled_backbuffer_;
  // Cached descriptor identity for lazy TemporalScaler recreate.
  WMTPixelFormat temporal_depth_format_ = WMTPixelFormatInvalid;
  WMTPixelFormat temporal_motion_format_ = WMTPixelFormatInvalid;
  WMTPixelFormat temporal_color_format_ = WMTPixelFormatInvalid;
  WMTPixelFormat temporal_output_format_ = WMTPixelFormatInvalid;
  uint32_t temporal_input_width_ = 0;
  uint32_t temporal_input_height_ = 0;
  uint32_t temporal_output_width_ = 0;
  uint32_t temporal_output_height_ = 0;
  bool temporal_needs_reset_ = true;
  float scale_factor = 1.0;
};

HRESULT
CreateSwapChain(
    IDXGIFactory1 *pFactory, MTLD3D11Device *pDevice, HWND hWnd, const DXGI_SWAP_CHAIN_DESC1 *pDesc,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC *pFullscreenDesc, IDXGISwapChain1 **ppSwapChain
) {
  if (pDesc == NULL)
    return DXGI_ERROR_INVALID_CALL;
  if (ppSwapChain == NULL)
    return DXGI_ERROR_INVALID_CALL;
  InitReturnPtr(ppSwapChain);

  DWORD window_process_id;
  GetWindowThreadProcessId(hWnd, &window_process_id);
  if (GetProcessId(GetCurrentProcess()) != window_process_id) {
    ERR("CreateSwapChain: cross-process swapchain not supported yet");
    return E_FAIL;
  }

  Com<IMTLDXGIDevice> layer_factory;
  if (FAILED(pDevice->QueryInterface(IID_PPV_ARGS(&layer_factory)))) {
    ERR("CreateSwapChain: failed to get IMTLDXGIDevice");
    return E_FAIL;
  }
  if ((pDesc->SwapEffect != DXGI_SWAP_EFFECT_DISCARD &&
       pDesc->SwapEffect != DXGI_SWAP_EFFECT_FLIP_DISCARD) &&
      pDesc->BufferCount != 1) {
    WARN("CreateSwapChain: unsupported swap effect ", pDesc->SwapEffect, " with backbuffer size ", pDesc->BufferCount);
  }
  // Unified temporal+interp Present (ticket 02: temporal scaler only; interp = 03).
  // Prefer over spatial so the same tick never runs spatial+sidecar / hybrid.
  if (env::getEnvVar("DXMT_METALFX_UNIFIED_TEMPORAL_INTERP") == "1") {
    if (pDevice->GetMTLDevice().supportsFXTemporalScaler()) {
      *ppSwapChain = new MTLD3D11SwapChain<SwapChainMetalFXMode::Temporal>(
          pFactory, pDevice, layer_factory.ptr(), hWnd, pDesc, pFullscreenDesc
      );
      return S_OK;
    } else {
      WARN("MetalFX temporal scaler is not supported on this device (unified path)");
    }
  }
  if (env::getEnvVar("DXMT_METALFX_SPATIAL_SWAPCHAIN") == "1") {
    if (pDevice->GetMTLDevice().supportsFXSpatialScaler()) {
      *ppSwapChain = new MTLD3D11SwapChain<SwapChainMetalFXMode::Spatial>(
          pFactory, pDevice, layer_factory.ptr(), hWnd, pDesc, pFullscreenDesc
      );
      return S_OK;
    } else {
      WARN("MetalFX spatial scaler is not supported on this device");
    }
  }
  *ppSwapChain = new MTLD3D11SwapChain<SwapChainMetalFXMode::Off>(
      pFactory, pDevice, layer_factory.ptr(), hWnd, pDesc, pFullscreenDesc
  );
  return S_OK;
};

} // namespace dxmt