#include "dxmt_scaler.hpp"

namespace dxmt {

void
SpatialScaler::incRef() {
  refcount_.fetch_add(1u, std::memory_order_acquire);
}

void
SpatialScaler::decRef() {
  if (refcount_.fetch_sub(1u, std::memory_order_release) == 1u)
    delete this;
}

SpatialScaler::SpatialScaler(WMT::Device device, const WMTFXSpatialScalerInfo &info) {
  scaler_ = device.newSpatialScaler(info);
  fence_ = device.newFence();
};

void
TemporalScaler::incRef() {
  refcount_.fetch_add(1u, std::memory_order_acquire);
}

void
TemporalScaler::decRef() {
  if (refcount_.fetch_sub(1u, std::memory_order_release) == 1u)
    delete this;
}

TemporalScaler::TemporalScaler(WMT::Device device, const WMTFXTemporalScalerInfo &info) {
  scaler_ = device.newTemporalScaler(info);
  fence_ = device.newFence();
}

void
FrameInterpolator::incRef() {
  refcount_.fetch_add(1u, std::memory_order_acquire);
}

void
FrameInterpolator::decRef() {
  if (refcount_.fetch_sub(1u, std::memory_order_release) == 1u)
    delete this;
}

FrameInterpolator::FrameInterpolator(WMT::Device device, const WMTFXFrameInterpolatorInfo &info) {
  interpolator_ = device.newFrameInterpolator(info);
  fence_ = device.newFence();
}

} // namespace dxmt
