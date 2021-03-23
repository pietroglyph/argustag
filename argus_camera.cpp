#include "argus_camera.h"

// #include "EGL/egl.h"
#include "utils/ArgusHelpers.h"
#include "utils/color_conversion.h"

#include <cuda/api/array.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/unique_ptr.hpp>
#include <cuda_egl_interop.h>

#include <npp.h>
#include <nppi.h>

#include <fmt/core.h>

#include <chrono>
#include <limits>
#include <mutex>
#include <stdexcept>

// This beautifies our interactions with Argus so that we don't have to use
// their auto_ptr lookalike... 🤢 Note that this is not an owning pointer;
// i.e. the unique_ptr still owns the memory this points to.
namespace Argus {
template <typename Interface, typename Object>
inline Interface *interface_cast(
    const std::unique_ptr<Object, std::function<void(Object *)>> &obj) {
  return interface_cast<Interface>(obj.get());
}
} // namespace Argus

// Note: anywhere I use Y or y to refer to the luma plane I really mean Y'

namespace cuco {

argus_camera::argus_camera(uint32_t camera_index, uint32_t sensor_mode_index)
    : current_cuda_device{cuda::device::current::get()} {
  // We assume that the CUDA Runtime API is already initialized and has chosen a
  // default device for this thread

  // Create camera provider to query connected cameras and version
  camera_provider.reset(ag::CameraProvider::create());
  ag::ICameraProvider *i_camera_provider =
      ag::interface_cast<ag::ICameraProvider>(camera_provider);
  if (!i_camera_provider)
    throw std::runtime_error("Failed to create camera provider");

  // Get the CameraDevice for the given camera_index
  ag::CameraDevice *camera_device = ArgusSamples::ArgusHelpers::getCameraDevice(
      camera_provider.get(), camera_index);
  if (!camera_device)
    throw std::runtime_error(
        fmt::format("Argus camera {} is not available", camera_index));

  // For the selected CameraDevice and the given sensor_mode_index, get the
  // sensor mode
  ag::SensorMode *sensor_mode = ArgusSamples::ArgusHelpers::getSensorMode(
      camera_device, sensor_mode_index);
  ag::ISensorMode *i_sensor_mode =
      ag::interface_cast<ag::ISensorMode>(sensor_mode);
  if (!i_sensor_mode)
    throw std::runtime_error(
        fmt::format("Sensor mode {} is not valid for Argus camera {}",
                    sensor_mode_index, camera_index));

  ArgusSamples::ArgusHelpers::printSensorModeInfo(sensor_mode, "");

  output_frame_width = i_sensor_mode->getResolution().width();
  output_frame_height = i_sensor_mode->getResolution().height();
  output_frame_channels = 4;

  // Create a capture session using the selected device
  capture_session.reset(i_camera_provider->createCaptureSession(camera_device));
  ag::ICaptureSession *i_capture_session =
      ag::interface_cast<ag::ICaptureSession>(capture_session);
  if (!i_capture_session)
    throw std::runtime_error("Failed to create Argus capture session");

  // Make settings for our OutputStream to correspond with the selected video
  // mode
  stream_settings.reset(
      i_capture_session->createOutputStreamSettings(ag::STREAM_TYPE_EGL));
  ag::IEGLOutputStreamSettings *i_stream_settings =
      ag::interface_cast<ag::IEGLOutputStreamSettings>(stream_settings);
  if (!i_stream_settings)
    throw std::runtime_error("Failed to create Argus OutputStreamSettings");
  i_stream_settings->setPixelFormat(ag::PIXEL_FMT_YCbCr_420_888);
  i_stream_settings->setResolution(i_sensor_mode->getResolution());
  i_stream_settings->setMode(
      ag::EGL_STREAM_MODE_MAILBOX); // Only keep the latest frame; TODO: FIFO
                                    // might be better?

  // Create an OutputStream for our given settings
  output_stream.reset(
      i_capture_session->createOutputStream(stream_settings.get()));
  ag::IEGLOutputStream *i_output_stream =
      ag::interface_cast<ag::IEGLOutputStream>(output_stream);
  if (!i_output_stream)
    throw std::runtime_error("Failed to create EGL OutputStream");
  egl_stream = i_output_stream->getEGLStream();

  // Create a capture request
  capture_request.reset(i_capture_session->createRequest());
  ag::IRequest *i_request = ag::interface_cast<ag::IRequest>(capture_request);
  if (!i_request)
    throw std::runtime_error("Failed to create Argus capture request");
  i_request->enableOutputStream(output_stream.get());

  // We also have to set the sensor mode on the request
  ag::ISourceSettings *i_source_settings =
      ag::interface_cast<ag::ISourceSettings>(capture_request);
  if (!i_source_settings)
    throw std::runtime_error("Failed to get source settings request interface");
  i_source_settings->setSensorMode(sensor_mode);
  //i_source_settings->setExposureTimeRange({1000, 13000 + 1}/*{i_source_settings->getExposureTimeRange().min(), i_source_settings->getExposureTimeRange().min() + 2000}*/); // 2 ms leeway
  i_source_settings->setFrameDurationRange({8333333, 8333333 + 1000});
}

void argus_camera::start_capture() {
  if (capture_thread.joinable())
    throw std::logic_error("Can't start a camera that is already capturing");

  // Now we connect CUDA to the EGLStreamKHR
  // We assume that the Runtime API has attached a context which matches to the
  // current device

  // Connect our CUDA consumer (via stream_connection) to the EGL stream
  cudaError_t res =
      cudaEGLStreamConsumerConnect(&stream_connection, egl_stream);
  if (res != cudaError::cudaSuccess)
    throw std::runtime_error(
        fmt::format("Unable to connect CUDA to EGLStream as a consumer: {}",
                    cudaGetErrorString(res)));

  should_capture = true;

  capture_thread = std::thread([&]() {
    // We bind the CUDA device that was active for the spawning thread (usually
    // the main thread) to this new thread
    current_cuda_device.make_current();

    // The "correct" way to do this is to query the stream state with
    // eglQueryStreamKHR. Unfortunately, eglQueryStreamKHR doesn't work without
    // a valid EGLDisplay, and we want to run headless.
    while (!frame_producer_ready) {
      std::this_thread::yield();
    }

    auto image_plane_ptrs = cuda::memory::managed::make_unique<uint8_t *[]>(2);
    cuda::memory::device::unique_ptr<uint8_t[]> y_plane;
    cuda::memory::device::unique_ptr<uint8_t[]> cbcr_plane;

    auto argus_async_cuda_stream = current_cuda_device.create_stream(cuda::stream::async);

    while (should_capture) {
      // Acquire a frame from the EGLStream as a CUDA resource
      cudaGraphicsResource_t frame_resource{};
      cudaError_t res = cudaEGLStreamConsumerAcquireFrame(
          &stream_connection, &frame_resource,
          nullptr /* Send request over the default CUDA stream */,
          -1 /* No timeout */);
      cuda::throw_if_error(res, "Unable to acquire a frame from the EGLStream");

      // Get an EGLFrame from the CUDA resource
      cudaEglFrame egl_frame{};
      res = cudaGraphicsResourceGetMappedEglFrame(&egl_frame, frame_resource, 0,
                                                  0);
      cuda::throw_if_error(res, "Unable to get EGLFrame from a CUDA resource");

      // Y'CbCr 4:2:0 Semiplanar (2-plane) Extended Range is more commonly known
      // as NV12
      if (egl_frame.eglColorFormat !=
          cudaEglColorFormat::cudaEglColorFormatYUV420SemiPlanar_ER)
        throw std::runtime_error(
            fmt::format("Frame color format (ID {}) is unsupported; only "
                        "extended-range Y'CbCr 4:2:0 semiplanar is supported",
                        egl_frame.eglColorFormat));
      else if (egl_frame.frameType != cudaEglFrameType::cudaEglFrameTypeArray)
        throw std::runtime_error(
            "Only array-type (non-pitched) frames are supported");

      // Note: we assume no padding; the width * channels = the pitch of the Y' and
      // CbCr planes
      std::size_t pitch_y = egl_frame.planeDesc[0].width;
      std::size_t pitch_cbcr =
          egl_frame.planeDesc[1].width *
          2; // the chroma plane is effectively two channel; one channel for Cb and one for Cr

      if (!y_plane || !cbcr_plane) {
        y_plane = cuda::memory::device::make_unique<uint8_t[]>(
            current_cuda_device, pitch_y * egl_frame.planeDesc[0].height);
        cbcr_plane = cuda::memory::device::make_unique<uint8_t[]>(
            current_cuda_device, pitch_cbcr * egl_frame.planeDesc[1].height);
      }

      std::size_t pitch_rgba =
          output_frame_channels * egl_frame.planeDesc[0].width;

      // Copy from the array we get back to a contiguous block of memory, row
      // major... Could use the C++ wrapper copy function here, but the
      // unwrapped C-style call is actually more concise (we'd have to make a
      // non-owning cuda::array_t to use the wrapper function.)
      // TODO: A kernel that converts as it reads from a surface bound to the
      // array would probably be faster.
      res = cudaMemcpy2DFromArray(y_plane.get(), pitch_y, egl_frame.frame.pArray[0],
                            0, 0, pitch_y, egl_frame.planeDesc[0].height,
                            cudaMemcpyKind::cudaMemcpyDefault);
      cuda::throw_if_error(res, "Couldn't copy Y' plane from a surfae to a raw buffer");
      cudaMemcpy2DFromArray(cbcr_plane.get(), pitch_cbcr,
                            egl_frame.frame.pArray[1], 0, 0, pitch_cbcr,
                            egl_frame.planeDesc[1].height,
                            cudaMemcpyKind::cudaMemcpyDefault);
      cuda::throw_if_error(res, "Couldn't copy CbCr plane from a surface to a raw buffer");

      cuda::synchronize(current_cuda_device);

      {
        std::scoped_lock lk(frame_pool_mutex);

	// This effectively fills the frame pool until we reach an equilibrium where it's filled enough to not need new allocations
        if (frame_pool.empty()) {
          fmt::print("Frame pool was empty; allocating memory for a new one\n");

          frame_pool.emplace_back(cuda::memory::device::make_unique<uint8_t[]>(
              current_cuda_device, pitch_rgba * egl_frame.planeDesc[0].height));
        }

        cudaError_t res = cudaNV12ToRGBX(
            y_plane.get(), cbcr_plane.get(),
            reinterpret_cast<uchar4 *>(frame_pool.back().get()),
            egl_frame.planeDesc[0].width, egl_frame.planeDesc[0].height, argus_async_cuda_stream);
	cuda::throw_if_error(res, "Couldn't convert from NV12 to RGBX");

        cuda::synchronize(current_cuda_device);

        new_frame_available = true;
      }

      new_frame_available_cv.notify_all();

      res = cudaEGLStreamConsumerReleaseFrame(
          &stream_connection, frame_resource,
          nullptr /* Send request over the default CUDA stream */);
      cuda::throw_if_error(res, "Couldn't release frame to EGLStream for reuse");
    }
  });

    ag::ICaptureSession *i_capture_session =
        ag::interface_cast<ag::ICaptureSession>(capture_session);
    if (!i_capture_session)
      throw std::runtime_error("Failed to get Argus capture session interface");

    i_capture_session->repeat(capture_request.get());
    frame_producer_ready = true;
}

void argus_camera::stop_capture() {
  if (!capture_thread.joinable())
    throw std::logic_error("Can't stop a camera that isn't capturing");

  should_capture = false;
  frame_producer_ready = false;

  
    ag::ICaptureSession *i_capture_session =
        ag::interface_cast<ag::ICaptureSession>(capture_session);
    if (!i_capture_session)
      throw std::runtime_error("Failed to get Argus capture session interface");
    i_capture_session->stopRepeat();
  capture_thread.join();

  cudaError_t res = cudaEGLStreamConsumerDisconnect(&stream_connection);
  cuda::throw_if_error(res, "Unable to disconnect CUDA from EGLStream");
}

std::tuple<cuda::memory::device::unique_ptr<std::uint8_t[]>, unsigned int,
           unsigned int, unsigned int>
argus_camera::get_latest_frame() {
  std::unique_lock lk(frame_pool_mutex);
  new_frame_available_cv.wait(lk,
                              [&]() {
                                return new_frame_available.load();
                              });
  new_frame_available = false;

  // Should never happen, but we'll be defensive because calling std::vector::back on an empty vector is UB
  if (frame_pool.empty()) throw std::runtime_error("Frame pool was empty after condition variable triggered");
  auto ret = std::tuple{std::move(frame_pool.back()), output_frame_width, output_frame_height,
          output_frame_channels};
  frame_pool.pop_back();
  return ret;
}
} // namespace cuco
