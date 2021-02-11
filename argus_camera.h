#ifndef argus_camera_H
#define argus_camera_H

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include <cuda/runtime_api.hpp>

#include <Argus/Argus.h>

// We forward declare these because the headers they're defined in generally
// aren't included in any other compilation units (while other non-std types,
// e.g. the CUDA runtime C++ wrapper are)
typedef void *EGLStreamKHR;
typedef struct CUeglStreamConnection_st *cudaEGLStreamConnection;

namespace cuco {
namespace ag =
    Argus; // Fits our style conventions more, also shorter than Argus

class argus_camera {
public:
  argus_camera(std::uint32_t camera_index, std::uint32_t sensorModeIndex);

  void start_capture();

  void stop_capture();

  cuda::memory::device::unique_ptr<std::uint8_t[]> get_latest_frame();

private:
  // This is the device active for the thread that's instantiating and
  // controlling this class. We hold it so that it can be made active for the
  // frame receiving thread.
  cuda::device_t device;

  // Ew. FIXME.
  std::unique_ptr<ag::CameraProvider, std::function<void(ag::CameraProvider *)>>
      camera_provider{nullptr, [](auto p) { p->destroy(); }};
  std::unique_ptr<ag::CaptureSession, std::function<void(ag::CaptureSession *)>>
      capture_session{nullptr, [](auto p) { p->destroy(); }};
  std::unique_ptr<ag::OutputStreamSettings,
                  std::function<void(ag::OutputStreamSettings *)>>
      stream_settings{nullptr, [](auto p) { p->destroy(); }};
  std::unique_ptr<ag::OutputStream, std::function<void(ag::OutputStream *)>>
      output_stream{nullptr, [](auto p) { p->destroy(); }};
  std::unique_ptr<ag::Request, std::function<void(ag::Request *)>>
      capture_request{nullptr, [](auto p) { p->destroy(); }};

  EGLStreamKHR egl_stream;
  cudaEGLStreamConnection stream_connection;

  std::mutex latest_frame_mutex;
  cuda::memory::device::unique_ptr<std::uint8_t[]> latest_frame;

  std::atomic_bool frame_producer_ready{false};
  std::atomic_bool should_capture{false};
  std::thread capture_thread;
};

} // namespace cuco

#endif // argus_camera_H
