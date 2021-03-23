#ifndef argus_camera_H
#define argus_camera_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>

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

  std::tuple<cuda::memory::device::unique_ptr<std::uint8_t[]>, unsigned int,
             unsigned int, unsigned int>
  get_latest_frame();

  void return_frame(cuda::memory::device::unique_ptr<std::uint8_t[]> frame) {
    std::scoped_lock lk(frame_pool_mutex);

    static constexpr int max_frame_pool_size = 5;
    if (frame_pool.size() < max_frame_pool_size && frame)
      frame_pool.emplace_back(std::move(frame));
  }

private:
  // This is the device active for the thread that's instantiating and
  // controlling this class. We hold it so that it can be made active for the
  // frame receiving thread.
  cuda::device_t current_cuda_device;

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

  std::mutex frame_pool_mutex;
  std::vector<cuda::memory::device::unique_ptr<std::uint8_t[]>> frame_pool;
  std::condition_variable new_frame_available_cv;
  std::atomic_bool new_frame_available{false};

  unsigned int output_frame_width, output_frame_height;
  int output_frame_channels; // Right now this is always 4, because we return
                             // RGBX; note: channels != planes

  std::atomic_bool frame_producer_ready{false};
  std::atomic_bool should_capture{false};
  std::thread capture_thread;
  std::thread capture_request_thread;
};

} // namespace cuco

#endif // argus_camera_H
