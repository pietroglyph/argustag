#ifndef file_camera_H
#define file_camera_H

#include <cuda/runtime_api.hpp>

#include <filesystem>

namespace cuco {
class file_camera {
public:
  file_camera(const std::filesystem::path &image_path);

  void start_capture(){};

  void stop_capture(){};

  std::tuple<cuda::memory::device::unique_ptr<std::uint8_t[]>, unsigned int,
             unsigned int, unsigned int>
  get_latest_frame();

private:
  cuda::device_t current_cuda_device;

  std::size_t frame_size;
  cuda::memory::device::unique_ptr<std::uint8_t[]> frame_device;

  unsigned int width, height;
  int depth; // Note this counts the number of packed channels, not planes
};
} // namespace cuco

#endif
