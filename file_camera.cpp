#include "file_camera.h"

#include <fmt/core.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace cuco {
file_camera::file_camera(const std::filesystem::path &image_path)
    : current_cuda_device{cuda::device::current::get()} {
  if (image_path.extension() != "png")
    throw std::runtime_error(fmt::format(
        "Only .png files are accepted by file_camera; {} is invalid",
        image_path.filename().c_str()));
  cv::Mat frame_host = cv::imread(image_path.c_str());
  if (frame_host.type() != CV_8UC4)
    throw std::runtime_error(fmt::format(
        "Only 8-bit 4-channel images are supported; yours is of type {}",
        frame_host.type()));

  frame_size = frame_host.total() * frame_host.elemSize();
  width = frame_host.cols;
  height = frame_host.rows;
  depth = frame_host.depth();

  frame_device = cuda::memory::device::make_unique<uint8_t[]>(
      current_cuda_device, frame_size);
  cuda::memory::copy(frame_device.get(), frame_host.data, frame_size);
}

std::tuple<cuda::memory::device::unique_ptr<std::uint8_t[]>, unsigned int,
           unsigned int, unsigned int>
file_camera::get_latest_frame() {
  // We should switch to a shared_ptr return for efficiency; this is just for
  // testing rn so haven't done it because it isn't convenient
  auto frame_to_return = cuda::memory::device::make_unique<uint8_t[]>(
      current_cuda_device, frame_size);
  cuda::memory::copy(frame_to_return.get(), frame_device.get(), frame_size);
  return {std::move(frame_to_return), width, height, depth};
}
} // namespace cuco
