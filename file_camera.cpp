#include "file_camera.h"

#include <fmt/core.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace cuco {
file_camera::file_camera(const std::filesystem::path &image_path)
    : current_cuda_device{cuda::device::current::get()} {
  if (image_path.extension() != ".png")
    throw std::runtime_error(fmt::format(
        "Only .png files are accepted by file_camera; {} is invalid",
        image_path.filename().c_str()));
  cv::Mat frame_host = cv::imread(image_path.c_str());

  // Kind of sloppy... assumes 3-channel is RGB (could be BGR or even Y'CbCr
  // 4:4:4)
  if (frame_host.type() == CV_8UC3)
    cv::cvtColor(frame_host, frame_host, cv::COLOR_RGB2RGBA);
  else if (frame_host.type() != CV_8UC4)
    throw std::runtime_error(fmt::format(
        "Only 8-bit 3 and 4-channel images are supported; yours is of type {}",
        frame_host.type()));

  frame_size = frame_host.total() * frame_host.elemSize();
  width = frame_host.cols;
  height = frame_host.rows;
  channels = frame_host.channels();

  frame_device = cuda::memory::device::make_unique<uint8_t[]>(
      current_cuda_device, frame_size);
  cuda::memory::copy(frame_device.get(), frame_host.data, frame_size);
}

std::tuple<cuda::memory::device::unique_ptr<std::uint8_t[]>, unsigned int,
           unsigned int, unsigned int>
file_camera::get_latest_frame() {
  std::scoped_lock lk(frame_to_return_mutex);

  if (!frame_to_return_device) {
    fmt::print("Had to make a copy\n");
    frame_to_return_device = cuda::memory::device::make_unique<uint8_t[]>(
        current_cuda_device, frame_size);
    cuda::memory::copy(frame_to_return_device.get(), frame_device.get(),
                       frame_size);
  }
  return {std::move(frame_to_return_device), width, height, channels};
}
} // namespace cuco
