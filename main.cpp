#include <cstdio>

#include <cuda/api/device.hpp>
#include <cuda/api/error.hpp>
#include <cuda/common/types.hpp>
#include <cuda/runtime_api.hpp>

#include <fmt/core.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nvapriltags/nvAprilTags.h>

#include <fstream>
#include <iostream>
#include <thread>

#include "argus_camera.h"
#include "file_camera.h"

int main(int argc, char **argv) {
  int sensor_mode = 0;
  if (argc > 1)
    sensor_mode = std::atoi(argv[1]);
  fmt::print("Using sensor mode {}\n", sensor_mode);

  if (cuda::device::count() == 0) {
    throw std::runtime_error("No CUDA devices on this system");
  }
  cuda::force_runtime_initialization();
  cuda::device::current::get().reset();

  nvAprilTagsHandle handle{};
  nvAprilTagsCameraIntrinsics_t camera_intrinsics = {
      .fx = 2399.1, .fy = 2390.1, .cx = 1618.7, .cy = 1235.5};

  cuco::argus_camera cam(0, sensor_mode);
  // std::filesystem::path img_path = "./calib_1.png";
  // cuco::file_camera cam(img_path);
  cam.start_capture();

  auto apriltag_stream =
      cuda::device::current::get().create_stream(cuda::stream::async);

  int i = 0;
  std::chrono::time_point<std::chrono::system_clock> last_loop_time;
  while (true) {
    auto [frame, width, height, depth] = cam.get_latest_frame();
    auto pitch = width * depth;

    if (!handle) {
      fmt::print("Making a new detector\n");

      int error = nvCreateAprilTagsDetector(
          &handle, width, height, NVAT_TAG36H11, &camera_intrinsics, 0.17272);
      if (error)
        throw std::runtime_error("Couldn't create ISAAC AprilTag detector");
    }

    if (!frame) {
      fmt::print("Got null frame\n");
      continue;
    }

    uint32_t num_tags_detected;
    std::array<nvAprilTagsID_t, 5> april_tags_detected{};
    nvAprilTagsImageInput_t april_tags_image{
        reinterpret_cast<uchar4 *>(frame.get()),
        static_cast<std::size_t>(pitch),
        static_cast<uint16_t>(
            width), // TODO: example code shows that so-called "pitch" is
                    // actually total image buffer length in bytes
        static_cast<uint16_t>(height)};

    int error =
        nvAprilTagsDetect(handle, &april_tags_image, april_tags_detected.data(),
                          &num_tags_detected, april_tags_detected.max_size(),
                          apriltag_stream.id());
    if (error)
      throw std::runtime_error("Couldn't detect AprilTags");
    else if (num_tags_detected > 0)
      fmt::print("Detected {} tags\n", num_tags_detected);

    fmt::print("Frame time of {} ms\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now() - last_loop_time)
                   .count());
    last_loop_time = std::chrono::system_clock::now();

    static constexpr int writing_period = 100;
    if (++i % writing_period == 0) {
      auto frame_host = std::make_unique<uint8_t[]>(pitch * height);
      cuda::memory::copy(frame_host.get(), frame.get(), pitch * height);
      cuda::device::current::get().synchronize();

      cv::Mat frame_mat{static_cast<int>(height), static_cast<int>(width),
                        static_cast<int>(CV_8UC(depth)), frame_host.get(),
                        pitch};
      cv::cvtColor(frame_mat, frame_mat,
                   cv::COLOR_RGBA2BGRA); // OpenCV wants BGRA
      cv::imwrite(fmt::format("calib_{}.png", i / writing_period), frame_mat);

      fmt::print("Wrote image {}\n", i / writing_period);
    } else {
      cam.return_frame(std::move(frame));
    }
  }

  int error = nvAprilTagsDestroy(handle);
  if (error)
    throw std::runtime_error("Couldn't destroy ISAAC AprilTag detector");

  cam.stop_capture();

  return EXIT_SUCCESS;
}
