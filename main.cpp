#include <cstdio>

#include <cuda/api/device.hpp>
#include <cuda/runtime_api.hpp>

#include <fmt/core.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <thread>

#include "argus_camera.h"

int main(int argc, char **argv) {
  int sensorMode = 0;
  if (argc > 1)
    sensorMode = std::atoi(argv[1]);

  if (cuda::device::count() == 0) {
    throw std::runtime_error("No CUDA devices on this system");
  }
  cuda::force_runtime_initialization();
  cuda::device::current::get().reset();

  cuco::argus_camera cam(0, sensorMode);
  cam.start_capture();

  int i = 0;
  while (true) {
    // Simulate some other processing so that we don't contend the lock
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(10ms);

    auto [frame, width, height, depth] = cam.get_latest_frame();
    auto pitch = width * depth;
    if (frame && ++i > 5) {
      auto frame_host = std::make_unique<uint8_t[]>(pitch * height);
      cuda::memory::copy(frame_host.get(), frame.get(), pitch * height);
      cuda::device::current::get().synchronize();

      cv::Mat frame_mat{static_cast<int>(height), static_cast<int>(width),
                        static_cast<int>(CV_8UC(depth)), frame_host.get(),
                        pitch};
      cv::cvtColor(frame_mat, frame_mat,
                   cv::COLOR_RGBA2BGRA); // OpenCV wants BGRA
      cv::imwrite("out.png", frame_mat);

      fmt::print("Written\n");

      break;
    }
  }

  cam.stop_capture();

  return EXIT_SUCCESS;
}
