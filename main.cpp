#include <cstdio>

#include <cuda/api/device.hpp>
#include <cuda/runtime_api.hpp>

#include <fmt/core.h>

#include <fstream>
#include <iostream>
#include <thread>

#include "argus_camera.h"
#include "utils/CUDAHelper.h"

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

    auto frame = cam.get_latest_frame();
    if (frame && ++i > 20) {
      unsigned int dimx = 1280, dimy = 720;

      std::ofstream ofs("first.ppm", std::ios::out | std::ios::binary);
      ofs << "P6"
          << "\n"
          << dimx << ' ' << dimy << "\n"
          << "255"
          << "\n";

      auto frame_host = std::make_unique<uint8_t[]>(dimx * dimy * 3);
      cuda::memory::copy(frame_host.get(), frame.get(), dimx * dimy * 3);
      auto frame_host_buf = frame_host.get();

      for (auto y = 0u; y < dimy; ++y) {
        for (auto x = 0u; x < dimx; ++x) {
          auto i = y * dimy * 3 + x * 3;
          ofs << static_cast<char>(frame_host_buf[i])
              << static_cast<char>(frame_host_buf[i + 1])
              << static_cast<char>(frame_host_buf[i + 2]);
        }
      }

      break;
    }
  }

  return EXIT_SUCCESS;
}
