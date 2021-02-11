#include <cstdio>

#include <cuda/api/device.hpp>
#include <cuda/runtime_api.hpp>

#include <fmt/core.h>

#include <thread>

#include "argus_camera.h"
#include "utils/CUDAHelper.h"

int main(int argc, char **argv) {
  // First, initilize a context through the driver API
  // CUcontext cudaContext;
  // if (!ArgusSamples::initCUDA(&cudaContext)) {
  //     throw std::runtime_error("Couldn't initialize CUDA through the driver
  //     API");
  // }

  // From here on in we use the runtime API through the C++ wrappers
  if (cuda::device::count() == 0) {
    throw std::runtime_error("No CUDA devices on this system");
  }
  cuda::force_runtime_initialization();

  // cuda::device::current::set_to_default();
  // auto device = cuda::device::current::get();
  // device.make_current();
  // fmt::print("Device name: {}\n", device.name());

  // cuco::argus_camera cam(0, 0);
  // cam.start_capture();

  // while (true) {
  //     std::this_thread::yield();
  // }

  return EXIT_SUCCESS;
}