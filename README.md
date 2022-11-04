# argustag
 A C++17 wrapper for Nvidia Argus with support for zero-copy frame transfers to CUDA kernels and CUDA-accelerated AprilTag detection with ISAAC (no ROS required).

## Features
 - Modern C++ wrapper for Argus that makes it easy to get frames to a CUDA kernel as fast as possible, with no copying--I read Nvidia's examples and the Argus headers so you don't have to!
 - Example integration with the ISAAC GPU-accelerated AprilTag detector that does **not** require ROS

## Usage
This is excerpted from a closed-source SLAM system based on UcoSLAM that I wrote (called "cucoslam" for *C*UDA and *UcoSLAM*; this is why the top-level namespace is `cuco`). I took all the commits up to a certain data that are relevant to someone trying to use Argus with CUDA or trying to use Argus with the ISAAC AprilTag detector. This isn't really ready to be a nice library that you can plug and play with, but I'd like to think that it's pretty close. It should at least serve as a good starting point.

## Building
For the moment this *requires* Argus, and it will always require a valid CUDA installation (11 or later). For these two reasons we currently only support building on the Jetson Nano's Jetpack 4.3 image. You'll also need to install `libopencv-dev` through `apt`. All other dependencies are packaged internally.

Note: the NVIDIA AprilTag library is currently required but not included for licensing reasons. The *only* version I've found that has the right device-side symbols for use indepedant of ISAAC is [here](https://github.com/NVIDIA-AI-IOT/ros2-nvapriltags/tree/6e437ac605cb91c2849ddd05b986a123e8f77843/nvapriltags). You'll have to place it in the project manually.
