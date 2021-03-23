# cucoslam
A CUDA-accelerated reimplementation of UcoSLAM

## Building
For the moment this *requires* Argus, and it will always require a valid CUDA installation. For these two reasons we currently only support building on the Jetson Nano's Jetpack 4.3 image. You'll also need to install `libopencv-dev` through `apt`. All other dependencies are packaged internally.

Note: the NVIDIA AprilTag library is currently required but not included for licensing reasons. The *only* version I've found that has the right device-side symbols for use indepedant of ISAAC is [here](https://github.com/NVIDIA-AI-IOT/ros2-nvapriltags/tree/6e437ac605cb91c2849ddd05b986a123e8f77843/nvapriltags). You'll have to place it in the project manually.
