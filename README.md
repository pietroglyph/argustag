# cucoslam
A CUDA-accelerated reimplementation of UcoSLAM

## Building
For the moment this *requires* Argus, and it will always require a valid CUDA installation. For these two reasons we currently only support building on the Jetson Nano's Jetpack 4.3 image. You'll also need to install `libopencv-dev` through `apt`. All other dependencies are packaged internally.
