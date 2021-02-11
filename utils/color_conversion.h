#ifndef COLOR_CONVERSION_H
#define COLOR_CONVERSION_H

#include <cuda/runtime_api.hpp>

extern void ycbcr420er_to_rgb(const cudaSurfaceObject_t y_plane, const cudaSurfaceObject_t uv_plane, const unsigned int width, const unsigned int height, uint8_t *rgb_out);

#endif