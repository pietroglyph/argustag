#ifndef COLOR_CONVERSION_H
#define COLOR_CONVERSION_H

#include <cstddef>
#include <cstdint>

#include <cuda/runtime_api.hpp>
#include <vector_types.h>

/**
 * Converts Y'CbCr 4:2:0 semiplanar extended-range to RGBX in the sRGB
 * colorspace, with the X byte set to 0xFF.
 */
cudaError_t cudaNV12ToRGBX(const uint8_t *__restrict__ src_luma,
                           const uint8_t *__restrict__ src_chroma,
                           uchar4 *__restrict__ dst, unsigned int width,
                           unsigned int height, cuda::stream_t &stream);

// Used to allow dealing with generic vector types
// TODO: Remove or at least use decltype
template <class T> struct cudaVectorTypeInfo;

template <> struct cudaVectorTypeInfo<uchar1> { typedef uint8_t Base; };
template <> struct cudaVectorTypeInfo<uchar3> { typedef uint8_t Base; };
template <> struct cudaVectorTypeInfo<uchar4> { typedef uint8_t Base; };

template <> struct cudaVectorTypeInfo<float> { typedef float Base; };
template <> struct cudaVectorTypeInfo<float3> { typedef float Base; };
template <> struct cudaVectorTypeInfo<float4> { typedef float Base; };

template <typename T> struct cuda_assert_false : std::false_type {};

// make_vec and specializations
template <typename T>
inline __host__ __device__ T make_vec(typename cudaVectorTypeInfo<T>::Base,
                                      typename cudaVectorTypeInfo<T>::Base,
                                      typename cudaVectorTypeInfo<T>::Base,
                                      typename cudaVectorTypeInfo<T>::Base) {
  static_assert(cuda_assert_false<T>::value,
                "invalid vector type - supported types are uchar3, uchar4, "
                "float3, float4");
}

template <>
inline __host__ __device__ uchar1 make_vec(uint8_t x, uint8_t, uint8_t,
                                           uint8_t) {
  return make_uchar1(x);
}
template <>
inline __host__ __device__ uchar3 make_vec(uint8_t x, uint8_t y, uint8_t z,
                                           uint8_t) {
  return make_uchar3(x, y, z);
}
template <>
inline __host__ __device__ uchar4 make_vec(uint8_t x, uint8_t y, uint8_t z,
                                           uint8_t w) {
  return make_uchar4(x, y, z, w);
}

template <>
inline __host__ __device__ float make_vec(float x, float, float, float) {
  return x;
}
template <>
inline __host__ __device__ float3 make_vec(float x, float y, float z, float) {
  return make_float3(x, y, z);
}
template <>
inline __host__ __device__ float4 make_vec(float x, float y, float z, float w) {
  return make_float4(x, y, z, w);
}

// extern void ycbcr420er_to_rgb(const cudaSurfaceObject_t y_plane, const
// cudaSurfaceObject_t uv_plane, const unsigned int width, const unsigned int
// height, uint8_t *rgb_out);

#endif
