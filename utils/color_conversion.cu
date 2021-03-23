#include "color_conversion.h"

// Note that all the templatized functions labelled RGBX can also work in RGB if
// the template param is a 3-dimensional vector type (uint3)

static inline __device__ float clamp(float x) {
  return fminf(fmaxf(x, 0.0f), 255.0f);
}

template <typename T>
static inline __device__ T YCbCr_to_RGBX(const uint3 &yuvi) {
  const float luma = float(yuvi.x);
  const float cb = float(yuvi.y) - 128.0f;
  const float cr = float(yuvi.z) - 128.0f;

  // Convert from Y'CbCr extended range to RGB(X) in the sRGB colorspace
  // Coefficients
  // https://web.archive.org/web/20180421030430/http://www.equasys.de/colorconversion.html
  return make_vec<T>(clamp(luma + 1.400f * cr),
                     clamp(luma - 0.343f * cb - 0.711f * cr),
                     clamp(luma + 1.765f * cb), 0xFF);
}

template <typename T>
__global__ void
NV12ToRGBX(const __restrict__ uint8_t *src_luma,
           const __restrict__ uint8_t *src_chroma, std::size_t src_luma_pitch,
           std::size_t src_chroma_pitch, T *dst_image, std::size_t dst_pitch,
           uint32_t width, uint32_t height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  uint8_t luma, chroma_cb, chroma_cr;

  luma = src_luma[y * src_luma_pitch + x];
  chroma_cb = src_chroma[(y / 2) * src_chroma_pitch + 2 * (x / 2)];
  chroma_cr = src_chroma[(y / 2) * src_chroma_pitch + 2 * (x / 2) + 1];

  dst_image[y * dst_pitch + x] =
      YCbCr_to_RGBX<T>(make_uint3(luma, chroma_cb, chroma_cr));
}

static inline int div_round_up(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

template <typename T>
static cudaError_t launchNV12ToRGBX(const __restrict__ uint8_t *src_luma,
                                    const __restrict__ uint8_t *src_chroma,
                                    T *dst, std::size_t width,
                                    std::size_t height, cuda::stream_t& stream) {
  if (!src_luma || !src_chroma || !dst)
    return cudaError::cudaErrorInvalidDevicePointer;
  if (width == 0 || height == 0)
    return cudaError::cudaErrorInvalidValue;

  const auto src_luma_pitch = width * sizeof(uint8_t);
  const auto src_chroma_pitch =
      width * sizeof(uint8_t); // Chroma pitch = image width / 2 * pitch where
                               // pitch = 2, so chroma pitch = image width
  const auto dst_pitch =
      width; // Note: no sizeof(T) because we're indexing over T and not uint8_t

  // TODO: change the y block dim to achieve maximum occupancy in each SM...
  // right now each SM might have more than 8 warps
  static constexpr int warp_size = 32;
  const dim3 block_dim(warp_size, 8);
  const dim3 grid_dim(div_round_up(width, block_dim.x),
                      div_round_up(height, block_dim.y));

  //auto launch_config = cuda::make_launch_config(block_dim, grid_dim);
  //stream.enqueue.kernel_launch(cudaNV12ToRGBX<T>, launch_config, src_luma, src_chroma, src_luma_pitch, src_chroma_pitch, dst, dst_pitch, width, height);
  NV12ToRGBX<T><<<grid_dim, block_dim, 0, stream.id()>>>(src_luma, src_chroma, src_luma_pitch,
                                         src_chroma_pitch, dst, dst_pitch,
                                         width, height);

  return cudaGetLastError();
}

cudaError_t cudaNV12ToRGBX(const uint8_t *__restrict__ src_luma,
                           const uint8_t *__restrict__ src_chroma,
                           uchar4 *__restrict__ dst, unsigned int width,
                           unsigned int height, cuda::stream_t& stream) {
  return launchNV12ToRGBX(src_luma, src_chroma, dst, width, height, stream);
}
