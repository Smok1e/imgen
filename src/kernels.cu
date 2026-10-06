#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <kernels.hpp>

//========================================

const glm::uvec2 GLYPH_SIZE(8, 16);
const glm::mat4x4 BT601(
	1.16438,  0.00000,  1.59603, -0.87079,
	1.16438, -0.39176, -0.81297,  0.52959,
	1.16438,  2.01723,  0.00000, -1.08139,
	0.00000,  0.00000,  0.00000,  1.00000
);

__constant__ glm::uvec2  DEVICE_GLYPH_SIZE;
__constant__ glm::mat4x4 DEVICE_BT601;

/*
__constant__ constexpr float DEVICE_ORDERED_DITHER_COEFFS[] = {
     0.f / 64.f, 32.f / 64.f,  8.f / 64.f, 40.f / 64.f,  2.f / 64.f, 34.f / 64.f, 10.f / 64.f, 42.f / 64.f,
    48.f / 64.f, 16.f / 64.f, 56.f / 64.f, 24.f / 64.f, 50.f / 64.f, 18.f / 64.f, 58.f / 64.f, 26.f / 64.f,
    12.f / 64.f, 44.f / 64.f,  4.f / 64.f, 36.f / 64.f, 14.f / 64.f, 46.f / 64.f,  6.f / 64.f, 38.f / 64.f,
    60.f / 64.f, 28.f / 64.f, 52.f / 64.f, 20.f / 64.f, 62.f / 64.f, 30.f / 64.f, 54.f / 64.f, 22.f / 64.f,
     3.f / 64.f, 35.f / 64.f, 11.f / 64.f, 43.f / 64.f,  1.f / 64.f, 33.f / 64.f,  9.f / 64.f, 41.f / 64.f,
    51.f / 64.f, 19.f / 64.f, 59.f / 64.f, 27.f / 64.f, 49.f / 64.f, 17.f / 64.f, 57.f / 64.f, 25.f / 64.f,
    15.f / 64.f, 47.f / 64.f,  7.f / 64.f, 39.f / 64.f, 13.f / 64.f, 45.f / 64.f,  5.f / 64.f, 37.f / 64.f,
    63.f / 64.f, 31.f / 64.f, 55.f / 64.f, 23.f / 64.f, 61.f / 64.f, 29.f / 64.f, 53.f / 64.f, 21.f / 64.f
};

__constant__ constexpr size_t DEVICE_ORDERED_DITHER_POWER = 8;
*/

__constant__ constexpr float DEVICE_ORDERED_DITHER_COEFFS[] = {
	 0.f / 16,  8.f / 16,  2.f / 16, 10.f / 16, 
	12.f / 16,  4.f / 16, 14.f / 16,  6.f / 16, 
	 3.f / 16, 11.f / 16,  1.f / 16,  9.f / 16, 
	15.f / 16,  7.f / 16, 13.f / 16,  5.f / 16
};

__constant__ constexpr size_t DEVICE_ORDERED_DITHER_POWER = 4;

/*
__constant__ constexpr float DEVICE_ORDERED_DITHER_COEFFS[] = {
	0, 2,
	3, 1
};

__constant__ constexpr size_t DEVICE_ORDERED_DITHER_POWER = 2;
*/

//========================================

__global__ void ApproximationKernel(
	KernelArray<KernelGlyph> font,
	KernelArray<glm::u8vec3> palette,
	KernelArray2D<glm::u8vec4> reference_image,
	KernelArray2D<KernelCell> approximated_image,
	float r
);

__global__ void RasterizationKernel(
	KernelArray<KernelGlyph> font,
	KernelArray2D<KernelCell> approximated_image,
	KernelArray2D<glm::u8vec4> rasterized_image
);

__global__ void FrameDecodeKernel(
	KernelArray2D<uint8_t> y_plane,
	KernelArray2D<uint8_t> cb_plane,
	KernelArray2D<uint8_t> cr_plane,
	KernelArray2D<glm::u8vec4> frame
);

__global__ void ImageQuantizationKernel(
	KernelArray<glm::u8vec3> palette,
	KernelArray2D<glm::u8vec4> src,
	KernelArray2D<glm::u8vec4> dst
);

__global__ void ImageOrderedDitherKernel(
	KernelArray<glm::u8vec3> palette,
	KernelArray2D<glm::u8vec4> src,
	KernelArray2D<glm::u8vec4> dst,
	float r
);

__global__ void QuantizationKernel(
	KernelArray<glm::u8vec3> palette,
	KernelArray2D<KernelCell> image
);

__global__ void OrderedDitherKernel(
	KernelArray<glm::u8vec3> palette,
	KernelArray2D<KernelCell> image,
	float r
);

__device__ bool DeviceGetGlyphPixel(
	const uint8_t (&data)[16], 
	const glm::uvec2& position
);

__device__ glm::u8vec4& DeviceGetImagePixel(
	const KernelArray2D<glm::u8vec4>& image,
	const glm::uvec2& position
);

__device__ int DeviceCheckThreadIndex(
	const glm::uvec2& dim
);

__device__ glm::uvec2 DeviceCalculateArrayPosition(
	size_t index, 
	const glm::uvec2& dim
);

__device__ size_t DeviceFindClosestPaletteIndex(
	const KernelArray<glm::u8vec3>& palette,
	const glm::ivec3 color
);

//========================================

constexpr size_t CalculateBlockCount(size_t thread_count, size_t task_count)
{
	return (task_count + thread_count - 1) / thread_count;
}

//========================================

void InitializeKernelConstants()
{
	cudaMemcpyToSymbol(DEVICE_GLYPH_SIZE, &GLYPH_SIZE, sizeof(GLYPH_SIZE));
	cudaMemcpyToSymbol(DEVICE_BT601,      &BT601,      sizeof(BT601)     );
}

cudaError_t LaunchApproximation(
	const KernelArray<KernelGlyph>& font,
	const KernelArray<glm::u8vec3>& palette,
	const KernelArray2D<glm::u8vec4>& reference_image,
	const KernelArray2D<KernelCell>& approximated_image,
	float r
)
{
	constexpr size_t thread_count = 16;

	size_t cell_count = reference_image.size.x * reference_image.size.y;
	size_t block_count = CalculateBlockCount(thread_count, cell_count);

	ApproximationKernel<<<block_count, thread_count>>>(
		font, 
		palette,
		reference_image, 
		approximated_image,
		r
	);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}

cudaError_t LaunchRasterization(
	const KernelArray<KernelGlyph>& font,
	const KernelArray2D<KernelCell>& approximated_image,
	const KernelArray2D<glm::u8vec4>& rasterized_image
)
{
	constexpr size_t thread_count = 128;

	size_t cell_count = approximated_image.size.x * approximated_image.size.y;
	size_t block_count = CalculateBlockCount(thread_count, cell_count);

	RasterizationKernel<<<block_count, thread_count>>>(
		font,
		approximated_image,
		rasterized_image
	);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}

cudaError_t LaunchFrameDecode(
	const KernelArray2D<uint8_t>& y_plane,
	const KernelArray2D<uint8_t>& cb_plane,
	const KernelArray2D<uint8_t>& cr_plane,
	const KernelArray2D<glm::u8vec4>& frame
)
{
	constexpr size_t thread_count = 16;

	size_t pixel_count = frame.size.x * frame.size.y;
	size_t block_count = CalculateBlockCount(thread_count, pixel_count);

	FrameDecodeKernel<<<block_count, thread_count>>>(
		y_plane,
		cb_plane,
		cr_plane,
		frame
	);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}

cudaError_t LaunchImageQuantization(
	const KernelArray<glm::u8vec3>& palette,
	const KernelArray2D<glm::u8vec4>& src,
	const KernelArray2D<glm::u8vec4>& dst
)
{
	constexpr size_t thread_count = 128;

	size_t pixel_count = src.size.x * src.size.y;
	size_t block_count = CalculateBlockCount(thread_count, pixel_count);

	ImageQuantizationKernel<<<block_count, thread_count>>>(palette, src, dst);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}

cudaError_t LaunchImageOrderedDither(
	const KernelArray<glm::u8vec3>& palette,
	const KernelArray2D<glm::u8vec4>& src,
	const KernelArray2D<glm::u8vec4>& dst,
	float r
)
{
	constexpr size_t thread_count = 128;

	size_t pixel_count = src.size.x * src.size.y;
	size_t block_count = CalculateBlockCount(thread_count, pixel_count);

	ImageOrderedDitherKernel<<<block_count, thread_count>>>(palette, src, dst, r);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}

cudaError_t LaunchQuantization(
	const KernelArray<glm::u8vec3>& palette,
	const KernelArray2D<KernelCell>& image
)
{
	constexpr size_t thread_count = 128;

	size_t pixel_count = image.size.x * image.size.y;
	size_t block_count = CalculateBlockCount(thread_count, pixel_count);

	QuantizationKernel<<<block_count, thread_count>>>(palette, image);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}

cudaError_t LaunchOrderedDither(
	const KernelArray<glm::u8vec3>& palette,
	const KernelArray2D<KernelCell>& image,
	float r
)
{
	constexpr size_t thread_count = 128;

	size_t pixel_count = image.size.x * image.size.y;
	size_t block_count = CalculateBlockCount(thread_count, pixel_count);

	OrderedDitherKernel<<<block_count, thread_count>>>(palette, image, r);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}

//========================================

__global__ void ApproximationKernel(
	KernelArray<KernelGlyph> font,
	KernelArray<glm::u8vec3> palette,
	KernelArray2D<glm::u8vec4> reference_image,
	KernelArray2D<KernelCell> approximated_image,
	float r
)
{
	int index = DeviceCheckThreadIndex(approximated_image.size);
	if (index < 0)
		return;

	glm::uvec2 cell_position = DeviceCalculateArrayPosition(
		index, 
		approximated_image.size
	);

	glm::uvec2 image_offset = cell_position * glm::uvec2(8, 16);

	int best_err = -1;
	size_t best_glyph = 0;
	glm::u8vec3 best_background = {};
	glm::u8vec3 best_foreground = {};

	size_t coeff_index = (
		cell_position.y % DEVICE_ORDERED_DITHER_POWER
	) * DEVICE_ORDERED_DITHER_POWER + cell_position.x % DEVICE_ORDERED_DITHER_POWER;

	float coeff = DEVICE_ORDERED_DITHER_COEFFS[coeff_index];

	for (size_t i = 0; i < font.size; i++)
	{
		const auto& glyph = font.data[i];

		glm::ivec3 bg = {};
		size_t bg_count = 0;

		glm::ivec3 fg = {};
		size_t fg_count = 0;

		glm::uvec2 glyph_offset = {};
		for (glyph_offset.x = 0; glyph_offset.x < DEVICE_GLYPH_SIZE.x; glyph_offset.x++)
		{
			for (glyph_offset.y = 0; glyph_offset.y < DEVICE_GLYPH_SIZE.y; glyph_offset.y++)
			{
				glm::u8vec3 color = DeviceGetImagePixel(
					reference_image,
					image_offset + glyph_offset
				);

				if (DeviceGetGlyphPixel(glyph.data, glyph_offset))
				{
					fg += color;
					fg_count++;
				}

				else
				{
					bg += color;
					bg_count++;
				}
			}
		}

		bg /= bg_count;
		fg /= fg_count;

		if (r >= .001f)
		{
			for (auto* color: {&bg, &fg})
			{
				auto normalized = glm::vec3(*color) / static_cast<float>(0xFF);
				auto dithered = glm::clamp(
					normalized + r * (coeff - .5f),
					0.f,
					1.f
				) * static_cast<float>(0xFF);

				*color = palette.data[DeviceFindClosestPaletteIndex(palette, dithered)];
			}
		}

		float err = 0;
		for (glyph_offset.x = 0; glyph_offset.x < DEVICE_GLYPH_SIZE.x; glyph_offset.x++)
		{
			for (glyph_offset.y = 0; glyph_offset.y < DEVICE_GLYPH_SIZE.y; glyph_offset.y++)
			{
				auto color = DeviceGetGlyphPixel(glyph.data, glyph_offset)
					? fg
					: bg;

				auto diff = glm::ivec3(
					DeviceGetImagePixel(
						reference_image,
						image_offset + glyph_offset
					)
				) - color;

				err += diff.r*diff.r + diff.g*diff.g + diff.b*diff.b;
			}
		}

		if (best_err < 0 || err < best_err)
		{
			best_err = err;
			best_glyph = i;
			best_background = bg;
			best_foreground = fg;
		}
	}

	auto& cell = approximated_image.data[index];
	cell.glyph_index = best_glyph;
	cell.background = best_background;
	cell.foreground = best_foreground;
}

//========================================

__global__ void RasterizationKernel(
	KernelArray<KernelGlyph> font,
	KernelArray2D<KernelCell> approximated_image,
	KernelArray2D<glm::u8vec4> rasterized_image
)
{
	int index = DeviceCheckThreadIndex(approximated_image.size);
	if (index < 0)
		return;

	const auto& cell = approximated_image.data[index];
	const auto& glyph = font.data[cell.glyph_index];

	auto cell_position = DeviceCalculateArrayPosition(
		index, 
		approximated_image.size
	);

	glm::uvec2 image_offset = cell_position * glm::uvec2(DEVICE_GLYPH_SIZE.x, DEVICE_GLYPH_SIZE.y);
	glm::uvec2 glyph_offset = {};
	for (glyph_offset.x = 0; glyph_offset.x < DEVICE_GLYPH_SIZE.x; glyph_offset.x++)
	{
		for (glyph_offset.y = 0; glyph_offset.y < DEVICE_GLYPH_SIZE.y; glyph_offset.y++)
		{
			auto color = DeviceGetGlyphPixel(glyph.data, glyph_offset)
				? cell.foreground
				: cell.background;

			DeviceGetImagePixel(
				rasterized_image,
				image_offset + glyph_offset
			) = glm::u8vec4(color, 0xFF);
		}
	}
}

//========================================

__global__ void FrameDecodeKernel(
	KernelArray2D<uint8_t> y_plane,
	KernelArray2D<uint8_t> cb_plane,
	KernelArray2D<uint8_t> cr_plane,
	KernelArray2D<glm::u8vec4> frame
)
{
	int index = DeviceCheckThreadIndex(frame.size);
	if (index < 0)
		return;

	auto position = DeviceCalculateArrayPosition(index, frame.size);

	auto yuv_color = glm::vec4(
		y_plane.data [ position.y      * y_plane.size.x  + position.x    ],
		cb_plane.data[(position.y / 2) * cb_plane.size.x + position.x / 2],
		cr_plane.data[(position.y / 2) * cr_plane.size.x + position.x / 2],
		0xFF
	);

	frame.data[index] = glm::clamp(
		yuv_color * DEVICE_BT601,
		static_cast<float>(0x00), 
		static_cast<float>(0xFF)
	);
}

//========================================

__global__ void ImageQuantizationKernel(
	KernelArray<glm::u8vec3> palette,
	KernelArray2D<glm::u8vec4> src,
	KernelArray2D<glm::u8vec4> dst
)
{
	int index = DeviceCheckThreadIndex(src.size);
	if (index < 0)
		return;

	dst.data[index] = glm::u8vec4(
		palette.data[
			DeviceFindClosestPaletteIndex(
				palette, 
				src.data[index]
			)
		], 
		0xFF
	);
}

__global__ void ImageOrderedDitherKernel(
	KernelArray<glm::u8vec3> palette,
	KernelArray2D<glm::u8vec4> src,
	KernelArray2D<glm::u8vec4> dst,
	float r
)
{			  
	int index = DeviceCheckThreadIndex(src.size);
	if (index < 0)
		return;

	auto& pixel = src.data[index];
	auto position = DeviceCalculateArrayPosition(index, src.size);

	size_t coeff_index = (
		position.y % DEVICE_ORDERED_DITHER_POWER
	) * DEVICE_ORDERED_DITHER_POWER + position.x % DEVICE_ORDERED_DITHER_POWER;

	float coeff = DEVICE_ORDERED_DITHER_COEFFS[coeff_index];

	auto normalized = glm::vec3(pixel) / static_cast<float>(0xFF);
	auto dithered = glm::clamp(
		normalized + r * (coeff - .5f),
		0.f,
		1.f
	) * static_cast<float>(0xFF);

	dst.data[index] = glm::u8vec4(
		palette.data[DeviceFindClosestPaletteIndex(palette, dithered)],
		0xFF
	);
}

__global__ void QuantizationKernel(
	KernelArray<glm::u8vec3> palette,
	KernelArray2D<KernelCell> image
)
{
	int index = DeviceCheckThreadIndex(image.size);
	if (index < 0)
		return;

	auto& cell = image.data[index];
	cell.background = palette.data[DeviceFindClosestPaletteIndex(palette, cell.background)];
	cell.foreground = palette.data[DeviceFindClosestPaletteIndex(palette, cell.foreground)];
}

__global__ void OrderedDitherKernel(
	KernelArray<glm::u8vec3> palette,
	KernelArray2D<KernelCell> image,
	float r
)
{			  
	int index = DeviceCheckThreadIndex(image.size);
	if (index < 0)
		return;

	auto& cell = image.data[index];
	auto position = DeviceCalculateArrayPosition(index, image.size);

	size_t coeff_index = (
		position.y % DEVICE_ORDERED_DITHER_POWER
	) * DEVICE_ORDERED_DITHER_POWER + position.x % DEVICE_ORDERED_DITHER_POWER;

	float coeff = DEVICE_ORDERED_DITHER_COEFFS[coeff_index];
	for (auto* color: {&cell.background, &cell.foreground})
	{
		auto normalized = glm::vec3(*color) / static_cast<float>(0xFF);
		auto dithered = glm::clamp(
			normalized + r * (coeff - .5f),
			0.f,
			1.f
		) * static_cast<float>(0xFF);

		*color = palette.data[DeviceFindClosestPaletteIndex(palette, dithered)];
	}
}

//========================================

__device__ bool DeviceGetGlyphPixel(
	const uint8_t (&data)[16], 
	const glm::uvec2& position
)
{
	size_t offset = position.y * 8 + position.x;
	return ((data[offset / 8] >> (7 - (offset % 8)))) & 1;
}

__device__ glm::u8vec4& DeviceGetImagePixel(
	const KernelArray2D<glm::u8vec4>& image,
	const glm::uvec2& position
)
{
	return image.data[position.y * image.size.x + position.x];
}

__device__ int DeviceCheckThreadIndex(
	const glm::uvec2& dim
)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= dim.x * dim.y)
		return -1;

	return index;
}

__device__ glm::uvec2 DeviceCalculateArrayPosition(
	size_t index, 
	const glm::uvec2& dim
)
{
	return glm::uvec2(
		index % dim.x,
		index / dim.x
	);
}

__device__ size_t DeviceFindClosestPaletteIndex(
	const KernelArray<glm::u8vec3>& palette,
	const glm::ivec3 color
)
{
	int best_err = -1;
	size_t best_index = 0;
	for (size_t i = 0; i < palette.size; i++)
	{
		auto diff = glm::ivec3(palette.data[i]) - color;
		int err = diff.r*diff.r + diff.g*diff.g + diff.b*diff.b;
		
		if (err < best_err || best_err < 0)
		{
			best_err = err;
			best_index = i;
		}
	}

	return best_index;
}

//========================================