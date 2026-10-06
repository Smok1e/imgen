#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include <glm/glm.hpp>

//========================================

struct KernelCell
{
	size_t glyph_index;
	glm::u8vec3 background;
	glm::u8vec3 foreground;
};

struct KernelGlyph
{
	uint32_t codepoint;
	uint8_t data[16];
};

template<typename T>
struct KernelArray
{
	T* data;
	size_t size;
};

template<typename T>
struct KernelArray2D
{
	T* data;
	glm::uvec2 size;
};

//========================================

void InitializeKernelConstants();

cudaError_t LaunchApproximation(
	const KernelArray<KernelGlyph>& font,
	const KernelArray<glm::u8vec3>& palette,
	const KernelArray2D<glm::u8vec4>& reference_image,
	const KernelArray2D<KernelCell>& approximated_image,
	float r
);

cudaError_t LaunchRasterization(
	const KernelArray<KernelGlyph>& font,
	const KernelArray2D<KernelCell>& approximated_image,
	const KernelArray2D<glm::u8vec4>& rasterized_image
);

cudaError_t LaunchFrameDecode(
	const KernelArray2D<uint8_t>& y_plane,
	const KernelArray2D<uint8_t>& cb_plane,
	const KernelArray2D<uint8_t>& cr_plane,
	const KernelArray2D<glm::u8vec4>& frame
);

cudaError_t LaunchImageQuantization(
	const KernelArray<glm::u8vec3>& palette,
	const KernelArray2D<glm::u8vec4>& src,
	const KernelArray2D<glm::u8vec4>& dst
);

cudaError_t LaunchImageOrderedDither(
	const KernelArray<glm::u8vec3>& palette,
	const KernelArray2D<glm::u8vec4>& src,
	const KernelArray2D<glm::u8vec4>& dst,
	float r
);

cudaError_t LaunchQuantization(
	const KernelArray<glm::u8vec3>& palette,
	const KernelArray2D<KernelCell>& image
);

cudaError_t LaunchOrderedDither(
	const KernelArray<glm::u8vec3>& palette,
	const KernelArray2D<KernelCell>& image,
	float r
);

//========================================