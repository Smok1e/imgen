#include <ranges>

#include <palette.hpp>

//========================================

size_t FindClosestPaletteIndex(const glm::u8vec3& color)
{
	glm::ivec3 color_ivec(color);

	int best_err = -1;
	size_t best_index = 0;
	for (const auto& [index, palette_color]: Palette | std::views::enumerate)
	{
		auto diff = glm::ivec3(palette_color) - color_ivec;
		int err = diff.r*diff.r + diff.g*diff.g + diff.b*diff.b;

		if (err < best_err || best_err < 0)
		{
			best_err = err;
			best_index = index;
		}
	}

	return best_index;
}

//========================================