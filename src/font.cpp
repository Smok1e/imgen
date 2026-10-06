#include <fstream>
#include <ranges>

#include <font.hpp>

//========================================

constexpr uint8_t DecodeHexDigit(char digit)
{
	return digit > '9'
		? 10 + digit - 'A'
		: digit - '0';
}

constexpr uint8_t DecodeHexByte(const char* digits)
{
	return (DecodeHexDigit(digits[0]) << 4) | DecodeHexDigit(digits[1]);
}

//========================================

std::vector<KernelGlyph> LoadFont(std::istream& stream)
{
	std::vector<KernelGlyph> font_data;

	char line[128] = "";
	while (stream.peek() != EOF && stream.getline(line, sizeof(line)))
	{
		const char* delimiter = std::strchr(line, ':');
		if (!delimiter)
			throw std::runtime_error("delimiter not found");

		size_t delimiter_offset = delimiter - line;
		size_t line_length = std::strlen(line);

		size_t data_size = (line_length - delimiter_offset - 1) / 2;
		if (data_size != 16)
			continue;

		auto& glyph = font_data.emplace_back();
		glyph.codepoint = std::strtoul(line, nullptr, 16);

		for (size_t i = 0, offset = delimiter_offset + 1; i < data_size; i++, offset += 2)
			glyph.data[i] = DecodeHexByte(line + offset);
	}

	std::ranges::sort(
		font_data,
		[](const auto& a, const auto& b) -> bool
		{
			return memcmp(a.data, b.data, sizeof(a.data)) > 0;
		}
	);

	auto last = std::unique(
		font_data.begin(),
		font_data.end(),
		[](const auto& a, const auto& b) -> bool
		{
			return memcmp(a.data, b.data, sizeof(a.data)) == 0;
		}
	);

	font_data.erase(last, font_data.end());
	return font_data;
}

std::vector<KernelGlyph> LoadFont(const std::filesystem::path& path)
{
	std::ifstream stream;
	stream.exceptions(stream.exceptions() | std::ios::failbit);
	stream.open(path);

	return LoadFont(stream);
}

//========================================

bool GetGlyphPixel(const uint8_t (&data)[16], const glm::uvec2& position)
{
	size_t offset = position.y * 8 + position.x;
	return ((data[offset / 8] >> (7 - (offset % 8)))) & 1;	
}

//========================================