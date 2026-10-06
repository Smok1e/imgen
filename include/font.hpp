#pragma once

#include <filesystem>
#include <iostream>
#include <cstdint>
#include <map>

#include <kernels.hpp>

//========================================

std::vector<KernelGlyph> LoadFont(std::istream& stream);
std::vector<KernelGlyph> LoadFont(const std::filesystem::path& path);

bool GetGlyphPixel(const uint8_t (&data)[16], const glm::uvec2& position);

//========================================