#define PL_MPEG_IMPLEMENTATION

#include <vector>
#include <chrono>
#include <print>
#include <ranges>

#include <SFML/Graphics.hpp>

#include <pl_mpeg.h>

#include <font.hpp>
#include <palette.hpp>
#include <device_array.hpp>
#include <device_array_2d.hpp>
#include <kernels.hpp>

//========================================

constexpr float Scale = 2;

//========================================

class Main
{
public:
	Main() = default;

	void init();
	void run();

private:
	plm_t* m_plm = nullptr;
	
	std::vector<KernelGlyph> m_font;

	DeviceArray<KernelGlyph>   m_device_font;
	DeviceArray<glm::u8vec3>   m_device_palette;
	DeviceArray2D<KernelCell>  m_device_approximated_image;
	DeviceArray2D<glm::u8vec4> m_device_refernce_image;
	DeviceArray2D<glm::u8vec4> m_device_rasterized_image;
	DeviceArray2D<uint8_t>     m_device_y_plane;
	DeviceArray2D<uint8_t>     m_device_cb_plane;
	DeviceArray2D<uint8_t>     m_device_cr_plane;

	sf::RenderWindow m_window;

	std::chrono::duration<double> m_approximation_total_time = {};
	size_t m_approximation_count = 0;

	bool m_dither = false;
	bool m_quantize = false;
	float m_dither_r = .05f;

};

//========================================

std::string FormatDuration(std::chrono::duration<double> duration);

void QuantizeApproximatedImage(
	std::span<const KernelGlyph> font,
	DeviceArray2D<KernelCell>& device_image,
	bool dither
);

void QuantizeImage(
	DeviceArray2D<glm::u8vec4>& device_image,
	bool dither
);

void TransferImageToHost(
	sf::Image& image,
	DeviceArray2D<glm::u8vec4>& device_image
);

void TransferImageToDevice(
	DeviceArray2D<glm::u8vec4>& device_image,
	const sf::Image& image
);

//========================================

void Main::init()
{
	InitializeKernelConstants();

	// Video
	// m_plm = plm_create_with_filename("assets/apple.mpg");
	// plm_set_audio_enabled(m_plm, false);

	// Font
	auto view = std::views::all; // std::views::take(1000);

	/*
	auto view = std::views::filter(
		[](const auto& glyph) -> bool
		{
			return 0x2800 <= glyph.codepoint && glyph.codepoint <= 0x28FF;
		}
	) | std::ranges::to<std::vector>();	
	*/

	m_font = LoadFont("assets/font.hex") | view | std::ranges::to<std::vector>(); 
	m_device_font.create(m_font);

	// Palette
	m_device_palette.create(Palette);

	// Reference image
	sf::Image image;
	image.loadFromFile("assets/test.png");

	TransferImageToDevice(m_device_refernce_image, image);

	/*
	m_device_refernce_image.create(
		glm::uvec2(
			plm_get_width(m_plm), 
			plm_get_height(m_plm)
		)
	);
	*/

	// Approximated image
	m_device_approximated_image.create(
		m_device_refernce_image.getSize() / glm::uvec2(8, 16)
	);

	// Rasterized image
	m_device_rasterized_image.create(
		m_device_approximated_image.getSize() * glm::uvec2(8, 16)
	);

	// Window
	m_window.create(
		sf::VideoMode(
			Scale * m_device_rasterized_image.getSize().x,
			Scale * m_device_rasterized_image.getSize().y
		),
		"Anus view"
	);

	// m_window.setFramerateLimit(plm_get_framerate(m_plm));
	m_window.setVerticalSyncEnabled(true);
}

//========================================

void Main::run()
{		
	sf::Image image;
	sf::Texture texture;
	sf::Text text;

	sf::Sprite sprite;
	sprite.setScale(sf::Vector2f(Scale, Scale));

	while (m_window.isOpen())
	{
		sf::Event event;
		while (m_window.pollEvent(event))
		{
			switch (event.type)
			{
				case sf::Event::Closed:
					m_window.close();
					break;

				case sf::Event::KeyPressed:
					switch (event.key.code)
					{
						case sf::Keyboard::Escape:
							m_window.close();
							break;

						case sf::Keyboard::Space:
							m_dither ^= true;
							break;

						case sf::Keyboard::Q:
							m_quantize ^= true;
							break;

					}

					break;

				case sf::Event::MouseWheelMoved:
					m_dither_r = std::clamp<float>(m_dither_r + event.mouseWheel.delta * .001f, 0.f, 5.f);
					break;

			}
		}

		/*
		if (auto frame = plm_decode_video(m_plm); frame)
		{
			m_device_y_plane.create(
				glm::uvec2(
					frame->y.width, 
					frame->y.height
				), 
				frame->y.data
			);

			m_device_cb_plane.create(
				glm::uvec2(
					frame->cb.width, 
					frame->cb.height
				), 
				frame->cb.data
			);

			m_device_cr_plane.create(
				glm::uvec2(
					frame->cr.width, 
					frame->cr.height
				), 
				frame->cr.data
			);

			CudaCheck(
				LaunchFrameDecode(
					m_device_y_plane,
					m_device_cb_plane,
					m_device_cr_plane,
					m_device_refernce_image
				)
			);
		}

		else
			plm_seek(m_plm, 0, false);
		*/

		/*
		if (m_quantize)
		{
			if (m_dither)
			{
				CudaCheck(
					LaunchOrderedDither(
						m_device_palette,
						m_device_refernce_image
					)
				);
			}

			else
			{
				CudaCheck(
					LaunchQuantization(
						m_device_palette,
						m_device_refernce_image
					)
				);				
			}
		}
		*/

		/*
		auto start_time = std::chrono::system_clock::now();

		CudaCheck(
			LaunchApproximation(
				m_device_font,
				m_device_palette,
				m_device_refernce_image,
				m_device_approximated_image,
				m_dither_r * !m_quantize
			)
		);

		m_approximation_total_time += (std::chrono::system_clock::now() - start_time);
		m_approximation_count++;

		auto approximation_time_per_frame = m_approximation_total_time / m_approximation_count;
		auto approximation_time_per_cell = approximation_time_per_frame / (
			m_device_approximated_image.getSize().x * m_device_approximated_image.getSize().y
		);

		std::print(
			"\x1B[2K\r{}/frame, {}/cell average; output size is {}x{}; r = {:.4f}; dither {}; quantize {}",
			FormatDuration(approximation_time_per_frame),
			FormatDuration(approximation_time_per_cell),
			m_device_approximated_image.getSize().x,
			m_device_approximated_image.getSize().y,
			m_dither_r,
			m_dither,
			m_quantize
		);

		if (m_quantize)
		{
			if (m_dither)
			{
				CudaCheck(
					LaunchOrderedDither(
						m_device_palette,
						m_device_approximated_image,
						m_dither_r
					)
				);
			}

			else
			{
				CudaCheck(
					LaunchQuantization(
						m_device_palette,
						m_device_approximated_image
					)
				);				
			}
		}

		CudaCheck(
			LaunchRasterization(
				m_device_font,
				m_device_approximated_image,
				m_device_rasterized_image
			)
		);

		TransferImageToHost(image, m_device_rasterized_image);
		*/

		std::print(
			"\x1B[2K\routput size is {}x{}; r = {:.4f}; dither {}; quantize {}",
			m_device_approximated_image.getSize().x,
			m_device_approximated_image.getSize().y,
			m_dither_r,
			m_dither,
			m_quantize
		);

		if (m_dither)
		{
			CudaCheck(
				LaunchImageOrderedDither(
					m_device_palette, 
					m_device_refernce_image,
					m_device_rasterized_image,
					m_dither_r
				)
			);
		}

		else
		{
			CudaCheck(
				LaunchImageQuantization(
					m_device_palette, 
					m_device_refernce_image,
					m_device_rasterized_image
				)
			);			
		}

			TransferImageToHost(image, m_device_rasterized_image);

		texture.loadFromImage(image);
		sprite.setTexture(texture);

		m_window.clear();
		m_window.draw(sprite);
		m_window.display();
	}
}

//========================================

int main()
{
	try
	{
		Main instance;
		instance.init();
		instance.run();
	}

	catch (const CudaError& err)
	{
		std::println("Cuda error: {} ({})", err.getErrorMessage(), static_cast<unsigned>(err.getErrorCode()));
	}

	catch (const std::runtime_error& err)
	{
		std::println("Runtime error: {}", err.what());
	}
}

//========================================

std::string FormatDuration(std::chrono::duration<double> duration)
{
	constexpr const char* suffixes[] = {"s", "ms", "mcs", "ns"};
	auto seconds = duration.count();

	int power = -std::floor(std::log(seconds) / std::log(1000));
	auto suffix = suffixes[std::clamp<int>(power, 0, std::size(suffixes) - 1)];

	return std::format("{:.2f}{}", seconds * std::pow(1000, power), suffix);
}

void TransferImageToHost(
	sf::Image& image,
	DeviceArray2D<glm::u8vec4>& device_image
)
{
	auto data = std::make_unique<glm::u8vec4[]>(
		device_image.getSize().x * device_image.getSize().y
	);

	device_image.pull(data.get());
	image.create(
		device_image.getSize().x,
		device_image.getSize().y,
		reinterpret_cast<uint8_t*>(data.get())
	);
}

void TransferImageToDevice(
	DeviceArray2D<glm::u8vec4>& device_image,
	const sf::Image& image
)
{
	device_image.create(
		glm::uvec2(
			image.getSize().x, 
			image.getSize().y
		),
		reinterpret_cast<const glm::u8vec4*>(image.getPixelsPtr())
	);
}

//========================================