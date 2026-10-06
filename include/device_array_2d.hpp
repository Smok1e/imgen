#pragma once

#include <device_array.hpp>

#include <glm/glm.hpp>

//========================================

template<typename T>
class DeviceArray2D
{
public:
	DeviceArray2D() = default;
	DeviceArray2D(const DeviceArray2D& copy) = delete;

	glm::uvec2 getSize() const;
	size_t getDataSize() const;

	T* getData() const;

	void create(const glm::uvec2& size);
	void create(const glm::uvec2& size, const T* data);

	void push(const T* data);
	void pull(T* data) const;

	operator KernelArray2D<T>() const;

private:
	DeviceArray<T> m_array;
	glm::uvec2 m_size = {};

};

//========================================

template<typename T>
glm::uvec2 DeviceArray2D<T>::getSize() const
{
	return m_size;
}

template<typename T>
size_t DeviceArray2D<T>::getDataSize() const
{
	return m_array.getDataSize();
}

template<typename T>
T* DeviceArray2D<T>::getData() const
{
	return m_array.getData();
}

//========================================

template<typename T>
void DeviceArray2D<T>::create(const glm::uvec2& size)
{
	m_array.create(size.x * size.y);
	m_size = size;
}

template<typename T>
void DeviceArray2D<T>::create(const glm::uvec2& size, const T* data)
{
	m_array.create(size.x * size.y, data);
	m_size = size;
}

template<typename T>
void DeviceArray2D<T>::push(const T* data)
{
	m_array.push(data);
}

template<typename T>
void DeviceArray2D<T>::pull(T* data) const
{
	m_array.pull(data);
}

//========================================

template<typename T>
DeviceArray2D<T>::operator KernelArray2D<T>() const
{
	KernelArray2D<T> array = {};
	array.data = m_array.getData();
	array.size = m_size;

	return array;
}

//========================================