#pragma once

#include <span>

#include <cuda_runtime.h>

#include <cuda_error.hpp>
#include <kernels.hpp>

//========================================

template<typename T>
class DeviceArray
{
public:
	DeviceArray() = default;
	DeviceArray(const DeviceArray& copy) = delete;
	~DeviceArray();

	size_t getSize() const;
	size_t getDataSize() const;

	T* getData() const;

	void create(size_t size);
	void create(size_t size, const T* data);
	void create(std::span<const T> span);

	void push(const T* data);
	void pull(T* data) const;

	operator KernelArray<T>() const;

private:
	T* m_data = nullptr;
	size_t m_size = 0;

	void release();

};

//========================================

template<typename T>
DeviceArray<T>::~DeviceArray()
{
	release();
}

//========================================

template<typename T>
size_t DeviceArray<T>::getSize() const
{
	return m_size;
}

template<typename T>
size_t DeviceArray<T>::getDataSize() const
{
	return m_size * sizeof(T);
}

template<typename T>
T* DeviceArray<T>::getData() const
{
	return m_data;
}

//========================================

template<typename T>
void DeviceArray<T>::create(size_t size)
{
	if (m_size == size)
		return;

	release();
	CudaCheck(cudaMalloc(&m_data, sizeof(T) * size));
	m_size = size;
}

template<typename T>
void DeviceArray<T>::create(size_t size, const T* data)
{
	create(size);
	push(data);
}

template<typename T>
void DeviceArray<T>::create(std::span<const T> span)
{
	create(span.size(), span.data());
}

template<typename T>
void DeviceArray<T>::release()
{
	if (m_data)
		cudaFree(m_data);
}

template<typename T>
void DeviceArray<T>::push(const T* data)
{
	CudaCheck(
		cudaMemcpy(
			m_data, 
			data, 
			m_size * sizeof(T), 
			cudaMemcpyHostToDevice
		)
	);
}

template<typename T>
void DeviceArray<T>::pull(T* data) const
{
	CudaCheck(
		cudaMemcpy(
			data,
			m_data,
			m_size * sizeof(T),
			cudaMemcpyDeviceToHost
		)
	);
}

//========================================

template<typename T>
DeviceArray<T>::operator KernelArray<T>() const
{
	KernelArray<T> array = {};
	array.data = m_data;
	array.size = m_size;

	return array;
}

//========================================