#include <print>

#include <cuda_error.hpp>

//========================================

CudaError::CudaError(cudaError_t error_code):
	m_error_code(error_code)
{}

cudaError_t CudaError::getErrorCode() const
{
	return m_error_code;
}

const char* CudaError::getErrorMessage() const
{
	return cudaGetErrorString(m_error_code);
}

//========================================

void CudaCheck(cudaError_t result)
{
	if (result != cudaSuccess)
		throw CudaError(result);
}

//========================================