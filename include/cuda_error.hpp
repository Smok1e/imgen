#pragma once

#include <cuda_runtime.h>

//========================================

class CudaError
{
public:
	CudaError(cudaError_t error_code);

	cudaError_t getErrorCode() const;
	const char* getErrorMessage() const;

private:
	cudaError_t m_error_code;

};

//========================================

void CudaCheck(cudaError_t result);

//========================================