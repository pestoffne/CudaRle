#include <cstdio>

#include <cuda_runtime.h>

#include "encode_gpu.h"

#define RUN(x) (run(x, __FILE__, __LINE__))

__host__ void run(cudaError_t err, const char *file, int line)
{
	if (err) {
		fprintf(stderr, "Error in file %s at line %d:\n%s\n%s\n",
				file, line, cudaGetErrorName(err), cudaGetErrorString(err));
		EXIT;
	}
}

__device__ void merge(const byte_t *const left_bytes, const natural_t left_size,
					  const byte_t *const right_bytes, const natural_t right_size,
					  byte_t *&output_bytes, natural_t output_size)
{
	output_size = 100;
	output_bytes = MALLOC(output_size);
}

__global__ void kernel(const natural_t input_size)
{
	natural_t temp_size, index;
	byte_t *temp_bytes;

	index = blockDim.x * blockIdx.x + threadIdx.x;

	temp_size = input_size;
	temp_bytes = (byte_t *)(input_size * index);



	// merge see github c#
	//if (index % ? == ?) { ...
}

__host__ void encode_gpu(const byte_t *const input_bytes,
						 const natural_t input_size,
						 byte_t *&output_bytes, natural_t &output_size)
{
	byte_t *input_bytes_d = NULL;
	RUN(cudaMalloc((void **)&input_bytes_d, input_size));
	RUN(cudaMemcpy(input_bytes_d, input_bytes, input_size,
				   cudaMemcpyHostToDevice));

	byte_t *temp_d;
	RUN(cudaMalloc((void **)&temp_d, 32 * 2 * input_size));

	kernel<<<1, 32>>>(input_size);
	RUN(cudaGetLastError());

	// extract output_size
	output_size = 100;
	output_bytes = MALLOC(output_size);

	RUN(cudaMemcpy(output_bytes, temp_d, output_size, cudaMemcpyDeviceToHost));

	output_size = 0;
	output_bytes = nullptr;
}
