#include <algorithm>
#include <cstdio>

#include <cuda_runtime.h>

#include "encode_gpu.h"

const int threadsCount = 32;

#define RUN(x) (run(x, __FILE__, __LINE__))

__host__ void run(cudaError_t err, const char *file, int line)
{
	if (err) {
		fprintf(stderr, "Error in file %s at line %d:\n%s\n%s\n",
				file, line, cudaGetErrorName(err), cudaGetErrorString(err));
		EXIT;
	}
}

__device__ void encode(const byte_t *const input_bytes, const natural_t input_size,
					   byte_t *&output_bytes, natural_t &output_size)
{
	natural_t i, j, k;
	byte_t *buffer;

	*(buffer = MALLOC(input_size * 2)) = *input_bytes;
	i = j = 0;
	k = *(1 + buffer) = 1;

	while (i++, input_size - i) {
		if (*(j + buffer) == *(i + input_bytes) && *(k + buffer) - 255) {
			++*(k + buffer);
		}
		else {
			*((j = (++k)++) + buffer) = *(input_bytes + i);
			*(buffer + k) = 1;
		}
	}

	output_bytes = MALLOC(++k);
	std::copy(buffer, buffer + k, output_bytes);
	free(buffer);
	output_size = k;
}

__device__ void merge(const byte_t *const left_bytes, const natural_t left_size,
					  const byte_t *const right_bytes, const natural_t right_size,
					  byte_t *&output_bytes, natural_t output_size)
{
	output_size = 100;
	output_bytes = MALLOC(output_size);
}

__global__ void kernel(const byte_t *const input_bytes, const natural_t input_size,
					   byte_t *&output_bytes, natural_t &output_size)
{
	natural_t index, input_part_size, part_end_index;
	byte_t *input_part_bytes;

	index = blockDim.x * blockIdx.x + threadIdx.x;
	input_part_bytes = (byte_t *)(input_bytes + DIVIDE_CEIL(input_size, threadsCount) * index);
	part_end_index = natural_t(input_bytes + DIVIDE_CEIL(input_size, threadsCount) * (index + 1));
	input_part_size = natural_t(input_bytes + std::min(input_size, part_end_index));

	//encode(input_part_bytes, input_part_size, )

	// merge see github c#
	//if (index % ? == ?) { ...

	if (index) {
		return;
	}

	output_size = temp_size;
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
	RUN(cudaMalloc((void **)&temp_d, threadsCount * 2 * input_size));

	natural_t *output_size_d;
	RUN(cudaMalloc((void**)&output_size_d, sizeof(natural_t)));

	kernel<<<1, threadsCount>>>(input_bytes_d, input_size, temp_d, *output_size_d);
	RUN(cudaGetLastError());

	RUN(cudaMemcpy(&output_size, output_size_d, sizeof(natural_t), cudaMemcpyDeviceToHost));
	fprintf(stderr, "output_size = %lu.\n", output_size);
	output_bytes = MALLOC(output_size);
	RUN(cudaMemcpy(output_bytes, temp_d, output_size, cudaMemcpyDeviceToHost));
}
