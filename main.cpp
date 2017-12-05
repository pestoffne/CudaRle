#include <algorithm>
#include <fstream>
#include <random>

#include "encode_gpu.h"

// Generates input text data using cpu with exponentional distribution equal   \
characters sequences lengths. If size is big then mean equal characters		   \
sequence length will be approximately equals to mean_sequence parameter.	   \
Result text consists of Latin letters with uniform distributaion.
byte_t *generate_bytes(const natural_t size, const double mean_sequence)
{
	natural_t i, j;
	byte_t b, c;
	byte_t *output_bytes;

	if (mean_sequence < 1.0) {
		printf("mean_sequence = %.3f is too smal\n", mean_sequence);
		EXIT;
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	const double lambda = 1.0 / (mean_sequence - 1.0);
	std::exponential_distribution<double> exp_dist(lambda);

	std::srand(std::time(0));

	output_bytes = MALLOC(size);

	i = 0;
	b = (byte_t)'\0';

	for (;;) {
		j = i;
		++(i += (natural_t)std::round((double)exp_dist(gen)));
		c = b;

		do {
			b = rand() % (byte_t)('z' - 'a' + 1) + (byte_t)'a';
		} while (b == c);

		if (i < size) {
			std::fill(output_bytes + j, output_bytes + i, b);
		} else {
			std::fill(output_bytes + j, output_bytes + size, b);
			break;
		}
	}

	return output_bytes;
}

// Simple RLE encode using cpu.
void encode_cpu(const byte_t *const input_bytes, const natural_t input_size,
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
		} else {
			*((j = (++k)++) + buffer) = *(input_bytes + i);
			*(buffer + k) = 1;
		}
	}

	output_bytes = MALLOC(++k);
	std::copy(buffer, buffer + k, output_bytes);
	free(buffer);
	output_size = k;
}

// Simple RLE decode using cpu
void decode_cpu(const byte_t *const input_bytes, const natural_t input_size,
				byte_t *&output_bytes, natural_t &output_size)
{
	natural_t i;
	byte_t *k, *t;

	output_size = 0;

	for (i = 1; i < input_size; i += 2) {
		output_size += *(input_bytes + i);
	}

	output_bytes = MALLOC(output_size);

	i = 0;
	k = output_bytes;

	while (input_size - i) {
		std::fill(k, t = *(input_bytes + i + 1) + k, *(input_bytes + i));
		k = t;
		i += 2;
	}
}

void print_as_text(const byte_t *const bytes, const natural_t size)
{
	natural_t i;

	printf("size = %lu\n", size);

	for (i = 0; i < size; i++) {
		printf("%c", *(bytes + i));
	}

	printf("\n");
}

void print_as_code(const byte_t *const bytes, const natural_t size)
{
	natural_t i;

	printf("size = %lu\n", size);

	for (i = 0; i < size; i += 2) {
		printf("%c%d ", *(i + bytes), *(i + 1 + bytes));
	}

	printf("\n");
}

int main(int argc, char **argv)
{
	UNUSED(argc);
	UNUSED(argv);

	const natural_t origin_size = 1024 * 1024;
	const byte_t *origin_bytes = generate_bytes(origin_size, 100.0);

	natural_t cpu_size;
	byte_t *cpu_bytes;
	encode_cpu(origin_bytes, origin_size, cpu_bytes, cpu_size);

#if 1
	// check mean, max sequence in generated origin
	{
		natural_t sum, count, max, index, value;
		sum = count = max = 0;

		for (index = 1; index < cpu_size; index += 2) {
			sum += *(cpu_bytes + index);

			if (index == 1 || *(cpu_bytes + index - 1) - *(cpu_bytes + index -
				3)) {
				count++;
				value = *(cpu_bytes + index);
			} else {
				value += *(cpu_bytes + index);
			}

			if (value > max) {
				max = value;
			}
		}

		printf("mean sequence = %.3f\n", (double)sum / (double)count);
		printf("max sequence = %lu\n", max);
	}
#endif

	natural_t cpu_check_size;
	byte_t *cpu_check_bytes;
	decode_cpu(cpu_bytes, cpu_size, cpu_check_bytes, cpu_check_size);

	free(cpu_bytes);

	if (std::equal(origin_bytes, origin_bytes + origin_size, cpu_check_bytes)
		&& origin_size == cpu_check_size) {
		printf("Cpu test passed.\n");
	} else {
		fprintf(stderr, "Cpu test failed!\n");
		EXIT;
	}

	free(cpu_check_bytes);

	natural_t gpu_size;
	byte_t *gpu_bytes;

	encode_gpu(origin_bytes, origin_size, gpu_bytes, gpu_size);

	natural_t gpu_check_size;
	byte_t *gpu_check_bytes;

	decode_cpu(gpu_bytes, gpu_size, gpu_check_bytes, gpu_check_size);

	free(gpu_bytes);

	if (std::equal(origin_bytes, origin_bytes + origin_size, gpu_check_bytes)
		&& origin_size == gpu_check_size) {
		printf("Gpu test passed.\n");
	} else {
		fprintf(stderr, "Gpu test failed!\n");
		EXIT;
	}

	free(gpu_check_bytes);

	END;
}
