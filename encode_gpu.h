#pragma once

#include <cstdlib>
#include <ctime>

#define UNUSED(x) (void)(x)
#define TIME_INTERVAL(b, e) (static_cast<float>((e) - (b)) / CLOCKS_PER_SEC)
#define MALLOC (byte_t *)malloc
#define DIVIDE_CEIL(x, y) (((x) + (y) - 1) / (y))

#define MSVS 1

#if MSVS
#define EXIT \
	getchar(); \
	exit(EXIT_FAILURE)

#define END \
	getchar(); \
	exit(EXIT_SUCCESS)
#else
#define EXIT exit(EXIT_FAILURE)
#define END exit(EXIT_SUCCESS)
#endif

typedef unsigned char byte_t;
typedef unsigned long natural_t;

void encode(const byte_t *const input_bytes, const natural_t input_size,
			byte_t *&output_bytes, natural_t &output_size);
void encode_gpu(const byte_t *const input_bytes, const natural_t input_size,
				byte_t *&output_bytes, natural_t &output_size);
