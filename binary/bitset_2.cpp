#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <nmmintrin.h>
#include <stdint.h>

using namespace std;

const uint8_t bit_num_in_uint8[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, \
	4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, \
	2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, \
	4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3,\
	3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, \
	4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, \
	6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};


uint16_t popcnt_cpu_64bit(const uint8_t* lhs, const uint8_t* rhs, const size_t n) {
	uint16_t res = 0;
	uint64_t v1 = 0;
	uint64_t v2 = 0;
	uint64_t v = 0;
	uint16_t i = 0;
	#define ITER { \
		v1 = *reinterpret_cast<const uint64_t*>(lhs + i); \
		v2 = *reinterpret_cast<const uint64_t*>(rhs + i); \
		v = v1 ^ v2; \
		res += _mm_popcnt_u64(v); \
		i += 8; \
	}

	while (i + 8 <= n) {
		ITER
	}

	#undef ITER
	
	if (i + 4 <= n) {
		res += _mm_popcnt_u32((*reinterpret_cast<const uint32_t*>(lhs + i)) ^ (*reinterpret_cast<const uint32_t*>(rhs + i)));
		i += 4;
	}
	
	while (i < n) {
		res += bit_num_in_uint8[lhs[i]^rhs[i]];
		i++;
	}
	
	return res;
}

	
void hamming_dist_mat(size_t code_len, uint8_t* code_Y, size_t size_Y, size_t block_sz_Y, uint8_t* code_X, size_t size_X, size_t block_sz_X, uint16_t* ret) {
	uint32_t code_per_char = (code_len%8 == 0 ? (code_len/8) : (code_len/8) + 1);
	uint32_t block_loop_Y = (size_Y%block_sz_Y == 0 ? size_Y/block_sz_Y : size_Y/block_sz_Y + 1);
	uint32_t block_loop_X = (size_X%block_sz_X == 0 ? size_X/block_sz_X : size_X/block_sz_X + 1);
	uint32_t begin_Y = 0, end_Y = 0;
	uint32_t begin_X = 0, end_X = 0;
	for (uint32_t bi = 0; bi < block_loop_Y-1; ++bi) {
		begin_Y = bi * block_sz_Y;
		end_Y = (bi+1) * block_sz_Y;
		for (uint32_t bj = 0; bj < block_loop_X-1; ++bj) {
			begin_X = bj * block_sz_X;
			end_X = (bj+1) * block_sz_X;
			for (uint32_t i = begin_Y; i < end_Y; ++i) {
				for (uint32_t j = begin_X; j < end_X; ++j) {
					ret[i*size_X+j] = popcnt_cpu_64bit(code_Y+i*code_per_char, code_X+j*code_per_char, code_per_char);
				}
			}
		}
		begin_X = end_X;
		end_X = size_X;
		for (uint32_t i = begin_Y; i < end_Y; ++i) {
			for (uint32_t j = begin_X; j < end_X; ++j) {
				ret[i*size_X+j] = popcnt_cpu_64bit(code_Y+i*code_per_char, code_X+j*code_per_char, code_per_char);
			}
		}
	}
	
	begin_Y = end_Y;
	end_Y = size_Y;
	for (uint32_t bj = 0; bj < block_loop_X-1; ++bj) {
		begin_X = bj * block_sz_X;
		end_X = (bj+1) * block_sz_X;
		for (uint32_t i = begin_Y; i < end_Y; ++i) {
			for (uint32_t j = begin_X; j < end_X; ++j) {
				ret[i*size_X+j] = popcnt_cpu_64bit(code_Y+i*code_per_char, code_X+j*code_per_char, code_per_char);
			}
		}
	}
	begin_X = end_X;
	end_X = size_X;
	for (uint32_t i = begin_Y; i < end_Y; ++i) {
		for (uint32_t j = begin_X; j < end_X; ++j) {
			ret[i*size_X+j] = popcnt_cpu_64bit(code_Y+i*code_per_char, code_X+j*code_per_char, code_per_char);
		}
	}	
}

int main(int argc, char** argv)
{
	int n = atoi(argv[1]);
	srand((int)time(NULL));
	int *a = new int[n];
	int *b = new int[n];
	for (int i = 0; i < n; ++i) {
		a[i] = rand() % 2;
		b[i] = rand() % 2;
	}
	int hamm1 = 0;
	for (int i = 0; i < n; ++i) {
		hamm1 += (a[i] != b[i]);
	}
	std::cout << hamm1 << endl;
	for (int i = 0; i < n; ++i) {
		std::cout << a[i];
	}
	std::cout << endl;
	for (int i = 0; i < n; ++i) {
		std::cout << b[i];
	}
	std::cout << endl;

	
	uint8_t *c = new uint8_t[n/8];
	uint8_t *d = new uint8_t[n/8];
	for (int i = 0; i < n/8; ++i) {
		uint8_t tmp = 0;
		for (int j = 0; j < 8; ++j) {
			tmp <<= 1;
			tmp += a[i*8+j];
		}
		c[i] = tmp;
	}
	for (int i = 0; i < n/8; ++i) {
		uint8_t tmp = 0;
		for (int j = 0; j < 8; ++j) {
			tmp <<= 1;
			tmp += b[i*8+j];
		}
		d[i] = tmp;
	}
	uint16_t *res = new uint16_t[1];
	hamming_dist_mat(n, d, 1, 1, c, 1, 1, res);
	std::cout << res[0] << endl;
	return 0;
}