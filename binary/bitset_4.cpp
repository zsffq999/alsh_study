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

void hamming_dist_mat_1(size_t code_len, uint8_t* code_Y, size_t size_Y, size_t block_sz_Y, uint8_t* code_X, size_t size_X, size_t block_sz_X, uint16_t* ret) {
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

void hamming_dist_mat(size_t code_len, uint8_t* code_Y, size_t size_Y, uint8_t* code_X, size_t size_X, uint16_t* ret)
{
	uint32_t code_per_char = (code_len%8 == 0 ? (code_len/8) : (code_len/8) + 1);
	for (uint32_t i = 0; i < size_Y; ++i) {
		for (uint32_t j = 0; j < size_X; ++j) {
			ret[i*size_X+j] = popcnt_cpu_64bit(code_Y+i*code_per_char, code_X+j*code_per_char, code_per_char);
		}
	}
}

int main(int argc, char** argv)
{
	int r = atoi(argv[1]);
	int m = atoi(argv[2]);
	int n = atoi(argv[3]);
	int block_sz = atoi(argv[4]);
	srand((int)time(NULL));
	
	cout << "preparing..." << endl;
	
	int bit_r = (r % 8 == 0 ? r/8 : r/8+1);

	uint8_t *c = new uint8_t[m*bit_r];
	uint8_t *d = new uint8_t[n*bit_r];
	for (int t = 0; t < m; ++t) {
		for (int i = 0; i < bit_r; ++i) {
			uint8_t tmp = rand() % 256;
			c[t*bit_r+i] = tmp;
		}
	}
	
	for (int t = 0; t < n; ++t) {
		for (int i = 0; i < bit_r; ++i) {
			uint8_t tmp = rand() % 256;
			d[t*bit_r+i] = tmp;
		}
	}

	
	uint16_t *res = new uint16_t[m*n];
	uint16_t *res_2 = new uint16_t[m*n];
	int tic = clock();
	hamming_dist_mat_1(r, c, m, block_sz, d, n, block_sz, res);
	int toc = clock();
	
	int tic2 = clock();
	hamming_dist_mat(r, c, m, d, n, res_2);
	int toc2 = clock();
	
	std::cout << "time 1: " << toc-tic << "/" << CLOCKS_PER_SEC << endl;
	std::cout << "time 2: " << toc2-tic2 << "/" << CLOCKS_PER_SEC << endl;
	return 0;
}