#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <nmmintrin.h>
#include <stdint.h>

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
		v1 = *(const uint64_t*)(lhs + i); /**reinterpret_cast<const uint64_t*>(lhs + i);*/ \
		v2 = *(const uint64_t*)(rhs + i); /**reinterpret_cast<const uint64_t*>(rhs + i);*/ \
		v = v1 ^ v2; \
		res += _mm_popcnt_u64(v); \
		i += 8; \
	}

	while (i + 8 <= n) {
		ITER
	}

	#undef ITER
	
	if (i + 4 <= n) {
		res += _mm_popcnt_u32((*(const uint32_t*)(lhs + i)) ^ (*(const uint32_t*)(rhs + i)));
		i += 4;
	}
	
	while (i < n) {
		res += bit_num_in_uint8[lhs[i]^rhs[i]];
		i++;
	}
	
	return res;
}

void hamming_sort(size_t code_len, uint16_t* dist_mat, size_t size_Y, size_t size_X, bool row_wise, uint32_t* ret)
{
	size_t bin_size = code_len+1;
	// uint32_t* rank_bin = new uint32_t[size_Y*bin_size];
	uint32_t* rank_bin = (uint32_t*)malloc(sizeof(uint32_t)*size_Y*bin_size);
	memset(rank_bin, 0x0, sizeof(uint32_t)*size_Y*bin_size);
	size_t i, k, j;
	if (row_wise) {
		for (i = 0; i < size_Y; ++i) {
			uint32_t begin_B = i*bin_size;
			uint32_t begin_X = i*size_X;
			// counting
			for (j = 0; j < size_X; ++j) {
				rank_bin[begin_B+dist_mat[begin_X+j]]++;
			}
		}
	}
	else {
		for (j = 0; j < size_X; ++j) {
			// counting
			for (i = 0; i < size_Y; ++i) {
				rank_bin[i*bin_size+dist_mat[i*size_X+j]]++;
			}
		}
	}
	
	for (i = 0; i < size_Y; ++i) {
		uint32_t begin_B = i*bin_size;
		uint32_t begin_X = i*size_X;

		for (k = 1; k < bin_size; ++k) {
			rank_bin[begin_B+k] += rank_bin[begin_B+k-1];
		}
		// sorting
		for (j = 0; j < size_X; ++j) {
			rank_bin[begin_B+dist_mat[begin_X+j]]--;
			ret[begin_X+rank_bin[begin_B+dist_mat[begin_X+j]]] = j;
		}
	}
	
	free(rank_bin);
}

void hamming_rank(size_t code_len, uint8_t* code_Y, size_t size_Y, uint8_t* code_X, size_t size_X, uint32_t* ret)
{
	uint32_t code_per_char = (code_len%8 == 0 ? (code_len/8) : (code_len/8) + 1);
	size_t bin_size = code_len+1;
	uint32_t* rank_bin = (uint32_t*)malloc(sizeof(uint32_t)*size_Y*bin_size);
	memset(rank_bin, 0x0, sizeof(uint32_t)*size_Y*bin_size);
	size_t i, j, k;
	if (size_Y >= size_X) {
		// TODO: parallelization
		for (i = 0; i < size_Y; ++i) {
			uint32_t begin_B = i*bin_size;
			// counting
			for (j = 0; j < size_X; ++j) {
				rank_bin[begin_B+popcnt_cpu_64bit(code_Y+i*code_per_char, code_X+j*code_per_char, code_per_char)]++;
			}
		}
	}
	else {
		// TODO: parallelization
		for (j = 0; j < size_X; ++j) {
			for (i = 0; i < size_Y; ++i) {
				rank_bin[i*bin_size+popcnt_cpu_64bit(code_Y+i*code_per_char, code_X+j*code_per_char, code_per_char)]++;
			}
		}
	}
	
	// TODO: parallelization
	for (i = 0; i < size_Y; ++i) {
		uint32_t begin_B = i*bin_size;
		uint32_t begin_X = i*size_X;

		for (k = 1; k < bin_size; ++k) {
			rank_bin[begin_B+k] += rank_bin[begin_B+k-1];
		}
		// sorting
		for (j = 0; j < size_X; ++j) {
			rank_bin[begin_B+dist_mat[begin_X+j]]--;
			ret[begin_X+rank_bin[begin_B+dist_mat[begin_X+j]]] = j;
		}
	}
	
	free(rank_bin);
	// delete [] rank_bin;
}

int main(int argc, char** argv)
{
	int r = atoi(argv[1]);
	int m = atoi(argv[2]);
	int n = atoi(argv[3]);
	int block_sz = atoi(argv[4]);
	srand((int)time(NULL));
	
	// cout << "preparing..." << endl;
	
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
	
	// std::cout << "time 1: " << toc-tic << "/" << CLOCKS_PER_SEC << endl;
	// std::cout << "time 2: " << toc2-tic2 << "/" << CLOCKS_PER_SEC << endl;
	return 0;
}