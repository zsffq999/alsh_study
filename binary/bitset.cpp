#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

const unsigned char bit_num_in_uint8[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, \
    4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, \
    2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, \
    4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3,\
    3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, \
    4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, \
    6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

std::uint32 popcnt_cpu_64bit(const uint8_t* data, const size_t n) {
	
}
	
void hamming_dist_mat(int code_len, unsigned char* code_Y, int size_Y, int block_sz_Y, unsigned char* code_X, int size_X, int block_sz_X, unsigned short* ret) {
	unsigned int code_per_char = (code_len%8 == 0 ? (code_len/8) : (code_len/8) + 1);
	unsigned int block_loop_Y = (size_Y%block_sz_Y == 0 ? size_Y/block_sz_Y : size_Y/block_sz_Y + 1);
	unsigned int block_loop_X = (size_X%block_sz_X == 0 ? size_X/block_sz_X : size_X/block_sz_X + 1);
	unsigned int begin_Y = 0, end_Y = 0;
	unsigned int begin_X = 0, end_X = 0;
	for (unsigned int bi = 0; bi < block_loop_Y-1; ++bi) {
		begin_Y = bi * block_sz_Y;
		end_Y = (bi+1) * block_sz_Y;
		for (unsigned int bj = 0; bj < block_loop_X-1; ++bj) {
			begin_X = bj * block_sz_X;
			end_X = (bj+1) * block_sz_X;
			for (unsigned int i = begin_Y; i < end_Y; ++i) {
				for (unsigned int j = begin_X; j < end_X; ++j) {
					unsigned short hamm = 0;
					for (unsigned int k = 0; k < code_per_char; ++k) {
						hamm += bit_num_in_uint8[code_Y[i*code_per_char+k] ^ code_X[j*code_per_char+k]];
					}
					ret[i*size_X+j] = hamm;
				}
			}
		}
		begin_X = end_X;
		end_X = size_X;
		for (unsigned int i = begin_Y; i < end_Y; ++i) {
			for (unsigned int j = begin_X; j < end_X; ++j) {
				unsigned short hamm = 0;
				for (unsigned int k = 0; k < code_per_char; ++k) {
					hamm += bit_num_in_uint8[code_Y[i*code_per_char+k] ^ code_X[j*code_per_char+k]];
				}
				ret[i*size_X+j] = hamm;
			}
		}
	}
	
	begin_Y = end_Y;
	end_Y = size_Y;
	for (unsigned int bj = 0; bj < block_loop_X-1; ++bj) {
		begin_X = bj * block_sz_X;
		end_X = (bj+1) * block_sz_X;
		for (unsigned int i = begin_Y; i < end_Y; ++i) {
			for (unsigned int j = begin_X; j < end_X; ++j) {
				unsigned short hamm = 0;
				for (unsigned int k = 0; k < code_per_char; ++k) {
					hamm += bit_num_in_uint8[code_Y[i*code_per_char+k] ^ code_X[j*code_per_char+k]];
				}
				ret[i*size_X+j] = hamm;
			}
		}
	}
	begin_X = end_X;
	end_X = size_X;
	for (unsigned int i = begin_Y; i < end_Y; ++i) {
		for (unsigned int j = begin_X; j < end_X; ++j) {
			unsigned short hamm = 0;
			for (unsigned int k = 0; k < code_per_char; ++k) {
				hamm += bit_num_in_uint8[code_Y[i*code_per_char+k] ^ code_X[j*code_per_char+k]];
			}
			ret[i*size_X+j] = hamm;
		}
	}	
}

int main()
{
	srand((int)time(NULL));
	int *a = new int[128];
	int *b = new int[128];
	for (int i = 0; i < 128; ++i) {
		a[i] = rand() % 2;
		b[i] = rand() % 2;
	}
	int hamm1 = 0;
	for (int i = 0; i < 128; ++i) {
		hamm1 += (a[i] != b[i]);
	}
	cout << hamm1 << endl;
	for (int i = 0; i < 128; ++i) {
		cout << a[i];
	}
	cout << endl;
	for (int i = 0; i < 128; ++i) {
		cout << b[i];
	}
	cout << endl;

	
	unsigned char *c = new unsigned char[16];
	unsigned char *d = new unsigned char[16];
	for (int i = 0; i < 16; ++i) {
		unsigned char tmp = 0;
		for (int j = 0; j < 8; ++j) {
			tmp <<= 1;
			tmp += a[i*8+j];
		}
		c[i] = tmp;
	}
	for (int i = 0; i < 16; ++i) {
		unsigned char tmp = 0;
		for (int j = 0; j < 8; ++j) {
			tmp <<= 1;
			tmp += b[i*8+j];
		}
		d[i] = tmp;
	}
	unsigned short *res = new unsigned short[1];
	hamming_dist_mat(128, d, 1, 1, c, 1, 1, res);
	cout << res[0] << endl;
	return 0;
}