#include "matrix.hpp"

#include <tbb/tick_count.h>

#include <random>
#include <vector>
#include <iostream>


/**
 * Naive implementation of serial transposition.
 */
template<typename T, class O>
void serial_transpose(Matrix<T, O> &matrix)
{
	for (std::size_t i = 0; i < matrix.m(); ++i)
		for (std::size_t j = i + 1; j < matrix.n(); ++j)
			std::swap(matrix.at(i, j), matrix.at(j, i));
}


/**
 * Verify that two matrices hold the same values.
 */
template<typename T, class O>
bool verify(const Matrix<T, O> &m1, const Matrix<T, O> &m2)
{
	if (m1.m() != m2.m() || m1.n() != m1.n()) {
		std::cerr << "The verified matrices differ in size (" << m1.m() << "x" << m1.n() << " vs " << m2.m() << "x" << m2.n() << ")." << std::endl;
		return false;
	}
	
	std::size_t errors = 0;
	for (std::size_t i = 0; i < m1.m(); ++i)
		for (std::size_t j = i + 1; j < m2.n(); ++j) {
			if (m1.at(i, j) != m2.at(i, j)) {
				if (errors == 0) std::cerr << "FAILED" << std::endl;
				if (++errors <= 20) {	// preview first 20 errors
					std::cerr << "Mismatch at [" << i << "," << j << "]: " << m1.at(i, j) << " != " << m2.at(i, j) << std::endl;
				}
			}
		}

	if (errors) {
		std::cerr << "Total " << errors << " errors found." << std::endl;
		return false;
	}
	return true;
}


/**
 * Parallel matrix transposition using TBB
 */
template<typename T, class O>
void parallel_transpose(Matrix<T, O> &matrix)
{

	/*
	 * TODO - Fake implementation, replace with real parallel implementation
	 */
	
	serial_transpose(matrix);
}



int main(int argc, char *argv[])
{
	std::size_t n = 16*1024;
	if ((n | (n - 1)) != (2 * n - 1))
		std::cerr << "Warining: n (" << n << ") is not a power of 2." << std::endl;

	// Prepare data ...
	Matrix<std::uint64_t> matrix(n, n), matrixTmp(n, n), matrixVerif(n, n);
	matrix_fill_seq(matrix);

	// Run and time serial version algorithms ...
	matrixTmp.copyFrom(matrix);
	matrixTmp.touch();
	tbb::tick_count tstart1 = tbb::tick_count::now();
	serial_transpose(matrixTmp);
	tbb::tick_count tend1 = tbb::tick_count::now();
	matrixVerif.copyFrom(matrixTmp);

	// Run and time parallel version ...
	matrixTmp.copyFrom(matrix);
	matrixTmp.touch();
	tbb::tick_count tstart2 = tbb::tick_count::now();
	parallel_transpose(matrixTmp);
	tbb::tick_count tend2 = tbb::tick_count::now();

	// Print times ...
	std::cout << "serial time: " << (tend1 - tstart1).seconds() << "s" << std::endl;
	std::cout << "tbb time :   " << (tend2 - tstart2).seconds() << "s" << std::endl;

	// Verify results ...
	std::cout << "Verifying ... ";
	std::cout.flush();
	if (verify(matrixTmp, matrixVerif)) {
		std::cout << "OK" << std::endl;
		return 0;
	}
	else
		return 1;
}
