#ifndef NPRG042_MATRIX_HPP
#define NPRG042_MATRIX_HPP

#include <memory>
#include <vector>
#include <random>
#include <limits>
#include <stdexcept>
#include <cassert>


namespace traits {
	template<typename T>
	class MatrixOrganizationLinear
	{
	public:
		static std::size_t offset(std::size_t i, std::size_t j, std::size_t m, std::size_t n)
		{
			return i*m + j;
		}


		static std::size_t size(std::size_t m, std::size_t n)
		{
			return m*n;
		}


		static std::size_t alignment() { return 4096 / sizeof(T); }
	};
}



/**
 * A mathematical matrix holding m rows and n columns of T values.
 * \tparam T Numerical type that represents the items of the matrix.
 * \tparam ORGTRAIT Trait class that takes care of internal memory organization (items layout).
 */
template<typename T, class ORGTRAIT = traits::MatrixOrganizationLinear<T>>
class Matrix
{
private:
	std::size_t mM, mN, mSize;
	std::vector<T> mAllocated;
	T *mData;

public:
	typedef T value_t;

	Matrix() : mM(0), mN(0), mSize(0), mData(nullptr) {}
	Matrix(std::size_t m, std::size_t n) : mM(0), mN(0), mSize(0), mData(nullptr) { resize(m, n); }
	Matrix(const Matrix<T, ORGTRAIT> &matrix) : mM(0), mN(0), mSize(0), mData(nullptr)
	{
		resize(matrix.m(), matrix.n());
		copyFrom(matrix);
	}


	/**
	 * Change the size of the matrix. Items are not preserved, nor their values reinitialized.
	 */
	void resize(std::size_t m, std::size_t n)
	{
		mM = m;
		mN = n;
		if (m && n) {
			mSize = ORGTRAIT::size(m, n);
			mAllocated.resize(mSize + ORGTRAIT::alignment() - 1);
			void *tmp = &mAllocated[0];
			std::size_t size = mAllocated.size() * sizeof(T);
			mData = (T*)std::align(ORGTRAIT::alignment() * sizeof(T), mSize * sizeof(T), tmp, size);
		}
		else {
			mSize = 0;
			mAllocated.clear();
			mData = nullptr;
		}
	}


	std::size_t m() const { return mM; }
	std::size_t n() const { return mN; }


	T& at(std::size_t i, std::size_t j)
	{
		assert(i < mM && j < mN);
		return mData[ORGTRAIT::offset(i, j, mM, mN)];
	}

	const T& at(std::size_t i, std::size_t j) const
	{
		assert(i < mM && j < mN);
		return mData[ORGTRAIT::offset(i, j, mM, mN)];
	}


	/**
	 * Copy data from another matrix.
	 */
	void copyFrom(const Matrix<T, ORGTRAIT> &matrix)
	{
		if (m() != matrix.m() || n() != matrix.n())
			throw std::runtime_error("Matrix size mismatch. Unable to copy.");

		if (mSize) {
			for (std::size_t i = 0; i < mSize; ++i)
				mData[i] = matrix.mData[i];
		}
	}


	/**
	 * Read the data of the whole matrix (attempt to keep as much as possible in caches, TLB, ...)
	 */
	void touch()
	{
		std::size_t counter = 0;
		for (std::size_t i = 0; i < mSize; ++i)
			if (mData[i] > T(0)) ++counter;
	}
};



/**
 * Helper function that fills the matix with seq numbers from 1 line by line.
 */
template<typename T, class O>
void matrix_fill_seq(Matrix<T, O> &matrix)
{
	std::size_t counter = 0;
	for (std::size_t i = 0; i < matrix.m(); ++i)
		for (std::size_t j = 0; j < matrix.n(); ++j)
			matrix.at(i, j) = (T)++counter;
}


/**
 * Helper function that fills the matix with random numbers.
 */
template<typename T, class O>
void matrix_fill_random(Matrix<T, O> &matrix)
{
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<T> distribution(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

	std::size_t counter = 0;
	for (std::size_t i = 0; i < matrix.m(); ++i)
		for (std::size_t j = 0; j < matrix.n(); ++j)
			matrix.at(i, j) = distribution(generator);
}


// Partial specialization for floats
template<class O>
void matrix_fill_random(Matrix<float, O> &matrix)
{
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<float> distribution(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());

	std::size_t counter = 0;
	for (std::size_t i = 0; i < matrix.m(); ++i)
		for (std::size_t j = 0; j < matrix.n(); ++j)
			matrix.at(i, j) = distribution(generator);
}


// Partial specialization for floats
template<class O>
void matrix_fill_random(Matrix<double, O> &matrix)
{
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<double> distribution(std::numeric_limits<double>::min(), std::numeric_limits<double>::max());

	std::size_t counter = 0;
	for (std::size_t i = 0; i < matrix.m(); ++i)
		for (std::size_t j = 0; j < matrix.n(); ++j)
			matrix.at(i, j) = distribution(generator);
}


#endif
