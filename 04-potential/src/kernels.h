#ifndef CUDA_POTENTIAL_IMPLEMENTATION_KERNELS_H
#define CUDA_POTENTIAL_IMPLEMENTATION_KERNELS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <cstdint>

#include "data.hpp"

using coord_t = double;
using real_t = coord_t;
using index_t = std::uint32_t;
using length_t = std::uint32_t;
using point_t = Point<coord_t>;
using edge_t = Edge<index_t>;
using params_t = ModelParameters<real_t>;

struct neighbor_t
{
	index_t neighborIdx;
	length_t length;

	neighbor_t() = default;

	neighbor_t(index_t neighborIdx, length_t length)
		: neighborIdx(neighborIdx)
		, length(length)
	{}
};



/**
 * A stream exception that is base for all runtime errors.
 */
class CudaError : public std::exception
{
protected:
	std::string mMessage;	///< Internal buffer where the message is kept.
	cudaError_t mStatus;

public:
	CudaError(cudaError_t status = cudaSuccess) : std::exception(), mStatus(status) {}
	CudaError(const char *msg, cudaError_t status = cudaSuccess) : std::exception(), mMessage(msg), mStatus(status) {}
	CudaError(const std::string &msg, cudaError_t status = cudaSuccess) : std::exception(), mMessage(msg), mStatus(status) {}
	virtual ~CudaError() throw() {}

	virtual const char* what() const throw()
	{
		return mMessage.c_str();
	}

	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	CudaError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};


/**
 * CUDA error code check. This is internal function used by CUCH macro.
 */
inline void _cuda_check(cudaError_t status, int line, const char *srcFile, const char *errMsg = NULL)
{
	if (status != cudaSuccess) {
		throw (CudaError(status) << "CUDA Error (" << status << "): " << cudaGetErrorString(status) << "\n"
			<< "at " << srcFile << "[" << line << "]: " << errMsg);
	}
}

/**
 * Macro wrapper for CUDA calls checking.
 */
#define CUCH(status) _cuda_check(status, __LINE__, __FILE__, #status)



/*
 * Kernel wrapper declarations.
 */

void runComputeRepulsiveForces(
	const index_t pointsCount,
	const point_t *cuPoints,
	const params_t params,
	point_t *cuRepulsiveForces
);


void runComputeCompulsiveForces(
	const index_t pointsCount,
	const index_t neighborsCount,
	const point_t *cuPoints,
	const neighbor_t *cuNeighbors,
	const index_t *cuNeighborsStart,
	const params_t params,
	point_t *cuCompulsiveForces
);

void runComputeVelocitiesAndPositions(
	const index_t pointsCount,
	const params_t params,
	point_t *cuRepulsiveForces,
	point_t *cuCompulsiveForces,
	point_t *cuVelocities,
	point_t *cuPoints
);

#endif
