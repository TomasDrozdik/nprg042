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

struct MatrixNode
{
	length_t edgeLength{0};  // 0 means no edge
	point_t dforce{0, 0}; // delta force
	//point_t dRepulsiveForce{0, 0}; // delta force
	//point_t dCompulsiveForce{0, 0}; // delta force
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

void runComputeForces(
	const index_t pointsCount,
	const params_t params,
	const point_t *cuPoints,
	MatrixNode **cuGraph
);

void runComputePositions(
	const index_t pointsCount,
	const params_t params,
	MatrixNode **cuGraph,
	point_t *cuVelocities,
	point_t *cuPoints
);

#endif
