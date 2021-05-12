#ifndef CUDA_POTENTIAL_IMPLEMENTATION_HPP
#define CUDA_POTENTIAL_IMPLEMENTATION_HPP

#include "kernels.h"

#include <interface.hpp>
#include <data.hpp>

#include <cuda_runtime.h>


/*
 * Final implementation of the tested program.
 */
template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class ProgramPotential : public IProgramPotential<F, IDX_T, LEN_T>
{
public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t;	// Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;

private:



public:
	virtual void initialize(index_t points, const std::vector<edge_t>& edges, const std::vector<length_t> &lengths, index_t iterations)
	{
		/*
		 * Initialize your implementation.
		 * Allocate/initialize buffers, transfer initial data to GPU...
		 */
	}


	virtual void iteration(std::vector<point_t> &points)
	{
		/*
		 * Perform one iteration of the simulation and update positions of the points.
		 */
	}


	virtual void getVelocities(std::vector<point_t> &velocities)
	{
		/*
		 * Retrieve the velocities buffer from the GPU.
		 * This operation is for vreification only and it does not have to be efficient.
		 */
	}
};


#endif
