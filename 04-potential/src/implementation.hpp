#ifndef CUDA_POTENTIAL_IMPLEMENTATION_HPP
#define CUDA_POTENTIAL_IMPLEMENTATION_HPP

#include <algorithm>
#include <iostream>
#include <cassert>

#include <cuda_runtime.h>

#include "kernels.h"
#include <interface.hpp>
#include <data.hpp>


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
	index_t pointsCount{};
	index_t neighborsCount{};
	index_t iterationsCount{};
	index_t iterationCurrent{};

	point_t *cuPoints1{};
	point_t *cuPoints2{};
	neighbor_t *cuNeighbors{}; // effectively const
	index_t *cuNeighborsStart{}; // effectively const

	point_t *cuVelocities{};

public:
	virtual void initialize(index_t points, const std::vector<edge_t>& edges, const std::vector<length_t> &lengths, index_t iterations)
	{
		this->pointsCount = points;
		this->iterationsCount = iterations;
		this->iterationCurrent = 0;

		// Place egdes in a different structure - vector where each edge has its neighbors one after another.
		// Since the vector is continuous in order to know where neighbors of "next in line" start we keep
		// a separate vector that keeps information about how many points an egde is connected to".
		//
		// For example: for edges (0, 1); (0, 2); (1, 2):
		// Continuous neighbors vector (semicolons represent where neighbors of next point in line begin):
		// ;1, 2; 0, 2; 0, 2
		// 0    1     2
		// In order to store these points keep a separate array edgesStart: 0, 2, 5
		// This vector has size 2 * neighborsCount. However the size is equal since to edges vector contain edge_t type
		// Of course there is additional vector of neighbors start indices of size pointsCount * sizeof(index_t)
		
		// Count number of edges connected to each node
		std::vector<std::vector<neighbor_t>> neighborsList;
		neighborsList.resize(pointsCount);
		for (index_t i = 0; i < edges.size(); ++i) {
			neighborsList[edges[i].p1].emplace_back(edges[i].p2, lengths[i]);
			neighborsList[edges[i].p2].emplace_back(edges[i].p1, lengths[i]);
		}

		// Sort the neighbor list
		for (auto &neighbors : neighborsList) {
			std::sort(neighbors.begin(), neighbors.end(),
				[](const neighbor_t &n1, const neighbor_t &n2) -> bool {
					return n1.neighborIdx < n2.neighborIdx;
				}
			);
		}

		// These are the mentioned structures that we will copy to GPU
		std::vector<neighbor_t> neighborsListFlat(edges.size() * 2);
		std::vector<index_t> neighborsStart(pointsCount + 1);

		index_t neighborsStartIdx = 0;
		for (index_t i = 0; i < neighborsList.size(); ++i) {
			neighborsStart[i] = neighborsStartIdx;
			for (index_t neighborIdx = 0; neighborIdx < neighborsList[i].size(); ++neighborIdx) {
				neighborsListFlat[neighborsStartIdx + neighborIdx] = neighborsList[i][neighborIdx];
			}
			neighborsStartIdx += neighborsList[i].size();
		}
		this->neighborsCount = edges.size() * 2;
		assert(neighborsStartIdx == neighborsCount);

		neighborsStart[pointsCount] = neighborsCount; // neighborsStart[pointsCount - 1] will have and at neighborsStart[pointsCount]

		// Allocate memory on the device
		CUCH(cudaSetDevice(0));

		CUCH(cudaMalloc(&cuPoints1, pointsCount * sizeof(*cuPoints1)));
		CUCH(cudaMalloc(&cuPoints2, pointsCount * sizeof(*cuPoints2)));

		CUCH(cudaMalloc(&cuNeighbors, neighborsCount * sizeof(*cuNeighbors)));
		CUCH(cudaMemcpy(cuNeighbors, neighborsListFlat.data(), neighborsCount * sizeof(*cuNeighbors),
				cudaMemcpyHostToDevice));

		CUCH(cudaMalloc(&cuNeighborsStart, neighborsStart.size() * sizeof(*cuNeighborsStart)));
		CUCH(cudaMemcpy(cuNeighborsStart, neighborsStart.data(), neighborsStart.size() * sizeof(*cuNeighborsStart),
				cudaMemcpyHostToDevice));

		CUCH(cudaMalloc(&cuVelocities, pointsCount * sizeof(*cuVelocities)));
		CUCH(cudaMemset(cuVelocities, 0, pointsCount * sizeof(*cuVelocities)));
	}


	virtual void iteration(std::vector<point_t> &points)
	{
		/*
		 * Perform one iteration of the simulation and update positions of the points.
		 */
		assert(points.size() == pointsCount);
		assert(iterationCurrent < iterationsCount);

		if (iterationCurrent == 0) {
			CUCH(cudaMemcpy(cuPoints1, points.data(), pointsCount * sizeof(*cuPoints1), cudaMemcpyHostToDevice));
		}

		point_t *cuPointsOld{};
		point_t *cuPointsNew{};

		if (iterationCurrent % 2) {
			cuPointsOld = cuPoints2;
			cuPointsNew = cuPoints1;
		} else {
			cuPointsOld = cuPoints1;
			cuPointsNew = cuPoints2;
		}

		runComputePositions(
			this->mParams,
			cuPointsOld,
			pointsCount,
			cuNeighbors,
			cuNeighborsStart,
			neighborsCount,
			cuVelocities,
			cuPointsNew
		);
		CUCH(cudaGetLastError());

		CUCH(cudaMemcpy(points.data(), cuPointsNew, pointsCount * sizeof(*cuPointsNew), cudaMemcpyDeviceToHost));

		++iterationCurrent;
	}


	virtual void getVelocities(std::vector<point_t> &velocities)
	{
		velocities.resize(pointsCount);
		CUCH(cudaMemcpy(velocities.data(), cuVelocities, pointsCount * sizeof(*cuVelocities), cudaMemcpyDeviceToHost));
	}
};


#endif
