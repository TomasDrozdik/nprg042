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
	index_t iterationsCount{};
	index_t iterationCurrent{};

	point_t *cuPoints{};
	MatrixNode **cuGraph{};
	point_t *cuVelocities{};

public:
	virtual void initialize(index_t points, const std::vector<edge_t>& edges, const std::vector<length_t> &lengths, index_t iterations)
	{
		this->pointsCount = points;
		this->iterationsCount = iterations;
		this->iterationCurrent = 0;

		std::size_t size = 0;

		// Prepare the graph matrix
		std::vector<std::vector<MatrixNode>> graph;
		graph.resize(pointsCount);
		for (index_t rowIdx = 1; rowIdx < pointsCount; ++rowIdx) {
			index_t rowSize = rowIdx;
			graph[rowIdx].resize(rowSize);
		}

		for (std::size_t edgeIdx = 0; edgeIdx < edges.size(); ++edgeIdx) {
			auto &edge = edges[edgeIdx];
			auto p1 = std::max(edge.p1, edge.p2);
			auto p2 = std::min(edge.p1, edge.p2);
			graph[p1][p2].edgeLength = lengths[edgeIdx];
		}

		CUCH(cudaSetDevice(0));

		CUCH(cudaMalloc(&cuPoints, pointsCount * sizeof(*cuPoints)));
		printf("cuPoints: %lu\n", pointsCount * sizeof(*cuPoints));
		size += pointsCount * sizeof(*cuPoints);

		// Move graph to cuGraph
		// First create a host vector of cu pointers
		std::vector<MatrixNode *> graphRowCuPointers(pointsCount);
		for (index_t rowIdx = 0; rowIdx < pointsCount; ++rowIdx) {
			index_t rowSize = rowIdx;
			CUCH(cudaMalloc(&graphRowCuPointers[rowIdx], rowSize * sizeof(MatrixNode)));
			CUCH(cudaMemcpy(graphRowCuPointers[rowIdx], graph[rowIdx].data(), rowSize * sizeof(MatrixNode),
					cudaMemcpyHostToDevice));
			printf("cuGraph[%u]: %lu\n", rowIdx, rowSize * sizeof(MatrixNode));
			size += rowSize * sizeof(MatrixNode);
		}

		// Then allocate a vector that holds these pointers and copy preallocated cuda pointers to it
		CUCH(cudaMalloc(&cuGraph, pointsCount * sizeof(*cuGraph)));
		CUCH(cudaMemcpy(cuGraph, graphRowCuPointers.data(), pointsCount * sizeof(*cuGraph),
				cudaMemcpyHostToDevice));

		printf("cuGraph: %lu\n", pointsCount * sizeof(*cuGraph));
		size += pointsCount * sizeof(*cuGraph);

		// Additionaly allocate temporary help fields for velocity
		CUCH(cudaMalloc(&cuVelocities, pointsCount * sizeof(*cuVelocities)));
		CUCH(cudaMemset(cuVelocities, (real_t)0.0, pointsCount * sizeof(*cuVelocities)));

		printf("cuVelocities: %lu\n", pointsCount * sizeof(*cuVelocities));
		size += pointsCount * sizeof(*cuVelocities);
		printf("TOTAL SIZE: %lu B\n", size);
	}


	virtual void iteration(std::vector<point_t> &points)
	{
		/*
		 * Perform one iteration of the simulation and update positions of the points.
		 */
		assert(points.size() == pointsCount);
		assert(iterationCurrent < iterationsCount);

		if (iterationCurrent == 0) {
			CUCH(cudaMemcpy(cuPoints, points.data(), pointsCount * sizeof(*cuPoints), cudaMemcpyHostToDevice));
		}
		++iterationCurrent;


		runComputeForces(
			pointsCount,
			this->mParams,
			cuPoints,
			cuGraph
		);
		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize());

		runComputePositions(
			pointsCount,
			this->mParams,
			cuGraph,
			cuVelocities,
			cuPoints
		);
		CUCH(cudaGetLastError());

		CUCH(cudaMemcpy(points.data(), cuPoints, pointsCount * sizeof(*cuPoints), cudaMemcpyDeviceToHost));
	}


	virtual void getVelocities(std::vector<point_t> &velocities)
	{
		velocities.resize(pointsCount);
		CUCH(cudaMemcpy(velocities.data(), cuVelocities, pointsCount * sizeof(*cuVelocities), cudaMemcpyDeviceToHost));
	}
};


#endif
