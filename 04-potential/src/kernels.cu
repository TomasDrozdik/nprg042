#include "kernels.h"
#include "math.h"

constexpr index_t blockSize = 256;

/*
__global__ void computeRepulsiveForces(
	const index_t pointsCount,
	const point_t *cuPoints,
	const params_t params,
	point_t *cuForces)
{
	const index_t p1 = blockIdx.x * blockDim.x + threadIdx.x;
	if (p1 >= pointsCount) {
		return;
	}

	const point_t point = cuPoints[p1];
	for (index_t p2 = 0; p2 < pointsCount; ++p2) {
		real_t dx = (real_t)point.x - (real_t)cuPoints[p2].x;
		real_t dy = (real_t)point.y - (real_t)cuPoints[p2].y;
		const real_t sqLen = max(dx*dx + dy*dy, (real_t)0.0001);
		const real_t fact = params.vertexRepulsion / (sqLen * (real_t)std::sqrt(sqLen));
		dx *= fact;
		dy *= fact;
		cuForces[p1].x += dx;
		cuForces[p1].y += dy;
	}
}

__global__ void computeCompulsiveForces(
	const index_t pointsCount,
	const index_t neighborsCount,
	const point_t *cuPoints,
	const neighbor_t *cuNeighbors,
	const index_t *cuNeighborsStart,
	const params_t params,
	point_t *cuForces)
{
	const index_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pointIdx >= pointsCount) {
		return;
	}
	
	const point_t point = cuPoints[pointIdx];

	const index_t neighborsIdxBegin = cuNeighborsStart[pointIdx];
	const index_t neighborsIdxEnd = (pointIdx == pointsCount - 1) ? neighborsCount : cuNeighborsStart[pointIdx + 1];

	for (index_t neighborsIdx = neighborsIdxBegin; neighborsIdx  < neighborsIdxEnd; ++neighborsIdx) {
		const neighbor_t neighbor = cuNeighbors[neighborsIdx];
		const point_t neighborPoint = cuPoints[neighbor.neighborIdx];
		const length_t length = cuNeighbors[neighborsIdx].length;

		real_t dx = point.x - neighborPoint.x;
		real_t dy = point.y - neighborPoint.y;
		const real_t sqLen = dx*dx + dy*dy;
		const real_t fact = (real_t)std::sqrt(sqLen) * params.edgeCompulsion / (real_t)(length);
		dx *= fact;
		dy *= fact;
		cuForces[pointIdx].x -= dx;
		cuForces[pointIdx].y -= dy;
	}
}

__global__ void computeVelocitiesAndPositions(
	const index_t pointsCount,
	const params_t params,
	point_t *cuRepulsiveForces,
	point_t *cuCompulsiveForces,
	point_t *cuVelocities,
	point_t *cuPoints)
{
	const index_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pointIdx >= pointsCount) {
		return;
	}

	// Compute result force
	point_t &repulsiveForce = cuRepulsiveForces[pointIdx];
	point_t &compulsiveForce = cuCompulsiveForces[pointIdx];
	coord_t resultForceX = compulsiveForce.x + repulsiveForce.x;
	coord_t resultForceY = compulsiveForce.y + repulsiveForce.y;

	// Compute velocities (requiered by getVelocities)
	real_t fact = params.timeQuantum / params.vertexMass;	// v = Ft/m  => t/m is mul factor for F.
	cuVelocities[pointIdx].x = (cuVelocities[pointIdx].x + resultForceX * fact) * params.slowdown;
	cuVelocities[pointIdx].y = (cuVelocities[pointIdx].y + resultForceY * fact) * params.slowdown;

	// Copute new positions
	cuPoints[pointIdx].x += cuVelocities[pointIdx].x * params.timeQuantum;
	cuPoints[pointIdx].y += cuVelocities[pointIdx].y * params.timeQuantum;

	// Reset Forces for given point
	repulsiveForce.x = 0;
	repulsiveForce.y = 0;
	compulsiveForce.x = 0;
	compulsiveForce.y = 0;
}
*/

__device__
void computeForcesInRow(
	const index_t rowIdx,
	const params_t params,
	const point_t *cuPoints,
	MatrixNode **cuGraph)
{
	const point_t point = cuPoints[rowIdx];
	for (index_t colIdx = 0; colIdx < rowIdx; ++colIdx) {
		const real_t dx =  cuPoints[colIdx].x - point.x;
		const real_t dy =  cuPoints[colIdx].y - point.y;
		const real_t sqLen = max(dx * dx + dy * dy, (real_t)0.0001);

		const real_t repulsiveFact = params.vertexRepulsion / (sqLen * (real_t)std::sqrt(sqLen));

		MatrixNode &node = cuGraph[rowIdx][colIdx];
		node.dforce.x = dx * repulsiveFact;
		node.dforce.y = dy * repulsiveFact;

		if (node.edgeLength) { // if has edge
			const real_t compulsiveFact = (real_t)std::sqrt(sqLen) * params.edgeCompulsion / (real_t)node.edgeLength;
			node.dforce.x -= dx * compulsiveFact;
			node.dforce.y -= dy * compulsiveFact;
		}
	}
}

__global__
void computeForces(
	const index_t pointsCount,
	const params_t params,
	const point_t *cuPoints,
	MatrixNode **cuGraph)
{
	const index_t rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowIdx >= pointsCount) {
		return;
	}

	computeForcesInRow(rowIdx, params, cuPoints, cuGraph);
}

void runComputeForces(
	const index_t pointsCount,
	const params_t params,
	const point_t *cuPoints,
	MatrixNode **cuGraph)
{
	const index_t blockCount = (pointsCount + blockSize - 1) / blockSize;
	computeForces<<<blockCount, blockSize>>>(
		pointsCount,
		params,
		cuPoints,
		cuGraph
	);
}

__global__
void computePositions(
	const index_t pointsCount,
	const params_t params,
	MatrixNode **cuGraph,
	point_t *cuVelocities,
	point_t *cuPoints)
{
	const index_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pointIdx >= pointsCount) {
		return;
	}

	const bool print = false; // pointIdx < 5;

	// Accumulate resulting force
	coord_t forceX = 0;
	coord_t forceY = 0;

	// First go in rows (cache friendly)
	const index_t rowSize = pointIdx;
	for (index_t colIdx = 0; colIdx < rowSize; ++colIdx) {
		forceX -= cuGraph[pointIdx][colIdx].dforce.x;
		forceY -= cuGraph[pointIdx][colIdx].dforce.y;

		//if (print)
		//printf("[%d, %d] dforce(%f, %f)\n",
		//	pointIdx, colIdx, cuGraph[pointIdx][colIdx].dforce.x, cuGraph[pointIdx][colIdx].dforce.y);
	}

	// Then go in columns (cache un-friendly)
	for (index_t rowIdx = pointIdx + 1; rowIdx < pointsCount; ++rowIdx) {
		forceX += cuGraph[rowIdx][pointIdx].dforce.x;
		forceY += cuGraph[rowIdx][pointIdx].dforce.y;

		//if (print)
		//printf("[%d, %d] dforce(%f, %f)\n",
		//	rowIdx, pointIdx, cuGraph[rowIdx][pointIdx].dforce.x, cuGraph[rowIdx][pointIdx].dforce.y);
	}

	if (print)
	printf("cuda[%d] force(%f, %f)\n", pointIdx, forceX, forceY);

	// Update velocities
	const real_t fact = params.timeQuantum / params.vertexMass;	// v = Ft/m  => t/m is mul factor for F.
	cuVelocities[pointIdx].x = (cuVelocities[pointIdx].x + forceX * fact) * params.slowdown;
	cuVelocities[pointIdx].y = (cuVelocities[pointIdx].y + forceY * fact) * params.slowdown;

	// Compute new positions
	cuPoints[pointIdx].x += cuVelocities[pointIdx].x * params.timeQuantum;
	cuPoints[pointIdx].y += cuVelocities[pointIdx].y * params.timeQuantum;
}

void runComputePositions(
	const index_t pointsCount,
	const params_t params,
	MatrixNode **cuGraph,
	point_t *cuVelocities,
	point_t *cuPoints)
{
	const index_t blockCount = (pointsCount + blockSize - 1) / blockSize;
	computePositions<<<blockCount, blockSize>>>(
		pointsCount,
		params,
		cuGraph,
		cuVelocities,
		cuPoints
	);
}
