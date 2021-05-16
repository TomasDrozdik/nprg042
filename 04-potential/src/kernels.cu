#include "kernels.h"
#include "math.h"

constexpr index_t blockSize = 256;

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

	//printf("compute_repulsive_forces for points %d\n", p1);

	const point_t point = cuPoints[p1];
	for (index_t p2 = p1 + 1; p2 < pointsCount; ++p2) {
		real_t dx = (real_t)point.x - (real_t)cuPoints[p2].x;
		real_t dy = (real_t)point.y - (real_t)cuPoints[p2].y;
		const real_t sqLen = max(dx*dx + dy*dy, (real_t)0.0001);
		const real_t fact = params.vertexRepulsion / (sqLen * (real_t)std::sqrt(sqLen));
		dx *= fact;
		dy *= fact;
		cuForces[p1].x += dx;
		cuForces[p1].y += dy;
		cuForces[p2].x -= dx;
		cuForces[p2].y -= dy;

		//if (p1 == 1 || p2 == 1) {
		//	printf("compute_repulsive_forces for points %d and %d \t distance[%f, %f] \t dforces[%f, %f]\n",
		//		p1, p2,
		//		(real_t)point.x - (real_t)cuPoints[p2].x, 
		//		(real_t)point.y - (real_t)cuPoints[p2].y, 
		//		dx, dy
		//	);
		//}
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

		//if (pointIdx == 0 || pointIdx == 31) {
		//	printf("compute_compulsive_force for %d %d \t dforces[%f, %f] \t cforces[%f, %f]\n",
		//		pointIdx, neighbor.neighborIdx,
		//		dx, dy,
		//		cuForces[pointIdx].x, cuForces[pointIdx].y
		//	);
		//}
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

	if (pointIdx == 13) {
		printf("computeVelocitiesAndPositions: point %d rf(%f, %f) cf(%f, %f) f(%f, %f)\n",
			pointIdx,
			repulsiveForce.x, repulsiveForce.y,
			compulsiveForce.x, compulsiveForce.y,
			resultForceX, resultForceY
		);
	}

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

void runComputeRepulsiveForces(
	const index_t pointsCount,
	const point_t *cuPoints,
	const params_t params,
	point_t *cuRepulsiveForces)
{
	const index_t blockCount = (pointsCount + blockSize - 1) / blockSize;
	computeRepulsiveForces<<<blockCount, blockSize>>>(
		pointsCount,
		cuPoints,
		params,
		cuRepulsiveForces
	);
}


void runComputeCompulsiveForces(
	const index_t pointsCount,
	const index_t neighborsCount,
	const point_t *cuPoints,
	const neighbor_t *cuNeighbors,
	const index_t *cuNeighborsStart,
	const params_t params,
	point_t *cuCompulsiveForces)
{
	const index_t blockCount = (pointsCount + blockSize - 1) / blockSize;
	computeCompulsiveForces<<<blockCount, blockSize>>>(
		pointsCount,
		neighborsCount,
		cuPoints,
		cuNeighbors,
		cuNeighborsStart,
		params,
		cuCompulsiveForces
	);
}

void runComputeVelocitiesAndPositions(
	const index_t pointsCount,
	const params_t params,
	point_t *cuRepulsiveForces,
	point_t *cuCompulsiveForces,
	point_t *cuVelocities,
	point_t *cuPoints)
{
	const index_t blockCount = (pointsCount + blockSize - 1) / blockSize;
	computeVelocitiesAndPositions<<<blockCount, blockSize>>>(
		pointsCount,
		params,
		cuRepulsiveForces,
		cuCompulsiveForces,
		cuVelocities,
		cuPoints
	);
}
