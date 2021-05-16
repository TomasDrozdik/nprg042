#include "assert.h"
#include "kernels.h"
#include "math.h"

constexpr index_t blockSize = 64;

__global__
void computePositions(
	const params_t params,
	const point_t *cuPointsOld,
	const index_t pointsCount,
	const neighbor_t *cuNeighbors,
	const index_t *cuNeighborsStart,
	const index_t neighborsCount,
	point_t *cuVelocities,
	point_t *cuPointsNew)
{
	const index_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pointIdx >= pointsCount) {
		return;
	}

	const point_t point = cuPointsOld[pointIdx];
	real_t forceX = (real_t)0.0;
	real_t forceY = (real_t)0.0;
	
	const index_t neighborsIdxBegin = cuNeighborsStart[pointIdx];
	const index_t neighborsIdxEnd = cuNeighborsStart[pointIdx + 1];
	index_t neighborsIdx = neighborsIdxBegin;
	neighbor_t nextNeighbor = cuNeighbors[neighborsIdx];

	for (index_t p2 = 0; p2 < pointsCount; ++p2) {
		if (pointIdx == p2) {
			//assert(p2 != nextNeighbor.neighborIdx);
			continue;
		}

		const real_t dx = (real_t)point.x - (real_t)cuPointsOld[p2].x;
		const real_t dy = (real_t)point.y - (real_t)cuPointsOld[p2].y;
		const real_t sqLen = max(dx*dx + dy*dy, (real_t)0.0001);
		const real_t repulsiveFact = params.vertexRepulsion / (sqLen * (real_t)std::sqrt(sqLen));
		forceX += dx * repulsiveFact;
		forceY += dy * repulsiveFact;

		if (neighborsIdx < neighborsIdxEnd && nextNeighbor.neighborIdx == p2) {
			const real_t compulsiveFact =
				(real_t)std::sqrt(sqLen) * params.edgeCompulsion / (real_t)(nextNeighbor.length);
			forceX -= dx * compulsiveFact;
			forceY -= dy * compulsiveFact;

			++neighborsIdx;
			if (neighborsIdx < neighborsIdxEnd) {
				nextNeighbor = cuNeighbors[neighborsIdx];
			}
		}
	}

	//assert(neighborsIdx == neighborsIdxEnd);

	real_t fact = params.timeQuantum / params.vertexMass;	// v = Ft/m  => t/m is mul factor for F.
	cuVelocities[pointIdx].x = (cuVelocities[pointIdx].x + (real_t)forceX * fact) * params.slowdown;
	cuVelocities[pointIdx].y = (cuVelocities[pointIdx].y + (real_t)forceY * fact) * params.slowdown;

	cuPointsNew[pointIdx].x = point.x + (cuVelocities[pointIdx].x * params.timeQuantum);
	cuPointsNew[pointIdx].y = point.y + (cuVelocities[pointIdx].y * params.timeQuantum);
}

void runComputePositions(
	const params_t params,
	const point_t *cuPointsOld,
	const index_t pointsCount,
	const neighbor_t *cuNeighbors,
	const index_t *cuNeighborsStart,
	const index_t neighborsCount,
	point_t *cuVelocities,
	point_t *cuPointsNew)
{
	const index_t blockCount = (pointsCount + blockSize - 1) / blockSize;
	computePositions<<<blockCount, blockSize>>>(
		params,
		cuPointsOld,
		pointsCount,
		cuNeighbors,
		cuNeighborsStart,
		neighborsCount,
		cuVelocities,
		cuPointsNew
	);
}
