#include "image.hpp"
#include "stopwatch.hpp"
#include "cuda_helpers.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <cstdlib>

/**
 * Serial reference implementation of the blur algorithm.
 */
template<typename T>
void serial_blur(const Image<T> &src, Image<T> &dest, int radius = 5)
{
	if (src.width() != dest.width() || src.height() != dest.height()) {
		dest.resize(src.width(), src.height());
	}

	const T* srcData = src.getRawData();
	T* destData = dest.getRawData();

	int width = (int)src.width();
	int height = (int)src.height();
	for (int y = 0; y < height; ++y)
		for (int x = 0; x < width; ++x) {
			// Process pixel [x,y] ...
			T sum = (T)0;
			T weights = (T)0;

			// Computing weighted average of the pixels in the radius...
			for (int dy = -radius; dy <= radius; ++dy)
				for (int dx = -radius; dx <= radius; ++dx) {
					int srcX = x + dx;
					int srcY = y + dy;
					if (srcX >= 0 && srcY >= 0 && srcX < width && srcY < height) {
						int distance = std::abs(dx) + std::abs(dy); // Manhattan distance
						T weight = (distance > 0) ? 1 / (T)distance : (T)5;
						weights += weight;
						sum += srcData[srcY*width + srcX] * weight;
					}
				}

			destData[y*width + x] = sum / weights;
		}
}


template<typename T>
__global__ void blur_kernel(const T *src, T *dest, int width, int height, int radius)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= width) return;

	// Process pixel [x,y] ...
	T sum = (T)0;
	T weights = (T)0;

	// Computing weighted average of the pixels in the radius...
	for (int dy = -radius; dy <= radius; ++dy)
		for (int dx = -radius; dx <= radius; ++dx) {
			int srcX = x + dx;
			int srcY = y + dy;
			if (srcX >= 0 && srcY >= 0 && srcX < width && srcY < height) {
				int distance = std::abs(dx) + std::abs(dy); // Manhattan distance
				T weight = (distance > 0) ? 1 / (T)distance : (T)5;
				weights += weight;
				sum += src[srcY*width + srcX] * weight;
			}
		}

	dest[y*width + x] = sum / weights;
}


/**
 * CUDA implementation of the blur stencil.
 */
template<typename T>
void cuda_blur(const Image<T> &src, Image<T> &dest, int radius = 5)
{
	if (src.width() != dest.width() || src.height() != dest.height()) {
		dest.resize(src.width(), src.height());
	}

	CUCH(cudaSetDevice(0));
	
	T* cuSrc = nullptr;
	T* cuDest = nullptr;
	CUCH(cudaMalloc(&cuSrc, src.size() * sizeof(T)));
	CUCH(cudaMalloc(&cuDest, dest.size() * sizeof(T)));

	bpp::Stopwatch stopwatch(true);
	CUCH(cudaMemcpy(cuSrc, src.getRawData(), src.size() * sizeof(T), cudaMemcpyHostToDevice));

	int width = (int)src.width();
	int height = (int)src.height();

	const int blockSize = 32;
	dim3 dimGrid((width+blockSize-1) / blockSize, (height+blockSize-1) / blockSize);
	blur_kernel<T><<<dimGrid, dim3(blockSize, blockSize)>>>(cuSrc, cuDest, width, height, radius);
	CUCH(cudaGetLastError());

	CUCH(cudaDeviceSynchronize());
	CUCH(cudaMemcpy(dest.getRawData(), cuDest, dest.size() * sizeof(T), cudaMemcpyDeviceToHost));
	stopwatch.stop();
	std::cout << stopwatch.getMiliseconds() << " ms" << std::endl;
}


/*
 * Application Entry Point
 */
int main(int argc, char *argv[])
{
	if (argc != 5) {
		std::cout << "Four arguments expected: algorithm radius input_file output_file." << std::endl;
		return 1;
	}

	try {
		std::string algorithm(argv[1]);
		int radius = std::abs(std::atoi(argv[2]));
		std::string inFile(argv[3]), outFile(argv[4]);

		// Prepare the data...
		std::cout << "Loading " << inFile << " ..." << std::endl;
		Image<float> src, dest;
		src.loadNetpbm(inFile);
		dest.resize(src.width(), src.height());

		bpp::Stopwatch stopwatch(true); // true = start it right away

		// Yeah, this is an ugly if-else, but what the heck, we have only 2 options...
		if (algorithm == "serial") {
			serial_blur(src, dest, radius);
		}
		else if (algorithm == "cuda") {
			cuda_blur(src, dest, radius);
		}
		else {
			throw std::runtime_error("Invalid algorithm '" + algorithm + "' selected.");
		}

		stopwatch.stop();
		std::cout << "Algorithm " << algorithm << " with radius " << radius << " took " << stopwatch.getMiliseconds() << " ms." << std::endl;

		// Write down the results...
		std::cout << "Saving results to " << outFile << " ..." << std::endl;
		dest.saveNetpbm(outFile);

		return 0;
	}
	catch (std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 2;
	}
}
