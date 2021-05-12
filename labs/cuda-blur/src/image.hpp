#ifndef CUDA_BLUR_STENCIL_IMAGE_HPP
#define CUDA_BLUR_STENCIL_IMAGE_HPP

#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include <cstdio>


/**
 * Wrapper and container for the greyscale image data.
 * All data are represented by T values, which should normelly be float or double.
 */
template<typename T = float>
class Image
{
private:
	std::size_t mWidth, mHeight;
	std::vector<T> mData; ///< Actual values of the pixels (stored rowise).

public:
	Image() : mWidth(0), mHeight(0) {}

	std::size_t width() const { return mWidth; }
	std::size_t height() const { return mHeight; }
	std::size_t size() const { return mWidth * mHeight; }

	const T* getRawData() const { return &mData[0]; }
	T* getRawData() { return &mData[0]; }

	bool isEmpty() const { return mWidth == 0 || mHeight == 0; }


	/**
	 * Change the proportions of the image.
	 */
	void resize(std::size_t width, std::size_t height)
	{
		mWidth = width;
		mHeight = height;
		mData.resize(width * height);

		// Reset the bitmap contents.
		for (auto && x : mData) {
			x = (T)0;
		}
	}


	/**
	 * Load the bitmap from Netpbm file (onlo P5 type is implemented).
	 */
	void loadNetpbm(const std::string &fileName)
	{
		std::FILE *fp = std::fopen(fileName.c_str(), "rb");
		if (!fp) {
			throw std::runtime_error("Unable to open " + fileName + " file for reading.");
		}

		// Check the magic header.
		char signature[2];
		if (std::fread(signature, 1, sizeof(signature), fp) != 2
			|| signature[0] != 'P' || signature[1] != '5') {
			throw std::runtime_error("Given pbm file has invalid header.");
		}

		// Get (text) metadata.
		unsigned width, height, maxVal;
		std::fscanf(fp, "%u %u %u\n", &width, &height, &maxVal);
		if (width > 65536 || height > 65536) {
			throw std::runtime_error("The image is too large. Maximal size in each dimension is 65536.");
		}

		if (maxVal != 255) {
			throw std::runtime_error("The implementation supports only images with 8 bits per pixel.");
		}

		// Load raw data.
		std::size_t totalSize = width * height;
		std::vector<unsigned char> rawData(totalSize);
		std::size_t res = std::fread(&rawData[0], 1, totalSize, fp);
		std::fclose(fp);
		if (res != totalSize) {
			throw std::runtime_error("Unable to load all image data. The file is probably corrupted.");
		}

		// Convert the data into internal format.
		resize(width, height);
		for (std::size_t i = 0; i < totalSize; ++i) {
			mData[i] = (T)rawData[i] / (T)maxVal;
		}
	}


	/**
	 * Save the data into P5 Netpbm file.
	 */
	void saveNetpbm(const std::string &fileName)
	{
		if (isEmpty()) {
			throw std::runtime_error("Cannot save empty image.");
		}

		std::FILE *fp = std::fopen(fileName.c_str(), "wb");
		if (!fp) {
			throw std::runtime_error("Unable to open " + fileName + " file for writing.");
		}

		// Write header metadata.
		unsigned w = (unsigned)width();
		unsigned h = (unsigned)height();
		unsigned maxVal = 255;
		std::fprintf(fp, "P5\n%u %u\n%u\n", w, h, maxVal);

		// Convert internal data into 8-bit representation.
		std::size_t totalSize = size();
		std::vector<unsigned char> rawData(totalSize);
		for (std::size_t i = 0; i < totalSize; ++i) {
			T x = std::min(std::max(mData[i], (T)0), (T)1); // saturate in [0,1] interval
			rawData[i] = (unsigned char)std::round(x * (T)255);
		}

		// Write the data into the file.
		if (fwrite(&rawData[0], 1, totalSize, fp) != totalSize) {
			throw std::runtime_error("Error occurred while writing the file.");
		}
		std::fclose(fp);
	}
};

#endif
