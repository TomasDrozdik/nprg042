#define _CRT_SECURE_NO_WARNINGS
/*
 * K-means Clustering Algorithm
 */
#include <implementation.hpp>

#include <exception.hpp>
#include <stopwatch.hpp>
#include <interface.hpp>

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdio>



void print_usage()
{
	std::cout << "Arguments: [ -debug ] <points_file> <k> <iters> <centroids_file> <assignments_file>" << std::endl;
	std::cout << "  -debug             - flag for debugging output" << std::endl;
	std::cout << "  <points_file>      - input file containing point coordinates" << std::endl;
	std::cout << "  <k>                - desired number of clusters (1-256)" << std::endl;
	std::cout << "  <iters>            - number of refining iterations (1-1000)" << std::endl;
	std::cout << "  <centroids_file>   - output file where final centroids are stored" << std::endl;
	std::cout << "  <assignments_file> - output file where final assignment is stored" << std::endl;
}


/*
 * \bried Convert string to unsigned number. Zero is returned on error.
 */
std::size_t getNumArg(const std::string &str)
{
	std::size_t idx;
	std::size_t res = (std::size_t)std::stoul(str, &idx);
	return (idx != str.length()) ? 0 : res;
}


/*
 * \bried Load an entire file into a vector of points.
 */
void load_file(const std::string &fileName, std::vector<point_t> &res)
{
	// Open the file.
	std::FILE *fp = std::fopen(fileName.c_str(), "rb");
	if (fp == nullptr)
		throw (bpp::RuntimeError() << "File '" << fileName << "' cannot be opened for reading.");

	// Determine length of the file and 
	std::fseek(fp, 0, SEEK_END);
	std::size_t count = (std::size_t)(std::ftell(fp) / sizeof(point_t));
	std::fseek(fp, 0, SEEK_SET);
	res.resize(count);

	// Read the entire file.
	std::size_t offset = 0;
	while (offset < count) {
		std::size_t batch = std::min<std::size_t>(count - offset, 1024*1024);
		if (std::fread(&res[offset], sizeof(point_t), batch, fp) != batch)
			throw (bpp::RuntimeError() << "Error while reading from file '" << fileName << "'.");
		offset += batch;
	}

	std::fclose(fp);
}


/*
* \bried Load an entire file into a vector of points.
*/
template<typename T>
void save_file(const std::string &fileName, const std::vector<T> &data)
{
	// Open the file.
	std::FILE *fp = std::fopen(fileName.c_str(), "wb");
	if (fp == nullptr)
		throw (bpp::RuntimeError() << "File '" << fileName << "' cannot be opened for writing.");

	// Write the entire vector to the file.
	std::size_t offset = 0;
	while (offset < data.size()) {
		std::size_t batch = std::min<std::size_t>(data.size() - offset, 1024*1024);
		if (std::fwrite(&data[offset], sizeof(T), batch, fp) != batch)
			throw (bpp::RuntimeError() << "Error while writing data to file '" << fileName << "'.");
		offset += batch;
	}

	std::fclose(fp);
}



// Main routine that performs the computation.
template<bool DEBUG>
void runKmeans(const std::vector<point_t> &points, std::size_t k, std::size_t iters,
	std::vector<point_t> &centroids, std::vector<std::uint8_t> &assignments)
{
	// Initialize distance functor.
	KMeans<point_t, std::uint8_t, DEBUG> kMeans;
	kMeans.init(points.size(), k, iters);
	
	// Preallocate results.
	centroids.clear();
	centroids.reserve(k);
	assignments.clear();
	assignments.reserve(points.size());
		
	// Compute the distance.
	bpp::Stopwatch stopwatch(true);
	kMeans.compute(points, k, iters, centroids, assignments);
	stopwatch.stop();
	if (centroids.size() != k)
		throw (bpp::RuntimeError() << "Invalid number of centroids (" << centroids.size() <<", but " << k << "expected).");
	if (assignments.size() != points.size())
		throw (bpp::RuntimeError() << "Invalid number of assignments (" << assignments.size() <<", but " << points.size() << "expected).");

	std::cout << stopwatch.getMiliseconds() << std::endl;
}


/*
 * Application Entry Point
 */
int main(int argc, char **argv)
{
	// Process arguments.
	--argc; ++argv;
	bool debug = false;
	if (argc == 6 && std::string(*argv) == std::string("-debug")) {
		--argc; ++argv;
		debug = true;
	}

	if (argc != 5) {
		print_usage();
		return 0;
	}

	std::size_t k = getNumArg(argv[1]);
	std::size_t iters = getNumArg(argv[2]);
	if (k == 0 || iters == 0 || k > 256 || iters > 1000) {
		print_usage();
		return 0;
	}

	// Load files.
	std::vector<point_t> points;
	try {
		load_file(argv[0], points);
	}
	catch (std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		print_usage();
		return 1;
	}


	// Run the algorithm.
	std::vector<point_t> centroids;
	std::vector<std::uint8_t> assignment;
	try {
		if (debug)
			runKmeans<true>(points, k, iters, centroids, assignment);
		else
			runKmeans<false>(points, k, iters, centroids, assignment);
		
		// Save outputs.
		save_file(argv[3], centroids);
		save_file(argv[4], assignment);
	}
	catch (std::exception &e) {
		std::cout << "FAILED" << std::endl;
		std::cerr << e.what() << std::endl;
		return 2;
	}

	return 0;
}
