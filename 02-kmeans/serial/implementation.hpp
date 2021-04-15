#ifndef KMEANS_SERIAL_IMPLEMENTATION_HPP
#define KMEANS_SERIAL_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>



template<typename POINT = point_t, typename ASGN = std::uint8_t, bool DEBUG = false>
class KMeans : public IKMeans<POINT, ASGN, DEBUG>
{
private:
	typedef typename POINT::coord_t coord_t;

	std::vector<POINT> sums;
	std::vector<std::size_t> counts;


	static coord_t distance(const POINT &point, const POINT &centroid)
	{
		std::int64_t dx = (std::int64_t)point.x - (std::int64_t)centroid.x;
		std::int64_t dy = (std::int64_t)point.y - (std::int64_t)centroid.y;
		return (coord_t)(dx*dx + dy*dy);
	}

	static std::size_t getNearestCluster(const POINT &point, const std::vector<POINT> &centroids)
	{
		coord_t minDist = distance(point, centroids[0]);
		std::size_t nearest = 0;
		for (std::size_t i = 1; i < centroids.size(); ++i) {
			coord_t dist = distance(point, centroids[i]);
			if (dist < minDist) {
				minDist = dist;
				nearest = i;
			}
		}

		return nearest;
	}


public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param points Number of points being clustered.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 */
	virtual void init(std::size_t points, std::size_t k, std::size_t iters)
	{
		sums.resize(k);
		counts.resize(k);
	}


	/*
	 * \brief Perform the clustering and return the cluster centroids and point assignment
	 *		yielded by the last iteration.
	 * \note First k points are taken as initial centroids for first iteration.
	 * \param points Vector with input points.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 * \param centroids Vector where the final cluster centroids should be stored.
	 * \param assignments Vector where the final assignment of the points should be stored.
	 *		The indices should correspond to point indices in 'points' vector.
	 */
	virtual void compute(const std::vector<POINT> &points, std::size_t k, std::size_t iters,
		std::vector<POINT> &centroids, std::vector<ASGN> &assignments)
	{
		// Prepare for the first iteration
		centroids.resize(k);
		assignments.resize(points.size());
		for (std::size_t i = 0; i < k; ++i)
			centroids[i] = points[i];

		// Run the k-means refinements
		while (iters > 0) {
			--iters;

			// Prepare empty tmp fields.
			for (std::size_t i = 0; i < k; ++i) {
				sums[i].x = sums[i].y = 0;
				counts[i] = 0;
			}
			
			for (std::size_t i = 0; i < points.size(); ++i) {
				std::size_t nearest = getNearestCluster(points[i], centroids);
				assignments[i] = (ASGN)nearest;
				sums[nearest].x += points[i].x;
				sums[nearest].y += points[i].y;
				++counts[nearest];
			}

			for (std::size_t i = 0; i < k; ++i) {
				if (counts[i] == 0) continue;	// If the cluster is empty, keep its previous centroid.
				centroids[i].x = sums[i].x / (std::int64_t)counts[i];
				centroids[i].y = sums[i].y / (std::int64_t)counts[i];
			}
		}
	}
};


#endif
