#ifndef KMEANS_FRAMEWORK_INTERNAL_INTERFACE_HPP
#define KMEANS_FRAMEWORK_INTERNAL_INTERFACE_HPP

#include <vector>
#include <utility>
#include <cstdint>


/*
 * \brief Structure representing 2d point coordinates.
 */
struct point_t
{
	typedef std::int64_t coord_t;
	coord_t x, y;
};



/*
 * \brief Interface defining the k-means algorithm wrapper.
 * \tparam POINT Structure type representing points (and centroids).
 * \tparam ASGN Numeric type that holds cluster index (for assignment).
 * \tparam DEBUG Flag used for debugging output. If false, the class should
 *		not write anything to the output.
 */
template<typename POINT = point_t, typename ASGN = std::uint8_t, bool DEBUG = false>
class IKMeans
{
public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param points Number of points being clustered.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 */
	virtual void init(std::size_t points, std::size_t k, std::size_t iters) {}

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
		std::vector<POINT> &centroids, std::vector<ASGN> &assignments) = 0;
};


#endif
