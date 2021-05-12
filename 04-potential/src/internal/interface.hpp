#ifndef CUDA_POTENTIAL_INTERNAL_INTERFACE_HPP
#define CUDA_POTENTIAL_INTERNAL_INTERFACE_HPP


#include <data.hpp>
#include <exception.hpp>

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>


/**
 * \brief Exception for reporting user exceptions form the implementation.
 *
 * This exception may be used by the students for debugging.
 */
class UserException : public bpp::RuntimeError
{
public:
	UserException() : RuntimeError() {}
	UserException(const char *msg) : RuntimeError(msg) {}
	UserException(const std::string &msg) : RuntimeError(msg) {}
	virtual ~UserException() throw() {}

	/*
	 * Overloading << operator that uses stringstream to append data to mMessage.
	 * Note that this overload is necessary so the operator returns object of exactly this class.
	 */
	template<typename T> UserException& operator<<(const T &data)
	{
		RuntimeError::operator<<(data);
		return *this;
	}
};



/*
 * \brief Abstract class for all tested applications that compute designated problem.
 * \tparam F Floating point num type used for point coordinates (float or double).
 * \tparam IDX_T Type used for various indices (e.g., referencing vertices in graph).
 * \tparam LEN_T Type in which lengths of edges is represented.
 */
template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class IProgramPotential
{
public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t;	// Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;

protected:
	ModelParameters<real_t> mParams;
	bool mVerbose;	///< Whether verbose (debugging mode) is turned on.


public:
	virtual ~IProgramPotential() {}	// enforce virtual destructor


	/*
	 * \brief This method is called prior to initialization
			and it SHOULD NOT be modified by the implementation.
	 * \param params Model parameters
	 * \param verbose Whether we are in verbose (debug) mode
	 */
	void preinitialize(const ModelParameters<real_t> &params, bool verbose)
	{
		mParams = params;
		mVerbose = verbose;
	}


	/*
	 * \brief Initialization that is specific to the implementation.
	 * \param points Number of points in the graph.
	 * \param edges The edge data (how the points are interconnected). Note that edges are symmetric.
	 * \param lengths Lengts of the edges (i-th value corresponds to i-th edge).
	 * \param iterations How many iterations will be performed.
	 */
	virtual void initialize(index_t points, const std::vector<edge_t>& edges,
		const std::vector<length_t> &lengths, index_t iterations) = 0;


	/*
	 * \brief Compute one iteration that update point values.
	 * \note It is guaranteed that points yielded by one iteration are
	 *		passed on to the following iteration.
	 * \param points Vector where the point values are stored.
	 */
	virtual void iteration(std::vector<point_t> &points) = 0;


	/**
	 * \brief Return the internal velocities.
	 * \param velocities A vector where the results should be stored.
	 */
	virtual void getVelocities(std::vector<point_t> &velocities) = 0;
};


#endif
