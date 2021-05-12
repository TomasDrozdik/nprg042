#ifndef CUDA_POTENTIAL_INTERNAL_DATA_HPP
#define CUDA_POTENTIAL_INTERNAL_DATA_HPP

#include <exception.hpp>

#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>



/**
 * \brief Specific exception thrown when data loading/management fails.
 */
class DataException : public bpp::RuntimeError
{
public:
	DataException() : RuntimeError() {}
	DataException(const char *msg) : RuntimeError(msg) {}
	DataException(const std::string &msg) : RuntimeError(msg) {}
	virtual ~DataException() throw() {}

	/*
	 * Overloading << operator that uses stringstream to append data to mMessage.
	 * Note that this overload is necessary so the operator returns object of exactly this class.
	 */
	template<typename T> DataException& operator<<(const T &data)
	{
		RuntimeError::operator<<(data);
		return *this;
	}
};



/*
 * \brief Physical model parameters.
 * \tparam F Real type used for parameter representation (float or double).
 */
template<typename F = float>
struct ModelParameters
{
	typedef F real_t;

	real_t vertexRepulsion;
	real_t vertexMass;
	real_t edgeCompulsion;
	real_t slowdown;
	real_t timeQuantum;
};



/**
 * \brief Structure representing 2D point with real coordinates.
 */
template<typename F = float>
struct Point
{
	typedef F coord_t;
	coord_t x, y;
	Point() : x(), y() {}
	Point(coord_t _x, coord_t _y) : x(_x), y(_y) {}
};



/**
 * \brief Structure representing one edge (indices to adjacent vertices).
 */
template<typename IDX_T = std::uint32_t>
struct Edge
{
	IDX_T p1, p2;
	Edge() {}
	Edge(IDX_T _p1, IDX_T _p2) : p1(_p1), p2(_p2) {}
};



/**
 * \brief Class that encapsulates all necessary data and their loading from a file.
 * \tparam F Floating point num type used for point coordinates (float or double).
 * \tparam IDX_T Type used for various indices (e.g., referencing vertices in graph).
 * \tparam LEN_T Type in which lengths of edges is represented.
 */
template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class Graph
{
public:
	typedef F coord_t;
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;

private:
	std::vector<point_t> mPoints;			///< List of points.
	std::vector<edge_t> mEdges;			///< List of edges (indices to points array).
	std::vector<LEN_T> mLengths;		///< Length values (i-th value corresponds to i-th edge).

	struct FileHeader
	{
		std::uint32_t magic;
		std::uint32_t floatSize;
		std::uint32_t indexSize;
		std::uint32_t lengthSize;
		index_t pointsCount;
		index_t edgesCount;

		static const std::uint32_t MAGIC_VALUE = 0x67aff11e;
	};

public:
	Graph() {}


	/**
	 * \brief Load all graph data from a file.
	 */
	void load(const std::string &fileName)
	{
		std::FILE *fp = std::fopen(fileName.c_str(), "rb");
		if (fp == NULL)
			throw (DataException() << "Unable to open file '" << fileName << "' for reading.");

		FileHeader header;
		if (std::fread(&header, sizeof(FileHeader), 1, fp) != 1)
			throw (DataException() << "Unable to read header from '" << fileName << "' file.");
		if (header.magic != FileHeader::MAGIC_VALUE)
			throw (DataException() << "Input file '" << fileName << "' has wrong header format.");

		if (header.floatSize != sizeof(coord_t) || header.indexSize != sizeof(index_t) || header.lengthSize != sizeof(length_t))
			throw (DataException() << "Input file '" << fileName << "' uses different numeric precision of coordinates or indices.");

		mPoints.resize(header.pointsCount);
		mEdges.resize(header.edgesCount);
		mLengths.resize(header.edgesCount);

		if (std::fread(&mPoints[0], sizeof(point_t), mPoints.size(), fp) != mPoints.size())
			throw (DataException() << "Unable to read points from '" << fileName << "' file.");

		if (std::fread(&mEdges[0], sizeof(edge_t), mEdges.size(), fp) != mEdges.size())
			throw (DataException() << "Unable to read edges from '" << fileName << "' file.");

		if (std::fread(&mLengths[0], sizeof(length_t), mLengths.size(), fp) != mLengths.size())
			throw (DataException() << "Unable to read lengths from '" << fileName << "' file.");

		std::fclose(fp);
	}


	/*
	 * Internal Data Accessors
	 */
	index_t pointCount() const						{ return (index_t)mPoints.size(); }
	index_t edgeCount() const						{ return (index_t)mEdges.size(); }

	const point_t& getPoint(index_t i) const		{ return mPoints[i]; }
	point_t& getPoint(index_t i)					{ return mPoints[i]; }
	const std::vector<point_t>& getPoints() const	{ return mPoints; }
	std::vector<point_t>& getPoints()				{ return mPoints; }

	const edge_t& getEdge(index_t i) const			{ return mEdges[i]; }
	edge_t& getEdge(index_t i)						{ return mEdges[i]; }
	const std::vector<edge_t>& getEdges() const		{ return mEdges; }
	std::vector<edge_t>& getEdges()					{ return mEdges; }

	length_t getLength(index_t i) const				{ return mLengths[i]; }
	const std::vector<length_t>& getLengths() const	{ return mLengths; }
	std::vector<length_t>& getLengths()				{ return mLengths; }
};

#endif
