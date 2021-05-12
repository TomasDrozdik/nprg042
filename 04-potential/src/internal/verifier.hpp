#ifndef CUDA_POTENTIAL_INTERNAL_VERIFIER_HPP
#define CUDA_POTENTIAL_INTERNAL_VERIFIER_HPP

#include <data.hpp>
#include <serial.hpp>

#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>


/**
 * \brief Base class for all result verifiers which also defines the interface.
 * \tparam F Floating point num type used for point coordinates (float or double).
 * \tparam IDX_T Type used for various indices (e.g., referencing vertices in graph).
 * \tparam LEN_T Type in which lengths of edges is represented.
 */
template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class Verifier
{
public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t;	// Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;


protected:
	// Verification Parameters
	real_t mTolerance;			///< Relative tolerance for float numbers.
	bool mVerbose;				///< Whether additional information should be reported.
	std::size_t mMaxReportError;		///< Maximal number of errors fully reported.

public:
	Verifier(real_t tolerance, bool verbose, std::size_t maxReportError)
		: mTolerance(std::fabs(tolerance)), mVerbose(verbose), mMaxReportError(maxReportError) {}
	virtual ~Verifier() {}

	/**
	 * \brief Initialize the verifier (saves data for serial verification).
	 */
	virtual void init(const Graph<F, IDX_T, LEN_T> &graph, const ModelParameters<real_t> &params) {}


	/**
	 * \brief Check next iteration. Computes results serially and checks with given results.
	 * \param points Next version of point positions computed by the tested solution.
	 * \param velocities Next version of velocities computed by the tested solution.
	 * \return True if the results checked out, false otherwise. Note that any errors
	 *		are printed immediately, if in verbose mode.
	 */
	virtual bool check(const std::vector<point_t> &points, std::vector<point_t> &velocities) { return true; }


	/**
	 * \brief Print statistics accumulated by the verifier (e.g., greatest deviation from the correct results).
	 */
	virtual void printStats() const {}
};





/**
 * \brief Full version of the verifier that implements prescribed interface using serial simulation.
 */
template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class VerifierFull : public Verifier<F, IDX_T, LEN_T>
{
public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t;	// Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;
	typedef SerialSimulator<coord_t, index_t, length_t> simulator_t;

	/**
	 * \brief Compare two floats and return relative deviation.
	 * \param value1 One of the values being compared.
	 * \param value2 One of the values being compared.
	 * \return Relative deviation (e.g., value 0.05 means difference ~ 5% of absolute values).
	 */
	static double compare(real_t value1, real_t value2)
	{
		double v1 = (double)value1;
		double v2 = (double)value2;
		
		double len = std::fabs(v1 - v2);
		double greater = std::max<double>(std::fabs(v1), std::fabs(v2));

		if (greater < 0.000001) {
			// Cheating -- if the absolute values are too small, we take absolute difference.
			return len;
		}
		return len / greater;
	}



private:
	// Data computed by serial version.
	std::unique_ptr<simulator_t> mSerial;
	std::vector<point_t> mPoints;

	// Statistics
	double mPointsMaxDeviation;
	double mPointsDeviationSum;
	std::size_t mPointsDeviationCount;

	double mVelocitiesMaxDeviation;
	double mVelocitiesDeviationSum;
	std::size_t mVelocitiesDeviationCount;


	/**
	 * \brief Compare two vectors of points or velocities item by item. The errors are directly
	 *		printed to stderr (if verbose is set) up to the max errors limit.
	 * \param vec1 Vector of coordinates to be compared.
	 * \param vec2 Vector of coordinates to be compared.
	 * \param vectorName Name of the compared vectors (so the errors can be reported properly).
	 * \param deviationMax Variable were the maximal deviation is kept (for statistics).
	 * \param deviationSum Variable were the sum of all deviations is accumulated (for statistics).
	 * \param deviationCount Variable where the number of sumed deviations is kept (for statistics).
	 * \param errors Number of encountered errors (incremented with each error).
	 */
	void compareVectors(const std::vector<point_t>& vec1, const std::vector<point_t>& vec2,
		const char *vectorName, double &deviationMax, double &deviationSum, std::size_t &deviationCount, std::size_t &errors)
	{
		if (vec1.size() != vec2.size())
			throw (bpp::RuntimeError() << "Error in vector verification. Vectors of different lengths ("
				<< vec1.size() << " and " << vec2.size() << ") were passed on for testing.");

		for (std::size_t i = 0; i < vec1.size(); ++i) {
			// Get deviations (and update statistics);
			double dx = compare(vec1[i].x, vec2[i].x);
			double dy = compare(vec1[i].y, vec2[i].y);
			double maxD = std::max<double>(dx, dy);
			deviationMax = std::max<double>(deviationMax, maxD);
			deviationSum += dx + dy;
			deviationCount += 2;

			if (maxD > this->mTolerance) {
				if (this->mVerbose && errors < this->mMaxReportError) {
					std::cerr << "Coordinates in " << vectorName << "[" << i << "] are out of tolerance. ("
						<< vec1[i].x << ", " << vec1[i].y << ") =/= (" << vec2[i].x << ", " << vec2[i].y << ")" << std::endl;
				}
				++errors;
			}
		}
	}


public:
	VerifierFull(real_t tolerance, bool verbose, std::size_t maxReportError)
		: Verifier<F, IDX_T, LEN_T>(tolerance, verbose, maxReportError),
		mPointsMaxDeviation(0.0), mPointsDeviationSum(0.0), mPointsDeviationCount(0),
		mVelocitiesMaxDeviation(0.0), mVelocitiesDeviationSum(0.0), mVelocitiesDeviationCount(0)
	{}


	void init(const Graph<coord_t, index_t, length_t> &graph, const ModelParameters<real_t> &params) override
	{
		mSerial = std::make_unique<simulator_t>(graph.pointCount(), graph.getEdges(), graph.getLengths(), params);
		mPoints.assign(graph.getPoints().begin(), graph.getPoints().end());
	}


	bool check(const std::vector<point_t> &points, std::vector<point_t> &velocities) override
	{
		mSerial->updatePoints(mPoints);

		for (auto&& p : points) {
			if (std::isnan(p.x) || std::isnan(p.y)) {
				std::cerr << "NaN value found in points vector!" << std::endl;
				return false;
			}
		}

		for (auto&& v : velocities) {
			if (std::isnan(v.x) || std::isnan(v.y)) {
				std::cerr << "NaN value found in points vector!" << std::endl;
				return false;
			}
		}

		std::size_t errors = 0;	// total error counter (one point out of tolerance ~ one error)
		compareVectors(mPoints, points, "points", mPointsMaxDeviation, mPointsDeviationSum, mPointsDeviationCount, errors);
		compareVectors(mSerial->getVelocities(), velocities, "velocities", mVelocitiesMaxDeviation, mVelocitiesDeviationSum, mVelocitiesDeviationCount, errors);

		// Check error results.
		if (errors > 0) {
			if (this->mVerbose)
				std::cerr << "Total " << errors << " errors found!" << std::endl;
			return false;
		}

		// Update verifier for the next iteration.
		mPoints = points;
		mSerial->swapVelocities(velocities);
		return true;
	}


	void printStats() const override
	{
		if (!this->mVerbose) return;

		std::cout << "Points max diff: " << mPointsMaxDeviation << ", avg diff: " << (mPointsDeviationSum / (double)mPointsDeviationCount) << std::endl;
		std::cout << "Velocities max diff: " << mVelocitiesMaxDeviation << ", avg diff: " << (mVelocitiesDeviationSum / (double)mVelocitiesDeviationCount) << std::endl;
	}
};

#endif
