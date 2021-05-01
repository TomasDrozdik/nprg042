#ifndef LEVENSHTEIN_SERIAL_IMPLEMENTATION_HPP
#define LEVENSHTEIN_SERIAL_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>

#include <utility>


template<typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class EditDistance : public IEditDistance<C, DIST, DEBUG>
{
private:
	std::vector<DIST> mRow;

public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	virtual void init(DIST len1, DIST len2)
	{
		mRow.resize((std::size_t)std::min<DIST>(len1, len2));
		for (std::size_t i = 0; i < mRow.size(); ++i)
			mRow[i] = i+1;
	}


	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2)
	{
		// Number of rows (size of the larger of the two strings).
		std::size_t rows = std::max<std::size_t>(str1.size(), str2.size());

		// Special case (one of the strings is empty).
		if (mRow.size() == 0)
			return rows;

		// Make sure s1 is the shorter string and s2 is the longer one.
		const C* s1 = &str1[0];
		const C* s2 = &str2[0];
		if (str1.size() > str2.size())
			std::swap(s1, s2);

		// Traverse the distanece matrix keeping exactly one row in memory.
		for (std::size_t row = 0; row < rows; ++row) {
			DIST lastUpper = row;
			DIST lastLeft = lastUpper + 1;
			for (std::size_t i = 0; i < mRow.size(); ++i) {
				DIST dist1 = std::min<DIST>(mRow[i], lastLeft) + 1;
				DIST dist2 = lastUpper + (s1[i] == s2[row] ? 0 : 1);
				lastUpper = mRow[i];
				lastLeft = mRow[i] = std::min<DIST>(dist1, dist2);
			}
		}

		// Last item of the last row is the result.
		return mRow.back();
	}
};


#endif
