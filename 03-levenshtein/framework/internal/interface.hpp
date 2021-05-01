#ifndef LEVENSHTEIN_FRAMEWORK_INTERNAL_INTERFACE_HPP
#define LEVENSHTEIN_FRAMEWORK_INTERNAL_INTERFACE_HPP

#include <vector>
#include <utility>



/*
 * \brief Interface defining the functor for Levenshtein edit distance.
 * \tparam C Type of the character used in strings (e.g., char, uint32_t, ...).
 * \tparam DIST Numeric type in which the distance is measured (e.g., size_t).
 * \tparam DEBUG Flag used for debugging output. If false, the class should
 *		not write anything to the output.
 */
template<typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class IEditDistance
{
public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	virtual void init(DIST len1, DIST len2) {}

	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2) = 0;
};


#endif
