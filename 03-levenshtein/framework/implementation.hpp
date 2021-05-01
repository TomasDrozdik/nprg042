#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>



template<typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class EditDistance : public IEditDistance<C, DIST, DEBUG>
{
public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	virtual void init(DIST len1, DIST len2)
	{
		/*
			Your core goes here ...
		 */
	}


	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2)
	{
		/*
			Your core goes here ...
		*/
		throw bpp::RuntimeError("Solution not implemented yet.");
	}
};


#endif
