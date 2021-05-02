#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>

#include <array>
#include <iostream>
#include <cassert>

template<typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class EditDistance : public IEditDistance<C, DIST, DEBUG> {
private:
    std::vector<DIST> col;
    std::vector<DIST> row;
    std::vector<DIST> prev_cross_diag;

    static constexpr std::size_t block_size = 1024;

    //static void print(const std::vector<DIST> v, size_t from, size_t to, char delim = ' ', bool newline = true) {
    //    for (size_t i = from ; i < to; ++i) {
    //        std::cout << v[i] << delim;
    //    }
    //    if (newline) {
    //        std::cout << '\n';
    //    }
    //}

    void process_block(std::size_t block_row_idx, std::size_t block_col_idx, std::size_t cross_idx,
                       const std::vector<C> &str1, const std::vector<C> &str2)
    {
        std::size_t start_row_idx = block_row_idx * block_size + 1;
        std::size_t start_col_idx = block_col_idx * block_size + 1;

        DIST cross = prev_cross_diag[cross_idx];

        for (std::size_t row_idx = start_row_idx; row_idx < start_row_idx + block_size; ++row_idx) {
            DIST next_cross = col[row_idx];

            for (std::size_t col_idx = start_col_idx; col_idx < start_col_idx + block_size; ++col_idx) {
                DIST dist1 = std::min<DIST>(row[col_idx], col[row_idx]) + 1;
                DIST dist2 = cross + (str1[col_idx - 1] == str2[row_idx - 1] ? 0 : 1);
                cross = row[col_idx];
                row[col_idx] = col[row_idx] = std::min<DIST>(dist1, dist2);
            }

            cross = next_cross;
        }
        prev_cross_diag[cross_idx] = cross;
    }

public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	virtual void init(DIST len1, DIST len2)
	{
		col.resize(len2 + 1);
		row.resize(len1 + 1);
		prev_cross_diag.resize(len2);

		for (std::size_t i = 0; i < col.size(); ++i) {
			col[i] = i;
		}

		for (std::size_t i = 0; i < row.size(); ++i) {
			row[i] = i;
		}

		prev_cross_diag[0] = 0;
	}


	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */
	virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2)
	{
        assert(str1.size() % block_size == 0);
        assert(str2.size() % block_size == 0);

		const std::size_t block_row_count = str2.size() / block_size;
        const std::size_t block_col_count = str1.size() / block_size;

        for (std::size_t diag = 1; diag < block_row_count + block_col_count; ++diag) {

            #pragma omp parallel for
            for (std::size_t col_idx = (diag < block_row_count) ? 0 : diag - block_row_count;
                 col_idx <= std::min(diag - 1, block_col_count - 1); ++col_idx) {

                std::size_t row_idx = diag - col_idx - 1;

                process_block(row_idx, col_idx, col_idx, str1, str2);
            }

            if (diag < block_row_count) {
                prev_cross_diag[diag] = diag * block_size;
            }
        }

        return row.back();
	}
};

