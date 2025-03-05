#pragma once

#include <Eigen/Dense>

namespace operational_space_controller {
    namespace aliases {
        template <int Rows_, int Cols_>
        using Matrix = Eigen::Matrix<double, Rows_, Cols_, Eigen::RowMajor>;
        
        template <int Rows_>
        using Vector = Eigen::Matrix<double, Rows_, 1>;

        template <int Rows_, int Cols_>
        using MatrixColMajor = Eigen::Matrix<double, Rows_, Cols_, Eigen::ColMajor>;
    }
}