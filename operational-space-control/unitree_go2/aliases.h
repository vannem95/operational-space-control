#pragma once

#include <Eigen/Dense>

#include "operational-space-control/unitree_go2/constants.h"

using namespace operational_space_controller::constants;


namespace operational_space_controller {
    namespace aliases {
        template <int Rows_, int Cols_>
        using Matrix = Eigen::Matrix<double, Rows_, Cols_, Eigen::RowMajor>;
        
        template <int Rows_>
        using Vector = Eigen::Matrix<double, Rows_, 1>;

        template <int Rows_, int Cols_>
        using MatrixColMajor = Eigen::Matrix<double, Rows_, Cols_, Eigen::ColMajor>;

        using TaskspaceTargets = Matrix<model::site_ids_size, 6>;

        using OptimizationSolution = Vector<optimization::design_vector_size>;

        using dv_sequence = Eigen::seqN(0, optimization::dv_size);
        using u_sequence = Eigen::seqN(optimization::dv_idx, optimization::u_size);
        using z_sequence = Eigen::seqN(optimization::u_idx, optimization::z_size);
    }
}