#include <iostream>

#include "Eigen/Dense"


template <int Rows_, int Cols_>
using Matrix = Eigen::Matrix<double, Rows_, Cols_, Eigen::RowMajor>;


int main(int argc, char** argv) {
    // Test Template Alias:
    Matrix<3, 3> A = Matrix<3, 3>::Zero();

    std::cout < A << std::endl;

    return 0;
}
