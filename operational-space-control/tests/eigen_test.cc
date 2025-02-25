#include <iostream>
#include <string_view>

#include "Eigen/Dense"

#include "src/unitree_go2/autogen/autogen_defines.h"


template <int Rows_, int Cols_>
using Matrix = Eigen::Matrix<double, Rows_, Cols_, Eigen::RowMajor>;


int main(int argc, char** argv) {
    // Test Template Alias:
    Matrix<3, 3> A = Matrix<3, 3>::Zero();

    std::cout << A << std::endl;

    // String View Test:
    for(const std::string_view& site : constants::model::site_list) {
        std::cout << site << std::endl;
    }


    return 0;
}
