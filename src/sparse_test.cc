#include <iostream>

#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "osqp.h"


int main(int argc, char **argv) {
    // Set Dense matrix to sparse matrix:
    Eigen::Matrix<double, 3, 2> A;
    A << 1, 1,
         1, 0,
         0, 1;
    
    Eigen::SparseMatrix<double> As = A.sparseView();
    As.makeCompressed();

    std::cout << "Dense Matrix A: " << A << std::endl;
    std::cout << "Inner Index Ptr: " << std::endl;
    for(int i = 0; i <= As.innerSize(); i++)
        std::cout << As.innerIndexPtr()[i] << std::endl;
    std::cout << "Out Index Ptr: " << std::endl;
    for(int i = 0; i <= As.outerSize(); i++)
        std::cout << As.outerIndexPtr()[i] << std::endl;
    std::cout << "Value Ptr: " << std::endl;
    for(int i = 0; i < As.nonZeros(); i++)
        std::cout << As.valuePtr()[i] << std::endl;

    return 0;
}