#include <iostream>

#include "Eigen/Dense"

#include "src/unitree_go2/autogen/autogen_functions.h"
#include "src/unitree_go2/autogen/autogen_defines.h"
#include "src/unitree_go2/utilities.h"

using namespace constants;

int main(int argc, char** argv) {
    // Declare Function Operations Struct and Assign Function Pointers:
    FunctionOperations Aeq_ops {
        .incref = Aeq_incref,
        .checkout = Aeq_checkout,
        .eval = Aeq,
        .release = Aeq_release,
        .decref = Aeq_decref
    };

    // Test Function Evaluation:
    using AeqFunctionParams = 
        FunctionParams<Aeq_SZ_ARG, Aeq_SZ_RES, Aeq_SZ_IW, Aeq_SZ_W, optimization::Aeq_rows, optimization::Aeq_cols, optimization::Aeq_sz, 4>;

    // Create Dummy Arguments:
    Eigen::VectorXd design_vector = Eigen::VectorXd::Random(optimization::design_vector_size);
    Eigen::MatrixXd mass_matrix = Eigen::MatrixXd::Random(model::nv_size, model::nv_size);
    Eigen::VectorXd coriolis_matrix = Eigen::VectorXd::Random(model::nv_size);
    Eigen::MatrixXd contact_jacobian = Eigen::MatrixXd::Random(model::nv_size, optimization::z_size);

    // Evaluate Function:
    auto Aeq_matrix = evaluate_function<AeqFunctionParams>(Aeq_ops, {design_vector.data(), mass_matrix.data(), coriolis_matrix.data(), contact_jacobian.data()});

    std::cout << "Aeq_matrix: " << Aeq_matrix << std::endl;

    return 0;
}