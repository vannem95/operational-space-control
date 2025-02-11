#include "src/unitree_go2/autogen/autogen_functions.h"
#include "src/unitree_go2/autogen/autogen_defines.h"

#include "src/utilities.h"


struct FunctionOperations {
    void (*func_incref)();
    int (*func_checkout)();
    int (*eval_func)(const double* args[], double* res[], casadi_int iw[], double w[], int mem);
    void (*func_release)(int mem);
    void (*func_decref)();
};

template<size_t sz_args, size_t sz_res, size_t sz_iw, size_t sz_w, int rows, int cols, size_t output_size, size_t N>
Eigen::Matrix<double, rows, cols, Eigen::ColMajor> evaluate_function(FunctionOperations& ops, const std::array<double*, N> arguments) {
    // Allocate Work Vectors:
    const double *args[sz_args];
    double *res[sz_res];
    casadi_int iw[sz_iw];
    double w[sz_w];

    // Place result pointer in the result array:
    double result[output_size];
    res[0] = result;

    // Increase the reference count:
    ops.func_incref();

    // Copy arguments into args array:
    std::copy(arguments.begin(), arguments.end(), args);

    // Initialize Memory:
    int mem = ops.func_checkout();

    // Evaluate the Function:
    if (ops.eval_func(args, res, iw, w, mem)) 
        return 1;

    // Release Memory:
    ops.func_release(mem);

    // Decrease the reference count:
    ops.func_decref();

    return Eigen::Map<Eigen::Matrix<double, rows, cols, Eigen::ColMajor>>(result);
}