// Template struct to hold CasADi function sizes
template<size_t ArgSize, size_t ResSize, size_t IwSize, size_t WSize, size_t ResultSize>
struct CasadiFunctionSizes {
    static constexpr size_t sz_arg = ArgSize;
    static constexpr size_t sz_res = ResSize;
    static constexpr size_t sz_iw = IwSize;
    static constexpr size_t sz_w = WSize;
    static constexpr size_t result_size = ResultSize;
};

// Template struct to hold CasADi function pointers
template<typename Sizes>
struct CasadiFunctionPointers {
    using CasadiFunc = void (*)(const double**, double**, casadi_int*, double*, int);
    using IncrefFunc = void (*)();
    using DecrefFunc = void (*)();
    using AllocMemFunc = int (*)();
    using InitMemFunc = void (*)(int);
    using FreeMemFunc = void (*)(int);

    CasadiFunc func;
    IncrefFunc incref;
    DecrefFunc decref;
    AllocMemFunc alloc_mem;
    InitMemFunc init_mem;
    FreeMemFunc free_mem;
};

// Helper function to evaluate CasADi function
template<typename ReturnType, typename Sizes, typename... Args>
ReturnType evaluate_casadi_function(
    const CasadiFunctionPointers<Sizes>& func_ptrs,
    const std::vector<const double*>& args_vec) {
    
    // Allocate result object
    double res0[Sizes::result_size];

    // Allocate CasADi work vectors
    const double* args[Sizes::sz_arg];
    double* res[Sizes::sz_res];
    casadi_int iw[Sizes::sz_iw];
    double w[Sizes::sz_w];

    // Copy arguments
    std::copy(args_vec.begin(), args_vec.end(), args);
    res[0] = res0;

    // Reference counting
    func_ptrs.incref();

    // Initialize memory
    int mem = func_ptrs.alloc_mem();
    func_ptrs.init_mem(mem);

    // Evaluate function
    func_ptrs.func(args, res, iw, w, mem);

    // Map result to return type
    ReturnType result = Eigen::Map<ReturnType>(res0);

    // Free memory
    func_ptrs.free_mem(mem);
    func_ptrs.decref();

    return result;
}

// Example usage for Aeq function
using AeqSizes = CasadiFunctionSizes<Aeq_SZ_ARG, Aeq_SZ_RES, Aeq_SZ_IW, Aeq_SZ_W, optimization::Aeq_sz>;
using AeqReturnType = MatrixColMaj<optimization::Aeq_rows, optimization::Aeq_cols>;

MatrixColMaj<optimization::Aeq_rows, optimization::Aeq_cols> Aeq_function(OSCData& osc_data) {
    // Map osc data to Column Major for Casadi
    MatrixColMajor<model::nv_size, model::nv_size> mass_matrix = 
        Eigen::Map<MatrixColMajor<model::nv_size, model::nv_size>>(osc_data.mass_matrix.data());
    MatrixColMajor<model::nv_size, optimization::z_size> contact_jacobian = 
        Eigen::Map<MatrixColMajor<model::nv_size, optimization::z_size>>(osc_data.contact_jacobian.data());

    // Create function pointers struct
    CasadiFunctionPointers<AeqSizes> func_ptrs{
        Aeq,
        Aeq_incref,
        Aeq_decref,
        Aeq_alloc_mem,
        Aeq_init_mem,
        Aeq_free_mem
    };

    // Prepare arguments
    std::vector<const double*> args = {
        design_vector.data(),
        mass_matrix.data(),
        osc_data.coriolis_matrix.data(),
        contact_jacobian.data()
    };

    return evaluate_casadi_function<AeqReturnType, AeqSizes>(func_ptrs, args);
}