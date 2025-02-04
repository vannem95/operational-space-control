#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include <Eigen/SparseCore>

#include "src/unitree_go2/operational_space_controller.h"
#include "src/utilities.h"
#include "src/unitree_go2/autogen/equality_constraint_function.h"


int main(int argc, char** argv){
    OperationalSpaceController osc;
    std::filesystem::path xml_path = "models/unitree_go2/scene_mjx_torque.xml";
    osc.initialize(xml_path);

    Eigen::VectorXd q_init =  Eigen::Map<Eigen::VectorXd>(osc.model->key_qpos, osc.model->nq);
    Eigen::VectorXd qd_init =  Eigen::Map<Eigen::VectorXd>(osc.model->key_qvel, osc.model->nv);
    Eigen::VectorXd ctrl =  Eigen::Map<Eigen::VectorXd>(osc.model->key_ctrl, osc.model->nu);

    // Set initial state:
    osc.data->qpos = q_init.data();
    osc.data->qvel = qd_init.data();
    osc.data->ctrl = ctrl.data();

    // Desired Motor States:
    Eigen::VectorXd q_desired(osc.model->nu);
    q_desired << 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8;
    Eigen::VectorXd qd_desired = Eigen::VectorXd::Zero(q_desired.size());

    mj_forward(osc.model, osc.data);

    Matrix points = Matrix::Zero(5, 3);
    points = MapMatrix(osc.data->site_xpos, 5, 3);
    OSCData osc_data = osc.get_data(points);

    const int num_contacts = 4;
    int design_vector_size = osc.model->nv + osc.model->nu + 3 * num_contacts;
    Eigen::VectorXd q = Eigen::VectorXd::Zero(42);


    // Allocated input/output buffers and work vectors:
    const double *args[A_eq_function_SZ_ARG];
    double *res[A_eq_function_SZ_RES];
    casadi_int iw[A_eq_function_SZ_IW];
    double w[A_eq_function_SZ_W];

    double res0[756];

    A_eq_function_incref();

    // Copy to C Arrays:
    double *design_vector = q.data();
    double *mass_matrix = osc_data.mass_matrix.data();
    double *coriolis_matrix = osc_data.coriolis_matrix.data();
    double *contact_jacobian = osc_data.contact_jacobian.data();

    args[0] = design_vector;
    args[1] = mass_matrix;
    args[2] = coriolis_matrix;
    args[3] = contact_jacobian;
    res[0] = res0;

    // 
    casadi_int n_in = A_eq_function_n_in();
    casadi_int n_out = A_eq_function_n_out();

    std::cout << n_out << std::endl;

    int mem = A_eq_function_alloc_mem();
    A_eq_function_init_mem(mem);

    A_eq_function(args, res, iw, w, mem);

    casadi_int i;
    std::vector<double> array;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    for(i=0; i<n_in + n_out; ++i){
        // Retrieve the sparsity pattern - CasADi uses column compressed storage (CCS)
        const casadi_int *sp_i;
        if (i<n_in) {
            printf("Input %lld\n", i);
            sp_i = A_eq_function_sparsity_in(i);
        } 
        else {
            printf("Output %lld\n", i-n_in);
            sp_i = A_eq_function_sparsity_out(i-n_in);
        }
        if (sp_i==0) 
            return 1;
        casadi_int nrow = *sp_i++; /* Number of rows */
        casadi_int ncol = *sp_i++; /* Number of columns */
        const casadi_int *colind = sp_i; /* Column offsets */
        const casadi_int *row = sp_i + ncol+1; /* Row nonzero */
        casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

        casadi_int rr, cc, el;
        int iter = 0;
        for(cc=0; cc<ncol; ++cc){                           /* loop over columns */
            for(el=colind[cc]; el<colind[cc+1]; ++el){      /* loop over the nonzeros entries of the column */
                rr = row[el];                               /* Get the row */
                row_idx.push_back(rr);
                col_idx.push_back(cc);
                array.push_back(res0[iter]);
                iter++;
            }
        }
        std::cout << iter << std::endl;
    }
    
    Matrix A = MapMatrix(res0, 18, 42);
    // Eigen::MatrixXd A_col_major = Eigen::Map<Eigen::MatrixXd>(res0, 18, 42);
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(A_col_major.data(), 18, 42);

    // int iter = 0;
    // for(int i = 0; i < 42; i++){
    //     for(int j = 0; j < 18; j++){
    //         A(j, i) = res0[iter];
    //         iter++;
    //     }
    // }

    // Eigen::MatrixXd A = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(array.data(), 18, 42);

    // std::cout << res0[23] << std::endl;
    // std::cout << array[23] << std::endl;

    // for (int i = 0; i < 18*42; i++){
    //     std::cout << res0[i] << std::endl;
    // }

    // Eigen::MatrixXd A = Eigen::MatrixXd::Zero(18, 42);
    // for(int i = 0; i < 18 * 42; i++){
    //     A.data()[i] = res0[i];
    // }
    // A = MapMatrix(res0, 18, 42);

    // std::cout << A.data()[23] << std::endl;

    {
        std::ofstream file("A.csv");
        if (file.is_open()) {
            file << A.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    A_eq_function_free_mem(mem);

    A_eq_function_decref();

    return 0;
}