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
    Eigen::VectorXd q = Eigen::VectorXd::Zero(design_vector_size);

    // auto outpattern = A_eq_function_sparsity_out(0);
    // Eigen::Vector<casadi_int, 42> outpattern_eigen;
    // std::cout << outpattern_eigen[2] << std::endl;
    // for (int i = 0; i < 42; i++){
    //     outpattern_eigen(i) = outpattern[i+3];
    // }
    // std::cout << outpattern_eigen << std::endl;

    // Allocated input/output buffers and work vectors:
    const double *args[A_eq_function_SZ_ARG];
    double *res[A_eq_function_SZ_RES];
    casadi_int iw[A_eq_function_SZ_IW];
    double w[A_eq_function_SZ_W];

    double res0;

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
    res[0] = &res0;

    int mem = A_eq_function_alloc_mem();
    A_eq_function_init_mem(mem);

    A_eq_function(args, res, iw, w, mem);

    Matrix A = Matrix::Zero(18, 42);
    A = MapMatrix(&res0, 18, 42);

    {
        std::ofstream file("A_matrix_sparse.csv");
        if (file.is_open()) {
            file << A.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    A_eq_function_free_mem(mem);

    A_eq_function_decref();

    // std::cout << osc_data.mass_matrix.data() << std::endl;


    return 0;
}