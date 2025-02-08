#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "mujoco/mujoco.h"
#include "Eigen/Dense"

#include "src/unitree_go2/operational_space_controller_.h"

#include "src/unitree_go2/autogen/autogen_functions.h"
#include "src/unitree_go2/autogen/autogen_defines.h"


int main(int argc, char** argv){
    // Real Data:
    OperationalSpaceController osc;
    std::filesystem::path xml_path = "models/unitree_go2/scene_mjx_torque.xml";
    osc.initialize(xml_path);

    Eigen::VectorXd q_init =  Eigen::Map<Eigen::VectorXd>(osc.mj_model->key_qpos, constants::model::nq_size);
    Eigen::VectorXd qd_init =  Eigen::Map<Eigen::VectorXd>(osc.mj_model->key_qvel, constants::model::nv_size);
    Eigen::VectorXd ctrl =  Eigen::Map<Eigen::VectorXd>(osc.mj_model->key_ctrl, constants::model::nu_size);

    // Set initial state:
    osc.mj_data->qpos = q_init.data();
    osc.mj_data->qvel = qd_init.data();
    osc.mj_data->ctrl = ctrl.data();

    // Desired Motor States:
    Eigen::VectorXd q_desired(osc.mj_model->nu);
    q_desired << 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8;
    Eigen::VectorXd qd_desired = Eigen::VectorXd::Zero(q_desired.size());

    mj_forward(osc.mj_model, osc.mj_data);

    Eigen::Matrix<double, constants::model::site_ids_size, 3, Eigen::RowMajor> points = 
        Eigen::Matrix<double, constants::model::site_ids_size, 3, Eigen::RowMajor>::Zero();
    points = Eigen::Map<Eigen::Matrix<double, constants::model::site_ids_size, 3, Eigen::RowMajor>>(
        osc.mj_data->site_xpos
    );
    OSCData osc_data = osc.get_data(points);

    Eigen::VectorXd design_vector = Eigen::VectorXd::Zero(constants::optimization::design_vector_size);
    Eigen::MatrixXd M = Eigen::Map<Eigen::MatrixXd>(osc_data.mass_matrix.data(), constants::model::nv_size, constants::model::nv_size);
    Eigen::VectorXd C = osc_data.coriolis_matrix;
    Eigen::MatrixXd J = Eigen::Map<Eigen::MatrixXd>(osc_data.contact_jacobian.data(), constants::model::nv_size, constants::optimization::z_size);

    // Save matrix to CSV:
    {
        std::ofstream file("M.csv");
        if (file.is_open()) {
            file << M.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    // Save C vector to CSV:
    {
        std::ofstream file("C.csv");
        if (file.is_open()) {
            file << C.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    // Save J matrix to CSV:
    {
        std::ofstream file("J.csv");
        if (file.is_open()) {
            file << J.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    // Allocate Result Object:
    double res0[756];

    // Casadi Work Vector and Memory:
    const double *args[A_eq_function_SZ_ARG];
    double *res[A_eq_function_SZ_RES];
    casadi_int iw[A_eq_function_SZ_IW];
    double w[A_eq_function_SZ_W];

    res[0] = res0;

    A_eq_function_incref();

    // Copy the C Arrays:
    args[0] = design_vector.data();
    args[1] = M.data();
    args[2] = C.data();
    args[3] = J.data();

    // Initialize Memory:
    int mem = A_eq_function_alloc_mem();
    A_eq_function_init_mem(mem);

    // Evaluate the Function:
    A_eq_function(args, res, iw, w, mem);

    std::cout << "Size of res0: " << sizeof(res0) / sizeof(res0[0]) << std::endl;
    
    // Map the Result to Eigen Matrix:
    Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd>(res0, constants::model::nv_size, constants::optimization::design_vector_size);

    std::cout << A << std::endl;

    {
        std::ofstream file("A.csv");
        if (file.is_open()) {
            file << A.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    // Free Memory:
    A_eq_function_free_mem(mem);
    A_eq_function_decref();

    return 0;
}