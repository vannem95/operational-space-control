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
    // Real Data:
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

    Eigen::VectorXd design_vector = Eigen::VectorXd::Zero(42);
    Eigen::MatrixXd M = Eigen::Map<Eigen::MatrixXd>(osc_data.mass_matrix.data(), 18, 18);
    Eigen::VectorXd C = osc_data.coriolis_matrix;
    Eigen::MatrixXd J = Eigen::Map<Eigen::MatrixXd>(osc_data.contact_jacobian.data(), 18, 12);

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

    // Dummy Data: Zeros
    // Eigen::VectorXd design_vector = Eigen::VectorXd::Zero(42);
    // Eigen::MatrixXd M = Eigen::MatrixXd::Zero(18, 18);
    // Eigen::VectorXd C = Eigen::VectorXd::Zero(18);
    // Eigen::MatrixXd J = Eigen::MatrixXd::Zero(18, 12);

    // // Dummy Data: Ones
    // Eigen::VectorXd design_vector = Eigen::VectorXd::Zero(42);
    // Eigen::Matrix<double, 18, 18> M = Eigen::Matrix<double, 18, 18>::Identity();
    // Eigen::VectorXd C = Eigen::VectorXd::Ones(18);
    // Eigen::Matrix<double, 18, 12> J = Eigen::Matrix<double, 18, 12>::Identity();

    // Allocate Result Object:
    double res0[756];

    // Casadi Work Vector and Memory:
    // const double *args[4];
    // double *res[1] = {res0};
    // casadi_int iw[A_eq_function_SZ_IW];
    // double w[A_eq_function_SZ_W];

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
    Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd>(res0, 18, 42);

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