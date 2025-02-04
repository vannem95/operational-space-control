#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "mujoco/mujoco.h"
#include "Eigen/Dense"

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
    OSCData osc_data_b = osc_data;

    const int num_contacts = 4;
    int design_vector_size = osc.model->nv + osc.model->nu + 3 * num_contacts;
    Eigen::VectorXd q = Eigen::VectorXd::Zero(design_vector_size);
    
    // Allocated input/output buffers and work vectors:
    const double *args[b_eq_function_SZ_ARG];
    double *res[b_eq_function_SZ_RES];
    casadi_int iw[b_eq_function_SZ_IW];
    double w[b_eq_function_SZ_W];

    double res0;

    b_eq_function_incref();

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

    int mem = b_eq_function_checkout();

    b_eq_function(args, res, iw, w, mem);

    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(&res0, 18);
    std::cout << "b Vector: \n" << b << std::endl;

    b_eq_function_release(mem);

    b_eq_function_decref();

    {
        std::ofstream file("b_matrix.csv");
        if (file.is_open()) {
            file << b.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }


    return 0;
}