#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <ranges>
#include <thread>
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
    {
        // Allocated input/output buffers and work vectors:
        const double *arg[b_eq_function_SZ_ARG];
        double *res[b_eq_function_SZ_RES];
        casadi_int iw[b_eq_function_SZ_IW];
        double w[b_eq_function_SZ_W];

        // Initialize input and outputs:
        const int num_contacts = 4;
        int design_vector_size = osc.model->nv + osc.model->nu + 3 * num_contacts;
        const Eigen::VectorXd q = Eigen::VectorXd::Zero(design_vector_size);
        double res0;

        b_eq_function_incref();

        /* Evaluate Function: */
        arg[0] = q.data();
        arg[1] = osc_data.mass_matrix.data();;
        arg[2] = osc_data.coriolis_matrix.data();;
        arg[3] = osc_data.contact_jacobian.data();
        res[0] = &res0;

        int mem = b_eq_function_checkout();

        b_eq_function(arg, res, iw, w, mem);

        b_eq_function_release(mem);

        Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(&res0, osc.model->nv);
        std::cout << "B Matrix: \n" << b << std::endl;

        b_eq_function_decref();
    }
    // Now Try A_eq_function:
    {
        // Allocated input/output buffers and work vectors:
        const double *arg[A_eq_function_SZ_ARG];
        double *res[A_eq_function_SZ_RES];
        casadi_int iw[A_eq_function_SZ_IW];
        double w[A_eq_function_SZ_W];

        // Initialize input and outputs:
        const int num_contacts = 4;
        int design_vector_size = osc.model->nv + osc.model->nu + 3 * num_contacts;
        const Eigen::VectorXd q = Eigen::VectorXd::Zero(design_vector_size);
        double res0;

        A_eq_function_incref();

        /* Evaluate Function: */
        arg[0] = q.data();
        arg[1] = osc_data.mass_matrix.data();;
        arg[2] = osc_data.coriolis_matrix.data();;
        arg[3] = osc_data.contact_jacobian.data();
        res[0] = &res0;

        int mem = A_eq_function_checkout();

        A_eq_function(arg, res, iw, w, mem);

        A_eq_function_release(mem);

        Eigen::VectorXd A = Eigen::Map<Eigen::VectorXd>(&res0, 18 * 42);
        std::cout << "A Matrix: \n" << A << std::endl;

        A_eq_function_decref();
    }

    return 0;
}