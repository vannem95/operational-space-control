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


int main(int argc, char** argv) {
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
        std::ofstream file("mass_matrix.csv");
        if (file.is_open()) {
            file << osc_data.mass_matrix.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    {
        std::ofstream file("coriolis_matrix.csv");
        if (file.is_open()) {
            file << osc_data.coriolis_matrix.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    {
        std::ofstream file("contact_jacobian.csv");
        if (file.is_open()) {
            file << osc_data.contact_jacobian.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    {
        std::ofstream file("taskspace_jacobian.csv");
        if (file.is_open()) {
            file << osc_data.taskspace_jacobian.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    {
        std::ofstream file("taskspace_bias.csv");
        if (file.is_open()) {
            file << osc_data.taskspace_bias.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    {
        std::ofstream file("contact_mask.csv");
        if (file.is_open()) {
            file << osc_data.contact_mask.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    {
        std::ofstream file("previous_q.csv");
        if (file.is_open()) {
            file << osc_data.previous_q.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    {
        std::ofstream file("previous_qd.csv");
        if (file.is_open()) {
            file << osc_data.previous_qd.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
            file.close();
        }
    }

    /* Free Memory */
    osc.close();

    return 0;
}