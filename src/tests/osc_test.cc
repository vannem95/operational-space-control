#include <iostream>
#include <filesystem>

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "GLFW/glfw3.h"

#include "src/unitree_go2/operational_space_controller.h"


char error[1000];


int main(int argc, char** argv) {
    // Simulation model and data:
    mjModel* mj_model = mj_loadXML("models/unitree_go2/scene_mjx.xml", nullptr, error, 1000);
    if( !mj_model ) {
        printf("%s\n", error);
        return 1;
    }
    mjData* mj_data = mj_makeData(mj_model);

    // Initialize mj_data:
    mj_data->qpos = mj_model->key_qpos.data();
    mj_data->qvel = mj_model->key_qvel.data();

    mj_forward(mj_model, mj_data);

    Eigen::VectorXd qpos = Eigen::Map<Eigen::VectorXd>(mj_data->qpos, mj_model->nq);
    Eigen::VectorXd qvel = Eigen::Map<Eigen::VectorXd>(mj_data->qvel, mj_model->nv);

    Eigen::VectorXd body_rotation = qpos(Eigen::seqN(3, 4));
    Eigen::VectorXd joint_pos = qpos(Eigen::seqN(7, mj_model->nu));

    Eigen::VectorXd body_velocity = qvel(Eigen::seqN(3, 3));
    Eigen::VectorXd joint_vel = qvel(Eigen::seqN(6, mj_model->nu));

    int num_contacts = 4
    Eigen::VectorXd contact_mask = Eigen::VectorXd::Zero(num_contacts);
    double contact_threshold = 1e-3;
    for(int i = 0; i < num_contacts; i++) {
        auto contact = mj_data->contact[i];
        contact_mask(i) = contact.dist < contact_threshold;
    }

    // Required initial arguments for OSC:
    State initial_state;
    initial_state.motor_position = joint_pos;
    initial_state.motor_velocity = joint_vel
    initial_state.motor_acceleration = Eigen::VectorXd::Zero(model::nu_size);
    initial_state.torque_estimate = Eigen::VectorXd::Zero(model::nu_size);
    initial_state.body_rotation = body_rotation;
    initial_state.body_velocity = body_velocity;
    initial_state.body_acceleration = Eigen::VectorXd::Zero(3);
    initial_state.contact_mask = contact_mask;
    int control_rate = 2;
    
    // Get model path:
    std::filesystem::path model_path = "models/unitree_go2/go2_mjx.xml";

    // Initialize OSC:
    OperationalSpaceController osc(initial_state, control_rate);
    osc.initialize(model_path);

    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);

    return 0;
}
