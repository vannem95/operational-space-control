#include <filesystem>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "GLFW/glfw3.h"

#include "operational-space-control/unitree_go2/aliases.h"
#include "operational-space-control/unitree_go2/containers.h"
#include "operational-space-control/unitree_go2/constants.h"
#include "operational-space-control/unitree_go2/operational_space_controller.h"

using namespace operational_space_controller::aliases;
using namespace operational_space_controller::containers;
using namespace operational_space_controller::constants;
using rules_cc::cc::runfiles::Runfiles;


int main(int argc, char** argv) {
    // Use runfiles to find the path to the model file
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path osc_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/go2.xml");
    
    std::filesystem::path simulation_model_path = 
        runfiles->Rlocation("mujoco-models/models/unitree_go2/scene_go2.xml");

    // Load Simulation Model
    char mj_error[1000];
    mjModel* mj_model = mj_loadXML(simulation_model_path.c_str(), nullptr, mj_error, 1000);
    if (!mj_model) {
        printf("%s\n", mj_error);
        return 1;
    }
    mjData* mj_data = mj_makeData(mj_model);

    // Initialize mj_data:
    mj_data->qpos = mj_model->key_qpos;
    mj_data->qvel = mj_model->key_qvel;
    mj_data->ctrl = mj_model->key_ctrl;
    mj_forward(mj_model, mj_data);

    // Visualization:
    mjvCamera cam;
    mjvPerturb pert;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;

    // Create GLFW window
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "Operational Space Control Example", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Initialize visualization data structures:
    mjv_defaultCamera(&cam);
    mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(mj_model, &scn, 1000);
    mjr_makeContext(mj_model, &con, mjFONTSCALE_150);

    // Framebuffer viewport:
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // Swap OpenGL buffers:
    glfwSwapBuffers(window);

    // Process pendings GUI events:
    glfwPollEvents();

    // Initialize Operational Space Controller
    OperationalSpaceController controller(
        osc_model_path
    );

    Vector<model::nq_size> qpos = Eigen::Map<Vector<model::nq_size>>(mj_data->qpos);
    Vector<model::nv_size> qvel = Eigen::Map<Vector<model::nv_size>>(mj_data->qvel);
    Vector<model::nv_size> qfrc_actuator = Eigen::Map<Vector<model::nv_size>>(mj_data->qfrc_actuator);
    Vector<3> initial_position = qpos(Eigen::seqN(0, 3));

    State initial_state;
    initial_state.motor_position = qpos(Eigen::seqN(7, model::nu_size));
    initial_state.motor_velocity = qvel(Eigen::seqN(6, model::nu_size));
    initial_state.torque_estimate = qfrc_actuator(Eigen::seqN(6, model::nu_size));
    initial_state.body_rotation = qpos(Eigen::seqN(3, 4));
    initial_state.linear_body_velocity = qvel(Eigen::seqN(0, 3));
    initial_state.angular_body_velocity = qvel(Eigen::seqN(3, 3));
    initial_state.contact_mask = Vector<model::contact_site_ids_size>::Constant(1.0);

    TaskspaceTargets taskspace_targets = Matrix<model::site_ids_size, 6>::Zero();

    absl::Status result;
    result.Update(controller.initialize(initial_state));
    result.Update(controller.initialize_optimization());
    ABSL_CHECK(result.ok()) << result.message();

    // Initalize Controller Thread:
    controller.update_taskspace_targets(taskspace_targets);
    result.Update(controller.initialize_thread());
    ABSL_CHECK(result.ok()) << result.message();

    // Control loop
    double visualization_timer = mj_data->time;
    double visualization_start_time = visualization_timer;
    double visualization_interval = 0.01;
    double simulation_time = 20.0;
    auto current_time = mj_data->time;
    while(current_time < simulation_time) {
        current_time = mj_data->time;
        visualization_timer = current_time - visualization_start_time;

        // Update State Struct:
        Vector<model::nq_size> qpos = Eigen::Map<Vector<model::nq_size>>(mj_data->qpos);
        Vector<model::nv_size> qvel = Eigen::Map<Vector<model::nv_size>>(mj_data->qvel);
        Vector<model::nv_size> qfrc_actuator = Eigen::Map<Vector<model::nv_size>>(mj_data->qfrc_actuator);

        State state;
        state.motor_position = qpos(Eigen::seqN(7, model::nu_size));
        state.motor_velocity = qvel(Eigen::seqN(6, model::nu_size));
        state.torque_estimate = qfrc_actuator(Eigen::seqN(6, model::nu_size));
        state.body_rotation = qpos(Eigen::seqN(3, 4));
        state.linear_body_velocity = qvel(Eigen::seqN(0, 3));
        state.angular_body_velocity = qvel(Eigen::seqN(3, 3));
        state.contact_mask = Vector<model::contact_site_ids_size>::Constant(1.0);

        controller.update_state(state);
        
        // Update Taskspace Targets:
        TaskspaceTargets taskspace_targets = TaskspaceTargets::Zero();

        // Position and Velocity:
        Eigen::Quaternion<double> body_rotation = Eigen::Quaternion<double>(state.body_rotation(0), state.body_rotation(1), state.body_rotation(2), state.body_rotation(3));
        Vector<3> body_position = qpos(Eigen::seqN(0, 3));
        Vector<3> position_error = initial_position - body_position;
        Vector<3> velocity_error = Vector<3>::Zero() - state.linear_body_velocity;
        Vector<3> rotation_error = (Eigen::Quaternion<double>(1, 0, 0, 0) * body_rotation.conjugate()).vec();
        Vector<3> angular_velocity_error = Vector<3>::Zero() - state.angular_body_velocity;
        Vector<3> linear_control = 150.0 * (position_error) + 25.0 * (velocity_error);
        Vector<3> angular_control = 50.0 * (rotation_error) + 10.0 * (angular_velocity_error);
        Eigen::Vector<double, 6> cmd {linear_control(0), linear_control(1), linear_control(2), angular_control(0), angular_control(1), angular_control(2)};
        taskspace_targets.row(0) = cmd;

        controller.update_taskspace_targets(taskspace_targets);

        // Get Torque Command:
        Vector<model::nu_size> torque_command = controller.get_torque_command();

        // Update Mujoco Data and Step:
        mj_data->ctrl = torque_command.data();
        mj_step(mj_model, mj_data);

        if(visualization_timer > visualization_interval) {
            visualization_start_time = mj_data->time;

            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    // Clean up visualization:
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Stop Threads and Clean up:
    result.Update(controller.stop_thread());
    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);
    ABSL_CHECK(result.ok()) << result.message();

    return 0;
}