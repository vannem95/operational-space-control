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

#include "GLFW/glfw3.h"

#include "src/unitree_go2/operational_space_controller.h"
#include "src/utilities.h"

// Camera and Scene Setup:
mjvCamera cam;                      // abstract camera
mjvPerturb pert;                    // perturbation object
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


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

    // Visualization:
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "Simulation Test", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
    mjv_makeScene(osc.model, &scn, 1000);                     // space for 1000 objects
    mjr_makeContext(osc.model, &con, mjFONTSCALE_100);     // model-specific context

    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    double kp = 45.0;
    double kd = 5.0;

    auto start = std::chrono::high_resolution_clock::now();
    while(osc.data->time < 5) {
        auto loop_start = std::chrono::high_resolution_clock::now();

        // PD Control:
        int num_iterations = 20;
        for(const int i : std::views::iota(0, num_iterations)) {
            std::ignore = i;
            Eigen::VectorXd qpos = Eigen::Map<Eigen::VectorXd>(osc.data->qpos, osc.model->nq)(Eigen::seq(7, Eigen::placeholders::last));
            Eigen::VectorXd qvel = Eigen::Map<Eigen::VectorXd>(osc.data->qvel, osc.model->nv)(Eigen::seq(6, Eigen::placeholders::last));
            Eigen::VectorXd ctrl = kp * (q_desired - qpos) + kd * (qd_desired - qvel);
            osc.data->ctrl = ctrl.data();

            mj_step(osc.model, osc.data);
        }

        Matrix points = Matrix::Zero(5, 3);
        auto data = osc.data;
        points = MapMatrix(data->site_xpos, 5, 3);
        auto osc_data = osc.get_data(points);

        // update scene and render
        mjv_updateScene(osc.model, osc.data, &opt, &pert, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

        // Simulation Loop Run Realtime:
        auto loop_end = std::chrono::high_resolution_clock::now();
        auto loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_time = std::chrono::milliseconds(40) - loop_duration;
        if (sleep_time < std::chrono::milliseconds(0))
            sleep_time = std::chrono::milliseconds(0);
        std::this_thread::sleep_for(sleep_time);
    }

    // Stop the clock:
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::chrono::seconds sec =  std::chrono::duration_cast<std::chrono::seconds>(duration);
    std::cout << "Wall Time: " << sec.count() << " seconds" << std::endl;
    std::cout << "Simulation Time: " << osc.data->time << " seconds" << std::endl;

    /* Free Memory */
    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free model and data, deactivate
    osc.close();

    return 0;
}