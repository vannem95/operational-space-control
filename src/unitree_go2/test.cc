#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

#include "mujoco/mujoco.h"
#include "Eigen/Dense"

#include "GLFW/glfw3.h"

#include "src/unitree_go2/operational_space_controller.h"

// Camera and Scene Setup:
mjvCamera cam;                      // abstract camera
mjvPerturb pert;                    // perturbation object
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


int main(int argc, char** argv) {
    OperationalSpaceController osc;
    std::filesystem::path xml_path = "models/unitree_go2/scene_mjx.xml";
    osc.initialize(xml_path);

    Eigen::VectorXd q_init =  Eigen::Map<Eigen::VectorXd>(osc.model->key_qpos, osc.model->nq);
    Eigen::VectorXd qd_init =  Eigen::Map<Eigen::VectorXd>(osc.model->key_qvel, osc.model->nv);
    Eigen::VectorXd ctrl =  Eigen::Map<Eigen::VectorXd>(osc.model->key_ctrl, osc.model->nu);

    // Visualization:
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
   //  mjv_makeScene(const mjModel* model, mjvScene* scn, int maxgeom);
    mjv_makeScene(osc.model, &scn, 1000);                     // space for 1000 objects
    mjr_makeContext(osc.model, &con, mjFONTSCALE_100);     // model-specific context

    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    while(osc.data->time < 5) {

        // PD Control:
        Eigen::VectorXd qpos = Eigen::Map<Eigen::VectorXd>(osc.data->qpos, osc.model->nq);
        Eigen::VectorXd qvel = Eigen::Map<Eigen::VectorXd>(osc.data->qvel, osc.model->nv);

        mj_step(osc.model, osc.data);

        Eigen::MatrixXd points = Eigen::MatrixXd::Zero(5, 3);
        auto data = osc.data;
        typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MapMatrix;
        points = MapMatrix(data->site_xpos, 5, 3);
        std::cout << points << std::endl;

        auto osc_data = osc.get_data(points);

        // update scene and render
        mjv_updateScene(osc.model, osc.data, &opt, &pert, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    /* Free Memory */
    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free model and data, deactivate
    osc.close();

    return 0;
}