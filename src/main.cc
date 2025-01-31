#include <iostream>

#include "mujoco/mujoco.h"
#include "GLFW/glfw3.h"
#include "Eigen/Dense"
#include "string.h"

char error[1000];
mjModel* m;
mjData* d;

mjvCamera cam;                      // abstract camera
mjvPerturb pert;                    // perturbation object
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

int main(void)
{
    // load model from file and check for errors
    m = mj_loadXML("models/unitree_go2/scene_mjx.xml", nullptr, error, 1000);
    if( !m )
    {
        printf("%s\n", error);
        return 1;
    }

    // make data corresponding to model
    d = mj_makeData(m);

    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
   //  mjv_makeScene(const mjModel* m, mjvScene* scn, int maxgeom);
    mjv_makeScene(m, &scn, 1000);                     // space for 1000 objects
    mjr_makeContext(m, &con, mjFONTSCALE_100);     // model-specific context


    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // run simulation for 10 seconds
    auto i = 0;
    while(d->time< 5)
    {
        mj_step(m, d);

        if(i == 0) {
            auto mass_matrix = Eigen::Map<Eigen::MatrixXd>(d->qM, m->nv, m->nv);;
            std::cout << mass_matrix << std::endl;

            Eigen::MatrixXd jacp = Eigen::MatrixXd::Zero(3, m->nv);
            Eigen::MatrixXd jacr = Eigen::MatrixXd::Zero(3, m->nv);
            Eigen::MatrixXd points = Eigen::MatrixXd::Zero(3,3);
            points << 0, 0, 0,
                      1, 0, 0,
                      2, 0, 0;
            int body_id = 1;

            // Calculate Jacobian:
            Eigen::VectorXd point = Eigen::Map<Eigen::VectorXd>(points.row(0).data(), 3);

            mj_jac(m, d, jacp.data(), jacr.data(), points.row(1).data(), body_id);

            std::cout << jacp << std::endl;
            std::cout << jacr << std::endl;

            // Calculate Taskspace Bias Acceleration:
            Eigen::VectorXd joint_velocity = Eigen::Map<Eigen::VectorXd>(d->qvel, m->nv);
            Eigen::MatrixXd taskspace_bias = Eigen::MatrixXd::Zero(12, 1);
            
            int num_body_ids = 2;
            Eigen::MatrixXd jacobian_dot_translation = Eigen::MatrixXd::Zero(6, 18);
            Eigen::MatrixXd jacobian_dot_rotation = Eigen::MatrixXd::Zero(6, 18);
            Eigen::MatrixXd jacobian_dot = Eigen::MatrixXd::Zero(12, 18);

            jacobian_dot.block<6, 18>(0, 0) = jacobian_dot_translation;
            jacobian_dot.block<6, 18>(6, 0) = jacobian_dot_rotation;

            std::cout << joint_velocity << std::endl;

            taskspace_bias = jacobian_dot * joint_velocity;
            std::cout << taskspace_bias << std::endl;

        }
        i++;

        // Reshape leading axis -> num_body_ids x 6
        // taskspace_bias = taskspace_bias.transpose().reshaped(num_body_ids, 6);

        // std::cout << taskspace_bias << std::endl;

        // Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 3);
        // for(int i = 0; i < 2; i++) {
        //     Eigen::MatrixXd B(3, 3);
        //     B << 1, 2, 3,
        //          4, 5, 6,
        //          7, 8, 9;
        //     int row_idx = i * 3;
        //     int col_idx = 0;
        //     A.block<3, 3>(row_idx, col_idx) = B;
        // }
        // std::cout << A << std::endl;

        // update scene and render
        mjv_updateScene(m, d, &opt, &pert, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);

    return 0;
}