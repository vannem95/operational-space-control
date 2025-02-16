#include <iostream>
#include <filesystem>

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "GLFW/glfw3.h"

#include "src/unitree_go2/operational_space_controller.h"

// For debugging:
#include "src/unitree_go2/autogen/autogen_functions.h"


char error[1000];

mjvCamera cam;                      // abstract camera
mjvPerturb pert;                    // perturbation object
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


int main(int argc, char** argv) {
    // Simulation model and data:
    mjModel* mj_model = mj_loadXML("models/unitree_go2/scene_mjx_torque.xml", nullptr, error, 1000);
    if( !mj_model ) {
        printf("%s\n", error);
        return 1;
    }
    mjData* mj_data = mj_makeData(mj_model);

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
    mjv_makeScene(mj_model, &scn, 1000);                     // space for 1000 objects
    mjr_makeContext(mj_model, &con, mjFONTSCALE_100);     // model-specific context


    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // Initialize mj_data:
    mj_data->qpos = mj_model->key_qpos;
    mj_data->qvel = mj_model->key_qvel;

    mj_forward(mj_model, mj_data);

    // site_xpos matches the site_id mappings
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> site_xpos = 
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(mj_data->site_xpos, mj_model->nsite, 3);
    std::cout << site_xpos << std::endl;

    // Compare to site_id mappings:
    std::vector<std::string> sites;
    std::vector<std::string> bodies;
    std::vector<std::string> noncontact_sites;
    std::vector<std::string> contact_sites;
    std::vector<int> site_ids;
    std::vector<int> noncontact_site_ids;
    std::vector<int> contact_site_ids;
    std::vector<int> body_ids;

    for(const std::string_view& site : model::site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model, mjOBJ_SITE, site_str.data());
        assert(id != -1 && "Site not found in model.");
        sites.push_back(site_str);
        site_ids.push_back(id);
    }
    for(const std::string_view& site : model::noncontact_site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model, mjOBJ_SITE, site_str.data());
        assert(id != -1 && "Site not found in model.");
        noncontact_sites.push_back(site_str);
        noncontact_site_ids.push_back(id);
    }
    for(const std::string_view& site : model::contact_site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model, mjOBJ_SITE, site_str.data());
        assert(id != -1 && "Site not found in model.");
        contact_sites.push_back(site_str);
        contact_site_ids.push_back(id);
    }

    std::cout << "Sites: " << std::endl;
    for(int i = 0; i < sites.size(); i++) {
        std::cout << sites[i] << " : " << site_ids[i] << std::endl;
    }


    Eigen::VectorXd qpos = Eigen::Map<Eigen::VectorXd>(mj_data->qpos, mj_model->nq);
    Eigen::VectorXd qvel = Eigen::Map<Eigen::VectorXd>(mj_data->qvel, mj_model->nv);

    Eigen::VectorXd body_rotation = qpos(Eigen::seqN(3, 4));
    Eigen::VectorXd joint_pos = qpos(Eigen::seqN(7, mj_model->nu));

    Eigen::VectorXd body_velocity = qvel(Eigen::seqN(3, 3));
    Eigen::VectorXd joint_vel = qvel(Eigen::seqN(6, mj_model->nu));

    int num_contacts = 4;
    Eigen::VectorXd contact_mask = Eigen::VectorXd::Zero(num_contacts);
    double contact_threshold = 1e-3;
    for(int i = 0; i < num_contacts; i++) {
        auto contact = mj_data->contact[i];
        contact_mask(i) = contact.dist < contact_threshold;
    }

    // Required initial arguments for OSC:
    State initial_state;
    initial_state.motor_position = joint_pos;
    initial_state.motor_velocity = joint_vel;
    initial_state.motor_acceleration = Eigen::VectorXd::Zero(model::nu_size);
    initial_state.torque_estimate = Eigen::VectorXd::Zero(model::nu_size);
    initial_state.body_rotation = body_rotation;
    initial_state.body_velocity = body_velocity;
    initial_state.body_acceleration = Eigen::VectorXd::Zero(3);
    initial_state.contact_mask = contact_mask;
    int control_rate = 2;
    
    // Get model path:
    std::filesystem::path model_path = "models/unitree_go2/go2_mjx_torque.xml";

    // Initialize OSC:
    OperationalSpaceController osc(initial_state, control_rate);
    osc.initialize(model_path);

    // Update Taskspace Targets:
    Eigen::Matrix<double, model::site_ids_size, 6, Eigen::RowMajor> taskspace_targets = Eigen::Matrix<double, model::site_ids_size, 6, Eigen::RowMajor>::Zero();
    taskspace_targets << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    osc.update_taskspace_targets(taskspace_targets);

    // Initialize 
    osc.initialize_control_thread();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Loop:
    int visualize_iter = 0;
    while(mj_data->time < 5) {
        // Step simulation model:
        mj_step(mj_model, mj_data);

        // Get data to make next_state:
        Eigen::VectorXd qpos = Eigen::Map<Eigen::VectorXd>(mj_data->qpos, mj_model->nq);
        Eigen::VectorXd qvel = Eigen::Map<Eigen::VectorXd>(mj_data->qvel, mj_model->nv);

        Eigen::VectorXd body_rotation = qpos(Eigen::seqN(3, 4));
        Eigen::VectorXd joint_pos = qpos(Eigen::seqN(7, mj_model->nu));

        Eigen::VectorXd body_velocity = qvel(Eigen::seqN(3, 3));
        Eigen::VectorXd joint_vel = qvel(Eigen::seqN(6, mj_model->nu));

        int num_contacts = 4;
        Eigen::VectorXd contact_mask = Eigen::VectorXd::Zero(num_contacts);
        double contact_threshold = 1e-3;
        for(int i = 0; i < num_contacts; i++) {
            auto contact = mj_data->contact[i];
            contact_mask(i) = contact.dist < contact_threshold;
        }

        // Create state struct:
        State next_state;
        next_state.motor_position = joint_pos;
        next_state.motor_velocity = joint_vel;
        next_state.motor_acceleration = Eigen::VectorXd::Zero(model::nu_size);
        next_state.torque_estimate = Eigen::VectorXd::Zero(model::nu_size);
        next_state.body_rotation = body_rotation;
        next_state.body_velocity = body_velocity;
        next_state.body_acceleration = Eigen::VectorXd::Zero(3);
        next_state.contact_mask = contact_mask;

        // Update OSC state:
        osc.update_state(next_state);

        if(visualize_iter % 10 == 0) {
            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            // swap OpenGL buffers (blocking call due to v-sync)
            glfwSwapBuffers(window);

            // process pending GUI events, call GLFW callbacks
            glfwPollEvents();
        }
        visualize_iter++;
    }

    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Stop thread and free resources:
    osc.stop_control_thread();
    osc.close();
    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);

    return 0;
}
