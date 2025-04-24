#include <filesystem>
#include <cmath>
#include <fstream>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "GLFW/glfw3.h"

#include "operational-space-control/walter_sr/aliases.h"
#include "operational-space-control/walter_sr/containers.h"
#include "operational-space-control/walter_sr/constants.h"
#include "operational-space-control/walter_sr/operational_space_controller.h"

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
        runfiles->Rlocation("mujoco-models/models/walter_sr/WaLTER_Senior.xml");
    
    std::filesystem::path simulation_model_path = 
        runfiles->Rlocation("mujoco-models/models/walter_sr/scene_walter_sr.xml");

    // Load Simulation Model
    char mj_error[1000];
    mjModel* mj_model = mj_loadXML(simulation_model_path.c_str(), nullptr, mj_error, 1000);
    if (!mj_model) {
        printf("%s\n", mj_error);
        return 1;
    }
    mjData* mj_data = mj_makeData(mj_model);

    // Reset Data to match Keyframe 2
    mj_resetDataKeyframe(mj_model, mj_data, 0);


    // Initialize mj_data:
    // mj_data->qpos = mj_model->key_qpos;
    // mj_data->qvel = mj_model->key_qvel;
    // mj_data->ctrl = mj_model->key_ctrl;
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
    //===================================================
    //                  edits -- print qpos to see whats going on
    //===================================================
    Vector<3> initial_position = qpos(Eigen::seqN(0, 3));

    Eigen::Matrix<double, model::site_ids_size, 3> site_data;
    Eigen::Matrix<double, model::site_ids_size, 3> initial_site_data;

    Eigen::Vector<double, model::contact_site_ids_size> contact_check;
    double wheel_contact_check_height = 0.0615;

    Eigen::Matrix<double, model::site_ids_size, 9> site_rotational_data;


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
    double simulation_time = 40.0;
    auto current_time = mj_data->time;

    // to get points / site position --> we need to build site-ids using sites
    std::vector<std::string> sites;
    std::vector<int> site_ids;

    // record shin site to plot
    std::vector<double> target_tl_shin_data;
    std::vector<double> tl_shin_data;

    for(const std::string_view& site : model::site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model, mjOBJ_SITE, site_str.data());
        assert(id != -1 && "Site not found in model.");
        sites.push_back(site_str);
        site_ids.push_back(id);
    }
    initial_site_data = Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->site_xpos)(site_ids, Eigen::placeholders::all);

    while(current_time < simulation_time) {
        current_time = mj_data->time;
        visualization_timer = current_time - visualization_start_time;

        // Update State Struct:
        Vector<model::nq_size> qpos = Eigen::Map<Vector<model::nq_size>>(mj_data->qpos);
        Vector<model::nv_size> qvel = Eigen::Map<Vector<model::nv_size>>(mj_data->qvel);
        Vector<model::nv_size> qfrc_actuator = Eigen::Map<Vector<model::nv_size>>(mj_data->qfrc_actuator);

        // Eigen::Matrix<double, model::site_ids_size, 3> site_data;
        site_data = Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->site_xpos)(site_ids, Eigen::placeholders::all);
        site_rotational_data = Eigen::Map<Matrix<model::site_ids_size, 9>>(mj_data->site_xmat)(site_ids, Eigen::placeholders::all);

        //===================================================
        //                  print qpos
        //===================================================                
        std::cout << "site data: " << site_data << std::endl;
        // std::cout << "site rotational data: " << site_rotational_data << std::endl;
        // std::cout << "initial_site_data data: " << initial_site_data << std::endl;


        //===================================================
        // find contact mask based on dist between wheel and ground
        //===================================================
        contact_check = {(site_data(5,2)<wheel_contact_check_height),
            (site_data(6,2)<wheel_contact_check_height),
            (site_data(7,2)<wheel_contact_check_height),
            (site_data(8,2)<wheel_contact_check_height),
            (site_data(9,2)<wheel_contact_check_height),
            (site_data(10,2)<wheel_contact_check_height),
            (site_data(11,2)<wheel_contact_check_height),
            (site_data(12,2)<wheel_contact_check_height)};
        // std::cout << "contact_check data: " << contact_check << std::endl;
        // std::cout << "contact_mask data: " << state.contact_mask << std::endl;
        
    


        State state;
        state.motor_position = qpos(Eigen::seqN(7, model::nu_size));
        state.motor_velocity = qvel(Eigen::seqN(6, model::nu_size));
        state.torque_estimate = qfrc_actuator(Eigen::seqN(6, model::nu_size));
        state.body_rotation = qpos(Eigen::seqN(3, 4));
        state.linear_body_velocity = qvel(Eigen::seqN(0, 3));
        state.angular_body_velocity = qvel(Eigen::seqN(3, 3));
        // state.contact_mask = Vector<model::contact_site_ids_size>::Constant(0.0);
        state.contact_mask = contact_check;



        controller.update_state(state);
        
        // Update Taskspace Targets:
        TaskspaceTargets taskspace_targets = TaskspaceTargets::Zero();

        // Sinusoidal Position and Velocity Tracking:
        double amplitude = 0.04;
        double frequency = 0.1;

        // ------------------------------------------------------------------
        //       z-axis
        // ------------------------------------------------------------------
        // targets
        Vector<3> tl_position_target = Vector<3>(
            initial_site_data(1,0)+1.0, initial_site_data(1,1), initial_site_data(1,2)
        );
        Vector<3> tr_position_target = Vector<3>(
            initial_site_data(2,0)+1.0, initial_site_data(2,1), initial_site_data(2,2)
        );
        Vector<3> hl_position_target = Vector<3>(
            initial_site_data(3,0)+1.0, initial_site_data(3,1), initial_site_data(3,2)
        );
        Vector<3> hr_position_target = Vector<3>(
            initial_site_data(4,0)+1.0, initial_site_data(4,1), initial_site_data(4,2)
        );


        // record target and actual (torso - left) shin site data
        target_tl_shin_data.push_back(tl_position_target(2));
        tl_shin_data.push_back(site_data(1,2));
        
        // Vector<3> velocity_target = Vector<3>(
        //     // sine wave in x-axis --> front/back
        //     // 2.0 * M_PI * amplitude * frequency * std::cos(2.0 * M_PI * frequency * current_time),0.0, 0.0
        //     0.0,0.0,0.0
        // );
        // Eigen::Quaternion<double> body_rotation = Eigen::Quaternion<double>(state.body_rotation(0), state.body_rotation(1), state.body_rotation(2), state.body_rotation(3));
        Vector<3> tl_body_position = site_data(1,Eigen::seqN(0, 3));
        Vector<3> tr_body_position = site_data(2,Eigen::seqN(0, 3));
        Vector<3> hl_body_position = site_data(3,Eigen::seqN(0, 3));
        Vector<3> hr_body_position = site_data(4,Eigen::seqN(0, 3));
        Vector<3> tl_position_error = (tl_position_target - tl_body_position); 
        Vector<3> tr_position_error = (tr_position_target - tr_body_position); 
        Vector<3> hl_position_error = (hl_position_target - hl_body_position);
        Vector<3> hr_position_error = (hr_position_target - hr_body_position);
        // Vector<3> velocity_error = velocity_target - state.linear_body_velocity;
        // Vector<3> rotation_error = (Eigen::Quaternion<double>(1, 0, 0, 0) * body_rotation.conjugate()).vec();
        // Vector<3> angular_velocity_error = Vector<3>::Zero() - state.angular_body_velocity;
        // Vector<3> linear_control = 150.0 * (position_error) + 25.0 * (velocity_error);


        Vector<3> tl_linear_control = 150.0 * (tl_position_error);
        Vector<3> tr_linear_control = 150.0 * (tr_position_error);
        Vector<3> hl_linear_control = 150.0 * (hl_position_error);
        Vector<3> hr_linear_control = 150.0 * (hr_position_error);
        // Vector<3> angular_control = 50.0 * (rotation_error) + 10.0 * (angular_velocity_error);
        // // Eigen::Vector<double, 6> cmd {linear_control(0), linear_control(1), linear_control(2), angular_control(0), angular_control(1), angular_control(2)};
        // Eigen::Vector<double, 6> cmd1 {tl_linear_control(0), tl_linear_control(1), tl_linear_control(2), 0, 1000, 0};        
        // Eigen::Vector<double, 6> cmd2 {tr_linear_control(0), tr_linear_control(1), tr_linear_control(2), 0, 1000, 0};        
        // Eigen::Vector<double, 6> cmd3 {hl_linear_control(0), hl_linear_control(1), hl_linear_control(2), 0, 1000, 0};        
        // Eigen::Vector<double, 6> cmd4 {hr_linear_control(0), hr_linear_control(1), hr_linear_control(2), 0, 1000, 0};        

        

        Eigen::Vector<double, 6> cmd1 {0, 0, tl_linear_control(2), 0, 700, 0};        
        Eigen::Vector<double, 6> cmd2 {0, 0, tr_linear_control(2), 0, 700, 0};        
        Eigen::Vector<double, 6> cmd3 {0, 0, hl_linear_control(2), 0, 700, 0};        
        Eigen::Vector<double, 6> cmd4 {0, 0, hr_linear_control(2), 0, 700, 0};        

        // Eigen::Vector<double, 6> cmd1 {0, 0, 0, 0, 0.7*1e3, 0};        
        // Eigen::Vector<double, 6> cmd2 {0, 0, 0, 0, 0.7*1e3, 0};        
        // Eigen::Vector<double, 6> cmd3 {0, 0, 0, 0, 0.7*1e3, 0};        
        // Eigen::Vector<double, 6> cmd4 {0, 0, 0, 0, 0.7*1e3, 0};        


        taskspace_targets.row(1) = cmd1;
        taskspace_targets.row(2) = cmd2;
        taskspace_targets.row(3) = cmd3;
        taskspace_targets.row(4) = cmd4;


        // ------------------------------------------------------------------
        //       track head height
        // ------------------------------------------------------------------
        Vector<3> position_target = Vector<3>(
            initial_position(0), initial_position(1), initial_position(2)
        );
        Vector<3> velocity_target = Vector<3>(
            0.0,0.0,0.0
        );

        Eigen::Quaternion<double> body_rotation = Eigen::Quaternion<double>(state.body_rotation(0), state.body_rotation(1), state.body_rotation(2), state.body_rotation(3));
        Vector<3> body_position = qpos(Eigen::seqN(0, 3));
        Vector<3> position_error = position_target - body_position;
        Vector<3> velocity_error = velocity_target - state.linear_body_velocity;
        Vector<3> rotation_error = (Eigen::Quaternion<double>(1, 0, 0, 0) * body_rotation.conjugate()).vec();
        Vector<3> angular_velocity_error = Vector<3>::Zero() - state.angular_body_velocity;
        Vector<3> linear_control = 200.0 * (position_error) + 0.0 * (velocity_error);
        Vector<3> angular_control = 50.0 * (rotation_error) + 10.0 * (angular_velocity_error);
        Eigen::Vector<double, 6> cmd {0, 0, linear_control(2), angular_control(0), angular_control(1), angular_control(2)};
        taskspace_targets.row(0) = cmd;
        
        // ------------------------------------------------------------------
        //       camera track head x
        // ------------------------------------------------------------------
        cam.lookat[0] = body_position(0);
        // mjv_updateScene(mj_model, mj_data, &scn, nullptr, &cam);        

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

    // save data to file
    std::ofstream outfile("tl_shin_sine_data.txt");
    if (outfile.is_open()) {
        for (size_t i = 0; i < target_tl_shin_data.size(); ++i) {
            outfile << target_tl_shin_data[i] << " " << tl_shin_data[i] << std::endl;
        }
        outfile.close();
        std::cout << "Data saved to data.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }

    return 0;
}