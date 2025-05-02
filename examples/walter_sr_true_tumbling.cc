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


double wrapToPi(double angle) {
    if (angle < -M_PI) {
        angle += 2 * M_PI;
    }
    if (angle >= M_PI) {
        angle -= 2 * M_PI;
    }
    return angle;
}



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
    //           initialize variables
    //===================================================
    Vector<3> initial_position = qpos(Eigen::seqN(0, 3));

    // site positions (0-torso, 1-4 -> shin, 5-8 -> thigh)
    Eigen::Matrix<double, model::site_ids_size, 3> site_data;
    Eigen::Matrix<double, model::site_ids_size, 3> initial_site_data;

    // used to check distance between wheel height to determine contact
    Eigen::Vector<double, model::contact_site_ids_size> contact_check;
    double wheel_contact_check_height = 0.0615;

    // site orientation
    Eigen::Matrix<double, model::site_ids_size, 9> site_rotational_data;
    Eigen::Matrix<double, model::site_ids_size, 9> initial_site_rotational_data;


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

    // TEST TIME
    double simulation_time = 40.0;
    auto current_time = mj_data->time;

    // to get points / site position --> we need to build site-ids using sites
    std::vector<std::string> sites;
    std::vector<int> site_ids;

    // initialize data array variables to record
    std::vector<double> target_tl_shin_data;
    std::vector<double> time_data;
    std::vector<double> tl_shin_data;

    // loop to get site ids
    for(const std::string_view& site : model::site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model, mjOBJ_SITE, site_str.data());
        assert(id != -1 && "Site not found in model.");
        sites.push_back(site_str);
        site_ids.push_back(id);
    }

    // site id remapping because mjdata gets data from top to bottom and we need site ids as we assign in the config
    initial_site_data = Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->site_xpos)(site_ids, Eigen::placeholders::all);
    initial_site_rotational_data = Eigen::Map<Matrix<model::site_ids_size, 9>>(mj_data->site_xmat)(site_ids, Eigen::placeholders::all);

    //  last (for velo) and initial angular position of the shin 
    double last_tl_angular_position = acos(initial_site_rotational_data(1,0));
    double last_tr_angular_position = acos(initial_site_rotational_data(2,0));
    double last_hl_angular_position = acos(initial_site_rotational_data(3,0));
    double last_hr_angular_position = acos(initial_site_rotational_data(4,0));

    double initial_tl_angular_position = acos(initial_site_rotational_data(1,0));
    double initial_tr_angular_position = acos(initial_site_rotational_data(2,0));
    double initial_hl_angular_position = acos(initial_site_rotational_data(3,0));
    double initial_hr_angular_position = acos(initial_site_rotational_data(4,0));

    //  last (for velocity) and initial angular position of the thigh sites 
    double last_tlh_angular_position = acos(initial_site_rotational_data(5,0));
    double last_trh_angular_position = acos(initial_site_rotational_data(6,0));
    double last_hlh_angular_position = acos(initial_site_rotational_data(7,0));
    double last_hrh_angular_position = acos(initial_site_rotational_data(8,0));

    double initial_tlh_angular_position = acos(initial_site_rotational_data(5,0));
    double initial_trh_angular_position = acos(initial_site_rotational_data(6,0));
    double initial_hlh_angular_position = acos(initial_site_rotational_data(7,0));
    double initial_hrh_angular_position = acos(initial_site_rotational_data(8,0));

    //  last (for velocity) linear position of the thigh sites 
    Vector<3> last_tlh_linear_position = initial_site_data(5,Eigen::seqN(0, 3));
    Vector<3> last_trh_linear_position = initial_site_data(6,Eigen::seqN(0, 3));
    Vector<3> last_hlh_linear_position = initial_site_data(7,Eigen::seqN(0, 3));
    Vector<3> last_hrh_linear_position = initial_site_data(8,Eigen::seqN(0, 3));

    double last_time = current_time;


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
        // find contact mask based on dist between wheel and ground --> 0 - body , 1-4 -> shin, 5-8 -> thigh , 9-16 -> wheels
        //===================================================
        contact_check = {(site_data(9,2)<wheel_contact_check_height),
            (site_data(10,2)<wheel_contact_check_height),
            (site_data(11,2)<wheel_contact_check_height),
            (site_data(12,2)<wheel_contact_check_height),
            (site_data(13,2)<wheel_contact_check_height),
            (site_data(14,2)<wheel_contact_check_height),
            (site_data(15,2)<wheel_contact_check_height),
            (site_data(16,2)<wheel_contact_check_height)};
        
    


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

        // ------------------------------------------------------------------------------------------------------------------------------------
        //       shin z-axis height tracking
        // ------------------------------------------------------------------------------------------------------------------------------------
        // targets
        Vector<3> tl_position_target = Vector<3>(
            initial_site_data(1,0), initial_site_data(1,1), initial_site_data(1,2)
        );
        Vector<3> tr_position_target = Vector<3>(
            initial_site_data(2,0), initial_site_data(2,1), initial_site_data(2,2)
        );
        Vector<3> hl_position_target = Vector<3>(
            initial_site_data(3,0), initial_site_data(3,1), initial_site_data(3,2)
        );
        Vector<3> hr_position_target = Vector<3>(
            initial_site_data(4,0), initial_site_data(4,1), initial_site_data(4,2)
        );


        
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


        // Vector<3> tl_linear_control = 150.0 * (tl_position_error);
        // Vector<3> tr_linear_control = 150.0 * (tr_position_error);
        // Vector<3> hl_linear_control = 150.0 * (hl_position_error);
        // Vector<3> hr_linear_control = 150.0 * (hr_position_error);
        
        // Vector<3> angular_control = 50.0 * (rotation_error) + 10.0 * (angular_velocity_error);
        // // Eigen::Vector<double, 6> cmd {linear_control(0), linear_control(1), linear_control(2), angular_control(0), angular_control(1), angular_control(2)};
        // Eigen::Vector<double, 6> cmd1 {tl_linear_control(0), tl_linear_control(1), tl_linear_control(2), 0, 1000, 0};        
        // Eigen::Vector<double, 6> cmd2 {tr_linear_control(0), tr_linear_control(1), tr_linear_control(2), 0, 1000, 0};        
        // Eigen::Vector<double, 6> cmd3 {hl_linear_control(0), hl_linear_control(1), hl_linear_control(2), 0, 1000, 0};        
        // Eigen::Vector<double, 6> cmd4 {hr_linear_control(0), hr_linear_control(1), hr_linear_control(2), 0, 1000, 0};        

        // ------------------------------------------------------------------------------------------------------------------------------------
        //       shin angular position
        // ------------------------------------------------------------------------------------------------------------------------------------
        // shin angular position
        // Sinusoidal Position and Velocity Tracking:
        double shin_rot_vel = 0.5;
        // double shin_rot_frequency = 0.1;

        double tl_angular_position = acos(site_rotational_data(1,0));
        double tr_angular_position = acos(site_rotational_data(2,0));
        double hl_angular_position = acos(site_rotational_data(3,0));
        double hr_angular_position = acos(site_rotational_data(4,0));

        double tl_angular_velocity = (tl_angular_position - last_tl_angular_position)/(current_time - last_time);
        double tr_angular_velocity = (tr_angular_position - last_tr_angular_position)/(current_time - last_time);
        double hl_angular_velocity = (hl_angular_position - last_hl_angular_position)/(current_time - last_time);
        double hr_angular_velocity = (hr_angular_position - last_hr_angular_position)/(current_time - last_time);

        // targets
        double tl_angular_position_target = wrapToPi(initial_tl_angular_position + shin_rot_vel * current_time);
        double tr_angular_position_target = wrapToPi(initial_tr_angular_position + shin_rot_vel * current_time);
        double hl_angular_position_target = wrapToPi(initial_hl_angular_position + shin_rot_vel * current_time);
        double hr_angular_position_target = wrapToPi(initial_hr_angular_position + shin_rot_vel * current_time);

        double tl_angular_velocity_target = shin_rot_vel;
        double tr_angular_velocity_target = shin_rot_vel;
        double hl_angular_velocity_target = shin_rot_vel;
        double hr_angular_velocity_target = shin_rot_vel;

        double tl_angular_position_error = (tl_angular_position_target - tl_angular_position);
        double tr_angular_position_error = (tr_angular_position_target - tr_angular_position);
        double hl_angular_position_error = (hl_angular_position_target - hl_angular_position);
        double hr_angular_position_error = (hr_angular_position_target - hr_angular_position);
        
        double tl_angular_velocity_error = (tl_angular_velocity_target - tl_angular_velocity);
        double tr_angular_velocity_error = (tr_angular_velocity_target - tr_angular_velocity);
        double hl_angular_velocity_error = (hl_angular_velocity_target - hl_angular_velocity);
        double hr_angular_velocity_error = (hr_angular_velocity_target - hr_angular_velocity);

        double shin_kp = 100;
        double shin_kv = 200;

        double tl_angular_control = shin_kp * (tl_angular_position_error) + shin_kv * (tl_angular_velocity_error);
        double tr_angular_control = shin_kp * (tr_angular_position_error) + shin_kv * (tr_angular_velocity_error);
        double hl_angular_control = shin_kp * (hl_angular_position_error) + shin_kv * (hl_angular_velocity_error);
        double hr_angular_control = shin_kp * (hr_angular_position_error) + shin_kv * (hr_angular_velocity_error);
        
        double last_tl_angular_position = tl_angular_position;
        double last_tr_angular_position = tr_angular_position;
        double last_hl_angular_position = hl_angular_position;
        double last_hr_angular_position = hr_angular_position;

        // =================================         

        // ------------------------------------------------------------------
        //       feedback angular velocity tracking
        // ------------------------------------------------------------------        

        Eigen::Vector<double, 6> cmd1 {0, 0, 0, 0, tl_angular_control, 0};        
        Eigen::Vector<double, 6> cmd2 {0, 0, 0, 0, tr_angular_control, 0};        
        Eigen::Vector<double, 6> cmd3 {0, 0, 0, 0, hl_angular_control, 0};        
        Eigen::Vector<double, 6> cmd4 {0, 0, 0, 0, hr_angular_control, 0};        

        // ------------------------------------------------------------------
        //       old feedforward angular acceleration
        // ------------------------------------------------------------------

        // Eigen::Vector<double, 6> cmd1 {0, 0, 0, 0, 0.8*1e3, 0};        
        // Eigen::Vector<double, 6> cmd2 {0, 0, 0, 0, 0.8*1e3, 0};        
        // Eigen::Vector<double, 6> cmd3 {0, 0, 0, 0, 0.8*1e3, 0};        
        // Eigen::Vector<double, 6> cmd4 {0, 0, 0, 0, 0.8*1e3, 0};        


        taskspace_targets.row(1) = cmd1;
        taskspace_targets.row(2) = cmd2;
        taskspace_targets.row(3) = cmd3;
        taskspace_targets.row(4) = cmd4;


        // ------------------------------------------------------------------------------------------------------------------------------------
        //       thigh angular position
        // ------------------------------------------------------------------------------------------------------------------------------------
        // thigh angular position
        // constant position and zero velocity Tracking:
        double thigh_rot_vel = 0.0;
        double thigh_kp = 500;
        double thigh_kv = 50;

        double tlh_angular_position = acos(site_rotational_data(5,0));
        double trh_angular_position = acos(site_rotational_data(6,0));
        double hlh_angular_position = acos(site_rotational_data(7,0));
        double hrh_angular_position = acos(site_rotational_data(8,0));

        double tlh_angular_velocity = (tlh_angular_position - last_tlh_angular_position)/(current_time - last_time);
        double trh_angular_velocity = (trh_angular_position - last_trh_angular_position)/(current_time - last_time);
        double hlh_angular_velocity = (hlh_angular_position - last_hlh_angular_position)/(current_time - last_time);
        double hrh_angular_velocity = (hrh_angular_position - last_hrh_angular_position)/(current_time - last_time);

        // targets
        double tlh_angular_position_target = wrapToPi(initial_tlh_angular_position + thigh_rot_vel * current_time);
        double trh_angular_position_target = wrapToPi(initial_trh_angular_position + thigh_rot_vel * current_time);
        double hlh_angular_position_target = wrapToPi(initial_hlh_angular_position + thigh_rot_vel * current_time);
        double hrh_angular_position_target = wrapToPi(initial_hrh_angular_position + thigh_rot_vel * current_time);

        double tlh_angular_velocity_target = thigh_rot_vel;
        double trh_angular_velocity_target = thigh_rot_vel;
        double hlh_angular_velocity_target = thigh_rot_vel;
        double hrh_angular_velocity_target = thigh_rot_vel;

        double tlh_angular_position_error = (tlh_angular_position_target - tlh_angular_position);
        double trh_angular_position_error = (trh_angular_position_target - trh_angular_position);
        double hlh_angular_position_error = (hlh_angular_position_target - hlh_angular_position);
        double hrh_angular_position_error = (hrh_angular_position_target - hrh_angular_position);
        
        double tlh_angular_velocity_error = (tlh_angular_velocity_target - tlh_angular_velocity);
        double trh_angular_velocity_error = (trh_angular_velocity_target - trh_angular_velocity);
        double hlh_angular_velocity_error = (hlh_angular_velocity_target - hlh_angular_velocity);
        double hrh_angular_velocity_error = (hrh_angular_velocity_target - hrh_angular_velocity);

        double tlh_angular_control = thigh_kp * (tlh_angular_position_error) + thigh_kv * (tlh_angular_velocity_error);
        double trh_angular_control = thigh_kp * (trh_angular_position_error) + thigh_kv * (trh_angular_velocity_error);
        double hlh_angular_control = thigh_kp * (hlh_angular_position_error) + thigh_kv * (hlh_angular_velocity_error);
        double hrh_angular_control = thigh_kp * (hrh_angular_position_error) + thigh_kv * (hrh_angular_velocity_error);
        
        double last_tlh_angular_position = tlh_angular_position;
        double last_trh_angular_position = trh_angular_position;
        double last_hlh_angular_position = hlh_angular_position;
        double last_hrh_angular_position = hrh_angular_position;
        // ------------------------------------------------------------------------------------------------------------------------------------
        //       thigh linear position
        // ------------------------------------------------------------------------------------------------------------------------------------
        double thigh_lin_vel = 0.0;
        double thigh_lin_kp = 1000*4.2;
        double thigh_lin_kv = 100*4.2;

        Vector<3> tlh_linear_position = site_data(5,Eigen::seqN(0, 3));
        Vector<3> trh_linear_position = site_data(6,Eigen::seqN(0, 3));
        Vector<3> hlh_linear_position = site_data(7,Eigen::seqN(0, 3));
        Vector<3> hrh_linear_position = site_data(8,Eigen::seqN(0, 3));
    
        Vector<3> tlh_linear_velocity = (tlh_linear_position - last_tlh_linear_position)/(current_time - last_time);
        Vector<3> trh_linear_velocity = (trh_linear_position - last_trh_linear_position)/(current_time - last_time);
        Vector<3> hlh_linear_velocity = (hlh_linear_position - last_hlh_linear_position)/(current_time - last_time);
        Vector<3> hrh_linear_velocity = (hrh_linear_position - last_hrh_linear_position)/(current_time - last_time);

        // targets
        double tlh_linear_velocity_target = thigh_lin_vel;
        double trh_linear_velocity_target = thigh_lin_vel;
        double hlh_linear_velocity_target = thigh_lin_vel;
        double hrh_linear_velocity_target = thigh_lin_vel;

        double tlh_linear_position_error = (initial_site_data(5,2) - tlh_linear_position(2));
        double trh_linear_position_error = (initial_site_data(6,2) - trh_linear_position(2));
        double hlh_linear_position_error = (initial_site_data(7,2) - hlh_linear_position(2));
        double hrh_linear_position_error = (initial_site_data(8,2) - hrh_linear_position(2));
        
        double tlh_linear_velocity_error = (tlh_linear_velocity_target - tlh_linear_velocity(2));
        double trh_linear_velocity_error = (trh_linear_velocity_target - trh_linear_velocity(2));
        double hlh_linear_velocity_error = (hlh_linear_velocity_target - hlh_linear_velocity(2));
        double hrh_linear_velocity_error = (hrh_linear_velocity_target - hrh_linear_velocity(2));

        double tlh_linear_control = thigh_lin_kp * (tlh_linear_position_error) + thigh_lin_kv * (tlh_linear_velocity_error);
        double trh_linear_control = thigh_lin_kp * (trh_linear_position_error) + thigh_lin_kv * (trh_linear_velocity_error);
        double hlh_linear_control = thigh_lin_kp * (hlh_linear_position_error) + thigh_lin_kv * (hlh_linear_velocity_error);
        double hrh_linear_control = thigh_lin_kp * (hrh_linear_position_error) + thigh_lin_kv * (hrh_linear_velocity_error);
        
        Vector<3> last_tlh_linear_position = tlh_linear_position;
        Vector<3> last_trh_linear_position = trh_linear_position;
        Vector<3> last_hlh_linear_position = hlh_linear_position;
        Vector<3> last_hrh_linear_position = hrh_linear_position;
        // =================================         
        double last_time = current_time;
        // =================================         

        // ------------------------------------------------------------------
        //       feedback angular velocity tracking
        // ------------------------------------------------------------------        

        // Eigen::Vector<double, 6> cmd5 {0, 0, tlh_linear_control, 0, tlh_angular_control, 0};        
        // Eigen::Vector<double, 6> cmd6 {0, 0, trh_linear_control, 0, trh_angular_control, 0};        
        // Eigen::Vector<double, 6> cmd7 {0, 0, hlh_linear_control, 0, hlh_angular_control, 0};        
        // Eigen::Vector<double, 6> cmd8 {0, 0, hrh_linear_control, 0, hrh_angular_control, 0};        

        // Eigen::Vector<double, 6> cmd5 {0, 0, tlh_linear_control, 0, 0, 0};        
        // Eigen::Vector<double, 6> cmd6 {0, 0, trh_linear_control, 0, 0, 0};        
        // Eigen::Vector<double, 6> cmd7 {0, 0, hlh_linear_control, 0, 0, 0};        
        // Eigen::Vector<double, 6> cmd8 {0, 0, hrh_linear_control, 0, 0, 0};        


        // Eigen::Vector<double, 6> cmd5 {0, 0, 0, 0, tlh_angular_control, 0};        
        // Eigen::Vector<double, 6> cmd6 {0, 0, 0, 0, trh_angular_control, 0};        
        // Eigen::Vector<double, 6> cmd7 {0, 0, 0, 0, hlh_angular_control, 0};        
        // Eigen::Vector<double, 6> cmd8 {0, 0, 0, 0, hrh_angular_control, 0};        

        // Eigen::Vector<double, 6> cmd5 {0, 0, 0, 0, 0.1, 0};        
        // Eigen::Vector<double, 6> cmd6 {0, 0, 0, 0, 0.1, 0};        
        // Eigen::Vector<double, 6> cmd7 {0, 0, 0, 0, 0.1, 0};        
        // Eigen::Vector<double, 6> cmd8 {0, 0, 0, 0, 0.1, 0};        
        

        // ------------------------------------------------------------------
        //       old feedforward angular acceleration
        // ------------------------------------------------------------------

        // Eigen::Vector<double, 6> cmd1 {0, 0, 0, 0, 0.8*1e3, 0};        
        // Eigen::Vector<double, 6> cmd2 {0, 0, 0, 0, 0.8*1e3, 0};        
        // Eigen::Vector<double, 6> cmd3 {0, 0, 0, 0, 0.8*1e3, 0};        
        // Eigen::Vector<double, 6> cmd4 {0, 0, 0, 0, 0.8*1e3, 0};        


        // taskspace_targets.row(5) = cmd5;
        // taskspace_targets.row(6) = cmd6;
        // taskspace_targets.row(7) = cmd7;
        // taskspace_targets.row(8) = cmd8;        


        // ------------------------------------------------------------------------------------------------------------------------------------
        //       track head height
        // ------------------------------------------------------------------------------------------------------------------------------------
        Vector<3> position_target = Vector<3>(
            initial_position(0), initial_position(1), initial_position(2)-0.03
        );
        Vector<3> velocity_target = Vector<3>(
            0.0,0.0,0.0
        );

        Eigen::Quaternion<double> body_rotation = Eigen::Quaternion<double>(state.body_rotation(0), state.body_rotation(1), state.body_rotation(2), state.body_rotation(3));
        Vector<3> body_position = qpos(Eigen::seqN(0, 3));
        Vector<3> position_error = position_target - body_position;
        Vector<3> velocity_error = Vector<3>(0.0, 0.0, 0.0-state.linear_body_velocity(2));
        Vector<3> rotation_error = (Eigen::Quaternion<double>(1, 0, 0, 0) * body_rotation.conjugate()).vec();
        Vector<3> angular_velocity_error = Vector<3>::Zero() - state.angular_body_velocity;

        double torso_lin_kp = 20000.0;
        double torso_lin_kv = 200.0;

        double torso_ang_kp = 100.0;
        double torso_ang_kv = 100.0;

        Vector<3> linear_control = torso_lin_kp * (position_error) + torso_lin_kv * (velocity_error);
        Vector<3> angular_control = torso_ang_kp * (rotation_error) + torso_ang_kv * (angular_velocity_error);
        Eigen::Vector<double, 6> cmd {0, 0, linear_control(2), angular_control(0), angular_control(1), angular_control(2)};
        taskspace_targets.row(0) = cmd;
        

        // ------------------------------------------------------------------------------------------------------------------------------------
        //       camera track head x
        // ------------------------------------------------------------------------------------------------------------------------------------
        cam.lookat[0] = body_position(0);


        // ------------------------------------------------------------------
        //       record tlh angular position
        // ------------------------------------------------------------------
        // std::cout << "tlh_angular_position_target: " << initial_position(2)-0.1 << std::endl;
        // std::cout << "tlh_angular_control: " << body_position(2) << std::endl;
        
        target_tl_shin_data.push_back(position_target(2));
        tl_shin_data.push_back(body_position(2));
        time_data.push_back(current_time);


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
            outfile << target_tl_shin_data[i] << " " << tl_shin_data[i] << " " << time_data[i] << std::endl;
        }
        outfile.close();
        std::cout << "Data saved to data.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }

    return 0;
}