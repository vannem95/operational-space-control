#include <filesystem>
#include <cmath>
#include <fstream>
#include <vector>

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



//  to wrap angles higher than 2pi - angular positions of hip and shins
double wrapToPi(double angle) {
    const double twoPi = 2.0 * M_PI;
    while (angle > M_PI) {
      angle -= twoPi;
    }
    while (angle <= -M_PI) {
      angle += twoPi;
    }
    return angle;
}



// checks if the <value> is in the <vector>
template <typename T>
bool contains(const std::vector<T>& vec, const T& value) {
    // std::find returns an iterator to the first occurrence of the value,
    // or vec.end() if the value is not found.
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}



// prints the geom name of the id
void printGeomName(const mjModel* m, int geom_id) {
    // Check if the geom_id is valid
    if (geom_id >= 0 && geom_id < m->ngeom) {
        // mjOBJ_GEOM specifies that we are looking for a geom object
        const char* geom_name = mj_id2name(m, mjOBJ_GEOM, geom_id);

        if (geom_name) { // mj_id2name returns NULL if the ID is invalid or unnamed
            std::cout << "Geom with ID " << geom_id << " has name: " << geom_name << std::endl;
        } else {
            std::cout << "Geom with ID " << geom_id << " has no name (or ID is invalid)." << std::endl;
        }
    } else {
        std::cout << "Invalid geom ID: " << geom_id << std::endl;
    }
}  



// finds geoms on the body of the site id
void findGeomsOnSameBodyAsSite(const mjModel* m, int site_id) {
    if (site_id < 0 || site_id >= m->nsite) {
        std::cerr << "Error: Invalid site ID." << std::endl;
        return;
    }

    const char* site_name = mj_id2name(m, mjOBJ_SITE, site_id);
    if (!site_name) site_name = "Unnamed Site";

    int site_body_id = m->site_bodyid[site_id];
    const char* site_body_name = mj_id2name(m, mjOBJ_BODY, site_body_id);
    if (!site_body_name) site_body_name = "Unnamed Body";


    std::cout << "\nSite '" << site_name << "' (ID: " << site_id
              << ") is attached to Body '" << site_body_name << "' (ID: " << site_body_id << ")." << std::endl;
    std::cout << "Geoms on the same body as Site '" << site_name << "':" << std::endl;

    bool found_geom = false;
    for (int i = 0; i < m->ngeom; ++i) {
        if (m->geom_bodyid[i] == site_body_id) {
            const char* geom_name = mj_id2name(m, mjOBJ_GEOM, i);
            if (!geom_name) geom_name = "Unnamed Geom";
            std::cout << "  - Geom '" << geom_name << "' (ID: " << i << ")" << std::endl;
            found_geom = true;
        }
    }

    if (!found_geom) {
        std::cout << "  No geoms found on the same body as Site '" << site_name << "'." << std::endl;
    }
}



// find sites on the same geom id
std::vector<int> getSiteIdsOnSameBodyAsGeom(const mjModel* m, int geom_id) {
    std::vector<int> associated_site_ids; // Vector to store the found site IDs

    if (geom_id < 0 || geom_id >= m->ngeom) {
        std::cerr << "Error: Invalid geom ID: " << geom_id << std::endl;
        return associated_site_ids; // Return empty vector
    }

    const char* geom_name = mj_id2name(m, mjOBJ_GEOM, geom_id);
    if (!geom_name) geom_name = "Unnamed Geom";

    int geom_body_id = m->geom_bodyid[geom_id];
    const char* geom_body_name = mj_id2name(m, mjOBJ_BODY, geom_body_id);
    if (!geom_body_name) geom_body_name = "Unnamed Body";

    // std::cout << "\nChecking for sites on the same body as Geom '" << geom_name
    //           << "' (ID: " << geom_id << ", Body ID: " << geom_body_id << ", Body Name: '" << geom_body_name << "'):" << std::endl;


    for (int i = 0; i < m->nsite; ++i) {
        if (m->site_bodyid[i] == geom_body_id) {
            associated_site_ids.push_back(i); // Add the site ID to the vector
        }
    }

    if (associated_site_ids.empty()) {
        std::cout << "  No sites found on the same body." << std::endl;
    } else {
        // std::cout << "  Found " << associated_site_ids.size() << " site(s) on the same body. IDs: ";
        for (int site_id : associated_site_ids) {
            // std::cout << site_id << " ";
            // Optionally, print the site name as well
            const char* site_name = mj_id2name(m, mjOBJ_SITE, site_id);
            // if (site_name) {
            //     std::cout << "('" << site_name << "') ";
            // }
        }
        // std::cout << std::endl;
    }

    return associated_site_ids; // Return the vector of IDs
}



// outputs vector C as binary of if vector A elements are in vector B
std::vector<int> getBinaryRepresentation_std_find(const std::vector<int>& A, const std::vector<int>& B) {
    std::vector<int> C;
    C.reserve(B.size());

    for (int b_element : B) {
        // std::find returns an iterator to the element if found, or A.end() if not.
        auto it = std::find(A.begin(), A.end(), b_element);
        C.push_back((it != A.end()) ? 1 : 0);
    }
    return C;
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
    mj_resetDataKeyframe(mj_model, mj_data, 3);


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
    GLFWwindow* window = glfwCreateWindow(800, 600, "Slow tumbling", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    std::cout << "--------------here--------------";


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
    Eigen::Vector<double, model::contact_site_ids_size> contact_check2;
    // double wheel_contact_check_height = 0.065;
    double wheel_contact_check_height = 0.0615;
    double wheel_contact_check_height2 = 0.0;

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
    double simulation_time = 30.0;
    auto current_time = mj_data->time;

    // to get points / site position --> we need to build site-ids using sites
    std::vector<std::string> sites;
    std::vector<int> site_ids;

    // initialize data array variables to record
    std::vector<double> data1_1;
    std::vector<double> data1_2;
    std::vector<double> data1_3;
    std::vector<double> data1_4;
    std::vector<double> data1t_1;
    std::vector<double> data1t_2;
    std::vector<double> data1t_3;
    std::vector<double> data1t_4;

    std::vector<double> data2_1;
    std::vector<double> data2_2;
    std::vector<double> data2_3;
    std::vector<double> data2_4;
    std::vector<double> data2t_1;
    std::vector<double> data2t_2;
    std::vector<double> data2t_3;
    std::vector<double> data2t_4;    

    std::vector<double> data3_1;
    std::vector<double> data3_2;
    std::vector<double> data3_3;
    std::vector<double> data3_4;
    std::vector<double> data3t_1;
    std::vector<double> data3t_2;
    std::vector<double> data3t_3;
    std::vector<double> data3t_4;    

    std::vector<double> data4_1;
    std::vector<double> data4_2;
    std::vector<double> data4_3;
    std::vector<double> data4_4;
    std::vector<double> data4t_1;
    std::vector<double> data4t_2;
    std::vector<double> data4t_3;
    std::vector<double> data4t_4;    

    std::vector<double> data5_1;
    std::vector<double> data5_2;
    std::vector<double> data5_3;
    std::vector<double> data5t_1;
    std::vector<double> data5t_2;
    std::vector<double> data5t_3;

    std::vector<double> data6_1;
    std::vector<double> data6_2;
    std::vector<double> data6_3;
    std::vector<double> data6t_1;
    std::vector<double> data6t_2;
    std::vector<double> data6t_3;

    std::vector<double> data_time;
    

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

    //===================================================
    //  last (for velo) and initial angular position of the shin 
    //===================================================
    // double last_tl_angular_position = acos(initial_site_rotational_data(1,0));
    // double last_tr_angular_position = acos(initial_site_rotational_data(2,0));
    // double last_hl_angular_position = acos(initial_site_rotational_data(3,0));
    // double last_hr_angular_position = acos(initial_site_rotational_data(4,0));

    // double last_tl_angular_position = atan2(initial_site_rotational_data(1,2),initial_site_rotational_data(1,0));
    // double last_tr_angular_position = atan2(initial_site_rotational_data(2,2),initial_site_rotational_data(2,0));
    // double last_hl_angular_position = atan2(initial_site_rotational_data(3,2),initial_site_rotational_data(3,0));
    // double last_hr_angular_position = atan2(initial_site_rotational_data(4,2),initial_site_rotational_data(4,0));

    double last_tl_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[2]];        
    double last_tr_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[4]];        
    double last_hl_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[6]];        
    double last_hr_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[8]];        
    
    double last_tl_angular_position = last_tl_shin_angle;
    double last_tr_angular_position = last_tr_shin_angle;
    double last_hl_angular_position = last_hl_shin_angle;
    double last_hr_angular_position = last_hr_shin_angle;

    // double initial_tl_angular_position = atan2(initial_site_rotational_data(1,2),initial_site_rotational_data(1,0));
    // double initial_tr_angular_position = atan2(initial_site_rotational_data(2,2),initial_site_rotational_data(2,0));
    // double initial_hl_angular_position = atan2(initial_site_rotational_data(3,2),initial_site_rotational_data(3,0));
    // double initial_hr_angular_position = atan2(initial_site_rotational_data(4,2),initial_site_rotational_data(4,0));

    double initial_tl_angular_position = last_tl_shin_angle;
    double initial_tr_angular_position = last_tr_shin_angle;
    double initial_hl_angular_position = last_hl_shin_angle;
    double initial_hr_angular_position = last_hr_shin_angle;
    
    //===================================================
    //  last (for velocity) and initial angular position of the thigh sites 
    //===================================================
    // double last_tlh_angular_position = acos(initial_site_rotational_data(5,0));
    // double last_trh_angular_position = acos(initial_site_rotational_data(6,0));
    // double last_hlh_angular_position = acos(initial_site_rotational_data(7,0));
    // double last_hrh_angular_position = acos(initial_site_rotational_data(8,0));

    double last_tlh_angular_position = atan2(initial_site_rotational_data(5,2),initial_site_rotational_data(5,0));
    double last_trh_angular_position = atan2(initial_site_rotational_data(6,2),initial_site_rotational_data(6,0));
    double last_hlh_angular_position = atan2(initial_site_rotational_data(7,2),initial_site_rotational_data(7,0));
    double last_hrh_angular_position = atan2(initial_site_rotational_data(8,2),initial_site_rotational_data(8,0));

    // double initial_tlh_angular_position = acos(initial_site_rotational_data(5,0));
    // double initial_trh_angular_position = acos(initial_site_rotational_data(6,0));
    // double initial_hlh_angular_position = acos(initial_site_rotational_data(7,0));
    // double initial_hrh_angular_position = acos(initial_site_rotational_data(8,0));

    double initial_tlh_angular_position = atan2(initial_site_rotational_data(5,2),initial_site_rotational_data(5,0));
    double initial_trh_angular_position = atan2(initial_site_rotational_data(6,2),initial_site_rotational_data(6,0));
    double initial_hlh_angular_position = atan2(initial_site_rotational_data(7,2),initial_site_rotational_data(7,0));
    double initial_hrh_angular_position = atan2(initial_site_rotational_data(8,2),initial_site_rotational_data(8,0));

    //  last (for velocity) linear position of the thigh sites 
    Vector<3> last_tlh_linear_position = initial_site_data(5,Eigen::seqN(0, 3));
    Vector<3> last_trh_linear_position = initial_site_data(6,Eigen::seqN(0, 3));
    Vector<3> last_hlh_linear_position = initial_site_data(7,Eigen::seqN(0, 3));
    Vector<3> last_hrh_linear_position = initial_site_data(8,Eigen::seqN(0, 3));

    Vector<3> last_tls_linear_position = initial_site_data(1,Eigen::seqN(0, 3));
    Vector<3> last_trs_linear_position = initial_site_data(2,Eigen::seqN(0, 3));
    Vector<3> last_hls_linear_position = initial_site_data(3,Eigen::seqN(0, 3));
    Vector<3> last_hrs_linear_position = initial_site_data(4,Eigen::seqN(0, 3));

    double last_time = current_time;

    // std::vector<int> wheel_sites_mujoco = {3, 4, 7, 8, 11, 12, 15, 16};
    std::vector<int> wheel_sites_mujoco = {3, 4, 7, 8, 11, 12, 15, 16};

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
        // contact_check = {(site_data(9,2)<wheel_contact_check_height),
        //     (site_data(10,2)<wheel_contact_check_height),
        //     (site_data(11,2)<wheel_contact_check_height),
        //     (site_data(12,2)<wheel_contact_check_height),
        //     (site_data(13,2)<wheel_contact_check_height),
        //     (site_data(14,2)<wheel_contact_check_height),
        //     (site_data(15,2)<wheel_contact_check_height),
        //     (site_data(16,2)<wheel_contact_check_height)};

        // contact_data = Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->contact)(site_ids, Eigen::placeholders::all);

        // std::cout << "Vector elements (range-based for loop): ";
        // std::cout << "--------------here--------------";
        // for (int element : site_ids) {
        //     std::cout << element << " ";
        // }
        // std::cout << std::endl;
                
        const mjContact& contact0 = mj_data->contact[0];
        const mjContact& contact1 = mj_data->contact[1];
        const mjContact& contact2 = mj_data->contact[2];
        const mjContact& contact3 = mj_data->contact[3];
        const mjContact& contact4 = mj_data->contact[4];
        const mjContact& contact5 = mj_data->contact[5];
        const mjContact& contact6 = mj_data->contact[6];
        const mjContact& contact7 = mj_data->contact[7];

        // std::cout << "mj_data->contact0: " << mj_data->contact[0].geom[1] << std::endl;
        // std::cout << "mj_data->contact1: " << mj_data->contact[1].geom[1] << std::endl;
        // std::cout << "mj_data->contact2: " << mj_data->contact[2].geom[1] << std::endl;
        // std::cout << "mj_data->contact3: " << mj_data->contact[3].geom[1] << std::endl;
        // std::cout << "mj_data->contact4: " << mj_data->contact[4].geom[1] << std::endl;
        // std::cout << "mj_data->contact5: " << mj_data->contact[5].geom[1] << std::endl;
        // std::cout << "mj_data->contact6: " << mj_data->contact[6].geom[1] << std::endl;
        // std::cout << "mj_data->contact7: " << mj_data->contact[7].geom[1] << std::endl;
        // std::cout << "mj_data->contact7: " << mj_data->contact[8].geom[1] << std::endl;
        // std::cout << "mj_data->contact7: " << mj_data->contact[9].geom[1] << std::endl;
        // std::cout << "mj_data->contact1: " << mj_data->contact[10].geom[1] << std::endl;
        // std::cout << "mj_data->contact1: " << mj_data->contact[11].geom[1] << std::endl;

        // std::cout << "mj_data->contact0: " << mj_data->contact[0].geom[0] << std::endl;
        // std::cout << "mj_data->contact1: " << mj_data->contact[1].geom[0] << std::endl;
        // std::cout << "mj_data->contact2: " << mj_data->contact[2].geom[0] << std::endl;
        // std::cout << "mj_data->contact3: " << mj_data->contact[3].geom[0] << std::endl;
        // std::cout << "mj_data->contact4: " << mj_data->contact[4].geom[0] << std::endl;
        // std::cout << "mj_data->contact5: " << mj_data->contact[5].geom[0] << std::endl;
        // std::cout << "mj_data->contact6: " << mj_data->contact[6].geom[0] << std::endl;
        // std::cout << "mj_data->contact7: " << mj_data->contact[7].geom[0] << std::endl;
        // std::cout << "mj_data->contact7: " << mj_data->contact[8].geom[0] << std::endl;
        // std::cout << "mj_data->contact7: " << mj_data->contact[9].geom[0] << std::endl;
        // std::cout << "mj_data->contact1: " << mj_data->contact[10].geom[0] << std::endl;
        // std::cout << "mj_data->contact1: " << mj_data->contact[11].geom[0] << std::endl;


         // Should print "floor"
        for (int i = 0; i < mj_model->ngeom; ++i) {
            printGeomName(mj_model, i);
        }
        

        // printGeomName(mj_model, 8); // Should print "box"
        // printGeomName(mj_model, 14); // Should indicate invalid ID or no name        
        // printGeomName(mj_model, 18); // Should indicate invalid ID or no name        

        // std::cout << "mj_data->ncon: " << mj_data->ncon << std::endl;
        // std::cout << "current_time: " << current_time << std::endl;


        std::vector<int> contact_site_ids_test;
        for (int i = 0; i < mj_data->ncon; ++i) {
            
            if (contains(wheel_sites_mujoco,mj_data->contact[i].geom[1])){
            // std::cout << "mj_data->contact[i].geom[1]: " << mj_data->contact[i].geom[1] << std::endl;

            std::vector<int> site_of_geom = getSiteIdsOnSameBodyAsGeom(mj_model, mj_data->contact[i].geom[1]);
            contact_site_ids_test.push_back(site_of_geom[0]);
            }
            else{}
        }

        for (int i = 0; i < mj_data->ncon; ++i) {
            // std::cout << "i: " << i << std::endl;
            // std::cout << "mj_data->contact[i].geom[0]: " << mj_data->contact[i].geom[0] << std::endl;
            if (contains(wheel_sites_mujoco,mj_data->contact[i].geom[0])){
            // std::cout << "mj_data->contact[i].geom[0]: " << mj_data->contact[i].geom[0] << std::endl;

            std::vector<int> site_of_geom = getSiteIdsOnSameBodyAsGeom(mj_model, mj_data->contact[i].geom[0]);
            contact_site_ids_test.push_back(site_of_geom[0]);
            }
            else{}
        }        

        // std::cout << "contact_site_ids_test: ";
        // for (int element : contact_site_ids_test) {
        //     std::cout << element << " ";
        // }
        // std::cout << std::endl;

        
        std::vector<int> contact_check2_temp = getBinaryRepresentation_std_find(contact_site_ids_test,wheel_sites_mujoco);


        // Create a temporary Eigen::Map of the int data, then cast to double and assign
        contact_check2 = Eigen::Map<Eigen::VectorXi>(contact_check2_temp.data(), contact_check2_temp.size()).cast<double>();
        

        // std::cout << "mj_data->contact8: " << mj_data->contact[8].geom[1] << std::endl;
        // std::cout << "mj_data->contact9: " << mj_data->contact[9].geom[1] << std::endl;
        // std::cout << "mj_data->contact10: " << mj_data->contact[10].geom[1] << std::endl;
        // std::cout << "mj_data->contact11: " << mj_data->contact[11].geom[1] << std::endl;
        // std::cout << "mj_data->contact12: " << mj_data->contact[12].geom[1] << std::endl;
        // std::cout << "mj_data->contact13: " << mj_data->contact[13].geom[1] << std::endl;
        // std::cout << "mj_data->contact14: " << mj_data->contact[14].geom[1] << std::endl;
        // std::cout << "mj_data->contact15: " << mj_data->contact[15].geom[1] << std::endl;
        // std::cout << "mj_data->contact16: " << mj_data->contact[16].geom[1] << std::endl;
        // std::cout << "mj_data->contact17: " << mj_data->contact[17].geom[1] << std::endl;
        // std::cout << "mj_data->contact18: " << mj_data->contact[18].geom[1] << std::endl;
        // std::cout << "mj_data->contact19: " << mj_data->contact[19].geom[1] << std::endl;
        // std::cout << "mj_data->contact20: " << mj_data->contact[20].geom[1] << std::endl;
        // std::cout << "mj_data->contact21: " << mj_data->contact[21].geom[1] << std::endl;
        // std::cout << "mj_data->contact22: " << mj_data->contact[22].geom[1] << std::endl;
        // std::cout << "mj_data->contact23: " << mj_data->contact[23].geom[1] << std::endl;
        // std::cout << "mj_data->contact24: " << mj_data->contact[24].geom[1] << std::endl;
        // std::cout << "mj_data->contact25: " << mj_data->contact[25].geom[1] << std::endl;
        // std::cout << "mj_data->contact26: " << mj_data->contact[26].geom[1] << std::endl;
        // std::cout << "mj_data->contact27: " << mj_data->contact[27].geom[1] << std::endl;
        // std::cout << "mj_data->contact28: " << mj_data->contact[28].geom[1] << std::endl;
        // std::cout << "mj_data->contact29: " << mj_data->contact[29].geom[1] << std::endl;
        // std::cout << "mj_data->contact30: " << mj_data->contact[30].geom[1] << std::endl;
        // std::cout << "mj_data->contact31: " << mj_data->contact[31].geom[1] << std::endl;
        // std::cout << "mj_data->contact32: " << mj_data->contact[32].geom[1] << std::endl;
        // std::cout << "mj_data->contact33: " << mj_data->contact[33].geom[1] << std::endl;
        // std::cout << "mj_data->contact34: " << mj_data->contact[34].geom[1] << std::endl;
        // std::cout << "mj_data->contact35: " << mj_data->contact[35].geom[1] << std::endl;
        // std::cout << "mj_data->contact36: " << mj_data->contact[36].geom[1] << std::endl;
        // std::cout << "mj_data->contact37: " << mj_data->contact[37].geom[1] << std::endl;
        // std::cout << "mj_data->contact38: " << mj_data->contact[38].geom[1] << std::endl;
        // std::cout << "mj_data->contact39: " << mj_data->contact[39].geom[1] << std::endl;



        // std::cout << "contact_check: " << contact_check << std::endl;
        // std::cout << "contact_check2: " << contact_check2 << std::endl;

        // std::cout << "contact_check2: ";
        // for (int element : contact_check2) {
        //     std::cout << element << " ";
        // }
        // std::cout << std::endl;


        State state;
        state.motor_position = qpos(Eigen::seqN(7, model::nu_size));
        state.motor_velocity = qvel(Eigen::seqN(6, model::nu_size));
        state.torque_estimate = qfrc_actuator(Eigen::seqN(6, model::nu_size));
        state.body_rotation = qpos(Eigen::seqN(3, 4));
        state.linear_body_velocity = qvel(Eigen::seqN(0, 3));
        state.angular_body_velocity = qvel(Eigen::seqN(3, 3));
        // state.contact_mask = Vector<model::contact_site_ids_size>::Constant(0.0);
        state.contact_mask = contact_check2;

        
        controller.update_state(state);
        
        // Update Taskspace Targets:
        TaskspaceTargets taskspace_targets = TaskspaceTargets::Zero();

        // Sinusoidal Position and Velocity Tracking:
        double amplitude = 0.04;
        double frequency = 0.1;

        //____________________________________________________________________________________________________________________________        
        //       shin z-axis height tracking - off
        //____________________________________________________________________________________________________________________________        
        // targets
        double shin_lin_vel = 0.2;        

        // shin linear position target
        Vector<3> tls_linear_target = Vector<3>(
            initial_site_data(1,0) + shin_lin_vel*current_time, initial_site_data(1,1), initial_site_data(1,2)
        );
        Vector<3> trs_linear_target = Vector<3>(
            initial_site_data(2,0) + shin_lin_vel*current_time, initial_site_data(2,1), initial_site_data(2,2)
        );
        Vector<3> hls_linear_target = Vector<3>(
            initial_site_data(3,0) + shin_lin_vel*current_time, initial_site_data(3,1), initial_site_data(3,2)
        );
        Vector<3> hrs_linear_target = Vector<3>(
            initial_site_data(4,0) + shin_lin_vel*current_time, initial_site_data(4,1), initial_site_data(4,2)
        );

        // shin linear position
        Vector<3> tls_linear_position = site_data(1,Eigen::seqN(0, 3));
        Vector<3> trs_linear_position = site_data(2,Eigen::seqN(0, 3));
        Vector<3> hls_linear_position = site_data(3,Eigen::seqN(0, 3));
        Vector<3> hrs_linear_position = site_data(4,Eigen::seqN(0, 3));

        // shin linear position error
        Vector<3> tls_linear_position_error = (tls_linear_target - tls_linear_position); 
        Vector<3> trs_linear_position_error = (trs_linear_target - trs_linear_position); 
        Vector<3> hls_linear_position_error = (hls_linear_target - hls_linear_position);
        Vector<3> hrs_linear_position_error = (hrs_linear_target - hrs_linear_position);

        // shin linear velocity
        Vector<3> tls_linear_velocity = (tls_linear_position - last_tls_linear_position)/(current_time - last_time);
        Vector<3> trs_linear_velocity = (trs_linear_position - last_trs_linear_position)/(current_time - last_time);
        Vector<3> hls_linear_velocity = (hls_linear_position - last_hls_linear_position)/(current_time - last_time);
        Vector<3> hrs_linear_velocity = (hrs_linear_position - last_hrs_linear_position)/(current_time - last_time);

        // shin linear velocity target
        Vector<3> tls_linear_velocity_target = Vector<3> (shin_lin_vel,0,0);
        Vector<3> trs_linear_velocity_target = Vector<3> (shin_lin_vel,0,0);
        Vector<3> hls_linear_velocity_target = Vector<3> (shin_lin_vel,0,0);
        Vector<3> hrs_linear_velocity_target = Vector<3> (shin_lin_vel,0,0);

        // shin linear velocity error
        Vector<3> tls_linear_velocity_error = (tls_linear_velocity_target - tls_linear_velocity);
        Vector<3> trs_linear_velocity_error = (trs_linear_velocity_target - trs_linear_velocity);
        Vector<3> hls_linear_velocity_error = (hls_linear_velocity_target - hls_linear_velocity);
        Vector<3> hrs_linear_velocity_error = (hrs_linear_velocity_target - hrs_linear_velocity);

        double shin_lin_kp = 100.0; 
        double shin_lin_kv = 10.0;

        Vector<3> tls_linear_control = shin_lin_kp * (tls_linear_position_error) + shin_lin_kv * (tls_linear_velocity_error);
        Vector<3> trs_linear_control = shin_lin_kp * (trs_linear_position_error) + shin_lin_kv * (trs_linear_velocity_error);
        Vector<3> hls_linear_control = shin_lin_kp * (hls_linear_position_error) + shin_lin_kv * (hls_linear_velocity_error);
        Vector<3> hrs_linear_control = shin_lin_kp * (hrs_linear_position_error) + shin_lin_kv * (hrs_linear_velocity_error);        

        Vector<3> last_tls_linear_position = tls_linear_position;
        Vector<3> last_trs_linear_position = trs_linear_position;
        Vector<3> last_hls_linear_position = hls_linear_position;
        Vector<3> last_hrs_linear_position = hrs_linear_position;     

        // ------------------------------------------------------------------------------------------------------------------------------------
        //       shin angular position - off - feedforward 1000 test
        // ------------------------------------------------------------------------------------------------------------------------------------
        // shin angular position
        // Sinusoidal Position and Velocity Tracking:
        double shin_rot_pos = 0.1*5.0;
        double shin_rot_vel = 0.1*8.0*5.0;
        // double shin_rot_frequency = 0.1;

        double tl_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[2]];        
        double tr_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[4]];        
        double hl_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[6]];        
        double hr_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[8]];        
        
        // double tl_angular_position = atan2(site_rotational_data(1,2),site_rotational_data(1,0));
        // double tr_angular_position = atan2(site_rotational_data(2,2),site_rotational_data(2,0));
        // double hl_angular_position = atan2(site_rotational_data(3,2),site_rotational_data(3,0));
        // double hr_angular_position = atan2(site_rotational_data(4,2),site_rotational_data(4,0));

        double tl_angular_position = tl_shin_angle;
        double tr_angular_position = tr_shin_angle;
        double hl_angular_position = hl_shin_angle;
        double hr_angular_position = hr_shin_angle;

        double tl_angular_velocity = (tl_angular_position - last_tl_angular_position)/(current_time - last_time);
        double tr_angular_velocity = (tr_angular_position - last_tr_angular_position)/(current_time - last_time);
        double hl_angular_velocity = (hl_angular_position - last_hl_angular_position)/(current_time - last_time);
        double hr_angular_velocity = (hr_angular_position - last_hr_angular_position)/(current_time - last_time);

        // targets
        double tl_angular_position_target = initial_tl_angular_position + shin_rot_vel * current_time;
        double tr_angular_position_target = initial_tr_angular_position + shin_rot_vel * current_time;
        double hl_angular_position_target = initial_hl_angular_position + shin_rot_vel * current_time;
        double hr_angular_position_target = initial_hr_angular_position + shin_rot_vel * current_time;

        // double tl_angular_position_target = wrapToPi(tl_angular_position + shin_rot_pos);
        // double tr_angular_position_target = wrapToPi(tr_angular_position + shin_rot_pos);
        // double hl_angular_position_target = wrapToPi(hl_angular_position + shin_rot_pos);
        // double hr_angular_position_target = wrapToPi(hr_angular_position + shin_rot_pos);

        
        // double tl_angular_position_target = (initial_tl_angular_position + shin_rot_vel * current_time);
        // double tr_angular_position_target = (initial_tr_angular_position + shin_rot_vel * current_time);
        // double hl_angular_position_target = (initial_hl_angular_position + shin_rot_vel * current_time);
        // double hr_angular_position_target = (initial_hr_angular_position + shin_rot_vel * current_time);
        

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

        // double shin_kp = 850; 
        // double shin_kv = 200;
        // double shin_kp = 10.0; 
        // double shin_kv = 800.0;

        double shin_kp = 800.0 * 3.0; 
        double shin_kv = 800.0 * 3.0;

        // double shin_kp = 1000.0; 
        // double shin_kv = 200.0;

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

        // Eigen::Vector<double, 6> cmd1 {tls_linear_control(0), 0, 0, 0, 0, 0};        
        // Eigen::Vector<double, 6> cmd2 {trs_linear_control(0), 0, 0, 0, 0, 0};        
        // Eigen::Vector<double, 6> cmd3 {hls_linear_control(0), 0, 0, 0, 0, 0};        
        // Eigen::Vector<double, 6> cmd4 {hrs_linear_control(0), 0, 0, 0, 0, 0};        


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
        //       thigh angular position - on // thigh linear position - on
        // ------------------------------------------------------------------------------------------------------------------------------------
        // thigh angular position
        // constant position and zero velocity Tracking:
        double thigh_rot_vel = 0.0;
        double thigh_kp = 1000.0;
        double thigh_kv = 100.0;

        // double tlh_angular_position = acos(site_rotational_data(5,0));
        // double trh_angular_position = acos(site_rotational_data(6,0));
        // double hlh_angular_position = acos(site_rotational_data(7,0));
        // double hrh_angular_position = acos(site_rotational_data(8,0));

        double tlh_angular_position = atan2(site_rotational_data(5,2),site_rotational_data(5,0));
        double trh_angular_position = atan2(site_rotational_data(6,2),site_rotational_data(6,0));
        double hlh_angular_position = atan2(site_rotational_data(7,2),site_rotational_data(7,0));
        double hrh_angular_position = atan2(site_rotational_data(8,2),site_rotational_data(8,0));

        // double tlh_angular_position_new = atan2(site_rotational_data(5,2),site_rotational_data(5,0));


        double tlh_angular_velocity = (tlh_angular_position - last_tlh_angular_position)/(current_time - last_time);
        double trh_angular_velocity = (trh_angular_position - last_trh_angular_position)/(current_time - last_time);
        double hlh_angular_velocity = (hlh_angular_position - last_hlh_angular_position)/(current_time - last_time);
        double hrh_angular_velocity = (hrh_angular_position - last_hrh_angular_position)/(current_time - last_time);

        // targets
        double tlh_angular_position_target = wrapToPi(initial_tlh_angular_position + M_PI/5.0 + thigh_rot_vel * current_time);
        double trh_angular_position_target = wrapToPi(initial_trh_angular_position + M_PI/5.0 + thigh_rot_vel * current_time);
        double hlh_angular_position_target = wrapToPi(initial_hlh_angular_position - M_PI/3.0 + thigh_rot_vel * current_time);
        double hrh_angular_position_target = wrapToPi(initial_hrh_angular_position - M_PI/3.0 + thigh_rot_vel * current_time);

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
        // double thigh_lin_kp = 400.0;
        // double thigh_lin_kv = 60.0;

        // double thigh_lin_kp = 4000.0 * 2.0;
        // double thigh_lin_kv = 600.0 * 2.0;

        double thigh_lin_kp = 4000.0 * 0.5;
        double thigh_lin_kv = 600.0 * 0.5;



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
        Vector<3> body_position = qpos(Eigen::seqN(0, 3));

        // double thigh_height_increase_stairs = (body_position(0)>1)*(body_position(0)-1 - 0.2)*(0.05/0.2);
        // double thigh_height_increase_stairs = (body_position(0)>0.5)*(body_position(0)-0.5)*(0.05/0.2);
        double thigh_height_increase_stairs = -0.025;
        // double thigh_height_increase_stairs = 0.0;

        double tlh_linear_position_error = ( (initial_site_data(5,2) - 0.0 + thigh_height_increase_stairs) - tlh_linear_position(2));
        double trh_linear_position_error = ( (initial_site_data(6,2) - 0.0 + thigh_height_increase_stairs) - trh_linear_position(2));
        double hlh_linear_position_error = ( (initial_site_data(7,2) - 0.0 + thigh_height_increase_stairs) - hlh_linear_position(2));
        double hrh_linear_position_error = ( (initial_site_data(8,2) - 0.0 + thigh_height_increase_stairs) - hrh_linear_position(2));

        
        // double tlh_linear_position_error = ( (initial_site_data(5,2) + 0.5*qpos(0) - 1.0) - tlh_linear_position(2));
        // double trh_linear_position_error = ( (initial_site_data(6,2) + 0.5*qpos(0) - 1.0) - trh_linear_position(2));
        // double hlh_linear_position_error = ( (initial_site_data(7,2) + 0.5*qpos(0) - 1.0) - hlh_linear_position(2));
        // double hrh_linear_position_error = ( (initial_site_data(8,2) + 0.5*qpos(0) - 1.0) - hrh_linear_position(2));

        // double tlh_linear_position_error = ( (initial_site_data(5,2) + 0.5*qpos(0) - 1.0) - tlh_linear_position(2));
        // double trh_linear_position_error = ( (initial_site_data(6,2) + 0.5*qpos(0) - 1.0) - trh_linear_position(2));
        // double hlh_linear_position_error = ( (initial_site_data(7,2) + 0.5*qpos(0) - 1.0) - hlh_linear_position(2));
        // double hrh_linear_position_error = ( (initial_site_data(8,2) + 0.5*qpos(0) - 1.0) - hrh_linear_position(2));
        
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

        Eigen::Vector<double, 6> cmd5 {0, 0, tlh_linear_control, 0, 0, 0};        
        Eigen::Vector<double, 6> cmd6 {0, 0, trh_linear_control, 0, 0, 0};        
        Eigen::Vector<double, 6> cmd7 {0, 0, hlh_linear_control, 0, 0, 0};        
        Eigen::Vector<double, 6> cmd8 {0, 0, hrh_linear_control, 0, 0, 0};        


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


        taskspace_targets.row(5) = cmd5;
        taskspace_targets.row(6) = cmd6;
        taskspace_targets.row(7) = cmd7;
        taskspace_targets.row(8) = cmd8;        


        // ------------------------------------------------------------------------------------------------------------------------------------
        //       track head height - off // orientation - on
        // ------------------------------------------------------------------------------------------------------------------------------------
        // initial_position(0), initial_position(1), initial_position(2)-0.02

        Vector<3> position_target = Vector<3>(
            initial_position(0) + 0.2*current_time, initial_position(1), initial_position(2)
        );
        Vector<3> velocity_target = Vector<3>(
            0.2,0.0,0.0
        );

        Eigen::Quaternion<double> body_rotation = Eigen::Quaternion<double>(state.body_rotation(0), state.body_rotation(1), state.body_rotation(2), state.body_rotation(3));
        // Vector<3> body_position = qpos(Eigen::seqN(0, 3));
        Vector<3> position_error = position_target - body_position;
        Vector<3> velocity_error = velocity_target - state.linear_body_velocity;
        // Vector<3> velocity_error = Vector<3>(0.0, 0.0, 0.0-state.linear_body_velocity(2));
        Vector<3> rotation_error = (Eigen::Quaternion<double>(1, 0, 0, 0) * body_rotation.conjugate()).vec();
        Vector<3> angular_velocity_error = Vector<3>::Zero() - state.angular_body_velocity;



        // double torso_lin_kp = 1000.0;
        // double torso_lin_kv = 10.0;

        double torso_lin_kp = 0.0;
        double torso_lin_kv = 0.0;

        // double torso_ang_kp = 100.0;
        // double torso_ang_kv = 100.0;

        // double torso_ang_kp = 10.0;
        // double torso_ang_kv = 10.0;

        // double torso_ang_kp = 5.0;
        // double torso_ang_kv = 1.0;        

        double torso_ang_kp = 0.0;
        double torso_ang_kv = 0.0;        

        Vector<3> linear_control = torso_lin_kp * (position_error) + torso_lin_kv * (velocity_error);
        Vector<3> angular_control = torso_ang_kp * (rotation_error) + torso_ang_kv * (angular_velocity_error);
        Eigen::Vector<double, 6> cmd {linear_control(0), 0, 0, angular_control(0), angular_control(1), angular_control(2)};
        taskspace_targets.row(0) = cmd;
        

        // ------------------------------------------------------------------------------------------------------------------------------------
        //       camera track head x
        // ------------------------------------------------------------------------------------------------------------------------------------
        cam.lookat[0] = body_position(0);
        // cam.lookat[4] = cam.lookat[4] + 0.01;


        // ------------------------------------------------------------------
        //       record tlh angular position
        // ------------------------------------------------------------------
        // std::cout << "tlh_angular_position_target: " << initial_position(2)-0.1 << std::endl;
        // std::cout << "contact9.dist: " << contact9.dist << std::endl;
        
        // target_tl_shin_data.push_back(position_target(2)); // body height data
        // tl_shin_data.push_back(body_position(2));
        // time_data.push_back(current_time);
        

        // shin angular position 
        data1_1.push_back(tl_angular_position);
        data1_2.push_back(tr_angular_position);
        data1_3.push_back(hl_angular_position);
        data1_4.push_back(hr_angular_position);
        data1t_1.push_back(tl_angular_position_target);
        data1t_2.push_back(tr_angular_position_target);
        data1t_3.push_back(hl_angular_position_target);
        data1t_4.push_back(hr_angular_position_target);

        // shin angular velocity
        data2_1.push_back(tl_angular_velocity);
        data2_2.push_back(tr_angular_velocity);
        data2_3.push_back(hl_angular_velocity);
        data2_4.push_back(hr_angular_velocity);
        data2t_1.push_back(tl_angular_velocity_target);
        data2t_2.push_back(tr_angular_velocity_target);
        data2t_3.push_back(hl_angular_velocity_target);
        data2t_4.push_back(hr_angular_velocity_target);

        // thigh z position
        data3_1.push_back(tlh_linear_position(2));
        data3_2.push_back(trh_linear_position(2));
        data3_3.push_back(hlh_linear_position(2));
        data3_4.push_back(hrh_linear_position(2));
        data3t_1.push_back(initial_site_data(5,2) - 0.0 + thigh_height_increase_stairs);
        data3t_2.push_back(initial_site_data(6,2) - 0.0 + thigh_height_increase_stairs);
        data3t_3.push_back(initial_site_data(7,2) - 0.0 + thigh_height_increase_stairs);
        data3t_4.push_back(initial_site_data(8,2) - 0.0 + thigh_height_increase_stairs);

        // thigh z velocity
        data4_1.push_back(tlh_linear_velocity(2));
        data4_2.push_back(trh_linear_velocity(2));
        data4_3.push_back(hlh_linear_velocity(2));
        data4_4.push_back(hrh_linear_velocity(2));
        data4t_1.push_back(tlh_linear_velocity_target);
        data4t_2.push_back(trh_linear_velocity_target);
        data4t_3.push_back(hlh_linear_velocity_target);
        data4t_4.push_back(hrh_linear_velocity_target);

        // rotation error
        data5_1.push_back(rotation_error(0));
        data5_2.push_back(rotation_error(1));
        data5_3.push_back(rotation_error(2));

        
        data5t_1.push_back(0);
        data5t_2.push_back(0);
        data5t_3.push_back(0);

        // angular velocity error
        data6_1.push_back(angular_velocity_error(0));
        data6_2.push_back(angular_velocity_error(1));
        data6_3.push_back(angular_velocity_error(2));
        data6t_1.push_back(0);
        data6t_2.push_back(0);
        data6t_3.push_back(0);

        // time
        data_time.push_back(current_time);


        // print joints and joint ids
        // for (int i = 0; i < mj_model->njnt; ++i) {
        //     // Get the joint ID (which is 'i' in this loop)
        //     int joint_id = i;

        //     // Get the joint name using mj_id2name
        //     // mj_id2name returns a const char*
        //     const char* joint_name = mj_id2name(mj_model, mjOBJ_JOINT, joint_id);

        //     // Print the ID and name
        //     if (joint_name) {
        //         std::cout << "ID: " << joint_id << ", Name: " << joint_name << std::endl;
        //     } else {
        //         // This case typically indicates an unnamed joint
        //         std::cout << "ID: " << joint_id << ", Name: (Unnamed Joint)" << std::endl;
        //     }
        // }

        // std::vector<int> wheel_sites_mujoco = {3, 4, 7, 8, 11, 12, 15, 16};
        findGeomsOnSameBodyAsSite(mj_model,3);
        findGeomsOnSameBodyAsSite(mj_model,4);
        findGeomsOnSameBodyAsSite(mj_model,7);
        findGeomsOnSameBodyAsSite(mj_model,8);
        findGeomsOnSameBodyAsSite(mj_model,11);
        findGeomsOnSameBodyAsSite(mj_model,12);
        findGeomsOnSameBodyAsSite(mj_model,15);
        findGeomsOnSameBodyAsSite(mj_model,16);

        controller.update_taskspace_targets(taskspace_targets);

        // Get Torque Command:
        Vector<model::nu_size> torque_command = controller.get_torque_command();


        data5t_1.push_back(torque_command[0]);
        data5t_2.push_back(torque_command[1]);
        data5t_3.push_back(torque_command[2]);

        data6t_1.push_back(torque_command[3]);
        data6t_2.push_back(torque_command[4]);
        data6t_3.push_back(torque_command[5]);        

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
    std::ofstream outfile("osc_test_slowtumbling.txt");
    if (outfile.is_open()) {
        
        outfile << "time data1_1 data1_2 data1_3 data1_4 data1t_1 data1t_2 data1t_3 data1t_4 "
            << "data2_1 data2_2 data2_3 data2_4 data2t_1 data2t_2 data2t_3 data2t_4 "
            << "data3_1 data3_2 data3_3 data3_4 data3t_1 data3t_2 data3t_3 data3t_4 "
            << "data4_1 data4_2 data4_3 data4_4 data4t_1 data4t_2 data4t_3 data4t_4 "
            << "data5_1 data5_2 data5_3 data5t_1 data5t_2 data5t_3 "
            << "data6_1 data6_2 data6_3 data6t_1 data6t_2 data6t_3 " << std::endl;

        for (size_t i = 0; i < data1_1.size(); ++i) {
            // Inside your loop where 'i' iterates from 0 to data1_1.size() - 1
            outfile << data_time[i] << " " << 
                       data1_1[i] << " " << data1_2[i] << " " << data1_3[i] << " " << data1_4[i] << " "
                    << data1t_1[i] << " " << data1t_2[i] << " " << data1t_3[i] << " " << data1t_4[i] << " "

                    << data2_1[i] << " " << data2_2[i] << " " << data2_3[i] << " " << data2_4[i] << " "
                    << data2t_1[i] << " " << data2t_2[i] << " " << data2t_3[i] << " " << data2t_4[i] << " "

                    << data3_1[i] << " " << data3_2[i] << " " << data3_3[i] << " " << data3_4[i] << " "
                    << data3t_1[i] << " " << data3t_2[i] << " " << data3t_3[i] << " " << data3t_4[i] << " "

                    << data4_1[i] << " " << data4_2[i] << " " << data4_3[i] << " " << data4_4[i] << " "
                    << data4t_1[i] << " " << data4t_2[i] << " " << data4t_3[i] << " " << data4t_4[i] << " "

                    << data5_1[i] << " " << data5_2[i] << " " << data5_3[i] << " "
                    << data5t_1[i] << " " << data5t_2[i] << " " << data5t_3[i] << " "

                << data6_1[i] << " " << data6_2[i] << " " << data6_3[i]  << " "
                << data6t_1[i] << " " << data6t_2[i] << " " << data6t_3[i]  << std::endl;            
        }
        outfile.close();
        std::cout << "Data saved to data.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }

    return 0;
}