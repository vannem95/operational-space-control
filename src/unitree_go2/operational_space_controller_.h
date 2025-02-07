#include <filesystem>
#include <vector>
#include <string>
#include <cstdlib>

#include "mujoco/mujoco.h"
#include "Eigen/Dense"

#include "src/utilities.h"
#include "src/unitree_go2/autogen/autogen_defines.h"

using namespace constants;

// Anonymous Namespace for shorthand constants:
namespace {
    // s_size : Size of fully spatial vector representation for all bodies
    constexpr int s_size = 6 * model::body_ids_size;
    // p_size : Size of translation component to a spatial vector:
    constexpr int p_size = 3 * model::body_ids_size;
    // r_size : Size of rotation component to a spatial vector:
    constexpr int r_size = 3 * model::body_ids_size;
}

struct OSCData {
    Eigen::Matrix<double, model::nv_size, model::nv_size, Eigen::RowMajor> mass_matrix;    
    Eigen::Vector<double, model::nv_size> coriolis_matrix;
    Eigen::Matrix<double, model::nv_size, optimization::z_size, Eigen::RowMajor> contact_jacobian;
    Eigen::Matrix<double, s_size, s_size, Eigen::RowMajor> taskspace_jacobian;
    Eigen::Matrix<double, model::body_ids_size, 6, Eigen::RowMajor> taskspace_bias;
    Eigen::Vector<double, model::contact_site_ids_size> contact_mask;
    Eigen::Vector<double, model::nq_size> previous_q;
    Eigen::Vector<double, model::nv_size> previous_qd;
};

class OperationalSpaceController {
    public:
        OperationalSpaceController() {}
        ~OperationalSpaceController() {}

        void initialize(std::filesystem::path xml_path) {
            char error[1000];
            mj_model = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
            if( !mj_model ) {
                printf("%s\n", error);
                std::exit(EXIT_FAILURE);
            }
            // Physics timestep:
            mj_model->opt.timestep = 0.002;
            
            mj_data = mj_makeData(mj_model);

            for(const std::string& site : sites){
                site_ids.push_back(mj_name2id(mj_model, mjOBJ_SITE, site.c_str()));
            }
            for(const std::string& body : bodies){
                body_ids.push_back(mj_name2id(mj_model, mjOBJ_BODY, body.c_str()));
            }
            // Assert Number of Sites and Bodies are equal:
            assert(site_ids.size() == body_ids.size() && "Number of Sites and Bodies must be equal.");
            num_body_ids = body_ids.size();
        }

        void close() {
            mj_deleteData(mj_data);
            mj_deleteModel(mj_model);
        }

        OSCData get_data(Eigen::Matrix<double, model::body_ids_size, 3>& points) {
            // Mass Matrix:
            Eigen::Matrix<double, model::nv_size, model::nv_size, Eigen::RowMajor> mass_matrix = 
                Eigen::Matrix<double, model::nv_size, model::nv_size, Eigen::RowMajor>::Zero();
            mj_fullM(mj_model, mass_matrix.data(), mj_data->qM);

            // Coriolis Matrix:
            Eigen::Vector<double, model::nv_size> coriolis_matrix = 
                Eigen::Map<Eigen::Vector<double, model::nv_size>>(mj_data->qfrc_bias);

            // Generalized Positions and Velocities:
            Eigen::Vector<double, model::nq_size> generalized_positions = 
                Eigen::Map<Eigen::Vector<double, model::nq_size> >(mj_data->qpos);
            Eigen::Vector<double, model::nv_size> generalized_velocities = 
                Eigen::Map<Eigen::Vector<double, model::nv_size>>(mj_data->qvel);

            // Jacobian Calculation:
            Eigen::Matrix<double, p_size, model::nv_size> jacobian_translation = 
                Eigen::Matrix<double, p_size, model::nv_size>::Zero();
            Eigen::Matrix<double, r_size, model::nv_size> jacobian_rotation = 
                Eigen::Matrix<double, r_size, model::nv_size>::Zero();
            Eigen::Matrix<double, p_size, model::nv_size> jacobian_dot_translation = 
                Eigen::Matrix<double, p_size, model::nv_size>::Zero();
            Eigen::Matrix<double, r_size, model::nv_size> jacobian_dot_rotation = 
                Eigen::Matrix<double, r_size, model::nv_size>::Zero();
            for (int i = 0; i < model::body_ids_size; i++) {
                // Temporary Jacobian Matrices:
                Eigen::Matrix<double, 3, model::nv_size> jacp = Eigen::Matrix<double, 3, model::nv_size>::Zero();
                Eigen::Matrix<double, 3, model::nv_size> jacr = Eigen::Matrix<double, 3, model::nv_size>::Zero();
                Eigen::Matrix<double, 3, model::nv_size> jacp_dot = Eigen::Matrix<double, 3, model::nv_size>::Zero();
                Eigen::Matrix<double, 3, model::nv_size> jacr_dot = Eigen::Matrix<double, 3, model::nv_size>::Zero();

                // Calculate Jacobian:
                mj_jac(mj_model, mj_data, jacp.data(), jacr.data(), points.row(i).data(), body_ids[i]);

                // Calculate Jacobian Dot:
                mj_jacDot(mj_model, mj_data, jacp_dot.data(), jacr_dot.data(), points.row(i).data(), body_ids[i]);

                // Append to Jacobian Matrices:
                int row_offset = i * 3;
                for(int row_idx = 0; row_idx < 3; row_idx++) {
                    for(int col_idx = 0; col_idx < mj_model->nv; col_idx++) {
                        jacobian_translation(row_idx + row_offset, col_idx) = jacp(row_idx, col_idx);
                        jacobian_rotation(row_idx + row_offset, col_idx) = jacr(row_idx, col_idx);
                        jacobian_dot_translation(row_idx + row_offset, col_idx) = jacp_dot(row_idx, col_idx);
                        jacobian_dot_rotation(row_idx + row_offset, col_idx) = jacr_dot(row_idx, col_idx);
                    }
                }
            }

            // Stack Jacobian Matrices: Taskspace Jacobian: [jacp; jacr], Jacobian Dot: [jacp_dot; jacr_dot]
            Matrix taskspace_jacobian = Matrix::Zero(num_body_ids * 6, mj_model->nv);
            Matrix jacobian_dot = Matrix::Zero(num_body_ids * 6, mj_model->nv);
            int row_offset = num_body_ids * 3;
            for(int row_idx = 0; row_idx < num_body_ids * 3; row_idx++) {
                for(int col_idx = 0; col_idx < mj_model->nv; col_idx++) {
                    taskspace_jacobian(row_idx, col_idx) = jacobian_translation(row_idx, col_idx);
                    taskspace_jacobian(row_idx + row_offset, col_idx) = jacobian_rotation(row_idx, col_idx);
                    jacobian_dot(row_idx, col_idx) = jacobian_dot_translation(row_idx, col_idx);
                    jacobian_dot(row_idx + row_offset, col_idx) = jacobian_dot_rotation(row_idx, col_idx);
                }
            }

            // Calculate Taskspace Bias Acceleration:
            Matrix bias = Matrix::Zero(num_body_ids * 6, 1);
            bias = jacobian_dot * generalized_velocities;
            // Reshape leading axis -> num_body_ids x 6
            Matrix taskspace_bias = bias.reshaped<Eigen::RowMajor>(num_body_ids, 6);

            // Contact Jacobian: Shape (NV, 3 * num_contacts) 
            // TODO(jeh15): This assumes the contact frames come directly after the body frame...
            Matrix contact_jacobian = Matrix::Zero(mj_model->nv, num_contacts * 3);
            contact_jacobian = taskspace_jacobian(Eigen::seqN(3, num_contacts * 3), Eigen::placeholders::all)
                .transpose();

            // Contact Mask: Shape (num_contacts, 1)
            Eigen::VectorXd contact_mask = Eigen::VectorXd::Zero(num_contacts);
            double contact_threshold = 1e-3;
            for(int i = 0; i < num_contacts; i++) {
                auto contact = mj_data->contact[i];
                contact_mask(i) = contact.dist < contact_threshold;
            }

            OSCData osc_data{
                .mass_matrix = mass_matrix,
                .coriolis_matrix = coriolis_matrix,
                .contact_jacobian = contact_jacobian,
                .taskspace_jacobian = taskspace_jacobian,
                .taskspace_bias = taskspace_bias,
                .contact_mask = contact_mask,
                .previous_q = generalized_positions,
                .previous_qd = generalized_velocities  
            };

            return osc_data;
        }

        // Chnage this to private after testing:
        public:
            mjModel* mj_model;
            mjData* mj_data;
            std::vector<std::string> sites = {
                "imu", "front_left_foot", "front_right_foot", "hind_left_foot", "hind_right_foot"
            };
            std::vector<std::string> bodies = {
                "base_link", "front_left_calf", "front_right_calf", "hind_left_calf", "hind_right_calf"
            };
            std::vector<int> site_ids;
            std::vector<int> body_ids;
            int num_body_ids;
            const int num_contacts = 4; // num_contacts should match the number of sites intended to be used as contact points and preferably the xml is setup such that mj_data->ncon also equals this value.

};
