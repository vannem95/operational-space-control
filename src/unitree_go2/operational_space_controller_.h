#include <filesystem>
#include <vector>
#include <string>
#include <cstdlib>

#include "mujoco/mujoco.h"
#include "Eigen/Dense"

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

    template <int Rows_, int Cols_>
    using Matrix = Eigen::Matrix<double, Rows_, Cols_, Eigen::RowMajor>;

    template <int Rows_>
    using Vector = Eigen::Matrix<double, Rows_, 1>;
}

struct OSCData {
    Matrix<model::nv_size, model::nv_size> mass_matrix;    
    Vector<model::nv_size> coriolis_matrix;
    Matrix<model::nv_size, optimization::z_size> contact_jacobian;
    Matrix<s_size, model::nv_size> taskspace_jacobian;
    Matrix<model::body_ids_size, 6> taskspace_bias;
    Vector<model::contact_site_ids_size> contact_mask;
    Vector<model::nq_size> previous_q;
    Vector<model::nv_size> previous_qd;
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
            for(const std::string_view& body : model::body_list) {
                std::string body_str = std::string(body);
                int id = mj_name2id(mj_model, mjOBJ_BODY, body_str.data());
                assert(id != -1 && "Body not found in model.");
                bodies.push_back(body_str);
                body_ids.push_back(id);
            }
            // Assert Number of Sites and Bodies are equal:
            assert(site_ids.size() == body_ids.size() && "Number of Sites and Bodies must be equal.");
        }

        void close() {
            mj_deleteData(mj_data);
            mj_deleteModel(mj_model);
        }

        OSCData get_data(Eigen::Matrix<double, model::body_ids_size, 3, Eigen::RowMajor>& points) {
            // Mass Matrix:
            Matrix<model::nv_size, model::nv_size> mass_matrix = 
                Matrix<model::nv_size, model::nv_size>::Zero();
            mj_fullM(mj_model, mass_matrix.data(), mj_data->qM);

            // Coriolis Matrix:
            Vector<model::nv_size> coriolis_matrix = 
                Eigen::Map<Vector<model::nv_size>>(mj_data->qfrc_bias);

            // Generalized Positions and Velocities:
            Vector<model::nq_size> generalized_positions = 
                Eigen::Map<Vector<model::nq_size> >(mj_data->qpos);
            Vector<model::nv_size> generalized_velocities = 
                Eigen::Map<Vector<model::nv_size>>(mj_data->qvel);

            // Jacobian Calculation:
            Matrix<p_size, model::nv_size> jacobian_translation = 
                Matrix<p_size, model::nv_size>::Zero();
            Matrix<r_size, model::nv_size> jacobian_rotation = 
                Matrix<r_size, model::nv_size>::Zero();
            Matrix<p_size, model::nv_size> jacobian_dot_translation = 
                Matrix<p_size, model::nv_size>::Zero();
            Matrix<r_size, model::nv_size> jacobian_dot_rotation = 
                Matrix<r_size, model::nv_size>::Zero();
            for (int i = 0; i < model::body_ids_size; i++) {
                // Temporary Jacobian Matrices:
                Matrix<3, model::nv_size> jacp = Matrix<3, model::nv_size>::Zero();
                Matrix<3, model::nv_size> jacr = Matrix<3, model::nv_size>::Zero();
                Matrix<3, model::nv_size> jacp_dot = Matrix<3, model::nv_size>::Zero();
                Matrix<3, model::nv_size> jacr_dot = Matrix<3, model::nv_size>::Zero();

                // Calculate Jacobian:
                mj_jac(mj_model, mj_data, jacp.data(), jacr.data(), points.row(i).data(), body_ids[i]);

                // Calculate Jacobian Dot:
                mj_jacDot(mj_model, mj_data, jacp_dot.data(), jacr_dot.data(), points.row(i).data(), body_ids[i]);

                // Append to Jacobian Matrices:
                int row_offset = i * 3;
                for(int row_idx = 0; row_idx < 3; row_idx++) {
                    for(int col_idx = 0; col_idx < model::nv_size; col_idx++) {
                        jacobian_translation(row_idx + row_offset, col_idx) = jacp(row_idx, col_idx);
                        jacobian_rotation(row_idx + row_offset, col_idx) = jacr(row_idx, col_idx);
                        jacobian_dot_translation(row_idx + row_offset, col_idx) = jacp_dot(row_idx, col_idx);
                        jacobian_dot_rotation(row_idx + row_offset, col_idx) = jacr_dot(row_idx, col_idx);
                    }
                }
            }

            // Stack Jacobian Matrices: Taskspace Jacobian: [jacp; jacr], Jacobian Dot: [jacp_dot; jacr_dot]
            Matrix<s_size, model::nv_size> taskspace_jacobian = Matrix<s_size, model::nv_size>::Zero();
            Matrix<s_size, model::nv_size> jacobian_dot = Matrix<s_size, model::nv_size>::Zero();
            int row_offset = model::body_ids_size * 3;
            for(int row_idx = 0; row_idx < model::body_ids_size * 3; row_idx++) {
                for(int col_idx = 0; col_idx < model::nv_size; col_idx++) {
                    taskspace_jacobian(row_idx, col_idx) = jacobian_translation(row_idx, col_idx);
                    taskspace_jacobian(row_idx + row_offset, col_idx) = jacobian_rotation(row_idx, col_idx);
                    jacobian_dot(row_idx, col_idx) = jacobian_dot_translation(row_idx, col_idx);
                    jacobian_dot(row_idx + row_offset, col_idx) = jacobian_dot_rotation(row_idx, col_idx);
                }
            }

            // Calculate Taskspace Bias Acceleration:
            Vector<s_size> bias = Vector<s_size>::Zero();
            bias = jacobian_dot * generalized_velocities;
            // Reshape leading axis -> num_body_ids x 6
            Matrix<model::body_ids_size, 6> taskspace_bias = bias.reshaped<Eigen::RowMajor>(model::body_ids_size, 6);

            // Contact Jacobian: Shape (NV, 3 * num_contacts) 
            // This assumes contact frames are the last rows of the translation component of the taskspace_jacobian (jacobian_translation).
            // contact_jacobian = jacobian_translation[end-(3 * contact_site_ids_size):end, :].T
            Matrix<model::nv_size, optimization::z_size> contact_jacobian = 
                Matrix<model::nv_size, optimization::z_size>::Zero();

            contact_jacobian = jacobian_translation(
                Eigen::seq(Eigen::placeholders::end - Eigen::fix<optimization::z_size>, Eigen::placeholders::last),
                Eigen::placeholders::all
            ).transpose();

            // Contact Mask: Shape (num_contacts, 1)
            Vector<model::contact_site_ids_size> contact_mask = Vector<model::contact_site_ids_size>::Zero();
            double contact_threshold = 1e-3;
            for(int i = 0; i < model::contact_site_ids_size; i++) {
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
            std::vector<std::string> sites;
            std::vector<std::string> bodies;
            std::vector<std::string> noncontact_sites;
            std::vector<std::string> contact_sites;
            std::vector<int> site_ids;
            std::vector<int> noncontact_site_ids;
            std::vector<int> contact_site_ids;
            std::vector<int> body_ids;

};
