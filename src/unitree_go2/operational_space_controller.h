#include <filesystem>
#include <vector>
#include <string>
#include <cstdlib>

#include "mujoco/mujoco.h"
#include "Eigen/Dense"

#include "src/utilities.h"

struct OSCData {
    Matrix mass_matrix;    
    Eigen::VectorXd coriolis_matrix;
    Matrix contact_jacobian;
    Matrix taskspace_jacobian;
    Matrix taskspace_bias;
    Eigen::VectorXd contact_mask;
    Eigen::VectorXd previous_q;
    Eigen::VectorXd previous_qd;
};

class OperationalSpaceController {
    public:
        OperationalSpaceController() {}
        ~OperationalSpaceController() {}

        void initialize(std::filesystem::path xml_path) {
            char error[1000];
            model = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
            if( !model ) {
                printf("%s\n", error);
                std::exit(EXIT_FAILURE);
            }
            // Physics timestep:
            model->opt.timestep = 0.002;
            
            data = mj_makeData(model);

            for(const std::string& site : sites){
                site_ids.push_back(mj_name2id(model, mjOBJ_SITE, site.c_str()));
            }
            for(const std::string& body : bodies){
                body_ids.push_back(mj_name2id(model, mjOBJ_BODY, body.c_str()));
            }
            // Assert Number of Sites and Bodies are equal:
            assert(site_ids.size() == body_ids.size() && "Number of Sites and Bodies must be equal.");
            num_body_ids = body_ids.size();
        }

        void close() {
            mj_deleteData(data);
            mj_deleteModel(model);
        }

        OSCData get_data(Matrix& points) {
            // Mass Matrix:
            Matrix mass_matrix = Matrix::Zero(model->nv, model->nv);
            // double *mass_matrix_data = mass_matrix.data();
            mj_fullM(model, mass_matrix.data(), data->qM);
            // mj_fullM(model, mass_matrix_data, data->qM);
            // mass_matrix = MapMatrix(mass_matrix_data, model->nv, model->nv);

            // Coriolis Matrix:
            Matrix coriolis_matrix = Eigen::Map<Eigen::VectorXd>(data->qfrc_bias, model->nv);

            // Joint Position and Velocity:
            Eigen::VectorXd joint_position = Eigen::Map<Eigen::VectorXd>(data->qpos, model->nq);
            Eigen::VectorXd joint_velocity = Eigen::Map<Eigen::VectorXd>(data->qvel, model->nv);

            // Jacobian Calculation:
            Matrix jacobian_translation = Matrix::Zero(num_body_ids * 3, model->nv);
            Matrix jacobian_rotation = Matrix::Zero(num_body_ids * 3, model->nv);
            Matrix jacobian_dot_translation = Matrix::Zero(num_body_ids * 3, model->nv);
            Matrix jacobian_dot_rotation = Matrix::Zero(num_body_ids * 3, model->nv);
            for (int i = 0; i < num_body_ids; i++) {
                // Temporary Jacobian Matrices:
                Matrix jacp = Matrix::Zero(3, model->nv);
                Matrix jacr = Matrix::Zero(3, model->nv);
                Matrix jacp_dot = Matrix::Zero(3, model->nv);
                Matrix jacr_dot = Matrix::Zero(3, model->nv);

                // Points Data:
                // double* point_data = points.row(i).data();

                // Calculate Jacobian:
                // double *jacp_data = jacp.data();
                // double *jacr_data = jacr.data();
                mj_jac(model, data, jacp.data(), jacr.data(), points.row(i).data(), body_ids[i]);
                // mj_jac(model, data, jacp_data, jacr_data, point_data, body_ids[i]);
                // jacp = MapMatrix(jacp_data, 3, model->nv);
                // jacr = MapMatrix(jacr_data, 3, model->nv);

                // Calculate Jacobian Dot:
                // double *jacp_dot_data = jacp_dot.data();
                // double *jacr_dot_data = jacr_dot.data();
                mj_jacDot(model, data, jacp_dot.data(), jacr_dot.data(), points.row(i).data(), body_ids[i]);
                // mj_jacDot(model, data, jacp_dot_data, jacr_dot_data, point_data, body_ids[i]);
                // jacp_dot = MapMatrix(jacp_dot_data, 3, model->nv);
                // jacr_dot = MapMatrix(jacr_dot_data, 3, model->nv);

                // Append to Jacobian Matrices:
                int row_offset = i * 3;
                for(int row_idx = 0; row_idx < 3; row_idx++) {
                    for(int col_idx = 0; col_idx < model->nv; col_idx++) {
                        jacobian_translation(row_idx + row_offset, col_idx) = jacp(row_idx, col_idx);
                        jacobian_rotation(row_idx + row_offset, col_idx) = jacr(row_idx, col_idx);
                        jacobian_dot_translation(row_idx + row_offset, col_idx) = jacp_dot(row_idx, col_idx);
                        jacobian_dot_rotation(row_idx + row_offset, col_idx) = jacr_dot(row_idx, col_idx);
                    }
                }
            }

            // Stack Jacobian Matrices: Taskspace Jacobian: [jacp; jacr], Jacobian Dot: [jacp_dot; jacr_dot]
            Matrix taskspace_jacobian = Matrix::Zero(num_body_ids * 6, model->nv);
            Matrix jacobian_dot = Matrix::Zero(num_body_ids * 6, model->nv);
            int row_offset = num_body_ids * 3;
            for(int row_idx = 0; row_idx < num_body_ids * 3; row_idx++) {
                for(int col_idx = 0; col_idx < model->nv; col_idx++) {
                    taskspace_jacobian(row_idx, col_idx) = jacobian_translation(row_idx, col_idx);
                    taskspace_jacobian(row_idx + row_offset, col_idx) = jacobian_rotation(row_idx, col_idx);
                    jacobian_dot(row_idx, col_idx) = jacobian_dot_translation(row_idx, col_idx);
                    jacobian_dot(row_idx + row_offset, col_idx) = jacobian_dot_rotation(row_idx, col_idx);
                }
            }

            // Calculate Taskspace Bias Acceleration:
            Matrix bias = Matrix::Zero(num_body_ids * 6, 1);
            bias = jacobian_dot * joint_velocity;
            // Reshape leading axis -> num_body_ids x 6
            Matrix taskspace_bias = bias.reshaped<Eigen::RowMajor>(num_body_ids, 6);

            // Contact Jacobian: Shape (NV, 3 * num_contacts) 
            // TODO(jeh15): This assumes the contact frames come directly after the body frame...
            Matrix contact_jacobian = Matrix::Zero(model->nv, num_contacts * 3);
            contact_jacobian = taskspace_jacobian(Eigen::seqN(3, num_contacts * 3), Eigen::placeholders::all)
                .transpose();

            // Contact Mask: Shape (num_contacts, 1)
            Eigen::VectorXd contact_mask = Eigen::VectorXd::Zero(num_contacts);
            double contact_threshold = 1e-3;
            for(int i = 0; i < num_contacts; i++) {
                auto contact = data->contact[i];
                contact_mask(i) = contact.dist < contact_threshold;
            }

            OSCData osc_data{
                .mass_matrix = mass_matrix,
                .coriolis_matrix = coriolis_matrix,
                .contact_jacobian = contact_jacobian,
                .taskspace_jacobian = taskspace_jacobian,
                .taskspace_bias = taskspace_bias,
                .contact_mask = contact_mask,
                .previous_q = joint_position,
                .previous_qd = joint_velocity  
            };

            return osc_data;
        }

        // Chnage this to private after testing:
        public:
            mjModel* model;
            mjData* data;
            std::vector<std::string> sites = {
                "imu", "front_left_foot", "front_right_foot", "hind_left_foot", "hind_right_foot"
            };
            std::vector<std::string> bodies = {
                "base_link", "front_left_calf", "front_right_calf", "hind_left_calf", "hind_right_calf"
            };
            std::vector<int> site_ids;
            std::vector<int> body_ids;
            int num_body_ids;
            const int num_contacts = 4; // num_contacts should match the number of sites intended to be used as contact points and preferably the xml is setup such that data->ncon also equals this value.

};
