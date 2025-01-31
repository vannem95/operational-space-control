#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "Eigen/CX11/Tensor"
#include "stdio.h"
#include "string.h"


class OperationSpaceController {
    public:
        OperationSpaceController();
        ~OperationSpaceController();
        OSCData get_data(MatrixXd points) {
            // Mass Matrix:
            Eigen::MatrixXd mass_matrix = Eigen::Map<Eigen::MatrixXd>(data->qM, model->nv, model->nv);

            // Coriolis Matrix:
            Eigen::MatrixXd coriolis_matrix = Eigen::Map<Eigen::MatrixXd>(data->qfrc_bias, model->nv, model->nv);

            // Joint Velocity:
            Eigen::VectorXd joint_velocity = Eigen::Map<Eigen::VectorXd>(data->qvel, model->nv);

            // Jacobian Calculation:
            Eigen::MatrixXd jacobian_translation = Eigen::MatrixXd::Zero(num_body_ids * 3, model->nv);
            Eigen::MatrixXd jacobian_rotation = Eigen::MatrixXd::Zero(num_body_ids * 3, model->nv);
            Eigen::MatrixXd jacobian_dot_translation = Eigen::MatrixXd::Zero(num_body_ids * 3, model->nv);
            Eigen::MatrixXd jacobian_dot_rotation = Eigen::MatrixXd::Zero(num_body_ids * 3, model->nv);
            for (int i = 0; i < num_body_ids; i++) {
                // Temporary Jacobian Matrices:
                Eigen::MatrixXd jacp = Eigen::MatrixXd::Zero(3, model->nv);
                Eigen::MatrixXd jacr = Eigen::MatrixXd::Zero(3, model->nv);
                Eigen::MatrixXd jacp_dot = Eigen::MatrixXd::Zero(3, model->nv);
                Eigen::MatrixXd jacr_dot = Eigen::MatrixXd::Zero(3, model->nv);

                // Calculate Jacobian:
                mj_jac(model, data, jacp.data(), jacr.data(), points.row(i).data(), body_ids[i]);

                // Calculate Jacobian Dot:
                mj_jacDot(model, data, jacp_dot.data(), jacr_dot.data(), points.row(i).data(), body_ids[i]);

                // Append to Jacobian Matrices:
                int row_idx = i * 3;
                jacobian_translation.block<3, model->nv>(row_idx, 0) = jacp;
                jacobian_rotation.block<3, model->nv>(row_idx, 0) = jacr;
                jacobian_dot_translation.block<3, model->nv>(row_idx, 0) = jacp_dot;
                jacobian_dot_rotation.block<3, model->nv>(row_idx, 0) = jacr_dot;
            }

            // Stack Jacobian Matrices:
            Eigen::MatrixXd taskspace_jacobian = Eigen::MatrixXd::Zero(num_body_ids * 6, model->nv);
            Eigen::MatrixXd jacobian_dot = Eigen::MatrixXd::Zero(num_body_ids * 6, model->nv);
            taskspace_jacobian.block<num_body_ids * 3, model->nv>(0, 0) = jacobian_translation;
            taskspace_jacobian.block<num_body_ids * 3, model->nv>(num_body_ids * 3, 0) = jacobian_rotation;
            jacobian_dot.block<num_body_ids * 3, model->nv>(0, 0) = jacobian_dot_translation;
            jacobian_dot.block<num_body_ids * 3, model->nv>(num_body_ids * 3, 0) = jacobian_dot_rotation;

            // Calculate Taskspace Bias Acceleration:
            Eigen::MatrixXd taskspace_bias = Eigen::MatrixXd::Zero(num_body_ids * 6, 1);
            taskspace_bias = jacobian_dot * joint_velocity;
            // Reshape leading axis -> num_body_ids x 6
            taskspace_bias = taskspace_bias.transpose().reshape(num_body_ids, 6);


        }

        const int num_body_ids = 5;
};