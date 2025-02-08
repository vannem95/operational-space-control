import os
from absl import app

import yaml

import numpy as np
import casadi
import mujoco

from casadi import MX, DM


class AutoGen():
    def __init__(self, mj_model: mujoco.MjModel):
        self.mj_model = mj_model

        # Parse Configuration YAML File:
        with open("config/unitree_go2/unitree_go2_config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Get Weight Configuration:
        self.weights_config = config['weights_config']

        # Get Body and Site IDs:
        self.body_list = [f'"{body}"sv' for body in config['body_list']]
        self.noncontact_site_list = [f'"{noncontact_site}"sv' for noncontact_site in config['noncontact_site_list']]
        self.contact_site_list = [f'"{contact_site}"sv' for contact_site in config['contact_site_list']]
        self.site_list = self.noncontact_site_list + self.contact_site_list
        self.num_body_ids = len(self.body_list)
        self.num_site_ids = len(self.site_list)
        self.num_noncontact_site_ids = len(self.noncontact_site_list)
        self.num_contact_site_ids = len(self.contact_site_list)
        self.mu = config['friction_coefficient']

        assert self.num_body_ids == self.num_site_ids, "Number of body IDs and site IDs must be equal."

        self.dv_size = self.mj_model.nv
        self.u_size = self.mj_model.nu
        self.z_size = self.num_contact_site_ids * 3
        self.design_vector_size = self.dv_size + self.u_size + self.z_size

        self.dv_idx = self.dv_size
        self.u_idx = self.dv_idx + self.u_size
        self.z_idx = self.u_idx + self.z_size

        self.B: DM = casadi.vertcat(
            DM.zeros((6, self.u_size)),
            DM.eye(self.u_size),
        )

    def equality_constraints(
        self,
        q: MX,
        M: MX,
        C: MX,
        J_contact: MX,
    ) -> MX:
        """Equality constraints for the dynamics of a system.

        Args:
            q: design vector.
            M: The mass matrix.
            C: The Coriolis matrix.
            J_contact: Contact Jacobian.
            split_indx: The index at which the optimization
                variables are split in dv and u.

        Returns:
            The equality constraints.

            Dynamics:
            M @ dv + C - B @ u - J_contact.T @ z = 0
        """
        # Unpack Design Variables:
        dv = q[:self.dv_idx]
        u = q[self.dv_idx:self.u_idx]
        z = q[self.u_idx:self.z_idx]

        # Dynamics:
        equality_constraints = M @ dv + C - self.B @ u - J_contact @ z

        return equality_constraints

    def inequality_constraints(
        self,
        q: MX,
    ) -> MX:
        """Compute inequality constraints for the Operational Space Controller.

        Args:
            q: design vector.

        Returns:
            MX: Inequality constraints.

            Friction Cone Constraints:
            |f_x| + |f_y| <= mu * f_z

        """
        # Unpack Design Variables:
        dv = q[:self.dv_idx] 
        u = q[self.dv_idx:self.u_idx]
        z = q[self.u_idx:self.z_idx]

        def translational_friction(x: MX) -> MX:
            constraint_1 = x[0] + x[1] - self.mu * x[2]
            constraint_2 = -x[0] + x[1] - self.mu * x[2]
            constraint_3 = x[0] - x[1] - self.mu * x[2]
            constraint_4 = -x[0] - x[1] - self.mu * x[2]
            return casadi.vertcat(
                constraint_1, constraint_2, constraint_3, constraint_4
            )

        contact_forces = casadi.reshape(z, self.num_contact_site_ids, 3)

        inequality_constraints = []
        for i in range(self.num_contact_site_ids):
            contact_force = contact_forces[i, :]
            friction_constraints = translational_friction(contact_force)
            inequality_constraints.append(friction_constraints)

        inequality_constraints = casadi.vertcat(*inequality_constraints)

        return inequality_constraints

    def objective(
        self,
        q: MX,
        desired_task_ddx: MX,
        J_task: MX,
        task_bias: MX,
    ) -> MX:
        """Compute the Task Space Tracking Objective.

        Args:
            q: Design vector.
            desired_task_ddx: Desired task acceleration.
            J_task: Taskspace Jacobian.
            task_bias: Taskspace bias acceleration.

        Returns:
            MX: Objective function.

        """
        # Unpack Design Variables:
        dv = q[:self.dv_idx]
        u = q[self.dv_idx:self.u_idx]
        z = q[self.u_idx:self.z_idx]

        # Compute Task Space Tracking Objective:
        ddx_task = J_task @ dv + task_bias

        # Split into Translational and Rotational components:
        ddx_task_p, ddx_task_r = casadi.vertsplit_n(ddx_task, 2)
        ddx_base_p, ddx_fl_p, ddx_fr_p, ddx_hl_p, ddx_hr_p = casadi.vertsplit_n(ddx_task_p, self.num_site_ids)
        ddx_base_r, ddx_fl_r, ddx_fr_r, ddx_hl_r, ddx_hr_r = casadi.vertsplit_n(ddx_task_r, self.num_site_ids)

        # Split Desired Task Acceleration:
        desired_task_p, desired_task_r = casadi.vertsplit_n(desired_task_ddx, 2)
        desired_base_p, desired_fl_p, desired_fr_p, desired_hl_p, desired_hr_p = casadi.vertsplit_n(desired_task_p, self.num_site_ids)
        desired_base_r, desired_fl_r, desired_fr_r, desired_hl_r, desired_hr_r = casadi.vertsplit_n(desired_task_r, self.num_site_ids)

        # I could make this more general at the cost of readability...
        # ddx_task_p = casadi.vertsplit_n(ddx_task_p, self.num_site_ids)
        # ddx_task_r = casadi.vertsplit_n(ddx_task_r, self.num_site_ids)
        # desired_task_p = casadi.vertsplit_n(desired_task_p, self.num_site_ids)
        # desired_task_r = casadi.vertsplit_n(desired_task_r, self.num_site_ids)

        objective_terms = {
            'base_translational_tracking': self._objective_tracking(
                ddx_base_p,
                desired_base_p,
            ),
            'base_rotational_tracking': self._objective_tracking(
                ddx_base_r,
                desired_base_r,
            ),
            'fl_translational_tracking': self._objective_tracking(
                ddx_fl_p,
                desired_fl_p,
            ),
            'fl_rotational_tracking': self._objective_tracking(
                ddx_fl_r,
                desired_fl_r,
            ),
            'fr_translational_tracking': self._objective_tracking(
                ddx_fr_p,
                desired_fr_p,
            ),
            'fr_rotational_tracking': self._objective_tracking(
                ddx_fr_r,
                desired_fr_r,
            ),
            'hl_translational_tracking': self._objective_tracking(
                ddx_hl_p,
                desired_hl_p,
            ),
            'hl_rotational_tracking': self._objective_tracking(
                ddx_hl_r,
                desired_hl_r,
            ),
            'hr_translational_tracking': self._objective_tracking(
                ddx_hr_p,
                desired_hr_p,
            ),
            'hr_rotational_tracking': self._objective_tracking(
                ddx_hr_r,
                desired_hr_r,
            ),
            'torque': self._objective_regularization(u),
            'regularization': self._objective_regularization(q),
        }

        objective_terms = {
            k: v * self.weights_config[k] for k, v in objective_terms.items()
        }
        objective_value = sum(objective_terms.values())

        return objective_value

    def _objective_tracking(
        self, q: MX, task_target: MX,
    ) -> MX:
        """Tracking Objective Function."""
        return casadi.sumsqr(q - task_target)

    def _objective_regularization(
        self, q: MX,
    ) -> MX:
        """Regularization Objective Function."""
        return casadi.sumsqr(q)

    def generate_functions(self):
        # Define symbolic variables:
        dv = casadi.MX.sym("dv", self.dv_size)
        u = casadi.MX.sym("u", self.u_size)
        z = casadi.MX.sym("z", self.z_size)

        design_vector = casadi.vertcat(dv, u, z)

        M = casadi.MX.sym("M", self.dv_size, self.dv_size)
        C = casadi.MX.sym("C", self.dv_size)
        J_contact = casadi.MX.sym("J_contact", self.dv_size, self.z_size)
        desired_task_ddx = casadi.MX.sym("desired_task_ddx", self.num_site_ids * 6)
        J_task = casadi.MX.sym("J_task", self.num_site_ids * 6, self.dv_size)
        task_bias = casadi.MX.sym("task_bias", self.num_site_ids * 6)

        equality_constraint_input = [
            design_vector,
            M,
            C,
            J_contact,
        ]

        inequality_constraint_input = [
            design_vector,
        ]

        objective_input = [
            design_vector,
            desired_task_ddx,
            J_task,
            task_bias,
        ]

        # Convert to CasADi Function:
        beq = casadi.Function(
            "beq",
            equality_constraint_input,
            [-self.equality_constraints(*equality_constraint_input)],
        )

        Aeq = casadi.Function(
            "Aeq",
            equality_constraint_input,
            [casadi.densify(casadi.jacobian(
                self.equality_constraints(*equality_constraint_input),
                design_vector,
            ))],
        )

        bineq = casadi.Function(
            "bineq",
            inequality_constraint_input,
            [-self.inequality_constraints(*inequality_constraint_input)],
        )

        Aineq = casadi.Function(
            "Aineq",
            inequality_constraint_input,
            [casadi.densify(casadi.jacobian(
                self.inequality_constraints(*inequality_constraint_input),
                design_vector,
            ))],
        )

        hessian, gradient = casadi.hessian(
            self.objective(*objective_input),
            design_vector,
        )

        H = casadi.Function(
            "H",
            objective_input,
            [casadi.densify(hessian)],
        )

        f = casadi.Function(
            "f",
            objective_input,
            [casadi.densify(gradient)],
        )

        # Actual Values:
        M = np.loadtxt("debug/M.csv", delimiter=",")
        C = np.loadtxt("debug/C.csv", delimiter=",")
        Jc = np.loadtxt("debug/J.csv", delimiter=",")
        dummy_q = np.zeros(self.dv_size + self.u_size + self.z_size)

        res = Aeq(dummy_q, M, C, Jc)
        Apy = res.toarray()

        # Compare:
        A = np.loadtxt("debug/A.csv", delimiter=",")
        print(np.allclose(A, Apy, atol=1e-3))

        np.savetxt("debug/Apy.csv", Apy, delimiter=",")


def main(argv=None):
    # Initialize Mujoco Model:
    filename = "models/unitree_go2/scene_mjx_torque.xml"
    filepath = os.path.join(
        os.path.dirname(
            os.path.dirname(__file__),
        ),
        filename,
    )
    mj_model = mujoco.MjModel.from_xml_path(filepath)

    autogen = AutoGen(mj_model)

    # Generate Functions:
    autogen = AutoGen(mj_model)
    autogen.generate_functions()


if __name__ == "__main__":
    app.run(main)
