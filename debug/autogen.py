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

        contact_forces = casadi.vertsplit_n(z, self.num_contact_site_ids)

        inequality_constraints = []
        for i in range(self.num_contact_site_ids):
            contact_force = contact_forces[i]
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
        desired_task_p, desired_task_r = casadi.horzsplit_n(desired_task_ddx, 2)
        desired_task_p = casadi.vertsplit_n(desired_task_p, self.num_site_ids)
        desired_task_r = casadi.vertsplit_n(desired_task_r, self.num_site_ids)
        desired_base_p, desired_fl_p, desired_fr_p, desired_hl_p, desired_hr_p = map(lambda x: x.T, desired_task_p)
        desired_base_r, desired_fl_r, desired_fr_r, desired_hl_r, desired_hr_r = map(lambda x: x.T, desired_task_r)

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
        dv = casadi.SX.sym("dv", self.dv_size)
        u = casadi.SX.sym("u", self.u_size)
        z = casadi.SX.sym("z", self.z_size)

        design_vector = casadi.vertcat(dv, u, z)

        M = casadi.SX.sym("M", self.dv_size, self.dv_size)
        C = casadi.SX.sym("C", self.dv_size)
        J_contact = casadi.SX.sym("J_contact", self.dv_size, self.z_size)
        desired_task_ddx = casadi.SX.sym("desired_task_ddx", self.num_site_ids, 6)
        J_task = casadi.SX.sym("J_task", self.num_site_ids * 6, self.dv_size)
        task_bias = casadi.SX.sym("task_bias", self.num_site_ids * 6)

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
            [hessian],
        )

        f = casadi.Function(
            "f",
            objective_input,
            [gradient],
        )

        beq_size = beq.size_out(0)
        self.beq_sz = beq_size[0] * beq_size[1]
        Aeq_size = Aeq.size_out(0)
        self.Aeq_sz = Aeq_size[0] * Aeq_size[1]
        self.Aeq_rows = Aeq_size[0]
        self.Aeq_cols = Aeq_size[1]
        bineq_size = bineq.size_out(0)
        self.bineq_sz = bineq_size[0] * bineq_size[1]
        Aineq_size = Aineq.size_out(0)
        self.Aineq_sz = Aineq_size[0] * Aineq_size[1]
        self.Aineq_rows = Aineq_size[0]
        self.Aineq_cols = Aineq_size[1]
        H_size = H.size_out(0)
        self.H_sz = H_size[0] * H_size[1]
        self.H_rows = H_size[0]
        self.H_cols = H_size[1]
        f_size = f.size_out(0)
        self.f_sz = f_size[0] * f_size[1]

        # Load Test Values:
        dummy_q = np.zeros(self.dv_size + self.u_size + self.z_size)
        M = np.loadtxt("debug/M.csv", delimiter=",")
        C = np.loadtxt("debug/C.csv", delimiter=",")
        Jc = np.loadtxt("debug/Jc.csv", delimiter=",")
        J = np.loadtxt("debug/J.csv", delimiter=",")
        b = np.loadtxt("debug/b.csv", delimiter=",")
        ddx_desired = np.loadtxt("debug/ddx_desired.csv", delimiter=",")

        # Evaluate Functions:
        Aeqpy = Aeq(dummy_q, M, C, Jc).toarray()
        beqpy = beq(dummy_q, M, C, Jc).toarray().flatten()
        Aineqpy = Aineq(dummy_q).toarray()
        bineqpy = bineq(dummy_q).toarray().flatten()
        Hpy = H(dummy_q, ddx_desired, J, b).toarray()
        fpy = f(dummy_q, ddx_desired, J, b).toarray().flatten()

        # Compare:
        Aeq = np.loadtxt("debug/Aeq.csv", delimiter=",")
        print(f"Aeq == Aeqpy: {np.allclose(Aeq, Aeqpy, atol=1e-3)}")
        beq = np.loadtxt("debug/beq.csv", delimiter=",")
        print(f"beq == beqpy: {np.allclose(beq, beqpy, atol=1e-3)}")
        Aineq = np.loadtxt("debug/Aineq.csv", delimiter=",")
        print(f"Aineq == Aineqpy: {np.allclose(Aineq, Aineqpy, atol=1e-3)}")
        bineq = np.loadtxt("debug/bineq.csv", delimiter=",")
        print(f"bineq == bineqpy: {np.allclose(bineq, bineqpy, atol=1e-3)}")
        H = np.loadtxt("debug/H.csv", delimiter=",")
        print(f"H == Hpy: {np.allclose(H, Hpy, atol=1e-3)}")
        f = np.loadtxt("debug/f.csv", delimiter=",")
        print(f"f == fpy: {np.allclose(f, fpy, atol=1e-3)}")
        np.savetxt("debug/Aeqpy.csv", Aeqpy, fmt='%.1e', delimiter=",")
        pass


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
