import os
from absl import app

import numpy as np
import casadi
import mujoco

from casadi import MX, DM


class AutoGen():
    def __init__(self, mj_model: mujoco.MjModel, num_contacts: int):
        self.dv_size = mj_model.nv
        self.u_size = mj_model.nu
        self.z_size = num_contacts * 3
        
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

    def generate_functions(self):
        # Define symbolic variables:
        dv = casadi.MX.sym("dv", self.dv_size)
        u = casadi.MX.sym("u", self.u_size)
        z = casadi.MX.sym("z", self.z_size)

        design_vector = casadi.vertcat(dv, u, z)

        M_matrix = casadi.MX.sym("M", self.dv_size, self.dv_size)
        C_matrix = casadi.MX.sym("C", self.dv_size)
        J_contact_matrix = casadi.MX.sym("J_contact", self.dv_size, self.z_size)

        equality_constraint_input = [
            design_vector,
            M_matrix,
            C_matrix,
            J_contact_matrix,
        ]

        # Convert to CasADi Function:
        b_eq_function = casadi.Function(
            "b_eq_function",
            equality_constraint_input,
            [-self.equality_constraints(*equality_constraint_input)],
        )

        # Wrapped with densify:
        A_eq_function = casadi.Function(
            "A_eq_function",
            equality_constraint_input,
            [casadi.densify(casadi.jacobian(
                self.equality_constraints(*equality_constraint_input),
                design_vector,
            ))],
        )

        # Actual Values:
        M = np.loadtxt("debug/M.csv", delimiter=",")
        C = np.loadtxt("debug/C.csv", delimiter=",")
        Jc = np.loadtxt("debug/J.csv", delimiter=",")
        dummy_q = np.zeros(self.dv_size + self.u_size + self.z_size)

        res = A_eq_function(dummy_q, M, C, Jc)
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
    num_contacts = 4

    # Generate Functions:
    autogen = AutoGen(mj_model, num_contacts)
    autogen.generate_functions()


if __name__ == "__main__":
    app.run(main)
