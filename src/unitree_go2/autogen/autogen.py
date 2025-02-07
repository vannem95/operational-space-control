from absl import app
from absl import flags

import os
import yaml

import casadi
import mujoco

from casadi import MX, DM

FLAGS = flags.FLAGS
flags.DEFINE_string("filepath", None, "Bazel filepath to the autogen folder (This should be automatically determinded by the genrule).")


class AutoGen():
    def __init__(self, mj_model: mujoco.MjModel):
        self.mj_model = mj_model

        # Parse Configuration YAML File:
        with open("config/unitree_go2/unitree_go2_config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Get Body and Site IDs:
        self.body_list = [f'"{body}"sv' for body in config['body_list']]
        self.noncontact_site_list = [f'"{noncontact_site}"sv' for noncontact_site in config['noncontact_site_list']]
        self.contact_site_list = [f'"{contact_site}"sv' for contact_site in config['contact_site_list']]
        self.site_list = self.noncontact_site_list + self.contact_site_list
        self.num_body_ids = len(self.body_list)
        self.num_site_ids = len(self.site_list)
        self.num_noncontact_site_ids = len(self.noncontact_site_list)
        self.num_contact_site_ids = len(self.contact_site_list)

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

        A_eq_function = casadi.Function(
            "A_eq_function",
            equality_constraint_input,
            [casadi.densify(casadi.jacobian(
                self.equality_constraints(*equality_constraint_input),
                design_vector,
            ))],
        )

        # Generate C++ Code:
        opts = {
            "cpp": True,
            "with_header": True,
        }
        filenames = [
            "autogen_functions",
        ]
        casadi_functions = [
            [b_eq_function, A_eq_function],
        ]
        loop_iterables = zip(
            filenames,
            casadi_functions,
        )

        for filename, casadi_function in loop_iterables:
            generator = casadi.CodeGenerator(f"{filename}.cc", opts)
            for function in casadi_function:
                generator.add(function)
        generator.generate(FLAGS.filepath+"/")

    def generate_defines(self):
        cc_code = f"""#pragma once
#include <string_view>

using namespace std::string_view_literals;

namespace constants {{
    namespace model {{
        // Mujoco Model Constants:
        constexpr int nq_size  = {self.mj_model.nq};
        constexpr int nv_size = {self.mj_model.nv};
        constexpr int nu_size  = {self.mj_model.nu};
        constexpr int body_ids_size = {self.num_body_ids};
        constexpr int site_ids_size = {self.num_site_ids};
        constexpr int noncontact_site_ids_size = {self.num_noncontact_site_ids};
        constexpr int contact_site_ids_size = {self.num_contact_site_ids};
        constexpr std::array body_list = {{{", ".join(self.body_list)}}};
        constexpr std::array site_list = {{{", ".join(self.site_list)}}};
        constexpr std::array noncontact_site_list = {{{", ".join(self.noncontact_site_list)}}};
        constexpr std::array contact_site_list = {{{", ".join(self.contact_site_list)}}};
    }}
    namespace optimization {{
        // Optimization Constants:
        constexpr int dv_size = {self.dv_size};
        constexpr int u_size = {self.u_size};
        constexpr int z_size = {self.z_size};
        constexpr int design_vector_size = {self.design_vector_size};
        constexpr int dv_idx = {self.dv_idx};
        constexpr int u_idx = {self.u_idx};
        constexpr int z_idx = {self.z_idx};
    }}
}}
        """

        filepath = os.path.join(FLAGS.filepath, "autogen_defines.h")
        with open(filepath, "w") as f:
            f.write(cc_code)


def main(argv):
    # Initialize Mujoco Model:
    mj_model = mujoco.MjModel.from_xml_path("models/unitree_go2/scene_mjx_torque.xml")

    # Generate Functions:
    autogen = AutoGen(mj_model)
    autogen.generate_functions()
    autogen.generate_defines()


if __name__ == "__main__":
    app.run(main)
