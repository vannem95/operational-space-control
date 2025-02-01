#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

#include "mujoco/mujoco.h"
#include "Eigen/Dense"

#include "src/unitree_go2/operational_space_controller.h"

int main(int argc, char** argv) {
    OperationalSpaceController osc;
    std::filesystem::path xml_path = "models/unitree_go2/scene_mjx.xml";
    osc.initialize(xml_path);

    mj_step(osc.model, osc.data);

    Eigen::MatrixXd points = Eigen::MatrixXd::Zero(5, 3);
    auto data = osc.data;
    typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MapMatrix;
    points = MapMatrix(data->site_xpos, 5, 3);
    std::cout << points << std::endl;

    auto osc_data = osc.get_data(points);

    

    return 0;
}