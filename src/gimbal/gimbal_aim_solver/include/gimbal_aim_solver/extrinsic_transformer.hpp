#pragma once

#include <Eigen/Core>
#include <vector>

namespace gimbal_aim_solver
{

class ExtrinsicTransformer
{
public:
  ExtrinsicTransformer() = default;

  bool init(const std::vector<double>& rvec_data,
            const std::vector<double>& tvec_data);

  bool initialized() const { return initialized_; }

  // world -> camera
  Eigen::Vector3d worldToCamera(const Eigen::Vector3d& p_world) const;

private:
  Eigen::Matrix3d R_cw_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_cw_{Eigen::Vector3d::Zero()};
  bool initialized_{false};
};

}  // namespace gimbal_aim_solver