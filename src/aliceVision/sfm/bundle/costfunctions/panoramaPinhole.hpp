// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/geometry/lie.hpp>
#include <Eigen/Core>
#include <ceres/ceres.h>

namespace aliceVision {
namespace sfm {

class CostPanoramaPinHole : public ceres::CostFunction
{
  public:
    CostPanoramaPinHole(Vec2 fi, Vec2 fj, std::shared_ptr<camera::Pinhole>& intrinsic)
      : _fi(fi),
        _fj(fj),
        _intrinsic(intrinsic)
    {
        set_num_residuals(2);

        mutable_parameter_block_sizes()->push_back(6);
        mutable_parameter_block_sizes()->push_back(6);
        mutable_parameter_block_sizes()->push_back(intrinsic->getParams().size());
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        Vec2 pt_i = _fi;
        Vec2 pt_j = _fj;

        const double* parameter_pose_i = parameters[0];
        const double* parameter_pose_j = parameters[1];
        const double* parameter_intrinsics = parameters[2];

        const Eigen::Map<const Eigen::Vector3d> iro(parameter_pose_i);
        const Eigen::Map<const Eigen::Vector3d> jro(parameter_pose_j);

        Eigen::Matrix<double, 3, 3> iRo = SO3::expm(iro);
        Eigen::Matrix<double, 3, 3> jRo = SO3::expm(jro);

        _intrinsic->setScale({parameter_intrinsics[0], parameter_intrinsics[1]});
        _intrinsic->setOffset({parameter_intrinsics[2], parameter_intrinsics[3]});

        size_t params_size = _intrinsic->getParamsSize();
        size_t disto_size = _intrinsic->getDistortionParamsSize();
        size_t offset = params_size - disto_size;

        _intrinsic->setDistortionParamsFn(disto_size, [&](auto index) { return parameter_intrinsics[offset + index]; });

        Eigen::Matrix3d R = jRo * iRo.transpose();
        geometry::Pose3 T_pose3(R, Vec3({0, 0, 0}));
        Eigen::Matrix4d T = T_pose3.getHomogeneous();

        Vec2 pt_i_cam = _intrinsic->ima2cam(pt_i);
        Vec2 pt_i_undist = _intrinsic->removeDistortion(pt_i_cam);
        Vec4 pt_i_sphere = _intrinsic->toUnitSphere(pt_i_undist).homogeneous();

        Vec2 pt_j_est = _intrinsic->project(T_pose3, pt_i_sphere, true);

        residuals[0] = pt_j_est(0) - pt_j(0);
        residuals[1] = pt_j_est(1) - pt_j(1);

        if (jacobians == nullptr)
        {
            return true;
        }

        if (jacobians[0] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(jacobians[0]);

            Eigen::Matrix<double, 2, 3> J3 = _intrinsic->getDerivativeProjectWrtRotation(T, pt_i_sphere) *
                                             getJacobian_AB_wrt_B<3, 3, 3>(jRo, iRo.transpose()) * getJacobian_At_wrt_A<3, 3>() *
                                             SO3::dexpmdr(iro);

            J.fill(0);
            J.block<2, 3>(0, 0) = J3.block<2, 3>(0, 0);
        }

        if (jacobians[1] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(jacobians[1]);

            Eigen::Matrix<double, 2, 3> J3 = _intrinsic->getDerivativeProjectWrtRotation(T, pt_i_sphere) *
                                             getJacobian_AB_wrt_A<3, 3, 3>(jRo, iRo.transpose()) *
                                             SO3::dexpmdr(jro);

            J.fill(0);
            J.block<2, 3>(0, 0) = J3.block<2, 3>(0, 0);
        }

        if (jacobians[2] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[2], 2, params_size);

            Eigen::Matrix<double, 4, 3> Jhomogenous = Eigen::Matrix<double, 4, 3>::Identity();

            Eigen::Matrix<double, 2, 2> Jscale =
              _intrinsic->getDerivativeProjectWrtScale(T, pt_i_sphere) +
              _intrinsic->getDerivativeProjectWrtPoint(T, pt_i_sphere) * Jhomogenous * _intrinsic->getDerivativetoUnitSphereWrtPoint(pt_i_undist) *
                _intrinsic->getDerivativeRemoveDistoWrtPt(pt_i_cam) * _intrinsic->getDerivativeIma2CamWrtScale(pt_i);
            Eigen::Matrix<double, 2, 2> Jpp =
              _intrinsic->getDerivativeProjectWrtPrincipalPoint(T, pt_i_sphere) +
              _intrinsic->getDerivativeProjectWrtPoint(T, pt_i_sphere) * Jhomogenous * _intrinsic->getDerivativetoUnitSphereWrtPoint(pt_i_undist) *
                _intrinsic->getDerivativeRemoveDistoWrtPt(pt_i_cam) * _intrinsic->getDerivativeIma2CamWrtPrincipalPoint();

            J.block<2, 2>(0, 0) = Jscale;
            J.block<2, 2>(0, 2) = Jpp;

            if (disto_size > 0)
            {
                Eigen::Matrix<double, 2, Eigen::Dynamic> Jdisto =
                _intrinsic->getDerivativeProjectWrtDisto(T, pt_i_sphere) + _intrinsic->getDerivativeProjectWrtPoint(T, pt_i_sphere) * Jhomogenous *
                                                                           _intrinsic->getDerivativetoUnitSphereWrtPoint(pt_i_undist) *
                                                                           _intrinsic->getDerivativeRemoveDistoWrtDisto(pt_i_cam);
                J.block(0, 4, 2, disto_size) = Jdisto;
            }
        }

        return true;
    }

  private:
    Vec2 _fi;
    Vec2 _fj;
    std::shared_ptr<camera::Pinhole> _intrinsic;
};

}  // namespace sfm
}  // namespace aliceVision