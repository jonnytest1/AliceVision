// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/geometry/lie.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <ceres/ceres.h>

namespace aliceVision {
namespace sfm {

class SO3Manifold : public ceres::Manifold
{
  public:
    ~SO3Manifold() override = default;

    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override
    {
        double* ptrBase = (double*)x;
        double* ptrResult = (double*)x_plus_delta;
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotation(ptrBase);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotationResult(ptrResult);

        Eigen::Vector3d axis;
        axis(0) = delta[0];
        axis(1) = delta[1];
        axis(2) = delta[2];
        double angle = axis.norm();

        axis.normalize();

        Eigen::AngleAxisd aa(angle, axis);
        Eigen::Matrix3d Rupdate;
        Rupdate = aa.toRotationMatrix();

        rotationResult = Rupdate * rotation;

        return true;
    }

    bool PlusJacobian(const double* /*x*/, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobian);
        // Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(x);

        J.fill(0);

        J(1, 2) = 1;
        J(2, 1) = -1;
        J(3, 2) = -1;
        J(5, 0) = 1;
        J(6, 1) = 1;
        J(7, 0) = -1;

        return true;
    }

    bool Minus(const double* y, const double* x, double* delta) const override
    {
        throw std::invalid_argument("SO3::Manifold::Minus() should never be called");
    }

    bool MinusJacobian(const double* x, double* jacobian) const override
    {
        throw std::invalid_argument("SO3::Manifold::MinusJacobian() should never be called");
    }

    int AmbientSize() const override { return 9; }

    int TangentSize() const override { return 3; }
};

}  // namespace sfm

}  // namespace aliceVision
