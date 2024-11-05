// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <Eigen/Dense>

namespace aliceVision {

namespace SO2 {

using Matrix = Eigen::Matrix<double, 2, 2, Eigen::RowMajor>;

/**
 * @brief Compute the exponential map of the given algebra on the group.
 * @param algebra the 1D vector
 * @return a 2*2 S0(2) matrix
 */
inline Eigen::Matrix2d expm(double algebra)
{
    Eigen::Matrix2d ret;

    ret(0, 0) = cos(algebra);
    ret(0, 1) = -sin(algebra);
    ret(1, 0) = sin(algebra);
    ret(1, 1) = cos(algebra);

    return ret;
}
}  // namespace SO2

namespace SO3 {

using Matrix = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

/**
 * @brief Compute the skew symmetric matrix of the given vector 3D.
 * @param in the 3D vector
 * @return a skew symmetric matrix
 */
inline Eigen::Matrix3d skew(const Eigen::Vector3d& in)
{
    Eigen::Matrix3d ret;

    ret.fill(0);

    ret(0, 1) = -in(2);
    ret(1, 0) = in(2);
    ret(0, 2) = in(1);
    ret(2, 0) = -in(1);
    ret(1, 2) = -in(0);
    ret(2, 1) = in(0);

    return ret;
}

/**
 * @brief Compute the exponential map of the given algebra on the group.
 * @param algebra the 3D vector
 * @return a 3*3 SO(3) matrix
 */
inline Eigen::Matrix3d expm(const Eigen::Vector3d& algebra)
{
    const double angle = algebra.norm();

    if (angle < std::numeric_limits<double>::epsilon())
    {
        return Eigen::Matrix3d::Identity();
    }

    const Eigen::Matrix3d omega = skew(algebra);

    Eigen::Matrix3d ret;
    ret = Eigen::Matrix3d::Identity() + (sin(angle) / angle) * omega + ((1.0 - cos(angle)) / (angle * angle)) * omega * omega;

    return ret;
}

/**
 * @brief Compute the algebra related to a given rotation matrix.
 * @param R the input rotation matrix
 * @return the algebra
 */
inline Eigen::Vector3d logm(const Eigen::Matrix3d& R)
{
    Eigen::Vector3d ret;

    const double p1 = R(2, 1) - R(1, 2);
    const double p2 = R(0, 2) - R(2, 0);
    const double p3 = R(1, 0) - R(0, 1);

    double costheta = (R.trace() - 1.0) / 2.0;
    if (costheta < -1.0)
    {
        costheta = -1.0;
    }

    if (costheta > 1.0)
    {
        costheta = 1.0;
    }

    if (1.0 - costheta < 1e-24)
    {
        ret.fill(0);
        return ret;
    }

    const double theta = acos(costheta);
    const double scale = theta / (2.0 * sin(theta));

    ret(0) = scale * p1;
    ret(1) = scale * p2;
    ret(2) = scale * p3;

    return ret;
}

/**
Compute the jacobian of the logarithm wrt changes in the rotation matrix values
@param R the input rotation matrix
@return the jacobian matrix (3*9 matrix)
*/
inline Eigen::Matrix<double, 3, 9, Eigen::RowMajor> dlogmdr(const Eigen::Matrix3d& R)
{
    double p1 = R(2, 1) - R(1, 2);
    double p2 = R(0, 2) - R(2, 0);
    double p3 = R(1, 0) - R(0, 1);

    double costheta = (R.trace() - 1.0) / 2.0;
    if (costheta > 1.0)
        costheta = 1.0;
    else if (costheta < -1.0)
        costheta = -1.0;

    double theta = acos(costheta);

    if (fabs(theta) < std::numeric_limits<float>::epsilon())
    {
        Eigen::Matrix<double, 3, 9> J;
        J.fill(0);
        J(0, 5) = 1;
        J(0, 7) = -1;
        J(1, 2) = -1;
        J(1, 6) = 1;
        J(2, 1) = 1;
        J(2, 3) = -1;
        return J;
    }

    double scale = theta / (2.0 * sin(theta));

    Eigen::Vector3d resnoscale;
    resnoscale(0) = p1;
    resnoscale(1) = p2;
    resnoscale(2) = p3;

    Eigen::Matrix<double, 3, 3> dresdp = Eigen::Matrix3d::Identity() * scale;
    Eigen::Matrix<double, 3, 9> dpdmat;
    dpdmat.fill(0);
    dpdmat(0, 5) = 1;
    dpdmat(0, 7) = -1;
    dpdmat(1, 2) = -1;
    dpdmat(1, 6) = 1;
    dpdmat(2, 1) = 1;
    dpdmat(2, 3) = -1;

    double dscaledtheta = -0.5 * theta * cos(theta) / (sin(theta) * sin(theta)) + 0.5 / sin(theta);
    double dthetadcostheta = -1.0 / sqrt(-costheta * costheta + 1.0);

    Eigen::Matrix<double, 1, 9> dcosthetadmat;
    dcosthetadmat << 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5;
    Eigen::Matrix<double, 1, 9> dscaledmat = dscaledtheta * dthetadcostheta * dcosthetadmat;

    return dpdmat * scale + resnoscale * dscaledmat;
}



/**
Compute the jacobian of the exponential wrt changes in the rotation vector values
@param vecr the rotation vector
@return the jacobian matrix (9*3 matrix)
*/
inline Eigen::Matrix<double, 9, 3, Eigen::RowMajor> dexpmdr(const Eigen::Vector3d & vecr)
{
    double angle = vecr.norm();  

    if (angle < 1e-24)
    {
        Eigen::Matrix<double, 9, 3> d_K_d_vecr;
        d_K_d_vecr.fill(0);
        d_K_d_vecr(1, 2) = 1;
        d_K_d_vecr(2, 1) = -1;
        d_K_d_vecr(3, 2) = -1;
        d_K_d_vecr(5, 0) = 1;
        d_K_d_vecr(6, 1) = 1;
        d_K_d_vecr(7, 0) = -1;
        return d_K_d_vecr;
    }

    const double x = vecr(0);
    const double y = vecr(1);
    const double z = vecr(2);

    double angle2 = angle * angle;
    double angle3 = angle2 * angle;
    double sina = sin(angle);
    double cosa = cos(angle);
    double mcosa = 1.0 - cosa;
    double c = mcosa/ angle2;
    double s = sina / angle;

    double d_s_d_angle = cosa/angle - sina/angle2;
    double d_c_d_angle = sina/angle2 - 2.0*mcosa/angle3;
    
    double d_angle_d_x = x / angle;
    double d_angle_d_y = y / angle;
    double d_angle_d_z = z / angle;

    //Jacobian of s*K + c*K*K
    Eigen::Matrix<double, 9, 5> J;
    J(0,0)=0;
    J(1,0)=c*y;
    J(2,0)=c*z;
    J(3,0)=c*y;
    J(4,0)=-2*c*x;
    J(5,0)=s;
    J(6,0)=c*z;
    J(7,0)=-s;
    J(8,0)=-2*c*x;
    J(0,1)=-2*c*y;
    J(1,1)=c*x;
    J(2,1)=-s;
    J(3,1)=c*x;
    J(4,1)=0;
    J(5,1)=c*z;
    J(6,1)=s;
    J(7,1)=c*z;
    J(8,1)=-2*c*y;
    J(0,2)=-2*c*z;
    J(1,2)=s;
    J(2,2)=c*x;
    J(3,2)=-s;
    J(4,2)=-2*c*z;
    J(5,2)=c*y;
    J(6,2)=c*x;
    J(7,2)=c*y;
    J(8,2)=0;
    J(0,3)=0;
    J(1,3)=z;
    J(2,3)=-y;
    J(3,3)=-z;
    J(4,3)=0;
    J(5,3)=x;
    J(6,3)=y;
    J(7,3)=-x;
    J(8,3)=0;
    J(0,4)=-y*y - z*z;
    J(1,4)=x*y;
    J(2,4)=x*z;
    J(3,4)=x*y;
    J(4,4)=-x*x - z*z;
    J(5,4)=y*z;
    J(6,4)=x*z;
    J(7,4)=y*z;
    J(8,4)=-x*x - y*y;

    //Jacobian of [x y z s c] wrt [x y z]
    Eigen::Matrix<double, 5, 3> M = Eigen::Matrix<double, 5, 3>::Zero();
    M(0, 0) = 1.0;
    M(1, 1) = 1.0;
    M(2, 2) = 1.0;
    M(3, 0) = d_s_d_angle * d_angle_d_x;
    M(3, 1) = d_s_d_angle * d_angle_d_y;
    M(3, 2) = d_s_d_angle * d_angle_d_z;
    M(4, 0) = d_c_d_angle * d_angle_d_x;
    M(4, 1) = d_c_d_angle * d_angle_d_y;
    M(4, 2) = d_c_d_angle * d_angle_d_z;

    return J * M;
}

}  // namespace SO3

namespace SE3 {

using Matrix = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;

/**
 * @brief Compute the exponential map of the given algebra on the group.
 * @param algebra the 6D vector
 * @return a 4*4 SE(3) matrix
 */
inline Eigen::Matrix4d expm(const Eigen::Matrix<double, 6, 1>& algebra)
{
    Eigen::Matrix4d ret;
    ret.setIdentity();

    const Eigen::Vector3d vecR = algebra.block<3, 1>(0, 0);
    const Eigen::Vector3d vecT = algebra.block<3, 1>(3, 0);

    double angle = vecR.norm();
    if (angle < std::numeric_limits<double>::epsilon())
    {
        ret.setIdentity();
        ret.block<3, 1>(0, 3) = vecT;
        return ret;
    }

    const Eigen::Matrix3d omega = SO3::skew(vecR);
    const Eigen::Matrix3d V =
      Eigen::Matrix3d::Identity() + ((1.0 - cos(angle)) / (angle * angle)) * omega + ((angle - sin(angle)) / (angle * angle * angle)) * omega * omega;

    ret.block<3, 3>(0, 0) = SO3::expm(vecR);
    ret.block<3, 1>(0, 3) = V * vecT;

    return ret;
}

}  // namespace SE3

}  // namespace aliceVision