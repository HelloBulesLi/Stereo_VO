// think how to add damp to avoid the non observerty? first run, then change
//
// Created by xiang on 12/21/17.
//
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "sophus/se3.hpp"

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

// use g2o do optimize

// string p3d_file = "./p3d.txt";
// string p2d_file = "./p2d.txt";
string p3d_file = "../p3d.txt";
string p2d_file = "../p2d.txt";


int main(int argc, char **argv) {

    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d 
    // START YOUR CODE HERE
    ifstream input_file;
    input_file.open(p2d_file);
    if(!input_file.is_open())
    {
        cout << "open file failed" << endl;
    }
    while(!input_file.eof())
    {
        Vector2d tmp;
        input_file >> tmp[0] >> tmp[1];
        p2d.push_back(tmp);
    }
    input_file.close();

    input_file.open(p3d_file);
    if(!input_file.is_open())
    {
        cout << "open file failed" << endl;
    }
    while(!input_file.eof())
    {
        Vector3d tmp;
        input_file >> tmp[0] >> tmp[1] >> tmp[2];
        p3d.push_back(tmp);
    }
    input_file.close();

    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;

    Sophus::SE3<double> T_esti; // estimated pose

    Vector6d se3_esti = Vector6d::Zero();
    Matrix3d R;
    R<< 1,0,0,0,1,0,0,0,1;
    T_esti.so3() = R;
    T_esti.translation() = Vector3d::Zero();
    lastCost = 3e7;

    // double currentLambda_ = -1;
    double currentLambda_ = 0.01;
    double ni_ = 2.0;
    double stopThresholdLM_ = -1;
    for (int iter = 0; iter < iterations; iter++) {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        Vector3d fx = Vector3d::Zero();
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
            // START YOUR CODE HERE 
            Vector3d Z = T_esti*p3d[i];
            Vector3d z;
            z = 1/Z[2]*K*Z;
            z[2] = 1;
            Vector3d p2d_homo(p2d[i][0], p2d[i][1], 1);
            fx = p2d_homo -z;
            cost += fx.transpose()*fx;
            // END YOUR CODE HERE

            // compute jacobian
            Matrix<double, 2, 6> J;
            // START YOUR CODE HERE 
            // Matrix<double, 3, 6> Jacob = 
            Matrix<double, 3, 6> tmp;
            tmp.block(0,0,3,3)<< 1,0,0,0,1,0,0,0,1;
            // tmp.block(0,3,6,6) = -Sophus::SO3d::hat(T_esti.so3().matrix()*Z + T_esti.translation());
            tmp.block(0,3,3,3) = -Sophus::SO3d::hat(Z);
            Matrix<double, 3, 6> j_tmp = -K*tmp;
            J = j_tmp.block(0,0,2,6);
            // END YOUR CODE HERE

            H += J.transpose() * J;
            b += -J.transpose() * fx.head(2);
        }

        if (currentLambda_ < 0)
        {
            double maxDiagonal = 0;
            uint32_t size = H.cols();
            assert(H.rows() == H.cols() && "Hessian is not square");
            for (uint32_t i = 0; i < size; ++i) {
                maxDiagonal = std::max(fabs(H(i, i)), maxDiagonal);
            }
            double tau = 1e-5;
            currentLambda_ = tau * maxDiagonal;
            stopThresholdLM_ = 1e-6*cost;
            cout << "init lambda is " << currentLambda_ << endl;
        }
        else
        {
            for (uint32_t i = 0; i < H.cols(); ++i) {
                H(i, i) += currentLambda_;
            }
        }

	    // solve dx 
        Vector6d dx;

        // START YOUR CODE HERE
        dx = H.ldlt().solve(b);
        // END YOUR CODE HERE

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        // if (iter > 0 && cost >= lastCost) {
        //     // cost increase, update is not good
        //     cout << "cost: " << cost << ", last cost: " << lastCost << endl;
        //     break;
        // }

        // update your estimation
        // START YOUR CODE HERE

        se3_esti += dx;
        T_esti.so3() = Sophus::SO3<double>::exp(se3_esti.tail(3));
        T_esti.translation() = se3_esti.head(3);

        // judge is accept the update ?
        double scale = dx.transpose() * (currentLambda_ * dx + b);
        scale += 1e-3;    // make sure it's non-zero :)

        double new_cost = 0;
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
            // START YOUR CODE HERE 
            Vector3d Z = T_esti*p3d[i];
            Vector3d z;
            z = 1/Z[2]*K*Z;
            z[2] = 1;
            Vector3d p2d_homo(p2d[i][0], p2d[i][1], 1);
            fx = p2d_homo -z;
            new_cost += fx.transpose()*fx;
            // END YOUR CODE HERE
        }

        if (new_cost  < stopThresholdLM_)
        {
            break;
        }

        double rho = (cost - new_cost) / scale;

        // if (rho > 0 && isfinite(new_cost))   // last step was good, 误差在下降
        // {
        //     double alpha = 1. - pow((2 * rho - 1), 3);
        //     alpha = std::min(alpha, 2. / 3.);
        //     double scaleFactor = (std::max)(1. / 3., alpha);
        //     currentLambda_ *= scaleFactor;
        //     ni_ = 2;
        // } else {
        //     currentLambda_ *= ni_;
        //     ni_ *= 2;
        //     se3_esti -= dx;
        //     T_esti.so3() = Sophus::SO3<double>::exp(se3_esti.tail(3));
        //     T_esti.translation() = se3_esti.head(3);
        //     cout << "roll back occure" << endl;
        // }

        if (rho > 0 && isfinite(new_cost))   // last step was good, 误差在下降
        {
            // statistic currentChi and scale
            // input_file << tempChi << ' ' << currentChi_ - (0.5*scale) << ' ' << currentLambda_ << ' ' << endl;

            currentLambda_ = (std::max)(currentLambda_/9.0, 1e-7);
        } else {
            currentLambda_ = (std::min)(currentLambda_*11.0, 1e7);
            se3_esti -= dx;
            T_esti.so3() = Sophus::SO3<double>::exp(se3_esti.tail(3));
            T_esti.translation() = se3_esti.head(3);
            cout << "roll back occure" << endl;
            continue;
        }
        cout << "se3 esti" << endl << se3_esti << endl;
        // END YOUR CODE HERE
        
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }


    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
