#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>
#include <boost/format.hpp>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/edge_se3_prior.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <chrono>
#include <execution>

using namespace std;
using namespace Eigen;
using namespace cv;

// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 基线
double b = 0.573;

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 16>> BlockSloverType;
typedef g2o::LinearSolverEigen<BlockSloverType::PoseMatrixType> LinearSolverType_Cur;

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

// add wrap for source and target image
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSophus() {}

    ~VertexSophus() {}

    bool read(std::istream &is) {}

    bool write(std::ostream &os) const {}

    virtual void setToOriginImpl() {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update_) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        setEstimate(Sophus::SE3d::exp(update) * estimate());
    }
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
/*
class ProjectionEdge : public g2o::BaseUnaryEdge<2, Vector2d, VertexSophus> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ProjectionEdge(Vector3d &Pr){_Pr = Pr;}
  // 计算曲线模型误差
  virtual void computeError() override {
    VertexSophus *pose = (VertexSophus *)_vertices[0];
    Vector3d Pc = pose->estimate()*_Pr;
    double z = Pc[2];
    double u = fx*Pc[0]/z + cx;
    double v = fy*Pc[1]/z + cy;

    _error = _measurement - Vector2d(u, v);
    // _error = Vector2d(u, v) - _measurement;
  }

  // 计算雅可比矩阵
//   virtual void linearizeOplus() override {
//     auto v0 = (VertexSophus *)_vertices[0];
//     Eigen::Vector3d Pc = v0->estimate()*_Pr;
//     double Z_2 = Pc(2,0)*Pc(2,0);
//     Matrix<double, 2, 3> Jacob_Pc;
//     Jacob_Pc << fx/Pc(2,0), 0, -fx*Pc(0,0)/Z_2, 0, fy/Pc(2,0), -fy*Pc(1,0)/Z_2;
//     Matrix<double, 3, 6> Jacob_Pc_Tj;
//     Jacob_Pc_Tj.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
//     Jacob_Pc_Tj.block<3,3>(0,3) = -Sophus::SO3d::hat(Pc);

//     _jacobianOplusXi = Jacob_Pc * Jacob_Pc_Tj;
//     // cout << "cur pc is " << Pc.transpose() << endl;
//   }

  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}

public:
  Vector3d _Pr;  // x 值， y 值为 _measurement
};
*/

class ProjectionEdge : public g2o::BaseUnaryEdge<2, Vector2d, VertexSophus> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ProjectionEdge(Vector3d &Pr){_Pr = Pr;}
  // 计算曲线模型误差
  virtual void computeError() override {
    VertexSophus *pose = (VertexSophus *)_vertices[0];
    Vector3d Pc = pose->estimate()*_Pr;
    double z = Pc[2];
    double u = fx*Pc[0]/z + cx;
    double v = fy*Pc[1]/z + cy;

    _error = _measurement - Vector2d(u, v);
    // _error = Vector2d(u, v) - _measurement;
  }

  // 计算雅可比矩阵
  /*
  virtual void linearizeOplus() override {
    // const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
    // const Eigen::Vector3d abc = v->estimate();
    // double y = v->estimate();
    auto v0 = (VertexSophus *)_vertices[0];
    Eigen::Vector3d Pc = v0->estimate()*_Pr;
    double Z_2 = Pc(2,0)*Pc(2,0);
    Matrix<double, 2, 3> Jacob_Pc;
    Jacob_Pc << fx/Pc(2,0), 0, -fx*Pc(0,0)/Z_2, 0, fy/Pc(2,0), -fy*Pc(1,0)/Z_2;
    Matrix<double, 3, 6> Jacob_Pc_Tj;
    Jacob_Pc_Tj.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    Jacob_Pc_Tj.block<3,3>(0,3) = -Sophus::SO3d::hat(Pc);

    _jacobianOplusXi = Jacob_Pc * Jacob_Pc_Tj;
    // cout << "cur pc is " << Pc.transpose() << endl;
  }
  */

  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}

public:
  Vector3d _Pr;  // x 值， y 值为 _measurement
};

typedef Eigen::Matrix<double,16,16> Matrix16d;
typedef Eigen::Matrix<double,64,64> Matrix64d;
// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
typedef Eigen::Matrix<double,16,1> Vector16d;
typedef Eigen::Matrix<double,64,1> Vector64d;

uint8_t patch_size = 4;

class EdgeDirectProjection : public g2o::BaseUnaryEdge<16, Vector16d, VertexSophus> {
// class EdgeDirectProjection : public g2o::BaseUnaryEdge<64, Vector64d, VertexSophus> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // EdgeDirectProjection(float *color, cv::Mat &target) {
    //     this->origColor = color;
    //     this->targetImg = target;
    // }

    EdgeDirectProjection(Vector3d &Pr, cv::Mat &target) {
        _Pr = Pr;
        targetImg = target;
    }

    ~EdgeDirectProjection() {}

    virtual void computeError() override {

        // currently not think wrap
        // TODO START YOUR CODE HERE
        auto v0 = (VertexSophus *)_vertices[0];

        Eigen::Vector3d Pc = v0->estimate()*_Pr;
        double u = fx*Pc(0,0)/Pc(2,0) + cx;
        double v = fy*Pc(1,0)/Pc(2,0) + cy;

        Vector16d proj;
        Vector16d proj_back;
        uint32_t index = 0;
        if (u < patch_size/2 || u + patch_size/2 > targetImg.cols
            || v < patch_size/2 || v + patch_size/2 > targetImg.rows)
        {
            _error.setZero();
            return;
        }

        for (int dx = -patch_size/2; dx < patch_size/2; dx++)
        {
            for (int dy = -patch_size/2; dy < patch_size/2; dy++)
            {
                proj(index, 0) = GetPixelValue(targetImg, u+dx, v+dy);
                // proj(index, 0) = targetImg.at<uchar>(u+dx, v+dy);
                index++;
            }
        }
        
        // if (proj(0) >= 254)
        // {
        //     _error.setZero();
        //     return ;
        // }
        _error = proj - _measurement;
        // cout << "cur measure " << _measurement.transpose() << endl;
        // cout << "cur proj pixel " << proj.transpose() << endl;
        // cout << "cur proj pos " << u << ' ' << v << endl;
        // TODO END YOUR CODE HERE
    }

    // calculate jabcobain by chain rule
    
    virtual void linearizeOplus() override {
        // binary edge,so jacobian xi to v0, jacobian xj to v1 
        // _jacobianOplusXi;
        auto v0 = (VertexSophus *)_vertices[0];
        auto v1 = (g2o::VertexPointXYZ *)_vertices[1];
        Sophus::SE3d Tj = v0->estimate();
        Eigen::Vector3d Pc = Tj*_Pr;
        double Z_2 = Pc(2,0)*Pc(2,0);
        Matrix<double, 2, 3> Jacob_Pc;
        Jacob_Pc << fx/Pc(2,0), 0, -fx*Pc(0,0)/Z_2, 0, fy/Pc(2,0), -fy*Pc(1,0)/Z_2;
        
        double u = fx*Pc(0,0)/Pc(2,0) + cx;
        double v = fy*Pc(1,0)/Pc(2,0) + cy;

        Matrix<double, 3, 6> Jacob_Pc_Tj;
        Jacob_Pc_Tj.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
        Jacob_Pc_Tj.block<3,3>(0,3) = -Sophus::SO3d::hat(Pc);

        Vector16d proj;
        uint8_t patch_size = 4;
        uint8_t index = 0;
        if (u < patch_size/2 || u+patch_size/2 > targetImg.cols
            || v < patch_size/2 || v+patch_size/2 > targetImg.rows)
        {
            _jacobianOplusXi.setZero();
            // _jacobianOplusXj.setZero();
            return;
        }

        for (int dx = -patch_size/2; dx < patch_size/2; dx++)
        {
            for (int dy = -patch_size/2; dy < patch_size/2; dy++)
            {
                // double Ix = (GetPixelValue(targetImg, u+dx+1, v+dy) - GetPixelValue(targetImg, u+dx-1, v+dy))/2.0;
                // double Iy = (GetPixelValue(targetImg, u+dx, v+dy+1) - GetPixelValue(targetImg, u+dx, v+dy-1))/2.0;
                double Ix = ((GetPixelValue(targetImg, u+dx+1, v+dy-1) - GetPixelValue(targetImg, u+dx-1, v+dy-1)) + 
                            2*(GetPixelValue(targetImg, u+dx+1, v+dy) - GetPixelValue(targetImg, u+dx-1, v+dy)) +
                            (GetPixelValue(targetImg, u+dx+1, v+dy+1) - GetPixelValue(targetImg, u+dx-1, v+dy+1)))/4.0;
                double Iy = ((GetPixelValue(targetImg, u+dx-1, v+dy+1) - GetPixelValue(targetImg, u+dx-1, v+dy-1)) + 
                             2*(GetPixelValue(targetImg, u+dx, v+dy+1) - GetPixelValue(targetImg, u+dx, v+dy-1)) + 
                             (GetPixelValue(targetImg, u+dx+1, v+dy+1) - GetPixelValue(targetImg, u+dx+1, v+dy-1))) /4.0;
                
                Eigen::Vector2d cur_grad(Ix, Iy);
                Matrix<double,1,6> cur_Jacob_Tj;
                cur_Jacob_Tj = cur_grad.transpose()*Jacob_Pc*Jacob_Pc_Tj;
                _jacobianOplusXi.block<1,6>(index,0) = cur_Jacob_Tj;

                // Eigen::Matrix3d Rj = Tj.so3().matrix();
                // Eigen::Matrix<double,1,3> cur_Jacob_Pj = cur_grad.transpose()*Jacob_Pc*Rj.transpose();
                // _jacobianOplusXj.block<1,3>(index,0) = cur_Jacob_Pj;
                index++;
            }
        }
    }
    
    
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

private:
    // cv::Mat targetImg;  // the target image
    // float *origColor = nullptr;   // 16 floats, the color of this point
    cv::Mat targetImg;  // the target image
    Vector3d _Pr;
};

void extract_patch(cv::Mat &img, Vector2d &pos, int cur_patch_size, Vector16d &pixel_patch)
{
    if(pos(0) < cur_patch_size/2 || pos(0) + cur_patch_size/2 > img.cols || pos(1) < cur_patch_size/2 || pos(1) + cur_patch_size/2 > img.rows)
    {
        cout << "invalid pos " << pos.transpose() << endl;
        return;
    }

    if(pixel_patch.rows() < cur_patch_size)
    {
        cout << "invalid vector size " << pixel_patch.rows() << endl;
    }

    int index = 0;
    for (int dx = -cur_patch_size/2; dx < cur_patch_size/2; dx++)
        for (int dy = -cur_patch_size/2; dy < cur_patch_size/2; dy++)
        {
            pixel_patch(index) = img.at<uchar>(pos(1)+dy, pos(0)+dx);
            index++;
        }
}

void opt_point_select(cv::Mat &img, VecVec3d &opt_points)
{
    int cols = img.cols;
    int rows = img.rows;

    int index_x = 25;
    int index_y = 25;

    int grid_num = 30;

    int grid_cols = floor(cols/grid_num);
    int grid_rows = floor(rows/grid_num);


    for (int i = 0; i < grid_num; i++)
        for (int j = 0; j < grid_num; j++)
    {
        Vector3d pts;
        pts(0) = j*grid_cols + grid_cols/2;
        pts(1) = i*grid_rows + grid_rows/2;
        if(pts(1) < patch_size/2 || pts(1) + patch_size/2 > rows || pts(0) < patch_size/2 || pts(0) + patch_size/2 > cols )
        {
            continue;
        }

        pts(2) = img.at<uchar>(pts(1), pts(0));

        opt_points.push_back(pts);
    }
}
// string image_left_path = "../../data_set/00/image_2";
// string image_right_path = "../../data_set/00/image_3";

// string image_left_path = "../../data_set_gray/01/image_0";
// string image_right_path = "../../data_set_gray/01/image_1";

string image_left_path = "../../data_set_gray/00/image_0";
string image_right_path = "../../data_set_gray/00/image_1";

// string left_file = "../left.png";
// string right_file = "../right.png";

// cv::Ptr<cv::StereoSGBM> sgbm;

void TrackLastFrame(g2o::OptimizationAlgorithmLevenberg* solver,
        cv::Mat &last_frame, cv::Mat &cur_frame, cv::Mat &last_depth, Sophus::SE3d &rel_pose, Sophus::Vector6d &res);

void TrackLastFrameDirSingle(
    cv::Mat &last_frame, cv::Mat &cur_frame, cv::Mat &last_depth, Sophus::SE3d &rel_pose, Sophus::Vector6d &res);

void TrackLastFrameDirSingleMT(
    cv::Mat &last_frame, cv::Mat &cur_frame, cv::Mat &last_depth, Sophus::SE3d &rel_pose, Sophus::Vector6d &res);

void Draw(const VecSE3 &poses, const VecVec3d &points);

void depth_generate(cv::Mat &left, cv::Mat &right, cv::Mat &depth, vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud)
{
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8*9*9, 32*9*9, 1, 63, 10, 100, 32);
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0/16.0f);

    // generate depth of left image

    // depth.resize(disparity.size(), -10);
    depth.create(disparity.rows, disparity.cols, CV_64F);
    
    // vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;
    int count = 0;
    for(int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++){
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0)
            {
                depth.at<double>(v, u) = -10.0;
            }
            else
            {
                Vector4d point(0, 0, 0, left.at<uchar>(v,u)/255.0);

                double x = (u - cx)/fx;
                double y = (v - cy)/fy;
                double cur_depth = fx * b / (disparity.at<float>(v, u));
                point[0] = x * cur_depth;
                point[1] = y * cur_depth;
                point[2] = cur_depth;

                if (cur_depth < 1e-6)
                {
                    cout << "small depth " << cur_depth << endl;
                }

                pointcloud.push_back(point);
                depth.at<double>(v, u) = cur_depth;
            }
        }
}

void pointcloud_generate(cv::Mat &left, cv::Mat &depth, vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud)
{
    int count = 0;
    for(int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++){
            if (depth.at<double>(v, u) < 0)
            {
                count++;
                continue;
            }

            Vector4d point(0, 0, 0, left.at<uchar>(v,u)/255.0);

            double x = (u - cx)/fx;
            double y = (v - cy)/fy;
            point[2] = depth.at<uchar>(v,u);
            point[0] = x * point[2];
            point[1] = y * point[2];
            pointcloud.push_back(point);
        }

    cout << "resolution of image is " << left.cols << "x" << left.rows << endl;
    cout << "unknow disparity point num is " << count << endl;
}

void showPointCloud(
    const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main()
{
    string left_file = image_left_path + "/000000.png";
    string right_file = image_right_path + "/000000.png";
    cv::Mat last_left = cv::imread(left_file, 0);
    cv::Mat last_right = cv::imread(right_file, 0);

    // string left_file1 = image_left_path + "/001248.png";
    // string right_file1 = image_right_path + "/001248.png";

    // cv::Mat left_match = cv::imread(left_file1, 0);
    // cv::Mat right_match = cv::imread(right_file1, 0);
    cv::Mat last_depth,depth1;
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud,pointcloud1;

    double total_time = 0;

    depth_generate(last_left, last_right, last_depth, pointcloud);
    // depth_generate(left_match, right_match, depth1, pointcloud1);

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
                  g2o::make_unique<BlockSloverType>(g2o::make_unique<LinearSolverType_Cur>()));

    VecSE3 opt_poses;
    opt_poses.push_back(Sophus::SE3d());
    VecVec3d points;
    points.push_back(Vector3d::Zero());
    fstream traj_store;
    traj_store.open("track_result.txt", ios::app | ios::out );
    for (uint32_t i = 1; i < 200; i++)
    {
        stringstream ss;
        ss << setw(6) << setfill('0') << i ;
        string sequnce_num;
        ss >> sequnce_num;         //将字符流传给 str
        string cur_left_file = 	image_left_path + "/" + sequnce_num + ".png";
        string cur_right_file = image_right_path + "/" + sequnce_num + ".png";
        cout << cur_left_file << endl;
        cout << cur_right_file << endl;

        cv::Mat cur_left = cv::imread(cur_left_file, 0);
        cv::Mat cur_right = cv::imread(cur_right_file, 0);
        cv::Mat cur_depth;
        vector<Vector4d, Eigen::aligned_allocator<Vector4d>> cur_pointcloud;

        depth_generate(cur_left, cur_right, cur_depth, cur_pointcloud);

        Sophus::SE3d rel_pose = Sophus::SE3d();
        Sophus::Vector6d res = Sophus::Vector6d::Zero();
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        // TrackLastFrame(solver, last_left, cur_left, last_depth, rel_pose, res);
        // TrackLastFrameDirSingle(last_left, cur_left, last_depth, rel_pose, res);
        TrackLastFrameDirSingleMT(last_left, cur_left, last_depth, rel_pose, res);
        auto t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - now);
        // std::cout << "time_span is " << time_span << std::endl;
        opt_poses.push_back(opt_poses[i - 1]*rel_pose.inverse());
        last_left = cur_left;
        last_right = cur_right;
        last_depth = cur_depth;

        total_time += time_span.count();
        std::cout << "It took " << time_span.count() << " seconds." << endl;

        traj_store << i << ' ' << opt_poses[i].log().transpose() << ' ' << endl;
        traj_store << i << ' ' << res.transpose() << endl;

        // cout << "error befor opti " << res.block<2,1>(0,0).transpose()/res(4) << ", error after opt " << res.block<2,1>(2,0).transpose()/res(5) << endl;
        // cout << "all used keypoint " << res(4) << "valid matches " << res(5) << endl;
        cout << "error befor opti " << res(0)/res(4) << ", error after opt " << res(1)/res(5) << endl;
        cout << "all used keypoint " << res(4) << "valid matches " << res(5) << endl;
    }

    for (uint32_t i = 0; i < 10; i++)
    {
        cout << i << " pose matrix is " << endl << opt_poses[i].matrix() << endl;
    }

    traj_store.close();

    std::cout << "It total took " << total_time << " seconds." << endl;

    Draw(opt_poses, points);
/*
    Sophus::SE3d rel_pose = Sophus::SE3d();
    Vector8d res = Vector8d::Zero();
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    TrackLastFrame(solver, left, left_match, depth, rel_pose, res);
    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - now);
    // std::cout << "time_span is " << time_span << std::endl;
    std::cout << "It took " << time_span.count() << " seconds." << endl;

    cout << "error befor opti " << res.block<3,1>(0,0).transpose()/res(7) << ", error after opt " << res.block<3,1>(3,0).transpose()/res(7) << endl;
    cout << "all good matched " << res(7) << "valid matches " << res(6) << endl;
*/
    // showPointCloud(pointcloud1);
}

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}

void Draw(const VecSE3 &poses, const VecVec3d &points) {
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto &Tcw: poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
            glMultMatrixf((GLfloat *) m.data());
            glColor3f(1, 0, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glEnd();
            glPopMatrix();
        }

        // points
        glPointSize(2);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < points.size(); i++) {
            glColor3f(0.0, points[i][2]/4, 1.0-points[i][2]/4);
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

bool cmp_max(cv::KeyPoint k1, cv::KeyPoint k2){
	return k1.response > k2.response;
}

void TrackLastFrame(g2o::OptimizationAlgorithmLevenberg* solver,
        cv::Mat &last_frame, cv::Mat &cur_frame, cv::Mat &last_depth, Sophus::SE3d &rel_pose, Sophus::Vector6d &res)
{
    // extract fast corner in last frame

    int threshold = 30;
    bool nonmaxSuppression = true;
    int type = FastFeatureDetector::TYPE_9_16;
    Ptr< FastFeatureDetector > 	fast_dector = FastFeatureDetector::create(threshold,
                                                    nonmaxSuppression,
                                                    type );

    Ptr<FeatureDetector> ORB_detector = ORB::create();
    
    // cv::Mat mask(last_frame.rows, last_frame.cols, CV_8U, 0);
    // cv::Rect roi(30,30, last_frame.cols - 30, last_frame.rows - 30);
    // cv::Mat roi_mask(last_frame.rows - 60, last_frame.cols - 60, CV_8U, 1);
    // roi_mask.copyTo(mask);

    std::vector<cv::KeyPoint> keypoints;
    // fast_dector->detect(last_frame, keypoints, mask);
    fast_dector->detect(last_frame, keypoints);
    // ORB_detector->detect(last_frame, keypoints);


    sort(keypoints.begin(),keypoints.end(),cmp_max);

    // cout << "keypoint num is " << keypoints.size() << endl;
    // cout << "the most strong response is " << keypoints[0].response << endl;

    // cout << "the last most strong response is " << keypoints[keypoints.size() -1].response << endl;

    // Mat outimg1;
    // drawKeypoints( last_frame, keypoints, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    // imshow("FAST_FEATURE",outimg1);
    // waitKey(0);


    // choose only 300 fast points
    VecVec3d opt_points;
    opt_point_select(last_frame, opt_points);
    
    // solve for second frame pose
    // auto solver1 = new g2o::OptimizationAlgorithmLevenberg(
    //               g2o::make_unique<BlockSloverType>(g2o::make_unique<LinearSolverType_Cur>()));
    auto solver1 = new g2o::OptimizationAlgorithmGaussNewton(
                        g2o::make_unique<BlockSloverType>(g2o::make_unique<LinearSolverType_Cur>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver1);
    optimizer.setVerbose(true);


    // add vertex to graph
    VertexSophus *pose1 = new VertexSophus();

    pose1->setEstimate(Sophus::SE3d());
    pose1->setId(0);
    optimizer.addVertex(pose1);


    double error_old = 0;
    double error_sq_old = 0;
    int valid_depth_match = 0;
    // add edge
    Matrix16d Information;
    Information.setIdentity();
    int count = 0;
    // int key_size = keypoints.size();
    int key_size = opt_points.size();
    int maxkey_size = 4000;
    for (int i = 0; i < min(key_size, maxkey_size); i++) {

        Vector3d Pr = Vector3d::Zero();
        // int u = keypoints[i].pt.x;
        // int v = keypoints[i].pt.y;
        int u = opt_points[i](0);
        int v = opt_points[i](1);
        Pr(0) = last_depth.at<double>(v, u) *(u - cx)/fx;
        Pr(1) = last_depth.at<double>(v, u) *(v - cy)/fy;
        Pr(2) = last_depth.at<double>(v, u);

        // remove the minus depth point
        if (Pr(2) < 1 || u < patch_size/2 || u + patch_size/2 > last_frame.cols || v < patch_size/2 || v + patch_size/2 > last_frame.rows)
        {
            // cout << "cur depth is " << last_depth.at<double>(v, u) << endl;
            // cout << "cur position is " << Pr.transpose() << "cur image pos is " <<u << ' ' <<v << endl;
            count++;
            continue;
        }

        // error_old += Vector2d(u- uc, v - vc);
        // error_sq_old = error_old.transpose() * error_old;
        Vector2d cur_pos(u,v); 
        Vector16d cur_patch;
        extract_patch(cur_frame, cur_pos, patch_size, cur_patch);

        Vector2d last_pos(u,v);
        Vector16d last_patch;
        extract_patch(last_frame, last_pos, patch_size, last_patch);

        Vector16d diff = last_patch - cur_patch;
        
        error_old += diff.transpose() * diff;
        error_sq_old += diff.transpose() * diff;


        valid_depth_match++;

// Rect roiRect(x0,y0, roiW, roiH);

// if (toUseROI)
//     img = originImg(roiRect).clone();

        // cout << "cur key point measure is " << last_patch.transpose() << endl;
        // cout << "cur key pos is " << u << ' ' << v << endl;
        EdgeDirectProjection *edge = new EdgeDirectProjection(Pr, cur_frame);
        // edge->setId(i);
        // edge->setId(valid_depth_match);
        edge->setVertex(0, pose1);
        edge->setMeasurement(last_patch);
        edge->setInformation(Information); // information matrix, the inverse of coverance matrix
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        bool add_succ = optimizer.addEdge(edge);
        if (!add_succ)
        {
            cout << "add edge failed" << endl;
        }
    }

    // print the projection error befor optimize

    cout << " mean error is " << error_old/min(key_size, maxkey_size) << endl;
    cout << " mean error sq is " << error_sq_old/min(key_size, maxkey_size) << endl;

    cout << "total used keypoint num is " << min(key_size, maxkey_size) << endl;
    cout << "depth little than 1e-6 cout is " << count << endl;


    // perform optimization, max iter time set 20
    optimizer.initializeOptimization(0);
    optimizer.optimize(200);

    // calculate the projection error after optimization
    auto estimate = pose1->estimate();

    Matrix3d K;
    K << fx,0,cx,0,fy,cy,0,0,1;
    double error = 0;
    double error_sq = 0;
    uint32_t final_match = 0;
    double dx,dy;
    dx = 0;
    dy = 0;
    Vector2d pos_error = Vector2d::Zero();
    double pos_square = 0;

    for (int i = 0; i < min(key_size, maxkey_size); i++)
    {
        Vector3d Pr = Vector3d::Zero();
        // int u = keypoints[i].pt.x;
        // int v = keypoints[i].pt.y;
        int u = opt_points[i](0);
        int v = opt_points[i](1);
        Pr(0) = last_depth.at<double>(v, u) *(u - cx)/fx;
        Pr(1) = last_depth.at<double>(v, u) *(v - cy)/fy;
        Pr(2) = last_depth.at<double>(v, u);
        Vector3d Pc = estimate*Pr;
        Vector3d p = K*Pc/Pc(2);
        // cout << "p imag position " << p.transpose() << endl;
        if (p(0) < patch_size/2 || p(0) + patch_size/2 > cur_frame.cols
            || p(1) < patch_size/2 || p(1) + patch_size/2 > cur_frame.rows)
        {
            continue;
        }

        pos_error += Vector2d(p(0)-u, p(1)-v);
        pos_square += (p(0)-u)*(p(0)-u) + (p(1)-v)*(p(1)-v);

        Vector16d cur_patch,last_patch;
        Vector2d cur_pos(p(0), p(1)), last_pos(u, v);
        extract_patch(cur_frame, cur_pos, patch_size, cur_patch);
        extract_patch(last_frame, last_pos, patch_size, last_patch);

        Vector16d  diff = last_patch - cur_patch;
        error += diff.transpose() * diff;
        error_sq += diff.transpose() * diff;
        final_match++;
    }

    res(0) = error_old;
    res(1) = error_sq_old;
    res(2) = error;
    res(3) = error_sq;
    res(4) = min(key_size, maxkey_size);
    res(5) = final_match;

    rel_pose = estimate;

    // print the result
    cout << " opt pose rotation is " << endl << estimate.so3().matrix() << endl;
    cout << " opt pose translation is " << endl << estimate.translation() << endl;
    cout << " mean error is " << error/final_match << endl;
    cout << " mean error sq is " << error_sq/final_match << endl;
    cout << " pos error is " << pos_error.transpose()/final_match << endl;
    cout << " pos error sq is " << pos_square/final_match << endl;

    // cout << "calculate error is " << optimizer.activeChi2() << ' ' << optimizer.activeRobustChi2() << endl;
}

//GN iteration times:100, max used point 1000 
void TrackLastFrameDirSingle(
    cv::Mat &last_frame, cv::Mat &cur_frame, cv::Mat &last_depth, Sophus::SE3d &rel_pose, Sophus::Vector6d &res)
{
    VecVec3d opt_points;
    // opt_point_select(last_frame, opt_points);

    cv::RNG rng;
    int nPoints=1000;
    int boarder = 40;

    // generate opt points
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, last_frame.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, last_frame.rows - boarder);  // don't pick pixels close to boarder
        Vector3d cur_points;
        cur_points(0) = x;
        cur_points(1) = y;
        cur_points(2) = last_frame.at<uchar>(y, x);
        opt_points.push_back(cur_points);
    }

    // Ptr<FeatureDetector> ORB_detector = ORB::create();

    // std::vector<cv::KeyPoint> keypoints;

    // ORB_detector->detect(last_frame, keypoints);

    // cout << "keypoint size is " << keypoints.size() << endl;
    int iter_time = 100;
    int iter_count = 0;
    Matrix3d K;
    K << fx,0,cx,0,fy,cy,0,0,1;
    double cost = 0;
    double last_cost = 0;
    while(iter_count < iter_time)
    {
        // calculate hessian and b
        Sophus::Matrix6d Hessian = Sophus::Matrix6d::Zero();
        Sophus::Vector6d b = Sophus::Vector6d::Zero();
        Sophus::Vector6d update = Sophus::Vector6d::Zero();
        cost = 0;

        // int key_size = keypoints.size();
        // int key_size = opt_points.size();
        // int maxkey_size = 1000;
        for (int i = 0; i < opt_points.size(); i++)
        // for (int i = 0; i < min(key_size, maxkey_size); i++)
        {
            Vector16d cur_pixel_patch = Vector16d::Zero();
            Vector16d last_pixel_patch = Vector16d::Zero();
            double u = opt_points[i](0);
            double v = opt_points[i](1);

            // double u = keypoints[i].pt.x;
            // double v = keypoints[i].pt.y;
            Vector3d Pr;
            double d = last_depth.at<double>(v,u);
            Pr(0) = (u-cx)*d/fx;
            Pr(1) = (v-cy)*d/fy;
            Pr(2) = d;
            if ((Pr(2) < 0) || u < patch_size/2 || u + patch_size/2 > last_frame.cols || v < patch_size/2 || v + patch_size/2 > last_frame.rows)
            {
                continue;
            }
            Vector3d Pc = rel_pose*Pr;
            Vector3d p = K*Pc/Pc(2);
            if ((Pr(2) < 0) || p(0) < patch_size/2 || p(0) + patch_size/2 > last_frame.cols || p(1) < patch_size/2 || p(1) + patch_size/2 > last_frame.rows)
            {
                continue;
            }

            // extract_patch(last_frame, opt_point.block<2,1>(0,0), patch_size, last_pixel_patch);
            // extract_patch(cur_frame, p.block<2,1>(0,0), patch_size, cur_pixel_patch);
            Matrix<double, 2, 3> Jacob_Pc;
            double Z_2 = Pc(2)*Pc(2);
            Jacob_Pc << fx/Pc(2,0), 0, -fx*Pc(0,0)/Z_2, 0, fy/Pc(2,0), -fy*Pc(1,0)/Z_2;

            Matrix<double, 3, 6> Jacob_Pc_Tj;
            Jacob_Pc_Tj.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
            Jacob_Pc_Tj.block<3,3>(0,3) = -Sophus::SO3d::hat(Pc);
            for (int dx = -patch_size/2; dx < patch_size/2; dx++)
            {
                for (int dy = -patch_size/2; dy < patch_size/2; dy++)
                {
                    double Ix = (GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy) - GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy))/2.0;
                    double Iy = (GetPixelValue(cur_frame, p(0)+dx, p(1)+dy+1) - GetPixelValue(cur_frame, p(0)+dx, p(1)+dy-1))/2.0;
                    // double Ix = ((GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy-1) - GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy-1)) + 
                    //             2*(GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy) - GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy)) +
                    //             (GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy+1) - GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy+1)))/4.0;
                    // double Iy = ((GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy+1) - GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy-1)) + 
                    //                 2*(GetPixelValue(cur_frame, p(0)+dx, p(1)+dy+1) - GetPixelValue(cur_frame, p(0)+dx, p(1)+dy-1)) + 
                    //                 (GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy+1) - GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy-1))) /4.0;

                    Vector2d cur_grad(Ix, Iy);
                    double error = GetPixelValue(cur_frame, p(0)+dx, p(1)+dy) - GetPixelValue(last_frame, u+dx, v+dy);

                    Matrix<double,1,6> cur_Jacob_Tj;
                    cur_Jacob_Tj = cur_grad.transpose()*Jacob_Pc*Jacob_Pc_Tj;
                    b += -cur_Jacob_Tj.transpose()*error;
                    Hessian += cur_Jacob_Tj.transpose()*cur_Jacob_Tj;
                    cost += error*error;
                }
            }
        }

        if(iter_count == 0)
        {
            res(0) = cost;
        }

        update = Hessian.ldlt().solve(b);
        
        if ((iter_count > 0)  && cost > last_cost)
        {
            cout << "iter count is" << iter_count << " cur cost greater than last" << cost << ' ' << last_cost << endl;
            break;
        }
        rel_pose = Sophus::SE3d::exp(update) * rel_pose;
        last_cost = cost;
        iter_count++;
    }

    res(1) = cost;
    res(2) = 1.0;
    res(3) = 1.0;
    res(4) = opt_points.size();
    res(5) = opt_points.size();
}

typedef struct Direct_Pose_MT{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector2d ref_p;
    Eigen::Vector3d ref_Pc;
    Sophus::Matrix6d H;
    Sophus::Vector6d b;
    Eigen::Vector2d cur_point;
    double cost;
    bool good;
} Direct_Pose_MT_t;

//GN iteration times:100, max used point 1000 
void TrackLastFrameDirSingleMT(
    cv::Mat &last_frame, cv::Mat &cur_frame, cv::Mat &last_depth, Sophus::SE3d &rel_pose, Sophus::Vector6d &res)
{
    VecVec3d opt_points;
    // opt_point_select(last_frame, opt_points);

    cv::RNG rng;
    int nPoints=1000;
    int boarder = 40;

    // generate opt points
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, last_frame.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, last_frame.rows - boarder);  // don't pick pixels close to boarder
        Vector3d cur_points;
        cur_points(0) = x;
        cur_points(1) = y;
        cur_points(2) = last_frame.at<uchar>(y, x);
        opt_points.push_back(cur_points);
    }

    // Ptr<FeatureDetector> ORB_detector = ORB::create();

    // std::vector<cv::KeyPoint> keypoints;

    // ORB_detector->detect(last_frame, keypoints);

    // cout << "keypoint size is " << keypoints.size() << endl;
    int iter_time = 100;
    int iter_count = 0;
    Matrix3d K;
    K << fx,0,cx,0,fy,cy,0,0,1;
    double cost = 0;
    double last_cost = 0;

    vector<Direct_Pose_MT_t> points_MT_opt;
    points_MT_opt.resize(opt_points.size());

    for (int i = 0; i < opt_points.size(); i++)
    {
        points_MT_opt[i].ref_p = opt_points[i].block<2,1>(0,0);
        points_MT_opt[i].cur_point = opt_points[i].block<2,1>(0,0);
        double u = opt_points[i](0);
        double v = opt_points[i](1);
        double d = last_depth.at<double>(v,u);
        Vector3d Pr;
        Pr(0) = (u-cx)*d/fx;
        Pr(1) = (v-cy)*d/fy;
        Pr(2) = d;
        points_MT_opt[i].ref_Pc = Pr;
        points_MT_opt[i].H = Sophus::Matrix6d::Zero();
        points_MT_opt[i].b = Sophus::Vector6d::Zero();
        points_MT_opt[i].cost = 0;
        points_MT_opt[i].good = false;
    }

    while(iter_count < iter_time)
    {
        // calculate hessian and b
        Sophus::Matrix6d Hessian = Sophus::Matrix6d::Zero();
        Sophus::Vector6d b = Sophus::Vector6d::Zero();
        Sophus::Vector6d update = Sophus::Vector6d::Zero();
        cost = 0;

        for (int i = 0; i < opt_points.size(); i++)
        {
            points_MT_opt[i].H = Sophus::Matrix6d::Zero();
            points_MT_opt[i].b = Sophus::Vector6d::Zero();
            points_MT_opt[i].cost = 0;
            points_MT_opt[i].good = false;
        }

        std::for_each(std::execution::par_unseq, points_MT_opt.begin(), points_MT_opt.end(),
            [&patch_size, &last_frame, &cur_frame, &K, &rel_pose](auto &cur_ref_p) {
                double u = cur_ref_p.ref_p(0);
                double v = cur_ref_p.ref_p(1);

                if ((cur_ref_p.ref_Pc(2) < 0) || u < patch_size/2 || u + patch_size/2 > last_frame.cols || v < patch_size/2 || v + patch_size/2 > last_frame.rows)
                {
                    cur_ref_p.good = false;
                    return;
                }
                Vector3d Pc = rel_pose*cur_ref_p.ref_Pc;
                Vector3d p = K*Pc/Pc(2);
                if (p(0) < patch_size/2 || p(0) + patch_size/2 > last_frame.cols || p(1) < patch_size/2 || p(1) + patch_size/2 > last_frame.rows)
                {
                    cur_ref_p.good = false;
                    return;
                }
                
                cur_ref_p.good = true;
                cur_ref_p.cur_point = p.block<2,1>(0, 0);

                // extract_patch(last_frame, opt_point.block<2,1>(0,0), patch_size, last_pixel_patch);
                // extract_patch(cur_frame, p.block<2,1>(0,0), patch_size, cur_pixel_patch);
                Matrix<double, 2, 3> Jacob_Pc;
                double Z_2 = Pc(2)*Pc(2);
                Jacob_Pc << fx/Pc(2,0), 0, -fx*Pc(0,0)/Z_2, 0, fy/Pc(2,0), -fy*Pc(1,0)/Z_2;

                Matrix<double, 3, 6> Jacob_Pc_Tj;
                Jacob_Pc_Tj.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
                Jacob_Pc_Tj.block<3,3>(0,3) = -Sophus::SO3d::hat(Pc);
                for (int dx = -patch_size/2; dx < patch_size/2; dx++)
                {
                    for (int dy = -patch_size/2; dy < patch_size/2; dy++)
                    {
                        double Ix = (GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy) - GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy))/2.0;
                        double Iy = (GetPixelValue(cur_frame, p(0)+dx, p(1)+dy+1) - GetPixelValue(cur_frame, p(0)+dx, p(1)+dy-1))/2.0;
                        // double Ix = ((GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy-1) - GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy-1)) + 
                        //             2*(GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy) - GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy)) +
                        //             (GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy+1) - GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy+1)))/4.0;
                        // double Iy = ((GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy+1) - GetPixelValue(cur_frame, p(0)+dx-1, p(1)+dy-1)) + 
                        //                 2*(GetPixelValue(cur_frame, p(0)+dx, p(1)+dy+1) - GetPixelValue(cur_frame, p(0)+dx, p(1)+dy-1)) + 
                        //                 (GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy+1) - GetPixelValue(cur_frame, p(0)+dx+1, p(1)+dy-1))) /4.0;

                        Vector2d cur_grad(Ix, Iy);
                        double error = GetPixelValue(cur_frame, p(0)+dx, p(1)+dy) - GetPixelValue(last_frame, u+dx, v+dy);

                        Matrix<double,1,6> cur_Jacob_Tj;
                        cur_Jacob_Tj = cur_grad.transpose()*Jacob_Pc*Jacob_Pc_Tj;
                        cur_ref_p.b += -cur_Jacob_Tj.transpose()*error;
                        cur_ref_p.H += cur_Jacob_Tj.transpose()*cur_Jacob_Tj;
                        cur_ref_p.cost += error*error;
                    }
                }
        });
        // int key_size = keypoints.size();
        // int key_size = opt_points.size();
        // int maxkey_size = 1000;
        for (int i = 0; i < opt_points.size(); i++)
        // for (int i = 0; i < min(key_size, maxkey_size); i++)
        {
            if(points_MT_opt[i].good)
            {
                Hessian += points_MT_opt[i].H;
                b += points_MT_opt[i].b;
                cost += points_MT_opt[i].cost;
            }
        }

        if(iter_count == 0)
        {
            res(0) = cost;
        }

        update = Hessian.ldlt().solve(b);
        
        if ((iter_count > 0)  && cost > last_cost)
        {
            cout << "iter count is" << iter_count << " cur cost greater than last" << cost << ' ' << last_cost << endl;
            break;
        }
        rel_pose = Sophus::SE3d::exp(update) * rel_pose;
        last_cost = cost;
        iter_count++;
    }

    res(1) = cost;
    res(2) = 1.0;
    res(3) = 1.0;
    res(4) = opt_points.size();
    res(5) = opt_points.size();

    // it's a good display way
    // for (auto &px: goodProjection) {
    //     cv::rectangle(img2_show, cv::Point2f(px[0] - 2, px[1] - 2), cv::Point2f(px[0] + 2, px[1] + 2),
    //                   cv::Scalar(0, 250, 0));
    // }
}

void TrackLastFrameDirMultiMT(
    cv::Mat &last_frame, cv::Mat &cur_frame, cv::Mat &last_depth, Sophus::SE3d &rel_pose, Sophus::Vector6d &res)
{
    
}