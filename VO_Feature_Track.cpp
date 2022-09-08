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
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/edge_se3_prior.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace cv;

// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 基线
double b = 0.573;

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> BlockSloverType;
typedef g2o::LinearSolverEigen<BlockSloverType::PoseMatrixType> LinearSolverType_Cur;

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;

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
        cv::Mat &last_frame, cv::Mat &cur_frame, cv::Mat &last_depth, Sophus::SE3d &rel_pose, Vector8d &res);

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
    cout << "cur image size is " << last_left.rows << ' ' << last_left.cols << endl;

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
        Vector8d res = Vector8d::Zero();
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        TrackLastFrame(solver, last_left, cur_left, last_depth, rel_pose, res);
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

        cout << "error befor opti " << res.block<3,1>(0,0).transpose()/res(7) << ", error after opt " << res.block<3,1>(3,0).transpose()/res(7) << endl;
        cout << "all good matched " << res(7) << "valid matches " << res(6) << endl;
    }

    traj_store.close();

    std::cout << "It total took " << total_time << " seconds." << endl;

    cout << "last poses is " << endl << opt_poses[199].matrix() << endl;
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

void TrackLastFrame(g2o::OptimizationAlgorithmLevenberg* solver,
        cv::Mat &last_frame, cv::Mat &cur_frame, cv::Mat &last_depth, Sophus::SE3d &rel_pose, Vector8d &res)
{
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;

    // use orb
    
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);

    //-- detect fast point
    detector->detect ( last_frame,keypoints_1 );
    detector->detect ( cur_frame,keypoints_2 );

    //-- calculate descriptor
    descriptor->compute ( last_frame, keypoints_1, descriptors_1 );
    descriptor->compute ( cur_frame, keypoints_2, descriptors_2 );

    // display the feature extract result
    // Mat outimg1;
    // drawKeypoints( left, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    // imshow("ORB_FEATURE",outimg1);


    //-- match ORB descriptor
    vector<DMatch> matches;
    matcher->match( descriptors_1, descriptors_2, matches);

    
    //-- remove outlier of match point
    double min_dist=10000, max_dist=0;

    // find min and max hamming distance
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }


    cout << "-- Max dist : " << max_dist << endl;
    cout <<  "-- Min dist : " << min_dist << endl;

    // judge outlier 
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 20.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }

    //-- display the match result befor and after remove outlier
    Mat img_match;
    Mat img_goodmatch;
    // drawMatches ( left, keypoints_1, left_match, keypoints_2, matches, img_match );
    // drawMatches ( last_frame, keypoints_1, cur_frame, keypoints_2, good_matches, img_goodmatch );
    // imshow ( "all match point", img_match );
    // imshow ( "match_point after remove outlier", img_goodmatch );
    // waitKey(0);


    // solve for second frame pose
    auto solver1 = new g2o::OptimizationAlgorithmLevenberg(
                  g2o::make_unique<BlockSloverType>(g2o::make_unique<LinearSolverType_Cur>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver1);
    optimizer.setVerbose(true);


    // add vertex to graph
    VertexSophus *pose1 = new VertexSophus();

    pose1->setEstimate(Sophus::SE3d());
    pose1->setId(0);
    optimizer.addVertex(pose1);


    Vector2d error_old = Vector2d::Zero();
    double error_sq_old = 0;
    int valid_match = 0;
    // add edge
    Matrix2d Information;
    Information.setIdentity();
    int count = 0;
    for (int i = 0; i < good_matches.size(); i++) {
        DMatch cur_match = good_matches[i];
        Vector3d Pr = Vector3d::Zero();
        int u = keypoints_1[good_matches[i].queryIdx].pt.x;
        int v = keypoints_1[good_matches[i].queryIdx].pt.y;
        Pr(0) = last_depth.at<double>(v, u) *(u - cx)/fx;
        Pr(1) = last_depth.at<double>(v, u) *(v - cy)/fy;
        Pr(2) = last_depth.at<double>(v, u);

        double uc = keypoints_2[good_matches[i].trainIdx].pt.x;
        double vc = keypoints_2[good_matches[i].trainIdx].pt.y;

        error_old += Vector2d(u- uc, v - vc);
        error_sq_old = error_old.transpose() * error_old;

        // remove the minus depth point
        if (Pr(2) < 1)
        {
            // cout << "cur depth is " << last_depth.at<double>(v, u) << endl;
            // cout << "cur position is " << Pr.transpose() << "cur image pos is " <<u << ' ' <<v << endl;
            count++;
            continue;
        }

        valid_match++;

        ProjectionEdge *edge = new ProjectionEdge(Pr);
        // edge->setId(i);
        edge->setId(valid_match);
        edge->setVertex(0, pose1);
        edge->setMeasurement(Vector2d(uc, vc));
        edge->setInformation(Information); // information matrix, the inverse of coverance matrix
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    // print the projection error befor optimize
    /* 
    cout << " mean error is " << error_old.transpose()/valid_match << endl;
    cout << " mean error sq is " << error_sq_old/valid_match << endl;

    cout << "total good match num is " << good_matches.size() << endl;
    cout << "little than 1e-6 cout is " << count << endl;
    */

    // perform optimization, max iter time set 20
    optimizer.initializeOptimization(0);
    optimizer.optimize(100);

    // calculate the projection error after optimization
    auto estimate = pose1->estimate();

    Matrix3d K;
    K << fx,0,cx,0,fy,cy,0,0,1;
    Vector2d error = Vector2d::Zero();
    double error_sq = 0;

    for (int i = 0; i < good_matches.size(); i++)
    {
        DMatch cur_match = good_matches[i];
        Vector3d Pr = Vector3d::Zero();
        int u = keypoints_1[good_matches[i].queryIdx].pt.x;
        int v = keypoints_1[good_matches[i].queryIdx].pt.y;
        Pr(0) = last_depth.at<double>(v, u) *(u - cx)/fx;
        Pr(1) = last_depth.at<double>(v, u) *(v - cy)/fy;
        Pr(2) = last_depth.at<double>(v, u);
        Vector3d Pc = estimate*Pr;
        Vector3d p = K*Pc/Pc(2);
        // cout << "p imag position " << p.transpose() << endl;

        double uc = keypoints_2[good_matches[i].trainIdx].pt.x;
        double vc = keypoints_2[good_matches[i].trainIdx].pt.y;

        error += Vector2d(p(0)- uc, p(1) - vc);
        error_sq = error.transpose() * error;
    }

    res(0) = error_old(0);
    res(1) = error_old(1);
    res(2) = error_sq_old;
    res(3) = error(0);
    res(4) = error(1);
    res(5) = error_sq;
    res(6) = valid_match;
    res(7) = good_matches.size();

    rel_pose = estimate;

    // print the result
    cout << " opt pose rotation is " << endl << estimate.so3().matrix() << endl;
    cout << " opt pose translation is " << endl << estimate.translation() << endl;
    cout << " mean error is " << error.transpose()/good_matches.size() << endl;
    cout << " mean error sq is " << error_sq/good_matches.size() << endl;

    // cout << "calculate error is " << optimizer.activeChi2() << ' ' << optimizer.activeRobustChi2() << endl;
}