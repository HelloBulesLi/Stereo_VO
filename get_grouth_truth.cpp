#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <eigen3/Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>
#include <fstream>
#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;

// 内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// 基线
double b = 0.573;

typedef vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> VecMatrix4d;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;

string gt_path = "/home/libo/slam_work/final_project/gt_result/data_odometry_poses/dataset/poses";

void Draw(const VecMatrix4d &poses, const VecVec3d &points);

int main()
{
    gt_path += "/00.txt";

    fstream gt_file;
    gt_file.open(gt_path, ios::app | ios::in);

    VecMatrix4d poses ;

    VecVec3d points;
    points.push_back(Vector3d::Zero());

    while(!gt_file.eof())
    {
        Matrix4d cur_pose = Eigen::Matrix4d::Identity();
        gt_file >> cur_pose(0,0) >> cur_pose(0,1) >> cur_pose(0,2) >> cur_pose(0,3)
                >> cur_pose(1,0) >> cur_pose(1,1) >> cur_pose(1,2) >> cur_pose(1,3)
                >> cur_pose(2,0) >> cur_pose(2,1) >> cur_pose(2,2) >> cur_pose(2,3);
        cur_pose.block<1,4>(3,0) = Vector4d(0,0,0,1);
        poses.push_back(cur_pose);
    }

    cout << "file size is " << poses.size() << endl;

    Matrix4d rel_pos = poses[3].inverse() * poses[0];
    cout << "first pose trans is " << endl << poses[0].block<3,1>(0,3) << endl;
    cout << "second pose trans is " << endl << poses[1].block<3,1>(0,3) << endl;
    cout << "thrid pose trans is " << endl << poses[2].block<3,1>(0,3) << endl;

    // cout << "second pose trans is " << endl << poses[1].block<3,1>(0,3) << endl;
    // cout << "thrid pose trans is " << endl << poses[2].block<3,1>(0,3) << endl;
    cout << "fourth pose trans is " << endl << poses[3].block<3,1>(0,3) << endl;
    cout << "fiveth pose trans is " << endl << poses[4].block<3,1>(0,3) << endl;
    cout << "sixth pose trans is " << endl << poses[5].block<3,1>(0,3) << endl;
    cout << "last pose trans is " << endl << poses[199].block<3,1>(0,3) << endl;
    
    cout << "rel rotate is " << endl << rel_pos.block<3,3>(0,0) << endl;
    cout << "rel trans is " << endl << rel_pos.block<3,1>(0,3) << endl;

    VecMatrix4d poses_partial(poses.begin(), poses.begin()+200);
    Draw(poses_partial, points);

    return 0;
}

void Draw(const VecMatrix4d &poses, const VecVec3d &points) {
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
            Sophus::Matrix4f m = Tcw.inverse().cast<float>();
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