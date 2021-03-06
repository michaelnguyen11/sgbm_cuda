#include <memory>
#include <time.h>
#include <sys/stat.h>
#include <signal.h>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <numeric>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

// Sample includes
#include <time.h>

#include "common.h"
#include "camera/stereo_camera.hpp"

static bool is_streaming = true;
static void sig_handler(int sig)
{
    is_streaming = false;
}

static cv::CommandLineParser getConfig(int argc, char **argv)
{
    const char *params = "{ help           | false              | print usage          }"
                         "{ fps            | 30                 | (int) Frame rate }"
                         "{ width          | 1280               | (int) Image width }"
                         "{ height         | 720                | (int) Image height }"
                         "{ max_disparity  | 128                | (int) Maximum disparity }"
                         "{ subpixel       | true               | Compute subpixel accuracy }"
                         "{ num_path       | 4                  | (int) Num path to optimize, 4 or 8 }";

    cv::CommandLineParser config(argc, argv, params);
    if (config.get<bool>("help"))
    {
        config.printMessage();
        exit(0);
    }

    return config;
}

void free_gpu_mem();
cv::Mat compute_disparity(cv::Mat *left_img, cv::Mat *right_img);
void cuda_init(SGM_PARAMS *params);

int main(int argc, char **argv)
{
    // handle signal by user
    struct sigaction act;
    act.sa_handler = sig_handler;
    sigaction(SIGINT, &act, NULL);

    // parse config from cmd
    cv::CommandLineParser config = getConfig(argc, argv);

    StereoCameraConfig camConfig;
    camConfig.fps = config.get<int>("fps");
    camConfig.width = config.get<int>("width");
    camConfig.height = config.get<int>("height");

    StereoCamera::Ptr camera = std::make_shared<StereoCamera>(camConfig);
    if (!camera->checkCameraStarted())
    {
        std::cout << "Camera open fail..." << std::endl;
        return 0;
    }

    // Stereo Camera
    cv::Mat frame_0, frame_1, frame_0_rect, frame_1_rect;
    cv::Mat disp16, disp32;
    StereoCameraData camDataCamera;

    cv::FileStorage fs("ocams_calibration_720p.xml", cv::FileStorage::READ);

    cv::Mat D_L, K_L, D_R, K_R;
    cv::Mat Rect_L, Proj_L, Rect_R, Proj_R, Q;
    cv::Mat baseline;
    cv::Mat Rotation, Translation;

    fs["D_L"] >> D_L;
    fs["K_L"] >> K_L;
    fs["D_R"] >> D_R;
    fs["K_R"] >> K_R;
    fs["baseline"] >> baseline;
    fs["Rotation"] >> Rotation;
    fs["Translation"] >> Translation;

    // Code to calculate Rotation matrix and Projection matrix for each camera
    cv::Vec3d Translation_2((double *)Translation.data);

    cv::stereoRectify(K_L, D_L, K_R, D_R, cv::Size(camConfig.width, camConfig.height), Rotation, Translation_2,
                      Rect_L, Rect_R, Proj_L, Proj_R, Q, cv::CALIB_ZERO_DISPARITY);

    cv::Mat map11, map12, map21, map22;

    cv::initUndistortRectifyMap(K_L, D_L, Rect_L, Proj_L, cv::Size(camConfig.width, camConfig.height), CV_32FC1, map11, map12);
    cv::initUndistortRectifyMap(K_R, D_R, Rect_R, Proj_R, cv::Size(camConfig.width, camConfig.height), CV_32FC1, map21, map22);

    SGM_PARAMS params;
    params.preFilterCap = 63;
    params.BlockSize = 9;
    params.P1 = 8 * params.BlockSize * params.BlockSize;
    params.P2 = 32 * params.BlockSize * params.BlockSize;
    params.uniquenessRatio = 1;
    params.disp12MaxDiff = 15;
    cuda_init(&params);

    while (1)
    {
        if (!is_streaming)
        {
            std::cout << "Exit by user signal" << std::endl;
            break;
        }
        if (camera->getCamData(camDataCamera))
        {
            frame_0 = camDataCamera.frame_0;
            frame_1 = camDataCamera.frame_1;

            cv::remap(frame_0, frame_0_rect, map11, map12, cv::INTER_LINEAR);
            cv::remap(frame_1, frame_1_rect, map21, map22, cv::INTER_LINEAR);

            cv::Mat disparity16S = compute_disparity(&frame_0_rect, &frame_1_rect);

            cv::Mat disp8U;
            disparity16S.convertTo(disp8U, CV_8U, 255 / (MAX_DISPARITY * 16.));

            cv::Mat dispColor;
            cv::applyColorMap(disp8U, dispColor, cv::COLORMAP_JET);

            cv::imshow("Disparity map", disp8U);
            cv::waitKey(0);
        }
    }

    free_gpu_mem();

    return 0;
}
